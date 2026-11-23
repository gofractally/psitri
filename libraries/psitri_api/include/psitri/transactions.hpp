#pragma once
// Transaction types that work on any engine satisfying the concept.
//
//   transient_transaction<Snapshot>
//       Never commits. Writes go to a private delta; reads see the delta
//       merged over a snapshot. It takes no lock and touches nothing shared,
//       so any number run at once, each on its own thread. Use it to try a
//       change that will be thrown away.
//
//   delta_transaction<Engine>
//       Holds the root's writer for its whole life, so writers queue. Writes
//       go to a private delta and the tree is not touched until commit, which
//       applies the delta in key order and publishes the root once. Readers
//       of the root are unaffected until that publication.
//
// The engine's own transaction is the third kind: it changes a private copy
// of the tree as it goes and publishes it on commit.
//
// Reads through either type, including cursors in both directions, see the
// transaction's own writes. A cursor is a snapshot, as an engine's cursor is:
// it shows the state at the moment it was created and does not change when
// the transaction is written to afterwards. It must not be used after the
// transaction commits, aborts or is reset.

#include <psitri/concepts.hpp>

#include <art/art_map.hpp>
#include <art/heap_arena.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace psitri
{
   namespace detail
   {
      // One changed key: a new value (stored after this header) or a removal.
      struct delta_entry
      {
         std::uint8_t  removed = 0;
         std::uint32_t size    = 0;
         std::string_view value() const noexcept
         {
            return {reinterpret_cast<const char*>(this) + sizeof(delta_entry), size};
         }
      };
   }  // namespace detail

   /// The changes of one transaction, in key order. Memory comes from one
   /// region that is reused, not freed, when the store is cleared.
   class delta_store
   {
     public:
      using map_type = art::art_map<detail::delta_entry, art::heap_arena>;
      using iterator = art::art_iterator<detail::delta_entry, art::heap_arena>;

      explicit delta_store(std::uint32_t initial_bytes = 1u << 16) : _map(initial_bytes) {}

      void put(std::string_view key, std::string_view value)
      {
         if (value.size() > UINT32_MAX)
            throw std::length_error("value too large for a transaction delta");
         _map.upsert_inline(key, detail::delta_entry{0, static_cast<std::uint32_t>(value.size())},
                            value);
      }
      void erase(std::string_view key)
      {
         _map.upsert_inline(key, detail::delta_entry{1, 0}, {});
      }
      const detail::delta_entry* find(std::string_view key) const noexcept
      {
         return const_cast<map_type&>(_map).get(key);
      }

      /// The contents at one moment. Later changes to the store do not show
      /// through it. Valid until the store is cleared.
      class view
      {
        public:
         view() = default;
         iterator begin() const noexcept { return art::make_begin<detail::delta_entry>(*_arena, _root); }
         iterator end() const noexcept { return art::make_end<detail::delta_entry>(*_arena, _root); }
         /// First entry with key >= `key`.
         iterator at_or_after(std::string_view key) const noexcept
         {
            return art::make_lower_bound<detail::delta_entry>(*_arena, _root, key);
         }
         /// First entry with key > `key`.
         iterator after(std::string_view key) const noexcept
         {
            auto it = at_or_after(key);
            if (it != end() && it.key() == key)
               ++it;
            return it;
         }
         const detail::delta_entry* find(std::string_view key) const noexcept
         {
            return art::get<detail::delta_entry>(*_arena, _root, key);
         }

        private:
         friend class delta_store;
         view(art::heap_arena* arena, art::offset_t root) : _arena(arena), _root(root) {}
         art::heap_arena* _arena = nullptr;
         art::offset_t    _root  = art::null_offset;
      };
      /// The contents as they are now, unaffected by later writes.
      view freeze() noexcept
      {
         // Entries written from here on never overwrite ones that exist now.
         _map.bump_cow_seq();
         return view(&_map.get_arena(), _map.root());
      }

      bool          empty() const noexcept { return _map.empty(); }
      std::uint32_t size() const noexcept { return _map.size(); }

      /// A point the contents can be returned to.
      struct mark
      {
         art::offset_t root;
         std::uint32_t size;
      };
      mark save() noexcept
      {
         // Entries written from here on never overwrite ones that exist now.
         _map.bump_cow_seq();
         return {_map.root(), _map.size()};
      }
      void restore(const mark& m) noexcept
      {
         _map.root() = m.root;
         _map.set_size(m.size);
         _map.bump_cow_seq();
      }
      /// The contents as they were at a mark that has not been rolled back
      /// past. Valid until the store is cleared.
      view at(const mark& m) noexcept { return view(&_map.get_arena(), m.root); }

      /// Bytes of the store's region in use. Rolling back does not lower it;
      /// clearing does. The region holds at most 4 GiB.
      std::uint32_t bytes() const noexcept
      {
         return const_cast<map_type&>(_map).get_arena().bytes_used();
      }

      /// Bytes of memory the store holds on to, in use or not.
      std::uint64_t bytes_held() const noexcept
      {
         return const_cast<map_type&>(_map).get_arena().bytes_held();
      }

      void clear() noexcept { _map.clear(); }
      /// Forget the contents but keep their memory, so that views frozen
      /// earlier can still be read.
      void drop_contents() noexcept
      {
         _map.root() = art::null_offset;
         _map.set_size(0);
         _map.bump_cow_seq();
      }

     private:
      map_type _map;
   };

   namespace detail
   {
      // Stores are kept per thread and handed out one per live transaction.
      inline std::vector<std::unique_ptr<delta_store>>& idle_delta_stores()
      {
         thread_local std::vector<std::unique_ptr<delta_store>> stores;
         return stores;
      }
      inline constexpr std::uint64_t max_idle_delta_bytes = 8u << 20;
      inline std::unique_ptr<delta_store> take_delta_store()
      {
         auto& idle = idle_delta_stores();
         if (idle.empty())
            return std::make_unique<delta_store>();
         auto result = std::move(idle.back());
         idle.pop_back();
         return result;
      }
      inline void give_delta_store(std::unique_ptr<delta_store> store) noexcept
      {
         if (!store)
            return;
         // A store that grew large is freed, not kept: an idle store holds on
         // to its region, and one outsized transaction should not pin that
         // much memory on the thread for good.
         if (store->bytes_held() > max_idle_delta_bytes)
            return;
         store->clear();
         try
         {
            idle_delta_stores().push_back(std::move(store));
         }
         catch (...)
         {
         }
      }
   }  // namespace detail

   /// A delta_store taken from the calling thread's idle stores and handed
   /// back, cleared, to the thread that destroys this. For an embedder that
   /// keeps changes aside itself instead of through the transaction types.
   class pooled_delta_store
   {
     public:
      pooled_delta_store() : _store(detail::take_delta_store()) {}
      pooled_delta_store(pooled_delta_store&&) noexcept = default;
      pooled_delta_store& operator=(pooled_delta_store&& other) noexcept
      {
         if (this != &other)
         {
            detail::give_delta_store(std::move(_store));
            _store = std::move(other._store);
         }
         return *this;
      }
      ~pooled_delta_store() { detail::give_delta_store(std::move(_store)); }

      delta_store&       operator*() noexcept { return *_store; }
      const delta_store& operator*() const noexcept { return *_store; }
      delta_store*       operator->() noexcept { return _store.get(); }
      const delta_store* operator->() const noexcept { return _store.get(); }

     private:
      std::unique_ptr<delta_store> _store;
   };

   /// A cursor over a frozen delta merged with a base cursor. It has the
   /// positioning rules of an engine cursor: `lower_bound` finds the first key
   /// at or after its argument and `upper_bound` the last key at or before it.
   template <adb::mergeable_cursor Base>
   class merged_cursor
   {
     public:
      merged_cursor(delta_store::view delta, Base base) : _delta(delta), _base(std::move(base)) {}

      bool seek_begin() { return forward_from({}, true); }
      bool lower_bound(std::string_view key) { return forward_from(key, true); }
      bool seek_last()
      {
         _d      = _delta.end();
         _d_live = step_delta_back();
         _b_live = _base.seek_last();
         return settle_backward();
      }
      bool upper_bound(std::string_view key) { return backward_from(key, true); }

      bool next()
      {
         if (_where == position::after_last)
            return false;
         if (_where == position::before_first)
            return seek_begin();
         if (!_forward || _moved)
            return forward_from(std::string(_key), false);
         if (_on_delta)
            _d_live = (++_d != _delta.end());
         else
            _b_live = _base.next();
         return settle_forward();
      }
      bool prev()
      {
         if (_where == position::before_first)
            return false;
         if (_where == position::after_last)
            return seek_last();
         if (_forward || _moved)
            return backward_from(std::string(_key), false);
         if (_on_delta)
            _d_live = step_delta_back();
         else
            _b_live = _base.prev();
         return settle_backward();
      }

      bool is_end() const noexcept { return _where == position::after_last; }
      bool is_rend() const noexcept { return _where == position::before_first; }

      std::string_view key() const noexcept { return _key; }

      template <class T = std::string>
      std::optional<T> value()
      {
         if (!on_key())
            return std::nullopt;
         if (_on_delta)
         {
            auto v = _d.value().value();
            T    result;
            result.resize(v.size());
            if (!v.empty())
               std::memcpy(result.data(), v.data(), v.size());
            return result;
         }
         return _base.template value<T>();
      }
      template <class F>
      bool get_value(F&& callback)
      {
         if (!on_key())
            return false;
         if (_on_delta)
         {
            callback(_d.value().value());
            return true;
         }
         return _base.get_value(std::forward<F>(callback));
      }
      bool is_subtree() { return on_key() && !_on_delta && _base.is_subtree(); }

      /// Point reads. They do not change where the cursor is.
      template <class F>
      bool get(std::string_view key, F&& callback)
      {
         if (auto* entry = _delta.find(key))
         {
            if (entry->removed)
               return false;
            callback(entry->value());
            return true;
         }
         _moved = true;  // the base cursor is used for the read
         return _base.get(key, std::forward<F>(callback));
      }
      template <class T = std::string>
      std::optional<T> get(std::string_view key)
      {
         std::optional<T> result;
         get(key,
             [&](std::string_view v)
             {
                result.emplace();
                result->resize(v.size());
                if (!v.empty())
                   std::memcpy(result->data(), v.data(), v.size());
             });
         return result;
      }

     private:
      enum class position : std::uint8_t
      {
         before_first,
         on_key,
         after_last
      };

      // On a key, with both sources where a step expects them.
      bool on_key()
      {
         if (_where != position::on_key)
            return false;
         if (_moved)
         {
            std::string key(_key);
            if (_forward)
               forward_from(key, true);
            else
               backward_from(key, true);
         }
         return _where == position::on_key;
      }

      bool step_delta_back()
      {
         if (_d == _delta.begin())
            return false;
         --_d;
         return true;
      }

      // Position on the first key >= `key` (or > `key` when not inclusive).
      bool forward_from(std::string_view key, bool inclusive)
      {
         std::string held(key);
         if (inclusive)
         {
            _d      = _delta.at_or_after(held);
            _b_live = held.empty() ? _base.seek_begin() : _base.lower_bound(held);
         }
         else
         {
            _d      = _delta.after(held);
            _b_live = _base.lower_bound(held);
            if (_b_live && _base.key() == held)
               _b_live = _base.next();
         }
         _d_live = _d != _delta.end();
         return settle_forward();
      }
      // Position on the last key <= `key` (or < `key` when not inclusive).
      bool backward_from(std::string_view key, bool inclusive)
      {
         std::string held(key);
         _d      = inclusive ? _delta.after(held) : _delta.at_or_after(held);
         _d_live = step_delta_back();
         _b_live = _base.upper_bound(held);
         if (!inclusive && _b_live && _base.key() == held)
            _b_live = _base.prev();
         return settle_backward();
      }

      // Both sources are at their first candidate. Pick the smaller key; the
      // delta wins a tie, and a removal in the delta hides the key.
      bool settle_forward()
      {
         _forward = true;
         _moved   = false;
         for (;;)
         {
            if (!_d_live && !_b_live)
            {
               _where = position::after_last;
               _key.clear();
               return false;
            }
            if (_d_live && (!_b_live || _d.key() <= _base.key()))
            {
               if (_b_live && _d.key() == _base.key())
                  _b_live = _base.next();
               if (_d.value().removed)
               {
                  _d_live = (++_d != _delta.end());
                  continue;
               }
               _on_delta = true;
               _key.assign(_d.key());
            }
            else
            {
               _on_delta = false;
               _key.assign(_base.key());
            }
            _where = position::on_key;
            return true;
         }
      }
      bool settle_backward()
      {
         _forward = false;
         _moved   = false;
         for (;;)
         {
            if (!_d_live && !_b_live)
            {
               _where = position::before_first;
               _key.clear();
               return false;
            }
            if (_d_live && (!_b_live || _d.key() >= _base.key()))
            {
               if (_b_live && _d.key() == _base.key())
                  _b_live = _base.prev();
               if (_d.value().removed)
               {
                  _d_live = step_delta_back();
                  continue;
               }
               _on_delta = true;
               _key.assign(_d.key());
            }
            else
            {
               _on_delta = false;
               _key.assign(_base.key());
            }
            _where = position::on_key;
            return true;
         }
      }

      delta_store::view     _delta;
      Base                  _base;
      delta_store::iterator _d{};
      bool                  _d_live   = false;
      bool                  _b_live   = false;
      bool                  _on_delta = false;
      bool                  _forward  = true;
      bool                  _moved    = false;  // the base cursor is no longer at its place
      position              _where    = position::before_first;
      std::string           _key;
   };

   namespace detail
   {
      // Shared by both transaction types: a delta over something readable.
      template <class Derived, class BaseCursor>
      class delta_operations
      {
        public:
         using cursor = merged_cursor<BaseCursor>;

         /// A point to roll back to. Savepoints nest; release or roll back the
         /// most recent one first.
         struct savepoint
         {
            delta_store::mark mark;
            std::uint32_t     depth;
         };
         savepoint save()
         {
            return {store().save(), ++_depth};
         }
         /// Undo everything written since the savepoint.
         void rollback(const savepoint& point)
         {
            check_innermost(point);
            store().restore(point.mark);
            --_depth;
         }
         /// Keep everything written since the savepoint.
         void release(const savepoint& point)
         {
            check_innermost(point);
            --_depth;
         }

         /// Write the changes, in key order, to anything that has `upsert` and
         /// `remove`: an engine's transaction, an editor of a tree that is not
         /// installed in any root, or another transaction of these types. The
         /// changes stay in this transaction as well.
         template <class Target>
         void apply_to(Target& target)
         {
            auto changes = store().freeze();
            for (auto it = changes.begin(), end = changes.end(); it != end; ++it)
            {
               if (it.value().removed)
                  target.remove(it.key());
               else
                  target.upsert(it.key(), it.value().value());
            }
         }
         /// How many keys this transaction has written or removed.
         std::uint32_t changed_keys() const noexcept { return store().size(); }

        protected:
         delta_store& store() { return *static_cast<Derived*>(this)->_store; }
         const delta_store& store() const { return *static_cast<const Derived*>(this)->_store; }
         std::uint32_t depth() const noexcept { return _depth; }
         void reset_depth() noexcept { _depth = 0; }

         template <class C>
         static bool read_through(const delta_store& delta, C&& base, std::string_view key,
                                  auto&& callback)
         {
            if (auto* entry = delta.find(key))
            {
               if (entry->removed)
                  return false;
               callback(entry->value());
               return true;
            }
            return base.get(key, callback);
         }

         // Remove every key in [lower, upper) by recording a removal for each.
         std::uint64_t remove_each(cursor scan, std::string_view lower, std::string_view upper)
         {
            std::vector<std::string> keys;
            for (bool more = lower.empty() ? scan.seek_begin() : scan.lower_bound(lower);
                 more && (upper.empty() || scan.key() < upper); more = scan.next())
               keys.emplace_back(scan.key());
            for (auto& key : keys)
               store().erase(key);
            return keys.size();
         }

        private:
         void check_innermost(const savepoint& point) const
         {
            if (point.depth != _depth)
               throw std::logic_error("savepoints must be closed innermost first");
         }
         std::uint32_t _depth = 0;
      };
   }  // namespace detail

   /// A transaction that is never committed. See the top of this file.
   template <adb::shared_snapshot Snapshot>
   class transient_transaction
       : public detail::delta_operations<
             transient_transaction<Snapshot>,
             decltype(std::declval<const Snapshot&>().snapshot_cursor())>
   {
      using base_cursor = decltype(std::declval<const Snapshot&>().snapshot_cursor());
      using operations  = detail::delta_operations<transient_transaction<Snapshot>, base_cursor>;
      friend operations;

     public:
      using cursor = merged_cursor<base_cursor>;

      explicit transient_transaction(Snapshot base)
          : _base(std::move(base)), _store(detail::take_delta_store())
      {
      }
      transient_transaction(transient_transaction&&) noexcept            = default;
      transient_transaction& operator=(transient_transaction&&) noexcept = delete;
      ~transient_transaction() { detail::give_delta_store(std::move(_store)); }

      void upsert(std::string_view key, std::string_view value) { _store->put(key, value); }
      /// 0 if the key was there, -1 if not.
      int remove(std::string_view key)
      {
         if (!get(key, [](std::string_view) {}))
            return -1;
         _store->erase(key);
         return 0;
      }
      /// Removes every key in [lower, upper); an empty bound is open. Returns
      /// how many keys were removed.
      std::uint64_t remove_range(std::string_view lower, std::string_view upper)
      {
         return this->remove_each(snapshot_cursor(), lower, upper);
      }
      /// The name the engines' transactions use.
      std::uint64_t remove_range_counted(std::string_view lower, std::string_view upper)
      {
         return remove_range(lower, upper);
      }

      template <class F>
      bool get(std::string_view key, F&& callback) const
      {
         // One cursor on the snapshot serves every point read.
         if (!_reader)
            _reader.emplace(_base.snapshot_cursor());
         return operations::read_through(*_store, *_reader, key, callback);
      }
      template <class T = std::string>
      std::optional<T> get(std::string_view key) const
      {
         std::optional<T> result;
         get(key, [&](std::string_view v) { result.emplace(v.data(), v.size()); });
         return result;
      }

      /// A cursor over the snapshot with this transaction's changes applied.
      cursor snapshot_cursor() const { return cursor(_store->freeze(), _base.snapshot_cursor()); }

      /// Throw away every change and start again on the same snapshot.
      void reset()
      {
         _store->clear();
         this->reset_depth();
      }

     private:
      Snapshot                           _base;
      std::unique_ptr<delta_store>       _store;
      mutable std::optional<base_cursor> _reader;
   };

   /// A transaction that keeps its changes in a delta and applies them at
   /// commit. See the top of this file.
   template <psitri::engine E>
   class delta_transaction
       : public detail::delta_operations<
             delta_transaction<E>,
             decltype(std::declval<const snapshot_for<E>&>().snapshot_cursor())>
   {
      using snapshot    = snapshot_for<E>;
      using base_cursor = decltype(std::declval<const snapshot&>().snapshot_cursor());
      using operations  = detail::delta_operations<delta_transaction<E>, base_cursor>;
      friend operations;

     public:
      using cursor = merged_cursor<base_cursor>;

      /// Waits for the root's writer, as the engine's own transaction does.
      delta_transaction(typename E::write_session& session, std::uint32_t root)
          : _inner(session.start_transaction(root)),
            _base(session.get_root(root)),
            _store(detail::take_delta_store())
      {
      }
      delta_transaction(delta_transaction&&) noexcept            = default;
      delta_transaction& operator=(delta_transaction&&) noexcept = delete;
      ~delta_transaction()
      {
         if (_store)
            abort();
      }

      void upsert(std::string_view key, std::string_view value) { _store->put(key, value); }
      /// 0 if the key was there, -1 if not.
      int remove(std::string_view key)
      {
         if (!get(key, [](std::string_view) {}))
            return -1;
         _store->erase(key);
         return 0;
      }
      /// Removes every key in [lower, upper); an empty bound is open. Returns
      /// how many keys were removed.
      ///
      /// With both bounds given and no savepoint open, the changes made so far
      /// are written to the engine's transaction and the range is removed there
      /// in one operation; the transaction then carries on keeping its changes
      /// aside as before. Otherwise each key in the range is removed singly.
      std::uint64_t remove_range(std::string_view lower, std::string_view upper)
      {
         if (lower.empty() || upper.empty() || this->depth() != 0)
            return this->remove_each(snapshot_cursor(), lower, upper);
         this->apply_to(_inner);
         _store->drop_contents();
         _written = true;
         const auto removed = _inner.remove_range_counted(lower, upper);
         _reader.reset();  // what the changes sit on has moved
         return removed;
      }
      /// The name the engines' transactions use.
      std::uint64_t remove_range_counted(std::string_view lower, std::string_view upper)
      {
         return remove_range(lower, upper);
      }

      template <class F>
      bool get(std::string_view key, F&& callback) const
      {
         // One cursor on what the changes sit on serves every point read.
         if (!_reader)
            _reader.emplace(base_position());
         return operations::read_through(*_store, *_reader, key, callback);
      }
      template <class T = std::string>
      std::optional<T> get(std::string_view key) const
      {
         std::optional<T> result;
         get(key, [&](std::string_view v) { result.emplace(v.data(), v.size()); });
         return result;
      }

      /// A cursor over the root as it was when this transaction started, with
      /// this transaction's changes applied.
      cursor snapshot_cursor() const { return cursor(_store->freeze(), base_position()); }

      /// Apply the changes in key order and publish the root.
      void commit()
      {
         this->apply_to(_inner);
         _inner.commit();
         finish();
      }
      /// Discard the changes. The root is as it was.
      void abort() noexcept
      {
         _inner.abort();
         finish();
      }

     private:
      // What the changes kept aside sit on: the root as it was at the start,
      // or the engine's transaction once a range remove has written to it.
      base_cursor base_position() const
      {
         if (_written)
            return const_cast<typename E::transaction&>(_inner).snapshot_cursor();
         return _base.snapshot_cursor();
      }
      void finish() noexcept
      {
         _reader.reset();
         this->reset_depth();
         detail::give_delta_store(std::move(_store));
      }

      typename E::transaction            _inner;
      snapshot                           _base;
      std::unique_ptr<delta_store>       _store;
      mutable std::optional<base_cursor> _reader;
      bool                               _written = false;
   };
}  // namespace psitri

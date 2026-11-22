#pragma once
// Internal byte-key/value adapter for DWAL consumers such as the MDBX shim.
// This is not the public psitri::engine concept: persistent subtree snapshots
// and synchronous commit policies are not exposed by this adapter.
#include <psitri/dwal/dwal_database_impl.hpp>
#include <psitri/dwal/merge_cursor_impl.hpp>
#include <psitri/transactions.hpp>
#include <map>
#include <mutex>
#include <unordered_map>
#include <variant>
#include <type_traits>

namespace psitri::dwal
{
   namespace detail
   {
      template <class Base, bool = std::is_copy_constructible_v<typename Base::cursor>>
      struct byte_base_snapshot
      {
         std::shared_ptr<typename Base::read_session> session;
         typename Base::cursor point;
         byte_base_snapshot(typename Base::database& db, uint32_t index)
             : session(db.start_read_session()), point(session->create_cursor(index)) {}
         typename Base::cursor create_cursor() const { return point; }
      };

      template <class Base>
      struct byte_base_snapshot<Base, false>
      {
         // Multiple independent cursors must never reuse a single hazard slot.
         decltype(std::declval<typename Base::database&>().get_root_owned(0)) root;
         typename Base::cursor point;
         byte_base_snapshot(typename Base::database& db, uint32_t index)
             : root(db.get_root_owned(index)), point(root.snapshot_cursor()) {}
         typename Base::cursor create_cursor() const { return root.snapshot_cursor(); }
      };
   }

   template <class Base>
   struct byte_engine
   {
      using dwal_type = basic_dwal_database<std_lock_policy, Base>;
      struct config
      {
         typename Base::config base;
         dwal_config buffering;
      };
      class database;

      struct snapshot
      {
         std::shared_ptr<database> owner;
         std::shared_ptr<btree_layer> rw, ro;
         std::optional<btree_layer_view> rw_view, ro_view;
         std::optional<detail::byte_base_snapshot<Base>> base;

         snapshot(std::shared_ptr<database> db, uint32_t index, bool writer_owned = false)
             : owner(std::move(db))
         {
            auto& state = owner->dwal.root(index);
            std::unique_lock lock(state.tx_mutex, std::defer_lock);
            if (!writer_owned) lock.lock();
            rw = state.rw_layer;
            if (rw)
            {
               rw_view.emplace(*rw);
               rw->map.bump_cow_seq();
            }
            std::lock_guard buffered_lock(state.buffered_mutex);
            ro = state.buffered_ptr;
            if (ro) ro_view.emplace(*ro);
            base.emplace(*owner->dwal.underlying_db(), index);
         }

         template <class F>
         bool get(std::string_view key, F&& fn)
         {
            for (auto* layer : {&rw_view, &ro_view})
            {
               if (!*layer) continue;
               if (auto* value = (*layer)->map.get(key))
               {
                  if (!value->is_data()) return false;
                  fn(value->data);
                  return true;
               }
               if ((*layer)->tombstones.is_deleted(key)) return false;
            }
            return base->point.get(key, std::forward<F>(fn));
         }
      };

      // A cursor over the committed layers of one root.
      class layer_cursor
      {
         std::shared_ptr<snapshot> view_;
         std::optional<basic_merge_cursor<Base>> merge_;
         auto& ordered()
         {
            if (!merge_)
               merge_.emplace(view_->rw_view, view_->ro_view, view_->base->create_cursor());
            return *merge_;
         }
       public:
         explicit layer_cursor(std::shared_ptr<snapshot> view) : view_(std::move(view)) {}
         bool seek_begin() { return ordered().seek_begin(); }
         bool seek_last() { return ordered().seek_last(); }
         bool lower_bound(std::string_view k) { return ordered().lower_bound(k); }
         // The shim's base protocol defines upper_bound as the floor.
         bool upper_bound(std::string_view k)
         {
            if (lower_bound(k)) return key() == k || prev();
            return seek_last();
         }
         bool next() { return ordered().next(); }
         bool prev() { return ordered().prev(); }
         bool is_end() const { return !merge_ || merge_->is_end(); }
         bool is_rend() const { return !merge_ || merge_->is_rend(); }
         std::string_view key() const { return merge_->key(); }
         template <class F> bool get(std::string_view k, F&& fn)
         { return view_->get(k, std::forward<F>(fn)); }
         template <class T> std::optional<T> get(std::string_view k)
         {
            std::optional<T> result;
            get(k, [&](std::string_view v) { result.emplace(v); });
            return result;
         }
         template <class T> std::optional<T> value()
         {
            if (is_end() || is_rend() || merge_->is_subtree()) return {};
            if (merge_->current_source() == basic_merge_cursor<Base>::source::tri)
               return merge_->tri_cursor()->template value<T>();
            return T(merge_->current_value().data);
         }
         bool is_subtree() { return merge_ && !is_end() && !is_rend() && merge_->is_subtree(); }
         template <class F> bool get_value(F&& fn)
         {
            if (is_end() || is_rend() || merge_->is_subtree()) return false;
            if (merge_->current_source() == basic_merge_cursor<Base>::source::tri)
               return merge_->tri_cursor()->get_value(std::forward<F>(fn));
            fn(std::string_view(merge_->current_value().data));
            return true;
         }
      };

      // What a transaction's changes sit on: the root as committed when the
      // transaction started.
      struct committed_view
      {
         std::shared_ptr<snapshot> view;
         layer_cursor snapshot_cursor() const { return layer_cursor(view); }
      };
      using changes_type = psitri::transient_transaction<committed_view>;

      // The engine's cursor: over a root's committed layers, or over a
      // transaction's changes on top of them.
      class cursor
      {
         std::variant<layer_cursor, typename changes_type::cursor> impl_;
       public:
         explicit cursor(std::shared_ptr<snapshot> view) : impl_(layer_cursor(std::move(view))) {}
         explicit cursor(typename changes_type::cursor merged) : impl_(std::move(merged)) {}
         bool seek_begin() { return std::visit([](auto& c) { return bool(c.seek_begin()); }, impl_); }
         bool seek_last() { return std::visit([](auto& c) { return bool(c.seek_last()); }, impl_); }
         bool lower_bound(std::string_view k)
         { return std::visit([&](auto& c) { return bool(c.lower_bound(k)); }, impl_); }
         bool upper_bound(std::string_view k)
         { return std::visit([&](auto& c) { return bool(c.upper_bound(k)); }, impl_); }
         bool next() { return std::visit([](auto& c) { return bool(c.next()); }, impl_); }
         bool prev() { return std::visit([](auto& c) { return bool(c.prev()); }, impl_); }
         bool is_end() const { return std::visit([](auto& c) { return bool(c.is_end()); }, impl_); }
         bool is_rend() const { return std::visit([](auto& c) { return bool(c.is_rend()); }, impl_); }
         std::string_view key() const
         { return std::visit([](auto& c) { return std::string_view(c.key()); }, impl_); }
         bool is_subtree() { return std::visit([](auto& c) { return bool(c.is_subtree()); }, impl_); }
         template <class F> bool get(std::string_view k, F&& fn)
         { return std::visit([&](auto& c) { return bool(c.get(k, fn)); }, impl_); }
         template <class T> std::optional<T> get(std::string_view k)
         {
            std::optional<T> result;
            get(k, [&](std::string_view v) { result.emplace(v); });
            return result;
         }
         template <class F> bool get_value(F&& fn)
         { return std::visit([&](auto& c) { return bool(c.get_value(fn)); }, impl_); }
         template <class T> std::optional<T> value()
         { return std::visit([](auto& c) { return c.template value<T>(); }, impl_); }
      };

      class read_session
      {
         std::shared_ptr<database> db_;
       public:
         explicit read_session(std::shared_ptr<database> db) : db_(std::move(db)) {}
         // Each cursor is a snapshot of the root as committed when it is created.
         // (One snapshot used to be kept per root for the life of the session, so
         // later cursors never showed later commits.)
         cursor create_cursor(uint32_t index)
         {
            return cursor(std::make_shared<snapshot>(db_, index));
         }
      };

      // A transaction keeps its changes aside and writes them to the buffered
      // layer when it commits. It holds its root's writer for its whole life,
      // so writers queue, but it touches the shared layer only at commit, so a
      // reader is never held up behind an open transaction.
      class transaction
      {
         std::shared_ptr<database>    db_;
         uint32_t                     index_;
         std::unique_lock<std::mutex> writer_;
         changes_type                 changes_;
       public:
         transaction(std::shared_ptr<database> db, uint32_t index)
             : db_(std::move(db)),
               index_(index),
               writer_(db_->writer(index)),
               changes_(committed_view{std::make_shared<snapshot>(db_, index)})
         {
         }
         transaction(transaction&&) = default;
         ~transaction() = default;
         void upsert(std::string_view k, std::string_view v) { changes_.upsert(k, v); }
         bool remove(std::string_view k) { return changes_.remove(k) == 0; }
         template <class F> bool get(std::string_view k, F&& fn) const
         { return changes_.get(k, std::forward<F>(fn)); }
         template <class T> std::optional<T> get(std::string_view k) const
         {
            std::optional<T> result;
            get(k, [&](std::string_view v) { result.emplace(v); });
            return result;
         }
         cursor create_cursor() { return cursor(changes_.snapshot_cursor()); }
         cursor snapshot_cursor() { return create_cursor(); }
         void commit()
         {
            if (!writer_.owns_lock()) return;
            {
               auto tx = db_->dwal.start_write_transaction(index_);
               changes_.apply_to(tx);
               tx.commit(sal::sync_type::none);
            }
            changes_.reset();
            writer_.unlock();
         }
         void abort()
         {
            if (!writer_.owns_lock()) return;
            changes_.reset();
            writer_.unlock();
         }
      };

      class write_session
      {
         std::shared_ptr<database> db_;
       public:
         explicit write_session(std::shared_ptr<database> db) : db_(std::move(db)) {}
         transaction start_transaction(uint32_t index) { return transaction(db_, index); }
      };

      class database : public std::enable_shared_from_this<database>
      {
         std::mutex                    writers_mutex_;
         std::map<uint32_t, std::mutex> writers_;
       public:
         dwal_type dwal;
         // The writer lock of root `index`: one transaction at a time.
         std::mutex& writer(uint32_t index)
         {
            std::lock_guard guard(writers_mutex_);
            return writers_[index];
         }
         database(std::shared_ptr<typename Base::database> base,
                  const std::filesystem::path& path, const dwal_config& cfg)
             : dwal(std::move(base), path / "wal", cfg) {}
         static std::shared_ptr<database> open(const std::filesystem::path& path,
                                               sal::open_mode mode, const config& cfg)
         {
            auto base = Base::database::open(path, mode, cfg.base);
            return std::make_shared<database>(std::move(base), path, cfg.buffering);
         }
         std::shared_ptr<write_session> start_write_session()
         { return std::make_shared<write_session>(this->shared_from_this()); }
         std::shared_ptr<read_session> start_read_session()
         { return std::make_shared<read_session>(this->shared_from_this()); }
      };
   };
}

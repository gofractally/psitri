#pragma once
// heart's NATIVE psitri-shaped public API + the LOCKED adb typed-value layer.
// The same database → write_session → transaction → cursor structure psitri
// exposes; sal_art_ref::radix::database carries the base shape natively, and
// this header adds the registry, three-outcome typed reads, and any_value on
// top — so the one templated kv-compat body drives heart identically to the
// other engines.
//
// Storage: plain bytes go to the engine untouched. A typed value is stored
// through the engine's typed API (upsert_typed: a type number and a token,
// kept under the engine's own value kind), and the engine fires the type's
// retain/release hooks from its node lifecycle (value_hooks.hpp registry).
// This header only converts between a storable type and its token.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <sal/config.hpp>  // sal::sync_type

#include "radix_engine.hpp"
#include <sal_art_ref/value_hooks.hpp>

namespace sal_art_ref::kv
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   struct runtime_config
   {
      sal::sync_type sync_mode               = sal::sync_type::none;
      bool           write_protect_on_commit = false;
      bool           checksum_on_commit      = false;
   };

   using sal_art_ref::invalid_type;  // thrown by the engine's byte reads too

   template <class T>
   concept storable = requires(T t, const typename T::t_pod& p, void* ses) {
      { T::type_id } -> std::convertible_to<uint16_t>;
      requires std::is_trivially_copyable_v<typename T::t_pod>;
      { std::move(t).take_pod() } -> std::same_as<typename T::t_pod>;
      { T::make_from_pod(p, true) } -> std::same_as<T>;
      T::t_pod::retain(ses, p);
      T::t_pod::release(ses, p);
   };

   // What the engine hands back for a value of any kind: its type number (0 =
   // plain bytes) and the bytes or the token.
   struct slot
   {
      uint16_t    type = kRawType;
      std::string payload;
   };
   // Read through any native reader with get_any(key, fn(typenum, bytes)).
   template <class Reader>
   std::optional<slot> read_slot(Reader&& r, key_view k)
   {
      std::optional<slot> out;
      r.get_any(k, [&](uint16_t t, value_view v) { out = slot{t, std::string(v)}; });
      return out;
   }
   template <class S>
   S slot_decode(const slot& s)
   {
      if constexpr (storable<S>)
      {
         if (s.type != S::type_id)
            throw invalid_type{s.type, S::type_id};
         typename S::t_pod pod;
         if (s.payload.size() != sizeof(pod))
            throw invalid_type{s.type, S::type_id};
         std::memcpy(&pod, s.payload.data(), sizeof(pod));
         return S::make_from_pod(pod, true);  // a RETAINED handle out
      }
      else
      {
         if (s.type != kRawType)
            throw invalid_type{s.type, kRawType};
         return S(s.payload.data(), s.payload.size());
      }
   }
   // Store a typed value: take_pod() CONSUMES v — custody moves into the database.
   template <class Writer, class T>
   void store_typed(Writer&& w, key_view k, T&& v)
   {
      using U                     = std::remove_cvref_t<T>;
      const typename U::t_pod pod = std::move(v).take_pod();
      w.upsert_typed(k, uint16_t(U::type_id),
                     value_view(reinterpret_cast<const char*>(&pod), sizeof(pod)));
   }

   // any_value — heart backs it with an OWNED copy (the leaf_page may
   // be relocated by sal, so a raw slot pointer is unsafe past the read lock).
   // The interface matches every other engine; as<T> retains, bytes() on a
   // typed slot is invalid_type, raw_pod() is the expert/merger path.
   class any_value
   {
     public:
      any_value() = default;
      uint16_t type() const noexcept { return _has ? _slot.type : kRawType; }
      template <class T>
      bool is() const noexcept
      {
         return _has && _slot.type == T::type_id;
      }
      template <class T>
      T as() const
      {
         if (!is<T>())
            throw invalid_type{type(), T::type_id};
         return slot_decode<T>(_slot);
      }
      std::string_view bytes() const
      {
         if (type() != kRawType)
            throw invalid_type{type(), kRawType};
         return _slot.payload;
      }
      std::string_view raw_pod() const noexcept
      {
         return _has ? std::string_view{_slot.payload} : std::string_view{};
      }
      explicit operator bool() const noexcept { return _has; }

     private:
      friend class transaction;
      explicit any_value(slot s) : _slot(std::move(s)), _has(true) {}
      slot _slot;
      bool _has = false;
   };

   // Typed cursor: delegates positioning to the native cursor; value reads are
   // three-outcome. Used by the main CRUD battery.
   class cursor
   {
     public:
      explicit cursor(sal_art_ref::radix::radix_tree::cursor&& c) : _c(std::move(c)) {}

      bool seek_begin() { return _c.seek_begin(); }
      bool lower_bound(key_view k) { return _c.lower_bound(k); }
      bool upper_bound(key_view k) { return _c.upper_bound(k); }  // floor: largest v <= k
      void find(key_view k) { _c.find(k); }
      bool next() { return _c.next(); }
      bool is_end() const { return _c.is_end(); }
      key_view key() const { return _c.key(); }

      template <class S = std::string>
      std::optional<S> value() const
      {
         std::optional<slot> e;
         _c.value_any([&](uint16_t t, value_view v) { e = slot{t, std::string(v)}; });
         if (!e)
            return std::nullopt;
         return slot_decode<S>(*e);
      }
      template <class F>
      bool get_value(F&& f) const
      {
         return _c.get_value(std::forward<F>(f));  // a typed value throws invalid_type
      }
      uint16_t value_type() const { return _c.value_typenum(); }
      template <class T>
      bool is_type() const
      {
         return _c.value_typenum() == T::type_id;
      }

      template <class S = std::string>
      std::optional<S> get(key_view k)
      {
         auto e = read_slot(_c, k);
         if (!e)
            return std::nullopt;
         return slot_decode<S>(*e);
      }
      template <class F>
      bool get(key_view k, F&& f)
      {
         return _c.get(k, std::forward<F>(f));  // a typed value throws invalid_type
      }

     private:
      mutable sal_art_ref::radix::radix_tree::cursor _c;
   };

   // A tree IS a snapshot: an owned, frozen handle to a committed root (the
   // native owning tree holds a smart_ptr ref, so COW keeps the version alive
   // while this exists). It is the single owned-tree type — what get_root
   // returns, what set_root exchanges, and what typed reads go through.
   // Default = empty.
   class tree
   {
     public:
      tree() = default;
      explicit tree(sal_art_ref::radix::radix_tree::tree t) : _t(std::move(t)) {}
      cursor snapshot_cursor() const { return cursor(_t.snapshot_cursor()); }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         auto e = read_slot(_t, k);
         if (!e)
            return std::nullopt;
         return slot_decode<S>(*e);
      }
      explicit operator bool() const noexcept { return bool(_t); }

     private:
      friend class root_handle;
      sal_art_ref::radix::radix_tree::tree _t;
   };

   // Targets one root inside a multi-root transaction (adb root_handle).
   // Non-owning: points at a native tree_handle held by the transaction.
   class root_handle
   {
     public:
      explicit root_handle(sal_art_ref::radix::radix_tree::tree_handle* h) : _h(h) {}
      void insert(key_view k, value_view v) { _h->upsert(k, v); }
      void upsert(key_view k, value_view v) { _h->upsert(k, v); }
      template <class T>
         requires storable<std::remove_cvref_t<T>>
      void upsert(key_view k, T&& v)
      {
         store_typed(*_h, k, std::forward<T>(v));
      }
      int remove(key_view k) { return _h->remove(k); }
      template <class T = std::string>
      std::optional<T> get(key_view k) const
      {
         auto e = read_slot(*_h, k);
         if (!e)
            return std::nullopt;
         return slot_decode<T>(*e);
      }
      // THE exchange: assign, return the displaced tree (adb set_root).
      tree set_root(tree t) { return tree(_h->set_root(std::move(t._t))); }

     private:
      sal_art_ref::radix::radix_tree::tree_handle* _h;
   };

   class transaction
   {
     public:
      explicit transaction(sal_art_ref::radix::radix_tree::transaction&& t) : _tx(std::move(t)) {}
      void insert(key_view k, value_view v) { _tx.upsert(k, v); }
      void upsert(key_view k, value_view v) { _tx.upsert(k, v); }
      // ── multi-root (adb): primary()/open_root() target additional roots ──
      root_handle primary() { return root_handle(&_tx.primary()); }
      root_handle open_root(uint32_t idx) { return root_handle(&_tx.open_root(idx)); }
      // typed store: take_pod() CONSUMES v — custody moves into the database.
      template <class T>
         requires storable<std::remove_cvref_t<T>>
      void upsert(key_view k, T&& v)
      {
         store_typed(_tx, k, std::forward<T>(v));  // custody -> db
      }
      int  remove(key_view k) { return _tx.remove(k); }
      void commit() { _tx.commit(); }
      void abort() noexcept { _tx.abort(); }

      template <class T = std::string>
      std::optional<T> get(key_view k)
      {
         auto e = read_slot(_tx, k);
         if (!e)
            return std::nullopt;
         return slot_decode<T>(*e);
      }
      template <class F>
      bool get(key_view k, F&& fn)
      {
         return _tx.get(k, std::forward<F>(fn));  // a typed value throws invalid_type
      }
      any_value get_any(key_view k)
      {
         auto e = read_slot(_tx, k);
         if (!e)
            return {};
         return any_value{std::move(*e)};
      }

     private:
      sal_art_ref::radix::radix_tree::transaction _tx;
   };

   class write_session
   {
     public:
      explicit write_session(std::shared_ptr<sal_art_ref::radix::write_session> ws)
          : _ws(std::move(ws))
      {
      }
      transaction start_transaction(uint32_t root_index)
      {
         return transaction(_ws->start_transaction(root_index));
      }
      // get_root(i) IS the snapshot primitive: an owned, frozen tree of root
      // i's committed state (the native owning tree holds a smart_ptr ref, so
      // COW keeps that version alive while the handle exists). Call it again
      // to observe later state.
      tree get_root(uint32_t root_index) { return tree(_ws->get_root(root_index)); }

     private:
      std::shared_ptr<sal_art_ref::radix::write_session> _ws;
   };

   // Reader-thread handle (psitri-shaped): hands out typed snapshot
   // cursors. CREATE ON THE READER THREAD (one SAL session per
   // thread). Reads run lock-free against the writer (native contract).
   class read_session
   {
     public:
      explicit read_session(std::shared_ptr<sal_art_ref::radix::read_session> rs)
          : _rs(std::move(rs))
      {
      }
      cursor create_cursor(uint32_t root_index) { return cursor(_rs->create_cursor(root_index)); }

     private:
      std::shared_ptr<sal_art_ref::radix::read_session> _rs;
   };

   class database
   {
     public:
      static std::shared_ptr<database> create(std::filesystem::path dir, runtime_config = {})
      {
         std::filesystem::create_directories(dir);
         return std::shared_ptr<database>(new database(sal_art_ref::radix::database::open(dir)));
      }
      std::shared_ptr<write_session> start_write_session()
      {
         return std::make_shared<write_session>(_db->start_write_session());
      }
      std::shared_ptr<read_session> start_read_session()
      {
         return std::make_shared<read_session>(_db->start_read_session());
      }

      // ── typed-value registry (locked adb API) ───────────────────────────
      // Populates the engine-global registry (value_hooks.hpp) the leaf_page
      // lifecycle consults; fires on the writer AND sal GC threads.
      template <class... Ts>
      void register_value_types()
      {
         (reg_type_<Ts>(), ...);
      }

     private:
      explicit database(std::shared_ptr<sal_art_ref::radix::database> db) : _db(std::move(db)) {}

      template <class T>
      static void reg_type_()
      {
         static_assert(storable<T>);
         static_assert(T::type_id != kRawType, "type_id 0 is the raw-bytes builtin");
         sal_art_ref::g_value_types[T::type_id] = {&retain_t_<T>, &release_t_<T>};
      }
      template <class T>
      static void retain_t_(void* ses, const char* p, size_t n)
      {
         typename T::t_pod pod;
         (void)n;
         std::memcpy(&pod, p, sizeof(pod));
         T::t_pod::retain(ses, pod);
      }
      template <class T>
      static void release_t_(void* ses, const char* p, size_t n)
      {
         typename T::t_pod pod;
         (void)n;
         std::memcpy(&pod, p, sizeof(pod));
         T::t_pod::release(ses, pod);
      }

      std::shared_ptr<sal_art_ref::radix::database> _db;
   };
}  // namespace sal_art_ref::kv

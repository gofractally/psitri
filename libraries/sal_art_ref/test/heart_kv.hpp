#pragma once
// heart's NATIVE psitri-shaped public API + the LOCKED adb typed-value layer.
// The same database → write_session → transaction → cursor structure psitri
// exposes; sal_art_ref::radix::database carries the base shape natively, and
// this header adds the [u16 typenum][payload] value envelope, the registry,
// three-outcome typed reads, and any_value on top — so the one templated
// kv-compat body drives heart identically to the other engines.
//
// Storage: every value heart stores is the envelope (typenum 0 = raw bytes).
// The engine fires the user retain/release hooks from leaf_page's lifecycle
// (value_hooks.hpp registry) keyed by the embedded typenum; this header only
// builds the envelope on write and strips it on read.
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

   struct invalid_type : std::runtime_error
   {
      invalid_type(uint16_t stored, uint16_t asked)
          : std::runtime_error("sal_art_ref: value is type " + std::to_string(stored) +
                               ", read as type " + std::to_string(asked))
      {
      }
   };

   template <class T>
   concept storable = requires(T t, const typename T::t_pod& p, void* ses) {
      { T::type_id } -> std::convertible_to<uint16_t>;
      requires std::is_trivially_copyable_v<typename T::t_pod>;
      { std::move(t).take_pod() } -> std::same_as<typename T::t_pod>;
      { T::make_from_pod(p, true) } -> std::same_as<T>;
      T::t_pod::retain(ses, p);
      T::t_pod::release(ses, p);
   };

   inline uint16_t slot_type(value_view s) noexcept { return sal_art_ref::env_typenum(s); }
   inline value_view slot_payload(value_view s) noexcept { return s.substr(2); }
   inline value_view env_raw_(value_view v)  // [0][bytes], thread_local scratch
   {
      thread_local std::string buf;
      buf.resize(2 + v.size());
      buf[0] = buf[1] = 0;
      std::memcpy(buf.data() + 2, v.data(), v.size());
      return {buf.data(), buf.size()};
   }
   template <class S>
   S slot_decode(value_view slot)
   {
      const uint16_t t = slot_type(slot);
      if constexpr (storable<S>)
      {
         if (t != S::type_id)
            throw invalid_type{t, S::type_id};
         typename S::t_pod pod;
         if (slot.size() - 2 != sizeof(pod))
            throw invalid_type{t, S::type_id};
         std::memcpy(&pod, slot.data() + 2, sizeof(pod));
         return S::make_from_pod(pod, true);  // a RETAINED handle out
      }
      else
      {
         if (t != kRawType)
            throw invalid_type{t, kRawType};
         const value_view p = slot_payload(slot);
         return S(p.data(), p.size());
      }
   }

   // any_value — heart backs it with an OWNED envelope copy (the leaf_page may
   // be relocated by sal, so a raw slot pointer is unsafe past the read lock).
   // The interface matches every other engine; as<T> retains, bytes() on a
   // typed slot is invalid_type, raw_pod() is the expert/merger path.
   class any_value
   {
     public:
      any_value() = default;
      uint16_t type() const noexcept { return _has ? slot_type(_env) : kRawType; }
      template <class T>
      bool is() const noexcept
      {
         return _has && slot_type(_env) == T::type_id;
      }
      template <class T>
      T as() const
      {
         if (!is<T>())
            throw invalid_type{type(), T::type_id};
         typename T::t_pod pod;
         std::memcpy(&pod, _env.data() + 2, sizeof(pod));
         return T::make_from_pod(pod, true);
      }
      std::string_view bytes() const
      {
         if (type() != kRawType)
            throw invalid_type{type(), kRawType};
         return {_env.data() + 2, _env.size() - 2};
      }
      std::string_view raw_pod() const noexcept
      {
         return _has ? std::string_view{_env.data() + 2, _env.size() - 2} : std::string_view{};
      }
      explicit operator bool() const noexcept { return _has; }

     private:
      friend class transaction;
      explicit any_value(std::string env) : _env(std::move(env)), _has(true) {}
      std::string _env;
      bool        _has = false;
   };

   // Stripping cursor: delegates positioning to the native cursor; value reads
   // peel the envelope (three-outcome). Used by the main CRUD battery.
   class cursor
   {
     public:
      explicit cursor(sal_art_ref::radix::radix_tree::cursor&& c) : _c(std::move(c)) {}

      bool seek_begin() { return _c.seek_begin(); }
      bool lower_bound(key_view k) { return _c.lower_bound(k); }
      void upper_bound(key_view k) { _c.upper_bound(k); }
      void find(key_view k) { _c.find(k); }
      bool next() { return _c.next(); }
      bool is_end() const { return _c.is_end(); }
      key_view key() const { return _c.key(); }

      template <class S = std::string>
      std::optional<S> value() const
      {
         auto e = _c.template value<std::string>();
         if (!e)
            return std::nullopt;
         return slot_decode<S>(*e);
      }
      template <class F>
      bool get_value(F&& f) const
      {
         auto e = _c.template value<std::string>();
         if (!e)
            return false;
         if (slot_type(*e) != kRawType)
            throw invalid_type{slot_type(*e), kRawType};
         f(slot_payload(*e));
         return true;
      }
      uint16_t value_type() const
      {
         auto e = _c.template value<std::string>();
         return e ? slot_type(*e) : kRawType;
      }
      template <class T>
      bool is_type() const
      {
         auto e = _c.template value<std::string>();
         return e && slot_type(*e) == T::type_id;
      }

      template <class S = std::string>
      std::optional<S> get(key_view k)
      {
         auto e = _c.template get<std::string>(k);
         if (!e)
            return std::nullopt;
         return slot_decode<S>(*e);
      }
      template <class F>
      bool get(key_view k, F&& f)
      {
         auto e = _c.template get<std::string>(k);
         if (!e)
            return false;
         if (slot_type(*e) != kRawType)
            throw invalid_type{slot_type(*e), kRawType};
         f(slot_payload(*e));
         return true;
      }

     private:
      mutable sal_art_ref::radix::radix_tree::cursor _c;
   };

   // A tree IS a snapshot: an owned, frozen handle to a committed root (the
   // native owning tree holds a smart_ptr ref, so COW keeps the version alive
   // while this exists). It is the single owned-tree type — what get_root
   // returns, what set_root exchanges, and what reads strip the envelope
   // through. Default = empty.
   class tree
   {
     public:
      tree() = default;
      explicit tree(sal_art_ref::radix::radix_tree::tree t) : _t(std::move(t)) {}
      cursor snapshot_cursor() const { return cursor(_t.snapshot_cursor()); }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         auto e = _t.template get<std::string>(k);
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
      void insert(key_view k, value_view v) { _h->upsert(k, env_raw_(v)); }
      void upsert(key_view k, value_view v) { _h->upsert(k, env_raw_(v)); }
      template <class T>
         requires storable<std::remove_cvref_t<T>>
      void upsert(key_view k, T&& v)
      {
         using U                     = std::remove_cvref_t<T>;
         const typename U::t_pod pod = std::move(v).take_pod();
         char                    buf[2 + sizeof(pod)];
         const uint16_t          id = U::type_id;
         std::memcpy(buf, &id, 2);
         std::memcpy(buf + 2, &pod, sizeof(pod));
         _h->upsert(k, value_view(buf, sizeof(buf)));
      }
      int remove(key_view k) { return _h->remove(k); }
      template <class T = std::string>
      std::optional<T> get(key_view k) const
      {
         auto e = _h->template get<std::string>(k);
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
      void insert(key_view k, value_view v) { _tx.upsert(k, env_raw_(v)); }
      void upsert(key_view k, value_view v) { _tx.upsert(k, env_raw_(v)); }
      // ── multi-root (adb): primary()/open_root() target additional roots ──
      root_handle primary() { return root_handle(&_tx.primary()); }
      root_handle open_root(uint32_t idx) { return root_handle(&_tx.open_root(idx)); }
      // typed store: take_pod() CONSUMES v — custody moves into the database.
      template <class T>
         requires storable<std::remove_cvref_t<T>>
      void upsert(key_view k, T&& v)
      {
         using U                     = std::remove_cvref_t<T>;
         const typename U::t_pod pod = std::move(v).take_pod();  // custody -> db
         char                    buf[2 + sizeof(pod)];
         const uint16_t          id = U::type_id;
         std::memcpy(buf, &id, 2);
         std::memcpy(buf + 2, &pod, sizeof(pod));
         _tx.upsert(k, value_view(buf, sizeof(buf)));
      }
      int  remove(key_view k) { return _tx.remove(k); }
      void commit() { _tx.commit(); }
      void abort() noexcept { _tx.abort(); }

      template <class T = std::string>
      std::optional<T> get(key_view k)
      {
         auto e = _tx.template get<std::string>(k);
         if (!e)
            return std::nullopt;
         return slot_decode<T>(*e);
      }
      template <class F>
      bool get(key_view k, F&& fn)
      {
         auto e = _tx.template get<std::string>(k);
         if (!e)
            return false;
         if (slot_type(*e) != kRawType)
            throw invalid_type{slot_type(*e), kRawType};
         fn(slot_payload(*e));
         return true;
      }
      any_value get_any(key_view k)
      {
         auto e = _tx.template get<std::string>(k);
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

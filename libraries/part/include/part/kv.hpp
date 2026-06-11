// part::kv — part's NATIVE psitri-shaped public API (the same database →
// write_session → transaction → cursor structure psitri and heart::kv expose),
// so one templated test/bench body drives all three engines through identical
// calls. See heart/test/heart_kv.hpp for the canonical shape this mirrors and
// programs/kv_compat_test.cpp for the conformance driver.
//
// Engine: the verbatim-hkv tree on the part arena (part/hkv/tree.hpp +
// part_alloc), with the part COW extension supplying snapshot semantics:
// commit retains the new root for the database's live-root slot and releases
// the old one; snapshot cursors pin the live root (the 191..254 refcount band)
// and everything reachable under the pin is immutable by the COW discipline
// (SPEC §7), so cursor traversal needs no further refcounting. Values are POD
// inline byte strings (<= 126 B): COW clones are memcpys and every value byte
// lives in the arena.
#pragma once
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <part/hkv/tree.hpp>
#include <part/tree_alloc.hpp>

namespace part::kv
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   // Mirrors psitri::runtime_config's relevant fields (same sal types).
   struct runtime_config
   {
      sal::sync_type sync_mode               = sal::sync_type::none;
      bool           write_protect_on_commit = false;
      bool           checksum_on_commit      = false;
      uint32_t       max_segments            = 4096;  // x 8MB arena ceiling (32 GB default)
   };

   // Inline POD value: len-prefixed bytes, trivially copyable (COW memcpy).
   // 80 B total keeps a typical leaf (4 B header + value + short suffix) inside
   // ONE 128 B line — at 128 B every 8-byte bank balance burned a 2-line leaf,
   // which alone doubled resident memory. Variable-length leaf values (design5
   // val_len) are the real fix and live in the tree layer's future.
   struct pval
   {
      uint16_t len = 0;
      char     b[78];
      static pval from(value_view v)
      {
         if (v.size() > sizeof(b))
            throw std::length_error("part::kv value exceeds 78 bytes");
         pval p;
         p.len = uint16_t(v.size());
         std::memcpy(p.b, v.data(), v.size());
         return p;
      }
      value_view view() const noexcept { return value_view(b, len); }
      // composed-bucket bridge: cells store live bytes, T converts at the edge
      static pval       from_bytes(value_view v) { return from(v); }
      value_view        to_bytes() const noexcept { return view(); }
   };
   static_assert(std::is_trivially_copyable_v<pval> && sizeof(pval) == 80);

   using engine_t = hkv::kv_tree<std::string_view, pval, hkv::mode::none, part_alloc<pval>>;
   using root_ptr = part::detail::line_ptr;

   class database;

   // Snapshot cursor: a pinned root + a non-owning view tree + an hkv iterator.
   // The dtor drops the pin; if that was the generation's last reference the
   // old version drains back to the arena.
   class cursor
   {
     public:
      cursor(cursor&& o) noexcept
          : _a(o._a), _hp(o._hp), _root(o._root), _view(std::move(o._view)), _it(_view.end())
      {
         o._a  = nullptr;  // moved-from cursor owns no pin; position resets
         o._hp = nullptr;
      }
      cursor(const cursor&) = delete;
      ~cursor()
      {
         if (!_a)
            return;
         const root_ptr r = _root;
         _view.adopt_root(root_ptr::null());  // the view owns nothing
         if (_hp)
            _hp->store(0, std::memory_order_seq_cst);  // readers only clear their slot:
         else if (!r.is_null())                        // reclamation is the writer's job
            _view.cow_release(r);  // write-side pinned cursor (scan_all path)
      }

      void seek_begin() { _it = _view.begin(); }
      void lower_bound(key_view k) { _it = _view.lower_bound(k); }
      void upper_bound(key_view k)
      {
         _it = _view.lower_bound(k);
         if (_it != _view.end() && (*_it).first == k)
            ++_it;
      }
      void find(key_view k)
      {
         if constexpr (engine_t::cb_enabled)
         {  // pages: position at lower_bound and demand an exact key match
            _it = _view.lower_bound(k);
            if (_it != _view.end() && (*_it).first != k)
               _it = _view.end();
         }
         else
            _it = _view.find(k);
      }
      void next() { ++_it; }
      bool is_end() const { return _it == _view.end(); }

      key_view key() const { return (*_it).first; }

      template <class S = std::string>
      std::optional<S> value() const
      {
         if (is_end())
            return std::nullopt;
         const value_view v = (*_it).second.view();
         return S(v.data(), v.size());
      }
      template <class F>
      bool get_value(F&& f) const
      {
         if (is_end())
            return false;
         f((*_it).second.view());
         return true;
      }

      // Point read against the snapshot (the bank reader hot path).
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         pval out;
         const bool hit = engine_t::cb_enabled ? _view.pg_find(k, out) : _view.find(k, out);
         if (!hit)
            return std::nullopt;
         return S(out.b, out.len);
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         pval out;
         const bool hit = engine_t::cb_enabled ? _view.pg_find(k, out) : _view.find(k, out);
         if (!hit)
            return false;
         f(out.view());
         return true;
      }

     private:
      friend class database;
      friend class read_session;
      cursor(part::arena* a, root_ptr root, std::atomic<uint64_t>* hp = nullptr)
          : _a(a), _hp(hp), _root(root), _view(part_alloc<pval>(a)), _it(_view.end())
      {
         _view.adopt_root(_root);
      }

      part::arena*             _a;
      std::atomic<uint64_t>*   _hp;  // sal paradigm: the reader's published slot
      root_ptr                 _root;
      mutable engine_t         _view;
      engine_t::const_iterator _it;
   };

   // A reader thread's handle: one part::read_session slot for the lifetime of
   // the thread; cursors pin the live root per batch.
   class read_session
   {
     public:
      cursor create_cursor(uint32_t root_index);

     private:
      friend class database;
      explicit read_session(database* db);
      database*          _db;
      part::read_session _rs;
   };

   class write_session;

   class transaction
   {
     public:
      void insert(key_view k, value_view v) { upsert(k, v); }
      void upsert(key_view k, value_view v);
      int  remove(key_view k);
      void commit();
      void abort() noexcept {}

      template <class F>
      bool get(key_view k, F&& fn) const;
      template <class S>
      std::optional<S> get(key_view k) const;

     private:
      friend class write_session;
      transaction(write_session* ws, part::transaction&& tx) : _ws(ws), _tx(std::move(tx)) {}
      write_session*    _ws;
      part::transaction _tx;
   };

   // Root handle for read-only access through the write session (kv_compat's
   // get_root(i).snapshot_cursor()).
   class tree
   {
     public:
      explicit tree(write_session* ws) noexcept : _ws(ws) {}
      cursor   snapshot_cursor() const;
      explicit operator bool() const noexcept { return _ws != nullptr; }

     private:
      write_session* _ws;
   };

   class write_session
   {
     public:
      transaction start_transaction(uint32_t root_index)
      {
         (void)root_index;  // single root in v1, like heart_kv
         return transaction(this, _ses.start_transaction(0));
      }
      tree get_root(uint32_t root_index)
      {
         (void)root_index;
         return tree(this);
      }

      // Drop the working tree's reference the COW way: nodes shared with the
      // live-root slot or reader pins survive; hkv's unconditional teardown
      // must never see them.
      ~write_session()
      {
         _t.cow_release(_t.root_handle());
         _t.adopt_root(root_ptr::null());
      }

      engine_t& tree_dbg() noexcept { return _t; }  // debug/tools only

     private:
      friend class database;
      friend class transaction;
      friend class tree;
      explicit write_session(database* db);

      database*     _db;
      part::session _ses;
      engine_t      _t;  // the working tree (continues across transactions)
   };

   class database
   {
     public:
      static std::shared_ptr<database> create(std::filesystem::path dir, runtime_config cfg = {})
      {
         return std::shared_ptr<database>(new database(std::move(dir), cfg));
      }
      static std::shared_ptr<database> open(std::filesystem::path dir, runtime_config cfg = {})
      {
         return create(std::move(dir), cfg);
      }

      std::shared_ptr<write_session> start_write_session()
      {
         return std::shared_ptr<write_session>(new write_session(this));
      }
      std::shared_ptr<read_session> start_read_session()
      {
         return std::shared_ptr<read_session>(new read_session(this));
      }

      void sync(sal::sync_type = sal::sync_type::none) {}  // commits sync per cfg already

      ~database()
      {
         // No readers remain by contract: force-drain retired roots, then the
         // live root's slot reference.
         for (uint64_t raw : _retired)
         {
            root_ptr       p;
            const uint32_t r32 = uint32_t(raw);
            std::memcpy(p.b, &r32, 4);
            engine_t v{part_alloc<pval>(&_arena)};
            v.cow_release(p);
         }
         if (!_live_root.is_null())
         {
            engine_t v{part_alloc<pval>(&_arena)};
            v.cow_release(_live_root);
         }
      }

      part::arena& store() noexcept { return _arena; }
      // debug/tools only: the raw live root (no pin — caller must hold off writers)
      root_ptr     live_root_dbg() const noexcept { return _live_root; }

      // Atomic live-root handoff: commit swaps under the mutex; snapshot
      // acquisition pins under the same mutex, so a root can never be released
      // between a reader's load and its pin.
      cursor pin_snapshot()
      {
         std::lock_guard g(_root_mx);
         const root_ptr  r = _live_root;
         if (!r.is_null())
            _arena.add_ref_or_throw(uint32_t(r.raw() >> 4));
         return cursor(&_arena, r);
      }

     private:
      friend class write_session;
      friend class read_session;
      friend class transaction;

      database(std::filesystem::path dir, runtime_config cfg)
          : _arena(make_cfg_(std::move(dir), cfg)), _live_root(root_ptr::null())
      {
      }
      static part::arena_config make_cfg_(std::filesystem::path dir, runtime_config cfg)
      {
         part::arena_config c;
         c.dir          = std::move(dir);
         c.sync_mode    = cfg.sync_mode;
         c.max_segments = cfg.max_segments;
         // COW churn frees whole generations but long-lived survivors pepper
         // old segments; demanding 50%-in-8-contig before reuse lets the file
         // grow without bound. 1/8 keeps the can-this-segment-serve-allocs
         // guarantee while reclaiming aggressively.
         c.reuse_threshold = 0.125;
         return c;
      }

      // Retired-root release is DEFERRED to the writer: a root leaves the
      // queue only when no reader slot publishes it (sal's reader paradigm —
      // readers never touch refcounts or locks).
      void drain_retired_()  // caller holds _root_mx
      {
         for (size_t i = 0; i < _retired.size();)
         {
            if (_arena.root_published(_retired[i]))
            {
               ++i;
               continue;
            }
            root_ptr p;
            const uint32_t r32 = uint32_t(_retired[i]);
            std::memcpy(p.b, &r32, 4);
            engine_t v{part_alloc<pval>(&_arena)};
            v.cow_release(p);
            _retired[i] = _retired.back();
            _retired.pop_back();
         }
      }

      part::arena           _arena;
      std::mutex            _root_mx;
      root_ptr              _live_root;
      std::atomic<uint64_t> _live_raw{0};
      std::vector<uint64_t> _retired;
   };

   // ── bodies ────────────────────────────────────────────────────────────────
   inline write_session::write_session(database* db)
       : _db(db), _ses(db->_arena.start_session()), _t(part_alloc<pval>(&_ses))
   {
   }

   inline read_session::read_session(database* db)
       : _db(db), _rs(db->_arena.start_read_session())
   {
   }

   inline cursor read_session::create_cursor(uint32_t)
   {
      auto& slot = _db->_arena.reader_slot(_rs.slot());
      for (;;)
      {
         const uint64_t r = _db->_live_raw.load(std::memory_order_seq_cst);
         slot.store(r, std::memory_order_seq_cst);  // publish FIRST...
         if (_db->_live_raw.load(std::memory_order_seq_cst) == r)
         {  // ...then validate: r was live after our publish, so the writer's
            // drain (which runs after replacing the root) must see our slot.
            root_ptr p = root_ptr::null();
            if (r)
            {
               const uint32_t r32 = uint32_t(r);
               std::memcpy(p.b, &r32, 4);
            }
            return cursor(&_db->_arena, p, &slot);
         }
      }
   }

   inline cursor tree::snapshot_cursor() const
   {
      return _ws->_db->pin_snapshot();
   }

   inline void transaction::upsert(key_view k, value_view v)
   {
      if constexpr (engine_t::cb_enabled)
         _ws->_t.pg_upsert(k, pval::from(v));  // diver rule: keys live in pages
      else
         _ws->_t.cow_insert(k, pval::from(v));
   }
   inline int transaction::remove(key_view k)
   {
      if constexpr (engine_t::cb_enabled)
         return _ws->_t.pg_remove(k) ? 0 : -1;
      else
         return _ws->_t.cow_remove(k) ? 0 : -1;
   }
   template <class F>
   inline bool transaction::get(key_view k, F&& fn) const
   {
      pval out;
      const bool hit = engine_t::cb_enabled ? _ws->_t.pg_find(k, out) : _ws->_t.find(k, out);
      if (!hit)
         return false;
      fn(out.view());
      return true;
   }
   template <class S>
   inline std::optional<S> transaction::get(key_view k) const
   {
      pval out;
      const bool hit = engine_t::cb_enabled ? _ws->_t.pg_find(k, out) : _ws->_t.find(k, out);
      if (!hit)
         return std::nullopt;
      return S(out.b, out.len);
   }

   inline void transaction::commit()
   {
      _tx.commit();  // data msync + floor advance (layer 1)
      database* db = _ws->_db;
      root_ptr  nr = _ws->_t.root_handle();
      std::lock_guard g(db->_root_mx);
      const root_ptr  old = db->_live_root;
      // UNCONDITIONAL handoff: a raw-equality skip is unsound under line
      // recycling — the new root GENERATION can land on the very line the
      // retired root freed (fcache ABA), and skipping then strands _live's
      // reference on a dead subtree. retain-new then retire-old is a net
      // no-op for a genuinely unchanged root and correct for the collision.
      if (!nr.is_null() && !_ws->_t.cow_retain(nr))
         throw std::runtime_error("part::kv: root retain refused");
      db->_live_root = nr;
      db->_live_raw.store(nr.is_null() ? 0 : nr.raw(), std::memory_order_seq_cst);
      db->_arena.set_root(0, nr.is_null() ? 0 : nr.raw());
      if (!old.is_null())
         db->_retired.push_back(old.raw());  // strictly after the root moved on
      db->drain_retired_();  // all reclamation on the writer thread (SPEC §7)
   }
}  // namespace part::kv

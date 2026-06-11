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
#include <chrono>
#include <condition_variable>
#include <thread>

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
      // SPEC §19: when false, no releaser thread is spawned and the defer hook
      // is never registered — every release runs synchronously on its caller
      // (identical semantics, for deployments that cannot tolerate a thread).
      bool           background_release      = true;
   };

   // Inline POD value: len-prefixed bytes, trivially copyable (COW memcpy).
   // 80 B total keeps a typical leaf (4 B header + value + short suffix) inside
   // ONE 128 B line — at 128 B every 8-byte bank balance burned a 2-line leaf,
   // which alone doubled resident memory. Variable-length leaf values (design5
   // val_len) are the real fix and live in the tree layer's future.
   // (pval[80] retired: values are raw bytes in the leaf — hkv::bytes_value)

   // ELEGANT value mode: the leaf IS the value (raw bytes sized to the data);
   // reads hand back views — pval[80] retired.
   using bval     = hkv::bytes_value;
   using engine_t = hkv::kv_tree<std::string_view, bval, hkv::mode::none, part_alloc<bval>>;
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
         if constexpr (engine_t::cb_create)
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
         const value_view v = (*_it).second.v;
         return S(v.data(), v.size());
      }
      template <class F>
      bool get_value(F&& f) const
      {
         if (is_end())
            return false;
         f((*_it).second.v);
         return true;
      }

      // Point read against the snapshot (the bank reader hot path).
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if constexpr (engine_t::cb_create)
         {
            std::string_view out;
            if (!_view.pg_find(k, out))
               return std::nullopt;
            return S(out.data(), out.size());
         }
         bval out;
         if (!_view.find(k, out))
            return std::nullopt;
         return S(out.v.data(), out.v.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         if constexpr (engine_t::cb_create)
         {
            std::string_view out;
            if (!_view.pg_find(k, out))
               return false;
            f(out);
            return true;
         }
         bval out;
         if (!_view.find(k, out))
            return false;
         f(out.v);
         return true;
      }

     private:
      friend class database;
      friend class read_session;
      cursor(part::arena* a, root_ptr root, std::atomic<uint64_t>* hp = nullptr)
          : _a(a), _hp(hp), _root(root), _view(part_alloc<bval>(a)), _it(_view.end())
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
      // CONTRACT (kv-compat 4.7b): dropping or aborting an uncommitted
      // transaction ROLLS BACK — the session tree returns to the txn-start
      // root and the diverged generation is released (shared nodes keep the
      // committed generation's references).
      void abort() noexcept;
      ~transaction() { abort(); }
      transaction(transaction&& o) noexcept
          : _ws(o._ws), _tx(std::move(o._tx)), _start_root(o._start_root),
            _start_count(o._start_count), _done(o._done)
      {
         o._done = true;  // moved-from: nothing to roll back
      }
      transaction(const transaction&)            = delete;
      transaction& operator=(const transaction&) = delete;

      template <class F>
      bool get(key_view k, F&& fn) const;
      template <class S>
      std::optional<S> get(key_view k) const;

     private:
      friend class write_session;
      transaction(write_session* ws, part::transaction&& tx);
      write_session*    _ws;
      part::transaction _tx;
      root_ptr          _start_root;
      size_t            _start_count = 0;
      bool              _done        = false;
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
         // clean shutdown (SPEC §19): ring drained, releaser stopped and
         // JOINED before any member teardown; the rest runs foreground/exact
         flush_releases();
         _rel_stop.store(true, std::memory_order_release);
         _rel_cv.notify_one();
         if (_rel_thread.joinable())
            _rel_thread.join();
         _arena.set_defer(nullptr, nullptr);  // remaining releases are inline
         // No readers remain by contract: force-drain retired roots, then the
         // live root's slot reference.
         for (uint64_t raw : _retired)
         {
            root_ptr       p;
            const uint32_t r32 = uint32_t(raw);
            std::memcpy(p.b, &r32, 4);
            engine_t v{part_alloc<bval>(&_arena)};
            v.cow_release(p);
         }
         if (!_live_root.is_null())
         {
            engine_t v{part_alloc<bval>(&_arena)};
            v.cow_release(_live_root);
         }
      }

      part::arena& store() noexcept { return _arena; }
      // SPEC §19: wait until the background releaser has emptied its ring
      // (drain gates and shutdown ordering call this)
      void flush_releases() noexcept
      {
         while (_rel_tail.load(std::memory_order_acquire) !=
                _rel_head.load(std::memory_order_acquire))
            std::this_thread::yield();
      }
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
         // background releaser (SPEC §19): the writer pushes release-to-zero
         // subtree roots; this thread runs the cascades; a full ring makes the
         // writer release inline (graceful degradation to the old behavior).
         // OPTIONAL: without the thread the hook is never registered and the
         // queue is permanently "blocked" — everything runs synchronously.
         if (cfg.background_release)
         {
            _arena.set_defer(&defer_cb_, this);
            _rel_thread = std::thread([this] { rel_loop_(); });
         }
      }
      static bool defer_cb_(void* ctx, uint32_t raw) noexcept
      {
         return static_cast<database*>(ctx)->rel_push_(raw);
      }
      bool rel_push_(uint32_t raw) noexcept
      {
         const uint32_t h = _rel_head.load(std::memory_order_relaxed);
         const uint32_t t = _rel_tail.load(std::memory_order_acquire);
         if (h - t >= kRelRing)
            return false;  // full: the caller releases in the foreground
         _rel[h % kRelRing].store(raw, std::memory_order_relaxed);
         _rel_head.store(h + 1, std::memory_order_release);
         if (h == t)
            _rel_cv.notify_one();  // was empty: wake the releaser
         return true;
      }
      void rel_loop_() noexcept
      {
         engine_t v{part_alloc<bval>(&_arena, /*defer=*/false)};  // inline worker
         for (;;)
         {
            const uint32_t t = _rel_tail.load(std::memory_order_relaxed);
            if (t == _rel_head.load(std::memory_order_acquire))
            {
               if (_rel_stop.load(std::memory_order_acquire))
                  return;
               std::unique_lock lk(_rel_mx);
               _rel_cv.wait_for(lk, std::chrono::milliseconds(2));
               continue;
            }
            const uint32_t raw = uint32_t(_rel[t % kRelRing].load(std::memory_order_relaxed));
            root_ptr       p;
            std::memcpy(p.b, &raw, 4);
            v.cow_release(p);  // non-deferring allocator: full cascade here
            _rel_tail.store(t + 1, std::memory_order_release);
         }
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
         const bool tm = getenv("PART_DRAIN_TIME") != nullptr;
         const auto t0 = tm ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
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
            engine_t v{part_alloc<bval>(&_arena, /*defer=*/true)};
            v.cow_release(p);  // big generations route to the releaser
            _retired[i] = _retired.back();
            _retired.pop_back();
            ++_drain_roots;
         }
         if (tm)
            _drain_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
      }

     public:
      uint64_t drain_ns_dbg() const noexcept { return _drain_ns; }
      uint64_t drain_roots_dbg() const noexcept { return _drain_roots; }

     private:
      uint64_t _drain_ns    = 0;
      uint64_t _drain_roots = 0;

      private:
      static constexpr uint32_t kRelRing = 4096;  // big cascades shard into many subtree chunks
      std::atomic<uint32_t>     _rel[kRelRing] = {};
      std::atomic<uint32_t>     _rel_head{0}, _rel_tail{0};
      std::mutex                _rel_mx;
      std::condition_variable   _rel_cv;
      std::thread               _rel_thread;
      std::atomic<bool>         _rel_stop{false};

      part::arena           _arena;
      std::mutex            _root_mx;
      root_ptr              _live_root;
      std::atomic<uint64_t> _live_raw{0};
      std::vector<uint64_t> _retired;
   };

   // ── bodies ────────────────────────────────────────────────────────────────
   inline write_session::write_session(database* db)
       : _db(db), _ses(db->_arena.start_session()), _t(part_alloc<bval>(&_ses))
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
      if constexpr (engine_t::cb_create)
         _ws->_t.pg_upsert(k, v);  // bytes straight to the page
      else
         _ws->_t.cow_insert(k, bval{v});  // view in; the leaf copies once
   }
   inline int transaction::remove(key_view k)
   {
      if constexpr (engine_t::cb_create)
         return _ws->_t.pg_remove(k) ? 0 : -1;
      else
         return _ws->_t.cow_remove(k) ? 0 : -1;
   }
   template <class F>
   inline bool transaction::get(key_view k, F&& fn) const
   {
      if constexpr (engine_t::cb_create)
      {
         std::string_view out;
         if (!_ws->_t.pg_find(k, out))
            return false;
         fn(out);
         return true;
      }
      bval out;
      if (!_ws->_t.find(k, out))
         return false;
      fn(out.v);
      return true;
   }
   template <class S>
   inline std::optional<S> transaction::get(key_view k) const
   {
      if constexpr (engine_t::cb_create)
      {
         std::string_view out;
         if (!_ws->_t.pg_find(k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      bval out;
      if (!_ws->_t.find(k, out))
         return std::nullopt;
      return S(out.v.data(), out.v.size());
   }

   inline transaction::transaction(write_session* ws, part::transaction&& tx)
       : _ws(ws), _tx(std::move(tx)), _start_root(ws->_t.root_handle()),
         _start_count(ws->_t.size_hint())
   {
   }
   inline void transaction::abort() noexcept
   {
      if (_done)
         return;
      _done   = true;
      auto& t = _ws->_t;
      if (t.root_handle().raw() != _start_root.raw())
      {
         t.cow_retain(_start_root);  // the tree takes an owned reference
         const root_ptr diverged = t.root_handle();
         t.adopt_root(_start_root, _start_count);
         t.cow_release(diverged);  // drop the uncommitted generation
      }
   }
   inline void transaction::commit()
   {
      _done = true;
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

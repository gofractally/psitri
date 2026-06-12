// vpart::kv — part's NATIVE psitri-shaped public API (the same database →
// write_session → transaction → cursor structure psitri and heart::kv expose),
// so one templated test/bench body drives all engines through identical calls.
//
// Engine: the verbatim-hkv tree on the vpart arena, with STAMP LIVENESS
// (docs/unified_tier_design.md) instead of refcounts:
//   • the version IS the transaction — one boundary published per commit;
//   • the in-place fast path is within-transaction only (stamp == txn), so
//     the last committed root is stable for readers with ZERO coordination;
//   • superseded nodes ride per-transaction clone-logs (shallow = clone
//     husks, deep = dropped subtrees); commit publishes the chain under the
//     new version; abort frees the create-log and drops the chain unpublished;
//   • readers pin a VERSION by publishing (version<<32|root) in their reader
//     slot — they never mint version roots and never wait on the writer;
//   • a background drain frees every chain whose death version is <= the
//     floor (the oldest pinned version; everything, when nothing is pinned).
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

#include <vpart/hkv/tree.hpp>
#include <vpart/tree_alloc.hpp>

namespace vpart::kv
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
      // When false, no drain thread is spawned: every committed chain drains
      // synchronously on the writer after commit (identical semantics).
      bool           background_release      = true;
      // Segment-reuse bar: only segments at least this empty are reclaimed
      // for the write window; emptier = more bump-like allocation. Dan:
      // standalone expects 50%; the DELTA role (tiered) overrides to 80%
      // (flush+retire drains segments nearly clean there).
      double         reuse_threshold         = 0.50;
   };

   // ELEGANT value mode: the leaf IS the value (raw bytes sized to the data);
   // reads hand back views.
   using bval = vhkv::bytes_value;
   // DEFAULT = the c2/c4/c8 ladder, matching part's form exactly (Dan: the
   // only part/vpart diff is supposed to be refcount vs version header; the
   // census proved straight-widen was +19 MB of node_full on rand 1M, ~the
   // whole "extra space" — the stamps themselves cost zero extra lines).
   // VPART_STRAIGHT_WIDEN=1 keeps the form experiment buildable: widen goes
   // setlist -> node_full directly (fastest dispatch, fatter sparse wides);
   // the placement/mover design owns the per-node choice long-term.
#if defined(VPART_STRAIGHT_WIDEN) && VPART_STRAIGHT_WIDEN
   using engine_t = vhkv::kv_tree<std::string_view, bval, vhkv::mode::none, part_alloc<bval>>;
#else
   using engine_t =
       vhkv::kv_tree<std::string_view, bval, vhkv::mode::dense_tiers, part_alloc<bval>>;
#endif
   using root_ptr = vpart::detail::line_ptr;

   class database;

   // Snapshot cursor: a version pin (reader slot publishing version<<32|root)
   // + a non-owning view tree + an hkv iterator. The dtor clears the slot;
   // reclamation is the drain thread's job (floor rule) — readers never free.
   class cursor
   {
     public:
      cursor(cursor&& o) noexcept
          : _hp(o._hp), _rs(std::move(o._rs)), _root(o._root), _view(std::move(o._view)),
            _it(_view.end()), _rend(o._rend)
      {
         o._hp = nullptr;
      }
      cursor(const cursor&) = delete;
      ~cursor()
      {
         _view.adopt_root(root_ptr::null());  // the view owns nothing
         if (_hp)
            _hp->store(0, std::memory_order_seq_cst);  // drop the version pin
      }

      bool seek_begin()
      {
         _rend = false;
         _it   = _view.begin();
         return !is_end();
      }
      bool seek_last()
      {
         _rend = false;
         _it   = _view.last();
         return !is_end();
      }
      bool lower_bound(key_view k)
      {
         _rend = false;
         _it   = _view.lower_bound(k);
         return !is_end();
      }
      bool upper_bound(key_view k)
      {
         _rend = false;
         _it   = _view.lower_bound(k);
         if (_it != _view.end() && (*_it).first == k)
            ++_it;
         return !is_end();
      }
      bool find(key_view k)
      {
         _rend = false;
         _it   = _view.find(k);
         return !is_end();
      }
      bool next()
      {
         if (_rend)
            return seek_begin();  // psitri convention: next from rend = first key
         ++_it;
         return !is_end();
      }
      bool prev()
      {
         if (_rend)
            return false;
         if (_it == _view.end())
            return seek_last();  // prev from end = last key
         --_it;
         if (_it == _view.end())
         {  // retreated past the first key
            _rend = true;
            return false;
         }
         return true;
      }
      bool is_end() const { return !_rend && _it == _view.end(); }
      bool is_rend() const { return _rend; }
      bool is_subtree() const { return false; }  // vpart has no subtree values

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
         bval out;
         if (!_view.find(k, out))
            return std::nullopt;
         return S(out.v.data(), out.v.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         bval out;
         if (!_view.find(k, out))
            return false;
         f(out.v);
         return true;
      }

     private:
      friend class database;
      friend class read_session;
      cursor(vpart::arena* a, root_ptr root, std::atomic<uint64_t>* hp,
             std::optional<vpart::read_session> rs = std::nullopt)
          : _hp(hp), _rs(std::move(rs)), _root(root), _view(part_alloc<bval>(a)), _it(_view.end())
      {
         _view.adopt_root(_root);
      }

      std::atomic<uint64_t>*             _hp;  // the reader's published pin slot
      std::optional<vpart::read_session> _rs;  // owned slot (writer-side snapshots)
      root_ptr                           _root;
      mutable engine_t                   _view;
      engine_t::const_iterator           _it;
      bool                               _rend = false;  // retreated past the first key
   };

   // A reader thread's handle: one vpart::read_session slot for the lifetime of
   // the thread; cursors pin (publish) the latest committed version per batch.
   class read_session
   {
     public:
      cursor create_cursor(uint32_t root_index);

     private:
      friend class database;
      explicit read_session(database* db);
      database*           _db;
      vpart::read_session _rs;
   };

   class write_session;

   class transaction
   {
     public:
      void insert(key_view k, value_view v) { upsert(k, v); }
      void upsert(key_view k, value_view v);
      int  remove(key_view k);
      // Retire primitive (L5): remove ONLY if the key's last write-version is
      // <= vmax (i.e., it was already flushed at the merge floor). Keys
      // rewritten after the floor — and keys whose stamp is unknowable
      // (inline terminals) — are left alone. 0 = removed, -1 = kept/absent.
      int remove_if_older(key_view k, uint32_t vmax);
      // L5 retirement, Dan's stamp-bounded subtree algorithm (3 phases; A and
      // C run in (separate) write transactions under the caller's writer
      // serialization; B is lock-free — the detached list is private).
      size_t   retire_detach(uint32_t floor);
      uint64_t retire_count() const;
      void     retire_release(uint64_t counted);
      void commit();
      // CONTRACT (kv-compat 4.7b): dropping or aborting an uncommitted
      // transaction ROLLS BACK — the session tree returns to the txn-start
      // root, every node the transaction created is freed (create-log), and
      // its supersession chain is dropped unpublished.
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
      transaction(write_session* ws, vpart::transaction&& tx);
      write_session*     _ws;
      vpart::transaction _tx;
      root_ptr           _start_root;
      size_t             _start_count = 0;
      bool               _done        = false;
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
      transaction start_transaction(uint32_t root_index);
      // Rotation (Dan's double buffer): freeze the last committed delta as
      // the WARM tree and continue on a NEWBORN EMPTY hot tree. Publishes the
      // reset as a new version; the warm (version, root) persists in root
      // slots 2/3 until release_warm() so a crash re-flushes it (idempotent).
      // Caller holds the writer serialization. Returns {V_f, warm_root_raw}.
      std::pair<uint32_t, uint32_t> freeze_and_reset();
      // After the warm tree is durably flushed to the base: clear the slots
      // and deep-log the warm root — the DRAIN frees the whole tree once no
      // reader version-ref holds it (per-version release).
      void release_warm(uint32_t warm_root_raw);
      tree        get_root(uint32_t root_index)
      {
         (void)root_index;
         return tree(this);
      }

      // vpart: the working tree's nodes are the committed state (or were
      // logged when superseded) — teardown owns nothing to free.
      ~write_session() { _t.adopt_root(root_ptr::null()); }

      engine_t& tree_dbg() noexcept { return _t; }  // debug/tools only

     private:
      friend class database;
      friend class transaction;
      friend class tree;
      explicit write_session(database* db);

      database*      _db;
      vpart::session _ses;
      vlog_state     _vlog;  // per-transaction clone/create logs
      engine_t       _t;     // the working tree (continues across transactions)
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
      // Unified open semantics (kv_engine concept construction clause).
      static std::shared_ptr<database> open(std::filesystem::path dir,
                                            sal::open_mode        mode,
                                            runtime_config        cfg = {})
      {
         const bool exists = std::filesystem::exists(dir / "data.part");
         switch (mode)
         {
            case sal::open_mode::create_only:
               if (exists)
                  throw std::runtime_error("database already exists: " + dir.generic_string());
               break;
            case sal::open_mode::open_existing:
               if (!exists)
                  throw std::runtime_error("database does not exist: " + dir.generic_string());
               break;
            case sal::open_mode::read_only:
               throw std::runtime_error("vpart: read-only opens not implemented yet");
            case sal::open_mode::create_or_open:
               break;
         }
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
         // clean shutdown: stop and join the drain thread FIRST (single
         // consumer), then drain every committed chain (no readers remain by
         // contract, so the floor is the last committed version).
         _drain_stop.store(true, std::memory_order_release);
         _drain_cv.notify_one();
         if (_drain_thread.joinable())
            _drain_thread.join();
         drain_(committed_());
         {  // flush ring-parked runs back to the freeset
            uint32_t l, n;
            while (_recycle.pop(l, n))
               _arena.release_run(l, n);
         }
         // The live tree is durable state — nothing to release at teardown.
      }

      vpart::arena& store() noexcept { return _arena; }
      // wait until the drain ring is empty (tests / shutdown ordering)
      void flush_releases() noexcept
      {
         auto spill_empty = [&] {
            std::lock_guard<std::mutex> g(_reg_spill_mx);
            return _reg_spill.empty();
         };
         while (_reg_tail.load(std::memory_order_acquire) !=
                    _reg_head.load(std::memory_order_acquire) ||
                !spill_empty())
         {
            _drain_cv.notify_one();
            std::this_thread::yield();
         }
         // return ring-parked runs to the freeset (the gate counts free BITS;
         // ring entries are claimed-but-reusable). Caller is the consumer side.
         uint32_t l, n;
         while (_recycle.pop(l, n))
            _arena.release_run(l, n);
      }
      // debug/tools only: the raw live root (no pin — caller must hold off writers)
      root_ptr live_root_dbg() const noexcept
      {
         root_ptr       p    = root_ptr::null();
         const uint64_t live = _live.load(std::memory_order_seq_cst);
         // raw 0 is a VALID root (line 0, tag value_ptr): "nothing published"
         // is version 0 — test the WHOLE packed word, never the root half.
         if (live != 0)
         {
            const uint32_t r32 = uint32_t(live);
            std::memcpy(p.b, &r32, 4);
         }
         return p;
      }

      // A crashed rotation leaves the frozen warm tree in root slots 2/3 —
      // recovery re-flushes it (idempotent) and releases. {0,0} = none.
      std::pair<uint32_t, uint32_t> pending_warm() const
      {
         const uint32_t vf = uint32_t(_arena.get_root(3));
         if (!vf)
            return {0, 0};
         return {vf, uint32_t(_arena.get_root(2))};
      }

      // Pin a cursor at a SPECIFIC (version, root) boundary — the WARM leg of
      // tiered readers. Publish-validate against the floor: valid only while
      // a master pin (the compactor's) holds floor <= version, which the
      // rotation protocol guarantees for the whole flush window.
      cursor pin_at(uint32_t version, uint32_t root_raw)
      {
         auto  rs   = _arena.start_read_session();
         auto& slot = _arena.reader_slot(rs.slot());
         slot.store((uint64_t(version) << 32) | root_raw, std::memory_order_seq_cst);
         root_ptr p = root_ptr::null();
         std::memcpy(p.b, &root_raw, 4);
         return cursor(&_arena, p, &slot, std::move(rs));
      }

      // Writer-side snapshot: claims a read-session slot and pins the latest
      // committed boundary exactly like a reader.
      cursor pin_snapshot()
      {
         auto rs = _arena.start_read_session();
         auto& slot = _arena.reader_slot(rs.slot());
         for (;;)
         {
            const uint64_t live = _live.load(std::memory_order_seq_cst);
            slot.store(live, std::memory_order_seq_cst);
            if (_live.load(std::memory_order_seq_cst) == live)
            {
               root_ptr p = root_ptr::null();
               if (live != 0)  // raw 0 is a valid root; version 0 = unpublished
               {
                  const uint32_t r32 = uint32_t(live);
                  std::memcpy(p.b, &r32, 4);
               }
               return cursor(&_arena, p, &slot, std::move(rs));
            }
         }
      }

     private:
      friend class write_session;
      friend class read_session;
      friend class transaction;

      database(std::filesystem::path dir, runtime_config cfg)
          : _arena(make_cfg_(std::move(dir), cfg))
      {
         // resume the version counter past every stamp that could be durable,
         // and re-expose the durable root (slot 0) as the live boundary.
         const uint64_t v0 = _arena.get_root(0);
         const uint64_t v1 = _arena.get_root(1);
         _committed        = uint32_t(v1);
         _live.store((uint64_t(_committed) << 32) | uint32_t(v0), std::memory_order_seq_cst);
         if (cfg.background_release)
            _drain_thread = std::thread([this] { drain_loop_(); });
      }

      uint32_t committed_() const noexcept { return _committed; }

     public:
      // Last published version — the merge floor F a flusher captures BEFORE
      // pinning its snapshot (F <= pin version: conservative-safe for retire).
      uint32_t committed_version() const noexcept { return _committed; }

     private:

      // floor = oldest pinned version (committed when nothing is pinned).
      // seq_cst slot publication (reader) vs seq_cst scan (drain) closes the
      // race exactly like part's root_published dance.
      uint32_t floor_() noexcept
      {
         uint32_t f = _committed;
         for (uint32_t i = 0; i < max_read_sessions; ++i)
         {
            const uint64_t s = _arena.reader_slot(i).load(std::memory_order_seq_cst);
            if (!s)
               continue;
            const uint32_t v = uint32_t(s >> 32);  // pin at V sees state V:
            if (v < f)                             // deaths <= V stay eligible
               f = v;
         }
         return f;
      }

      // SPSC registry: writer pushes (version, chain head) at commit; the
      // drain owner (thread, or the writer inline when no thread) consumes.
      void reg_push_(uint32_t ver, uint32_t chain_head)
      {
         if (chain_head == vlog_state::npos)
            return;
         for (;;)
         {
            const uint32_t h = _reg_head.load(std::memory_order_relaxed);
            const uint32_t t = _reg_tail.load(std::memory_order_acquire);
            if (h - t < kRegRing)
            {
               _reg[h % kRegRing] = {ver, chain_head};
               _reg_head.store(h + 1, std::memory_order_release);
               _drain_cv.notify_one();
               return;
            }
            // ring full: a pin is blocking the floor. NEVER spin here — the
            // writer may hold locks a pin-holder needs to UNPIN (measured
            // 3-thread deadlock: writer spins in commit, drain floor-gated,
            // merger waits on the writer lock). Spill unbounded instead.
            {
               std::lock_guard g(_reg_spill_mx);
               _reg_spill.push_back({ver, chain_head});
            }
            _drain_cv.notify_one();
            return;
         }
      }

      // Drain every committed chain with death <= floor. Single consumer.
      void drain_(uint32_t floor) noexcept
      {
         engine_t v{part_alloc<bval>(&_arena, /*reclaim=*/true)};
         v.allocator_mut().vring = &_recycle;  // freed runs go back to the writer
         auto eat_chain = [&](uint32_t head) noexcept {
            // walk the chain newest-line-first; entries within a chain share
            // the death version, so order inside is irrelevant
            uint32_t line = head;
            while (line != vlog_state::npos)
            {
               const uint32_t* w = static_cast<const uint32_t*>(_arena.get(line));
               const uint32_t  prev = w[0], n = w[1];
               for (uint32_t i = 0; i < n; ++i)
                  v.vdrain_entry(w[2 + i * 2], w[3 + i * 2]);
               _arena.release_run(line, 1);  // the chain line itself
               line = prev;
            }
         };
         for (;;)
         {
            const uint32_t t = _reg_tail.load(std::memory_order_relaxed);
            if (t == _reg_head.load(std::memory_order_acquire))
               break;
            const reg_entry e = _reg[t % kRegRing];
            if (e.ver > floor)
               break;  // strictly ordered: nothing behind it is eligible either
            eat_chain(e.head);
            _reg_tail.store(t + 1, std::memory_order_release);
         }
         // ring-overflow spill (unordered): eat what the floor allows, keep the rest
         std::vector<reg_entry> local;
         {
            std::lock_guard g(_reg_spill_mx);
            local.swap(_reg_spill);
         }
         if (!local.empty())
         {
            std::vector<reg_entry> keep;
            for (const reg_entry& e : local)
               if (e.ver <= floor)
                  eat_chain(e.head);
               else
                  keep.push_back(e);
            if (!keep.empty())
            {
               std::lock_guard g(_reg_spill_mx);
               _reg_spill.insert(_reg_spill.end(), keep.begin(), keep.end());
            }
         }
      }
      void drain_loop_() noexcept
      {
         for (;;)
         {
            if (_reg_tail.load(std::memory_order_acquire) ==
                _reg_head.load(std::memory_order_acquire))
            {
               if (_drain_stop.load(std::memory_order_acquire))
                  return;
               std::unique_lock lk(_drain_mx);
               _drain_cv.wait_for(lk, std::chrono::milliseconds(2));
               continue;
            }
            drain_(floor_());
            if (_drain_stop.load(std::memory_order_acquire) &&
                _reg_tail.load(std::memory_order_acquire) ==
                    _reg_head.load(std::memory_order_acquire))
               return;
            std::this_thread::yield();
         }
      }

      static vpart::arena_config make_cfg_(std::filesystem::path dir, runtime_config cfg)
      {
         vpart::arena_config c;
         c.dir          = std::move(dir);
         c.sync_mode    = cfg.sync_mode;
         c.max_segments = cfg.max_segments;
         // COW churn frees whole generations but long-lived survivors pepper
         // old segments; 1/8 keeps the can-this-segment-serve-allocs
         // guarantee while reclaiming aggressively.
         c.reuse_threshold = cfg.reuse_threshold;
         return c;
      }

      struct reg_entry
      {
         uint32_t ver;
         uint32_t head;
      };
      static constexpr uint32_t kRegRing = 1u << 16;  // 512 KB; covers deep pin backlogs
      reg_entry                 _reg[kRegRing];
      std::atomic<uint32_t>     _reg_head{0}, _reg_tail{0};
      std::mutex                _reg_spill_mx;  // ring-overflow side list (writer never blocks)
      std::vector<reg_entry>    _reg_spill;
      std::mutex                _drain_mx;
      std::condition_variable   _drain_cv;
      std::thread               _drain_thread;
      std::atomic<bool>         _drain_stop{false};

      vpart::recycle_ring   _recycle;  // drain -> writer exact-size run reuse
      vpart::arena          _arena;
      std::mutex            _root_mx;             // serializes commit publication
      std::atomic<uint64_t> _live{0};             // (version << 32) | root raw
      uint32_t              _committed = 0;  // last published version (writer-owned)
   };

   // ── bodies ────────────────────────────────────────────────────────────────
   inline write_session::write_session(database* db)
       : _db(db), _ses(db->_arena.start_session()), _t(part_alloc<bval>(&_ses))
   {
      _t.allocator_mut().vlog = &_vlog;
      // re-adopt the durable root on reopen (parity plus: part started empty)
      const uint64_t live = db->_live.load(std::memory_order_seq_cst);
      if (live != 0)  // raw 0 is a valid root; version 0 = unpublished
      {
         const uint32_t r32 = uint32_t(live);
         root_ptr       p;
         std::memcpy(p.b, &r32, 4);
         _t.adopt_root(p);
      }
   }

   inline transaction write_session::start_transaction(uint32_t root_index)
   {
      (void)root_index;  // single root in v1, like heart_kv
      _vlog.reset();
      {  // adopt drain-recycled runs: exact-size O(1) reuse of COW churn
         uint32_t l, n, k = 0;
         while (k++ < 512 && _db->_recycle.pop(l, n))
            _ses.recycle_drained(l, n);
      }
      _t.allocator_mut().vtxn = _db->_committed + 1;  // the version IS the txn
      return transaction(this, _ses.start_transaction(0));
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
         const uint64_t live = _db->_live.load(std::memory_order_seq_cst);
         slot.store(live, std::memory_order_seq_cst);  // publish the pin FIRST...
         if (_db->_live.load(std::memory_order_seq_cst) == live)
         {  // ...then validate: the boundary was live after our publish, so
            // any drain that could free it must see our pinned version.
            root_ptr p = root_ptr::null();
            if (live != 0)  // raw 0 is a valid root; version 0 = unpublished
            {
               const uint32_t r32 = uint32_t(live);
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

   inline std::pair<uint32_t, uint32_t> write_session::freeze_and_reset()
   {
      database*      db   = _db;
      const root_ptr warm = _t.root_handle();
      const uint32_t wraw = uint32_t(warm.raw());
      const uint32_t vf   = db->_committed;
      // a fresh, empty hot tree published as its own version (the rotation
      // is itself a commit: readers pinning "latest" from here see hot=empty
      // and reach the frozen data through the WARM leg)
      const uint32_t v = vf + 1;
      _t.adopt_root(root_ptr::null(), 0);
      {
         std::lock_guard g(db->_root_mx);
         db->_arena.set_root(2, wraw);  // warm survives a crash: recovery re-flushes
         db->_arena.set_root(3, vf);
         db->_arena.set_root(0, uint32_t(root_ptr::null().raw()));
         db->_arena.set_root(1, v);
         db->_committed = v;
         db->_live.store((uint64_t(v) << 32) | uint32_t(root_ptr::null().raw()),
                         std::memory_order_seq_cst);
      }
      return {vf, wraw};
   }

   inline void write_session::release_warm(uint32_t warm_root_raw)
   {
      database* db = _db;
      {
         std::lock_guard g(db->_root_mx);
         db->_arena.set_root(2, uint32_t(root_ptr::null().raw()));
         db->_arena.set_root(3, 0);
      }
      // deep-log the warm root inside a txn: the drain cascades the whole
      // tree once the floor passes every reader still holding its versions
      root_ptr wp = root_ptr::null();
      std::memcpy(wp.b, &warm_root_raw, 4);
      auto tx = start_transaction(0);
      _t.release_root(wp);
      tx.commit();
   }

   inline void transaction::upsert(key_view k, value_view v)
   {
      _ws->_t.cow_insert(k, bval{v});  // view in; the leaf copies once
   }
   inline int transaction::remove(key_view k)
   {
      return _ws->_t.cow_remove(k) ? 0 : -1;
   }
   inline int transaction::remove_if_older(key_view k, uint32_t vmax)
   {
      const uint32_t st = _ws->_t.write_stamp(k);
      if (st == engine_t::no_stamp || st > vmax)
         return -1;
      return _ws->_t.cow_remove(k) ? 0 : -1;
   }
   inline size_t transaction::retire_detach(uint32_t floor)
   {
      return _ws->_t.retire_detach(floor);
   }
   inline uint64_t transaction::retire_count() const
   {
      return _ws->_t.retire_count();
   }
   inline void transaction::retire_release(uint64_t counted)
   {
      _ws->_t.retire_release(counted);
   }
   template <class F>
   inline bool transaction::get(key_view k, F&& fn) const
   {
      bval out;
      if (!_ws->_t.find(k, out))
         return false;
      fn(out.v);
      return true;
   }
   template <class S>
   inline std::optional<S> transaction::get(key_view k) const
   {
      bval out;
      if (!_ws->_t.find(k, out))
         return std::nullopt;
      return S(out.v.data(), out.v.size());
   }

   inline transaction::transaction(write_session* ws, vpart::transaction&& tx)
       : _ws(ws), _tx(std::move(tx)), _start_root(ws->_t.root_handle()),
         _start_count(ws->_t.size_hint())
   {
   }

   namespace detail2
   {
      inline void free_chain_lines_(vpart::arena& a, uint32_t head) noexcept
      {
         uint32_t line = head;
         while (line != vlog_state::npos)
         {
            const uint32_t* w    = static_cast<const uint32_t*>(a.get(line));
            const uint32_t  prev = w[0];
            a.release_run(line, 1);
            line = prev;
         }
      }
   }  // namespace detail2

   inline void transaction::abort() noexcept
   {
      if (_done)
         return;
      _done    = true;
      auto& t  = _ws->_t;
      auto& vl = _ws->_vlog;
      auto& a  = _ws->_db->_arena;
      // free the diverged window by WALKING it (recursion stops at pre-txn
      // nodes — they are shared with the restored root), then restore the
      // boundary root and drop the supersession chain unpublished.
      const root_ptr diverged = t.root_handle();
      if (diverged.raw() != _start_root.raw())
         t.vabort_release(diverged);
      t.adopt_root(_start_root, _start_count);
      detail2::free_chain_lines_(a, vl.clone_head);
      vl.reset();
      t.allocator_mut().vtxn = 0;
   }

   inline void transaction::commit()
   {
      _done = true;
      _tx.commit();  // data msync + floor advance (layer 1)
      database*      db = _ws->_db;
      auto&          vl = _ws->_vlog;
      const uint32_t v  = _ws->_t.allocator_mut().vtxn;
      root_ptr       nr = _ws->_t.root_handle();
      {
         std::lock_guard g(db->_root_mx);
         // null root publishes its REAL bit pattern (all-FF): raw 0 is a VALID
         // root (line 0, tag value_ptr) — encoding null as 0 made a committed
         // empty tree read as a phantom leaf at line 0 (and, before the reader
         // fix, made a real line-0 leaf root invisible). Version 0 alone means
         // "nothing ever published".
         const uint32_t  r32 = uint32_t(nr.raw());
         db->_arena.set_root(0, r32);
         db->_arena.set_root(1, v);
         db->_committed = v;
         db->_live.store((uint64_t(v) << 32) | r32, std::memory_order_seq_cst);
      }
      db->reg_push_(v, vl.clone_head);  // publish the supersession chain
      vl.reset();
      _ws->_t.allocator_mut().vtxn = 0;
      if (!db->_drain_thread.joinable())
         db->drain_(db->floor_());  // no thread: drain inline on the writer
   }
   // kv_engine concept surface (psitri::dwal::kv_engine): one aggregate naming
   // the five types + config. tiered_kv<W, Base> and the conformance
   // static_assert bind to THIS, never to the classes directly.
   struct engine
   {
      using database      = vpart::kv::database;
      using write_session = vpart::kv::write_session;
      using read_session  = vpart::kv::read_session;
      using transaction   = vpart::kv::transaction;
      using cursor        = vpart::kv::cursor;
      using config        = vpart::kv::runtime_config;
   };
}  // namespace vpart::kv

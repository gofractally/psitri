// gcart::kv — adb-style adapter over the spec engine (gcart), with
// lock-free MVCC. Mirrors the v1 gcart::legacy::kv discipline (smart_ptr IS the pin,
// try_retain+verify acquire, COW-to-root, give() hand-off) but the root is a
// (branch_type tag, u32 ptr) pair packed into an atomic u64 published slot.
#pragma once
#include <gcart/spec_tree.hpp>

#include <sal/config.hpp>  // sal::open_mode (embedder open factory)

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gcart::kv
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   inline uint64_t pack(bt t, uint32_t p) noexcept
   {
      return (uint64_t(uint8_t(t)) << 32) | p;
   }
   inline bt rtag(uint64_t r) noexcept
   {
      return bt(uint8_t(r >> 32));
   }
   inline uint32_t rptr(uint64_t r) noexcept
   {
      return uint32_t(r);
   }
   inline constexpr uint64_t null_root = (uint64_t(uint8_t(bt::null)) << 32);

   // ── smart_ptr: a held (tag,ptr) root IS the snapshot pin ───────────────────
   class snap
   {
     public:
      snap() = default;
      static snap acquire(gcart::engine& e, std::atomic<uint64_t>& slot)
      {
         for (;;)
         {
            const uint64_t r = slot.load(std::memory_order_acquire);
            const bt       t = rtag(r);
            const uint32_t p = rptr(r);
            if (t == bt::null)
               return snap(&e, bt::null, 0);
            if (!try_retain(e, t, p))
               continue;  // lost race with a freeing writer
            if (r == slot.load(std::memory_order_acquire))
               return snap(&e, t, p);
            release_subtree(e, t, p);  // root moved — undo, retry
         }
      }
      // ADOPT an existing owned (tag,ptr) WITHOUT retaining — the caller transfers a
      // ref it already holds (the set_root exchange: a work tree's root → a snapshot).
      static snap adopt(gcart::engine& e, bt t, uint32_t p) noexcept { return snap(&e, t, p); }
      snap(const snap& o) : _e(o._e), _t(o._t), _p(o._p)
      {
         if (_e && _t != bt::null)
            retain(*_e, _t, _p);
      }
      snap& operator=(const snap& o)
      {
         if (this != &o)
         {
            reset();
            _e = o._e;
            _t = o._t;
            _p = o._p;
            if (_e && _t != bt::null)
               retain(*_e, _t, _p);
         }
         return *this;
      }
      snap(snap&& o) noexcept : _e(o._e), _t(o._t), _p(o._p) { o._t = bt::null; }
      snap& operator=(snap&& o) noexcept
      {
         if (this != &o)
         {
            reset();
            _e   = o._e;
            _t   = o._t;
            _p   = o._p;
            o._t = bt::null;
         }
         return *this;
      }
      ~snap() { reset(); }

      bt       tag() const noexcept { return _t; }
      uint32_t ptr() const noexcept { return _p; }
      gcart::engine* eng() const noexcept { return _e; }
      void     give() noexcept { _t = bt::null; }  // relinquish without releasing
      // the pin OWNS its release. (A/B: SYNC release to isolate the bump allocator
      // from the async bg cascade.)
      void reset() noexcept
      {
         if (_e && _t != bt::null)
            release_subtree(*_e, _t, _p);
         _t = bt::null;
      }

     private:
      snap(gcart::engine* e, bt t, uint32_t p) : _e(e), _t(t), _p(p) {}
      gcart::engine* _e = nullptr;
      bt       _t = bt::null;
      uint32_t _p = 0;
   };

   class database;

   // ── cursor: a pinned snapshot + point get + ORDERED iteration (adb::kv_cursor)
   //    Iteration is restart-based: each positioning is one root descent (cur_ge =
   //    smallest key >= target); next() = cur_ge(curkey + '\0'). lower_bound is
   //    therefore as cheap as any engine's; multi-step scans pay one O(depth)
   //    descent per step (a frame-stack optimization is a follow-up). ────────────
   class cursor
   {
     public:
      cursor() = default;
      explicit cursor(snap s) : _s(std::move(s)) {}

      // point get (no positioning)
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return std::nullopt;
         std::string out;
         if (!find_in(*_s.eng(), _s.tag(), _s.ptr(), k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      // point get, lambda form (adb::mergeable_cursor)
      template <class F>
      bool get(key_view k, F&& f) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return false;
         std::string out;
         if (!find_in(*_s.eng(), _s.tag(), _s.ptr(), k, out))
            return false;
         f(value_view(out.data(), out.size()));
         return true;
      }

      // ── ordered forward iteration (adb::kv_cursor) ──────────────────────────
      // lower_bound/seek_begin do ONE descent (a 1-entry batch → point positioning
      // stays single-descent fast). next() amortizes: when the batch is consumed it
      // collects the next `batch` keys in one descent (cur_scan), so a scan is one
      // descent per `batch` steps instead of per step.
      static constexpr size_t batch_min = 8, batch_max = 512;
      bool                    seek_begin() { return seek_ge_(key_view{}); }
      bool                    lower_bound(key_view k) { return seek_ge_(k); }
      bool                    next()
      {
         if (_end)
            return false;
         if (_rend)
            return seek_ge_(key_view{});
         if (++_cur < _buf.size())
            return true;
         if (!_s.eng() || _s.tag() == bt::null)
         {
            _end = true;
            return false;
         }
         std::string t = _buf.back().first;
         t.push_back('\0');  // refill after the last emitted key
         _buf.clear();
         _cur = 0;
         cur_scan(*_s.eng(), _s.tag(), resolve(*_s.eng(), _s.tag(), _s.ptr()), t, 0, std::string(),
                  _buf, _bsz);
         _bsz = _bsz < batch_max ? _bsz * 2
                                 : batch_max;  // grow: short ranges collect little, scans ramp up
         if (_buf.empty())
         {
            _end = true;
            return false;
         }
         return true;
      }
      bool     is_end() const { return _end; }
      key_view key() const { return (_end || _rend) ? key_view{} : key_view(_buf[_cur].first); }
      template <class S = std::string>
      std::optional<S> value() const
      {
         if (_end || _rend)
            return std::nullopt;
         const auto& v = _buf[_cur].second;
         return S(v.data(), v.size());
      }
      // value at the current position, lambda form (adb::mergeable_cursor)
      template <class F>
      bool get_value(F&& f) const
      {
         if (_end || _rend)
            return false;
         const auto& v = _buf[_cur].second;
         f(value_view(v.data(), v.size()));
         return true;
      }
      // gcart has no composable subtrees (yet) — a value is never a subtree.
      bool is_subtree() const { return false; }

      // ── ordered reverse iteration (adb::mergeable_cursor) ───────────────────
      bool seek_last()
      {
         _buf.clear();
         _cur  = 0;
         _rend = false;
         if (!_s.eng() || _s.tag() == bt::null)
         {
            _end = true;
            return false;
         }
         std::string k, v;
         cur_rightmost(*_s.eng(), _s.tag(), resolve(*_s.eng(), _s.tag(), _s.ptr()), std::string(),
                       k, v);
         _buf.emplace_back(std::move(k), std::move(v));
         _end = false;
         return true;
      }
      bool upper_bound(key_view k)
      {
         std::string t(k);
         t.push_back('\0');
         return seek_ge_(t);
      }            // smallest > k
      bool prev()  // restart-based reverse (forward scan is the buffered hot path)
      {
         if (_rend)
            return false;
         if (_end)
            return seek_last();
         if (!_s.eng() || _s.tag() == bt::null)
         {
            _rend = true;
            return false;
         }
         std::string nk, nv;
         if (cur_lt(*_s.eng(), _s.tag(), resolve(*_s.eng(), _s.tag(), _s.ptr()), _buf[_cur].first,
                    0, std::string(), nk, nv))
         {
            _buf.clear();
            _buf.emplace_back(std::move(nk), std::move(nv));
            _cur = 0;
            _end = false;
            return true;
         }
         _rend = true;
         return false;
      }
      bool is_rend() const { return _rend; }

     private:
      bool seek_ge_(key_view target)
      {
         _buf.clear();
         _cur  = 0;
         _rend = false;
         _bsz  = batch_min;
         if (!_s.eng() || _s.tag() == bt::null)
         {
            _end = true;
            return false;
         }
         std::string k, v;
         if (cur_ge(*_s.eng(), _s.tag(), resolve(*_s.eng(), _s.tag(), _s.ptr()), target, 0,
                    std::string(), k, v))
         {
            _buf.emplace_back(std::move(k), std::move(v));
            _end = false;
            return true;
         }
         _end = true;
         return false;
      }
      snap     _s;
      kv_batch _buf;              // current run of (key,value), in order
      size_t   _cur = 0;          // index into _buf
      size_t   _bsz = batch_min;  // next refill size (grows on each refill)
      bool     _end = true, _rend = false;
   };

   // ── snapshot: an owned, FROZEN root (adb kv_tree + copyable shared_snapshot).
   //    get_root(i) returns one; it keeps its version alive via the pin, and
   //    snapshot_cursor() walks it. Named `snapshot` (not `tree`) — `tree` is the
   //    mutable engine work-tree used by transactions. ───────────────────────────
   class snapshot
   {
     public:
      snapshot() = default;
      explicit snapshot(snap s) : _s(std::move(s)) {}
      cursor   snapshot_cursor() const { return cursor(_s); }  // copies the pin (retains)
      explicit operator bool() const noexcept { return _s.eng() != nullptr; }

      // point read of the frozen snapshot (used by the snapshot-isolation battery)
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return std::nullopt;
         std::string out;
         if (!find_in(*_s.eng(), _s.tag(), _s.ptr(), k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      // root-exchange primitives (set_root): the snapshot IS the owned tree.
      bt       tag() const noexcept { return _s.tag(); }
      uint32_t ptr() const noexcept { return _s.ptr(); }
      void     give() noexcept { _s.give(); }  // relinquish the ref (it moved into a work tree)
      static snapshot adopt(gcart::engine& e, bt t, uint32_t p) { return snapshot(snap::adopt(e, t, p)); }

     private:
      snap _s;
   };

   class transaction;

   // ── root_handle: a view onto ONE root's work tree within a multi-root txn.
   //    primary()/open_root(i) return these; set_root is the owned-tree exchange. ──
   class root_handle
   {
     public:
      root_handle() = default;
      root_handle(gcart::engine* e, tree* w) : _e(e), _w(w) {}
      // (node engine, not the kv::engine bundle)
      void insert(key_view k, value_view v) { _w->upsert(k, v); }
      void upsert(key_view k, value_view v) { _w->upsert(k, v); }
      int  remove(key_view k) { return _w->remove(k); }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         std::string out;
         if (!_w->find(k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         std::string out;
         if (!_w->find(k, out))
            return false;
         f(value_view(out.data(), out.size()));
         return true;
      }

      // EXCHANGE: adopt `incoming`'s root into this work tree (transferring its ref),
      // and return the displaced root as an owned snapshot. set_root({}) clears this
      // root and hands back its old contents; the rotation composes from this.
      snapshot set_root(snapshot incoming)
      {
         const bt       dt = _w->root_tag();
         const uint32_t dp = _w->root_ptr();            // displaced
         _w->set_root(incoming.tag(), incoming.ptr());  // adopt incoming's root
         incoming.give();                               // its ref moved into _w
         return snapshot::adopt(*_e, dt, dp);           // own the displaced
      }

     private:
      gcart::engine* _e = nullptr;
      tree*   _w = nullptr;
   };

   // ── transaction: COW over one PRIMARY root (single-root API) + optional extra
   //    roots opened via open_root (multi-root). All opened roots commit/abort
   //    together. ─────────────────────────────────────────────────────────────────
   class transaction
   {
      struct root_state
      {
         uint32_t slot;
         tree*    work;
         snap     snapshot;
      };

     public:
      transaction(transaction&& o) noexcept
          : _db(o._db),
            _t(o._t),
            _slot(o._slot),
            _snap(std::move(o._snap)),
            _extra(std::move(o._extra)),
            _done(o._done)
      {
         o._db = nullptr;
      }
      transaction(const transaction&)            = delete;
      transaction& operator=(const transaction&) = delete;
      ~transaction();

      // multi-root (adb::multi_root_transaction): primary = the root opened at
      // start_transaction; open_root(i) opens additional roots in this txn.
      // (bodies out-of-line — they touch database, defined below.)
      root_handle primary();
      root_handle open_root(uint32_t idx);

      void insert(key_view k, value_view v) { _t->upsert(k, v); }
      void upsert(key_view k, value_view v) { _t->upsert(k, v); }
      int  remove(key_view k) { return _t->remove(k); }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         std::string out;
         if (!_t->find(k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      // lambda form (adb::kv_transaction): invoke f(value) on a hit, return found.
      template <class F>
      bool get(key_view k, F&& f) const
      {
         std::string out;
         if (!_t->find(k, out))
            return false;
         f(value_view(out.data(), out.size()));
         return true;
      }
      void commit();
      void abort() noexcept;

     private:
      friend class write_session;
      transaction(database* db, tree* t, uint32_t slot);
      database*               _db   = nullptr;
      tree*                   _t    = nullptr;
      uint32_t                _slot = 0;
      snap                    _snap;
      std::vector<root_state> _extra;  // roots opened beyond the primary
      bool                    _done = false;
   };

   class write_session
   {
     public:
      explicit write_session(database* db) : _db(db) {}
      transaction start_transaction(uint32_t idx = 0);
      snapshot    get_root(uint32_t idx = 0);  // owned, FROZEN snapshot of root idx
     private:
      database* _db;
   };

   class read_session
   {
     public:
      explicit read_session(database* db) : _db(db) {}
      cursor create_cursor(uint32_t idx);

     private:
      database* _db;
   };

   // ── database ───────────────────────────────────────────────────────────────
   class database
   {
     public:
      static std::shared_ptr<database> create(std::filesystem::path dir)
      {
         return std::shared_ptr<database>(new database(std::move(dir)));
      }
      static std::shared_ptr<database> open(std::filesystem::path dir)
      {
         return create(std::move(dir));
      }
      // Unified open semantics (adb::openable_database — the embedder contract the
      // MDBX shim / delta_art tier binds to). The spec engine's segment files live
      // under `dir` (cl1.dat is the first); mode gates create-vs-open.
      static std::shared_ptr<database> open(std::filesystem::path dir, sal::open_mode mode)
      {
         const bool exists = std::filesystem::exists(dir / "cl1.dat");
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
               throw std::runtime_error("gcart: read-only opens not implemented yet");
            case sal::open_mode::create_or_open:
               break;
         }
         return create(std::move(dir));
      }

      std::shared_ptr<write_session> start_write_session()
      {
         return std::make_shared<write_session>(this);
      }
      std::shared_ptr<read_session> start_read_session()
      {
         return std::make_shared<read_session>(this);
      }
      void sync(uint32_t = 0) noexcept {}

      gcart::engine&            eng() noexcept { return _e; }
      std::atomic<uint64_t>& root_slot(uint32_t i) noexcept { return _roots[i]; }
      tree&                  work_tree(uint32_t i)
      {
         if (!_work[i])
            _work[i] = std::make_unique<tree>(_e);
         return *_work[i];
      }
      // settled once the engine's background cascade has caught up.
      uint64_t live_bytes() noexcept { return _e.live_bytes(); }
      void     drain() noexcept { _e.drain(); }  // measurement barrier

     private:
      explicit database(std::filesystem::path dir) : _e(std::move(dir))
      {
         for (auto& r : _roots)
            r.store(null_root, std::memory_order_relaxed);
      }

      gcart::engine                            _e;
      std::array<std::atomic<uint64_t>, 64> _roots;
      std::array<std::unique_ptr<tree>, 64> _work{};
   };

   // ── bodies ─────────────────────────────────────────────────────────────────
   inline transaction::transaction(database* db, tree* t, uint32_t slot)
       : _db(db), _t(t), _slot(slot), _snap(snap::acquire(db->eng(), db->root_slot(slot)))
   {
      _t->set_root(_snap.tag(), _snap.ptr());  // working tree starts from the committed root
   }
   inline transaction::~transaction()
   {
      if (_db && !_done)
         abort();
   }

   inline void transaction::commit()
   {
      if (_done)
         return;
      // publish the new working root atomically, then drop the snapshot pin on
      // the old committed root. _snap.reset() routes through the engine's
      // background reclaimer (the cascade runs off the writer's critical path);
      // the smart_ptr owns its own release.
      // publish every opened root atomically (each store is atomic; cross-root is
      // ordered, not atomic — adb §multi-root), then drop the snapshot pins.
      _db->root_slot(_slot).store(pack(_t->root_tag(), _t->root_ptr()), std::memory_order_release);
      for (auto& rs : _extra)
         _db->root_slot(rs.slot).store(pack(rs.work->root_tag(), rs.work->root_ptr()),
                                       std::memory_order_release);
      _snap.reset();
      for (auto& rs : _extra)
         rs.snapshot.reset();
      _done = true;
   }
   inline void transaction::abort() noexcept
   {
      if (_done)
         return;
      auto rollback = [&](tree* w, snap& s)
      {
         const bt       wt = w->root_tag();
         const uint32_t wp = w->root_ptr();
         w->set_root(s.tag(), s.ptr());  // restore committed
         s.give();                       // the snap pin becomes the live tree's ownership ref
         release_subtree(_db->eng(), wt,
                         wp);  // free the discarded work (no-op-safe if ==committed)
      };
      rollback(_t, _snap);  // primary
      for (auto& rs : _extra)
         rollback(rs.work, rs.snapshot);  // all roots roll back together
      _done = true;
   }

   inline root_handle transaction::primary()
   {
      return root_handle(&_db->eng(), _t);
   }
   inline root_handle transaction::open_root(uint32_t idx)
   {
      if (idx == _slot)
         return root_handle(&_db->eng(), _t);
      for (auto& rs : _extra)
         if (rs.slot == idx)
            return root_handle(&_db->eng(), rs.work);
      tree& w = _db->work_tree(idx);
      snap  s = snap::acquire(_db->eng(), _db->root_slot(idx));
      w.set_root(s.tag(), s.ptr());
      _extra.push_back(root_state{idx, &w, std::move(s)});
      return root_handle(&_db->eng(), &w);
   }

   inline transaction write_session::start_transaction(uint32_t idx)
   {
      return transaction(_db, &_db->work_tree(idx), idx);
   }

   inline snapshot write_session::get_root(uint32_t idx)
   {
      return snapshot(snap::acquire(_db->eng(), _db->root_slot(idx)));
   }

   inline cursor read_session::create_cursor(uint32_t idx)
   {
      return cursor(snap::acquire(_db->eng(), _db->root_slot(idx)));
   }

   // ── adb engine bundle (adb::kv_engine) ───────────────────────────────────────
   // The five role types as nested typedefs — the shape a tiered engine
   // (delta_art) or a drop-in embedder (psitrimdbx: `using TK = engine;`) binds
   // to. `engine` here is the BUNDLE; the actual node/storage engine is
   // gcart::engine (the node/storage engine, written qualified throughout this
   // file to keep the bare name `engine` free for this bundle — the X::kv::engine
   // convention every engine uses).
   struct engine
   {
      using database      = gcart::kv::database;
      using write_session = gcart::kv::write_session;
      using read_session  = gcart::kv::read_session;
      using transaction   = gcart::kv::transaction;
      using cursor        = gcart::kv::cursor;
   };
}  // namespace gcart::kv

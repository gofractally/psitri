#pragma once
// walart — part-backed delta WAL over a heart base (docs/pdwal_design.md).
//
// v1 scope (banking app): single logical root, W/S/H pipeline with
// transaction-boundary steals, 3-tier point lookups (working ∪ stolen ∪
// heart), background key-sorted merge applied as ONE heart transaction with a
// single root publish, flush_barrier() for pure-heart artifacts (validation
// scans run heart-only behind a barrier). NOT yet: 3-tier merge ITERATION,
// range tombstones, multi-root, crash recovery (gated on the part crash-fuzz
// harness — see spec §5.1).
//
// Steal policy is GROW-BIASED per design review: deltas accumulate until
// steal_keys (default 256k entries) so the merge gains economy of scale —
// durability comes from part itself, so unlike dwal there is no WAL-size or
// RAM pressure forcing small generations. Knob: WALART_STEAL_KEYS.
//
// Value encoding in the part delta: [tag byte][payload]; tag 0x01 = data,
// 0x00 = tombstone. Values are raw leaf bytes (hkv::bytes_value) — no length
// cap; reads hand back views valid while the pinned root lives.
#include <radix_engine.hpp>  // heart engine + native psitri-shaped API
#include <part/kv.hpp>       // part engine family (pval, hkv tree, arena)

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace walart
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   namespace detail
   {
      using engine_t = part::kv::engine_t;  // hkv tree over part_alloc<bval>
      using root_ptr = part::kv::root_ptr;
      using bval     = part::kv::bval;  // raw leaf bytes, view-in/view-out

      // [tag][payload] staged in caller scratch; the leaf copies once on insert.
      inline bval encode(value_view v, bool tombstone, std::string& scratch)
      {
         scratch.resize(v.size() + 1);
         scratch[0] = tombstone ? char(0x00) : char(0x01);
         std::memcpy(scratch.data() + 1, v.data(), v.size());
         return bval{std::string_view(scratch.data(), scratch.size())};
      }
      inline bool is_tombstone(const bval& p) { return !p.v.empty() && p.v[0] == 0x00; }
      inline value_view payload(const bval& p) { return value_view(p.v.data() + 1, p.v.size() - 1); }
   }  // namespace detail

   class database;

   // Snapshot cursor: pins (W, S) part roots + a heart cursor at creation.
   // Point lookups only in v1 (the bank reader's hot path).
   class cursor
   {
     public:
      // Hand-written move (defined after database): the engine views must
      // NEVER be member-moved — a moved-from kv_tree that still holds an
      // adopted root runs hkv's unconditional teardown on the pinned snapshot
      // at destruction (part::kv's cursor steals state for the same reason).
      cursor(cursor&& o);
      cursor(const cursor&) = delete;
      ~cursor();

      template <class S = std::string>
      std::optional<S> get(key_view k);

     private:
      friend class database;
      friend class read_session;
      cursor(database* db, heart::radix::cursor&& hc);

      database*                          _db;
      detail::root_ptr                   _w = detail::root_ptr::null();
      detail::root_ptr                   _s = detail::root_ptr::null();
      mutable detail::engine_t           _wview;  // non-owning views (adopt/release null)
      mutable detail::engine_t           _sview;
      mutable heart::radix::cursor       _hc;
   };

   class read_session
   {
     public:
      cursor create_cursor(uint32_t idx = 0);

     private:
      friend class database;
      explicit read_session(database* db);
      database*                                  _db;
      std::shared_ptr<heart::radix::read_session> _hrs;
   };

   class transaction
   {
     public:
      void upsert(key_view k, value_view v);
      void insert(key_view k, value_view v) { upsert(k, v); }
      int  remove(key_view k);
      void commit();
      void abort();

      template <class S = std::string>
      std::optional<S> get(key_view k);

     private:
      friend class write_session;
      explicit transaction(database* db);
      database*                        _db;
      std::optional<part::transaction> _ptx;  // session floor: MUST be open
                                              // across mutations (hot-hole
                                              // recycling keys off it)
      bool                             _open = true;
   };

   class write_session
   {
     public:
      transaction start_transaction(uint32_t idx = 0);

     private:
      friend class database;
      explicit write_session(database* db) : _db(db) {}
      database* _db;
   };

   class database
   {
     public:
      static std::shared_ptr<database> open(const std::filesystem::path& dir,
                                            sal::sync_type sync = sal::sync_type::none)
      {
         return std::shared_ptr<database>(new database(dir, sync));
      }
      ~database() { shutdown_(); }

      std::shared_ptr<write_session> start_write_session()
      {
         return std::shared_ptr<write_session>(new write_session(this));
      }
      std::shared_ptr<read_session> start_read_session()
      {
         return std::shared_ptr<read_session>(new read_session(this));
      }

      // Drain the delta into heart and wait: after this, heart's published
      // root covers every committed walart transaction (spec §4 flush barrier).
      void flush_barrier()
      {
         {
            std::unique_lock g(_mx);
            if (_w_root.raw() != detail::root_ptr::null().raw() && !_have_s)
               steal_locked_();
         }
         _cv.notify_all();
         std::unique_lock g(_mx);
         _merge_done.wait(g, [&] { return !_have_s; });
      }

      heart::radix::database& base() { return *_heart; }
      part::arena&            delta_store() { return _arena; }
      uint64_t                delta_keys() const { return _delta_keys.load(); }
      uint64_t                merges() const { return _merges.load(); }

     private:
      friend class transaction;
      friend class cursor;
      friend class read_session;

      database(const std::filesystem::path& dir, sal::sync_type sync)
          : _sync(sync),
            _arena(make_arena_cfg_(dir / "delta", sync)),
            _wses(_arena.start_session()),
            _wtree(part::part_alloc<detail::bval>(&_wses))
      {
         if (const char* sk = std::getenv("WALART_STEAL_KEYS"); sk && atoll(sk) > 0)
            _steal_keys = uint64_t(atoll(sk));
         // heart is constructed, written, and destroyed ON the merge thread —
         // SAL sessions are thread-bound and the merger is heart's only writer.
         _merger = std::thread([this, base = dir / "base"] { merge_loop_(base); });
         std::unique_lock g(_mx);
         _cv.wait(g, [&] { return _heart != nullptr; });
      }

      static part::arena_config make_arena_cfg_(const std::filesystem::path& d,
                                                sal::sync_type               sync)
      {
         std::filesystem::remove_all(d);
         part::arena_config c;
         c.dir             = d;
         c.sync_mode       = sync;
         c.reuse_threshold = 0.125;
         return c;
      }

      void shutdown_()
      {
         flush_barrier();
         _writer_hrs.reset();  // before the merger tears heart down
         {
            std::unique_lock g(_mx);
            _stop = true;
         }
         _cv.notify_all();
         if (_merger.joinable())
            _merger.join();  // merger drains, tears heart down on its own thread
         drain_releases_();
         // writer-tree teardown dance (part::kv ~write_session)
         _wtree.cow_release(_wtree.root_handle());
         _wtree.adopt_root(detail::root_ptr::null());
         if (_w_root.raw() != detail::root_ptr::null().raw())
            _wtree.cow_release(_w_root);  // published pin
      }

      // ── writer side (single writer thread, the thread that created us) ──
      void drain_releases_()  // writer thread only
      {
         std::vector<detail::root_ptr> q;
         {
            std::unique_lock g(_mx);
            q.swap(_release_q);
         }
         for (auto r : q)
            _wtree.cow_release(r);
      }

      void writer_reset_if_stolen_()
      {
         drain_releases_();
         std::unique_lock g(_mx);
         if (!_writer_reset)
            return;
         _writer_reset = false;
         g.unlock();
         // the lineage survives via the S pin; drop the working ref, start empty
         _wtree.cow_release(_wtree.root_handle());
         _wtree.adopt_root(detail::root_ptr::null());
         _delta_keys.store(0, std::memory_order_relaxed);
      }

      void commit_()
      {
         const detail::root_ptr nr = _wtree.root_handle();
         bool                   want_steal = false;
         {
            std::unique_lock g(_mx);
            if (nr.raw() != _w_root.raw())
            {
               if (nr.raw() != detail::root_ptr::null().raw() && !_wtree.cow_retain(nr))
                  throw std::runtime_error("walart: W retain refused");
               const detail::root_ptr old = _w_root;
               _w_root                    = nr;
               _arena.set_root(0, nr.is_null() ? 0 : nr.raw());
               if (old.raw() != detail::root_ptr::null().raw())
                  _wtree.cow_release(old);
            }
            // grow-biased steal: only at txn boundary, only when merger is free
            if (!_have_s && _delta_keys.load(std::memory_order_relaxed) >= _steal_keys &&
                _w_root.raw() != detail::root_ptr::null().raw())
            {
               steal_locked_();
               want_steal = true;
            }
         }
         if (want_steal)
            _cv.notify_all();
      }

      void steal_locked_()  // _mx held; transfers the W publish-pin to S
      {
         _s_root = _w_root;
         _w_root = detail::root_ptr::null();
         _have_s = true;
         _arena.set_root(1, _s_root.raw());
         _arena.set_root(0, 0);
         _writer_reset = true;  // writer starts empty at its next txn boundary
      }

      // ── merge thread: owns heart end-to-end; S → heart, single publish ──
      void merge_loop_(std::filesystem::path base_dir)
      {
         auto heart_db = heart::radix::database::open(base_dir);
         auto hws      = heart_db->start_write_session();
         {
            std::unique_lock g(_mx);
            _heart = heart_db;
         }
         _cv.notify_all();
         while (true)
         {
            detail::root_ptr s;
            {
               std::unique_lock g(_mx);
               _cv.wait(g, [&] { return _stop || _have_s; });
               if (_stop && !_have_s)
                  break;
               s = _s_root;
            }
            {
               detail::engine_t view{part::part_alloc<detail::bval>(&_arena)};
               view.adopt_root(s);
               auto htx = hws->start_transaction(0);
               for (auto it = view.begin(); it != view.end(); ++it)
               {
                  const auto&         kv = *it;
                  const detail::bval& p  = kv.second;
                  if (detail::is_tombstone(p))
                     htx.remove(kv.first);
                  else
                     htx.upsert(kv.first, detail::payload(p));
               }
               htx.commit();  // single heart root publish for the whole delta
               view.adopt_root(detail::root_ptr::null());
            }
            {
               std::unique_lock g(_mx);
               _release_q.push_back(s);  // reclaimed by the WRITER (part SPEC §7)
               _s_root = detail::root_ptr::null();
               _have_s = false;
               _arena.set_root(1, 0);
               _merges.fetch_add(1, std::memory_order_relaxed);
            }
            _merge_done.notify_all();
         }
         heart_db->wait_for_compactor();
         hws.reset();
         // readers must be torn down before the walart database (bank teardown
         // order guarantees this); last heart ref dies here, on its home thread
         {
            std::unique_lock g(_mx);
            _heart.reset();
         }
      }

      // ── lookup plumbing ──
      bool find_delta_(const detail::engine_t& view, key_view k, std::string& out,
                       bool& tombstone) const
      {
         detail::bval p;
         if (!view.find(k, p))  // p.v: view into the pinned tree (copied below)
            return false;
         tombstone = detail::is_tombstone(p);
         if (!tombstone)
         {
            auto v = detail::payload(p);
            out.assign(v.data(), v.size());
         }
         return true;
      }

      sal::sync_type                          _sync;
      std::shared_ptr<heart::radix::database> _heart;  // set/cleared by merger
      std::shared_ptr<heart::radix::read_session> _writer_hrs;  // writer-thread heart reads
      part::arena                             _arena;
      part::session                           _wses;   // writer thread session
      detail::engine_t                        _wtree;  // writer working tree

      mutable std::mutex      _mx;
      std::condition_variable _cv, _merge_done;
      detail::root_ptr        _w_root = detail::root_ptr::null();  // published, pinned
      detail::root_ptr        _s_root = detail::root_ptr::null();  // stolen, pinned
      bool                    _have_s = false, _writer_reset = false, _stop = false;
      std::atomic<uint64_t>   _delta_keys{0}, _merges{0};
      uint64_t                _steal_keys = 256'000;
      std::thread             _merger;
      // part's reclamation contract (part::kv SPEC §7): release cascades run
      // ONLY on the writer thread — readers and the merger enqueue, the
      // writer drains at its next boundary.
      std::vector<detail::root_ptr> _release_q;
   };

   // ── inline impls ─────────────────────────────────────────────────────────
   inline transaction::transaction(database* db) : _db(db)
   {
      _ptx.emplace(_db->_wses.start_transaction(0));  // floor FIRST
      _db->writer_reset_if_stolen_();                 // then releases (floored)
   }
   inline void transaction::upsert(key_view k, value_view v)
   {
      thread_local std::string scratch;  // writer thread only
      _db->_wtree.cow_insert(k, detail::encode(v, false, scratch));
      _db->_delta_keys.fetch_add(1, std::memory_order_relaxed);
   }
   inline int transaction::remove(key_view k)
   {
      thread_local std::string scratch;
      _db->_wtree.cow_insert(k, detail::encode({}, true, scratch));
      _db->_delta_keys.fetch_add(1, std::memory_order_relaxed);
      return 0;
   }
   inline void transaction::commit()
   {
      _ptx->commit();  // part durability + floor close
      _ptx.reset();
      _db->commit_();
      _open = false;
   }
   inline void transaction::abort()
   {
      _ptx.reset();
      // re-adopt the last committed W (O(1) COW rollback — spec §2.3)
      std::unique_lock g(_db->_mx);
      detail::root_ptr w = _db->_w_root;
      g.unlock();
      auto& t = _db->_wtree;
      if (w.raw() != detail::root_ptr::null().raw() && !t.cow_retain(w))
         throw std::runtime_error("walart: abort retain refused");
      t.cow_release(t.root_handle());
      t.adopt_root(w);
      _open = false;
   }
   template <class S>
   inline std::optional<S> transaction::get(key_view k)
   {
      std::string out;
      bool        tomb = false;
      if (_db->find_delta_(_db->_wtree, k, out, tomb))  // working (read-your-writes)
         return tomb ? std::nullopt : std::optional<S>(S(out.data(), out.size()));
      detail::root_ptr s;
      {
         std::unique_lock g(_db->_mx);
         s = _db->_s_root;
         if (_db->_have_s && !_db->_wtree.cow_retain(s))
            s = detail::root_ptr::null();
      }
      if (s.raw() != detail::root_ptr::null().raw())
      {
         detail::engine_t v{part::part_alloc<detail::bval>(&_db->_arena)};
         v.adopt_root(s);
         bool hit = _db->find_delta_(v, k, out, tomb);
         v.adopt_root(detail::root_ptr::null());
         v.cow_release(s);
         if (hit)
            return tomb ? std::nullopt : std::optional<S>(S(out.data(), out.size()));
      }
      // heart read from the writer thread: lazily-created session owned by
      // the database (NOT thread_local — TLS finalizers outlive the engine),
      // torn down explicitly in shutdown_ on this same thread.
      if (!_db->_writer_hrs)
         _db->_writer_hrs = _db->_heart->start_read_session();
      return _db->_writer_hrs->create_cursor(0).template get<S>(k);
   }

   inline transaction write_session::start_transaction(uint32_t) { return transaction(_db); }

   inline read_session::read_session(database* db)
       : _db(db), _hrs(db->_heart->start_read_session())
   {
   }
   inline cursor read_session::create_cursor(uint32_t) { return cursor(_db, _hrs->create_cursor(0)); }

   inline cursor::cursor(database* db, heart::radix::cursor&& hc)
       : _db(db),
         _wview(part::part_alloc<detail::bval>(&db->_arena)),
         _sview(part::part_alloc<detail::bval>(&db->_arena)),
         _hc(std::move(hc))
   {
      std::unique_lock g(db->_mx);
      if (db->_w_root.raw() != detail::root_ptr::null().raw() &&
          _wview.cow_retain(db->_w_root))
         _w = db->_w_root;
      if (db->_have_s && _sview.cow_retain(db->_s_root))
         _s = db->_s_root;
      _wview.adopt_root(_w);
      _sview.adopt_root(_s);
   }
   inline cursor::cursor(cursor&& o)
       : _db(o._db),
         _w(o._w),
         _s(o._s),
         _wview(part::part_alloc<detail::bval>(&o._db->_arena)),
         _sview(part::part_alloc<detail::bval>(&o._db->_arena)),
         _hc(std::move(o._hc))
   {
      o._wview.adopt_root(detail::root_ptr::null());  // source owns nothing
      o._sview.adopt_root(detail::root_ptr::null());
      o._w  = detail::root_ptr::null();
      o._s  = detail::root_ptr::null();
      o._db = nullptr;
      _wview.adopt_root(_w);
      _sview.adopt_root(_s);
   }

   inline cursor::~cursor()
   {
      if (!_db)
         return;  // moved-from
      _wview.adopt_root(detail::root_ptr::null());
      _sview.adopt_root(detail::root_ptr::null());
      const bool have_w = _w.raw() != detail::root_ptr::null().raw();
      const bool have_s = _s.raw() != detail::root_ptr::null().raw();
      if (have_w || have_s)
      {
         std::unique_lock g(_db->_mx);  // reader threads NEVER run release
         if (have_w)                    // cascades (part SPEC §7) — enqueue
            _db->_release_q.push_back(_w);
         if (have_s)
            _db->_release_q.push_back(_s);
      }
   }
   template <class S>
   inline std::optional<S> cursor::get(key_view k)
   {
      std::string out;
      bool        tomb = false;
      if (_w.raw() != detail::root_ptr::null().raw() &&
          _db->find_delta_(_wview, k, out, tomb))
         return tomb ? std::nullopt : std::optional<S>(S(out.data(), out.size()));
      if (_s.raw() != detail::root_ptr::null().raw() &&
          _db->find_delta_(_sview, k, out, tomb))
         return tomb ? std::nullopt : std::optional<S>(S(out.data(), out.size()));
      return _hc.template get<S>(k);
   }
}  // namespace walart

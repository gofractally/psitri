// tiered_kv<W, B> — the L5 pipeline as ONE kv_engine (unified_tier_design §7-L5).
//
//   writes ──► W (vpart hot delta: accumulated writes + tombstones, enveloped)
//   merger ──► hysteresis-triggered BURST flush (measured: burst beats trickle,
//              logs/2026-06-12_sustained-matrix): pin snapshot at floor F,
//              key-sorted bulk chunks into B, publish per chunk
//   retire ──► writer-side pump at commit boundaries: remove_if_older(k, F)
//              (keys rewritten after F are untouched; the (R,F] band is cache)
//   reads  ──► root_index 0 = `latest`  (W ∪ B merge; W shadows; tombstones mask)
//              root_index 1 = `stable`  (B only: one plain cursor, no merge tax)
//
// Both W and B satisfy psitri::dwal::kv_engine; so does the composition —
// the static_assert in the consumer is the contract (cursors included).
// Values in W are ENVELOPED (§2.8: stored values are encodings): one tag
// byte [1=data, 0=tombstone] ahead of the payload. B stores raw payloads.
// No WAL anywhere: W's committed root IS the durable state; recovery =
// reopen and re-flush whatever W still holds (idempotent upserts/removes).
#pragma once

#include <sal/config.hpp>  // sal::open_mode (the unified open clause)

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace delta_art
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   inline constexpr char TAG_DATA = 1, TAG_TOMB = 0;

   struct knobs
   {
      // ARENA budget for the delta (Dan): the observer flushes when the
      // arena FOOTPRINT crosses half of this — free space is the sensor,
      // not bytes written (footprint includes COW churn, and a large arena
      // also poisons segment-claim scans). Keep the delta SMALL.
      uint64_t arena_budget_bytes = 1ull << 30;
      uint64_t hi_water_bytes = 512ull << 20;  // legacy bytes-written backstop
      uint64_t lo_water_bytes = 64ull << 20;   // flush down to here (hysteresis band)
      uint64_t flush_chunk    = 10'000'000;    // keys per B txn (plateau >= 100k; Dan: big)
      uint64_t repin_keys     = 1'000'000;     // re-pin the flush snapshot every N keys:
                                               // the drain floor advances DURING long flushes
      bool     background     = true;          // false: no merger thread (flush_now only; tests)
   };

   template <class W, class B>
   class tiered_kv
   {
      // The double-buffer rotation needs W's freeze/release/pin primitives
      // (Dan's design: cutoff at flush start — the flusher owns an IMMUTABLE
      // warm tree, the writer owns a newborn hot tree; nobody shares).
      // Rotation rides W's refcounted version-root handles: freeze returns a
      // tree_root; holders (merger, reader warm legs) just copy/drop it and
      // W's OWN reaper reclaims on the last drop (Dan: GC is vpart's job).
      using warm_root_t = decltype(std::declval<typename W::write_session&>().freeze_and_reset());
      static_assert(requires(warm_root_t& h) {
                       { h.version() } -> std::convertible_to<uint32_t>;
                       { h.root_raw() } -> std::convertible_to<uint32_t>;
                       { bool(h) };
                    },
                    "W's freeze_and_reset must return a refcounted tree_root handle");
      static_assert(requires(typename W::write_session& ws) { ws.clear_warm_slots(); },
                    "W must expose clear_warm_slots (crash-recovery marker)");
      static_assert(requires(typename W::database& d, uint32_t v, uint32_t r) {
                       { d.pin_at(v, r) } -> std::same_as<typename W::cursor>;
                       { d.pending_warm() } -> std::convertible_to<std::pair<uint32_t, uint32_t>>;
                    },
                    "W must expose pin_at/pending_warm — the warm-leg reader primitives");
      static_assert(requires(typename W::database& d) {
                       { d.committed_version() } -> std::convertible_to<uint32_t>;
                    },
                    "W must expose committed_version() — the merge floor F");

     public:
      class database;
      class write_session;
      class read_session;
      class transaction;
      class cursor;
      struct config
      {
         typename W::config wcfg{};
         typename B::config bcfg{};
         delta_art::knobs      knobs{};
      };

      // ── cursor: 2-tier ordered merge (latest) or B passthrough (stable) ──────
      // Precedence: W shadows B on equal keys; a W tombstone hides both sides.
      // Reverse iteration mirrors forward (max instead of min); end/rend follow
      // the psitri positioning convention through the underlying cursors.
      class cursor
      {
        public:
         bool seek_begin()
         {
            if (!_w)
               return _b.seek_begin();
            _w->seek_begin();
            if (_m)
               _m->seek_begin();
            _b.seek_begin();
            return settle_fwd_();
         }
         bool seek_last()
         {
            if (!_w)
               return _b.seek_last();
            _w->seek_last();
            if (_m)
               _m->seek_last();
            _b.seek_last();
            return settle_bwd_();
         }
         bool lower_bound(key_view k)
         {
            if (!_w)
               return _b.lower_bound(k);
            _w->lower_bound(k);
            if (_m)
               _m->lower_bound(k);
            _b.lower_bound(k);
            return settle_fwd_();
         }
         bool upper_bound(key_view k)
         {
            if (!_w)
               return _b.upper_bound(k);
            _w->upper_bound(k);
            if (_m)
               _m->upper_bound(k);
            _b.upper_bound(k);
            return settle_fwd_();
         }
         bool next()
         {
            if (!_w)
               return _b.next();
            if (_rend)  // psitri convention: next from rend = first
               return seek_begin();
            if (_on_w)
               _w->next();
            if (_on_m)
               _m->next();
            if (_on_b)
               _b.next();
            return settle_fwd_();
         }
         bool prev()
         {
            if (!_w)
               return _b.prev();
            if (is_end())  // prev from end = last
               return seek_last();
            if (_rend)
               return false;
            if (_on_w)
               _w->prev();
            if (_on_m)
               _m->prev();
            if (_on_b)
               _b.prev();
            return settle_bwd_();
         }
         bool is_end() const
         {
            if (!_w)
               return _b.is_end();
            return !_rend && !_on_w && !_on_m && !_on_b;
         }
         bool is_rend() const { return _w ? _rend : _b.is_rend(); }
         bool is_subtree() const
         {
            return _w ? ((_on_w || _on_m) ? false : _b.is_subtree()) : _b.is_subtree();
         }
         key_view key() const
         {
            if (!_w)
               return _b.key();
            return _on_w ? _w->key() : _on_m ? _m->key() : _b.key();
         }
         template <class S = std::string>
         std::optional<S> value() const
         {
            if (_w && (_on_w || _on_m))
            {  // enveloped legs; settle skipped tombstones already
               auto v = (_on_w ? *_w : *_m).template value<std::string>();
               if (!v || v->empty())
                  return std::nullopt;
               return S(v->data() + 1, v->size() - 1);
            }
            auto v = _b.template value<std::string>();
            if (!v)
               return std::nullopt;
            return S(v->data(), v->size());
         }
         template <class F>
         bool get_value(F&& f) const
         {
            if (_w && (_on_w || _on_m))
               return (_on_w ? *_w : *_m).get_value([&](value_view v) {
                  f(value_view(v.data() + 1, v.size() - 1));
               });
            return _b.get_value(std::forward<F>(f));
         }
         // Point read: W first (tombstone = definitely-deleted, stop), then B.
         // Non-const: the base cursor REPOSITIONS on a point get (heart).
         template <class F>
         bool get(key_view k, F&& f)
         {
            auto env_get = [&](typename W::cursor& c, bool& hit, bool& dead) {
               c.get(k, [&](value_view v) {
                  if (!v.empty() && v[0] == TAG_DATA)
                  {
                     hit = true;
                     f(value_view(v.data() + 1, v.size() - 1));
                  }
                  else
                     dead = true;
               });
            };
            if (_w)
            {
               bool hit = false, dead = false;
               env_get(*_w, hit, dead);
               if (hit)
                  return true;
               if (dead)
                  return false;
               if (_m)
               {
                  env_get(*_m, hit, dead);
                  if (hit)
                     return true;
                  if (dead)
                     return false;
               }
            }
            return _b.get(k, std::forward<F>(f));
         }
         template <class S = std::string>
         std::optional<S> get(key_view k)
         {
            std::optional<S> out;
            get(k, [&](value_view v) { out.emplace(v.data(), v.size()); });
            return out;
         }

        private:
         friend class read_session;
         explicit cursor(typename B::cursor&& bc) : _b(std::move(bc)) {}  // stable
         cursor(typename W::cursor&& wc, typename B::cursor&& bc)        // latest, 2-way
             : _w(std::move(wc)), _b(std::move(bc))
         {
         }
         template <class H>
         cursor(typename W::cursor&& wc, typename W::cursor&& mc, H&& mh,
                typename B::cursor&& bc)  // latest with a WARM leg (flush in flight)
             : _w(std::move(wc)), _m(std::move(mc)), _mh(std::forward<H>(mh)), _b(std::move(bc))
         {
         }
         static bool tomb_(const typename W::cursor& c)
         {
            bool tomb = false;
            c.get_value([&](value_view v) { tomb = v.empty() || v[0] == TAG_TOMB; });
            return tomb;
         }
         bool w_tomb_() const { return tomb_(*_w); }
         // Position state -> (_on_w, _on_b): which side(s) the current key
         // lives on. Tombstoned keys advance past both sides and re-settle.
         bool settle_fwd_()
         {
            for (;;)
            {
               _rend         = false;
               const bool we = _w->is_end();
               const bool me = !_m || _m->is_end();
               const bool be = _b.is_end();
               if (we && me && be)
               {
                  _on_w = _on_m = _on_b = false;
                  return false;
               }
               // the minimum key among live legs wins; precedence hot>warm>cold
               key_view mk;
               bool     have = false;
               auto     consider = [&](bool end, key_view k) {
                  if (end)
                     return;
                  if (!have || cmp_(k, mk) < 0)
                  {
                     mk   = k;
                     have = true;
                  }
               };
               consider(we, we ? key_view{} : _w->key());
               consider(me, me ? key_view{} : _m->key());
               consider(be, be ? key_view{} : _b.key());
               _on_w = !we && cmp_(_w->key(), mk) == 0;
               _on_m = !me && cmp_(_m->key(), mk) == 0;
               _on_b = !be && cmp_(_b.key(), mk) == 0;
               const bool dead = (_on_w && w_tomb_()) || (!_on_w && _on_m && tomb_(*_m));
               if (dead)
               {  // a tombstone in the winning enveloped leg masks ALL legs here
                  if (_on_w)
                     _w->next();
                  if (_on_m)
                     _m->next();
                  if (_on_b)
                     _b.next();
                  continue;
               }
               return true;
            }
         }
         bool settle_bwd_()
         {
            for (;;)
            {
               const bool wr = _w->is_rend() || _w->is_end();
               const bool mr = !_m || _m->is_rend() || _m->is_end();
               const bool br = _b.is_rend() || _b.is_end();
               if (wr && mr && br)
               {
                  _on_w = _on_m = _on_b = false;
                  _rend = true;
                  return false;
               }
               key_view mk;
               bool     have = false;
               auto     consider = [&](bool ex, key_view k) {
                  if (ex)
                     return;
                  if (!have || cmp_(k, mk) > 0)
                  {
                     mk   = k;
                     have = true;
                  }
               };
               consider(wr, wr ? key_view{} : _w->key());
               consider(mr, mr ? key_view{} : _m->key());
               consider(br, br ? key_view{} : _b.key());
               _on_w = !wr && cmp_(_w->key(), mk) == 0;
               _on_m = !mr && cmp_(_m->key(), mk) == 0;
               _on_b = !br && cmp_(_b.key(), mk) == 0;
               const bool dead = (_on_w && w_tomb_()) || (!_on_w && _on_m && tomb_(*_m));
               if (dead)
               {
                  if (_on_w)
                     _w->prev();
                  if (_on_m)
                     _m->prev();
                  if (_on_b)
                     _b.prev();
                  continue;
               }
               return true;
            }
         }
         static int cmp_(key_view a, key_view b)
         {
            const int c = std::memcmp(a.data(), b.data(), std::min(a.size(), b.size()));
            if (c)
               return c;
            return a.size() < b.size() ? -1 : a.size() > b.size() ? 1 : 0;
         }
         std::optional<typename W::cursor> _w;  // hot leg; absent = stable view
         std::optional<typename W::cursor> _m;  // WARM leg (flush in flight)
         warm_root_t _mh;  // keeps the warm tree alive: the reaper counts handles
         typename B::cursor                _b;
         bool _on_w = false, _on_m = false, _on_b = false, _rend = false;
      };

      class read_session
      {
        public:
         // root_index 0 = latest (W ∪ B), 1 = stable (B only).
         cursor create_cursor(uint32_t root_index)
         {
            if (root_index == 1)
               return cursor(_brs->create_cursor(0));
            std::lock_guard rg(_db->_rot_m);  // legs sampled atomically vs rotation
            if (_db->_warm_active)
               return cursor(_wrs->create_cursor(0),
                             _db->_w->pin_at(_db->_warm_vf, _db->_warm_root),
                             _db->_warm_handle, _brs->create_cursor(0));
            return cursor(_wrs->create_cursor(0), _brs->create_cursor(0));
         }

        private:
         friend class database;
         read_session(database* db, std::shared_ptr<typename W::read_session> w,
                      std::shared_ptr<typename B::read_session> b)
             : _db(db), _wrs(std::move(w)), _brs(std::move(b))
         {
         }
         database*                                  _db;
         std::shared_ptr<typename W::read_session> _wrs;
         std::shared_ptr<typename B::read_session> _brs;
      };

      class transaction
      {
        public:
         void insert(key_view k, value_view v) { upsert(k, v); }
         void upsert(key_view k, value_view v)
         {
            _buf.resize(v.size() + 1);
            _buf[0] = TAG_DATA;
            std::memcpy(_buf.data() + 1, v.data(), v.size());
            _tx->upsert(k, value_view(_buf.data(), _buf.size()));
            _bytes += k.size() + v.size() + 1;
         }
         void remove(key_view k)
         {  // composed mode: a tombstone (it must MASK the base); retire drops it
            const char t = TAG_TOMB;
            _tx->upsert(k, value_view(&t, 1));
            _bytes += k.size() + 1;
         }
         void commit()
         {
            _tx->commit();
            _db->after_commit_(_bytes);
            _bytes = 0;
            _lk.unlock();
         }
         void abort() noexcept
         {
            _tx->abort();
            _bytes = 0;
            if (_lk.owns_lock())
               _lk.unlock();
         }
         template <class F>
         bool get(key_view k, F&& f) const
         {  // Dan's 3-way writer cursor: open txn (hot) -> warm -> base
            bool hit = false, dead = false;
            _tx->get(k, [&](value_view v) {
               if (!v.empty() && v[0] == TAG_DATA)
               {
                  hit = true;
                  f(value_view(v.data() + 1, v.size() - 1));
               }
               else
                  dead = true;
            });
            if (hit)
               return true;
            if (dead)
               return false;
            if (_db->warm_get_(_warm, _warm_h, k, hit, dead, f))
               return hit && !dead ? true : hit;
            if (dead)
               return false;
            return _db->base_get_(k, std::forward<F>(f));
         }
         template <class S = std::string>
         std::optional<S> get(key_view k) const
         {
            std::optional<S> out;
            get(k, [&](value_view v) { out.emplace(v.data(), v.size()); });
            return out;
         }

        private:
         friend class write_session;
         transaction(database* db, write_session* ws, std::unique_lock<std::mutex>&& lk,
                     typename W::transaction&& tx)
             : _db(db), _ws(ws), _lk(std::move(lk)), _tx(std::move(tx))
         {
         }
         database*                          _db;
         write_session*                     _ws;
         std::unique_lock<std::mutex>       _lk;  // writer serialization (user <-> merger)
         mutable std::optional<typename W::cursor> _warm;  // lazy warm leg (this txn)
         mutable warm_root_t                        _warm_h;  // keeps it alive
         std::optional<typename W::transaction>    _tx;
         std::string                        _buf;  // envelope scratch (amortized)
         uint64_t                           _bytes = 0;
      };

      class write_session
      {
        public:
         transaction start_transaction(uint32_t root_index)
         {
            std::unique_lock lk(_db->_writer_m);  // released at commit/abort
            return transaction(_db, this, std::move(lk), _ws->start_transaction(root_index));
         }

        private:
         friend class database;
         friend class transaction;
         friend class tiered_kv;
         write_session(database* db, std::shared_ptr<typename W::write_session> ws)
             : _db(db), _ws(std::move(ws))
         {
         }
         database*                                  _db;
         std::shared_ptr<typename W::write_session> _ws;
      };

      class database
      {
        public:
         static std::shared_ptr<database> open(const std::filesystem::path& dir,
                                               sal::open_mode               mode,
                                               const config&                cfg = {})
         {
            return std::shared_ptr<database>(new database(dir, mode, cfg));
         }
         std::shared_ptr<write_session> start_write_session()
         {
            return std::shared_ptr<write_session>(new write_session(this, _wws));
         }
         std::shared_ptr<read_session> start_read_session()
         {
            return std::shared_ptr<read_session>(
                new read_session(this, _w->start_read_session(), _b->start_read_session()));
         }
         ~database()
         {
            {
               std::lock_guard lk(_m);
               _stop = true;
            }
            _cv.notify_all();
            if (_merger.joinable())
               _merger.join();
         }

         // One full flush, synchronously (tests + the no-thread mode): pin a
         // snapshot at floor F, stream it into B in big sorted chunks, then arm
         // the retire pump at F. Returns keys flushed.
         uint64_t flush_now()
         {
            // ── Dan's double buffer ─────────────────────────────────────────
            // CUTOFF: freeze the committed delta as the WARM tree and reset
            // the writer to a newborn hot tree — the flusher owns an
            // IMMUTABLE tree, the writer owns a private one; nobody shares a
            // mutating delta (the in-place retire WAS the write-amp).
            warm_root_t handle;
            uint32_t    vf = 0, wroot = 0;
            {
               std::scoped_lock lk(_writer_m, _rot_m);
               if (_warm_active)
                  return 0;  // a flush is already in flight
               if (auto pend = _w->pending_warm(); pend.first)
                  handle = _w->make_orphan_root(pend.first, pend.second);  // crashed run
               else
                  handle = _wws->freeze_and_reset();
               vf     = handle.version();
               wroot  = handle.root_raw();
               _warm_active = true;
               _warm_vf     = vf;
               _warm_root   = wroot;
               _warm_handle = handle;  // reader warm legs copy THIS
            }
            // master pin: holds the floor <= vf for every reader warm leg.
            // optional: it MUST drop before the final writer-lock acquisition
            // (deadlock cycle otherwise: writer spins on the full supersession
            // ring inside commit while HOLDING _writer_m; the drain can't eat
            // the ring while the floor sits at vf; we'd wait on _writer_m).
            std::optional<typename W::cursor> warm;
            warm.emplace(_w->pin_at(vf, wroot));
            if (!_bws)
               _bws = _b->start_write_session();  // ONCE per flushing thread
            uint64_t n    = 0;
            bool     more = warm->seek_begin();
            while (more)
            {
               auto     btx   = _bws->start_transaction(0);
               uint64_t in_tx = 0;
               while (!warm->is_end() && in_tx < _k.flush_chunk)
               {
                  warm->get_value([&](value_view v) {
                     if (!v.empty() && v[0] == TAG_DATA)
                        btx.upsert(warm->key(), value_view(v.data() + 1, v.size() - 1));
                     else
                        btx.remove(warm->key());  // tombstone: applied, never stored in B
                  });
                  ++in_tx;
                  warm->next();
               }
               btx.commit();
               _base_epoch.fetch_add(1, std::memory_order_release);
               n += in_tx;
               more = !warm->is_end();
            }
            warm.reset();  // DROP the floor pin before touching _writer_m
            // the warm data is durable in B: stop handing out warm legs and
            // clear the crash slots, then cascade-free the tree INLINE on
            // this (merger) thread once every warm reader is gone — the
            // drain thread cannot keep pace with multi-GB tree cascades
            // (measured: generation backlog to the 32 GB wall at 142M keys).
            {
               std::scoped_lock lk(_writer_m, _rot_m);
               _warm_active = false;
               _warm_handle.reset();  // OUR ref; readers may still hold theirs
               _wws->clear_warm_slots();
            }
            _unflushed.store(0, std::memory_order_relaxed);
            // `handle` (this frame's copy) drops at return: when the LAST
            // holder (reader warm legs included) lets go, W's reaper frees
            // the tree — the merger does no garbage collection (Dan).
            return n;
         }

        private:
         friend class transaction;
         friend class read_session;
         friend class tiered_kv;
         database(const std::filesystem::path& dir, sal::open_mode mode, const config& cfg)
             : _k(cfg.knobs)
         {
            std::filesystem::create_directories(dir / "delta");
            std::filesystem::create_directories(dir / "base");
            auto wcfg = cfg.wcfg;
            // Delta role (Dan): flush+retire drains segments nearly clean, so
            // only near-empty segments are worth reusing — 80% bar keeps
            // allocation bump-like (long runs, virgin locality). Standalone
            // W keeps its own default (50%).
            if constexpr (requires { wcfg.reuse_threshold; })
               wcfg.reuse_threshold = 0.80;
            _w   = W::database::open(dir / "delta", mode, wcfg);
            _b   = B::database::open(dir / "base", mode, cfg.bcfg);
            _wws = _w->start_write_session();  // THE single W writer, shared
                                               // (user txns and merger retire
                                               // txns serialize on _writer_m;
                                               // sessions are thread-agnostic)
            if (_k.background)
               _merger = std::thread([this] { merge_loop_(); });
         }
         void after_commit_(uint64_t bytes)
         {
            const uint64_t un = _unflushed.fetch_add(bytes, std::memory_order_relaxed) + bytes;
            // primary trigger: arena footprint at HALF the budget (flush
            // EARLY — memory keeps growing while the flush runs). The
            // footprint read is two loads; cheap enough per commit.
            if (un >= _k.hi_water_bytes ||
                _w->footprint_bytes() >= _k.arena_budget_bytes / 2)
               _cv.notify_one();
         }
         // warm-leg point get for the writer's 3-way cursor: lazily pin the
         // current warm boundary for THIS transaction; envelope-decoded.
         template <class F>
         bool warm_get_(std::optional<typename W::cursor>& leg, warm_root_t& legh, key_view k,
                        bool& hit, bool& dead, F&& f)
         {
            if (!leg)
            {
               std::lock_guard rg(_rot_m);
               if (!_warm_active)
                  return false;
               legh = _warm_handle;  // hold the tree (reaper counts handles)
               leg.emplace(_w->pin_at(_warm_vf, _warm_root));
            }
            leg->get(k, [&](value_view v) {
               if (!v.empty() && v[0] == TAG_DATA)
               {
                  hit = true;
                  f(value_view(v.data() + 1, v.size() - 1));
               }
               else
                  dead = true;
            });
            return hit || dead;
         }
         template <class F>
         bool base_get_(key_view k, F&& f)
         {
            if (!_bget_rs)
               _bget_rs = _b->start_read_session();
            // The cached cursor pins ONE base root: it must be re-seated every
            // time the merger publishes, or post-retire reads see PRE-FLUSH
            // values (caught by the bank balance check: stale reads inflate).
            const uint64_t e = _base_epoch.load(std::memory_order_acquire);
            if (!_bget_cur || _bget_epoch != e)
            {
               _bget_cur.reset();
               _bget_cur.emplace(_bget_rs->create_cursor(0));
               _bget_epoch = e;
            }
            return _bget_cur->get(k, std::forward<F>(f));
         }
         void merge_loop_()
         {
            for (;;)
            {
               std::unique_lock lk(_m);
               _cv.wait(lk, [&] {
                  return _stop ||
                         _unflushed.load(std::memory_order_relaxed) >= _k.hi_water_bytes ||
                         _w->footprint_bytes() >= _k.arena_budget_bytes / 2;
               });
               if (_stop)
                  return;
               lk.unlock();
               flush_now();  // BURST, flat out (measured: burst beats trickle)
            }
         }

         std::shared_ptr<typename W::database>      _w;
         std::shared_ptr<typename B::database>      _b;
         std::shared_ptr<typename W::write_session> _wws;
         std::shared_ptr<typename B::write_session> _bws;  // merger-thread, lazy
         knobs                                 _k;
         std::atomic<uint64_t>                 _unflushed{0};
         std::mutex                            _writer_m;  // user txns <-> merger rotation txns
         std::mutex  _rot_m;          // warm-state flips vs reader-leg sampling
         bool        _warm_active = false;
         uint32_t    _warm_vf = 0, _warm_root = 0;
         warm_root_t _warm_handle;
         std::shared_ptr<typename B::read_session>        _bget_rs;   // writer-thread base reads
         std::optional<typename B::cursor>                _bget_cur;
         std::atomic<uint64_t>                            _base_epoch{0};
         uint64_t                                         _bget_epoch = 0;
         std::thread             _merger;
         std::mutex              _m;
         std::condition_variable _cv;
         bool                    _stop = false;
      };
   };

   // The composed engine as a kv_engine aggregate.
   template <class W, class B>
   struct engine
   {
      using database      = typename tiered_kv<W, B>::database;
      using write_session = typename tiered_kv<W, B>::write_session;
      using read_session  = typename tiered_kv<W, B>::read_session;
      using transaction   = typename tiered_kv<W, B>::transaction;
      using cursor        = typename tiered_kv<W, B>::cursor;
      using config        = typename tiered_kv<W, B>::config;
   };
}  // namespace delta_art

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
#include <sal/diag.hpp>    // PSITRI_DEBUG(...) gate + sal::env cached env reader

#include <cstdlib>
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
      // The double-buffer rotation rides W's adb multi-root + snapshot
      // primitives ONLY (concept-only — no contraband shims). Rotation is the
      // set_root exchange in one multi-root transaction (h1.set_root(h0
      // .set_root({}))): the writer's hot tree (root 0) becomes the IMMUTABLE
      // warm tree (root 1), root 0 is a newborn — the flusher and the writer
      // never share a mutating delta. The warm tree is read through
      // get_root(1), an owned, FROZEN snapshot (a tree IS a snapshot); copies
      // of it (merger flush cursor, reader/writer warm legs) keep the version
      // alive, and W's OWN GC reclaims it when the last handle drops.
      using warm_tree_t = decltype(std::declval<typename W::write_session&>().get_root(1u));
      static_assert(requires(typename W::transaction& tx, uint32_t i) {
                       { tx.primary() };
                       { tx.open_root(i) };
                    },
                    "W must be a multi-root engine (primary()/open_root())");
      static_assert(requires(warm_tree_t& t) {
                       { t.snapshot_cursor() } -> std::same_as<typename W::cursor>;
                       { bool(t) };
                    },
                    "W::get_root must return an owned, readable snapshot tree");
      // The warm handle is COPIED to many holders (merger flush cursor +
      // reader/writer warm legs); a move-only one-ref snapshot can't be shared.
      static_assert(std::is_copy_constructible_v<warm_tree_t>,
                    "W::get_root's snapshot must be COPYABLE (warm legs share it)");
      // Both tiers' cursors drive the bidirectional merge: positioning returns
      // bool (landed-on-a-key), plus reverse iteration. A void-returning cursor
      // passes the battery but cannot merge — catch it HERE, not deep in a
      // template error.
      static constexpr auto mergeable_ = [](auto* cp) {
         using C = std::remove_pointer_t<decltype(cp)>;
         return requires(C c, std::string_view k) {
            { c.seek_begin() } -> std::convertible_to<bool>;
            { c.seek_last() } -> std::convertible_to<bool>;
            { c.next() } -> std::convertible_to<bool>;
            { c.prev() } -> std::convertible_to<bool>;
            { c.is_end() } -> std::convertible_to<bool>;
            { c.is_rend() } -> std::convertible_to<bool>;
         };
      };
      static_assert(mergeable_(static_cast<typename W::cursor*>(nullptr)),
                    "W::cursor must be a bool-returning bidirectional (mergeable) cursor");
      static_assert(mergeable_(static_cast<typename B::cursor*>(nullptr)),
                    "B::cursor must be a bool-returning bidirectional (mergeable) cursor");

     public:
      class database;
      class write_session;
      class read_session;
      class transaction;
      class cursor;
      struct config
      {
         // adb-level commit durability — fanned out to BOTH tiers (hot W + cold B) at open.
         // Default fsync matches MDBX's MDBX_SYNC_DURABLE: a real commit boundary every batch,
         // which writes the seal record so segments compact/recycle (no seal => unbounded growth).
         // The shim derives this from the MDBX env durability flags.
         sal::sync_type     commit_sync = sal::sync_type::fsync;
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
            _dir_fwd_ = true;
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
            _dir_fwd_ = false;
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
            _dir_fwd_ = true;
            _w->lower_bound(k);
            if (_m)
               _m->lower_bound(k);
            _b.lower_bound(k);
            return settle_fwd_();
         }
         // Mathematical upper bound: the LARGEST live key v <= k (the floor; the dual
         // of lower_bound). Reuses the merge-aware lower_bound/prev/seek_last so the
         // W∪B(∪M) dedup + direction handling is not re-derived here.
         bool upper_bound(key_view k)
         {
            if (!_w)
               return _b.upper_bound(k);  // base passthrough (the base leg is floor too)
            if (lower_bound(k))           // smallest live v >= k
            {
               if (key() == k)            // k present (live) → it is its own floor
                  return true;
               return prev();             // landed strictly > k → predecessor = largest v < k
            }
            return seek_last();           // no live key >= k → the last key is the floor
         }
         bool next()
         {
            if (!_w)
               return _b.next();
            if (_rend)  // psitri convention: next from rend = first
               return seek_begin();
            if (!_dir_fwd_)
            {  // direction change (backward -> forward): non-current legs sit at keys < current;
               // move each to the smallest key > current before merging.
               if (is_end())
                  return false;
               std::string cur(key());
               reseek_fwd_(*_w, cur);
               if (_m)
                  reseek_fwd_(*_m, cur);
               reseek_fwd_(_b, cur);
               _dir_fwd_ = true;
               return settle_fwd_();
            }
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
            if (_dir_fwd_)
            {  // direction change (forward -> backward): non-current legs sit at keys > current;
               // move each to the largest key < current before merging.
               std::string cur(key());
               reseek_bwd_(*_w, cur);
               if (_m)
                  reseek_bwd_(*_m, cur);
               reseek_bwd_(_b, cur);
               _dir_fwd_ = false;
               return settle_bwd_();
            }
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
         // Reposition a leg on a DIRECTION CHANGE. After a forward op (lower_bound/upper_bound/
         // next/seek_begin) every non-current leg sits at a key > current; after a backward op,
         // at a key < current. On the first opposite step those "stranded" legs must be moved to
         // the correct side of `cur`, or settle_*() re-selects one of them on the wrong side
         // (the seek_last-of-namespace bug: lower_bound(successor)+prev returned a next-namespace
         // key). fwd: smallest key > cur; bwd: largest key < cur.
         template <class C>
         static void reseek_fwd_(C& c, key_view cur)
         {  // smallest key > cur (strictly greater): lower_bound then skip an exact
            // match. (A leg's upper_bound is now the floor, so it can't be used here.)
            if (c.lower_bound(cur) && c.key() == cur)
               c.next();
         }
         template <class C>
         static void reseek_bwd_(C& c, key_view cur)
         {
            if (c.lower_bound(cur))
               c.prev();
            else
               c.seek_last();
         }
         std::optional<typename W::cursor> _w;  // hot leg; absent = stable view
         std::optional<typename W::cursor> _m;  // WARM leg (flush in flight)
         warm_tree_t _mh;  // keeps the warm snapshot alive: W's GC counts handles
         typename B::cursor                _b;
         bool _on_w = false, _on_m = false, _on_b = false, _rend = false;
         bool _dir_fwd_ = true;  // last positioning direction (for direction-change reseek)
      };

      class read_session
      {
        public:
         // root_index 0 = latest (W ∪ B), 1 = stable (B only).
         // Snapshot order under _rot_m (consistency vs rotation/clear): HOT
         // first, then WARM, then BASE — a rotation between hot and warm makes
         // an in-flight key appear in both (dedup), and clear strictly follows
         // the base commit, so a base cursor taken after warm never misses it.
         cursor create_cursor(uint32_t root_index)
         {
            if (root_index == 1)
               return cursor(_brs->create_cursor(0));
            std::lock_guard rg(_db->_rot_m);  // legs sampled atomically vs rotation
            auto hot = _wrs->create_cursor(0);
            if (_db->_warm_active)
               return cursor(std::move(hot), _db->_warm_handle.snapshot_cursor(),
                             _db->_warm_handle, _brs->create_cursor(0));
            return cursor(std::move(hot), _brs->create_cursor(0));
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
            // Destroy the inner W transaction BEFORE releasing _writer_m: its
            // ~transaction() touches the shared write session, and the merge
            // thread's flush_now borrows that same session under _writer_m.
            // Unlocking first let the inner dtor race flush_now (TSan-confirmed).
            _tx.reset();
            _lk.unlock();
         }
         void abort() noexcept
         {
            _bytes = 0;
            _tx.reset();  // inner ~transaction() rolls back — under _writer_m (see commit)
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
         mutable warm_tree_t                        _warm_h;  // keeps the snapshot alive
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
         // debug/tuning: (used_lines, total_lines) of the DELTA (hot) arena.
         std::pair<uint64_t, uint64_t> delta_occupancy_dbg() noexcept
         {
            if constexpr (requires { _w->occupancy_dbg(); })
               return _w->occupancy_dbg();
            else
               return {0, 0};
         }
         std::pair<uint64_t, uint64_t> delta_seg_stats_dbg() noexcept
         {
            if constexpr (requires { _w->seg_stats_dbg(); })
               return _w->seg_stats_dbg();
            else
               return {0, 0};
         }
         // Completed base transactions since open (including recovery).
         uint64_t base_commit_count() const noexcept
         { return _base_epoch.load(std::memory_order_acquire); }
         uint64_t flush_now()
         {
            // ── Dan's double buffer, in adb vocabulary ──────────────────────
            // CUTOFF = the set_root exchange: in one multi-root transaction
            // move the writer's hot tree (root 0) to the WARM slot (root 1)
            // and leave root 0 a newborn. The flusher then reads root 1 as a
            // FROZEN snapshot (get_root(1)); the writer mutates the private
            // newborn root 0 — nobody shares a mutating delta (the in-place
            // retire WAS the write-amp). No freeze/pin/clear shims.
            warm_tree_t warm_snap;
            {
               std::scoped_lock lk(_writer_m, _rot_m);
               if (_warm_active)
                  return 0;  // a flush is already in flight
               // rotate hot -> warm (only when root 0 actually holds data; an
               // empty rotation just wastes a version)
               auto tx = _wws->start_transaction(0);
               auto h0 = tx.primary();
               auto h1 = tx.open_root(1);
               h1.set_root(h0.set_root({}));  // THE exchange
               tx.commit();
               warm_snap    = _wws->get_root(1);  // owned, frozen warm snapshot
               _warm_handle = warm_snap;          // reader/writer warm legs copy THIS
               _warm_active = true;
            }
            // Stream the frozen warm snapshot into B (independent of _wws now;
            // the writer keeps mutating root 0 the whole time).
            const uint64_t n = drain_warm_(warm_snap);
            // The warm data is durable in B. CLEAR the warm slot (set_root({})
            // on root 1) — the displaced warm tree ARMS and W's GC reclaims it
            // when the last snapshot handle (reader warm legs + our copies)
            // drops. Stop handing out warm legs under _rot_m. Snapshot order
            // (clear AFTER the base commits) is what lets a reader's base
            // cursor never miss an in-flight key.
            {
               std::scoped_lock lk(_writer_m, _rot_m);
               _warm_active = false;
               _warm_handle = warm_tree_t{};  // OUR shared ref; readers keep theirs
               auto tx = _wws->start_transaction(1);
               tx.primary().set_root({});  // empty the warm slot
               tx.commit();
            }
            _unflushed.store(0, std::memory_order_relaxed);
            // warm_snap drops at return: when the LAST holder (reader warm
            // legs included) lets go, W's GC frees the tree — the merger does
            // no garbage collection (Dan).
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
            // adb-level commit durability fans out to BOTH tiers — the commit boundary that
            // seals segments for compaction/recycling (no seal => the base grows by the full
            // COW-garbage rate). Hot W commits sync per cfg; cold B's publish now honors it too.
            wcfg.sync_mode = cfg.commit_sync;
            auto bcfg      = cfg.bcfg;
            bcfg.sync_mode = cfg.commit_sync;
            // Delta role (Dan): flush+retire drains segments nearly clean, so
            // only near-empty segments are worth reusing — 80% bar keeps
            // allocation bump-like (long runs, virgin locality). Standalone
            // W keeps its own default (50%).
            if constexpr (requires { wcfg.reuse_threshold; })
            {
               wcfg.reuse_threshold = 0.80;
               PSITRI_DEBUG(if (const char* rt = sal::env::get("DELTA_REUSE"))  // experiment knob
                               wcfg.reuse_threshold = atof(rt);)
            }
            _w   = W::database::open(dir / "delta", mode, wcfg);
            _b   = B::database::open(dir / "base", mode, bcfg);
            _wws = _w->start_write_session();  // THE single W writer, shared
                                               // (user txns and merger retire
                                               // txns serialize on _writer_m;
                                               // sessions are thread-agnostic)
            recover_();  // a crashed flush may have left warm data in root 1
            if (_k.background)
               _merger = std::thread([this] { merge_loop_(); });
         }
         // Stream a frozen warm snapshot into B in sorted chunks (the flush
         // body, shared by flush_now and recovery). Returns keys applied.
         uint64_t drain_warm_(warm_tree_t& warm_snap)
         {
            if (!_bws)
               _bws = _b->start_write_session();
            uint64_t n    = 0;
            auto     warm = warm_snap.snapshot_cursor();
            bool     more = warm.seek_begin();
            while (more)
            {
               auto     btx   = _bws->start_transaction(0);
               uint64_t in_tx = 0;
               while (!warm.is_end() && in_tx < _k.flush_chunk)
               {
                  warm.get_value([&](value_view v) {
                     if (!v.empty() && v[0] == TAG_DATA)
                        btx.upsert(warm.key(), value_view(v.data() + 1, v.size() - 1));
                     else
                        btx.remove(warm.key());  // tombstone: applied, never stored in B
                  });
                  ++in_tx;
                  warm.next();
               }
               btx.commit();
               _base_epoch.fetch_add(1, std::memory_order_release);
               n += in_tx;
               more = !warm.is_end();
            }
            return n;
         }
         // Crash recovery: if root 1 (the warm slot) still holds data from a
         // run that died mid-flush, re-flush it (idempotent upserts/removes)
         // and clear the slot. No WAL — re-applying the warm tree is the
         // recovery. Empty root 1 = clean shutdown, nothing to do.
         void recover_()
         {
            warm_tree_t warm_snap = _wws->get_root(1);
            auto        probe     = warm_snap.snapshot_cursor();
            if (!probe.seek_begin())
               return;  // empty warm slot
            drain_warm_(warm_snap);
            auto tx = _wws->start_transaction(1);
            tx.primary().set_root({});
            tx.commit();
         }
         // Flush trigger: bytes-written backstop, plus (when W exposes it) the
         // arena footprint at HALF the budget — flush EARLY, memory keeps
         // growing while the flush runs. footprint_bytes is an OPTIONAL W
         // sensor (if constexpr): the concept requires only the byte counter.
         bool over_water_() const
         {
            if (_unflushed.load(std::memory_order_relaxed) >= _k.hi_water_bytes)
               return true;
            if constexpr (requires(typename W::database& d) { d.footprint_bytes(); })
               return _w->footprint_bytes() >= _k.arena_budget_bytes / 2;
            return false;
         }
         void after_commit_(uint64_t bytes)
         {
            _unflushed.fetch_add(bytes, std::memory_order_relaxed);
            if (over_water_())
               _cv.notify_one();
         }
         // warm-leg point get for the writer's 3-way cursor: lazily pin the
         // current warm boundary for THIS transaction; envelope-decoded.
         template <class F>
         bool warm_get_(std::optional<typename W::cursor>& leg, warm_tree_t& legh, key_view k,
                        bool& hit, bool& dead, F&& f)
         {
            if (!leg)
            {
               std::lock_guard rg(_rot_m);
               if (!_warm_active)
                  return false;
               legh = _warm_handle;  // hold the snapshot (W's GC counts handles)
               leg.emplace(legh.snapshot_cursor());
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
               _cv.wait(lk, [&] { return _stop || over_water_(); });
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
         warm_tree_t _warm_handle;    // the shared frozen warm snapshot (get_root(1))
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

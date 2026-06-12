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

namespace tiered
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   inline constexpr char TAG_DATA = 1, TAG_TOMB = 0;

   struct knobs
   {
      uint64_t hi_water_bytes = 512ull << 20;  // burst trigger (Dan: ~512 MB keeps W shallow)
      uint64_t lo_water_bytes = 64ull << 20;   // flush down to here (hysteresis band)
      uint64_t flush_chunk    = 10'000'000;    // keys per B txn (plateau >= 100k; Dan: big)
      uint64_t retire_batch   = 4096;          // remove_if_older per writer commit
      bool     background     = true;          // false: no merger thread (flush_now only; tests)
   };

   template <class W, class B>
   class tiered_kv
   {
      // The retire pump needs W's stamp hooks; fail loudly if W lacks them.
      static_assert(requires(typename W::transaction& t, key_view k, uint32_t v) {
                       { t.remove_if_older(k, v) } -> std::convertible_to<int>;
                    },
                    "W must expose remove_if_older(key, vmax) — the L5 retire primitive");
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
         tiered::knobs      knobs{};
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
            _b.seek_begin();
            return settle_fwd_();
         }
         bool seek_last()
         {
            if (!_w)
               return _b.seek_last();
            _w->seek_last();
            _b.seek_last();
            return settle_bwd_();
         }
         bool lower_bound(key_view k)
         {
            if (!_w)
               return _b.lower_bound(k);
            _w->lower_bound(k);
            _b.lower_bound(k);
            return settle_fwd_();
         }
         bool upper_bound(key_view k)
         {
            if (!_w)
               return _b.upper_bound(k);
            _w->upper_bound(k);
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
            if (_on_b)
               _b.prev();
            return settle_bwd_();
         }
         bool is_end() const
         {
            if (!_w)
               return _b.is_end();
            return !_rend && !_on_w && !_on_b;
         }
         bool is_rend() const { return _w ? _rend : _b.is_rend(); }
         bool is_subtree() const { return _w ? (_on_w ? false : _b.is_subtree()) : _b.is_subtree(); }
         key_view key() const
         {
            if (!_w)
               return _b.key();
            return _on_w ? _w->key() : _b.key();
         }
         template <class S = std::string>
         std::optional<S> value() const
         {
            if (!_w || !_on_w)
            {
               auto v = _b.template value<std::string>();
               if (!v)
                  return std::nullopt;
               return S(v->data(), v->size());
            }
            auto v = _w->template value<std::string>();  // enveloped; settle skipped tombs
            if (!v || v->empty())
               return std::nullopt;
            return S(v->data() + 1, v->size() - 1);
         }
         template <class F>
         bool get_value(F&& f) const
         {
            if (!_w || !_on_w)
               return _b.get_value(std::forward<F>(f));
            return _w->get_value([&](value_view v) { f(value_view(v.data() + 1, v.size() - 1)); });
         }
         // Point read: W first (tombstone = definitely-deleted, stop), then B.
         // Non-const: the base cursor REPOSITIONS on a point get (heart).
         template <class F>
         bool get(key_view k, F&& f)
         {
            if (_w)
            {
               bool hit = false, dead = false;
               _w->get(k, [&](value_view v) {
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
         cursor(typename W::cursor&& wc, typename B::cursor&& bc)        // latest
             : _w(std::move(wc)), _b(std::move(bc))
         {
         }
         bool w_tomb_() const
         {
            bool tomb = false;
            _w->get_value([&](value_view v) { tomb = v.empty() || v[0] == TAG_TOMB; });
            return tomb;
         }
         // Position state -> (_on_w, _on_b): which side(s) the current key
         // lives on. Tombstoned keys advance past both sides and re-settle.
         bool settle_fwd_()
         {
            for (;;)
            {
               const bool we = _w->is_end(), be = _b.is_end();
               _rend = false;
               if (we && be)
               {
                  _on_w = _on_b = false;
                  return false;
               }
               int c = we ? 1 : be ? -1 : cmp_(_w->key(), _b.key());
               _on_w = c <= 0;
               _on_b = c >= 0;
               if (_on_w && w_tomb_())
               {  // the tombstone masks an equal B key; skip both
                  _w->next();
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
               const bool wr = _w->is_rend() || _w->is_end();  // exhausted backward
               const bool br = _b.is_rend() || _b.is_end();
               if (wr && br)
               {
                  _on_w = _on_b = false;
                  _rend = true;
                  return false;
               }
               int c = wr ? -1 : br ? 1 : cmp_(_w->key(), _b.key());
               _on_w = c >= 0;  // backward: larger key wins
               _on_b = c <= 0;
               if (_on_w && w_tomb_())
               {
                  _w->prev();
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
         std::optional<typename W::cursor> _w;  // absent = stable view
         typename B::cursor                _b;
         bool                              _on_w = false, _on_b = false, _rend = false;
      };

      class read_session
      {
        public:
         // root_index 0 = latest (W ∪ B), 1 = stable (B only).
         cursor create_cursor(uint32_t root_index)
         {
            if (root_index == 1)
               return cursor(_brs->create_cursor(0));
            return cursor(_wrs->create_cursor(0), _brs->create_cursor(0));
         }

        private:
         friend class database;
         read_session(std::shared_ptr<typename W::read_session> w,
                      std::shared_ptr<typename B::read_session> b)
             : _wrs(std::move(w)), _brs(std::move(b))
         {
         }
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
            _db->retire_pump_(*_ws);
         }
         void abort() noexcept
         {
            _tx->abort();
            _bytes = 0;
         }
         template <class F>
         bool get(key_view k, F&& f) const
         {  // read-your-writes through the OPEN txn, then the base
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
         transaction(database* db, write_session* ws, typename W::transaction&& tx)
             : _db(db), _ws(ws), _tx(std::move(tx))
         {
         }
         database*                          _db;
         write_session*                     _ws;
         std::optional<typename W::transaction> _tx;
         std::string                        _buf;  // envelope scratch (amortized)
         uint64_t                           _bytes = 0;
      };

      class write_session
      {
        public:
         transaction start_transaction(uint32_t root_index)
         {
            return transaction(_db, this, _ws->start_transaction(root_index));
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
            return std::shared_ptr<write_session>(
                new write_session(this, _w->start_write_session()));
         }
         std::shared_ptr<read_session> start_read_session()
         {
            return std::shared_ptr<read_session>(
                new read_session(_w->start_read_session(), _b->start_read_session()));
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
            const uint32_t F   = _w->committed_version();  // BEFORE the pin: F <= pin
            auto           wrs = _w->start_read_session();
            auto           bws = _b->start_write_session();
            auto           cur = wrs->create_cursor(0);
            uint64_t       n   = 0;
            bool           more = cur.seek_begin();
            while (more)
            {
               auto     btx   = bws->start_transaction(0);
               uint64_t in_tx = 0;
               while (!cur.is_end() && in_tx < _k.flush_chunk)
               {
                  cur.get_value([&](value_view v) {
                     if (!v.empty() && v[0] == TAG_DATA)
                        btx.upsert(cur.key(), value_view(v.data() + 1, v.size() - 1));
                     else
                        btx.remove(cur.key());  // tombstone: applied, never stored in B
                  });
                  ++in_tx;
                  cur.next();
               }
               btx.commit();
               _base_epoch.fetch_add(1, std::memory_order_release);  // re-seat cached base cursors
               n += in_tx;
               more = !cur.is_end();
            }
            {
               std::lock_guard rg(_retire_m);  // pump reads resume/floor under the same lock
               _retire_floor  = F;
               _retire_resume.clear();  // pump restarts from the smallest key
               _retire_live.store(true, std::memory_order_release);
            }
            _unflushed.store(0, std::memory_order_relaxed);
            return n;
         }

        private:
         friend class transaction;
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
            _w = W::database::open(dir / "delta", mode, wcfg);
            _b = B::database::open(dir / "base", mode, cfg.bcfg);
            if (_k.background)
               _merger = std::thread([this] { merge_loop_(); });
         }
         void after_commit_(uint64_t bytes)
         {
            if (_unflushed.fetch_add(bytes, std::memory_order_relaxed) + bytes >=
                _k.hi_water_bytes)
               _cv.notify_one();
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
         // Writer-side retire pump (the writer thread calls this at commit
         // boundaries — vpart is single-writer, removes never run elsewhere).
         void retire_pump_(write_session& ws)
         {
            if (!_retire_live.load(std::memory_order_acquire))
               return;
            // collect the next sorted batch of candidate keys from a snapshot;
            // resume/floor are shared with the merger's arm step -> same lock
            uint32_t F;
            _pump_keys.clear();
            {
               std::lock_guard rg(_retire_m);
               F        = _retire_floor;
               auto rs  = _w->start_read_session();
               auto cur = rs->create_cursor(0);
               bool ok  = _retire_resume.empty() ? cur.seek_begin()
                                                 : cur.lower_bound(_retire_resume);
               if (ok && !_retire_resume.empty() && cur.key() == key_view(_retire_resume))
                  ok = cur.next();
               for (uint64_t i = 0; ok && i < _k.retire_batch; ++i)
               {
                  _pump_keys.emplace_back(cur.key());
                  ok = cur.next();
               }
               if (_pump_keys.empty())
               {
                  _retire_live.store(false, std::memory_order_release);
                  return;
               }
               _retire_resume = _pump_keys.back();
            }
            auto           tx = ws._ws->start_transaction(0);
            for (const auto& k : _pump_keys)
               tx.remove_if_older(k, F);  // rewritten-after-F keys stay (the cache band)
            tx.commit();
         }
         void merge_loop_()
         {
            for (;;)
            {
               std::unique_lock lk(_m);
               _cv.wait(lk, [&] {
                  return _stop || _unflushed.load(std::memory_order_relaxed) >= _k.hi_water_bytes;
               });
               if (_stop)
                  return;
               lk.unlock();
               flush_now();  // BURST, flat out (measured: burst beats trickle)
            }
         }

         std::shared_ptr<typename W::database> _w;
         std::shared_ptr<typename B::database> _b;
         knobs                                 _k;
         std::atomic<uint64_t>                 _unflushed{0};
         std::mutex                            _retire_m;  // floor/resume handoff (merger <-> pump)
         uint32_t                              _retire_floor = 0;
         std::atomic<bool>                     _retire_live{false};
         std::string                           _retire_resume;
         std::vector<std::string>              _pump_keys;  // writer-thread scratch
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
}  // namespace tiered

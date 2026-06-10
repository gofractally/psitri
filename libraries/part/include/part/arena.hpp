// part::arena — layer 1 of the persistent ART (SPEC §5, §7-§11).
//
// Four files in one directory, all contiguous-VA block_allocator/mapping backed:
//   data.part  segments of 128 B lines        — msync'd at commit (dirty runs only)
//   free.part  one seg_freeset per segment    — derived, never synced
//   refs.part  one u8 refcount per line       — derived, never synced
//   meta.part  header + roots + segment dir   — fsync'd at commit (root flips)
//
// Placement policy: each writer session owns one segment at a time (claimed
// under a mutex — a once-per-~8MB slow path). Within it, allocation is
// first-fit-forward from the transaction floor (the disk page where the txn
// began writing): pages behind the floor are synced & logically read-only,
// pages at/after it are dirty-anyway, so hole filling there is free (SPEC §7).
// When a segment can't satisfy a request the session releases it and claims
// the next: adjacent-continuation first, else the longest contiguous run of
// qualifying segments (free 8-line groups >= threshold — raw free lines don't
// qualify a segment, an every-other-line 50%-free segment must never be
// claimed), else the file grows.
//
// Refcounts are per LINE (not per node): u8, writers clone at 190, reader pins
// own 191..254, and the increment that would store 255 throws ref_overflow —
// counts can never overflow and nothing is ever pinned-stuck (SPEC §9).
#pragma once
#include <sys/mman.h>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <sal/block_allocator.hpp>
#include <sal/mapping.hpp>

#include <part/config.hpp>
#include <part/freeset.hpp>

namespace part
{
   struct ref_overflow : std::runtime_error
   {
      ref_overflow() : std::runtime_error("part: refcount would reach the 255 sentinel") {}
   };

   struct arena_config
   {
      std::filesystem::path dir;
      uint32_t              max_segments    = 4096;  // x 8MB = 32 GB ceiling
      double                reuse_threshold = 0.50;  // of capacity, in free 8-groups
      uint32_t              grow_batch      = 4;     // segments per file growth
      uint32_t              msync_gap_pages = 1;     // merge dirty runs across <= G clean pages
      sync_type             sync_mode       = sync_type::none;
   };

   struct arena_stats
   {
      std::atomic<uint64_t> bytes_allocated{0};
      std::atomic<uint64_t> bytes_freed{0};
      std::atomic<uint64_t> bytes_synced{0};
      std::atomic<uint64_t> sync_calls{0};
      std::atomic<uint64_t> segments_grown{0};
      std::atomic<uint64_t> segments_reused{0};
      std::atomic<uint64_t> range_switches{0};
      std::atomic<uint64_t> clones_forced{0};  // retain refused at writer_clone_at
      double write_amp() const noexcept
      {
         const auto a = bytes_allocated.load(std::memory_order_relaxed);
         return a ? double(bytes_synced.load(std::memory_order_relaxed)) / double(a) : 0.0;
      }
   };

   class arena;

   // One of <=64 reader slots (a u64 bitmap). pin/unpin bracket a snapshot's
   // lifetime; the pin IS the smart pointer for everything reachable under it.
   class read_session
   {
     public:
      read_session(read_session&& o) noexcept : _a(o._a), _slot(o._slot) { o._a = nullptr; }
      read_session(const read_session&) = delete;
      ~read_session();

      void pin(line_idx i);  // +1 in the reserved band; throws ref_overflow at 255
      template <typename Destroy>
      void unpin(line_idx i, Destroy&& d);  // exact release (may cascade via d)

     private:
      friend class arena;
      read_session(arena* a, uint32_t slot) noexcept : _a(a), _slot(slot) {}
      arena*   _a;
      uint32_t _slot;
   };

   class transaction;

   // One per writer thread. Owns the active segment claim and the txn floor.
   class session
   {
     public:
      // Not movable: start_session() returns a prvalue (guaranteed elision) and
      // outstanding transactions hold a session*.
      session(session&&)      = delete;
      session(const session&) = delete;
      ~session();

      transaction start_transaction(uint32_t root_slot = 0);

     private:
      friend class arena;
      friend class transaction;
      explicit session(arena* a) noexcept : _a(a) {}

      arena*                _a;
      uint32_t              _segment      = UINT32_MAX;  // current claim
      line_idx              _floor        = 0;           // first-fit scan start (global line)
      line_idx              _fill_end     = 0;           // one past the highest alloc this txn
      bool                  _txn_open     = false;
      std::vector<uint32_t> _touched;  // segments dirtied by the open txn
   };

   // First-class txn (harness-shaped): alloc_lines + commit/abort. The tree
   // layer composes durable commits as: txn.commit() -> arena.set_root() ->
   // release(old root) — in that order (SPEC §7).
   class transaction
   {
     public:
      transaction(transaction&& o) noexcept : _s(o._s) { o._s = nullptr; }
      transaction(const transaction&) = delete;
      ~transaction();

      line_idx alloc_lines(uint32_t n);  // 1..64, word-confined; all n refs = 1
      void     commit();
      void     abort() noexcept {}  // immediate frees make abort free (SPEC §12.1)

      arena& store() noexcept;

     private:
      friend class session;
      explicit transaction(session* s) noexcept : _s(s) {}
      session* _s;
   };

   class arena
   {
     public:
      static constexpr uint64_t meta_magic   = 0x7061727461726e61ull;  // "partarna"
      static constexpr uint32_t meta_version = 1;
      static constexpr uint32_t max_roots    = 16;
      static constexpr uint8_t  SEG_UNOWNED = 0, SEG_OWNED = 1;

      explicit arena(arena_config cfg) : _cfg(std::move(cfg))
      {
         if (uint64_t(_cfg.max_segments) * lines_per_segment > sentinel_base)
            throw std::runtime_error("part: max_segments crosses the sentinel index band");
         namespace fs = std::filesystem;
         fs::create_directories(_cfg.dir);

         _data = std::make_unique<sal::block_allocator>(_cfg.dir / "data.part", segment_bytes,
                                                        _cfg.max_segments, true);
         _free = std::make_unique<sal::block_allocator>(
             _cfg.dir / "free.part", std::bit_ceil(sizeof(seg_freeset)), _cfg.max_segments, true);
         _refs = std::make_unique<sal::block_allocator>(_cfg.dir / "refs.part",
                                                        uint64_t(lines_per_segment) /* 1B/line */,
                                                        _cfg.max_segments, true);
         _meta = std::make_unique<sal::mapping>(_cfg.dir / "meta.part",
                                                sal::access_mode::read_write);

         const size_t meta_bytes = sizeof(meta_hdr) + size_t(_cfg.max_segments) * sizeof(seg_entry);
         const bool   fresh      = _meta->size() < sizeof(meta_hdr);
         _meta->resize(meta_bytes);
         _hdr  = _meta->as<meta_hdr>();
         _segs = reinterpret_cast<seg_entry*>(reinterpret_cast<char*>(_hdr) + sizeof(meta_hdr));

         if (fresh)
         {
            _hdr->magic   = meta_magic;
            _hdr->version = meta_version;
            _hdr->seg_bytes = segment_bytes;
            _hdr->page_bytes = disk_page_bytes;
            _hdr->num_segments.store(0, std::memory_order_relaxed);
            _hdr->clean.store(1, std::memory_order_relaxed);
            for (auto& r : _hdr->roots)
               r.store(0, std::memory_order_relaxed);
         }
         else if (_hdr->magic != meta_magic || _hdr->version != meta_version ||
                  _hdr->seg_bytes != segment_bytes || _hdr->page_bytes != disk_page_bytes)
            throw std::runtime_error("part: meta geometry mismatch");

         _was_clean = _hdr->clean.load(std::memory_order_relaxed) != 0;
         _hdr->clean.store(0, std::memory_order_relaxed);

         // Re-attach existing segments (maps existing file contents).
         const uint32_t nseg = _hdr->num_segments.load(std::memory_order_relaxed);
         if (nseg)
         {
            _data->reserve(nseg);
            _free->reserve(nseg);
            _refs->reserve(nseg);
         }
         _base      = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _free_base = static_cast<std::byte*>(const_cast<void*>(_free->mapped_base()));
         _ref_base  = reinterpret_cast<std::atomic<uint8_t>*>(
             const_cast<void*>(_refs->mapped_base()));
         for (uint32_t s = 0; s < nseg; ++s)
            _segs[s].state.store(SEG_UNOWNED, std::memory_order_relaxed);
      }

      ~arena()
      {
         // Clean shutdown: derived state on disk is now trustworthy.
         if (_hdr)
         {
            _hdr->clean.store(1, std::memory_order_relaxed);
            _meta->sync(sync_type::full);
         }
      }

      bool was_clean() const noexcept { return _was_clean; }

      // ── addressing ─────────────────────────────────────────────────────────
      void* get(line_idx i) noexcept
      {
         assert(!is_sentinel(i));
         return _base + size_t(i) * line_bytes;
      }
      const void* get(line_idx i) const noexcept
      {
         assert(!is_sentinel(i));
         return _base + size_t(i) * line_bytes;
      }
      std::byte* base() const noexcept { return _base; }

      // ── refcounts (SPEC §9) ────────────────────────────────────────────────
      uint8_t ref(line_idx i) const noexcept
      {
         return _ref_base[i].load(std::memory_order_relaxed);
      }
      // State-reference retain (writers): false at >= writer_clone_at — caller
      // clones the 128B line instead (always correct under COW).
      [[nodiscard]] bool retain(line_idx i) noexcept
      {
         auto& r   = _ref_base[i];
         auto  cur = r.load(std::memory_order_relaxed);
         while (cur < writer_clone_at)
            if (r.compare_exchange_weak(cur, uint8_t(cur + 1), std::memory_order_relaxed))
               return true;
         _stats.clones_forced.fetch_add(1, std::memory_order_relaxed);
         return false;
      }
      // Returns how many of [i, i+n) were retained; < n means line i+count was
      // refused — clone exactly that one and continue (SPEC §9).
      uint32_t retain_range(line_idx i, uint32_t n) noexcept
      {
         for (uint32_t k = 0; k < n; ++k)
            if (!retain(i + k))
               return k;
         return n;
      }
      // Pin-band increment (read sessions, smart_ptr copies): throws at the
      // increment that would store the 255 sentinel.
      void add_ref_or_throw(line_idx i)
      {
         auto& r   = _ref_base[i];
         auto  cur = r.load(std::memory_order_relaxed);
         do
         {
            if (cur >= ref_ceiling)
               throw ref_overflow();
         } while (!r.compare_exchange_weak(cur, uint8_t(cur + 1), std::memory_order_relaxed));
      }
      // Exact decrement; on 1->0 runs destroy(i) while the line is still valid,
      // then frees the line immediately (crash safety is the COW contract's
      // job — SPEC §7). Returns true when the line died.
      template <typename Destroy>
      bool release(line_idx i, Destroy&& destroy) noexcept(noexcept(destroy(i)))
      {
         const uint8_t prior = _ref_base[i].fetch_sub(1, std::memory_order_acq_rel);
         assert(prior >= 1 && "release of a dead line");
         if (prior != 1)
            return false;
         destroy(i);
         free_lines_now(i, 1);
         return true;
      }
      bool release(line_idx i) noexcept
      {
         return release(i, [](line_idx) {});
      }

      // Free without refcount involvement (n contiguous lines whose counts are
      // already zero — e.g. the tail lines of a dying node, batched).
      void free_lines_now(line_idx i, uint32_t n) noexcept
      {
         const uint32_t s = i / lines_per_segment;
         freeset_(s).free_run(i % lines_per_segment, n);
         _stats.bytes_freed.fetch_add(uint64_t(n) * line_bytes, std::memory_order_relaxed);
      }

      // ── sessions ───────────────────────────────────────────────────────────
      session start_session() { return session(this); }

      read_session start_read_session()
      {
         auto cur = _read_slots.load(std::memory_order_relaxed);
         while (true)
         {
            if (cur == ~0ull)
               throw std::runtime_error("part: all 64 read-session slots in use");
            const uint32_t slot = uint32_t(std::countr_one(cur));
            if (_read_slots.compare_exchange_weak(cur, cur | (1ull << slot),
                                                  std::memory_order_acquire))
               return read_session(this, slot);
         }
      }

      // ── durable roots ──────────────────────────────────────────────────────
      void set_root(uint32_t slot, uint64_t v)
      {
         assert(slot < max_roots);
         _hdr->roots[slot].store(v, std::memory_order_release);
         if (_cfg.sync_mode != sync_type::none)
            _meta->sync(_cfg.sync_mode);
      }
      uint64_t get_root(uint32_t slot) const noexcept
      {
         assert(slot < max_roots);
         return _hdr->roots[slot].load(std::memory_order_acquire);
      }

      // ── recovery (SPEC §10) — single-threaded by contract ─────────────────
      void rebuild_begin()
      {
         const uint32_t nseg = num_segments();
         for (uint32_t s = 0; s < nseg; ++s)
            freeset_(s).init(true);
         std::memset(reinterpret_cast<void*>(_ref_base), 0,
                     size_t(nseg) * lines_per_segment);
      }
      // First mark claims the line (ref=1); later marks add a parent ref.
      void mark_line(line_idx i)
      {
         auto&   fs    = freeset_(i / lines_per_segment);
         uint32_t local = i % lines_per_segment;
         if (fs.is_free(local))
         {
            fs.claim_exact(local, 1);
            _ref_base[i].store(1, std::memory_order_relaxed);
         }
         else
         {
            const uint8_t cur = _ref_base[i].load(std::memory_order_relaxed);
            if (cur >= ref_ceiling)
               throw ref_overflow();  // a well-formed image never exceeds the bands
            _ref_base[i].store(uint8_t(cur + 1), std::memory_order_relaxed);
         }
      }
      void rebuild_end()
      {
         const uint32_t nseg = num_segments();
         for (uint32_t s = 0; s < nseg; ++s)
         {
            freeset_(s).rebuild_from_l0();
            _segs[s].state.store(SEG_UNOWNED, std::memory_order_relaxed);
            for (auto& d : _segs[s].dirty)
               d.store(0, std::memory_order_relaxed);
         }
      }

      uint32_t num_segments() const noexcept
      {
         return _hdr->num_segments.load(std::memory_order_acquire);
      }
      uint32_t    segment_of(line_idx i) const noexcept { return i / lines_per_segment; }
      seg_freeset& freeset_of(uint32_t s) noexcept { return freeset_(s); }
      const arena_stats& stats() const noexcept { return _stats; }
      const arena_config& config() const noexcept { return _cfg; }

     private:
      friend class session;
      friend class transaction;
      friend class read_session;

      struct meta_hdr
      {
         uint64_t              magic;
         uint32_t              version;
         uint32_t              seg_bytes;
         uint32_t              page_bytes;
         std::atomic<uint8_t>  clean;
         uint8_t               pad[3];
         std::atomic<uint32_t> num_segments;
         std::atomic<uint64_t> roots[max_roots];
      };
      struct seg_entry
      {
         std::atomic<uint8_t>  state;
         uint8_t               pad[7];
         std::atomic<uint64_t> dirty[pages_per_segment / 64];  // 1 bit per disk page
      };

      seg_freeset& freeset_(uint32_t s) noexcept
      {
         return *reinterpret_cast<seg_freeset*>(_free_base +
                                                size_t(s) * std::bit_ceil(sizeof(seg_freeset)));
      }
      std::atomic<uint8_t>* refs_of_(uint32_t s) noexcept
      {
         return _ref_base + size_t(s) * lines_per_segment;
      }

      uint32_t thr8_() const noexcept
      {
         return uint32_t(_cfg.reuse_threshold * (lines_per_segment / 8));
      }
      bool qualifies_(uint32_t s) noexcept
      {
         return _segs[s].state.load(std::memory_order_relaxed) == SEG_UNOWNED &&
                freeset_(s).free_groups8() >= thr8_();
      }

      // Slow path, mutex-held: adjacent continuation, else longest qualifying
      // run (tie: nearest forward of `cur`), else grow. Returns segment index.
      // `cur` (the segment that just failed to satisfy) is excluded — claiming
      // it again would loop when it qualifies by count but lacks the needed run.
      uint32_t claim_(uint32_t cur)
      {
         std::lock_guard g(_claim_mutex);
         const uint32_t  nseg = num_segments();
         if (cur != UINT32_MAX && cur + 1 < nseg && qualifies_(cur + 1))
         {
            _segs[cur + 1].state.store(SEG_OWNED, std::memory_order_relaxed);
            _stats.segments_reused.fetch_add(1, std::memory_order_relaxed);
            return cur + 1;
         }
         // longest run of qualifying segments; tie-break nearest forward of cur
         uint32_t best_start = UINT32_MAX, best_len = 0;
         uint32_t run_start = UINT32_MAX, run_len = 0;
         for (uint32_t s = 0; s < nseg; ++s)
         {
            if (s != cur && qualifies_(s))
            {
               if (run_len == 0)
                  run_start = s;
               ++run_len;
            }
            else
               run_len = 0;
            if (run_len > best_len ||
                (run_len == best_len && run_len && cur != UINT32_MAX && run_start > cur &&
                 (best_start == UINT32_MAX || best_start <= cur)))
            {
               best_len   = run_len;
               best_start = run_start;
            }
         }
         if (best_len)
         {
            _segs[best_start].state.store(SEG_OWNED, std::memory_order_relaxed);
            _stats.segments_reused.fetch_add(1, std::memory_order_relaxed);
            return best_start;
         }
         return grow_locked_();
      }

      uint32_t grow_locked_()
      {
         const uint32_t first = num_segments();
         if (first >= _cfg.max_segments)
            throw std::runtime_error("part: arena is full (max_segments)");
         const uint32_t batch = std::min(_cfg.grow_batch, _cfg.max_segments - first);
         for (uint32_t k = 0; k < batch; ++k)
         {
            _data->alloc();
            _free->alloc();
            _refs->alloc();
            const uint32_t s = first + k;
            new (&freeset_(s)) seg_freeset;
            freeset_(s).init(true);
            _segs[s].state.store(SEG_UNOWNED, std::memory_order_relaxed);
            for (auto& d : _segs[s].dirty)
               d.store(0, std::memory_order_relaxed);
            _stats.segments_grown.fetch_add(1, std::memory_order_relaxed);
         }
         _hdr->num_segments.store(first + batch, std::memory_order_release);
         _segs[first].state.store(SEG_OWNED, std::memory_order_relaxed);
         return first;
      }

      void release_claim_(uint32_t s) noexcept
      {
         if (s != UINT32_MAX)
            _segs[s].state.store(SEG_UNOWNED, std::memory_order_release);
      }

      void mark_dirty_(uint32_t seg, uint32_t line_lo, uint32_t line_hi /*incl*/) noexcept
      {
         const uint32_t p_lo = line_lo / lines_per_page, p_hi = line_hi / lines_per_page;
         for (uint32_t p = p_lo; p <= p_hi; ++p)
            _segs[seg].dirty[p / 64].fetch_or(1ull << (p % 64), std::memory_order_relaxed);
      }

      // msync the dirty page runs of `seg`, merging runs separated by <= gap
      // clean pages into one syscall. Returns bytes synced.
      uint64_t sync_segment_(uint32_t seg)
      {
         const int flags = (_cfg.sync_mode == sync_type::msync_async) ? MS_ASYNC : MS_SYNC;
         uint64_t  pages[pages_per_segment / 64];
         bool      any = false;
         for (uint32_t w = 0; w < pages_per_segment / 64; ++w)
         {
            pages[w] = _segs[seg].dirty[w].exchange(0, std::memory_order_relaxed);
            any |= pages[w] != 0;
         }
         if (!any)
            return 0;
         uint64_t synced = 0;
         std::byte* segbase = _base + size_t(seg) * segment_bytes;
         int32_t    run_start = -1;
         uint32_t   last_set  = 0;
         auto       flush = [&](uint32_t end_incl)
         {
            const size_t off = size_t(run_start) * disk_page_bytes;
            const size_t len = size_t(end_incl - uint32_t(run_start) + 1) * disk_page_bytes;
            ::msync(segbase + off, len, flags);
            _stats.sync_calls.fetch_add(1, std::memory_order_relaxed);
            synced += len;
         };
         for (uint32_t p = 0; p < pages_per_segment; ++p)
         {
            const bool d = (pages[p / 64] >> (p % 64)) & 1;
            if (!d)
               continue;
            if (run_start < 0)
               run_start = int32_t(p);
            else if (p - last_set > 1 + _cfg.msync_gap_pages)
            {
               flush(last_set);
               run_start = int32_t(p);
            }
            last_set = p;
         }
         if (run_start >= 0)
            flush(last_set);
         _stats.bytes_synced.fetch_add(synced, std::memory_order_relaxed);
         return synced;
      }

      arena_config                          _cfg;
      std::unique_ptr<sal::block_allocator> _data, _free, _refs;
      std::unique_ptr<sal::mapping>         _meta;
      meta_hdr*                             _hdr  = nullptr;
      seg_entry*                            _segs = nullptr;
      std::byte*                            _base = nullptr;
      std::byte*                            _free_base = nullptr;
      std::atomic<uint8_t>*                 _ref_base  = nullptr;
      std::mutex                            _claim_mutex;
      std::atomic<uint64_t>                 _read_slots{0};
      arena_stats                           _stats;
      bool                                  _was_clean = false;
   };

   // ── session / transaction / read_session bodies ──────────────────────────
   inline session::~session()
   {
      if (_a && _segment != UINT32_MAX)
         _a->release_claim_(_segment);
   }

   inline transaction session::start_transaction(uint32_t)
   {
      assert(!_txn_open && "one open transaction per session");
      _txn_open = true;
      // floor = the disk page the txn starts writing on (SPEC §7)
      _floor = (_fill_end / lines_per_page) * lines_per_page;
      return transaction(this);
   }

   inline transaction::~transaction()
   {
      if (_s)
         _s->_txn_open = false;
   }
   inline arena& transaction::store() noexcept
   {
      return *_s->_a;
   }

   inline line_idx transaction::alloc_lines(uint32_t n)
   {
      assert(n >= 1 && n <= max_alloc_lines);
      session& s = *_s;
      arena&   a = *s._a;
      for (int attempts = 0;; ++attempts)
      {
         if (s._segment == UINT32_MAX)
         {
            s._segment = a.claim_(UINT32_MAX);
            s._floor   = std::max(s._floor, line_idx(s._segment) * lines_per_segment);
         }
         const line_idx seg_base = line_idx(s._segment) * lines_per_segment;
         uint32_t local_from = s._floor > seg_base ? uint32_t(s._floor - seg_base) : 0u;
         if (local_from >= lines_per_segment)
            local_from = lines_per_segment;  // floor past this segment: forces a switch
         const uint32_t local = a.freeset_(s._segment).alloc(n, local_from);
         if (local != UINT32_MAX)
         {
            const line_idx g = seg_base + local;
            for (uint32_t k = 0; k < n; ++k)
               a._ref_base[g + k].store(1, std::memory_order_relaxed);
            if (a._cfg.sync_mode != sync_type::none)
            {
               a.mark_dirty_(s._segment, local, local + n - 1);
               if (s._touched.empty() || s._touched.back() != s._segment)
                  s._touched.push_back(s._segment);
            }
            if (g + n > s._fill_end)
               s._fill_end = g + n;
            a._stats.bytes_allocated.fetch_add(uint64_t(n) * line_bytes,
                                               std::memory_order_relaxed);
            return g;
         }
         // This segment can't satisfy n ahead of the floor: switch (SPEC §8).
         const uint32_t prev = s._segment;
         a.release_claim_(prev);
         s._segment = a.claim_(prev);
         s._floor   = line_idx(s._segment) * lines_per_segment;
         a._stats.range_switches.fetch_add(1, std::memory_order_relaxed);
         if (attempts > 1024)
            throw std::runtime_error("part: allocation livelock");
      }
   }

   inline void transaction::commit()
   {
      session& s = *_s;
      arena&   a = *s._a;
      if (a._cfg.sync_mode != sync_type::none)
      {
         for (uint32_t seg : s._touched)
            a.sync_segment_(seg);
         s._touched.clear();
         if (a._cfg.sync_mode == sync_type::fsync || a._cfg.sync_mode == sync_type::full)
            a._data->fsync(a._cfg.sync_mode == sync_type::full);
      }
      // Advance the floor to the page containing the fill position: the next
      // txn re-dirties that mid-commit page and may fill holes from it forward.
      s._floor = (s._fill_end / lines_per_page) * lines_per_page;
   }

   inline read_session::~read_session()
   {
      if (_a)
         _a->_read_slots.fetch_and(~(1ull << _slot), std::memory_order_release);
   }
   inline void read_session::pin(line_idx i)
   {
      _a->add_ref_or_throw(i);
   }
   template <typename Destroy>
   inline void read_session::unpin(line_idx i, Destroy&& d)
   {
      _a->release(i, static_cast<Destroy&&>(d));
   }
}  // namespace part

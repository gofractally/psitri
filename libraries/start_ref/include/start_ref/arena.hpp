// start_ref::arena — layer 1 of the persistent ART (SPEC §5, §7-§11).
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
//
// Grain-parameterized (large-value second arena): arena_t<Geo> derives all its
// size constants from Geo. line_geometry (128 B grain) reproduces today's exact
// constants verbatim — the default `arena = arena_t<line_geometry>` alias is
// byte-identical to the pre-template arena. large_geometry (32 KiB grain) is the
// same allocator at a coarser grain (max object 2 MiB). The geometry constants
// shadow the file-scope globals so the class bodies are byte-identical.
#pragma once
#include <sys/mman.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <sal/block_allocator.hpp>
#include <sal/mapping.hpp>

#include <start_ref/config.hpp>
#include <start_ref/freeset.hpp>

namespace start_ref
{
   struct ref_overflow : std::runtime_error
   {
      ref_overflow()
          : std::runtime_error("start_ref: a node has reached its limit of 254 references "
                               "(too many owned snapshots or copies of one version); release "
                               "some before taking more")
      {
      }
   };

   // The arena has reached its configured ceiling (arena_config::max_segments)
   // and no claimed segment has room. Thrown from allocation, so from any
   // write; the caller's transaction is still intact and can be aborted.
   struct arena_full : std::runtime_error
   {
      using std::runtime_error::runtime_error;
   };

   struct arena_config
   {
      std::filesystem::path dir;
      uint32_t              max_segments    = 4096;  // x 8MB = 32 GB ceiling
      double                reuse_threshold = 0.75;  // of capacity, in free 8-groups
                                                     // higher => only reuse near-empty segments
                                                     // (big contiguous runs) -> multi-line allocs
                                                     // reuse instead of growing -> less bloat
      uint32_t              grow_batch      = 4;     // segments per file growth
      uint32_t              msync_gap_pages = 1;     // merge dirty runs across <= G clean pages
      sync_type             sync_mode       = sync_type::none;
      // Units kept back from ordinary allocations once the arena can no
      // longer grow, for the allocations of operations that free more than
      // they take (a remove copies the path to its key before the old path
      // is freed). UINT32_MAX: a sixty-fourth of the arena, at least 1,024
      // and at most 8,192 units. 0: no reserve.
      uint32_t              reserve_lines   = UINT32_MAX;
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

   // Grain-parameterized: line_geometry (128 B) reproduces today's constants
   // verbatim; large_geometry (32 KiB) is the same allocator at a coarser grain.
   template <class Geo>
   class arena_t;

   // One of <=64 reader slots (a u64 bitmap). pin/unpin bracket a snapshot's
   // lifetime; the pin IS the smart pointer for everything reachable under it.
   template <class Geo>
   class read_session_t
   {
     public:
      read_session_t(read_session_t&& o) noexcept : _a(o._a), _slot(o._slot) { o._a = nullptr; }
      read_session_t(const read_session_t&) = delete;
      ~read_session_t();

      void pin(line_idx i);  // +1 in the reserved band; throws ref_overflow at 255
      template <typename Destroy>
      void unpin(line_idx i, Destroy&& d);  // exact release (may cascade via d)
      uint32_t slot() const noexcept { return _slot; }

     private:
      friend class arena_t<Geo>;
      read_session_t(arena_t<Geo>* a, uint32_t slot) noexcept : _a(a), _slot(slot) {}
      arena_t<Geo>* _a;
      uint32_t      _slot;
   };

   template <class Geo>
   class transaction_t;

   // One per writer thread. Owns the active segment claim and the txn floor.
   template <class Geo>
   class session_t
   {
      static constexpr uint32_t line_bytes        = Geo::grain_bytes;
      static constexpr uint32_t lines_per_segment = Geo::units_per_segment;
      static constexpr uint32_t lines_per_page    = Geo::units_per_page;
      using seg_freeset = seg_freeset_t<Geo::units_per_segment>;

     public:
      // Not movable: start_session() returns a prvalue (guaranteed elision) and
      // outstanding transactions hold a session*.
      session_t(session_t&&)      = delete;
      session_t(const session_t&) = delete;
      ~session_t();

      transaction_t<Geo> start_transaction(uint32_t root_slot = 0);

      // The allocation entry point (1..64 word-confined lines, all refs = 1).
      // Lives on the session (not the transaction) so a long-lived tree can
      // hold one allocator handle across many transactions; the transaction
      // object only marks the floor/commit boundaries.
      line_idx alloc_lines(uint32_t n);
      template <uint32_t N>
      [[gnu::always_inline]] inline line_idx alloc_lines_c();

      // Release a node's run through this session: per-line refcounts as
      // arena::release_run, but runs that die entirely at/after the txn floor
      // land in the session's hot-hole cache instead of the freeset.
      template <typename Destroy>
      void release_run(line_idx i, uint32_t n, Destroy&& destroy);
      void release_run(line_idx i, uint32_t n)
      {
         release_run(i, n, [](line_idx) {});
      }

      arena_t<Geo>& store() noexcept { return *_a; }
      bool          in_transaction() const noexcept { return _txn_open; }

      // While one of these is alive, this session's allocations may come out
      // of the arena's reserve. For operations that end up freeing more than
      // they allocate (removes), so that a full store can always shrink.
      class reserve_use
      {
        public:
         explicit reserve_use(session_t* s) noexcept : _s(s) { ++_s->_reserve_ok; }
         reserve_use(const reserve_use&)            = delete;
         reserve_use& operator=(const reserve_use&) = delete;
         ~reserve_use() { --_s->_reserve_ok; }

        private:
         session_t* _s;
      };
      [[nodiscard]] reserve_use use_reserve() noexcept { return reserve_use(this); }

     private:
      friend class arena_t<Geo>;
      friend class transaction_t<Geo>;
      explicit session_t(arena_t<Geo>* a) noexcept : _a(a) {}

      arena_t<Geo>*         _a;
      uint32_t              _segment      = UINT32_MAX;  // current claim
      line_idx              _floor        = 0;           // first-fit scan start (global line)
      line_idx              _cursor       = 0;           // next-scan hint (global line)
      line_idx              _fill_end     = 0;           // one past the highest alloc this txn
      uint64_t              _alloc_bytes  = 0;           // batched into arena stats at commit
      bool                  _txn_open     = false;
      bool                  _seg_idle     = false;  // _segment was marked SEG_IDLE at txn end
      uint32_t              _reserve_ok   = 0;      // live reserve_use guards
      std::vector<uint32_t> _touched;  // segments dirtied by the open txn

      // Word window: the slow path reserves the remainder of the l0 word it
      // allocated from (one CAS + one propagation per 64 lines); handouts then
      // cost zero shared-memory traffic. low_water intrusion flushes the
      // window so first-fit-from-floor placements stay exact.
      uint64_t _wmask  = 0;           // private free bits, positions within the word
      line_idx _wbase  = 0;           // global line of the window word's bit 0
      uint32_t _wseg   = UINT32_MAX;  // segment the window belongs to

      // Hot-hole cache (SPEC §7): exact-size LIFOs of runs freed AT/AFTER the
      // txn floor — their pages are dirty anyway, so recycling them is free and
      // skips the freeset round trip entirely. Flushed at commit: holes behind
      // the next floor must wait for segment recycle (two-state page model).
      static constexpr uint32_t max_reclaim_rounds = 16;  // per allocation, before giving up
      static constexpr uint32_t fcache_max_size = 16;  // run length, in lines
      std::vector<line_idx>     _fcache[fcache_max_size + 1];

      bool     cache_freed_(line_idx i, uint32_t n) noexcept;
      void     flush_fcache_() noexcept;
      line_idx take_from_window_(uint32_t n) noexcept;
      void     flush_window_() noexcept;
      line_idx alloc_slow_(uint32_t n);
      void     finish_alloc_(line_idx g, uint32_t n) noexcept;
      void     flush_stats_() noexcept;
   };

   // First-class txn (harness-shaped): alloc_lines + commit/abort. The tree
   // layer composes durable commits as: txn.commit() -> arena.set_root() ->
   // release(old root) — in that order (SPEC §7).
   template <class Geo>
   class transaction_t
   {
     public:
      transaction_t(transaction_t&& o) noexcept : _s(o._s) { o._s = nullptr; }
      transaction_t(const transaction_t&) = delete;
      ~transaction_t();
      /// The transaction is finished: its session may start another one even
      /// while this object is still alive.
      void close() noexcept;

      line_idx alloc_lines(uint32_t n);  // 1..64, word-confined; all n refs = 1
      void     commit();
      void     abort() noexcept {}  // immediate frees make abort free (SPEC §12.1)

      arena_t<Geo>& store() noexcept;

     private:
      friend class session_t<Geo>;
      explicit transaction_t(session_t<Geo>* s) noexcept : _s(s) {}
      session_t<Geo>* _s;
   };

   template <class Geo>
   class arena_t
   {
      // Grain geometry — line_geometry (default) reproduces the original
      // constants verbatim; the names shadow the file-scope globals so the
      // bodies below are byte-identical to the pre-template arena.
      static constexpr uint32_t line_bytes        = Geo::grain_bytes;
      static constexpr uint32_t segment_bytes     = Geo::seg_bytes;
      static constexpr uint32_t disk_page_bytes   = Geo::page_bytes;
      static constexpr uint32_t lines_per_segment = Geo::units_per_segment;
      static constexpr uint32_t pages_per_segment = Geo::pages_per_segment_;
      static constexpr uint32_t lines_per_page    = Geo::units_per_page;
      using seg_freeset = seg_freeset_t<Geo::units_per_segment>;

     public:
      static constexpr uint64_t meta_magic   = 0x7061727461726e61ull;  // "partarna"
      static constexpr uint32_t meta_version = 1;
      static constexpr uint32_t max_roots    = 16;
      // OWNED: a session is allocating in it. IDLE: a session holds it but
      // is between transactions; it keeps the segment for its next one
      // unless, at the ceiling, another session needs it first.
      static constexpr uint8_t  SEG_UNOWNED = 0, SEG_OWNED = 1, SEG_IDLE = 2;

      explicit arena_t(arena_config cfg) : _cfg(std::move(cfg))
      {
         if (_cfg.reserve_lines == UINT32_MAX)
            _cfg.reserve_lines = uint32_t(std::clamp<uint64_t>(
                uint64_t(_cfg.max_segments) * lines_per_segment / 64, 1024, 8192));
         if (uint64_t(_cfg.max_segments) * lines_per_segment > sentinel_base)
            throw std::runtime_error("part: max_segments crosses the sentinel index band");
         namespace fs = std::filesystem;
         fs::create_directories(_cfg.dir);

         _data = std::make_unique<sal::block_allocator>(_cfg.dir / "data.part", segment_bytes,
                                                        _cfg.max_segments,
                                                        sal::access_mode::read_write);
         _free = std::make_unique<sal::block_allocator>(_cfg.dir / "free.part",
                                                        std::bit_ceil(sizeof(seg_freeset)),
                                                        _cfg.max_segments,
                                                        sal::access_mode::read_write);
         _refs = std::make_unique<sal::block_allocator>(_cfg.dir / "refs.part",
                                                        uint64_t(lines_per_segment) /* 1B/line */,
                                                        _cfg.max_segments,
                                                        sal::access_mode::read_write);
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

         _was_clean  = _hdr->clean.load(std::memory_order_relaxed) != 0;
         _derived_ok = _was_clean;
         _hdr->clean.store(0, std::memory_order_relaxed);
         // The free map and reference counts are never synced, so after a
         // power loss they are only trustworthy if the "open" mark got to
         // storage before anything changed them. Only stores that asked for
         // synchronization pay for that.
         if (_cfg.sync_mode != sync_type::none && _cfg.sync_mode != sync_type::mprotect)
            _meta->sync(_cfg.sync_mode);

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
         if (getenv("PART_PRINT_REFBASE"))
            fprintf(stderr, "REFBASE %p DATABASE %p\n", (void*)_ref_base, (void*)_base);
      }

      ~arena_t()
      {
         // Clean shutdown: derived state on disk is trustworthy, provided it
         // was when this arena opened or has been rebuilt since. An arena
         // that was found unclean and not rebuilt (recovery failed, or the
         // owner gave up part-way through opening) stays marked unclean.
         if (_hdr)
         {
            // The free map and reference counts are not flushed by commits.
            // A store that asked for synchronization must not carry a clean
            // mark on storage ahead of them: after a power loss the next open
            // would trust maps that never arrived.
            // The mark below is always synced, whatever the mode, so the
            // maps are always written out first, and synchronously: an
            // asynchronous request could still let the mark arrive first.
            if (_derived_ok && _free && _refs)
            {
               const sync_type st = std::max(_cfg.sync_mode, sync_type::msync_sync);
               _free->sync(st);
               _refs->sync(st);
            }
            _hdr->clean.store(_derived_ok ? 1 : 0, std::memory_order_relaxed);
            _meta->sync(sync_type::full);
         }
      }

      bool was_clean() const noexcept { return _was_clean; }

      // Whether the arena in `dir` was closed by its destructor, read from its
      // header without mapping or changing anything. Empty when there is no
      // arena there (or not one this build can read).
      static std::optional<bool> read_clean_flag(const std::filesystem::path& dir)
      {
         std::FILE* f = std::fopen((dir / "meta.part").c_str(), "rb");
         if (!f)
            return std::nullopt;
         alignas(meta_hdr) unsigned char raw[sizeof(meta_hdr)];
         const size_t got = std::fread(raw, 1, sizeof(raw), f);
         std::fclose(f);
         uint64_t magic;
         std::memcpy(&magic, raw, sizeof(magic));
         if (got != sizeof(raw) || magic != meta_magic)
            return std::nullopt;
         return raw[offsetof(meta_hdr, clean)] != 0;
      }

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
         if (long(i) == trace_line_())
            fprintf(stderr, "TRACE line %u retain from=%u ra0=%p ra1=%p ra2=%p\n", i, cur,
                    __builtin_return_address(0), __builtin_return_address(1),
                    __builtin_return_address(2));
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
         if (long(i) == trace_line_())
            fprintf(stderr, "TRACE line %u pin from=%u\n", i, cur);
         do
         {
            if (cur >= ref_ceiling)
               throw ref_overflow();
         } while (!r.compare_exchange_weak(cur, uint8_t(cur + 1), std::memory_order_relaxed));
      }
      // Exact decrement; on 1->0 runs destroy(i) while the line is still valid,
      // then frees the line immediately (crash safety is the COW contract's
      // job — SPEC §7). Returns true when the line died.
      // PART_TRACE_LINE=<n>: stderr trace of one line's count lifecycle (debug)
      static long trace_line_()
      {
         static long v = [] {
            const char* e = getenv("PART_TRACE_LINE");
            return e ? atol(e) : -1;
         }();
         return v;
      }
      template <typename Destroy>
      bool release(line_idx i, Destroy&& destroy) noexcept(noexcept(destroy(i)))
      {
         const uint8_t prior = _ref_base[i].fetch_sub(1, std::memory_order_acq_rel);
         if (long(i) == trace_line_())
            fprintf(stderr, "TRACE line %u release prior=%u\n", i, prior);
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

      // Release n contiguous lines (one allocation's run): per-line counts, but
      // contiguous lines that die together are freed with ONE freeset op — the
      // common exclusive-node free costs one CAS + one propagation, not n.
      template <typename Destroy>
      void release_run(line_idx i, uint32_t n, Destroy&& destroy) noexcept(noexcept(destroy(i)))
      {
         uint32_t run_start = 0, run_len = 0;
         for (uint32_t k = 0; k < n; ++k)
         {
            const uint8_t prior = _ref_base[i + k].fetch_sub(1, std::memory_order_acq_rel);
            if (long(i + k) == trace_line_())
               fprintf(stderr, "TRACE line %u release_run prior=%u\n", i + k, prior);
            assert(prior >= 1 && "release of a dead line");
            if (prior == 1)
            {
               destroy(i + k);
               if (run_len == 0)
                  run_start = k;
               ++run_len;
            }
            else if (run_len)
            {
               free_lines_now(i + run_start, run_len);
               run_len = 0;
            }
         }
         if (run_len)
            free_lines_now(i + run_start, run_len);
      }
      void release_run(line_idx i, uint32_t n) noexcept
      {
         release_run(i, n, [](line_idx) {});
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
      session_t<Geo> start_session() { return session_t<Geo>(this); }

      std::atomic<uint64_t>& reader_slot(uint32_t i) noexcept { return _reader_roots[i]; }
      bool root_published(uint64_t raw) const noexcept
      {
         for (uint32_t i = 0; i < max_read_sessions; ++i)
            if (_reader_roots[i].load(std::memory_order_seq_cst) == raw)
               return true;
         return false;
      }

      read_session_t<Geo> start_read_session()
      {
         auto cur = _read_slots.load(std::memory_order_relaxed);
         while (true)
         {
            if (cur == ~0ull)
               throw std::runtime_error(
                   "start_ref: all 64 reader slots are in use (each read session holds one, "
                   "and so does each cursor a session has open beyond its first)");
            const uint32_t slot = uint32_t(std::countr_one(cur));
            if (_read_slots.compare_exchange_weak(cur, cur | (1ull << slot),
                                                  std::memory_order_acquire))
               return read_session_t<Geo>(this, slot);
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
         _derived_ok = false;  // until rebuild_end: a rebuild that stops half-way is not clean
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
      // Recovery marking for one allocation of n units whose head is i. The
      // first mark claims all n (each with one reference) and returns true;
      // a later mark is another reference to the head and returns false.
      // Throws on an index or run the allocator could not have produced:
      // that is a damaged image, not a state to guess at.
      void check_index(uint64_t i, uint32_t n = 1) const
      {
         if (n == 0 || n > max_alloc_lines ||
             i + n > uint64_t(num_segments()) * lines_per_segment || (i % 64) + n > 64)
            throw std::runtime_error("part: recovery reached storage outside the arena");
      }
      bool mark_run(line_idx i, uint32_t n)
      {
         check_index(i, n);
         auto&          fs    = freeset_(i / lines_per_segment);
         const uint32_t local = i % lines_per_segment;
         if (!fs.is_free(local))
         {
            const uint8_t cur = _ref_base[i].load(std::memory_order_relaxed);
            if (cur == 0 || cur >= ref_ceiling)
               throw std::runtime_error("part: recovery found an impossible reference count");
            _ref_base[i].store(uint8_t(cur + 1), std::memory_order_relaxed);
            return false;
         }
         claim_run(i, n);
         return true;
      }
      // Claim n units that nothing has claimed yet, one reference each.
      void claim_run(line_idx i, uint32_t n)
      {
         check_index(i, n);
         auto&          fs    = freeset_(i / lines_per_segment);
         const uint32_t local = i % lines_per_segment;
         for (uint32_t k = 0; k < n; ++k)
            if (!fs.is_free(local + k))
               throw std::runtime_error("part: recovery found two nodes on one line");
         fs.claim_exact(local, n);
         for (uint32_t k = 0; k < n; ++k)
            _ref_base[i + k].store(1, std::memory_order_relaxed);
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
         _derived_ok = true;
      }

      uint32_t num_segments() const noexcept
      {
         return _hdr->num_segments.load(std::memory_order_acquire);
      }
      uint32_t    segment_of(line_idx i) const noexcept { return i / lines_per_segment; }
      // Space accounting, in bytes. Reads one counter per segment; safe from any
      // thread while writers run (the sum is then a moment-in-time estimate).
      // "Used" counts every claimed unit: live nodes, nodes of retained
      // snapshots, nodes waiting on the releaser, and the few units a write
      // session holds privately between commits.
      uint64_t used_bytes() const noexcept
      {
         const uint32_t nseg  = num_segments();
         uint64_t       freeU = 0;
         for (uint32_t s = 0; s < nseg; ++s)
            freeU += const_cast<arena_t*>(this)->freeset_(s).free_lines();
         return (uint64_t(nseg) * lines_per_segment - freeU) * line_bytes;
      }
      // Segments are never returned to the OS: this is the high-water mark.
      uint64_t mapped_bytes() const noexcept { return uint64_t(num_segments()) * segment_bytes; }
      uint64_t limit_bytes() const noexcept { return uint64_t(_cfg.max_segments) * segment_bytes; }
      uint64_t reserve_bytes() const noexcept { return uint64_t(_cfg.reserve_lines) * line_bytes; }
      seg_freeset& freeset_of(uint32_t s) noexcept { return freeset_(s); }
      const arena_stats& stats() const noexcept { return _stats; }
      const void* refs_base_dbg() const noexcept { return _ref_base; }
      // background-release hook (SPEC §19): the kv layer registers a push-to-
      // ring callback; cow_release routes release-to-zero CASCADES through it
      using defer_fn = bool (*)(void*, uint64_t);  // raw is the full (now 40-bit) handle
      void set_defer(defer_fn f, void* ctx) noexcept
      {
         _defer_fn  = f;
         _defer_ctx = ctx;
      }
      bool try_defer(uint64_t raw) noexcept
      {
         return _defer_fn && _defer_fn(_defer_ctx, raw);
      }
      const arena_config& config() const noexcept { return _cfg; }
      // debug/tests: a segment's claim state
      uint8_t segment_state(uint32_t s) const noexcept
      {
         return _segs[s].state.load(std::memory_order_relaxed);
      }

      // Last-chance hook before an allocation reports arena_full: the owner
      // may know of storage that is about to be freed (the kv layer's
      // background releaser has a queue of dead subtrees). The hook waits for
      // that work and returns true if there was any, in which case the
      // allocation is tried again. Called on the allocating thread with no
      // arena lock held.
      using reclaim_fn = bool (*)(void*);
      void set_reclaim(reclaim_fn f, void* ctx) noexcept
      {
         _reclaim_fn  = f;
         _reclaim_ctx = ctx;
      }
      bool try_reclaim() noexcept { return _reclaim_fn && _reclaim_fn(_reclaim_ctx); }

      // Companion link to the large-value arena (the 32 KiB-grain second instance).
      // The line arena holds it so view/drain allocators (which carry only an
      // arena*) can resolve and free value_large leaves; null when no large
      // arena is attached. Stored type-erased: arena_t<line_geometry> cannot
      // name arena_t<large_geometry> without an instantiation cycle.
      void  set_companion(void* large) noexcept { _companion = large; }
      void* companion() const noexcept { return _companion; }

     private:
      friend class session_t<Geo>;
      friend class transaction_t<Geo>;
      friend class read_session_t<Geo>;

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

      bool all_refs_one_(line_idx i, uint32_t n) const noexcept
      {
         for (uint32_t k = 0; k < n; ++k)
            if (_ref_base[i + k].load(std::memory_order_relaxed) != 1)
               return false;
         return true;
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
      //
      // When the arena cannot grow any further the reuse threshold stops being
      // useful: it exists to keep writes in mostly-empty segments, and would
      // otherwise declare the store full while most of its lines are free but
      // scattered. So, at the ceiling, `sweep` (or simply finding nothing that
      // qualifies) takes ANY unowned segment with at least n free lines,
      // going round from the one after `cur` and ending with `cur` itself, so
      // that repeated calls visit every segment once per lap. Throws
      // arena_full when no segment has n free lines.
      uint32_t claim_(uint32_t cur, bool prefer_grow = false, uint32_t n = 1, bool sweep = false)
      {
         std::lock_guard g(_claim_mutex);
         const uint32_t  nseg = num_segments();
         const bool      can_grow = nseg < _cfg.max_segments;
         if (sweep && !can_grow)
            return sweep_locked_(cur, n);
         // Anti-livelock: a large (multi-line) alloc that keeps failing on
         // "qualifying" but FRAGMENTED reuse segments (free lines exist but no
         // contiguous run) must grow a fresh segment to make progress, rather
         // than cycling reuse candidates forever. The caller flips this on after
         // a few failed switches.
         if (prefer_grow && nseg < _cfg.max_segments)
            return grow_locked_();
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
         if (can_grow)
            return grow_locked_();
         return sweep_locked_(cur, n);
      }

      // At the ceiling: the next unowned segment after `cur` (wrapping, `cur`
      // last) that has n free lines at all.
      uint32_t sweep_locked_(uint32_t cur, uint32_t n)
      {
         const uint32_t nseg  = num_segments();
         const uint32_t start = cur == UINT32_MAX ? 0 : cur + 1;
         for (uint32_t k = 0; k < nseg; ++k)
         {
            const uint32_t s = (start + k) % nseg;
            if (freeset_(s).free_lines() < n)
               continue;
            // An idle session's segment may be taken: its owner notices at
            // its next transaction and claims afresh. (Its owner may also be
            // starting a transaction this instant, hence the exchange.)
            uint8_t state = _segs[s].state.load(std::memory_order_relaxed);
            if (state != SEG_OWNED &&
                _segs[s].state.compare_exchange_strong(state, SEG_OWNED,
                                                       std::memory_order_acq_rel))
            {
               _stats.segments_reused.fetch_add(1, std::memory_order_relaxed);
               return s;
            }
         }
         throw_full_();
      }
      // Whether at least `need` units are free across the arena. Stops at
      // the first segments that add up, so it is cheap unless the arena is
      // nearly full.
      bool free_at_least_(uint64_t need) noexcept
      {
         const uint32_t nseg = num_segments();
         uint64_t       sum  = 0;
         for (uint32_t s = 0; s < nseg; ++s)
            if ((sum += freeset_(s).free_lines()) >= need)
               return true;
         return false;
      }
      [[gnu::cold, noreturn]] void throw_full_() const
      {
         throw arena_full("start_ref: storage limit reached: all " +
                          std::to_string(_cfg.max_segments) + " segments (max_segments, " +
                          std::to_string(uint64_t(_cfg.max_segments) * segment_bytes /
                                         (1024 * 1024)) +
                          " MiB) of " + _cfg.dir.generic_string() +
                          " are in use and none has room for this allocation");
      }

      uint32_t grow_locked_()
      {
         const uint32_t first = num_segments();
         if (first >= _cfg.max_segments)
            throw_full_();
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
      defer_fn                              _defer_fn  = nullptr;
      void*                                 _defer_ctx = nullptr;
      reclaim_fn                            _reclaim_fn  = nullptr;
      void*                                 _reclaim_ctx = nullptr;
      std::byte*                            _free_base = nullptr;
      std::atomic<uint8_t>*                 _ref_base  = nullptr;
      std::mutex                            _claim_mutex;
      std::atomic<uint64_t>                 _read_slots{0};
      // Hazard-pointer table (sal's reader paradigm): slot i publishes the
      // root the reader currently traverses; readers never touch refcounts or
      // locks, and the WRITER defers retired-root release until no slot shows
      // it. Transient: rebuilt empty every open.
      std::atomic<uint64_t> _reader_roots[max_read_sessions] = {};
      arena_stats                           _stats;
      bool                                  _was_clean = false;
      bool                                  _derived_ok = false;  // free map + counts may be marked clean at close
      void*                                 _companion = nullptr;  // -> large_arena (type-erased)
   };

   // ── session / transaction / read_session bodies ──────────────────────────
   template <class Geo>
   inline session_t<Geo>::~session_t()
   {
      if (!_a)
         return;
      flush_fcache_();
      flush_window_();
      flush_stats_();
      if (_segment != UINT32_MAX)
      {
         // Give the segment back only if it is still ours: an idle one may
         // have been taken at the ceiling.
         uint8_t expected = _seg_idle ? arena_t<Geo>::SEG_IDLE : arena_t<Geo>::SEG_OWNED;
         _a->_segs[_segment].state.compare_exchange_strong(expected, arena_t<Geo>::SEG_UNOWNED,
                                                           std::memory_order_acq_rel);
      }
   }

   template <class Geo>
   inline transaction_t<Geo> session_t<Geo>::start_transaction(uint32_t)
   {
      assert(!_txn_open && "one open transaction per session");
      _txn_open = true;
      if (_seg_idle)
      {
         // Take the segment back from idle; if another session needed it
         // meanwhile, claim afresh at the first allocation.
         _seg_idle        = false;
         uint8_t expected = arena_t<Geo>::SEG_IDLE;
         if (!_a->_segs[_segment].state.compare_exchange_strong(
                 expected, arena_t<Geo>::SEG_OWNED, std::memory_order_acq_rel))
            _segment = UINT32_MAX;
      }
      // floor = the disk page the txn starts writing on (SPEC §7)
      _floor = (_fill_end / lines_per_page) * lines_per_page;
      return transaction_t<Geo>(this);
   }

   template <class Geo>
   inline transaction_t<Geo>::~transaction_t()
   {
      close();
   }
   template <class Geo>
   inline void transaction_t<Geo>::close() noexcept
   {
      if (_s)
      {
         _s->_txn_open = false;
         if (_s->_segment != UINT32_MAX && !_s->_seg_idle)
         {
            // A plain store: while the session is in a transaction the
            // segment is OWNED by it and nobody else may change that.
            _s->_a->_segs[_s->_segment].state.store(arena_t<Geo>::SEG_IDLE,
                                                    std::memory_order_release);
            _s->_seg_idle = true;
         }
      }
      _s = nullptr;
   }
   template <class Geo>
   inline arena_t<Geo>& transaction_t<Geo>::store() noexcept
   {
      return *_s->_a;
   }

   template <class Geo>
   inline line_idx transaction_t<Geo>::alloc_lines(uint32_t n)
   {
      return _s->alloc_lines(n);
   }

   // bit i set iff `v` has n consecutive set bits starting at i (in-word).
   inline constexpr uint64_t runs_in_word_(uint64_t v, uint32_t n) noexcept
   {
      uint64_t r   = v;
      uint32_t got = 1;
      while (got * 2 <= n)
      {
         r &= r >> got;
         got *= 2;
      }
      if (got < n)
         r &= r >> (n - got);
      return r;
   }

   template <class Geo>
   inline void session_t<Geo>::finish_alloc_(line_idx g, uint32_t n) noexcept
   {
      arena_t<Geo>& a = *_a;
      if (arena_t<Geo>::trace_line_() >= 0 && uint32_t(arena_t<Geo>::trace_line_()) >= g &&
          uint32_t(arena_t<Geo>::trace_line_()) < g + n)
         fprintf(stderr, "TRACE line %ld alloc (run %u+%u)\n", arena_t<Geo>::trace_line_(), g, n);
      for (uint32_t k = 0; k < n; ++k)
         a._ref_base[g + k].store(1, std::memory_order_relaxed);
      if (a._cfg.sync_mode != sync_type::none)
      {
         // derive the segment from g: cache hits can land in a previously
         // claimed segment after a switch
         const uint32_t seg   = g / lines_per_segment;
         const uint32_t local = g % lines_per_segment;
         a.mark_dirty_(seg, local, local + n - 1);
         if (_touched.empty() || _touched.back() != seg)
            _touched.push_back(seg);
      }
      if (g + n > _fill_end)
         _fill_end = g + n;
      _cursor = g + n;
      _alloc_bytes += uint64_t(n) * line_bytes;
   }

   template <class Geo>
   template <typename Destroy>
   inline void session_t<Geo>::release_run(line_idx i, uint32_t n, Destroy&& destroy)
   {
      if (n <= fcache_max_size && i >= _floor && _a->all_refs_one_(i, n))
      {
         for (uint32_t k = 0; k < n; ++k)
         {
            if (long(i + k) == arena_t<Geo>::trace_line_())
               fprintf(stderr, "TRACE line %u ses_release_run(1->0)\n", i + k);
            _a->_ref_base[i + k].store(0, std::memory_order_release);
            destroy(i + k);
         }
         if (cache_freed_(i, n))
            return;
         _a->free_lines_now(i, n);
         return;
      }
      _a->release_run(i, n, static_cast<Destroy&&>(destroy));
   }

   template <class Geo>
   inline bool session_t<Geo>::cache_freed_(line_idx i, uint32_t n) noexcept
   {
      auto& v = _fcache[n];
      if (v.size() >= 4096)
         return false;
      v.push_back(i);
      return true;
   }

   template <class Geo>
   inline void session_t<Geo>::flush_fcache_() noexcept
   {
      for (uint32_t n = 1; n <= fcache_max_size; ++n)
      {
         for (line_idx i : _fcache[n])
            _a->free_lines_now(i, n);
         _fcache[n].clear();
      }
   }

   template <class Geo>
   inline line_idx session_t<Geo>::take_from_window_(uint32_t n) noexcept
   {
      const uint64_t r = n == 1 ? _wmask : runs_in_word_(_wmask, n);
      if (!r)
         return null_idx;
      const uint32_t pos = uint32_t(std::countr_zero(r));
      _wmask &= ~(((n >= 64 ? ~0ull : ((1ull << n) - 1))) << pos);
      return _wbase + pos;
   }

   template <class Geo>
   inline void session_t<Geo>::flush_window_() noexcept
   {
      if (!_wmask)
      {
         _wseg = UINT32_MAX;
         return;
      }
      // Return unused reserved lines to the freeset (one op).
      seg_freeset&   fs    = _a->freeset_(_wseg);
      const uint32_t local = uint32_t(_wbase % lines_per_segment);
      const uint32_t w     = local / 64;
      const uint64_t prior = fs.l0[w].fetch_or(_wmask, std::memory_order_release);
      assert((prior & _wmask) == 0);
      fs.apply_word_transition(w, prior, prior | _wmask);
      auto lw = fs.low_water.load(std::memory_order_relaxed);
      const uint32_t lo = local + uint32_t(std::countr_zero(_wmask));
      while (lo < lw && !fs.low_water.compare_exchange_weak(lw, lo, std::memory_order_relaxed))
         ;
      _wmask = 0;
      _wseg  = UINT32_MAX;
   }

   template <class Geo>
   inline line_idx session_t<Geo>::alloc_lines(uint32_t n)
   {
      assert(n >= 1 && n <= max_alloc_lines);
      // Hot-hole cache: exact-size recycle of this-txn frees (dirty pages).
      if (n <= fcache_max_size && !_fcache[n].empty())
      {
         const line_idx g = _fcache[n].back();
         _fcache[n].pop_back();
         finish_alloc_(g, n);
         _alloc_bytes -= uint64_t(n) * line_bytes;  // reuse, not new allocation
         return g;
      }
      // Window fast path: zero shared-memory traffic. A free announced below
      // the window's next handout flushes back to the exact first-fit slow
      // path — UNLESS it is below the txn floor: those holes are unreachable
      // until segment recycle (two-state page model) and must not defeat the
      // window. Old-generation cascades from snapshot releases land there.
      if (_wmask && _wseg == _segment)
      {
         seg_freeset&   fs = _a->freeset_(_segment);
         const uint32_t lw = fs.low_water.load(std::memory_order_relaxed);
         const line_idx seg_base = line_idx(_segment) * lines_per_segment;
         const uint32_t floor_local =
             _floor > seg_base ? uint32_t(_floor - seg_base) : 0u;
         const uint32_t next_local =
             uint32_t(_wbase % lines_per_segment) + uint32_t(std::countr_zero(_wmask));
         if (lw >= next_local || lw < floor_local)
         {
            const line_idx g = take_from_window_(n);
            if (g != null_idx)
            {
               finish_alloc_(g, n);
               return g;
            }
         }
         else
            flush_window_();
      }
      return alloc_slow_(n);
   }

   template <class Geo>
   template <uint32_t N>
   [[gnu::always_inline]] inline line_idx session_t<Geo>::alloc_lines_c()
   {
      static_assert(N >= 1 && N <= max_alloc_lines);
      if (N <= fcache_max_size && !_fcache[N].empty())
      {
         const line_idx g = _fcache[N].back();
         _fcache[N].pop_back();
         finish_alloc_(g, N);
         _alloc_bytes -= uint64_t(N) * line_bytes;
         return g;
      }
      if (_wmask && _wseg == _segment)
      {
         seg_freeset&   fs       = _a->freeset_(_segment);
         const uint32_t lw       = fs.low_water.load(std::memory_order_relaxed);
         const line_idx seg_base = line_idx(_segment) * lines_per_segment;
         const uint32_t floor_local =
             _floor > seg_base ? uint32_t(_floor - seg_base) : 0u;
         const uint32_t next_local =
             uint32_t(_wbase % lines_per_segment) + uint32_t(std::countr_zero(_wmask));
         if (lw >= next_local || lw < floor_local)
         {
            const line_idx g = take_from_window_(N);
            if (g != null_idx)
            {
               finish_alloc_(g, N);
               return g;
            }
         }
         else
            flush_window_();
      }
      return alloc_slow_(N);
   }

   template <class Geo>
   inline line_idx session_t<Geo>::alloc_slow_(uint32_t n)
   {
      session_t<Geo>& s = *this;
      arena_t<Geo>&   a = *s._a;
      if (s._wmask)
         s.flush_window_();
      // Claiming can end in arena_full. Before letting that out, give the
      // owner's reclaim hook one chance per round to finish frees that are
      // already queued; only if it had nothing to do is the arena full.
      // The hook reporting work does not mean this allocation can now be
      // met (another writer may keep the releaser busy for ever while n
      // lines in a row exist nowhere), so the number of rounds is bounded.
      uint32_t   reclaims       = 0;
      const auto reclaim_helped = [&]() noexcept
      {
         return reclaims++ < max_reclaim_rounds && a.try_reclaim();
      };
      auto claim = [&](uint32_t cur, bool prefer_grow, bool sweep) -> uint32_t
      {
         for (;;)
         {
            try
            {
               return a.claim_(cur, prefer_grow, n, sweep);
            }
            catch (const arena_full&)
            {
               if (!reclaim_helped())
                  throw;
            }
         }
      };
      // Once the arena cannot grow, ordinary allocations leave the reserve
      // alone. (The few lines a session already holds privately are outside
      // this accounting; the reserve is far larger than those.)
      if (a._cfg.reserve_lines && !s._reserve_ok &&
          a.num_segments() >= a._cfg.max_segments)
         while (!a.free_at_least_(uint64_t(a._cfg.reserve_lines) + n))
            if (!reclaim_helped())
               a.throw_full_();
      uint32_t swept = 0;  // segments tried in sweep mode since the last reclaim
      for (int attempts = 0;; ++attempts)
      {
         if (s._segment == UINT32_MAX)
         {
            s._segment = claim(UINT32_MAX, false, false);
            const line_idx base = line_idx(s._segment) * lines_per_segment;
            // A segment claimed afresh is written from its start: the floor
            // only ever applied to the segment it was set in.
            s._floor  = base;
            s._cursor = base;
         }
         const line_idx seg_base = line_idx(s._segment) * lines_per_segment;
         seg_freeset&   fs       = a.freeset_(s._segment);
         uint32_t local_floor = s._floor > seg_base ? uint32_t(s._floor - seg_base) : 0u;
         if (local_floor >= lines_per_segment)
            local_floor = lines_per_segment;  // floor past this segment: forces a switch
         // First-fit-from-floor without the rescan: holes freed behind the
         // cursor announce themselves via low_water, so the scan starts at
         // min(cursor, low_water) — same placements, bump-like cost.
         const uint32_t cur_local =
             s._cursor > seg_base ? uint32_t(s._cursor - seg_base) : 0u;
         const uint32_t lw = fs.low_water.exchange(UINT32_MAX, std::memory_order_relaxed);
         const uint32_t local_from = std::max(local_floor, std::min(cur_local, lw));
         const uint32_t local      = fs.alloc(n, local_from);
         if (local != UINT32_MAX)
         {
            // Put the mark back only when the hole is reachable this txn
            // (>= floor); below-floor marks are noise until segment recycle.
            if (lw != UINT32_MAX && lw >= local_floor && local > lw)
            {
               auto cur = fs.low_water.load(std::memory_order_relaxed);
               while (lw < cur && !fs.low_water.compare_exchange_weak(
                                      cur, lw, std::memory_order_relaxed))
                  ;
            }
            const line_idx g = seg_base + local;
            // Reserve the remainder of this word into the private window: the
            // bits ABOVE the allocation (bits below it were skipped by the
            // first-fit scan for a reason and stay public).
            {
               const uint32_t w   = local / 64;
               const uint32_t hi  = (local % 64) + n;
               if (hi < 64)
               {
                  const uint64_t above = ~0ull << hi;
                  uint64_t       cur   = fs.l0[w].load(std::memory_order_relaxed);
                  uint64_t       want  = cur & above;
                  if (want &&
                      fs.l0[w].compare_exchange_strong(cur, cur & ~want,
                                                       std::memory_order_acquire,
                                                       std::memory_order_relaxed))
                  {
                     fs.apply_word_transition(w, cur, cur & ~want);
                     _wmask = want;
                     _wbase = seg_base + w * 64;
                     _wseg  = s._segment;
                  }
               }
            }
            finish_alloc_(g, n);
            return g;
         }
         // This segment can't satisfy n ahead of the floor: switch (SPEC §8).
         // After a few failed reuse switches the qualifying segments are too
         // fragmented to hold this (multi-line) alloc — grow a fresh segment so a
         // large alloc always makes progress instead of livelocking at scale.
         // At the ceiling there is nothing to grow: go round every segment
         // once instead, and if none can hold n lines in a row the arena is
         // full for this allocation.
         const uint32_t prev = s._segment;
         a.release_claim_(prev);
         s._segment = UINT32_MAX;  // nothing is claimed if the next claim throws
         const bool at_ceiling = a.num_segments() >= a._cfg.max_segments;
         const bool sweep      = at_ceiling && attempts >= 8;
         if (sweep && ++swept > a.num_segments())
         {
            if (!reclaim_helped())
               a.throw_full_();
            swept = 0;
         }
         s._segment = claim(prev, /*prefer_grow=*/attempts >= 8, sweep);
         s._floor   = line_idx(s._segment) * lines_per_segment;
         s._cursor  = s._floor;
         a._stats.range_switches.fetch_add(1, std::memory_order_relaxed);
         if (!at_ceiling && attempts > 1024)
            throw std::runtime_error("part: allocation livelock");
      }
   }

   template <class Geo>
   inline void session_t<Geo>::flush_stats_() noexcept
   {
      if (_alloc_bytes)
      {
         _a->_stats.bytes_allocated.fetch_add(_alloc_bytes, std::memory_order_relaxed);
         _alloc_bytes = 0;
      }
   }

   template <class Geo>
   inline void transaction_t<Geo>::commit()
   {
      session_t<Geo>& s = *_s;
      arena_t<Geo>&   a = *s._a;
      // The window/cache may hold lines behind the post-commit floor; returning
      // them keeps "writes never land behind the floor" exact for the next txn.
      s.flush_fcache_();
      s.flush_window_();
      s.flush_stats_();
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
      s._floor = (s._fill_end / Geo::units_per_page) * Geo::units_per_page;
   }

   template <class Geo>
   inline read_session_t<Geo>::~read_session_t()
   {
      if (_a)
         _a->_read_slots.fetch_and(~(1ull << _slot), std::memory_order_release);
   }
   template <class Geo>
   inline void read_session_t<Geo>::pin(line_idx i)
   {
      _a->add_ref_or_throw(i);
   }
   template <class Geo>
   template <typename Destroy>
   inline void read_session_t<Geo>::unpin(line_idx i, Destroy&& d)
   {
      _a->release(i, static_cast<Destroy&&>(d));
   }

   // ── default (128 B grain) aliases — byte-identical to the pre-template API ──
   // Every existing user (kv.hpp, tree_alloc.hpp, tests, benches) names these.
   using arena        = arena_t<line_geometry>;
   using session      = session_t<line_geometry>;
   using transaction  = transaction_t<line_geometry>;
   using read_session = read_session_t<line_geometry>;

   // The large-value second arena (32 KiB grain): same allocator, same COW /
   // refcount / recovery machinery, max object large_max_object_bytes.
   using large_arena        = arena_t<large_geometry>;
   using large_session      = session_t<large_geometry>;
   using large_transaction  = transaction_t<large_geometry>;
   using large_read_session = read_session_t<large_geometry>;
}  // namespace start_ref

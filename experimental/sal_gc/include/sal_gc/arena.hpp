// sal_gc::arena — the FIX-region line store (design §6.1, v1).
//
// Three contiguous-VA block_allocator files, mirroring start_ref's proven core
// but with a deliberately SIMPLE writer session (the design rejects start_ref's
// arena session: no std::vector hot path, no window/fcache, no O(n) claim scan):
//   data.sal2gc  segments of 128 B lines   (the nodes; base + (idx<<line_shift))
//   free.sal2gc  one seg_freeset / segment  (lock-free recycle bitmap; derived)
//   refs.sal2gc  one u8 / line              (per-node refcount, head line only)
//
// Liveness is per-node refcount (v1): a node shared by N references has ref N at
// its head line; COW clones at ref>1, mutates in place at ref==1; release frees
// the node's whole run at ref 0 (the tree recurses into children). Version-GC is
// the design's later phase. Reads never touch refs.
#pragma once
#include <sys/mman.h>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>

#include <sal/block_allocator.hpp>

#include <sal_gc/config.hpp>
#include <sal_gc/freeset.hpp>

namespace sal_gc
{
   struct ref_overflow : std::runtime_error
   {
      ref_overflow() : std::runtime_error("sal_gc: refcount reached the 255 sentinel") {}
   };

   using seg_freeset = seg_freeset_t<lines_per_segment>;

   struct arena_config
   {
      std::filesystem::path dir;
      uint32_t              max_segments = 16384;  // x 8 MB = 128 GB ceiling
      uint32_t              max_roots    = 16;
   };

   class arena;

   // One writer. Owns the active segment + a first-fit cursor within it. Not
   // movable (the tree holds a session*). Single-writer by contract (v1).
   class session
   {
     public:
      session(const session&)            = delete;
      session& operator=(const session&) = delete;

      // Allocate n consecutive lines (1..64), all refs=1, head line stamped.
      // Returns the global line index. Throws on out-of-space (max_segments).
      line_idx alloc_lines(uint32_t n);
      arena&   store() noexcept { return *_a; }

     private:
      friend class arena;
      explicit session(arena* a) noexcept : _a(a) {}
      line_idx alloc_slow_(uint32_t n);
      line_idx finish_(uint32_t local);

      arena*   _a;
      uint32_t _seg    = UINT32_MAX;  // current segment claim
      uint32_t _cursor = 0;           // next-fit line within _seg (local)
   };

   class arena
   {
     public:
      explicit arena(arena_config cfg) : _cfg(std::move(cfg))
      {
         std::filesystem::create_directories(_cfg.dir);
         _data = std::make_unique<sal::block_allocator>(
             _cfg.dir / "data.sal2gc", segment_bytes, _cfg.max_segments,
             sal::access_mode::read_write);
         _free = std::make_unique<sal::block_allocator>(
             _cfg.dir / "free.sal2gc", uint64_t(std::bit_ceil(sizeof(seg_freeset))),
             _cfg.max_segments, sal::access_mode::read_write);
         _refs = std::make_unique<sal::block_allocator>(
             _cfg.dir / "refs.sal2gc", uint64_t(lines_per_segment) /* 1 B/line */,
             _cfg.max_segments, sal::access_mode::read_write);
         // grow one segment so the mapped bases exist, then cache them.
         grow_();
         _base      = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _free_base = static_cast<std::byte*>(const_cast<void*>(_free->mapped_base()));
         _ref_base  = reinterpret_cast<std::atomic<uint8_t>*>(
             const_cast<void*>(_refs->mapped_base()));
         for (auto& r : _roots)
            r.store(null_idx, std::memory_order_relaxed);
      }

      // ── addressing ─────────────────────────────────────────────────────────
      void* get(line_idx i) noexcept
      {
         assert(!is_sentinel(i));
         return _base + (size_t(i) << line_shift);
      }
      const void* get(line_idx i) const noexcept
      {
         assert(!is_sentinel(i));
         return _base + (size_t(i) << line_shift);
      }
      template <class T>
      T* as(line_idx i) noexcept { return reinterpret_cast<T*>(get(i)); }
      template <class T>
      const T* as(line_idx i) const noexcept { return reinterpret_cast<const T*>(get(i)); }

      // ── per-node refcount (head line) ───────────────────────────────────────
      uint8_t ref(line_idx i) const noexcept
      {
         return _ref_base[i].load(std::memory_order_relaxed);
      }
      bool unique(line_idx i) const noexcept { return ref(i) == 1; }

      // Pin: +1, throws at the 255 sentinel (reader/snapshot pins, COW shares).
      // Caller must already hold the node alive (ref >= 1) — used on the COW /
      // writer path and to copy a pin the caller already owns.
      void retain(line_idx i)
      {
         auto& r   = _ref_base[i];
         auto  cur = r.load(std::memory_order_relaxed);
         do
         {
            if (cur >= ref_ceiling)
               throw ref_overflow();
         } while (!r.compare_exchange_weak(cur, uint8_t(cur + 1), std::memory_order_relaxed));
      }

      // Lock-free snapshot acquire (concurrent readers, no mutex): bump the
      // refcount ONLY if it is currently non-zero — i.e. the node is still
      // alive. Returns false if the node is already dead (ref 0), so a reader
      // that lost the race with a freeing writer simply retries with the newer
      // root. The refcount array is a stable mmap, so probing ref(i) for any
      // line is always memory-safe even after the line's data is freed/recycled;
      // a benign ABA (probing a recycled line) is balanced by the caller's
      // matching release and never resurrects a dead node. Pair with a re-read
      // of the root slot to confirm the pinned root is the one intended.
      [[nodiscard]] bool try_retain(line_idx i) noexcept
      {
         auto& r   = _ref_base[i];
         auto  cur = r.load(std::memory_order_acquire);
         do
         {
            if (cur == 0)
               return false;               // already dead — caller retries
            if (cur >= ref_ceiling)
               return false;               // saturated; treat as un-pinnable
         } while (!r.compare_exchange_weak(cur, uint8_t(cur + 1), std::memory_order_acq_rel,
                                           std::memory_order_acquire));
         return true;
      }

      // Release one reference to the node at head line i. Returns true iff it hit
      // zero (the caller then recurses into children and frees the run). The
      // caller owns the run length (from the node header).
      [[nodiscard]] bool release_one(line_idx i) noexcept
      {
         const uint8_t prior = _ref_base[i].fetch_sub(1, std::memory_order_acq_rel);
         assert(prior >= 1 && "release of a dead node");
         return prior == 1;
      }

      // Free n lines at i (refs already zero / freshly allocated rollback).
      void free_lines(line_idx i, uint32_t n) noexcept
      {
         const uint32_t s = i / lines_per_segment;
         freeset_(s).free_run(i % lines_per_segment, n);
      }

      // ── sessions / roots ────────────────────────────────────────────────────
      session start_session() { return session(this); }

      line_idx get_root(uint32_t slot) const noexcept
      {
         assert(slot < _cfg.max_roots);
         return _roots[slot].load(std::memory_order_acquire);
      }
      void set_root(uint32_t slot, line_idx v) noexcept
      {
         assert(slot < _cfg.max_roots);
         _roots[slot].store(v, std::memory_order_release);
      }

      // ── stats (for the memory metric) ───────────────────────────────────────
      uint32_t num_segments() const noexcept
      {
         return _nseg.load(std::memory_order_acquire);
      }
      seg_freeset& freeset_of(uint32_t s) noexcept { return freeset_(s); }
      uint64_t     live_lines() noexcept
      {
         const uint32_t ns = num_segments();
         uint64_t       used = 0;
         for (uint32_t s = 0; s < ns; ++s)
            used += lines_per_segment - freeset_(s).free_lines();
         return used;
      }
      uint64_t live_bytes() noexcept { return live_lines() * uint64_t(line_bytes); }

     private:
      friend class session;

      seg_freeset& freeset_(uint32_t s) noexcept
      {
         return *reinterpret_cast<seg_freeset*>(
             _free_base + size_t(s) * std::bit_ceil(sizeof(seg_freeset)));
      }

      // Grow by one segment (slow path, under the grow mutex). Returns its index.
      uint32_t grow_()
      {
         std::lock_guard g(_grow_mx);
         const uint32_t s = _nseg.load(std::memory_order_relaxed);
         if (s >= _cfg.max_segments)
            throw std::runtime_error("sal_gc: arena full (max_segments)");
         _data->alloc();
         _free->alloc();
         _refs->alloc();
         // mapped bases are stable (PROT_NONE reserve + MAP_FIXED); recompute the
         // free base lazily on first use is unnecessary — block_allocator keeps it.
         std::byte* fb = static_cast<std::byte*>(const_cast<void*>(_free->mapped_base()));
         new (fb + size_t(s) * std::bit_ceil(sizeof(seg_freeset))) seg_freeset();
         reinterpret_cast<seg_freeset*>(fb + size_t(s) * std::bit_ceil(sizeof(seg_freeset)))
             ->init(true);
         std::byte* rb = static_cast<std::byte*>(const_cast<void*>(_refs->mapped_base()));
         std::memset(rb + size_t(s) * lines_per_segment, 0, lines_per_segment);
         _nseg.store(s + 1, std::memory_order_release);
         return s;
      }

      // Find a segment (other than `except`) that can hold n lines, else grow.
      uint32_t claim_(uint32_t except)
      {
         const uint32_t ns = num_segments();
         for (uint32_t s = 0; s < ns; ++s)
            if (s != except && freeset_(s).free_lines() >= max_alloc_lines)
               return s;
         return grow_();
      }

      arena_config                          _cfg;
      std::unique_ptr<sal::block_allocator> _data, _free, _refs;
      std::byte*                            _base      = nullptr;
      std::byte*                            _free_base = nullptr;
      std::atomic<uint8_t>*                 _ref_base  = nullptr;
      std::atomic<uint32_t>                 _nseg{0};
      std::mutex                            _grow_mx;
      std::atomic<line_idx>                 _roots[64] = {};
   };

   // ── session bodies ──────────────────────────────────────────────────────
   inline line_idx session::alloc_lines(uint32_t n)
   {
      assert(n >= 1 && n <= max_alloc_lines);
      if (_seg != UINT32_MAX)
      {
         seg_freeset&   fs    = _a->freeset_(_seg);
         const uint32_t local = fs.alloc(n, _cursor);
         if (local != UINT32_MAX)
         {
            _cursor = local + n;
            return finish_(local);
         }
      }
      return alloc_slow_(n);
   }

   inline line_idx session::alloc_slow_(uint32_t n)
   {
      // The cursor region can't fit an n-line run. Scan every segment for a real
      // n-contiguous run (a segment may have many free lines but no n-run under
      // fragmentation); grow a fresh segment if none fits (fresh = all-free, so it
      // always satisfies n<=64).
      const uint32_t ns = _a->num_segments();
      for (uint32_t s = 0; s < ns; ++s)
      {
         seg_freeset& fs = _a->freeset_(s);
         if (fs.free_lines() < n)
            continue;
         const uint32_t local = fs.alloc(n, 0);
         if (local != UINT32_MAX)
         {
            _seg    = s;
            _cursor = local + n;
            return finish_(local);
         }
      }
      const uint32_t s     = _a->grow_();
      const uint32_t local = _a->freeset_(s).alloc(n, 0);  // fresh seg: cannot fail
      _seg                 = s;
      _cursor              = local + n;
      return finish_(local);
   }

   // (declared inline in the class? no — define finish_ here)
   inline line_idx session::finish_(uint32_t local)  // NOLINT
   {
      const line_idx g = line_idx(_seg) * lines_per_segment + local;
      _a->_ref_base[g].store(1, std::memory_order_relaxed);  // head line ref=1
      return g;
   }
}  // namespace sal_gc

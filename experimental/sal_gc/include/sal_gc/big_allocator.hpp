// sal_gc::big_allocator<Grain> — the BIG region (sal_gc_design.md §6.3): large
// (> ~2KB) objects in a FIXED, never-relocated mmap region, handed out by a fast
// SIZE-CLASS FREE-LIST allocator (the "boost-shmem-malloc style" the spec asks
// for). Two properties the spec pins down:
//
//   * Fixed regions that don't move — the backing store is sal::block_allocator
//     (PROT_NONE reserve + MAP_FIXED commit): a big object's address never
//     changes, so a parent branch can hold a verbatim 4-byte handle and readers
//     deref lock-free as base + handle*Grain.
//   * A real heap allocator (not a bump) — each grain-count has a LIFO free-list
//     of returned blocks, so a freed big value is reused at the OBJECT level
//     immediately (no waiting for a whole segment to recycle). A request with no
//     free block of its size bumps the contiguous region tail.
//
// Liveness is a per-object 32-bit DEATH version in an INLINE header
// ([u32 ngrains|u32 death] in the block's first grain), like the FIX/SAL regions:
// the COW path stamp_death()s the superseded object and the GC reclaim()s it once
// no held root covers [birth, death). No refcount; reads touch nothing.
// Big allocs/frees are rare relative to FIX/SAL traffic, so the alloc/free path
// is a single mutex; get/ref/retain/release_one are lock-free.
//
// v1 reuses a freed block only for a request of the SAME grain-count (exact-fit
// size classes — no split/coalesce). That fits the shape of large values
// (write-once contract code; same-size value overwrites). Split+coalesce (boost
// rbtree_best_fit style) is a later density refinement.
#pragma once
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <sal/block_allocator.hpp>

#include <sal_gc/config.hpp>  // version_t, death_alive / death_free sentinels

namespace sal_gc
{
   template <uint32_t Grain = 64>
   class big_allocator
   {
     public:
      static constexpr uint32_t grain          = Grain;
      static constexpr uint64_t segment_bytes  = 8u * 1024u * 1024u;  // 8 MB commit unit
      static constexpr uint32_t grains_per_seg = uint32_t(segment_bytes / Grain);
      static constexpr uint32_t max_bucket     = 8192;  // free-list size classes
      static constexpr uint32_t hdr_bytes      = 8;     // [u32 ngrains | u32 ref]
      static constexpr uint32_t null_grain     = UINT32_MAX;
      static_assert(Grain >= hdr_bytes, "grain must hold the inline header");
      static_assert((segment_bytes % Grain) == 0, "segment must be a whole number of grains");

      struct config
      {
         std::filesystem::path dir;
         std::string           name;
         uint32_t              max_segments = 4096;
      };

      explicit big_allocator(config cfg) : _cfg(std::move(cfg))
      {
         std::filesystem::create_directories(_cfg.dir);
         _data = std::make_unique<sal::block_allocator>(_cfg.dir / (_cfg.name + ".big"),
                                                        segment_bytes, _cfg.max_segments,
                                                        sal::access_mode::read_write);
         _base = static_cast<std::byte*>(
             const_cast<void*>(_data->mapped_base()));  // reservation base (fixed)
         for (auto& h : _flist)
            h.store(null_grain, std::memory_order_relaxed);
         adopt_existing_();
      }

      // ── reopen recovery ───────────────────────────────────────────────────────
      // block_allocator remmaps the .big file, so a clean reopen's object bytes +
      // inline [ngrains|ref] headers survive — old big objects are reachable
      // (get(g) = base + g·grain) and releasable for free. The bump tail can't be
      // reconstructed from the data (object boundaries are only walkable from a
      // known tail), so we resume the cursor at the END of the committed region:
      // new objects append there; the unused tail of the last old segment is
      // wasted (negligible). Free-lists start empty (old freed-big holes are not
      // re-pooled). live_bytes is a stat only — left 0 here.
      void adopt_existing_() noexcept
      {
         const uint32_t existing = uint32_t(_data->mapped_blocks());
         if (existing == 0)
            return;
         _data->resize(existing);
         _base             = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _nseg             = existing;
         _committed_grains = existing * grains_per_seg;
         _tail             = _committed_grains;  // append after committed (safe)
      }

      // inline block header at the block's first grain.
      struct hdr
      {
         uint32_t              ngrains;
         std::atomic<version_t> death;  // version-liveness (was refcount): [birth in node, death here)
      };

      hdr* hdr_at(uint32_t g) noexcept { return reinterpret_cast<hdr*>(_base + size_t(g) * Grain); }
      const hdr* hdr_at(uint32_t g) const noexcept
      {
         return reinterpret_cast<const hdr*>(_base + size_t(g) * Grain);
      }
      void* get(uint32_t g) noexcept { return _base + size_t(g) * Grain + hdr_bytes; }  // body
      const void* get(uint32_t g) const noexcept { return _base + size_t(g) * Grain + hdr_bytes; }

      static uint32_t grains_for(uint32_t body_bytes) noexcept
      {
         return (hdr_bytes + body_bytes + Grain - 1) / Grain;
      }

      // ── version liveness (death stamp; no refcount) — see fixed_allocator ──────
      version_t death(uint32_t g) const noexcept
      {
         return hdr_at(g)->death.load(std::memory_order_relaxed);
      }
      bool alive(uint32_t g) const noexcept { return death(g) == death_alive; }
      bool dead(uint32_t g) const noexcept
      {
         const version_t d = death(g);
         return d != death_alive && d != death_free;
      }
      void stamp_death(uint32_t g, version_t v) noexcept
      {
         hdr_at(g)->death.store(v, std::memory_order_release);
      }

      // allocate a body of `body_bytes` (born death_alive). LOCKED — big allocs are rare.
      uint32_t alloc(uint32_t body_bytes)
      {
         const uint32_t              ng = grains_for(body_bytes);
         std::lock_guard<std::mutex> lk(_mx);
         uint32_t                    g;
         if (ng < max_bucket && _flist[ng].load(std::memory_order_relaxed) != null_grain)
         {  // size-class free-list hit: reuse a returned block of exactly ng grains
            g = _flist[ng].load(std::memory_order_relaxed);
            uint32_t next;
            std::memcpy(&next, _base + size_t(g) * Grain + hdr_bytes, 4);  // unthread the LIFO next
            _flist[ng].store(next, std::memory_order_relaxed);
         }
         else
         {  // bump the contiguous region tail, committing segments as needed
            ensure_committed_(uint64_t(_tail) + ng);
            g = _tail;
            _tail += ng;
         }
         hdr* h     = hdr_at(g);
         h->ngrains = ng;
         h->death.store(death_alive, std::memory_order_relaxed);  // born live
         _live_grains += ng;
         return g;
      }

      // GC reclaim: return a block to its size-class free-list. Called by the GC once
      // death-stamped + uncovered (or by abort). Marks death_free, then LIFO-pushes.
      void reclaim(uint32_t g) noexcept
      {
         assert(death(g) != death_free && "double reclaim of a big object");
         hdr_at(g)->death.store(death_free, std::memory_order_release);
         const uint32_t              ng = hdr_at(g)->ngrains;
         std::lock_guard<std::mutex> lk(_mx);
         _live_grains -= ng;
         if (ng < max_bucket)
         {  // LIFO push: thread the old head through this (now free) block's body
            const uint32_t head = _flist[ng].load(std::memory_order_relaxed);
            std::memcpy(_base + size_t(g) * Grain + hdr_bytes, &head, 4);
            _flist[ng].store(g, std::memory_order_relaxed);
         }
         // ng >= max_bucket: oversized — not pooled (rare); its grains stay in-region.
      }

      uint64_t live_bytes() noexcept
      {
         std::lock_guard<std::mutex> lk(_mx);
         return uint64_t(_live_grains) * Grain;
      }
      uint32_t num_segments() const noexcept { return _nseg; }

     private:
      void ensure_committed_(uint64_t need_grains)  // caller holds _mx
      {
         while (uint64_t(_committed_grains) < need_grains)
         {
            if (_nseg >= _cfg.max_segments)
               throw std::runtime_error("sal_gc: big_allocator full");
            _data->alloc();  // commit the next contiguous segment (MAP_FIXED in the reservation)
            _base = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
            _committed_grains += grains_per_seg;
            ++_nseg;
         }
      }

      config                                _cfg;
      std::unique_ptr<sal::block_allocator> _data;
      std::byte*                            _base = nullptr;  // fixed reservation base
      std::mutex                            _mx;
      uint32_t                              _tail             = 0;  // next fresh grain
      uint32_t                              _committed_grains = 0;
      uint32_t                              _nseg             = 0;
      uint64_t                              _live_grains      = 0;
      std::atomic<uint32_t>                 _flist[max_bucket];  // per-grain-count free-list heads
   };
}  // namespace sal_gc

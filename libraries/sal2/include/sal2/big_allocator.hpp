// sal2::big_allocator<Grain> — the BIG region (sal2_design.md §6.3): large
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
// Liveness is a per-object refcount in an INLINE header ([u32 ngrains|u32 ref] in
// the block's first grain), so retain/release_one drive freeing exactly like the
// FIX/SAL regions: free() is reached ONLY via release_one()->0 (it asserts ref==0).
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

namespace sal2
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
      }

      // inline block header at the block's first grain.
      struct hdr
      {
         uint32_t              ngrains;
         std::atomic<uint32_t> ref;
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

      uint32_t ref(uint32_t g) const noexcept
      {
         return hdr_at(g)->ref.load(std::memory_order_relaxed);
      }
      bool unique(uint32_t g) const noexcept { return ref(g) == 1; }
      void retain(uint32_t g) { hdr_at(g)->ref.fetch_add(1, std::memory_order_relaxed); }
      [[nodiscard]] bool try_retain(uint32_t g) noexcept
      {
         auto& r = hdr_at(g)->ref;
         auto  c = r.load(std::memory_order_acquire);
         do
         {
            if (c == 0)
               return false;
         } while (!r.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel,
                                           std::memory_order_acquire));
         return true;
      }
      [[nodiscard]] bool release_one(uint32_t g) noexcept
      {
         const uint32_t prior = hdr_at(g)->ref.fetch_sub(1, std::memory_order_acq_rel);
         assert(prior >= 1 && "release of a dead big object");
         return prior == 1;
      }

      // allocate a body of `body_bytes` (ref stamped 1). LOCKED — big allocs are rare.
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
         h->ref.store(1, std::memory_order_relaxed);
         _live_grains += ng;
         return g;
      }

      // return a dead block (ref already 0) to its size-class free-list.
      void free(uint32_t g) noexcept
      {
         assert(hdr_at(g)->ref.load(std::memory_order_relaxed) == 0 &&
                "big free() of an object whose ref != 0 — route the drop through release");
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
               throw std::runtime_error("sal2: big_allocator full");
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
}  // namespace sal2

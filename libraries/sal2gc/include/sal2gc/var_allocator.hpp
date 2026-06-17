// sal2gc::var_allocator<Grain> — the SAL leaf region as per-size-class slabs.
//
// Spec: suffix_value leaf objects (<4KB) live in a 32-byte-grain region (denser
// than the 128B FIX line — the leaf-density lever). Per Dan's SAL segment model
// every segment is EXACT-SIZED: it holds objects of exactly ONE grain-count, so
//  • alloc is a pure forward bump by the class size (no freeset, no CAS),
//  • liveness is the refcount array — ref==0 at a slot boundary IS a free slot
//    (no continuation grains are ever inspected; the walk steps by the class),
//  • a freed slot is reused only when the WHOLE segment recycles: a SEALED
//    segment that crosses >50% free joins a per-class recyclable set, and the
//    next new-segment request for that class pulls it and writes AROUND the live
//    leftovers (skip ref!=0). No fragmentation, no run-finding, no infinite loop.
//
// One l0 word per segment would cap slots at 64·64·64 = 262144 grains, so a
// Grain-byte segment is 262144·Grain bytes (8 MB at Grain=32); a single object
// is ≤64 grains (2 KB at Grain=32) — larger values go BIG. The slot index (SAL
// ptr_address) is a GLOBAL grain index: addr = base + slot·Grain.
#pragma once
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <sal/block_allocator.hpp>

#include <sal2gc/config.hpp>

namespace sal2gc
{
   template <uint32_t Grain>
   class var_allocator
   {
     public:
      static constexpr uint32_t grain             = Grain;
      static constexpr uint32_t slots_per_segment = 64u * 64u * 64u;  // grains / segment
      static constexpr uint64_t segment_bytes     = uint64_t(slots_per_segment) * Grain;
      static constexpr uint32_t max_run           = max_alloc_lines;    // 64 grains/object
      static constexpr uint32_t max_object_bytes  = max_run * Grain;    // 2 KB at Grain=32
      static constexpr uint32_t recycle_below = slots_per_segment / 2;  // >50% free ⇒ recyclable
      static_assert((segment_bytes & (segment_bytes - 1)) == 0, "segment must be a power of 2");

      struct config
      {
         std::filesystem::path dir;
         std::string           name;
         uint32_t              max_segments = 65536;
      };
      class session;

      explicit var_allocator(config cfg) : _cfg(std::move(cfg))
      {
         std::filesystem::create_directories(_cfg.dir);
         _data = std::make_unique<sal::block_allocator>(_cfg.dir / (_cfg.name + ".dat"),
                                                        segment_bytes, _cfg.max_segments,
                                                        sal::access_mode::read_write);
         _refs = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".rc"), std::bit_ceil(uint64_t(slots_per_segment)),
             _cfg.max_segments, sal::access_mode::read_write);
         _seg_alloced = std::make_unique<std::atomic<uint32_t>[]>(_cfg.max_segments);
         _seg_freed   = std::make_unique<std::atomic<uint32_t>[]>(_cfg.max_segments);
         _seg_sealed  = std::make_unique<std::atomic<uint8_t>[]>(_cfg.max_segments);
         _seg_class   = std::make_unique<std::atomic<uint8_t>[]>(_cfg.max_segments);
         _recyc       = std::make_unique<std::atomic<uint64_t>[]>((_cfg.max_segments + 63) / 64);
         cache_bases_();
         adopt_existing_();
      }

      // ── reopen recovery ───────────────────────────────────────────────────────
      // block_allocator remmaps the .dat + .rc files, so a clean reopen's object
      // bytes AND per-grain refcounts survive. Restore the in-memory view: resize
      // the block_allocators so the next alloc() appends after `existing` segments
      // (never overwrites), set _nseg, and RETIRE every existing segment with the
      // sentinel class 0xFF so take_recyclable_ (which only reuses a segment whose
      // grain-class matches the request) never re-fills it — we can't reconstruct a
      // segment's grain-class from refcounts alone, and reusing it under the wrong
      // stride would corrupt the walk. Old objects stay reachable + releasable via
      // the preserved refcounts; new objects grow fresh segments.
      void adopt_existing_() noexcept
      {
         const uint32_t existing = uint32_t(_data->mapped_blocks());
         if (existing == 0)
            return;
         _data->resize(existing);
         _refs->resize(existing);
         cache_bases_();
         _nseg.store(existing, std::memory_order_relaxed);
         for (uint32_t s = 0; s < existing; ++s)
         {
            _seg_class[s].store(0xFF, std::memory_order_relaxed);  // sentinel: never recyclable
            _seg_sealed[s].store(1, std::memory_order_relaxed);
            _seg_alloced[s].store(slots_per_segment, std::memory_order_relaxed);
            _seg_freed[s].store(0, std::memory_order_relaxed);
         }
      }

      void*           get(uint32_t slot) noexcept { return _base + size_t(slot) * Grain; }
      const void*     get(uint32_t slot) const noexcept { return _base + size_t(slot) * Grain; }
      static uint32_t grains_for(uint32_t bytes) noexcept { return (bytes + Grain - 1) / Grain; }

      uint8_t ref(uint32_t i) const noexcept
      {
         return _ref_base[i].load(std::memory_order_relaxed);
      }
      bool unique(uint32_t i) const noexcept { return ref(i) == 1; }
      void retain(uint32_t i)
      {
         auto& r = _ref_base[i];
         auto  c = r.load(std::memory_order_relaxed);
         do
         {
            if (c >= ref_ceiling)
               throw std::runtime_error("sal2gc: var_allocator refcount overflow");
         } while (!r.compare_exchange_weak(c, uint8_t(c + 1), std::memory_order_relaxed));
      }
      [[nodiscard]] bool try_retain(uint32_t i) noexcept
      {
         auto& r = _ref_base[i];
         auto  c = r.load(std::memory_order_acquire);
         do
         {
            if (c == 0 || c >= ref_ceiling)
               return false;
         } while (!r.compare_exchange_weak(c, uint8_t(c + 1), std::memory_order_acq_rel,
                                           std::memory_order_acquire));
         return true;
      }
      // Last ref drops to the DYING sentinel (not 0): the slot is being torn down
      // (the releaser still reads `bytes` from its content) and must NOT be reused
      // until free() sets ref=0. Write-around reuses only ref==0. CAS so there is
      // no transient ref==0 window before the sentinel. See fixed_allocator.
      [[nodiscard]] bool release_one(uint32_t i) noexcept
      {
         auto&   r = _ref_base[i];
         uint8_t c = r.load(std::memory_order_acquire);
         for (;;)
         {
            assert(c >= 1 && c != dying_ref && "release of a dead/dying object");
            const uint8_t nv = (c == 1) ? dying_ref : uint8_t(c - 1);
            if (r.compare_exchange_weak(c, nv, std::memory_order_acq_rel, std::memory_order_acquire))
               return c == 1;
         }
      }
      // Mark an object dead (refcount already 0). bg writes _seg_freed, foreground
      // writes _seg_alloced (separate arrays = separate cachelines, no false
      // sharing); a sealed segment crossing >50% free joins its class's recyclable
      // set. The alloc hot path never reads this, so frees don't contend with it.
      void free(uint32_t i, uint32_t bytes) noexcept
      {
         // Publish REUSABLE (ref dying→0) with RELEASE so the write-around alloc's
         // ACQUIRE load of ref==0 orders the releaser's content reads (done before
         // free()) ahead of any reuse-write. See fixed_allocator::free.
         _ref_base[i].store(0, std::memory_order_release);
         const uint32_t n = grains_for(bytes);
         _freed_grains.fetch_add(n, std::memory_order_relaxed);
         const uint32_t s     = i / slots_per_segment;
         const uint32_t freed = _seg_freed[s].fetch_add(n, std::memory_order_relaxed) + n;
         const uint32_t live  = _seg_alloced[s].load(std::memory_order_acquire) - freed;
         if (live <= recycle_below && _seg_sealed[s].load(std::memory_order_acquire))
            _recyc[s >> 6].fetch_or(1ull << (s & 63), std::memory_order_relaxed);
      }

      session  start_session() { return session(this); }
      uint32_t num_segments() const noexcept { return _nseg.load(std::memory_order_acquire); }
      uint64_t live_slots() noexcept
      {
         return _allocated_grains.load(std::memory_order_relaxed) -
                _freed_grains.load(std::memory_order_relaxed);
      }
      uint64_t live_bytes() noexcept { return live_slots() * uint64_t(Grain); }

     private:
      friend class session;
      void cache_bases_() noexcept
      {
         _base = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _ref_base =
             reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
      }
      // Grow a fresh segment dedicated to grain-count `cls`.
      uint32_t grow_(uint8_t cls)
      {
         std::lock_guard<std::mutex> g(_grow_mx);
         const uint32_t              s = _nseg.load(std::memory_order_relaxed);
         if (s >= _cfg.max_segments)
            throw std::runtime_error("sal2gc: var_allocator full");
         _data->alloc();
         _refs->alloc();
         cache_bases_();
         _seg_alloced[s].store(0, std::memory_order_relaxed);
         _seg_freed[s].store(0, std::memory_order_relaxed);
         _seg_sealed[s].store(0, std::memory_order_relaxed);
         _seg_class[s].store(cls, std::memory_order_relaxed);
         _nseg.store(s + 1, std::memory_order_release);
         return s;
      }
      // Pull a recyclable (>50%-free, sealed) segment OF CLASS `cls`, or MAX. A
      // recycled segment keeps its class: its live leftovers sit on class-aligned
      // slot boundaries, so it can only be re-filled at the same size.
      uint32_t take_recyclable_(uint8_t cls) noexcept
      {
         const uint32_t words = (_nseg.load(std::memory_order_acquire) + 63) >> 6;
         for (uint32_t w = 0; w < words; ++w)
         {
            uint64_t m = _recyc[w].load(std::memory_order_relaxed);
            while (m)
            {
               const uint32_t s   = w * 64 + uint32_t(std::countr_zero(m));
               const uint64_t bit = 1ull << (s & 63);
               if (_seg_class[s].load(std::memory_order_relaxed) != cls)
               {
                  m &= ~bit;
                  continue;
               }
               if (_recyc[w].fetch_and(~bit, std::memory_order_acquire) & bit)
               {
                  _seg_sealed[s].store(0, std::memory_order_relaxed);
                  return s;
               }
               m = _recyc[w].load(std::memory_order_relaxed);
            }
         }
         return UINT32_MAX;
      }

      config                                _cfg;
      std::unique_ptr<sal::block_allocator> _data, _refs;
      std::byte*                            _base     = nullptr;
      std::atomic<uint8_t>*                 _ref_base = nullptr;
      std::atomic<uint32_t>                 _nseg{0};
      std::mutex                            _grow_mx;
      alignas(64) std::atomic<uint64_t> _allocated_grains{0};  // foreground
      alignas(64) std::atomic<uint64_t> _freed_grains{0};      // bg release thread
      std::unique_ptr<std::atomic<uint32_t>[]> _seg_alloced;   // foreground: grains / segment
      std::unique_ptr<std::atomic<uint32_t>[]> _seg_freed;     // bg: freed grains / segment
      std::unique_ptr<std::atomic<uint8_t>[]>  _seg_sealed;    // 1 = full/retired
      std::unique_ptr<std::atomic<uint8_t>[]>  _seg_class;     // grain-count this segment serves
      std::unique_ptr<std::atomic<uint64_t>[]> _recyc;         // bitset of recyclable segments
   };

   // One writer's per-size-class bump cursors. Each grain-count has its own active
   // segment, so a segment is never mixed-size and the recycle walk is exact.
   template <uint32_t Grain>
   class var_allocator<Grain>::session
   {
     public:
      session(const session&)            = delete;
      session& operator=(const session&) = delete;

      // allocate an object of `bytes` (≤ max_object_bytes); ref stamped to 1.
      uint32_t alloc(uint32_t bytes)
      {
         const uint32_t g = grains_for(bytes);
         assert(g >= 1 && g <= max_run);
         return alloc_g_(g);
      }

     private:
      friend class var_allocator<Grain>;
      explicit session(var_allocator* a) noexcept : _a(a) {}

      struct cursor
      {
         uint32_t seg    = UINT32_MAX;
         uint32_t pos    = 0;
         bool     fresh  = true;
         bool     active = false;
      };

      // Forward bump within class g's active segment: a FRESH segment is a pure
      // cursor bump by g; a RECYCLED one write-arounds by stepping over live
      // (ref!=0) slots. Every slot is exactly g grains, so ref==0 == free slot.
      uint32_t alloc_g_(uint32_t g)
      {
         cursor& c = _cls[g];
         if (c.active)
         {
            if (c.fresh)
            {
               if (c.pos + g <= slots_per_segment)
               {
                  const uint32_t l = c.pos;
                  c.pos += g;
                  return finish_(c.seg, l, g);
               }
            }
            else
            {
               const std::atomic<uint8_t>* refs = _a->_ref_base + size_t(c.seg) * slots_per_segment;
               while (c.pos + g <= slots_per_segment)
               {
                  // ACQUIRE (see fixed_allocator): orders the releaser's pre-release
                  // content reads before our reuse of a ref==0 slot.
                  if (refs[c.pos].load(std::memory_order_acquire) == 0)
                  {
                     const uint32_t l = c.pos;
                     c.pos += g;
                     return finish_(c.seg, l, g);
                  }
                  c.pos += g;  // live leftover — write around it
               }
            }
         }
         return alloc_slow_(g);
      }
      // class g's segment is full → seal it (eligible for recycle), pull a
      // recyclable g-class segment (write around its leftovers) else grow fresh.
      uint32_t alloc_slow_(uint32_t g)
      {
         cursor& c = _cls[g];
         if (c.active)
         {
            _a->_seg_sealed[c.seg].store(1, std::memory_order_release);
            const uint32_t live = _a->_seg_alloced[c.seg].load(std::memory_order_relaxed) -
                                  _a->_seg_freed[c.seg].load(std::memory_order_acquire);
            if (live <= recycle_below)
               _a->_recyc[c.seg >> 6].fetch_or(1ull << (c.seg & 63), std::memory_order_relaxed);
         }
         const uint32_t r = _a->take_recyclable_(uint8_t(g));
         if (r != UINT32_MAX)
         {
            c.seg   = r;
            c.pos   = 0;
            c.fresh = false;
         }
         else
         {
            c.seg   = _a->grow_(uint8_t(g));
            c.pos   = 0;
            c.fresh = true;
         }
         c.active = true;
         return alloc_g_(g);
      }
      uint32_t finish_(uint32_t seg, uint32_t local, uint32_t g) noexcept
      {
         const uint32_t slot = seg * slots_per_segment + local;
         _a->_ref_base[slot].store(1, std::memory_order_relaxed);
         _a->_seg_alloced[seg].fetch_add(g, std::memory_order_release);
         _a->_allocated_grains.fetch_add(g, std::memory_order_relaxed);
         return slot;
      }
      var_allocator* _a;
      cursor         _cls[max_run + 1];  // one active segment per grain-count (1..64)
   };
}  // namespace sal2gc

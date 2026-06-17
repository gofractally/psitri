// sal2::fixed_allocator<SlotBytes> — the FIX-region per-size-class slab.
//
// Spec (libart++/docs/sal2.md, dev-plan step 1): "first build and test
// FixedAllocator<N> ... robust against random alloc/free." One complete
// allocator for ONE fixed slot size; the FIX region is three of them:
//   FixedAllocator<128>  = cl1  (setlist / nested_setlist)
//   FixedAllocator<256>  = cl2  (double_setlist / big_nested)
//   FixedAllocator<1152> = cl9  (full / node256)
//
// 32 MB segments (block_allocator-backed, grown on demand); per-segment
// seg_freeset = one bit per slot (free=1), lock-free O(1) find-first-free;
// per-slot refcount (Dan: refcount liveness, `created`/birth dropped — denser).
// A write session owns an active segment and bump-claims forward; a freed slot
// returns to the bitmap immediately; a segment is re-offered once its free count
// crosses the reuse threshold. Slot index is GLOBAL: addr = base + slot·SlotBytes.
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

#include <sal2/config.hpp>
#include <sal2/freeset.hpp>

namespace sal2
{
   // Slots per 32 MB segment, rounded DOWN to a multiple of 64 (the freeset l0
   // word). cl9's 1152 B slot leaves a small tail; cl1/cl2 divide evenly.
   inline constexpr uint32_t fa_segment_bytes = 32u * 1024 * 1024;
   constexpr uint32_t        slots_per_seg(uint32_t slot_bytes) noexcept
   {
      return (fa_segment_bytes / slot_bytes) & ~63u;
   }

   template <uint32_t SlotBytes>
   class fixed_allocator
   {
     public:
      static constexpr uint32_t slot_bytes = SlotBytes;
      static constexpr uint32_t slots_per_segment = slots_per_seg(SlotBytes);
      using freeset = seg_freeset_t<slots_per_segment>;
      static_assert(slots_per_segment >= 64 && slots_per_segment % 64 == 0);

      struct config
      {
         std::filesystem::path dir;
         std::string           name;          // file stem, e.g. "cl1"
         uint32_t              max_segments = 16384;
         double                reuse_at     = 0.5;  // re-offer a segment at ≥50% free
      };

      class session;

      explicit fixed_allocator(config cfg) : _cfg(std::move(cfg))
      {
         std::filesystem::create_directories(_cfg.dir);
         _data = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".dat"), fa_segment_bytes /* power-of-2; cl9 wastes a tiny tail */,
             _cfg.max_segments, sal::access_mode::read_write);
         _free = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".fs"), uint64_t(std::bit_ceil(sizeof(freeset))),
             _cfg.max_segments, sal::access_mode::read_write);
         _refs = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".rc"), std::bit_ceil(uint64_t(slots_per_segment)) /* 1 B/slot, pow2 */,
             _cfg.max_segments, sal::access_mode::read_write);
         grow_();
         _base      = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _free_base = static_cast<std::byte*>(const_cast<void*>(_free->mapped_base()));
         _ref_base  = reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
      }

      // ── addressing (slot index → bytes) ──────────────────────────────────────
      void*       get(uint32_t slot) noexcept { return _base + size_t(slot) * SlotBytes; }
      const void* get(uint32_t slot) const noexcept { return _base + size_t(slot) * SlotBytes; }

      // ── refcount ─────────────────────────────────────────────────────────────
      uint8_t ref(uint32_t i) const noexcept { return _ref_base[i].load(std::memory_order_relaxed); }
      bool    unique(uint32_t i) const noexcept { return ref(i) == 1; }
      void    retain(uint32_t i)
      {
         auto& r = _ref_base[i]; auto c = r.load(std::memory_order_relaxed);
         do { if (c >= ref_ceiling) throw std::runtime_error("sal2: fixed_allocator refcount overflow"); }
         while (!r.compare_exchange_weak(c, uint8_t(c + 1), std::memory_order_relaxed));
      }
      [[nodiscard]] bool try_retain(uint32_t i) noexcept
      {
         auto& r = _ref_base[i]; auto c = r.load(std::memory_order_acquire);
         do { if (c == 0 || c >= ref_ceiling) return false; }
         while (!r.compare_exchange_weak(c, uint8_t(c + 1), std::memory_order_acq_rel, std::memory_order_acquire));
         return true;
      }
      // returns true iff this drops the slot to free; the caller then free()s it.
      [[nodiscard]] bool release_one(uint32_t i) noexcept
      {
         const uint8_t prior = _ref_base[i].fetch_sub(1, std::memory_order_acq_rel);
         assert(prior >= 1 && "release of a dead slot");
         return prior == 1;
      }
      // return a slot to its segment's free bitmap (immediate reclaim).
      void free(uint32_t i) noexcept
      {
         const uint32_t s = i / slots_per_segment;
         freeset_(s).free1(i % slots_per_segment);
      }

      // ── sessions ───────────────────────────────────────────────────────────
      session start_session() { return session(this); }

      // ── stats ────────────────────────────────────────────────────────────────
      uint32_t num_segments() const noexcept { return _nseg.load(std::memory_order_acquire); }
      freeset& freeset_of(uint32_t s) noexcept { return freeset_(s); }
      uint64_t live_slots() noexcept
      {
         const uint32_t ns = num_segments();
         uint64_t used = 0;
         for (uint32_t s = 0; s < ns; ++s) used += slots_per_segment - freeset_(s).free_lines();
         return used;
      }
      uint64_t live_bytes() noexcept { return live_slots() * uint64_t(SlotBytes); }

     private:
      friend class session;

      freeset& freeset_(uint32_t s) noexcept
      {
         return *reinterpret_cast<freeset*>(_free_base + size_t(s) * std::bit_ceil(sizeof(freeset)));
      }
      uint32_t grow_()
      {
         std::lock_guard<std::mutex> g(_grow_mx);
         const uint32_t s = _nseg.load(std::memory_order_relaxed);
         if (s >= _cfg.max_segments) throw std::runtime_error("sal2: fixed_allocator full");
         _data->alloc(); _free->alloc(); _refs->alloc();
         _base      = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _free_base = static_cast<std::byte*>(const_cast<void*>(_free->mapped_base()));
         _ref_base  = reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
         new (&freeset_(s)) freeset();
         freeset_(s).init(/*all_free=*/true);
         _nseg.store(s + 1, std::memory_order_release);
         return s;
      }

      config                                _cfg;
      std::unique_ptr<sal::block_allocator> _data, _free, _refs;
      std::byte*                            _base = nullptr;
      std::byte*                            _free_base = nullptr;
      std::atomic<uint8_t>*                 _ref_base = nullptr;
      std::atomic<uint32_t>                 _nseg{0};
      std::mutex                            _grow_mx;
   };

   // One writer's bump cursor within an active segment (spec req 4/5/7).
   template <uint32_t SlotBytes>
   class fixed_allocator<SlotBytes>::session
   {
     public:
      session(const session&)            = delete;
      session& operator=(const session&) = delete;

      // Allocate one slot (global index), ref stamped to 1.
      uint32_t alloc()
      {
         if (_seg != UINT32_MAX)
         {
            const uint32_t local = _a->freeset_(_seg).alloc1(_cursor);
            if (local != UINT32_MAX) { _cursor = local + 1; return finish_(local); }
         }
         return alloc_slow_();
      }

     private:
      friend class fixed_allocator<SlotBytes>;
      explicit session(fixed_allocator* a) noexcept : _a(a) {}

      uint32_t alloc_slow_()
      {
         // adopt the lowest segment with a free slot, else grow a fresh one.
         const uint32_t ns = _a->num_segments();
         for (uint32_t s = (_seg + 1 < ns ? _seg + 1 : 0), seen = 0; seen < ns; ++seen, s = (s + 1 < ns ? s + 1 : 0))
         {
            if (_a->freeset_(s).free_lines() == 0) continue;
            const uint32_t local = _a->freeset_(s).alloc1(0);
            if (local != UINT32_MAX) { _seg = s; _cursor = local + 1; return finish_(local); }
         }
         const uint32_t s     = _a->grow_();
         const uint32_t local = _a->freeset_(s).alloc1(0);
         _seg = s; _cursor = local + 1;
         return finish_(local);
      }
      uint32_t finish_(uint32_t local)
      {
         const uint32_t g = _seg * slots_per_segment + local;
         _a->_ref_base[g].store(1, std::memory_order_relaxed);
         return g;
      }
      fixed_allocator* _a;
      uint32_t         _seg    = UINT32_MAX;
      uint32_t         _cursor = 0;
   };
}  // namespace sal2

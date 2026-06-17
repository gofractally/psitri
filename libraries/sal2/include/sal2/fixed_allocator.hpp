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

namespace sal2
{
   // Slots per 32 MB segment, rounded DOWN to a multiple of 64. cl9's 1152 B slot
   // leaves a small tail; cl1/cl2 divide evenly.
   inline constexpr uint32_t fa_segment_bytes = 32u * 1024 * 1024;
   constexpr uint32_t        slots_per_seg(uint32_t slot_bytes) noexcept
   {
      return (fa_segment_bytes / slot_bytes) & ~63u;
   }

   template <uint32_t SlotBytes>
   class fixed_allocator
   {
     public:
      static constexpr uint32_t slot_bytes        = SlotBytes;
      static constexpr uint32_t slots_per_segment = slots_per_seg(SlotBytes);
      static_assert(slots_per_segment >= 64 && slots_per_segment % 64 == 0);

      struct config
      {
         std::filesystem::path dir;
         std::string           name;  // file stem, e.g. "cl1"
         uint32_t              max_segments = 16384;
         double                reuse_at     = 0.5;  // re-offer a segment at ≥50% free
      };

      class session;

      explicit fixed_allocator(config cfg) : _cfg(std::move(cfg))
      {
         std::filesystem::create_directories(_cfg.dir);
         _data = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".dat"),
             fa_segment_bytes /* power-of-2; cl9 wastes a tiny tail */, _cfg.max_segments,
             sal::access_mode::read_write);
         _refs = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".rc"),
             std::bit_ceil(uint64_t(slots_per_segment)) /* 1 B/slot, pow2 */, _cfg.max_segments,
             sal::access_mode::read_write);
         _seg_alloced = std::make_unique<std::atomic<uint32_t>[]>(_cfg.max_segments);
         _seg_freed   = std::make_unique<std::atomic<uint32_t>[]>(_cfg.max_segments);
         _seg_sealed  = std::make_unique<std::atomic<uint8_t>[]>(_cfg.max_segments);
         _recyc       = std::make_unique<std::atomic<uint64_t>[]>((_cfg.max_segments + 63) / 64);
         _base        = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _ref_base =
             reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
      }

      // ── addressing (slot index → bytes) ──────────────────────────────────────
      void*       get(uint32_t slot) noexcept { return _base + size_t(slot) * SlotBytes; }
      const void* get(uint32_t slot) const noexcept { return _base + size_t(slot) * SlotBytes; }

      // ── refcount ─────────────────────────────────────────────────────────────
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
               throw std::runtime_error("sal2: fixed_allocator refcount overflow");
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
      // Drop one ref. Returns true iff this was the LAST ref — in which case the
      // slot is left in the DYING sentinel (not 0): the caller then reads its
      // content to recurse children and finally free()s it (which sets ref=0 =
      // reusable). The write-around alloc reuses ONLY ref==0, so a dying slot is
      // never grabbed mid-teardown (the reader-vs-writer reclaim race). CAS (not
      // fetch_sub) so there is no transient ref==0 window before the sentinel.
      [[nodiscard]] bool release_one(uint32_t i) noexcept
      {
         auto&   r = _ref_base[i];
         uint8_t c = r.load(std::memory_order_acquire);
         for (;;)
         {
            assert(c >= 1 && c != dying_ref && "release of a dead/dying slot");
            const uint8_t nv = (c == 1) ? dying_ref : uint8_t(c - 1);
            if (r.compare_exchange_weak(c, nv, std::memory_order_acq_rel, std::memory_order_acquire))
               return c == 1;
         }
      }
      // Mark a slot dead (refcount already 0). Per the SAL segment model a freed
      // slot is NOT reused until the WHOLE segment recycles: when a SEALED (full)
      // segment crosses >50% free it joins the recyclable bitset, and the next
      // new-segment request pulls it (writing around the live leftovers) before
      // growing fresh. The alloc hot path is a pure bump that never consults this,
      // so the bg release thread's frees never contend with allocations.
      void free(uint32_t i) noexcept
      {
         // Publish the slot as REUSABLE (ref dying→0) with RELEASE: the write-around
         // alloc's ACQUIRE load of ref==0 then happens-after this, ordering the
         // releaser's content reads (done before free()) ahead of any reuse-write.
         // Until this store the slot is the dying sentinel (write-around skips it),
         // so the teardown reads it race-free.
         _ref_base[i].store(0, std::memory_order_release);
         _freed.fetch_add(1, std::memory_order_relaxed);
         const uint32_t s = i / slots_per_segment;
         // bg writes _seg_freed; foreground writes _seg_alloced — separate arrays
         // (separate cachelines) so the two never false-share. live = the diff.
         const uint32_t freed = _seg_freed[s].fetch_add(1, std::memory_order_relaxed) + 1;
         const uint32_t live  = _seg_alloced[s].load(std::memory_order_acquire) - freed;
         if (live <= recycle_below && _seg_sealed[s].load(std::memory_order_acquire))
            _recyc[s >> 6].fetch_or(1ull << (s & 63), std::memory_order_relaxed);
      }

      // ── sessions ───────────────────────────────────────────────────────────
      session start_session() { return session(this); }

      // ── stats ────────────────────────────────────────────────────────────────
      uint32_t num_segments() const noexcept { return _nseg.load(std::memory_order_acquire); }
      // live = bump-allocated minus freed (no freeset; the bump cursor never
      // returns slots, so a counter pair is exact).
      uint64_t live_slots() noexcept
      {
         return _allocated.load(std::memory_order_relaxed) - _freed.load(std::memory_order_relaxed);
      }
      uint64_t live_bytes() noexcept { return live_slots() * uint64_t(SlotBytes); }

     private:
      friend class session;

      uint32_t grow_()
      {
         std::lock_guard<std::mutex> g(_grow_mx);
         const uint32_t              s = _nseg.load(std::memory_order_relaxed);
         if (s >= _cfg.max_segments)
            throw std::runtime_error("sal2: fixed_allocator full");
         _data->alloc();
         _refs->alloc();
         _base = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _ref_base =
             reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
         _seg_alloced[s].store(0, std::memory_order_relaxed);
         _seg_freed[s].store(0, std::memory_order_relaxed);
         _seg_sealed[s].store(0, std::memory_order_relaxed);
         _nseg.store(s + 1, std::memory_order_release);
         return s;
      }

      // Pull the lowest-numbered recyclable (>50%-free, sealed) segment, or
      // UINT32_MAX. Clears its bit + sealed flag (the caller re-fills it).
      uint32_t take_recyclable_() noexcept
      {
         const uint32_t words = (_nseg.load(std::memory_order_acquire) + 63) >> 6;
         for (uint32_t w = 0; w < words; ++w)
         {
            uint64_t m = _recyc[w].load(std::memory_order_relaxed);
            while (m)
            {
               const uint32_t s = w * 64 + uint32_t(std::countr_zero(m));
               if (_recyc[w].fetch_and(~(1ull << (s & 63)), std::memory_order_acquire) &
                   (1ull << (s & 63)))
               {
                  _seg_sealed[s].store(0, std::memory_order_relaxed);
                  return s;
               }
               m = _recyc[w].load(std::memory_order_relaxed);  // lost the race — re-read
            }
         }
         return UINT32_MAX;
      }

      static constexpr uint32_t recycle_below = slots_per_segment / 2;  // >50% free ⇒ recyclable

      config                                _cfg;
      std::unique_ptr<sal::block_allocator> _data, _refs;
      std::byte*                            _base     = nullptr;
      std::atomic<uint8_t>*                 _ref_base = nullptr;
      std::atomic<uint32_t>                 _nseg{0};
      std::mutex                            _grow_mx;
      // bump-model accounting + recycle (the freeset is no longer the authority;
      // liveness is the refcount array, ref==0 ⇒ a free slot to write around).
      // alloc-side and free-side counters live on separate cachelines so the
      // foreground (alloc) and the bg release thread (free) never false-share.
      alignas(64) std::atomic<uint64_t> _allocated{0};        // foreground
      alignas(64) std::atomic<uint64_t> _freed{0};            // bg release thread
      std::unique_ptr<std::atomic<uint32_t>[]> _seg_alloced;  // foreground: allocs / segment
      std::unique_ptr<std::atomic<uint32_t>[]> _seg_freed;    // bg: frees / segment
      std::unique_ptr<std::atomic<uint8_t>[]>  _seg_sealed;   // 1 = full/retired
      std::unique_ptr<std::atomic<uint64_t>[]> _recyc;        // bitset of recyclable segments
   };

   // One writer's bump cursor within an active segment (spec req 4/5/7).
   template <uint32_t SlotBytes>
   class fixed_allocator<SlotBytes>::session
   {
     public:
      session(const session&)            = delete;
      session& operator=(const session&) = delete;

      // Allocate one slot (global index), ref stamped to 1. Forward BUMP within
      // the active segment: a FRESH segment is a pure cursor bump; a RECYCLED
      // segment write-arounds by skipping live (ref!=0) leftovers. Either way no
      // freeset CAS, and freed slots are reused only via whole-segment recycle.
      uint32_t alloc()
      {
         if (_fresh)
         {
            if (_pos < slots_per_segment)
               return finish_(_pos++);
            return alloc_slow_();
         }
         const std::atomic<uint8_t>* refs = _a->_ref_base + size_t(_seg) * slots_per_segment;
         while (_pos < slots_per_segment)
         {
            // ACQUIRE: seeing ref==0 must happen-after the releaser's release_one
            // (release-store of 0), so the releaser's reads of the slot's content
            // (done BEFORE its release_one) are ordered before we reuse+overwrite it.
            if (refs[_pos].load(std::memory_order_acquire) == 0)
               return finish_(_pos++);  // dead → write around
            ++_pos;                     // skip live leftover
         }
         return alloc_slow_();
      }

     private:
      friend class fixed_allocator<SlotBytes>;
      explicit session(fixed_allocator* a) noexcept : _a(a) {}

      // Active segment exhausted → seal it (now eligible for recycle), then pull a
      // recyclable (>50%-free) segment and write around its leftovers; only if
      // none is available do we grow a fresh segment.
      uint32_t alloc_slow_()
      {
         if (_seg != UINT32_MAX)
         {
            _a->_seg_sealed[_seg].store(1, std::memory_order_release);
            // catch a segment that went >50% free while still active.
            const uint32_t live = _a->_seg_alloced[_seg].load(std::memory_order_relaxed) -
                                  _a->_seg_freed[_seg].load(std::memory_order_acquire);
            if (live <= recycle_below)
               _a->_recyc[_seg >> 6].fetch_or(1ull << (_seg & 63), std::memory_order_relaxed);
         }
         const uint32_t r = _a->take_recyclable_();
         if (r != UINT32_MAX)
         {
            _seg   = r;
            _pos   = 0;
            _fresh = false;
         }
         else
         {
            _seg   = _a->grow_();
            _pos   = 0;
            _fresh = true;
         }
         return alloc();
      }
      uint32_t finish_(uint32_t local) noexcept
      {
         const uint32_t g = _seg * slots_per_segment + local;
         _a->_ref_base[g].store(1, std::memory_order_relaxed);
         _a->_seg_alloced[_seg].fetch_add(1, std::memory_order_release);
         _a->_allocated.fetch_add(1, std::memory_order_relaxed);
         return g;
      }
      fixed_allocator* _a;
      uint32_t         _seg   = UINT32_MAX;
      uint32_t         _pos   = slots_per_segment;  // forces a fresh segment on first alloc
      bool             _fresh = true;
   };
}  // namespace sal2

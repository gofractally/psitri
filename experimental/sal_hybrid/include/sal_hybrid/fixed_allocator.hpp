// sal_hybrid::fixed_allocator<SlotBytes> — the FIX-region per-size-class slab.
//
// Spec (libart++/docs/sal_hybrid.md, dev-plan step 1): "first build and test
// FixedAllocator<N> ... robust against random alloc/free." One complete
// allocator for ONE fixed slot size; the FIX region is three of them:
//   FixedAllocator<128>  = cl1  (setlist / nested_setlist)
//   FixedAllocator<256>  = cl2  (double_setlist / big_nested)
//   FixedAllocator<1152> = cl9  (full / node256)
//
// ALLOCATION AUTHORITY = the freeset (start_ref::arena's model — arena.hpp:5).
// Each segment owns one IN-MEMORY seg_freeset (1 bit per fixed slot, free=1):
// find-first-free-at-or-after a floor, first-fit-forward. The bump position is
// ONLY a next-scan HINT for where that scan starts; freed slots inside the
// current word are refilled (recycled) BEFORE the scan advances — that "stay in
// the word until 100% used" behaviour falls out of the bitmap, it is not coded.
// This REVERTS the prior pure-bump deviation (which grew fresh segments past
// freed slots — the measured rcart bulk-load deficit vs start_ref).
//
// The freeset is DERIVED, NEVER SYNCED (arena.hpp:5): in-memory only, rebuilt
// on reopen from ref>0 — no .fs file (that avoids sal_hybrid's prior sub-page .fs
// EINVAL). Per-slot refcount = liveness (Dan: `created`/birth dropped, denser).
// A write session owns an active segment, scans the freeset forward from the
// txn floor, and caches at/after-floor frees in a per-session hot-hole stack
// (start_ref::session::_fcache); cold (behind-floor) frees go to the freeset and
// wait for the cursor/low_water to revisit. Slot index is GLOBAL: addr = base +
// slot·SlotBytes. SAFETY: a freeset bit is set free ONLY when the slot is truly
// free (ref hit 0 / released = no reader); find-first-free returns only free
// bits, so it never hands out a slot a reader holds (same invariant as
// start_ref). The _floor split is hot/cold locality, NOT safety.
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

#include <sal_hybrid/config.hpp>
#include <sal_hybrid/freeset.hpp>

namespace sal_hybrid
{
   // Slots per 32 MB segment, rounded DOWN to a multiple of 64. cl9's 1152 B slot
   // leaves a small tail; cl1/cl2 divide evenly.
   inline constexpr uint32_t fa_segment_bytes = 32u * 1024 * 1024;
   constexpr uint32_t        slots_per_seg(uint32_t slot_bytes) noexcept
   {
      return (fa_segment_bytes / slot_bytes) & ~63u;
   }

   // WithBirth (gcart, compile-time): allocate a parallel per-slot u32 BIRTH array — the
   // .bv sibling of .rc — so the COW birth==V latch reads a dense array (denser GC sweep
   // + more app-data room) instead of a node-header field. rcart leaves it false (refcount
   // engine): the birth files/members/methods compile out entirely (no runtime branch,
   // and a stray birth() call on a WithBirth=false instance is a static_assert error).
   template <uint32_t SlotBytes, bool WithBirth = false>
   class fixed_allocator
   {
     public:
      static constexpr uint32_t slot_bytes        = SlotBytes;
      static constexpr uint32_t slots_per_segment = slots_per_seg(SlotBytes);
      static_assert(slots_per_segment >= 64 && slots_per_segment % 64 == 0);

      // One freeset per segment: 1 bit per FIXED SLOT (single-size slab ⇒ 1 unit ==
      // 1 slot; NOT line-granular). Authority for who is free. In-memory only.
      using freeset_t = seg_freeset_t<slots_per_segment>;

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
         // gcart only (WithBirth): the parallel u32 BIRTH array (.bv). Persisted (the COW
         // birth==V latch on reopened committed nodes needs birth < the new write version),
         // grown block-for-block with .dat/.rc. One block per data segment, sized to hold
         // all of a segment's slots' u32 births (rounded up to a pow2 the block_allocator
         // accepts). For rcart (WithBirth=false) this whole branch compiles out.
         if constexpr (WithBirth)
            _births = std::make_unique<sal::block_allocator>(
                _cfg.dir / (_cfg.name + ".bv"),
                std::bit_ceil(uint64_t(slots_per_segment) * 4u) /* 4 B/slot, pow2 */,
                _cfg.max_segments, sal::access_mode::read_write);
         _seg_alloced = std::make_unique<std::atomic<uint32_t>[]>(_cfg.max_segments);
         _seg_freed   = std::make_unique<std::atomic<uint32_t>[]>(_cfg.max_segments);
         _seg_sealed  = std::make_unique<std::atomic<uint8_t>[]>(_cfg.max_segments);
         _recyc       = std::make_unique<std::atomic<uint64_t>[]>((_cfg.max_segments + 63) / 64);
         // Freeset is in-memory + derived, NEVER on disk (arena.hpp:5): one slot per
         // segment in this array of unique_ptrs (heap so 32 KB/segment stays off the
         // class footprint and is allocated only for segments that exist).
         _freeset = std::make_unique<std::unique_ptr<freeset_t>[]>(_cfg.max_segments);
         _base    = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _ref_base =
             reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
         if constexpr (WithBirth)
            _birth_base = reinterpret_cast<uint32_t*>(const_cast<void*>(_births->mapped_base()));
         adopt_existing_();
      }

      // ── reopen recovery ───────────────────────────────────────────────────────
      // The block_allocator already remmaps the .dat (data) + .rc (refcounts) files
      // on construction, so a clean reopen's slot contents AND per-slot refcounts
      // survive for free. What's lost is the in-memory view: the segment count, the
      // bump/recycle accounting, and the (derived, never-synced) freeset. Restore
      // it: tell the block_allocators they already hold `existing` blocks (so the
      // next alloc() appends, never overwrites), set _nseg, then SCAN the preserved
      // refcounts to rebuild each segment's live count AND its freeset (free bit =
      // ref==0). Sealed existing segments that are >50% dead rejoin the recyclable
      // set, so their dead slots are written around (space reclaimed) while live
      // slots (ref!=0) are stepped over — no data loss. This is the "rebuild by
      // scan" path; cheap at the small DB sizes where reopen happens.
      void adopt_existing_() noexcept
      {
         const uint32_t existing = uint32_t(_data->mapped_blocks());
         if (existing == 0)
            return;
         _data->resize(existing);  // _num_blocks = existing → alloc() appends after
         _refs->resize(existing);
         _base     = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _ref_base = reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
         if constexpr (WithBirth)
         {
            // The .bv mmap already restored its persisted births (birth survives reopen
            // so committed nodes read birth < the next write version and clone).
            _births->resize(existing);
            _birth_base = reinterpret_cast<uint32_t*>(const_cast<void*>(_births->mapped_base()));
         }
         _nseg.store(existing, std::memory_order_relaxed);
         uint64_t total_live = 0;
         for (uint32_t s = 0; s < existing; ++s)
         {
            _freeset[s]   = std::make_unique<freeset_t>();
            freeset_t& fs = *_freeset[s];
            fs.init(/*all_free=*/false);  // start fully USED; mark free what is ref==0
            uint32_t       live = 0;
            const uint32_t base = s * slots_per_segment;
            for (uint32_t k = 0; k < slots_per_segment; ++k)
            {
               const uint8_t r = _ref_base[base + k].load(std::memory_order_relaxed);
               if (r != 0 && r != dying_ref)
                  ++live;
               else
                  fs.free1(k);  // ref 0 (or dying) ⇒ a free slot the freeset may hand out
            }
            _seg_alloced[s].store(live, std::memory_order_relaxed);
            _seg_freed[s].store(0, std::memory_order_relaxed);
            _seg_sealed[s].store(1, std::memory_order_relaxed);  // retired; sessions grow/recycle
            total_live += live;
            if (live <= recycle_below)
               _recyc[s >> 6].fetch_or(1ull << (s & 63), std::memory_order_relaxed);
         }
         _allocated.store(total_live, std::memory_order_relaxed);
         _freed.store(0, std::memory_order_relaxed);
      }

      // ── addressing (slot index → bytes) ──────────────────────────────────────
      void*       get(uint32_t slot) noexcept { return _base + size_t(slot) * SlotBytes; }
      const void* get(uint32_t slot) const noexcept { return _base + size_t(slot) * SlotBytes; }

      // ── birth version (gcart parallel array; the .bv sibling of .rc) ───────────
      // Writer-thread-only: the COW birth==V latch reads it; build/clone stamp it.
      // Readers NEVER touch birth (hazard-pointer design) and the bg reaper decides
      // by death, so birth needs no atomic. A freed slot's entry is stale until the
      // next alloc re-stamps it (every fresh build calls set_birth) — harmless. Only
      // exists when WithBirth (gcart); a call on a WithBirth=false (rcart) instance is
      // a compile error (the static_asserts below). Slot index is GLOBAL.
      uint32_t birth(uint32_t slot) const noexcept
      {
         static_assert(WithBirth, "birth() requires fixed_allocator<…, WithBirth=true>");
         return _birth_base[slot];
      }
      void set_birth(uint32_t slot, uint32_t v) noexcept
      {
         static_assert(WithBirth, "set_birth() requires fixed_allocator<…, WithBirth=true>");
         _birth_base[slot] = v;
      }
      // Prefetch a slot's birth line so the COW latch's array read overlaps the node
      // load (Dan: a prefetch of the node before reading birth hides the array touch).
      void prefetch_birth(uint32_t slot) const noexcept
      {
         static_assert(WithBirth, "prefetch_birth() requires fixed_allocator<…, WithBirth=true>");
#if defined(__GNUC__) || defined(__clang__)
         __builtin_prefetch(_birth_base + slot, 0, 3);
#else
         (void)slot;
#endif
      }

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
               throw std::runtime_error("sal_hybrid: fixed_allocator refcount overflow");
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
      // content to recurse children and finally free()s it (which marks the freeset
      // bit free = reusable). find-first-free skips the dying slot (its freeset bit
      // is still 0/used until free() runs), so it is never grabbed mid-teardown (the
      // reader-vs-writer reclaim race). CAS (not fetch_sub) so there is no transient
      // ref==0 window before the sentinel.
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
      // Free a slot whose refcount already hit 0 (dying sentinel). Sets the freeset
      // bit FREE (the authority) — find-first-free may now hand it out. SAFETY: this
      // runs only after release_one returned the last-ref; no reader holds it. The
      // store(0) publishes "reusable" with RELEASE so a reader's content reads (done
      // before release_one) are ordered ahead of any reuse-overwrite; the freeset's
      // own free1/alloc1 carry the matching release/acquire on l0.
      void free(uint32_t i) noexcept
      {
         _ref_base[i].store(0, std::memory_order_release);
         _freed.fetch_add(1, std::memory_order_relaxed);
         const uint32_t s     = i / slots_per_segment;
         const uint32_t local = i % slots_per_segment;
         // Authority: return the bit to the segment's freeset. low_water inside
         // free1 lets the owning session's next scan revisit this hole (recycle).
         _freeset[s]->free1(local);
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
      // live = allocated minus freed (the counter pair is exact; the freeset is the
      // placement authority but the counters are the cheap census).
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
            throw std::runtime_error("sal_hybrid: fixed_allocator full");
         _data->alloc();
         _refs->alloc();
         _base = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _ref_base =
             reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
         if constexpr (WithBirth)
         {
            _births->alloc();  // one .bv block per data segment (re-fetch the base: may remap)
            _birth_base = reinterpret_cast<uint32_t*>(const_cast<void*>(_births->mapped_base()));
         }
         _seg_alloced[s].store(0, std::memory_order_relaxed);
         _seg_freed[s].store(0, std::memory_order_relaxed);
         _seg_sealed[s].store(0, std::memory_order_relaxed);
         // Fresh segment: every slot free (the freeset is init all-free).
         _freeset[s] = std::make_unique<freeset_t>();
         _freeset[s]->init(/*all_free=*/true);
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

      freeset_t& freeset_(uint32_t s) noexcept { return *_freeset[s]; }

      static constexpr uint32_t recycle_below = slots_per_segment / 2;  // >50% free ⇒ recyclable

      config                                _cfg;
      std::unique_ptr<sal::block_allocator> _data, _refs;
      std::unique_ptr<sal::block_allocator> _births;  // gcart: per-slot u32 BIRTH (.bv); null for rcart
      std::byte*                            _base       = nullptr;
      std::atomic<uint8_t>*                 _ref_base   = nullptr;
      uint32_t*                             _birth_base = nullptr;  // _births->mapped_base() (gcart)
      std::atomic<uint32_t>                 _nseg{0};
      std::mutex                            _grow_mx;
      // ALLOCATION AUTHORITY: one in-memory freeset per segment (derived, never
      // synced). The counters below are the cheap live census — the freeset's l0
      // bitmap is the placement truth.
      std::unique_ptr<std::unique_ptr<freeset_t>[]> _freeset;
      // alloc-side and free-side counters live on separate cachelines so the
      // foreground (alloc) and the bg release thread (free) never false-share.
      alignas(64) std::atomic<uint64_t> _allocated{0};        // foreground
      alignas(64) std::atomic<uint64_t> _freed{0};            // bg release thread
      std::unique_ptr<std::atomic<uint32_t>[]> _seg_alloced;  // foreground: allocs / segment
      std::unique_ptr<std::atomic<uint32_t>[]> _seg_freed;    // bg: frees / segment
      std::unique_ptr<std::atomic<uint8_t>[]>  _seg_sealed;   // 1 = full/retired
      std::unique_ptr<std::atomic<uint64_t>[]> _recyc;        // bitset of recyclable segments
   };

   // One writer's freeset cursor within an active segment (start_ref::session).
   template <uint32_t SlotBytes, bool WithBirth>
   class fixed_allocator<SlotBytes, WithBirth>::session
   {
     public:
      session(const session&)            = delete;
      session& operator=(const session&) = delete;
      ~session() { flush_fcache_(); }

      // Mark the transaction boundary (start_ref::session::start_transaction):
      // _floor = _cursor = the current frontier. Frees AT/AFTER the floor this txn
      // recycle hot through _fcache (their pages are dirty anyway); frees BEHIND it
      // go to the freeset and wait for the cursor/segment to revisit. begin_txn must
      // be called at write-transaction start (the engine forwards it to cl1/cl2/cl9).
      void begin_txn() noexcept
      {
         flush_fcache_();  // leftovers from the prior txn → freeset (cold now)
         _floor = _frontier;
      }

      // Allocate one slot (global index), ref stamped to 1. start_ref's model:
      //  1. pop the per-session hot-hole cache (at/after-floor freed this txn) —
      //     its page is already dirty, so reuse is free and skips the freeset.
      //  2. else find-first-free in the active segment's freeset, scanning from the
      //     cursor hint (>= floor); low_water pulls the scan back to a behind-cursor
      //     hole, so freed slots in the current word REFILL before the cursor
      //     advances (the recycle — the bump never returns slots, the freeset does).
      uint32_t alloc()
      {
         if (_fcache_n)
            return finish_(_fcache[--_fcache_n]);
         if (_seg != UINT32_MAX)
         {
            const uint32_t g = alloc_from_seg_();
            if (g != UINT32_MAX)
               return finish_(g);
         }
         return alloc_slow_();
      }

      // Hot-hole free: a slot freed AT/AFTER this txn's floor (its page is already
      // dirty) is cached for free reuse this txn; below-floor (cold) frees go to the
      // allocator's freeset directly. The engine may route the foreground COW reclaim
      // here; the allocator's free() is the always-correct any-thread authority path.
      void free(uint32_t i) noexcept
      {
         if (i >= _floor && _fcache_n < fcache_cap)
         {
            // born this txn / page dirty: stamp ref reusable + cache the index.
            _a->_ref_base[i].store(0, std::memory_order_release);
            _a->_freed.fetch_add(1, std::memory_order_relaxed);
            const uint32_t s = i / slots_per_segment;
            const uint32_t freed = _a->_seg_freed[s].fetch_add(1, std::memory_order_relaxed) + 1;
            const uint32_t live  = _a->_seg_alloced[s].load(std::memory_order_acquire) - freed;
            if (live <= recycle_below && _a->_seg_sealed[s].load(std::memory_order_acquire))
               _a->_recyc[s >> 6].fetch_or(1ull << (s & 63), std::memory_order_relaxed);
            _fcache[_fcache_n++] = i;
            return;
         }
         _a->free(i);  // cold or cache full → freeset authority
      }

     private:
      friend class fixed_allocator<SlotBytes, WithBirth>;
      explicit session(fixed_allocator* a) noexcept : _a(a) {}

      // find-first-free in the active segment from the cursor hint, refilling holes
      // behind the cursor via low_water (exact first-fit, bump-like cost). Returns a
      // GLOBAL slot index, or UINT32_MAX if the segment can't satisfy at/after floor.
      uint32_t alloc_from_seg_() noexcept
      {
         const uint32_t seg_base = _seg * slots_per_segment;
         freeset_t&     fs       = _a->freeset_(_seg);
         uint32_t local_floor = _floor > seg_base ? (_floor - seg_base) : 0u;
         if (local_floor >= slots_per_segment)
            return UINT32_MAX;  // floor past this segment → force a switch
         const uint32_t cur_local = _cursor > seg_base ? (_cursor - seg_base) : 0u;
         // low_water exchange: a hole freed behind the cursor announces itself, so
         // the scan starts at min(cursor, low_water) but never before the floor.
         const uint32_t lw   = fs.low_water.exchange(UINT32_MAX, std::memory_order_relaxed);
         uint32_t       from = cur_local < lw ? cur_local : lw;
         if (from < local_floor)
            from = local_floor;
         const uint32_t local = fs.alloc1(from);
         if (local == UINT32_MAX)
            return UINT32_MAX;
         // Put the low_water mark back if it pointed at a still-unconsumed hole
         // at/after the floor below where we landed (the next alloc revisits it).
         if (lw != UINT32_MAX && lw >= local_floor && local > lw)
         {
            auto cur = fs.low_water.load(std::memory_order_relaxed);
            while (lw < cur && !fs.low_water.compare_exchange_weak(cur, lw,
                                                                   std::memory_order_relaxed))
               ;
         }
         _cursor = seg_base + local + 1;  // hint: next scan starts just past here
         return seg_base + local;
      }

      // Active segment can't satisfy at/after the floor → seal it (eligible for
      // recycle), pull a recyclable (>50%-free) segment whose freeset still has
      // free bits, else grow fresh. Then retry the freeset scan.
      uint32_t alloc_slow_()
      {
         if (_seg != UINT32_MAX)
         {
            _a->_seg_sealed[_seg].store(1, std::memory_order_release);
            const uint32_t live = _a->_seg_alloced[_seg].load(std::memory_order_relaxed) -
                                  _a->_seg_freed[_seg].load(std::memory_order_acquire);
            if (live <= recycle_below)
               _a->_recyc[_seg >> 6].fetch_or(1ull << (_seg & 63), std::memory_order_relaxed);
         }
         for (;;)
         {
            const uint32_t r = _a->take_recyclable_();
            if (r != UINT32_MAX)
               _seg = r;
            else
               _seg = _a->grow_();
            const uint32_t seg_base = _seg * slots_per_segment;
            // Switching segments RESETS the floor + cursor to the new segment's base
            // (start_ref::alloc_slow_): the floor is the first-fit scan start WITHIN
            // the active segment, not a monotonic global — a recycled low-numbered
            // segment must be scannable from its base or its freed slots never reuse.
            _floor  = seg_base;
            _cursor = seg_base;
            const uint32_t g = alloc_from_seg_();
            if (g != UINT32_MAX)
               return finish_(g);
            // Recycled segment had no free bit at/after floor — seal + try again.
            _a->_seg_sealed[_seg].store(1, std::memory_order_release);
         }
      }

      uint32_t finish_(uint32_t g) noexcept
      {
         const uint32_t s = g / slots_per_segment;
         _a->_ref_base[g].store(1, std::memory_order_relaxed);
         _a->_seg_alloced[s].fetch_add(1, std::memory_order_release);
         _a->_allocated.fetch_add(1, std::memory_order_relaxed);
         if (g + 1 > _frontier)
            _frontier = g + 1;
         return g;
      }

      void flush_fcache_() noexcept
      {
         // Leftover at/after-floor holes return to the freeset authority (cold for
         // the next txn — start_ref::flush_fcache_). The ref store(0) + counters
         // already happened in free(); only the freeset bit is outstanding.
         while (_fcache_n)
         {
            const uint32_t i = _fcache[--_fcache_n];
            _a->_freeset[i / slots_per_segment]->free1(i % slots_per_segment);
         }
      }

      static constexpr uint32_t fcache_cap = 256;  // hot-hole stack depth (start_ref §7)

      fixed_allocator* _a;
      uint32_t         _seg      = UINT32_MAX;            // current active segment
      uint32_t         _floor    = 0;                     // txn start (global slot)
      uint32_t         _cursor   = 0;                     // next-scan hint (global slot)
      uint32_t         _frontier = 0;                     // one past the highest alloc seen
      uint32_t         _fcache_n = 0;                     // hot-hole stack depth
      uint32_t         _fcache[fcache_cap];               // freed slots at/after floor this txn
   };
}  // namespace sal_hybrid

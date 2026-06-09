#pragma once
#include <cstdint>

// Range-refcount nibble math — the pure, side-effect-free core of the COW child-retain
// batching scheme (design: doc/range_refcount.md). Kept separate from the engine and the
// SAL block so the arithmetic is unit-tested in isolation ("correct by construction")
// before it is wired into the refcount-bearing engine paths.
//
// A big-tier node points at a shared "range block" (a custom SAL control_block): its
// share count = control_block.ref, and its packed user value holds NIB_COUNT 4-bit
// deferred-per-range counters n[r]. A child c in range r has
//     true_ref(c) = cb.ref(c) + n[r]
// and per live block the invariant n[r] = share - 1 holds (the block's one base retain
// per child covers version #1; the nibble covers the rest). A block never overflows: a
// bump that would hit NIB_MAX forks a fresh block instead, so releases are always a
// cheap decrement until the block's last version (block death releases the base).

namespace heart::rr
{
   inline constexpr int      nib_count = 8;    // 8 ranges (256 branches / 8 = 32 per range)
   inline constexpr uint64_t nib_max   = 15;   // 4-bit counter ceiling
   inline constexpr uint64_t nib_mask  = 0xFu;

   /// Range index for a branch number (top 3 bits of the 0..255 branch space).
   inline constexpr int range_of(unsigned branch) noexcept { return int((branch >> 5) & 7u); }

   /// Read / write one 4-bit counter inside the packed value (low 32 bits used).
   inline constexpr uint64_t get(uint64_t packed, int r) noexcept
   {
      return (packed >> (4 * r)) & nib_mask;
   }
   inline constexpr uint64_t with(uint64_t packed, int r, uint64_t v) noexcept
   {
      return (packed & ~(nib_mask << (4 * r))) | ((v & nib_mask) << (4 * r));
   }

   /// Would incrementing range r overflow the nibble? (→ caller must fork instead.)
   inline constexpr bool bump_overflows(uint64_t packed, int r) noexcept
   {
      return get(packed, r) >= nib_max;
   }

   /// Clone share: +1 deferred owner for every range. Returns the new packed value.
   /// Precondition: no range is at nib_max (caller checks needs_fork first).
   inline constexpr uint64_t bump_all(uint64_t packed) noexcept
   {
      uint64_t out = 0;
      for (int r = 0; r < nib_count; ++r)
         out = with(out, r, get(packed, r) + 1);
      return out;
   }
   inline constexpr bool needs_fork(uint64_t packed) noexcept
   {
      for (int r = 0; r < nib_count; ++r)
         if (get(packed, r) >= nib_max)
            return true;
      return false;
   }

   /// Release: -1 deferred owner for range r. If the nibble is already 0 the deferred
   /// pool is empty and the caller must do a real cb.release on each child in r
   /// (returns false → "materialized release needed"); otherwise decrements (returns true).
   inline constexpr bool dec_or_materialize(uint64_t& packed, int r) noexcept
   {
      uint64_t v = get(packed, r);
      if (v == 0)
         return false;
      packed = with(packed, r, v - 1);
      return true;
   }

   /// Settle range r (divergence): the caller pushes n[r] real retains onto each child in
   /// r, then resets the nibble. Returns the count to push and zeroes the nibble in place.
   inline constexpr uint64_t settle(uint64_t& packed, int r) noexcept
   {
      uint64_t v = get(packed, r);
      packed     = with(packed, r, 0);
      return v;
   }
}  // namespace heart::rr

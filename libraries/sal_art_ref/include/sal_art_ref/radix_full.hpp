#pragma once
#include <cassert>

#include <sal_art_ref/smart_address.hpp>
#include <sal/numbers.hpp>

namespace sal_art_ref
{
   /**
    * The dense top tier: a byte-indexed array of `Bits` branches (smart_address —
    * each a child node pointer or a local inline tail allocation), with a null
    * smart_address marking bytes that have no child. Routing is a branchless
    * `branches[byte]` — no bitset, no rank, no dedup — at the cost of always
    * occupying the full size:
    *
    *   Bits=64  -> 256 B  (2 cachelines)
    *   Bits=128 -> 512 B  (4 cachelines)
    *   Bits=256 -> 1024 B (8 cachelines, 32 ptr_addresses each)
    *
    * A radix_bitset expands into this when it grows too dense to stay compact (see
    * radix_bitset::expand / ::contract_from). This is a generic container: it holds
    * no version/refcount/COW state — those belong to the node that embeds it.
    *
    * Trivially copyable — no constructors; call init() (which fills null) after
    * placement.
    */
   template <int Bits>
   struct radix_full
   {
      static_assert(Bits == 64 || Bits == 128 || Bits == 256, "radix_full Bits in {64,128,256}");
      static constexpr int range = Bits;

      smart_address branches[Bits];  // byte-indexed; a null smart_address == absent

      /// Fill every slot with a null smart_address.
      void init() noexcept
      {
         for (int i = 0; i < Bits; ++i)
            branches[i] = smart_address::null();
      }

      bool contains(int byte) const noexcept
      {
         assert(byte >= 0 && byte < Bits);
         return !branches[byte].is_null();
      }

      /// Branch for `byte` (child node pointer or local tail allocation), or a null
      /// smart_address if absent. EXACT — does not scan; use next_present /
      /// prev_present to seek to a neighbour.
      smart_address find(int byte) const noexcept
      {
         assert(byte >= 0 && byte < Bits);
         return branches[byte];
      }

      /// Seek: lowest present (non-null) byte at or above `from` — i.e. jump to
      /// `from` and scan RIGHT past nulls. Returns -1 if none. Mirrors
      /// bitset::next_set; iterate with next_present(b + 1).
      int next_present(int from) const noexcept
      {
         for (int i = from < 0 ? 0 : from; i < Bits; ++i)
            if (!branches[i].is_null())
               return i;
         return -1;
      }

      /// Seek: highest present (non-null) byte at or below `from` — jump to `from`
      /// and scan LEFT past nulls. Returns -1 if none. Mirrors bitset::prev_set;
      /// iterate with prev_present(b - 1).
      int prev_present(int from) const noexcept
      {
         for (int i = from >= Bits ? Bits - 1 : from; i >= 0; --i)
            if (!branches[i].is_null())
               return i;
         return -1;
      }

      int first_present() const noexcept { return next_present(0); }
      int last_present() const noexcept { return prev_present(Bits - 1); }

      /// Set `byte`'s branch (pass a null smart_address to clear). Always succeeds.
      void set_branch(int byte, smart_address a) noexcept
      {
         assert(byte >= 0 && byte < Bits);
         branches[byte] = a;
      }
      void clear_branch(int byte) noexcept { set_branch(byte, smart_address::null()); }

      /// Number of non-null entries (O(Bits) scan).
      int size() const noexcept
      {
         int n = 0;
         for (int i = 0; i < Bits; ++i)
            n += !branches[i].is_null();
         return n;
      }

      /// Mathematical lower/upper bound (NOT STL): bounding *keys*, found by the
      /// jump-then-scan seek. lower_bound(byte) = greatest present byte at or below
      /// (the floor) == prev_present; upper_bound(byte) = least present byte at or
      /// above (the ceiling) == next_present; -1 if none.
      int lower_bound(int byte) const noexcept { return prev_present(byte); }
      int upper_bound(int byte) const noexcept { return next_present(byte); }
   };

   static_assert(sizeof(radix_full<64>) == 256, "radix_full<64> = 2 cachelines");
   static_assert(sizeof(radix_full<128>) == 512, "radix_full<128> = 4 cachelines");
   static_assert(sizeof(radix_full<256>) == 1024, "radix_full<256> = 8 cachelines");
}  // namespace sal_art_ref

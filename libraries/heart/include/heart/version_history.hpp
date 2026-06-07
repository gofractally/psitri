#pragma once
#include <cstdint>

#include <sal/alloc_header.hpp>
#include <sal/control_block_alloc.hpp>  // version_t

namespace heart
{
   /**
    * One row of a version_history: the version this snapshot was created at, the
    * (owning) address of the frozen node at that version, and the (non-owning)
    * handle to the version_node whose liveness gates the row.
    *
    *   version_t (6) + node (4, owning) + handle (4, non-owning) = 14 B
    */
   struct version_history_entry
   {
      sal::version_t   version;  // creation version of the frozen snapshot
      sal::ptr_address node;     // owning: the frozen node at `version`
      sal::ptr_address handle;   // non-owning: the version_node defining the version's life
   } __attribute__((packed));
   static_assert(sizeof(version_history_entry) == 14);

   /**
    * A radix node's table of older versions (the thing versioned::version_history
    * points at). Built by clone+modify on each MVCC update; a reader at snapshot R
    * picks the largest `version <= R` and navigates to its frozen `node`. Pruned
    * (owning `node` released) when an entry's `handle` is detected dead.
    *
    * A plain COW node (not itself `versioned`). `N` entries fit a given allocation:
    * alloc_header(16) + count(1) + 14*N; N = 7 fills a 128 B cacheline (115 B).
    * When full, the update COWs one level up instead of extending (see design).
    */
   template <int N>
   struct version_history : sal::alloc_header
   {
      uint8_t               count;
      version_history_entry entries[N];

      void init_history(uint32_t asize, sal::header_type t, sal::ptr_address_seq seq) noexcept
      {
         sal::alloc_header::init(asize, t, seq);
         count = 0;
      }
      bool full() const noexcept { return count >= N; }
   } __attribute__((packed));

   static_assert(sizeof(version_history<7>) == 16 + 1 + 14 * 7, "alloc_header + count + 7 entries");
}  // namespace heart

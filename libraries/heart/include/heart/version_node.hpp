#pragma once
#include <sal/alloc_header.hpp>

namespace heart
{
   /**
    * A per-transaction version handle. Allocated at the start of a write txn; its
    * version number is its alloc_header creation version, and its *lifetime defines
    * the version's lifetime* — readers at that snapshot hold a reference to it.
    *
    * Dead-version detection (impl phase): a version_history entry's non-owning
    * handle points here; a version is dead once that handle no longer resolves to a
    * version_node whose version() matches (the control block was freed / reused).
    * Monotonic 48-bit versions make a reused block impossible to mistake for live.
    *
    * Minimal by design — existence + version number are the whole payload.
    */
   struct version_node : sal::alloc_header
   {
      uint64_t version() const noexcept { return sequence(); }

      void init_version_node(uint32_t asize, sal::header_type t, sal::ptr_address_seq seq) noexcept
      {
         sal::alloc_header::init(asize, t, seq);
      }
   } __attribute__((packed));

   static_assert(sizeof(version_node) == sizeof(sal::alloc_header), "version_node is a bare versioned handle");
}  // namespace heart

#pragma once
#include <sal/alloc_header.hpp>
#include <sal/numbers.hpp>

namespace sal_art_ref
{
   /**
    * The MVCC layer: any node that derives `versioned` participates in the generic
    * version-history protocol. It adds one pointer — to a version_history table of
    * older versions — on top of the SAL alloc_header. The node's *creation version*
    * is NOT stored here; it reuses the 48-bit `alloc_header` version field
    * (exposed as created_version()).
    *
    * Update protocol (impl phase): a write relocates the old node to a fresh
    * address, records {old created_version -> that address} in the version_history,
    * builds the new state, and repoints the stable control block at it — so the
    * stable ptr_address always resolves to the latest version and parents are not
    * touched (no COW-to-root). A reader at snapshot R uses a node directly when
    * created_version() <= R, else rewinds through version_history to the largest
    * entry <= R.
    *
    * Layout: alloc_header (16 B) + version_history ptr (4 B) = 20 B, packed.
    */
   struct versioned : sal::alloc_header
   {
      sal::ptr_address _version_history;  // -> version_history table, or null_ptr_address

      /// Node creation version — reuses the SAL alloc_header 48-bit field.
      uint64_t created_version() const noexcept { return sequence(); }

      bool             has_version_history() const noexcept { return _version_history != sal::null_ptr_address; }
      sal::ptr_address version_history() const noexcept { return _version_history; }
      void             set_version_history(sal::ptr_address a) noexcept { _version_history = a; }

      /// Initialize the alloc_header identity and an empty (null) version history.
      void init_versioned(uint32_t asize, sal::header_type t, sal::ptr_address_seq seq) noexcept
      {
         sal::alloc_header::init(asize, t, seq);
         _version_history = sal::null_ptr_address;
      }
   } __attribute__((packed));

   static_assert(sizeof(versioned) == sizeof(sal::alloc_header) + 4, "alloc_header(16) + version_history(4) = 20");
}  // namespace sal_art_ref

#pragma once
#include <sal/alloc_header.hpp>

namespace heart
{
   /**
    * SAL header_type ids for the radix node taxonomy (one per concrete stored type,
    * starting at header_type::start_user_type). Kept in one place so leaf_node,
    * radix_nodes, and the tree engine agree. These identify the *format* (what CL0
    * holds); the *dispatch shape* (r1/r2/r4/r8) is carried separately in the control
    * block, so the descent routes without reading the node header.
    */
   namespace radix_type_ids
   {
      inline constexpr auto base = unsigned(sal::header_type::start_user_type);  // 3

      inline constexpr sal::header_type leaf        = sal::header_type(base + 0);
      inline constexpr sal::header_type setlist_u8  = sal::header_type(base + 1);
      inline constexpr sal::header_type setlist_u16 = sal::header_type(base + 2);
      inline constexpr sal::header_type setlist_u32 = sal::header_type(base + 3);
      inline constexpr sal::header_type bitset1     = sal::header_type(base + 4);
      inline constexpr sal::header_type bitset_r2   = sal::header_type(base + 5);
      inline constexpr sal::header_type bitset_r4   = sal::header_type(base + 6);
      inline constexpr sal::header_type full_radix  = sal::header_type(base + 7);
      inline constexpr sal::header_type bucket      = sal::header_type(base + 8);  // 2-3-4 leaf bucket
   }  // namespace radix_type_ids
}  // namespace heart

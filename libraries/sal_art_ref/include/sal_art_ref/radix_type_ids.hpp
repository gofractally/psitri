#pragma once
#include <sal/alloc_header.hpp>

namespace sal_art_ref
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
      inline constexpr sal::header_type bitset1     = sal::header_type(base + 4);  // reserved: never produced
      inline constexpr sal::header_type bitset_r2   = sal::header_type(base + 5);
      inline constexpr sal::header_type bitset_r4   = sal::header_type(base + 6);
      inline constexpr sal::header_type full_radix  = sal::header_type(base + 7);
      inline constexpr sal::header_type bucket      = sal::header_type(base + 8);  // 2-3-4 leaf bucket
      // Single-cacheline narrow-window bitset tier (dense-in-a-narrow-range fan-out),
      // between setlist and the multi-CL R2/R4. Window [base, base+Width); r1-dispatched.
      inline constexpr sal::header_type bitset_c1_32  = sal::header_type(base + 9);
      inline constexpr sal::header_type bitset_c1_64  = sal::header_type(base + 10);
      inline constexpr sal::header_type bitset_c1_128 = sal::header_type(base + 11);
      // 2KB COW leaf page (leaf_page.hpp) — the page-leaf terminal tier of the
      // radix-indexed-leaf-pages model: the radix above stores DIVIDERS whose branch
      // targets / term_values point directly at pages.
      inline constexpr sal::header_type leaf_page = sal::header_type(base + 12);
      // Standalone (key,value) record (key_val_node.hpp): a page item whose suffix+value
      // exceed the in-page cap lives in its own allocation; the page keeps the suffix (or a
      // comparison fragment of it) + cbase code. NOT in radix_kind (never dispatched as a router).
      inline constexpr sal::header_type key_val = sal::header_type(base + 13);
      // Raw continuation payload; the directory and first payload live in key_val.
      inline constexpr sal::header_type value_chunk = sal::header_type(base + 14);

      // The same taxonomy as a STRONG enum (enumerator order == the header_type offsets above).
      // Engine dispatch switches on this — never on a raw int — so -Wswitch flags any switch that
      // forgets a case when a node type is added here. Add a kind → the compiler points at every
      // dispatch that must handle it.
      enum class radix_kind : uint8_t
      {
         leaf = 0,
         setlist_u8,
         setlist_u16,
         setlist_u32,
         bitset1,
         bitset_r2,
         bitset_r4,
         full,
         bucket,
         c1_32,
         c1_64,
         c1_128,
         page,  // 2KB leaf_page terminal (radix-indexed-leaf-pages model)
      };
      // A node's kind from its SAL header type (the radix tier ids are a contiguous block at base).
      inline constexpr radix_kind kind_of(sal::header_type t) noexcept
      {
         return radix_kind(unsigned(t) - base);
      }
   }  // namespace radix_type_ids
}  // namespace sal_art_ref

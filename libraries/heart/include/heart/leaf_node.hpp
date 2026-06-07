#pragma once
#include <heart/inline_value.hpp>
#include <heart/versioned.hpp>

namespace heart
{
   /**
    * A standalone key→value leaf: versioned (so it participates in MVCC like any
    * radix node) but NOT paged and not a radix node — it just carries one inline
    * suffix+value. Same `inline_value` shape a radix node tail-allocates, so the two
    * are interchangeable.
    *
    * Layout: versioned (20 B) + inline_value (3 B header + flexible data).
    */
   struct leaf_node : versioned
   {
      inline_value value;

      void init_leaf(uint32_t asize, sal::header_type t, sal::ptr_address_seq seq) noexcept
      {
         init_versioned(asize, t, seq);
      }
   } __attribute__((packed));

   static_assert(sizeof(leaf_node) == 23, "versioned(20) + inline_value header(3)");
}  // namespace heart

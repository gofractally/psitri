#pragma once
#include <cstring>
#include <span>

#include <heart/inline_value.hpp>
#include <heart/radix_type_ids.hpp>
#include <heart/versioned.hpp>
#include <ucc/round.hpp>

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
      static constexpr sal::header_type type_id = radix_type_ids::leaf;

      inline_value value;

      void init_leaf(uint32_t asize, sal::header_type t, sal::ptr_address_seq seq) noexcept
      {
         init_versioned(asize, t, seq);
      }

      // ---- SAL construction contract ----
      static uint32_t alloc_size(std::span<const uint8_t> suffix,
                                 std::span<const uint8_t> val) noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(uint32_t(
             sizeof(versioned) + inline_value::byte_size(int(suffix.size()), int(val.size()))));
      }
      leaf_node(uint32_t asize, sal::ptr_address_seq seq, std::span<const uint8_t> suffix,
                std::span<const uint8_t> val) noexcept
      {
         init_leaf(asize, type_id, seq);
         value.suffix_len = uint8_t(suffix.size());
         value.value_len  = uint8_t(val.size());
         value.value_type = 0;
         std::memcpy(value.data, suffix.data(), suffix.size());
         std::memcpy(value.data + suffix.size(), val.data(), val.size());
      }

      std::span<const uint8_t> suffix() const noexcept { return value.suffix(); }
      std::span<const uint8_t> value_bytes() const noexcept { return value.value(); }
      // A leaf owns no node children; alloc_header's default destroy()/visit_children
      // (no-ops) are correct.
   } __attribute__((packed));

   static_assert(sizeof(leaf_node) == 23, "versioned(20) + inline_value header(3)");
}  // namespace heart

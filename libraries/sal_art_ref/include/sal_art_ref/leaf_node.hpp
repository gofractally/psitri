#pragma once
#include <cstring>
#include <span>

#include <sal_art_ref/inline_value.hpp>
#include <sal_art_ref/radix_type_ids.hpp>
#include <sal_art_ref/versioned.hpp>
#include <ucc/round.hpp>

namespace sal_art_ref
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
      bool                     is_subtree() const noexcept { return value.is_subtree(); }
      sal::ptr_address         subtree_root() const noexcept { return value.subtree_root(); }

      // A byte-valued leaf owns no node children (default no-op). A SUBTREE leaf holds one ref on
      // its nested tree root (value_type == vt_subtree, value bytes = the root address) and must
      // release it when the leaf dies — the cascade that frees a composed subtree.
      template <typename Session>
      void destroy(const Session& s) const
      {
         if (value.is_subtree())
            s->release(value.subtree_root());
      }
   } __attribute__((packed));

   static_assert(sizeof(leaf_node) == 23, "versioned(20) + inline_value header(3)");
}  // namespace sal_art_ref

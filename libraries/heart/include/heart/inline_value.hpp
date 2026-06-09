#pragma once
#include <cstdint>
#include <cstring>
#include <span>

#include <sal/numbers.hpp>  // sal::ptr_address (subtree-root value bytes)

namespace heart
{
   /**
    * A suffix + value pair stored inline — the dynamic object both a leaf_node and
    * a radix node's tail allocation hold. A radix branch that consumes the rest of
    * the key points (via a smart_address tail offset) at one of these; a leaf_node
    * embeds one directly.
    *
    *   [ suffix_len | value_len | value_type | suffix[suffix_len] value[value_len] ]
    *
    * Capped so a single inline object stays cheap to copy-on-write:
    *   suffix_len + value_len + 3 <= max_inline_size (64).
    */
   struct inline_value
   {
      static constexpr int max_inline_size = 64;

      // value_type discriminator: what the value bytes MEAN.
      //   vt_bytes   — opaque user bytes (the default).
      //   vt_subtree — the value bytes are a 4-byte sal::ptr_address: the root of a nested tree
      //                (a first-class subtree value). Lifecycle code that retains/releases a
      //                terminal must, for vt_subtree, retain/release that embedded root.
      static constexpr uint8_t vt_bytes   = 0;
      static constexpr uint8_t vt_subtree = 1;

      uint8_t suffix_len;
      uint8_t value_len;
      uint8_t value_type;
      uint8_t data[];  // suffix[suffix_len] then value[value_len]

      std::span<const uint8_t> suffix() const noexcept { return {data, suffix_len}; }
      std::span<const uint8_t> value() const noexcept { return {data + suffix_len, value_len}; }
      bool                     is_subtree() const noexcept { return value_type == vt_subtree; }
      // The embedded subtree root (valid only when is_subtree()). Value bytes are a raw ptr_address.
      sal::ptr_address subtree_root() const noexcept
      {
         uint32_t raw = 0;
         std::memcpy(&raw, data + suffix_len, sizeof(raw));
         return sal::ptr_address(raw);
      }

      /// Total bytes occupied (header + suffix + value).
      int byte_size() const noexcept { return 3 + suffix_len + value_len; }

      static int byte_size(int suffix_n, int value_n) noexcept { return 3 + suffix_n + value_n; }
   } __attribute__((packed));

   static_assert(sizeof(inline_value) == 3, "fixed prefix is 3 bytes; data is a flexible array");
}  // namespace heart

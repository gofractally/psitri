#pragma once
#include <cstdint>
#include <span>

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

      uint8_t suffix_len;
      uint8_t value_len;
      uint8_t value_type;
      uint8_t data[];  // suffix[suffix_len] then value[value_len]

      std::span<const uint8_t> suffix() const noexcept { return {data, suffix_len}; }
      std::span<const uint8_t> value() const noexcept { return {data + suffix_len, value_len}; }

      /// Total bytes occupied (header + suffix + value).
      int byte_size() const noexcept { return 3 + suffix_len + value_len; }

      static int byte_size(int suffix_n, int value_n) noexcept { return 3 + suffix_n + value_n; }
   } __attribute__((packed));

   static_assert(sizeof(inline_value) == 3, "fixed prefix is 3 bytes; data is a flexible array");
}  // namespace heart

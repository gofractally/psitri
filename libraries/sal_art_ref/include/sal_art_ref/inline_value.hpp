#pragma once
#include <cstdint>
#include <cstring>
#include <span>

#include <sal/numbers.hpp>  // sal::ptr_address (subtree-root value bytes)

namespace sal_art_ref
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
      //   vt_extern  — the value lives in an external SAL object; the value bytes are a 2-byte
      //                cbase code private to the containing leaf_page (which owns the cbase
      //                table that decodes it to a ptr_address). Lifecycle code must
      //                retain/release the decoded address.
      //   vt_extern_val — VALUE-ONLY external: the FULL post-cpfx suffix stays in THIS record
      //                (the page's common prefix keeps it short), and only the VALUE lives in an
      //                external node (a key_val_node with key_len==0, possibly chunked). Payload is
      //                the same 2-byte cbase code as vt_extern, but the suffix is COMPLETE — so the
      //                key never needs the external object (exact-match compares the in-page
      //                suffix). Used when suffix <= ext_frag_cap; a longer suffix falls back to
      //                vt_extern (full key stored in the external node).
      static constexpr uint8_t vt_bytes      = 0;
      static constexpr uint8_t vt_subtree    = 1;
      static constexpr uint8_t vt_extern     = 2;
      static constexpr uint8_t vt_extern_val = 3;
      //   vt_typed   — a typed value (value_hooks.hpp): the value bytes are
      //                [u16 LE typenum][token], written by the engine. Stored inline like
      //                vt_bytes; its lifecycle fires the type's retain/release hooks. When the
      //                record does not fit in the page it is external (vt_extern /
      //                vt_extern_val) and the key_val_node carries vt_typed instead.
      static constexpr uint8_t vt_typed = 4;

      // The value bytes are stored in the record itself and sized by the caller.
      static constexpr bool inline_payload(uint8_t vt) noexcept
      {
         return vt == vt_bytes || vt == vt_typed;
      }

      // The value is stored in an external SAL object (record holds a 2-byte cbase code).
      static constexpr bool external_value(uint8_t vt) noexcept
      {
         return vt == vt_extern || vt == vt_extern_val;
      }
      // The in-page suffix is the COMPLETE post-cpfx suffix (only vt_extern keeps a fragment).
      static constexpr bool suffix_complete(uint8_t vt) noexcept { return vt != vt_extern; }

      uint8_t suffix_len;
      uint8_t value_len;
      uint8_t value_type;
      uint8_t data[];  // suffix[suffix_len] then value[value_len]

      std::span<const uint8_t> suffix() const noexcept { return {data, suffix_len}; }
      std::span<const uint8_t> value() const noexcept { return {data + suffix_len, value_len}; }
      bool                     is_subtree() const noexcept { return value_type == vt_subtree; }
      bool                     is_typed() const noexcept { return value_type == vt_typed; }
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
}  // namespace sal_art_ref

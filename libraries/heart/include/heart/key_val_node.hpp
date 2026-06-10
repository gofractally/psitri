#pragma once
#include <cstring>
#include <span>

#include <heart/inline_value.hpp>  // vt_* discriminators
#include <heart/radix_type_ids.hpp>
#include <heart/versioned.hpp>
#include <ucc/round.hpp>

namespace heart
{
   /**
    * A standalone (key, value) record: the page model's big-item overflow. A leaf_page
    * record is capped at 128 bytes; when suffix+value exceed that, the WHOLE item moves
    * here — its own allocation, independent of any page — and the page keeps only a
    * comparison FRAGMENT (the first bytes of the suffix) plus a 2-byte cbase code.
    *
    * Stores the FULL KEY (not a page-relative suffix): pages refactor their common
    * prefix on split/reprefix, but this object is immutable and shared — full-key
    * storage makes it valid under any cpfx (the same reason page fingerprints hash the
    * full key). u16 lengths: keys to max_key_len (1024, psitri parity), values to 64K-.
    *
    * value_type is the REAL discriminator (vt_bytes | vt_subtree) — the page-side
    * record says only vt_extern; a big-keyed subtree value lives here as vt_subtree
    * with the 4-byte root, released by destroy() like any subtree terminal.
    */
   struct key_val_node : versioned
   {
      static constexpr sal::header_type type_id     = radix_type_ids::key_val;
      static constexpr uint32_t         max_key_len = 1024;  // psitri max-key parity

      uint16_t key_len;
      uint16_t value_len;
      uint8_t  value_type;  // vt_bytes | vt_subtree (never vt_extern)
      uint8_t  data[];      // key[key_len] then value[value_len]

      static uint32_t alloc_size(std::span<const uint8_t> key, std::span<const uint8_t> val,
                                 uint8_t /*vtype*/) noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(
             uint32_t(sizeof(key_val_node)) + uint32_t(key.size()) + uint32_t(val.size()));
      }
      key_val_node(uint32_t asize, sal::ptr_address_seq seq, std::span<const uint8_t> key,
                   std::span<const uint8_t> val, uint8_t vtype) noexcept
      {
         init_versioned(asize, type_id, seq);
         assert(key.size() <= max_key_len);
         key_len    = uint16_t(key.size());
         value_len  = uint16_t(val.size());
         value_type = vtype;
         std::memcpy(data, key.data(), key.size());
         std::memcpy(data + key.size(), val.data(), val.size());
      }

      std::span<const uint8_t> key() const noexcept { return {data, key_len}; }
      std::span<const uint8_t> value_bytes() const noexcept
      {
         return {data + key_len, value_len};
      }
      bool             is_subtree() const noexcept { return value_type == inline_value::vt_subtree; }
      sal::ptr_address subtree_root() const noexcept
      {
         uint32_t raw = 0;
         std::memcpy(&raw, data + key_len, sizeof(raw));
         return sal::ptr_address(raw);
      }

      // SAL destroy hook: a subtree value owns one ref on its nested root.
      template <typename Session>
      void destroy(const Session& s) const
      {
         if (is_subtree())
            s->release(subtree_root());
      }
   } __attribute__((packed));

   static_assert(sizeof(key_val_node) == 25, "versioned(20) + key_len(2)+value_len(2)+vt(1)");
}  // namespace heart

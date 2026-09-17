#pragma once
#include <cstring>
#include <limits>
#include <span>

#include <sal_art_ref/inline_value.hpp>
#include <sal_art_ref/radix_type_ids.hpp>
#include <sal_art_ref/versioned.hpp>
#include <ucc/round.hpp>

namespace sal_art_ref
{
   struct chunk_entry
   {
      uint32_t         start;  // logical start of this continuation (never zero)
      sal::ptr_address child;
   } __attribute__((packed));
   static_assert(sizeof(chunk_entry) == 8);

   /**
    * External page value: optional full key, continuation directory, first payload.
    * Zero continuations means the entire value is contiguous in this allocation.
    * Each directory entry names a subsequent chunk; the first entry's start is
    * the first payload's length. The last continuation ends at value_len.
    *
    * Pages share this immutable object across COW/snapshots. Full keys survive
    * page reprefixing; vt_extern_val records instead keep their key in the page.
    * Subtree values have no continuations and hold one nested-root reference.
    */
   struct key_val_node : versioned
   {
      static constexpr sal::header_type type_id       = radix_type_ids::key_val;
      static constexpr uint32_t         max_key_len   = 1024;
      static constexpr uint32_t         max_value_len = std::numeric_limits<uint32_t>::max();

      uint32_t value_len;
      uint16_t key_len;
      uint16_t continuation_count;
      uint8_t  value_type;  // vt_bytes | vt_subtree
      uint8_t  data[];      // key, chunk_entry[continuation_count], first payload

      static uint32_t alloc_size(std::span<const uint8_t> key,
                                 std::span<const uint8_t> first,
                                 uint8_t,
                                 uint32_t                             = 0,
                                 std::span<const chunk_entry> entries = {}) noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(
             uint32_t(sizeof(key_val_node) + key.size() + entries.size_bytes() + first.size()));
      }
      key_val_node(uint32_t                     asize,
                   sal::ptr_address_seq         seq,
                   std::span<const uint8_t>     key,
                   std::span<const uint8_t>     first,
                   uint8_t                      vtype,
                   uint32_t                     total_size = 0,
                   std::span<const chunk_entry> entries    = {}) noexcept
      {
         init_versioned(asize, type_id, seq);
         assert(key.size() <= max_key_len);
         assert(entries.size() <= std::numeric_limits<uint16_t>::max());
         assert(entries.empty() ||
                (vtype == inline_value::vt_bytes && entries.front().start == first.size() &&
                 total_size > first.size()));
         value_len          = entries.empty() ? uint32_t(first.size()) : total_size;
         key_len            = uint16_t(key.size());
         continuation_count = uint16_t(entries.size());
         value_type         = vtype;
         if (!key.empty())
            std::memcpy(data, key.data(), key.size());
         if (!entries.empty())
            std::memcpy(data + key_len, entries.data(), entries.size_bytes());
         if (!first.empty())
            std::memcpy(data + key_len + entries.size_bytes(), first.data(), first.size());
      }

      bool                         chunked() const noexcept { return continuation_count != 0; }
      uint32_t                     value_size() const noexcept { return value_len; }
      std::span<const uint8_t>     key() const noexcept { return {data, key_len}; }
      std::span<const chunk_entry> chunks() const noexcept
      {
         return {reinterpret_cast<const chunk_entry*>(data + key_len), continuation_count};
      }
      std::span<const uint8_t> first_bytes() const noexcept
      {
         return {data + key_len + size_t(continuation_count) * sizeof(chunk_entry),
                 chunked() ? chunks().front().start : value_len};
      }
      std::span<const uint8_t> value_bytes() const noexcept
      {
         assert(!chunked());
         return first_bytes();
      }
      bool is_subtree() const noexcept { return value_type == inline_value::vt_subtree; }
      sal::ptr_address subtree_root() const noexcept
      {
         assert(is_subtree() && !chunked());
         uint32_t raw = 0;
         std::memcpy(&raw, data + key_len, sizeof(raw));
         return sal::ptr_address(raw);
      }

      template <typename Session>
      void destroy(const Session& s) const
      {
         if (is_subtree())
            s->release(subtree_root());
         for (const auto& entry : chunks())
            if (entry.child != sal::null_ptr_address)
               s->release(entry.child);
      }
      void visit_children(const std::function<void(sal::ptr_address)>& visit) const
      {
         if (is_subtree())
            visit(subtree_root());
         for (const auto& entry : chunks())
            if (entry.child != sal::null_ptr_address)
               visit(entry.child);
      }
   } __attribute__((packed));

   static_assert(sizeof(key_val_node) == 29,
                 "versioned(20) + value(4) + key(2) + count(2) + type(1)");
}  // namespace sal_art_ref

#pragma once

#include <algorithm>
#include <sal/allocator_session_impl.hpp>
#include <sal_art_ref/key_val_node.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace sal_art_ref
{
   struct value_chunk : sal::alloc_header
   {
      static constexpr auto type_id = radix_type_ids::value_chunk;
      uint32_t              payload_size;
      uint8_t               data[];
      static uint32_t       alloc_size(std::span<const uint8_t> value) noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(
             uint32_t(sizeof(value_chunk) + value.size()));
      }
      value_chunk(uint32_t size, sal::ptr_address_seq seq, std::span<const uint8_t> value) noexcept
          : sal::alloc_header(size, type_id, seq), payload_size(uint32_t(value.size()))
      {
         std::memcpy(data, value.data(), value.size());
      }
   } __attribute__((packed));
   static_assert(sizeof(value_chunk) == 20);
   static_assert(sal::allocator_session::fresh_allocation_capacity == sal::max_object_size);

   namespace large_value_detail
   {
      inline constexpr uint32_t split_threshold = 2 * 1024 * 1024;
      inline constexpr uint32_t minimum_tail    = 64 * 1024;
      inline constexpr uint32_t chunk_capacity =
          sal::allocator_session::fresh_allocation_capacity - sizeof(value_chunk);
      inline constexpr uint64_t max_value_size = key_val_node::max_value_len;

      struct allocation_plan
      {
         uint32_t first_size;
         uint16_t continuation_count;
         bool     start_fresh = false;
      };

      // Solve V <= (space - header - N * entry_size) + N * chunk_capacity
      // for the smallest N. Directory space is charged to the first chunk, so
      // this also handles values at the exact one-/two-segment boundaries.
      constexpr allocation_plan layout(uint32_t value_size,
                                       uint16_t key_size,
                                       uint32_t space) noexcept
      {
         const uint32_t header = sizeof(key_val_node) + key_size;
         if (space <= header)
            return {};
         const uint32_t available = space - header;
         if (value_size <= available)
            return {value_size, 0};
         constexpr uint32_t net_capacity = chunk_capacity - sizeof(chunk_entry);
         const uint64_t     count =
             (uint64_t(value_size) - available + net_capacity - 1) / net_capacity;
         const uint32_t directory_size = uint32_t(count * sizeof(chunk_entry));
         if (directory_size >= available)
            return {};
         assert(count <= std::numeric_limits<uint16_t>::max());
         return {available - directory_size, uint16_t(count)};
      }

      constexpr allocation_plan plan(uint32_t value_size,
                                     uint16_t key_size,
                                     uint32_t space) noexcept
      {
         auto tail = layout(value_size, key_size, space);
         if (tail.first_size == value_size && space >= sizeof(key_val_node) + key_size)
            return tail;
         auto fresh =
             layout(value_size, key_size, sal::allocator_session::fresh_allocation_capacity);
         fresh.start_fresh = true;
         if (tail.first_size < minimum_tail)
            return fresh;
         if (value_size <= split_threshold)
            return tail;
         if (!fresh.continuation_count)
            return fresh;
         auto score = [value_size](allocation_plan p, uint32_t skipped)
         {
            const uint64_t rest  = uint64_t(value_size) - p.first_size;
            const uint64_t large = (p.first_size > split_threshold) + rest / chunk_capacity +
                                   (rest % chunk_capacity > split_threshold);
            return std::tuple{large, uint32_t(p.continuation_count) + 1, skipped};
         };
         return score(fresh, space) < score(tail, 0) ? fresh : tail;
      }

      inline sal::ptr_address allocate(sal::allocator_session&  ses,
                                       std::span<const uint8_t> key,
                                       std::span<const uint8_t> value,
                                       uint8_t                  type)
      {
         if (key.size() > key_val_node::max_key_len || value.size() > max_value_size)
            throw std::invalid_argument("value or key exceeds the external-record limit");
         const auto placement =
             plan(uint32_t(value.size()), uint16_t(key.size()), ses.current_allocation_capacity());
         if (!placement.continuation_count)
            return ses.alloc<key_val_node>(sal::alloc_hint{}, key, value, type);
         assert(type == inline_value::vt_bytes);

         // Allocate all transient metadata before touching persistent state. A
         // partially constructed head owns only non-null children; each child is
         // transferred immediately from its temporary RAII owner into the head.
         std::vector<chunk_entry> entries(placement.continuation_count);
         uint64_t                 offset = placement.first_size;
         for (auto& entry : entries)
         {
            entry = {uint32_t(offset), sal::null_ptr_address};
            offset += std::min<uint64_t>(value.size() - offset, chunk_capacity);
         }
         assert(offset == value.size());
         if (placement.start_fresh)
            ses.next_allocation_segment();
         auto head = ses.smart_alloc<key_val_node>(
             sal::alloc_hint{}, key, value.first(placement.first_size), type,
             uint32_t(value.size()), std::span<const chunk_entry>(entries));
         SAL_CRASH_POINT("large:head_allocated");
         auto pin = ses.lock();
         auto ref = ses.get_ref<key_val_node>(head.address());
         assert(!ses.is_read_only(head.address()));
         auto  write     = ref.modify();
         auto* directory = reinterpret_cast<chunk_entry*>(write->data + key.size());
         // Rollover only finalizes segments; sync makes them read-only/eligible
         // for compaction. No sync or publication occurs until this head is
         // complete, so the first allocation remains writable and cannot move.
         for (size_t i = 0; i < entries.size(); ++i)
         {
            const uint32_t start = entries[i].start;
            const uint32_t end =
                i + 1 < entries.size() ? entries[i + 1].start : uint32_t(value.size());
            auto child =
                ses.smart_alloc<value_chunk>(sal::alloc_hint{}, value.subspan(start, end - start));
            SAL_CRASH_POINT("large:chunk_allocated");
            directory[i].child = child.take();
            SAL_CRASH_POINT("large:chunk_linked");
         }
         return head.take();
      }
   }  // namespace large_value_detail
}  // namespace sal_art_ref

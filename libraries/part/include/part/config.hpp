// part — persistent ART, layer 1 geometry. One constant rules everything:
// PART_SEGMENT_BYTES (4-16 MB intended); every size below derives from it.
// See SPEC.md §3-§4.
#pragma once
#include <cstddef>
#include <cstdint>

#include <sal/config.hpp>  // sal::sync_type — shared enum family with psitri/heart

#ifndef PART_SEGMENT_BYTES
#define PART_SEGMENT_BYTES (8u * 1024 * 1024)
#endif
#ifndef PART_DISK_PAGE_BYTES
#define PART_DISK_PAGE_BYTES (16u * 1024)
#endif

namespace part
{
   using sal::sync_type;

   inline constexpr uint32_t line_bytes      = 128;
   inline constexpr uint32_t segment_bytes   = PART_SEGMENT_BYTES;
   inline constexpr uint32_t disk_page_bytes = PART_DISK_PAGE_BYTES;

   inline constexpr uint32_t lines_per_segment = segment_bytes / line_bytes;
   inline constexpr uint32_t pages_per_segment = segment_bytes / disk_page_bytes;
   inline constexpr uint32_t lines_per_page    = disk_page_bytes / line_bytes;

   static_assert((segment_bytes & (segment_bytes - 1)) == 0, "segment size is a power of 2");
   static_assert((disk_page_bytes & (disk_page_bytes - 1)) == 0, "disk page is a power of 2");
   static_assert(segment_bytes % disk_page_bytes == 0 && disk_page_bytes % line_bytes == 0);
   static_assert(lines_per_segment >= 64 && lines_per_segment % 64 == 0);

   // Global line index: address = base + idx * 128. The top 2^16 values are
   // reserved sentinels (never minted by the allocator): every ptr-shaped field
   // is an index iff it is below the band — one unsigned compare (SPEC §4).
   using line_idx                            = uint32_t;
   inline constexpr line_idx sentinel_base   = 0xFFFF0000u;
   inline constexpr line_idx null_idx        = 0xFFFFFFFFu;
   constexpr bool            is_sentinel(line_idx i) noexcept { return i >= sentinel_base; }

   // An allocation is 1..64 consecutive lines that never cross a u64 word of the
   // level-0 freeset (an 8 KB-aligned region) — every alloc/free is one CAS.
   inline constexpr uint32_t max_alloc_lines = 64;

   // Refcount bands (SPEC §9): u8 per line; 255 is the overflow THROW sentinel
   // (never a stored count), readers get the top band, writers clone at the rest.
   inline constexpr uint32_t max_read_sessions = 64;  // one u64 slot bitmap
   inline constexpr uint8_t  ref_ceiling       = 254;
   inline constexpr uint8_t  writer_clone_at   = ref_ceiling - max_read_sessions;  // 190

   static_assert(max_read_sessions <= 64, "read-session slots are one u64 bitmap");
}  // namespace part

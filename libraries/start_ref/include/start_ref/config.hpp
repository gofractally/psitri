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

namespace start_ref
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

   // ── grain geometry (large-value second arena) ──────────────────────────
   // The arena/freeset are grain-AGNOSTIC: the freeset allocates runs of UNITS
   // (bits), <= 64 per u64 word; the byte-size-per-unit lives only here. A
   // `geometry` instantiation freezes one grain. The DEFAULT (`line_geometry`)
   // reproduces today's exact constants verbatim — the existing line arena is
   // byte-identical. A second instance at a 32 KiB grain reuses the SAME freeset
   // and gets 1..64 grains/word = 2 MiB max allocation, with no allocator changes.
   template <uint32_t GrainBytes,
             uint32_t SegBytes  = segment_bytes,
             uint32_t PageBytes = disk_page_bytes>
   struct geometry
   {
      static constexpr uint32_t grain_bytes        = GrainBytes;
      static constexpr uint32_t seg_bytes          = SegBytes;
      static constexpr uint32_t page_bytes         = PageBytes;
      static constexpr uint32_t units_per_segment  = SegBytes / GrainBytes;
      static constexpr uint32_t units_per_page     = PageBytes / GrainBytes;
      static constexpr uint32_t pages_per_segment_ = SegBytes / PageBytes;

      static_assert((SegBytes & (SegBytes - 1)) == 0, "segment size is a power of 2");
      static_assert((PageBytes & (PageBytes - 1)) == 0, "disk page is a power of 2");
      static_assert(SegBytes % PageBytes == 0 && PageBytes % GrainBytes == 0);
      static_assert(units_per_segment >= 64 && units_per_segment % 64 == 0);
   };

   // The default line grain (128 B): every constant matches the globals above.
   using line_geometry = geometry<line_bytes>;
   static_assert(line_geometry::grain_bytes == line_bytes);
   static_assert(line_geometry::units_per_segment == lines_per_segment);
   static_assert(line_geometry::units_per_page == lines_per_page);
   static_assert(line_geometry::pages_per_segment_ == pages_per_segment);

   // The large-value grain: 32 KiB units. Max allocation = max_alloc_lines (64)
   // x 32768 = 2 MiB (the single-CAS cap — a run stays in one freeset u64 word).
   // The byte API chunks larger logical values across these fixed-address runs.
   // The segment keeps units_per_segment == lines_per_segment (the default
   // geometry's): the
   // per-segment metadata (seg_freeset, the 1 B/unit refcount block) is then
   // BYTE-IDENTICAL in size to the line arena's — critically, a whole-page
   // multiple, so the contiguous-VA block_allocator's MAP_FIXED grows succeed on
   // 16 KB-page hosts (a sub-page metadata block size makes mmap(MAP_FIXED) fail
   // with EINVAL). 32 KB x 65536 = 2 GB segments (PROT_NONE-reserved VA).
   inline constexpr uint32_t large_grain_bytes   = 32u * 1024;
   inline constexpr uint32_t large_segment_bytes = large_grain_bytes * lines_per_segment;
   // The large arena's disk page == its grain (a 32 KB grain exceeds the 16 KB
   // line-arena disk page, so it cannot reuse it: PageBytes % GrainBytes must be
   // 0). One grain per page keeps msync granularity == the allocation unit, which
   // is exactly right for whole-object large values.
   inline constexpr uint32_t large_page_bytes = large_grain_bytes;
   using large_geometry = geometry<large_grain_bytes, large_segment_bytes, large_page_bytes>;
   inline constexpr uint32_t large_max_object_bytes = max_alloc_lines * large_grain_bytes;
   static_assert(large_geometry::units_per_segment == lines_per_segment,
                 "large per-segment metadata matches the line arena's (page-aligned)");

   // Refcount bands (SPEC §9): u8 per line; 255 is the overflow THROW sentinel
   // (never a stored count), readers get the top band, writers clone at the rest.
   inline constexpr uint32_t max_read_sessions = 64;  // one u64 slot bitmap
   inline constexpr uint8_t  ref_ceiling       = 254;
   inline constexpr uint8_t  writer_clone_at   = ref_ceiling - max_read_sessions;  // 190

   static_assert(max_read_sessions <= 64, "read-session slots are one u64 bitmap");
}  // namespace start_ref

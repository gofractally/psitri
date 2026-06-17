// sal2::config — geometry for the SAL 2.0 FIX-region line allocator.
//
// One knob rules the grain (design §4.1): SAL2_GRAIN_BYTES is the allocation
// quantum (the "line"). Everything else derives from it + the 32 MB segment.
// v1 builds the FIX region only (direct line index, no control blocks); the
// movable + BIG regions are deferred (design §6.2/§6.3).
#pragma once
#include <cstdint>

#ifndef SAL2_GRAIN_BYTES
#define SAL2_GRAIN_BYTES 128u  // default 128 B; design targets a 32 B knob later
#endif
#ifndef SAL2_SEGMENT_BYTES
#define SAL2_SEGMENT_BYTES (8u * 1024 * 1024)  // 8 MB segments (file-grow unit)
#endif
#ifndef SAL2_DISK_PAGE_BYTES
#define SAL2_DISK_PAGE_BYTES (16u * 1024)
#endif

namespace sal2
{
   inline constexpr uint32_t line_bytes      = SAL2_GRAIN_BYTES;   // the grain
   inline constexpr uint32_t segment_bytes   = SAL2_SEGMENT_BYTES;
   inline constexpr uint32_t disk_page_bytes = SAL2_DISK_PAGE_BYTES;

   inline constexpr uint32_t lines_per_segment = segment_bytes / line_bytes;
   inline constexpr uint32_t pages_per_segment = segment_bytes / disk_page_bytes;
   inline constexpr uint32_t lines_per_page    = disk_page_bytes / line_bytes;

   static_assert((line_bytes & (line_bytes - 1)) == 0, "grain is a power of 2");
   static_assert((segment_bytes & (segment_bytes - 1)) == 0, "segment is a power of 2");
   static_assert(segment_bytes % disk_page_bytes == 0 && disk_page_bytes % line_bytes == 0);
   static_assert(lines_per_segment >= 64 && lines_per_segment % 64 == 0,
                 "freeset l0 is whole u64 words");

   inline constexpr uint32_t line_shift = (line_bytes == 128)  ? 7
                                          : (line_bytes == 64)  ? 6
                                          : (line_bytes == 32)  ? 5
                                                                : 0;
   static_assert(line_shift != 0 || line_bytes == 1, "supported grains: 32/64/128");
   static_assert((1u << line_shift) == line_bytes);

   // Global line index: address = base + (idx << line_shift). The top 2^16
   // values are reserved sentinels (never minted) — a null/leaf-absent marker is
   // recognized by one unsigned compare (design §7 / start_ref §4).
   using line_idx                          = uint32_t;
   inline constexpr line_idx sentinel_base = 0xFFFF0000u;
   inline constexpr line_idx null_idx      = 0xFFFFFFFFu;
   constexpr bool is_sentinel(line_idx i) noexcept { return i >= sentinel_base; }

   // An allocation is 1..64 consecutive lines confined to one freeset u64 word
   // (one CAS per alloc/free). A FIX node never exceeds this (cl9 = 9 lines).
   inline constexpr uint32_t max_alloc_lines = 64;

   // Per-line u8 refcount reclamation (v1; version-GC is the design's later
   // phase). 255 is the overflow sentinel (never stored); the top band is reader
   // pins, the rest is writer clones (start_ref's proven banding, design §9).
   inline constexpr uint32_t max_read_sessions = 64;
   inline constexpr uint8_t  ref_ceiling       = 254;
   inline constexpr uint8_t  writer_clone_at   = uint8_t(ref_ceiling - max_read_sessions);  // 190
   // release_one() drops the LAST ref to this DYING sentinel (not 0): the slot is
   // being torn down (its content still read by the releaser to recurse children)
   // and must NOT be reusable yet. free() then sets it to 0 = reusable. The
   // write-around alloc reuses ONLY ref==0, so it never grabs a dying slot mid-
   // teardown (the reader-vs-writer reclaim race) — and the release path needs no
   // gather-before-release. 255 ≥ ref_ceiling so try_retain/retain already reject it.
   inline constexpr uint8_t  dying_ref         = 255;
}  // namespace sal2

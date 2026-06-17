// gcart::branch_type + 3-bit packed tag helpers — the FIX-region per-branch
// dispatch selector for the spec node taxonomy. Generic 3-bit packing, shared
// (the spec engine + any node format that stores per-branch kinds); it carries
// no engine state, so it lives in `gcart` proper rather than inside any one
// node header.
#pragma once
#include <cstdint>
#include <cstring>

namespace gcart
{
   // ── §7 per-branch type tag (3 bits, 8 kinds) ──────────────────────────────
   // The dispatch selector stored per branch in an inner node: it says WHAT a
   // branch points to / holds, so the descent needs no child header read and a
   // terminal value can live INLINE (no separate leaf node). Region per kind:
   //   full/setlist/nested_setlist/double_setlist/big_nested → FIX,
   //   suffix_value → inline-in-node (MOVABLE in the full design),
   //   big_suffix_value → BIG, null → empty.
   enum class branch_type : uint8_t
   {
      full = 0,           // child is a node256 (dense 256-way)
      setlist,            // child is a separate setlist node (a pointer)
      nested_setlist,     // child setlist inlined in this node's tail-heap
      double_setlist,     // a cl2 (256B) setlist (a pointer)
      big_nested,         // a large nested region
      suffix_value,       // TERMINAL (suffix,value) inlined in this node's tail-heap
      big_suffix_value,   // terminal whose value spilled to the BIG/chain store
      null                // empty
   };

   // 3-bit packed tag array (⌈3n/8⌉ bytes). Read = one unaligned u16 load, shift,
   // mask — branchless, never straddles two reads (a 3-bit field starting at
   // intra-byte offset 0..7 ends by bit ≤10). The ≤1-byte tail over-read is
   // absorbed by the following in-node array (the offsets) or a pad byte — same
   // convention as the padded SWAR scans. (Validated: 16→6B, 256→96B, RMW-safe.)
   inline uint32_t tag3_bytes(uint32_t n) noexcept { return (3u * n + 7u) / 8u; }
   inline branch_type tag3_get(const uint8_t* packed, uint32_t i) noexcept
   {
      uint16_t w;
      std::memcpy(&w, packed + ((3u * i) >> 3), 2);
      return branch_type((w >> ((3u * i) & 7u)) & 0x7u);
   }
   inline void tag3_set(uint8_t* packed, uint32_t i, branch_type v) noexcept
   {
      const uint32_t bit = 3u * i, byte = bit >> 3, sh = bit & 7u;
      uint16_t       w;
      std::memcpy(&w, packed + byte, 2);
      w = uint16_t((w & ~(uint16_t(0x7) << sh)) | (uint16_t(uint8_t(v) & 0x7) << sh));
      std::memcpy(packed + byte, &w, 2);
   }
}  // namespace gcart

// gcart::page — the §7 tagged page node (the density model).
//
// THE unifying insight (Dan): there is no leaf node type — a leaf is a
// `suffix_value` with a 0-length suffix. So every terminal is one inline format
// stored in the node's tail-heap; only ROUTING is a separate node (setlist /
// node256-full) and only LARGE values are a pointer (big_suffix_value → chain).
// This collapses the v1 "one 128 B leaf node per key + one setlist per binary
// divergence" (the 2× bloat vs sal_art_ref) into inline entries packed per node.
//
// Node layout (one tagged page; node256-`full` is the dense variant, later):
//   [page_hdr] keys[n]:u8  types[n]:3-bit  slots[n]:u32  prefix[plen] …free… heap↓
//     types[i] = branch_type:  setlist/full/double_setlist/big_nested → pointer (slots[i]=nptr)
//                              suffix_value                           → inline   (slots[i]=heap offset)
//                              big_suffix_value                       → pointer  (slots[i]=chain nptr) + inline suffix
//                              nested_setlist                         → inline   (slots[i]=heap offset to a sub-page)
//   term (key ends here, 0-len suffix) = a suffix_value referenced by hdr.term_off.
//   tail-heap grows DOWN from the node end; suffix_value entry = [u16 vlen][u8 slen][value][suffix].
//
// Refcount COW liveness (v1, kept by Dan): pointer children are refcounted;
// inline entries are bytes (no refcount). Mutation is copy-construct-insert into
// a right-sized node (stable source husk → new node, child pointers transferred),
// the required style.
#pragma once
#include <gcart/node.hpp>   // branch_type, tag3_*, nptr, ntype, value_chunk, LINE

#include <cstring>
#include <string_view>

namespace gcart::pg
{
   using sv = std::string_view;

   // ── page header (16 B) ────────────────────────────────────────────────────
   struct page_hdr
   {
      uint8_t  type;       // ntype::setlist (sparse page) — node256 is the dense variant
      uint8_t  nlines;     // allocation size in lines
      uint16_t n;          // branch count
      uint16_t plen;       // prefix length
      uint16_t heap;       // bytes used by the tail-heap (from the node end)
      uint16_t cap;        // branch capacity (slots provisioned)
      uint16_t term_off;   // tail-heap offset of the 0-len-suffix term, or 0 = none
      uint32_t _rsv;
   };
   static_assert(sizeof(page_hdr) == 16);

   // an inline terminal: value then suffix, length-prefixed.
   //   [u16 vlen][u8 slen][value bytes][suffix bytes]
   struct sv_entry
   {
      static uint32_t bytes(uint32_t vlen, uint32_t slen) noexcept { return 3 + vlen + slen; }
   };

   struct page
   {
      page_hdr h;

      static constexpr uint32_t hdr      = sizeof(page_hdr);
      static constexpr uint32_t max_branches = 256;

      // ── geometry (cap branches, plen prefix) ───────────────────────────────
      static uint32_t keys_off() noexcept { return hdr; }
      static uint32_t types_off(uint32_t cap) noexcept { return hdr + cap; }
      static uint32_t slots_off(uint32_t cap) noexcept
      {
         return (hdr + cap + tag3_bytes(cap) + 3u) & ~3u;  // 4-byte align slots
      }
      static uint32_t prefix_off(uint32_t cap) noexcept
      {
         return slots_off(cap) + cap * uint32_t(sizeof(nptr));
      }
      // bytes for the fixed part (header..prefix), before the tail-heap.
      static uint32_t fixed_bytes(uint32_t cap, uint32_t plen) noexcept
      {
         return prefix_off(cap) + plen;
      }

      uint8_t*       keys() noexcept { return reinterpret_cast<uint8_t*>(this) + keys_off(); }
      const uint8_t* keys() const noexcept { return reinterpret_cast<const uint8_t*>(this) + keys_off(); }
      uint8_t*       types() noexcept { return reinterpret_cast<uint8_t*>(this) + types_off(h.cap); }
      const uint8_t* types() const noexcept { return reinterpret_cast<const uint8_t*>(this) + types_off(h.cap); }
      nptr*          slots() noexcept { return reinterpret_cast<nptr*>(reinterpret_cast<char*>(this) + slots_off(h.cap)); }
      const nptr*    slots() const noexcept { return reinterpret_cast<const nptr*>(reinterpret_cast<const char*>(this) + slots_off(h.cap)); }
      char*          prefix_ptr() noexcept { return reinterpret_cast<char*>(this) + prefix_off(h.cap); }
      const char*    prefix_ptr() const noexcept { return reinterpret_cast<const char*>(this) + prefix_off(h.cap); }
      sv             prefix() const noexcept { return {prefix_ptr(), h.plen}; }
      uint32_t       line_count() const noexcept { return h.nlines; }

      // tail-heap: byte offsets are measured from the node start; the heap
      // occupies [node_end - heap, node_end).
      uint32_t node_bytes() const noexcept { return h.nlines * LINE; }
      char*    at(uint32_t off) noexcept { return reinterpret_cast<char*>(this) + off; }
      const char* at(uint32_t off) const noexcept { return reinterpret_cast<const char*>(this) + off; }

      branch_type btype(uint32_t i) const noexcept { return tag3_get(types(), i); }

      // decode an inline suffix_value entry at heap offset `off`.
      sv entry_value(uint32_t off) const noexcept
      {
         uint16_t vlen; std::memcpy(&vlen, at(off), 2);
         return {at(off) + 3, vlen};
      }
      sv entry_suffix(uint32_t off) const noexcept
      {
         uint16_t vlen; std::memcpy(&vlen, at(off), 2);
         const uint8_t slen = uint8_t(*at(off + 2));
         return {at(off) + 3 + vlen, slen};
      }

      // sorted lower_bound on the (zero-padded) key array — branchless SWAR.
      int find_index(uint8_t byte) const noexcept
      {
         int pos = int(ucc::lower_bound_padded(keys(), h.n, byte));
         return (pos < int(h.n) && keys()[pos] == byte) ? pos : -1;
      }
   };
}  // namespace gcart::pg

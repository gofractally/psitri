#pragma once
#include <cstdint>
#include <cstring>
#include <span>

#include <heart/radix_bitset.hpp>
#include <heart/radix_full.hpp>
#include <heart/radix_node.hpp>
#include <heart/radix_type_ids.hpp>
#include <heart/smart_address.hpp>

namespace heart
{
   // The concrete node taxonomy (type-only; layouts byte-pinned by static_assert,
   // no algorithm bodies). Maps to ART's N4/N16/N48/N256 dispatch classes:
   //
   //   encoding         N        total CL   dispatch_shape   refcount
   //   suffix_value     terminal     1       r1              none
   //   prefix_next      1            1       r1              none
   //   setlist_tiny    <4            1       r1 (linear)     none
   //   setlist<T>      <=16          1       r1 (SIMD)       none
   //   bitset1         ~16..50       1       r1              large_radix K=1
   //   bitset_r2       ~45..179      3       r2 (b>>7)       large_radix K=2
   //   bitset_r4       ~96..256      5       r4 (b>>6)       large_radix K=4
   //   full_radix      256           9       r8 (b>>5)       large_radix K=4
   //
   // The CL0 "variable region" (radix_impl::prefix_data) holds the inline prefix,
   // and for the in-CL0 encodings (setlist*, bitset1, suffix_value) the branch
   // payload as well; the multi-cacheline encodings put branches in trailing CLs.

   // ---- fixed-field structs (separate complete types for the CRTP param) ----
   struct empty_fixed
   {
   };
   struct prefix_next_fixed
   {
      smart_address next;  // the single child (node ptr or inline tail)
   };
   struct setlist_tiny_fixed
   {
      uint8_t nstems;
   };
   struct setlist_fixed
   {
      uint16_t nstems;
   };

   // ---- R1, sparse (no refcount tree) ----

   /// A terminal: inline key suffix + value live in the CL0 variable region.
   struct suffix_value_node : radix_impl<suffix_value_node, empty_fixed>
   {
      static constexpr dispatch_shape tier = dispatch_shape::r1;
   };

   /// Path compression: shared prefix + a single `next` child.
   struct prefix_next_node : radix_impl<prefix_next_node, prefix_next_fixed>
   {
      static constexpr dispatch_shape tier = dispatch_shape::r1;
   };

   /// ART Node4 analogue: < 4 stems, linear (unrolled) dispatch.
   struct setlist_tiny_node : radix_impl<setlist_tiny_node, setlist_tiny_fixed>
   {
      static constexpr dispatch_shape tier = dispatch_shape::r1;
   };

   /// ART Node16 analogue (also the multi-byte stem variant): up to ~16 stems,
   /// SIMD dispatch. StemType u8/u16/u32 consumes 1/2/4 key bytes per node.
   template <typename StemType>
   struct setlist_node : radix_impl<setlist_node<StemType>, setlist_fixed>
   {
      static constexpr dispatch_shape tier       = dispatch_shape::r1;
      static constexpr int            stem_width = sizeof(StemType);
   };

   // ---- bitset family (large_radix; group-counter COW) ----

   /// 1 cacheline: present<256> + compact branches share the CL0 variable region.
   struct bitset1_node : large_radix<bitset1_node, 1>
   {
      static constexpr dispatch_shape tier = dispatch_shape::r1;
   };

   /// 3 cachelines: CL0 header, then 2 sub-range cachelines [0,128)/[128,256),
   /// each a self-contained radix_bitset<128> reached directly by `byte >> 7`.
   struct bitset_r2_node : large_radix<bitset_r2_node, 2>
   {
      static constexpr dispatch_shape tier = dispatch_shape::r2;
      radix_bitset<128, 128> half[2];
   };

   /// 5 cachelines: CL0 header, then 4 sub-range cachelines (64 each), reached by
   /// `byte >> 6`.
   struct bitset_r4_node : large_radix<bitset_r4_node, 4>
   {
      static constexpr dispatch_shape tier = dispatch_shape::r4;
      radix_bitset<64, 128> quad[4];
   };

   /// 9 cachelines: CL0 header, then radix_full<256> direct index in CL1–CL8,
   /// reached by `byte >> 5`. The dense R8 fallback.
   struct full_radix_node : large_radix<full_radix_node, 4>
   {
      static constexpr dispatch_shape   tier    = dispatch_shape::r8;
      static constexpr sal::header_type type_id = radix_type_ids::full_radix;

      radix_full<256> branches;  // @128 (CL1..8): 32 ptr_addresses per cacheline

      full_radix_node() noexcept = default;  // for placement layout tests

      // Offset of the segment region (CL1) — the skip-header descent reads
      // branches[byte] at base + branch_offset + 4*byte without touching CL0.
      // Pinned to 128 by the offsetof static_assert below the struct.
      static constexpr uint32_t branch_offset = 128;

      // ---- SAL construction contract (branches/term are pure pointers; no tail
      // heap, so the allocation is exactly the fixed 9-cacheline struct). ----
      static uint32_t alloc_size(std::span<const uint8_t> prefix) noexcept
      {
         (void)prefix;  // prefix lives in the CL0 variable region (prefix_data)
         return uint32_t(sizeof(full_radix_node));  // 1152, already a cacheline multiple
      }
      full_radix_node(uint32_t asize, sal::ptr_address_seq seq,
                      std::span<const uint8_t> prefix) noexcept
      {
         init_page(asize, type_id, seq);
         _rtype         = radix_node_type::full256;
         _term_value    = smart_address::null();
         _prefix_length = 0;
         set_prefix(prefix);
         branches.init();
      }

      smart_address find(uint8_t b) const noexcept { return branches.find(b); }
      bool          set_branch(uint8_t b, smart_address sa) noexcept
      {
         branches.set_branch(b, sa);  // direct index — always has room
         return true;
      }
      bool branch_has_room(uint8_t) const noexcept { return true; }

      std::span<const sal::ptr_address> branch_hint() const noexcept
      {
         return {reinterpret_cast<const sal::ptr_address*>(&branches.branches[0]), 256};
      }

      template <typename F>
      void for_each_child(F&& f) const
      {
         if (_term_value.is_pointer())
            f(_term_value.as_pointer());
         for (int b = 0; b < 256; ++b)
            if (branches.branches[b].is_pointer())
               f(branches.branches[b].as_pointer());
      }
      template <typename Session>
      void destroy(const Session& s) const
      {
         for_each_child([&](sal::ptr_address c) { s->release(c); });
      }
   };

   static_assert(sizeof(suffix_value_node) == 128);
   static_assert(sizeof(prefix_next_node) == 128);
   static_assert(sizeof(setlist_tiny_node) == 128);
   static_assert(sizeof(setlist_node<uint8_t>) == 128);
   static_assert(sizeof(setlist_node<uint32_t>) == 128);
   static_assert(sizeof(bitset1_node) == 128, "1 cacheline");
   static_assert(sizeof(bitset_r2_node) == 384, "3 cachelines");
   static_assert(sizeof(bitset_r4_node) == 640, "5 cachelines");
   static_assert(sizeof(full_radix_node) == 1152, "9 cachelines");
}  // namespace heart

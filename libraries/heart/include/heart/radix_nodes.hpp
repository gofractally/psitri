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
   ///
   /// CL0 variable region (radix_impl::prefix_data, `var_capacity` bytes) packs the
   /// prefix, the stems, and the branches. To keep prefix changes (prefix_split)
   /// cheap, branches are anchored to the END of the region (branch i at
   /// var_capacity - (i+1)*4) so they NEVER move; only the prefix (front) and the
   /// stems (just after the prefix) shift. A null branch slot marks an absent stem.
   template <typename StemType>
   struct setlist_node : radix_impl<setlist_node<StemType>, setlist_fixed>
   {
      using base = radix_impl<setlist_node<StemType>, setlist_fixed>;
      static constexpr dispatch_shape tier         = dispatch_shape::r1;
      static constexpr int            stem_width   = sizeof(StemType);
      static constexpr int            var_capacity = base::prefix_capacity;
      static constexpr sal::header_type type_id =
          stem_width == 1 ? radix_type_ids::setlist_u8
                          : (stem_width == 2 ? radix_type_ids::setlist_u16
                                             : radix_type_ids::setlist_u32);
      static constexpr radix_node_type rtype_val =
          stem_width == 1 ? radix_node_type::setlist_u8
                          : (stem_width == 2 ? radix_node_type::setlist_u16
                                             : radix_node_type::setlist_u32);

      setlist_node() noexcept = default;

      int  nstems() const noexcept { return this->fixed.nstems; }
      void set_nstems(int n) noexcept { this->fixed.nstems = uint16_t(n); }

      StemType stem_at(int i) const noexcept
      {
         StemType s;
         std::memcpy(&s, this->prefix_data + this->_prefix_length + i * stem_width, stem_width);
         return s;
      }
      void set_stem(int i, StemType s) noexcept
      {
         std::memcpy(this->prefix_data + this->_prefix_length + i * stem_width, &s, stem_width);
      }
      smart_address branch_at(int i) const noexcept
      {
         smart_address sa;
         std::memcpy(&sa, this->prefix_data + var_capacity - (i + 1) * 4, 4);
         return sa;
      }
      void set_branch_slot(int i, smart_address sa) noexcept
      {
         std::memcpy(this->prefix_data + var_capacity - (i + 1) * 4, &sa, 4);
      }

      bool room_for_one() const noexcept
      {
         return int(this->_prefix_length) + (nstems() + 1) * (stem_width + 4) <= var_capacity;
      }
      smart_address find(StemType s) const noexcept
      {
         const int n = nstems();
         for (int i = 0; i < n; ++i)
            if (stem_at(i) == s)
               return branch_at(i);
         return smart_address::null();
      }
      bool branch_has_room(StemType s) const noexcept
      {
         return !find(s).is_null() || room_for_one();
      }
      // Uniform tier interface (setlist room is count-based, address-independent).
      bool can_set(uint8_t b, smart_address) const noexcept { return branch_has_room(StemType(b)); }
      // Alloc-hint: branch slots are an end-anchored contiguous block, so a new
      // child can be placed near the siblings (clusters cachelines for dedup).
      std::span<const sal::ptr_address> branch_hint(uint8_t = 0) const noexcept
      {
         const int n = nstems();
         return {reinterpret_cast<const sal::ptr_address*>(this->prefix_data + var_capacity - n * 4),
                 size_t(n)};
      }
      // Update-or-append; false (no change) if a new stem won't fit.
      bool set_branch(StemType s, smart_address sa) noexcept
      {
         const int n = nstems();
         for (int i = 0; i < n; ++i)
            if (stem_at(i) == s)
            {
               set_branch_slot(i, sa);
               return true;
            }
         if (!room_for_one())
            return false;
         set_stem(n, s);          // stems grow forward (after the prefix)
         set_branch_slot(n, sa);  // branches grow backward (from the region end)
         set_nstems(n + 1);
         return true;
      }

      // Replace the prefix, preserving stems+branches. Branches are end-anchored so
      // they stay put; only the stems (which follow the prefix) are restowed.
      void reprefix(std::span<const uint8_t> p) noexcept
      {
         const int n = nstems();
         StemType  tmp[64];
         for (int i = 0; i < n; ++i)
            tmp[i] = stem_at(i);
         std::memcpy(this->prefix_data, p.data(), p.size());
         this->_prefix_length = uint16_t(p.size());
         for (int i = 0; i < n; ++i)
            set_stem(i, tmp[i]);
      }

      template <typename F>
      void for_each_child(F&& f) const
      {
         if (this->_term_value.is_pointer())
            f(this->_term_value.as_pointer());
         const int n = nstems();
         for (int i = 0; i < n; ++i)
            if (branch_at(i).is_pointer())
               f(branch_at(i).as_pointer());
      }
      // (stem, branch) pairs — used by upgrade to repopulate a wider tier. Only
      // meaningful for the single-byte (u8) stem variant.
      template <typename F>
      void for_each_entry(F&& f) const
      {
         const int n = nstems();
         for (int i = 0; i < n; ++i)
            f(uint8_t(stem_at(i)), branch_at(i));
      }
      template <typename Session>
      void destroy(const Session& s) const
      {
         for_each_child([&](sal::ptr_address c) { s->release(c); });
      }

      // ---- SAL construction contract (single fixed cacheline) ----
      static uint32_t alloc_size(std::span<const uint8_t> prefix) noexcept
      {
         (void)prefix;  // prefix lives in the CL0 variable region
         return uint32_t(sizeof(setlist_node));  // 128
      }
      setlist_node(uint32_t asize, sal::ptr_address_seq seq, std::span<const uint8_t> prefix) noexcept
      {
         this->init_page(asize, type_id, seq);
         this->_rtype         = rtype_val;
         this->_term_value    = smart_address::null();
         this->_prefix_length = 0;
         set_nstems(0);
         this->set_prefix(prefix);
      }
   };

   // ---- bitset family (large_radix; group-counter COW) ----

   /// 1 cacheline: present<256> + compact branches share the CL0 variable region.
   struct bitset1_node : large_radix<bitset1_node, 1>
   {
      static constexpr dispatch_shape tier = dispatch_shape::r1;
   };

   /// 3 cachelines: CL0 header, then 2 sub-range cachelines [0,128)/[128,256),
   /// each a self-contained radix_bitset<128> reached directly by `byte >> 7`.
   /// Skip-header descent reads segment (base + 128 + (byte>>7)*128) and never CL0.
   struct bitset_r2_node : large_radix<bitset_r2_node, 2>
   {
      static constexpr dispatch_shape   tier          = dispatch_shape::r2;
      static constexpr sal::header_type type_id       = radix_type_ids::bitset_r2;
      static constexpr uint32_t         branch_offset = 128;  // segment cachelines at CL1
      static constexpr int              nseg          = 2;
      static constexpr int              seg_shift     = 7;
      static constexpr int              seg_mask      = 127;
      using segment                                   = radix_bitset<128, 128>;

      segment half[2];  // @128 (CL1), @256 (CL2)

      bitset_r2_node() noexcept = default;

      smart_address find(uint8_t b) const noexcept { return half[b >> 7].find(b & 127); }
      bool          can_set(uint8_t b, smart_address sa) const noexcept
      {
         const segment& h = half[b >> 7];
         return h.contains(b & 127) || h.can_insert(sa);
      }
      bool set_branch(uint8_t b, smart_address sa) noexcept
      {
         return half[b >> 7].set_branch(b & 127, sa);
      }
      // Alloc-hint: the cacheline bases of the segment this byte routes to, so a new
      // child clusters with its segment-mates (keeps compact_addresses bases <= 16).
      std::span<const sal::ptr_address> branch_hint(uint8_t b) const noexcept
      {
         return half[b >> 7].get_hint();
      }
      template <typename F>
      void for_each_entry(F&& f) const
      {
         for (int s = 0; s < 2; ++s)
            for (int lb = half[s].present.first_set(); lb >= 0; lb = half[s].present.next_set(lb + 1))
               f(uint8_t(s * 128 + lb), half[s].find(lb));
      }
      template <typename F>
      void for_each_child(F&& f) const
      {
         if (this->_term_value.is_pointer())
            f(this->_term_value.as_pointer());
         for_each_entry([&](uint8_t, smart_address br) {
            if (br.is_pointer())
               f(br.as_pointer());
         });
      }
      template <typename Session>
      void destroy(const Session& s) const
      {
         for_each_child([&](sal::ptr_address c) { s->release(c); });
      }

      static uint32_t alloc_size(std::span<const uint8_t> prefix) noexcept
      {
         (void)prefix;
         return uint32_t(sizeof(bitset_r2_node));  // 384
      }
      bitset_r2_node(uint32_t asize, sal::ptr_address_seq seq, std::span<const uint8_t> prefix) noexcept
      {
         this->init_page(asize, type_id, seq);
         this->_rtype         = radix_node_type::bitset128;
         this->_term_value    = smart_address::null();
         this->_prefix_length = 0;
         half[0].init();
         half[1].init();
         this->set_prefix(prefix);
      }
   };

   /// 5 cachelines: CL0 header, then 4 sub-range cachelines (64 each), reached by
   /// `byte >> 6`.
   struct bitset_r4_node : large_radix<bitset_r4_node, 4>
   {
      static constexpr dispatch_shape   tier          = dispatch_shape::r4;
      static constexpr sal::header_type type_id       = radix_type_ids::bitset_r4;
      static constexpr uint32_t         branch_offset = 128;
      static constexpr int              nseg          = 4;
      static constexpr int              seg_shift     = 6;
      static constexpr int              seg_mask      = 63;
      using segment                                   = radix_bitset<64, 128>;

      segment quad[4];  // @128, @256, @384, @512

      bitset_r4_node() noexcept = default;

      smart_address find(uint8_t b) const noexcept { return quad[b >> 6].find(b & 63); }
      bool          can_set(uint8_t b, smart_address sa) const noexcept
      {
         const segment& q = quad[b >> 6];
         return q.contains(b & 63) || q.can_insert(sa);
      }
      bool set_branch(uint8_t b, smart_address sa) noexcept
      {
         return quad[b >> 6].set_branch(b & 63, sa);
      }
      std::span<const sal::ptr_address> branch_hint(uint8_t b) const noexcept
      {
         return quad[b >> 6].get_hint();
      }
      template <typename F>
      void for_each_entry(F&& f) const
      {
         for (int s = 0; s < 4; ++s)
            for (int lb = quad[s].present.first_set(); lb >= 0; lb = quad[s].present.next_set(lb + 1))
               f(uint8_t(s * 64 + lb), quad[s].find(lb));
      }
      template <typename F>
      void for_each_child(F&& f) const
      {
         if (this->_term_value.is_pointer())
            f(this->_term_value.as_pointer());
         for_each_entry([&](uint8_t, smart_address br) {
            if (br.is_pointer())
               f(br.as_pointer());
         });
      }
      template <typename Session>
      void destroy(const Session& s) const
      {
         for_each_child([&](sal::ptr_address c) { s->release(c); });
      }

      static uint32_t alloc_size(std::span<const uint8_t> prefix) noexcept
      {
         (void)prefix;
         return uint32_t(sizeof(bitset_r4_node));  // 640
      }
      bitset_r4_node(uint32_t asize, sal::ptr_address_seq seq, std::span<const uint8_t> prefix) noexcept
      {
         this->init_page(asize, type_id, seq);
         this->_rtype         = radix_node_type::bitset64;
         this->_term_value    = smart_address::null();
         this->_prefix_length = 0;
         for (auto& q : quad)
            q.init();
         this->set_prefix(prefix);
      }
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
      bool can_set(uint8_t, smart_address) const noexcept { return true; }

      std::span<const sal::ptr_address> branch_hint(uint8_t = 0) const noexcept
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

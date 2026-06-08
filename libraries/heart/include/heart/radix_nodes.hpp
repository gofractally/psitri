#pragma once
#include <cstdint>
#include <cstring>
#include <span>

#include <heart/radix_bitset.hpp>
#include <heart/radix_full.hpp>
#include <heart/radix_node.hpp>
#include <heart/radix_type_ids.hpp>
#include <heart/smart_address.hpp>
#include <ucc/round.hpp>

namespace heart
{
   // radix_node_headroom + the default node lifecycle (alloc_size/head_bytes/head_size/
   // free_heap/destroy) live on radix_impl in radix_node.hpp; nodes inherit them and only
   // setlist overrides head_size()/free_heap() for its wide (index-past-CL0) layout.
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
      using stem_type                              = StemType;
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

      // Key bytes consumed/compared per hop. u8->1, u16->2; the u32 cell consumes 4 or,
      // when re-strided to the 3-byte variant (rtype setlist_u24), 3 (low byte ignored).
      int consume_width() const noexcept
      {
         if constexpr (stem_width != 4)
            return stem_width;
         else
            return this->_rtype == radix_node_type::setlist_u24 ? 3 : 4;
      }

      // ---- index region (prefix + stems + branches) ----
      // u8 (stem_width==1): the fixed 1-CL block; branches END-anchored at var_capacity so
      // prefix edits never move them. wide (stem_width>1): forward parallel arrays — stems
      // then branches — that GROW across cachelines, bounded below by the tail heap, with
      // capacity derived from the allocation size (up to ~4 KB). All wide-only code is
      // if-constexpr'd so the u8 type is byte-identical to before.
      uint32_t index_cap() const noexcept  // bytes available for prefix+stems+branches
      {
         if constexpr (stem_width == 1)
            return uint32_t(var_capacity);
         else
            return uint32_t((this->tail() - this->tail_used()) -
                            reinterpret_cast<const char*>(this->prefix_data));
      }
      const uint8_t* branch_base() const noexcept  // wide: branches follow the stems
      {
         return this->prefix_data + this->_prefix_length + size_t(nstems()) * stem_width;
      }
      uint8_t* branch_base() noexcept
      {
         return this->prefix_data + this->_prefix_length + size_t(nstems()) * stem_width;
      }

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
         if constexpr (stem_width == 1)
            std::memcpy(&sa, this->prefix_data + var_capacity - (i + 1) * 4, 4);
         else
            std::memcpy(&sa, branch_base() + size_t(i) * 4, 4);
         return sa;
      }
      void set_branch_slot(int i, smart_address sa) noexcept
      {
         if constexpr (stem_width == 1)
            std::memcpy(this->prefix_data + var_capacity - (i + 1) * 4, &sa, 4);
         else
            std::memcpy(branch_base() + size_t(i) * 4, &sa, 4);
      }

      bool room_for_one() const noexcept
      {
         return int(this->_prefix_length) + (nstems() + 1) * (stem_width + 4) <= int(index_cap());
      }
      // Point lookup. u8 uses a branchless LINEAR scan — on the small fan-outs radix nodes
      // hold it beats binary search (no mispredicts), and dict is u8-dominated. u16/u32 use
      // the SIMD lower_bound (lower_bound_index) + an equality check. (Earlier this was a
      // unified lower_bound+eq, but that routed u8 through binary search and regressed dict
      // random by ~12%.)
      smart_address find(StemType s) const noexcept
      {
         const int n = nstems();
         if constexpr (stem_width == 1)
         {
            for (int i = 0; i < n; ++i)
               if (stem_at(i) == s)
                  return branch_at(i);
            return smart_address::null();
         }
         else
         {
            const int i = lower_bound_index(s);
            return (i < n && stem_at(i) == s) ? branch_at(i) : smart_address::null();
         }
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
         if constexpr (stem_width == 1)
            return {reinterpret_cast<const sal::ptr_address*>(this->prefix_data + var_capacity - n * 4),
                    size_t(n)};
         else
            return {reinterpret_cast<const sal::ptr_address*>(branch_base()), size_t(n)};
      }
      // Stems are kept in ascending key order. Cells are big-endian, left-justified, so
      // an unsigned integer compare on the cell is exact lexicographic key order (u24's
      // zero low byte is constant across stems, so it sorts on its 3 consumed bytes).
      // First index whose stem is >= s; nstems() if every stem is < s. Stems are a
      // contiguous cell array right after the prefix, so this is a SIMD count-of-less
      // (ucc::lower_bound_uN) for the wide widths — faster than binary search on the
      // small arrays radix nodes hold. u8 keeps the branchless binary search.
      int lower_bound_index(StemType s) const noexcept
      {
         const int      n = nstems();
         const uint8_t* p = this->prefix_data + this->_prefix_length;
         // _padded over-reads the final partial chunk (safe: the branch array follows the
         // stems in-node) to stay fully branchless — fastest for these small arrays.
         if constexpr (stem_width == 4)
            return ucc::lower_bound_u32_padded(reinterpret_cast<const uint32_t*>(p), n, uint32_t(s));
         else if constexpr (stem_width == 2)
            return ucc::lower_bound_u16_padded(reinterpret_cast<const uint16_t*>(p), n, uint16_t(s));
         else
         {
            int lo = 0, hi = n;
            while (lo < hi)
            {
               const int mid = (lo + hi) >> 1;
               if (stem_at(mid) < s)
                  lo = mid + 1;
               else
                  hi = mid;
            }
            return lo;
         }
      }
      // First index whose stem is > s; nstems() if every stem is <= s.
      int upper_bound_index(StemType s) const noexcept
      {
         const int      n = nstems();
         const uint8_t* p = this->prefix_data + this->_prefix_length;
         if constexpr (stem_width == 4)
            return ucc::upper_bound_u32_padded(reinterpret_cast<const uint32_t*>(p), n, uint32_t(s));
         else if constexpr (stem_width == 2)
            return ucc::upper_bound_u16_padded(reinterpret_cast<const uint16_t*>(p), n, uint16_t(s));
         else
         {
            int lo = 0, hi = n;
            while (lo < hi)
            {
               const int mid = (lo + hi) >> 1;
               if (stem_at(mid) <= s)
                  lo = mid + 1;
               else
                  hi = mid;
            }
            return lo;
         }
      }

      // Sorted update-or-insert; false (no change) if a new stem won't fit. Inserting at
      // the lower-bound position keeps the stems (and the rank-aligned branches) ordered.
      bool set_branch(StemType s, smart_address sa) noexcept
      {
         const int n = nstems();
         const int p = lower_bound_index(s);
         if (p < n && stem_at(p) == s)
         {
            set_branch_slot(p, sa);  // update in place
            return true;
         }
         if (!room_for_one())
            return false;
         if constexpr (stem_width == 1)
         {
            // stems forward (open a gap at p); branches end-anchored (shift [p,n) one slot
            // toward the head). The two regions never overlap (room_for_one guarantees it).
            uint8_t* st   = this->prefix_data + this->_prefix_length;
            uint8_t* base = this->prefix_data;
            std::memmove(st + p + 1, st + p, size_t(n - p));
            st[p] = uint8_t(s);
            std::memmove(base + var_capacity - (n + 1) * 4, base + var_capacity - n * 4,
                         size_t(n - p) * 4);
            std::memcpy(base + var_capacity - (p + 1) * 4, &sa, 4);
         }
         else
         {
            // forward parallel arrays: the stem array grows by one cell (branch base moves
            // right by stem_width) and both arrays open a gap at p. Move highest addresses
            // first (branches upper, branches lower, then stems upper) so nothing clobbers.
            uint8_t*  pe     = this->prefix_data + this->_prefix_length;
            const int sw     = stem_width;
            uint8_t*  old_bb = pe + size_t(n) * sw;      // current branch base
            uint8_t*  new_bb = pe + size_t(n + 1) * sw;  // base once the stem array grows
            std::memmove(new_bb + size_t(p + 1) * 4, old_bb + size_t(p) * 4, size_t(n - p) * 4);
            std::memmove(new_bb, old_bb, size_t(p) * 4);
            std::memmove(pe + size_t(p + 1) * sw, pe + size_t(p) * sw, size_t(n - p) * sw);
            std::memcpy(pe + size_t(p) * sw, &s, sw);
            std::memcpy(new_bb + size_t(p) * 4, &sa, 4);
         }
         set_nstems(n + 1);
         return true;
      }

      // Replace the prefix, preserving stems+branches.
      void reprefix(std::span<const uint8_t> p) noexcept
      {
         const int n = nstems();
         if constexpr (stem_width == 1)
         {
            // u8: branches are end-anchored (stay put); only the stems are restowed.
            StemType tmp[64];
            for (int i = 0; i < n; ++i)
               tmp[i] = stem_at(i);
            std::memcpy(this->prefix_data, p.data(), p.size());
            this->_prefix_length = uint16_t(p.size());
            for (int i = 0; i < n; ++i)
               set_stem(i, tmp[i]);
         }
         else
         {
            // wide: stems+branches are one contiguous forward block; shift it to the new
            // prefix end (no fixed buffer — n can be large).
            uint8_t* old_block = this->prefix_data + this->_prefix_length;
            std::memcpy(this->prefix_data, p.data(), p.size());
            this->_prefix_length = uint16_t(p.size());
            std::memmove(this->prefix_data + p.size(), old_block, size_t(n) * (stem_width + 4));
         }
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
      // head_size()/free_heap() OVERRIDE the radix_impl defaults: the wide variant grows
      // prefix+stems+branches past CL0, so the head (and thus the tail-heap start) is
      // variable. u8 keeps the sizeof() head. (alloc_size/head_bytes/destroy inherited.)
      uint32_t head_size() const noexcept
      {
         if constexpr (stem_width == 1)
            return uint32_t(sizeof(setlist_node));
         else
            return uint32_t((reinterpret_cast<const char*>(this->prefix_data) -
                             reinterpret_cast<const char*>(this)) +
                            this->_prefix_length + size_t(nstems()) * (stem_width + 4));
      }
      uint32_t free_heap() const noexcept
      {
         if constexpr (stem_width == 1)
            return this->free_bytes(reinterpret_cast<const char*>(this) + sizeof(setlist_node));
         else
            return this->free_bytes(reinterpret_cast<const char*>(this) + head_size());
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

   /// The segmented bitset tier, parameterized by `SegShift` (key bits per segment):
   ///   SegShift=7 → r2 (2 × radix_bitset<128>, byte>>7), 3 cachelines;
   ///   SegShift=6 → r4 (4 × radix_bitset<64>,  byte>>6), 5 cachelines.
   /// CL0 is the header (large_radix fills it to exactly 128 B), then `nseg` self-
   /// contained segment cachelines at CL1+. Skip-header descent reads segment
   /// (base + 128 + (byte>>SegShift)*128) and never touches CL0. nseg = 256>>SegShift
   /// (also the refcount-root count K), seg_width = 1<<SegShift, seg_mask = width-1.
   template <int SegShift, sal::header_type TypeId, dispatch_shape Tier, radix_node_type RType>
   struct bitset_rN_node : large_radix<bitset_rN_node<SegShift, TypeId, Tier, RType>, (256 >> SegShift)>
   {
      static constexpr dispatch_shape   tier          = Tier;
      static constexpr int              stem_width    = 1;  // byte-indexed (1 key byte / hop)
      static constexpr sal::header_type type_id       = TypeId;
      static constexpr uint32_t         branch_offset = 128;  // segment cachelines at CL1
      static constexpr int              nseg          = 256 >> SegShift;
      static constexpr int              seg_shift     = SegShift;
      static constexpr int              seg_width     = 1 << SegShift;
      static constexpr int              seg_mask      = seg_width - 1;
      using segment                                   = radix_bitset<seg_width, 128>;

      segment segs[nseg];  // @128, @256, ... one cacheline each

      bitset_rN_node() noexcept = default;

      const segment& seg(int s) const noexcept { return segs[s]; }  // uniform ordered-walk access
      smart_address  find(uint8_t b) const noexcept { return segs[b >> seg_shift].find(b & seg_mask); }
      bool           can_set(uint8_t b, smart_address sa) const noexcept
      {
         const segment& h = segs[b >> seg_shift];
         return h.contains(b & seg_mask) || h.can_insert(sa);
      }
      bool set_branch(uint8_t b, smart_address sa) noexcept
      {
         return segs[b >> seg_shift].set_branch(b & seg_mask, sa);
      }
      // Alloc-hint: the cacheline bases of the segment this byte routes to, so a new
      // child clusters with its segment-mates (keeps compact_addresses bases <= 16).
      std::span<const sal::ptr_address> branch_hint(uint8_t b) const noexcept
      {
         return segs[b >> seg_shift].get_hint();
      }
      template <typename F>
      void for_each_entry(F&& f) const
      {
         for (int s = 0; s < nseg; ++s)
            for (int lb = segs[s].present.first_set(); lb >= 0; lb = segs[s].present.next_set(lb + 1))
               f(uint8_t(s * seg_width + lb), segs[s].find(lb));
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
      // alloc_size / head_bytes / head_size / free_heap / destroy: inherited from radix_impl.
      bitset_rN_node(uint32_t asize, sal::ptr_address_seq seq, std::span<const uint8_t> prefix) noexcept
      {
         this->init_page(asize, type_id, seq);
         this->_rtype         = RType;
         this->_term_value    = smart_address::null();
         this->_prefix_length = 0;
         for (auto& s : segs)
            s.init();
         this->set_prefix(prefix);
      }
   };

   using bitset_r2_node =
       bitset_rN_node<7, radix_type_ids::bitset_r2, dispatch_shape::r2, radix_node_type::bitset128>;
   using bitset_r4_node =
       bitset_rN_node<6, radix_type_ids::bitset_r4, dispatch_shape::r4, radix_node_type::bitset64>;

   /// 9 cachelines: CL0 header, then radix_full<256> direct index in CL1–CL8,
   /// reached by `byte >> 5`. The dense R8 fallback.
   struct full_radix_node : large_radix<full_radix_node, 4>
   {
      static constexpr dispatch_shape   tier       = dispatch_shape::r8;
      static constexpr int              stem_width = 1;  // byte-indexed (1 key byte / hop)
      static constexpr sal::header_type type_id    = radix_type_ids::full_radix;

      radix_full<256> branches;  // @128 (CL1..8): 32 ptr_addresses per cacheline

      full_radix_node() noexcept = default;  // for placement layout tests

      // Offset of the segment region (CL1) — the skip-header descent reads
      // branches[byte] at base + branch_offset + 4*byte without touching CL0.
      // Pinned to 128 by the offsetof static_assert below the struct.
      static constexpr uint32_t branch_offset = 128;

      // alloc_size / head_bytes / head_size / free_heap / destroy: inherited from radix_impl
      // (branches/term are pure pointers; no tail heap, so the alloc is the fixed struct).
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
      void for_each_entry(F&& f) const
      {
         for (int b = 0; b < 256; ++b)
            if (!branches.branches[b].is_null())
               f(uint8_t(b), branches.branches[b]);
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
      // alloc_size / head_bytes / head_size / free_heap / destroy: inherited from radix_impl.
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

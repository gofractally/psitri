#pragma once
#include <cassert>
#include <cstdint>
#include <span>

#include <sal_art_ref/paged.hpp>
#include <sal_art_ref/smart_address.hpp>
#include <sal/colo_alloc.hpp>  // implicit_ref_addr / implicit_ref_region / combine (range counters)
#include <sal/config.hpp>      // cacheline_size
#include <ucc/round.hpp>       // round_up_multiple (alloc_size default)

namespace sal_art_ref
{
   // Spare tail-heap headroom every radix node is allocated beyond its fixed struct, so inline
   // terminal values have somewhere to grow before a COW-grow is forced. 128 (was 256): measured
   // avg setlist tail_used is ~51 B, so 256 over-reserved ~205 B/node on the DOMINANT tier.
   // Dropping to 128 shrinks setlist 384→256 (-33%) and bitset_r2 640→512 (-20%) with NO object
   // increase (nothing spilled to external leaves — values >60 B are external regardless) and
   // same-or-faster lookups (smaller nodes pack denser); dense nodes still COW-grow on demand.
   inline constexpr uint32_t radix_node_headroom = 128;

   /**
    * The R1 dispatch *format* — what CL0 holds once you've committed to reading the
    * header. (The control_block carries the cacheline-division *shape* below; this
    * is the concrete encoding.) Concrete values are assigned in the impl phase.
    */
   enum class radix_node_type : uint8_t
   {
      suffix_value,  // inline key suffix + terminal value
      prefix_next,   // shared prefix + a single next pointer
      setlist_u8,
      setlist_u16,
      setlist_u24,  // u32 cell, but consumes/compares only 3 key bytes (low byte ignored)
      setlist_u32,
      bitset256,
      bitset128,
      bitset64,
      full256,
   };

   /**
    * How the 256-byte branch range is divided across cachelines — stamped into 2 of
    * SAL's 3 control-block user bits at alloc time, so a reader routes a byte to the
    * right cacheline without reading CL0. A common prefix forces r1 (CL0 must be
    * read to consume the prefix first), even for a full node.
    */
   enum class dispatch_shape : uint8_t
   {
      r1 = 0,  // branches in CL0 (compact), or prefix present -> read CL0
      r2 = 1,  // 2 CL x 128-range
      r4 = 2,  // 4 CL x 64-range
      r8 = 3,  // 8 CL x 32-range (full, direct index)
   };

   /**
    * Fields common to every radix node, on the MVCC arena. The node's version lives
    * in alloc_header (versioned::created_version); version_history in versioned;
    * this adds the radix routing metadata. `term_value` is a smart_address so a
    * terminal value may be a node pointer or a local inline allocation.
    *
    * Layout: paged (22 B) + rtype (1) + term_value (4) + prefix_length (2) = 29 B.
    */
   struct radix_base : paged
   {
      radix_node_type _rtype;
      smart_address   _term_value;        // child value when a key ends here; null if none
      // The 2-byte prefix word is split: 12 bits of length (prefix_capacity < 4096, so ample)
      // + a 4-bit segment-presence map. seg_presence is the PERSISTENT truth for the bitset
      // tiers (which 64-wide quarters have a live cacheline); the control-block `type` field
      // mirrors it (high nibble) as a skip-header hint, but control blocks are rebuilt on
      // recovery so the node must carry it. Other tiers leave it 0.
      uint16_t _prefix_length : 12;
      uint16_t _seg_presence : 4;

      radix_node_type rtype() const noexcept { return _rtype; }
      void            set_rtype(radix_node_type t) noexcept { _rtype = t; }

      bool          has_term_value() const noexcept { return !_term_value.is_null(); }
      smart_address term_value() const noexcept { return _term_value; }
      void          set_term_value(smart_address v) noexcept { _term_value = v; }

      uint16_t prefix_length() const noexcept { return _prefix_length; }
      uint8_t  seg_presence() const noexcept { return uint8_t(_seg_presence); }
      void     set_seg_presence(uint8_t p) noexcept { _seg_presence = p; }
   } __attribute__((packed));

   static_assert(sizeof(radix_base) == 29, "paged(22) + rtype(1) + term_value(4) + prefix_length(2)");

   /**
    * CRTP radix-node header. A derived node passes its fixed-field struct as
    * `FixedHeader` (a separate template parameter, not Derived::fixed_header, so it
    * is complete while Derived is still incomplete). CL0 layout:
    *
    *   [ radix_base 29 | FixedHeader | inline prefix[] ]  == one cacheline
    *
    * Branch storage lives in CL1+ (R2/R4/R8) or in the prefix/free region (R1).
    * `Derived` supplies a `static constexpr dispatch_shape tier` (the no-prefix
    * cacheline division); the stamped shape is `prefix_length ? r1 : tier`.
    */
   template <typename Derived, typename FixedHeader>
   struct radix_impl : radix_base
   {
      FixedHeader fixed;

      static constexpr int prefix_capacity =
          int(sal::cacheline_size) - int(sizeof(radix_base)) - int(sizeof(FixedHeader));
      static_assert(prefix_capacity > 0, "FixedHeader leaves no room for an inline prefix");

      uint8_t prefix_data[prefix_capacity];

      std::span<const uint8_t> prefix() const noexcept
      {
         const int n = _prefix_length < prefix_capacity ? _prefix_length : prefix_capacity;
         return {prefix_data, static_cast<size_t>(n)};
      }
      void set_prefix(std::span<const uint8_t> p) noexcept
      {
         assert(int(p.size()) <= prefix_capacity);
         for (size_t i = 0; i < p.size(); ++i)
            prefix_data[i] = p[i];
         _prefix_length = uint16_t(p.size());
      }

      /// The cacheline-division shape to stamp on this node's control block.
      dispatch_shape stamp_shape() const noexcept
      {
         return _prefix_length ? dispatch_shape::r1 : Derived::tier;
      }

      Derived&       self() noexcept { return *static_cast<Derived*>(this); }
      const Derived& self() const noexcept { return *static_cast<const Derived*>(this); }

      // ── default node lifecycle (a node is sizeof(Derived) + a trailing tail heap) ──
      // The leaf page (a concrete, non-template node) writes the same shape inline; only
      // the radix node *template family* needs sizeof(Derived) via CRTP, so it lives here.
      // setlist overrides head_size()/free_heap() (wide index past CL0).
      static constexpr uint32_t headroom = radix_node_headroom;
      static constexpr uint32_t head_bytes() noexcept { return uint32_t(sizeof(Derived)); }
      uint32_t                  head_size() const noexcept { return uint32_t(sizeof(Derived)); }
      uint32_t                  free_heap() const noexcept
      {
         return this->free_bytes(reinterpret_cast<const char*>(this) + sizeof(Derived));
      }
      static uint32_t alloc_size(std::span<const uint8_t> prefix) noexcept
      {
         (void)prefix;  // prefix lives in CL0's variable region; size is fixed + headroom
         return ucc::round_up_multiple<sal::cacheline_size>(uint32_t(sizeof(Derived)) + headroom);
      }
      template <typename Session>
      void destroy(const Session& s) const
      {
         self().for_each_child([&](sal::ptr_address c) { s->release(c); });
      }
   } __attribute__((packed));

   // The 256-branch space is split into RANGE_COUNT ranges (top 3 bits of the branch byte); each
   // range's branch-SET has one refcount in the SAL implicit_ref_pool (how many node-versions
   // share that set). 8 ranges × 32 branches.
   // Number of per-range branch-set refcounts a big node carries (over the 256-branch space).
   // Tunable resolution knob: 8 ranges × 32 branches (header 18B) or 16 × 16 (header 34B). Higher
   // resolution halves the per-divergence kept-children retain but doubles clone/destroy/birth
   // counter ops — measure both. range_of is branch-uniform: byte / (256/range_count).
   inline constexpr int range_count = 16;
   inline constexpr int range_of(unsigned byte) noexcept
   {
      return int((byte * unsigned(range_count)) >> 8);
   }

   /**
    * Per-range counter handles for the large_radix family: ONE region + RANGE_COUNT offsets
    * (sal::implicit_ref_addr) — the compact form of 8 per-range counters living in the SAL
    * implicit_ref_pool. Replaces the packed-nibble refcount_roots. 18 bytes. region == no_region
    * means "no counters yet" (fresh/half-built node → destroy falls back to direct release).
    * See doc/micro_refcount.md + doc/task24_handoff.md.
    */
   struct range_ctrs
   {
      static constexpr sal::implicit_ref_region no_region = 0xFFFF;
      sal::implicit_ref_region region = no_region;
      sal::implicit_ref_addr   off[range_count] = {};  // offset within `region`, one per range
   };
   static_assert(sizeof(range_ctrs) == 2 + 2 * range_count, "region(2) + range_count offsets(2 each)");

   /**
    * Base for dense radix nodes (bitset1/r2/r4, full_radix) that batch child retain/release into
    * RANGE_COUNT per-range refcounts. FixedHeader is the range_ctrs handle. K is vestigial (the
    * old packed tier count) — kept only so existing `large_radix<Derived, K>` instantiations
    * compile and `refcount_block_count` stays a marker for is_big. Sparse tiers (setlist*,
    * single-child) skip this and retain their few branches directly.
    */
   template <typename Derived, int K>
   struct large_radix : radix_impl<Derived, range_ctrs>
   {
      static constexpr int refcount_block_count = range_count;  // is_big marker (8 ranges)

      sal::implicit_ref_region rc_region() const noexcept { return this->fixed.region; }
      void rc_set_region(sal::implicit_ref_region r) noexcept { this->fixed.region = r; }
      bool rc_has_ctrs() const noexcept { return this->fixed.region != range_ctrs::no_region; }
      /// Full ptr_address of range r's counter (region + offset).
      sal::ptr_address rc_ctr(int r) const noexcept
      {
         return sal::combine(this->fixed.region, this->fixed.off[r]);
      }
      void rc_set_off(int r, sal::implicit_ref_addr o) noexcept { this->fixed.off[r] = o; }
      const sal::implicit_ref_addr* rc_offs() const noexcept { return this->fixed.off; }  // 8 offsets

      /// Destroy: release each of the RANGE_COUNT per-range counters. The version that drives a
      /// range's counter to 0 is the range-set's last owner → it releases that range's branch
      /// pointers. No nibble math — each counter is its own self-freeing lifetime. term_value is
      /// held directly (not part of a range).
      template <typename Session>
      void destroy(const Session& s) const
      {
         if (not rc_has_ctrs())  // never got counters (defensive) → direct per-child release
         {
            this->self().for_each_child([&](sal::ptr_address c) { s->release(c); });
            return;
         }
         const smart_address tv = this->term_value();
         if (tv.is_pointer())
            s->release(tv.as_pointer());
         for (int r = 0; r < range_count; ++r)
         {
            if (s->rr_release(rc_ctr(r)))  // last owner of this range's set → free its branches
               this->self().for_each_entry(
                   [&](uint8_t byte, smart_address sa)
                   {
                      if (sa.is_pointer() && range_of(byte) == r)
                         s->release(sa.as_pointer());
                   });
         }
      }
   } __attribute__((packed));
}  // namespace sal_art_ref

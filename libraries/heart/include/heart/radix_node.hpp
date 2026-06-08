#pragma once
#include <cassert>
#include <cstdint>
#include <span>

#include <heart/paged.hpp>
#include <heart/smart_address.hpp>
#include <sal/config.hpp>  // cacheline_size

namespace heart
{
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
      smart_address   _term_value;     // child value when a key ends here; null if none
      uint16_t        _prefix_length;  // logical shared-prefix length

      radix_node_type rtype() const noexcept { return _rtype; }
      void            set_rtype(radix_node_type t) noexcept { _rtype = t; }

      bool          has_term_value() const noexcept { return !_term_value.is_null(); }
      smart_address term_value() const noexcept { return _term_value; }
      void          set_term_value(smart_address v) noexcept { _term_value = v; }

      uint16_t prefix_length() const noexcept { return _prefix_length; }
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
   } __attribute__((packed));

   /**
    * The refcount-tree roots for the large_radix family: K packed-count control
    * blocks. Each block reinterprets its control_block's 41-bit location field
    * (flagged "not a location" via the spare control-block type bit) as 10 x 4-bit
    * deferred-share counters — an implicit B-tree over the 256 branch-numbers
    * (fan-8 leaf + fan-4 upper). K = 1/2/4 by tier. The group-counter retain/release
    * COW is impl-phase.
    */
   template <int K>
   struct refcount_roots
   {
      sal::ptr_address blocks[K];
   };
   static_assert(sizeof(refcount_roots<1>) == 4);
   static_assert(sizeof(refcount_roots<4>) == 16);

   /**
    * Base for radix nodes dense enough to want group-counter COW (N >= ~16):
    * bitset1 (K=1), bitset_r2 (K=2), bitset_r4 / full_radix (K=4). Its FixedHeader
    * is the refcount roots; the branch storage is the Derived's trailing cachelines
    * (or, for bitset1, the CL0 variable region). Sparse nodes (setlist*, single-
    * child) skip this and retain their few branches directly.
    */
   template <typename Derived, int K>
   struct large_radix : radix_impl<Derived, refcount_roots<K>>
   {
      static constexpr int refcount_block_count = K;

      sal::ptr_address*       refcount_blocks() noexcept { return this->fixed.blocks; }
      const sal::ptr_address* refcount_blocks() const noexcept { return this->fixed.blocks; }
   } __attribute__((packed));
}  // namespace heart

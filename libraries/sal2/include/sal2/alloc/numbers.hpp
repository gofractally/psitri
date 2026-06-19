#pragma once
#include <ucc/typed_int.hpp>

namespace sal2
{
   using segment_number           = ucc::typed_int<uint32_t, struct segment_number_tag>;
   using allocator_session_number = ucc::typed_int<uint32_t, struct allocator_session_number_tag>;
   using root_object_number       = ucc::typed_int<uint32_t, struct root_object_number_tag>;
   static constexpr root_object_number null_root_index = root_object_number(-1u);

   /**
    * The index to a control_block of a shared pointer.
    */
   using ptr_address                             = ucc::typed_int<uint32_t, struct ptr_address_tag>;
   static constexpr ptr_address null_ptr_address = ptr_address(-1u);

   /// A tree reference: just the root node address (MVCC removed — the version
   /// half was dropped). Kept as a distinct type (not a bare ptr_address) for
   /// call-site clarity and because the root-object slot packs it into its
   /// atomic u64 via pack()/unpack() (root in the low 32 bits, high 32 unused).
   struct tree_id
   {
      ptr_address root;

      constexpr tree_id() noexcept : root(null_ptr_address) {}
      constexpr tree_id(ptr_address r) noexcept : root(r) {}

      /// Pack into u64 for atomic load/store (root in low 32, high 32 = 0).
      uint64_t pack() const noexcept { return uint64_t(*root); }
      static tree_id unpack(uint64_t packed) noexcept { return {ptr_address(uint32_t(packed))}; }

      bool is_valid() const noexcept { return root != null_ptr_address; }

      friend bool operator==(const tree_id&, const tree_id&) noexcept = default;
   };
   static constexpr tree_id null_tree_id{};
   // Reserved high ptr_address range for application sentinels and tags (null,
   // tombstones, transient promote markers, inline tail-offset tags, …). The
   // control-block allocator hands out indices from 0 up, so it can only collide
   // with this range after ~4.3B live control blocks (impossible in practice).
   // SAL treats any value in the range as a non-address: the alloc-hint path SKIPS
   // it instead of resolving it to a (wildly out-of-bounds) cacheline, and the
   // deref/prefetch helpers reject it — so a pointer column that doubles as an
   // alloc hint can safely hold sentinels.
   //
   // The range is the top 2^16 addresses so applications can pack a 16-bit tag
   // into a reserved address (e.g. heart::smart_address encodes an inline tail
   // offset as 0xFFFFFFFF - offset, all of which fall in this range).
   static constexpr uint32_t    reserved_ptr_address_count = 1u << 16;  // 65536
   static constexpr ptr_address first_reserved_ptr_address =
       ptr_address(uint32_t(-1) - (reserved_ptr_address_count - 1));   // 0xFFFF0000
   inline constexpr bool is_reserved_ptr_address(ptr_address a) noexcept
   {
      return *a >= *first_reserved_ptr_address;
   }

   /// Opaque 64-bit payload published with each root. SAL stores it
   /// atomically alongside the root_object_number; the user chooses the
   /// encoding (psitri packs root_addr+ver_addr into it). SAL never
   /// inspects the bits.
   using root_payload                              = std::uint64_t;
   inline constexpr root_payload null_root_payload = 0ull;
}  // namespace sal2

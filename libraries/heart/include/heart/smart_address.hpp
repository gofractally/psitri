#pragma once
#include <cstdint>

#include <sal/numbers.hpp>

namespace heart
{
   /**
    * A typed offset measured from the tail of the node it lives in. Used by
    * smart_address to encode "the value is inline on this page" rather than "the
    * value is at another ptr_address". `offset_from_tail == 0` is reserved to
    * mean null (see smart_address), so a live tail_ptr has offset 1..65535.
    */
   template <typename T = void>
   struct tail_ptr
   {
      uint16_t offset_from_tail;
      bool     is_null() const noexcept { return offset_from_tail == 0; }
   };

   /**
    * A pointer column entry that is *either* a normal sal::ptr_address *or* an
    * inline tail offset, distinguished purely by value (design3.md).
    *
    * The top 2^16 of the u32 address space encode tail offsets:
    *
    *     value >= tail_base (0xFFFF0000)  ->  tail_ptr, offset = 0xFFFFFFFF - value
    *     value <  tail_base               ->  a real ptr_address
    *
    * Because null_ptr_address == 0xFFFFFFFF maps to tail offset 0, null is null
    * in both interpretations — a tail_ptr never legitimately has offset 0.
    *
    * SAL reserves the top 2^16 addresses (sal::first_reserved_ptr_address ==
    * 0xFFFF0000, reserved_ptr_address_count == 1<<16) precisely so this tag space
    * is safe: SAL's alloc-hint path skips reserved values and the deref/prefetch
    * helpers reject them, so a branch column holding tail tags can still double as
    * an alloc-hint source without SAL ever treating a tag as a real address.
    */
   struct smart_address
   {
      static constexpr uint32_t tail_base = 0xFFFF0000u;  // values >= this are tail offsets
      static constexpr uint16_t max_tail  = 0xFFFFu;      // largest representable tail offset

      uint32_t raw;

      // ---- construction ----------------------------------------------------
      static smart_address null() noexcept { return {0xFFFFFFFFu}; }
      static smart_address from_pointer(sal::ptr_address a) noexcept { return {*a}; }
      static smart_address from_tail(uint16_t offset) noexcept { return {0xFFFFFFFFu - offset}; }
      template <typename T>
      static smart_address from_tail(tail_ptr<T> t) noexcept
      {
         return from_tail(t.offset_from_tail);
      }

      // ---- classification --------------------------------------------------
      bool is_null() const noexcept { return raw == 0xFFFFFFFFu; }
      bool is_tail() const noexcept { return raw >= tail_base; }
      bool is_pointer() const noexcept { return raw < tail_base; }

      // ---- decode ----------------------------------------------------------
      sal::ptr_address as_pointer() const noexcept { return sal::ptr_address(raw); }
      uint16_t         tail_offset() const noexcept { return uint16_t(0xFFFFFFFFu - raw); }
      template <typename T = void>
      tail_ptr<T> as_tail() const noexcept
      {
         return tail_ptr<T>{tail_offset()};
      }

      friend bool operator==(const smart_address&, const smart_address&) noexcept = default;
   };

   static_assert(sizeof(smart_address) == 4, "smart_address must be 4 bytes");
   static_assert(smart_address::tail_base == (0xFFFFFFFFu - smart_address::max_tail),
                 "tail encoding and tail_base must agree");
   // The whole tail range must live inside SAL's reserved range, so SAL never
   // mistakes a tail tag for a real address. If SAL's reserved count shrinks, this
   // fires instead of letting tags collide with live addresses.
   static_assert(smart_address::tail_base >= *sal::first_reserved_ptr_address,
                 "smart_address tail range must be within SAL's reserved ptr_address range");
}  // namespace heart

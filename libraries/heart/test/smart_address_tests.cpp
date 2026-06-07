#include <catch2/catch_test_macros.hpp>

#include <heart/smart_address.hpp>

using heart::smart_address;
using heart::tail_ptr;
using sal::ptr_address;

TEST_CASE("smart_address pointer round-trip", "[smart_address]")
{
   auto sa = smart_address::from_pointer(ptr_address(0x12345));
   REQUIRE(sa.is_pointer());
   REQUIRE_FALSE(sa.is_tail());
   REQUIRE_FALSE(sa.is_null());
   REQUIRE(*sa.as_pointer() == 0x12345u);
}

TEST_CASE("smart_address null is null in both interpretations", "[smart_address]")
{
   auto sa = smart_address::null();
   REQUIRE(sa.is_null());
   REQUIRE(sa.is_tail());          // 0xFFFFFFFF is in the tail range
   REQUIRE_FALSE(sa.is_pointer());
   REQUIRE(sa.tail_offset() == 0);  // offset 0 == null sentinel
   REQUIRE(sa.as_tail().is_null());

   // null_ptr_address must encode to the same thing
   auto from_null_ptr = smart_address::from_pointer(sal::null_ptr_address);
   REQUIRE(from_null_ptr == sa);
}

TEST_CASE("smart_address tail offsets round-trip", "[smart_address]")
{
   for (uint32_t off : {1u, 2u, 100u, 4096u, 65534u, 65535u})
   {
      auto sa = smart_address::from_tail(uint16_t(off));
      REQUIRE(sa.is_tail());
      REQUIRE_FALSE(sa.is_pointer());
      REQUIRE_FALSE(sa.is_null());  // only offset 0 is null
      REQUIRE(sa.tail_offset() == off);
      REQUIRE(sa.as_tail().offset_from_tail == off);
   }
}

TEST_CASE("smart_address tail/pointer boundary", "[smart_address]")
{
   // Largest pointer value that is still a pointer.
   auto last_ptr = smart_address::from_pointer(ptr_address(smart_address::tail_base - 1));
   REQUIRE(last_ptr.is_pointer());

   // First value that is a tail (== max representable offset).
   auto first_tail = smart_address{smart_address::tail_base};
   REQUIRE(first_tail.is_tail());
   REQUIRE(first_tail.tail_offset() == smart_address::max_tail);
   REQUIRE(first_tail == smart_address::from_tail(smart_address::max_tail));
}

TEST_CASE("smart_address tail_ptr typed alias carries type", "[smart_address]")
{
   struct value_blob;
   auto sa = smart_address::from_tail(uint16_t(42));
   tail_ptr<value_blob> tp = sa.as_tail<value_blob>();
   REQUIRE(tp.offset_from_tail == 42);
   REQUIRE_FALSE(tp.is_null());

   // construct a smart_address back from a typed tail_ptr
   REQUIRE(smart_address::from_tail(tp) == sa);
}

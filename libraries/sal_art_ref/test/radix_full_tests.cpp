#include <catch2/catch_test_macros.hpp>
#include <map>

#include <sal_art_ref/radix_bitset.hpp>
#include <sal_art_ref/radix_full.hpp>
#include <sal_art_ref/smart_address.hpp>

using sal_art_ref::smart_address;
using sal::ptr_address;

namespace
{
   smart_address addr(uint32_t cacheline, uint32_t slot)
   {
      return smart_address{(cacheline << 4) | (slot & 0xF)};
   }
}  // namespace

TEST_CASE("radix_full basic byte-indexed access", "[radix_full]")
{
   sal_art_ref::radix_full<256> rf;
   rf.init();
   REQUIRE(rf.size() == 0);
   for (int b = 0; b < 256; ++b)
      REQUIRE_FALSE(rf.contains(b));

   rf.set_branch(0, addr(0x10, 1));
   rf.set_branch(200, addr(0x20, 2));
   REQUIRE(rf.size() == 2);
   REQUIRE(rf.contains(0));
   REQUIRE(rf.contains(200));
   REQUIRE_FALSE(rf.contains(1));
   REQUIRE(rf.find(0) == addr(0x10, 1));
   REQUIRE(rf.find(200) == addr(0x20, 2));
   REQUIRE(rf.find(1) == smart_address::null());  // absent

   rf.clear_branch(0);
   REQUIRE_FALSE(rf.contains(0));
   REQUIRE(rf.size() == 1);
}

TEST_CASE("radix_full next_present / prev_present seek (jump then scan)", "[radix_full]")
{
   sal_art_ref::radix_full<256> rf;
   rf.init();
   rf.set_branch(10, addr(0x10, 0));
   rf.set_branch(20, addr(0x10, 1));
   rf.set_branch(200, addr(0x10, 2));

   // exact find does NOT scan
   REQUIRE(rf.find(15) == smart_address::null());

   // next_present: jump to byte, scan right past nulls (inclusive)
   REQUIRE(rf.next_present(10) == 10);   // present -> itself
   REQUIRE(rf.next_present(11) == 20);   // null -> scan right
   REQUIRE(rf.next_present(0) == 10);
   REQUIRE(rf.next_present(21) == 200);
   REQUIRE(rf.next_present(201) == -1);  // none to the right
   REQUIRE(rf.first_present() == 10);

   // prev_present: jump to byte, scan left past nulls (inclusive)
   REQUIRE(rf.prev_present(200) == 200);  // present -> itself
   REQUIRE(rf.prev_present(199) == 20);   // null -> scan left
   REQUIRE(rf.prev_present(15) == 10);
   REQUIRE(rf.prev_present(9) == -1);     // none to the left
   REQUIRE(rf.last_present() == 200);

   // full ascending / descending walks
   std::vector<int> up;
   for (int b = rf.first_present(); b >= 0; b = rf.next_present(b + 1))
      up.push_back(b);
   REQUIRE(up == std::vector<int>{10, 20, 200});

   std::vector<int> down;
   for (int b = rf.last_present(); b >= 0; b = rf.prev_present(b - 1))
      down.push_back(b);
   REQUIRE(down == std::vector<int>{200, 20, 10});
}

TEST_CASE("radix_full lower_bound / upper_bound are bounding keys (= prev/next_present)",
          "[radix_full]")
{
   sal_art_ref::radix_full<256> rf;
   rf.init();
   rf.set_branch(10, addr(0x10, 0));
   rf.set_branch(20, addr(0x10, 1));
   rf.set_branch(200, addr(0x10, 2));

   // lower_bound = floor (greatest present <= x) == prev_present
   REQUIRE(rf.lower_bound(9) == -1);
   REQUIRE(rf.lower_bound(10) == 10);
   REQUIRE(rf.lower_bound(15) == 10);
   REQUIRE(rf.lower_bound(255) == 200);
   REQUIRE(rf.lower_bound(15) == rf.prev_present(15));

   // upper_bound = ceiling (least present >= x) == next_present
   REQUIRE(rf.upper_bound(0) == 10);
   REQUIRE(rf.upper_bound(11) == 20);
   REQUIRE(rf.upper_bound(201) == -1);
   REQUIRE(rf.upper_bound(11) == rf.next_present(11));
}

TEST_CASE("radix_bitset expand fills null where absent", "[radix_full][radix_bitset]")
{
   sal_art_ref::radix_bitset<256, 128> rb;
   rb.init();
   const std::map<int, smart_address> entries{
       {0, addr(0x10, 0)}, {7, addr(0x10, 1)}, {128, addr(0x20, 0)}, {255, addr(0x30, 0)}};
   for (auto& [b, a] : entries)
      REQUIRE(rb.set_branch(b, a));

   sal_art_ref::radix_full<256> rf;
   sal_art_ref::expand(rf, rb);

   REQUIRE(rf.size() == (int)entries.size());
   for (int b = 0; b < 256; ++b)
   {
      auto it = entries.find(b);
      if (it == entries.end())
         REQUIRE(rf.find(b) == smart_address::null());  // null where not any
      else
         REQUIRE(rf.find(b) == it->second);
   }
}

TEST_CASE("radix_full assembles from two radix_bitset<128> halves", "[radix_full][radix_bitset]")
{
   sal_art_ref::radix_bitset<128, 128> lo, hi;
   lo.init();
   hi.init();
   REQUIRE(lo.set_branch(0, addr(0x10, 0)));
   REQUIRE(lo.set_branch(127, addr(0x10, 1)));
   REQUIRE(hi.set_branch(0, addr(0x20, 0)));    // -> byte 128
   REQUIRE(hi.set_branch(127, addr(0x20, 1)));  // -> byte 255

   sal_art_ref::radix_full<256> rf;
   sal_art_ref::expand(rf, lo, hi);

   REQUIRE(rf.size() == 4);
   REQUIRE(rf.find(0) == addr(0x10, 0));
   REQUIRE(rf.find(127) == addr(0x10, 1));
   REQUIRE(rf.find(128) == addr(0x20, 0));
   REQUIRE(rf.find(255) == addr(0x20, 1));
   REQUIRE(rf.find(64) == smart_address::null());
}

TEST_CASE("radix_full assembles from four radix_bitset<64> quarters", "[radix_full][radix_bitset]")
{
   sal_art_ref::radix_bitset<64, 128> q0, q1, q2, q3;
   q0.init();
   q1.init();
   q2.init();
   q3.init();
   REQUIRE(q0.set_branch(1, addr(0x10, 0)));   // byte 1
   REQUIRE(q1.set_branch(2, addr(0x10, 1)));   // byte 66
   REQUIRE(q2.set_branch(3, addr(0x10, 2)));   // byte 131
   REQUIRE(q3.set_branch(63, addr(0x10, 3)));  // byte 255

   sal_art_ref::radix_full<256> rf;
   sal_art_ref::expand(rf, q0, q1, q2, q3);

   REQUIRE(rf.size() == 4);
   REQUIRE(rf.find(1) == addr(0x10, 0));
   REQUIRE(rf.find(66) == addr(0x10, 1));
   REQUIRE(rf.find(131) == addr(0x10, 2));
   REQUIRE(rf.find(255) == addr(0x10, 3));
   REQUIRE(rf.find(0) == smart_address::null());
}

TEST_CASE("radix_bitset expand / contract_from round-trips", "[radix_full][radix_bitset]")
{
   sal_art_ref::radix_bitset<256, 128> rb;
   rb.init();
   const std::map<int, smart_address> entries{
       {3, addr(0x10, 0)}, {64, addr(0x10, 1)}, {65, addr(0x11, 0)},
       {200, addr(0x11, 1)}, {255, addr(0x12, 0)}};
   for (auto& [b, a] : entries)
      REQUIRE(rb.set_branch(b, a));

   sal_art_ref::radix_full<256> rf;
   sal_art_ref::expand(rf, rb);

   sal_art_ref::radix_bitset<256, 128> back;
   back.init();
   REQUIRE(back.contract_from(rf));
   REQUIRE(back.size() == (int)entries.size());
   for (auto& [b, a] : entries)
      REQUIRE(back.find(b) == a);
}

TEST_CASE("radix_full contract_from fails when too dense for one cacheline", "[radix_full]")
{
   // 120 entries across 8 cachelines -> far more than radix_bitset<256,128> holds.
   sal_art_ref::radix_full<256> rf;
   rf.init();
   for (int b = 0; b < 120; ++b)
      rf.set_branch(b, addr(0x10 + b / 16, b % 16));
   REQUIRE(rf.size() == 120);

   sal_art_ref::radix_bitset<256, 128> rb;
   rb.init();
   REQUIRE_FALSE(rb.contract_from(rf));
}

TEST_CASE("radix_full works at 64 and 128 widths too", "[radix_full]")
{
   {
      sal_art_ref::radix_bitset<64, 128> rb;
      rb.init();
      rb.set_branch(0, addr(0x1, 0));
      rb.set_branch(63, addr(0x1, 1));
      sal_art_ref::radix_full<64> rf;
      sal_art_ref::expand(rf, rb);
      REQUIRE(rf.size() == 2);
      REQUIRE(rf.find(0) == addr(0x1, 0));
      REQUIRE(rf.find(63) == addr(0x1, 1));
      REQUIRE(rf.find(30) == smart_address::null());
   }
   {
      sal_art_ref::radix_full<128> rf;
      rf.init();
      rf.set_branch(127, addr(0x2, 3));
      sal_art_ref::radix_bitset<128, 128> rb;
      rb.init();
      REQUIRE(rb.contract_from(rf));
      REQUIRE(rb.size() == 1);
      REQUIRE(rb.find(127) == addr(0x2, 3));
   }
}

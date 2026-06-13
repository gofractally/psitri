#include <catch2/catch_test_macros.hpp>
#include <map>
#include <random>

#include <sal_art_ref/radix_bitset.hpp>
#include <sal_art_ref/smart_address.hpp>

using sal_art_ref::smart_address;
using sal::ptr_address;

namespace
{
   // Build a node-pointer branch with a given cacheline group and slot (keeps
   // inserts dense enough that compact_addresses dedups rather than running out).
   smart_address addr(uint32_t cacheline, uint32_t slot)
   {
      return smart_address{(cacheline << 4) | (slot & 0xF)};
   }
}  // namespace

TEST_CASE("radix_bitset set_branch / find / contains / clear_branch", "[radix_bitset]")
{
   sal_art_ref::radix_bitset<256, 128> rb;
   rb.init();
   REQUIRE(rb.size() == 0);
   REQUIRE(rb.empty());

   REQUIRE(rb.set_branch(50, addr(0x10, 1)));
   REQUIRE(rb.set_branch(10, addr(0x10, 2)));
   REQUIRE(rb.set_branch(200, addr(0x20, 3)));

   REQUIRE(rb.size() == 3);
   REQUIRE(rb.contains(10));
   REQUIRE(rb.contains(50));
   REQUIRE(rb.contains(200));
   REQUIRE_FALSE(rb.contains(11));

   REQUIRE(rb.find(10) == addr(0x10, 2));
   REQUIRE(rb.find(50) == addr(0x10, 1));
   REQUIRE(rb.find(200) == addr(0x20, 3));
   REQUIRE(rb.find(123) == smart_address::null());  // absent

   smart_address removed = smart_address::null();
   REQUIRE(rb.clear_branch(50, removed));
   REQUIRE(removed == addr(0x10, 1));
   REQUIRE(rb.size() == 2);
   REQUIRE_FALSE(rb.contains(50));
   REQUIRE(rb.find(10) == addr(0x10, 2));
   REQUIRE(rb.find(200) == addr(0x20, 3));

   // clearing an absent bit fails and leaves `removed` untouched
   smart_address sentinel = addr(0xDEAD, 0);
   REQUIRE_FALSE(rb.clear_branch(50, sentinel));
   REQUIRE(sentinel == addr(0xDEAD, 0));
}

TEST_CASE("radix_bitset set_branch replaces an existing branch", "[radix_bitset]")
{
   sal_art_ref::radix_bitset<256, 128> rb;
   rb.init();
   REQUIRE(rb.set_branch(42, addr(0x10, 1)));
   REQUIRE(rb.size() == 1);

   // same cacheline, new slot — in-place rewrite, size unchanged
   REQUIRE(rb.set_branch(42, addr(0x10, 7)));
   REQUIRE(rb.size() == 1);
   REQUIRE(rb.find(42) == addr(0x10, 7));

   // different cacheline — re-homes the branch, still size 1
   REQUIRE(rb.set_branch(42, addr(0x55, 2)));
   REQUIRE(rb.size() == 1);
   REQUIRE(rb.find(42) == addr(0x55, 2));
   REQUIRE(rb.present.count() == 1);
}

TEST_CASE("radix_bitset set_branch fails when full, changing nothing", "[radix_bitset]")
{
   // Tiny branch capacity: bitset<64> = 8 bytes, so branches = compact_addresses<8>
   // (data_size 6) — room for ~1 base + a branch.
   sal_art_ref::radix_bitset<64, 16> rb;
   rb.init();
   int    n = 0;
   for (int i = 0; i < 64; ++i)
      if (rb.set_branch(i, addr(0x30 + i, 0)))  // each a new cacheline
         ++n;
      else
         break;
   REQUIRE(n >= 1);
   const int sz = rb.size();

   // The next distinct-cacheline branch must not fit; state is unchanged.
   const int next = sz;  // some absent bit
   REQUIRE_FALSE(rb.set_branch(next + 1, addr(0xFEED, 0)));
   REQUIRE(rb.size() == sz);
   REQUIRE_FALSE(rb.contains(next + 1));
}

TEST_CASE("radix_bitset lower_bound / upper_bound are bounding keys", "[radix_bitset]")
{
   sal_art_ref::radix_bitset<256, 128> rb;
   rb.init();
   rb.set_branch(10, addr(0x10, 0));
   rb.set_branch(20, addr(0x10, 1));
   rb.set_branch(200, addr(0x10, 2));

   // lower_bound = greatest present byte <= x (floor)
   REQUIRE(rb.lower_bound(9) == -1);
   REQUIRE(rb.lower_bound(10) == 10);
   REQUIRE(rb.lower_bound(15) == 10);
   REQUIRE(rb.lower_bound(20) == 20);
   REQUIRE(rb.lower_bound(255) == 200);

   // upper_bound = least present byte >= x (ceiling)
   REQUIRE(rb.upper_bound(0) == 10);
   REQUIRE(rb.upper_bound(10) == 10);
   REQUIRE(rb.upper_bound(11) == 20);
   REQUIRE(rb.upper_bound(201) == -1);

   // the dense branch index of a present byte is present.rank(byte)
   REQUIRE(rb.branches.at(rb.present.rank(20)) == addr(0x10, 1));
}

TEST_CASE("radix_bitset keeps branch order aligned with bit rank", "[radix_bitset]")
{
   sal_art_ref::radix_bitset<256, 128> rb;
   rb.init();
   std::map<int, uint32_t> expect{{200, 0}, {5, 1}, {99, 2}, {6, 3}, {0, 4}};
   for (auto& [byte, slot] : expect)
      REQUIRE(rb.set_branch(byte, addr(0x10, slot)));

   for (auto& [byte, slot] : expect)
      REQUIRE(rb.find(byte) == addr(0x10, slot));

   // Iterating ascending byte (select) walks branches[0..n) in order.
   for (int i = 0; i < rb.size(); ++i)
   {
      const int byte = rb.present.select(i);
      REQUIRE(rb.branches.at(i) == rb.find(byte));
   }
}

TEST_CASE("radix_bitset works at all three widths", "[radix_bitset]")
{
   {
      sal_art_ref::radix_bitset<64, 128> rb;
      rb.init();
      REQUIRE(rb.set_branch(0, addr(0x1, 0)));
      REQUIRE(rb.set_branch(63, addr(0x1, 1)));
      REQUIRE(rb.find(0) == addr(0x1, 0));
      REQUIRE(rb.find(63) == addr(0x1, 1));
      REQUIRE(rb.size() == 2);
   }
   {
      sal_art_ref::radix_bitset<128, 128> rb;
      rb.init();
      REQUIRE(rb.set_branch(127, addr(0x2, 5)));
      REQUIRE(rb.contains(127));
      REQUIRE(rb.find(127) == addr(0x2, 5));
   }
}

TEST_CASE("radix_bitset split into two half-width halves", "[radix_bitset]")
{
   sal_art_ref::radix_bitset<128, 128> rb;
   rb.init();
   rb.set_branch(10, addr(0x10, 0));   // lo
   rb.set_branch(63, addr(0x20, 0));   // lo boundary
   rb.set_branch(64, addr(0x20, 1));   // hi boundary -> byte 0
   rb.set_branch(70, addr(0x10, 1));   // hi -> byte 6
   rb.set_branch(127, addr(0x30, 0));  // hi -> byte 63

   sal_art_ref::radix_bitset<64, 128> lo, hi;
   lo.init();
   hi.init();
   REQUIRE(rb.split(lo, hi));

   REQUIRE(lo.size() == 2);
   REQUIRE(lo.find(10) == addr(0x10, 0));
   REQUIRE(lo.find(63) == addr(0x20, 0));

   REQUIRE(hi.size() == 3);
   REQUIRE(hi.find(0) == addr(0x20, 1));
   REQUIRE(hi.find(6) == addr(0x10, 1));
   REQUIRE(hi.find(63) == addr(0x30, 0));
}

TEST_CASE("radix_bitset merge is the inverse of split", "[radix_bitset]")
{
   sal_art_ref::radix_bitset<128, 128> rb;
   rb.init();
   const std::map<int, smart_address> entries{
       {0, addr(0x10, 0)}, {31, addr(0x10, 1)}, {63, addr(0x11, 0)},
       {64, addr(0x11, 1)}, {100, addr(0x12, 0)}, {127, addr(0x12, 1)}};
   for (auto& [b, a] : entries)
      REQUIRE(rb.set_branch(b, a));

   sal_art_ref::radix_bitset<64, 128> lo, hi;
   lo.init();
   hi.init();
   REQUIRE(rb.split(lo, hi));

   sal_art_ref::radix_bitset<128, 128> back;
   back.init();
   REQUIRE(back.merge(lo, hi));
   REQUIRE(back.size() == (int)entries.size());
   for (auto& [b, a] : entries)
      REQUIRE(back.find(b) == a);
}

TEST_CASE("radix_bitset merge fails when combined content won't fit", "[radix_bitset]")
{
   // Pack each half densely (4 cachelines x 16 slots = 64 entries) so the two
   // together blow past a single radix_bitset<128,128>'s branch capacity.
   sal_art_ref::radix_bitset<64, 128> lo, hi;
   lo.init();
   hi.init();
   for (int b = 0; b < 64; ++b)
   {
      const auto a = addr(0x10 + b / 16, b % 16);
      REQUIRE(lo.set_branch(b, a));
      REQUIRE(hi.set_branch(b, a));
   }
   REQUIRE(lo.size() == 64);
   REQUIRE(hi.size() == 64);

   sal_art_ref::radix_bitset<128, 128> rb;
   rb.init();
   REQUIRE_FALSE(rb.merge(lo, hi));  // 128 branches can't fit one cacheline
}

TEST_CASE("radix_bitset fuzz against std::map", "[radix_bitset][fuzz]")
{
   std::mt19937 rng(99);
   for (int trial = 0; trial < 300; ++trial)
   {
      sal_art_ref::radix_bitset<256, 128> rb;
      rb.init();
      std::map<int, smart_address> ref;

      for (int op = 0; op < 60; ++op)
      {
         const int byte = rng() % 256;
         const int roll = rng() % 3;  // 0 set/replace, 1 set, 2 clear
         if (roll == 2)
         {
            smart_address removed = smart_address::null();
            const bool  ok      = rb.clear_branch(byte, removed);
            REQUIRE(ok == (bool)ref.count(byte));
            if (ok)
            {
               REQUIRE(removed == ref[byte]);
               ref.erase(byte);
            }
         }
         else
         {
            const auto a = addr(0x10 + (rng() % 6), rng() % 16);
            // replace always fits (no new branch); a fresh insert may not
            if (ref.count(byte) || rb.can_insert(a))
            {
               REQUIRE(rb.set_branch(byte, a));
               ref[byte] = a;
            }
         }

         REQUIRE(rb.size() == (int)ref.size());
         for (auto& [b, a] : ref)
         {
            REQUIRE(rb.contains(b));
            REQUIRE(rb.find(b) == a);
         }
      }
   }
}

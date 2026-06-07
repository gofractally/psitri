#include <catch2/catch_test_macros.hpp>
#include <random>
#include <set>
#include <vector>

#include <heart/bitset.hpp>

using heart::bitset;

TEST_CASE("bitset basic set/test/reset", "[bitset]")
{
   bitset<256> b;
   b.clear();
   REQUIRE(b.count() == 0);
   REQUIRE(b.none());
   REQUIRE_FALSE(b.any());

   b.set(0);
   b.set(63);
   b.set(64);
   b.set(255);
   REQUIRE(b.count() == 4);
   REQUIRE(b.test(0));
   REQUIRE(b.test(63));
   REQUIRE(b.test(64));
   REQUIRE(b.test(255));
   REQUIRE_FALSE(b.test(1));
   REQUIRE_FALSE(b.test(254));

   b.reset(64);
   REQUIRE_FALSE(b.test(64));
   REQUIRE(b.count() == 3);
}

TEST_CASE("bitset set_if_unset reports genuine inserts", "[bitset]")
{
   bitset<64> b;
   b.clear();
   REQUIRE(b.set_if_unset(10));
   REQUIRE_FALSE(b.set_if_unset(10));
   REQUIRE(b.count() == 1);
}

TEST_CASE("bitset rank (count of set bits below)", "[bitset]")
{
   bitset<256> b;
   b.clear();
   b.set(3);
   b.set(8);
   b.set(200);

   REQUIRE(b.rank(0) == 0);
   REQUIRE(b.rank(3) == 0);   // nothing strictly below 3
   REQUIRE(b.rank(4) == 1);   // {3}
   REQUIRE(b.rank(8) == 1);   // {3}
   REQUIRE(b.rank(9) == 2);   // {3,8}
   REQUIRE(b.rank(200) == 2); // {3,8}
   REQUIRE(b.rank(201) == 3); // {3,8,200}
   REQUIRE(b.rank(256) == 3); // full range, no OOB read
}

TEST_CASE("bitset lower_bound/upper_bound are bounding keys (floor/ceiling)", "[bitset]")
{
   bitset<256> b;
   b.clear();
   b.set(3);
   b.set(8);
   b.set(200);

   // lower_bound = greatest set bit <= x (floor)
   REQUIRE(b.lower_bound(2) == -1);
   REQUIRE(b.lower_bound(3) == 3);
   REQUIRE(b.lower_bound(7) == 3);
   REQUIRE(b.lower_bound(8) == 8);
   REQUIRE(b.lower_bound(199) == 8);
   REQUIRE(b.lower_bound(255) == 200);

   // upper_bound = least set bit >= x (ceiling)
   REQUIRE(b.upper_bound(0) == 3);
   REQUIRE(b.upper_bound(3) == 3);
   REQUIRE(b.upper_bound(4) == 8);
   REQUIRE(b.upper_bound(9) == 200);
   REQUIRE(b.upper_bound(201) == -1);
}

TEST_CASE("bitset select is the inverse of rank", "[bitset]")
{
   bitset<256> b;
   b.clear();
   b.set(3);
   b.set(8);
   b.set(64);
   b.set(200);

   REQUIRE(b.select(0) == 3);
   REQUIRE(b.select(1) == 8);
   REQUIRE(b.select(2) == 64);
   REQUIRE(b.select(3) == 200);
   REQUIRE(b.select(4) == -1);  // only 4 set bits

   // round-trips
   for (int bit : {3, 8, 64, 200})
      REQUIRE(b.select(b.rank(bit)) == bit);
   for (int n = 0; n < b.count(); ++n)
      REQUIRE(b.rank(b.select(n)) == n);
}

TEST_CASE("bitset select matches brute force (fuzz)", "[bitset][fuzz]")
{
   std::mt19937 rng(7);
   for (int trial = 0; trial < 200; ++trial)
   {
      bitset<256> b;
      b.clear();
      std::set<int> ref;
      const int     k = rng() % 256;
      for (int i = 0; i < k; ++i)
      {
         const int bit = rng() % 256;
         b.set(bit);
         ref.insert(bit);
      }

      std::vector<int> ordered(ref.begin(), ref.end());
      for (int n = 0; n < (int)ordered.size(); ++n)
         REQUIRE(b.select(n) == ordered[n]);
      REQUIRE(b.select((int)ordered.size()) == -1);
   }
}

TEST_CASE("bitset next_set / first_set iteration", "[bitset]")
{
   bitset<128> b;
   b.clear();
   REQUIRE(b.first_set() == -1);

   b.set(5);
   b.set(70);
   b.set(127);
   REQUIRE(b.first_set() == 5);
   REQUIRE(b.next_set(0) == 5);
   REQUIRE(b.next_set(5) == 5);
   REQUIRE(b.next_set(6) == 70);
   REQUIRE(b.next_set(71) == 127);
   REQUIRE(b.next_set(127) == 127);
   REQUIRE(b.next_set(128) == -1);

   // walk every set bit
   std::vector<int> seen;
   for (int i = b.first_set(); i != -1; i = b.next_set(i + 1))
      seen.push_back(i);
   REQUIRE(seen == std::vector<int>{5, 70, 127});
}

TEST_CASE("bitset prev_set / last_set descending iteration", "[bitset]")
{
   bitset<128> b;
   b.clear();
   REQUIRE(b.last_set() == -1);

   b.set(5);
   b.set(70);
   b.set(127);
   REQUIRE(b.last_set() == 127);
   REQUIRE(b.prev_set(128) == 127);  // clamped to N-1
   REQUIRE(b.prev_set(127) == 127);  // inclusive (at or below)
   REQUIRE(b.prev_set(126) == 70);
   REQUIRE(b.prev_set(70) == 70);
   REQUIRE(b.prev_set(69) == 5);
   REQUIRE(b.prev_set(5) == 5);
   REQUIRE(b.prev_set(4) == -1);
   REQUIRE(b.prev_set(0) == -1);

   // walk every set bit high -> low (step with b-1)
   std::vector<int> seen;
   for (int i = b.last_set(); i >= 0; i = b.prev_set(i - 1))
      seen.push_back(i);
   REQUIRE(seen == std::vector<int>{127, 70, 5});
}

TEST_CASE("bitset prev_set matches brute force (fuzz)", "[bitset][fuzz]")
{
   std::mt19937 rng(4321);
   for (int trial = 0; trial < 200; ++trial)
   {
      bitset<256> b;
      b.clear();
      std::set<int> ref;
      const int     k = rng() % 256;
      for (int i = 0; i < k; ++i)
      {
         const int bit = rng() % 256;
         b.set(bit);
         ref.insert(bit);
      }

      // descending walk must equal the reference set in reverse
      std::vector<int> seen;
      for (int i = b.last_set(); i >= 0; i = b.prev_set(i - 1))
         seen.push_back(i);
      REQUIRE(seen == std::vector<int>(ref.rbegin(), ref.rend()));
   }
}

TEST_CASE("bitset rank matches brute force (fuzz)", "[bitset][fuzz]")
{
   std::mt19937 rng(1337);
   for (int trial = 0; trial < 200; ++trial)
   {
      bitset<256> b;
      b.clear();
      std::set<int> ref;
      const int     k = rng() % 256;
      for (int i = 0; i < k; ++i)
      {
         const int bit = rng() % 256;
         b.set(bit);
         ref.insert(bit);
      }
      REQUIRE(b.count() == (int)ref.size());

      for (int probe = 0; probe <= 256; ++probe)
      {
         int expected = 0;
         for (int v : ref)
            if (v < probe)
               ++expected;
         REQUIRE(b.rank(probe) == expected);
      }

      // iteration matches the reference set in order
      std::vector<int> seen;
      for (int i = b.first_set(); i != -1; i = b.next_set(i + 1))
         seen.push_back(i);
      REQUIRE(seen == std::vector<int>(ref.begin(), ref.end()));
   }
}

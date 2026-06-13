#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "radix_tree.hpp"

using sal_art_ref::proto::radix_tree;

namespace
{
   std::span<const uint8_t> key_of(const uint64_t& k)
   {
      return {reinterpret_cast<const uint8_t*>(&k), sizeof(k)};
   }
}  // namespace

TEST_CASE("radix_tree insert/find correctness", "[radix_tree]")
{
   radix_tree         t(128ull << 20);
   std::mt19937_64           rng(20260607);
   std::vector<uint64_t>     keys;
   std::unordered_set<uint64_t> seen;
   while (keys.size() < 50000)
   {
      uint64_t k = rng();
      if (seen.insert(k).second)
         keys.push_back(k);
   }

   for (size_t i = 0; i < keys.size(); ++i)
      t.insert(key_of(keys[i]), keys[i] ^ 0x9e3779b97f4a7c15ull);

   // every inserted key resolves to its stored value
   for (uint64_t k : keys)
   {
      uint64_t v = 0;
      REQUIRE(t.find(key_of(k), v));
      REQUIRE(v == (k ^ 0x9e3779b97f4a7c15ull));
   }

   // update in place
   t.insert(key_of(keys[0]), 42);
   uint64_t v = 0;
   REQUIRE(t.find(key_of(keys[0]), v));
   REQUIRE(v == 42);

   // miss: a key guaranteed absent
   uint64_t absent = rng();
   while (seen.count(absent))
      absent = rng();
   uint64_t miss = 0;
   REQUIRE_FALSE(t.find(key_of(absent), miss));

   // variable-length keys + shared prefixes (one a prefix of another)
   radix_tree s(8ull << 20);
   const char* ks[] = {"a", "ab", "abc", "abd", "abce", "x", ""};
   for (uint32_t i = 0; i < 7; ++i)
      s.insert({reinterpret_cast<const uint8_t*>(ks[i]), std::strlen(ks[i])}, 1000 + i);
   for (uint32_t i = 0; i < 7; ++i)
   {
      uint64_t got = 0;
      REQUIRE(s.find({reinterpret_cast<const uint8_t*>(ks[i]), std::strlen(ks[i])}, got));
      REQUIRE(got == 1000 + i);
   }
}

TEST_CASE("radix_tree random insert/find benchmark", "[.][bench]")
{
   const int N = 1'000'000;

   std::mt19937_64       rng(1);
   std::vector<uint64_t> keys(N);
   for (auto& k : keys)
      k = rng();

   auto now = [] { return std::chrono::steady_clock::now(); };
   auto ns  = [](auto a, auto b)
   { return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };

   // ---- heart radix_tree ----
   radix_tree t(3ull << 30);
   auto       t0 = now();
   for (int i = 0; i < N; ++i)
      t.insert(key_of(keys[i]), uint64_t(i));
   auto t1 = now();

   uint64_t checksum = 0;
   auto     t2       = now();
   for (int i = 0; i < N; ++i)
   {
      uint64_t v = 0;
      if (t.find(key_of(keys[i]), v))
         checksum += v;
   }
   auto t3 = now();

   // ---- std::unordered_map baseline ----
   std::unordered_map<uint64_t, uint64_t> m;
   m.reserve(N);
   auto u0 = now();
   for (int i = 0; i < N; ++i)
      m.emplace(keys[i], uint64_t(i));
   auto u1 = now();
   uint64_t uchk = 0;
   auto     u2   = now();
   for (int i = 0; i < N; ++i)
   {
      auto it = m.find(keys[i]);
      if (it != m.end())
         uchk += it->second;
   }
   auto u3 = now();

   REQUIRE(checksum == uchk);  // both saw every key

   std::printf("\n  radix_tree  : insert %6.1f ns/op   find %6.1f ns/op   (%zu MB arena)\n",
               double(ns(t0, t1)) / N, double(ns(t2, t3)) / N, t.bytes_used() >> 20);
   std::printf("  unordered_map: insert %6.1f ns/op   find %6.1f ns/op\n\n",
               double(ns(u0, u1)) / N, double(ns(u2, u3)) / N);
   std::fflush(stdout);
}

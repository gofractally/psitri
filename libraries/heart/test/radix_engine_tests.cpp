// Correctness for the rebuilt engine on the DESIGNED segmented taxonomy
// (heart::radix::radix_tree): skip-header dispatch + COW-to-root, Phase 1
// (leaf_node + full_radix_node, prefix compression).
//
// Values are arbitrary, dynamically-sized byte arrays (psitri value_view), NOT
// fixed u64s — the tests exercise variable-length values on purpose.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "radix_engine.hpp"

using heart::radix::radix_tree;
using heart::radix::value_view;

namespace
{
   std::string u64key(uint64_t x)  // 8 raw key bytes
   {
      return std::string(reinterpret_cast<const char*>(&x), sizeof(x));
   }
   uint64_t hashed(uint64_t i) { return (i * 0x9E3779B97F4A7C15ull) ^ (i >> 29); }
   // A value whose CONTENT and LENGTH both vary with the seed (arbitrary bytes).
   std::string make_value(uint64_t seed)
   {
      size_t      len = 1 + (seed % 40);  // 1..40 bytes
      std::string v(len, '\0');
      for (size_t j = 0; j < len; ++j)
         v[j] = char(uint8_t(seed * 31 + j));
      return v;
   }
   std::filesystem::path tmp(const char* name)
   {
      return std::filesystem::temp_directory_path() / name;
   }
}  // namespace

TEST_CASE("radix_engine: random u64 keys, variable-size values vs std::map", "[radix_engine]")
{
   radix_tree                      t(tmp("heart_re_u64"));
   std::map<uint64_t, std::string> oracle;

   const int N = 20000;
   for (int i = 0; i < N; ++i)
   {
      uint64_t    k = hashed(uint64_t(i));
      std::string v = make_value(uint64_t(i));
      t.upsert(u64key(k), v);
      oracle[k] = v;
   }
   for (auto& [k, v] : oracle)
   {
      value_view got;
      REQUIRE(t.find(u64key(k), got));
      REQUIRE(std::string(got) == v);
   }
   for (int i = N; i < N + 5000; ++i)  // misses
   {
      value_view got;
      REQUIRE_FALSE(t.find(u64key(hashed(uint64_t(i))), got));
   }
}

TEST_CASE("radix_engine: string keys, prefixes, and terminals", "[radix_engine]")
{
   radix_tree               t(tmp("heart_re_str"));
   std::vector<std::string> keys = {"a",     "ab",   "abc",     "abcd", "abd", "b", "ba",
                                    "hello", "help", "helping", "",     "x",   "xy", "xyz",
                                    "xyzzy"};
   for (size_t i = 0; i < keys.size(); ++i)
      t.upsert(keys[i], make_value(i + 1));
   for (size_t i = 0; i < keys.size(); ++i)
   {
      value_view got;
      REQUIRE(t.find(keys[i], got));
      REQUIRE(std::string(got) == make_value(i + 1));
   }
   value_view got;
   REQUIRE_FALSE(t.find("abcde", got));
   REQUIRE_FALSE(t.find("hel", got));
   REQUIRE_FALSE(t.find("z", got));
}

TEST_CASE("radix_engine: forced fan-out exercises bitset_r2/r4 tiers", "[radix_engine]")
{
   // Keys share a 3-byte prefix then diverge on one byte over K distinct values,
   // forcing the child node through setlist -> bitset_r2 -> bitset_r4 as K grows.
   for (int K : {30, 100, 250})
   {
      radix_tree t(tmp("heart_re_fanout"));
      for (int i = 0; i < K; ++i)
      {
         std::string key = "AAA";
         key.push_back(char(uint8_t(i)));
         key += "tail";
         t.upsert(key, make_value(uint64_t(i)));
      }
      for (int i = 0; i < K; ++i)
      {
         std::string key = "AAA";
         key.push_back(char(uint8_t(i)));
         key += "tail";
         value_view got;
         REQUIRE(t.find(key, got));
         REQUIRE(std::string(got) == make_value(uint64_t(i)));
      }
      auto tc = t.count_tiers();
      if (K == 30)
         REQUIRE(tc.bitset_r2 >= 1);  // 30 > setlist cap but fits 2 half-cachelines
      if (K == 250)
         REQUIRE((tc.bitset_r4 >= 1 || tc.full >= 1));  // 250 distinct → r4 or full
   }
}

TEST_CASE("radix_engine: overwrite returns prior size + COW commit cycle", "[radix_engine]")
{
   radix_tree t(tmp("heart_re_cow"));
   const int  N = 4000;
   for (int round = 0; round < 6; ++round)
   {
      for (int i = 0; i < N; ++i)
      {
         uint64_t    k = hashed(uint64_t(i));
         std::string v = make_value(uint64_t(round * 100000 + i));  // size varies by round
         int         prv = t.upsert(u64key(k), v);
         if (round == 0)
            REQUIRE(prv == -1);  // new key
         else
            REQUIRE(prv == int(make_value(uint64_t((round - 1) * 100000 + i)).size()));
      }
      t.commit();
      for (int i = 0; i < N; ++i)
      {
         value_view got;
         REQUIRE(t.find(u64key(hashed(uint64_t(i))), got));
         REQUIRE(std::string(got) == make_value(uint64_t(round * 100000 + i)));
      }
   }
}

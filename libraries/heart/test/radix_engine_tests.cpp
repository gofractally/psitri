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
#include <random>
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
      // With inline terminals, the fan-out node widens past setlist; the exact tier
      // depends on how inline tails fill compact bases (each tail = one base), so just
      // require it widened beyond setlist (a wider segmented tier appeared).
      auto tc = t.count_tiers();
      REQUIRE((tc.bitset_r2 + tc.bitset_r4 + tc.full) >= 1);
   }
}

TEST_CASE("radix_engine: wide stems create + grow on collision (gate on, no split)",
          "[radix_engine]")
{
   // Keys 'A' + byte(i) + byte(i*7) + "CDfg" (7 bytes). They collide on 'A' at the root;
   // the two 6-byte suffixes diverge at byte 0 (≥4 each) → a wide setlist_u32 child is
   // created with 4-byte stems [byte(i), byte(i*7), 'C', 'D'] and a small inline tail "fg".
   // Stems stay distinct, values are tiny, 60 entries stay well under 4 KB → no split. This
   // validates create + wide route + grow, and that a u32 wide node SURVIVES.
   auto mk = [](int i) {
      std::string s(1, 'A');
      s.push_back(char(uint8_t(i)));
      s.push_back(char(uint8_t(i * 7)));
      s += "CDfg";
      return s;  // 7 bytes
   };
   const int N = 60;

   radix_tree t(tmp("heart_re_wide_on"));
   t.set_wide_stems(true);
   for (int i = 0; i < N; ++i)
      t.upsert(mk(i), make_value(uint64_t(i % 4)));  // tiny values, no overflow
   for (int i = 0; i < N; ++i)
   {
      value_view got;
      REQUIRE(t.find(mk(i), got));
      REQUIRE(std::string(got) == make_value(uint64_t(i % 4)));
   }
   value_view g;
   REQUIRE_FALSE(t.find("Zzzzzzzz", g));
   REQUIRE(t.count_tiers().setlist_w32 >= 1);  // a u32 wide node survives (no split)
   REQUIRE(t.counters().restrides == 0);

   radix_tree off(tmp("heart_re_wide_off"));  // gate off: same keys, NO wide nodes, still correct
   for (int i = 0; i < N; ++i)
      off.upsert(mk(i), make_value(uint64_t(i % 4)));
   for (int i = 0; i < N; ++i)
   {
      value_view got;
      REQUIRE(off.find(mk(i), got));
   }
   auto oc = off.count_tiers();
   REQUIRE(oc.setlist_w16 + oc.setlist_w32 == 0);
}

TEST_CASE("radix_engine: wide stems split (re_stride) on 4 KB overflow stays correct",
          "[radix_engine]")
{
   // 200 keys 'A' + byte(i) + "BCDEFG" with values up to 40 bytes. The wide u32 child
   // overflows 4 KB and re_strides; because all stems differ only in byte 0, it cascades
   // u32 → u16 → full_radix (every split group is a singleton). The point is that re_stride
   // FIRES and every key still resolves to the right value through the split structure.
   auto mk = [](int i) {
      std::string s(1, 'A');
      s.push_back(char(uint8_t(i)));
      s += "BCDEFG";
      return s;  // 8 bytes
   };
   const int N = 200;

   radix_tree t(tmp("heart_re_wide_split"));
   t.set_wide_stems(true);
   for (int i = 0; i < N; ++i)
      t.upsert(mk(i), make_value(uint64_t(i)));
   for (int i = 0; i < N; ++i)
   {
      value_view got;
      REQUIRE(t.find(mk(i), got));
      REQUIRE(std::string(got) == make_value(uint64_t(i)));
   }
   value_view g;
   REQUIRE_FALSE(t.find("Zzzzzzzz", g));
   REQUIRE(t.counters().restrides >= 1);  // a split actually happened
}

TEST_CASE("radix_engine: ordered container — in-order, lower_bound, upper_bound vs std::map",
          "[radix_engine]")
{
   // This is a SORTED container: in-order traversal and lower/upper_bound must match
   // std::map exactly, over mixed keys (random binary, shared-prefix families, varied
   // lengths) that exercise every tier — including wide stems when the gate is on.
   for (bool wide : {false, true})
   {
      radix_tree t(tmp(wide ? "heart_ord_wide" : "heart_ord_u8"));
      t.set_wide_stems(wide);
      std::map<std::string, std::string> oracle;
      std::mt19937_64                    rng(424242);
      const int                          N = 6000;
      for (int i = 0; i < N; ++i)
      {
         std::string k;
         switch (int(rng() % 3))
         {
            case 0:
               for (int j = 0, len = 1 + int(rng() % 12); j < len; ++j)
                  k.push_back(char(uint8_t(rng())));
               break;
            case 1:
               k = "pre";
               for (int j = 0, len = 1 + int(rng() % 10); j < len; ++j)
                  k.push_back(char('a' + int(rng() % 6)));
               break;
            default:
               k = "ABCDEFGH";
               for (int j = 0, len = int(rng() % 8); j < len; ++j)
                  k.push_back(char(uint8_t(rng())));
         }
         std::string v = make_value(rng());
         t.upsert(k, v);
         oracle[k] = v;
      }

      // in-order == oracle order
      auto it    = oracle.begin();
      int  count = 0;
      t.for_each_inorder([&](std::string_view key, std::string_view val) {
         REQUIRE(it != oracle.end());
         REQUIRE(std::string(key) == it->first);
         REQUIRE(std::string(val) == it->second);
         ++it;
         ++count;
      });
      REQUIRE(count == int(oracle.size()));

      auto check = [&](const std::string& q) {
         std::string ck;
         value_view  cv;
         bool        got = t.lower_bound(q, ck, cv);
         auto        oi  = oracle.lower_bound(q);
         REQUIRE(got == (oi != oracle.end()));
         if (got)
         {
            REQUIRE(ck == oi->first);
            REQUIRE(std::string(cv) == oi->second);
         }
         got = t.upper_bound(q, ck, cv);
         oi  = oracle.upper_bound(q);
         REQUIRE(got == (oi != oracle.end()));
         if (got)
         {
            REQUIRE(ck == oi->first);
            REQUIRE(std::string(cv) == oi->second);
         }
      };
      for (auto& [k, v] : oracle)
      {
         check(k);                                        // present
         if (!k.empty())
            check(k.substr(0, k.size() - 1));             // a prefix
         check(k + '\0');                                 // just past a key
      }
      for (int i = 0; i < 4000; ++i)                      // random (mostly absent)
      {
         std::string q;
         for (int j = 0, len = int(rng() % 14); j < len; ++j)
            q.push_back(char(uint8_t(rng())));
         check(q);
      }
      check("");                                          // before everything
      check(std::string(20, char(0xFF)));                 // past everything
      for (int b = 0; b < 256; ++b)
         check(std::string(1, char(uint8_t(b))));
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

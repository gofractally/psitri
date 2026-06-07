#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hash/xxhash.h>

#include "sal_radix_tree.hpp"

using heart::sal_proto::sal_radix_tree;

namespace
{
   std::span<const uint8_t> key_of(const uint64_t& k)
   {
      return {reinterpret_cast<const uint8_t*>(&k), sizeof(k)};
   }
   // Hash-derived key: the benchmark stores NO input array — the i-th key is
   // hash(i), recomputed identically on the find/verify pass.
   uint64_t hkey(uint64_t i) noexcept { return XXH3_64bits(&i, sizeof(i)); }

   std::filesystem::path tmp(const char* name)
   {
      return std::filesystem::temp_directory_path() / name;
   }
}  // namespace

TEST_CASE("sal_radix_tree lower/upper bound vs std::map oracle", "[sal_radix_tree]")
{
   auto span_of = [](const std::string& s)
   { return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(s.data()), s.size()); };

   sal_radix_tree                  t(tmp("heart_lub"));
   std::map<std::string, uint64_t> oracle;
   std::mt19937                    rng(2024);
   for (int i = 0; i < 8000; ++i)
   {
      int         len = 1 + int(rng() % 12);
      std::string k(size_t(len), 'a');
      for (auto& ch : k) ch = char('a' + (rng() % 5));  // small alphabet → dense ordering
      oracle[k] = uint64_t(i);              // last write wins (matches t.insert)
      t.insert(span_of(k), uint64_t(i));
   }

   auto probe = [&](const std::string& q)
   {
      uint64_t lv = 0, uv = 0;
      bool     lf = t.lower_bound(span_of(q), lv);
      bool     uf = t.upper_bound(span_of(q), uv);
      auto     ol = oracle.lower_bound(q);
      auto     ou = oracle.upper_bound(q);
      REQUIRE(lf == (ol != oracle.end()));
      if (lf) REQUIRE(lv == ol->second);
      REQUIRE(uf == (ou != oracle.end()));
      if (uf) REQUIRE(uv == ou->second);
   };

   // exact keys, near-misses, prefixes, and random probes
   for (auto& [k, v] : oracle)
   {
      probe(k);                 // exact: lower_bound hits it, upper_bound the next
      probe(k + "a");           // just past
      if (k.size() > 1) probe(k.substr(0, k.size() - 1));  // a prefix of it
   }
   for (int i = 0; i < 3000; ++i)
   {
      int         len = 1 + int(rng() % 13);
      std::string q(size_t(len), 'a');
      for (auto& ch : q) ch = char('a' + (rng() % 6));  // wider alphabet → many misses
      probe(q);
   }
   probe("");           // before everything
   probe("zzzzzzzz");   // after everything
}

TEST_CASE("sal_radix_tree stem corner cases", "[sal_radix_tree]")
{
   auto span_of = [](const std::string& s)
   { return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(s.data()), s.size()); };

   // Insert a key set in a given order, then verify every key round-trips with
   // value = its index in `keys` (NOT insertion index), plus a guaranteed miss.
   auto check_set = [&](const char* dir, std::vector<std::string> keys,
                        std::vector<int> insert_order, const std::string& absent)
   {
      sal_radix_tree t(tmp(dir));
      for (int idx : insert_order)
         t.insert(span_of(keys[idx]), uint64_t(idx));
      for (size_t i = 0; i < keys.size(); ++i)
      {
         uint64_t v = ~0ull;
         REQUIRE(t.find(span_of(keys[i]), v));
         REQUIRE(v == i);
      }
      uint64_t miss = 0;
      REQUIRE_FALSE(t.find(span_of(absent), miss));
   };

   SECTION("terminal falls within the 4-byte stem range (nested prefixes)")
   {
      // a,ab,abc,abcd,abcde,abcdef,abcdefgh — each a prefix of the next, crossing
      // the 4- and 8-byte stem boundaries. A short key terminates inside a longer
      // key's stem window.
      std::vector<std::string> ks = {"a",      "ab",      "abc",     "abcd",
                                     "abcde",  "abcdef",  "abcdefg", "abcdefgh"};
      std::vector<int>         fwd(ks.size()), rev(ks.size());
      for (size_t i = 0; i < ks.size(); ++i) { fwd[i] = int(i); rev[i] = int(ks.size() - 1 - i); }
      check_set("heart_corner_nest_fwd", ks, fwd, "abcdefghi");  // grow into stems
      check_set("heart_corner_nest_rev", ks, rev, "abcdefghi");  // split when shorter arrives
   }

   SECTION("4-byte stems in the middle of long words")
   {
      // Long shared prefix, divergence past the first stem (byte 8) and at byte 4.
      std::vector<std::string> ks;
      for (int i = 0; i < 60; ++i)
      {
         char buf[32];
         std::snprintf(buf, sizeof(buf), "abcdefgh%02d", i);  // diverge at byte 8 (mid-word)
         ks.push_back(buf);
      }
      ks.push_back("abcdWXYZ0000");  // diverge at byte 4
      ks.push_back("abcdefghijklmnop");
      std::vector<int> ord(ks.size());
      for (size_t i = 0; i < ks.size(); ++i) ord[i] = int(i);
      check_set("heart_corner_midword", ks, ord, "abcdefgh60");
   }

   SECTION("stem value is a pointer (continuation), not a terminal")
   {
      // At the 4-byte point "stem"/"wxyz" there is BOTH a terminal key AND keys
      // that continue past it (so the branch must be a pointer, not inline value).
      std::vector<std::string> ks = {"stem",     "stemABCD", "stemABCE", "stemXYZW",
                                     "wxyz",     "wxyzabcd", "wxyzabce"};
      std::vector<int>         ord(ks.size()), rev(ks.size());
      for (size_t i = 0; i < ks.size(); ++i) { ord[i] = int(i); rev[i] = int(ks.size()-1-i); }
      check_set("heart_corner_ptr_fwd", ks, ord, "ste");
      check_set("heart_corner_ptr_rev", ks, rev, "stemAB");
   }

   SECTION("heavy random variable-length keys (1..16 bytes)")
   {
      // Stress terminals-in-range + mid-word stems + pointer continuations at scale,
      // in random insert order.
      sal_radix_tree               t(tmp("heart_corner_varlen"));
      std::mt19937                 rng(777);
      std::unordered_map<std::string, uint64_t> ref;
      for (int i = 0; i < 30000; ++i)
      {
         int         len = 1 + int(rng() % 16);
         std::string k(size_t(len), 'a');
         for (auto& c : k) c = char('a' + (rng() % 6));  // small alphabet → dense prefixes/collisions
         ref[k] = uint64_t(i);  // last write wins
         t.insert(span_of(k), uint64_t(i));
      }
      for (auto& [k, v] : ref)
      {
         uint64_t got = ~0ull;
         REQUIRE(t.find(span_of(k), got));
         REQUIRE(got == v);
      }
   }
}

TEST_CASE("sal_radix_tree path compression", "[sal_radix_tree]")
{
   auto span_of = [](const std::string& s)
   { return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(s.data()), s.size()); };

   SECTION("long shared prefix compresses to few nodes")
   {
      sal_radix_tree t(tmp("heart_sal_radix_pathcomp"));
      const std::string prefix = "the/quick/brown/fox/jumps/over/";  // 31 shared bytes
      const int         n      = 2000;
      for (int i = 0; i < n; ++i)
      {
         char buf[8];
         std::snprintf(buf, sizeof(buf), "%04d", i);
         t.insert(span_of(prefix + buf), uint64_t(i) ^ 0xABCDull);
      }
      for (int i = 0; i < n; ++i)
      {
         char buf[8];
         std::snprintf(buf, sizeof(buf), "%04d", i);
         uint64_t v = 0;
         REQUIRE(t.find(span_of(prefix + buf), v));
         REQUIRE(v == (uint64_t(i) ^ 0xABCDull));
      }
      // Compression proof: a naive byte-per-node trie would need ~ n*len nodes
      // (>70k). The 31-byte shared prefix must be stored on a handful of nodes —
      // the only branching is over the 4 trailing digits.
      std::printf("  [pathcomp] %d keys, %d-byte shared prefix -> %llu nodes\n", n,
                  int(prefix.size()), (unsigned long long)t.object_count());
      std::fflush(stdout);
      REQUIRE(t.object_count() < 1500);
      // A non-prefix that shares 30 of 31 bytes must miss.
      uint64_t miss = 0;
      REQUIRE_FALSE(t.find(span_of("the/quick/brown/fox/jumps/oveX/0000"), miss));
   }

   SECTION("key that is a strict prefix of others, plus the empty key")
   {
      sal_radix_tree t(tmp("heart_sal_radix_prefixkeys"));
      const char*    ks[] = {"", "a", "ab", "abc", "abcd", "abce", "abcdef", "b"};
      for (uint32_t i = 0; i < std::size(ks); ++i)
         t.insert(span_of(ks[i]), 7000 + i);
      for (uint32_t i = 0; i < std::size(ks); ++i)
      {
         uint64_t v = 0;
         REQUIRE(t.find(span_of(ks[i]), v));
         REQUIRE(v == 7000 + i);
      }
      // "ab" and "abz" share a 2-byte prefix; "abz" was never inserted → miss.
      uint64_t miss = 0;
      REQUIRE_FALSE(t.find(span_of("abz"), miss));
      REQUIRE_FALSE(t.find(span_of("abcde"), miss));
   }

   SECTION("setlist with a prefix upgrades to full past 16 branches")
   {
      // A node reached with a non-empty prefix that accumulates >16 distinct
      // branches must upgrade tiers without losing the key that triggers it.
      // (Regression: the upgrade retry used the post-prefix depth, double-counting
      // the prefix and dropping the new branch.)
      sal_radix_tree t(tmp("heart_sal_radix_upgrade_prefix"));
      std::vector<std::string> ks;
      for (int c = 0; c < 40; ++c)  // "XY" + distinct 3rd byte → child node has prefix "Y"
         ks.push_back(std::string("XY") + char('A' + c));
      for (size_t i = 0; i < ks.size(); ++i)
         t.insert(span_of(ks[i]), 9000 + i);
      for (size_t i = 0; i < ks.size(); ++i)
      {
         uint64_t v = 0;
         REQUIRE(t.find(span_of(ks[i]), v));
         REQUIRE(v == 9000 + i);
      }
   }

   SECTION("re-splitting a shared prefix at progressively earlier points")
   {
      sal_radix_tree t(tmp("heart_sal_radix_resplit"));
      // Insert order forces a built-up prefix to be split at earlier offsets.
      const char* ks[] = {"aaaaaaaa", "aaaaaaab", "aaaaaab0", "aaaaab00", "aaaab000", "a0000000"};
      for (uint32_t i = 0; i < std::size(ks); ++i)
         t.insert(span_of(ks[i]), 5000 + i);
      for (uint32_t i = 0; i < std::size(ks); ++i)
      {
         uint64_t v = 0;
         REQUIRE(t.find(span_of(ks[i]), v));
         REQUIRE(v == 5000 + i);
      }
   }
}

TEST_CASE("sal_radix_tree insert/find correctness", "[sal_radix_tree]")
{
   sal_radix_tree               t(tmp("heart_sal_radix_correctness"));
   std::mt19937_64              rng(20260607);
   std::vector<uint64_t>        keys;
   std::unordered_set<uint64_t> seen;
   while (keys.size() < 50000)
   {
      uint64_t k = rng();
      if (seen.insert(k).second)
         keys.push_back(k);
   }

   for (size_t i = 0; i < keys.size(); ++i)
      t.insert(key_of(keys[i]), keys[i] ^ 0x9e3779b97f4a7c15ull);

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

   // miss
   uint64_t absent = rng();
   while (seen.count(absent))
      absent = rng();
   uint64_t miss = 0;
   REQUIRE_FALSE(t.find(key_of(absent), miss));

   // variable-length keys + shared prefixes (one a prefix of another)
   sal_radix_tree s(tmp("heart_sal_radix_varlen"));
   const char*    ks[] = {"a", "ab", "abc", "abd", "abce", "x", ""};
   for (uint32_t i = 0; i < 7; ++i)
      s.insert({reinterpret_cast<const uint8_t*>(ks[i]), std::strlen(ks[i])}, 1000 + i);
   for (uint32_t i = 0; i < 7; ++i)
   {
      uint64_t got = 0;
      REQUIRE(s.find({reinterpret_cast<const uint8_t*>(ks[i]), std::strlen(ks[i])}, got));
      REQUIRE(got == 1000 + i);
   }
}

TEST_CASE("sal_radix_tree insert scaling probe", "[.][scale]")
{
   auto now = [] { return std::chrono::steady_clock::now(); };
   auto ns  = [](auto a, auto b)
   { return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };

   sal_radix_tree t(tmp("heart_sal_radix_scale"));
   const int      batch = 50'000;
   uint64_t       next  = 0;
   for (int b = 0; b < 8; ++b)
   {
      auto s0 = now();
      for (int i = 0; i < batch; ++i)
      {
         uint64_t k = hkey(next);
         t.insert(key_of(k), next);
         ++next;
      }
      auto s1 = now();
      std::printf("  batch %d  (%llu..%llu): %6.1f ns/insert  (%zu MB)\n", b,
                  (unsigned long long)(next - batch), (unsigned long long)next,
                  double(ns(s0, s1)) / batch, t.bytes_used() >> 20);
      std::fflush(stdout);
   }
}

TEST_CASE("sal_radix_tree random insert/find benchmark", "[.][bench]")
{
   const uint64_t N = 1'000'000;

   auto now = [] { return std::chrono::steady_clock::now(); };
   auto ns  = [](auto a, auto b)
   { return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };

   // Keys are hash-derived (key[i] = hash(i)); value[i] = i. No input array, no
   // value array — the find pass recomputes hash(i) and checks the value is i.
   sal_radix_tree t(tmp("heart_sal_radix_bench"));
   auto           t0 = now();
   for (uint64_t i = 0; i < N; ++i)
   {
      uint64_t k = hkey(i);
      t.insert(key_of(k), i);
   }
   auto t1 = now();

   uint64_t checksum = 0;
   auto     t2       = now();
   for (uint64_t i = 0; i < N; ++i)
   {
      uint64_t k = hkey(i);
      uint64_t v = 0;
      if (t.find(key_of(k), v) && v == i)
         checksum += v;
   }
   auto t3 = now();

   std::unordered_map<uint64_t, uint64_t> m;
   m.reserve(N);
   auto u0 = now();
   for (uint64_t i = 0; i < N; ++i)
      m.emplace(hkey(i), i);
   auto u1 = now();
   uint64_t uchk = 0;
   auto     u2   = now();
   for (uint64_t i = 0; i < N; ++i)
   {
      auto it = m.find(hkey(i));
      if (it != m.end() && it->second == i)
         uchk += it->second;
   }
   auto u3 = now();

   const uint64_t expect = N * (N - 1) / 2;  // sum of 0..N-1
   REQUIRE(checksum == expect);
   REQUIRE(uchk == expect);

   std::printf(
       "\n  sal_radix_tree : insert %6.1f ns/op   find %6.1f ns/op   (%zu MB live, %zu MB compacted, "
       "%llu objects)\n",
       double(ns(t0, t1)) / double(N), double(ns(t2, t3)) / double(N), t.bytes_used() >> 20,
       t.bytes_compacted() >> 20, (unsigned long long)t.object_count());
   std::printf("  unordered_map  : insert %6.1f ns/op   find %6.1f ns/op\n\n",
               double(ns(u0, u1)) / double(N), double(ns(u2, u3)) / double(N));
   std::fflush(stdout);
}

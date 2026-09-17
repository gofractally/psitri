// start_ver::hkv::kv_tree over part_alloc — the verbatim hkv node layout + descent running
// on part's line arena. Model-based vs std::map across mixed key shapes.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>
#include <string>

#include <start_ver/hkv/tree.hpp>
#include <start_ver/tree_alloc.hpp>

using namespace start_ver;
namespace fs = std::filesystem;

namespace
{
   fs::path tmp_dir(const char* tag)
   {
      static std::mt19937_64 rng(std::random_device{}());
      auto p = fs::temp_directory_path() /
               (std::string("part_tree_") + tag + "_" + std::to_string(rng()));
      fs::remove_all(p);
      return p;
   }
}  // namespace

using tree_t = start_ver::hkv::kv_tree<std::string_view, uint64_t, start_ver::hkv::mode::none, part_alloc<uint64_t>>;

TEST_CASE("kv_tree on part arena: insert/find/iterate vs std::map")
{
   arena_config c;
   c.dir = tmp_dir("smoke");
   arena a(c);
   {
      auto   ses = a.start_session();
      auto   tx  = ses.start_transaction(0);
      tree_t t{part_alloc<uint64_t>(&ses)};

      std::map<std::string, uint64_t> oracle;
      std::mt19937_64                 rng(42);

      // Mixed key shapes: binary u64 (uniform), shared-prefix strings, short words.
      std::vector<std::string> keys;
      for (int i = 0; i < 20000; ++i)
      {
         const uint64_t r = rng();
         keys.emplace_back(reinterpret_cast<const char*>(&r), 8);
      }
      for (int i = 0; i < 20000; ++i)
         keys.push_back("user/profile/" + std::to_string(rng() % 100000) + "/name");
      for (int i = 0; i < 5000; ++i)
         keys.push_back(std::to_string(rng() % 1000000));

      uint64_t v = 0;
      for (auto& k : keys)
      {
         const bool fresh = t.insert(std::string_view(k), v) ;
         REQUIRE(fresh == !oracle.count(k));
         oracle[k] = v;
         ++v;
      }
      REQUIRE(t.size() == oracle.size());

      // Point lookups: every key present with the latest value; misses miss.
      for (auto& [k, val] : oracle)
      {
         uint64_t out = 0;
         REQUIRE(t.find(std::string_view(k), out));
         REQUIRE(out == val);
      }
      for (int i = 0; i < 10000; ++i)
      {
         std::string k = "missing/" + std::to_string(rng());
         REQUIRE(t.contains(std::string_view(k)) == (oracle.count(k) != 0));
      }

      // Overwrite half the keys, re-verify.
      size_t n = 0;
      for (auto& [k, val] : oracle)
      {
         if (++n % 2)
            continue;
         REQUIRE_FALSE(t.insert(std::string_view(k), val + 7'000'000));
         oracle[k] = val + 7'000'000;
      }
      for (auto& [k, val] : oracle)
      {
         uint64_t out = 0;
         REQUIRE(t.find(std::string_view(k), out));
         REQUIRE(out == val);
      }

      // Ordered iteration matches the oracle exactly.
      auto oit = oracle.begin();
      for (auto it = t.begin(); it != t.end(); ++it, ++oit)
      {
         REQUIRE(oit != oracle.end());
         REQUIRE((*it).first == oit->first);
         REQUIRE((*it).second == oit->second);
      }
      REQUIRE(oit == oracle.end());
   }  // tree dtor frees every node; session dtor returns its word window

   // Everything the tree allocated was released: the arena drains to empty.
   uint64_t free_total = 0;
   for (uint32_t s = 0; s < a.num_segments(); ++s)
      free_total += a.freeset_of(s).free_lines();
   REQUIRE(free_total == uint64_t(a.num_segments()) * lines_per_segment);
}

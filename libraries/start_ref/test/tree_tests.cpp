// start_ref::hkv::kv_tree over part_alloc — the verbatim hkv node layout + descent running
// on part's line arena. Model-based vs std::map across mixed key shapes.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>
#include <string>

#include <start_ref/hkv/tree.hpp>
#include <start_ref/tree_alloc.hpp>

using namespace start_ref;
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

using tree_t = start_ref::hkv::kv_tree<std::string_view, uint64_t, start_ref::hkv::mode::none, part_alloc<uint64_t>>;

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

// TEMP shape audit: do shared-prefix Ethereum-shaped keys create single-child setlists?
TEST_CASE("kv_tree shape audit: single-child setlists on shared-prefix keys")
{
   arena_config c;
   c.dir = tmp_dir("shape");
   arena a(c);
   auto   ses = a.start_session();
   auto   tx  = ses.start_transaction(0);
   tree_t t{part_alloc<uint64_t>(&ses)};

   // Ethereum storage shape: 2000 accounts (20-byte address) x 40 slots (12-byte) =>
   // every account's 40 keys share a 20-byte prefix (> PREFIX_CAP=8 => must path-compress).
   std::mt19937_64 rng(7);
   uint64_t        v = 0, nkeys = 0;
   for (int acct = 0; acct < 2000; ++acct)
   {
      uint8_t addr[20];
      for (auto& b : addr) b = uint8_t(rng());
      for (int slot = 0; slot < 40; ++slot)
      {
         std::string key(reinterpret_cast<char*>(addr), 20);
         uint64_t    s = uint64_t(slot);
         key.append(reinterpret_cast<char*>(&s), 8);
         t.insert(std::string_view(key), v++);
         ++nkeys;
      }
   }
   auto st = t.audit_shape();
   std::printf("\n[SHAPE] keys=%llu size_hint=%zu total_nodes=%zu leaves=%zu routers=%zu prefix=%zu\n",
               (unsigned long long)nkeys, t.size_hint(), st.total, st.leaves, st.routers, st.prefix);
   std::printf("[SHAPE] setlist=%zu  SINGLE-CHILD setlists=%zu  single-child routers(any)=%zu\n",
               st.setlist, st.setlist1, st.router1);
   std::printf("[SHAPE] avg leaf depth=%.2f  bytes/key(nodes*128/keys)=%.0f\n",
               st.nleaf ? double(st.leafdepth) / st.nleaf : 0.0,
               nkeys ? double(st.total) * 128.0 / nkeys : 0.0);
   REQUIRE(nkeys == 80000);
}

// TEMP shape audit #2: sequential big-endian keys — the classic single-child-chain trigger.
TEST_CASE("kv_tree shape audit: sequential BE keys (single-child trigger)")
{
   arena_config c; c.dir = tmp_dir("shape2"); arena a(c);
   auto ses = a.start_session(); auto tx = ses.start_transaction(0);
   tree_t t{part_alloc<uint64_t>(&ses)};
   uint64_t v = 0;
   for (uint64_t i = 0; i < 100000; ++i)   // 0..100k as 8-byte BE: long shared zero-prefix
   {
      uint8_t be[8]; for (int b = 0; b < 8; ++b) be[b] = uint8_t(i >> (8 * (7 - b)));
      t.insert(std::string_view(reinterpret_cast<char*>(be), 8), v++);
   }
   auto st = t.audit_shape();
   std::printf("\n[SHAPE2] seq-BE keys=%zu total=%zu leaves=%zu routers=%zu prefix=%zu setlist=%zu "
               "SINGLE-CHILD-setlist=%zu single-child-router=%zu avgdepth=%.2f bytes/key=%.0f\n",
               t.size_hint(), st.total, st.leaves, st.routers, st.prefix, st.setlist,
               st.setlist1, st.router1, st.nleaf ? double(st.leafdepth)/st.nleaf : 0.0,
               double(st.total)*128.0/100000.0);
   REQUIRE(t.size_hint() == 100000);
}

// WASTE AUDIT: open an existing datadir, walk live root-0, compare to on-disk total.
TEST_CASE("waste audit: live tree vs on-disk total")
{
   const char* dir = std::getenv("AUDIT_DIR");
   if (!dir) { SUCCEED("set AUDIT_DIR=<chaindata> to run"); return; }
   arena_config c; c.dir = dir; c.max_segments = 55000;
   arena a(c);
   const uint64_t rootraw = a.get_root(0);
   start_ref::detail::line_ptr root;
   const uint64_t v = rootraw & ((uint64_t(1) << 40) - 1);
   std::memcpy(root.b, &v, sizeof(root.b));
   std::printf("\n[WASTE] root0 raw=0x%llx tag=%d\n", (unsigned long long)rootraw, int(root.tag()));
   auto   ses = a.start_session();
   tree_t t{part_alloc<uint64_t>(&ses)};
   auto   st = t.audit_shape_from(root);

   const double seg_gb  = double(a.num_segments()) * lines_per_segment * 128.0 / 1e9;
   const double live_gb = double(st.total) * 128.0 / 1e9;
   std::printf("[WASTE] segments=%u  on-disk(data.part)=%.1f GB\n", a.num_segments(), seg_gb);
   std::printf("[WASTE] LIVE: nodes=%zu  leaves=%zu  routers=%zu  prefix=%zu  setlist1=%zu  router1=%zu\n",
               st.total, st.leaves, st.routers, st.prefix, st.setlist1, st.router1);
   std::printf("[WASTE] LIVE bytes = %.1f GB  (%.1f%% of disk)   WASTE = %.1f GB (%.0f%%)\n",
               live_gb, 100.0 * live_gb / seg_gb, seg_gb - live_gb,
               100.0 * (seg_gb - live_gb) / seg_gb);
   std::printf("[WASTE] avg leaf depth=%.2f  live bytes/leaf=%.0f\n",
               st.nleaf ? double(st.leafdepth) / st.nleaf : 0.0,
               st.leaves ? live_gb * 1e9 / st.leaves : 0.0);
   SUCCEED("audit done");
}

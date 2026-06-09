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

   // GENERAL ordered-surface correctness: validates EVERY read surface of a tree against a
   // std::map oracle — point find, for_each_inorder, the cursor (forward seek_begin/next,
   // reverse seek_last/prev, lower_bound/upper_bound), and tree-level lower_bound/upper_bound.
   // Each surface is guarded so an unhandled tier throws into a CHECK rather than aborting,
   // and reports independently. This is what the suite lacked: the old ordered test drove only
   // for_each_inorder + tree lower_bound, never the cursor, and never on a bucket tree.
   void check_ordered_surfaces(radix_tree& t, const std::map<std::string, std::string>& oracle,
                               const char* ctx)
   {
      using kv = std::pair<std::string, std::string>;
      INFO("context = " << ctx);
      std::vector<kv> expect(oracle.begin(), oracle.end());  // sorted ground truth

      // (A) point find — distinguishes an INSERT bug from a read/ordering bug.
      {
         bool ok = true;
         for (auto& [k, v] : oracle)
         {
            value_view g;
            if (!t.find(k, g) || std::string(g) != v) { ok = false; break; }
         }
         INFO("surface: find");
         CHECK(ok);
      }
      // (A2) cursor extras: get(key)->optional, full count_keys() == size, range count_keys.
      {
         radix_tree::cursor c(&t);
         INFO("surface: cursor get(key)/count_keys");
         CHECK(c.count_keys() == oracle.size());
         bool getok = true;
         int  s     = 0;
         for (auto& [k, v] : oracle)
         {
            if (s++ % 13) continue;
            auto got = c.template get<std::string>(k);
            if (!got || *got != v) { getok = false; break; }
         }
         CHECK(getok);
         if (oracle.size() > 2)  // range count vs oracle on a sampled [lo,hi)
         {
            auto lo = std::next(oracle.begin(), long(oracle.size() / 4))->first;
            auto hi = std::next(oracle.begin(), long(3 * oracle.size() / 4))->first;
            uint64_t want = 0;
            for (auto& [k, v] : oracle) if (k >= lo && k < hi) ++want;
            CHECK(c.count_keys(lo, hi) == want);
         }
      }
      // (B) tree for_each_inorder == sorted oracle.
      {
         std::vector<kv> got;
         bool            threw = false;
         try { t.for_each_inorder([&](std::string_view k, std::string_view v)
                                  { got.emplace_back(std::string(k), std::string(v)); }); }
         catch (const std::exception& e) { threw = true; INFO("threw: " << e.what()); }
         INFO("surface: for_each_inorder");
         CHECK_FALSE(threw);
         CHECK(got == expect);
      }
      // (C) cursor forward: seek_begin + next == oracle (keys AND values).
      {
         std::vector<kv> got;
         bool            threw = false;
         try {
            radix_tree::cursor c(&t);
            for (bool end = c.seek_begin(); !end; end = c.next())
            {
               std::string vv;
               c.get_value([&](value_view v) { vv.assign(v.data(), v.size()); });
               got.emplace_back(std::string(c.key()), vv);
            }
         } catch (const std::exception& e) { threw = true; INFO("threw: " << e.what()); }
         INFO("surface: cursor forward");
         CHECK_FALSE(threw);
         CHECK(got == expect);
      }
      // (D) cursor reverse: seek_last + prev == reverse(oracle) (keys).
      {
         std::vector<std::string> got;
         bool                     threw = false;
         try {
            radix_tree::cursor c(&t);
            for (bool r = c.seek_last(); !r; r = c.prev()) got.emplace_back(std::string(c.key()));
         } catch (const std::exception& e) { threw = true; INFO("threw: " << e.what()); }
         INFO("surface: cursor reverse");
         CHECK_FALSE(threw);
         std::vector<std::string> rexpect;
         for (auto it = expect.rbegin(); it != expect.rend(); ++it) rexpect.push_back(it->first);
         CHECK(got == rexpect);
      }
      // (E) cursor lower_bound and (F) tree lower_bound vs std::map, sampled — incl. the
      // key-is-a-prefix-of-another-key case (a query that is itself a terminal key).
      {
         auto check_q = [&](const std::string& q) {
            { // cursor lower_bound
               bool threw = false;
               try {
                  radix_tree::cursor c(&t);
                  bool end = c.lower_bound(q);
                  auto oi  = oracle.lower_bound(q);
                  CHECK(end == (oi == oracle.end()));
                  if (!end && oi != oracle.end()) CHECK(std::string(c.key()) == oi->first);
               } catch (const std::exception& e) { threw = true; INFO("threw: " << e.what()); }
               INFO("surface: cursor lower_bound, q.size=" << q.size());
               CHECK_FALSE(threw);
            }
            { // tree lower_bound
               bool threw = false;
               try {
                  std::string ck; value_view cv;
                  bool got = t.lower_bound(q, ck, cv);
                  auto oi  = oracle.lower_bound(q);
                  CHECK(got == (oi != oracle.end()));
                  if (got && oi != oracle.end()) { CHECK(ck == oi->first); CHECK(std::string(cv) == oi->second); }
               } catch (const std::exception& e) { threw = true; INFO("threw: " << e.what()); }
               INFO("surface: tree lower_bound, q.size=" << q.size());
               CHECK_FALSE(threw);
            }
         };
         int s = 0;
         for (auto& [k, v] : oracle)
         {
            if (s++ % 11) continue;  // sample to keep it quick
            check_q(k);                                          // present key
            if (!k.empty()) check_q(k.substr(0, k.size() - 1));  // a PREFIX of a key
            check_q(k + '\0');                                   // just past a key
         }
         check_q("");                                            // before everything
         check_q(std::string(20, char(0xFF)));                   // past everything
      }
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

TEST_CASE("radix_engine: bucket mode — radix routers + bucket leaves vs std::map",
          "[radix_engine]")
{
   // The hybrid: start as a bucket, split full buckets into radix nodes with bucket leaves.
   // Drives many splits + prefix_splits (shared "pre" family + random + long-common-prefix).
   radix_tree t(tmp("heart_bucket_mode"));
   t.set_bucket_mode(true);
   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(20240608);
   const int                          N = 40000;  // >> one bucket -> forces the split path
   for (int i = 0; i < N; ++i)
   {
      std::string k;
      switch (int(rng() % 3))
      {
         case 0:
            for (int j = 0, len = 1 + int(rng() % 12); j < len; ++j) k.push_back(char(uint8_t(rng())));
            break;
         case 1:
            k = "pre";
            for (int j = 0, len = 1 + int(rng() % 10); j < len; ++j) k.push_back(char('a' + int(rng() % 6)));
            break;
         default:
            k = "commonprefix/";
            for (int j = 0, len = 1 + int(rng() % 6); j < len; ++j) k.push_back(char(uint8_t(rng())));
      }
      std::string v = make_value(rng());
      if (oracle.emplace(k, v).second)  // unique-key insert path (v1 bucket mode: see note)
         t.upsert(k, v);
   }
   for (auto& [k, v] : oracle)
   {
      value_view got;
      REQUIRE(t.find(k, got));
      REQUIRE(std::string(got) == v);
   }
   for (int i = 0; i < 5000; ++i)  // misses
   {
      std::string q(1 + int(rng() % 14), 0);
      for (auto& c : q) c = char(uint8_t(rng()));
      value_view got;
      if (!oracle.count(q))
         REQUIRE_FALSE(t.find(q, got));
   }
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


// Drives ALL ordered surfaces (find/inorder/cursor fwd+rev/lower_bound) against std::map on the
// RADIX tiers, in plain and wide-stem mode, over data that DENSELY creates keys which are
// prefixes of other keys (the "pre"/"predcb…" families) — the case the old ordered test under-
// covered. Same data both modes; the only difference is set_wide_stems, so a mode-specific
// failure is a wide-stem bug.
TEST_CASE("radix_engine: ordered surfaces correct on radix tiers (plain + wide), prefix-dense",
          "[radix_engine][cursor]")
{
   for (bool wide : {false, true})
   {
      radix_tree t(tmp(wide ? "heart_ord_surf_wide" : "heart_ord_surf_plain"));
      t.set_wide_stems(wide);
      std::map<std::string, std::string> oracle;
      std::mt19937_64                    rng(0xBADC0FFEE0DDF00Dull);
      const int                          N = 8000;
      for (int i = 0; i < N; ++i)
      {
         std::string k;
         switch (int(rng() % 3))
         {
            case 0:
               for (int j = 0, len = 1 + int(rng() % 12); j < len; ++j)
                  k.push_back(char(uint8_t(rng())));
               break;
            case 1:  // dense short keys → many are prefixes of longer ones in the same family
               k = "pre";
               for (int j = 0, len = 1 + int(rng() % 8); j < len; ++j) k.push_back(char('a' + int(rng() % 5)));
               break;
            default: k = u64key(hashed(uint64_t(i)));
         }
         std::string v = make_value(rng());
         t.upsert(k, v);
         oracle[k] = v;
      }
      check_ordered_surfaces(t, oracle, wide ? "radix wide" : "radix plain");
   }
}

// Drives the SAME ordered surfaces on a BUCKET tree (set_bucket_mode → dense sorted bucket
// leaves). The old suite never ran ordered ops on a bucket tree at all. ASSERTS buckets exist
// so coverage can't silently lapse.
TEST_CASE("radix_engine: ordered surfaces correct on a bucket tree", "[radix_engine][cursor]")
{
   radix_tree t(tmp("heart_ord_surf_bucket"));
   t.set_bucket_mode(true);
   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(20240608);
   const int                          N = 20000;
   for (int i = 0; i < N; ++i)
   {
      std::string k;
      switch (int(rng() % 3))
      {
         case 0:
            for (int j = 0, len = 1 + int(rng() % 12); j < len; ++j) k.push_back(char(uint8_t(rng())));
            break;
         case 1:
            k = "pre";
            for (int j = 0, len = 1 + int(rng() % 8); j < len; ++j) k.push_back(char('a' + int(rng() % 6)));
            break;
         default: k = "commonprefix/" + std::to_string(int(rng() % 4000));
      }
      std::string v = make_value(rng());
      if (oracle.emplace(k, v).second) t.upsert(k, v);  // v1 bucket mode: unique-insert path
   }
   auto tc = t.count_tiers();
   INFO("buckets = " << tc.bucket << " entries " << tc.bucket_entries);
   REQUIRE(tc.bucket > 0);  // coverage guard: this test MUST exercise buckets
   check_ordered_surfaces(t, oracle, "bucket mode");
}

// First-class subtree values: a value that IS a nested tree, not opaque bytes. Exercises the
// discriminator, the cursor (is_subtree/subtree_cursor), write (create_subtree/upsert_subtree),
// the transitions (bytes<->subtree, remove), and the ref-counted free cascade.
// Transaction / sub-transaction / multi-root model (psitri-shaped). COW-to-root gives commit
// (publish working) / abort (drop working) / savepoint (retain+restore) for free.
TEST_CASE("radix_engine: transactions, savepoints, multi-root", "[radix_engine][txn]")
{
   radix_tree t(tmp("heart_txn"));

   // commit publishes; abort discards.
   {
      auto tx = t.start_transaction(0);
      tx.upsert("a", "1");
      tx.upsert("b", "2");
      tx.commit();
   }
   CHECK(t.get_root(0).get<std::string>("a").value_or("") == "1");
   {
      auto tx = t.start_transaction(0);
      tx.upsert("c", "3");
      tx.abort();  // discard
   }
   CHECK_FALSE(t.get_root(0).get<std::string>("c").has_value());
   CHECK(t.get_root(0).count_keys() == 2);
   {
      auto tx = t.start_transaction(0);  // implicit abort on scope exit (never committed)
      tx.upsert("d", "4");
   }
   CHECK_FALSE(t.get_root(0).get<std::string>("d").has_value());

   // savepoint: abort the frame rolls back to the savepoint; outer commit keeps the rest.
   {
      auto tx = t.start_transaction(0);
      tx.upsert("e", "5");
      {
         auto f = tx.sub_transaction();
         f.upsert("f", "6");
         f.abort();  // roll back "f"
      }
      {
         auto f = tx.sub_transaction();
         f.upsert("g", "7");
         f.commit();  // keep "g"
      }
      tx.commit();
   }
   CHECK(t.get_root(0).get<std::string>("e").value_or("") == "5");
   CHECK_FALSE(t.get_root(0).get<std::string>("f").has_value());  // rolled back
   CHECK(t.get_root(0).get<std::string>("g").value_or("") == "7");

   // multi-root: roots are independent; an untouched root is empty (falsy).
   CHECK_FALSE(bool(t.get_root(5)));
   {
      auto tx = t.start_transaction(5);
      tx.upsert("only-in-5", "x");
      tx.commit();
   }
   CHECK(bool(t.get_root(5)));
   CHECK(t.get_root(5).get<std::string>("only-in-5").value_or("") == "x");
   CHECK_FALSE(t.get_root(0).get<std::string>("only-in-5").has_value());  // isolated from root 0
   CHECK_FALSE(t.get_root(5).get<std::string>("a").has_value());

   // subtree stored via a transaction; read back through the published root.
   {
      auto tx  = t.start_transaction(0);
      auto sub = t.create_subtree();
      sub.upsert("one", "1");
      sub.upsert("two", "2");
      tx.upsert_subtree("kids", std::move(sub));
      tx.commit();
   }
   {
      auto rh = t.get_root(0);
      REQUIRE(rh.is_subtree("kids"));
      CHECK(rh.get_subtree("kids").count_keys() == 2);
      CHECK(rh.get_subtree("kids").get<std::string>("two").value_or("") == "2");
   }

   // abort discards a stored subtree with no leak (psitri's "abort discards subtree").
   const uint64_t before = t.object_count();
   {
      auto tx  = t.start_transaction(2);
      auto sub = t.create_subtree();
      for (int i = 0; i < 200; ++i) sub.upsert("k" + std::to_string(i), "v");
      tx.upsert_subtree("tree", std::move(sub));
      tx.abort();  // must release the whole subtree
   }
   CHECK_FALSE(bool(t.get_root(2)));
   CHECK(t.object_count() == before);  // aborted subtree fully released
}

// Regression gate for the collision-child subtree double-free: many prefix-sharing subtree keys
// force the root through tier transitions / prefix-splits / collisions while it holds subtree
// leaves — the path that re-created an existing subtree terminal from its value bytes without
// retaining the root. Must be build-di clean (no double-free) and every subtree readable.
TEST_CASE("radix_engine: many subtrees across tier transitions (no double-free)",
          "[radix_engine][subtree]")
{
   radix_tree                 t(tmp("heart_subtree_scale"));
   std::map<std::string, int> oracle;
   std::mt19937_64            rng(0x5);
   for (int i = 0; i < 1500; ++i)
   {
      std::string k;
      for (int j = 0, len = 1 + int(rng() % 9); j < len; ++j) k.push_back(char('a' + int(rng() % 4)));
      if (!oracle.emplace(k, i).second) continue;
      auto s = t.create_subtree();
      s.upsert("v", std::to_string(i));
      t.upsert_subtree(k, std::move(s));
   }
   int ok = 0;
   for (auto& [k, i] : oracle)
   {
      auto sh = t.get_subtree(k);
      if (sh && sh.get<std::string>("v").value_or("") == std::to_string(i)) ++ok;
   }
   CHECK(ok == int(oracle.size()));
   CHECK(t.get_root(0).count_keys() == oracle.size());
}

// tree_handle: in-place subtree editing (open_subtree/create_subtree) with bottom-up commit,
// dedup, nesting, and multi-root — psitri's tree_handle model.
TEST_CASE("radix_engine: tree_handle open/create subtree, nested, multi-root",
          "[radix_engine][tree_handle]")
{
   radix_tree t(tmp("heart_tree_handle"));
   {
      auto  tx = t.start_transaction(0);
      auto& s  = tx.create_subtree("child");
      s.upsert("a", "1");
      s.upsert("b", "2");
      tx.upsert("data", "root");
      tx.commit();
   }
   {
      auto  tx = t.start_transaction(0);
      auto& s  = tx.open_subtree("child");
      CHECK(s.get<std::string>("a").value_or("") == "1");
      s.upsert("c", "3");
      s.remove("a");
      tx.commit();
   }
   {
      auto sh = t.get_root(0).get_subtree("child");
      REQUIRE(bool(sh));
      CHECK(sh.count_keys() == 2);  // b, c
      CHECK(sh.get<std::string>("c").value_or("") == "3");
      CHECK_FALSE(sh.get<std::string>("a").has_value());
      CHECK(t.get_root(0).get<std::string>("data").value_or("") == "root");
   }
   {  // dedup: same key twice → same handle
      auto  tx = t.start_transaction(0);
      auto& s1 = tx.open_subtree("child");
      s1.upsert("x", "mod");
      auto& s2 = tx.open_subtree("child");
      CHECK(s2.get<std::string>("x").value_or("") == "mod");
      tx.commit();
   }
   CHECK(t.get_root(0).get_subtree("child").get<std::string>("x").value_or("") == "mod");
   {  // nested: create inner within a created outer (bottom-up write-back)
      auto  tx    = t.start_transaction(0);
      auto& outer = tx.create_subtree("outer");
      outer.upsert("od", "hi");
      auto& inner = outer.create_subtree("inner");
      inner.upsert("deep", "v");
      tx.commit();
   }
   {
      auto outer = t.get_root(0).get_subtree("outer");
      REQUIRE(bool(outer));
      CHECK(outer.get<std::string>("od").value_or("") == "hi");
      auto inner = outer.get_subtree("inner");
      REQUIRE(bool(inner));
      CHECK(inner.get<std::string>("deep").value_or("") == "v");
   }
   {  // multiple subtrees in one txn
      auto  tx = t.start_transaction(0);
      auto& s1 = tx.create_subtree("m1");
      s1.upsert("p", "10");
      auto& s2 = tx.create_subtree("m2");
      s2.upsert("q", "20");
      tx.commit();
   }
   CHECK(t.get_root(0).get_subtree("m1").get<std::string>("p").value_or("") == "10");
   CHECK(t.get_root(0).get_subtree("m2").get<std::string>("q").value_or("") == "20");
   {  // multi-root: subtree in additional root 5, isolated from root 0
      auto  tx = t.start_transaction(0);
      tx.upsert("in0", "x");
      auto& r5 = tx.open_root(5);
      r5.upsert("top", "5data");
      auto& nested = r5.create_subtree("nested");
      nested.upsert("ik", "iv");
      tx.commit();
   }
   CHECK(t.get_root(5).get<std::string>("top").value_or("") == "5data");
   CHECK(t.get_root(5).get_subtree("nested").get<std::string>("ik").value_or("") == "iv");
   CHECK(t.get_root(0).get<std::string>("in0").value_or("") == "x");
   CHECK_FALSE(t.get_root(5).get<std::string>("in0").has_value());
}

TEST_CASE("radix_engine: first-class subtree values", "[radix_engine][cursor][subtree]")
{
   radix_tree t(tmp("heart_subtree"));
   std::map<std::string, std::string> parent = {{"alpha", "A"}, {"beta", "B"}, {"zeta", "Z"}};
   for (auto& [k, v] : parent) t.upsert(k, v);

   const std::map<std::string, std::string> subkeys = {{"one", "1"}, {"three", "3"}, {"two", "2"}};
   {
      auto sub = t.create_subtree();
      for (auto& [k, v] : subkeys) sub.upsert(k, v);
      t.upsert_subtree("mid", std::move(sub));  // attach (consumes the handle)
   }

   // (1) discrimination: "mid" is a subtree, not a byte value; byte keys are not subtrees.
   value_view vv;
   CHECK(t.find("alpha", vv));
   CHECK(std::string(vv) == "A");
   CHECK_FALSE(t.find("mid", vv));  // a subtree is not a byte value
   CHECK(t.is_subtree("mid"));
   CHECK_FALSE(t.is_subtree("alpha"));
   {
      auto sh = t.get_subtree("mid");  // a handle, not a ptr_address
      REQUIRE(bool(sh));
      CHECK(sh.count_keys() == 3);
      CHECK(sh.get<std::string>("two").value_or("") == "2");
   }

   // (2) parent in-order walk: "mid" shows as a subtree, byte keys as bytes, all in key order.
   {
      radix_tree::cursor       c(&t);
      std::vector<std::string> order;
      for (bool e = c.seek_begin(); !e; e = c.next())
      {
         order.push_back(std::string(c.key()));
         if (c.key() == "mid")
         {
            CHECK(c.is_subtree());
            CHECK(c.value_size() == radix_tree::cursor::value_subtree);
         }
         else
            CHECK_FALSE(c.is_subtree());
      }
      CHECK(order == std::vector<std::string>{"alpha", "beta", "mid", "zeta"});
   }

   // (3) descend into the subtree via subtree_cursor and iterate it == subkeys.
   {
      radix_tree::cursor c(&t);
      REQUIRE_FALSE(c.find("mid"));  // find returns is_end (false = found)
      REQUIRE(c.is_subtree());
      auto                               sc = c.subtree_cursor();
      std::map<std::string, std::string> got;
      for (bool e = sc.seek_begin(); !e; e = sc.next())
      {
         std::string val;
         sc.get_value([&](value_view v) { val.assign(v.data(), v.size()); });
         got[std::string(sc.key())] = val;
      }
      CHECK(got == subkeys);
   }

   // (4) overwrite the subtree with a byte value → old subtree must be freed (cascade).
   const uint64_t before = t.object_count();
   t.upsert("mid", "now-bytes");
   CHECK_FALSE(t.is_subtree("mid"));
   CHECK(t.find("mid", vv));
   CHECK(std::string(vv) == "now-bytes");
   CHECK(t.object_count() < before);  // subtree root + its leaves released

   // (5) promote a byte key to a subtree, then remove it → frees the subtree too.
   {
      auto sub2 = t.create_subtree();
      sub2.upsert("x", "9");
      t.upsert_subtree("alpha", std::move(sub2));  // alpha was a byte value
   }
   CHECK(t.is_subtree("alpha"));
   const uint64_t before2 = t.object_count();
   CHECK(t.remove("alpha"));
   CHECK_FALSE(t.is_subtree("alpha"));
   CHECK_FALSE(t.find("alpha", vv));
   CHECK(t.object_count() < before2);  // removing the key freed the subtree
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

// ───────────────────────────── remove ─────────────────────────────
#include <algorithm>
#include <chrono>
#include <thread>

namespace
{
   // Releases of a sole-referenced object are deferred to a background thread; poll until
   // the queue is empty AND the live count is stable, so an exact leak assertion is sound.
   void drain(radix_tree& t)
   {
      for (int i = 0; i < 4000 && t.pending_releases() > 0; ++i)
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
      uint64_t prev = t.object_count();
      for (int i = 0; i < 100; ++i)
      {
         std::this_thread::sleep_for(std::chrono::milliseconds(2));
         uint64_t cur = t.object_count();
         if (cur == prev && t.pending_releases() == 0)
            return;
         prev = cur;
      }
   }
}

TEST_CASE("radix_engine: remove basic — present, absent, re-find", "[radix_engine][remove]")
{
   radix_tree               t(tmp("heart_rm_basic"));
   std::vector<std::string> keys = {"a",   "ab",  "abc", "abd",     "b",  "ba",
                                    "hello", "help", "helping", "",  "x", "xyz"};
   for (size_t i = 0; i < keys.size(); ++i)
      t.upsert(keys[i], make_value(i + 1));

   REQUIRE_FALSE(t.remove("nope"));   // absent
   REQUIRE_FALSE(t.remove("he"));     // a proper prefix of present keys, but not itself a key
   REQUIRE_FALSE(t.remove("abce"));   // diverges past a present key

   REQUIRE(t.remove("abc"));          // present
   value_view g;
   REQUIRE_FALSE(t.find("abc", g));   // gone
   REQUIRE(t.find("abd", g));         // sibling intact
   REQUIRE(t.find("ab", g));          // ancestor intact
   REQUIRE_FALSE(t.remove("abc"));    // second remove → absent

   REQUIRE(t.remove(""));             // the empty key (a root term)
   REQUIRE_FALSE(t.find("", g));
   REQUIRE(t.find("a", g));

   t.upsert("abc", make_value(999));  // re-insert after remove
   REQUIRE(t.find("abc", g));
   REQUIRE(std::string(g) == make_value(999));
}

TEST_CASE("radix_engine: remove all drains every tier with no leak", "[radix_engine][remove]")
{
   // Forced fan-out pushes a node through setlist → c1/bitset_r2/r4 → full; removing all
   // must drain them back to an empty root with the live object count at baseline.
   for (int K : {30, 120, 256})
   {
      radix_tree     t(tmp("heart_rm_tiers"));
      const uint64_t base = t.object_count();
      auto           mk   = [](int i) {
         std::string k = "AAA";
         k.push_back(char(uint8_t(i)));
         k += "tail";
         return k;
      };
      for (int i = 0; i < K; ++i)
         t.upsert(mk(i), make_value(uint64_t(i)));
      auto tc = t.count_tiers();
      REQUIRE((tc.bitset_c1 + tc.bitset_r2 + tc.bitset_r4 + tc.full) >= 1);  // widened

      for (int i = 0; i < K; ++i)
         REQUIRE(t.remove(mk(i)));
      value_view g;
      for (int i = 0; i < K; ++i)
         REQUIRE_FALSE(t.find(mk(i), g));
      drain(t);
      REQUIRE(t.object_count() == base);  // every node freed, no leak
   }
}

TEST_CASE("radix_engine: remove downgrades a drained wide node to a smaller tier",
          "[radix_engine][remove]")
{
   // A 256-way fan-out node widens to a big tier (r4/full); removing it down to a handful of
   // entries must shrink it back (hysteresis: ~25% below the regrow point), so a sparse node
   // doesn't keep its big-tier cacheline footprint. Verify via the tier census + correctness.
   radix_tree t(tmp("heart_rm_downgrade"));
   auto       mk = [](int i) {
      std::string k = "AAA";
      k.push_back(char(uint8_t(i)));
      k += "t";
      return k;
   };
   const int K = 256;
   for (int i = 0; i < K; ++i)
      t.upsert(mk(i), make_value(uint64_t(i)));
   auto wide = t.count_tiers();
   REQUIRE((wide.bitset_r4 + wide.full) >= 1);  // widened to a multi-CL tier

   for (int i = 0; i < K - 8; ++i)  // drain to 8 entries (well below every downgrade floor)
      REQUIRE(t.remove(mk(i)));
   auto narrow = t.count_tiers();
   REQUIRE(narrow.full == 0);                              // the full/r4 node is gone…
   REQUIRE(narrow.bitset_r4 < wide.bitset_r4 + 1);         // …shrunk
   REQUIRE((narrow.setlist + narrow.bitset_c1 + narrow.bitset_r2) >= 1);  // …into a smaller tier

   value_view g;  // surviving keys intact, removed keys gone
   for (int i = K - 8; i < K; ++i)
   {
      REQUIRE(t.find(mk(i), g));
      REQUIRE(std::string(g) == make_value(uint64_t(i)));
   }
   for (int i = 0; i < K - 8; ++i)
      REQUIRE_FALSE(t.find(mk(i), g));
}

TEST_CASE("radix_engine: remove on narrow c1 windows (promote on) — no leak",
          "[radix_engine][remove]")
{
   radix_tree t(tmp("heart_rm_c1"));
   t.set_c1_promote(true);
   const uint64_t                    base = t.object_count();
   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(7);
   const int                          N = 8000;
   for (int i = 0; i < N; ++i)
   {
      std::string k(4, 0);
      for (int j = 0; j < 4; ++j)
         k[j] = char(40 + int(rng() % 18));  // narrow byte window → c1 nodes
      std::string v = make_value(uint64_t(i));
      t.upsert(k, v);
      oracle[k] = v;
   }
   REQUIRE(t.count_tiers().bitset_c1 >= 1);  // c1 tier actually exercised
   value_view g;
   for (auto& [k, v] : oracle)
   {
      REQUIRE(t.find(k, g));
      REQUIRE(std::string(g) == v);
   }
   for (auto& [k, v] : oracle)
      REQUIRE(t.remove(k));
   for (auto& [k, v] : oracle)
      REQUIRE_FALSE(t.find(k, g));
   drain(t);
   REQUIRE(t.object_count() == base);
}

TEST_CASE("radix_engine: remove fuzz vs std::map, exact no-leak", "[radix_engine][remove]")
{
   radix_tree                         t(tmp("heart_rm_fuzz"));
   const uint64_t                     base = t.object_count();
   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(0xC0FFEE);
   const int                          OPS = 80000;
   for (int op = 0; op < OPS; ++op)
   {
      std::string k = u64key(hashed(rng() % 6000));  // a bounded key space → frequent hits
      if ((rng() & 3) == 0)                          // ~25% removes
      {
         bool exp = oracle.erase(k) > 0;
         REQUIRE(t.remove(k) == exp);
      }
      else
      {
         std::string v = make_value(rng());
         t.upsert(k, v);
         oracle[k] = v;
      }
      if ((op & 4095) == 0)  // periodic spot check of a handful of live keys
      {
         value_view g;
         int        n = 0;
         for (auto it = oracle.begin(); it != oracle.end() && n < 64; ++it, ++n)
         {
            REQUIRE(t.find(it->first, g));
            REQUIRE(std::string(g) == it->second);
         }
      }
   }
   // full cross-check, then remove everything → exact baseline (no reference leak)
   value_view g;
   for (auto& [k, v] : oracle)
   {
      REQUIRE(t.find(k, g));
      REQUIRE(std::string(g) == v);
   }
   for (auto& [k, v] : oracle)
      REQUIRE(t.remove(k));
   for (auto& [k, v] : oracle)
      REQUIRE_FALSE(t.find(k, g));
   drain(t);
   REQUIRE(t.object_count() == base);
}

TEST_CASE("radix_engine: remove under MVCC snapshots stays leak-bounded",
          "[radix_engine][remove]")
{
   radix_tree     t(tmp("heart_rm_mvcc"));
   const uint64_t base = t.object_count();
   const int      N    = 6000;
   for (int i = 0; i < N; ++i)
      t.upsert(u64key(hashed(uint64_t(i))), make_value(uint64_t(i)));
   t.commit();  // snapshot pins the whole tree (ref>1 → removes COW the path)

   for (int i = 0; i < N; ++i)  // remove all from the live root (shared-mode COW path)
      REQUIRE(t.remove(u64key(hashed(uint64_t(i)))));
   value_view g;
   for (int i = 0; i < N; ++i)
      REQUIRE_FALSE(t.find(u64key(hashed(uint64_t(i))), g));

   t.commit();  // drop the old snapshot → the original (full) tree is freed
   drain(t);
   REQUIRE(t.object_count() < base + 8);  // back to ~empty: no per-version leak
}

TEST_CASE("radix_engine: bucket-mode remove vs std::map", "[radix_engine][remove]")
{
   radix_tree t(tmp("heart_rm_bucket"));
   t.set_bucket_mode(true);
   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(123);
   const int                          N = 5000;
   for (int i = 0; i < N; ++i)
   {
      std::string k = u64key(hashed(uint64_t(i)));
      std::string v = make_value(uint64_t(i));
      t.upsert(k, v);
      oracle[k] = v;
   }
   // remove a random half, verify against the oracle
   std::vector<std::string> keys;
   for (auto& [k, v] : oracle)
      keys.push_back(k);
   std::shuffle(keys.begin(), keys.end(), rng);
   for (size_t i = 0; i < keys.size() / 2; ++i)
   {
      REQUIRE(t.remove(keys[i]));
      oracle.erase(keys[i]);
   }
   REQUIRE_FALSE(t.remove(keys[0]));  // already removed
   value_view g;
   for (auto& [k, v] : oracle)
   {
      REQUIRE(t.find(k, g));
      REQUIRE(std::string(g) == v);
   }
   for (size_t i = 0; i < keys.size() / 2; ++i)
      REQUIRE_FALSE(t.find(keys[i], g));
}

// BUCKET_SPEC TDD: the composed bucket's ops driven DIRECTLY against an
// oracle — cells/pages/SP lines, duplicate updates, shrink/grow, page
// lifecycle, multi-line equal-stem runs. Single txn (no sharing) first;
// sharing covered by the kv battery once PART_CBUCKET=1.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <map>
#include <random>
#include <string>

#include <part/kv.hpp>

using namespace part;
namespace fs = std::filesystem;
using tree_t = part::kv::engine_t;
using pval   = part::kv::pval;

namespace
{
   struct rig
   {
      arena                a;
      std::unique_ptr<session> ses;
      std::unique_ptr<part::transaction> tx;
      tree_t               t;
      rig()
          : a([] {
               arena_config c;
               c.dir = fs::temp_directory_path() / "part_cbu";
               fs::remove_all(c.dir);
               return c;
            }()),
            ses(new session(a.start_session())),
            tx(new part::transaction(ses->start_transaction(0))),
            t(part_alloc<pval>(ses.get()))
      {
      }
   };

   std::string val_of(uint64_t seed, size_t len)
   {
      std::string v(len, 0);
      for (size_t j = 0; j < len; ++j)
         v[j] = char(uint8_t(seed * 131 + j * 7 + 1));
      return v;
   }
}  // namespace

TEST_CASE("cbucket: oracle fuzz — insert/update/remove/find until split")
{
   rig r;
   auto root = r.t.cb_make2_("", "aa", "v1", "bb", "v2");
   auto* h   = static_cast<tree_t::cb_hdr*>(r.a.get(uint32_t(root.raw() >> 4)));

   std::map<std::string, std::string> o{{"aa", "v1"}, {"bb", "v2"}};
   std::mt19937_64                    rng(42);

   auto mkkey = [&](int style) -> std::string {
      switch (style % 5)
      {
         case 0:  // same-stem family (forces multi-line runs after rebalance)
            return "st" + std::to_string(rng() % 64);
         case 1:  // short
            return std::string(1, char('a' + rng() % 26));
         case 2:  // 2-byte exact stems
            return std::string{char('a' + rng() % 8), char('a' + rng() % 8)};
         case 3:  // longer suffix
            return "key/" + std::to_string(rng() % 256);
         default:  // binary-ish
         {
            std::string k(3 + rng() % 6, 0);
            for (auto& c : k)
               c = char(uint8_t(rng()));
            if (!k.empty() && k[0] == 0)
               k[0] = 1;
            return k;
         }
      }
   };

   int split_signals = 0;
   for (int step = 0; step < 20000 && split_signals == 0; ++step)
   {
      const int op = int(rng() % 100);
      if (op < 55 || o.empty())
      {  // insert or update with varying value length (shrink/grow/same)
         std::string k = mkkey(int(rng()));
         if (k.empty())
            continue;
         std::string v = val_of(rng(), 1 + rng() % 50);
         const int   rc = r.t.cb_insert_(h, k, v);
         if (rc < 0)
         {
            ++split_signals;  // capacity reached: stop fuzz, verify, done
            break;
         }
         const bool was = o.count(k) != 0;
         REQUIRE(rc == (was ? 0 : 1));
         o[k] = v;
      }
      else if (op < 80 && !o.empty())
      {  // update an EXISTING key (the duplicate-entry bug shape)
         auto it = o.begin();
         std::advance(it, rng() % o.size());
         std::string v  = val_of(rng(), 1 + rng() % 50);
         const int   rc = r.t.cb_insert_(h, it->first, v);
         if (rc < 0)
         {
            ++split_signals;
            break;
         }
         REQUIRE(rc == 0);  // never a fresh insert for an existing key
         it->second = v;
      }
      else
      {  // remove (existing or missing)
         std::string k = (rng() % 2 && !o.empty())
                             ? std::next(o.begin(), rng() % o.size())->first
                             : mkkey(int(rng()));
         const bool  was = o.count(k) != 0;
         REQUIRE(r.t.cb_remove_(h, k) == was);
         o.erase(k);
      }
      // full point-verify every 64 steps; spot-verify otherwise
      if (step % 64 == 0)
      {
         REQUIRE(h->rh.hdr.nbranch == o.size());
         for (auto& [k, v] : o)
         {
            const uint8_t* c = r.t.cb_find_(h, k);
            REQUIRE(c != nullptr);
            REQUIRE(tree_t::cb_cell{const_cast<uint8_t*>(c)}.value() == v);
         }
         // enumeration matches exactly (no duplicates, nothing lost)
         std::map<std::string, std::string> seen;
         r.t.cb_for_each_(h, [&](std::string_view s, std::string_view v)
                          { REQUIRE(seen.emplace(std::string(s), std::string(v)).second); });
         REQUIRE(seen == o);
      }
   }
   // final exhaustive verify
   REQUIRE(h->rh.hdr.nbranch == o.size());
   for (auto& [k, v] : o)
   {
      const uint8_t* c = r.t.cb_find_(h, k);
      REQUIRE(c != nullptr);
      REQUIRE(tree_t::cb_cell{const_cast<uint8_t*>(c)}.value() == v);
   }
}

TEST_CASE("cbucket: equal-stem run spanning lines — update never duplicates")
{
   rig r;
   // 20+ keys sharing the SAME 2-byte stem ("zz...") force rebalance to split
   // the run across SP lines; then update every key twice and verify counts.
   auto root = r.t.cb_make2_("", "zz000", "a", "zz001", "b");
   auto* h   = static_cast<tree_t::cb_hdr*>(r.a.get(uint32_t(root.raw() >> 4)));
   std::map<std::string, std::string> o{{"zz000", "a"}, {"zz001", "b"}};
   for (int i = 2; i < 40; ++i)
   {
      std::string k = "zz" + std::string(1, char('0' + i / 10)) + std::to_string(i % 10) +
                      std::to_string(i);
      REQUIRE(r.t.cb_insert_(h, k, "v") == 1);
      o[k] = "v";
   }
   for (int round = 0; round < 2; ++round)
      for (auto& [k, v] : o)
      {
         std::string nv = val_of(uint64_t(round * 1315423911u + k.size()), 1 + (round * 17 + k.size()) % 45);
         REQUIRE(r.t.cb_insert_(h, k, nv) == 0);
         v = nv;
      }
   REQUIRE(h->rh.hdr.nbranch == o.size());
   std::map<std::string, std::string> seen;
   r.t.cb_for_each_(h, [&](std::string_view s, std::string_view v)
                    { REQUIRE(seen.emplace(std::string(s), std::string(v)).second); });
   REQUIRE(seen == o);
}

TEST_CASE("cbucket: page lifecycle — dead pages release, slots reused")
{
   rig r;
   auto root = r.t.cb_make2_("", "k00", val_of(1, 40), "k01", val_of(2, 40));
   auto* h   = static_cast<tree_t::cb_hdr*>(r.a.get(uint32_t(root.raw() >> 4)));
   // ~5 cells of 48B fill a 248B page: insert 12, then delete the first 10 —
   // their pages must fully die and their table slots null out.
   for (int i = 2; i < 12; ++i)
      REQUIRE(r.t.cb_insert_(h, "k" + std::to_string(10 + i), val_of(uint64_t(i), 40)) == 1);
   unsigned pages_live_before = 0;
   for (unsigned i = 0; i < tree_t::CB_PAGES; ++i)
      pages_live_before += !h->page_line[i].is_null();
   REQUIRE(pages_live_before >= 2);
   for (int i = 2; i < 12; ++i)
      REQUIRE(r.t.cb_remove_(h, "k" + std::to_string(10 + i)));
   REQUIRE(r.t.cb_remove_(h, "k00"));
   REQUIRE(r.t.cb_remove_(h, "k01"));
   unsigned pages_live_after = 0;
   for (unsigned i = 0; i < tree_t::CB_PAGES; ++i)
      pages_live_after += !h->page_line[i].is_null();
   // the open page may linger; everything else must have been released
   REQUIRE(pages_live_after <= 1);
   REQUIRE(h->rh.hdr.nbranch == 0);
}

TEST_CASE("cbucket: shrink in place keeps capacity; regrow within it; grow past it")
{
   rig r;
   auto root = r.t.cb_make2_("", "kk", val_of(7, 40), "qq", "x");
   auto* h   = static_cast<tree_t::cb_hdr*>(r.a.get(uint32_t(root.raw() >> 4)));
   auto cell = [&] { return tree_t::cb_cell{const_cast<uint8_t*>(r.t.cb_find_(h, "kk"))}; };
   const uint8_t* before = cell().p;
   REQUIRE(r.t.cb_insert_(h, "kk", val_of(8, 10)) == 0);  // shrink: in place
   REQUIRE(cell().p == before);
   REQUIRE(cell().value() == val_of(8, 10));
   REQUIRE(tree_t::cb_cell_cap_(cell().p) == 40);          // capacity preserved
   REQUIRE(r.t.cb_insert_(h, "kk", val_of(9, 40)) == 0);   // regrow to capacity: in place
   REQUIRE(cell().p == before);
   REQUIRE(r.t.cb_insert_(h, "kk", val_of(10, 60)) == 0);  // past capacity: relocates
   REQUIRE(cell().value() == val_of(10, 60));
}

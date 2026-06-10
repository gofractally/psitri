// part::kv end-to-end vs oracle, mirroring the kv-compat battery's phases
// (batched commits — which retain the live root and force COW/demotion on the
// next batch — then updates, removes, fuzz). Stops at first divergence.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <part/kv.hpp>

namespace fs = std::filesystem;

namespace
{
   uint64_t    hashed(uint64_t i) { return (i * 0x9E3779B97F4A7C15ull) ^ (i >> 29); }
   std::string u64key(uint64_t x)
   {
      return std::string(reinterpret_cast<const char*>(&x), sizeof(x));
   }
   std::string mkval(uint64_t seed)
   {
      size_t      len = 1 + (seed % 40);
      std::string v(len, 0);
      for (size_t j = 0; j < len; ++j)
         v[j] = char(uint8_t(seed * 31 + j));
      return v;
   }
}  // namespace

namespace
{
   // Every line the engine ever allocated must come back once the sessions and
   // database are gone — the leak detector for the whole COW machinery.
   void require_drained(part::arena& a)
   {
      uint64_t free_total = 0;
      for (uint32_t s = 0; s < a.num_segments(); ++s)
         free_total += a.freeset_of(s).free_lines();
      REQUIRE(free_total == uint64_t(a.num_segments()) * part::lines_per_segment);
   }
}  // namespace

TEST_CASE("kv battery: batched commits, updates, removes vs oracle")
{
   auto dir = fs::temp_directory_path() / "part_kv_battery";
   fs::remove_all(dir);
   auto db  = part::kv::database::create(dir);
   auto ses = db->start_write_session();

   std::map<std::string, std::string> oracle;
   const int                          N = 20000, B = 2048;

   std::vector<std::pair<std::string, std::string>> kv;
   for (int i = 0; i < N; ++i)
   {
      std::string k = (i & 1) ? u64key(hashed(uint64_t(i)))
                              : ("str/" + std::to_string(hashed(uint64_t(i)) % 100000));
      kv.emplace_back(k, mkval(uint64_t(i)));
      oracle[k] = kv.back().second;
   }

   auto upsert_all = [&](auto& pairs)
   {
      for (size_t base = 0; base < pairs.size(); base += B)
      {
         auto   tx  = ses->start_transaction(0);
         size_t end = std::min(pairs.size(), base + size_t(B));
         for (size_t i = base; i < end; ++i)
            tx.upsert(pairs[i].first, pairs[i].second);
         tx.commit();
      }
   };
   auto verify_all = [&](const char* phase)
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
      {
         auto got = tx.template get<std::string>(k);
         INFO(phase << " key len=" << k.size());
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      tx.abort();
   };

   upsert_all(kv);
   verify_all("phase1");

   // Phase 2: update every key (forces COW through shared roots).
   for (auto& [k, v] : oracle)
      v = mkval(hashed(uint64_t(v.size() + 1)) + 13);
   std::vector<std::pair<std::string, std::string>> upd(oracle.begin(), oracle.end());
   upsert_all(upd);
   verify_all("phase2");

   // Phase 3: remove half, batched.
   std::vector<std::string> keys;
   for (auto& [k, v] : oracle)
      keys.push_back(k);
   const size_t half = keys.size() / 2;
   for (size_t base = 0; base < half; base += B)
   {
      auto   tx  = ses->start_transaction(0);
      size_t end = std::min(half, base + size_t(B));
      for (size_t i = base; i < end; ++i)
         tx.remove(keys[i]);
      tx.commit();
   }
   for (size_t i = 0; i < half; ++i)
      oracle.erase(keys[i]);
   verify_all("phase3-present");
   {
      auto tx = ses->start_transaction(0);
      for (size_t i = 0; i < half; ++i)
         REQUIRE_FALSE(tx.template get<std::string>(keys[i]).has_value());
      tx.abort();
   }

   // Phase 4: full forward scan matches the oracle.
   {
      auto c   = ses->get_root(0).snapshot_cursor();
      auto oit = oracle.begin();
      c.seek_begin();
      while (!c.is_end())
      {
         REQUIRE(oit != oracle.end());
         REQUIRE(std::string(c.key()) == oit->first);
         REQUIRE(*c.value<std::string>() == oit->second);
         c.next();
         ++oit;
      }
      REQUIRE(oit == oracle.end());
   }

   // Phase 5: remove EVERYTHING, then the arena must drain to empty once the
   // session and database release their roots.
   for (size_t base = half; base < keys.size(); base += B)
   {
      auto   tx  = ses->start_transaction(0);
      size_t end = std::min(keys.size(), base + size_t(B));
      for (size_t i = base; i < end; ++i)
         tx.remove(keys[i]);
      tx.commit();
   }
   // After remove-all the live root is null: every generation has drained
   // and the arena must be empty even while the database stays open.
   require_drained(db->store());
}

TEST_CASE("kv churn: repeated overwrites under generation turnover drain fully")
{
   auto dir = fs::temp_directory_path() / "part_kv_churn";
   fs::remove_all(dir);
   auto db  = part::kv::database::create(dir);
   auto ses = db->start_write_session();

   // Bank-shaped churn: a fixed key population, every txn overwrites a few
   // keys and commits (slot retain forces COW + demotion next txn).
   std::mt19937_64          rng(7);
   std::vector<std::string> keys;
   for (int i = 0; i < 4000; ++i)
      keys.push_back("acct/" + std::to_string(i));
   {
      auto tx = ses->start_transaction(0);
      for (auto& k : keys)
         tx.upsert(k, "00000000");
      tx.commit();
   }
   for (int round = 0; round < 3000; ++round)
   {
      auto tx = ses->start_transaction(0);
      for (int j = 0; j < 3; ++j)
      {
         auto& k = keys[rng() % keys.size()];
         char  v[9];
         snprintf(v, sizeof v, "%08x", uint32_t(rng()));
         tx.upsert(k, std::string_view(v, 8));
      }
      tx.commit();
   }
   // The live tree should be the ONLY generation: live bytes must be sane
   // (population is fixed; churn must not accumulate).
   const auto& st   = db->store().stats();
   const auto  live = st.bytes_allocated.load() - st.bytes_freed.load();
   INFO("live bytes after churn: " << live);
   REQUIRE(live < 16u * 1024 * 1024);  // 4k keys * ~0.5KB tree ≈ 2MB; 16MB = generous

   {  // remove everything: the arena must drain with the database still open
      auto tx = ses->start_transaction(0);
      for (auto& k : keys)
         tx.remove(k);
      tx.commit();
   }
   require_drained(db->store());
}

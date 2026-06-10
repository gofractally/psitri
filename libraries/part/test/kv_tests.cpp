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
}

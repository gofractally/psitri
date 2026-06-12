// tiered_kv differential tests: vpart hot delta over heart, vs std::map.
// Covers: envelope round-trip, tombstone masking (point + both scan
// directions), flush_now idempotence, the retire pump (keys rewritten after
// the floor SURVIVE), both views, and kv_engine conformance (cursors incl.).
#include <catch2/catch_test_macros.hpp>

#include <tiered/tiered_kv.hpp>

#include <heart_dwal_engine.hpp>  // heart::radix::dwal_engine (concept-checked)
#include <vpart/kv.hpp>

#include <filesystem>
#include <map>
#include <random>
#include <string>

namespace fs = std::filesystem;

using TK = tiered::tiered_kv<vpart::kv::engine, heart::radix::dwal_engine>;
using TE = tiered::engine<vpart::kv::engine, heart::radix::dwal_engine>;
static_assert(psitri::dwal::kv_engine<TE>,
              "the COMPOSITION must satisfy the engine concept, cursors included (Dan)");

namespace
{
   std::shared_ptr<TK::database> fresh(const char* tag, uint64_t retire_batch = 4096)
   {
      auto dir = fs::temp_directory_path() / (std::string("tiered_") + tag);
      fs::remove_all(dir);
      TK::config cfg;
      cfg.knobs.background   = false;  // tests drive flush_now() deterministically
      cfg.knobs.retire_batch = retire_batch;
      return TK::database::open(dir, sal::open_mode::create_or_open, cfg);
   }
   std::string key16(std::mt19937_64& rng)
   {
      char     b[16];
      uint64_t h = rng();
      for (int j = 0; j < 16; ++j, h >>= 4)
         b[j] = "0123456789abcdef"[h & 15];
      return std::string(b, 16);
   }
}  // namespace

TEST_CASE("tiered: write/read through the delta, flush to base, both views agree")
{
   auto db = fresh("basic");
   auto ws = db->start_write_session();

   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(7);
   for (int batch = 0; batch < 10; ++batch)
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 500; ++i)
      {
         auto k = key16(rng);
         auto v = "val:" + k + ":" + std::to_string(batch);
         tx.upsert(k, v);
         oracle[k] = v;
      }
      tx.commit();
   }

   auto rs = db->start_read_session();
   SECTION("latest view sees everything pre-flush; stable sees nothing")
   {
      auto lat = rs->create_cursor(0);
      for (auto& [k, v] : oracle)
         REQUIRE(lat.get<std::string>(k).value() == v);
      auto stb = rs->create_cursor(1);
      REQUIRE_FALSE(stb.seek_begin());
   }
   SECTION("flush: stable converges; latest unchanged; re-flush is idempotent")
   {
      REQUIRE(db->flush_now() == oracle.size());
      auto stb = rs->create_cursor(1);
      uint64_t n = 0;
      for (bool ok = stb.seek_begin(); ok; ok = stb.next(), void())
      {
         REQUIRE(oracle.at(std::string(stb.key())) == stb.value<std::string>().value());
         ++n;
      }
      REQUIRE(n == oracle.size());
      REQUIRE(db->flush_now() == oracle.size());  // idempotent re-apply
      auto lat = rs->create_cursor(0);
      for (auto& [k, v] : oracle)
         REQUIRE(lat.get<std::string>(k).value() == v);
   }
   SECTION("merge cursor: forward and backward scans match the oracle")
   {
      REQUIRE(db->flush_now() == oracle.size());
      // overwrite a third in the delta (post-flush versions shadow the base)
      auto tx = ws->start_transaction(0);
      int  j  = 0;
      for (auto& [k, v] : oracle)
         if (++j % 3 == 0)
         {
            v = v + ":new";
            tx.upsert(k, v);
         }
      tx.commit();
      auto lat = rs->create_cursor(0);
      auto it  = oracle.begin();
      for (bool ok = lat.seek_begin(); ok; ok = lat.next(), ++it)
      {
         REQUIRE(it != oracle.end());
         REQUIRE(lat.key() == it->first);
         REQUIRE(lat.value<std::string>().value() == it->second);
      }
      REQUIRE(it == oracle.end());
      auto rit = oracle.rbegin();
      for (bool ok = lat.seek_last(); ok; ok = lat.prev(), ++rit)
      {
         REQUIRE(rit != oracle.rend());
         REQUIRE(lat.key() == rit->first);
         REQUIRE(lat.value<std::string>().value() == rit->second);
      }
      REQUIRE(rit == oracle.rend());
      REQUIRE(lat.is_rend());
   }
}

TEST_CASE("tiered: tombstones mask the base in point reads and BOTH scan directions")
{
   auto db = fresh("tomb");
   auto ws = db->start_write_session();
   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(11);
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 2000; ++i)
      {
         auto k    = key16(rng);
         oracle[k] = "v" + k;
         tx.upsert(k, oracle[k]);
      }
      tx.commit();
   }
   REQUIRE(db->flush_now() == oracle.size());
   {  // delete every 4th key — tombstones land in the (now mostly empty) delta
      auto tx = ws->start_transaction(0);
      int  j  = 0;
      for (auto it = oracle.begin(); it != oracle.end();)
         if (++j % 4 == 0)
         {
            tx.remove(it->first);
            it = oracle.erase(it);
         }
         else
            ++it;
      tx.commit();
   }
   auto rs  = db->start_read_session();
   auto lat = rs->create_cursor(0);
   auto it  = oracle.begin();
   for (bool ok = lat.seek_begin(); ok; ok = lat.next(), ++it)
   {
      REQUIRE(it != oracle.end());
      REQUIRE(lat.key() == it->first);
   }
   REQUIRE(it == oracle.end());
   auto rit = oracle.rbegin();
   for (bool ok = lat.seek_last(); ok; ok = lat.prev(), ++rit)
      REQUIRE(lat.key() == rit->first);
   REQUIRE(rit == oracle.rend());
   // flush applies the deletes to the base and the tombstones never reach B
   db->flush_now();
   auto stb = rs->create_cursor(1);
   uint64_t n = 0;
   for (bool ok = stb.seek_begin(); ok; ok = stb.next(), void())
   {
      REQUIRE(oracle.count(std::string(stb.key())) == 1);
      ++n;
   }
   REQUIRE(n == oracle.size());
}

TEST_CASE("tiered: retire pump empties the flushed band; post-floor rewrites survive")
{
   auto db = fresh("retire", /*retire_batch=*/512);
   auto ws = db->start_write_session();
   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(23);
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 3000; ++i)
      {
         auto k    = key16(rng);
         oracle[k] = "v" + k;
         tx.upsert(k, oracle[k]);
      }
      tx.commit();
   }
   REQUIRE(db->flush_now() == oracle.size());  // arms the pump at floor F

   // rewrite 100 keys AFTER the floor — the pump must NOT retire these
   std::vector<std::string> hot;
   {
      auto tx = ws->start_transaction(0);
      int  j  = 0;
      for (auto& [k, v] : oracle)
      {
         if (++j > 100)
            break;
         v = v + ":hot";
         tx.upsert(k, v);
         hot.push_back(k);
      }
      tx.commit();
   }
   // drive the pump with empty commits until the retire pass completes
   for (int i = 0; i < 40; ++i)
   {
      auto tx = ws->start_transaction(0);
      tx.commit();
   }
   // hot keys still answer from the DELTA with their post-floor values;
   // retired keys now answer from the BASE — every read still correct.
   auto rs  = db->start_read_session();
   auto lat = rs->create_cursor(0);
   for (auto& [k, v] : oracle)
      REQUIRE(lat.get<std::string>(k).value() == v);
   // and the delta really emptied down to ~the hot band: a fresh flush
   // re-flushes only what survived the pump (hot keys + stragglers).
   const uint64_t left = db->flush_now();
   REQUIRE(left >= hot.size());
   REQUIRE(left < oracle.size() / 4);
}

TEST_CASE("tiered: read-your-writes inside an open transaction")
{
   auto db = fresh("ryw");
   auto ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("base-key", "base-val");
      tx.commit();
   }
   db->flush_now();
   auto tx = ws->start_transaction(0);
   tx.upsert("txn-key", "txn-val");
   REQUIRE(tx.get<std::string>("txn-key").value() == "txn-val");
   REQUIRE(tx.get<std::string>("base-key").value() == "base-val");  // falls to base
   tx.remove("base-key");
   REQUIRE_FALSE(tx.get<std::string>("base-key").has_value());  // tombstone masks
   tx.abort();
}

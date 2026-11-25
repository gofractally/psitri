// tiered_kv differential tests: vpart hot delta over heart, vs std::map.
// Covers: envelope round-trip, tombstone masking (point + both scan
// directions), flush_now idempotence, the retire pump (keys rewritten after
// the floor SURVIVE), both views, and kv_engine conformance (cursors incl.).
#include <catch2/catch_test_macros.hpp>

#include <adb/testing/kv_conformance.hpp>  // shared bounds conformance (one check, all engines)
#include <delta_art/tiered_kv.hpp>

#include <psitri/dwal/kv_engine.hpp>
#include <sal_art_ref/engine.hpp>
#include <start_ver/kv.hpp>

#include <filesystem>
#include <map>
#include <random>
#include <string>

namespace fs = std::filesystem;

using TK = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::engine>;
using TE = delta_art::engine<start_ver::kv::engine, sal_art_ref::radix::engine>;
static_assert(psitri::dwal::kv_engine<TE>,
              "the COMPOSITION must satisfy the engine concept, cursors included (Dan)");

namespace
{
   std::shared_ptr<TK::database> fresh(const char* tag)
   {
      auto dir = fs::temp_directory_path() / (std::string("tiered_") + tag);
      fs::remove_all(dir);
      TK::config cfg;
      cfg.knobs.background = false;  // tests drive flush_now() deterministically
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

TEST_CASE("delta_art: write/read through the delta, flush to base, both views agree")
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
      auto stb = rs->create_base_cursor(0);
      REQUIRE_FALSE(stb.seek_begin());
   }
   SECTION("flush: stable converges; latest unchanged; re-flush is idempotent")
   {
      REQUIRE(db->flush_now() == oracle.size());
      auto stb = rs->create_base_cursor(0);
      uint64_t n = 0;
      for (bool ok = stb.seek_begin(); ok; ok = stb.next(), void())
      {
         REQUIRE(oracle.at(std::string(stb.key())) == stb.value<std::string>().value());
         ++n;
      }
      REQUIRE(n == oracle.size());
      // retirement emptied the delta: re-flush is a no-op, and the latest
      // view still answers every key (fall-through to the base)
      REQUIRE(db->flush_now() == 0);
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

TEST_CASE("delta_art: tombstones mask the base in point reads and BOTH scan directions")
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
   auto stb = rs->create_base_cursor(0);
   uint64_t n = 0;
   for (bool ok = stb.seek_begin(); ok; ok = stb.next(), void())
   {
      REQUIRE(oracle.count(std::string(stb.key())) == 1);
      ++n;
   }
   REQUIRE(n == oracle.size());
}

TEST_CASE("delta_art: retirement empties the flushed band; post-floor rewrites survive")
{
   auto db = fresh("retire");
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
   // flush_now retires inline now (merger-side 3-phase); a second flush
   // re-arms the floor past the hot rewrites and retires what aged out
   db->flush_now();
   // hot keys still answer from the DELTA with their post-floor values;
   // retired keys now answer from the BASE — every read still correct.
   auto rs  = db->start_read_session();
   auto lat = rs->create_cursor(0);
   for (auto& [k, v] : oracle)
      REQUIRE(lat.get<std::string>(k).value() == v);
   // and the delta really emptied: everything <= each flush's floor is gone;
   // only data written AFTER the most recent floor can remain.
   const uint64_t left = db->flush_now();
   REQUIRE(left < oracle.size() / 4);
}

TEST_CASE("delta_art: read-your-writes inside an open transaction")
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

// Regression (Dan): the tiered merge cursor must handle DIRECTION CHANGES across the
// W (hot) and B (cold) tiers. Interleave keys so the two legs straddle every probe;
// after a forward op (lower_bound/upper_bound) the non-current leg sits PAST the probe,
// and prev() must move it back — else settle_bwd_ re-selects the stranded leg on the
// wrong side. This is exactly the Silkworm crash: seek_last-of-namespace
// (lower_bound(successor)+prev) returned a NEXT-namespace key -> MDBX_NOTFOUND.
TEST_CASE("delta_art: bidirectional merge cursor handles direction changes across tiers")
{
   auto db = fresh("dirchange");
   auto ws = db->start_write_session();
   std::map<std::string, std::string> oracle;
   auto                               put = [&](const std::string& k, const std::string& v)
   {
      auto tx = ws->start_transaction(0);
      tx.upsert(k, v);
      tx.commit();
      oracle[k] = v;
   };
   auto kn = [](int i)
   {
      char b[8];
      std::snprintf(b, sizeof b, "k%04d", i);
      return std::string(b);
   };

   // Evens -> cold base (flushed); odds -> hot delta (kept). Merged = 0..199 sorted,
   // every adjacent pair lives on a different tier (maximally straddled).
   for (int i = 0; i < 200; i += 2)
      put(kn(i), "b" + std::to_string(i));
   REQUIRE(db->flush_now() == 100);  // evens retired into the base
   for (int i = 1; i < 200; i += 2)
      put(kn(i), "w" + std::to_string(i));  // odds remain in the delta

   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);  // latest = W u B merge

   SECTION("forward scan == sorted oracle")
   {
      auto it = oracle.begin();
      for (bool ok = cur.seek_begin(); ok; ok = cur.next(), ++it)
      {
         REQUIRE(it != oracle.end());
         REQUIRE(std::string(cur.key()) == it->first);
      }
      REQUIRE(it == oracle.end());
   }
   SECTION("backward scan == reverse sorted oracle")
   {
      auto it = oracle.rbegin();
      for (bool ok = cur.seek_last(); ok; ok = cur.prev(), ++it)
      {
         REQUIRE(it != oracle.rend());
         REQUIRE(std::string(cur.key()) == it->first);
      }
      REQUIRE(it == oracle.rend());
   }
   SECTION("lower_bound(K) then prev() == predecessor at every boundary (the bug)")
   {
      for (int i = 0; i <= 200; ++i)
      {
         auto probe = kn(i);
         bool ok    = cur.lower_bound(probe);
         auto lb    = oracle.lower_bound(probe);
         if (lb == oracle.end())
            REQUIRE_FALSE(ok);
         else
         {
            REQUIRE(ok);
            REQUIRE(std::string(cur.key()) == lb->first);
         }
         bool pok = cur.prev();
         if (lb == oracle.begin())
            REQUIRE_FALSE(pok);  // no predecessor
         else if (lb == oracle.end())
         {
            REQUIRE(pok);  // prev from end == last
            REQUIRE(std::string(cur.key()) == oracle.rbegin()->first);
         }
         else
         {
            REQUIRE(pok);
            REQUIRE(std::string(cur.key()) == std::prev(lb)->first);
         }
      }
   }
   SECTION("seek-last-of-range shape: lower_bound(successor)+prev stays in range")
   {
      // The exact Silkworm failure: the last key < k0100 is k0099 (a HOT-delta key),
      // never the >= boundary key k0100 (a cold-base key).
      REQUIRE(cur.lower_bound("k0100"));
      REQUIRE(std::string(cur.key()) == "k0100");
      REQUIRE(cur.prev());
      REQUIRE(std::string(cur.key()) == "k0099");
   }
   SECTION("ordered bounds (lower_bound ceiling, upper_bound floor) vs oracle")
   {
      // The SHARED adb conformance, instantiated on the tiered W u B cursor — the same
      // bounds check every engine runs (defined once in adb::testing), not a
      // delta_art-specific copy. Its floor (upper_bound) is lower_bound+prev, so it
      // also exercises forward-op-then-prev across the hot/cold tier boundary.
      long bound_fails = 0;
      adb::testing::check_bounds(rs->create_cursor(0), oracle,
                                 [&](bool c) { if (!c) ++bound_fails; });
      REQUIRE(bound_fails == 0);
   }
}

#include <atomic>
#include <thread>

// Hypothesis #1b (Dan): in the TIERED engine, a reader's root-0 (latest = W u B merge)
// snapshot must never read a freed value across the ROTATION/DRAIN of the hot delta into
// the cold base. When hi_water trips, the hot buffer rotates to "warm" and the merge
// thread drains it; root-0 readers see HOT u WARM u COLD. If the warm snapshot (or a
// value in it) is reclaimed while a reader/merge still references it, the reader reads
// freed memory (the Silkworm "value is type N, read as type 0" / EXC_BAD_ACCESS, which
// only reproduced in the tiered config, not pure start_ver). Robust assertion: each key
// always carries the SAME content V_k, so any committed version a snapshot lands on must
// read exactly V_k; any other bytes / throw / fault = a freed value was read => bug.
TEST_CASE("delta_art: concurrent root-0 snapshot never reads a freed value across rotation")
{
   auto dir = fs::temp_directory_path() / "tiered_rotrace";
   fs::remove_all(dir);
   TK::config cfg;                         // background ON: rotation + merge + drain + compaction
   cfg.knobs.hi_water_bytes = 1u << 20;    // small => frequent rotation/drain (warm churn)
   auto db = TK::database::open(dir, sal::open_mode::create_or_open, cfg);

   constexpr int    NK  = 3000;
   constexpr size_t LEN = 9000;            // large (value_large in the cold base)
   auto vof = [&](int k) {
      std::string v(LEN, 0);
      for (size_t j = 0; j < LEN; ++j)
         v[j] = char(uint8_t(uint64_t(k) * 131u + j * 7u + (j >> 8)));
      return v;
   };
   auto kof = [&](int k) {
      char b[16];
      std::snprintf(b, sizeof b, "k%08d", k);
      return std::string(b);
   };

   {
      auto ws = db->start_write_session();
      for (int base = 0; base < NK; base += 200)
      {
         auto tx = ws->start_transaction(0);
         for (int k = base; k < base + 200 && k < NK; ++k)
            tx.upsert(kof(k), vof(k));
         tx.commit();
      }
   }

   std::atomic<bool>     stop{false}, bad{false};
   std::atomic<uint64_t> reads{0}, writes{0};

   std::thread writer([&] {
      std::mt19937_64 rng(7);
      auto            ws = db->start_write_session();
      while (!stop.load(std::memory_order_relaxed))
      {
         auto tx = ws->start_transaction(0);
         for (int i = 0; i < 64; ++i)
         {
            int k = int(rng() % NK);
            tx.upsert(kof(k), vof(k));  // same content -> readers always expect V_k
         }
         tx.commit();
         writes.fetch_add(64, std::memory_order_relaxed);
      }
   });

   auto reader = [&] {
      std::mt19937_64 rng(1234 + reads.load());
      while (!stop.load(std::memory_order_relaxed))
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(0);  // latest = W u B merge
         for (int i = 0; i < 256; ++i)
         {
            int k = int(rng() % NK);
            try
            {
               auto v = cur.get<std::string>(kof(k));
               if (!v || *v != vof(k))
               {
                  bad.store(true);
                  stop.store(true);
                  return;
               }
            }
            catch (...)
            {
               bad.store(true);
               stop.store(true);
               return;
            }
            reads.fetch_add(1, std::memory_order_relaxed);
         }
      }
   };
   std::vector<std::thread> rs;
   for (int i = 0; i < 4; ++i)
      rs.emplace_back(reader);

   while (!bad.load() && writes.load() < 2'000'000)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
   stop.store(true);
   writer.join();
   for (auto& t : rs)
      t.join();

   INFO("reads=" << reads.load() << " writes=" << writes.load());
   REQUIRE_FALSE(bad.load());
}

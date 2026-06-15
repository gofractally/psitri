// start_ver::kv large-value node kind (K::value_large) — values whose leaf
// exceeds the line arena's single-CAS cap (64 x 128 = 8192 B minus header/suffix)
// live in a SECOND arena at an 8 KB grain (max object 64 x 8192 = 524288 B).
// This drives the FULL lifecycle through the public kv API: insert / get /
// update / remove / iterate / reopen-recover, with sizes spanning the
// line<->large boundary and a >0.5 MB value that must throw a clear error.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <start_ver/kv.hpp>

namespace fs = std::filesystem;

namespace
{
   // Deterministic, size-keyed value: byte j = (seed*131 + j*7). Exact bytes are
   // checked on read, so a wrong-arena deref or a truncated copy is caught.
   std::string mkval(size_t len, uint64_t seed)
   {
      std::string v(len, 0);
      for (size_t j = 0; j < len; ++j)
         v[j] = char(uint8_t(seed * 131u + j * 7u + (j >> 8)));
      return v;
   }
   std::string key(int i)
   {
      char b[16];
      std::snprintf(b, sizeof(b), "k%08d", i);
      return std::string(b);
   }

   // Every line/grain the engine allocated must come back once writes are drained.
   void require_drained(start_ver::kv::database& db)
   {
      db.flush_releases();
      db.flush_reaper();
      db.flush_releases();
      auto& a = db.store();
      uint64_t free_total = 0;
      for (uint32_t s = 0; s < a.num_segments(); ++s)
         free_total += a.freeset_of(s).free_lines();
      REQUIRE(free_total == uint64_t(a.num_segments()) * start_ver::lines_per_segment);
      auto& la = db.large_store();
      uint64_t lfree = 0;
      for (uint32_t s = 0; s < la.num_segments(); ++s)
         lfree += la.freeset_of(s).free_lines();
      REQUIRE(lfree ==
              uint64_t(la.num_segments()) * start_ver::large_geometry::units_per_segment);
   }
}  // namespace

// Sizes that straddle the line<->large boundary. 8000 B fits the line arena
// (leaf <= ~8 KB), 9000/100000/524000 B force value_large; one >0.5 MB throws.
TEST_CASE("kv large values: round-trip across the line/large boundary", "[large_value]")
{
   auto dir = fs::temp_directory_path() / "part_kv_largeval";
   fs::remove_all(dir);

   const std::vector<size_t> sizes = {8000, 9000, 16384, 100000, 524000};
   std::map<std::string, std::string> oracle;

   {
      auto db  = start_ver::kv::database::create(dir);
      auto ses = db->start_write_session();

      // One key per size, interspersed with small keys so iteration order must
      // be correct with large leaves mixed among line leaves.
      {
         auto tx = ses->start_transaction(0);
         for (int i = 0; i < int(sizes.size()); ++i)
         {
            std::string k = key(i * 2);            // even slots: the sized values
            std::string v = mkval(sizes[size_t(i)], uint64_t(i + 1));
            tx.upsert(k, v);
            oracle[k] = v;
            std::string sk = key(i * 2 + 1);       // odd slots: tiny line-arena values
            std::string sv = mkval(5 + size_t(i), 7000 + uint64_t(i));
            tx.upsert(sk, sv);
            oracle[sk] = sv;
         }
         tx.commit();
      }

      // value_large actually fired (used grains > 0): the large arena is in use.
      REQUIRE(db->large_used_units_dbg() > 0);

      // get() returns exact bytes for every key, large and small.
      {
         auto tx = ses->start_transaction(0);
         for (auto& [k, v] : oracle)
         {
            auto got = tx.get<std::string>(k);
            INFO("get key=" << k << " expect len=" << v.size());
            REQUIRE(got.has_value());
            REQUIRE(got->size() == v.size());
            REQUIRE(*got == v);
         }
         tx.abort();
      }

      // Update a large value to a DIFFERENT large size (forces a leaf rebuild in
      // the large arena + free of the old large leaf), and shrink one large to a
      // line value (cross-arena transition), and grow a line value to large.
      {
         auto tx = ses->start_transaction(0);
         oracle[key(0)] = mkval(200000, 42);      // 100000-ish -> 200000 large
         tx.upsert(key(0), oracle[key(0)]);
         oracle[key(2)] = mkval(7000, 43);        // 9000 large -> 7000 line
         tx.upsert(key(2), oracle[key(2)]);
         oracle[key(1)] = mkval(60000, 44);       // a small key grows to large
         tx.upsert(key(1), oracle[key(1)]);
         tx.commit();
      }
      {
         auto tx = ses->start_transaction(0);
         for (auto& [k, v] : oracle)
         {
            auto got = tx.get<std::string>(k);
            REQUIRE(got.has_value());
            REQUIRE(*got == v);
         }
         tx.abort();
      }

      // Forward iteration over a snapshot matches the oracle (large values
      // interspersed must not break ordering or key reconstruction).
      {
         auto c   = ses->get_root(0).snapshot_cursor();
         auto oit = oracle.begin();
         c.seek_begin();
         while (!c.is_end())
         {
            REQUIRE(oit != oracle.end());
            REQUIRE(std::string(c.key()) == oit->first);
            auto v = c.value<std::string>();
            REQUIRE(v.has_value());
            REQUIRE(*v == oit->second);
            c.next();
            ++oit;
         }
         REQUIRE(oit == oracle.end());
      }
   }

   // Reopen recovers every value (large and small) with exact bytes.
   {
      auto db  = start_ver::kv::database::open(dir);
      auto ses = db->start_write_session();
      auto tx  = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
      {
         auto got = tx.get<std::string>(k);
         INFO("reopen key=" << k << " len=" << v.size());
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      tx.abort();
   }

   // Remove every key; the large objects must be freed (large arena drains to
   // fully free) and the line arena returns to fully free.
   {
      auto db  = start_ver::kv::database::open(dir);
      auto ses = db->start_write_session();
      {
         auto tx = ses->start_transaction(0);
         for (auto& [k, v] : oracle)
            REQUIRE(tx.remove(k) == 0);
         tx.commit();
      }
      {
         auto tx = ses->start_transaction(0);
         for (auto& [k, v] : oracle)
            REQUIRE_FALSE(tx.get<std::string>(k).has_value());
         tx.abort();
      }
      ses.reset();
      require_drained(*db);
   }
}

// A value larger than the large arena's 0.5 MB object cap throws a clear error.
TEST_CASE("kv large values: above the 0.5 MB cap throws", "[large_value]")
{
   auto dir = fs::temp_directory_path() / "part_kv_largeval_cap";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   // 524288 B is the exact cap (64 x 8192). The leaf adds header + suffix, so a
   // value at the cap already needs > 64 grains -> throw; one well past it
   // (600000) definitely throws.
   auto tx = ses->start_transaction(0);
   const std::string huge = mkval(600000, 99);
   REQUIRE_THROWS_AS(tx.upsert("toobig", huge), std::runtime_error);
   tx.abort();
}

// Heavy mix: many keys at randomly chosen sizes spanning the boundary, with
// updates and removes, all verified against a std::map oracle, then drained.
TEST_CASE("kv large values: mixed-size fuzz vs oracle", "[large_value]")
{
   auto dir = fs::temp_directory_path() / "part_kv_largeval_fuzz";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(0xC0FFEE);
   const int                          N = 400;

   auto rand_size = [&]() -> size_t {
      switch (rng() % 5)
      {
         case 0: return 4 + rng() % 60;        // tiny (inline/line)
         case 1: return 2000 + rng() % 5000;   // line
         case 2: return 9000 + rng() % 4000;   // small large
         case 3: return 40000 + rng() % 60000; // medium large
         default: return 150000 + rng() % 200000;  // big large
      }
   };

   // Insert phase (batched commits retain shared roots -> COW on next batch).
   for (int base = 0; base < N; base += 64)
   {
      auto tx = ses->start_transaction(0);
      for (int i = base; i < std::min(N, base + 64); ++i)
      {
         std::string k = key(i);
         std::string v = mkval(rand_size(), uint64_t(i + 100));
         tx.upsert(k, v);
         oracle[k] = v;
      }
      tx.commit();
   }
   // Update phase: re-size every key (cross-arena transitions both directions).
   for (int base = 0; base < N; base += 64)
   {
      auto tx = ses->start_transaction(0);
      for (int i = base; i < std::min(N, base + 64); ++i)
      {
         std::string k = key(i);
         std::string v = mkval(rand_size(), uint64_t(i + 5000));
         tx.upsert(k, v);
         oracle[k] = v;
      }
      tx.commit();
   }
   // Verify all.
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
      {
         auto got = tx.get<std::string>(k);
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      tx.abort();
   }
   // Remove every other key.
   {
      auto tx = ses->start_transaction(0);
      std::vector<std::string> dead;
      int i = 0;
      for (auto& [k, v] : oracle)
         if (i++ & 1) dead.push_back(k);
      for (auto& k : dead)
      {
         REQUIRE(tx.remove(k) == 0);
         oracle.erase(k);
      }
      tx.commit();
   }
   // Verify survivors and absence of removed.
   {
      auto tx = ses->start_transaction(0);
      for (int i = 0; i < N; ++i)
      {
         std::string k   = key(i);
         auto        got = tx.get<std::string>(k);
         auto        oit = oracle.find(k);
         if (oit == oracle.end())
            REQUIRE_FALSE(got.has_value());
         else
         {
            REQUIRE(got.has_value());
            REQUIRE(*got == oit->second);
         }
      }
      tx.abort();
   }
   // Remove the rest, then everything must drain to fully free (no leak).
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
         REQUIRE(tx.remove(k) == 0);
      tx.commit();
   }
   oracle.clear();
   ses.reset();
   require_drained(*db);
}


#include <atomic>
#include <thread>

// Hypothesis #1 (Dan): a LARGE (value_large) value that is overwritten must not be
// reclaimed while a concurrent reader's snapshot still sees it. Reader snapshots are
// version-pinned; the old value's death (at the overwrite's version) must be gated by
// the reader floor. If the large-value death bypasses the floor, a reader dereferences
// freed+reused memory — the Silkworm "value is type N, read as type 0" / EXC_BAD_ACCESS.
//
// Robust-under-concurrency assertion: each key always carries the SAME content V_k, so
// whichever committed version a reader's snapshot lands on, get(k) MUST equal V_k. Any
// other bytes (or an invalid_type throw / fault) = a freed value was read => bug.
TEST_CASE("kv large values: concurrent snapshot never reads a freed value", "[large_value][mt]")
{
   auto dir = fs::temp_directory_path() / "sv_largeval_mt";
   fs::remove_all(dir);
   start_ver::kv::runtime_config cfg;  // background_release ON (drain thread runs)
   auto db = start_ver::kv::database::open(dir, cfg);

   constexpr int NK  = 3000;
   constexpr size_t LEN = 9000;  // forces value_large (> ~8 KB line-arena leaf cap)
   auto vof = [&](int k) { return mkval(LEN, uint64_t(k)); };

   // pre-fill: every key k carries the fixed large value V_k
   {
      auto ws = db->start_write_session();
      for (int base = 0; base < NK; base += 200)
      {
         auto tx = ws->start_transaction(0);
         for (int k = base; k < base + 200 && k < NK; ++k)
            tx.upsert(key(k), vof(k));
         tx.commit();
      }
   }

   std::atomic<bool>     stop{false};
   std::atomic<bool>     bad{false};
   std::atomic<uint64_t> reads{0}, writes{0};

   // writer: redundantly rewrite random keys with the SAME content -> churns the COW
   // path + frees/reallocates the large values, without changing what a reader expects.
   std::thread writer([&] {
      std::mt19937_64 rng(7);
      auto            ws = db->start_write_session();
      while (!stop.load(std::memory_order_relaxed))
      {
         auto tx = ws->start_transaction(0);
         for (int i = 0; i < 64; ++i)
         {
            int k = int(rng() % NK);
            tx.upsert(key(k), vof(k));
         }
         tx.commit();
         writes.fetch_add(64, std::memory_order_relaxed);
      }
   });

   // readers: snapshot, read random keys, every value MUST be exactly V_k
   auto reader = [&] {
      std::mt19937_64 rng(1234 + reads.load());
      while (!stop.load(std::memory_order_relaxed))
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(0);
         for (int i = 0; i < 256; ++i)
         {
            int k = int(rng() % NK);
            try
            {
               auto v = cur.get<std::string>(key(k));
               if (!v || *v != vof(k))
               {
                  bad.store(true, std::memory_order_relaxed);
                  stop.store(true, std::memory_order_relaxed);
                  return;
               }
            }
            catch (...)
            {  // invalid_type etc. = a freed value was dereferenced
               bad.store(true, std::memory_order_relaxed);
               stop.store(true, std::memory_order_relaxed);
               return;
            }
            reads.fetch_add(1, std::memory_order_relaxed);
         }
      }
   };
   std::vector<std::thread> rthreads;
   for (int i = 0; i < 4; ++i)
      rthreads.emplace_back(reader);

   // run until enough churn has happened (the bench crashed by ~few hundred K writes)
   while (!bad.load(std::memory_order_relaxed) && writes.load() < 2'000'000)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
   stop.store(true);
   writer.join();
   for (auto& t : rthreads)
      t.join();

   INFO("reads=" << reads.load() << " writes=" << writes.load());
   REQUIRE_FALSE(bad.load());
}

// Regression: flush_releases() must NOT spin when there is no background drain
// thread (background_release=false) — it has to drain INLINE — and must not block on
// reader-pinned (gated) chains. Previously this hung forever. Also a clean single-
// threaded check of hypothesis #1: a snapshot pinned at v1 must keep reading V0 across
// overwrite+drain of that key's large value (the floor must gate the old value's death).
TEST_CASE("kv: flush_releases drains inline (no spin) + pinned snapshot keeps its value",
          "[large_value]")
{
   auto dir = fs::temp_directory_path() / "sv_flush_inline";
   fs::remove_all(dir);
   start_ver::kv::runtime_config cfg;
   cfg.background_release = false;  // no drain thread => flush_releases must drain inline
   auto db = start_ver::kv::database::open(dir, cfg);

   const std::string K   = key(42);
   const size_t      LEN = 100000;  // value_large
   const std::string V0  = mkval(LEN, 0);
   {
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.upsert(K, V0);
      tx.commit();
   }
   auto rs   = db->start_read_session();
   auto snap = rs->create_cursor(0);  // pin v1
   REQUIRE(snap.get<std::string>(K) == V0);

   auto ws = db->start_write_session();
   for (int i = 1; i <= 50; ++i)
   {
      auto tx = ws->start_transaction(0);
      tx.upsert(K, mkval(LEN, uint64_t(i)));
      tx.commit();
      db->flush_releases();  // MUST return (previously spun forever)
      auto got = snap.get<std::string>(K);
      REQUIRE(got.has_value());
      REQUIRE(*got == V0);  // floor must keep V0 alive while pinned
   }
}

// Reproduce the sal_art_ref cold-base corruption (the Silkworm drain SIGBUS) standalone.
// Crash = garbage child ptr_address in a cold-base router node, hit by route_to_page/
// max_page on the BACKGROUND merge thread (merge_loop_ -> drain_warm_) during the drain's
// upserts — concurrently with the writer filling the hot delta and the SAL background
// compactor/release relocating+freeing cold-base nodes. The crash point varied with
// timing (block 736k under TSan vs first flush under Release) => it is a RACE, so this
// repro maximizes concurrency: continuous churn (no synchronous flush_now), background
// drain ON, small hi_water (frequent rotations), large/vt_extern values, at scale, plus
// concurrent reader scans that walk the cold base (route_to_page/max_page).
#include <delta_art/tiered_kv.hpp>
#include <psitri/dwal/kv_engine.hpp>
#include <sal_art_ref/engine.hpp>
#include <start_ver/kv.hpp>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

using TK = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::engine>;

static std::string hashed_key(std::mt19937_64& rng, int nns)
{
   std::string k(32, '\0');
   k[0] = char(rng() % nns);
   for (int i = 1; i < 32; ++i)
      k[i] = char(rng() & 0xff);
   return k;
}
static std::string make_value(std::mt19937_64& rng, uint64_t tag)
{
   static const int sizes[] = {40, 64, 200, 700, 1500, 6600, 12000};
   int              n       = sizes[rng() % (sizeof(sizes) / sizeof(sizes[0]))];
   std::string      v(size_t(n), char('A' + int(tag & 31)));
   std::snprintf(v.data(), std::min<size_t>(v.size(), 20), "v%llu", (unsigned long long)tag);
   return v;
}

int main(int argc, char** argv)
{
   const std::string dir      = argc > 1 ? argv[1] : "/tmp/da_largeval_repro";
   const uint64_t    rounds   = argc > 2 ? strtoull(argv[2], nullptr, 10) : 400;
   const uint64_t    per      = argc > 3 ? strtoull(argv[3], nullptr, 10) : 4000;  // ops/txn
   const int         nreaders = argc > 4 ? atoi(argv[4]) : 3;
   const int         nns      = 6;
   std::filesystem::remove_all(dir);

   TK::config cfg;  // background drain ON (default); small hi_water => frequent drains
   cfg.knobs.hi_water_bytes =
       std::getenv("PSITRI_MDBX_HI_WATER_MB")
           ? (strtoull(std::getenv("PSITRI_MDBX_HI_WATER_MB"), nullptr, 10) << 20)
           : (2ull << 20);
   auto db = TK::database::open(dir, sal::open_mode::create_or_open, cfg);
   auto ws = db->start_write_session();

   std::printf("largeval CONCURRENT repro: rounds=%llu per=%llu readers=%d dir=%s\n",
               (unsigned long long)rounds, (unsigned long long)per, nreaders, dir.c_str());

   std::atomic<bool> stop{false};
   // concurrent readers: continuously scan the latest view (W u B) — walks the cold base
   auto reader = [&]()
   {
      std::mt19937_64 rng(99);
      while (!stop.load(std::memory_order_relaxed))
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_base_cursor(0);  // stable = cold base ONLY (sal_art_ref traversal)
         uint64_t n = 0;
         for (bool ok = cur.seek_begin(); ok && n < 200000; ok = cur.next())
         {
            (void)cur.key();
            (void)cur.value<std::string>();
            ++n;
         }
         if (cur.seek_last())
            (void)cur.key();
      }
   };
   std::vector<std::thread> readers;
   for (int i = 0; i < nreaders; ++i)
      readers.emplace_back(reader);

   std::mt19937_64          rng(1234);
   std::vector<std::string> live;
   uint64_t                 tag = 0;
   for (uint64_t r = 0; r < rounds; ++r)
   {
      auto tx = ws->start_transaction(0);
      for (uint64_t i = 0; i < per; ++i)
      {
         // Update-DOMINANT on a bounded working set: keep ~200k live keys, then mostly
         // rewrite them (+ churn removes/re-inserts). This makes the COLD base COW the
         // same nodes over and over -> heavy garbage -> aggressive compactor relocation
         // racing the drain (the Silkworm state-update shape).
         uint32_t roll = rng() % 100;
         const bool want_grow = live.size() < 200000;
         if ((roll < 15 && want_grow) || live.empty())
         {  // INSERT (only while growing the working set)
            auto k = hashed_key(rng, nns);
            auto v = make_value(rng, tag++);
            tx.upsert(k, v);
            live.push_back(k);
         }
         else if (roll < 90)
         {  // UPDATE existing (cold-base node COW churn)
            tx.upsert(live[rng() % live.size()], make_value(rng, tag++));
         }
         else
         {  // REMOVE + immediately re-insert a fresh key (collapse + regrow churn)
            size_t idx = rng() % live.size();
            tx.remove(live[idx]);
            auto k = hashed_key(rng, nns);
            tx.upsert(k, make_value(rng, tag++));
            live[idx] = k;
         }
      }
      tx.commit();  // background merge drains when hi_water trips — NO synchronous flush
      if (r % 20 == 0)
      {
         std::printf("round=%llu live=%zu (background draining)\n", (unsigned long long)r,
                     live.size());
         std::fflush(stdout);
      }
   }
   stop.store(true);
   for (auto& t : readers)
      t.join();
   std::printf("ALL OK — %llu concurrent churn rounds, no cold-base corruption\n",
               (unsigned long long)rounds);
   return 0;
}

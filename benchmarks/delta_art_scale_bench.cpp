// tiered progressive scale bench — the heart_scale_bench twin over the L5
// pipeline (vpart hot delta -> burst flush -> heart), so the decay curves
// overlay the past 1B heart run: splitmix64 u64 keys -> u64 values, BATCHED
// commits (Dan: 1000/txn), and after every 1M inserts 100k random point
// reads + 100k iteration steps from a random lower_bound — both on the
// LATEST view (delta-first, falls through to heart).
//
//   tiered-scale-bench [total_keys=1e9] [batch=1000] [probe_every=1e6]
//                      [dir=/tmp/tiered_scale_db]
//
// The delta never sees the 32 GB line_ptr wall: the retire pump keeps it at
// the hot band while heart absorbs the long tail — THE pipeline thesis.
#include <delta_art/tiered_kv.hpp>

#include <heart_dwal_engine.hpp>
#include <start_ver/kv.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;
using TK    = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>;

static uint64_t splitmix64(uint64_t x)
{
   x += 0x9E3779B97F4A7C15ull;
   x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
   x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
   return x ^ (x >> 31);
}
static std::string_view k8(const uint64_t& v)
{
   return {reinterpret_cast<const char*>(&v), 8};
}
static double ns_per(Clock::time_point a, Clock::time_point b, uint64_t n)
{
   return double(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()) /
          double(n ? n : 1);
}
static double rss_gb()
{
   rusage ru{};
   getrusage(RUSAGE_SELF, &ru);
   return double(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
}
static double dir_gb(const std::filesystem::path& p)
{
   uint64_t b = 0;
   std::error_code ec;
   for (auto& e : std::filesystem::recursive_directory_iterator(p, ec))
      if (e.is_regular_file(ec))
         b += e.file_size(ec);
   return double(b) / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char** argv)
{
   const uint64_t total = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1'000'000'000ull;
   const uint64_t batch = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000ull;
   const uint64_t probe = argc > 3 ? strtoull(argv[3], nullptr, 10) : 1'000'000ull;
   const char*    dirs  = argc > 4 ? argv[4] : "/tmp/tiered_scale_db";
   const uint64_t READS = 100'000, ITERS = 100'000;

   std::filesystem::path dir(dirs);
   std::filesystem::remove_all(dir);
   std::filesystem::create_directories(dir);
   TK::config cfg;
   if (const char* mb = getenv("DELTA_HI_WATER_MB"))
      cfg.knobs.hi_water_bytes = strtoull(mb, nullptr, 10) << 20;
   if (const char* ab = getenv("DELTA_ARENA_MB"))
      cfg.knobs.arena_budget_bytes = strtoull(ab, nullptr, 10) << 20;
   auto db  = TK::database::open(dir, sal::open_mode::create_or_open, cfg);
   auto ws  = db->start_write_session();
   auto rs  = db->start_read_session();

   printf("delta_art scale bench: %llu keys, batch %llu, probes every %llu "
          "(%llu rds + %llu iter steps), hi_water %llu MB\n",
          (unsigned long long)total, (unsigned long long)batch, (unsigned long long)probe,
          (unsigned long long)READS, (unsigned long long)ITERS,
          (unsigned long long)(cfg.knobs.hi_water_bytes >> 20));
   printf("%10s %9s %9s %7s %9s %9s %8s %8s\n", "keys", "ins ns", "rd ns", "hit%",
          "iter ns", "file GB", "rss GB", "chunk s");
   fflush(stdout);

   std::mt19937_64 rng(0xC0FFEE);
   uint64_t        done    = 0;
   auto            t_start = Clock::now();
   while (done < total)
   {
      const uint64_t end = std::min(total, done + probe);
      auto           t0  = Clock::now();
      try
      {
         uint64_t i = done;
         while (i < end)
         {
            const uint64_t bend = std::min(end, i + batch);
            auto           tx   = ws->start_transaction(0);
            for (; i < bend; ++i)
            {
               const uint64_t k = splitmix64(i);
               const uint64_t v = splitmix64(i ^ 0xDEADBEEFCAFEBABEull);
               tx.upsert(k8(k), k8(v));
            }
            tx.commit();
         }
      }
      catch (const std::exception& e)
      {
         printf("STOP at %llu committed keys: %s\n", (unsigned long long)done, e.what());
         break;
      }
      auto           t1    = Clock::now();
      const uint64_t n_ins = end - done;
      done                 = end;

      uint64_t found = 0;
      auto     t2    = Clock::now();
      {
         auto cur = rs->create_cursor(0);  // latest view
         for (uint64_t r = 0; r < READS; ++r)
         {
            const uint64_t k = splitmix64(rng() % done);
            if (cur.get<std::string>(k8(k)))
               ++found;
         }
      }
      auto t3 = Clock::now();

      uint64_t stepped = 0, ksum = 0;
      auto     t4      = Clock::now();
      {
         auto           cur = rs->create_cursor(0);
         const uint64_t k0  = splitmix64(rng() % done);
         cur.lower_bound(k8(k0));
         for (uint64_t s = 0; s < ITERS; ++s)
         {
            if (cur.is_end())
               cur.seek_begin();
            ksum += uint8_t(cur.key()[0]);
            cur.next();
            ++stepped;
         }
      }
      auto t5 = Clock::now();

      printf("%10llu %9.0f %9.0f %6.1f%% %9.1f %9.2f %8.2f %8.2f\n",
             (unsigned long long)done, ns_per(t0, t1, n_ins), ns_per(t2, t3, READS),
             100.0 * double(found) / double(READS), ns_per(t4, t5, stepped), dir_gb(dir),
             rss_gb(), std::chrono::duration<double>(t5 - t0).count());
      fflush(stdout);
      (void)ksum;
   }
   const double total_s = std::chrono::duration<double>(Clock::now() - t_start).count();
   {  // delta arena occupancy: used vs total lines (128 B each)
      auto occ   = db->delta_occupancy_dbg();
      auto used  = occ.first;
      auto total = occ.second;
      printf("delta arena: %.2f GB used / %.2f GB total (%.0f%% used)\n",
             double(used) * 128 / 1e9, double(total) * 128 / 1e9,
             total ? 100.0 * double(used) / double(total) : 0.0);
   }
   printf("DONE: %llu keys in %.1f min\n", (unsigned long long)done, total_s / 60.0);
   return 0;
}

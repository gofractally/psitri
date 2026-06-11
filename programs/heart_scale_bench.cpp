// heart progressive scale bench — the exact twin of part_scale_bench so the
// decay curves overlay: splitmix64 u64 keys -> u64 values inserted in 1M-chunk
// commits; after every chunk, 100k random point reads + a 100k-step iteration
// from a random lower_bound start; one flushed status line per chunk.
//
//   heart-scale-bench [total_keys=1e9] [chunk=1e6] [dir=/tmp/heart_scale_db]
//
// heart has no 32 GB address wall (SAL ptr_address is an object id, not a byte
// offset), so this is the engine the 1B-key curve comes from.
#include <radix_engine.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;

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
   return double(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);  // darwin: bytes
}

int main(int argc, char** argv)
{
   const uint64_t total = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1'000'000'000ull;
   const uint64_t chunk = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000'000ull;
   const char*    dirs  = argc > 3 ? argv[3] : "/tmp/heart_scale_db";
   const uint64_t READS = 100'000, ITERS = 100'000;

   std::filesystem::path dir(dirs);
   std::filesystem::remove_all(dir);
   std::filesystem::create_directories(dir);
   auto db  = heart::radix::database::open(dir.string());
   auto ses = db->start_write_session();
   auto rs  = db->start_read_session();

   printf("heart scale bench: %llu keys, chunk %llu, %llu rds + %llu iter steps per chunk\n",
          (unsigned long long)total, (unsigned long long)chunk, (unsigned long long)READS,
          (unsigned long long)ITERS);
   printf("%10s %9s %9s %7s %9s %9s %8s %7s %8s\n", "keys", "ins ns", "rd ns", "hit%",
          "iter ns", "file GB", "free%", "rss GB", "chunk s");
   fflush(stdout);

   std::mt19937_64 rng(0xC0FFEE);
   uint64_t        done    = 0;
   auto            t_start = Clock::now();
   while (done < total)
   {
      const uint64_t end = std::min(total, done + chunk);
      // ── insert chunk (one commit) ──
      auto t0 = Clock::now();
      try
      {
         auto tx = ses->start_transaction(0);
         for (uint64_t i = done; i < end; ++i)
         {
            const uint64_t k = splitmix64(i);
            const uint64_t v = splitmix64(i ^ 0xDEADBEEFCAFEBABEull);
            tx.upsert(k8(k), k8(v));
         }
         tx.commit();
      }
      catch (const std::exception& e)
      {
         printf("STOP at %llu committed keys: %s\n", (unsigned long long)done, e.what());
         break;
      }
      auto           t1    = Clock::now();
      const uint64_t n_ins = end - done;
      done                 = end;

      // ── 100k random point reads over everything inserted so far ──
      uint64_t found = 0;
      auto     t2    = Clock::now();
      {
         auto cur = rs->create_cursor(0);
         for (uint64_t r = 0; r < READS; ++r)
         {
            const uint64_t k = splitmix64(rng() % done);
            if (cur.get<std::string>(k8(k)))
               ++found;
         }
      }
      auto t3 = Clock::now();

      // ── 100k iteration steps from a random start (wraps at end) ──
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

      // ── SAL segment census ──
      auto     d       = db->dump();
      uint64_t live    = 0;
      for (auto& seg : d.segments)
         if (!seg.is_free && seg.alloc_pos > 0)
            live += seg.alloc_pos - seg.freed_bytes;
      const double file_gb =
          double(d.total_segments) * sal::segment_size / (1024.0 * 1024 * 1024);
      const double freep =
          file_gb > 0 ? 100.0 * (1.0 - double(live) / (double(d.total_segments) *
                                                       sal::segment_size))
                      : 0.0;

      printf("%9.0fM %9.1f %9.1f %6.1f%% %9.1f %9.2f %7.1f%% %7.1f %8.1f\n",
             double(done) / 1e6, ns_per(t0, t1, n_ins), ns_per(t2, t3, READS),
             100.0 * double(found) / double(READS), ns_per(t4, t5, stepped), file_gb, freep,
             rss_gb(), std::chrono::duration<double>(t5 - t0).count());
      fflush(stdout);
      (void)ksum;
   }
   db->wait_for_compactor();
   printf("total wall: %.1f s\n", std::chrono::duration<double>(Clock::now() - t_start).count());
   return 0;
}

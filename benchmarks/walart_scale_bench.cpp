// walart progressive scale bench — the delta_art_scale_bench twin over the
// WAL + art_map write-set engine (sal_art_ref base + start_ref delta). Same
// splitmix64 u64 workload, BATCHED commits, and after every 1M inserts 100k
// random point reads + 100k iteration steps from a random lower_bound — so
// the curves overlay the delta_art run.
//
//   walart-scale-bench [total_keys=1e8] [batch=1000] [probe_every=1e6]
//                      [dir=/tmp/walart_scale_db]
#include <walart/walart.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
   const uint64_t total = argc > 1 ? strtoull(argv[1], nullptr, 10) : 100'000'000ull;
   const uint64_t batch = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000ull;
   const uint64_t probe = argc > 3 ? strtoull(argv[3], nullptr, 10) : 1'000'000ull;
   const char*    dirs  = argc > 4 ? argv[4] : "/tmp/walart_scale_db";
   const uint64_t READS = 100'000;

   std::filesystem::path dir(dirs);
   std::filesystem::remove_all(dir);
   std::filesystem::create_directories(dir);

   auto db = walart::database::open(dir);
   auto ws = db->start_write_session();
   auto rs = db->start_read_session();

   printf("walart scale bench: %llu keys, batch %llu, probes every %llu (%llu rds)\n",
          (unsigned long long)total, (unsigned long long)batch, (unsigned long long)probe,
          (unsigned long long)READS);
   printf("%10s %9s %9s %7s %9s %8s %8s\n", "keys", "ins ns", "rd ns", "hit%", "file GB",
          "rss GB", "chunk s");
   fflush(stdout);

   std::mt19937_64 rng(0xC0FFEE);
   uint64_t        done = 0;
   auto            t0g  = Clock::now();
   while (done < total)
   {
      const uint64_t end = std::min(total, done + probe);
      auto           t0  = Clock::now();
      uint64_t       i   = done;
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
      auto           t1    = Clock::now();
      const uint64_t n_ins = end - done;
      done                 = end;

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

      printf("%10llu %9.0f %9.0f %6.1f%% %9.2f %8.2f %8.2f\n", (unsigned long long)done,
             ns_per(t0, t1, n_ins), ns_per(t2, t3, READS), 100.0 * double(found) / double(READS),
             dir_gb(dir), rss_gb(), std::chrono::duration<double>(t3 - t0).count());
      fflush(stdout);
   }
   printf("DONE: %llu keys in %.1f min\n", (unsigned long long)done,
          std::chrono::duration<double>(Clock::now() - t0g).count() / 60.0);
   return 0;
}

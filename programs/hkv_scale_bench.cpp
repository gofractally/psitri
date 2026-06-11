// hkv (standalone, ~/myartlib) progressive scale bench — the in-memory leg of
// the scale family: same splitmix64 u64 keys/values, 1M-key chunks, per-chunk
// 100k random point reads + 100k iteration steps from a random lower_bound.
// No persistence, no transactions: hkv here is a plain in-memory container on
// the default std::allocator, so handles are the native 6-byte packed_ptr
// (48-bit direct pointers) — exactly the u48 configuration part would inherit.
//
// Build (standalone — hkv is header-only outside this repo):
//   clang++ -O3 -DNDEBUG -std=c++23 -march=native -I ~/myartlib/src \
//       programs/hkv_scale_bench.cpp -o hkv-scale-bench
//
//   hkv-scale-bench [total_keys=2e9] [chunk=1e6]
//
// Columns match the persistent benches minus file/free%: rss tracks the heap.
#include "hkv/tree.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
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
   const uint64_t total = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2'000'000'000ull;
   const uint64_t chunk = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000'000ull;
   const uint64_t READS = 100'000, ITERS = 100'000;

   #ifdef HKV_BUCKETS
   hkv::kv_tree<std::string_view, uint64_t, hkv::mode::buckets> t;
#else
   hkv::kv_tree<std::string_view, uint64_t> t;
#endif

   printf("hkv scale bench (in-memory, u48 packed_ptr): %llu keys, chunk %llu\n",
          (unsigned long long)total, (unsigned long long)chunk);
   printf("%10s %9s %9s %7s %9s %7s %8s\n", "keys", "ins ns", "rd ns", "hit%", "iter ns",
          "rss GB", "chunk s");
   fflush(stdout);

   std::mt19937_64 rng(0xC0FFEE);
   uint64_t        done    = 0;
   auto            t_start = Clock::now();
   while (done < total)
   {
      const uint64_t end = std::min(total, done + chunk);
      auto           t0  = Clock::now();
      for (uint64_t i = done; i < end; ++i)
      {
         const uint64_t k = splitmix64(i);
         t.insert(k8(k), splitmix64(i ^ 0xDEADBEEFCAFEBABEull));
      }
      auto           t1    = Clock::now();
      const uint64_t n_ins = end - done;
      done                 = end;

      // ── 100k random point reads over everything inserted so far ──
      uint64_t found = 0;
      auto     t2    = Clock::now();
      for (uint64_t r = 0; r < READS; ++r)
      {
         const uint64_t k = splitmix64(rng() % done);
         uint64_t       v;
         if (t.find(k8(k), v))
            ++found;
      }
      auto t3 = Clock::now();

      // ── 100k iteration steps from a random start (wraps at end) ──
      uint64_t stepped = 0, ksum = 0;
      auto     t4      = Clock::now();
      {
         const uint64_t k0 = splitmix64(rng() % done);
         auto           it = t.lower_bound(k8(k0));
         for (uint64_t s = 0; s < ITERS; ++s)
         {
            if (it == t.end())
               it = t.begin();
            ksum += uint8_t(it.key()[0]);
            ++it;
            ++stepped;
         }
      }
      auto t5 = Clock::now();

      printf("%9.0fM %9.1f %9.1f %6.1f%% %9.1f %7.1f %8.1f\n", double(done) / 1e6,
             ns_per(t0, t1, n_ins), ns_per(t2, t3, READS),
             100.0 * double(found) / double(READS), ns_per(t4, t5, stepped), rss_gb(),
             std::chrono::duration<double>(t5 - t0).count());
      fflush(stdout);
      (void)ksum;
   }
   printf("total wall: %.1f s\n", std::chrono::duration<double>(Clock::now() - t_start).count());
   return 0;
}

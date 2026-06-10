// part progressive scale bench: random u64 keys -> u64 values, inserted in 1M chunks;
// after every chunk, 100k random point reads + a 100k-step iteration from a random
// start. One status line per chunk so the decay curve is visible as the working set
// outgrows cache tiers (L2 -> LLC -> DRAM -> page cache). Target: 1B keys.
//
//   part-scale-bench [total_keys=1e9] [chunk=1e6] [dir=/tmp/part_scale_db]
//
// Keys are splitmix64(i) — a bijection on u64, so all keys are unique and uniformly
// random-ordered. Values are 8 bytes of splitmix64(i ^ seed). Commits once per chunk.
#include <part/kv.hpp>

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
   const char*    dirs  = argc > 3 ? argv[3] : "/tmp/part_scale_db";
   const uint64_t READS = 100'000, ITERS = 100'000;

   std::filesystem::path dir(dirs);
   std::filesystem::remove_all(dir);
   part::kv::runtime_config cfg;
   cfg.sync_mode    = sal::sync_type::none;
   // 4096 x 8MB = 2^28 lines: detail::line_ptr persists a 4-bit kind tag + 28-bit
   // line index in 4 bytes, so 32 GB is the HARD address-space ceiling — raising
   // max_segments past it silently truncates pointers (the from_off assert is
   // NDEBUG'd out). The run ends at "arena is full" instead of reaching 1B keys.
   cfg.max_segments = 4096;
   auto db  = part::kv::database::create(dir, cfg);
   auto ses = db->start_write_session();
   auto rs  = db->start_read_session();

   printf("part scale bench: %llu keys, chunk %llu, %llu rds + %llu iter steps per chunk\n",
          (unsigned long long)total, (unsigned long long)chunk, (unsigned long long)READS,
          (unsigned long long)ITERS);
   printf("%10s %9s %9s %7s %9s %9s %8s %7s %8s\n", "keys", "ins ns", "rd ns", "hit%",
          "iter ns", "file GB", "free%", "rss GB", "chunk s");
   fflush(stdout);

   std::mt19937_64 rng(0xC0FFEE);
   uint64_t        done     = 0;
   auto            t_start  = Clock::now();
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
      auto           t1     = Clock::now();
      const uint64_t n_ins  = end - done;
      done                  = end;

      // ── 100k random point reads over everything inserted so far ──
      uint64_t found = 0;
      auto     t2    = Clock::now();
      {
         auto cur = rs->create_cursor(0);
         for (uint64_t r = 0; r < READS; ++r)
         {
            const uint64_t k = splitmix64(rng() % done);
            uint64_t       out;
            if (cur.get(k8(k), [&](part::kv::value_view v)
                        { std::memcpy(&out, v.data(), 8); }))
               ++found;
         }
      }
      auto t3 = Clock::now();

      // ── 100k iteration steps from a random start (wraps at end) ──
      uint64_t stepped = 0, ksum = 0;
      auto     t4      = Clock::now();
      {
         auto           cur = ses->get_root(0).snapshot_cursor();
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

      // ── arena census ──
      auto&    arena   = db->store();
      const uint32_t nseg = arena.num_segments();
      uint64_t free_ln = 0;
      for (uint32_t sg = 0; sg < nseg; ++sg)
         free_ln += arena.freeset_of(sg).free_lines();
      const double file_gb = double(nseg) * part::segment_bytes / (1024.0 * 1024 * 1024);
      const double freep =
          nseg ? 100.0 * double(free_ln) / (double(nseg) * part::lines_per_segment) : 0.0;

      printf("%9.0fM %9.1f %9.1f %6.1f%% %9.1f %9.2f %7.1f%% %7.1f %8.1f\n",
             double(done) / 1e6, ns_per(t0, t1, n_ins), ns_per(t2, t3, READS),
             100.0 * double(found) / double(READS), ns_per(t4, t5, stepped), file_gb, freep,
             rss_gb(), std::chrono::duration<double>(t5 - t0).count());
      fflush(stdout);
      (void)ksum;
   }
   printf("total wall: %.1f s\n", std::chrono::duration<double>(Clock::now() - t_start).count());
   return 0;
}

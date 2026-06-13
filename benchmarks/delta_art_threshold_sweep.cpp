// delta_art threshold sweep — Dan's thesis: keeping the hot delta SMALL beats
// letting it grow (a big delta accumulates COW churn that poisons writes via
// locality + claim_ scans; flushing early/often keeps the working footprint
// tight). The effect needs a GROWING key set (distinct splitmix64 keys) — an
// overwrite workload's live set is bounded and never degrades. Run the SAME
// growing-key insert workload at a range of flush high-water marks and report
// the STEADY-STATE insert rate (last 25%, where the delta is at its widest
// per cycle), read latency, and on-disk footprint.
//
//   delta_art-threshold-sweep [total_keys=5e7] [batch=1000] [unused]
//                             [dir=/tmp/da_sweep]  (SWEEP_MB="32 64 128" env)
#include <delta_art/tiered_kv.hpp>

#include <heart_dwal_engine.hpp>
#include <start_ver/kv.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using TK    = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>;

namespace
{
   constexpr uint64_t INIT = 1'000'000ull, AMOUNT = 1;
   std::string acct_key(uint64_t i) { return std::string(reinterpret_cast<const char*>(&i), 8); }
   std::string enc_bal(uint64_t b) { return std::string(reinterpret_cast<const char*>(&b), 8); }
   uint64_t    dec_bal(std::string_view s)
   {
      uint64_t b = 0;
      std::memcpy(&b, s.data(), std::min(s.size(), sizeof(b)));
      return b;
   }
   uint64_t pick(std::mt19937_64& rng, uint64_t n)
   {
      std::uniform_int_distribution<uint64_t> d(0, n / 2 - 1);
      return d(rng) + d(rng);
   }
   double secs(Clock::time_point a, Clock::time_point b)
   {
      return std::chrono::duration<double>(b - a).count();
   }
   double dir_gb(const std::filesystem::path& p)
   {
      uint64_t        b = 0;
      std::error_code ec;
      for (auto& e : std::filesystem::recursive_directory_iterator(p, ec))
         if (e.is_regular_file(ec))
            b += e.file_size(ec);
      return double(b) / (1024.0 * 1024.0 * 1024.0);
   }
}  // namespace

uint64_t splitmix64(uint64_t x)
{
   x += 0x9E3779B97F4A7C15ull;
   x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
   x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
   return x ^ (x >> 31);
}

int main(int argc, char** argv)
{
   const uint64_t TOTAL = argc > 1 ? strtoull(argv[1], nullptr, 10) : 50'000'000ull;
   const uint64_t BATCH = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000ull;
   const std::string base = argc > 4 ? argv[4] : "/tmp/da_sweep";
   const uint64_t READS = 200'000;

   std::vector<uint64_t> waters = {16, 32, 64, 128, 256, 512, 1024};
   if (const char* ml = getenv("SWEEP_MB"))  // e.g. SWEEP_MB="32 64 128"
   {
      waters.clear();
      std::string s(ml);
      size_t      p = 0;
      while (p < s.size())
      {
         size_t q = s.find(' ', p);
         if (q == std::string::npos)
            q = s.size();
         if (q > p)
            waters.push_back(strtoull(s.substr(p, q - p).c_str(), nullptr, 10));
         p = q + 1;
      }
   }

   printf("delta_art threshold sweep: %llu growing keys, batch %llu\n\n",
          (unsigned long long)TOTAL, (unsigned long long)BATCH);
   printf("%8s %12s %12s %10s %8s\n", "hi_wMB", "ins ns(ss)", "ins ns(all)", "rd ns", "file GB");
   fflush(stdout);

   const uint64_t ss_start = TOTAL - TOTAL / 4;  // last 25% = steady state

   for (uint64_t mb : waters)
   {
      std::filesystem::path dir = std::filesystem::path(base) / std::to_string(mb);
      std::filesystem::remove_all(dir);
      std::filesystem::create_directories(dir);
      TK::config cfg;
      cfg.knobs.hi_water_bytes     = mb << 20;
      cfg.knobs.arena_budget_bytes = (mb << 20) * 2;  // footprint sensor tracks hi_water
      auto db = TK::database::open(dir, sal::open_mode::create_or_open, cfg);
      auto ws = db->start_write_session();

      // growing-key inserts; time the whole run AND the steady-state tail.
      auto t_all = Clock::now();
      auto t_ss  = t_all;
      for (uint64_t i = 0; i < TOTAL;)
      {
         if (i == ss_start)
            t_ss = Clock::now();
         auto           tx  = ws->start_transaction(0);
         const uint64_t end = std::min(TOTAL, i + BATCH);
         for (; i < end; ++i)
         {
            const uint64_t k = splitmix64(i);
            const uint64_t v = splitmix64(i ^ 0xDEADBEEFCAFEBABEull);
            tx.upsert(acct_key(k), enc_bal(v));
         }
         tx.commit();
      }
      const auto   t_end  = Clock::now();
      const double ins_ss = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       t_end - t_ss)
                                       .count()) /
                            double(TOTAL - ss_start);
      const double ins_all = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        t_end - t_all)
                                        .count()) /
                             double(TOTAL);

      // read pass (latest view, through the merge)
      std::mt19937_64 rng(0xF00D);
      uint64_t        found = 0;
      auto            rs    = db->start_read_session();
      auto            tr    = Clock::now();
      {
         auto cur = rs->create_cursor(0);
         for (uint64_t r = 0; r < READS; ++r)
            if (cur.get<std::string>(acct_key(splitmix64(rng() % TOTAL))))
               ++found;
      }
      const double rdns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     Clock::now() - tr)
                                     .count()) /
                          double(READS);

      printf("%8llu %12.0f %12.0f %10.0f %8.2f\n", (unsigned long long)mb, ins_ss, ins_all, rdns,
             dir_gb(dir));
      fflush(stdout);
      (void)found;
   }
   return 0;
}

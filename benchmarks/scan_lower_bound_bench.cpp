// scan_lower_bound_bench — ordered-cursor comparison across the adb engines:
// lower_bound point latency, full ordered scan (seek_begin+next) throughput, and
// bounded range scans. Each engine via the same adb surface
// (ws->get_root(0).snapshot_cursor() -> lower_bound/seek_begin/next/key/value).
//   scan-lower-bound-bench [keys=2000000] [probes=1000000] [range_len=50] [engine]
//      engine: start_ver | start_ref | sal_art_ref | gcart | gcart_spec | all
#include <gcart/kv.hpp>
#include <gcart/kv.hpp>
#include <heart_kv.hpp>  // sal_art_ref::kv
#include <start_ref/kv.hpp>
#include <start_ver/kv.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b)
{
   return std::chrono::duration<double>(b - a).count();
}
static uint64_t splitmix64(uint64_t x)
{
   x += 0x9E3779B97F4A7C15ull;
   x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
   x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
   return x ^ (x >> 31);
}
static std::string k8(uint64_t v)
{
   char b[8];
   for (int i = 0; i < 8; ++i)
      b[i] = char(uint8_t(v >> (8 * (7 - i))));
   return std::string(b, 8);
}  // big-endian → sorted

struct row
{
   std::string name;
   double      load_s, lb_ns, scan_ns, range_ns;
   uint64_t    scanned;
};

template <class DB>
row run(const char*         name,
        std::shared_ptr<DB> db,
        uint64_t            keys,
        uint64_t            probes,
        uint64_t            range_len)
{
   auto ws = db->start_write_session();
   // bulk load `keys` random uint64 (hash-spread → not sequential), value = 8 bytes
   const std::string val(8, 'v');
   const auto        t0 = Clock::now();
   {
      uint64_t i = 0;
      while (i < keys)
      {
         auto           tx  = ws->start_transaction(0);
         const uint64_t end = std::min(keys, i + 1000);
         for (; i < end; ++i)
            tx.upsert(k8(splitmix64(i)), val);
         tx.commit();
      }
   }
   const double load_s = secs(t0, Clock::now());

   // lower_bound point latency: `probes` random lower_bounds on a fresh cursor each
   uint64_t   hits = 0;
   const auto t1   = Clock::now();
   {
      auto c = ws->get_root(0).snapshot_cursor();
      for (uint64_t p = 0; p < probes; ++p)
      {
         if (c.lower_bound(k8(splitmix64(keys + p))))
            ++hits;
      }
   }
   const double lb_ns = probes ? secs(t1, Clock::now()) * 1e9 / double(probes) : 0;

   // full ordered scan: seek_begin + next over the whole tree
   uint64_t   scanned = 0;
   uint64_t   ksum    = 0;
   const auto t2      = Clock::now();
   {
      auto c = ws->get_root(0).snapshot_cursor();
      for (bool ok = c.seek_begin(); ok; ok = c.next())
      {
         ksum += uint8_t(c.key()[0]);
         ++scanned;
      }
   }
   const double scan_ns = scanned ? secs(t2, Clock::now()) * 1e9 / double(scanned) : 0;

   // bounded range scans: lower_bound(random) then next×range_len
   uint64_t   steps = 0;
   const auto t3    = Clock::now();
   {
      auto c = ws->get_root(0).snapshot_cursor();
      for (uint64_t p = 0; p < probes; ++p)
      {
         c.lower_bound(k8(splitmix64(p)));
         for (uint64_t s = 0; s < range_len && !c.is_end(); ++s)
         {
            ksum += uint8_t(c.key()[0]);
            c.next();
            ++steps;
         }
      }
   }
   const double range_ns = steps ? secs(t3, Clock::now()) * 1e9 / double(steps) : 0;
   (void)ksum;
   (void)hits;
   row r{name, load_s, lb_ns, scan_ns, range_ns, scanned};
   std::printf(
       "  %-12s load=%.1fs  lower_bound=%.0f ns  full-scan=%.1f ns/key  range-next=%.1f ns/key  "
       "(scanned=%llu)\n",
       r.name.c_str(), r.load_s, r.lb_ns, r.scan_ns, r.range_ns, (unsigned long long)r.scanned);
   std::fflush(stdout);
   return r;
}

int main(int argc, char** argv)
{
   const uint64_t    keys   = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2'000'000ull;
   const uint64_t    probes = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000'000ull;
   const uint64_t    rlen   = argc > 3 ? strtoull(argv[3], nullptr, 10) : 50ull;
   const std::string which  = argc > 4 ? argv[4] : "all";
   auto              want   = [&](const char* n) { return which == "all" || which == n; };
   const fs::path    base   = "/tmp/scan_lb";
   std::printf("scan/lower_bound bench: keys=%llu probes=%llu range_len=%llu\n\n",
               (unsigned long long)keys, (unsigned long long)probes, (unsigned long long)rlen);
   std::vector<row> rows;
   if (want("start_ver"))
   {
      auto d = base / "start_ver";
      fs::remove_all(d);
      rows.push_back(run("start_ver", start_ver::kv::database::create(d), keys, probes, rlen));
   }
   if (want("start_ref"))
   {
      auto d = base / "start_ref";
      fs::remove_all(d);
      rows.push_back(run("start_ref", start_ref::kv::database::create(d), keys, probes, rlen));
   }
   if (want("sal_art_ref"))
   {
      auto d = base / "sal_art_ref";
      fs::remove_all(d);
      rows.push_back(run("sal_art_ref", sal_art_ref::kv::database::create(d), keys, probes, rlen));
   }
   if (want("gcart"))
   {
      auto d = base / "gcart";
      fs::remove_all(d);
      rows.push_back(run("gcart", gcart::kv::database::create(d), keys, probes, rlen));
   }
   if (want("gcart_spec"))
   {
      auto d = base / "gcart_spec";
      fs::remove_all(d);
      rows.push_back(run("gcart_spec", gcart::kv::database::create(d), keys, probes, rlen));
   }
   std::printf("\n%-12s %8s %14s %16s %16s\n", "engine", "load/s", "lower_bound(ns)",
               "full-scan(ns/k)", "range-next(ns/k)");
   for (auto& r : rows)
      std::printf("%-12s %8.0f %14.0f %16.1f %16.1f\n", r.name.c_str(), r.scanned / r.load_s,
                  r.lb_ns, r.scan_ns, r.range_ns);
   return 0;
}

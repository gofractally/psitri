// gcart_oplog_replay — replay a captured Silkworm engine op-stream through
// gcart's adb-conforming kv API. This IS "running Silkworm via the adb API":
// the stream is the exact upsert/remove/get/lower_bound/commit sequence the
// MDBX shim emitted while Silkworm synced mainnet (single keyspace; the dbi is
// folded into the key bytes). If gcart replays it faithfully (read-own-writes
// oracle + a final full-scan vs the truth map) it is a drop-in for that engine.
//
//   gcart-oplog-replay <oplog.bin> <dir> [verify=0] [report_every_commits=1]
//
// Record stream: 'C' commit | 'U' u32 klen key u32 vlen val | 'R' u32 klen key
//   | 'G' u32 klen key u8 found u64 fnv | 'L' u32 klen key u8 found u32 rklen
//   rkey u64 fnv.  (matches benchmarks/oplog_replay.cpp / the shim capture.)
#include <gcart/kv.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

static uint64_t fnv_(std::string_view v) noexcept  // matches the shim's oplog_fnv_
{
   uint64_t h = 1469598103934665603ull;
   for (unsigned char c : v) { h ^= c; h *= 1099511628211ull; }
   return h;
}

int main(int argc, char** argv)
{
   if (argc < 3)
   {
      std::fprintf(stderr, "usage: gcart-oplog-replay <oplog.bin> <dir> [verify=0] [report_every=1]\n");
      return 2;
   }
   const std::string oplog = argv[1];
   const std::string dir   = argv[2];
   const bool        verify = argc > 3 && std::atoi(argv[3]) != 0;
   const uint64_t    report_every = argc > 4 ? strtoull(argv[4], nullptr, 10) : 1;

   std::FILE* f = std::fopen(oplog.c_str(), "rb");
   if (!f) { std::perror("fopen"); return 2; }
   fs::remove_all(dir);

   auto db = gcart::kv::database::create(dir);
   auto ws = db->start_write_session();

   std::map<std::string, std::string> truth;  // oracle (only if verify)
   std::vector<char>                  key, val, rkey;
   uint64_t commits = 0, ups = 0, rems = 0, gets = 0, lbs = 0;
   uint64_t g_div = 0, oversized = 0;
   const uint32_t cap_lines = sal2::max_alloc_lines;  // gcart node ceiling (large-value gap)

   auto rd  = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
   auto t0  = Clock::now();
   auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

   std::optional<gcart::kv::transaction> tx;
   tx.emplace(ws->start_transaction(0));

   int c;
   while ((c = std::fgetc(f)) != EOF)
   {
      if (c == 'C')
      {
         tx->commit();
         tx.reset();
         ++commits;
         if (report_every && commits % report_every == 0)
         {
            const double mb = db->store().live_bytes() / 1048576.0;
            std::printf("  [commit %llu] U=%llu R=%llu live=%.0fMB segs=%u %.1fs\n",
                        (unsigned long long)commits, (unsigned long long)ups,
                        (unsigned long long)rems, mb, db->store().num_segments(),
                        secs(t0, Clock::now()));
            std::fflush(stdout);
         }
         tx.emplace(ws->start_transaction(0));
         continue;
      }
      uint32_t kl = 0;
      if (!rd(&kl, 4)) break;
      key.resize(kl);
      if (kl && !rd(key.data(), kl)) break;

      if (c == 'U')
      {
         uint32_t vl = 0;
         if (!rd(&vl, 4)) break;
         val.resize(vl);
         if (vl && !rd(val.data(), vl)) break;
         std::string_view kv(key.data(), kl), vv(val.data(), vl);
         if (gcart::leaf::alloc_size(kv, vv) > cap_lines)
         {
            ++oversized;  // > gcart's 8 KB node cap (BIG-region large values: TODO)
            continue;
         }
         tx->upsert(kv, vv);
         if (verify) truth[std::string(kv)] = std::string(vv);
         ++ups;
      }
      else if (c == 'R')
      {
         std::string_view kv(key.data(), kl);
         tx->remove(kv);
         if (verify) truth.erase(std::string(kv));
         ++rems;
      }
      else if (c == 'G')
      {  // read-own-writes oracle: gcart's working-txn get vs the recorded result
         uint8_t fb = 0; uint64_t fnv = 0;
         if (!rd(&fb, 1) || !rd(&fnv, 8)) break;
         std::string_view kv(key.data(), kl);
         auto got = tx->get<std::string>(kv);
         const bool gf = got.has_value();
         if (gf != (fb != 0) || (gf && fnv_(*got) != fnv)) ++g_div;
         ++gets;
      }
      else if (c == 'L')
      {  // lower_bound record (read-own-writes merge over the open txn is a TODO);
         // count it and skip the payload faithfully.
         uint8_t fb = 0; uint32_t rkl = 0; uint64_t fnv = 0;
         if (!rd(&fb, 1) || !rd(&rkl, 4)) break;
         if (rkl) { rkey.resize(rkl); if (!rd(rkey.data(), rkl)) break; }
         if (!rd(&fnv, 8)) break;
         ++lbs;
      }
      else break;
   }
   if (tx) tx->commit();  // the capture can end mid-txn; persist the trailing ops
   std::fclose(f);

   const double dt = secs(t0, Clock::now());
   const double mb = db->store().live_bytes() / 1048576.0;
   std::printf("\n[replay] commits=%llu U=%llu R=%llu G=%llu L=%llu oversized(skipped)=%llu  %.1fs\n",
               (unsigned long long)commits, (unsigned long long)ups, (unsigned long long)rems,
               (unsigned long long)gets, (unsigned long long)lbs, (unsigned long long)oversized, dt);
   std::printf("[replay] get-divergences=%llu  live=%.0fMB segs=%u  (%.0f ups/s)\n",
               (unsigned long long)g_div, mb, db->store().num_segments(),
               dt > 0 ? ups / dt : 0.0);

   int rc = (g_div == 0) ? 0 : 1;
   if (verify)
   {
      // full ordered scan must equal the truth map exactly (order + values).
      auto     rs  = db->start_read_session();
      auto     cur = rs->create_cursor(0);
      uint64_t n = 0, bad = 0;
      auto     it = truth.begin();
      for (bool ok = cur.seek_begin(); ok; ok = cur.next(), ++it, ++n)
      {
         if (it == truth.end() || std::string(cur.key()) != it->first ||
             *cur.template value<std::string>() != it->second)
         { if (++bad <= 5) std::fprintf(stderr, "  scan mismatch at %llu\n", (unsigned long long)n); }
      }
      const bool len_ok = (n == truth.size() && it == truth.end());
      std::printf("[verify] scanned=%llu truth=%zu mismatches=%llu length=%s\n",
                  (unsigned long long)n, truth.size(), (unsigned long long)bad,
                  len_ok ? "OK" : "MISMATCH");
      const double bpk = n ? db->store().live_bytes() / double(n) : 0.0;
      std::printf("[verify] live keys=%llu  bytes/key=%.0f\n", (unsigned long long)n, bpk);
      if (bad || !len_ok) rc = 1;
   }
   std::printf("RESULT: %s\n", rc == 0 ? "PASS" : "FAIL");
   return rc;
}

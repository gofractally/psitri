// gcart_spec_oplog_replay — replay a captured Silkworm op-stream through the NEW
// spec engine (gcart::st::kv) and VALIDATE its reads against the recorded
// results. This is the read+write validation of the spec engine on the real
// mainnet-sync stream (the MDBX shim's capture); gcart_oplog_replay.cpp does the
// same for V1 gcart::kv. Every 'G' op carries Silkworm's observed result
// (found-byte + fnv value-hash); we do tx->get (read-own-writes over the open
// txn) and count divergences. 'L' (lower_bound) records are skipped faithfully
// (the spec cursor has no forward/range surface yet — that's the remaining gap).
//
//   gcart-spec-oplog-replay <oplog.bin> <dir> [report_every_commits=0] [verify_sample=0]
//
// These captures turned out to be write-only (no 'G'/'L' read records), so the
// recorded-read oracle has nothing to check. With verify_sample=N (>0) we DERIVE
// a read+write validation instead: track the final value of every Nth key (by
// hash) through the U/R stream, then point-read each back through a read cursor at
// the end and compare — validating the read path against the write history on the
// real mainnet stream, including large (BIG) values.
//
// Stream: 'C' | 'U' u32 klen key u32 vlen val | 'R' u32 klen key
//   | 'G' u32 klen key u8 found u64 fnv | 'L' u32 klen key u8 found u32 rklen rkey u64 fnv
#include <gcart/spec_kv.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

static uint64_t fnv_(std::string_view v) noexcept  // matches the shim's oplog_fnv_
{
   uint64_t h = 1469598103934665603ull;
   for (unsigned char c : v)
   {
      h ^= c;
      h *= 1099511628211ull;
   }
   return h;
}

int main(int argc, char** argv)
{
   if (argc < 3)
   {
      std::fprintf(stderr, "usage: gcart-spec-oplog-replay <oplog.bin> <dir> [report_every=0]\n");
      return 2;
   }
   const std::string oplog        = argv[1];
   const std::string dir          = argv[2];
   const uint64_t    report_every = argc > 3 ? strtoull(argv[3], nullptr, 10) : 0;
   const uint64_t    verify_sample =
       argc > 4 ? strtoull(argv[4], nullptr, 10) : 0;              // 0=off; else keep 1-in-N keys
   const bool skip_removes = argc > 5 && std::atoi(argv[5]) != 0;  // replay U-only subset (diag)
   std::unordered_map<std::string, std::string> truth;  // sampled key -> final value (verify mode)

   std::FILE* f = std::fopen(oplog.c_str(), "rb");
   if (!f)
   {
      std::perror("fopen");
      return 2;
   }
   fs::remove_all(dir);

   auto db = gcart::st::kv::database::create(dir);
   auto ws = db->start_write_session();

   std::vector<char> key, val, rkey;
   uint64_t          commits = 0, ups = 0, rems = 0, gets = 0, lbs = 0, g_div = 0;
   auto              rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
   auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

   std::optional<gcart::st::kv::transaction> tx;
   tx.emplace(ws->start_transaction(0));
   const auto t0 = Clock::now();

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
            std::printf("  [commit %llu] U=%llu R=%llu G=%llu div=%llu live=%.0fMB %.1fs\n",
                        (unsigned long long)commits, (unsigned long long)ups,
                        (unsigned long long)rems, (unsigned long long)gets,
                        (unsigned long long)g_div, db->live_bytes() / 1048576.0,
                        secs(t0, Clock::now()));
            std::fflush(stdout);
         }
         tx.emplace(ws->start_transaction(0));
         continue;
      }
      uint32_t kl = 0;
      if (!rd(&kl, 4))
         break;
      key.resize(kl);
      if (kl && !rd(key.data(), kl))
         break;

      if (c == 'U')
      {
         uint32_t vl = 0;
         if (!rd(&vl, 4))
            break;
         val.resize(vl);
         if (vl && !rd(val.data(), vl))
            break;
         std::string_view kv(key.data(), kl), vv(val.data(), vl);
         tx->upsert(kv, vv);
         if (verify_sample && (fnv_(kv) % verify_sample) == 0)
            truth[std::string(kv)] = std::string(vv);
         ++ups;
      }
      else if (c == 'R')
      {
         std::string_view kv(key.data(), kl);
         if (!skip_removes)
         {
            tx->remove(kv);
            if (verify_sample && (fnv_(kv) % verify_sample) == 0)
               truth.erase(std::string(kv));
         }
         ++rems;
      }
      else if (c == 'G')
      {  // read-own-writes oracle: spec get vs Silkworm's recorded result
         uint8_t  fb  = 0;
         uint64_t fnv = 0;
         if (!rd(&fb, 1) || !rd(&fnv, 8))
            break;
         auto       got = tx->get<std::string>(std::string_view(key.data(), kl));
         const bool gf  = got.has_value();
         if (gf != (fb != 0) || (gf && fnv_(*got) != fnv))
            ++g_div;
         ++gets;
      }
      else if (c == 'L')
      {  // range read — spec cursor has no forward surface yet; skip payload faithfully
         uint8_t  fb  = 0;
         uint32_t rkl = 0;
         uint64_t fnv = 0;
         if (!rd(&fb, 1) || !rd(&rkl, 4))
            break;
         if (rkl)
         {
            rkey.resize(rkl);
            if (!rd(rkey.data(), rkl))
               break;
         }
         if (!rd(&fnv, 8))
            break;
         ++lbs;
      }
      else
         break;
   }
   if (tx)
      tx->commit();
   std::fclose(f);

   const double dt = secs(t0, Clock::now());
   std::printf("\n[spec-replay] commits=%llu U=%llu R=%llu G=%llu L=%llu  %.1fs  (%.0f ups/s)\n",
               (unsigned long long)commits, (unsigned long long)ups, (unsigned long long)rems,
               (unsigned long long)gets, (unsigned long long)lbs, dt, dt > 0 ? ups / dt : 0.0);
   std::printf("[spec-replay] get-divergences=%llu / %llu gets   live=%.0fMB\n",
               (unsigned long long)g_div, (unsigned long long)gets, db->live_bytes() / 1048576.0);
   int rc = (g_div == 0) ? 0 : 1;

   // derived read+write validation: point-read every sampled key vs its truth value
   if (verify_sample)
   {
      auto     rs      = db->start_read_session();
      auto     cur     = rs->create_cursor(0);
      uint64_t checked = 0, bad = 0, big = 0;
      for (auto& [k, v] : truth)
      {
         auto got = cur.get<std::string>(k);
         if (!got || *got != v)
         {
            if (++bad <= 5)
               std::fprintf(stderr, "  verify mismatch key.len=%zu val.len=%zu got=%s\n", k.size(),
                            v.size(), got ? "diff" : "(none)");
         }
         if (v.size() > 2045)
            ++big;  // landed in the BIG region
         ++checked;
      }
      std::printf(
          "[spec-verify] sample=1/%llu  checked=%llu  mismatches=%llu  (big-value keys "
          "verified=%llu)\n",
          (unsigned long long)verify_sample, (unsigned long long)checked, (unsigned long long)bad,
          (unsigned long long)big);
      if (bad)
         rc = 1;
   }
   std::printf("RESULT: %s\n", rc == 0 ? "PASS" : "FAIL");
   return rc;
}

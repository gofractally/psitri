// bank comparison bench — the SAME transfer workload + balance-conservation
// check run on three engines back-to-back, so the composed pipeline can be
// read against its parts:
//   delta_art<start_ver, sal_art_ref>  the tiered pipeline (hot delta -> cold)
//   start_ver                          the hot engine ALONE (no tiering)
//   sal_art_ref                        the cold engine ALONE (direct to heart)
// One logical writer batches transfers; reader threads read the latest view
// concurrently; the run ends summing every account == N * initial.
//
//   bank-compare-bench [accounts=1e6] [transactions=5e6] [batch=1000]
//                      [readers=4] [dir=/tmp/bank_cmp]
//   env: DELTA_HI_WATER_MB, DELTA_ARENA_MB  (delta_art tier only)
#include <delta_art/tiered_kv.hpp>

#include <heart_dwal_engine.hpp>  // sal_art_ref::radix::dwal_engine (cold base)
#include <heart_kv.hpp>           // sal_art_ref::kv (standalone)
#include <start_ver/kv.hpp>       // start_ver::kv (hot, standalone)

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using TK    = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>;

namespace
{
   constexpr uint64_t INIT   = 1'000'000ull;
   constexpr uint64_t AMOUNT = 1;

   std::string acct_key(uint64_t i) { return std::string(reinterpret_cast<const char*>(&i), 8); }
   std::string enc_bal(uint64_t b) { return std::string(reinterpret_cast<const char*>(&b), 8); }
   uint64_t    dec_bal(std::string_view s)
   {
      uint64_t b = 0;
      std::memcpy(&b, s.data(), std::min(s.size(), sizeof(b)));
      return b;
   }
   uint64_t pick(std::mt19937_64& rng, uint64_t n)  // triangular: hot accounts
   {
      std::uniform_int_distribution<uint64_t> d(0, n / 2 - 1);
      return d(rng) + d(rng);
   }
   double secs(Clock::time_point a, Clock::time_point b)
   {
      return std::chrono::duration<double>(b - a).count();
   }

   struct row
   {
      const char* name;
      double      load, wtx, rd_ww, wtx_ww, rd_solo;
      bool        pass;
   };

   struct params
   {
      uint64_t N, NTX, BATCH, READERS;
   };

   template <class DB>
   row run_bank(const char* name, std::shared_ptr<DB> db, params p)
   {
      auto ws = db->start_write_session();
      row  r{name, 0, 0, 0, 0, 0, false};

      // bulk load
      {
         auto     t0 = Clock::now();
         uint64_t i  = 0;
         while (i < p.N)
         {
            auto           tx  = ws->start_transaction(0);
            const uint64_t end = std::min(p.N, i + p.BATCH);
            for (; i < end; ++i)
               tx.upsert(acct_key(i), enc_bal(INIT));
            tx.commit();
         }
         r.load = double(p.N) / secs(t0, Clock::now());
      }

      auto writer = [&](std::atomic<bool>* stop) -> uint64_t
      {
         std::mt19937_64 rng(0xBA11C0DE);
         uint64_t        done = 0;
         while (done < p.NTX && (!stop || !stop->load(std::memory_order_relaxed)))
         {
            auto           tx   = ws->start_transaction(0);
            const uint64_t bend = std::min(p.NTX, done + p.BATCH);
            for (; done < bend; ++done)
            {
               const uint64_t s = pick(rng, p.N), d = pick(rng, p.N);
               if (s == d)
                  continue;
               auto sv = tx.template get<std::string>(acct_key(s));
               auto dv = tx.template get<std::string>(acct_key(d));
               if (!sv || !dv)
                  continue;
               uint64_t sb = dec_bal(*sv), dbal = dec_bal(*dv);
               if (sb < AMOUNT)
                  continue;
               tx.upsert(acct_key(s), enc_bal(sb - AMOUNT));
               tx.upsert(acct_key(d), enc_bal(dbal + AMOUNT));
            }
            tx.commit();
         }
         return done;
      };

      // write-only
      {
         auto     t0 = Clock::now();
         uint64_t n  = writer(nullptr);
         r.wtx       = double(n) / secs(t0, Clock::now());
      }

      auto reader = [&](std::atomic<bool>* stop, std::atomic<uint64_t>* count)
      {
         auto            rs    = db->start_read_session();
         std::mt19937_64 rng(0xF00D);
         uint64_t        local = 0;
         while (!stop->load(std::memory_order_relaxed))
         {
            auto cur = rs->create_cursor(0);
            for (int b = 0; b < 1000; ++b)
               if (cur.template get<std::string>(acct_key(pick(rng, p.N))))
                  ++local;
         }
         count->fetch_add(local, std::memory_order_relaxed);
      };

      // readers WITH a concurrent writer
      {
         std::atomic<bool>        stop{false};
         std::atomic<uint64_t>    rc{0};
         std::vector<std::thread> rt;
         for (uint64_t i = 0; i < p.READERS; ++i)
            rt.emplace_back(reader, &stop, &rc);
         auto     t0 = Clock::now();
         uint64_t wn = writer(&stop);
         stop.store(true);
         for (auto& t : rt)
            t.join();
         const double dt = secs(t0, Clock::now());
         r.rd_ww  = double(rc.load()) / dt;
         r.wtx_ww = double(wn) / dt;
      }

      // readers SOLO
      {
         std::atomic<bool>        stop{false};
         std::atomic<uint64_t>    rc{0};
         std::vector<std::thread> rt;
         auto                     t0 = Clock::now();
         for (uint64_t i = 0; i < p.READERS; ++i)
            rt.emplace_back(reader, &stop, &rc);
         std::this_thread::sleep_for(std::chrono::seconds(2));
         stop.store(true);
         for (auto& t : rt)
            t.join();
         r.rd_solo = double(rc.load()) / secs(t0, Clock::now());
      }

      // balance conservation
      {
         auto     rs    = db->start_read_session();
         auto     cur   = rs->create_cursor(0);
         uint64_t total = 0;
         for (uint64_t i = 0; i < p.N; ++i)
            if (auto v = cur.template get<std::string>(acct_key(i)))
               total += dec_bal(*v);
         r.pass = (total == p.N * INIT);
      }
      return r;
   }
}  // namespace

int main(int argc, char** argv)
{
   params p;
   p.N       = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1'000'000ull;
   p.NTX     = argc > 2 ? strtoull(argv[2], nullptr, 10) : 5'000'000ull;
   p.BATCH   = argc > 3 ? strtoull(argv[3], nullptr, 10) : 1'000ull;
   p.READERS = argc > 4 ? strtoull(argv[4], nullptr, 10) : 4ull;
   const std::string base = argc > 5 ? argv[5] : "/tmp/bank_cmp";

   printf("bank compare: %llu accts x %llu, %llu transfers, batch %llu, %llu readers\n\n",
          (unsigned long long)p.N, (unsigned long long)INIT, (unsigned long long)p.NTX,
          (unsigned long long)p.BATCH, (unsigned long long)p.READERS);
   fflush(stdout);

   std::vector<row> rows;

   {  // delta_art<start_ver, sal_art_ref>
      std::filesystem::remove_all(base + "/tiered");
      std::filesystem::create_directories(base + "/tiered");
      TK::config cfg;
      if (const char* mb = getenv("DELTA_HI_WATER_MB"))
         cfg.knobs.hi_water_bytes = strtoull(mb, nullptr, 10) << 20;
      if (const char* ab = getenv("DELTA_ARENA_MB"))
         cfg.knobs.arena_budget_bytes = strtoull(ab, nullptr, 10) << 20;
      auto db = TK::database::open(base + "/tiered", sal::open_mode::create_or_open, cfg);
      rows.push_back(run_bank("delta_art(sv→sar)", db, p));
   }
   {  // start_ver standalone
      std::filesystem::remove_all(base + "/start_ver");
      auto db = start_ver::kv::database::create(base + "/start_ver");
      rows.push_back(run_bank("start_ver", db, p));
   }
   {  // sal_art_ref standalone
      std::filesystem::remove_all(base + "/sal_art_ref");
      auto db = sal_art_ref::kv::database::create(base + "/sal_art_ref");
      rows.push_back(run_bank("sal_art_ref", db, p));
   }

   printf("\n%-20s %12s %12s %14s %12s %14s  %s\n", "engine", "load/s", "wtx/s",
          "rd/s(+wr)", "wtx/s(+rd)", "rd/s(solo)", "balance");
   for (auto& r : rows)
      printf("%-20s %12.0f %12.0f %14.0f %12.0f %14.0f  %s\n", r.name, r.load, r.wtx, r.rd_ww,
             r.wtx_ww, r.rd_solo, r.pass ? "PASS" : "FAIL");
   return 0;
}

// delta_art bank bench — the classic transfer workload over the composed
// pipeline (start_ver hot delta -> sal_art_ref dwal cold base). One logical
// writer (delta_art serializes writes internally) batches transfers; reader
// threads read concurrently through the 2/3-way merge; the run ends with a
// balance-conservation check (sum of every account == N * initial).
//
//   delta_art-bank-bench [accounts=1e6] [transactions=1e7] [batch=1000]
//                        [readers=4] [dir=/tmp/delta_art_bank]
//   env: DELTA_HI_WATER_MB, DELTA_ARENA_MB
#include <delta_art/tiered_kv.hpp>

#include <heart_dwal_engine.hpp>  // sal_art_ref::radix::dwal_engine (cold base)
#include <start_ver/kv.hpp>       // start_ver::kv::engine (hot delta)

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

static std::string acct_key(uint64_t i)
{
   return std::string(reinterpret_cast<const char*>(&i), sizeof(i));
}
static std::string enc_bal(uint64_t b)
{
   return std::string(reinterpret_cast<const char*>(&b), sizeof(b));
}
static uint64_t dec_bal(std::string_view s)
{
   uint64_t b = 0;
   std::memcpy(&b, s.data(), std::min(s.size(), sizeof(b)));
   return b;
}
// Triangular pick (hot accounts) over [0, n/2): sum of two uniforms — matches
// the legacy bank bench's contention profile.
static uint64_t pick(std::mt19937_64& rng, uint64_t n)
{
   std::uniform_int_distribution<uint64_t> d(0, n / 2 - 1);
   return d(rng) + d(rng);
}
static double secs(Clock::time_point a, Clock::time_point b)
{
   return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv)
{
   const uint64_t N       = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1'000'000ull;
   const uint64_t NTX     = argc > 2 ? strtoull(argv[2], nullptr, 10) : 10'000'000ull;
   const uint64_t BATCH   = argc > 3 ? strtoull(argv[3], nullptr, 10) : 1'000ull;
   const uint64_t READERS = argc > 4 ? strtoull(argv[4], nullptr, 10) : 4ull;
   const char*    dirs    = argc > 5 ? argv[5] : "/tmp/delta_art_bank";
   const uint64_t INIT    = 1'000'000ull;
   const uint64_t AMOUNT  = 1;

   std::filesystem::path dir(dirs);
   std::filesystem::remove_all(dir);
   std::filesystem::create_directories(dir);

   TK::config cfg;
   if (const char* mb = getenv("DELTA_HI_WATER_MB"))
      cfg.knobs.hi_water_bytes = strtoull(mb, nullptr, 10) << 20;
   if (const char* ab = getenv("DELTA_ARENA_MB"))
      cfg.knobs.arena_budget_bytes = strtoull(ab, nullptr, 10) << 20;

   auto db = TK::database::open(dir, sal::open_mode::create_or_open, cfg);
   auto ws = db->start_write_session();

   printf("delta_art bank bench: %llu accounts x %llu initial, %llu transfers, batch %llu, "
          "%llu readers\n",
          (unsigned long long)N, (unsigned long long)INIT, (unsigned long long)NTX,
          (unsigned long long)BATCH, (unsigned long long)READERS);
   fflush(stdout);

   // ── bulk load ──
   {
      auto t0 = Clock::now();
      uint64_t i = 0;
      while (i < N)
      {
         auto tx = ws->start_transaction(0);
         const uint64_t end = std::min(N, i + BATCH);
         for (; i < end; ++i)
            tx.upsert(acct_key(i), enc_bal(INIT));
         tx.commit();
      }
      printf("bulk load: %.0f accts/sec\n", double(N) / secs(t0, Clock::now()));
      fflush(stdout);
   }

   // The single logical writer: NTX transfers in batched transactions.
   auto writer = [&](std::atomic<bool>* stop) -> uint64_t
   {
      std::mt19937_64 rng(0xBA11C0DE);
      uint64_t        done = 0;
      while (done < NTX && (!stop || !stop->load(std::memory_order_relaxed)))
      {
         auto           tx   = ws->start_transaction(0);
         const uint64_t bend = std::min(NTX, done + BATCH);
         for (; done < bend; ++done)
         {
            const uint64_t s = pick(rng, N), d = pick(rng, N);
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

   // ── write-only phase ──
   {
      auto     t0 = Clock::now();
      uint64_t n  = writer(nullptr);
      printf("write-only: %.0f tx/sec\n", double(n) / secs(t0, Clock::now()));
      fflush(stdout);
   }

   // reader: random account reads on the LATEST view through the merge.
   auto reader = [&](std::atomic<bool>* stop, std::atomic<uint64_t>* count)
   {
      auto            rs  = db->start_read_session();
      std::mt19937_64 rng(0xF00D);
      uint64_t        local = 0;
      while (!stop->load(std::memory_order_relaxed))
      {
         auto cur = rs->create_cursor(0);
         for (int b = 0; b < 1000; ++b)
            if (cur.template get<std::string>(acct_key(pick(rng, N))))
               ++local;
      }
      count->fetch_add(local, std::memory_order_relaxed);
   };

   // ── readers WITH a concurrent writer ──
   {
      std::atomic<bool>     stop{false};
      std::atomic<uint64_t> rc{0};
      std::vector<std::thread> rt;
      for (uint64_t r = 0; r < READERS; ++r)
         rt.emplace_back(reader, &stop, &rc);
      auto     t0 = Clock::now();
      uint64_t wn = writer(&stop);  // a fresh batch of transfers under read load
      stop.store(true);
      for (auto& t : rt)
         t.join();
      const double dt = secs(t0, Clock::now());
      printf("reader (with writer): %.0f reads/sec  [writer %.0f tx/sec]\n",
             double(rc.load()) / dt, double(wn) / dt);
      fflush(stdout);
   }

   // ── readers SOLO ──
   {
      std::atomic<bool>     stop{false};
      std::atomic<uint64_t> rc{0};
      std::vector<std::thread> rt;
      auto t0 = Clock::now();
      for (uint64_t r = 0; r < READERS; ++r)
         rt.emplace_back(reader, &stop, &rc);
      std::this_thread::sleep_for(std::chrono::seconds(2));
      stop.store(true);
      for (auto& t : rt)
         t.join();
      printf("reader (solo):        %.0f reads/sec\n", double(rc.load()) / secs(t0, Clock::now()));
      fflush(stdout);
   }

   // ── balance conservation ──
   {
      auto     rs    = db->start_read_session();
      auto     cur   = rs->create_cursor(0);
      uint64_t total = 0, seen = 0;
      for (uint64_t i = 0; i < N; ++i)
         if (auto v = cur.template get<std::string>(acct_key(i)))
         {
            total += dec_bal(*v);
            ++seen;
         }
      const uint64_t expect = N * INIT;
      printf("balance: %llu (expected %llu over %llu/%llu accts)  %s\n",
             (unsigned long long)total, (unsigned long long)expect, (unsigned long long)seen,
             (unsigned long long)N, total == expect ? "PASS" : "FAIL");
      return total == expect ? 0 : 1;
   }
}

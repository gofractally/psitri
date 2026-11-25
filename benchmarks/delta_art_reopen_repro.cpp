// Minimal reopen race repro: open a delta_art tiered DB, populate (flush to the
// cold base), CLOSE, then REOPEN + write repeatedly — the path the Silkworm node
// hits on startup (open schema -> close -> open sync -> write) that the steady-
// state bank bench never exercises. Run under ThreadSanitizer.
#include <delta_art/tiered_kv.hpp>
#include <psitri/dwal/kv_engine.hpp>
#include <sal_art_ref/engine.hpp>
#include <start_ver/kv.hpp>       // start_ver::kv::engine (hot delta)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

using TK = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::engine>;

static void write_some(TK::database& db, uint64_t base, uint64_t n)
{
   auto ws = db.start_write_session();
   for (uint64_t i = 0; i < n; i += 1000)
   {
      auto tx = ws->start_transaction(0);
      for (uint64_t j = i; j < i + 1000 && j < n; ++j)
      {
         char     k[24];
         int      kn = std::snprintf(k, sizeof k, "key/%llu", (unsigned long long)(base + j));
         std::string v(64, char('a' + (j & 15)));
         tx.upsert(std::string_view(k, kn), std::string_view(v.data(), v.size()));
      }
      tx.commit();
   }
}

int main(int argc, char** argv)
{
   const std::string dir    = argc > 1 ? argv[1] : "/tmp/da_reopen_repro";
   const uint64_t    N      = argc > 2 ? strtoull(argv[2], nullptr, 10) : 60000;
   const int         rounds = argc > 3 ? atoi(argv[3]) : 6;
   std::filesystem::remove_all(dir);

   auto open = [&]() {
      TK::config cfg;
      cfg.knobs.hi_water_bytes = 1u << 20;  // force frequent flushes -> cold-base churn
      return TK::database::open(dir, sal::open_mode::create_or_open, cfg);
   };

   { auto db = open(); write_some(*db, 0, N); }  // populate + CLEAN close (dtor)

   for (int r = 0; r < rounds; ++r)              // reopen + write (recovery + drain + writes)
   {
      auto db = open();
      write_some(*db, uint64_t(r + 1) * N, N);
      std::printf("round %d ok\n", r);
   }
   std::printf("ALL OK\n");
   return 0;
}

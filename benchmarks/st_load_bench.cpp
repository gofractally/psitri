// Focused gcart (spec engine) load profile: sequential uint64 keys, 8-byte
// values, batched transactions — the bank LOAD phase in isolation so `sample`
// attributes time to the spec engine alone (not the 6 other bank engines).
//   st-load-bench [keys=2000000] [batch=1000] [dir=/tmp/st_load]
#include <gcart/spec_kv.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv)
{
   const uint64_t    N     = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2'000'000ull;
   const uint64_t    BATCH = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000ull;
   const std::string dir   = argc > 3 ? argv[3] : "/tmp/st_load";

   std::filesystem::remove_all(dir);
   auto db = gcart::kv::database::create(dir);
   auto ws = db->start_write_session();

   const uint64_t val = 1'000'000ull;
   auto           t0  = Clock::now();
   uint64_t       i   = 0;
   while (i < N)
   {
      auto           tx  = ws->start_transaction(0);
      const uint64_t end = std::min(N, i + BATCH);
      for (; i < end; ++i)
      {
         std::string k(reinterpret_cast<const char*>(&i), 8);
         std::string v(reinterpret_cast<const char*>(&val), 8);
         tx.upsert(k, v);
      }
      tx.commit();
   }
   const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
   printf("st-load: %llu keys in %.3fs = %.0f keys/s, %.1f ns/key\n",
          (unsigned long long)N, dt, double(N) / dt, dt * 1e9 / double(N));
   return 0;
}

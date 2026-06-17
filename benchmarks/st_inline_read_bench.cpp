// Focused inline-leaf READ microbench for gcart::st (nested_setlist). Inserts
// PAIRS of keys that each form a surviving 2-leaf node (base[7] + 'a'/'b'); with
// nested ON each pair is one cl1 line holding two INLINE leaves (read = 1 line),
// with nested OFF (GCART_NO_NESTED=1) it's a flat setlist + 2 SAL leaves (read =
// 2 lines). Times random point reads so we can see whether eliminating the SAL
// cacheline makes inline reads FASTER (as it should) — Dan's question.
//   st-inline-read-bench [pairs=2000000] [readloops=4] [dir=/tmp/st_inl]
#include <gcart/spec_kv.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::string mk(uint64_t i, char suf)
{
   uint64_t h = (i + 0x9E3779B97F4A7C15ull) * 0xC2B2AE3D27D4EB4Full;  // spread the base
   char b[8];
   std::memcpy(b, &h, 7);  // 7-byte base
   b[7] = suf;             // divergent last byte
   return std::string(b, 8);
}

int main(int argc, char** argv)
{
   const uint64_t    pairs = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2'000'000ull;
   const uint64_t    loops = argc > 2 ? strtoull(argv[2], nullptr, 10) : 4ull;
   const std::string dir   = argc > 3 ? argv[3] : "/tmp/st_inl";

   std::filesystem::remove_all(dir);
   auto db = gcart::st::kv::database::create(dir);
   auto ws = db->start_write_session();

   const std::string val(8, 'V');
   {
      uint64_t i = 0;
      while (i < pairs)
      {
         auto           tx  = ws->start_transaction(0);
         const uint64_t end = std::min(pairs, i + 1000);
         for (; i < end; ++i) { tx.upsert(mk(i, 'a'), val); tx.upsert(mk(i, 'b'), val); }
         tx.commit();
      }
   }
   const uint64_t nkeys = pairs * 2;
   printf("inserted %llu keys, %.1f B/key (%s)\n", (unsigned long long)nkeys,
          double(db->live_bytes()) / double(nkeys),
          getenv("GCART_NO_NESTED") ? "nested OFF" : "nested ON");
   fflush(stdout);

   // timed random point reads
   auto     rs = db->start_read_session();
   auto     cur = rs->create_cursor(0);
   uint64_t hits = 0;
   auto     t0 = Clock::now();
   for (uint64_t l = 0; l < loops; ++l)
   {
      uint64_t r = 0x12345;
      for (uint64_t k = 0; k < nkeys; ++k)
      {
         r = r * 6364136223846793005ull + 1442695040888963407ull;
         const uint64_t i = (r >> 17) % pairs;
         const char     s = (r & 1) ? 'a' : 'b';
         if (cur.get<std::string>(mk(i, s))) ++hits;
      }
   }
   const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
   const uint64_t reads = loops * nkeys;
   printf("reads: %llu in %.3fs = %.0f reads/s, %.1f ns/read (hits=%llu)\n",
          (unsigned long long)reads, dt, double(reads) / dt, dt * 1e9 / double(reads),
          (unsigned long long)hits);
   return 0;
}

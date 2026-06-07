// Apples-to-apples psitri baseline for the heart sal_radix_tree benchmark:
// 1M hash-derived 8-byte keys → 8-byte values, single writer, insert then find.
// Keys are key[i] = hash(i) (no input array, no value array); the find pass
// recomputes hash(i) and checks the value is i — same honest scheme as the
// heart bench (libraries/heart/test/sal_radix_tree_tests.cpp).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

#include <hash/xxhash.h>

#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_session_impl.hpp>

static uint64_t hkey(uint64_t i) noexcept { return XXH3_64bits(&i, sizeof(i)); }

int main(int argc, char** argv)
{
   const uint64_t N    = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 1'000'000;
   const uint32_t root = 0;

   auto path = std::filesystem::temp_directory_path() / "psitri_radix_baseline";
   std::filesystem::remove_all(path);

   psitri::runtime_config cfg;
   cfg.sync_mode               = sal::sync_type::none;
   cfg.write_protect_on_commit = false;
   cfg.checksum_on_commit      = false;

   auto db  = psitri::database::create(path.string(), cfg);
   auto ses = db->start_write_session();

   auto now = [] { return std::chrono::steady_clock::now(); };
   auto ns  = [](auto a, auto b)
   { return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };

   // ---- insert (one transaction, single writer = the unique-mode analogue) ----
   auto t0 = now();
   {
      auto tx = ses->start_transaction(root);
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t k = hkey(i);
         tx.insert(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)),
                   psitri::value_view(reinterpret_cast<const char*>(&i), sizeof(i)));
      }
      tx.commit();
   }
   auto t1 = now();

   // ---- find ----
   uint64_t checksum = 0;
   auto     t2       = now();
   {
      auto        rootp = ses->get_root(root);
      std::string val;
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t       k = hkey(i);
         psitri::cursor cur(rootp);
         val.clear();
         if (cur.get(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)), &val) >= 0 &&
             val.size() == sizeof(uint64_t))
         {
            uint64_t v;
            std::memcpy(&v, val.data(), sizeof(v));
            if (v == i)
               checksum += v;
         }
      }
   }
   auto t3 = now();

   const uint64_t expect = N * (N - 1) / 2;
   std::printf("\n  psitri (1 tx)  : insert %6.1f ns/op   find %6.1f ns/op   [%s]\n",
               double(ns(t0, t1)) / double(N), double(ns(t2, t3)) / double(N),
               checksum == expect ? "OK" : "CHECKSUM MISMATCH");
   std::fflush(stdout);
   return checksum == expect ? 0 : 1;
}

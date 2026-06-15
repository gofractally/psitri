// churn_leak — reproduce the cold-base bloat with a controlled overwrite workload.
//
// Insert N keys, then OVERWRITE all N keys K rounds (key count stays fixed), then
// REMOVE all N. After each phase, report on-disk footprint. A healthy COW engine
// plateaus (superseded nodes freed + segments recycled); a leaking one grows
// ~linearly with rounds. Localizes the leak: run on sal_art_ref (cold base alone,
// plain COW overwrite) vs delta_art (full tiered pipeline incl. drain/retire).
//
//   churn-leak <engine> [N=1000000] [rounds=40] [dir=/tmp/churn]
//   engine = sal_art_ref | delta_art
#include <delta_art/tiered_kv.hpp>
#include <heart_dwal_engine.hpp>
#include <heart_kv.hpp>
#include <start_ver/kv.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using TK     = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>;

static uint64_t splitmix64(uint64_t x)
{
   x += 0x9E3779B97F4A7C15ull;
   x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
   x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
   return x ^ (x >> 31);
}
static std::string k8(uint64_t v) { return std::string(reinterpret_cast<const char*>(&v), 8); }
static double      dir_gb(const fs::path& p)
{
   uint64_t        b = 0;
   std::error_code ec;
   for (auto& e : fs::recursive_directory_iterator(p, ec))
      if (e.is_regular_file(ec))
         b += e.file_size(ec);
   return double(b) / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char** argv)
{
   const std::string eng    = argc > 1 ? argv[1] : "delta_art";
   const uint64_t    N      = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000'000ull;
   const uint32_t    rounds = argc > 3 ? uint32_t(strtoul(argv[3], nullptr, 10)) : 40;
   const std::string dir    = (argc > 4 ? std::string(argv[4]) : "/tmp/churn") + "/" + eng;

   fs::remove_all(dir);
   fs::create_directories(dir);
   printf("churn-leak engine=%s N=%llu rounds=%u dir=%s\n", eng.c_str(),
          (unsigned long long)N, rounds, dir.c_str());
   printf("%-8s %10s %12s\n", "phase", "round", "footprint_GB");
   fflush(stdout);

   auto report = [&](const char* phase, long round) {
      printf("%-8s %10ld %12.3f\n", phase, round, dir_gb(dir));
      fflush(stdout);
   };

   if (eng == "delta_art")
   {
      auto db = TK::database::open(dir, sal::open_mode::create_or_open);
      auto ws = db->start_write_session();
      // insert
      for (uint64_t i = 0; i < N;)
      {
         auto tx = ws->start_transaction(0);
         for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
            tx.upsert(k8(splitmix64(i)), k8(i));
         tx.commit();
      }
      db->flush_now();
      report("insert", 0);
      // overwrite rounds
      for (uint32_t r = 1; r <= rounds; ++r)
      {
         for (uint64_t i = 0; i < N;)
         {
            auto tx = ws->start_transaction(0);
            for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
               tx.upsert(k8(splitmix64(i)), k8(i ^ (uint64_t(r) << 40)));
            tx.commit();
         }
         db->flush_now();
         report("overwrite", r);
      }
      // remove all
      for (uint64_t i = 0; i < N;)
      {
         auto tx = ws->start_transaction(0);
         for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
            tx.remove(k8(splitmix64(i)));
         tx.commit();
      }
      db->flush_now();
      report("removeall", -1);
   }
   else if (eng == "sal_art_ref")
   {
      auto db = sal_art_ref::kv::database::create(dir);
      auto ws = db->start_write_session();
      for (uint64_t i = 0; i < N;)
      {
         auto tx = ws->start_transaction(0);
         for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
            tx.upsert(k8(splitmix64(i)), k8(i));
         tx.commit();
      }
      report("insert", 0);
      for (uint32_t r = 1; r <= rounds; ++r)
      {
         for (uint64_t i = 0; i < N;)
         {
            auto tx = ws->start_transaction(0);
            for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
               tx.upsert(k8(splitmix64(i)), k8(i ^ (uint64_t(r) << 40)));
            tx.commit();
         }
         std::this_thread::sleep_for(std::chrono::milliseconds(300));  // let compactor free
         report("overwrite", r);
      }
      for (uint64_t i = 0; i < N;)
      {
         auto tx = ws->start_transaction(0);
         for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
            tx.remove(k8(splitmix64(i)));
         tx.commit();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      report("removeall", -1);
   }
   else
   {
      fprintf(stderr, "unknown engine: %s (sal_art_ref|delta_art)\n", eng.c_str());
      return 2;
   }

   printf("\nfinal footprint %.3f GB  (clean close ...)\n", dir_gb(dir));
   fflush(stdout);
   return 0;
}

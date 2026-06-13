// memory + recovery comparison — ONE engine per process (peak RSS is
// process-wide, so each engine must run isolated). Loads N keys, reports peak
// RSS + on-disk footprint, then CLEANLY closes and REOPENS, timing the reopen
// (recovery) and verifying a key sample survived.
//
//   mem-recovery-bench <engine> [keys=20e6] [dir=/tmp/memrec]
//   engine = start_ver | sal_art_ref | delta_art
//
// Recovery model note: these are COW engines with NO WAL — the committed root
// is the durable state, so a clean reopen is just mmap + root-table read (no
// replay). delta_art additionally re-drains a dirty warm slot (root 1) on
// open. start_ref is excluded: its per-line refcounts are "derived, never
// synced" (arena.hpp), so a crash needs a refcount REBUILD from the committed
// roots — a different recovery path than start_ver's stamp-in-root-table.
#include <delta_art/tiered_kv.hpp>
#include <heart_dwal_engine.hpp>
#include <heart_kv.hpp>
#include <start_ver/kv.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;
using TK    = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>;

namespace
{
   uint64_t splitmix64(uint64_t x)
   {
      x += 0x9E3779B97F4A7C15ull;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
      return x ^ (x >> 31);
   }
   std::string k8(uint64_t v) { return std::string(reinterpret_cast<const char*>(&v), 8); }
   double      rss_gb()
   {
      rusage ru{};
      getrusage(RUSAGE_SELF, &ru);
      return double(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
   }
   double dir_gb(const std::filesystem::path& p)
   {
      uint64_t        b = 0;
      std::error_code ec;
      for (auto& e : std::filesystem::recursive_directory_iterator(p, ec))
         if (e.is_regular_file(ec))
            b += e.file_size(ec);
      return double(b) / (1024.0 * 1024.0 * 1024.0);
   }
   double secs(Clock::time_point a, Clock::time_point b)
   {
      return std::chrono::duration<double>(b - a).count();
   }

   // Load N keys via a fresh db, return after a CLEAN close (db dtor). Then a
   // second open times the reopen + verifies a sample. Templated on a make(dir,
   // fresh) factory + a ws-from-db + read-from-db.
   template <class Open, class Load, class Verify>
   void run(const char* name, const std::string& dir, uint64_t N, Open open, Load load,
            Verify verify)
   {
      std::filesystem::remove_all(dir);
      std::filesystem::create_directories(dir);
      double load_rss = 0, file_gb = 0;
      {
         auto db = open(dir);
         auto t0 = Clock::now();
         load(db, N);
         const double load_s = secs(t0, Clock::now());
         load_rss            = rss_gb();
         file_gb             = dir_gb(dir);
         printf("%-14s load %llu in %.1fs  peak RSS %.2f GB  file %.2f GB\n", name,
                (unsigned long long)N, load_s, load_rss, file_gb);
         fflush(stdout);
      }  // CLEAN close (dtor joins GC, flushes)
      // reopen + verify
      auto   t0 = Clock::now();
      auto   db = open(dir);
      double reopen_s = secs(t0, Clock::now());
      const bool ok = verify(db, N);
      printf("%-14s reopen %.3fs  verify %s\n", name, reopen_s, ok ? "OK" : "FAIL");
      fflush(stdout);
   }
}  // namespace

int main(int argc, char** argv)
{
   const std::string eng = argc > 1 ? argv[1] : "start_ver";
   const uint64_t    N   = argc > 2 ? strtoull(argv[2], nullptr, 10) : 20'000'000ull;
   const std::string dir = (argc > 3 ? std::string(argv[3]) : "/tmp/memrec") + "/" + eng;

   auto sample_ok = [N](auto& cur) {
      // 1000 spaced keys must all be present
      for (uint64_t i = 0; i < 1000; ++i)
         if (!cur.template get<std::string>(k8(splitmix64((i * N / 1000) % N))))
            return false;
      return true;
   };

   if (eng == "start_ver")
   {
      run(
          "start_ver", dir, N,
          [](const std::string& d) { return start_ver::kv::database::create(d); },
          [](auto& db, uint64_t n) {
             auto ws = db->start_write_session();
             for (uint64_t i = 0; i < n;)
             {
                auto tx = ws->start_transaction(0);
                for (uint64_t e = std::min(n, i + 1000); i < e; ++i)
                   tx.upsert(k8(splitmix64(i)), k8(i));
                tx.commit();
             }
          },
          [&](auto& db, uint64_t) {
             auto rs  = db->start_read_session();
             auto cur = rs->create_cursor(0);
             return sample_ok(cur);
          });
   }
   else if (eng == "sal_art_ref")
   {
      run(
          "sal_art_ref", dir, N,
          [](const std::string& d) { return sal_art_ref::kv::database::create(d); },
          [](auto& db, uint64_t n) {
             auto ws = db->start_write_session();
             for (uint64_t i = 0; i < n;)
             {
                auto tx = ws->start_transaction(0);
                for (uint64_t e = std::min(n, i + 1000); i < e; ++i)
                   tx.upsert(k8(splitmix64(i)), k8(i));
                tx.commit();
             }
          },
          [&](auto& db, uint64_t) {
             auto rs  = db->start_read_session();
             auto cur = rs->create_cursor(0);
             return sample_ok(cur);
          });
   }
   else if (eng == "delta_art")
   {
      run(
          "delta_art", dir, N,
          [](const std::string& d) {
             return TK::database::open(d, sal::open_mode::create_or_open);
          },
          [](auto& db, uint64_t n) {
             auto ws = db->start_write_session();
             for (uint64_t i = 0; i < n;)
             {
                auto tx = ws->start_transaction(0);
                for (uint64_t e = std::min(n, i + 1000); i < e; ++i)
                   tx.upsert(k8(splitmix64(i)), k8(i));
                tx.commit();
             }
          },
          [&](auto& db, uint64_t) {
             auto rs  = db->start_read_session();
             auto cur = rs->create_cursor(0);
             return sample_ok(cur);
          });
   }
   else
   {
      fprintf(stderr, "unknown engine: %s (start_ver|sal_art_ref|delta_art)\n", eng.c_str());
      return 2;
   }
   return 0;
}

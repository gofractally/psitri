// oplog_compare_bench — replay the SAME captured Silkworm op-stream through each
// standalone adb engine and compare wall-time + on-disk footprint. This is the
// "running Silkworm via the adb API" workload measured head-to-head.
//
//   oplog-compare-bench <oplog.bin> <dir-base> [engine]
//      engine (optional): gcart | start_ver | start_ref | sal_art_ref | all
//   (run one engine per process to isolate footprint/RSS and survive a crash.)
#include <gcart/kv.hpp>
#include <gcart/spec_kv.hpp>  // gcart::st::kv (the new spec engine)
#include <start_ver/kv.hpp>
#include <start_ref/kv.hpp>
#include <heart_kv.hpp>  // sal_art_ref::kv

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

static double secs(Clock::time_point a, Clock::time_point b)
{
   return std::chrono::duration<double>(b - a).count();
}
static double dir_gb(const fs::path& p)
{
   uint64_t        b = 0;
   std::error_code ec;
   for (auto& e : fs::recursive_directory_iterator(p, ec))
      if (e.is_regular_file(ec)) b += e.file_size(ec);
   return double(b) / (1024.0 * 1024.0 * 1024.0);
}

struct row { std::string name; double secs, ups_s; uint64_t ups, rems, commits; double gb; };

template <class DB>
row replay(const char* name, const std::string& oplog, const fs::path& dir, std::shared_ptr<DB> db)
{
   auto       ws = db->start_write_session();
   std::FILE* f  = std::fopen(oplog.c_str(), "rb");
   if (!f) { std::perror("fopen"); std::exit(2); }
   std::vector<char> key, val, rkey;
   auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
   uint64_t ups = 0, rems = 0, commits = 0;

   using TX = decltype(ws->start_transaction(0u));
   std::optional<TX> tx;
   tx.emplace(ws->start_transaction(0u));
   const auto t0 = Clock::now();
   int c;
   while ((c = std::fgetc(f)) != EOF)
   {
      if (c == 'C') { tx->commit(); tx.reset(); ++commits; tx.emplace(ws->start_transaction(0u)); continue; }
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
         tx->upsert(std::string_view(key.data(), kl), std::string_view(val.data(), vl));
         ++ups;
      }
      else if (c == 'R') { tx->remove(std::string_view(key.data(), kl)); ++rems; }
      else if (c == 'G') { uint8_t fb = 0; uint64_t fnv = 0; if (!rd(&fb, 1) || !rd(&fnv, 8)) break; }
      else if (c == 'L')
      {
         uint8_t fb = 0; uint32_t rkl = 0; uint64_t fnv = 0;
         if (!rd(&fb, 1) || !rd(&rkl, 4)) break;
         if (rkl) { rkey.resize(rkl); if (!rd(rkey.data(), rkl)) break; }
         if (!rd(&fnv, 8)) break;
      }
      else break;
   }
   if (tx) tx->commit();
   const double dt = secs(t0, Clock::now());
   std::fclose(f);
   row r{name, dt, dt > 0 ? ups / dt : 0.0, ups, rems, commits, dir_gb(dir)};
   std::printf("  %-12s  %8.1fs  %10.0f ups/s  U=%llu R=%llu C=%llu  disk=%.1fGB\n",
               r.name.c_str(), r.secs, r.ups_s, (unsigned long long)r.ups,
               (unsigned long long)r.rems, (unsigned long long)r.commits, r.gb);
   std::fflush(stdout);
   return r;
}

int main(int argc, char** argv)
{
   if (argc < 3)
   {
      std::fprintf(stderr, "usage: oplog-compare-bench <oplog.bin> <dir-base> [gcart|start_ver|start_ref|sal_art_ref|all]\n");
      return 2;
   }
   const std::string oplog = argv[1];
   const fs::path    base  = argv[2];
   const std::string which = argc > 3 ? argv[3] : "all";
   auto want = [&](const char* n) { return which == "all" || which == n; };

   std::printf("oplog-compare: %s\n\n", oplog.c_str());
   std::vector<row> rows;

   if (want("start_ver"))
   { auto d = base / "start_ver"; fs::remove_all(d); rows.push_back(replay("start_ver", oplog, d, start_ver::kv::database::create(d))); }
   if (want("start_ref"))
   { auto d = base / "start_ref"; fs::remove_all(d); rows.push_back(replay("start_ref", oplog, d, start_ref::kv::database::create(d))); }
   if (want("gcart"))
   { auto d = base / "gcart"; fs::remove_all(d); rows.push_back(replay("gcart", oplog, d, gcart::kv::database::create(d))); }
   if (want("gcart_spec"))
   { auto d = base / "gcart_spec"; fs::remove_all(d); rows.push_back(replay("gcart_spec", oplog, d, gcart::st::kv::database::create(d))); }
   if (want("sal_art_ref"))
   { auto d = base / "sal_art_ref"; fs::remove_all(d); rows.push_back(replay("sal_art_ref", oplog, d, sal_art_ref::kv::database::create(d))); }

   std::printf("\n%-12s %10s %14s %10s\n", "engine", "secs", "ups/s", "disk(GB)");
   for (auto& r : rows)
      std::printf("%-12s %10.1f %14.0f %10.1f\n", r.name.c_str(), r.secs, r.ups_s, r.gb);
   return 0;
}

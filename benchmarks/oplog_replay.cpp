// oplog_replay — replay a captured Silkworm engine op-stream into delta_art and
// track on-disk footprint vs live keys. Reproduces the real workload (variable
// namespaced keys, interleaved upsert/remove, real commit batching) standalone —
// no Silkworm. The shim captures these via MDBX_OPLOG=<path>.
//
// Format: 'U' kl[4] key[kl] vl[4] val[vl] | 'R' kl[4] key[kl] | 'C' (commit)
//
//   oplog-replay <oplog.bin> <dir> [report_every_commits=200]
#include <delta_art/tiered_kv.hpp>
#include <heart_dwal_engine.hpp>
#include <start_ver/kv.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using TK     = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>;

static double dir_gb(const fs::path& p)
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
   if (argc < 3)
   {
      fprintf(stderr, "usage: oplog-replay <oplog.bin> <dir> [report_every_commits=200]\n");
      return 2;
   }
   const std::string oplog  = argv[1];
   const std::string dir    = argv[2];
   const uint64_t    report = argc > 3 ? strtoull(argv[3], nullptr, 10) : 200;
   const std::string syncs  = argc > 4 ? argv[4] : "fsync";
   const sal::sync_type sync =
       syncs == "none"    ? sal::sync_type::none
     : syncs == "async"   ? sal::sync_type::msync_async
     : syncs == "msync"   ? sal::sync_type::msync_sync
     : syncs == "full"    ? sal::sync_type::full
                          : sal::sync_type::fsync;

   fs::remove_all(dir);
   fs::create_directories(dir);
   std::FILE* f = std::fopen(oplog.c_str(), "rb");
   if (!f)
   {
      fprintf(stderr, "cannot open %s\n", oplog.c_str());
      return 1;
   }
   std::setvbuf(f, nullptr, _IOFBF, 1 << 22);

   TK::config cfg;
   cfg.commit_sync = sync;
   auto db = TK::database::open(dir, sal::open_mode::create_or_open, cfg);
   auto ws = db->start_write_session();

   printf("oplog-replay %s -> %s  (commit_sync=%s)\n", oplog.c_str(), dir.c_str(), syncs.c_str());
   printf("%12s %12s %10s %10s %12s %12s\n", "commits", "ops", "U", "R", "base_GB", "delta_GB");
   fflush(stdout);

   std::vector<char>              key, val;
   uint64_t                       commits = 0, ops = 0, ups = 0, rems = 0;
   std::optional<TK::transaction> tx;
   tx.emplace(ws->start_transaction(0));
   int  c;
   auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };

   while ((c = std::fgetc(f)) != EOF)
   {
      if (c == 'C')
      {
         if (tx)
            tx->commit();
         tx.reset();
         ++commits;
         if (commits % report == 0)
         {
            printf("%12llu %12llu %10llu %10llu %12.3f %12.3f\n", (unsigned long long)commits,
                   (unsigned long long)ops, (unsigned long long)ups, (unsigned long long)rems,
                   dir_gb(fs::path(dir) / "base"), dir_gb(fs::path(dir) / "delta"));
            fflush(stdout);
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
      ++ops;
      if (c == 'U')
      {
         uint32_t vl = 0;
         if (!rd(&vl, 4))
            break;
         val.resize(vl);
         if (vl && !rd(val.data(), vl))
            break;
         tx->upsert(std::string_view(key.data(), kl), std::string_view(val.data(), vl));
         ++ups;
      }
      else if (c == 'R')
      {
         tx->remove(std::string_view(key.data(), kl));
         ++rems;
      }
      else
      {
         break;  // unknown op — corrupt/eof
      }
   }
   if (tx)
      tx->commit();
   db->flush_now();
   std::fclose(f);

   printf("\nDONE commits=%llu ops=%llu (U=%llu R=%llu)\n", (unsigned long long)commits,
          (unsigned long long)ops, (unsigned long long)ups, (unsigned long long)rems);
   printf("final footprint base=%.3f GB delta=%.3f GB total=%.3f GB\n",
          dir_gb(fs::path(dir) / "base"), dir_gb(fs::path(dir) / "delta"), dir_gb(dir));
   fflush(stdout);
   return 0;
}

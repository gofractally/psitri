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
#include <start_ref/kv.hpp>
#include <start_ver/kv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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
// vsz-byte value derived from a seed (>128 bytes forces a key_val_node external)
static size_t      g_vsz = 8;
static std::string vbig(uint64_t seed)
{
   std::string s(g_vsz, '\0');
   for (size_t j = 0; j < g_vsz; j += 8)
   {
      uint64_t w = splitmix64(seed + j);
      std::memcpy(&s[j], &w, std::min(size_t(8), g_vsz - j));
   }
   return s;
}
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
   g_vsz                    = argc > 5 ? size_t(strtoul(argv[5], nullptr, 10)) : 8;

   // [lbtest] focused reproduction of the 1.24M lower_bound bug: build a tree with
   // a long shared prefix branching into a '00' child (K0) and a '01' child (K1),
   // then lower_bound(prefix+'01') — must land on K1 (the '01' subtree), NOT K0.
   // `rounds` = number of EXTRA sibling children at the branch byte (varies the
   // router node type: 0 -> setlist, larger -> bitset/full).
   if (eng == "lbtest")
   {
      auto h2b = [](const std::string& hx) { std::string b; for (size_t i = 0; i + 1 < hx.size(); i += 2) b.push_back((char)strtoul(hx.substr(i, 2).c_str(), nullptr, 16)); return b; };
      const std::string sh = "1e2faa316fc4624ec39adc2ef7b5301124cfb6877700ff00ff00ff00ff00ff00ff00ff0100000111d7b6f76f3db92d8929f6f5787875e57311763e5a029c7d7fc94a118a0c2f01";
      std::string search = h2b(sh);
      std::string K0     = h2b(sh.substr(0, sh.size() - 2) + "00ff0ada08");  // prefix + 00...
      std::string K1     = h2b(sh + "0ada40");                               // prefix + 01...
      fs::remove_all(dir); fs::create_directories(dir);
      auto db = start_ref::kv::database::create(dir);
      auto ws = db->start_write_session();
      {
         auto tx = ws->start_transaction(0);
         tx.upsert(K0, std::string("v0"));
         tx.upsert(K1, std::string("v1"));
         // extra siblings at the branch byte (prefix + byte b + tail) to grow the router
         for (uint32_t b = 2; b < 2 + rounds && b < 256; ++b)
         {
            std::string k = search; k.back() = (char)b; k += std::string(3, '\x10');
            tx.upsert(k, std::string("vs"));
         }
         tx.commit();
      }
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      bool f   = cur.lower_bound(search);
      std::string land = f ? std::string(cur.key()) : std::string();
      auto hex = [](const std::string& s){ std::string o; char x[3]; for(unsigned char c:s){snprintf(x,3,"%02x",c);o+=x;} return o; };
      printf("[lbtest] siblings=%u  lower_bound(search) found=%d  landed_ok(>=search)=%d  ==K1? %d  ==K0? %d\n",
             rounds, (int)f, (int)(f && land >= search), (int)(land == K1), (int)(land == K0));
      printf("[lbtest] landed=%s\n", hex(land).c_str());
      printf("[lbtest] %s\n", (land == K1) ? "PASS" : "*** BUG REPRODUCED (lower_bound mispositioned) ***");
      return (land == K1) ? 0 : 1;
   }

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
            tx.upsert(k8(splitmix64(i)), vbig(i));
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
               tx.upsert(k8(splitmix64(i)), vbig(i ^ (uint64_t(r) << 40)));
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
            tx.upsert(k8(splitmix64(i)), vbig(i));
         tx.commit();
      }
      report("insert", 0);
      for (uint32_t r = 1; r <= rounds; ++r)
      {
         for (uint64_t i = 0; i < N;)
         {
            auto tx = ws->start_transaction(0);
            for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
               tx.upsert(k8(splitmix64(i)), vbig(i ^ (uint64_t(r) << 40)));
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
   else if (eng == "start_ref")
   {
      auto db = start_ref::kv::database::create(dir);
      auto ws = db->start_write_session();
      for (uint64_t i = 0; i < N;)
      {
         auto tx = ws->start_transaction(0);
         for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
            tx.upsert(k8(splitmix64(i)), vbig(i));
         tx.commit();
      }
      report("insert", 0);
      {  // [isolate engine] read-after-commit via a FRESH read_session + cursor
         // (the exact path the shim's RO txn uses). If this misses, the bug is
         // in the engine, not the shim.
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(0);
         auto v0  = cur.template get<std::string>(k8(splitmix64(0)));
         auto vN  = cur.template get<std::string>(k8(splitmix64(N - 1)));
         printf("VERIFY after-commit: key0 found=%d match=%d ; keyN found=%d match=%d\n",
                (int)v0.has_value(), (int)(v0 && *v0 == vbig(0)),
                (int)vN.has_value(), (int)(vN && *vN == vbig(N - 1)));
         fflush(stdout);
      }
      {  // [isolate shim pattern] multi-root: write root 0 AND root 1 in ONE txn
         // (the shim writes user data to root 0 + the DBI catalog to root 1,
         // committed together). Then read both back via fresh cursors.
         {
            auto tx = ws->start_transaction(0);
            tx.upsert(k8(0xAAAA), vbig(0xAAAA));  // primary root 0
            auto h1 = tx.open_root(1);
            h1.upsert(k8(0xBBBB), vbig(0xBBBB));  // root 1 (catalog-like)
            tx.commit();
         }
         auto rs = db->start_read_session();
         auto c0 = rs->create_cursor(0);
         auto c1 = rs->create_cursor(1);
         auto a  = c0.template get<std::string>(k8(0xAAAA));
         auto b  = c1.template get<std::string>(k8(0xBBBB));
         printf("MULTIROOT after-commit: root0 found=%d ; root1 found=%d\n",
                (int)a.has_value(), (int)b.has_value());
         fflush(stdout);
      }
      for (uint32_t r = 1; r <= rounds; ++r)
      {
         for (uint64_t i = 0; i < N;)
         {
            auto tx = ws->start_transaction(0);
            for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
               tx.upsert(k8(splitmix64(i)), vbig(i ^ (uint64_t(r) << 40)));
            tx.commit();
         }
         report("overwrite", r);
      }
      {  // [isolate engine] read-after-many-COW: after `rounds` overwrites + the
         // background releaser draining retired generations, every key must read
         // back its LAST written value. A refcount over-release (freeing a node
         // still shared with the live root) shows up here as a miss/mismatch.
         std::this_thread::sleep_for(std::chrono::milliseconds(300));  // let the releaser drain
         auto     rs   = db->start_read_session();
         auto     cur  = rs->create_cursor(0);
         uint64_t miss = 0, bad = 0;
         for (uint64_t i = 0; i < N; ++i)
         {
            auto v = cur.template get<std::string>(k8(splitmix64(i)));
            if (!v)
               ++miss;
            else if (*v != vbig(i ^ (uint64_t(rounds) << 40)))
               ++bad;
         }
         printf("VERIFY after-overwrite(rounds=%u): N=%llu miss=%llu mismatch=%llu\n",
                rounds, (unsigned long long)N, (unsigned long long)miss,
                (unsigned long long)bad);
         fflush(stdout);
      }
      for (uint64_t i = 0; i < N;)
      {
         auto tx = ws->start_transaction(0);
         for (uint64_t e = std::min(N, i + 1000); i < e; ++i)
            tx.remove(k8(splitmix64(i)));
         tx.commit();
      }
      report("removeall", -1);
   }
   else
   {
      fprintf(stderr, "unknown engine: %s (sal_art_ref|delta_art|start_ref)\n", eng.c_str());
      return 2;
   }

   printf("\nfinal footprint %.3f GB  (clean close ...)\n", dir_gb(dir));
   fflush(stdout);
   return 0;
}

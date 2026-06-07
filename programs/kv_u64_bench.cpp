// u64-key head-to-head: heart sal_radix_tree vs default psitri, on two workloads
//   * RANDOM      : key = XXH3(i)            (scattered 8-byte keys)
//   * SEQUENTIAL  : key = bswap64(i)         (big-endian ascending = sorted insert)
// For each: insert, then SEQUENTIAL-order queries and RANDOM-order queries.
// Keys are generated on the fly (no stored key array); value[i]=i, verified by
// checksum. The random query order is a shared shuffle.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <hash/xxhash.h>

#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_session_impl.hpp>

#include <sal_radix_tree.hpp>

using heart::sal_proto::sal_radix_tree;

namespace
{
   using clk = std::chrono::steady_clock;
   auto   now() { return clk::now(); }
   double ns(clk::time_point a, clk::time_point b, uint64_t n)
   {
      return double(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()) / double(n);
   }
   uint64_t rand_key(uint64_t i) { return XXH3_64bits(&i, sizeof(i)); }
   uint64_t seq_key(uint64_t i) { return __builtin_bswap64(i); }  // big-endian ascending

   std::string_view kspan(const uint64_t& k)  // key_view over the 8 key bytes
   {
      return {reinterpret_cast<const char*>(&k), sizeof(k)};
   }

   struct row
   {
      double ins = 0, seq = 0, rnd = 0, miss = 0;
      bool   ok = false;
   };

   template <class KeyFn>
   row bench_heart(uint64_t N, KeyFn keyfn, const std::vector<uint32_t>& order, const char* tag)
   {
      auto dir = std::filesystem::temp_directory_path() / (std::string("heart_u64_") + tag);
      sal_radix_tree t(dir);
      row            r;
      auto           t0 = now();
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t k = keyfn(i);
         t.insert(kspan(k), i);
      }
      r.ins   = ns(t0, now(), N);
      uint64_t s = 0, q = 0;
      auto     s0 = now();
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t k = keyfn(i), v = 0;
         if (t.find(kspan(k), v))
            s += v;
      }
      r.seq    = ns(s0, now(), N);
      auto r0  = now();
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t k = keyfn(order[i]), v = 0;
         if (t.find(kspan(k), v))
            q += v;
      }
      r.rnd = ns(r0, now(), N);
      // MISS: keys keyfn(N..2N-1) were never inserted → guaranteed not-found.
      uint64_t missfound = 0;
      auto     m0        = now();
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t k = keyfn(N + order[i]), v = 0;
         if (t.find(kspan(k), v))
            ++missfound;
      }
      r.miss                = ns(m0, now(), N);
      const uint64_t expect = N * (N - 1) / 2;
      r.ok                  = (s == expect && q == expect && missfound == 0);
      return r;
   }

   template <class KeyFn>
   row bench_psitri(uint64_t N, KeyFn keyfn, const std::vector<uint32_t>& order, const char* tag)
   {
      auto dir = std::filesystem::temp_directory_path() / (std::string("psitri_u64_") + tag);
      std::filesystem::remove_all(dir);
      psitri::runtime_config cfg;
      cfg.sync_mode               = sal::sync_type::none;
      cfg.write_protect_on_commit = false;
      cfg.checksum_on_commit      = false;
      auto db  = psitri::database::create(dir.string(), cfg);
      auto ses = db->start_write_session();
      row  r;

      auto t0 = now();
      {
         auto tx = ses->start_transaction(0);
         for (uint64_t i = 0; i < N; ++i)
         {
            uint64_t k = keyfn(i);
            tx.insert(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)),
                      psitri::value_view(reinterpret_cast<const char*>(&i), sizeof(i)));
         }
         tx.commit();
      }
      r.ins      = ns(t0, now(), N);
      auto rootp = ses->get_root(0);
      auto query = [&](uint64_t k, uint64_t& acc)
      {
         psitri::cursor cur(rootp);
         std::string    val;
         if (cur.get(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)), &val) >= 0 &&
             val.size() == 8)
         {
            uint64_t v;
            std::memcpy(&v, val.data(), 8);
            acc += v;
         }
      };
      uint64_t s = 0, q = 0;
      auto     s0 = now();
      for (uint64_t i = 0; i < N; ++i)
         query(keyfn(i), s);
      r.seq    = ns(s0, now(), N);
      auto r0  = now();
      for (uint64_t i = 0; i < N; ++i)
         query(keyfn(order[i]), q);
      r.rnd = ns(r0, now(), N);
      uint64_t missfound = 0;
      auto     m0        = now();
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t       k = keyfn(N + order[i]);
         psitri::cursor cur(rootp);
         std::string    val;
         if (cur.get(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)), &val) >= 0)
            ++missfound;
      }
      r.miss                = ns(m0, now(), N);
      const uint64_t expect = N * (N - 1) / 2;
      r.ok                  = (s == expect && q == expect && missfound == 0);
      return r;
   }

   // Progressive: insert in `batch`-sized chunks; before each chunk sample
   // `sample` queries over what's already there. psitri COMMITS every chunk so the
   // SAL compactor can reclaim COW garbage (a single giant tx would pin it all).
   // Prints a row per chunk so growth-driven slowdowns are visible.
   void prog_heart(uint64_t N, uint64_t batch, uint64_t sample)
   {
      auto dir = std::filesystem::temp_directory_path() / "heart_u64_prog";
      std::filesystem::remove_all(dir);
      sal_radix_tree  t(dir);
      std::mt19937_64 qrng(123);
      std::printf("  heart : keys(M)  insert_ns  query_ns   miss_ns   objects   live_MB\n");
      for (uint64_t base = 0; base < N; base += batch)
      {
         double   qns = 0, mns = 0;
         if (base > 0)
         {
            uint64_t f = 0;
            auto     q0 = now();
            for (uint64_t i = 0; i < sample; ++i)
            {
               uint64_t j = qrng() % base, v = 0, k = rand_key(j);
               if (t.find(kspan(k), v) && v == j)
                  ++f;
            }
            qns = ns(q0, now(), sample);
            uint64_t mf = 0;
            auto     m0 = now();
            for (uint64_t i = 0; i < sample; ++i)
            {
               uint64_t k = rand_key(N + qrng() % N), v = 0;  // guaranteed absent
               if (t.find(kspan(k), v))
                  ++mf;
            }
            mns = ns(m0, now(), sample);
            (void)f;
            (void)mf;
         }
         auto i0 = now();
         for (uint64_t i = base; i < base + batch; ++i)
            t.upsert(kspan(rand_key(i)), i);
         t.commit();  // COW-to-root version boundary (snapshot) — like psitri's commit
         double ins = ns(i0, now(), batch);
         // bytes_used() walks the whole tree (O(n)) — only do it every 10 chunks.
         uint64_t chunk = (base + batch) / batch;
         long     mb    = (chunk % 10 == 0) ? long(t.bytes_used() >> 20) : -1;
         std::printf("        %7.1f  %9.1f  %9.1f  %9.1f  %8llu  %7ld\n", double(base + batch) / 1e6,
                     ins, qns, mns, (unsigned long long)t.object_count(), mb);
         std::fflush(stdout);
      }
   }

   void prog_psitri(uint64_t N, uint64_t batch, uint64_t sample)
   {
      auto dir = std::filesystem::temp_directory_path() / "psitri_u64_prog";
      std::filesystem::remove_all(dir);
      psitri::runtime_config cfg;
      cfg.sync_mode               = sal::sync_type::none;
      cfg.write_protect_on_commit = false;
      cfg.checksum_on_commit      = false;
      auto            db  = psitri::database::create(dir.string(), cfg);
      auto            ses = db->start_write_session();
      std::mt19937_64 qrng(123);
      std::printf("  psitri: keys(M)  insert_ns  query_ns   miss_ns\n");
      for (uint64_t base = 0; base < N; base += batch)
      {
         double qns = 0, mns = 0;
         if (base > 0)
         {
            auto        rootp = ses->get_root(0);
            std::string val;
            auto        q0 = now();
            for (uint64_t i = 0; i < sample; ++i)
            {
               uint64_t       j = qrng() % base, k = rand_key(j);
               psitri::cursor cur(rootp);
               val.clear();
               cur.get(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)), &val);
            }
            qns         = ns(q0, now(), sample);
            auto m0 = now();
            for (uint64_t i = 0; i < sample; ++i)
            {
               uint64_t       k = rand_key(N + qrng() % N);
               psitri::cursor cur(rootp);
               val.clear();
               cur.get(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)), &val);
            }
            mns = ns(m0, now(), sample);
         }
         auto i0 = now();
         {
            auto tx = ses->start_transaction(0);
            for (uint64_t i = base; i < base + batch; ++i)
            {
               uint64_t k = rand_key(i);
               tx.insert(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)),
                         psitri::value_view(reinterpret_cast<const char*>(&i), sizeof(i)));
            }
            tx.commit();  // close the tx → compactor can reclaim
         }
         double ins = ns(i0, now(), batch);
         std::printf("        %7.1f  %9.1f  %9.1f  %9.1f\n", double(base + batch) / 1e6, ins, qns, mns);
         std::fflush(stdout);
      }
   }

   void print_pair(const char* title, row h, row p)
   {
      std::printf("  %s\n", title);
      std::printf("    heart  : insert %6.1f   seq %6.1f   rand %6.1f   MISS %6.1f ns   [%s]\n", h.ins,
                  h.seq, h.rnd, h.miss, h.ok ? "OK" : "BAD");
      std::printf("    psitri : insert %6.1f   seq %6.1f   rand %6.1f   MISS %6.1f ns   [%s]\n", p.ins,
                  p.seq, p.rnd, p.miss, p.ok ? "OK" : "BAD");
   }
}  // namespace

int main(int argc, char** argv)
{
   const uint64_t    N    = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 2'000'000;
   const std::string mode = (argc > 2) ? argv[2] : "both";  // "rand" | "seq" | "both" | "prog"

   std::printf("N = %llu  (8-byte u64 keys, value=i)\n\n", (unsigned long long)N);

   if (mode == "prog")  // progressive random: 1M-insert chunks, 100k sampled queries, commit/chunk
   {
      std::printf("RANDOM keys, progressive (commit every chunk):\n");
      prog_heart(N, 1'000'000, 100'000);
      std::printf("\n");
      prog_psitri(N, 1'000'000, 100'000);
      std::fflush(stdout);
      return 0;
   }

   std::vector<uint32_t> order(N);
   for (uint64_t i = 0; i < N; ++i)
      order[i] = uint32_t(i);
   std::shuffle(order.begin(), order.end(), std::mt19937(98765));

   if (mode == "rand" || mode == "both")
   {
      row hr = bench_heart(N, rand_key, order, "rand");
      row pr = bench_psitri(N, rand_key, order, "rand");
      print_pair("RANDOM keys (XXH3(i)):", hr, pr);
      std::printf("\n");
   }
   if (mode == "seq" || mode == "both")
   {
      row hs = bench_heart(N, seq_key, order, "seq");
      row ps = bench_psitri(N, seq_key, order, "seq");
      print_pair("SEQUENTIAL big-endian keys (bswap64(i), ascending insert):", hs, ps);
   }

   std::fflush(stdout);
   return 0;
}

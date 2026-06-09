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
#include <unordered_map>
#include <vector>

#include <hash/xxhash.h>

#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_session_impl.hpp>

#include <sal_radix_tree.hpp>
#include <radix_engine.hpp>

using heart::sal_proto::sal_radix_tree;
using radix_engine_tree = heart::radix::radix_tree;

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
      double ins = 0, seq = 0, rnd = 0, miss = 0, iter = 0, del = 0;
      bool   ok = false;
   };

   // Apples-to-apples txn granularity: BOTH engines commit every TX_BATCH ops (psitri via
   // start_transaction/commit per chunk, heart via commit() per chunk → shared/COW mode).
   // A realistic bulk cadence — avoids the single-giant-txn pathology (which pins all of
   // psitri's COW garbage) and makes both pay equal commit/COW cost.
   constexpr uint64_t TX_BATCH = 4096;

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
      auto tc               = t.count_tiers();
      std::printf("    [%s tiers: %llu leaf, %llu setlist, %llu bitset, %llu full(R8)]\n", tag,
                  (unsigned long long)tc.leaf, (unsigned long long)tc.setlist,
                  (unsigned long long)tc.bitset, (unsigned long long)tc.full);
      return r;
   }

   // New engine (heart::radix on the DESIGNED segmented taxonomy). Values are the
   // 8-byte index (arbitrary byte array via value_view).
   template <class KeyFn>
   row bench_radix(uint64_t N, KeyFn keyfn, const std::vector<uint32_t>& order, const char* tag,
                   bool bucket = false)
   {
      auto dir = std::filesystem::temp_directory_path() /
                 (std::string("heart_radix_") + tag + (bucket ? "_bkt" : ""));
      radix_engine_tree t(dir);
      if (bucket)
         t.set_adaptive(true);
      row               r;
      auto              vview = [](const uint64_t& v) {
         return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v));
      };
      auto t0 = now();
      for (uint64_t base = 0; base < N; base += TX_BATCH)
      {
         const uint64_t end = std::min(N, base + TX_BATCH);
         for (uint64_t i = base; i < end; ++i)
         {
            uint64_t k = keyfn(i);
            t.upsert(kspan(k), vview(i));
         }
         t.commit();  // batch boundary → subsequent writes COW (shared mode), like a psitri txn
      }
      r.ins = ns(t0, now(), N);
      uint64_t s = 0, q = 0;
      auto     s0 = now();
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t          k = keyfn(i);
         std::string_view  v;
         if (t.find(kspan(k), v))
         {
            uint64_t got = 0;
            std::memcpy(&got, v.data(), std::min<size_t>(8, v.size()));
            s += got;
         }
      }
      r.seq    = ns(s0, now(), N);
      auto r0  = now();
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t         k = keyfn(order[i]);
         std::string_view v;
         if (t.find(kspan(k), v))
         {
            uint64_t got = 0;
            std::memcpy(&got, v.data(), std::min<size_t>(8, v.size()));
            q += got;
         }
      }
      r.rnd                 = ns(r0, now(), N);
      uint64_t missfound    = 0;
      auto     m0           = now();
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t         k = keyfn(N + order[i]);
         std::string_view v;
         if (t.find(kspan(k), v))
            ++missfound;
      }
      r.miss                = ns(m0, now(), N);
      const uint64_t expect = N * (N - 1) / 2;
      r.ok                  = (s == expect && q == expect && missfound == 0);
      if (bucket)  // count_tiers walks radix node types; bucket terminals would be misread
         std::printf("    [%s ADAPTIVE: %llu obj]\n", tag, (unsigned long long)t.object_count());
      else
      {
         auto tc = t.count_tiers();
         std::printf("    [%s tiers: %llu leaf, %llu setlist, %llu r2, %llu r4, %llu full(R8); %llu obj]\n",
                     tag, (unsigned long long)tc.leaf, (unsigned long long)tc.setlist,
                     (unsigned long long)tc.bitset_r2, (unsigned long long)tc.bitset_r4,
                     (unsigned long long)tc.full, (unsigned long long)t.object_count());
         std::printf("    [%s bitset: ptr %llu / %llu CLs (s_ptr=%.2f) | tails %llu]\n",
                     tag, (unsigned long long)tc.cptr, (unsigned long long)tc.cptr_cl,
                     tc.cptr_cl ? double(tc.cptr) / double(tc.cptr_cl) : 0.0,
                     (unsigned long long)tc.ctail);
         std::printf("    [%s setlist: ptr %llu / %llu CLs (s_setlist=%.2f) | tails %llu | "
                     "avg tail_used %.0f B of 256 headroom over %llu setlists]\n",
                     tag, (unsigned long long)tc.slptr, (unsigned long long)tc.slptr_cl,
                     tc.slptr_cl ? double(tc.slptr) / double(tc.slptr_cl) : 0.0,
                     (unsigned long long)tc.sltail,
                     tc.setlist ? double(tc.sl_tail_used) / double(tc.setlist) : 0.0,
                     (unsigned long long)tc.setlist);
      }
      // ITERATE: full forward scan via the engine's native cursor (skip in adaptive/bucket mode,
      // whose bucket leaves the ordered cursor doesn't traverse). Sums the 8-byte values.
      if (!bucket)
      {
         uint64_t isum = 0, icnt = 0;
         auto     i0 = now();
         heart::radix::radix_tree::cursor c(&t);
         for (bool end = c.seek_begin(); !end; end = c.next())
         {
            isum += c.key().size();  // pure traversal: touch the key, no value access
            ++icnt;
         }
         r.iter = ns(i0, now(), icnt ? icnt : 1);
         r.ok   = r.ok && icnt == N;
         (void)isum;
      }
      // REMOVE: delete every key in keyfn order → empties the tree (ascending = true
      // sequential delete for the seq keyfn; scattered for random). Committed per batch.
      auto d0 = now();
      for (uint64_t base = 0; base < N; base += TX_BATCH)
      {
         const uint64_t end = std::min(N, base + TX_BATCH);
         for (uint64_t i = base; i < end; ++i)
            t.remove(kspan(keyfn(i)));
         t.commit();
      }
      r.del          = ns(d0, now(), N);
      uint64_t still = 0;
      for (uint64_t i = 0; i < N; ++i)
      {
         std::string_view v;
         if (t.find(kspan(keyfn(i)), v))
            ++still;
      }
      r.ok = r.ok && still == 0;
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
      for (uint64_t base = 0; base < N; base += TX_BATCH)
      {
         const uint64_t end = std::min(N, base + TX_BATCH);
         auto           tx  = ses->start_transaction(0);
         for (uint64_t i = base; i < end; ++i)
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
      // ITERATE: full forward scan, exactly psitri's native iterate_test pattern.
      uint64_t iter_cnt = 0;
      {
         auto           i0 = now();
         psitri::cursor cur(rootp);
         cur.seek_begin();
         while (!cur.is_end())
         {
            ++iter_cnt;
            cur.next();
         }
         r.iter = ns(i0, now(), iter_cnt ? iter_cnt : 1);
      }
      // REMOVE: per-batch transactions delete every key in keyfn order (ascending for the seq
      // keyfn = true sequential delete; scattered for random) — same cadence as heart.
      auto d0 = now();
      for (uint64_t base = 0; base < N; base += TX_BATCH)
      {
         const uint64_t end = std::min(N, base + TX_BATCH);
         auto           tx  = ses->start_transaction(0);
         for (uint64_t i = base; i < end; ++i)
         {
            uint64_t k = keyfn(i);
            tx.remove(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)));
         }
         tx.commit();
      }
      r.del          = ns(d0, now(), N);
      auto     rootp2 = ses->get_root(0);
      uint64_t still  = 0;
      for (uint64_t i = 0; i < N; ++i)
      {
         uint64_t       k = keyfn(i);
         psitri::cursor cur(rootp2);
         std::string    val;
         if (cur.get(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)), &val) >= 0)
            ++still;
      }
      r.ok = (s == expect && q == expect && missfound == 0 && still == 0 && iter_cnt == N);
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

   void print_row(const char* name, row x)
   {
      std::printf(
          "    %-7s: insert %6.1f   seq %6.1f   rand %6.1f   MISS %6.1f   ITER %5.1f   REMOVE %6.1f ns   [%s]\n",
          name, x.ins, x.seq, x.rnd, x.miss, x.iter, x.del, x.ok ? "OK" : "BAD");
   }
   void print_trio(const char* title, row e, row eb, row h, row p)
   {
      std::printf("  %s\n", title);
      print_row("engine", e);   // new engine, bucket OFF (designed segmented taxonomy)
      print_row("adaptive", eb); // new engine, adaptive radix->bucket on deep-narrow
      print_row("proto", h);    // old prototype (sal_radix_tree)
      print_row("psitri", p);
   }
}  // namespace

int main(int argc, char** argv)
{
   const uint64_t    N    = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 2'000'000;
   const std::string mode = (argc > 2) ? argv[2] : "both";  // "rand" | "seq" | "both" | "prog"

   std::printf("N = %llu  (8-byte u64 keys, value=i)\n\n", (unsigned long long)N);

   if (mode == "worst")  // WORST CASE: commit every BATCH updates (default 1 → COW-to-root every op).
   {                     // Head-to-head heart radix engine vs psitri (both on SAL), random keys.
      const uint64_t batch = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 1;
      uint64_t (*keyfn)(uint64_t) =
          (argc > 4 && std::string_view(argv[4]) == "seq") ? +seq_key : +rand_key;
      std::printf("  (commit every %llu updates, %s keys)\n", (unsigned long long)batch,
                  (argc > 4 && std::string_view(argv[4]) == "seq") ? "SEQUENTIAL" : "random");
      auto vview = [](const uint64_t& v)
      { return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v)); };

      // ── heart radix engine: upsert + commit per BATCH (COW-to-root on the first op of a batch) ──
      {
         auto dir = std::filesystem::temp_directory_path() / "heart_worst";
         std::filesystem::remove_all(dir);
         radix_engine_tree t(dir);
         auto              t0 = now();
         for (uint64_t i = 0; i < N; ++i)
         {
            uint64_t k = keyfn(i);
            t.upsert(kspan(k), vview(i));
            if ((i + 1) % batch == 0) t.commit();
         }
         if (N % batch) t.commit();
         double   ins = ns(t0, now(), N);
         uint64_t s   = 0;
         for (uint64_t i = 0; i < N; ++i)
         {
            uint64_t k = keyfn(i); std::string_view v;
            if (t.find(kspan(k), v)) { uint64_t g = 0; std::memcpy(&g, v.data(), 8); s += g; }
         }
         std::printf("    heart  : update %8.1f ns/txn   [%s]  clones=%llu (%.2f/op)\n", ins,
                     s == N * (N - 1) / 2 ? "OK" : "BAD",
                     (unsigned long long)t.counters().clones,
                     double(t.counters().clones) / double(N));
         auto& c = t.counters();
         std::printf("             clones %llu  in_place %llu  grows %llu  child_retains %llu  (%.2f retains/txn, %.1f/clone)\n",
                     (unsigned long long)c.clones, (unsigned long long)c.in_place, (unsigned long long)c.grows,
                     (unsigned long long)c.child_retains,
                     double(c.child_retains) / double(N), double(c.child_retains) / double(c.clones ? c.clones : 1));
      }

      // ── psitri (single-version COW on SAL): one transaction per op ──
      {
         auto dir = std::filesystem::temp_directory_path() / "psitri_worst";
         std::filesystem::remove_all(dir);
         psitri::runtime_config cfg;
         cfg.sync_mode = sal::sync_type::none;
         cfg.write_protect_on_commit = false;
         cfg.checksum_on_commit = false;
         auto db  = psitri::database::create(dir.string(), cfg);
         auto ses = db->start_write_session();
         auto t0  = now();
         for (uint64_t base = 0; base < N; base += batch)
         {
            auto tx  = ses->start_transaction(0);
            uint64_t end = std::min(N, base + batch);
            for (uint64_t i = base; i < end; ++i)
            {
               uint64_t k = keyfn(i);
               tx.insert(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k)),
                         psitri::value_view(reinterpret_cast<const char*>(&i), sizeof(i)));
            }
            tx.commit();
         }
         double   ins = ns(t0, now(), N);
         uint64_t s = 0;
         auto     rootp = ses->get_root(0);
         for (uint64_t i = 0; i < N; ++i)
         {
            uint64_t k = keyfn(i);
            psitri::cursor cur(rootp);
            if (cur.find(psitri::key_view(reinterpret_cast<const char*>(&k), sizeof(k))))
               cur.get_value([&](psitri::value_view v) { uint64_t g = 0; std::memcpy(&g, v.data(), std::min<size_t>(8, v.size())); s += g; });
         }
         std::printf("    psitri : update %8.1f ns/txn   [%s]\n", ins,
                     s == N * (N - 1) / 2 ? "OK" : "BAD");
      }

      std::printf("WORST CASE: 1 committed txn per update (COW-to-root each op), random %llu keys\n",
                  (unsigned long long)N);
      std::fflush(stdout);
      return 0;
   }

   if (mode == "uniq")  // UNIQUE-MODE bulk: NO commit (no live snapshot) → must stay all in-place.
   {
      auto vview = [](const uint64_t& v)
      { return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v)); };
      auto dir = std::filesystem::temp_directory_path() / "heart_uniq";
      std::filesystem::remove_all(dir);
      radix_engine_tree t(dir);
      auto     t0 = now();
      for (uint64_t i = 0; i < N; ++i) { uint64_t k = rand_key(i); t.upsert(kspan(k), vview(i)); }
      double   ins = ns(t0, now(), N);
      uint64_t s = 0; std::string_view v;
      for (uint64_t i = 0; i < N; ++i)
      { uint64_t k = rand_key(i); if (t.find(kspan(k), v)) { uint64_t g=0; std::memcpy(&g,v.data(),8); s+=g; } }
      auto& c = t.counters();
      std::printf("UNIQUE bulk (no commit), %llu keys: insert %.1f ns/op  [%s]\n",
                  (unsigned long long)N, ins, s == N*(N-1)/2 ? "OK" : "BAD");
      std::printf("  clones=%llu  in_place=%llu  grows=%llu  child_retains=%llu  (clones==grows => 0 shared-path clones)\n",
                  (unsigned long long)c.clones, (unsigned long long)c.in_place, (unsigned long long)c.grows,
                  (unsigned long long)c.child_retains);
      std::fflush(stdout);
      return 0;
   }

   if (mode == "cowtest")  // DIRECT COW verification: insert 1000, commit, insert 1; check child refs.
   {
      auto vview = [](const uint64_t& v)
      { return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v)); };
      auto dir = std::filesystem::temp_directory_path() / "heart_cowtest";
      std::filesystem::remove_all(dir);
      radix_engine_tree t(dir);
      for (uint64_t i = 0; i < 1000; ++i) { uint64_t k = rand_key(i); t.upsert(kspan(k), vview(i)); }
      t.commit();
      auto root_before = t.root_addr();
      t.dump_root_child_refs("after commit, before the +1 insert");
      // Insert ONE brand-new key → should COW-to-root: clone root (retain all children), clone path.
      {
         uint64_t rb = t.counters().child_retains;
         uint64_t k = rand_key(1000);
         t.upsert(kspan(k), vview(1000));
         uint64_t ra = t.counters().child_retains;
         auto root_after = t.root_addr();
         std::printf("  root address changed (cloned)? %s   child-retains for this 1 insert = %llu\n",
                     root_before != root_after ? "YES" : "NO", (unsigned long long)(ra - rb));
         t.dump_root_child_refs("after the +1 insert (NEW root)");
      }
      // Now the EXACT worst-bench pattern: commit AFTER EVERY insert, many cycles, measure retains/txn.
      std::printf("  --- now commit-every-insert (worst-bench pattern), measuring child-retains/txn ---\n");
      for (uint64_t base : {2000ull, 50000ull})
      {
         uint64_t rb = t.counters().child_retains;
         for (uint64_t i = base; i < base + 2000; ++i)
         { uint64_t k = rand_key(i); t.upsert(kspan(k), vview(i)); t.commit(); }
         uint64_t ra = t.counters().child_retains;
         std::printf("    after %llu keys total: %.1f child-retains/txn over 2000 commit-per-insert txns\n",
                     (unsigned long long)(base + 2000), double(ra - rb) / 2000.0);
         t.dump_root_child_refs("  steady state");
      }
      std::fflush(stdout);
      return 0;
   }

   if (mode == "rrstress")  // ADVERSARIAL multi-version refcount stress (run under build-di).
   {
      // Many concurrent pinned snapshots, divergent write+remove lineages forked off shared
      // ancestors, common-ancestor release, deep version stacking (forces nibble fork) — every
      // pinned snapshot is model-checked (count+checksum) after arbitrary churn elsewhere, so any
      // refcount over/under-count surfaces as a snapshot reading the wrong contents (and any
      // UAF/double-free/leak is caught by free_range_tracker under build-di).
      const uint64_t rounds = (N < 100000) ? N : 4000;
      auto vview = [](const uint64_t& v)
      { return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v)); };
      auto kspn  = [](const uint64_t& k)
      { return std::string_view(reinterpret_cast<const char*>(&k), sizeof(k)); };
      using model_t = std::unordered_map<uint64_t, uint64_t>;
      auto model_sig = [](const model_t& m) -> std::pair<uint64_t, uint64_t> {
         uint64_t sum = 0;
         for (auto& [k, v] : m)
         {
            uint64_t h = 1469598103934665603ull;
            for (int i = 0; i < 8; ++i) h = (h ^ uint8_t(k >> (8 * i))) * 1099511628211ull;
            for (int i = 0; i < 8; ++i) h = (h ^ uint8_t(v >> (8 * i))) * 1099511628211ull;
            sum += h;
         }
         return {m.size(), sum};
      };
      auto dir = std::filesystem::temp_directory_path() / "heart_rrstress";
      std::filesystem::remove_all(dir);
      radix_engine_tree t(dir);
      std::mt19937_64   rng(0xC0FFEEull);

      using paddr = decltype(t.take_snapshot());
      struct Ver { paddr root; model_t model; };  // pinned snapshot + its frozen contents
      std::vector<Ver> snaps;
      model_t          live;
      uint64_t         nextval = 1;
      uint64_t         fails = 0, checks = 0;

      auto do_write = [&](uint64_t target) {  // churn the live lineage toward ~target keys
         for (int op = 0; op < 50; ++op)
         {
            bool rem = (!live.empty()) && (live.size() > target || (rng() & 1));
            if (rem)
            {
               auto it = live.begin();
               std::advance(it, rng() % live.size());
               uint64_t k = it->first;
               t.remove(kspn(k));
               live.erase(k);
            }
            else
            {
               uint64_t k = rng(), v = nextval++;
               t.upsert(kspn(k), vview(v));
               live[k] = v;
            }
         }
      };
      auto verify = [&](const Ver& s) {
         auto got = t.sig_at(s.root);
         auto exp = model_sig(s.model);
         ++checks;
         if (got != exp) { ++fails; if (fails <= 5)
            std::printf("  MISMATCH snap: got{n=%llu,sum=%llu} exp{n=%llu,sum=%llu}\n",
               (unsigned long long)got.first,(unsigned long long)got.second,
               (unsigned long long)exp.first,(unsigned long long)exp.second); }
      };

      for (int i = 0; i < 1500; ++i) { uint64_t k = rng(), v = nextval++; t.upsert(kspn(k), vview(v)); live[k] = v; }
      t.commit();

      for (uint64_t r = 0; r < rounds; ++r)
      {
         do_write(3000);
         t.commit();
         if ((rng() % 3) == 0 && snaps.size() < 40)            // pin a new snapshot
            snaps.push_back({t.take_snapshot(), live});
         if ((rng() % 6) == 0 && !snaps.empty())               // diverge a new lineage off an ancestor
         {
            auto& s = snaps[rng() % snaps.size()];
            t.adopt_root(s.root);
            live = s.model;
         }
         if ((rng() % 4) == 0 && snaps.size() > 6)             // release a snapshot (incl. ancestors)
         {
            size_t idx = rng() % snaps.size();
            t.release_snapshot(snaps[idx].root);
            snaps.erase(snaps.begin() + idx);
         }
         for (auto& s : snaps) verify(s);                      // ALL pinned snapshots stay isolated
      }
      for (auto& s : snaps) { verify(s); t.release_snapshot(s.root); }
      std::printf("rrstress: %llu rounds, %llu checks, %llu MISMATCHES  -> %s\n",
                  (unsigned long long)rounds, (unsigned long long)checks, (unsigned long long)fails,
                  fails == 0 ? "OK" : "FAIL");
      std::fflush(stdout);
      return 0;
   }

   if (mode == "snaptest")  // CORRECTNESS: does a committed snapshot survive later writes (true COW)?
   {
      auto vview = [](const uint64_t& v)
      { return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v)); };
      auto getv  = [](std::string_view v) { uint64_t g = 0; std::memcpy(&g, v.data(), std::min<size_t>(8, v.size())); return g; };
      auto dir = std::filesystem::temp_directory_path() / "heart_snaptest";
      std::filesystem::remove_all(dir);
      radix_engine_tree t(dir);
      // Insert 1000 keys with value = i, commit, pin that snapshot.
      for (uint64_t i = 0; i < 1000; ++i) { uint64_t k = rand_key(i); t.upsert(kspan(k), vview(i)); }
      t.commit();
      auto snap = t.take_snapshot();
      // (a) UPDATE an existing key to a new value, commit.
      uint64_t k7 = rand_key(7);
      t.upsert(kspan(k7), vview(999999)); t.commit();
      // (b) INSERT 1000 brand-new keys, commit.
      for (uint64_t i = 1000; i < 2000; ++i) { uint64_t k = rand_key(i); t.upsert(kspan(k), vview(i)); }
      t.commit();
      // (c) REMOVE 500 keys that EXIST in the snapshot (keys 0..499) from the LIVE lineage, commit.
      //     A correct COW-to-root remove must clone the path — the snapshot keeps these keys.
      for (uint64_t i = 0; i < 500; ++i) { uint64_t k = rand_key(i); t.remove(kspan(k)); }
      t.commit();
      // Now interrogate the PINNED snapshot. Correct COW → it must still show the old world.
      std::string_view v;
      bool got7   = t.find_at(snap, kspan(k7), v);
      uint64_t v7 = got7 ? getv(v) : 0;
      uint64_t snap_keys = t.count_keys_at(snap);          // COW-correct → still exactly 1000
      uint64_t live_keys = t.count_keys_at(t.root_addr()); // 1000 + 1000 inserted - 500 removed = 1500
      std::printf("SNAPSHOT ISOLATION TEST:\n");
      std::printf("  key#7 update isolation: snapshot value=%llu (expect 7)  -> %s\n",
                  (unsigned long long)v7, v7 == 7 ? "ISOLATED" : "CORRUPTED");
      // count is the unambiguous signal: removes on the live lineage (incl. keys present in the
      // snapshot) must NOT change the snapshot's count, while live drops by the 500 removed.
      std::printf("  snapshot count = %llu (expect 1000)  |  live count = %llu (expect 1500)  -> %s\n",
                  (unsigned long long)snap_keys, (unsigned long long)live_keys,
                  (snap_keys == 1000 && live_keys == 1500) ? "ISOLATED" : "CORRUPTED");
      (void)got7;
      t.release_snapshot(snap);
      std::fflush(stdout);
      return 0;
   }

   if (mode == "drop")  // BIG-SUBTREE-DROP: build N keys uniquely-owned, then release the root ONCE
   {                    // and time (a) the calling-thread latency and (b) the drain to quiescent.
      auto vview = [](const uint64_t& v)
      { return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v)); };
      auto dir = std::filesystem::temp_directory_path() / "heart_drop";
      std::filesystem::remove_all(dir);
      radix_engine_tree t(dir);
      for (uint64_t i = 0; i < N; ++i)  // NO per-op commit → interior nodes uniquely owned
      {
         uint64_t k = rand_key(i);
         t.upsert(kspan(k), vview(i));
      }
      uint64_t live_before = t.object_count();
      auto     t0  = now();
      t.drop_root();                                  // single release() → cascade
      auto     t1  = now();                            // calling-thread latency ends here
      while (t.pending_releases() != 0) { /* spin until bg drains spilled remainder */ }
      auto     t2  = now();                            // all memory actually freed
      double   caller_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
      double   total_ns  = std::chrono::duration<double, std::nano>(t2 - t0).count();
      std::printf("BIG DROP of %llu keys (%llu live objects):\n",
                  (unsigned long long)N, (unsigned long long)live_before);
      std::printf("    caller-thread latency : %10.1f us   (%6.1f ns/obj)\n",
                  caller_ns / 1000.0, caller_ns / double(live_before ? live_before : 1));
      std::printf("    total drain to quiet  : %10.1f us   (%6.1f ns/obj)\n",
                  total_ns / 1000.0, total_ns / double(live_before ? live_before : 1));
      std::fflush(stdout);
      return 0;
   }

   if (mode == "scale")  // heart radix engine: N random keys, commit every BATCH, sampled queries.
   {
      const uint64_t batch  = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 1'000'000;
      const uint64_t sample = 100'000;
      auto vview = [](const uint64_t& v)
      { return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v)); };
      auto getv  = [](std::string_view v) { uint64_t g = 0; std::memcpy(&g, v.data(), 8); return g; };
      auto dir = std::filesystem::temp_directory_path() / "heart_scale";
      std::filesystem::remove_all(dir);
      radix_engine_tree t(dir);
      std::mt19937_64   qrng(123);
      std::printf("heart radix engine — %lluM random keys, commit every %lluK, %lluk rand+seq queries/chunk\n",
                  (unsigned long long)(N / 1'000'000), (unsigned long long)(batch / 1000),
                  (unsigned long long)(sample / 1000));
      std::printf("  keys(M)   insert_ns  randQ_ns   seqQ_ns   objects\n");
      for (uint64_t base = 0; base < N; base += batch)
      {
         double rqns = 0, sqns = 0;
         if (base > 0)
         {
            // 100k RANDOM-order queries of random existing keys (scattered, cold).
            std::string_view v;
            uint64_t         hits = 0;
            auto             r0 = now();
            for (uint64_t i = 0; i < sample; ++i)
            {
               uint64_t j = qrng() % base;
               if (t.find(kspan(rand_key(j)), v) && getv(v) == j) ++hits;
            }
            rqns = ns(r0, now(), sample);
            // 100k SEQUENTIAL-index queries: keys rand_key(0..sample) in order (same set each chunk).
            auto s0 = now();
            for (uint64_t j = 0; j < sample; ++j)
               if (t.find(kspan(rand_key(j)), v) && getv(v) == j) ++hits;
            sqns = ns(s0, now(), sample);
            if (hits != 2 * sample) std::printf("  [WARN] query hits=%llu/%llu\n",
                                                (unsigned long long)hits, (unsigned long long)(2 * sample));
         }
         auto i0 = now();
         for (uint64_t i = base; i < base + batch && i < N; ++i)
            t.upsert(kspan(rand_key(i)), vview(i));
         t.commit();  // COW-to-root snapshot boundary
         double ins = ns(i0, now(), std::min(batch, N - base));
         std::printf("  %7.1f  %9.1f  %9.1f  %9.1f  %9llu\n", double(base + batch) / 1e6, ins, rqns,
                     sqns, (unsigned long long)t.object_count());
         std::fflush(stdout);
      }
      return 0;
   }

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
      row er  = bench_radix(N, rand_key, order, "rand");
      row erb = bench_radix(N, rand_key, order, "rand", /*bucket=*/true);
      row hr  = bench_heart(N, rand_key, order, "rand");
      row pr  = bench_psitri(N, rand_key, order, "rand");
      print_trio("RANDOM keys (XXH3(i)):", er, erb, hr, pr);
      std::printf("\n");
   }
   if (mode == "seq" || mode == "both")
   {
      row es  = bench_radix(N, seq_key, order, "seq");
      row esb = bench_radix(N, seq_key, order, "seq", /*bucket=*/true);
      row hs  = bench_heart(N, seq_key, order, "seq");
      row ps  = bench_psitri(N, seq_key, order, "seq");
      print_trio("SEQUENTIAL big-endian keys (bswap64(i), ascending insert):", es, esb, hs, ps);
   }

   std::fflush(stdout);
   return 0;
}

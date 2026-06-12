// dict_bench — ONE workload, FIVE engines: comparative technique analysis.
//
//   load      insert every key (value = u64 sequence number as 8 raw bytes)
//   find-hit  3 shuffled passes over every key
//   find-miss 2 passes with a mutated suffix (guaranteed absent)
//   iterate   3 full in-order scans (count + key-byte checksum)
//
// Keysets: --keys dict (/usr/share/dict/words) or --keys rand (1M x 8-byte
// binary). Engines: --engine hkv|artmap|part|vpart|heart. hkv/artmap are raw
// in-memory baselines (no durability, no versioning); part/vpart/heart run
// their full engine surface with batched commits (--batch, default 8192).
// The point (Dan): comparing the same phases across engines exposes which
// techniques pay and which implementations hide an unreported pathological
// gap (proof case: vpart's odd-size allocations falling off part's freeset
// fast classes — found by exactly this kind of side-by-side).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <part/hkv/tree.hpp>  // namespace hkv (heap mode baseline)
#include <art/art_map.hpp>
#include <art/heap_arena.hpp>
#include <art/iterator.hpp>
#include <part/kv.hpp>
#include <vpart/kv.hpp>
#include <radix_engine.hpp>  // heart::radix (heart/test include dir)

namespace
{
   using clock_t_ = std::chrono::steady_clock;
   double ns_per(clock_t_::time_point t0, clock_t_::time_point t1, uint64_t n)
   {
      return double(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
             double(n ? n : 1);
   }

   std::vector<std::string> load_keys(const std::string& mode)
   {
      std::vector<std::string> keys;
      if (mode == "dict")
      {
         std::ifstream f("/usr/share/dict/words");
         std::string   w;
         while (std::getline(f, w))
            if (!w.empty())
               keys.push_back(w);
      }
      else
      {
         std::mt19937_64 rng(0xD1C7);
         keys.reserve(1'000'000);
         for (int i = 0; i < 1'000'000; ++i)
         {
            uint64_t r = rng();
            keys.emplace_back(reinterpret_cast<const char*>(&r), 8);
         }
         std::sort(keys.begin(), keys.end());
         keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      }
      return keys;
   }

   std::string val_of(uint64_t i) { return std::string(reinterpret_cast<const char*>(&i), 8); }

   struct phases
   {
      std::vector<std::string> keys, shuffled, missing;
      explicit phases(std::vector<std::string> k) : keys(std::move(k))
      {
         shuffled = keys;
         std::mt19937_64 rng(42);
         std::shuffle(shuffled.begin(), shuffled.end(), rng);
         missing.reserve(keys.size());
         for (auto& s : keys)
            missing.push_back(s + "~");
      }
   };

   // run the four phases through engine-specific lambdas; print one row each
   template <class Put, class Commit, class Get, class Iter>
   void run(const char* engine, const phases& P, Put&& put, Commit&& commit, Get&& get,
            Iter&& iter, uint64_t batch)
   {
      const uint64_t N = P.keys.size();
      auto           t0 = clock_t_::now();
      for (uint64_t i = 0; i < N; ++i)
      {
         put(P.keys[i], i);
         if (batch && (i + 1) % batch == 0)
            commit();
      }
      commit();
      auto t1 = clock_t_::now();
      printf("%-7s load       %8.1f ns/op  (%llu keys)\n", engine, ns_per(t0, t1, N),
             (unsigned long long)N);

      uint64_t hits = 0;
      t0 = clock_t_::now();
      for (int pass = 0; pass < 3; ++pass)
         for (auto& k : P.shuffled)
            hits += get(k);
      t1 = clock_t_::now();
      printf("%-7s find-hit   %8.1f ns/op  (%llu found)\n", engine, ns_per(t0, t1, 3 * N),
             (unsigned long long)hits);
      if (hits != 3 * N)
         printf("%-7s   *** HIT COUNT MISMATCH: %llu != %llu\n", engine,
                (unsigned long long)hits, (unsigned long long)(3 * N));

      uint64_t false_hits = 0;
      t0 = clock_t_::now();
      for (int pass = 0; pass < 2; ++pass)
         for (auto& k : P.missing)
            false_hits += get(k);
      t1 = clock_t_::now();
      printf("%-7s find-miss  %8.1f ns/op  (%llu false hits)\n", engine, ns_per(t0, t1, 2 * N),
             (unsigned long long)false_hits);

      uint64_t cnt = 0, sum = 0;
      t0 = clock_t_::now();
      for (int pass = 0; pass < 3; ++pass)
         iter(cnt, sum);
      t1 = clock_t_::now();
      printf("%-7s iterate    %8.1f ns/op  (%llu seen, keysum %llx)\n", engine,
             ns_per(t0, t1, cnt ? cnt : 1), (unsigned long long)cnt, (unsigned long long)sum);
      if (cnt != 3 * N)
         printf("%-7s   *** ITERATE COUNT MISMATCH: %llu != %llu\n", engine,
                (unsigned long long)cnt, (unsigned long long)(3 * N));
   }

   std::filesystem::path tmpdb(const char* tag)
   {
      auto p = std::filesystem::temp_directory_path() / (std::string("dictbench_") + tag);
      std::filesystem::remove_all(p);
      return p;
   }
}  // namespace

int main(int argc, char** argv)
{
   std::string engine = "hkv", keys = "dict";
   uint64_t    batch  = 8192;
   for (int i = 1; i < argc - 1; ++i)
   {
      if (!strcmp(argv[i], "--engine")) engine = argv[i + 1];
      if (!strcmp(argv[i], "--keys")) keys = argv[i + 1];
      if (!strcmp(argv[i], "--batch")) batch = strtoull(argv[i + 1], nullptr, 10);
   }
   phases P(load_keys(keys));
   printf("# dict_bench engine=%s keys=%s n=%zu batch=%llu\n", engine.c_str(), keys.c_str(),
          P.keys.size(), (unsigned long long)batch);

   if (engine == "hkv")
   {
      hkv::kv_tree<std::string_view, uint64_t> t;
      run("hkv", P, [&](const std::string& k, uint64_t i) { t.insert(k, i); }, [] {},
          [&](const std::string& k) {
             uint64_t out;
             return t.find(std::string_view(k), out) ? 1u : 0u;
          },
          [&](uint64_t& cnt, uint64_t& sum) {
             for (auto it = t.begin(); it != t.end(); ++it)
             {
                ++cnt;
                sum += (*it).first.size();
             }
          },
          0);
   }
   else if (engine == "artmap")
   {
      art::art_map<uint64_t, art::heap_arena> m;
      run("artmap", P, [&](const std::string& k, uint64_t i) { m.upsert(k, i); }, [] {},
          [&](const std::string& k) { return m.get(k) ? 1u : 0u; },
          [&](uint64_t& cnt, uint64_t& sum) {
             auto it  = art::make_begin<uint64_t>(m.get_arena(), m.snapshot_root());
             auto end = art::make_end<uint64_t>(m.get_arena(), m.snapshot_root());
             for (; it != end; ++it)
             {
                ++cnt;
                sum += it.key().size();
             }
          },
          0);
   }
   else if (engine == "part" || engine == "vpart")
   {
      // identical body, two namespaces — the A/B Dan asked for
      auto drive = [&](auto db, const char* tag) {
         auto ws = db->start_write_session();
         using tx_t = decltype(ws->start_transaction(0));
         std::optional<tx_t> tx;
         tx.emplace(ws->start_transaction(0));
         run(tag, P,
             [&](const std::string& k, uint64_t i) { tx->upsert(k, val_of(i)); },
             [&] {
                tx->commit();
                tx.emplace(ws->start_transaction(0));
             },
             [&](const std::string& k) {
                return tx->template get<std::string>(k).has_value() ? 1u : 0u;
             },
             [&](uint64_t& cnt, uint64_t& sum) {
                auto cur = ws->get_root(0).snapshot_cursor();
                cur.seek_begin();
                while (!cur.is_end())
                {
                   ++cnt;
                   sum += cur.key().size();
                   cur.next();
                }
             },
             batch);
         tx->abort();
      };
      if (engine == "part")
         drive(part::kv::database::create(tmpdb("part")), "part");
      else
         drive(vpart::kv::database::create(tmpdb("vpart")), "vpart");
   }
   else if (engine == "heart")
   {
      auto db = heart::radix::database::open(tmpdb("heart"), sal::open_mode::create_or_open);
      auto ws = db->start_write_session();
      auto rs = db->start_read_session();
      using tx_t = decltype(ws->start_transaction(0));
      std::optional<tx_t> tx;
      tx.emplace(ws->start_transaction(0));
      using cur_t = decltype(rs->create_cursor(0));
      std::optional<cur_t> rcur;  // ONE pinned cursor per phase (matches the
                                  // tx->get the other engines use per find)
      run("heart", P,
          [&](const std::string& k, uint64_t i) { tx->upsert(k, val_of(i)); },
          [&] {
             tx->commit();
             tx.emplace(ws->start_transaction(0));
          },
          [&](const std::string& k) {
             if (!rcur)
                rcur.emplace(rs->create_cursor(0));
             return rcur->template get<std::string>(k).has_value() ? 1u : 0u;
          },
          [&](uint64_t& cnt, uint64_t& sum) {
             auto cur = rs->create_cursor(0);
             if (cur.seek_begin())
                do
                {
                   ++cnt;
                   sum += cur.key().size();
                } while (cur.next());
          },
          batch);
      tx->abort();
   }
   else
   {
      fprintf(stderr, "unknown engine %s\n", engine.c_str());
      return 1;
   }
   return 0;
}

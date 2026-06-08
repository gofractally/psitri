// System-dictionary query benchmark: load /usr/share/dict/words, insert every
// word, then time SEQUENTIAL (dictionary order) and RANDOM point queries — for
// both the heart sal_radix_tree (unique-mode, inline tail values, path
// compressed) and default psitri, so the two are directly comparable.
//
// Matches psitri's own dictionary-benchmark pattern (see
// libraries/psitri/tests/mvcc_tests.cpp load_dictionary / seed_dictionary).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_session_impl.hpp>

#include <sal_radix_tree.hpp>  // libraries/heart/test (added to include path)
#include <radix_engine.hpp>

using heart::sal_proto::sal_radix_tree;

namespace
{
   using clock_t_ = std::chrono::steady_clock;
   double ns_per(std::chrono::nanoseconds d, size_t n) { return double(d.count()) / double(n); }
   auto   now() { return clock_t_::now(); }
   std::chrono::nanoseconds el(clock_t_::time_point a, clock_t_::time_point b)
   {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a);
   }

   std::vector<std::string> load_dictionary(const char* path)
   {
      std::vector<std::string> words;
      std::ifstream            f(path);
      std::string              line;
      while (std::getline(f, line))
         if (!line.empty())
            words.push_back(std::move(line));
      return words;
   }

   std::string_view kspan(const std::string& s) { return s; }  // key_view
}  // namespace

int main(int argc, char** argv)
{
   const char* dict = (argc > 1) ? argv[1] : "/usr/share/dict/words";
   auto        words = load_dictionary(dict);
   if (words.size() < 1000)
   {
      std::fprintf(stderr, "could not load dictionary at %s\n", dict);
      return 2;
   }
   const size_t N = words.size();

   // A fixed random query order, shared by both engines.
   std::vector<uint32_t> order(N);
   for (size_t i = 0; i < N; ++i)
      order[i] = uint32_t(i);
   std::shuffle(order.begin(), order.end(), std::mt19937(12345));

   // Guaranteed-absent keys: each word with a 0xFF byte appended (no dictionary
   // word contains 0xFF, and "word\xff" is never another word). Precomputed so
   // the miss timing measures the lookup, not string building.
   std::vector<std::string> absent(N);
   for (size_t i = 0; i < N; ++i)
      absent[i] = words[i] + '\xff';

   std::printf("dictionary: %zu words from %s\n\n", N, dict);

   // ──────────────────────── new engine (designed taxonomy) ─────────────────
   {
      auto                          dir = std::filesystem::temp_directory_path() / "heart_radix_dict";
      heart::radix::radix_tree      t(dir);
      if (std::getenv("WIDE"))
         t.set_wide_stems(true);
      auto vview = [](const uint64_t& v) {
         return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v));
      };
      auto i0 = now();
      for (size_t i = 0; i < N; ++i)
         t.upsert(kspan(words[i]), vview(uint64_t(i)));
      auto i1 = now();
      auto query = [&](const std::string& w, uint64_t& acc) {
         std::string_view v;
         if (t.find(kspan(w), v))
         {
            uint64_t got = 0;
            std::memcpy(&got, v.data(), std::min<size_t>(8, v.size()));
            acc += got;
         }
      };
      uint64_t seq = 0;
      auto     s0  = now();
      for (size_t i = 0; i < N; ++i)
         query(words[i], seq);
      auto     s1  = now();
      uint64_t rnd = 0;
      auto     r0  = now();
      for (size_t i = 0; i < N; ++i)
         query(words[order[i]], rnd);
      auto     r1        = now();
      uint64_t missfound = 0;
      auto     m0        = now();
      for (size_t i = 0; i < N; ++i)
      {
         std::string_view v;
         if (t.find(kspan(absent[order[i]]), v))
            ++missfound;
      }
      auto           m1     = now();
      const uint64_t expect = uint64_t(N) * (N - 1) / 2;
      auto           tc     = t.count_tiers();
      std::printf("  heart engine         : insert %6.1f   seq %6.1f   rand %6.1f   MISS %6.1f ns"
                  "   [%s]  %llu objects\n",
                  ns_per(el(i0, i1), N), ns_per(el(s0, s1), N), ns_per(el(r0, r1), N),
                  ns_per(el(m0, m1), N),
                  (seq == expect && rnd == expect && missfound == 0) ? "OK" : "MISMATCH",
                  (unsigned long long)t.object_count());
      std::printf("    tiers: %llu leaf, %llu setlist, %llu r2, %llu r4, %llu full\n",
                  (unsigned long long)tc.leaf, (unsigned long long)tc.setlist,
                  (unsigned long long)tc.bitset_r2, (unsigned long long)tc.bitset_r4,
                  (unsigned long long)tc.full);
   }

   // ──────────────────────── new engine, BUCKET MODE ON (same harness) ──────────
   {
      heart::radix::radix_tree t(std::filesystem::temp_directory_path() / "heart_bucket_dict");
      t.set_adaptive(true);
      auto vview = [](const uint64_t& v) {
         return std::string_view(reinterpret_cast<const char*>(&v), sizeof(v));
      };
      auto i0 = now();
      for (size_t i = 0; i < N; ++i) t.upsert(kspan(words[i]), vview(uint64_t(i)));
      auto i1 = now();
      auto query = [&](const std::string& w, uint64_t& acc) {
         std::string_view v;
         if (t.find(kspan(w), v)) { uint64_t g = 0; std::memcpy(&g, v.data(), std::min<size_t>(8, v.size())); acc += g; }
      };
      uint64_t seq = 0; auto s0 = now();
      for (size_t i = 0; i < N; ++i) query(words[i], seq);
      auto s1 = now();
      uint64_t rnd = 0; auto r0 = now();
      for (size_t i = 0; i < N; ++i) query(words[order[i]], rnd);
      auto r1 = now();
      uint64_t missfound = 0; auto m0 = now();
      for (size_t i = 0; i < N; ++i) { std::string_view v; if (t.find(kspan(absent[order[i]]), v)) ++missfound; }
      auto           m1     = now();
      const uint64_t expect = uint64_t(N) * (N - 1) / 2;
      std::printf("  heart ADAPTIVE       : insert %6.1f   seq %6.1f   rand %6.1f   MISS %6.1f ns"
                  "   [%s]  %llu objects\n",
                  ns_per(el(i0, i1), N), ns_per(el(s0, s1), N), ns_per(el(r0, r1), N),
                  ns_per(el(m0, m1), N),
                  (seq == expect && rnd == expect && missfound == 0) ? "OK" : "MISMATCH",
                  (unsigned long long)t.object_count());
   }

   // ───────────────────────────── heart (prototype) ─────────────────────────
   {
      auto dir = std::filesystem::temp_directory_path() / "heart_dict_bench";
      sal_radix_tree t(dir);

      auto i0 = now();
      for (size_t i = 0; i < N; ++i)
         t.insert(kspan(words[i]), uint64_t(i));
      auto i1 = now();

      uint64_t seq = 0;
      auto     s0  = now();
      for (size_t i = 0; i < N; ++i)
      {
         uint64_t v = 0;
         if (t.find(kspan(words[i]), v))
            seq += v;
      }
      auto s1 = now();

      uint64_t rnd = 0;
      auto     r0  = now();
      for (size_t i = 0; i < N; ++i)
      {
         uint64_t v = 0;
         if (t.find(kspan(words[order[i]]), v))
            rnd += v;
      }
      auto r1 = now();

      uint64_t missfound = 0;
      auto     m0        = now();
      for (size_t i = 0; i < N; ++i)
      {
         uint64_t v = 0;
         if (t.find(kspan(absent[order[i]]), v))
            ++missfound;
      }
      auto m1 = now();

      const uint64_t expect = uint64_t(N) * (N - 1) / 2;
      auto           tc     = t.count_tiers();
      std::printf("  heart sal_radix_tree : insert %6.1f   seq %6.1f   rand %6.1f   MISS %6.1f ns"
                  "   [%s]  %llu objects, %zu MB\n",
                  ns_per(el(i0, i1), N), ns_per(el(s0, s1), N), ns_per(el(r0, r1), N),
                  ns_per(el(m0, m1), N),
                  (seq == expect && rnd == expect && missfound == 0) ? "OK" : "MISMATCH",
                  (unsigned long long)t.object_count(), t.bytes_used() >> 20);
      std::printf("    tiers: %llu leaf, %llu setlist, %llu bitset, %llu full\n",
                  (unsigned long long)tc.leaf, (unsigned long long)tc.setlist,
                  (unsigned long long)tc.bitset, (unsigned long long)tc.full);
      auto ds = t.measure_depth();
      std::printf("    depth: avg %.2f max %llu hops;  keys/leaf avg %.1f max %llu (%llu leaves)\n",
                  ds.nkeys ? double(ds.sum_depth) / double(ds.nkeys) : 0.0,
                  (unsigned long long)ds.max_depth,
                  ds.nleaves ? double(ds.leaf_key_sum) / double(ds.nleaves) : 0.0,
                  (unsigned long long)ds.leaf_max, (unsigned long long)ds.nleaves);
   }

   // ───────────────────────────── psitri ────────────────────────────
   {
      auto dir = std::filesystem::temp_directory_path() / "psitri_dict_bench";
      std::filesystem::remove_all(dir);
      psitri::runtime_config cfg;
      cfg.sync_mode               = sal::sync_type::none;
      cfg.write_protect_on_commit = false;
      cfg.checksum_on_commit      = false;
      auto db  = psitri::database::create(dir.string(), cfg);
      auto ses = db->start_write_session();

      auto i0 = now();
      {
         auto tx = ses->start_transaction(0);
         for (size_t i = 0; i < N; ++i)
         {
            uint64_t v = i;
            tx.insert(psitri::key_view(words[i].data(), words[i].size()),
                      psitri::value_view(reinterpret_cast<const char*>(&v), sizeof(v)));
         }
         tx.commit();
      }
      auto i1 = now();

      auto rootp = ses->get_root(0);
      auto query = [&](const std::string& w, uint64_t& acc)
      {
         psitri::cursor cur(rootp);
         std::string    val;
         if (cur.get(psitri::key_view(w.data(), w.size()), &val) >= 0 && val.size() == 8)
         {
            uint64_t v;
            std::memcpy(&v, val.data(), 8);
            acc += v;
         }
      };

      uint64_t seq = 0;
      auto     s0  = now();
      for (size_t i = 0; i < N; ++i)
         query(words[i], seq);
      auto s1 = now();

      uint64_t rnd = 0;
      auto     r0  = now();
      for (size_t i = 0; i < N; ++i)
         query(words[order[i]], rnd);
      auto r1 = now();

      uint64_t missfound = 0;
      auto     m0        = now();
      for (size_t i = 0; i < N; ++i)
      {
         const std::string& w = absent[order[i]];
         psitri::cursor     cur(rootp);
         std::string        val;
         if (cur.get(psitri::key_view(w.data(), w.size()), &val) >= 0)
            ++missfound;
      }
      auto m1 = now();

      const uint64_t expect = uint64_t(N) * (N - 1) / 2;
      std::printf("  psitri (1 tx)        : insert %6.1f   seq %6.1f   rand %6.1f   MISS %6.1f ns"
                  "   [%s]\n",
                  ns_per(el(i0, i1), N), ns_per(el(s0, s1), N), ns_per(el(r0, r1), N),
                  ns_per(el(m0, m1), N),
                  (seq == expect && rnd == expect && missfound == 0) ? "OK" : "MISMATCH");
   }

   std::fflush(stdout);
   return 0;
}

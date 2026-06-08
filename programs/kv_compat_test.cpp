// Engine-agnostic correctness suite: ONE templated body, parameterized on the engine's NATIVE
// database type (psitri::database or heart::kv::database) — no invented per-engine type. Every
// call is made natively and identically: DB::create → start_write_session → start_transaction →
// upsert/remove/commit, and reads via transaction::get(key, lambda). Each result is
// cross-checked against std::map. Exit 0 iff both engines pass every check.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "heart_kv.hpp"  // heart's native psitri-shaped API (heart::kv::database)

#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_session_impl.hpp>

namespace
{
   uint64_t    hashed(uint64_t i) { return (i * 0x9E3779B97F4A7C15ull) ^ (i >> 29); }
   std::string u64key(uint64_t x) { return std::string(reinterpret_cast<const char*>(&x), sizeof(x)); }
   std::string mkval(uint64_t seed)
   {
      size_t      len = 1 + (seed % 40);
      std::string v(len, 0);
      for (size_t j = 0; j < len; ++j)
         v[j] = char(uint8_t(seed * 31 + j));
      return v;
   }
   std::filesystem::path tmp(const char* n) { return std::filesystem::temp_directory_path() / n; }

   // Run the same battery on whatever native database type DB is (cursor deduced from get_root).
   template <class DB>
   bool run(const char* name, const std::filesystem::path& dir)
   {
      std::filesystem::remove_all(dir);
      long checks = 0, fails = 0;
      auto ok = [&](bool c) { ++checks; fails += !c; };

      auto db  = DB::create(dir);              // native, default config
      auto ses = db->start_write_session();    // native

      std::map<std::string, std::string> oracle;
      std::mt19937_64                    rng(0xBADC0FFEE);
      const int                          N = 20000, B = 2048;

      // present-with-value / absent checks against the latest committed state, via a read txn.
      auto verify_present = [&]() {
         auto tx = ses->start_transaction(0);
         for (auto& [k, v] : oracle)
         {
            bool match = false;
            bool found = tx.get(k, [&](std::string_view got) { match = (got == v); });
            ok(found && match);
         }
         tx.abort();
      };
      auto verify_absent = [&](const std::vector<std::string>& keys) {
         auto tx = ses->start_transaction(0);
         for (auto& k : keys)
            ok(!tx.get(k, [](std::string_view) {}));
         tx.abort();
      };
      auto upsert_pairs = [&](const std::vector<std::pair<std::string, std::string>>& kv) {
         for (size_t base = 0; base < kv.size(); base += B)
         {
            auto   tx  = ses->start_transaction(0);
            size_t end = std::min(kv.size(), base + size_t(B));
            for (size_t i = base; i < end; ++i)
               tx.upsert(kv[i].first, kv[i].second);  // overwrite semantics (set key=value)
            tx.commit();
         }
      };
      auto remove_keys = [&](const std::vector<std::string>& keys, size_t count) {
         for (size_t base = 0; base < count; base += B)
         {
            auto   tx  = ses->start_transaction(0);
            size_t end = std::min(count, base + size_t(B));
            for (size_t i = base; i < end; ++i)
               tx.remove(keys[i]);
            tx.commit();
         }
      };

      // ── Phase 1: batched upsert of mixed-shape keys (binary u64 + prefix-sharing strings)
      std::vector<std::pair<std::string, std::string>> kv;
      for (int i = 0; i < N; ++i)
      {
         std::string k = (i & 1) ? u64key(hashed(uint64_t(i)))
                                 : ("str/" + std::to_string(hashed(uint64_t(i)) % 100000));
         kv.emplace_back(k, mkval(uint64_t(i)));
         oracle[k] = kv.back().second;  // dup keys → last wins (matches upsert + std::map)
      }
      for (auto k : {std::string(""), std::string("p"), std::string("pre"), std::string("prefix")})
      {
         kv.emplace_back(k, mkval(k.size() + 7));
         oracle[k] = kv.back().second;
      }
      upsert_pairs(kv);
      verify_present();
      {
         std::vector<std::string> miss;
         for (int i = N; i < N + 3000; ++i)
            miss.push_back(u64key(hashed(uint64_t(i))));
         verify_absent(miss);
      }

      // ── Iterator: a full forward scan (get_root().snapshot_cursor() + seek_begin + next)
      // must enumerate keys in std::map order with matching values — the psitri pattern.
      {
         auto root = ses->get_root(0);
         auto c    = root.snapshot_cursor();
         c.seek_begin();
         auto it  = oracle.begin();
         long cnt = 0;
         while (!c.is_end() && it != oracle.end())
         {
            auto v = c.template value<std::string>();
            ok(std::string(c.key()) == it->first && v && *v == it->second);
            ++it;
            ++cnt;
            c.next();
         }
         std::printf("    [%s] forward scan enumerated %ld of %zu\n", name, cnt, oracle.size());
         ok(c.is_end() && it == oracle.end());  // both exhausted together — same length & order
      }
      // ── Iterator range: lower_bound(lo) .. up to (not incl) hi, vs the std::map range.
      {
         std::string lo = "str/5";  // a prefix-ish bound that lands mid-tree
         std::string hi = "str/8";
         auto        c  = ses->get_root(0).snapshot_cursor();
         c.lower_bound(lo);
         auto it = oracle.lower_bound(lo);
         while (!c.is_end() && std::string(c.key()) < hi)
         {
            auto v = c.template value<std::string>();
            ok(it != oracle.end() && std::string(c.key()) == it->first && v && *v == it->second);
            ++it;
            c.next();
         }
         ok(it == oracle.lower_bound(hi));  // oracle's matching range exhausted at hi too
      }

      // ── Phase 2: update every key with a new value (upsert overwrite)
      {
         std::vector<std::pair<std::string, std::string>> upd;
         for (auto& [k, v] : oracle)
            upd.emplace_back(k, mkval(rng()));
         upsert_pairs(upd);
         for (auto& [k, nv] : upd)
            oracle[k] = nv;
      }
      verify_present();

      // ── Phase 3: remove a random half
      {
         std::vector<std::string> keys;
         for (auto& [k, v] : oracle)
            keys.push_back(k);
         std::shuffle(keys.begin(), keys.end(), rng);
         size_t half = keys.size() / 2;
         remove_keys(keys, half);
         for (size_t i = 0; i < half; ++i)
            oracle.erase(keys[i]);
         verify_absent({keys.begin(), keys.begin() + half});
      }
      verify_present();

      // ── Phase 4: mixed upsert/remove fuzz over a bounded key space, batched
      {
         const int OPS = 40000;
         int       op  = 0;
         while (op < OPS)
         {
            auto tx = ses->start_transaction(0);
            for (int b = 0; b < B && op < OPS; ++b, ++op)
            {
               std::string k = u64key(hashed(rng() % 5000));
               if ((rng() & 3) == 0)
               {
                  tx.remove(k);
                  oracle.erase(k);
               }
               else
               {
                  std::string v = mkval(rng());
                  tx.upsert(k, v);
                  oracle[k] = v;
               }
            }
            tx.commit();
         }
      }
      verify_present();

      // ── Phase 5: remove everything → all miss
      {
         std::vector<std::string> keys;
         for (auto& [k, v] : oracle)
            keys.push_back(k);
         remove_keys(keys, keys.size());
         verify_absent(keys);
      }

      std::printf("  %-7s: %ld checks, %ld failures  [%s]\n", name, checks, fails,
                  fails == 0 ? "PASS" : "FAIL");
      return fails == 0;
   }
}  // namespace

int main()
{
   std::printf("KV compat suite — one templated body on each engine's NATIVE database type\n");
   bool h = run<heart::kv::database>("heart", tmp("kvcompat_heart"));
   bool p = run<psitri::database>("psitri", tmp("kvcompat_psitri"));
   std::printf("RESULT: heart %s, psitri %s\n", h ? "PASS" : "FAIL", p ? "PASS" : "FAIL");
   return (h && p) ? 0 : 1;
}

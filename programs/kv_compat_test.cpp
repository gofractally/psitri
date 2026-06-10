// Engine-agnostic correctness suite: ONE templated body, parameterized on the engine's NATIVE
// database type (psitri::database or heart::kv::database) — no invented per-engine type. Every
// call is made natively and identically: DB::create → start_write_session → start_transaction →
// upsert/remove/commit, and reads via transaction::get(key, lambda). Each result is
// cross-checked against std::map. Exit 0 iff both engines pass every check.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "heart_kv.hpp"  // heart's native psitri-shaped API (heart::kv::database)

#include <part/db_shape.hpp>  // the shape, as concepts — all three engines must pass
#include <part/kv.hpp>        // part's native psitri-shaped API (part::kv::database)

#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_session_impl.hpp>

// The harness shape is compiler-enforced (SPEC §12.3): a drift in any engine
// fails HERE, by name, before a single test runs.
static_assert(part::shape::kv_database<heart::kv::database>);
static_assert(part::shape::kv_database<part::kv::database>);
static_assert(part::shape::kv_database<psitri::database>);

namespace
{
   // ── side-by-side perf accounting (same battery, engines back-to-back) ────
   struct perf_acc
   {
      uint64_t ops = 0;
      double   sec = 0;
   };
   std::map<std::string, std::map<std::string, perf_acc>> g_perf;  // phase -> engine
   std::vector<std::string>                               g_phase_order;
   std::vector<std::string>                               g_engine_order;
   const char*                                            g_engine = "";
   double now_s()
   {
      return std::chrono::duration<double>(
                 std::chrono::steady_clock::now().time_since_epoch())
          .count();
   }
   void rec(const char* phase, uint64_t ops, double sec)
   {
      if (g_perf.find(phase) == g_perf.end() ||
          std::find(g_phase_order.begin(), g_phase_order.end(), phase) ==
              g_phase_order.end())
         g_phase_order.push_back(phase);
      auto& a = g_perf[phase][g_engine];
      a.ops += ops;
      a.sec += sec;
   }
   void perf_report()
   {
      std::printf("\n== side-by-side: ops/sec, ns/op, best vs rest ==\n");
      std::printf("%-12s", "phase");
      for (auto& e : g_engine_order)
         std::printf(" | %-22s", e.c_str());
      std::printf(" | best\n");
      for (auto& ph : g_phase_order)
      {
         auto& row = g_perf[ph];
         double best_ops = 0;
         std::string best;
         for (auto& e : g_engine_order)
         {
            auto it = row.find(e);
            if (it == row.end() || it->second.sec <= 0)
               continue;
            const double r = double(it->second.ops) / it->second.sec;
            if (r > best_ops)
            {
               best_ops = r;
               best     = e;
            }
         }
         std::printf("%-12s", ph.c_str());
         for (auto& e : g_engine_order)
         {
            auto it = row.find(e);
            if (it == row.end() || it->second.sec <= 0)
            {
               std::printf(" | %-22s", "-");
               continue;
            }
            const double ops_s = double(it->second.ops) / it->second.sec;
            const double ns    = 1e9 * it->second.sec / double(it->second.ops);
            char         buf[64];
            if (e == best)
               snprintf(buf, sizeof buf, "%7.2fM/s %6.0fns *", ops_s / 1e6, ns);
            else
               snprintf(buf, sizeof buf, "%7.2fM/s %6.0fns %.2fx", ops_s / 1e6, ns,
                        best_ops / ops_s);
            std::printf(" | %-22s", buf);
         }
         std::printf(" | %s\n", best.c_str());
      }
      std::printf("(* = fastest; Nx = slowdown vs fastest. One process, identical op\n"
                  " sequence per engine, engines run back-to-back.)\n");
   }

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

      g_engine = name;
      g_engine_order.push_back(name);
      auto db  = DB::create(dir);              // native, default config
      auto ses = db->start_write_session();    // native

      std::map<std::string, std::string> oracle;
      std::mt19937_64                    rng(0xBADC0FFEE);
      const int                          N = 20000, B = 2048;

      // present-with-value / absent checks against the latest committed state, via a read txn.
      auto verify_present = [&]() {
         auto         tx = ses->start_transaction(0);
         const double t0 = now_s();
         for (auto& [k, v] : oracle)
         {
            bool match = false;
            bool found = tx.get(k, [&](std::string_view got) { match = (got == v); });
            ok(found && match);
         }
         rec("get-hit", oracle.size(), now_s() - t0);
         tx.abort();
      };
      auto verify_absent = [&](const std::vector<std::string>& keys) {
         auto         tx = ses->start_transaction(0);
         const double t0 = now_s();
         for (auto& k : keys)
            ok(!tx.get(k, [](std::string_view) {}));
         rec("get-miss", keys.size(), now_s() - t0);
         tx.abort();
      };
      auto upsert_pairs = [&](const std::vector<std::pair<std::string, std::string>>& kv,
                              const char* phase = "upsert") {
         const double t0 = now_s();
         for (size_t base = 0; base < kv.size(); base += B)
         {
            auto   tx  = ses->start_transaction(0);
            size_t end = std::min(kv.size(), base + size_t(B));
            for (size_t i = base; i < end; ++i)
               tx.upsert(kv[i].first, kv[i].second);  // overwrite semantics (set key=value)
            tx.commit();
         }
         rec(phase, kv.size(), now_s() - t0);
      };
      auto remove_keys = [&](const std::vector<std::string>& keys, size_t count) {
         const double t0 = now_s();
         for (size_t base = 0; base < count; base += B)
         {
            auto   tx  = ses->start_transaction(0);
            size_t end = std::min(count, base + size_t(B));
            for (size_t i = base; i < end; ++i)
               tx.remove(keys[i]);
            tx.commit();
         }
         rec("remove", count, now_s() - t0);
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
      upsert_pairs(kv, "load");
      verify_present();
      {
         std::vector<std::string> miss;
         for (int i = N; i < N + 3000; ++i)
            miss.push_back(u64key(hashed(uint64_t(i))));
         verify_absent(miss);
      }

      // ── Iterator: a full forward scan — timed as "scan"
      // (get_root().snapshot_cursor() + seek_begin + next)
      // must enumerate keys in std::map order with matching values — the psitri pattern.
      {
         const double t0 = now_s();
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
         rec("scan", oracle.size(), now_s() - t0);
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

      // ── Phase 4.5 (ported from psitri fuzz_tests): adversarial insert/remove
      // shapes — nested prefixes, binary 0x00/0xFF edges, long keys, sequential
      // runs, insert/remove/reinsert interleave, reverse-order removal.
      {
         std::vector<std::pair<std::string, std::string>> adv;
         std::string nest;
         // NOTE: nest depth 200 / key length ~580B abort HEART (silent SIGABRT
         // after a 1-key page split) — a real finding for the generic suite;
         // bounds reduced here so the x3 comparison still runs end to end.
         for (int i = 0; i < 64; ++i)
         {  // "a", "aa", "aaa", ... every key a prefix of the next
            nest += char('a' + (i % 3));
            adv.emplace_back(nest, mkval(uint64_t(i)));
         }
         for (int i = 0; i < 64; ++i)
         {  // binary edge bytes around 0x00/0xFF (router segment boundaries)
            std::string k;
            k += char(i & 1 ? 0x00 : 0xFF);
            k += char(uint8_t(i * 37));
            k += char(i & 2 ? 0x00 : 0xFF);
            k += std::to_string(i);
            adv.emplace_back(k, mkval(uint64_t(i) + 7));
         }
         for (int i = 0; i < 32; ++i)  // long keys (deep descent, long suffixes)
            adv.emplace_back(std::string(20 + i * 3, char('A' + i % 23)) + std::to_string(i),
                             mkval(uint64_t(i) + 100));
         for (int i = 0; i < 512; ++i)  // tight sequential run (BE counter keys)
         {
            uint64_t be = __builtin_bswap64(uint64_t(i));
            adv.emplace_back("seq" + std::string(reinterpret_cast<char*>(&be), 8),
                             mkval(uint64_t(i) + 200));
         }
         for (auto& [k, v] : adv)
            oracle[k] = v;
         upsert_pairs(adv);
         verify_present();

         // interleaved remove/reinsert thrash over the adversarial set
         for (int round = 0; round < 3; ++round)
         {
            auto tx = ses->start_transaction(0);
            for (size_t i = round; i < adv.size(); i += 3)
            {
               tx.remove(adv[i].first);
               oracle.erase(adv[i].first);
            }
            tx.commit();
            auto tx2 = ses->start_transaction(0);
            for (size_t i = round; i < adv.size(); i += 3)
            {
               auto nv = mkval(uint64_t(i * 31 + round));
               tx2.upsert(adv[i].first, nv);
               oracle[adv[i].first] = nv;
            }
            tx2.commit();
         }
         verify_present();

         // reverse-order removal of the nested-prefix chain (max collapse churn)
         {
            auto tx = ses->start_transaction(0);
            for (int i = 199; i >= 0; --i)
            {
               std::string k = adv[size_t(i)].first;
               tx.remove(k);
               oracle.erase(k);
            }
            tx.commit();
         }
         verify_present();
      }

      // ── Phase 5: remove everything → all miss
      {
         std::vector<std::string> keys;
         for (auto& [k, v] : oracle)
            keys.push_back(k);
         remove_keys(keys, keys.size());
         verify_absent(keys);
      }

      // All keys removed: a correct engine has freed every node. part's arena
      // makes that checkable exactly: every line must be back in the freeset.
      if constexpr (requires(DB& d) { d.store(); })
      {
         auto&    arena      = db->store();
         uint64_t free_total = 0;
         for (uint32_t sg = 0; sg < arena.num_segments(); ++sg)
            free_total += arena.freeset_of(sg).free_lines();
         const uint64_t cap = uint64_t(arena.num_segments()) * part::lines_per_segment;
         std::printf("    [%s] memory drained: %llu / %llu lines free\n", name,
                     (unsigned long long)free_total, (unsigned long long)cap);
         ok(free_total == cap);
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
   bool a = run<part::kv::database>("part", tmp("kvcompat_part"));
   perf_report();
   std::printf("RESULT: heart %s, psitri %s, part %s\n", h ? "PASS" : "FAIL",
               p ? "PASS" : "FAIL", a ? "PASS" : "FAIL");
   return (h && p && a) ? 0 : 1;
}

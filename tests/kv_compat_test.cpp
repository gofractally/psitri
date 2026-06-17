// Engine-agnostic correctness suite: ONE templated body, parameterized on the engine's NATIVE
// database type (psitri::database or sal_art_ref::kv::database) — no invented per-engine type. Every
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

#include "heart_kv.hpp"  // heart's native psitri-shaped API (sal_art_ref::kv::database)

#include <adb/concepts.hpp>  // the shape, as concepts — every engine must pass
#include <start_ref/kv.hpp>        // part's native psitri-shaped API (start_ref::kv::database)

#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_session_impl.hpp>

#include <start_ver/kv.hpp>  // the 4th engine: vpart must pass standalone (Dan)
#include <gcart/kv.hpp>      // the 5th engine: gcart (sal2 GC-ART)
#include <gcart/spec_kv.hpp> // the 6th engine: gcart::st (spec engine)

// The harness shape is compiler-enforced (SPEC §12.3): a drift in any engine
// fails HERE, by name, before a single test runs.
static_assert(adb::kv_database<sal_art_ref::kv::database>);
static_assert(adb::kv_database<start_ref::kv::database>);
static_assert(adb::kv_database<psitri::database>);
static_assert(adb::kv_database<start_ver::kv::database>);
static_assert(adb::kv_database<gcart::kv::database>);
static_assert(adb::kv_database<gcart::st::kv::database>);  // spec engine: cursor + get_root enforced here
// the spec engine's snapshot cursor is fully mergeable (delta_art tier-capable).
static_assert(adb::mergeable_cursor<decltype(std::declval<gcart::st::kv::write_session>().get_root(0).snapshot_cursor())>);
static_assert(adb::shared_snapshot<decltype(std::declval<gcart::st::kv::write_session>().get_root(0))>);

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

   // ── the lifecycle test type: exact retain/release accounting ─────────────
   // Stored payload = 8-byte id. The registry hooks count per id; quiesced
   // counts must balance EXACTLY (retains == releases once the value is gone;
   // background reclaim contexts included). Thread-safe: hooks fire on GC
   // threads (the sal type-ops contract).
   struct hookstats
   {
      std::atomic<int64_t> retains{0}, releases{0};
      void reset()
      {
         retains  = 0;
         releases = 0;
      }
   };
   hookstats g_hooks;

   // The lifecycle test type in the LOCKED traits shape (§9): a refcounted
   // user handle whose pod is an 8-byte id; retain/release count globally
   // (GC threads included — relaxed atomics, non-blocking per the contract).
   struct test_ref
   {
      static constexpr uint16_t type_id = 7;
      struct t_pod
      {
         uint64_t id;
         static void retain(void*, const t_pod&)
         {
            g_hooks.retains.fetch_add(1, std::memory_order_relaxed);
         }
         static void release(void*, const t_pod&)
         {
            g_hooks.releases.fetch_add(1, std::memory_order_relaxed);
         }
      };
      uint64_t id    = 0;
      bool     owned = false;
      t_pod    take_pod() &&
      {
         owned = false;  // the database now owns this reference
         g_hooks.retains.fetch_add(1, std::memory_order_relaxed);  // custody -> pod
         return t_pod{id};
      }
      static test_ref make_from_pod(const t_pod& p, bool retain)
      {
         if (retain)
            t_pod::retain(nullptr, p);
         return test_ref{p.id, true};
      }
   };
   static_assert(adb::storable_type<test_ref>);

   // Phase T: typed values in the LOCKED API (§9/§10) — take/make traits,
   // THREE-OUTCOME reads, any_value. Engines without the feature fail RED.
   template <class DB>
   void phase_typed_values(DB& db, long& checks, long& fails, auto&& ok)
   {
      if constexpr (!adb::typed_values<DB, test_ref>)
      {
         ok(false);  // FEATURE ABSENT: typed values (traits/take-make/any_value)
         return;
      }
      else
      {
         g_hooks.reset();
         db.template register_value_types<test_ref>();
         auto ws = db.start_write_session();
         {
            auto tx = ws->start_transaction(0);
            tx.upsert("typed/a", test_ref{0xA1});  // take_pod: DB owns one ref
            tx.upsert("typed/b", test_ref{0xB2});
            tx.upsert("pod/x", "plain");  // typenum 0: NO hooks, fast path
            tx.commit();
         }
         {  // typed read: a RETAINED handle out
            auto tx = ws->start_transaction(0);
            auto r  = tx.template get<test_ref>("typed/a");
            ok(r.has_value() && r->id == 0xA1);
            // THREE OUTCOMES: wrong-pod read = invalid_type, NEVER not-found
            bool threw = false;
            try
            {
               (void)tx.template get<std::string>("typed/a");
            }
            catch (...)
            {
               threw = true;
            }
            ok(threw);
            bool threw2 = false;
            try
            {
               (void)tx.template get<test_ref>("pod/x");  // bytes read as T
            }
            catch (...)
            {
               threw2 = true;
            }
            ok(threw2);
            ok(!tx.template get<test_ref>("typed/absent").has_value());  // not-found stays optional
            // any_value: dynamic, lock-bundling
            auto v = tx.get_any("typed/b");
            ok(bool(v) && v.type() == test_ref::type_id && v.template is<test_ref>());
            auto h = v.template as<test_ref>();  // retains
            ok(h.id == 0xB2);
            bool threw3 = false;
            try
            {
               (void)v.bytes();  // typed pod as bytes = forgeable -> invalid_type
            }
            catch (...)
            {
               threw3 = true;
            }
            ok(threw3);
            ok(!bool(tx.get_any("typed/absent")));
            tx.abort();
         }
         {
            auto tx = ws->start_transaction(0);
            tx.upsert("typed/a", test_ref{0xA2});  // overwrite: release old, take new
            tx.commit();
         }
         {
            auto tx = ws->start_transaction(0);
            tx.remove("typed/b");  // remove: release
            tx.commit();
         }
         {
            auto tx = ws->start_transaction(0);
            tx.upsert("typed/c", test_ref{0xC3});
            tx.abort();  // abort: the taken pod is released (create-log symmetry)
         }
         // quiesce background reclaim; books balance to the live refs:
         // typed/a in the DB + the get<> copy + the as<> copy (test-held)
         if constexpr (requires { db.flush_releases(); })
            db.flush_releases();
         if constexpr (requires { db.flush_reaper(); })
            db.flush_reaper();
         ok(g_hooks.retains.load() - g_hooks.releases.load() == 3);
         {
            auto tx = ws->start_transaction(0);
            tx.remove("typed/a");
            tx.commit();
         }
         if constexpr (requires { db.flush_releases(); })
            db.flush_releases();
         ok(g_hooks.retains.load() - g_hooks.releases.load() == 2);  // only the test's handles
      }
   }

   // Phase M: multi-root transactions + the set_root exchange.
   template <class DB>
   void phase_multi_root(DB& db, long& checks, long& fails, auto&& ok)
   {
      auto ws = db.start_write_session();
      using TX = decltype(ws->start_transaction(0));
      if constexpr (!adb::multi_root_transaction<TX>)
      {
         ok(false);  // FEATURE ABSENT: multi-root transactions / set_root
         return;
      }
      else
      {
         {
            auto tx = ws->start_transaction(0);
            auto h0 = tx.primary();
            auto h1 = tx.open_root(1);
            h0.upsert("mr/u", "alice");
            h1.upsert("mr/o", "order");
            ok(h0.template get<std::string>("mr/u") == std::string("alice"));
            ok(!h0.template get<std::string>("mr/o").has_value());  // isolation
            tx.commit();
         }
         {
            auto tx = ws->start_transaction(0);
            auto h0 = tx.primary();
            auto h1 = tx.open_root(1);
            h0.upsert("mr/gone", "x");
            h1.upsert("mr/gone2", "y");
            tx.abort();  // all roots roll back
         }
         {
            auto tx = ws->start_transaction(0);
            ok(!tx.template get<std::string>("mr/gone").has_value());
            tx.abort();
         }
         // the rotation: hot -> warm in one exchange expression
         {
            auto tx = ws->start_transaction(0);
            auto h0 = tx.primary();
            auto h1 = tx.open_root(1);
            h1.set_root(h0.set_root({}));
            ok(!h0.template get<std::string>("mr/u").has_value());      // hot newborn
            ok(h1.template get<std::string>("mr/u") == std::string("alice"));  // warm holds it
            tx.commit();
         }
      }
   }

   // Phase S: a tree IS a snapshot. get_root(i) returns an owned, FROZEN
   // snapshot of root i; it keeps reading its as-of contents across a later
   // commit (COW keeps the version alive while the owned tree holds it).
   template <class DB>
   void phase_snapshot(DB& db, long& checks, long& fails, auto&& ok)
   {
      auto ws  = db.start_write_session();
      using WS = std::remove_reference_t<decltype(*ws)>;
      if constexpr (!adb::tree_snapshots<WS>)
      {
         ok(false);  // FEATURE ABSENT: get_root() owned frozen snapshot
         return;
      }
      else
      {
         {
            auto tx = ws->start_transaction(0);
            tx.upsert("snap/k", "v1");
            tx.commit();
         }
         auto snap = ws->get_root(0u);  // owned frozen snapshot at v1
         ok(bool(snap));
         {
            auto tx = ws->start_transaction(0);
            tx.upsert("snap/k", "v2");
            tx.commit();
         }
         // the snapshot still reads v1 — frozen by the owned reference
         auto c = snap.snapshot_cursor();
         auto v = c.template get<std::string>(std::string_view("snap/k"));
         ok(v && *v == std::string("v1"));
         // a fresh snapshot sees v2
         auto c2 = ws->get_root(0u).snapshot_cursor();
         auto v2 = c2.template get<std::string>(std::string_view("snap/k"));
         ok(v2 && *v2 == std::string("v2"));
      }
   }

   // Run the same battery on whatever native database type DB is (cursor deduced from get_root).
   template <class DB>
   bool run(const char* name, const std::filesystem::path& dir)
   {
      std::filesystem::remove_all(dir);
      long checks = 0, fails = 0;
      auto ok = [&](bool c)
      {
         ++checks;
         if (!c && fails < 8 && std::getenv("KV_TRACE"))
            std::fprintf(stderr, "  [%s] CHECK FAILED at ordinal %ld\n", g_engine, checks);
         fails += !c;
      };

      g_engine = name;
      g_engine_order.push_back(name);
      auto db  = DB::create(dir);              // native, default config
      auto ses = db->start_write_session();    // native

      std::map<std::string, std::string> oracle;
      std::mt19937_64                    rng(0xBADC0FFEE);
      const int                          N = 20000, B = 2048;

      // present-with-value / absent checks against the latest committed state, via a read txn.
      int  vp_seq         = 0;
      auto verify_present = [&]() {
         if (std::getenv("KV_TRACE"))
            std::fprintf(stderr, "  [%s] verify_present #%d (ordinal base %ld)\n", g_engine,
                         ++vp_seq, checks);
         auto         tx = ses->start_transaction(0);
         const double t0 = now_s();
         for (auto& [k, v] : oracle)
         {
            bool match = false;
            bool found = tx.get(k, [&](std::string_view got) { match = (got == v); });
            if (!(found && match) && std::getenv("KV_TRACE"))
               std::fprintf(stderr, "  [%s] get-hit MISMATCH key len=%zu found=%d first=%02x\n",
                            g_engine, k.size(), int(found), k.empty() ? 0 : uint8_t(k[0]));
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
         // Original harsher bounds RESTORED: heart's 1-key-page-split SIGABRT is fixed —
         // items whose suffix+value exceed the 128-byte in-page cap now live whole in
         // their own key_val_node (the page keeps a comparison fragment + cbase code).
         for (int i = 0; i < 200; ++i)
         {  // "a", "aa", "aaa", ... every key a prefix of the next (200-deep nesting)
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
         for (int i = 0; i < 188; ++i)  // long keys to ~580B (deep descent, big items)
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

      // ── Phase 4.7 (CONTRACT, engine-templated — Dan 2026-06-11): behaviors
      // every engine must enforce; new variants inherit these gates for free.
      {
         // (a) cursor-path depth at the key contract's limit: 600-deep nested
         // prefixes (one radix level per byte) — point find + full iteration.
         // psitri's cursor held a 128-frame path array and walked past it
         // (the find_impl SEGV); the contract is keys <= 1024 bytes, so the
         // path must survive ~1024 levels.
         if (!std::getenv("SKIP_DEEP"))
         {
            std::map<std::string, std::string> deep_oracle;
            std::string                        nest2;
            {
               auto tx = ses->start_transaction(0);
               for (int i = 0; i < 600; ++i)
               {
                  nest2 += char('a' + (i % 7));
                  std::string v = mkval(uint64_t(i) * 7);
                  tx.upsert(nest2, v);
                  deep_oracle[nest2] = v;
               }
               tx.commit();
            }
            {
               auto tx = ses->start_transaction(0);
               for (auto& [k, v] : deep_oracle)
                  ok(tx.template get<std::string>(k) == v);
               tx.abort();
            }
            {  // full iteration survives the 600-deep chain (path depth on ++)
               auto   c = ses->get_root(0).snapshot_cursor();
               size_t n = 0;
               c.seek_begin();
               while (!c.is_end())
               {
                  ++n;
                  c.next();
               }
               ok(n >= deep_oracle.size());
            }
            {
               auto tx = ses->start_transaction(0);
               for (auto& [k, v] : deep_oracle)
                  tx.remove(k);
               tx.commit();
            }
         }
         // (b) destroying an OPEN transaction must roll back, never abort the
         // process; the engine stays usable and the writes are absent.
         if (!std::getenv("SKIP_DTOR"))
         {
            const std::string ghost = "contract/uncommitted";
            {
               auto tx = ses->start_transaction(0);
               tx.upsert(ghost, "must-vanish");
               // dropped without commit
            }
            {
               auto tx = ses->start_transaction(0);
               ok(!tx.template get<std::string>(ghost).has_value());
               tx.upsert(ghost, "second-life");  // engine remains writable
               tx.commit();
            }
            {
               auto tx = ses->start_transaction(0);
               ok(tx.template get<std::string>(ghost) == std::string("second-life"));
               tx.remove(ghost);
               tx.commit();
            }
         }
         // (c) a key beyond an engine's contract must be REJECTED as input (a
         // structured throw, engine stays usable) or stored faithfully — never
         // trusted as a debug-only invariant. Heart's 1024-byte contract was an
         // assert: under NDEBUG an oversized key routed through fixed stack
         // buffers (silent smash). Part/psitri store it; heart now throws.
         if (!std::getenv("SKIP_BIGKEY"))
         {
            const std::string big(1500, 'K');
            bool              stored = false;
            try
            {
               auto tx = ses->start_transaction(0);
               tx.upsert(big, "big-key-value");
               tx.commit();
               stored = true;
            }
            catch (const std::exception&)
            {
               // structured rejection — usability is checked below
            }
            if (stored)
            {
               auto tx = ses->start_transaction(0);
               ok(tx.template get<std::string>(big) == std::string("big-key-value"));
               tx.remove(big);
               tx.commit();
            }
            {
               const std::string canary = "contract/after-bigkey";
               {
                  auto tx = ses->start_transaction(0);
                  tx.upsert(canary, "alive");
                  tx.commit();
               }
               auto tx = ses->start_transaction(0);
               ok(tx.template get<std::string>(canary) == std::string("alive"));
               tx.remove(canary);
               tx.commit();
            }
         }
      }

      // ── Phase 4.7: LARGE VALUES — spanning a typical inline node cap (gcart's
      // 8 KB) up to 128 KB (Silkworm contract code is ~24-38 KB). Any adb kv
      // engine must store, read back byte-exact (point get + cursor), overwrite
      // (large→small, small→large, large→large'), keep a snapshot frozen across
      // a large overwrite, and free them on remove (the drain gate below). These
      // keys go into `oracle` so Phase 5 removal + the drain census cover them.
      {
         auto bigval = [](size_t n, uint64_t seed) {
            std::string v(n, 0);
            uint64_t    x = seed * 0x9E3779B97F4A7C15ull + 1;
            for (size_t i = 0; i < n; ++i)
            { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v[i] = char(uint8_t(x)); }
            return v;
         };
         const size_t sizes[] = {64, 1000, 4096, 8000, 8192, 9000, 16384, 38813, 131072};
         std::vector<std::pair<std::string, std::string>> lv;
         for (size_t s : sizes)
         {
            std::string k = "large/" + std::to_string(s);
            std::string v = bigval(s, s + 1);
            lv.emplace_back(k, v);
            oracle[k] = v;
         }
         upsert_pairs(lv, "large-load");

         {  // byte-exact point get
            auto tx = ses->start_transaction(0);
            for (auto& [k, v] : lv)
               ok(tx.template get<std::string>(k) == v);
            tx.abort();
         }
         {  // byte-exact cursor value() at each large key
            auto c = ses->get_root(0).snapshot_cursor();
            for (auto& [k, v] : lv)
            { c.lower_bound(k); ok(!c.is_end() && std::string(c.key()) == k && *c.template value<std::string>() == v); }
         }
         {  // overwrites that cross the inline/large boundary in both directions
            auto tx = ses->start_transaction(0);
            std::string a = "large/38813", av = "small-now";          tx.upsert(a, av); oracle[a] = av;
            std::string b = "large/64",    bv = bigval(50000, 999);   tx.upsert(b, bv); oracle[b] = bv;
            std::string c = "large/16384", cv = bigval(20000, 888);   tx.upsert(c, cv); oracle[c] = cv;
            tx.commit();
            auto rx = ses->start_transaction(0);
            ok(rx.template get<std::string>(a) == oracle[a]);
            ok(rx.template get<std::string>(b) == oracle[b]);
            ok(rx.template get<std::string>(c) == oracle[c]);
            rx.abort();
         }
         {  // snapshot stays frozen across a large-value overwrite (MVCC + big val)
            std::string k    = "large/131072";
            auto        snap = ses->get_root(0);
            auto        sval = snap.template get<std::string>(k);
            ok(sval.has_value() && sval->size() == 131072);
            { auto tx = ses->start_transaction(0); tx.upsert(k, "tiny"); oracle[k] = "tiny"; tx.commit(); }
            ok(snap.template get<std::string>(k) == sval);  // old big value still readable via the pin
            ok(ses->start_transaction(0).template get<std::string>(k) == std::string("tiny"));
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

      // All keys removed: a correct engine has freed every node. part's arena
      // makes that checkable exactly: every line must be back in the freeset.
      // QUIESCENT means no live sessions: a session legitimately caches
      // claimed lines (fcache/window) while alive — end it before the census.
      ses.reset();
      if constexpr (requires(DB& d) { d.store(); })
      {
         if constexpr (requires(DB& d) { d.flush_releases(); })
            db->flush_releases();  // SPEC §19: the drain gate waits for the releaser
         auto&    arena      = db->store();
         uint64_t free_total = 0;
         for (uint32_t sg = 0; sg < arena.num_segments(); ++sg)
            free_total += arena.freeset_of(sg).free_lines();
         const uint64_t cap = uint64_t(arena.num_segments()) * start_ref::lines_per_segment;
         std::printf("    [%s] memory drained: %llu / %llu lines free\n", name,
                     (unsigned long long)free_total, (unsigned long long)cap);
         ok(free_total == cap);
      }
      // ── concept feature batteries (Dan: spec-first, every engine runs
      //    them; absent features fail RED until implemented) ───────────────
      {
         const long f0 = fails;
         {
            std::filesystem::remove_all(dir / "feat");
            auto fdb = DB::create(dir / "feat");
            phase_typed_values(*fdb, checks, fails, ok);
            phase_multi_root(*fdb, checks, fails, ok);
            phase_snapshot(*fdb, checks, fails, ok);
         }
         if (fails != f0)
            std::printf("    [%s] concept-extension phases: %ld failures "
                        "(typed-values / multi-root / snapshot)\n",
                        name, fails - f0);
      }
      std::printf("  %-7s: %ld checks, %ld failures  [%s]\n", name, checks, fails,
                  fails == 0 ? "PASS" : "FAIL");
      return fails == 0;
   }
}  // namespace

int main()
{
   std::printf("KV compat suite — one templated body on each engine's NATIVE database type\n");
   auto guarded = [](const char* name, auto&& fn) {
      try
      {
         return fn();
      }
      catch (const std::exception& e)
      {
         std::printf("  %-7s: EXCEPTION: %s  [FAIL]\n", name, e.what());
         return false;
      }
   };
   const char* only = std::getenv("KV_ONLY");  // fast single-engine iteration
   auto want = [&](const char* n) { return !only || std::string(only) == n; };
   bool h = !want("sal_art_ref") ||
            guarded("sal_art_ref", [&] { return run<sal_art_ref::kv::database>("sal_art_ref", tmp("kvcompat_sal_art_ref")); });
   bool p = !want("psitri") ||
            guarded("psitri", [&] { return run<psitri::database>("psitri", tmp("kvcompat_psitri")); });
   bool a = !want("start_ref") ||
            guarded("start_ref", [&] { return run<start_ref::kv::database>("start_ref", tmp("kvcompat_start_ref")); });
   bool v = !want("start_ver") ||
            guarded("start_ver", [&] { return run<start_ver::kv::database>("start_ver", tmp("kvcompat_start_ver")); });
   bool g = !want("gcart") ||
            guarded("gcart", [&] { return run<gcart::kv::database>("gcart", tmp("kvcompat_gcart")); });
   perf_report();
   std::printf("RESULT: sal_art_ref %s, psitri %s, start_ref %s, start_ver %s, gcart %s\n",
               h ? "PASS" : "FAIL", p ? "PASS" : "FAIL", a ? "PASS" : "FAIL", v ? "PASS" : "FAIL",
               g ? "PASS" : "FAIL");
   return (h && p && a && v && g) ? 0 : 1;
}

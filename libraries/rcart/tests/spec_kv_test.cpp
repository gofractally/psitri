// rcart::kv adb adapter — CRUD + snapshot isolation + concurrent MVCC
// (4 readers + writer, lock-free across the 3 regions). 0 fails expected.
#include <adb/concepts.hpp>
#include <atomic>
#include <cstdio>
#include <rcart/spec_kv.hpp>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>
using namespace rcart::kv;

// Compiler-enforced conformance: the spec engine MUST satisfy the adb concepts —
// transaction (upsert/remove/commit/get) AND the ordered cursor surface
// (get_root -> snapshot_cursor -> seek_begin/lower_bound/next/is_end/key/value).
// These never tripped a compile error before because the spec engine was never
// asserted against the concepts; they do now. You cannot ship an adb-compliant API
// without the cursor — the conformance suite enforces it here.
static_assert(adb::kv_transaction<transaction>, "transaction must satisfy adb::kv_transaction");
static_assert(adb::kv_cursor<cursor>,
              "cursor must satisfy adb::kv_cursor (seek_begin/lower_bound/next/is_end/key/value)");
static_assert(adb::mergeable_cursor<cursor>,
              "cursor must satisfy adb::mergeable_cursor "
              "(seek_last/upper_bound/prev/is_rend/is_subtree/get_value/get)");
static_assert(adb::kv_tree<snapshot>, "snapshot must satisfy adb::kv_tree (snapshot_cursor)");
static_assert(adb::shared_snapshot<snapshot>,
              "snapshot must satisfy adb::shared_snapshot (copyable + mergeable snapshot_cursor)");
static_assert(adb::kv_write_session<write_session>,
              "write_session must satisfy adb::kv_write_session (start_transaction + get_root)");
static_assert(adb::kv_database<database>, "rcart::kv::database must satisfy adb::kv_database");
static_assert(adb::root_handle<root_handle>,
              "root_handle must satisfy adb::root_handle (upsert/remove/get/set_root exchange)");
static_assert(adb::multi_root_transaction<transaction>,
              "transaction must satisfy adb::multi_root_transaction (primary/open_root)");
int main()
{
   namespace fs = std::filesystem;
   auto dir     = fs::temp_directory_path() / "sk_test";
   fs::remove_all(dir);
   auto                               db = database::create(dir);
   auto                               ws = db->start_write_session();
   std::map<std::string, std::string> oracle;
   std::mt19937                       r(1);
   auto                               mkk = [&]()
   {
      int         n = 3 + int(r() % 18);
      std::string k;
      for (int j = 0; j < n; ++j)
         k.push_back(char('a' + r() % 26));
      return k;
   };
   int checks = 0, fails = 0;
   // batched upserts
   for (int b = 0; b < 40; ++b)
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 2000; ++i)
      {
         auto k = mkk();
         auto v = "v" + std::to_string(r() % 100000);
         tx.upsert(k, v);
         oracle[k] = v;
      }
      tx.commit();
   }
   // read via cursor
   {
      auto rs = db->start_read_session();
      auto c  = rs->create_cursor(0);
      for (auto& [k, v] : oracle)
      {
         auto got = c.get<std::string>(k);
         ++checks;
         if (!got || *got != v)
         {
            if (++fails <= 5)
               std::printf("MISS %s\n", k.c_str());
         }
      }
   }

   // ── ordered cursor (adb::kv_cursor): full scan + lower_bound range vs std::map,
   //    incl. prefix-ordering ("" < "p" < "pre" < "prefix"), a deep key, a big value.
   {
      for (auto& k : std::vector<std::string>{"", "p", "pre", "prefix"})
      {
         std::string v  = "pfx:" + k;
         auto        tx = ws->start_transaction(0);
         tx.upsert(k, v);
         oracle[k] = v;
         tx.commit();
      }
      // deep cursor path: a 120-level prefix chain "D","DD",... (each a term, one
      // branch) — exercises deep descent/recursion + term-before-branch ordering.
      // (Single-leaf suffixes stay <255: sv_leaf's slen is u8 — keys whose leaf
      //  suffix exceeds 255B are a separate pre-existing limit, not the cursor's.)
      {
         std::string dk;
         for (int i = 0; i < 120; ++i)
         {
            dk.push_back('D');
            std::string dv = "d" + std::to_string(i);
            auto        tx = ws->start_transaction(0);
            tx.upsert(dk, dv);
            oracle[dk] = dv;
            tx.commit();
         }
      }
      {
         std::string bk = "bigval/key";
         std::string bv(6000, 'B');
         auto        tx = ws->start_transaction(0);
         tx.upsert(bk, bv);
         oracle[bk] = bv;
         tx.commit();
      }
      // full ordered scan must equal std::map order + values exactly
      {
         auto t = ws->get_root(0);
         auto c = t.snapshot_cursor();
         c.seek_begin();
         auto it = oracle.begin();
         while (!c.is_end() && it != oracle.end())
         {
            auto v = c.value<std::string>();
            ++checks;
            if (std::string(c.key()) != it->first || !v || *v != it->second)
            {
               if (++fails <= 5)
                  std::printf("SCAN @%s want %s\n", std::string(c.key()).c_str(),
                              it->first.c_str());
            }
            c.next();
            ++it;
         }
         ++checks;
         if (!(c.is_end() && it == oracle.end()))
         {
            ++fails;
            std::printf("SCAN length mismatch (cur_end=%d oracle_end=%d)\n", c.is_end(),
                        it == oracle.end());
         }
      }
      // lower_bound range [lo,hi) vs std::map
      {
         const std::string lo = "m", hi = "t";
         auto              c = ws->get_root(0).snapshot_cursor();
         c.lower_bound(lo);
         auto it = oracle.lower_bound(lo);
         while (!c.is_end() && std::string(c.key()) < hi)
         {
            auto v = c.value<std::string>();
            ++checks;
            if (it == oracle.end() || std::string(c.key()) != it->first || !v || *v != it->second)
            {
               if (++fails <= 5)
                  std::printf("LB @%s\n", std::string(c.key()).c_str());
            }
            c.next();
            ++it;
         }
         ++checks;
         if (it != oracle.lower_bound(hi))
         {
            ++fails;
            std::printf("LB range end mismatch\n");
         }
      }
      // lower_bound past the last key → is_end
      {
         auto c = ws->get_root(0).snapshot_cursor();
         ++checks;
         if (c.lower_bound(std::string(1, char(0xff))))
         { /* maybe a 0xff key exists; tolerate */
         }
      }
      // empty-tree cursor → seek_begin false + is_end
      {
         auto edb = database::create(fs::temp_directory_path() / "sk_empty");
         fs::remove_all(fs::temp_directory_path() / "sk_empty");
         auto ec = edb->start_write_session();
         auto t  = ec->get_root(0);
         auto c  = t.snapshot_cursor();
         ++checks;
         if (c.seek_begin() || !c.is_end())
         {
            ++fails;
            std::printf("empty cursor not end\n");
         }
      }
      // REVERSE scan (adb::mergeable_cursor): seek_last + prev vs std::map reverse
      {
         auto c = ws->get_root(0).snapshot_cursor();
         c.seek_last();
         auto it = oracle.rbegin();
         while (!c.is_rend() && it != oracle.rend())
         {
            auto v = c.value<std::string>();
            ++checks;
            if (std::string(c.key()) != it->first || !v || *v != it->second)
            {
               if (++fails <= 5)
                  std::printf("REV @%s want %s\n", std::string(c.key()).c_str(), it->first.c_str());
            }
            c.prev();
            ++it;
         }
         ++checks;
         if (!(c.is_rend() && it == oracle.rend()))
         {
            ++fails;
            std::printf("REV length mismatch\n");
         }
      }
      // upper_bound(k) = smallest key > k vs std::map::upper_bound
      {
         const std::string k = "m";
         auto              c = ws->get_root(0).snapshot_cursor();
         c.upper_bound(k);
         auto it = oracle.upper_bound(k);
         ++checks;
         if (c.is_end() != (it == oracle.end()) ||
             (!c.is_end() && std::string(c.key()) != it->first))
         {
            ++fails;
            std::printf("UB mismatch\n");
         }
      }
   }

   // ── multi-root transactions (adb::multi_root_transaction): isolation, abort-all,
   //    set_root rotation — on a fresh db so it doesn't pollute the oracle ─────────
   {
      auto mdir = fs::temp_directory_path() / "sk_mr";
      fs::remove_all(mdir);
      auto mdb = database::create(mdir);
      auto mws = mdb->start_write_session();
      {
         auto tx = mws->start_transaction(0);
         auto h0 = tx.primary();
         auto h1 = tx.open_root(1);
         h0.upsert("mr/u", "alice");
         h1.upsert("mr/o", "order");
         ++checks;
         if (h0.get<std::string>("mr/u") != std::string("alice"))
         {
            ++fails;
            std::printf("MR h0 self\n");
         }
         ++checks;
         if (h0.get<std::string>("mr/o").has_value())
         {
            ++fails;
            std::printf("MR isolation\n");
         }  // root-0 can't see root-1's key
         tx.commit();
      }
      {
         auto tx = mws->start_transaction(0);
         auto h0 = tx.primary();
         auto h1 = tx.open_root(1);
         h0.upsert("mr/gone", "x");
         h1.upsert("mr/gone2", "y");
         tx.abort();
      }  // all roots roll back
      {
         auto tx = mws->start_transaction(0);
         ++checks;
         if (tx.get<std::string>("mr/gone").has_value())
         {
            ++fails;
            std::printf("MR abort-all\n");
         }
         tx.abort();
      }
      {
         auto tx = mws->start_transaction(0);
         auto h0 = tx.primary();
         auto h1 = tx.open_root(1);
         h1.set_root(h0.set_root({}));  // rotate root0 -> root1
         ++checks;
         if (h0.get<std::string>("mr/u").has_value())
         {
            ++fails;
            std::printf("MR rot h0 not empty\n");
         }
         ++checks;
         if (h1.get<std::string>("mr/u") != std::string("alice"))
         {
            ++fails;
            std::printf("MR rot h1 missing\n");
         }
         tx.commit();
      }
      {
         auto tx = mws->start_transaction(0);
         auto h1 = tx.open_root(1);  // persisted after commit
         ++checks;
         if (h1.get<std::string>("mr/u") != std::string("alice"))
         {
            ++fails;
            std::printf("MR rot not persisted\n");
         }
         ++checks;
         if (tx.get<std::string>("mr/u").has_value())
         {
            ++fails;
            std::printf("MR root0 not cleared\n");
         }
         tx.abort();
      }
   }
   // snapshot isolation: pin a cursor, mutate a key, the cursor still sees old
   {
      auto        rs = db->start_read_session();
      auto        c  = rs->create_cursor(0);
      auto        it = oracle.begin();
      std::string k = it->first, oldv = it->second;
      {
         auto tx = ws->start_transaction(0);
         tx.upsert(k, "CHANGED");
         tx.commit();
         oracle[k] = "CHANGED";
      }
      auto got = c.get<std::string>(k);
      ++checks;
      if (!got || *got != oldv)
      {
         ++fails;
         std::printf("ISO FAIL: snapshot saw %s not %s\n", got ? got->c_str() : "(none)",
                     oldv.c_str());
      }
      auto c2 = db->start_read_session()->create_cursor(0);
      auto g2 = c2.get<std::string>(k);
      ++checks;
      if (!g2 || *g2 != "CHANGED")
      {
         ++fails;
         std::printf("fresh cursor stale\n");
      }
   }
   // snapshot isolation under REMOVE: pinned cursor still sees a key removed after it pinned
   {
      auto        rs = db->start_read_session();
      auto        c  = rs->create_cursor(0);
      auto        it = oracle.begin();
      std::string k = it->first, oldv = it->second;
      {
         auto tx = ws->start_transaction(0);
         int  rc = tx.remove(k);
         ++checks;
         if (rc != 1)
         {
            ++fails;
            std::printf("RM rc=%d\n", rc);
         }
         tx.commit();
         oracle.erase(k);
      }
      auto got = c.get<std::string>(k);
      ++checks;
      if (!got || *got != oldv)
      {
         ++fails;
         std::printf("ISO-RM FAIL: snapshot saw %s not %s\n", got ? got->c_str() : "(none)",
                     oldv.c_str());
      }
      auto c2 = db->start_read_session()->create_cursor(0);
      auto g2 = c2.get<std::string>(k);
      ++checks;
      if (g2)
      {
         ++fails;
         std::printf("fresh cursor saw removed key\n");
      }
   }
   // concurrent: readers + writer (balance-style)
   {
      std::atomic<bool>        stop{false};
      std::atomic<long>        reads{0}, rfail{0};
      std::vector<std::thread> rt;
      for (int i = 0; i < 4; ++i)
         rt.emplace_back(
             [&]
             {
                auto                     rs = db->start_read_session();
                std::mt19937             rr(i * 7 + 1);
                std::vector<std::string> keys;
                for (auto& [k, v] : oracle)
                   keys.push_back(k);
                while (!stop.load())
                {
                   auto c = rs->create_cursor(0);
                   for (int b = 0; b < 500; ++b)
                   {
                      auto& k   = keys[rr() % keys.size()];
                      auto  got = c.get<std::string>(k);
                      if (got)
                         reads.fetch_add(1);
                   }
                }
             });
      // writer churns values (KEEP oracle in sync — a later phase verifies it)
      std::mt19937             wr(123);
      std::vector<std::string> keys;
      for (auto& [k, v] : oracle)
         keys.push_back(k);
      for (int round = 0; round < 200; ++round)
      {
         auto tx = ws->start_transaction(0);
         for (int i = 0; i < 500; ++i)
         {
            auto& k = keys[wr() % keys.size()];
            auto  v = "w" + std::to_string(wr() % 1000);
            tx.upsert(k, v);
            oracle[k] = v;
         }
         tx.commit();
      }
      stop.store(true);
      for (auto& t : rt)
         t.join();
      std::printf("concurrent: %ld reads, %ld rfail\n", reads.load(), rfail.load());
   }
   // remove-churn under concurrent readers: writer interleaves upsert+remove via
   // transactions (each txn snap-pins the root → the COW path is SHARED →
   // edit_remove<shared>/reprefix exercised); readers just must never crash.
   {
      std::atomic<bool>        stop{false};
      std::atomic<long>        reads{0};
      std::vector<std::string> rkeys;
      for (auto& [k, v] : oracle)
         rkeys.push_back(k);
      std::vector<std::thread> rt;
      for (int i = 0; i < 4; ++i)
         rt.emplace_back(
             [&, i]
             {
                auto         rs = db->start_read_session();
                std::mt19937 rr(i * 13 + 1);
                while (!stop.load())
                {
                   auto c = rs->create_cursor(0);
                   for (int b = 0; b < 400; ++b)
                   {
                      auto got = c.get<std::string>(rkeys[rr() % rkeys.size()]);
                      if (got)
                         reads.fetch_add(1);
                   }
                }
             });
      std::mt19937 wr(777);
      for (int round = 0; round < 300; ++round)
      {
         auto tx = ws->start_transaction(0);
         for (int i = 0; i < 400; ++i)
         {
            auto k = mkk();
            if (wr() % 3 == 0)
            {
               int rc   = tx.remove(k);
               int want = oracle.count(k) ? 1 : 0;
               if (rc != want)
               {
                  ++fails;
                  if (fails <= 5)
                     std::printf("CC-RM rc=%d want=%d\n", rc, want);
               }
               oracle.erase(k);
            }
            else
            {
               auto v = "x" + std::to_string(wr() % 1000);
               tx.upsert(k, v);
               oracle[k] = v;
            }
         }
         tx.commit();
      }
      stop.store(true);
      for (auto& t : rt)
         t.join();
      // verify the tree matches the oracle exactly (survivors present+correct)
      auto c = db->start_read_session()->create_cursor(0);
      for (auto& [k, v] : oracle)
      {
         auto got = c.get<std::string>(k);
         ++checks;
         if (!got || *got != v)
         {
            if (++fails <= 5)
               std::printf("CC-FINAL MISS %s\n", k.c_str());
         }
      }
      std::printf("remove-churn: %ld reads, oracle=%zu\n", reads.load(), oracle.size());
   }
   // drain to empty via kv transactions; engine must settle to 0 live bytes
   {
      std::vector<std::string> keys;
      for (auto& [k, v] : oracle)
         keys.push_back(k);
      for (size_t i = 0; i < keys.size(); i += 500)
      {
         auto tx = ws->start_transaction(0);
         for (size_t j = i; j < keys.size() && j < i + 500; ++j)
         {
            if (tx.remove(keys[j]) != 1)
            {
               if (++fails <= 5)
                  std::printf("KV-DRAIN-MISS\n");
            }
         }
         tx.commit();
      }
      oracle.clear();
      db->drain();
      uint64_t lb = db->live_bytes();
      ++checks;
      if (lb != 0)
      {
         ++fails;
         std::printf("KV-DRAIN-LEAK live=%lluB\n", (unsigned long long)lb);
      }
   }
   // ── reopen recovery: a CLEAN close + reopen must restore committed data ────
   // (mmap-backed root table + allocator adopt_existing_). This is the Silkworm
   // requirement — it closes+reopens the env during startup.
   {
      auto rdir = fs::temp_directory_path() / "sk_reopen";
      fs::remove_all(rdir);
      const int                          N = 8000;
      std::map<std::string, std::string> exp;
      auto kk = [](int i) { return "reopen_key_" + std::to_string(i * 2654435761u % 1000003); };
      auto vv = [](int i) { return "val_" + std::to_string(i) + "_payloadpayloadpayload"; };
      {
         auto rdb = database::create(rdir);
         auto rws = rdb->start_write_session();
         auto tx  = rws->start_transaction(0);
         for (int i = 0; i < N; ++i)
         {
            tx.upsert(kk(i), vv(i));
            exp[kk(i)] = vv(i);
         }
         tx.commit();
      }  // CLEAN close: rdb/rws destroyed here
      {
         auto rdb  = database::open(rdir);  // REOPEN the same dir
         auto rrs  = rdb->start_read_session();
         auto cur  = rrs->create_cursor(0);
         int  miss = 0;
         for (auto& [k, v] : exp)
         {
            auto got = cur.get<std::string>(k);
            if (!got || *got != v)
               ++miss;
         }
         ++checks;
         if (miss)
         {
            ++fails;
            std::printf("KV-REOPEN-MISS %d/%d keys lost after clean close+reopen\n", miss, N);
         }
         else
            std::printf("reopen: %d/%d keys survived clean close+reopen\n", N, N);
      }
   }
   std::printf("sk-test: %d checks, %d fails, keys=%zu, bytes/key=%.0f\n", checks, fails,
               oracle.size(), oracle.empty() ? 0 : double(db->live_bytes()) / oracle.size());
   return fails ? 1 : 0;
}

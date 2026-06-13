// start_ver::kv end-to-end vs oracle, mirroring the kv-compat battery's phases
// (batched commits — which retain the live root and force COW/demotion on the
// next batch — then updates, removes, fuzz). Stops at first divergence.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <start_ver/kv.hpp>

namespace fs = std::filesystem;

namespace
{
   uint64_t    hashed(uint64_t i) { return (i * 0x9E3779B97F4A7C15ull) ^ (i >> 29); }
   std::string u64key(uint64_t x)
   {
      return std::string(reinterpret_cast<const char*>(&x), sizeof(x));
   }
   std::string mkval(uint64_t seed)
   {
      size_t      len = 1 + (seed % 40);
      std::string v(len, 0);
      for (size_t j = 0; j < len; ++j)
         v[j] = char(uint8_t(seed * 31 + j));
      return v;
   }
}  // namespace

namespace
{
   // Every line the engine ever allocated must come back once the sessions and
   // database are gone — the leak detector for the whole COW machinery.
   void require_drained(start_ver::kv::database& db)
   {
      db.flush_releases();  // SPEC §19: the gate waits for the releaser
      start_ver::arena& a = db.store();
      uint64_t free_total = 0;
      for (uint32_t s = 0; s < a.num_segments(); ++s)
         free_total += a.freeset_of(s).free_lines();
      REQUIRE(free_total == uint64_t(a.num_segments()) * start_ver::lines_per_segment);
   }
}  // namespace

TEST_CASE("kv battery: batched commits, updates, removes vs oracle")
{
   auto dir = fs::temp_directory_path() / "part_kv_battery";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   std::map<std::string, std::string> oracle;
   const int                          N = 20000, B = 2048;

   std::vector<std::pair<std::string, std::string>> kv;
   for (int i = 0; i < N; ++i)
   {
      std::string k = (i & 1) ? u64key(hashed(uint64_t(i)))
                              : ("str/" + std::to_string(hashed(uint64_t(i)) % 100000));
      kv.emplace_back(k, mkval(uint64_t(i)));
      oracle[k] = kv.back().second;
   }

   auto upsert_all = [&](auto& pairs)
   {
      for (size_t base = 0; base < pairs.size(); base += B)
      {
         auto   tx  = ses->start_transaction(0);
         size_t end = std::min(pairs.size(), base + size_t(B));
         for (size_t i = base; i < end; ++i)
            tx.upsert(pairs[i].first, pairs[i].second);
         tx.commit();
      }
   };
   auto verify_all = [&](const char* phase)
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
      {
         auto got = tx.template get<std::string>(k);
         INFO(phase << " key len=" << k.size());
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      tx.abort();
   };

   upsert_all(kv);
   verify_all("phase1");

   // Phase 2: update every key (forces COW through shared roots).
   for (auto& [k, v] : oracle)
      v = mkval(hashed(uint64_t(v.size() + 1)) + 13);
   std::vector<std::pair<std::string, std::string>> upd(oracle.begin(), oracle.end());
   upsert_all(upd);
   verify_all("phase2");

   // Phase 3: remove half, batched.
   std::vector<std::string> keys;
   for (auto& [k, v] : oracle)
      keys.push_back(k);
   const size_t half = keys.size() / 2;
   for (size_t base = 0; base < half; base += B)
   {
      auto   tx  = ses->start_transaction(0);
      size_t end = std::min(half, base + size_t(B));
      for (size_t i = base; i < end; ++i)
         tx.remove(keys[i]);
      tx.commit();
   }
   for (size_t i = 0; i < half; ++i)
      oracle.erase(keys[i]);
   verify_all("phase3-present");
   {
      auto tx = ses->start_transaction(0);
      for (size_t i = 0; i < half; ++i)
         REQUIRE_FALSE(tx.template get<std::string>(keys[i]).has_value());
      tx.abort();
   }

   // Phase 4: full forward scan matches the oracle.
   {
      auto c   = ses->get_root(0).snapshot_cursor();
      auto oit = oracle.begin();
      c.seek_begin();
      while (!c.is_end())
      {
         REQUIRE(oit != oracle.end());
         REQUIRE(std::string(c.key()) == oit->first);
         {
            auto got = *c.value<std::string>();
            INFO("scan key='" << oit->first << "' klen=" << oit->first.size()
                              << " got.len=" << got.size()
                              << " want.len=" << oit->second.size());
            REQUIRE(got == oit->second);
         }
         c.next();
         ++oit;
      }
      REQUIRE(oit == oracle.end());
   }

   // Phase 5: remove EVERYTHING, then the arena must drain to empty once the
   // session and database release their roots.
   for (size_t base = half; base < keys.size(); base += B)
   {
      auto   tx  = ses->start_transaction(0);
      size_t end = std::min(keys.size(), base + size_t(B));
      for (size_t i = base; i < end; ++i)
         tx.remove(keys[i]);
      tx.commit();
   }
   // After remove-all the live root is null: every generation has drained
   // and the arena must be empty even while the database stays open.
   require_drained(*db);
}

TEST_CASE("dense ladder: every tier boundary, narrow ranges, shared-seg widen")
{
   auto dir = fs::temp_directory_path() / "part_kv_ladder";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   // One subtree per fanout: keys "L<id>/" + byte b + "tail", fanout swept across
   // every ladder boundary (setlist 16 -> cx1 -> cx2 -> cx3 -> full_x 256) plus
   // narrow clusters (all bytes in one 32-range: lowercase ASCII shape).
   const int fan[] = {2, 16, 17, 31, 32, 62, 63, 124, 125, 248, 249, 256};
   std::map<std::string, std::string> oracle;
   {
      auto tx = ses->start_transaction(0);
      int  id = 0;
      for (int n : fan)
      {
         std::string base = "L" + std::to_string(id++) + "/";
         for (int b = 0; b < n; ++b)
         {
            std::string k = base + char(uint8_t(b)) + "tail";
            tx.upsert(k, mkval(uint64_t(b * 7 + n)));
            oracle[k] = mkval(uint64_t(b * 7 + n));
         }
      }
      // narrow: 32 keys all inside one 32-byte range (forces cx3 seg overflow -> fx)
      for (int b = 0; b < 32; ++b)
      {
         std::string k = "narrow/" + std::string(1, char(uint8_t(96 + b))) + "x";
         tx.upsert(k, mkval(uint64_t(b)));
         oracle[k] = mkval(uint64_t(b));
      }
      tx.commit();
   }
   auto verify = [&] {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
      {
         auto got = tx.template get<std::string>(k);
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      tx.abort();
   };
   verify();
   {  // scan order vs oracle
      auto c   = ses->get_root(0).snapshot_cursor();
      auto oit = oracle.begin();
      c.seek_begin();
      while (!c.is_end())
      {
         REQUIRE(oit != oracle.end());
         REQUIRE(std::string(c.key()) == oit->first);
         c.next();
         ++oit;
      }
      REQUIRE(oit == oracle.end());
   }
   {  // SHARED-SEG WIDEN: pin the committed root, then grow a 31-branch subtree
      // (cx) past every boundary while its seg lines are shared with the pin.
      auto pin = ses->get_root(0).snapshot_cursor();
      {
         auto tx = ses->start_transaction(0);
         for (int b = 0; b < 256; ++b)
         {
            std::string k = "L3/" + std::string(1, char(uint8_t(b))) + "tail2";  // L3 = fanout 31
            tx.upsert(k, mkval(uint64_t(b + 1000)));
            oracle[k] = mkval(uint64_t(b + 1000));
         }
         tx.commit();
      }
      verify();  // new generation complete
      // the pinned snapshot still sees the OLD subtree intact
      for (int b = 0; b < 31; ++b)
      {
         std::string k = "L3/" + std::string(1, char(uint8_t(b))) + "tail";
         REQUIRE(pin.get<std::string>(k).has_value());
      }
   }
   {  // removes: empty out ranges (seg lines must release), then everything
      auto tx = ses->start_transaction(0);
      std::vector<std::string> kill;
      for (auto& [k, v] : oracle)
         if (k.size() > 3 && uint8_t(k[3]) >= 128)  // upper half of each fanout
            kill.push_back(k);
      for (auto& k : kill)
      {
         tx.remove(k);
         oracle.erase(k);
      }
      tx.commit();
   }
   verify();
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
         tx.remove(k);
      tx.commit();
   }
   require_drained(*db);
}

TEST_CASE("pages: reprefix overflow splits the original (diverging key, fat cells)")
{
   // A page whose LCP is deep and whose cells are near the inline cap: a key
   // diverging early forces a reprefix whose longer suffixes can NOT fit the
   // rebuilt page — the engine must split the original and retry, never drop.
   auto dir = fs::temp_directory_path() / "part_kv_reprefix";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   std::map<std::string, std::string> oracle;
   const std::string deep(30, 'P');
   {
      auto tx = ses->start_transaction(0);
      for (int i = 0; i < 50; ++i)
      {
         std::string k = deep + "x" + std::to_string(1000 + i) + std::string(38, char('a' + i % 26));
         std::string v = mkval(uint64_t(i));
         tx.upsert(k, v);
         oracle[k] = v;
      }
      // the diverger: shares only 15 bytes with the page prefix
      std::string dk = deep.substr(0, 15) + "Q";
      tx.upsert(dk, "DIVERGED");
      oracle[dk] = "DIVERGED";
      tx.commit();
   }
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
      {
         auto got = tx.template get<std::string>(k);
         INFO("key.len=" << k.size());
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      tx.abort();
   }
   {  // scan agrees
      auto c   = ses->get_root(0).snapshot_cursor();
      auto oit = oracle.begin();
      c.seek_begin();
      while (!c.is_end())
      {
         REQUIRE(oit != oracle.end());
         REQUIRE(std::string(c.key()) == oit->first);
         c.next();
         ++oit;
      }
      REQUIRE(oit == oracle.end());
   }
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
         tx.remove(k);
      tx.commit();
   }
   require_drained(*db);
}

TEST_CASE("kv terms: a key that is a strict prefix of >64 same-byte siblings")
{
   // Burst shape that silently dropped the bucket term under NDEBUG: the
   // bucket holds a term ("base" == its prefix) and 64+ entries that all
   // share the next byte, so the rebuilt subtree tops with a prefix node.
   auto dir = fs::temp_directory_path() / "part_kv_terms";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   std::map<std::string, std::string> oracle;
   {
      auto tx = ses->start_transaction(0);
      for (int v = 0; v < 4; ++v)
      {
         std::string base = "trm/" + std::to_string(v) + "/key";
         tx.upsert(base, "TERMVAL" + std::to_string(v));
         oracle[base] = "TERMVAL" + std::to_string(v);
         for (int i = 0; i < 200; ++i)  // shared 'a' rides every suffix
         {
            std::string k = base + "a" + std::to_string(i);
            tx.upsert(k, mkval(uint64_t(v * 1000 + i)));
            oracle[k] = mkval(uint64_t(v * 1000 + i));
         }
      }
      tx.commit();
   }
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
      {
         auto got = tx.template get<std::string>(k);
         INFO("key='" << k << "'");
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      tx.abort();
   }
   {  // scan sees them all, in order
      auto c   = ses->get_root(0).snapshot_cursor();
      auto oit = oracle.begin();
      c.seek_begin();
      while (!c.is_end())
      {
         REQUIRE(oit != oracle.end());
         REQUIRE(std::string(c.key()) == oit->first);
         REQUIRE(*c.value<std::string>() == oit->second);
         c.next();
         ++oit;
      }
      REQUIRE(oit == oracle.end());
   }
   {  // lower_bound from every oracle key and from synthesized misses: the
      // cursor must land exactly where the oracle's lower_bound lands
      auto c = ses->get_root(0).snapshot_cursor();
      for (auto& [k, v] : oracle)
      {
         for (std::string probe : {k, k + "\x01", k.substr(0, k.size() - 1)})
         {
            auto oit = oracle.lower_bound(probe);
            c.lower_bound(probe);
            INFO("probe='" << probe << "'");
            if (oit == oracle.end())
               REQUIRE(c.is_end());
            else
            {
               REQUIRE_FALSE(c.is_end());
               REQUIRE(std::string(c.key()) == oit->first);
               REQUIRE(*c.value<std::string>() == oit->second);
            }
         }
      }
   }
   {  // remove every key; arena must drain (no leaked term leaves/wrappers)
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
         tx.remove(k);
      tx.commit();
   }
   require_drained(*db);
}

TEST_CASE("synchronous mode: background_release=false drains identically")
{
   // the queue reports blocked-forever; every cascade runs on its caller —
   // same battery shape, same drain gate, no second thread
   auto dir = fs::temp_directory_path() / "part_kv_sync";
   fs::remove_all(dir);
   start_ver::kv::runtime_config cfg;
   cfg.background_release = false;
   auto db  = start_ver::kv::database::create(dir, cfg);
   auto ses = db->start_write_session();
   std::map<std::string, std::string> oracle;
   for (int b = 0; b < 8; ++b)
   {
      auto tx = ses->start_transaction(0);
      for (int i = 0; i < 500; ++i)
      {
         std::string k = "s/" + std::to_string(b * 500 + i);
         tx.upsert(k, mkval(uint64_t(i)));
         oracle[k] = mkval(uint64_t(i));
      }
      tx.commit();
   }
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
         REQUIRE(*tx.get<std::string>(k) == v);
      tx.abort();
   }
   {
      auto tx = ses->start_transaction(0);
      for (auto& [k, v] : oracle)
         tx.remove(k);
      tx.commit();
   }
   require_drained(*db);
}

TEST_CASE("kv churn: repeated overwrites under generation turnover drain fully")
{
   auto dir = fs::temp_directory_path() / "part_kv_churn";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   // Bank-shaped churn: a fixed key population, every txn overwrites a few
   // keys and commits (slot retain forces COW + demotion next txn).
   std::mt19937_64          rng(7);
   std::vector<std::string> keys;
   for (int i = 0; i < 4000; ++i)
      keys.push_back("acct/" + std::to_string(i));
   {
      auto tx = ses->start_transaction(0);
      for (auto& k : keys)
         tx.upsert(k, "00000000");
      tx.commit();
   }
   for (int round = 0; round < 3000; ++round)
   {
      auto tx = ses->start_transaction(0);
      for (int j = 0; j < 3; ++j)
      {
         auto& k = keys[rng() % keys.size()];
         char  v[9];
         snprintf(v, sizeof v, "%08x", uint32_t(rng()));
         tx.upsert(k, std::string_view(v, 8));
      }
      tx.commit();
   }
   // The live tree should be the ONLY generation: live bytes must be sane
   // (population is fixed; churn must not accumulate).
   const auto& st   = db->store().stats();
   const auto  live = st.bytes_allocated.load() - st.bytes_freed.load();
   INFO("live bytes after churn: " << live);
   REQUIRE(live < 16u * 1024 * 1024);  // 4k keys * ~0.5KB tree ≈ 2MB; 16MB = generous

   {  // remove everything: the arena must drain with the database still open
      auto tx = ses->start_transaction(0);
      for (auto& k : keys)
         tx.remove(k);
      tx.commit();
   }
   require_drained(*db);
}

TEST_CASE("cursor bidirectional: seek_last/prev/rend conform to std::map order")
{
   auto dir = fs::temp_directory_path() / "part_kv_bidir";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   // Mixed-shape keys: random hex (fans wide), shared-prefix runs (prefix
   // nodes + hdr routers), and short keys (router terms) — every node kind
   // the reverse walk must mirror.
   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(0xbd1f);
   auto                               put = [&](const std::string& k) {
      std::string v = "v:" + k;
      auto        tx = ses->start_transaction(0);
      tx.upsert(k, v);
      tx.commit();
      oracle[k] = v;
   };
   for (int i = 0; i < 4000; ++i)
   {
      char b[17];
      uint64_t h = rng();
      for (int j = 0; j < 16; ++j, h >>= 4)
         b[j] = "0123456789abcdef"[h & 15];
      put(std::string(b, 16));
   }
   for (int i = 0; i < 1000; ++i)
      put("shared/prefix/run/" + std::to_string(rng() % 100000));
   for (int i = 0; i < 300; ++i)
      put(std::string(1 + i % 3, char('a' + i % 26)));

   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);

   SECTION("full backward walk == reverse oracle order")
   {
      REQUIRE(cur.seek_last());
      auto it = oracle.rbegin();
      uint64_t n = 0;
      do
      {
         REQUIRE(it != oracle.rend());
         REQUIRE(cur.key() == it->first);
         REQUIRE(cur.value<std::string>().value() == it->second);
         ++it;
         ++n;
      } while (cur.prev());
      REQUIRE(n == oracle.size());
      REQUIRE(it == oracle.rend());
      REQUIRE(cur.is_rend());
      REQUIRE_FALSE(cur.is_end());
      // psitri convention: next from rend re-seats at the first key
      REQUIRE(cur.next());
      REQUIRE(cur.key() == oracle.begin()->first);
   }
   SECTION("forward walk then prev from end seats at last")
   {
      REQUIRE(cur.seek_begin());
      while (cur.next()) {}
      REQUIRE(cur.is_end());
      REQUIRE(cur.prev());
      REQUIRE(cur.key() == oracle.rbegin()->first);
   }
   SECTION("mixed next/prev random walk tracks the oracle")
   {
      REQUIRE(cur.seek_begin());
      auto it = oracle.begin();
      for (int step = 0; step < 20000; ++step)
      {
         if (rng() & 1)
         {
            auto nx = std::next(it);
            if (nx == oracle.end())
               continue;  // stay in range: end/rend transitions tested above
            it = nx;
            REQUIRE(cur.next());
         }
         else
         {
            if (it == oracle.begin())
               continue;
            --it;
            REQUIRE(cur.prev());
         }
         REQUIRE(cur.key() == it->first);
      }
   }
   SECTION("lower_bound then backward walk")
   {
      const std::string probe = "shared/prefix/run/5";
      REQUIRE(cur.lower_bound(probe));
      auto it = oracle.lower_bound(probe);
      REQUIRE(cur.key() == it->first);
      while (it != oracle.begin())
      {
         --it;
         REQUIRE(cur.prev());
         REQUIRE(cur.key() == it->first);
      }
      REQUIRE_FALSE(cur.prev());
      REQUIRE(cur.is_rend());
   }
}

TEST_CASE("tree_root: refcounted version roots — held trees live, last drop reaps exactly")
{
   auto dir = fs::temp_directory_path() / "part_kv_treeroot";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   // ── fill the delta, then freeze it behind a handle ──
   std::map<std::string, std::string> frozen;
   {
      auto tx = ses->start_transaction(0);
      for (int i = 0; i < 20000; ++i)
      {
         auto k    = "frozen/" + std::to_string(i * 7919 % 100000);
         frozen[k] = "v" + k;
         tx.upsert(k, frozen[k]);
      }
      tx.commit();
   }
   // ── rotate hot -> warm in adb vocabulary: h1.set_root(h0.set_root({})) ──
   {
      auto tx = ses->start_transaction(0);
      auto h0 = tx.primary();
      auto h1 = tx.open_root(1);
      h1.set_root(h0.set_root(start_ver::kv::tree_root{}));  // hot -> warm, root 0 newborn
      tx.commit();
   }
   start_ver::kv::tree_root h1 = ses->get_root(1);  // owned, FROZEN warm snapshot
   REQUIRE(bool(h1));
   REQUIRE(h1.version() > 0);

   // hot tree (root 0) is NEWBORN: independent of the frozen warm tree
   {
      auto tx = ses->start_transaction(0);
      tx.upsert("hot/alpha", "a");
      tx.commit();
   }
   {
      auto rs  = db->start_read_session();
      auto hot = rs->create_cursor(0);
      REQUIRE(hot.get<std::string>("hot/alpha").value() == "a");
      REQUIRE_FALSE(hot.get<std::string>(frozen.begin()->first).has_value());
   }

   // ── the frozen warm tree reads correctly through ANY holder, cross-thread ──
   auto check_frozen = [&](const start_ver::kv::tree_root& h) {
      auto cur = h.snapshot_cursor();  // a tree IS a snapshot
      int  n   = 0;
      for (auto& [k, v] : frozen)
      {
         if (++n > 500)
            break;  // sample
         REQUIRE(cur.get<std::string>(k).value() == v);
      }
   };
   check_frozen(h1);
   start_ver::kv::tree_root h2 = h1;  // second holder (snapshots copy/share)
   h1.reset();                        // first drop: tree must SURVIVE (h2 lives)
   check_frozen(h2);
   {
      std::thread t([&] { check_frozen(h2); });  // handles cross threads
      t.join();
   }

   // get_root(1) snapshots are VIEWS of the live lineage (root 1 owns the
   // tree) — dropping every handle reaps NOTHING; the data stays readable ──
   h2.reset();
   db->flush_reaper();
   {
      auto rs2 = db->start_read_session();
      auto c1  = rs2->create_cursor(1);
      REQUIRE(c1.get<std::string>(frozen.begin()->first).has_value());
   }
   // reclamation = emptying root 1 (set_root({}); displaced tree arms -> reaper)
   {
      auto tx = ses->start_transaction(1);
      tx.primary().set_root(start_ver::kv::tree_root{});
      tx.commit();
   }
   db->flush_reaper();
   db->flush_releases();
   // everything the frozen tree held is back: only the 1-key hot tree (and
   // its bounded COW residue) remains allocated
   start_ver::arena& a          = db->store();
   uint64_t      free_total = 0, cap = 0;
   for (uint32_t s = 0; s < a.num_segments(); ++s)
   {
      free_total += a.freeset_of(s).free_lines();
      cap += start_ver::lines_per_segment;
   }
   const uint64_t used = cap - free_total;
   REQUIRE(used < 2000);  // 20k-key tree was ~10k+ lines; hot tree is ~tens
}

TEST_CASE("multi-root: basic commit, RYW per root, abort rolls back all roots")
{
   auto dir = fs::temp_directory_path() / "part_kv_multiroot";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   // basic commit across two roots in one txn
   {
      auto tx = ses->start_transaction(0);
      auto h0 = tx.primary();
      auto h1 = tx.open_root(1);
      h0.upsert("users:42", "alice");
      h1.upsert("orders:100", "order_data");
      REQUIRE(h0.get<std::string>("users:42").value() == "alice");   // RYW root 0
      REQUIRE(h1.get<std::string>("orders:100").value() == "order_data");
      REQUIRE_FALSE(h0.get<std::string>("orders:100").has_value());  // isolation
      tx.commit();
   }
   auto rs = db->start_read_session();
   {
      auto c0 = rs->create_cursor(0);
      auto c1 = rs->create_cursor(1);
      REQUIRE(c0.get<std::string>("users:42").value() == "alice");
      REQUIRE_FALSE(c0.get<std::string>("orders:100").has_value());
      REQUIRE(c1.get<std::string>("orders:100").value() == "order_data");
   }
   // abort rolls back BOTH roots
   {
      auto tx = ses->start_transaction(0);
      auto h0 = tx.primary();
      auto h1 = tx.open_root(1);
      h0.upsert("users:43", "bob");
      h1.upsert("orders:101", "x");
      tx.abort();
   }
   {
      auto c0 = rs->create_cursor(0);
      auto c1 = rs->create_cursor(1);
      REQUIRE_FALSE(c0.get<std::string>("users:43").has_value());
      REQUIRE_FALSE(c1.get<std::string>("orders:101").has_value());
      REQUIRE(c0.get<std::string>("users:42").value() == "alice");  // prior intact
   }
}

TEST_CASE("multi-root: set_root exchange — the rotation, abort-safe, leak-exact")
{
   auto dir = fs::temp_directory_path() / "part_kv_setroot";
   fs::remove_all(dir);
   auto db  = start_ver::kv::database::create(dir);
   auto ses = db->start_write_session();

   std::map<std::string, std::string> oracle;
   {
      auto tx = ses->start_transaction(0);
      for (int i = 0; i < 8000; ++i)
      {
         auto k    = "k" + std::to_string(i * 2654435761u % 1000000);
         oracle[k] = "v" + k;
         tx.upsert(k, oracle[k]);
      }
      tx.commit();
   }

   SECTION("aborted rotation leaves both roots untouched; nothing reaped")
   {
      {
         auto tx = ses->start_transaction(0);
         auto h0 = tx.primary();
         auto h1 = tx.open_root(1);
         h1.set_root(h0.set_root(start_ver::kv::tree_root{}));
         REQUIRE(h0.key_count() == 0);             // exchanged within the txn
         REQUIRE(h1.key_count() == oracle.size());
         tx.abort();
      }
      db->flush_reaper();
      auto rs = db->start_read_session();
      auto c0 = rs->create_cursor(0);
      auto c1 = rs->create_cursor(1);
      REQUIRE(c0.get<std::string>(oracle.begin()->first).has_value());  // restored
      REQUIRE_FALSE(c1.seek_begin());                                   // still empty
   }
   SECTION("committed rotation: hot reborn empty, warm holds the tree; emptying warm reaps")
   {
      {
         auto tx = ses->start_transaction(0);
         auto h0 = tx.primary();
         auto h1 = tx.open_root(1);
         h1.set_root(h0.set_root(start_ver::kv::tree_root{}));
         tx.commit();
      }
      auto rs = db->start_read_session();
      {
         auto c0 = rs->create_cursor(0);
         auto c1 = rs->create_cursor(1);
         REQUIRE_FALSE(c0.seek_begin());  // hot newborn
         uint64_t n = 0;
         for (bool ok = c1.seek_begin(); ok; ok = c1.next())
         {
            REQUIRE(oracle.at(std::string(c1.key())) == c1.value<std::string>().value());
            ++n;
         }
         REQUIRE(n == oracle.size());  // warm = the frozen tree, byte-exact
      }
      // hot keeps working independently
      {
         auto tx = ses->start_transaction(0);
         tx.upsert("hot/new", "x");
         tx.commit();
      }
      // empty the warm root: displaced tree arms at commit -> reaper frees it
      {
         auto tx = ses->start_transaction(1);
         (void)tx.primary().set_root(start_ver::kv::tree_root{});
         tx.commit();
      }
      db->flush_reaper();
      db->flush_releases();
      start_ver::arena& a          = db->store();
      uint64_t      free_total = 0, cap = 0;
      for (uint32_t s = 0; s < a.num_segments(); ++s)
      {
         free_total += a.freeset_of(s).free_lines();
         cap += start_ver::lines_per_segment;
      }
      REQUIRE(cap - free_total < 2000);  // 8k-key tree fully reclaimed
   }
}

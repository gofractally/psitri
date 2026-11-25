// start_ref::kv end-to-end vs oracle, mirroring the kv-compat battery's phases
// (batched commits — which retain the live root and force COW/demotion on the
// next batch — then updates, removes, fuzz). Stops at first divergence.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <start_ref/kv.hpp>

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
   void require_drained(start_ref::kv::database& db)
   {
      db.flush_releases();  // SPEC §19: the gate waits for the releaser
      start_ref::arena& a = db.store();
      uint64_t free_total = 0;
      for (uint32_t s = 0; s < a.num_segments(); ++s)
         free_total += a.freeset_of(s).free_lines();
      REQUIRE(free_total == uint64_t(a.num_segments()) * start_ref::lines_per_segment);
   }
}  // namespace

TEST_CASE("kv battery: batched commits, updates, removes vs oracle")
{
   auto dir = fs::temp_directory_path() / "part_kv_battery";
   fs::remove_all(dir);
   auto db  = start_ref::kv::database::create(dir);
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
   auto db  = start_ref::kv::database::create(dir);
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
   auto db  = start_ref::kv::database::create(dir);
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
   auto db  = start_ref::kv::database::create(dir);
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
   start_ref::kv::runtime_config cfg;
   cfg.background_release = false;
   auto db  = start_ref::kv::database::create(dir, cfg);
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
   auto db  = start_ref::kv::database::create(dir);
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

// ── large values (>8 KB): the second 8 KB-grain arena (value_large) ──────────
// Ethereum contract bytecode (up to ~0.5 MB) exceeds the line arena's single-CAS
// 8 KB object cap, so those leaves must route to the LARGE arena. This exercises
// the full value_large refcount lifecycle: create, read-back, COW overwrite
// (grows/shrinks across the line/large boundary), remove, and the snapshot-
// sharing path — and asserts the large arena DRAINS (no per-grain leak via the
// refcount cascade).
namespace
{
   std::string big_val(size_t len, uint64_t seed)
   {
      std::string v(len, 0);
      for (size_t j = 0; j < len; ++j)
         v[j] = char(uint8_t(seed * 2654435761u + j * 40503u + (j >> 8)));
      return v;
   }
}  // namespace

TEST_CASE("large values: >8KB and 0.5MB leaves route to the large arena and drain")
{
   auto dir = fs::temp_directory_path() / "part_kv_large";
   fs::remove_all(dir);
   auto db  = start_ref::kv::database::create(dir);
   auto ses = db->start_write_session();

   const std::string ka = "code/aaa";  // a >8KB value
   const std::string kb = "code/bbb";  // a 0.5MB-class value
   const std::string ks = "code/small";  // stays in the line arena (value_ptr)

   const std::string va  = big_val(20000, 1);     // 20 KB  -> value_large
   const std::string vb  = big_val(100000, 2);    // 100 KB -> value_large
   const std::string vsm = mkval(7);              // small  -> value_ptr (line arena)

   // 1) create + read back, verify bytes exactly.
   {
      auto tx = ses->start_transaction(0);
      tx.upsert(ka, va);
      tx.upsert(kb, vb);
      tx.upsert(ks, vsm);
      tx.commit();
   }
   REQUIRE(db->large_used_grains() > 0);  // the large arena actually fired
   {
      auto tx = ses->start_transaction(0);
      auto ga = tx.get<std::string>(ka);
      auto gb = tx.get<std::string>(kb);
      auto gs = tx.get<std::string>(ks);
      REQUIRE(ga.has_value());
      REQUIRE(gb.has_value());
      REQUIRE(gs.has_value());
      REQUIRE(*ga == va);
      REQUIRE(*gb == vb);
      REQUIRE(*gs == vsm);
      tx.abort();
   }

   // 2) a SNAPSHOT cursor pins the committed generation; overwriting under it
   //    must COW the large leaf (not corrupt the snapshot's bytes).
   {
      auto snap = ses->get_root(0).snapshot_cursor();
      {
         auto tx = ses->start_transaction(0);
         tx.upsert(ka, big_val(20000, 99));  // same size, new bytes -> COW clone
         tx.upsert(kb, big_val(250000, 99)); // grow -> stays large, COW clone
         tx.commit();
      }
      // the snapshot still sees the ORIGINAL bytes (point read against the pin)
      auto sa = snap.get<std::string>(ka);
      auto sb = snap.get<std::string>(kb);
      REQUIRE(sa.has_value());
      REQUIRE(sb.has_value());
      REQUIRE(*sa == va);
      REQUIRE(*sb == vb);
   }
   db->flush_releases();  // the snapshot is gone: its generation drains

   // 3) COW-overwrite churn across the line/large boundary, many rounds.
   for (int round = 0; round < 50; ++round)
   {
      auto tx = ses->start_transaction(0);
      // grow small->large and shrink large->small to exercise both leaf_alloc_
      // arena decisions and the cross-arena rebuild on overwrite
      tx.upsert(ka, (round & 1) ? big_val(40000, uint64_t(round))
                                : mkval(uint64_t(round)));         // large <-> line
      tx.upsert(kb, big_val(8193 + size_t(round) * 1000, uint64_t(round)));  // always large
      tx.commit();
   }
   {  // final read-back is consistent
      auto tx  = ses->start_transaction(0);
      auto gb  = tx.get<std::string>(kb);
      REQUIRE(gb.has_value());
      REQUIRE(*gb == big_val(8193 + 49 * 1000, 49));
      tx.abort();
   }

   // 4) remove everything: both arenas must drain to empty.
   {
      auto tx = ses->start_transaction(0);
      tx.remove(ka);
      tx.remove(kb);
      tx.remove(ks);
      tx.commit();
   }
   // End the write session so its line/large allocation windows flush their
   // reserved-but-unused grains back to the freesets (mirrors start_ver's test).
   ses.reset();
   require_drained(*db);             // line arena empty
   db->flush_releases();
   REQUIRE(db->large_used_grains() == 0);  // NO value_large leak
}

// A many-key large-value workload (Silkworm-shaped: a code table of contract
// bytecode), repeatedly overwritten, then removed — the large arena must not
// grow without bound and must drain fully.
TEST_CASE("large values: many-key code-table churn drains the large arena")
{
   auto dir = fs::temp_directory_path() / "part_kv_large_churn";
   fs::remove_all(dir);
   auto db  = start_ref::kv::database::create(dir);
   auto ses = db->start_write_session();

   const int                N = 200;
   std::vector<std::string> keys;
   for (int i = 0; i < N; ++i)
      keys.push_back("code/" + std::to_string(hashed(uint64_t(i)) % 1000000));
   std::sort(keys.begin(), keys.end());
   keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

   auto write_round = [&](uint64_t seed)
   {
      auto tx = ses->start_transaction(0);
      for (size_t i = 0; i < keys.size(); ++i)
         tx.upsert(keys[i], big_val(8200 + (i % 17) * 4096, seed + i));  // all value_large
      tx.commit();
   };

   write_round(1);
   const uint64_t after_first = db->large_used_grains();
   REQUIRE(after_first > 0);

   for (uint64_t r = 2; r < 12; ++r)  // 10 more full rewrites (COW turnover)
      write_round(r);
   db->flush_releases();

   // After turnover the live generation is the only one: grain use must be
   // bounded near the first round's footprint, not 10x it.
   const uint64_t after_churn = db->large_used_grains();
   INFO("large grains: first=" << after_first << " after_churn=" << after_churn);
   REQUIRE(after_churn <= after_first * 2);

   // verify the live values
   {
      auto tx = ses->start_transaction(0);
      for (size_t i = 0; i < keys.size(); ++i)
      {
         auto g = tx.get<std::string>(keys[i]);
         REQUIRE(g.has_value());
         REQUIRE(*g == big_val(8200 + (i % 17) * 4096, 11 + i));
      }
      tx.abort();
   }

   {
      auto tx = ses->start_transaction(0);
      for (auto& k : keys)
         tx.remove(k);
      tx.commit();
   }
   ses.reset();  // flush the allocation windows before the drain check
   require_drained(*db);
   db->flush_releases();
   REQUIRE(db->large_used_grains() == 0);
}

// gofractally/psitri#34. Dropping the last reference to a version destroys its
// tree, and used to descend one call level per level of that tree on the
// caller's own stack. A tree is as deep as its longest key, so a caller with
// a small stack (a service thread, a fibre) could overflow it by letting go of
// a snapshot.
TEST_CASE("kv: the last reference to a deep version is dropped on a small stack")
{
   namespace kv = start_ref::kv;
   const std::size_t small_stack = 96 * 1024;
   const int         depth       = 1024;  // 'a', 'aa', ...: one level per key
   auto build = [&](kv::database& db, kv::write_session& ws) {
      auto        tx = ws.start_transaction(0);
      std::string k;
      for (int i = 0; i < depth; ++i)
      {
         k.push_back('a');
         tx.upsert(k, "v");
      }
      tx.commit();
      (void)db;
   };
   auto clear = [&](kv::write_session& ws) {
      auto        tx = ws.start_transaction(0);
      std::string k;
      for (int i = 0; i < depth; ++i)
      {
         k.push_back('a');
         REQUIRE(tx.remove(k) == 0);
      }
      tx.commit();
   };
   for (bool background : {true, false})
   {
      INFO("background_release " << background);
      auto dir = fs::temp_directory_path() / "part_kv_deep_release";
      fs::remove_all(dir);
      kv::runtime_config cfg;
      cfg.background_release = background;
      auto db = kv::database::create(dir, cfg);
      auto ws = db->start_write_session();

      // an owned snapshot, dropped by a small-stack thread
      build(*db, *ws);
      {
         kv::tree snap = ws->get_root(0);
         clear(*ws);  // the old version now lives through `snap` alone
         db->flush_releases();
         kv::sized_thread th(small_stack, [&] { kv::tree t = std::move(snap); });
         th.join();
      }
      // a snapshot cursor, dropped by a small-stack thread
      build(*db, *ws);
      {
         auto cur = std::make_unique<kv::cursor>(ws->get_root(0).snapshot_cursor());
         clear(*ws);
         db->flush_releases();
         kv::sized_thread th(small_stack, [&] { cur.reset(); });
         th.join();
      }
      // the writer's own commit retiring a deep version, on a small stack
      build(*db, *ws);
      {
         kv::sized_thread th(small_stack, [&] { clear(*ws); });
         th.join();
      }
      ws.reset();
      require_drained(*db);
   }
}

// gofractally/psitri#35 item 4. A typed value may not use a type number the
// engine keeps for itself: upsert needs no registration, so the check has to
// be in what makes a type storable.
namespace
{
   template <uint16_t Id>
   struct numbered_value
   {
      static constexpr uint16_t type_id = Id;
      struct t_pod
      {
         uint64_t n;
         static void retain(void*, const t_pod&) {}
         static void release(void*, const t_pod&) {}
      };
      uint64_t n = 0;
      t_pod    take_pod() && { return {n}; }
      static numbered_value make_from_pod(const t_pod& p, bool) { return {p.n}; }
   };
   static_assert(start_ref::kv::storable<numbered_value<1>>);
   static_assert(start_ref::kv::storable<numbered_value<65534>>);
   static_assert(!start_ref::kv::storable<numbered_value<0>>,
                 "type number 0 is plain bytes");
   static_assert(!start_ref::kv::storable<numbered_value<65535>>,
                 "type number 65535 is the directory of a chunked value");
}  // namespace

// A cursor that is moved keeps its place (it used to come back at the end).
TEST_CASE("kv: a moved cursor stays on the key it stood on")
{
   namespace kv = start_ref::kv;
   auto dir = fs::temp_directory_path() / "part_kv_cursor_move";
   fs::remove_all(dir);
   auto db = kv::database::create(dir);
   auto ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 100; ++i)
         tx.upsert("k/" + std::to_string(1000 + i), "v" + std::to_string(i));
      tx.commit();
   }
   auto rs = db->start_read_session();
   {
      auto cur = rs->create_cursor(0);
      REQUIRE(cur.lower_bound("k/1040"));
      kv::cursor moved(std::move(cur));
      REQUIRE_FALSE(moved.is_end());
      REQUIRE(moved.key() == "k/1040");
      REQUIRE(*moved.value<std::string>() == "v40");
      REQUIRE(moved.next());
      REQUIRE(moved.key() == "k/1041");
   }
   {  // a cursor at the end, and one moved while holding the session's slot, too
      auto cur = rs->create_cursor(0);
      REQUIRE_FALSE(cur.lower_bound("z"));
      kv::cursor moved(std::move(cur));
      REQUIRE(moved.is_end());
      REQUIRE(moved.seek_begin());
      REQUIRE(moved.key() == "k/1000");
   }
   {  // a snapshot cursor returned by value from a function, as the API does
      auto make = [&] {
         auto c = ws->get_root(0).snapshot_cursor();
         c.lower_bound("k/1099");
         return c;
      };
      kv::cursor last = make();
      REQUIRE(last.key() == "k/1099");
      REQUIRE_FALSE(last.next());
   }
}

// oplog_replay — replay a captured Silkworm engine op-stream into delta_art and
// track on-disk footprint vs live keys. Reproduces the real workload (variable
// namespaced keys, interleaved upsert/remove, real commit batching) standalone —
// no Silkworm. The shim captures these via MDBX_OPLOG=<path>.
//
// Format: 'U' kl[4] key[kl] vl[4] val[vl] | 'R' kl[4] key[kl] | 'C' (commit)
//
//   oplog-replay <oplog.bin> <dir> [report_every_commits=200]
#include <delta_art/tiered_kv.hpp>
#include <psitri/dwal/kv_engine.hpp>
#include <sal_art_ref/engine.hpp>
#include <start_ref/kv.hpp>
#include <start_ver/kv.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using TK     = delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::engine>;

static uint64_t fnv_(std::string_view v) noexcept  // matches the shim's oplog_fnv_
{
   uint64_t h = 1469598103934665603ull;
   for (unsigned char c : v) { h ^= c; h *= 1099511628211ull; }
   return h;
}

static double dir_gb(const fs::path& p)
{
   uint64_t        b = 0;
   std::error_code ec;
   for (auto& e : fs::recursive_directory_iterator(p, ec))
      if (e.is_regular_file(ec))
         b += e.file_size(ec);
   return double(b) / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char** argv)
{
   if (argc < 3)
   {
      fprintf(stderr, "usage: oplog-replay <oplog.bin> <dir> [report_every_commits=200]\n");
      return 2;
   }
   const std::string oplog  = argv[1];
   const std::string dir    = argv[2];
   const uint64_t    report = argc > 3 ? strtoull(argv[3], nullptr, 10) : 200;
   const std::string syncs  = argc > 4 ? argv[4] : "fsync";
   const sal::sync_type sync =
       syncs == "none"    ? sal::sync_type::none
     : syncs == "async"   ? sal::sync_type::msync_async
     : syncs == "msync"   ? sal::sync_type::msync_sync
     : syncs == "full"    ? sal::sync_type::full
                          : sal::sync_type::fsync;

   fs::remove_all(dir);
   fs::create_directories(dir);
   std::FILE* f = std::fopen(oplog.c_str(), "rb");
   if (!f)
   {
      fprintf(stderr, "cannot open %s\n", oplog.c_str());
      return 1;
   }
   std::setvbuf(f, nullptr, _IOFBF, 1 << 22);

   // ── VERIFY MODE (OPLOG_ENGINE=start_ref) ─────────────────────────────────
   // Replay the real Silkworm WRITE stream into BOTH start_ref and a std::map
   // ground truth, then read every key back from start_ref and compare. A
   // divergent key reproduces the wrong-read (Silkworm's kInsufficientFunds)
   // deterministically — and proves it's the ENGINE (not the shim's DUPSORT/
   // cursor decode), because this path never touches the shim.
   if (const char* eng = std::getenv("OPLOG_ENGINE"); eng && std::string_view(eng) == "start_ref")
   {
      start_ref::kv::runtime_config rc;
      if (std::getenv("OPLOG_SYNC_REL")) rc.background_release = false;  // no releaser thread
      auto srdb = start_ref::kv::database::create(dir, rc);
      auto srws = srdb->start_write_session();
      std::map<std::string, std::string> truth;
      std::vector<char>                   key, val, rkey;
      uint64_t    commits = 0, ups = 0, rems = 0, g_n = 0, l_n = 0, g_div = 0;
      std::string g_first;
      // [mon] live-corruption monitor: pin a commit-MAXC snapshot, then replay the REAL
      // subsequent writes into an open txn and watch the snapshot's lower_bound change.
      auto mhex = [](const std::string& s){ std::string o; char b[3]; for(unsigned char c:s){snprintf(b,3,"%02x",c);o+=b;} return o; };
      const bool  do_mon = std::getenv("OPLOG_MON") != nullptr;
      std::string mon_sk;
      if (const char* lb = std::getenv("OPLOG_LB")) for (const char* p = lb; p[0] && p[1]; p += 2) mon_sk.push_back((char)strtoul(std::string{p[0], p[1]}.c_str(), nullptr, 16));
      std::shared_ptr<start_ref::kv::read_session> mon_rs;
      std::optional<start_ref::kv::cursor>          mon_C;  // ONE held snapshot cursor (not re-created)
      const bool  mon_hold = std::getenv("OPLOG_MON_HOLD") != nullptr;
      std::string mon_before; bool monitoring = false, mon_done = false; uint64_t mon_ops = 0;
      std::string mon_lastk; char mon_lasto = 0;
      auto mon_check = [&]() {
         if (!monitoring || mon_done) return;
         ++mon_ops;
         // coarse until near the known corruption window, then every op to pinpoint
         if (mon_ops < 835000 ? (mon_ops % 4096) : (mon_ops % 1)) return;
         bool f; std::string now;
         if (mon_hold) { f = mon_C->lower_bound(mon_sk); now = f ? std::string(mon_C->key()) : std::string(); }
         else { auto cc = mon_rs->create_cursor(0); f = cc.lower_bound(mon_sk); now = f ? std::string(cc.key()) : std::string(); }
         if (now != mon_before) {
            printf("[mon] *** SNAPSHOT CORRUPTED at batch op %llu: now %s search ***\n",
                   (unsigned long long)mon_ops, (f && now >= mon_sk) ? ">=" : "<");
            printf("[mon] corrupting write: op=%c key=%s\n", mon_lasto ? mon_lasto : '?', mhex(mon_lastk).c_str());
            printf("[mon] before=%s\n[mon] now   =%s\n", mhex(mon_before).c_str(), mhex(now).c_str());
            mon_done = true;
         }
         if (mon_ops > 4'000'000) { printf("[mon] no corruption in 4M batch ops\n"); mon_done = true; }
      };
      std::optional<start_ref::kv::transaction> tx;
      tx.emplace(srws->start_transaction(0));
      int  c;
      auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
      while ((c = std::fgetc(f)) != EOF)
      {
         if (c == 'C')
         {
            if (tx) tx->commit();
            tx.reset();
            ++commits;
            // [lb-repro] OPLOG_MAXC: stop after N commits (then probe), OR start the
            // live-corruption monitor (OPLOG_MON) and keep replaying into an open txn.
            if (const char* mc = std::getenv("OPLOG_MAXC"))
               if (commits >= strtoull(mc, nullptr, 10))
               {
                  if (do_mon && !monitoring)
                  {
                     mon_rs   = srdb->start_read_session();
                     mon_C.emplace(mon_rs->create_cursor(0));  // held snapshot cursor
                     mon_before = mon_C->lower_bound(mon_sk) ? std::string(mon_C->key()) : std::string();
                     monitoring = true;
                     printf("[mon] start at commit %llu: before=%s\n",
                            (unsigned long long)commits, mhex(mon_before).c_str());
                     tx.emplace(srws->start_transaction(0));
                     continue;
                  }
                  if (!do_mon) break;
               }
            tx.emplace(srws->start_transaction(0));
            continue;
         }
         uint32_t kl = 0;
         if (!rd(&kl, 4)) break;
         key.resize(kl);
         if (kl && !rd(key.data(), kl)) break;
         if (c == 'U')
         {
            uint32_t vl = 0;
            if (!rd(&vl, 4)) break;
            val.resize(vl);
            if (vl && !rd(val.data(), vl)) break;
            std::string_view kv(key.data(), kl), vv(val.data(), vl);
            tx->upsert(kv, vv);
            truth[std::string(kv)] = std::string(vv);
            ++ups;
            if (monitoring) { mon_lastk.assign(kv); mon_lasto = 'U'; }
            mon_check(); if (mon_done) break;
         }
         else if (c == 'R')
         {
            std::string_view kv(key.data(), kl);
            tx->remove(kv);
            truth.erase(std::string(kv));
            ++rems;
            if (monitoring) { mon_lastk.assign(kv); mon_lasto = 'R'; }
            mon_check(); if (mon_done) break;
         }
         else if (c == 'G')
         {  // point get: INLINE read-own-writes oracle — issue against start_ref's
            // working txn and compare to the result start_ver recorded at this point.
            uint8_t  fb  = 0;
            uint64_t fnv = 0;
            if (!rd(&fb, 1) || !rd(&fnv, 8)) break;
            std::string_view kv(key.data(), kl);
            auto             got = tx->get<std::string>(kv);
            const bool       gf  = got.has_value();
            if (gf != (fb != 0) || (gf && fnv_(*got) != fnv))
            {
               if (++g_div <= 1) g_first.assign(kv);
            }
            ++g_n;
         }
         else if (c == 'L')
         {  // lower bound: parse the record. Faithful read-own-writes L replay
            // needs the committed-cursor + write_set overlay merge (TODO); for now
            // we record it in the stream and count it.
            uint8_t  fb  = 0;
            uint32_t rkl = 0;
            uint64_t fnv = 0;
            if (!rd(&fb, 1) || !rd(&rkl, 4)) break;
            if (rkl) { rkey.resize(rkl); if (!rd(rkey.data(), rkl)) break; }
            if (!rd(&fnv, 8)) break;
            ++l_n;
         }
         else break;
      }
      if (tx) tx->commit();
      // [lb-repro] OPLOG_LB=<hexkey>: issue lower_bound against start_ref and compare to
      // the ground-truth std::map. Reproduces the 1.24M lower_bound positioning bug.
      if (const char* lbhex = std::getenv("OPLOG_LB"))
      {
         std::string sk;
         for (const char* p = lbhex; p[0] && p[1]; p += 2)
            sk.push_back((char)strtoul(std::string{p[0], p[1]}.c_str(), nullptr, 16));
         auto trs = srdb->start_read_session();
         auto tc  = trs->create_cursor(0);
         bool srf = tc.lower_bound(sk);
         std::string sr_land = srf ? std::string(tc.key()) : std::string();
         auto ti = truth.lower_bound(sk);
         bool tf = ti != truth.end();
         std::string tr_land = tf ? ti->first : std::string();
         auto hx = [](const std::string& s) { std::string o; char b[3]; for (unsigned char c : s) { snprintf(b, 3, "%02x", c); o += b; } return o; };
         printf("[lb-repro] search=%s\n", hx(sk).c_str());
         printf("[lb-repro] start_ref landed: found=%d key=%s  (>=search? %d)\n",
                srf, hx(sr_land).c_str(), srf ? (sr_land >= sk) : 0);
         printf("[lb-repro] truth     landed: found=%d key=%s\n", tf, hx(tr_land).c_str());
         printf("[lb-repro] %s\n", (srf == tf && sr_land == tr_land) ? "MATCH (no bug here)"
                                                                      : "*** DIVERGENCE REPRODUCED (fresh) ***");
         // [reuse] the shim pools cursors within a txn — a non-absolute lower_bound
         // that reuses a stale path stack lands wrong. Test resets from several prior
         // positions on the REAL (deep) tree.
         auto probe = [&](const char* tag, auto preposition) {
            auto rc = trs->create_cursor(0);
            preposition(rc);
            bool        rf = rc.lower_bound(sk);
            std::string rl = rf ? std::string(rc.key()) : std::string();
            printf("[lb-repro] reuse[%s]: found=%d (>=search? %d) %s\n", tag, rf,
                   rf ? (rl >= sk) : 0,
                   (rf == tf && rl == tr_land) ? "MATCH"
                                               : "*** DIVERGENCE REPRODUCED ***");
         };
         // [snap-iso] THE test: pin a snapshot, record its lower_bound, then do a COW
         // write that touches the same router, commit, and re-check the snapshot. If it
         // changed, the COW mutated a shared committed node in place (isolation broken).
         {
            auto srs  = srdb->start_read_session();
            auto snap = srs->create_cursor(0);
            bool        bf = snap.lower_bound(sk);
            std::string before = bf ? std::string(snap.key()) : std::string();
            // COW write held OPEN (uncommitted) — immediate cow_release path, as in the
            // live read-own-writes case (committed cursor read while a write txn is open).
            auto wtx = srws->start_transaction(0);
            for (int s = 5; s < 30; ++s)  // several siblings to force COW of the shared router
            {
               std::string sib = sk; sib.back() = (char)s; sib += std::string(4, '\x11');
               wtx.upsert(sib, std::string("x"));
            }
            bool        af = snap.lower_bound(sk);
            std::string after = af ? std::string(snap.key()) : std::string();
            printf("[snap-iso] before==after? %d  (before>=search %d  after>=search %d)  %s\n",
                   (bf == af && before == after), bf ? (before >= sk) : 0, af ? (after >= sk) : 0,
                   (bf == af && before == after) ? "snapshot stable"
                                                 : "*** SNAPSHOT MUTATED BY COW (isolation broken) ***");
            printf("[snap-iso] before=%s\n[snap-iso] after =%s\n", hx(before).c_str(), hx(after).c_str());
         }
         probe("lb-prefix-then-next-byte", [&](auto& c) { c.lower_bound(sk.substr(0, sk.size() - 1)); });
         probe("lb-K0exact-then", [&](auto& c) { std::string k0 = sk; k0.back() = '\x00'; k0 += std::string(2, '\xff'); c.lower_bound(k0); });
         probe("near-past", [&](auto& c) { c.lower_bound(sk + std::string(3, '\xff')); });
         probe("far-end",   [&](auto& c) { c.lower_bound(std::string(48, '\xff')); });
         probe("begin",     [&](auto& c) { c.seek_begin(); });
         probe("begin+10next", [&](auto& c) { c.seek_begin(); for (int i = 0; i < 10; ++i) c.next(); });
         probe("lb-K0-then-next", [&](auto& c) { std::string k0 = sk; k0.back() = '\x00'; c.lower_bound(k0); c.next(); c.next(); });
      }
      printf("[verify] replayed commits=%llu U=%llu R=%llu live=%zu | reads G=%llu L=%llu\n",
             (unsigned long long)commits, (unsigned long long)ups,
             (unsigned long long)rems, truth.size(),
             (unsigned long long)g_n, (unsigned long long)l_n);
      printf("[verify] read-own-writes POINT(G) oracle: issued=%llu divergent=%llu%s%s\n",
             (unsigned long long)g_n, (unsigned long long)g_div,
             g_div ? " first=" : "", g_div ? g_first.c_str() : "");
      fflush(stdout);
      // compare every live key against start_ref via a fresh read session
      auto     rs   = srdb->start_read_session();
      auto     cur  = rs->create_cursor(0);
      uint64_t miss = 0, bad = 0, ok = 0;
      std::string first_bad;
      for (auto& [k, v] : truth)
      {
         auto got = cur.get<std::string>(k);
         if (!got)            { if (++miss <= 1) first_bad = "MISS  " + k; }
         else if (*got != v)  { if (++bad  <= 1) first_bad = "WRONG " + k; }
         else                 ++ok;
      }
      printf("[verify] POINT get: ok=%llu MISS=%llu WRONG=%llu\n",
             (unsigned long long)ok, (unsigned long long)miss, (unsigned long long)bad);
      // ── cursor lower_bound (the account-read path Silkworm actually uses) ──
      {
         uint64_t lb_miss = 0, lb_bad = 0;
         std::string lb_first;
         for (auto& [k, v] : truth)
         {
            auto lc = rs->create_cursor(0);
            if (!lc.lower_bound(k) || lc.key() != std::string_view(k))
            { if (++lb_miss <= 1) lb_first = "LB-MISS " + k; continue; }
            auto vv = lc.value<std::string>();
            if (!vv || *vv != v) { if (++lb_bad <= 1) lb_first = "LB-WRONG " + k; }
         }
         printf("[verify] cursor lower_bound: MISS=%llu WRONG=%llu\n",
                (unsigned long long)lb_miss, (unsigned long long)lb_bad);
         if ((lb_miss || lb_bad))
         {
            std::fputs("[verify] LB first-bad key hex: ", stdout);
            for (size_t i = lb_first.find(' ') + 1; i < lb_first.size(); ++i)
               printf("%02x", (unsigned char)lb_first[i]);
            std::fputc('\n', stdout);
         }
      }
      // ── read via a WRITE transaction (read-own-writes path the shim uses for
      //    Silkworm execution: mdbx_get on a RW txn -> direct_tx->get) ──
      {
         auto     vtx  = srws->start_transaction(0);
         uint64_t t_miss = 0, t_bad = 0;
         for (auto& [k, v] : truth)
         {
            auto got = vtx.get<std::string>(k);
            if (!got)           ++t_miss;
            else if (*got != v) ++t_bad;
         }
         vtx.abort();
         printf("[verify] txn read-own-writes get: MISS=%llu WRONG=%llu\n",
                (unsigned long long)t_miss, (unsigned long long)t_bad);
      }
      // ── full forward iteration vs ordered truth ──
      {
         auto     ic = rs->create_cursor(0);
         uint64_t it_n = 0, it_bad = 0;
         auto     ti = truth.begin();
         for (bool ok2 = ic.seek_begin(); ok2; ok2 = ic.next(), ++ti, ++it_n)
         {
            if (ti == truth.end()) { ++it_bad; break; }
            auto vv = ic.value<std::string>();
            if (ic.key() != std::string_view(ti->first) || !vv || *vv != ti->second)
               ++it_bad;
         }
         printf("[verify] forward iterate: visited=%llu truth=%zu order/val_bad=%llu\n",
                (unsigned long long)it_n, truth.size(), (unsigned long long)it_bad);
      }
      if (miss || bad)
      {
         printf("[verify] first divergence: %s (klen=%zu)\n",
                first_bad.size() > 6 ? "(see below)" : "?", first_bad.size() - 6);
         // hex-dump the first divergent key
         std::fputs("[verify] key hex: ", stdout);
         for (size_t i = 6; i < first_bad.size(); ++i)
            printf("%02x", (unsigned char)first_bad[i]);
         std::fputc('\n', stdout);
      }
      fflush(stdout);
      std::fclose(f);
      return (miss || bad) ? 1 : 0;
   }

   TK::config cfg;
   cfg.commit_sync = sync;
   auto db = TK::database::open(dir, sal::open_mode::create_or_open, cfg);
   auto ws = db->start_write_session();

   printf("oplog-replay %s -> %s  (commit_sync=%s)\n", oplog.c_str(), dir.c_str(), syncs.c_str());
   printf("%12s %12s %10s %10s %12s %12s\n", "commits", "ops", "U", "R", "base_GB", "delta_GB");
   fflush(stdout);

   std::vector<char>              key, val;
   uint64_t                       commits = 0, ops = 0, ups = 0, rems = 0;
   std::optional<TK::transaction> tx;
   tx.emplace(ws->start_transaction(0));
   int  c;
   auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };

   while ((c = std::fgetc(f)) != EOF)
   {
      if (c == 'C')
      {
         if (tx)
            tx->commit();
         tx.reset();
         ++commits;
         if (commits % report == 0)
         {
            printf("%12llu %12llu %10llu %10llu %12.3f %12.3f\n", (unsigned long long)commits,
                   (unsigned long long)ops, (unsigned long long)ups, (unsigned long long)rems,
                   dir_gb(fs::path(dir) / "base"), dir_gb(fs::path(dir) / "delta"));
            fflush(stdout);
         }
         tx.emplace(ws->start_transaction(0));
         continue;
      }
      uint32_t kl = 0;
      if (!rd(&kl, 4))
         break;
      key.resize(kl);
      if (kl && !rd(key.data(), kl))
         break;
      ++ops;
      if (c == 'U')
      {
         uint32_t vl = 0;
         if (!rd(&vl, 4))
            break;
         val.resize(vl);
         if (vl && !rd(val.data(), vl))
            break;
         tx->upsert(std::string_view(key.data(), kl), std::string_view(val.data(), vl));
         ++ups;
      }
      else if (c == 'R')
      {
         tx->remove(std::string_view(key.data(), kl));
         ++rems;
      }
      else if (c == 'G')
      {  // read record: skip <found:u8><fnv:u64>
         uint8_t fb = 0; uint64_t fnv = 0;
         if (!rd(&fb, 1) || !rd(&fnv, 8)) break;
      }
      else if (c == 'L')
      {  // read record: skip <found:u8><rkl:u32> rkey <fnv:u64>
         uint8_t fb = 0; uint32_t rkl = 0; uint64_t fnv = 0;
         if (!rd(&fb, 1) || !rd(&rkl, 4)) break;
         if (rkl) { val.resize(rkl); if (!rd(val.data(), rkl)) break; }
         if (!rd(&fnv, 8)) break;
      }
      else
      {
         break;  // unknown op — corrupt/eof
      }
   }
   if (tx)
      tx->commit();
   db->flush_now();
   std::fclose(f);

   printf("\nDONE commits=%llu ops=%llu (U=%llu R=%llu)\n", (unsigned long long)commits,
          (unsigned long long)ops, (unsigned long long)ups, (unsigned long long)rems);
   printf("final footprint base=%.3f GB delta=%.3f GB total=%.3f GB\n",
          dir_gb(fs::path(dir) / "base"), dir_gb(fs::path(dir) / "delta"), dir_gb(dir));
   fflush(stdout);
   return 0;
}

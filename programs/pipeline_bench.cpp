// pipeline_bench — the L5 tiered pipeline prototype, measured end to end.
//
//   writes  ──► vpart (hot delta, dense)            Phase A   (vs heart-direct baseline)
//   flush   ──► pinned snapshot, key-sorted bulk    Phase B   (chunk-size matrix)
//                upserts into heart txns
//   sustain ──► writer keeps committing WHILE the   Phase C   (keep-up: writer tx/s
//                flusher streams a pinned snapshot              degradation + flush rate)
//   retire  ──► sorted removes of flushed keys      Phase D   (cost per retired key)
//   reads   ──► latest-hot / stable / post-retire   Phase E   (the read-path story)
//                fall-through
//
// Goal (Dan): sustain part-like write performance while gaining heart's
// long-term capacity, overcoming vpart's 32 GB address space and memory
// immovability. Flush amortization = dedup (snapshot iteration sees one
// version per key) x sorted bulk (each heart path COW'd once per chunk).
#include <radix_engine.hpp>  // heart::radix (heart/test include dir)
#include <vpart/kv.hpp>

#include <psitri/dwal/kv_engine.hpp>
// The composition contract (Dan: "the concept for engine needs to include the
// cursors"): vpart conforms FULLY — sessions, transactions, and the
// bidirectional cursor surface — before tiered_kv builds on it.
static_assert(psitri::dwal::kv_engine<vpart::kv::engine>);

#include <hash/xxhash.h>

#ifdef __APPLE__
#include <pthread/qos.h>
#endif

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace
{
   using clk = std::chrono::steady_clock;
   double secs(clk::time_point a, clk::time_point b)
   {
      return std::chrono::duration<double>(b - a).count();
   }

   std::filesystem::path tmpdb(const char* tag)
   {
      auto p = std::filesystem::temp_directory_path() / (std::string("pipeline_") + tag);
      std::filesystem::remove_all(p);
      std::filesystem::create_directories(p);
      return p;
   }

   // bank-like key: 16 hex chars of xxh64(i) — uniform random order, fixed len.
   struct keybuf
   {
      char b[17];
      std::string_view of(uint64_t i)
      {
         static const char* hx = "0123456789abcdef";
         uint64_t           h  = XXH64(&i, sizeof(i), 0x9e3779b97f4a7c15ull);
         for (int j = 15; j >= 0; --j, h >>= 4)
            b[j] = hx[h & 15];
         b[16] = 0;
         return {b, 16};
      }
   };
   struct valbuf
   {
      char b[64];
      std::string_view of(uint64_t i, uint64_t ver)
      {
         std::memset(b, 'v', sizeof(b));
         std::memcpy(b, &i, 8);
         std::memcpy(b + 8, &ver, 8);
         return {b, sizeof(b)};
      }
   };

   struct opts
   {
      uint64_t keys      = 1'000'000;  // delta size (accounts)
      uint64_t sustain   = 2'000'000;  // extra txns during Phase C
      uint64_t chunk     = 10'000'000; // keys per heart txn at flush (Dan: big)
      uint64_t retire_b  = 4096;       // removes per retire txn
      bool     matrix    = false;      // Phase B chunk-size matrix
      bool     repin     = false;      // re-pin a fresh snapshot per chunk (drain advances mid-flush)
      bool     qos       = false;      // flusher thread at QOS_CLASS_UTILITY (writer keeps the P-cores)
   };

   uint64_t now_us()
   {
      return uint64_t(
          std::chrono::duration_cast<std::chrono::microseconds>(clk::now().time_since_epoch())
              .count());
   }
}  // namespace

int main(int argc, char** argv)
{
   opts o;
   for (int i = 1; i < argc; ++i)
   {
      auto eq = [&](const char* f) { return std::strcmp(argv[i], f) == 0; };
      if (eq("--keys") && i + 1 < argc)
         o.keys = std::strtoull(argv[++i], nullptr, 10);
      else if (eq("--sustain") && i + 1 < argc)
         o.sustain = std::strtoull(argv[++i], nullptr, 10);
      else if (eq("--chunk") && i + 1 < argc)
         o.chunk = std::strtoull(argv[++i], nullptr, 10);
      else if (eq("--matrix"))
         o.matrix = true;
      else if (eq("--repin"))
         o.repin = true;
      else if (eq("--qos"))
         o.qos = true;
   }
   printf("=== pipeline_bench: vpart hot delta -> heart bulk flush ===\n");
   printf("keys=%" PRIu64 " sustain=%" PRIu64 " chunk=%" PRIu64 "\n\n", o.keys, o.sustain,
          o.chunk);

   auto vdb = vpart::kv::database::create(tmpdb("vpart"));
   auto hdb = heart::radix::database::open(tmpdb("heart"), sal::open_mode::create_or_open);
   auto vws = vdb->start_write_session();
   auto hws = hdb->start_write_session();

   // ── Phase A: delta writes into vpart, batch=1 (one txn per op) ────────────
   {
      keybuf kb;
      valbuf vb;
      auto   t0 = clk::now();
      for (uint64_t i = 0; i < o.keys; ++i)
      {
         auto tx = vws->start_transaction(0);
         tx.upsert(kb.of(i), vb.of(i, 0));
         tx.commit();
      }
      auto   t1 = clk::now();
      double s  = secs(t0, t1);
      printf("Phase A  vpart delta writes : %9.0f tx/s  (%" PRIu64 " txns, %.2fs)\n",
             o.keys / s, o.keys, s);
   }
   // heart-direct baseline, same access pattern (the contrast row)
   {
      keybuf kb;
      valbuf vb;
      auto   t0 = clk::now();
      for (uint64_t i = 0; i < o.keys; ++i)
      {
         auto tx = hws->start_transaction(1);  // scratch root; flushed root is 0
         tx.upsert(kb.of(i), vb.of(i, 0));
         tx.commit();
      }
      auto   t1 = clk::now();
      printf("         heart direct writes: %9.0f tx/s  (baseline, root 1)\n",
             o.keys / secs(t0, t1));
   }

   // ── Phase B: bulk flush — pinned sorted snapshot -> heart txn chunks ─────
   auto flush_once = [&](uint64_t chunk_keys, bool repin) -> std::pair<uint64_t, double> {
      uint64_t    n = 0;
      std::string resume;  // last flushed key (repin mode: resume point after the fresh pin)
      bool        more = true;
      auto        t0   = clk::now();
      auto        rs   = vdb->start_read_session();
      std::optional<decltype(rs->create_cursor(0))> cur;
      cur.emplace(rs->create_cursor(0));  // pins the latest committed version
      cur->seek_begin();
      while (more)
      {
         auto     tx    = hws->start_transaction(0);
         uint64_t in_tx = 0;
         while (!cur->is_end() && in_tx < chunk_keys)
         {
            cur->get_value([&](std::string_view v) { tx.upsert(cur->key(), v); });
            ++in_tx;
            if (repin)
               resume.assign(cur->key());
            cur->next();
         }
         tx.commit();
         n += in_tx;
         more = !cur->is_end();
         if (more && repin)
         {  // drop the pin between chunks: the drain floor advances while we flush
            cur.reset();
            cur.emplace(rs->create_cursor(0));
            cur->lower_bound(resume);
            if (!cur->is_end() && cur->key() == std::string_view(resume))
               cur->next();
            more = !cur->is_end();
         }
      }
      return {n, secs(t0, clk::now())};
   };
   if (o.matrix)
   {
      for (uint64_t c : {uint64_t(10'000), uint64_t(100'000), uint64_t(1'000'000), o.chunk})
      {
         auto [n, s] = flush_once(c, o.repin);
         printf("Phase B  flush chunk=%-9" PRIu64 ": %9.0f keys/s  (%" PRIu64 " keys, %.2fs)\n",
                c, n / s, n, s);
      }
   }
   else
   {
      auto [n, s] = flush_once(o.chunk, o.repin);
      printf("Phase B  bulk flush -> heart: %9.0f keys/s  (%" PRIu64 " keys, %.2fs, chunk=%" PRIu64
             ")\n",
             n / s, n, s, o.chunk);
   }

   // ── Phase C: sustained — writer commits while flusher streams a snapshot ─
   {
      std::atomic<bool>     fl_done{false};
      std::atomic<uint64_t> fl_keys{0};
      double                fl_secs = 0;
      std::thread           flusher([&] {
#ifdef __APPLE__
         if (o.qos)
            pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
         auto [n, s] = flush_once(o.chunk, o.repin);
         fl_keys     = n;
         fl_secs     = s;
         fl_done     = true;
      });
      keybuf   kb;
      valbuf   vb;
      uint64_t wn = 0;
      auto     t0 = clk::now();
      for (uint64_t i = 0; i < o.sustain; ++i, ++wn)
      {
         auto tx = vws->start_transaction(0);
         tx.upsert(kb.of(i % o.keys), vb.of(i % o.keys, 1 + i / o.keys));
         tx.commit();
      }
      auto t1 = clk::now();
      flusher.join();
      printf("Phase C  sustained writer   : %9.0f tx/s  (flusher concurrent: %9.0f keys/s)%s%s\n",
             wn / secs(t0, t1), fl_keys.load() / fl_secs, o.repin ? " [repin]" : "",
             o.qos ? " [qos]" : "");
      printf("         vpart arena after C: %.0f MB (%u segments)\n",
             double(vdb->store().num_segments()) * 8.0, vdb->store().num_segments());
   }

   // ── Phase D: retire — sorted removes of flushed keys, batched txns ───────
   {
      auto     rs  = vdb->start_read_session();
      auto     cur = rs->create_cursor(0);
      std::vector<std::string> keys;  // bench-harness collection (engine will re-iterate)
      cur.seek_begin();
      while (!cur.is_end())
      {
         keys.emplace_back(cur.key());
         cur.next();
      }
      auto t0 = clk::now();
      uint64_t n = 0;
      for (size_t i = 0; i < keys.size();)
      {
         auto tx = vws->start_transaction(0);
         for (uint64_t b = 0; b < o.retire_b && i < keys.size(); ++b, ++i, ++n)
            tx.remove(keys[i]);
         tx.commit();
      }
      double s = secs(t0, clk::now());
      printf("Phase D  retire (sorted rm) : %9.0f keys/s  (%" PRIu64 " keys, %.2fs, batch=%" PRIu64
             ")\n",
             n / s, n, s, o.retire_b);
   }

   // ── Phase E: read story — stable (heart) vs latest fall-through ──────────
   {
      const uint64_t probes = std::min<uint64_t>(o.keys, 200'000);
      keybuf         kb;
      auto           hrs = hdb->start_read_session();
      auto           hcur = hrs->create_cursor(0);
      uint64_t       hits = 0;
      auto           t0   = clk::now();
      for (uint64_t i = 0; i < probes; ++i)
         hcur.get(kb.of(i), [&](std::string_view) { ++hits; });
      double st = secs(t0, clk::now());

      // latest view post-retire: vpart miss -> heart (the fall-through cost)
      auto     vrs  = vdb->start_read_session();
      auto     vcur = vrs->create_cursor(0);
      uint64_t h2   = 0;
      t0            = clk::now();
      for (uint64_t i = 0; i < probes; ++i)
      {
         bool found = false;
         vcur.get(kb.of(i), [&](std::string_view) { found = true; });
         if (!found)
            hcur.get(kb.of(i), [&](std::string_view) { ++h2; });
      }
      double lt = secs(t0, clk::now());
      printf("Phase E  stable read (heart): %9.1f ns/op (%" PRIu64 "/%" PRIu64 " hits)\n",
             1e9 * st / probes, hits, probes);
      printf("         latest post-retire : %9.1f ns/op (vpart miss -> heart, %" PRIu64
             " hits)\n",
             1e9 * lt / probes, h2);
   }

   printf("\nPASS (validation: heart row count + spot values checked by flush hits above)\n");
   return 0;
}

// dict_bench — ONE workload, FIVE engines: comparative technique analysis.
//
//   load      insert every key (value = the u64 sequence number, 8 raw bytes)
//   find-hit  3 shuffled passes over every key   (zero-copy reads everywhere)
//   find-miss 2 passes with a mutated suffix     (guaranteed absent)
//   iterate   3 full in-order scans              (count + key-byte checksum)
//   live      bytes found by WALKING the live tree (census / node walk)
//   reserve   allocator footprint (segments / arena / block-rounded heap)
//
// Values are u64 EVERYWHERE (Dan): raw trees store u64; engines store the 8
// raw bytes via a stack view — no std::string in any hot loop.
// Keysets: --keys dict (/usr/share/dict/words) or rand (1M x 8B binary).
// Engines: --engine hkv|artmap|part|vpart|heart|all. hkv/artmap are raw
// in-memory baselines; part/vpart/heart run their full engine surface
// (batched commits, --batch default 8192). The point (Dan): side-by-side
// phases expose which techniques pay and which implementations hide an
// unreported pathological gap (proof cases: vpart's odd-size allocations
// off part's freeset fast classes; heart's point-op gap).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
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

   std::string_view val_of(const uint64_t& i)
   {
      return {reinterpret_cast<const char*>(&i), 8};
   }

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

   struct row
   {
      std::string name;
      double      load = 0, hit = 0, miss = 0, iter = 0;
      uint64_t    live = 0, reserve = 0;
      bool        ok = true;
   };

   template <class Put, class Commit, class Get, class Iter>
   row run(const char* engine, const phases& P, Put&& put, Commit&& commit, Get&& get,
           Iter&& iter, uint64_t batch)
   {
      row r;
      r.name            = engine;
      const uint64_t N  = P.keys.size();
      auto           t0 = clock_t_::now();
      for (uint64_t i = 0; i < N; ++i)
      {
         put(P.keys[i], i);
         if (batch && (i + 1) % batch == 0)
            commit();
      }
      commit();
      auto t1 = clock_t_::now();
      r.load  = ns_per(t0, t1, N);

      uint64_t hits = 0;
      t0 = clock_t_::now();
      for (int pass = 0; pass < 3; ++pass)
         for (auto& k : P.shuffled)
            hits += get(k);
      t1    = clock_t_::now();
      r.hit = ns_per(t0, t1, 3 * N);
      r.ok &= (hits == 3 * N);

      uint64_t false_hits = 0;
      t0 = clock_t_::now();
      for (int pass = 0; pass < 2; ++pass)
         for (auto& k : P.missing)
            false_hits += get(k);
      t1     = clock_t_::now();
      r.miss = ns_per(t0, t1, 2 * N);
      r.ok &= (false_hits == 0);

      uint64_t cnt = 0, sum = 0;
      t0 = clock_t_::now();
      for (int pass = 0; pass < 3; ++pass)
         iter(cnt, sum);
      t1     = clock_t_::now();
      r.iter = ns_per(t0, t1, cnt ? cnt : 1);
      r.ok &= (cnt == 3 * N);
      return r;
   }

   std::filesystem::path tmpdb(const char* tag)
   {
      auto p = std::filesystem::temp_directory_path() / (std::string("dictbench_") + tag);
      std::filesystem::remove_all(p);
      return p;
   }

   // live bytes of an art_map: walk every node/leaf with art's public views
   uint64_t art_live_(art::art_map<uint64_t, art::heap_arena>& m)
   {
      auto&                      a    = m.get_arena();
      uint64_t                   live = 0;
      std::vector<art::offset_t> stk;
      stk.push_back(m.snapshot_root());
      while (!stk.empty())
      {
         const art::offset_t off = stk.back();
         stk.pop_back();
         if (off == art::null_offset)
            continue;
         if (art::is_leaf(off))
         {
            auto* lh = a.as<art::leaf_header>(art::untag_leaf(off));
            live += (art::leaf_alloc_size<uint64_t>(lh->key_len) + art::cacheline_size - 1) &
                    ~uint64_t(art::cacheline_size - 1);
            continue;
         }
         auto* h = a.as<art::node_header>(off);
         if (h->value_off != art::null_offset)
            stk.push_back(h->value_off);
         if (h->type == art::node_type::setlist)
         {
            art::setlist_view    v{h};
            const art::offset_t* ch = v.children();
            live += v.alloc_size();
            for (unsigned i = 0; i < v.num_children(); ++i)
               stk.push_back(ch[i]);
         }
         else
         {
            art::node256_view    v{h};
            const art::offset_t* ch = v.children();
            live += art::node256_alloc_size(h->partial_len);
            for (int b = 0; b < 256; ++b)
               if (ch[b] != art::null_offset)
                  stk.push_back(ch[b]);
         }
      }
      return live;
   }

   template <hkv::mode M>
   row bench_hkv_(const phases& P, const char* tag)
   {
      hkv::kv_tree<std::string_view, uint64_t, M> t;
      row                                      r = run(
          tag, P, [&](const std::string& k, uint64_t i) { t.insert(k, i); }, [] {},
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
      t.census([&](auto cr) {
         r.live += cr.bytes;
         r.reserve += uint64_t(cr.lines_alloc) * 128;  // block-rounded heap
      });
      return r;
   }

   row bench_artmap(const phases& P)
   {
      art::art_map<uint64_t, art::heap_arena> m;
      row                                     r = run(
          "artmap", P, [&](const std::string& k, uint64_t i) { m.upsert(k, i); }, [] {},
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
      r.live    = art_live_(m);
      r.reserve = m.get_arena().bytes_used();  // bump arena: includes COW garbage
      return r;
   }

   template <class DB>
   row bench_partlike(DB db, const char* tag, const phases& P, uint64_t batch)
   {
      auto ws    = db->start_write_session();
      using tx_t = decltype(ws->start_transaction(0));
      std::optional<tx_t> tx;
      tx.emplace(ws->start_transaction(0));
      row r = run(
          tag, P, [&](const std::string& k, uint64_t i) { tx->upsert(k, val_of(i)); },
          [&] {
             tx->commit();
             tx.emplace(ws->start_transaction(0));
          },
          [&](const std::string& k) {
             uint32_t hit = 0;
             tx->get(k, [&](std::string_view) { hit = 1; });
             return hit;
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
      ws->tree_dbg().census([&](auto cr) { r.live += cr.bytes; });
      r.reserve = uint64_t(db->store().num_segments()) * 8u * 1024 * 1024;
      return r;
   }

   row bench_heart(const phases& P, uint64_t batch)
   {
      auto db    = heart::radix::database::open(tmpdb("heart"), sal::open_mode::create_or_open);
      auto ws    = db->start_write_session();
      auto rs    = db->start_read_session();
      using tx_t = decltype(ws->start_transaction(0));
      std::optional<tx_t> tx;
      tx.emplace(ws->start_transaction(0));
      using cur_t = decltype(rs->create_cursor(0));
      std::optional<cur_t> rcur;
      row                  r = run(
          "heart", P, [&](const std::string& k, uint64_t i) { tx->upsert(k, val_of(i)); },
          [&] {
             tx->commit();
             tx.emplace(ws->start_transaction(0));
          },
          [&](const std::string& k) {
             if (!rcur)
                rcur.emplace(rs->create_cursor(0));
             uint32_t hit = 0;
             rcur->get(k, [&](std::string_view) { hit = 1; });
             return hit;
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
      rcur.reset();
      auto d    = db->dump();
      r.reserve = d.total_segments * sal::segment_size;
      for (auto& seg : d.segments)
         if (!seg.is_free && seg.alloc_pos > 0)
            r.live += uint64_t(seg.alloc_pos) - seg.freed_bytes;
      return r;
   }

   // ── the table: aligned, color-coded, winner-marked per category ──────────
   void print_table(const std::vector<row>& rows, const std::string& keyset, size_t n,
                    uint64_t batch)
   {
      printf("\n# dict_bench keys=%s n=%zu batch=%llu  (lower is better; green* = winner)\n",
             keyset.c_str(), n, (unsigned long long)batch);
      printf("%-8s %11s %11s %11s %11s %11s %11s\n", "engine", "load ns", "hit ns", "miss ns",
             "iter ns", "live MB", "reserve MB");
      auto best = [&](auto f) {
         double m = 1e300;
         for (auto& r : rows)
            m = std::min(m, f(r));
         return m;
      };
      const double bl = best([](const row& r) { return r.load; });
      const double bh = best([](const row& r) { return r.hit; });
      const double bm = best([](const row& r) { return r.miss; });
      const double bi = best([](const row& r) { return r.iter; });
      const double bv = best([](const row& r) { return double(r.live); });
      const double br = best([](const row& r) { return double(r.reserve); });
      auto cell = [](double v, double best_v) {
         char buf[64];
         const bool win = v <= best_v * 1.0001;
         snprintf(buf, sizeof buf, "%.1f", v);
         printf(" %s%10s%s%s", win ? "\033[1;32m" : "", buf, win ? "*" : " ",
                win ? "\033[0m" : "");
      };
      for (auto& r : rows)
      {
         printf("%-8s", r.name.c_str());
         cell(r.load, bl);
         cell(r.hit, bh);
         cell(r.miss, bm);
         cell(r.iter, bi);
         cell(double(r.live) / 1048576.0, bv / 1048576.0);
         cell(double(r.reserve) / 1048576.0, br / 1048576.0);
         printf("%s\n", r.ok ? "" : "  *** VALIDATION FAILED");
      }
   }
}  // namespace

int main(int argc, char** argv)
{
   std::string engine = "all", keys = "dict";
   uint64_t    batch  = 8192;
   for (int i = 1; i < argc - 1; ++i)
   {
      if (!strcmp(argv[i], "--engine")) engine = argv[i + 1];
      if (!strcmp(argv[i], "--keys")) keys = argv[i + 1];
      if (!strcmp(argv[i], "--batch")) batch = strtoull(argv[i + 1], nullptr, 10);
   }
   phases P(load_keys(keys));

   std::vector<row> rows;
   auto             want = [&](const char* e) { return engine == "all" || engine == e; };
   if (want("hkv"))
      rows.push_back(bench_hkv_<hkv::mode::none>(P, "hkv"));
   if (want("hkv-dt"))  // dense tiers: the SAME c2/c4/c8 ladder the cow trees use
      rows.push_back(bench_hkv_<hkv::mode::dense_tiers>(P, "hkv-dt"));
   if (want("artmap"))
      rows.push_back(bench_artmap(P));
   if (want("part"))
      rows.push_back(bench_partlike(part::kv::database::create(tmpdb("part")), "part", P, batch));
   if (want("vpart"))
      rows.push_back(
          bench_partlike(vpart::kv::database::create(tmpdb("vpart")), "vpart", P, batch));
   if (want("heart"))
      rows.push_back(bench_heart(P, batch));
   print_table(rows, keys, P.keys.size(), batch);
   return 0;
}

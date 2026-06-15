// tree_shape_audit — READ-ONLY leak + tree-shape audit of a sal_art_ref cold base.
//
// Walks every node reachable from the committed roots (the TRUE live set), summing
// bytes + counting nodes, and compares to the allocator's allocated-live accounting
// (Σ over segments of alloc_pos−freed). The difference is LEAKED: nodes allocated +
// counted live but unreachable from any root — what crash-recovery would reclaim,
// measured with ZERO writes. Same pass yields tree-shape histograms.
//
//   tree-shape-audit <base-dir>      (cold base, e.g. <datadir>/chaindata/base)
#include <radix_engine.hpp>  // sal_art_ref::radix::database, radix_type_ids

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
namespace rti = sal_art_ref::radix_type_ids;

static double gb(double b) { return b / (1024.0 * 1024.0 * 1024.0); }

// splitmix64 — deterministic, well-mixed; gives Ethereum-like hashed keys.
static inline uint64_t smix(uint64_t x)
{
   x += 0x9e3779b97f4a7c15ull;
   x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
   x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
   return x ^ (x >> 31);
}

// AUDIT_BUILD=N: build a fresh, deterministic base of N random KLEN-byte keys so the
// tree shape can be measured on IDENTICAL input before/after the radix-shape fix.
// Single committed root at idx 0; same engine path the cold base uses in production.
static void build_base(const std::string& dir, uint64_t n)
{
   const uint64_t klen = []() { const char* e = getenv("AUDIT_BUILD_KLEN"); return e ? strtoull(e, nullptr, 10) : 20; }();
   printf("AUDIT_BUILD: building %llu keys (klen=%llu) into %s ...\n",
          (unsigned long long)n, (unsigned long long)klen, dir.c_str());
   fflush(stdout);
   fs::remove_all(dir);
   fs::create_directories(dir);
   {
      auto db = sal_art_ref::radix::database::open(dir, sal::open_mode::create_or_open);
      auto ws = db->start_write_session();
      uint8_t  kbuf[256], vbuf[16];
      uint64_t i = 0;
      const uint64_t BATCH = 200000;
      while (i < n)
      {
         auto           txn = ws->start_transaction(0);
         const uint64_t end = std::min(n, i + BATCH);
         for (; i < end; ++i)
         {
            // 3 mixed words → up to 24 key bytes (use klen), 16 value bytes
            uint64_t a = smix(i * 2 + 1), b = smix(a ^ 0xABCDEF), c = smix(b + i);
            std::memcpy(kbuf + 0, &a, 8);
            std::memcpy(kbuf + 8, &b, 8);
            for (uint64_t off = 16; off < klen; off += 8) { c = smix(c); std::memcpy(kbuf + off, &c, std::min<uint64_t>(8, klen - off)); }
            std::memcpy(vbuf + 0, &c, 8);
            std::memcpy(vbuf + 8, &a, 8);
            txn.upsert(std::string_view{reinterpret_cast<const char*>(kbuf), klen},
                       std::string_view{reinterpret_cast<const char*>(vbuf), sizeof(vbuf)});
         }
         txn.commit();
      }
   }  // close writer (flush + release flock) before the read-only audit reopens
   printf("AUDIT_BUILD: done.\n\n");
   fflush(stdout);
}

static const char* type_name(uint8_t t)
{
   using K = rti::radix_kind;
   switch (rti::kind_of(sal::header_type(t)))
   {
      case K::leaf: return "leaf";
      case K::setlist_u8: return "setlist_u8";
      case K::setlist_u16: return "setlist_u16";
      case K::setlist_u32: return "setlist_u32";
      case K::bitset1: return "bitset1";
      case K::bitset_r2: return "bitset_r2";
      case K::bitset_r4: return "bitset_r4";
      case K::full: return "full_radix";
      case K::bucket: return "bucket";
      case K::c1_32: return "bitset_c1_32";
      case K::c1_64: return "bitset_c1_64";
      case K::c1_128: return "bitset_c1_128";
      case K::page: return "leaf_page";
      default: return "other/key_val";
   }
}

int main(int argc, char** argv)
{
   std::string base = argc > 1 ? argv[1]
       : std::string(getenv("HOME")) + "/eth-bench/baseline-psitri/chaindata/base";

   // AUDIT_BUILD=N: synthesize a fresh deterministic base here, then audit it. Lets us
   // compare tree shape on IDENTICAL input across code changes (the radix-shape fix).
   if (const char* nb = getenv("AUDIT_BUILD"))
   {
      if (argc <= 1)
         base = std::string(getenv("HOME")) + "/eth-bench/audit_synth_base";
      build_base(base, strtoull(nb, nullptr, 10));
   }

   printf("tree-shape-audit (read-only) on %s\n\n", base.c_str());
   fflush(stdout);

   auto  db    = sal_art_ref::radix::database::open(base, sal::open_mode::read_only);
   auto& alloc = db->underlying_allocator();
   auto& eng   = db->engine_for_testing();

   // ---- allocated-live baseline: Σ over non-free segments of (alloc_pos − freed) ----
   auto     d              = db->dump();
   uint64_t allocated_live = 0, free_bytes = 0, used_segs = 0;
   for (auto& s : d.segments)
   {
      if (s.is_free)
         continue;
      if (s.alloc_pos > 0)
      {
         allocated_live += uint64_t(s.alloc_pos) - s.freed_bytes;
         ++used_segs;
      }
      free_bytes += s.freed_bytes;
   }
   const uint64_t segs_file = fs::file_size(fs::path(base) / "segs");

   printf("== allocator accounting (on disk) ==\n");
   printf("  segs file            %8.2f GB   (%llu segments)\n", gb(segs_file),
          (unsigned long long)d.total_segments);
   printf("  allocated-live       %8.2f GB   across %llu used segments\n", gb(allocated_live),
          (unsigned long long)used_segs);
   printf("  freed (reclaimable)  %8.2f GB\n", gb(free_bytes));
   printf("  ptr_alloc CBs (used) %8llu      distinct live node addresses\n",
          (unsigned long long)alloc.total_allocated_objects());
   fflush(stdout);

   // ---- walk reachable set from committed roots ----
   //  bytes   = allocated size()  |  compact = compact_size() (128-rounded tight size)
   //  payload = actual user key+value bytes  |  n_compacted = nodes already at compact size
   struct Stat
   {
      uint64_t count = 0, bytes = 0, compact = 0, payload = 0, n_compacted = 0;
   };
   Stat                      per_type[256];
   uint64_t                  total_compact = 0, total_payload = 0, total_compacted = 0;
   uint64_t                  total_page_keys = 0;  // Σ leaf_page::num_keys() → avg keys/page
   const uint64_t page_sample = std::getenv("AUDIT_PAGE_SAMPLE")
                                    ? strtoull(std::getenv("AUDIT_PAGE_SAMPLE"), nullptr, 10) : 0;
   bool                      stop_walk = false;
   uint64_t                  kv_key_bytes = 0, kv_value_bytes = 0;  // key_val key vs value split
   uint64_t                  size_hist[40] = {0};   // log2 buckets
   uint64_t                  depth_hist[64] = {0};  // terminal (leaf-of-DAG) depth
   uint64_t                  sl_fanout[8] = {0};    // setlist_u8 branch count: 1,2-4,5-8,9-16,17-32,33-64,65-128,129+
   // in-page suffix divergence: shared-prefix length between ADJACENT sorted suffixes.
   // cpl==0 → the two keys differ on the FIRST suffix byte (resolve in 1 byte); cpl>=1 → they
   // share byte(s), needing 2+ to resolve. + the page's chosen resolution column width (_pfxw).
   uint64_t adj_cpl_hist[8] = {0};
   uint64_t adj_pairs       = 0;
   uint64_t pfxw_hist[5]    = {0};  // index by _pfxw (1,2,4)
   uint64_t cpfx_len_sum    = 0;    // Σ leaf_page::_cpfx_len → avg factored common prefix
   uint64_t cpfx_hist[8]    = {0};  // cpfx length buckets: 0,1,2,3,4-7,8-15,16-31,32+
   uint64_t ncb_sum         = 0;    // Σ leaf_page::num_cbases() → avg cbases used (extern slots)
   uint64_t pages_zero_ncb  = 0;    // pages with NO externs (cbases[16] = 64B dead weight)
   uint64_t                  sl1_term = 0, sl1_inlineval = 0, sl1_childptr = 0;  // single-branch breakdown
   uint64_t                  total_nodes = 0, total_bytes = 0, terminals = 0;
   std::unordered_set<uint64_t> seen;
   seen.reserve(64u << 20);

   // First: enumerate the whole root table — which slots are non-null, and how big
   // is each root's immediate subtree (peek child count without recursing).
   printf("\n== root table scan ==\n");
   std::vector<uint32_t> populous;
   for (uint32_t idx = 0; idx < 1024; ++idx)
   {
      auto root = eng.get_root(idx);
      if (!root)
         continue;
      auto                     ses = root.session();
      const sal::alloc_header* hdr = ses->get_ref<sal::alloc_header>(root.address()).obj();
      int                      kids = 0;
      if (hdr)
         alloc.type_ops(hdr).visit_children(hdr, [&](sal::ptr_address) { ++kids; });
      // count_keys() walks the whole cursor — slow at billions of keys; gate behind env.
      const bool     ck    = std::getenv("AUDIT_COUNTKEYS") != nullptr;
      const uint64_t nkeys = ck ? root.count_keys() : 0;
      printf("  root[%4u] @ 0x%-9x type=%-14s immediate-children=%d  count_keys=%s\n", idx,
             *root.address(), hdr ? type_name(uint8_t(hdr->type())) : "(null)", kids,
             ck ? std::to_string(nkeys).c_str() : "(skipped; set AUDIT_COUNTKEYS)");
      populous.push_back(idx);
   }
   if (populous.empty())
      printf("  (no non-null roots)\n");

   for (uint32_t idx : populous)
   {
      auto root = eng.get_root(idx);
      if (!root)
         continue;
      auto        ses = root.session();
      printf("\n  walking root[%u] @ 0x%x ...\n", idx, *root.address());
      fflush(stdout);

      std::vector<std::pair<sal::ptr_address, int>> stack;
      stack.push_back({root.address(), 0});
      while (!stack.empty())
      {
         auto [a, depth] = stack.back();
         stack.pop_back();
         if (a == sal::null_ptr_address || !seen.insert(*a).second)
            continue;
         const sal::alloc_header* hdr = ses->get_ref<sal::alloc_header>(a).obj();
         if (!hdr)
            continue;
         const uint32_t sz = hdr->size();
         ++total_nodes;
         total_bytes += sz;
         const uint8_t ty = uint8_t(hdr->type());
         uint32_t      compact = alloc.type_ops(hdr).compact_size(hdr);  // compactor's tight size
         if (compact > sz)
            compact = sz;
         // user payload = actual key+value bytes (everything else — struct header, the 128B
         // skip-header headroom, router pointers, leaf-page columns, slack, rounding — is overhead
         // that compact_size does NOT subtract). routers/buckets hold no user K/V → payload 0.
         uint64_t payload = 0;
         if (ty == uint8_t(sal_art_ref::key_val_node::type_id))
         {
            const auto* n = ses->get_ref<sal_art_ref::key_val_node>(a).obj();
            payload       = uint64_t(n->key_len) + n->value_len;
            kv_key_bytes += n->key_len;     // FULL key stored (the prefix-duplication candidate)
            kv_value_bytes += n->value_len;
         }
         else if (ty == uint8_t(sal_art_ref::leaf_node::type_id))
         {
            const auto* n = ses->get_ref<sal_art_ref::leaf_node>(a).obj();
            payload       = uint64_t(n->value.suffix_len) + n->value.value_len;
         }
         else if (ty == uint8_t(sal_art_ref::leaf_page::type_id))
         {
            const auto* p = ses->get_ref<sal_art_ref::leaf_page>(a).obj();
            total_page_keys += p->num_keys();
            ncb_sum += p->num_cbases();
            if (p->num_cbases() == 0) ++pages_zero_ncb;
            if (p->prefix_width() < 5) pfxw_hist[p->prefix_width()]++;
            {
               const uint32_t cl = uint32_t(p->common_prefix().size());
               cpfx_len_sum += cl;
               const int cb = cl == 0 ? 0 : cl <= 3 ? cl : cl < 8 ? 4 : cl < 16 ? 5 : cl < 32 ? 6 : 7;
               cpfx_hist[cb]++;
            }
            const uint8_t* prev = nullptr; uint32_t prevlen = 0;
            for (uint32_t i = 0; i < p->num_keys(); ++i)
            {
               const auto* iv = p->record(i);
               payload += uint64_t(iv->suffix_len) + iv->value_len;
               if (prev)  // shared-prefix length with the previous (sorted) suffix
               {
                  uint32_t c = 0, m = std::min(prevlen, uint32_t(iv->suffix_len));
                  while (c < m && prev[c] == iv->data[c]) ++c;
                  adj_cpl_hist[c < 7 ? c : 7]++;
                  ++adj_pairs;
               }
               prev = iv->data; prevlen = iv->suffix_len;
            }
            if (page_sample && per_type[ty].count + 1 >= page_sample)
               stop_walk = true;  // enough pages sampled for a representative keys/page
         }
         if (rti::kind_of(hdr->type()) == rti::radix_kind::setlist_u8)
         {
            const auto* sl      = ses->get_ref<sal_art_ref::radix::setlist_u8>(a).obj();
            const bool  ht      = sl->has_term_value();
            const int   entries = sl->branch_count() + (ht ? 1 : 0);  // total entries (stems + term)
            const int   b = entries == 0 ? 0 : entries == 1 ? 1 : entries <= 4 ? 2 : entries <= 8 ? 3
                          : entries <= 16 ? 4 : entries <= 32 ? 5 : entries <= 64 ? 6 : 7;
            sl_fanout[b]++;
            if (entries == 1)  // a setlist holding ONE thing — should not be a router
            {
               bool tail = false;
               if (sl->branch_count() == 1)
                  sl->for_each_entry([&](uint8_t, auto br) { tail = br.is_tail(); });
               else
                  tail = sl->term_value().is_tail();  // the lone entry is the term value
               if (tail) sl1_inlineval++;  // single INLINE VALUE -> should be a suffix+value leaf
               else      sl1_childptr++;   // single CHILD/PAGE ptr -> should path-compress / inline
            }
            else if (entries == 0)
               sl1_term++;  // reuse counter: EMPTY setlist (no stems, no term) — pure bug
         }
         const bool compacted = (sz == compact);  // already at the 128-rounded tight size
         per_type[ty].count++;
         per_type[ty].bytes += sz;
         per_type[ty].compact += compact;
         per_type[ty].payload += payload;
         per_type[ty].n_compacted += compacted;
         total_compact += compact;
         total_payload += payload;
         total_compacted += compacted;
         size_hist[sz ? (63 - __builtin_clzll(sz)) : 0]++;

         int kids = 0;
         alloc.type_ops(hdr).visit_children(hdr, [&](sal::ptr_address c) {
            ++kids;
            stack.push_back({c, depth + 1});
         });
         if (kids == 0)
         {
            ++terminals;
            depth_hist[depth < 63 ? depth : 63]++;
         }
         if (stop_walk)
            break;
      }
      if (stop_walk)
         break;
   }

   // ---- results ----
   const int64_t leak_b = int64_t(allocated_live) - int64_t(total_bytes);
   const int64_t leak_n = int64_t(alloc.total_allocated_objects()) - int64_t(total_nodes);

   printf("\n== reachable tree (TRUE live) ==\n");
   printf("  reachable nodes      %8llu\n", (unsigned long long)total_nodes);
   printf("  reachable bytes      %8.2f GB\n", gb(total_bytes));
   printf("  terminals (leaves)   %8llu\n", (unsigned long long)terminals);

   printf("\n== LEAK ==\n");
   printf("  leaked nodes         %8lld   (%.1f%% of live CBs)\n", (long long)leak_n,
          alloc.total_allocated_objects() ? 100.0 * double(leak_n) / double(alloc.total_allocated_objects()) : 0);
   printf("  leaked bytes         %8.2f GB   (%.1f%% of allocated-live)\n", gb(leak_b > 0 ? leak_b : 0),
          allocated_live ? 100.0 * double(leak_b) / double(allocated_live) : 0);
   printf("  => segs %.1f GB = reachable %.1f GB + leaked %.1f GB + freed %.1f GB + tail\n",
          gb(segs_file), gb(total_bytes), gb(leak_b > 0 ? leak_b : 0), gb(free_bytes));

   printf("\n== reachable nodes by type — where the bytes go ==\n");
   printf("  %-14s %12s %9s %9s %7s %9s %9s\n", "type", "count", "alloc_GB", "payld_GB", "ovhd%",
          "slack_GB", "compact%");
   auto row = [&](const char* name, const Stat& s)
   {
      const uint64_t ovhd  = s.bytes - s.payload;       // header + padding + index + slack + rounding
      const uint64_t slack = s.bytes - s.compact;       // reclaimable by a compaction pass
      printf("  %-14s %12llu %9.2f %9.2f %6.0f%% %9.2f %7.0f%%\n", name,
             (unsigned long long)s.count, gb(s.bytes), gb(s.payload),
             100.0 * double(ovhd) / double(s.bytes ? s.bytes : 1), gb(slack),
             100.0 * double(s.n_compacted) / double(s.count ? s.count : 1));
   };
   for (int t = 0; t < 256; ++t)
      if (per_type[t].count)
         row(type_name(uint8_t(t)), per_type[t]);
   Stat tot{total_nodes, total_bytes, total_compact, total_payload, total_compacted};
   row("TOTAL", tot);

   {
      const uint64_t npages = per_type[uint8_t(sal_art_ref::leaf_page::type_id)].count;
      const uint64_t kvn    = per_type[uint8_t(sal_art_ref::key_val_node::type_id)].count;
      printf("\n  KEYS/PAGE: %llu keys across %llu leaf_pages = %.1f keys/page (%s)\n",
             (unsigned long long)total_page_keys, (unsigned long long)npages,
             npages ? double(total_page_keys) / double(npages) : 0.0,
             stop_walk ? "SAMPLED" : "full");
      printf("  KEY_VAL externals: %llu nodes (avg key %.0fB, val %.0fB) — vs %llu in-page page-keys\n",
             (unsigned long long)kvn,
             kvn ? double(kv_key_bytes) / double(kvn) : 0.0,
             kvn ? double(kv_value_bytes) / double(kvn) : 0.0,
             (unsigned long long)total_page_keys);
      // in-page suffix divergence: adjacent sorted suffixes' shared-prefix length
      printf("  SUFFIX DIVERGENCE (adjacent in-page keys, %llu pairs):\n", (unsigned long long)adj_pairs);
      const char* lbl[8] = {"differ @byte1 (0 shared)", "@byte2 (1 shared)", "@byte3 (2)",
                            "@byte4 (3)", "@byte5 (4)", "@byte6 (5)", "@byte7 (6)", ">=byte8 (7+)"};
      for (int c = 0; c < 8; ++c)
         if (adj_cpl_hist[c])
            printf("    %-26s %10llu  %5.1f%%\n", lbl[c], (unsigned long long)adj_cpl_hist[c],
                   adj_pairs ? 100.0 * double(adj_cpl_hist[c]) / double(adj_pairs) : 0.0);
      printf("  PAGE resolution width (_pfxw): 1B=%llu  2B=%llu  4B=%llu pages\n",
             (unsigned long long)pfxw_hist[1], (unsigned long long)pfxw_hist[2],
             (unsigned long long)pfxw_hist[4]);
      printf("  CBASES: avg %.2f used/page; %llu of %llu pages have ZERO externs "
             "(cbases[16]=64B dead → ~%.1f GB reclaimable if variable)\n",
             npages ? double(ncb_sum) / double(npages) : 0.0,
             (unsigned long long)pages_zero_ncb, (unsigned long long)npages,
             gb(double(pages_zero_ncb) * 64.0));
      printf("  CPFX (factored common prefix) avg %.1f B/page; buckets 0:%llu 1:%llu 2:%llu 3:%llu "
             "4-7:%llu 8-15:%llu 16-31:%llu 32+:%llu\n",
             npages ? double(cpfx_len_sum) / double(npages) : 0.0,
             (unsigned long long)cpfx_hist[0], (unsigned long long)cpfx_hist[1],
             (unsigned long long)cpfx_hist[2], (unsigned long long)cpfx_hist[3],
             (unsigned long long)cpfx_hist[4], (unsigned long long)cpfx_hist[5],
             (unsigned long long)cpfx_hist[6], (unsigned long long)cpfx_hist[7]);
      {
         double avg_sfx_cpl = 0;
         for (int c = 0; c < 8; ++c) avg_sfx_cpl += double(c) * double(adj_cpl_hist[c]);
         avg_sfx_cpl = adj_pairs ? avg_sfx_cpl / double(adj_pairs) : 0.0;
         printf("  => adjacent FULL keys share ~%.1f B (cpfx %.1f + suffix-overlap >=%.1f); "
                "cpfx captures %.0f%% of the shared run\n",
                (npages ? double(cpfx_len_sum) / double(npages) : 0.0) + avg_sfx_cpl,
                npages ? double(cpfx_len_sum) / double(npages) : 0.0, avg_sfx_cpl,
                ((npages ? double(cpfx_len_sum) / double(npages) : 0.0) + avg_sfx_cpl) > 0
                    ? 100.0 * (npages ? double(cpfx_len_sum) / double(npages) : 0.0) /
                          ((npages ? double(cpfx_len_sum) / double(npages) : 0.0) + avg_sfx_cpl)
                    : 0.0);
      }
   }

   const uint64_t total_ovhd  = total_bytes - total_payload;
   const uint64_t total_slack = total_bytes - total_compact;
   printf("\n  payload  = actual user key+value bytes (routers/buckets carry none).\n");
   printf("  overhead = alloc - payload = struct headers + 128B skip-header headroom +\n"
          "             router pointers + leaf-page columns + slack + rounding.\n");
   printf("  slack    = alloc - compact_size() = reclaimable by a full compaction pass\n"
          "             (does NOT include the headroom compact_size KEEPS, nor the %.2f GB already freed).\n",
          gb(free_bytes));
   printf("\n  EFFICIENCY: %.2f GB user data in %.2f GB allocated = %.1f%% payload, %.1f%% overhead.\n",
          gb(total_payload), gb(total_bytes),
          100.0 * double(total_payload) / double(total_bytes ? total_bytes : 1),
          100.0 * double(total_ovhd) / double(total_bytes ? total_bytes : 1));
   printf("  COMPACTED: %llu / %llu nodes (%.1f%%) already at their 128-rounded tight size;\n"
          "             %.2f GB still reclaimable as slack.\n",
          (unsigned long long)total_compacted, (unsigned long long)total_nodes,
          100.0 * double(total_compacted) / double(total_nodes ? total_nodes : 1), gb(total_slack));
   printf("\n  KEY_VAL split: %.2f GB FULL-keys + %.2f GB values  (avg key %.0f B, value %.0f B/node)\n",
          gb(kv_key_bytes), gb(kv_value_bytes),
          double(kv_key_bytes) / double(per_type[uint8_t(sal_art_ref::key_val_node::type_id)].count
                                            ? per_type[uint8_t(sal_art_ref::key_val_node::type_id)].count : 1),
          double(kv_value_bytes) / double(per_type[uint8_t(sal_art_ref::key_val_node::type_id)].count
                                            ? per_type[uint8_t(sal_art_ref::key_val_node::type_id)].count : 1));
   printf("  => the FULL-key bytes are the prefix-duplication ceiling: storing suffix-relative\n"
          "     could save at most %.2f GB (the part of each key already on the radix path+cpfx).\n",
          gb(kv_key_bytes));

   printf("\n== node size histogram (log2) ==\n");
   for (int i = 0; i < 40; ++i)
      if (size_hist[i])
         printf("  [%6u..%6u) %12llu\n", 1u << i, 1u << (i + 1),
                (unsigned long long)size_hist[i]);

   printf("\n== setlist_u8 fanout (branches per router — inline-children candidates) ==\n");
   {
      const char* lbl[8] = {"0(empty)", "1", "2-4", "5-8", "9-16", "17-32", "33-64", "65+"};
      uint64_t    tot = 0;
      for (int i = 0; i < 8; ++i) tot += sl_fanout[i];
      printf("  %-9s %12s\n", "entries", "setlists  (stems + term value)");
      for (int i = 0; i < 8; ++i)
         if (sl_fanout[i])
            printf("  %-9s %12llu  (%.1f%%)\n", lbl[i], (unsigned long long)sl_fanout[i],
                   100.0 * double(sl_fanout[i]) / double(tot ? tot : 1));
      printf("  setlist_u8 total %llu\n", (unsigned long long)tot);
      printf("  1-ENTRY setlists (Dan's bug — should be a leaf/path-compressed): %llu inline-value "
             "(-> suffix+value leaf) | %llu single-ptr (-> path-compress) ;  %llu empty(0-entry)\n",
             (unsigned long long)sl1_inlineval, (unsigned long long)sl1_childptr,
             (unsigned long long)sl1_term);
   }

   printf("\n== terminal depth histogram (node hops to a leaf-of-DAG) ==\n");
   for (int i = 0; i < 64; ++i)
      if (depth_hist[i])
         printf("  depth %2d  %12llu\n", i, (unsigned long long)depth_hist[i]);

   fflush(stdout);
   _exit(0);  // skip read-only teardown (it tries to truncate → throws)
}

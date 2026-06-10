// Tree-structure diagnostic for the heart radix engine.
//
// Loads a dictionary, inserts every word into heart::radix::radix_tree, then walks
// the tree and emits:
//   * an indented TEXT representation (heart_tree.txt) — one line per node/terminal,
//     inline (tail) values shown in place, child pointer lines annotated with the
//     SAL control-block cacheline they land in (so hint co-location is visible), and
//   * aggregate STATS to stdout: per-tier counts, node/alloc/heap bytes, depth and
//     fan-out histograms, inline-vs-leaf terminals, control-block co-location, and
//     the COW/alloc accounting (clones/grows/in-place) for the chosen commit cadence.
//
// Usage: dict-tree-dump [dict_path] [max_words] [commit_every] [out_prefix]
//   max_words=0   → all words   commit_every=0 → no commits during insert
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <sal/allocator.hpp>
#include <sal/allocator_session_impl.hpp>

#include <heart/inline_value.hpp>
#include <heart/leaf_node.hpp>
#include <heart/radix_nodes.hpp>

#include <radix_engine.hpp>



using heart::bitset_r2_node;
using heart::bitset_r4_node;
using heart::full_radix_node;
using heart::inline_value;
using heart::leaf_node;
using heart::smart_address;
using sal::ptr_address;
using setlist_u8 = heart::setlist_node<uint8_t>;

namespace
{
   std::vector<std::string> load_dictionary(const char* path)
   {
      std::vector<std::string> words;
      if (std::strncmp(path, "rand:", 5) == 0)  // rand:N -> N distinct random 8-byte u64 keys
      {
         size_t n = std::strtoul(path + 5, nullptr, 10);
         words.reserve(n);
         for (size_t i = 0; i < n; ++i)
         {
            uint64_t k = uint64_t(i) * 0x9E3779B97F4A7C15ull;
            words.emplace_back(reinterpret_cast<const char*>(&k), 8);
         }
         return words;
      }
      std::ifstream            f(path);
      std::string              line;
      while (std::getline(f, line))
         if (!line.empty())
            words.push_back(std::move(line));
      return words;
   }

   std::string escape(const uint8_t* p, size_t n)
   {
      std::string s;
      for (size_t i = 0; i < n; ++i)
      {
         uint8_t c = p[i];
         if (c >= 0x20 && c < 0x7f && c != '\\')
            s.push_back(char(c));
         else
         {
            char b[8];
            std::snprintf(b, sizeof(b), "\\x%02x", c);
            s += b;
         }
      }
      return s;
   }

   // SAL packs 8 control_blocks per 64-byte cacheline, indexed flat by ptr_address.
   uint64_t cacheline_of(ptr_address a) { return uint64_t(*a) >> 3; }

   struct Stats
   {
      uint64_t nodes = 0;
      // [0]=setlist [1]=r2 [2]=r4 [3]=full [4]=setlist_w16 [5]=setlist_w32
      static constexpr int NTIER = 6;
      uint64_t tier_nodes[NTIER] = {};
      uint64_t tier_alloc[NTIER] = {};
      uint64_t tier_heap[NTIER]  = {};  // tail_used bytes
      uint64_t tier_free[NTIER]  = {};  // free_heap bytes
      uint64_t leaf_holders  = 0;   // >60 SAL leaf_node value holders
      uint64_t leaf_bytes    = 0;
      uint64_t inline_terms  = 0;   // terminal values stored inline (tail)
      uint64_t leaf_terms    = 0;   // terminal values stored in a SAL leaf
      uint64_t keys          = 0;
      uint64_t key_depth_sum = 0;
      uint64_t depth_max     = 0;
      std::map<int, uint64_t> depth_hist;   // key terminals by descent depth (node hops)
      std::map<int, uint64_t> fanout_hist;  // routers by child count
      uint64_t                pfx_nodes = 0, pfx_bytes = 0;  // path-compression
      std::map<int, uint64_t> pfx_hist;     // node prefix length -> count
      // control-block co-location of sibling child nodes
      uint64_t colo_children   = 0;
      uint64_t colo_cachelines = 0;
      // compact_addresses utilization (one instance per bitset SEGMENT)
      uint64_t                seg_total     = 0;  // # compact_addresses instances seen
      uint64_t                seg_gt16      = 0;  // branches > 16  (max_bases — beyond a 4-bit base view)
      uint64_t                seg_gt_rawcap = 0;  // branches > data_size/4 (a raw ptr array wouldn't fit)
      uint64_t                seg_branch_sum = 0;
      uint64_t                seg_max       = 0;
      std::map<int, uint64_t> seg_size_hist;   // segment branch count -> count
      std::map<int, uint64_t> seg_bases_hist;  // live cacheline bases -> count
   };

   class Dumper
   {
     public:
      Dumper(sal::allocator_session_ptr ses, std::ostream& out, Stats& st)
          : _ses(ses), _out(out), _st(st)
      {
      }

      void run(ptr_address root) { walk(root, 0, 1); }

     private:
      sal::allocator_session_ptr _ses;
      std::ostream&              _out;
      Stats&                     _st;

      void indent(int d)
      {
         for (int i = 0; i < d; ++i)
            _out << "  ";
      }

      const inline_value* inline_of(const sal::alloc_header* node, smart_address sa)
      {
         return reinterpret_cast<const inline_value*>(reinterpret_cast<const char*>(node) +
                                                      node->size() - sa.tail_offset());
      }

      // Render a terminal value (inline tail OR SAL leaf) and account it. `hops` is the
      // node count from root to the holder node; a leaf adds one more node hop.
      void show_terminal(const sal::alloc_header* node, smart_address sa, int hops)
      {
         const inline_value* iv;
         int                 eff = hops;
         if (sa.is_tail())
         {
            iv = inline_of(node, sa);
            ++_st.inline_terms;
            _out << " = \"" << escape(iv->data, iv->suffix_len) << "\" :" << int(iv->value_len)
                 << "B (inline)";
         }
         else
         {
            auto ref = _ses->get_ref<leaf_node>(sa.as_pointer());
            iv       = &ref.obj()->value;
            eff      = hops + 1;
            ++_st.leaf_terms;
            ++_st.leaf_holders;
            _st.leaf_bytes += ref.obj()->size();
            _out << " -> leaf@cl" << cacheline_of(sa.as_pointer()) << " \""
                 << escape(iv->data, iv->suffix_len) << "\" :" << int(iv->value_len) << "B";
         }
         ++_st.keys;
         _st.key_depth_sum += uint64_t(eff);
         _st.depth_max = std::max<uint64_t>(_st.depth_max, uint64_t(eff));
         ++_st.depth_hist[eff];
      }

      template <class N>
      void account_node(const N* n, int tier)
      {
         ++_st.nodes;
         ++_st.tier_nodes[tier];
         _st.tier_alloc[tier] += n->size();
         _st.tier_heap[tier] += n->tail_used();
         _st.tier_free[tier] += n->free_heap();
      }

      template <class N>
      void walk_router(const N* n, ptr_address addr, int tier, int depth, int hops)
      {
         account_node(n, tier);
         auto pfx = n->prefix();
         ++_st.pfx_hist[int(pfx.size())];
         if (pfx.size())
         {
            ++_st.pfx_nodes;
            _st.pfx_bytes += pfx.size();
         }
         indent(depth);
         _out << tier_name(tier) << " #" << *addr << " @cl" << cacheline_of(addr) << " pfx=\""
              << escape(pfx.data(), pfx.size()) << "\" heap=" << n->tail_used() << "/"
              << (n->tail_used() + n->free_heap()) << "B size=" << n->size() << "B";

         // co-location: distinct control-block cachelines among child POINTERS
         uint64_t      cls[1100];
         int           ncl = 0, branches = 0;
         smart_address tv = n->term_value();
         if (tv.is_pointer())
            cls[ncl++] = cacheline_of(tv.as_pointer());
         n->for_each_entry([&](uint8_t, smart_address br) {
            ++branches;
            if (br.is_pointer())
               cls[ncl++] = cacheline_of(br.as_pointer());
         });
         std::sort(cls, cls + ncl);
         int distinct = 0;
         for (int i = 0; i < ncl; ++i)
            if (i == 0 || cls[i] != cls[i - 1])
               ++distinct;
         ++_st.fanout_hist[branches];
         if (ncl > 0)
         {
            _st.colo_children += uint64_t(ncl);
            _st.colo_cachelines += uint64_t(distinct);
         }
         _out << "  (" << branches << " br";
         if (ncl)
            _out << ", " << ncl << " kids/" << distinct << " cl";
         _out << ")\n";

         // terminal at this node (key ends exactly here)
         if (!tv.is_null())
         {
            indent(depth + 1);
            _out << "[$]";
            show_terminal(n, tv, hops);
            _out << "\n";
         }
         // branches, in byte order
         struct Ent
         {
            uint8_t       b;
            smart_address sa;
         };
         std::vector<Ent> ents;
         n->for_each_entry([&](uint8_t b, smart_address sa) { ents.push_back({b, sa}); });
         std::sort(ents.begin(), ents.end(), [](const Ent& a, const Ent& b) { return a.b < b.b; });
         for (auto& e : ents)
         {
            indent(depth + 1);
            _out << "[" << escape(&e.b, 1) << "]";
            bool is_node = e.sa.is_pointer() &&
                           _ses->get_ref<sal::alloc_header>(e.sa.as_pointer())->type() !=
                               leaf_node::type_id;
            if (is_node)
            {
               _out << " ->@cl" << cacheline_of(e.sa.as_pointer()) << "\n";
               walk(e.sa.as_pointer(), depth + 2, hops + 1);
            }
            else  // inline tail OR a >60 leaf holder — both terminals
            {
               show_terminal(n, e.sa, hops);
               _out << "\n";
            }
         }
      }

      // Per-segment compact_addresses accounting (one instance per bitset segment).
      template <class Seg>
      void account_seg(const Seg& seg)
      {
         const int sz     = seg.size();
         const int rawcap = Seg::branches_type::data_size / 4;  // raw 4B-ptr slots that would fit
         ++_st.seg_total;
         _st.seg_gt16 += (sz > 16);
         _st.seg_gt_rawcap += (sz > rawcap);
         _st.seg_branch_sum += uint64_t(sz);
         _st.seg_max = std::max<uint64_t>(_st.seg_max, uint64_t(sz));
         ++_st.seg_size_hist[sz];
         ++_st.seg_bases_hist[seg.branches.live_base_count()];
      }

      void walk(ptr_address addr, int depth, int hops)
      {
         auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
         if (t == setlist_u8::type_id)
            walk_router(_ses->get_ref<setlist_u8>(addr).obj(), addr, 0, depth, hops);
         else if (t == bitset_r2_node::type_id)
         {
            auto* n = _ses->get_ref<bitset_r2_node>(addr).obj();
            for (int s = 0; s < bitset_r2_node::nseg; ++s)
               account_seg(n->seg(s));
            walk_router(n, addr, 1, depth, hops);
         }
         else if (t == bitset_r4_node::type_id)
         {
            auto* n = _ses->get_ref<bitset_r4_node>(addr).obj();
            for (int s = 0; s < bitset_r4_node::nseg; ++s)
               account_seg(n->seg(s));
            walk_router(n, addr, 2, depth, hops);
         }
         else if (t == full_radix_node::type_id)
            walk_router(_ses->get_ref<full_radix_node>(addr).obj(), addr, 3, depth, hops);
         else if (t == heart::setlist_node<uint16_t>::type_id)
            walk_router(_ses->get_ref<heart::setlist_node<uint16_t>>(addr).obj(), addr, 4, depth,
                        hops);
         else if (t == heart::setlist_node<uint32_t>::type_id)
            walk_router(_ses->get_ref<heart::setlist_node<uint32_t>>(addr).obj(), addr, 5, depth,
                        hops);
         else if (t == leaf_node::type_id)
         {
            // a leaf reached as the root (degenerate); render as a key
            indent(depth);
            _out << "[$]";
            show_terminal(nullptr, smart_address::from_pointer(addr), hops);
            _out << "\n";
         }
      }

      static const char* tier_name(int tier)
      {
         static const char* n[6] = {"setlist",     "bitset_r2",   "bitset_r4",
                                    "full",        "setlist_w16", "setlist_w32"};
         return n[tier];
      }
   };
}  // namespace

int main(int argc, char** argv)
{
   const char* dict        = (argc > 1) ? argv[1] : "/usr/share/dict/words";
   size_t      max_words   = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 0;
   size_t      commit_every = (argc > 3) ? std::strtoul(argv[3], nullptr, 10) : 0;
   std::string prefix      = (argc > 4) ? argv[4] : "heart_tree";

   auto words = load_dictionary(dict);
   if (words.size() < 100)
   {
      std::fprintf(stderr, "could not load dictionary at %s\n", dict);
      return 2;
   }
   if (max_words && words.size() > max_words)
      words.resize(max_words);
   const size_t N = words.size();

   auto                     dir = std::filesystem::temp_directory_path() / "heart_tree_dump";
   heart::radix::radix_tree t(dir);
   sal::smart_ptr<sal::alloc_header> snap;  // a snapshot IS a smart_ptr copy
   auto vview = [](const uint64_t& v)
   { return std::string_view(reinterpret_cast<const char*>(&v), 8); };

   if (std::getenv("DICT_NUL"))  // NUL-terminate keys to match libart's key set exactly
      for (auto& w : words)
         w.push_back('\0');

   if (std::getenv("WIDE"))  // enable gated wide-stem (setlist_u16/u24/u32) nodes
      t.set_wide_stems(true);

   uint64_t commits = 0;
   for (size_t i = 0; i < N; ++i)
   {
      uint64_t v = i;
      t.upsert(std::string_view(words[i]), vview(v));
      if (commit_every && (i + 1) % commit_every == 0)
      {
         snap = t.root_snapshot();
         ++commits;
      }
   }

   // verify
   uint64_t found = 0;
   for (size_t i = 0; i < N; ++i)
   {
      std::string_view got;
      if (t.find(std::string_view(words[i]), got))
         ++found;
   }

   std::string  treefile = prefix + ".txt";
   std::ofstream out(treefile);
   Stats         st;
   Dumper        d(t.session(), out, st);
   d.run(t.root_addr());
   out.close();

   auto pct = [](uint64_t a, uint64_t b) { return b ? 100.0 * double(a) / double(b) : 0.0; };

   std::printf("dictionary: %zu words from %s   (verify found %llu/%zu)\n", N, dict,
               (unsigned long long)found, N);
   std::printf("commit_every=%zu -> %llu commits\n\n", commit_every, (unsigned long long)commits);

   std::printf("== TREE SHAPE ==\n");
   std::printf("  keys              : %llu\n", (unsigned long long)st.keys);
   std::printf(
       "  router nodes      : %llu  (setlist %llu, r2 %llu, r4 %llu, full %llu, w16 %llu, w32 "
       "%llu)\n",
       (unsigned long long)st.nodes, (unsigned long long)st.tier_nodes[0],
       (unsigned long long)st.tier_nodes[1], (unsigned long long)st.tier_nodes[2],
       (unsigned long long)st.tier_nodes[3], (unsigned long long)st.tier_nodes[4],
       (unsigned long long)st.tier_nodes[5]);
   std::printf("  leaf value holders: %llu  (>60B values)\n", (unsigned long long)st.leaf_holders);
   std::printf("  depth (node hops) : avg %.2f   max %llu\n",
               st.keys ? double(st.key_depth_sum) / double(st.keys) : 0.0,
               (unsigned long long)st.depth_max);
   std::printf("  terminals         : %llu inline (%.1f%%), %llu leaf (%.1f%%)\n",
               (unsigned long long)st.inline_terms, pct(st.inline_terms, st.keys),
               (unsigned long long)st.leaf_terms, pct(st.leaf_terms, st.keys));

   std::printf("\n== MEMORY ==\n");
   uint64_t total_alloc = st.leaf_bytes;
   uint64_t total_heap = 0, total_cap = 0;
   const char* tn[6] = {"setlist  ", "bitset_r2", "bitset_r4",
                        "full     ", "setlst_w16", "setlst_w32"};
   for (int i = 0; i < Stats::NTIER; ++i)
   {
      total_alloc += st.tier_alloc[i];
      total_heap += st.tier_heap[i];
      total_cap += st.tier_heap[i] + st.tier_free[i];
      if (st.tier_nodes[i])
         std::printf("  %s: %6llu nodes  %8llu KB alloc  heap %llu/%llu KB (%.0f%% full)\n", tn[i],
                     (unsigned long long)st.tier_nodes[i],
                     (unsigned long long)(st.tier_alloc[i] >> 10),
                     (unsigned long long)(st.tier_heap[i] >> 10),
                     (unsigned long long)((st.tier_heap[i] + st.tier_free[i]) >> 10),
                     pct(st.tier_heap[i], st.tier_heap[i] + st.tier_free[i]));
   }
   std::printf("  leaf     : %6llu nodes  %8llu KB alloc\n", (unsigned long long)st.leaf_holders,
               (unsigned long long)(st.leaf_bytes >> 10));
   std::printf("  TOTAL    : %llu KB allocated   tail-heap %llu/%llu KB (%.0f%% full)\n",
               (unsigned long long)(total_alloc >> 10), (unsigned long long)(total_heap >> 10),
               (unsigned long long)(total_cap >> 10), pct(total_heap, total_cap));
   std::printf("  bytes/key: %.1f\n", st.keys ? double(total_alloc) / double(st.keys) : 0.0);

   std::printf("\n== CONTROL-BLOCK CO-LOCATION (sibling child nodes) ==\n");
   std::printf("  %llu child ptrs land in %llu distinct cachelines  -> %.2f kids/cacheline\n",
               (unsigned long long)st.colo_children, (unsigned long long)st.colo_cachelines,
               st.colo_cachelines ? double(st.colo_children) / double(st.colo_cachelines) : 0.0);

   std::printf("\n== COMMON-PREFIX (path compression) ==\n");
   std::printf("  router nodes with a prefix : %llu / %llu  (%.1f%%)\n",
               (unsigned long long)st.pfx_nodes, (unsigned long long)st.nodes,
               st.nodes ? 100.0 * double(st.pfx_nodes) / double(st.nodes) : 0.0);
   std::printf("  total compressed bytes     : %llu  (avg %.2f / prefixed node)\n",
               (unsigned long long)st.pfx_bytes,
               st.pfx_nodes ? double(st.pfx_bytes) / double(st.pfx_nodes) : 0.0);
   std::printf("  PREFIX-LENGTH HISTOGRAM:\n");
   for (auto& [k, v] : st.pfx_hist)
      std::printf("    len %2d : %llu nodes\n", k, (unsigned long long)v);

   std::printf("\n== FAN-OUT (router child count) ==\n");
   {
      uint64_t nodes_gt16 = 0, nodes_gt27 = 0, ntot = 0;
      for (auto& [k, v] : st.fanout_hist)
      {
         ntot += v;
         if (k > 16)
            nodes_gt16 += v;
         if (k > 27)
            nodes_gt27 += v;
      }
      std::printf("  ALL router nodes: %llu;  fanout >16 : %llu (%.3f%%);  fanout >27 : %llu (%.3f%%)\n",
                  (unsigned long long)ntot, (unsigned long long)nodes_gt16, pct(nodes_gt16, ntot),
                  (unsigned long long)nodes_gt27, pct(nodes_gt27, ntot));
   }
   for (auto& [k, v] : st.fanout_hist)
      std::printf("  %3d branches : %llu nodes\n", k, (unsigned long long)v);

   std::printf("\n== COMPACT_ADDRESSES UTILIZATION (one instance per bitset segment) ==\n");
   std::printf("  segments seen        : %llu  (r2=110B/27slots, r4=118B/29slots per segment)\n",
               (unsigned long long)st.seg_total);
   std::printf("  mean branches/seg    : %.2f   max %llu\n",
               st.seg_total ? double(st.seg_branch_sum) / double(st.seg_total) : 0.0,
               (unsigned long long)st.seg_max);
   std::printf("  segments w/ >16 br   : %llu  (%.3f%%)  <- where a 4-bit base view earns capacity\n",
               (unsigned long long)st.seg_gt16, pct(st.seg_gt16, st.seg_total));
   std::printf("  segments w/ >cap br  : %llu  (%.3f%%)  <- where a RAW ptr array would NOT fit\n",
               (unsigned long long)st.seg_gt_rawcap, pct(st.seg_gt_rawcap, st.seg_total));
   std::printf("  SEGMENT BRANCH-COUNT HISTOGRAM:\n");
   for (auto& [k, v] : st.seg_size_hist)
      std::printf("    %3d br : %llu segs\n", k, (unsigned long long)v);
   std::printf("  LIVE-BASES HISTOGRAM (distinct cachelines / 16 max):\n");
   for (auto& [k, v] : st.seg_bases_hist)
      std::printf("    %2d bases : %llu segs\n", k, (unsigned long long)v);

   std::printf("\n== DEPTH HISTOGRAM (keys by node hops) ==\n");
   for (auto& [k, v] : st.depth_hist)
      std::printf("  hop %2d : %llu keys\n", k, (unsigned long long)v);

   auto& cc = t.counters();
   std::printf("\n== COW / ALLOC ACCOUNTING ==\n");
   std::printf("  node allocs   : %llu\n", (unsigned long long)cc.node_allocs);
   std::printf("  leaf allocs   : %llu\n", (unsigned long long)cc.leaf_allocs);
   std::printf("  inline values : %llu\n", (unsigned long long)cc.inline_values);
   std::printf("  in-place muts : %llu\n", (unsigned long long)cc.in_place);
   std::printf("  clones (COW)  : %llu   (of which %llu heap-full grows)\n",
               (unsigned long long)cc.clones, (unsigned long long)cc.grows);

   std::printf("\nwrote indented tree -> %s\n", treefile.c_str());
   return 0;
}

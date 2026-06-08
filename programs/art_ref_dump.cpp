// Reference structure dump for libart (the canonical Adaptive Radix Tree), so the
// heart stem engine can be validated to produce the SAME path-compressed prefixes and
// the SAME depth (only the node layout / dispatch should differ).
//
// Reports: key count, internal-node count + type histogram, per-key depth (internal
// nodes traversed to the leaf) avg/max/histogram, and an indented prefix tree for a
// small prefix of the input. Usage: art-ref-dump [dict] [max_words] [out_prefix]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

extern "C" {
#include "art.h"
}

// libart internals we need to walk the tree (mirror of art.c's tagging).
#define IS_LEAF(x) (((uintptr_t)(x)) & 1)
#define LEAF_RAW(x) ((art_leaf*)((void*)((uintptr_t)(x) & ~1)))

namespace
{
   std::vector<std::string> load_dictionary(const char* path)
   {
      std::vector<std::string> w;
      if (std::strncmp(path, "rand:", 5) == 0)  // rand:N -> N distinct random 8-byte u64 keys
      {
         size_t n = std::strtoul(path + 5, nullptr, 10);
         w.reserve(n);
         for (size_t i = 0; i < n; ++i)
         {
            uint64_t k = uint64_t(i) * 0x9E3779B97F4A7C15ull;
            w.emplace_back(reinterpret_cast<const char*>(&k), 8);
         }
         return w;
      }
      std::ifstream f(path);
      std::string   line;
      while (std::getline(f, line))
         if (!line.empty())
            w.push_back(std::move(line));
      return w;
   }
   std::string escape(const unsigned char* p, int n)
   {
      std::string s;
      for (int i = 0; i < n; ++i)
      {
         unsigned char c = p[i];
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

   struct Stats
   {
      uint64_t                internal = 0, leaves = 0;
      uint64_t                by_type[5] = {};  // 1..4 = NODE4/16/48/256
      uint64_t                depth_sum = 0, depth_max = 0;
      std::map<int, uint64_t> depth_hist;
      // path-compression: common-prefix (partial) length per internal node
      uint64_t                pfx_nodes = 0;   // internal nodes with partial_len > 0
      uint64_t                pfx_bytes = 0;   // total compressed bytes
      std::map<int, uint64_t> pfx_hist;        // partial_len -> count
   };

   // Visit each child pointer of an internal node (handles all four layouts).
   template <class F>
   void each_child(art_node* n, F&& f)
   {
      switch (n->type)
      {
         case NODE4:
         {
            auto* p = (art_node4*)n;
            for (int i = 0; i < n->num_children; ++i)
               f(p->children[i]);
            break;
         }
         case NODE16:
         {
            auto* p = (art_node16*)n;
            for (int i = 0; i < n->num_children; ++i)
               f(p->children[i]);
            break;
         }
         case NODE48:
         {
            auto* p = (art_node48*)n;
            for (int c = 0; c < 256; ++c)
               if (p->keys[c])
                  f(p->children[p->keys[c] - 1]);
            break;
         }
         case NODE256:
         {
            auto* p = (art_node256*)n;
            for (int c = 0; c < 256; ++c)
               if (p->children[c])
                  f(p->children[c]);
            break;
         }
      }
   }

   void walk(art_node* n, int depth, Stats& s, std::ostream* out, int indent)
   {
      if (!n)
         return;
      if (IS_LEAF(n))
      {
         ++s.leaves;
         s.depth_sum += uint64_t(depth);
         s.depth_max = std::max<uint64_t>(s.depth_max, uint64_t(depth));
         ++s.depth_hist[depth];
         if (out)
         {
            art_leaf* l = LEAF_RAW(n);
            for (int i = 0; i < indent; ++i)
               *out << "  ";
            *out << "leaf \"" << escape(l->key, int(l->key_len)) << "\"\n";
         }
         return;
      }
      ++s.internal;
      ++s.by_type[n->type];
      ++s.pfx_hist[int(n->partial_len)];
      if (n->partial_len)
      {
         ++s.pfx_nodes;
         s.pfx_bytes += n->partial_len;
      }
      if (out)
      {
         for (int i = 0; i < indent; ++i)
            *out << "  ";
         const char* tn[5] = {"?", "N4", "N16", "N48", "N256"};
         int         stored = std::min<int>(n->partial_len, MAX_PREFIX_LEN);
         *out << tn[n->type] << " pfx=\"" << escape(n->partial, stored) << "\""
              << (int(n->partial_len) > stored ? "(+opt)" : "") << " plen=" << n->partial_len
              << " nchild=" << int(n->num_children) << "\n";
      }
      each_child(n, [&](art_node* c) { walk(c, depth + 1, s, out, indent + 1); });
   }
}  // namespace

int main(int argc, char** argv)
{
   const char* dict      = (argc > 1) ? argv[1] : "/usr/share/dict/words";
   size_t      max_words = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 0;
   std::string prefix    = (argc > 3) ? argv[3] : "art_tree";

   auto words = load_dictionary(dict);
   if (words.size() < 100)
   {
      std::fprintf(stderr, "could not load dictionary at %s\n", dict);
      return 2;
   }
   if (max_words && words.size() > max_words)
      words.resize(max_words);
   const size_t N = words.size();

   art_tree t;
   art_tree_init(&t);
   for (size_t i = 0; i < N; ++i)
      art_insert(&t, (const unsigned char*)words[i].data(), int(words[i].size()) + 1,
                 (void*)(i + 1));  // +1 includes the NUL so no key is a prefix of another

   // optional indented tree for a browseable subset
   std::ofstream out(prefix + ".txt");
   Stats         s;
   walk(t.root, 0, s, &out, 0);
   out.close();

   auto pct = [](uint64_t a, uint64_t b) { return b ? 100.0 * double(a) / double(b) : 0.0; };
   std::printf("libart: %zu keys from %s\n\n", N, dict);
   std::printf("== TREE SHAPE ==\n");
   std::printf("  leaves (keys)     : %llu\n", (unsigned long long)s.leaves);
   std::printf("  internal nodes    : %llu  (N4 %llu, N16 %llu, N48 %llu, N256 %llu)\n",
               (unsigned long long)s.internal, (unsigned long long)s.by_type[1],
               (unsigned long long)s.by_type[2], (unsigned long long)s.by_type[3],
               (unsigned long long)s.by_type[4]);
   std::printf("  depth (internal nodes to leaf): avg %.2f   max %llu\n",
               s.leaves ? double(s.depth_sum) / double(s.leaves) : 0.0,
               (unsigned long long)s.depth_max);
   std::printf("  nodes/key         : %.3f   (internal nodes per key)\n",
               s.leaves ? double(s.internal) / double(s.leaves) : 0.0);
   (void)pct;
   std::printf("\n== COMMON-PREFIX (path compression) ==\n");
   std::printf("  internal nodes with a prefix : %llu / %llu  (%.1f%%)\n",
               (unsigned long long)s.pfx_nodes, (unsigned long long)s.internal,
               s.internal ? 100.0 * double(s.pfx_nodes) / double(s.internal) : 0.0);
   std::printf("  total compressed bytes       : %llu  (avg %.2f / prefixed node)\n",
               (unsigned long long)s.pfx_bytes,
               s.pfx_nodes ? double(s.pfx_bytes) / double(s.pfx_nodes) : 0.0);
   std::printf("  PREFIX-LENGTH HISTOGRAM (internal nodes by partial_len):\n");
   for (auto& [k, v] : s.pfx_hist)
      std::printf("    len %2d : %llu nodes\n", k, (unsigned long long)v);

   std::printf("\n== DEPTH HISTOGRAM (keys by internal-node depth) ==\n");
   for (auto& [k, v] : s.depth_hist)
      std::printf("  depth %2d : %llu keys\n", k, (unsigned long long)v);
   std::printf("\nwrote indented prefix tree -> %s.txt\n", prefix.c_str());
   return 0;
}

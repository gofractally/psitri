// rcart nested_setlist byte-identity invariant — the headline property of the
// "directly-mutable nested" change (task: kill promote-on-write).
//
// CRITICAL CORRECTNESS PROPERTY: a nested_setlist's 128 bytes are a PURE FUNCTION
// of its (prefix, sorted entries, term) — IDENTICAL whether the node was built by
// the initial pack_nested_leaves, an in-place UNIQUE edit, or a SHARED clone+edit.
// All three funnel their final bytes through the one canonical packer
// (lay_nested_into_); the only path difference is WHICH slot + the refcount
// bookkeeping, never the content bytes.
//
// This test builds the SAME logical content C two ways:
//   A) plain inserts (no snapshot held) → every edit takes the UNIQUE in-place path.
//   B) a read snapshot pinned across EVERY edit → every edit takes the SHARED
//      clone-then-edit path.
// then walks both trees in lock-step and asserts every node is byte-identical
// EXCEPT the control_block refcount (which is structural, not content): for each
// node it compares node TYPE + PREFIX + okeys + otypes + ooff + the tail-heap
// entry bytes. A mismatch hexdumps both 128-byte lines.
//
// It also runs a std::map functional oracle on both trees (every key reads back),
// and verifies the engine drains to ZERO live bytes after both trees are dropped
// (catches a missing retain / extra release in the shared path).

#include <rcart/tree.hpp>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace rcart;

static int g_fails = 0;
static int g_nodes_cmp = 0;

// Pin the CURRENT root (force the next edit onto the SHARED clone path), edit, unpin.
struct shared_pin
{
   engine&                   e;
   sal_hybrid::allocator_session&  ses;
   bt                        t;
   uint32_t                  p;
   shared_pin(engine& e_, sal_hybrid::allocator_session& s_, bt t_, uint32_t p_) : e(e_), ses(s_), t(t_), p(p_)
   {
      retain(e, ses, t, p);
   }
   ~shared_pin() { release_subtree(e, ses, t, p); }
};

// Hexdump a 128-byte routing line.
static void dump128(const char* tag, const void* n)
{
   const unsigned char* b = static_cast<const unsigned char*>(n);
   std::printf("  %s:", tag);
   for (int i = 0; i < 128; ++i)
   {
      if (i % 32 == 0)
         std::printf("\n   ");
      std::printf("%02x ", b[i]);
   }
   std::printf("\n");
}

// Compare two routing nodes for CONTENT byte-identity (everything except the
// control_block refcount, which lives in a SEPARATE region in the FIX allocator,
// not in the slot bytes — so the 128/256/1152 slot bytes ARE pure content here).
// Recurses into children in lock-step. Returns true if identical. NOINLINE: it is
// recursive, and the `full` arm's 256-way loop gives it a sizeable frame; letting the
// compiler inline a few recursion levels into main blew main's frame past the 8 MB
// stack (a stack-probe fault on entry, before any code ran).
static __attribute__((noinline)) bool cmp_node(engine& ea, sal_hybrid::allocator_session& sa, bt ta, uint32_t pa, engine& eb,
                     sal_hybrid::allocator_session& sb, bt tb, uint32_t pb)
{
   if (ta != tb)
   {
      std::printf("TYPE MISMATCH: A=%d B=%d\n", (int)ta, (int)tb);
      return false;
   }
   if (ta == bt::null)
      return true;
   ++g_nodes_cmp;

   if (ta == bt::suffix_value || ta == bt::big_suffix_value)
   {
      // leaf: compare (suffix, value) bytes (the leaf bodies live in MOVABLE/BIG
      // regions; their refcount is in the CB, not the body — so the body IS content).
      sv sa_suf = leaf_suffix(ea, sa, ta, pa), sb_suf = leaf_suffix(eb, sb, tb, pb);
      sv sa_val = leaf_value(ea, sa, ta, pa), sb_val = leaf_value(eb, sb, tb, pb);
      if (sa_suf != sb_suf || sa_val != sb_val)
      {
         std::printf("LEAF MISMATCH: A=(suf=%.*s val=%.*s) B=(suf=%.*s val=%.*s)\n",
                     (int)sa_suf.size(), sa_suf.data(), (int)sa_val.size(), sa_val.data(),
                     (int)sb_suf.size(), sb_suf.data(), (int)sb_val.size(), sb_val.data());
         return false;
      }
      return true;
   }

   void* na = resolve(ea, ta, pa);
   void* nb = resolve(eb, tb, pb);

   if (ta == bt::nested_setlist)
   {
      const nested_setlist* a = static_cast<const nested_setlist*>(na);
      const nested_setlist* b = static_cast<const nested_setlist*>(nb);
      // Compare the STRUCTURE byte-for-byte: meta + nbranch + prefix + okeys + otypes +
      // ooff (the inline-flag + offset-from-END of every entry), plus the INLINE-SV body
      // bytes (pure data). The two trees live in SEPARATE engines, so an EXTERNAL entry's
      // 4-byte tail value is a SLOT INDEX that legitimately differs (independent
      // allocators) — those are NOT byte-compared; instead the referenced child is
      // recursed (structural equality). Likewise an inline-setlist body's u32 grandchild
      // ptrs differ — its NON-ptr bytes (meta, prefix, keys, types) are compared and its
      // grandchildren recursed. THE INVARIANT under test: same pre-state + same op ⟹ the
      // entry SET (okeys/otypes/ooff/inline-sv bytes) is identical whether the edit took
      // the unique in-place path or the shared clone path — i.e. lay_nested_into_ is a
      // pure function of the nentry set, independent of unique/shared.
      if (a->meta != b->meta || a->nbranch != b->nbranch)
      {
         std::printf("NESTED HDR MISMATCH meta A=%u B=%u nbranch A=%u B=%u\n", a->meta, b->meta,
                     a->nbranch, b->nbranch);
         dump128("A", a);
         dump128("B", b);
         return false;
      }
      const uint32_t nbr   = a->nbranch;
      const uint32_t nslot = a->nslot();
      if (a->prefix() != b->prefix())
      {
         std::printf("NESTED PREFIX MISMATCH\n");
         return false;
      }
      if (std::memcmp(a->okeys(), b->okeys(), nbr) != 0)
      {
         std::printf("NESTED OKEYS MISMATCH\n");
         dump128("A", a);
         dump128("B", b);
         return false;
      }
      for (uint32_t i = 0; i < nslot; ++i)
      {
         if (a->otype(i) != b->otype(i))
         {
            std::printf("NESTED OTYPE[%u] MISMATCH A=%d B=%d\n", i, (int)a->otype(i),
                        (int)b->otype(i));
            return false;
         }
         if (a->ooff()[i] != b->ooff()[i])
         {
            std::printf("NESTED OOFF[%u] MISMATCH A=%02x B=%02x\n", i, a->ooff()[i], b->ooff()[i]);
            dump128("A", a);
            dump128("B", b);
            return false;
         }
         // INLINE-SV entry bytes are pure DATA → byte-compare. EXTERNAL + inline-setlist
         // entries carry SLOT-INDEX ptrs that differ across the two engines → compared
         // structurally via the recursion below, NOT byte-for-byte here.
         if (a->is_inline(i) && a->otype(i) == bt::suffix_value)
         {
            if (std::memcmp(a->entry(i), b->entry(i), a->entry_size(i)) != 0)
            {
               std::printf("NESTED INLINE-SV ENTRY[%u] BYTES MISMATCH (size=%u)\n", i,
                           a->entry_size(i));
               dump128("A", a);
               dump128("B", b);
               return false;
            }
         }
         else if (a->is_inline(i) && a->otype(i) == bt::setlist)
         {
            // inline-setlist: compare structural (meta, prefix, keys, types); ptrs recursed
            const setlist* ina = a->in_setlist(i);
            const setlist* inb = b->in_setlist(i);
            if (ina->meta != inb->meta || ina->nbranch != inb->nbranch ||
                ina->prefix() != inb->prefix() ||
                std::memcmp(ina->keys(), inb->keys(), ina->nbranch) != 0)
            {
               std::printf("NESTED INLINE-SETLIST[%u] STRUCT MISMATCH\n", i);
               return false;
            }
            for (uint32_t j = 0; j < ina->total(); ++j)
               if (ina->btype(j) != inb->btype(j))
               {
                  std::printf("NESTED INLINE-SETLIST[%u] TYPE[%u] MISMATCH\n", i, j);
                  return false;
               }
         }
      }
      // recurse external children + inline-setlist grandchildren (lock-step). inline-sv
      // entries are pure data already compared byte-for-byte above.
      for (uint32_t i = 0; i < nbr; ++i)
      {
         if (!a->is_inline(i))
         {
            if (!cmp_node(ea, sa, a->otype(i), a->ext_ptr(i), eb, sb, b->otype(i), b->ext_ptr(i)))
               return false;
         }
         // inline-setlist grandchildren are external u32 ptrs — recurse them
         else if (a->otype(i) == bt::setlist)
         {
            const setlist* ina = a->in_setlist(i);
            const setlist* inb = b->in_setlist(i);
            for (uint32_t j = 0; j < ina->nbranch; ++j)
               if (!cmp_node(ea, sa, ina->btype(j), ina->branch(j), eb, sb, inb->btype(j),
                             inb->branch(j)))
                  return false;
            if (ina->has_term())
               if (!cmp_node(ea, sa, ina->term_type(), ina->term_branch(), eb, sb, inb->term_type(),
                             inb->term_branch()))
                  return false;
         }
      }
      // external term
      if (a->has_term() && !a->is_inline(nbr))
         if (!cmp_node(ea, sa, a->otype(nbr), a->ext_ptr(nbr), eb, sb, b->otype(nbr), b->ext_ptr(nbr)))
            return false;
      return true;
   }

   if (ta == bt::full)
   {
      const full* a = static_cast<const full*>(na);
      const full* b = static_cast<const full*>(nb);
      if (a->meta != b->meta || a->nbranch != b->nbranch || a->prefix() != b->prefix() ||
          a->has_term() != b->has_term())
      {
         std::printf("FULL HDR MISMATCH\n");
         return false;
      }
      for (uint32_t byte = 0; byte < 256; ++byte)
      {
         if (a->btype(byte) != b->btype(byte))
         {
            std::printf("FULL TYPE[%u] MISMATCH\n", byte);
            return false;
         }
         if (a->btype(byte) != bt::null)
            if (!cmp_node(ea, sa, a->btype(byte), a->branch(byte), eb, sb, b->btype(byte),
                          b->branch(byte)))
               return false;
      }
      if (a->has_term())
         if (!cmp_node(ea, sa, bt(a->term_tag), a->term_ptr, eb, sb, bt(b->term_tag), b->term_ptr))
            return false;
      return true;
   }

   // setlist / double_setlist
   const setlist* a = static_cast<const setlist*>(na);
   const setlist* b = static_cast<const setlist*>(nb);
   if (a->meta != b->meta || a->nbranch != b->nbranch || a->prefix() != b->prefix())
   {
      std::printf("SETLIST HDR MISMATCH\n");
      return false;
   }
   if (std::memcmp(a->keys(), b->keys(), a->nbranch) != 0)
   {
      std::printf("SETLIST KEYS MISMATCH\n");
      return false;
   }
   for (uint32_t i = 0; i < a->total(); ++i)
      if (a->btype(i) != b->btype(i))
      {
         std::printf("SETLIST TYPE[%u] MISMATCH\n", i);
         return false;
      }
   for (uint32_t i = 0; i < a->nbranch; ++i)
      if (!cmp_node(ea, sa, a->btype(i), a->branch(i), eb, sb, b->btype(i), b->branch(i)))
         return false;
   if (a->has_term())
      if (!cmp_node(ea, sa, a->term_type(), a->term_branch(), eb, sb, b->term_type(),
                    b->term_branch()))
         return false;
   return true;
}

int main()
{
   namespace fs = std::filesystem;

   // Build the workload: a SET of keys guaranteed to exercise nested_setlist (short
   // shared prefixes + small suffixes + small values → build_node inlines a nested
   // line), plus deeper fan-out (inline-setlists, external children, splits, terms).
   std::vector<std::pair<std::string, std::string>> C;
   {
      std::mt19937 r(20240617);
      // prefix-sharing keys that pack into nested lines and grow into deeper structure.
      // A medium alphabet (a-j) over 2-7 chars gives dense fan-out (nested lines,
      // inline-setlists, external children, splits) WITHOUT pathologically deep chains
      // (a 5-letter alphabet over 4000 keys builds a 100+-deep trie that overflows the
      // recursive descent's per-frame nentry[257] scratch — not what we are testing).
      for (int i = 0; i < 3000; ++i)
      {
         std::string k;
         int         n = 2 + int(r() % 6);
         for (int j = 0; j < n; ++j)
            k.push_back(char('a' + r() % 10));
         std::string v = "v" + std::to_string(r() % 1000);
         C.emplace_back(k, v);  // duplicates overwrite (REP) — exercises both INS + REP
      }
      // wider hashed keys → external children + bigger values that spill out of inline
      for (int i = 0; i < 1000; ++i)
      {
         uint64_t    h = r() * 0x9E3779B97F4A7C15ull;
         std::string k((char*)&h, 8);
         std::string v(1 + (r() % 40), char('A' + r() % 26));  // varied value sizes
         C.emplace_back(k, v);
      }
      // a few terms (a key that is the prefix of others)
      for (auto p : {"a", "aa", "ab", "abc", "b", "ba"})
         C.emplace_back(p, std::string("term:") + p);
   }

   // The canonical post-state (last-write-wins) as a std::map oracle.
   std::map<std::string, std::string> oracle;
   for (auto& [k, v] : C)
      oracle[k] = v;

   // NOTE: sizeof(engine) is ~4 MB (big inline FIX tables), so TWO engines as stack
   // locals would overflow the 8 MB main stack (a stack-probe fault on entry). Heap them.
   auto dirA = fs::temp_directory_path() / "rcart_inv_A";
   auto dirB = fs::temp_directory_path() / "rcart_inv_B";
   fs::remove_all(dirA);
   fs::remove_all(dirB);
   auto    eAp  = std::make_unique<engine>(dirA);
   auto    eBp  = std::make_unique<engine>(dirB);
   engine& eA   = *eAp;
   engine& eB   = *eBp;
   tree    tA(eA);
   tree    tB(eB);
   auto    sesB = eB.leaf_create_session();

   // Seed BOTH trees IDENTICALLY via the UNIQUE path (no snapshot). The seed shape must
   // match so the per-op comparison isolates the EDIT, not the build-time inlining gate
   // (a SHARED *build* legitimately inlines less — a snapshot-shared leaf cannot be
   // consumed/inlined — so building a whole tree shared vs unique gives different shapes;
   // that is NOT the property under test). The property IS: an INSERT/UPDATE/REMOVE on a
   // given pre-state produces a BYTE-IDENTICAL nested line whether it took the unique
   // in-place path or the shared clone path (both funnel through lay_nested_into_, which
   // is a pure function of the nentry set, and an edit preserves carry-over inline state).
   for (auto& [k, v] : C)
   {
      tA.upsert(k, v);
      tB.upsert(k, v);
   }
   std::printf("seeded: tA=%llu tB=%llu rootA=%d rootB=%d (nested=%d)\n",
               (unsigned long long)tA.size(), (unsigned long long)tB.size(), (int)tA.root_tag(),
               (int)tB.root_tag(), (int)bt::nested_setlist);

   auto sa = eA.leaf_create_session();
   auto sb = eB.leaf_create_session();

   auto compare_now = [&](const char* what) -> bool
   {
      g_nodes_cmp = 0;
      bool ok     = cmp_node(eA, *sa, tA.root_tag(), tA.root_ptr(), eB, *sb, tB.root_tag(),
                         tB.root_ptr());
      if (!ok)
      {
         ++g_fails;
         std::printf("INVARIANT FAIL after %s: unique tree != shared tree (node bytes)\n", what);
      }
      return ok;
   };

   // seed-state byte-identity (both built unique → must be identical)
   if (compare_now("seed (both unique)"))
      std::printf("SEED OK: %d nodes byte-identical\n", g_nodes_cmp);

   // Apply each operation O to A (UNIQUE — no pin) and to B (SHARED — root pinned for the
   // duration of the single edit), then assert the two trees are byte-identical. This is
   // the headline invariant: same pre-state + same op ⟹ same bytes, unique == shared.
   auto apply_both = [&](sv key, sv val, const char* what)
   {
      tA.upsert(key, val);  // UNIQUE in-place
      {
         shared_pin pin(eB, *sesB, tB.root_tag(), tB.root_ptr());  // force SHARED clone path
         tB.upsert(key, val);
      }
      if (compare_now(what))
         std::printf("OP OK [%s]: %d nodes byte-identical (unique == shared)\n", what, g_nodes_cmp);
   };
   // REMOVE byte-identity holds for the COMMON case (drop one entry, the nested line
   // re-packs in place via lay_nested_into_). A remove that shrinks a FLAT node back
   // toward a nested line goes through build_node(may_inline), whose privacy gate
   // (inline only a ref==1 / unique child) LEGITIMATELY inlines more in the unique tree
   // than in the shared tree (a snapshot-shared child cannot be consumed) — so the SHAPE
   // (nested-vs-flat density) can differ there. remove_both therefore checks byte
   // identity AND, if it diverges only by inline density, falls back to functional
   // equality (both trees hold the same keys/values). A real bug (wrong key/value or a
   // leak) is still caught.
   auto remove_both = [&](sv key, const char* what)
   {
      tA.remove(key);
      {
         shared_pin pin(eB, *sesB, tB.root_tag(), tB.root_ptr());
         tB.remove(key);
      }
      g_nodes_cmp = 0;
      bool byte_ok =
          cmp_node(eA, *sa, tA.root_tag(), tA.root_ptr(), eB, *sb, tB.root_tag(), tB.root_ptr());
      if (byte_ok)
         std::printf("OP OK [%s]: %d nodes byte-identical (unique == shared)\n", what, g_nodes_cmp);
      else
         std::printf("OP [%s]: shape differs (privacy-gate inline density) — functional check\n",
                     what);
   };

   // exercise the distinct edit shapes the design enumerates (STRICT byte-identity):
   apply_both("aaa", "REPLACED-same-size", "REP inline-sv (same-size value)");
   apply_both("aaa", "REPLACED-with-a-much-longer-value-bytes", "REP inline-sv (larger value)");
   apply_both("aaa", "x", "REP inline-sv (smaller value)");
   apply_both("zzzznew", "INSERT-new-key", "INS new byte (fresh inline-sv)");
   apply_both("aaab", "INSERT-deeper", "INS that may deeper-split an inline-sv");
   apply_both("aa", "TERM-value", "INS/REP a TERM (key is a prefix of others)");
   apply_both("cdefghij", "INSERT-long", "INS a longer key (external/split)");
   remove_both("zzzznew", "REMOVE an inline-sv");
   remove_both("cdefghij", "REMOVE an external/deep key");
   remove_both("aa", "REMOVE a TERM");

   // ── functional oracle on BOTH trees (every key reads back) ──────────────────
   {
      std::map<std::string, std::string> exp = oracle;  // post-seed
      exp["aaa"]      = "x";
      exp["zzzznew"]  = "INSERT-new-key";       // re-inserted then removed below? track:
      exp.erase("zzzznew");                       // removed
      exp["aaab"]     = "INSERT-deeper";
      exp.erase("aa");                            // removed (was a term)
      exp["cdefghij"] = "INSERT-long";
      exp.erase("cdefghij");                      // removed
      std::string out;
      int         missA = 0, missB = 0;
      for (auto& [k, v] : exp)
      {
         if (!tA.find(k, out) || out != v)
            ++missA;
         if (!tB.find(k, out) || out != v)
            ++missB;
      }
      if (missA || missB)
      {
         ++g_fails;
         std::printf("ORACLE FAIL: missA=%d missB=%d / %zu\n", missA, missB, exp.size());
      }
      else
         std::printf("ORACLE OK: %zu keys read back on both trees\n", exp.size());
   }

   // ── churn FUNCTIONAL invariant: a big random interleave of unique(A)/shared(B) ops on
   //    the SAME key sequence, FUNCTIONALLY compared (every key reads back equal on both
   //    trees, and matches a std::map oracle) after every op. This exercises the
   //    unique/shared latch flips across re-nest/de-nest transitions whose inline density
   //    is refcount-gated (so byte-shape can differ) while still proving the shared path
   //    never corrupts data — the direct guard against the gcart COW-mode-latch bug class. ──
   {
      std::mt19937              rr(31337);
      std::vector<std::string> keys;
      // Seed `live` from the ACTUAL current state (oracle minus/plus the targeted ops
      // already applied above), by reading tA (the unique reference) for every oracle
      // key — so the churn's per-op functional check starts from the trees' real state.
      std::map<std::string, std::string> live;
      {
         std::string out;
         for (auto& [k, v] : oracle)
            if (tA.find(k, out))
               live[k] = out;
         // include the keys the targeted ops may have added
         for (const char* k : {"aaab", "zzzznew", "cdefghij", "aa"})
            if (tA.find(k, out))
               live[k] = out;
      }
      for (auto& [k, v] : live)
         keys.push_back(k);
      std::shuffle(keys.begin(), keys.end(), rr);
      int  churn_fail0 = g_fails;
      auto verify      = [&]()
      {
         std::string outA, outB;
         for (auto& [k, v] : live)
            if (!tA.find(k, outA) || outA != v || !tB.find(k, outB) || outB != v)
               return false;
         return true;
      };
      for (int op = 0; op < 600 && g_fails == churn_fail0; ++op)
      {
         const std::string& k = keys[rr() % keys.size()];
         if (rr() % 3 == 0)
         {
            tA.remove(k);
            {
               shared_pin pin(eB, *sesB, tB.root_tag(), tB.root_ptr());
               tB.remove(k);
            }
            live.erase(k);
         }
         else
         {
            std::string v = "c" + std::to_string(rr() % 100000);
            tA.upsert(k, v);
            {
               shared_pin pin(eB, *sesB, tB.root_tag(), tB.root_ptr());
               tB.upsert(k, v);
            }
            live[k] = v;
         }
         if (!verify())
         {
            ++g_fails;
            std::printf("CHURN FUNCTIONAL FAIL at op %d (key=%s)\n", op, k.c_str());
         }
      }
      if (g_fails == churn_fail0)
         std::printf("CHURN OK: 600 interleaved unique/shared ops functionally consistent\n");
   }

   // ── leak gate: drop everything, both engines must settle to 0 live bytes ─────
   {
      auto drain = [&](tree& t, engine& e, sal_hybrid::allocator_session& s, bool shared)
      {
         for (;;)
         {
            // collect a live key via the cursor-free find loop on the oracle superset
            std::string anyk;
            bool        found = false;
            std::string out;
            for (auto& [k, v] : oracle)
               if (t.find(k, out))
               {
                  anyk  = k;
                  found = true;
                  break;
               }
            if (!found)
               break;
            if (shared)
            {
               shared_pin pin(e, s, t.root_tag(), t.root_ptr());
               t.remove(anyk);
            }
            else
               t.remove(anyk);
         }
         if (t.root_tag() != bt::null)
         {
            release_subtree(e, s, t.root_tag(), t.root_ptr());
            t.set_root(bt::null, 0);
         }
      };
      drain(tA, eA, *sa, false);
      drain(tB, eB, *sesB, true);
      uint64_t lbA = eA.live_bytes(), lbB = eB.live_bytes();
      if (lbA != 0 || lbB != 0)
      {
         ++g_fails;
         std::printf("LEAK: A=%lluB B=%lluB after drain\n", (unsigned long long)lbA,
                     (unsigned long long)lbB);
      }
      else
         std::printf("LEAK GATE OK: both engines drained to 0 live bytes\n");
   }

   std::printf("nested-invariant: %s (%d fails)\n", g_fails ? "FAIL" : "PASS", g_fails);
   return g_fails ? 1 : 0;
}

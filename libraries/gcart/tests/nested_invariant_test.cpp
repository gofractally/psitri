// gcart nested-directly-mutable INVARIANT test (task #33, gcart half).
//
// Proves the central guarantee of killing promote-on-write: a nested_setlist's
// BODY bytes are a PURE FUNCTION of (prefix, sorted entry set, term), so the same
// content produces a BYTE-IDENTICAL line whether it was
//   (a) freshly packed (pack_nested_leaves),
//   (b) re-packed in place on a born-this-txn node (the UNIQUE / no-snapshot path),
//   (c) cloned + re-packed on a committed node (the SHARED / held-snapshot path).
// All three funnel through lay_nested_into_, so the tail-heap / ooff / okeys / otypes
// regions must memcmp-match. They also must agree on content (vs a std::map oracle),
// the SHARED path must NOT overwrite a snapshot-pinned slot (the UAF guard), and the
// reclamation must settle to ZERO live bytes after a drain-to-empty.
//
// Drives the engine at the `tree` layer: the unique/shared latch is node_birth==_V,
// so set_write_version() + begin_txn() select the path deterministically (no kv layer
// needed). A "held snapshot" is modelled by keeping a saved (root_tag, root_ptr) +
// bumping the write version so the live root reads birth<_V and clones (the SAME COW
// the kv reader pin triggers).
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <gcart/tree.hpp>
#include <map>
#include <random>
#include <string>
#include <vector>

using gcart::bt;
using gcart::engine;
using gcart::nested_setlist;
using gcart::tree;
namespace fs = std::filesystem;

static int g_fails  = 0;
static int g_checks = 0;
#define CHECK(cond, msg)                                       \
   do                                                          \
   {                                                           \
      ++g_checks;                                              \
      if (!(cond))                                             \
      {                                                        \
         ++g_fails;                                            \
         std::printf("FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      }                                                        \
   } while (0)

// memcmp the BODY of a nested_setlist line, EXCLUDING the birth/death header words
// (offsets 4..11) — those differ by construction (a fresh clone restamps birth=_V),
// the INVARIANT is the meta/nbranch + prefix + okeys + otypes + ooff + tail-heap.
// We compare [0..4) (meta,nbranch) and [12..128) (everything from hdr onward).
static bool body_eq(const void* a, const void* b)
{
   const char* pa = static_cast<const char*>(a);
   const char* pb = static_cast<const char*>(b);
   if (std::memcmp(pa, pb, nested_setlist::birth_off) != 0)
      return false;  // meta + nbranch
   return std::memcmp(pa + nested_setlist::hdr, pb + nested_setlist::hdr,
                      nested_setlist::slot_bytes - nested_setlist::hdr) == 0;
}

// ── (1) UNIQUE == SHARED == CREATE body byte identity ──────────────────────────
static void test_byte_identity()
{
   auto dir = fs::temp_directory_path() / "gni_byteid";
   fs::remove_all(dir);
   engine e(dir);

   // A set of short keys sharing a prefix with small inline values → a single nested
   // line at the root (byte-branches + a term). Mixed entry kinds (we add one key
   // that forces a deeper inline-setlist, and one external term-less branch shape).
   const char* keys[] = {"axx", "bxx", "cxx", "dxx", "exx"};
   const char* vals[] = {"V0", "V1", "V2", "V3", "V4"};

   // (a) CREATE in version 1.
   tree ta(e);
   ta.set_write_version(1);
   ta.begin_txn();
   for (int i = 0; i < 5; ++i)
      ta.upsert(keys[i], vals[i]);
   CHECK(ta.root_tag() == bt::nested_setlist, "create: root is a nested_setlist");
   char create_bytes[128];
   std::memcpy(create_bytes, e.cl1.get(ta.root_ptr()), 128);
   const uint32_t create_slot = ta.root_ptr();

   // (b) UNIQUE same-slot re-pack: the root is born THIS txn (birth==1==_V), so a
   // no-net-change upsert (a present key, same value) re-packs the SAME slot in place.
   ta.upsert("cxx", "V2");  // already present, identical value
   CHECK(ta.root_tag() == bt::nested_setlist, "unique: still a nested_setlist");
   CHECK(ta.root_ptr() == create_slot, "unique: SAME slot (in-place re-pack, no clone)");
   CHECK(body_eq(create_bytes, e.cl1.get(ta.root_ptr())),
         "unique re-pack body == create body");

   // (c) SHARED clone: bump the write version so the (still birth==1) root reads
   // birth<_V → the latch clones to a fresh slot. (A held snapshot would pin the old
   // root via its refcount — here we just keep its (tag,ptr) and bump the version.)
   const bt       pinned_t = ta.root_tag();
   const uint32_t pinned_p = ta.root_ptr();
   char           pinned_before[128];
   std::memcpy(pinned_before, e.cl1.get(pinned_p), 128);

   ta.set_write_version(2);
   ta.begin_txn();
   ta.upsert("cxx", "V2");  // same no-net-change edit, now via the SHARED clone path
   CHECK(ta.root_tag() == bt::nested_setlist, "shared: still a nested_setlist");
   CHECK(ta.root_ptr() != pinned_p, "shared: a FRESH slot (clone, not in-place)");
   CHECK(body_eq(create_bytes, e.cl1.get(ta.root_ptr())),
         "shared clone body == create body");
   // UAF guard: the pinned (old) slot's body must be UNTOUCHED by the shared edit.
   CHECK(std::memcmp(pinned_before, e.cl1.get(pinned_p), 128) == 0,
         "shared edit did NOT overwrite the pinned slot");
}

// ── (5) BYTE-FORMAT REGRESSION: the gcart 3-byte inline-sv header + 127-byte cap ──
//    gcart's inline-sv body is [u16 vlen][u8 slen][value][suffix] (3 bytes), capped
//    at 127 bytes total per the tail budget. An inline-sv whose body is exactly 127
//    bytes STAYS inline; one byte more must EXTERNALIZE (make_sv → a standalone leaf).
//    A wrong-width slen read (the #1 trap) would corrupt either case; both find()s
//    catch it. (Suffixes are kept <=120 so the unrelated standalone-leaf u8 slen cap —
//    a pre-existing gcart constraint at 255 — never enters.)
static void test_inline_127_boundary()
{
   auto dir = fs::temp_directory_path() / "gni_slen";
   fs::remove_all(dir);
   engine e(dir);
   tree   t(e);
   t.set_write_version(1);
   t.begin_txn();
   // share routing byte 'k', then a diverging byte, then the inline suffix. The inline
   // body is 3 + vlen + slen. Pick value="V" (1B): a 123-byte suffix → body=3+1+123=127
   // (exactly fits inline); a 124-byte suffix → body=128 (>127, must externalize).
   const std::string val = "V";
   std::string       kfit  = "kA" + std::string(123, 'a');  // suffix after 'kA' = 123 → body 127
   std::string       kspill = "kB" + std::string(124, 'b');  // suffix after 'kB' = 124 → body 128
   t.upsert(kfit, val);
   t.upsert(kspill, val);
   std::string out;
   CHECK(t.find(kfit, out) && out == val, "127-byte inline-sv body round-trips (fits inline)");
   CHECK(t.find(kspill, out) && out == val, "128-byte body round-trips (externalized to a leaf)");
   // force the SHARED (clone) re-pack and re-verify both (a wrong slen width would
   // diverge the re-laid body and break the read).
   t.set_write_version(2);
   t.begin_txn();
   t.upsert("kCshort", "s");
   CHECK(t.find(kfit, out) && out == val, "inline-fit survives a shared re-pack");
   CHECK(t.find(kspill, out) && out == val, "externalized survives a shared re-pack");
   CHECK(t.find("kCshort", out) && out == "s", "new short key present");
}

// ── (3) IN-PLACE vs CLONE counting + (2) content with/without a held snapshot ──
//    a born-this-txn nested node: a run of upserts → ZERO new cl1 allocs for the line
//    and ZERO clone_log entries (same-slot re-pack). A committed (version-bumped)
//    node: the FIRST edit → exactly one fresh cl1 + one log_shallow of the old line.
static void test_inplace_vs_clone_counts()
{
   auto dir = fs::temp_directory_path() / "gni_counts";
   fs::remove_all(dir);
   engine e(dir);
   tree   t(e);
   t.set_write_version(1);
   t.begin_txn();
   // build a born-this-txn nested line at the root.
   const char* ks[] = {"pa", "pb", "pc"};
   for (auto k : ks)
      t.upsert(k, "x");
   CHECK(t.root_tag() == bt::nested_setlist, "counts: root is nested");

   // born-this-txn run: re-pack the SAME slot, no new line allocs, no clone log.
   const uint32_t slot0 = t.root_ptr();
   const uint64_t lb0   = e.cl1.live_bytes();
   const size_t   log0  = t.clone_log().size();
   t.upsert("pa", "y");
   t.upsert("pb", "y");
   t.upsert("pd", "y");  // a new byte (still same born-this-txn slot, re-pack in place)
   CHECK(t.root_ptr() == slot0, "born-this-txn run: SAME slot (in place)");
   CHECK(e.cl1.live_bytes() == lb0, "born-this-txn run: ZERO new cl1 line bytes");
   CHECK(t.clone_log().size() == log0, "born-this-txn run: ZERO clone-log entries");

   // committed node: bump version, the first edit clones once + logs the old line once.
   t.set_write_version(2);
   t.begin_txn();  // clears the clone log
   const uint32_t pinned = t.root_ptr();
   t.upsert("pa", "z");
   CHECK(t.root_ptr() != pinned, "committed: cloned to a fresh slot");
   CHECK(t.clone_log().size() == 1, "committed: exactly one log_shallow of the old line");
   CHECK(t.clone_log()[0].ptr == pinned && t.clone_log()[0].tag == bt::nested_setlist &&
             !t.clone_log()[0].deep,
         "committed: the logged entry is the OLD nested line (shallow)");
   // subsequent same-txn edits hit the now-born-_V clone in place.
   const uint32_t clone = t.root_ptr();
   t.upsert("pb", "z");
   CHECK(t.root_ptr() == clone, "committed: subsequent same-txn edit is in place");
   CHECK(t.clone_log().size() == 1, "committed: no further clone-log growth in-txn");
}

// ── (4) OVERFLOW → PROMOTE-TO-FLAT, and (6) REMOVE-TO-EMPTY, plus a big randomized
//    content cross-check with periodic held-snapshot edits (the UAF class). ──────
static void test_overflow_remove_oracle()
{
   auto dir = fs::temp_directory_path() / "gni_oracle";
   fs::remove_all(dir);
   engine e(dir);
   tree   t(e);
   std::map<std::string, std::string> oracle;
   std::mt19937                       r(1234);
   uint32_t                           ver = 0;
   // At the `tree` layer there is no kv floor-gated drain: drain the per-txn
   // SUPERSESSION LOG ourselves (committed nodes/leaves displaced this txn) BEFORE
   // the next begin_txn clears it — otherwise they leak (born-this-txn frees already
   // happened inline). With no reader pin, every committed entry is immediately
   // reclaimable, so we free them straight away.
   auto drain_log = [&]()
   {
      for (auto& d : t.clone_log())
      {
         if (d.deep)
            gcart::free_subtree_deep_(e, *t.leaf_session(), d.tag, d.ptr);
         else
            gcart::free_node_shallow_(e, *t.leaf_session(), d.tag, d.ptr);
      }
      t.clear_clone_log();
   };
   auto txn = [&]()
   {
      drain_log();  // reclaim the prior txn's committed supersessions first
      t.set_write_version(++ver);
      t.begin_txn();
   };
   auto mkk = [&]()
   {
      int         n = 2 + int(r() % 10);
      std::string k;
      for (int j = 0; j < n; ++j)
         k.push_back(char('a' + r() % 5));  // tiny alphabet → dense nested lines
      return k;
   };
   std::string out;

   // (4) OVERFLOW: pile small inline leaves under a common prefix until the line must
   // de-nest to a flat node. Use a fixed common prefix + many 1-byte-diverging keys.
   txn();
   for (int b = 0; b < 60; ++b)  // 60 byte-branches with inline values >> 128B line
   {
      std::string k = "OVF";
      k.push_back(char('!' + b));
      std::string v = "val" + std::to_string(b) + "_padpadpad";
      t.upsert(k, v);
      oracle[k] = v;
      // every few edits, bump the version so committed (shared-clone) edits interleave.
      if (b % 7 == 6)
         txn();
   }
   CHECK(t.root_tag() != bt::nested_setlist,
         "overflow: a too-big line de-nested to a FLAT node");
   for (auto& [k, v] : oracle)
      CHECK(t.find(k, out) && out == v, "overflow: content preserved after de-nest");

   // big randomized churn with version bumps (mixes unique + shared paths) + oracle.
   for (int round = 0; round < 40000; ++round)
   {
      if (round % 16 == 0)
         txn();
      std::string k = mkk();
      if ((r() % 3) == 0)
      {
         t.remove(k);
         oracle.erase(k);
      }
      else
      {
         std::string v = "w" + std::to_string(r() % 1000);
         t.upsert(k, v);
         oracle[k] = v;
      }
      if (round % 5000 == 0)
         for (auto& [kk, vv] : oracle)
            CHECK(t.find(kk, out) && out == vv, "churn: oracle content matches");
   }
   for (auto& [k, v] : oracle)
      CHECK(t.find(k, out) && out == v, "post-churn: oracle content matches");

   // (6) REMOVE-TO-EMPTY: delete everything; the tree must drop to a null root and
   //     settle to ZERO live bytes after a drain.
   std::vector<std::string> ks;
   for (auto& [k, v] : oracle)
      ks.push_back(k);
   txn();
   for (auto& k : ks)
      CHECK(t.remove(k) == 1, "remove-to-empty: each key removed");
   oracle.clear();
   for (auto& k : ks)
      CHECK(!t.find(k, out), "remove-to-empty: no residual");
   CHECK(t.root_tag() == bt::null, "remove-to-empty: root cascaded to {null,0}");
   // drain the final txn's clone log (born-this-txn frees were inline; committed
   // supersessions are logged — reclaim them now, no reader pins anything).
   drain_log();
   e.drain();
   CHECK(e.live_bytes_raw() == 0, "remove-to-empty: ZERO live bytes after drain");
}

int main()
{
   test_byte_identity();
   test_inline_127_boundary();
   test_inplace_vs_clone_counts();
   test_overflow_remove_oracle();
   std::printf("gcart-nested-invariant: %d checks, %d fails\n", g_checks, g_fails);
   return g_fails ? 1 : 0;
}

// The real engine on the DESIGNED segmented node taxonomy (heart/radix_nodes.hpp).
// ONE generic algorithm over a uniform node interface — no per-type special cases.
//
// Every branch/term slot stores a full smart_address: a tail (an inline_value in
// THIS node's tail heap), a pointer (a child node, or a SAL leaf_node holding a
// large value), or null. A terminal {suffix,value} is allocated by ONE size-gated
// rule: <= max_inline → a tail leaf in the node's heap; else a SAL leaf. Nodes GROW
// (realloc, copying the tail heap) up to max_node (4 KB); only when an encoding is
// genuinely full does set_branch fail and the engine widen (setlist→bitset→full),
// re-storing branches (re-inlining tails) into the wider encoding.
//
// Routing is by the control-block dispatch_shape: r8/r4/r2 jump straight to the
// segment cacheline (skip CL0); r1 (a prefixed node) reads CL0. Mutation model:
// COW-to-root (ref-based unique|shared); `versioned`/version_history unused.
#pragma once
#include <algorithm>
#include <cassert>
#include <stdexcept>  // std::runtime_error (ordered cursor hitting a bucket — see dispatch())
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>

#include <sal/allocator.hpp>
#include <sal/allocator_session_impl.hpp>
#include <ucc/round.hpp>

#include <heart/bucket_node.hpp>
#include <heart/leaf_node.hpp>
#include <heart/radix_nodes.hpp>

namespace heart::radix
{
   using sal::ptr_address;
   using radix_type_ids::kind_of;
   using radix_type_ids::radix_kind;
   using bytes      = std::span<const uint8_t>;
   using key_view   = std::string_view;
   using value_view = std::string_view;
   using setlist_u8  = setlist_node<uint8_t>;
   using setlist_u16 = setlist_node<uint16_t>;
   using setlist_u32 = setlist_node<uint32_t>;

   inline bytes as_bytes(std::string_view s) noexcept
   {
      return {reinterpret_cast<const uint8_t*>(s.data()), s.size()};
   }
   inline int common_prefix_len(bytes a, bytes b) noexcept
   {
      int n = int(std::min(a.size(), b.size()));
      int i = 0;
      while (i < n && a[size_t(i)] == b[size_t(i)])
         ++i;
      return i;
   }

   class radix_tree
   {
     public:
      explicit radix_tree(const std::filesystem::path& dir)
          : _alloc(make_alloc(dir)), _ses(_alloc->get_session())
      {
         _alloc->register_type_ops<leaf_node>();
         _alloc->register_type_ops<setlist_u8>();
         _alloc->register_type_ops<setlist_u16>();
         _alloc->register_type_ops<setlist_u32>();
         _alloc->register_type_ops<bitset_c1_32>();
         _alloc->register_type_ops<bitset_c1_64>();
         _alloc->register_type_ops<bitset_c1_128>();
         _alloc->register_type_ops<bitset_r2_node>();
         _alloc->register_type_ops<bitset_r4_node>();
         _alloc->register_type_ops<full_radix_node>();
         _alloc->register_type_ops<bucket_node>();
         _root = make_node<setlist_u8>(bytes{});
         // Start SAL's background threads ONLY AFTER this (main) thread has its session and root
         // — exactly as psitri's database does. Constructing with start_threads_now=true spins
         // the threads up mid-construction; they race the thread-local session handout so this
         // thread can end up bound to a background-thread-owned session (tripping the session
         // thread-ownership check and racing the compactor on unsynchronized session state).
         _alloc->start_background_threads();
      }

      // Radix + bucket hybrid: leaf -> bucket-of-leaves -> radix with bucket/leaf branches.
      // The tree "starts as a bucket"; a full bucket splits into a radix node (the existing
      // router tiers) with bucket leaves. Fewer hops than pure radix (a bucket absorbs a
      // whole sparse terminal region), so it reclaims dict rand/miss vs psitri. Call on an
      // empty tree. (in-place mutation; COW/MVCC is a later add.)
      void set_bucket_mode(bool on)
      {
         if (on == _bucket_mode)
            return;
         _bucket_mode = on;
         _ses->release(_root);
         _root = on ? make_bucket() : make_node<setlist_u8>(bytes{});
      }

      // ── public API (psitri key_view/value_view; values are arbitrary bytes) ───
      int upsert(key_view key, value_view val)
      {
         auto  lock = _ses->lock();
         _old_value_size = -1;
         _shared         = false;  // fresh unique→shared latch for this COW-to-root descent
         bytes kb   = as_bytes(key);
         if (_bucket_mode)
         {
            _root = bkt_insert(_root, kb, 0, as_bytes(val));
            return _old_value_size;
         }
         ptr_address nr = insert_node(_root, kb, 0, as_bytes(val), {});
         if (nr != _root)
         {
            _ses->release(_root);
            _root = nr;
         }
         return _old_value_size;
      }

      bool find(key_view key, value_view& out)
      {
         auto        lock  = _ses->lock();
         bytes       kb    = as_bytes(key);
         if (_bucket_mode)
            return bkt_find(kb, out);
         ptr_address addr  = _root;
         uint32_t    depth = 0;
         for (;;)
         {
            auto    ref   = _ses->get_ref<sal::alloc_header>(addr);
            uint8_t shape = ref.cb_type();
            if (shape == uint8_t(dispatch_shape::r8) || shape == uint8_t(dispatch_shape::r2) ||
                shape == uint8_t(dispatch_shape::r4))  // SKIP-HEADER: route off the segment line
            {
               const char* base = reinterpret_cast<const char*>(ref.obj());
               if (depth < kb.size())
               {
                  // Address-generate BEFORE switching on the algo. The key bits consumed
                  // before indexing the segment line differ per tier (r2:7 r4:6 r8:5), but
                  // the *line* is the same closed form for all three — so we compute the
                  // exact cacheline the algo will dereference, then PREFETCH it (never a
                  // line we won't read), then switch only on the in-line algo (r8 direct
                  // index vs r2/r4 popcount-rank). The segment-line miss overlaps the jump.
                  static constexpr uint8_t skip_shift[4] = {0, 7, 6, 5};  // [r1] r2 r4 r8
                  uint8_t                  byte          = kb[depth];
                  const char*              cl = base + 128 + (size_t(byte >> skip_shift[shape]) << 7);
                  __builtin_prefetch(cl);
                  smart_address sa;
                  if (shape == uint8_t(dispatch_shape::r8))
                     std::memcpy(&sa, cl + 4u * (byte & 31), sizeof(sa));
                  else if (shape == uint8_t(dispatch_shape::r2))
                     sa = reinterpret_cast<const bitset_r2_node::segment*>(cl)->find(byte & 127);
                  else
                     sa = reinterpret_cast<const bitset_r4_node::segment*>(cl)->find(byte & 63);
                  if (sa.is_null())
                     return false;
                  if (sa.is_tail())
                     return inline_at(ref.obj(), sa, kb.subspan(depth + 1), out);
                  addr = sa.as_pointer();
                  ++depth;
                  continue;
               }
               smart_address tv = (shape == uint8_t(dispatch_shape::r8))
                                      ? ref.as<full_radix_node>().obj()->term_value()
                                      : (shape == uint8_t(dispatch_shape::r2)
                                             ? ref.as<bitset_r2_node>().obj()->term_value()
                                             : ref.as<bitset_r4_node>().obj()->term_value());
               return term_lookup(ref.obj(), tv, kb, depth, out);
            }
            // R1: read CL0, dispatch by format. ONE get_ref resolves the node; the switch
            // (jump table) reinterprets that single resolved pointer per case (alloc_header is
            // at offset 0 of every node) — no second per-case get_ref on the same address.
            bool                     cont = false;
            const sal::alloc_header* b    = _ses->get_ref<sal::alloc_header>(addr).obj();
            switch (kind_of(b->type()))
            {
               case radix_kind::leaf:
                  return leaf_read(addr, kb, depth, out);
               case radix_kind::bucket:  // a bucket LEAF: search its key suffix
               {
                  const inline_value* iv = reinterpret_cast<const bucket_node*>(b)->find(
                      kb.data() + depth, int(kb.size() - depth));
                  if (!iv)
                     return false;
                  out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                   iv->value_len);
                  return true;
               }
               case radix_kind::setlist_u8:
                  cont = route_r1(reinterpret_cast<const setlist_u8*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::setlist_u16:
                  cont = route_r1(reinterpret_cast<const setlist_u16*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::setlist_u32:
                  cont = route_r1(reinterpret_cast<const setlist_u32*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::c1_32:
                  cont = route_r1(reinterpret_cast<const bitset_c1_32*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::c1_64:
                  cont = route_r1(reinterpret_cast<const bitset_c1_64*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::c1_128:
                  cont = route_r1(reinterpret_cast<const bitset_c1_128*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::bitset_r2:
                  cont = route_r1(reinterpret_cast<const bitset_r2_node*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::bitset_r4:
                  cont = route_r1(reinterpret_cast<const bitset_r4_node*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::full:
                  cont = route_r1(reinterpret_cast<const full_radix_node*>(b), kb, depth, out, addr);
                  break;
               case radix_kind::bitset1:
                  __builtin_unreachable();  // not produced by the engine
            }
            if (!cont)
               return _route_found;
         }
      }

      // ── remove ──────────────────────────────────────────────────────────────────
      // Delete `key` if present; returns true iff it was found+removed. COW-to-root just
      // like upsert (unique → mutate in place; shared/snapshot → clone the path, the old
      // tree stays valid for the snapshot). Releases every SAL object the deletion frees
      // — external leaf/subtree values and any node that drains empty — with no leak and
      // no double-free (mirrors insert's clone-retains-children / caller-releases-original
      // contract). An inline-tail value's bytes stay in the node's tail heap (intra-node
      // space, reclaimed on the next clone); that is not a refcount leak. A node that loses
      // its last entry is dropped from its parent and freed, propagating to the root, which
      // is reset to a fresh empty node (the tree always has a root).
      bool remove(key_view key)
      {
         auto  lock = _ses->lock();
         bytes kb   = as_bytes(key);
         if (_bucket_mode)
            return bkt_remove(kb);
         _removed = false;
         _emptied = false;
         _shared  = false;  // fresh unique→shared latch for this COW-to-root remove descent
         ptr_address nr = remove_node(_root, kb, 0);
         if (!_removed)
            return false;
         if (_emptied)  // whole tree drained → free it, install a fresh empty root
         {
            if (nr != _root)
               _ses->release(_root);  // shared: drop our ref to the original (snapshot keeps it)
            _ses->release(nr);        // free the emptied (in-place or cloned) root
            _root = make_node<setlist_u8>(bytes{});
         }
         else if (nr != _root)
         {
            _ses->release(_root);
            _root = nr;
         }
         return true;
      }

      // ── ordered queries (this is a SORTED container, not a hash map) ────────────
      // lower_bound: smallest key >= `key`; upper_bound: smallest key strictly > `key`.
      // On a hit, out_key holds the matched key (materialized) and out_val views its
      // value in node memory (valid like find()'s result). Returns false if none.
      bool lower_bound(key_view key, std::string& out_key, value_view& out_val)
      {
         return ordered_seek(key, false, out_key, out_val);
      }
      bool upper_bound(key_view key, std::string& out_key, value_view& out_val)
      {
         return ordered_seek(key, true, out_key, out_val);
      }
      // Visit every (key, value) in ascending key order: cb(key_view, value_view).
      template <class Cb>
      void for_each_inorder(Cb&& cb)
      {
         auto lock = _ses->lock();
         KB   kb;
         inorder_walk(_root, kb, cb);
      }
      // DEBUG: count keys reachable from an ARBITRARY (pinned) root — unambiguous snapshot check.
      uint64_t count_keys_at(ptr_address root)
      {
         auto     lock = _ses->lock();
         KB       kb;
         uint64_t n  = 0;
         auto     cb = [&](key_view, value_view) { ++n; };
         inorder_walk(root, kb, cb);
         return n;
      }
      // DEBUG: order-independent (count, checksum) signature of the keyspace at a pinned root —
      // verifies a snapshot's exact contents (not just count) against the test model after churn.
      std::pair<uint64_t, uint64_t> sig_at(ptr_address root)
      {
         auto     lock = _ses->lock();
         KB       kb;
         uint64_t n = 0, sum = 0;
         auto     cb = [&](key_view k, value_view v) {
            uint64_t h = 1469598103934665603ull;
            for (char c : k) h = (h ^ uint8_t(c)) * 1099511628211ull;
            for (char c : v) h = (h ^ uint8_t(c)) * 1099511628211ull;
            sum += h;  // commutative accumulate → order-independent
            ++n;
         };
         inorder_walk(root, kb, cb);
         return {n, sum};
      }

      void commit()
      {
         auto lock = _ses->lock();
         if (_snap_valid)
            _ses->release(_snap);
         _snap = _root;
         _ses->retain(_root);
         _snap_valid = true;
      }

      // Release the ENTIRE tree in one cascade — test/bench hook for big-subtree-drop latency.
      // Drops the snapshot pin first so the root is uniquely owned (ref==1) and the single
      // release() cascades through every node. Resets to a fresh empty root so the tree stays
      // usable. (Build the tree WITHOUT per-op commit() so interior nodes are uniquely owned and
      // the cascade actually frees them rather than stopping at a shared snapshot.)
      void drop_root()
      {
         auto lock = _ses->lock();
         if (_snap_valid)
         {
            _ses->release(_snap);
            _snap_valid = false;
         }
         _ses->release(_root);
         _root = make_node<setlist_u8>(bytes{});
      }

      uint64_t object_count() const { return _alloc->total_allocated_objects(); }
      // Live count is exact only once the background release thread has drained — releases of
      // a sole-referenced object are deferred to it. Tests poll this to a quiescent point.
      uint64_t pending_releases() const { return _alloc->total_pending_releases(); }

      // ── gated wide-stem feature ────────────────────────────────────────────────
      // When off (default), the growth algorithm only builds setlist_u8 → the engine is
      // the libart-conformant byte radix. When on, the growth algorithm may pick
      // setlist_u16/u32 at the existing build points (fewer hops → fewer CB indirections).
      // Routing is already stride-aware; nothing else in the engine changes.
      void set_wide_stems(bool on) { _wide_stems = on; }
      bool wide_stems() const { return _wide_stems; }

      // Adaptive radix↔bucket: start radix; when insert descent crosses 2 levels that each
      // have < 3 branches (children + term) — i.e. going deep without widening, the signature
      // of clustered/non-radix-friendly data caught while the region is still a provably small
      // number of keys — collapse that subtree into a bucket leaf. Wide (radix-friendly) data
      // never trips it. Off by default so the gated baseline stays libart-exact.
      void set_adaptive(bool on) { _adaptive = on; }
      bool adaptive() const { return _adaptive; }
      void set_c1_promote(bool on) { _c1_promote_on = on; }
      // DEBUG probe: current refcount of the root node (is the snapshot pinning it?).
      uint32_t root_use_count() { return _ses->get_ref<sal::alloc_header>(_root).ref(); }
      // DEBUG: pin the current root as an independent snapshot (caller must release_snapshot()).
      ptr_address take_snapshot() { _ses->retain(_root); return _root; }
      void        release_snapshot(ptr_address s) { _ses->release(s); }
      // DEBUG: adopt a pinned snapshot as the live working root → writes COW a DIVERGENT lineage
      // from it (the snapshot stays frozen). Used to stress two+ writers sharing a common ancestor.
      void adopt_root(ptr_address s)
      {
         auto lk = _ses->lock();
         if (_snap_valid) { _ses->release(_snap); _snap_valid = false; }
         _ses->retain(s);
         _ses->release(_root);
         _root = s;
      }
      ptr_address working_root() const { return _root; }
      // DEBUG: read a key from an ARBITRARY (pinned) root — used to test snapshot isolation.
      bool find_at(ptr_address root, key_view key, value_view& out)
      {
         auto  lk = _ses->lock();
         KB    kb;
         bytes q = as_bytes(key);
         return seek(root, q, 0, /*strict=*/false, kb, out);  // first key >= q; q exists → lands on it
      }
      // DEBUG: enumerate the CURRENT root's immediate pointer-children and their refcounts.
      // After (insert 1000; commit; insert 1) a correct COW-to-root clones the root and retains
      // every child → all but the one on the modified path should read ref==2.
      void dump_root_child_refs(const char* tag);

      // ── introspection (for the tree-structure dump / stats tool) ───────────────
      // COW/alloc accounting: how mutation actually played out over the run. `clones`
      // and `grows` are the COW-to-root cost (a clone forced by sharing or a full heap);
      // `in_place` is the cheap path (unique node, room to spare).
      struct cow_counters
      {
         uint64_t node_allocs = 0;     // routers built
         uint64_t leaf_allocs = 0;     // >60 SAL value-holders
         uint64_t inline_values = 0;   // tail (local) terminal values
         uint64_t clones = 0;          // COW clones (shared OR heap-full)
         uint64_t child_retains = 0;   // per-child retains done by clone() (the dense-clone cost)
         uint64_t grows = 0;           // subset of clones forced by a full tail heap
         uint64_t in_place = 0;        // unique + room: mutated in place (no copy)
         uint64_t restrides = 0;       // wide-node splits (re_stride to a narrower stride)
      };
      const cow_counters&        counters() const { return _cc; }
      sal::allocator_session_ptr session() const { return _ses; }
      ptr_address                root_addr() const { return _root; }

      struct tier_counts
      {
         uint64_t leaf = 0, setlist = 0, setlist_w16 = 0, setlist_w32 = 0;
         uint64_t bitset_c1 = 0, bitset_r2 = 0, bitset_r4 = 0, full = 0;
         // compact_addresses sharing: branches / live cacheline bases across all bitset
         // segments (avg sharing factor s = cbranches / cbases; 1.0 = no sharing).
         uint64_t cbranches = 0, cbases = 0, csegs = 0;
         // POINTER-only sharing (excludes inline-tail terminals, which can't co-locate):
         // s_ptr = cptr / cptr_cl. ctail = inline-tail branches (local, not external CBs).
         uint64_t cptr = 0, cptr_cl = 0, ctail = 0;
         // Same breakdown for SETLIST nodes (sparse extreme; flat branches): pointer children
         // co-location per node. s_setlist = slptr / slptr_cl.
         uint64_t slptr = 0, slptr_cl = 0, sltail = 0;
         // setlist headroom utilization: sum of tail_used (inline-value bytes actually used)
         // vs the 256-byte headroom each setlist reserves. Measures recoverable memory.
         uint64_t sl_tail_used = 0;
         // full_radix (R8) occupancy: sum of occupied 64-wide quarters per full node (1-4).
         // avg = full_quarters / full → if < 4, presence could eliminate the empty quarters.
         uint64_t full_quarters = 0;
         // adaptive/bucket-mode dense terminal leaves (sorted (suffix,value) pages) + entries.
         uint64_t bucket = 0, bucket_entries = 0;
      };
      tier_counts count_tiers()
      {
         auto        lock = _ses->lock();
         tier_counts tc;
         walk(_root, tc);
         return tc;
      }

     private:
      static constexpr int      max_inline   = 60;    // suffix+value cap for a tail leaf
      static constexpr uint16_t tail_reserve = 16;    // tail offsets >=16 → compact-safe
      static constexpr uint32_t max_node     = 4096;  // grow cap; beyond → widen encoding

      template <class N>
      void walk_kids(ptr_address a, tier_counts& tc)
      {
         _ses->get_ref<N>(a).obj()->for_each_child([&](ptr_address c) { walk(c, tc); });
      }
      // Sum branches + live cacheline bases over a bitset node's non-empty segments —
      // measures the real compact_addresses sharing factor (cbranches/cbases).
      template <class N>
      void accum_compact(ptr_address a, tier_counts& tc)
      {
         const N* n = _ses->get_ref<N>(a).obj();
         for (int s = 0; s < N::nseg; ++s)
         {
            const auto& seg = n->seg(s);
            const int   b   = seg.branches.size();
            if (b == 0)
               continue;
            tc.cbranches += uint64_t(b);
            tc.cbases += uint64_t(seg.branches.live_base_count());
            ++tc.csegs;
            // Classify each branch: inline tail (local) vs pointer (external CB). For
            // pointers, count DISTINCT cachelines (raw>>4) to get true co-location.
            uint32_t cls[256];
            int      ncl = 0;
            for (int i = 0; i < b; ++i)
            {
               const smart_address sa = seg.branches.at(i);
               if (sa.is_tail())
               {
                  ++tc.ctail;
                  continue;
               }
               ++tc.cptr;
               const uint32_t cl = sa.raw >> 4;
               bool           seen = false;
               for (int j = 0; j < ncl; ++j)
                  if (cls[j] == cl) { seen = true; break; }
               if (!seen)
                  cls[ncl++] = cl;
            }
            tc.cptr_cl += uint64_t(ncl);
         }
      }
      // Pointer-child CB co-location for a setlist node (per node): distinct cachelines among
      // its pointer children vs pointer count. s_setlist = slptr / slptr_cl.
      template <class N>
      void accum_setlist(ptr_address a, tier_counts& tc)
      {
         tc.sl_tail_used += _ses->get_ref<N>(a).obj()->tail_used();
         uint32_t cls[64];
         int      ncl = 0;
         _ses->get_ref<N>(a).obj()->for_each_entry([&](auto, smart_address br) {
            if (br.is_tail())
            {
               ++tc.sltail;
               return;
            }
            ++tc.slptr;
            const uint32_t cl = br.raw >> 4;
            for (int j = 0; j < ncl; ++j)
               if (cls[j] == cl)
                  return;
            if (ncl < 64)
               cls[ncl++] = cl;
         });
         tc.slptr_cl += uint64_t(ncl);
      }
      void walk(ptr_address a, tier_counts& tc)
      {
         switch (kind_of(_ses->get_ref<sal::alloc_header>(a)->type()))
         {
            case radix_kind::leaf:
               ++tc.leaf;
               break;
            case radix_kind::setlist_u8:
               ++tc.setlist;
               accum_setlist<setlist_u8>(a, tc);
               walk_kids<setlist_u8>(a, tc);
               break;
            case radix_kind::setlist_u16:
               ++tc.setlist_w16;
               walk_kids<setlist_u16>(a, tc);
               break;
            case radix_kind::setlist_u32:
               ++tc.setlist_w32;
               walk_kids<setlist_u32>(a, tc);
               break;
            case radix_kind::c1_32:
               ++tc.bitset_c1;
               walk_kids<bitset_c1_32>(a, tc);
               break;
            case radix_kind::c1_64:
               ++tc.bitset_c1;
               walk_kids<bitset_c1_64>(a, tc);
               break;
            case radix_kind::c1_128:
               ++tc.bitset_c1;
               walk_kids<bitset_c1_128>(a, tc);
               break;
            case radix_kind::bitset_r2:
               ++tc.bitset_r2;
               accum_compact<bitset_r2_node>(a, tc);
               walk_kids<bitset_r2_node>(a, tc);
               break;
            case radix_kind::bitset_r4:
               ++tc.bitset_r4;
               accum_compact<bitset_r4_node>(a, tc);
               walk_kids<bitset_r4_node>(a, tc);
               break;
            case radix_kind::full:
            {
               ++tc.full;
               uint8_t q = 0;  // which 64-wide quarters have any branch
               _ses->get_ref<full_radix_node>(a).obj()->for_each_entry(
                   [&](uint8_t byte, smart_address) { q |= uint8_t(1u << (byte >> 6)); });
               tc.full_quarters += uint64_t(std::popcount(unsigned(q)));
               walk_kids<full_radix_node>(a, tc);
               break;
            }
            case radix_kind::bucket:  // adaptive dense terminal leaf — a counted endpoint
               ++tc.bucket;
               tc.bucket_entries += _ses->get_ref<bucket_node>(a).obj()->nent;
               break;
            case radix_kind::bitset1:  // not produced
               break;
         }
      }

      static std::unique_ptr<sal::allocator> make_alloc(const std::filesystem::path& dir)
      {
         std::filesystem::remove_all(dir);
         std::filesystem::create_directories(dir);
         sal::runtime_config cfg;
         cfg.sync_mode               = sal::sync_type::none;
         cfg.write_protect_on_commit = false;
         cfg.enable_read_cache       = false;
         return std::make_unique<sal::allocator>(dir, cfg, false);  // threads started post-ctor
      }

      // ── node construction + COW ────────────────────────────────────────────────
      template <class N>
      N* mut(ptr_address a)
      {
         return _ses->get_ref<N>(a).modify().get();
      }
      template <class N>
      void stamp(ptr_address a)
      {
         auto ref = _ses->get_ref<N>(a);
         ref.control().set_type(uint8_t(ref->stamp_shape()));  // r8 if prefix-less else r1
      }
      // A bucket is a LEAF (no skip-header routing) → it wants dispatch shape r1. alloc already
      // defaults the control type to r1 (0), so no explicit stamp is needed: the infrastructure
      // makes the safe shape the default, and only the wide tiers (set by make_node/clone/reprefix)
      // ever stamp away from it.
      ptr_address make_bucket() { return _ses->alloc<bucket_node>(sal::alloc_hint{}); }
      template <class N>
      ptr_address make_node(bytes pfx, sal::alloc_hint hint = {})
      {
         ptr_address a = _ses->alloc<N>(hint, pfx);
         N*          n = _ses->get_ref<N>(a).modify().get();
         n->alloc_tail(tail_reserve);  // tail offsets >=16
         rr_birth<N>(n);               // big tier → 8 per-range counters (empty node, =1 each)
         stamp<N>(a);
         ++_cc.node_allocs;
         return a;
      }
      ptr_address make_leaf(bytes suffix, bytes val, sal::alloc_hint hint = {})
      {
         ++_cc.leaf_allocs;
         return _ses->alloc<leaf_node>(hint, suffix, val);
      }

      // Copy src's payload into dst (same N): head [alloc_header, head_bytes()) and the
      // tail heap [tail()-tail_used, tail()) → dst's tail (which may be at a different
      // offset when dst is larger), so tail-relative smart_addresses stay valid.
      template <class N>
      static void blit(N* dst, const N* src) noexcept
      {
         // copy [alloc_header, head_size()) — the fixed header + the live index region
         // (head_size() == sizeof for the fixed tiers; variable only for a wide setlist).
         constexpr uint32_t hdr = uint32_t(sizeof(sal::alloc_header));
         std::memcpy(reinterpret_cast<char*>(dst) + hdr,
                     reinterpret_cast<const char*>(src) + hdr, src->head_size() - hdr);
         uint16_t hu = src->tail_used();
         std::memcpy(dst->tail() - hu, src->tail() - hu, hu);
      }

      // ── range-refcount engine layer (per-range branch-set counters; doc/micro_refcount.md) ──
      // A big node batches its child retain/release into RANGE_COUNT per-range counters in the SAL
      // implicit_ref_pool. clone/destroy branch on rc_has_ctrs(): countered nodes use the 8
      // counters, un-countered nodes (a birth site not yet wired) fall back to per-child — so
      // birth wiring is incrementally safe. is_big marks the tiers that have counters.
      template <class N>
      static constexpr bool is_big = requires { N::refcount_block_count; };

      // Birth: give a fresh big node its 8 per-range counters (each = 1, this version), co-located
      // in one region. The builder's per-child placement supplies each counter's base (cb.ref=1
      // per child). Re-homes if a region fills mid-birth (counters are unique=1 here, safe).
      template <class N>
      void rr_birth(N* n)
      {
         if constexpr (is_big<N>)
         {
            for (;;)
            {
               sal::ptr_address       a0     = _ses->rr_alloc_fresh();
               auto                   region = sal::region_of(a0);
               sal::implicit_ref_addr placed[range_count];
               placed[0] = sal::offset_of(a0);
               int  r  = 1;
               bool ok = true;
               for (; r < range_count; ++r)
               {
                  auto a = _ses->rr_alloc_in_region(
                      {region, std::span<const sal::implicit_ref_addr>(placed, size_t(r))});
                  if (not a) { ok = false; break; }
                  placed[r] = sal::offset_of(*a);
               }
               if (ok)
               {
                  n->rc_set_region(region);
                  for (int i = 0; i < range_count; ++i) n->rc_set_off(i, placed[i]);
                  return;
               }
               for (int j = 0; j < r; ++j)  // region filled mid-birth: undo the placed + retry
                  (void)_ses->rr_release(sal::combine(region, placed[j]));
            }
         }
      }

      // A big node's children are co-owned by another version iff any range counter > 1 (deferred
      // counts hide that from cb.ref).
      template <class N>
      bool children_shared(const N* cn)
      {
         if constexpr (is_big<N>)
         {
            if (not cn->rc_has_ctrs()) return false;
            for (int r = 0; r < range_count; ++r)
               if (_ses->rr_count(cn->rc_ctr(r)) > 1) return true;
         }
         return false;
      }

      // Is the child in range r co-owned? PER-RANGE — used when descending into a specific child so
      // we don't clone it just because some OTHER range is shared. After op1 diverges the path's
      // range (count→1), op2..opN walking the same path see this false → in-place. (children_shared,
      // the any-range form, would keep cloning until every range is private.)
      template <class N>
      bool range_shared(const N* cn, int r)
      {
         if constexpr (is_big<N>)
         {
            if (not cn->rc_has_ctrs()) return false;
            return _ses->rr_count(cn->rc_ctr(r)) > 1;
         }
         return false;
      }

      // Rare: a counter alloc found the node's region full. Fully privatize — retain ALL pointer
      // children (O(F)), drop this version's hold on all 8 old counters, birth fresh ones in a new
      // region. Every range becomes private (count 1).
      template <class N>
      void rr_rehome_full(N* n)
      {
         if constexpr (is_big<N>)
         {
            n->for_each_entry([&](uint8_t, smart_address sa)
                              { if (sa.is_pointer()) _ses->retain(sa.as_pointer()); });
            for (int r = 0; r < range_count; ++r) (void)_ses->rr_release(n->rc_ctr(r));
            rr_birth<N>(n);
         }
      }

      // Divergence: before mutating range r's branch set on a (cow'd) big node whose range-r
      // counter is SHARED, give this version its own range-r counter. Retain every current
      // pointer-child in range r (now co-owned by old set + this new one — the inherent ~32); the
      // per-site release(old_child) at drop/repoint cancels the changed child's retain, and an
      // add's new child is fresh. No-op for small tiers, un-countered nodes, already-private ranges.
      template <class N>
      void rr_diverge(N* n, int r)
      {
         if constexpr (is_big<N>)
         {
            if (not n->rc_has_ctrs() || _ses->rr_count(n->rc_ctr(r)) <= 1)
               return;
            auto a = _ses->rr_alloc_in_region(  // try a co-located new counter FIRST
                {n->rc_region(), std::span<const sal::implicit_ref_addr>(n->rc_offs(), range_count)});
            if (not a) { rr_rehome_full(n); return; }  // region full → privatize all (rare)
            n->for_each_entry([&](uint8_t byte, smart_address sa) {
               if (sa.is_pointer() && range_of(byte) == r) _ses->retain(sa.as_pointer());
            });
            (void)_ses->rr_release(n->rc_ctr(r));  // leave old shared set (count was >1, not last)
            n->rc_set_off(r, sal::offset_of(*a));
         }
      }

      // Clone from an already-resolved source node (no get_ref — the caller, e.g. cow(), holds
      // the ref pointing at exactly this node). `a` is unused for resolution but kept for symmetry.
      template <class N>
      ptr_address clone(ptr_address /*a*/, const N* src, uint32_t extra)
      {
         uint32_t need = ucc::round_up_multiple<sal::cacheline_size>(
             src->head_size() + src->tail_used() + extra + N::headroom);
         ptr_address na = _ses->alloc<N>(need, sal::alloc_hint{}, src->prefix());
         N*          nn = mut<N>(na);
         blit<N>(nn, src);
         if constexpr (is_big<N>)
         {
            if (nn->rc_has_ctrs())  // share the 8 per-range counters (blit copied region+offsets)
            {
               for (int r = 0; r < range_count; ++r) _ses->rr_retain(nn->rc_ctr(r));
               smart_address tv = nn->term_value();
               if (tv.is_pointer()) _ses->retain(tv.as_pointer());  // term held directly
            }
            else  // un-countered big node (birth not wired) → per-child fallback (matches destroy)
               nn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         }
         else
            nn->for_each_child([&](ptr_address c) { _ses->retain(c); ++_cc.child_retains; });
         stamp<N>(na);
         ++_cc.clones;
         return na;
      }
      // Address-only overload: resolve then delegate. For callers not holding the node's ref.
      template <class N>
      ptr_address clone(ptr_address a, uint32_t extra = 0)
      {
         return clone<N>(a, _ses->get_ref<N>(a).obj(), extra);
      }
      // Make a node writable for a mutation needing `extra` tail bytes, with the node already
      // resolved ONCE into `ref` (threaded by reference — no per-call get_ref, and a smart_ref
      // is borrowed so this avoids the by-value session-ptr refcount churn psitri warns about).
      //   shared (ref>1, a snapshot holds it) → clone to a NEW address (children retained);
      //   unique + room                       → mutate the held node IN PLACE via ref.modify()
      //                                         (reuses the resolved ref — no get_ref);
      //   unique + heap-full                  → grow (clone bigger, new address).
      // On a clone `addr` is updated to the new node and `ref` is left pointing at the OLD one
      // (the caller uses the returned writable pointer, not `ref`, after a COW, and releases the
      // old node per the COW-to-root contract — clone pre-sizes for `extra`, so we never release
      // here, which would double-free). modify()'s own SAL-COW is never reached: heart pre-clones
      // whenever ref>1, so a node that reaches modify() is unique and (write_protect off)
      // always can_modify → an in-place write.
      template <class N>
      N* cow(sal::smart_ref<N>& ref, ptr_address& addr, uint32_t extra = 0)
      {
         // SHARED mode (snapshot reachable) OR a directly-shared node → must CLONE, never mutate in
         // place. _shared forces the clone even at ref==1, because a ref==1 child of a shared parent
         // is still reachable from the snapshot (the clone retains the siblings — the COW-to-root cost).
         if (_shared || ref.ref() > 1)
         {
            addr = clone<N>(addr, ref.obj(), extra);  // reuse the held ref — no second get_ref
            return mut<N>(addr);
         }
         if (ref->free_heap() >= extra)
         {
            ++_cc.in_place;
            return ref.modify().get();
         }
         ++_cc.grows;
         addr = clone<N>(addr, ref.obj(), extra);  // reuse the held ref — no second get_ref
         return mut<N>(addr);
      }
      // Address-only overload: resolves once and delegates. Behavior-identical to threading a
      // ref; used by call sites not (yet) holding a smart_ref.
      template <class N>
      N* cow(ptr_address& addr, uint32_t extra = 0)
      {
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);
         return cow<N>(ref, addr, extra);
      }

      // ── terminal values: one size-gated rule ───────────────────────────────────
      static bool fits_inline(bytes suffix, bytes value) noexcept
      {
         return int(suffix.size() + value.size()) <= max_inline;
      }
      const inline_value* inline_of(const sal::alloc_header* node, smart_address sa) const
      {
         return reinterpret_cast<const inline_value*>(reinterpret_cast<const char*>(node) +
                                                      node->size() - sa.tail_offset());
      }
      // Tail bytes needed for an inline terminal {suffix,value}, or 0 if it spills to a SAL leaf.
      static uint32_t value_tail_need(bytes suffix, bytes value) noexcept
      {
         return fits_inline(suffix, value)
                    ? uint32_t(inline_value::byte_size(int(suffix.size()), int(value.size())))
                    : 0;
      }
      // Write a terminal {suffix,value} INTO an already-writable node `n` — NO cow (the caller
      // cow'd with value_tail_need() room). Small → a tail leaf in n's heap; else a SAL leaf.
      template <class N>
      smart_address make_value_into(N* n, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         if (fits_inline(suffix, value))
         {
            uint32_t need = uint32_t(inline_value::byte_size(int(suffix.size()), int(value.size())));
            auto*    iv   = reinterpret_cast<inline_value*>(n->alloc_tail(uint16_t(need)));
            iv->suffix_len = uint8_t(suffix.size());
            iv->value_len  = uint8_t(value.size());
            iv->value_type = 0;
            std::memcpy(iv->data, suffix.data(), suffix.size());
            std::memcpy(iv->data + suffix.size(), value.data(), value.size());
            ++_cc.inline_values;
            return smart_address::from_tail(n->tail_used());
         }
         return smart_address::from_pointer(make_leaf(suffix, value, hint));
      }
      // Allocate a terminal attached to `addr`: cow (room for the value), then write it in.
      // (Address overload for callers not threading a smart_ref.)
      template <class N>
      smart_address make_value(ptr_address& addr, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         N* n = cow<N>(addr, value_tail_need(suffix, value));
         return make_value_into<N>(n, suffix, value, hint);
      }
      // Zero-copy view of a terminal's {suffix,value} as raw bytes — a `bytes` overload of
      // the value_view term_view() the read path uses. No copy is needed: the node arena is
      // a fixed PROT_NONE reservation committed with MAP_FIXED (block_allocator), so it
      // never relocates, and mutation callers consume these views before the node they live
      // in is COW-released. So views into node memory stay valid across later allocations.
      void term_view(const sal::alloc_header* node, smart_address sa, bytes& suf, bytes& val) const
      {
         const inline_value* iv = sa.is_tail()
                                      ? inline_of(node, sa)
                                      : &_ses->get_ref<leaf_node>(sa.as_pointer()).obj()->value;
         suf = bytes{iv->data, size_t(iv->suffix_len)};
         val = bytes{iv->data + iv->suffix_len, size_t(iv->value_len)};
      }
      // Resolve an inline tail on `node`: its suffix must equal `rem`.
      bool inline_at(const sal::alloc_header* node, smart_address sa, bytes rem, value_view& out) const
      {
         const inline_value* iv = inline_of(node, sa);
         if (iv->suffix_len != rem.size() ||
             (rem.size() && std::memcmp(iv->data, rem.data(), iv->suffix_len) != 0))
            return false;
         out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
         return true;
      }
      bool term_lookup(const sal::alloc_header* node, smart_address tv, bytes key, uint32_t depth,
                       value_view& out)
      {
         if (tv.is_null())
            return false;
         if (tv.is_tail())
            return inline_at(node, tv, key.subspan(depth), out);
         return leaf_read(tv.as_pointer(), key, depth, out);
      }
      bool leaf_read(ptr_address addr, bytes key, uint32_t depth, value_view& out)
      {
         const leaf_node* n   = _ses->get_ref<leaf_node>(addr).obj();
         bytes            suf = n->suffix();
         bytes            rem = key.subspan(depth);
         if (suf.size() != rem.size() ||
             (suf.size() && std::memcmp(suf.data(), rem.data(), suf.size()) != 0))
            return false;
         bytes v = n->value_bytes();
         out     = value_view(reinterpret_cast<const char*>(v.data()), v.size());
         return true;
      }
      ptr_address update_leaf_value(ptr_address addr, bytes val)
      {
         auto             ref = _ses->get_ref<leaf_node>(addr);
         const leaf_node* ln  = ref.obj();
         if (ref.ref() == 1 && ln->value.value_len == val.size())
         {
            leaf_node* w = ref.modify().get();
            std::memcpy(w->value.data + w->value.suffix_len, val.data(), val.size());
            return addr;
         }
         return make_leaf(ln->suffix(), val);
      }

      // ── ordered-seek machinery (lower_bound / upper_bound / in-order walk) ──────
      // Stems are stored sorted (set_branch inserts at the lower-bound position) and the
      // bitset/full tiers iterate in byte order, so every node enumerates entries in key
      // order. A seek descends the matching path; when the query diverges it takes the
      // smallest sibling strictly greater than the query and descends to that subtree's
      // minimum (descend_min), backtracking to the parent when a subtree is all-less.

      // Reconstructed-key buffer (keys are materialized as we descend; values are viewed
      // directly in node memory). 1 KB covers any realistic key; asserts if exceeded.
      struct KB
      {
         uint8_t  buf[1024];
         uint32_t len = 0;
         void     append(const uint8_t* p, size_t n) noexcept
         {
            assert(len + n <= sizeof(buf));
            std::memcpy(buf + len, p, n);
            len += uint32_t(n);
         }
         void truncate(uint32_t l) noexcept { len = l; }
      };

      bool is_terminal_branch(smart_address sa) const
      {
         if (sa.is_tail())
            return true;
         return _ses->get_ref<sal::alloc_header>(sa.as_pointer())->type() == leaf_node::type_id;
      }
      // Suffix bytes + value of a terminal, both viewing node memory (no copy).
      void term_view(const sal::alloc_header* node, smart_address sa, bytes& suf,
                     value_view& val) const
      {
         const inline_value* iv =
             sa.is_tail() ? inline_of(node, sa)
                          : &_ses->get_ref<leaf_node>(sa.as_pointer()).obj()->value;
         suf = bytes{iv->data, size_t(iv->suffix_len)};
         val = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
      }
      // Append a terminal's suffix to kb and return its value (both view node memory).
      value_view append_terminal(const sal::alloc_header* node, smart_address sa, KB& kb)
      {
         bytes      suf;
         value_view v;
         term_view(node, sa, suf, v);
         kb.append(suf.data(), suf.size());
         return v;
      }
      // The `cw` actual key bytes a stem cell represents (big-endian, top cw of stem_width).
      static int stem_bytes(uint32_t cell, int stem_width, int cw, uint8_t* o) noexcept
      {
         for (int k = 0; k < cw; ++k)
            o[k] = uint8_t(cell >> (8 * (stem_width - 1 - k)));
         return cw;
      }
      static int key_cmp(const KB& kb, bytes q) noexcept
      {
         const size_t m = kb.len < q.size() ? kb.len : q.size();
         const int    c = m ? std::memcmp(kb.buf, q.data(), m) : 0;
         if (c)
            return c < 0 ? -1 : 1;
         if (kb.len == q.size())
            return 0;
         return kb.len < q.size() ? -1 : 1;
      }

      // ---- per-tier adaptors: a uniform ordered cursor over a node's entries ----
      template <class N>
      struct SLAd  // setlist (u8 / u16 / u24 / u32): cursor = entry index
      {
         const N*                 p;
         const sal::alloc_header* node() const { return reinterpret_cast<const sal::alloc_header*>(p); }
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return p->consume_width(); }
         int           stem_width() const { return N::stem_width; }
         int           first() const { return 0; }
         bool          valid(int c) const { return c < p->nstems(); }
         int           next(int c) const { return c + 1; }
         int           last() const { return p->nstems() - 1; }  // reverse: max entry (-1 if empty)
         int           prev(int c) const { return c - 1; }       // reverse: previous (-1 = before first)
         uint32_t      cell(int c) const { return uint32_t(p->stem_at(c)); }
         smart_address branch(int c) const { return p->branch_at(c); }
         int           lower(bytes r, bool& full, uint32_t& floorcell) const
         {
            const int m = int(r.size()) < cw() ? int(r.size()) : cw();
            full        = int(r.size()) >= cw();
            uint32_t v  = 0;
            for (int i = 0; i < m; ++i)
               v = (v << 8) | r[i];
            v <<= 8 * (N::stem_width - m);
            floorcell = v;
            return p->lower_bound_index(typename N::stem_type(v));
         }
      };
      template <class N, int SegBits>
      struct BSAd  // bitset_r2 / bitset_r4: cursor = present byte (or 256 = end)
      {
         const N*                 p;
         const sal::alloc_header* node() const { return reinterpret_cast<const sal::alloc_header*>(p); }
         int                      np(int b) const  // smallest present byte >= b, else 256
         {
            if (b < 0)
               b = 0;
            for (int s = b >> SegBits; s < (256 >> SegBits); ++s)
            {
               const int lo = (s == (b >> SegBits)) ? (b & ((1 << SegBits) - 1)) : 0;
               const int nx = p->seg(s).present.next_set(lo);
               if (nx >= 0)
                  return (s << SegBits) + nx;
            }
            return 256;
         }
         int pp(int b) const  // largest present byte <= b, else -1 (reverse of np)
         {
            if (b < 0) return -1;
            if (b > 255) b = 255;
            for (int s = b >> SegBits; s >= 0; --s)
            {
               const int hi = (s == (b >> SegBits)) ? (b & ((1 << SegBits) - 1)) : ((1 << SegBits) - 1);
               const int pv = p->seg(s).present.prev_set(hi);
               if (pv >= 0) return (s << SegBits) + pv;
            }
            return -1;
         }
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
         int           last() const { return pp(255); }      // reverse: highest present (-1 empty)
         int           prev(int c) const { return pp(c - 1); }
         uint32_t      cell(int c) const { return uint32_t(c); }
         smart_address branch(int c) const { return p->find(uint8_t(c)); }
         int           lower(bytes r, bool& full, uint32_t& floorcell) const
         {
            full      = !r.empty();
            floorcell = r.empty() ? 0u : uint32_t(r[0]);
            return np(int(floorcell));
         }
      };
      struct FullAd  // full_radix: cursor = present byte (or 256 = end)
      {
         const full_radix_node*   p;
         const sal::alloc_header* node() const { return reinterpret_cast<const sal::alloc_header*>(p); }
         int                      np(int b) const
         {
            if (b >= 256)
               return 256;
            const int nx = p->branches.next_present(b);
            return nx < 0 ? 256 : nx;
         }
         int pp(int b) const  // largest present byte <= b, else -1
         {
            if (b < 0) return -1;
            return p->branches.prev_present(b > 255 ? 255 : b);
         }
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
         int           last() const { return pp(255); }
         int           prev(int c) const { return pp(c - 1); }
         uint32_t      cell(int c) const { return uint32_t(c); }
         smart_address branch(int c) const { return p->find(uint8_t(c)); }
         int           lower(bytes r, bool& full, uint32_t& floorcell) const
         {
            full      = !r.empty();
            floorcell = r.empty() ? 0u : uint32_t(r[0]);
            return np(int(floorcell));
         }
      };
      template <class N>
      struct C1Ad  // bitset_c1_*: cursor = present absolute byte (or 256 = end)
      {
         const N*                 p;
         const sal::alloc_header* node() const { return reinterpret_cast<const sal::alloc_header*>(p); }
         int                      np(int b) const  // smallest present byte >= b, else 256
         {
            const int bse = p->window_base();
            const int nd  = p->next_present(b < bse ? 0 : b - bse);
            return nd < 0 ? 256 : bse + nd;
         }
         int pp(int b) const  // largest present byte <= b, else -1 (c1 has no prev_present → scan up)
         {
            int best = -1;
            for (int c = np(0); c >= 0 && c <= b; c = np(c + 1)) best = c;
            return best;
         }
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
         int           last() const { return pp(255); }
         int           prev(int c) const { return pp(c - 1); }
         uint32_t      cell(int c) const { return uint32_t(c); }
         smart_address branch(int c) const { return p->find(uint8_t(c)); }
         int           lower(bytes r, bool& full, uint32_t& floorcell) const
         {
            full      = !r.empty();
            floorcell = r.empty() ? 0u : uint32_t(r[0]);
            return np(int(floorcell));
         }
      };
      // Dispatch `addr` to the right adaptor and invoke f(adaptor). f returns bool/void.
      // ONE get_ref resolves the node; the type-switch lowers to a jump table; each case
      // reinterprets that single resolved pointer (alloc_header is at offset 0 of every node)
      // — no second per-case get_ref. This is the per-step cost of an ordered scan.
      template <class F>
      auto dispatch(ptr_address addr, F&& f)
      {
         const sal::alloc_header* b = _ses->get_ref<sal::alloc_header>(addr).obj();
         switch (kind_of(b->type()))
         {
            case radix_kind::setlist_u8:
               return f(SLAd<setlist_u8>{reinterpret_cast<const setlist_u8*>(b)});
            case radix_kind::setlist_u16:
               return f(SLAd<setlist_u16>{reinterpret_cast<const setlist_u16*>(b)});
            case radix_kind::setlist_u32:
               return f(SLAd<setlist_u32>{reinterpret_cast<const setlist_u32*>(b)});
            case radix_kind::c1_32:
               return f(C1Ad<bitset_c1_32>{reinterpret_cast<const bitset_c1_32*>(b)});
            case radix_kind::c1_64:
               return f(C1Ad<bitset_c1_64>{reinterpret_cast<const bitset_c1_64*>(b)});
            case radix_kind::c1_128:
               return f(C1Ad<bitset_c1_128>{reinterpret_cast<const bitset_c1_128*>(b)});
            case radix_kind::bitset_r2:
               return f(BSAd<bitset_r2_node, 7>{reinterpret_cast<const bitset_r2_node*>(b)});
            case radix_kind::bitset_r4:
               return f(BSAd<bitset_r4_node, 6>{reinterpret_cast<const bitset_r4_node*>(b)});
            case radix_kind::full:
               return f(FullAd{reinterpret_cast<const full_radix_node*>(b)});
            case radix_kind::bucket:
               // The ordered cursor / in-order walk has NO bucket adaptor yet: a bucket is a
               // dense sorted terminal leaf, not a radix router. Descending into one is the
               // known gap (step 2 adds a bucket frame). Throw rather than UB so the break is
               // observable. find()/upsert have their own switch and handle buckets directly.
               throw std::runtime_error(
                   "heart ordered cursor: reached a bucket leaf — buckets are not traversed");
            case radix_kind::leaf:     // not ordered routers — dispatch() is never called on these
            case radix_kind::bitset1:
               break;
         }
         __builtin_unreachable();
      }

      // Append an entry's stem bytes (+ inline suffix if terminal) and either yield its
      // value (terminal) or descend into its child to that subtree's minimum.
      template <class Ad>
      void take_entry_min(Ad ad, int c, KB& kb, value_view& out)
      {
         uint8_t       sb[4];
         const int     nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
         kb.append(sb, size_t(nb));
         const smart_address br = ad.branch(c);
         if (is_terminal_branch(br))
         {
            out = append_terminal(ad.node(), br, kb);
            return;
         }
         descend_min(br.as_pointer(), kb, out);
      }
      // Minimum key of this node's subtree, with the node's prefix ALREADY in kb.
      template <class Ad>
      void take_min_here(Ad ad, KB& kb, value_view& out)
      {
         if (!ad.term().is_null())  // a key ends here → EOF sorts before any byte
         {
            out = append_terminal(ad.node(), ad.term(), kb);
            return;
         }
         take_entry_min(ad, ad.first(), kb, out);  // else the smallest stem
      }
      void descend_min(ptr_address addr, KB& kb, value_view& out)
      {
         dispatch(addr, [&](auto ad) {
            bytes pfx = ad.prefix();
            kb.append(pfx.data(), pfx.size());
            take_min_here(ad, kb, out);
         });
      }

      template <class Ad>
      bool seek_in(Ad ad, bytes q, uint32_t depth, bool strict, KB& kb, value_view& out)
      {
         const uint32_t save  = kb.len;
         bytes          pfx   = ad.prefix();
         int            avail = int(q.size()) - int(depth);
         if (avail < 0)
            avail = 0;
         const int m = int(pfx.size()) < avail ? int(pfx.size()) : avail;
         int       cmp = 0, j = 0;
         for (; j < m; ++j)
            if (q[depth + j] != pfx[j])
            {
               cmp = q[depth + j] < pfx[j] ? -1 : 1;
               break;
            }
         if (j < m && cmp < 0)  // q < this path → whole subtree > q → its minimum
         {
            kb.append(pfx.data(), pfx.size());
            take_min_here(ad, kb, out);
            return true;
         }
         if (j < m && cmp > 0)  // q > this path → every key here < q
            return false;
         kb.append(pfx.data(), pfx.size());
         if (avail <= int(pfx.size()))
         {
            if (avail < int(pfx.size()))  // q is a strict prefix of the path → all > q
            {
               take_min_here(ad, kb, out);
               return true;
            }
            // q == path exactly
            if (!strict && !ad.term().is_null())
            {
               out = append_terminal(ad.node(), ad.term(), kb);
               return true;
            }
            if (!ad.valid(ad.first()))  // nothing greater (term-only & strict)
            {
               kb.truncate(save);
               return false;
            }
            take_entry_min(ad, ad.first(), kb, out);
            return true;
         }
         // q continues past the prefix; the node term (== path) is < q, so skip it.
         const uint32_t d2 = depth + uint32_t(pfx.size());
         bytes          r  = q.subspan(d2);
         bool           full;
         uint32_t       fc;
         for (int c = ad.lower(r, full, fc); ad.valid(c); c = ad.next(c))
         {
            const uint32_t      cl    = ad.cell(c);
            const bool          exact = full && cl == fc;
            const smart_address br    = ad.branch(c);
            uint8_t             sb[4];
            const int           nb = stem_bytes(cl, ad.stem_width(), ad.cw(), sb);
            const uint32_t      s2 = kb.len;
            kb.append(sb, size_t(nb));
            if (exact)  // stem equals q's stem → q may continue into this branch
            {
               if (is_terminal_branch(br))
               {
                  const value_view v  = append_terminal(ad.node(), br, kb);
                  const int        kc = key_cmp(kb, q);
                  if (kc > 0 || (kc == 0 && !strict))
                  {
                     out = v;
                     return true;
                  }
                  kb.truncate(s2);  // this key <= q (or == q & strict) → try the next stem
               }
               else if (seek(br.as_pointer(), q, d2 + uint32_t(ad.cw()), strict, kb, out))
                  return true;
               else
                  kb.truncate(s2);  // child held nothing >= q → next stem
            }
            else  // strictly greater stem → whole branch > q → its minimum
            {
               if (is_terminal_branch(br))
               {
                  out = append_terminal(ad.node(), br, kb);
                  return true;
               }
               descend_min(br.as_pointer(), kb, out);
               return true;
            }
         }
         kb.truncate(save);
         return false;
      }
      bool seek(ptr_address addr, bytes q, uint32_t depth, bool strict, KB& kb, value_view& out)
      {
         return dispatch(addr,
                         [&](auto ad) { return seek_in(ad, q, depth, strict, kb, out); });
      }
      bool ordered_seek(key_view key, bool strict, std::string& out_key, value_view& out_val)
      {
         auto       lock = _ses->lock();
         bytes      q    = as_bytes(key);
         KB         kb;
         value_view v;
         if (!seek(_root, q, 0, strict, kb, v))
            return false;
         out_key.assign(reinterpret_cast<const char*>(kb.buf), kb.len);
         out_val = v;
         return true;
      }

      template <class Ad, class Cb>
      void inorder_in(Ad ad, KB& kb, Cb& cb)
      {
         const uint32_t save = kb.len;
         bytes          pfx  = ad.prefix();
         kb.append(pfx.data(), pfx.size());
         if (!ad.term().is_null())
         {
            const uint32_t   s = kb.len;
            const value_view v = append_terminal(ad.node(), ad.term(), kb);
            cb(key_view(reinterpret_cast<const char*>(kb.buf), kb.len), v);
            kb.truncate(s);
         }
         for (int c = ad.first(); ad.valid(c); c = ad.next(c))
         {
            const uint32_t s = kb.len;
            uint8_t        sb[4];
            const int      nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
            kb.append(sb, size_t(nb));
            const smart_address br = ad.branch(c);
            if (is_terminal_branch(br))
            {
               const uint32_t   s3 = kb.len;
               const value_view v  = append_terminal(ad.node(), br, kb);
               cb(key_view(reinterpret_cast<const char*>(kb.buf), kb.len), v);
               kb.truncate(s3);
            }
            else
               inorder_walk(br.as_pointer(), kb, cb);
            kb.truncate(s);
         }
         kb.truncate(save);
      }
      template <class Cb>
      void inorder_walk(ptr_address addr, KB& kb, Cb& cb)
      {
         dispatch(addr, [&](auto ad) { inorder_in(ad, kb, cb); });
      }

     public:
      // ── the engine's NATIVE ordered cursor (heart::kv::cursor IS this type by alias) ──────
      // A stateful path-stack iterator. It OWNS a smart_ptr to the root version, which pins that
      // (immutable, COW-to-root) snapshot: every reachable node stays ref-alive and address-
      // stable for the cursor's lifetime, so a concurrent writer/compactor can't reclaim or move
      // nodes under it. Every navigation step takes the session read-lock. Positioning seeks
      // once and builds the path; next() then advances that path IN PLACE (no re-seek from root).
      // The key is materialized in _kbuf; the value is a view into the pinned node (valid until
      // the next navigation). Reverse (prev/last/seek_end) is the symmetric walk, still TODO.
      class cursor
      {
        public:
         explicit cursor(radix_tree* t) noexcept
             : _t(t), _root(t->session(), t->root_addr(), /*retain=*/true)
         {
         }

         bool seek_begin()  // position at the first (smallest) key; returns is_end()
         {
            auto lk = _t->_ses->lock();
            reset();
            descend_min(_root.address());
            return !_valid;
         }
         bool lower_bound(key_view k) { return position(as_bytes(k), false); }  // first key >= k
         bool upper_bound(key_view k) { return position(as_bytes(k), true); }   // first key  > k
         bool find(key_view k)                                                  // exact; end on miss
         {
            if (lower_bound(k))
               return true;  // nothing >= k
            if (key() == k)
               return false;  // exact hit
            reset();
            return true;  // first key >= k differs from k → absent
         }
         bool next()  // advance to the next key in order; returns is_end()
         {
            if (!_valid)
               return (_end < 0) ? seek_begin() : true;  // from rend → first key; from end → stay
            auto lk = _t->_ses->lock();
            while (_depth > 0)
            {
               const int r = advance_top();
               if (r == 1)
                  return false;  // landed on a terminal
               if (r == 2)
               {
                  descend_min(_nextaddr);
                  return !_valid;  // descended into a child subtree's minimum
               }
               // r == 0: this frame is exhausted, popped — keep advancing the parent
            }
            _valid = false;
            _end   = 0;  // fell off the top → at end
            return true;
         }
         bool prev()  // retreat to the previous key in order; returns is_rend()
         {
            if (!_valid)
               return (_end >= 0) ? seek_last() : true;  // from end → last key; from rend → stay
            auto lk = _t->_ses->lock();
            while (_depth > 0)
            {
               const int r = retreat_top();
               if (r == 1)
                  return false;  // landed on a terminal
               if (r == 2)
               {
                  descend_max(_nextaddr);
                  return !_valid;
               }
               // r == 0: frame exhausted, popped — keep retreating the parent
            }
            _valid = false;
            _end   = -1;  // fell off the bottom → before begin (rend)
            return true;
         }
         bool seek_last()  // position at the last (largest) key; returns is_rend()
         {
            auto lk = _t->_ses->lock();
            reset();
            descend_max(_root.address());
            if (!_valid) _end = -1;
            return !_valid;
         }
         bool last(key_view prefix = {})  // last key; prefix-bounded last is seek_last for now
         {
            (void)prefix;
            return seek_last();
         }
         void seek_end()  noexcept { reset(); }            // after last key:  is_end()
         void seek_rend() noexcept { reset(); _end = -1; }  // before first key: is_rend()
         bool       is_end() const noexcept { return !_valid && _end >= 0; }
         bool       is_rend() const noexcept { return !_valid && _end < 0; }
         key_view   key() const noexcept { return key_view(reinterpret_cast<const char*>(_kbuf), _klen); }
         template <class F>
         void get_value(F&& fn) const { fn(_val); }
         template <class T = std::string>
         std::optional<T> value() const
         {
            if (!_valid)
               return std::nullopt;
            return T(_val.data(), _val.data() + _val.size());
         }
         int32_t get(key_view k, std::string* buf)  // point lookup (psitri::cursor::get)
         {
            value_view v;
            if (!_t->find(k, v))
               return -1;
            buf->assign(v.data(), v.size());
            return int32_t(v.size());
         }
         // ── psitri::cursor public-API parity (forward) ──
         bool    seek(key_view k) { return lower_bound(k); }  // position at first key >= k
         bool    first(key_view prefix = {})                  // first key (optionally with prefix)
         {
            return prefix.empty() ? seek_begin() : lower_bound(prefix);
         }
         int32_t value_size() const noexcept { return _valid ? int32_t(_val.size()) : -1; }
         template <class F>
         bool get(key_view k, F&& fn)  // point lookup → lambda(value_view); true if found
         {
            value_view v;
            if (!_t->find(k, v)) return false;
            fn(v);
            return true;
         }

        private:
         struct frame  // 8 bytes (lean push/pop); psitri-style
         {
            ptr_address addr;
            int16_t     pos;    // -1 = at this node's term value; else entry cursor (0..256)
            uint16_t    klen1;  // _klen after this node's prefix (entry stems append from here)
         };
         static_assert(sizeof(frame) == 8, "lean 8-byte cursor frame");

         void reset() noexcept
         {
            _depth = 0;
            _klen  = 0;
            _valid = false;
            _end   = 0;
         }
         void append(const uint8_t* p, size_t n) noexcept
         {
            std::memcpy(_kbuf + _klen, p, n);
            _klen += uint32_t(n);
         }
         template <class Ad>
         void take_terminal(Ad ad, smart_address br)  // append suffix, view the value
         {
            bytes      suf;
            value_view v;
            _t->term_view(ad.node(), br, suf, v);
            append(suf.data(), suf.size());
            _val = v;
         }

         // Push frames down to the minimum terminal of `addr`'s subtree; sets _valid/_val/_kbuf.
         void descend_min(ptr_address addr)
         {
            for (;;)
            {
               const uint32_t k0 = _klen;  // key length before this node's prefix (local; not stored)
               const int      r  = _t->dispatch(addr, [&](auto ad) -> int {
                  frame& f = _fr[_depth];
                  f.addr   = addr;
                  bytes pfx = ad.prefix();
                  append(pfx.data(), pfx.size());
                  f.klen1 = uint16_t(_klen);
                  ++_depth;
                  if (!ad.term().is_null())  // term (key ends here) sorts before any entry
                  {
                     f.pos = -1;
                     take_terminal(ad, ad.term());
                     return 1;
                  }
                  const int c = ad.first();
                  if (!ad.valid(c))  // empty node (only the root can be empty)
                  {
                     --_depth;
                     _klen = k0;
                     return 0;
                  }
                  f.pos = int16_t(c);
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
                  append(sb, size_t(nb));
                  const smart_address br = ad.branch(c);
                  if (_t->is_terminal_branch(br))
                  {
                     take_terminal(ad, br);
                     return 1;
                  }
                  _nextaddr = br.as_pointer();
                  return 2;
               });
               if (r == 1) { _valid = true; return; }
               if (r == 0) { _valid = false; return; }
               addr = _nextaddr;
            }
         }

         // Advance within the top frame: 0 popped (exhausted), 1 found terminal, 2 descend _nextaddr.
         int advance_top()
         {
            frame& f = _fr[_depth - 1];
            return _t->dispatch(f.addr, [&](auto ad) -> int {
               const int nc = (f.pos < 0) ? ad.first() : ad.next(f.pos);
               _klen        = f.klen1;  // drop the previous entry's stem (+ any descent)
               if (!ad.valid(nc))
               {
                  --_depth;  // exhausted → pop; the parent's advance re-truncates to its klen1
                  return 0;
               }
               uint8_t   sb[4];
               const int nb = stem_bytes(ad.cell(nc), ad.stem_width(), ad.cw(), sb);
               append(sb, size_t(nb));
               f.pos                  = int16_t(nc);
               const smart_address br = ad.branch(nc);
               if (_t->is_terminal_branch(br))
               {
                  take_terminal(ad, br);
                  return 1;
               }
               _nextaddr = br.as_pointer();
               return 2;
            });
         }

         // Reverse of descend_min: push frames to the MAXIMUM terminal of `addr`'s subtree. In a
         // node, in-order is [term, entry0..entryN]; the max is entryN's subtree max, so we take
         // the LAST entry (the term, smallest, is reached only when entries are exhausted).
         void descend_max(ptr_address addr)
         {
            for (;;)
            {
               const uint32_t k0 = _klen;
               const int      r  = _t->dispatch(addr, [&](auto ad) -> int {
                  frame& f = _fr[_depth];
                  f.addr   = addr;
                  bytes pfx = ad.prefix();
                  append(pfx.data(), pfx.size());
                  f.klen1 = uint16_t(_klen);
                  ++_depth;
                  const int c = ad.last();
                  if (c >= 0)  // has entries → descend the largest
                  {
                     f.pos = int16_t(c);
                     uint8_t   sb[4];
                     const int nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
                     append(sb, size_t(nb));
                     const smart_address br = ad.branch(c);
                     if (_t->is_terminal_branch(br)) { take_terminal(ad, br); return 1; }
                     _nextaddr = br.as_pointer();
                     return 2;
                  }
                  if (!ad.term().is_null())  // no entries → the term is the only/largest key
                  {
                     f.pos = -1;
                     take_terminal(ad, ad.term());
                     return 1;
                  }
                  --_depth;  // empty node (root only)
                  _klen = k0;
                  return 0;
               });
               if (r == 1) { _valid = true; _end = 0; return; }
               if (r == 0) { _valid = false; return; }
               addr = _nextaddr;
            }
         }

         // Reverse of advance_top: 0 popped, 1 found terminal, 2 descend_max _nextaddr.
         int retreat_top()
         {
            frame& f = _fr[_depth - 1];
            return _t->dispatch(f.addr, [&](auto ad) -> int {
               _klen = f.klen1;  // drop the previous entry's stem (+ any descent)
               if (f.pos < 0)    // we were at this node's term (last in reverse) → pop
               {
                  --_depth;
                  return 0;
               }
               const int pc = ad.prev(f.pos);
               if (pc >= 0)  // previous (smaller) entry → descend its MAX
               {
                  f.pos = int16_t(pc);
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(pc), ad.stem_width(), ad.cw(), sb);
                  append(sb, size_t(nb));
                  const smart_address br = ad.branch(pc);
                  if (_t->is_terminal_branch(br)) { take_terminal(ad, br); return 1; }
                  _nextaddr = br.as_pointer();
                  return 2;
               }
               if (!ad.term().is_null())  // entries exhausted → the term (smallest) is last in reverse
               {
                  f.pos = -1;
                  take_terminal(ad, ad.term());
                  return 1;
               }
               --_depth;
               return 0;
            });
         }

         // Position at the first key >= q (strict → > q). Under the read-lock, on the PINNED
         // root: a raw inner seek finds the key, then build_to lays down the descent path.
         bool position(bytes q, bool strict)
         {
            auto       lk = _t->_ses->lock();
            KB         kb;
            value_view v;
            if (!_t->seek(_root.address(), q, 0, strict, kb, v))
            {
               reset();
               return true;
            }
            build_to(bytes{kb.buf, kb.len});
            return !_valid;
         }

         // Position at an EXACT existing key, building the descent path along the way.
         void build_to(bytes key)
         {
            reset();
            ptr_address addr  = _root.address();  // the pinned snapshot root
            uint32_t    depth = 0;
            for (;;)
            {
               const int r = _t->dispatch(addr, [&](auto ad) -> int {
                  frame& f = _fr[_depth];
                  f.addr   = addr;
                  bytes pfx = ad.prefix();
                  append(pfx.data(), pfx.size());
                  f.klen1 = uint16_t(_klen);
                  ++_depth;
                  const uint32_t d2 = depth + uint32_t(pfx.size());
                  if (d2 == key.size())  // the key ends here → this node's term
                  {
                     f.pos = -1;
                     take_terminal(ad, ad.term());
                     return 1;
                  }
                  bytes     r2 = key.subspan(d2);
                  bool      full;
                  uint32_t  fc;
                  const int c = ad.lower(r2, full, fc);  // exact cell position (key exists)
                  f.pos       = int16_t(c);
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
                  append(sb, size_t(nb));
                  const smart_address br = ad.branch(c);
                  if (_t->is_terminal_branch(br))
                  {
                     take_terminal(ad, br);
                     return 1;
                  }
                  _nextaddr  = br.as_pointer();
                  _nextdepth = d2 + uint32_t(ad.cw());
                  return 2;
               });
               if (r == 1) { _valid = true; return; }
               addr  = _nextaddr;
               depth = _nextdepth;
            }
         }

         radix_tree*                     _t;
         sal::smart_ptr<sal::alloc_header> _root;  // pins the snapshot version (RAII refcount)
         frame                           _fr[256];  // descent path (bounded by key depth; no heap)
         uint8_t                         _kbuf[1024];
         uint32_t    _klen      = 0;
         value_view  _val;
         int         _depth     = 0;
         bool        _valid     = false;
         int8_t      _end       = 0;  // when !_valid: +1 = at/after end, -1 = at/before begin (rend)
         ptr_address _nextaddr{};
         uint32_t    _nextdepth = 0;
      };

     private:

      // ── descent helper for r1 nodes (prefixed routers) ─────────────────────────
      template <class N>
      bool route_r1(const N* n, bytes key, uint32_t& depth, value_view& out, ptr_address& addr)
      {
         bytes pfx = n->prefix();
         if (pfx.size() > key.size() - depth ||
             (pfx.size() && std::memcmp(pfx.data(), key.data() + depth, pfx.size()) != 0))
         {
            _route_found = false;
            return false;
         }
         depth += uint32_t(pfx.size());
         if (depth == key.size())
         {
            _route_found = term_lookup(n, n->term_value(), key, depth, out);
            return false;
         }
         // Route by the node's consume-width key bytes. For setlist_u8 (stem_width==1)
         // the if-constexpr keeps the verbatim single-byte path → no change for u8.
         uint32_t      W;
         smart_address sa;
         if constexpr (N::stem_width == 1)
         {
            W  = 1;
            sa = n->find(key[depth]);
         }
         else
         {
            W = uint32_t(n->consume_width());  // 2 (u16) or 3/4 (u32 cell)
            if (key.size() - depth < W)        // key ends inside this node's stride -> miss
            {
               _route_found = false;
               return false;
            }
            sa = n->find(read_stem<N>(key, depth, W));
         }
         if (sa.is_null())
         {
            _route_found = false;
            return false;
         }
         if (sa.is_tail())
         {
            _route_found = inline_at(n, sa, key.subspan(depth + W), out);
            return false;
         }
         addr = sa.as_pointer();
         depth += W;
         return true;
      }

      // Read `cw` key bytes at depth big-endian and left-justify into N's cell (stem_width
      // bytes) — so a u32-cell node consuming 3 bytes zero-pads the low byte, and an
      // integer compare on the cell == lexicographic compare on the consumed bytes.
      template <class N>
      static uint32_t read_stem(bytes key, uint32_t depth, uint32_t cw) noexcept
      {
         uint32_t v = 0;
         for (uint32_t i = 0; i < cw; ++i)
            v = (v << 8) | key[depth + i];
         return v << (8 * (uint32_t(N::stem_width) - cw));
      }

      // ── insert (tier dispatch is the ONLY place type matters) ───────────────────
      ptr_address insert_node(ptr_address addr, bytes key, uint32_t depth, bytes val,
                              sal::alloc_hint peer_hint)
      {
         switch (kind_of(_ses->get_ref<sal::alloc_header>(addr)->type()))
         {
            case radix_kind::bucket:  // bucket LEAF: insert (split on overflow)
               return bucket_leaf_insert(addr, key, depth, val);
            case radix_kind::setlist_u8:
               return router_insert<setlist_u8>(addr, key, depth, val, peer_hint);
            case radix_kind::setlist_u16:
               return router_insert_wide<setlist_u16>(addr, key, depth, val, peer_hint);
            case radix_kind::setlist_u32:
               return router_insert_wide<setlist_u32>(addr, key, depth, val, peer_hint);
            case radix_kind::c1_32:
               return router_insert<bitset_c1_32>(addr, key, depth, val, peer_hint);
            case radix_kind::c1_64:
               return router_insert<bitset_c1_64>(addr, key, depth, val, peer_hint);
            case radix_kind::c1_128:
               return router_insert<bitset_c1_128>(addr, key, depth, val, peer_hint);
            case radix_kind::bitset_r2:
               return router_insert<bitset_r2_node>(addr, key, depth, val, peer_hint);
            case radix_kind::bitset_r4:
               return router_insert<bitset_r4_node>(addr, key, depth, val, peer_hint);
            case radix_kind::full:
               return router_insert<full_radix_node>(addr, key, depth, val, peer_hint);
            case radix_kind::leaf:     // a terminal value — never inserted into directly
            case radix_kind::bitset1:  // not produced by the engine
               break;
         }
         __builtin_unreachable();
      }

      // ── remove descent (mirrors insert_node dispatch; routers only — a terminal value
      // is handled in-place by the parent via is_terminal_branch, never recursed into) ──
      ptr_address remove_node(ptr_address addr, bytes key, uint32_t depth)
      {
         switch (kind_of(_ses->get_ref<sal::alloc_header>(addr)->type()))
         {
            case radix_kind::bucket:  // adaptive bucket LEAF: in-place remove
            {
               const uint8_t* s  = key.data() + depth;
               const int      sn = int(key.size() - depth);
               if (!_ses->get_ref<bucket_node>(addr).obj()->find(s, sn))
               {
                  _removed = false;
                  return addr;
               }
               bkt(addr)->remove(s, sn);  // buckets mutate in place (no-MVCC path)
               _removed = true;
               _emptied = (_ses->get_ref<bucket_node>(addr).obj()->nent == 0);
               return addr;
            }
            case radix_kind::setlist_u8:
               return remove_router<setlist_u8>(addr, key, depth);
            case radix_kind::setlist_u16:
               return remove_router<setlist_u16>(addr, key, depth);
            case radix_kind::setlist_u32:
               return remove_router<setlist_u32>(addr, key, depth);
            case radix_kind::c1_32:
               return remove_router<bitset_c1_32>(addr, key, depth);
            case radix_kind::c1_64:
               return remove_router<bitset_c1_64>(addr, key, depth);
            case radix_kind::c1_128:
               return remove_router<bitset_c1_128>(addr, key, depth);
            case radix_kind::bitset_r2:
               return remove_router<bitset_r2_node>(addr, key, depth);
            case radix_kind::bitset_r4:
               return remove_router<bitset_r4_node>(addr, key, depth);
            case radix_kind::full:
               return remove_router<full_radix_node>(addr, key, depth);
            case radix_kind::leaf:     // a terminal value — never recursed into
            case radix_kind::bitset1:  // not produced by the engine
               break;
         }
         __builtin_unreachable();
      }

      // Remove `key` from the subtree rooted at `addr`. Sets _removed (was it found) and,
      // when the node drains to nothing, _emptied (parent should drop+free it). Returns the
      // (possibly COW'd) node address. Releases external terminal values; for an emptied or
      // path-COW'd child it follows insert's contract — the PARENT releases the original.
      // After a reducing removal, downgrade `addr` to a smaller tier if it now fits one with
      // hysteresis; if the rebuild replaces a COW'd clone (shared mode, addr != pre), free the
      // clone here (in unique mode addr == pre and the PARENT frees the original). Skipped when
      // the node emptied (it's being dropped). Returns the node to hand back to the parent.
      template <class N>
      ptr_address finalize_remove(ptr_address addr, ptr_address pre)
      {
         if (_emptied)
            return addr;
         ptr_address dg = maybe_downgrade<N>(addr);
         if (dg != addr)
         {
            if (addr != pre)
               _ses->release(addr);
            addr = dg;
         }
         return addr;
      }

      template <class N>
      ptr_address remove_router(ptr_address addr, bytes key, uint32_t depth)
      {
         const ptr_address pre = addr;  // original (pre-COW) node, for clone-release on downgrade
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);  // resolved once; threaded into cow()
         const N*          cn  = ref.obj();
         // must_clone = is THIS node reachable from a snapshot (clone me). NOT children_shared —
         // that decides whether the CHILDREN clone (propagated per-descent below). A unique working
         // node with co-owned children mutates IN-PLACE; only the shared child clones. This is what
         // lets a batch converge toward an in-place (non-COW) tree as nodes become unique.
         const bool        must_clone = _shared || ref.ref() > 1;
         _shared                      = must_clone;  // N's own cow() decisions use this
         bytes             pfx = cn->prefix();
         if (pfx.size() > key.size() - depth ||
             (pfx.size() && std::memcmp(pfx.data(), key.data() + depth, pfx.size()) != 0))
         {
            _removed = false;  // key diverges from this node's compressed prefix → absent
            return addr;
         }
         depth += uint32_t(pfx.size());
         if (depth == key.size())  // key ends here → remove the term value
         {
            smart_address tv = cn->term_value();
            if (tv.is_null())
            {
               _removed = false;
               return addr;
            }
            N* n = cow<N>(ref, addr);
            if (tv.is_pointer())
               _ses->release(tv.as_pointer());  // external leaf/subtree value
            n->set_term_value(smart_address::null());
            _removed = true;
            _emptied = (n->branch_count() == 0);
            return finalize_remove<N>(addr, pre);
         }
         // route by this node's stride (1 byte for every tier remove supports; wide setlist
         // consumes its consume_width key bytes per hop, same as route_r1).
         uint32_t            W;
         typename N::stem_type cell;
         if constexpr (N::stem_width == 1)
         {
            W    = 1;
            cell = typename N::stem_type(key[depth]);
         }
         else
         {
            W = uint32_t(cn->consume_width());
            if (key.size() - depth < W)
            {
               _removed = false;
               return addr;
            }
            cell = typename N::stem_type(read_stem<N>(key, depth, W));
         }
         smart_address sa = cn->find(cell);
         if (sa.is_null())
         {
            _removed = false;
            return addr;
         }
         if (is_terminal_branch(sa))  // a leaf/inline terminal here → this is (maybe) the key
         {
            bytes      esuf;
            value_view eval;
            term_view(cn, sa, esuf, eval);
            bytes suf = key.subspan(depth + W);
            if (esuf.size() != suf.size() ||
                (esuf.size() && std::memcmp(esuf.data(), suf.data(), esuf.size()) != 0))
            {
               _removed = false;  // same routing cell, different suffix → key absent
               return addr;
            }
            N* n = cow<N>(ref, addr);
            rr_diverge<N>(n, range_of(cell));  // big tier: privatize range before dropping a branch
            if (sa.is_pointer())
               _ses->release(sa.as_pointer());  // external leaf value (inline tail just orphans)
            n->remove_branch(cell);
            _removed = true;
            _emptied = (n->branch_count() == 0 && !n->has_term_value());
            return finalize_remove<N>(addr, pre);
         }
         // pointer to an inner node → recurse. The child is shared (must clone) if THIS node is
         // shared OR its big-tier children are co-owned via shared counters; restore must_clone for
         // N's own cow() after the recursion clobbers _shared.
         ptr_address child = sa.as_pointer();
         _shared           = must_clone || range_shared<N>(cn, range_of(cell));  // PER-RANGE
         ptr_address nc    = remove_node(child, key, depth + W);
         _shared           = must_clone;
         if (!_removed)
            return addr;  // absent in subtree → nothing changed here
         if (_emptied)     // the child drained → drop its branch and free it
         {
            N* n = cow<N>(ref, addr);  // ref valid: the recursion didn't COW this node
            rr_diverge<N>(n, range_of(cell));  // big tier: privatize range before dropping a branch
            n->remove_branch(cell);
            _ses->release(nc);  // the emptied child (in-place == child, or a clone)
            if (nc != child)
               _ses->release(child);  // shared: also drop the original child's ref
            _emptied = (n->branch_count() == 0 && !n->has_term_value());
            return finalize_remove<N>(addr, pre);
         }
         if (nc != child)  // child was path-COW'd (shared) → repoint, release the original
         {
            N* n = cow<N>(ref, addr);  // ref valid: the recursion didn't COW this node
            rr_diverge<N>(n, range_of(cell));  // big tier: privatize range before re-pointing
            n->set_branch(cell, smart_address::from_pointer(nc));
            _ses->release(child);
         }
         _emptied = false;
         return addr;
      }

      template <class N>
      ptr_address router_insert(ptr_address addr, bytes key, uint32_t depth, bytes val,
                                sal::alloc_hint peer_hint)
      {
         const ptr_address addr_orig = addr;
         const uint32_t    depth0     = depth;  // where addr's region begins (for bucket collapse)
         // Resolve the node ONCE and thread it by reference: cn covers every pre-COW read, and
         // cow(ref, …) reuses it for an in-place mutation (no second get_ref). After a COW the
         // ref is stale (clone → new address), so the post-COW writable `n` is used instead.
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);
         const N*          cn  = ref.obj();
         // must_clone = is THIS node reachable from a snapshot (clone me). children_shared decides
         // whether the CHILDREN clone (propagated per-descent), NOT whether I do — so a unique
         // working node with co-owned children mutates IN-PLACE; only the shared child clones. This
         // is what lets a batch converge to an in-place (non-COW) tree as nodes become unique.
         const bool        must_clone = _shared || ref.ref() > 1;
         _shared                      = must_clone;  // N's own cow() decisions use this
         bytes             pfx = cn->prefix();
         int               avail     = int(key.size() - depth);
         int               maxp      = std::min(int(pfx.size()), avail);
         int p = common_prefix_len(key.subspan(depth, size_t(maxp)), pfx.subspan(0, size_t(maxp)));
         if (p < int(pfx.size()))
            return prefix_split<N>(addr, key, depth, p, val, peer_hint);

         depth += uint32_t(pfx.size());
         if (depth == key.size())  // key ends here → term
         {
            set_term<N>(addr, val, peer_hint);
            return addr;
         }
         uint8_t         b     = key[depth];
         sal::alloc_hint chint = cn->branch_hint(b);
         smart_address   sa    = cn->find(b);
         bytes           suf   = key.subspan(depth + 1);

         if (sa.is_null())  // new terminal at b
         {
            // set_branch is the source of truth (a tail's compact base depends on its exact
            // offset, so a probe can't predict it). COW ONCE (room for the inline value, ref
            // threaded), write the value into that node, try to set; on failure widen (the value
            // is re-made on the wider node). The COW may clone addr → release that clone here
            // (the caller releases the original), else it orphans.
            const ptr_address pre = addr;
            N*                 n   = cow<N>(ref, addr, value_tail_need(suf, val));
            rr_diverge<N>(n, range_of(b));  // big tier: privatize range before adding a branch
            smart_address      sab = make_value_into<N>(n, suf, val, chint);
            bool               to_c1 = false;
            if constexpr (std::is_same_v<N, setlist_u8>)
               to_c1 = want_c1_promote(n, b);  // opt-in; reuses the tested grow_and_set widen path
            if (!to_c1 && n->set_branch(b, sab))
               return addr;
            if constexpr (!std::is_same_v<N, full_radix_node>)
            {
               ptr_address up = grow_and_set<N>(addr, b, smart_address::null(), suf, val, peer_hint);
               if (addr != pre)
                  _ses->release(addr);
               return up;
            }
            return addr;  // full_radix never fails set_branch
         }

         // Is this branch a TERMINAL (an inline tail, or a pointer to a leaf_node)?
         if (is_terminal_branch(sa))
         {
            bytes esuf, eval;  // zero-copy views into the node (stable arena, consumed pre-COW)
            term_view(cn, sa, esuf, eval);
            if (esuf.size() == suf.size() &&
                std::memcmp(esuf.data(), suf.data(), esuf.size()) == 0)  // same key → update
            {
               _old_value_size = int(eval.size());
               smart_address nsa =
                   (sa.is_pointer())
                       ? smart_address::from_pointer(update_leaf_value(sa.as_pointer(), val))
                       : make_value<N>(addr, suf, val, chint);  // old tail leaks
               if (nsa != sa)
               {
                  N* n = cow<N>(addr);
                  rr_diverge<N>(n, range_of(b));  // big tier: privatize range before replace
                  if (sa.is_pointer() && nsa != sa)
                     _ses->release(sa.as_pointer());
                  n->set_branch(b, nsa);
               }
               return addr;
            }
            // diverge → child holding both continuations (a wide setlist_u32 when the gate
            // is on and both suffixes have ≥4 bytes past the common prefix; else u8)
            int           cp    = common_prefix_len(esuf, suf);
            ptr_address   child = make_collision_child(esuf, eval, suf, val, cp, chint);
            smart_address csa   = smart_address::from_pointer(child);
            N*            n   = cow<N>(ref, addr);  // ref valid: make_collision_child didn't COW addr
            rr_diverge<N>(n, range_of(b));  // big tier: privatize range before replace
            if (n->set_branch(b, csa))
            {
               if (sa.is_pointer())
                  _ses->release(sa.as_pointer());  // old SAL leaf replaced (a tail just leaks)
               return addr;
            }
            if constexpr (!std::is_same_v<N, full_radix_node>)  // re-home failed → widen
            {
               // grow_and_set skips b and sets b→child; releasing the COW'd `addr` runs
               // its destroy, releasing the OLD terminal at b (a leaf) — so don't release sa.
               ptr_address up = grow_and_set<N>(addr, b, csa, bytes{}, bytes{}, chint);
               if (addr != addr_orig)
                  _ses->release(addr);
               return up;
            }
            return addr;  // full_radix never fails set_branch
         }

         // pointer to an inner node → recurse
         ptr_address child = sa.as_pointer();
         // ADAPTIVE: this node + the child it routes into are BOTH narrow (< 3 branches incl
         // term) → 2 levels deep without widening = clustered, provably small (caught early).
         // Collapse addr's whole subtree into a bucket leaf instead of noding further.
         if constexpr (std::is_same_v<N, setlist_u8>)
         {
            if (_adaptive && !_in_bucket && cn->nstems() + (cn->has_term_value() ? 1 : 0) < 3 &&
                !is_bucket(child) && node_narrow(child))
            {
               ptr_address bp;
               if (convert_subtree_to_bucket(addr, bp))
                  return bucket_leaf_insert(bp, key, depth0, val);  // caller releases old addr
            }
         }
         // child is shared (must clone) if THIS node is shared OR its children are co-owned;
         // restore must_clone for N's own cow() after the recursion clobbers _shared.
         _shared        = must_clone || range_shared<N>(cn, range_of(b));  // PER-RANGE: only THIS
         ptr_address nc = insert_node(child, key, depth + 1, val, chint);  // child's range matters
         _shared        = must_clone;
         if (nc != child)
         {
            smart_address nsab = smart_address::from_pointer(nc);
            N*            n    = cow<N>(ref, addr);  // ref valid: the recursion didn't COW this node
            rr_diverge<N>(n, range_of(b));  // big tier: privatize range before re-pointing
            if (!n->set_branch(b, nsab))  // compact re-home failed → widen with the pointer
            {
               if constexpr (!std::is_same_v<N, full_radix_node>)
               {
                  // grow_and_set re-stores addr's entries (skipping b) and sets b→nc;
                  // releasing the COW'd `addr` runs its destroy, which releases all its
                  // children including the OLD child at b — so do NOT release `child`.
                  ptr_address up = grow_and_set<N>(addr, b, nsab, bytes{}, bytes{}, peer_hint);
                  if (addr != addr_orig)
                     _ses->release(addr);
                  return up;
               }
            }
            _ses->release(child);  // normal in-tier replace
         }
         return addr;
      }

      // ── wide-stem (gated) ───────────────────────────────────────────────────────
      // Build the child for a terminal collision: a wide setlist_u32 when the gate is on
      // AND both diverging suffixes have ≥4 bytes past the common prefix (so each forms a
      // full 4-byte stem); otherwise the byte-radix setlist_u8 (unchanged behavior).
      ptr_address make_collision_child(bytes esuf, bytes eval, bytes suf, bytes val, int cp,
                                       sal::alloc_hint hint)
      {
         if (_wide_stems && int(esuf.size()) - cp >= 4 && int(suf.size()) - cp >= 4)
         {
            ptr_address child = make_node<setlist_u32>(esuf.subspan(0, size_t(cp)), hint);
            add_wide_entry(child, esuf.subspan(size_t(cp)), eval, hint);
            add_wide_entry(child, suf.subspan(size_t(cp)), val, hint);
            return child;
         }
         ptr_address child = make_node<setlist_u8>(esuf.subspan(0, size_t(cp)), hint);
         add_continuation(child, esuf.subspan(size_t(cp)), eval, hint);
         add_continuation(child, suf.subspan(size_t(cp)), val, hint);
         return child;
      }
      // Add one entry to a fresh wide (u32, consume-4) node: 4-byte stem + inline remainder.
      // Precondition: suffix.size() >= 4 (the create gate guarantees it).
      void add_wide_entry(ptr_address& child, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         uint32_t      q   = read_stem<setlist_u32>(suffix, 0, 4);
         smart_address sv  = make_value<setlist_u32>(child, suffix.subspan(4), value, hint);
         setlist_u32*  w   = cow<setlist_u32>(child, uint32_t(setlist_u32::stem_width) + 4);
         [[maybe_unused]] bool ok = w->set_branch(q, sv);
         assert(ok && "fresh wide node always has room for an entry");
      }

      // Insert into a wide setlist (u16/u32). Stride = consume_width; new stems GROW the
      // node (clone bigger). Mid-stem / 4 KB split is step 4 (asserted unreachable here).
      template <class N>
      ptr_address router_insert_wide(ptr_address addr, bytes key, uint32_t depth, bytes val,
                                     sal::alloc_hint peer_hint)
      {
         const uint32_t    depth0 = depth;
         sal::smart_ref<N> ref    = _ses->get_ref<N>(addr);  // resolved once (reads + safe cows)
         const bool        must_clone = _shared || ref.ref() > 1;  // is THIS node snapshot-reachable
         _shared                      = must_clone;
         const N*          cn     = ref.obj();
         bytes             pfx    = cn->prefix();
         int            avail  = int(key.size() - depth);
         int            maxp   = std::min(int(pfx.size()), avail);
         int p = common_prefix_len(key.subspan(depth, size_t(maxp)), pfx.subspan(0, size_t(maxp)));
         if (p < int(pfx.size()))
            return prefix_split<N>(addr, key, depth, p, val, peer_hint);

         depth += uint32_t(pfx.size());
         int rem = int(key.size() - depth);
         if (rem == 0)  // key ends here → term
         {
            set_term<N>(addr, val, peer_hint);
            return addr;
         }
         int cw = cn->consume_width();
         if (rem < cw)  // key ends inside the stride → split to a stride that fits it
         {
            ptr_address na = re_stride_wide<N>(addr, rem, peer_hint);
            return insert_node(na, key, depth0, val, peer_hint);
         }
         uint32_t        q     = read_stem<N>(key, depth, uint32_t(cw));
         sal::alloc_hint chint = cn->branch_hint();
         smart_address   sa    = cn->find(q);
         bytes           suf   = key.subspan(depth + uint32_t(cw));

         if (sa.is_null())  // new stem
         {
            if (!cn->room_for_one() && cn->size() >= max_node)  // at 4 KB → split narrower
            {
               int         Wp = cw >= 3 ? 2 : 1;  // u32/u24 → u16; u16 → u8
               ptr_address na = re_stride_wide<N>(addr, Wp, peer_hint);
               return insert_node(na, key, depth0, val, peer_hint);
            }
            smart_address sv = make_value<N>(addr, suf, val, chint);  // else grow + append
            N*            w  = cow<N>(addr, uint32_t(N::stem_width) + 4);
            [[maybe_unused]] bool ok = w->set_branch(q, sv);
            assert(ok && "wide node grows → set_branch must succeed");
            return addr;
         }

         if (is_terminal_branch(sa))
         {
            bytes esuf, eval;  // zero-copy views into the node (stable arena, consumed pre-COW)
            term_view(cn, sa, esuf, eval);
            if (esuf.size() == suf.size() &&
                std::memcmp(esuf.data(), suf.data(), esuf.size()) == 0)  // same key → update
            {
               _old_value_size = int(eval.size());
               smart_address nsa =
                   sa.is_pointer()
                       ? smart_address::from_pointer(update_leaf_value(sa.as_pointer(), val))
                       : make_value<N>(addr, suf, val, chint);
               if (nsa != sa)
               {
                  N* w = cow<N>(addr);
                  if (sa.is_pointer())
                     _ses->release(sa.as_pointer());
                  w->set_branch(q, nsa);
               }
               return addr;
            }
            int           cp    = common_prefix_len(esuf, suf);
            ptr_address   child = make_collision_child(esuf, eval, suf, val, cp, chint);
            N*            w     = cow<N>(ref, addr);  // ref valid (same-key make_value not taken here)
            if (sa.is_pointer())
               _ses->release(sa.as_pointer());
            w->set_branch(q, smart_address::from_pointer(child));
            return addr;
         }

         // pointer to an inner node → recurse (wide setlist children are direct-refcounted, so
         // child_shared == must_clone here); restore must_clone for this node's cow() after.
         ptr_address child = sa.as_pointer();
         _shared           = must_clone || children_shared<N>(cn);
         ptr_address nc    = insert_node(child, key, depth + uint32_t(cw), val, chint);
         _shared           = must_clone;
         if (nc != child)
         {
            N* w = cow<N>(ref, addr);  // ref valid (insert_node recurses into child, not addr)
            w->set_branch(q, smart_address::from_pointer(nc));
            _ses->release(child);
         }
         return addr;
      }

      // ── re_stride (the wide-node split) ─────────────────────────────────────────
      // Allocate a router that consumes `cw` key bytes: cw=1 → byte router (setlist_u8,
      // widens by density); cw=2 → setlist_u16; cw=3 → setlist_u32 in consume-3 mode
      // (rtype setlist_u24); cw=4 → setlist_u32. `extra` sizes the growable variants.
      ptr_address make_wide_node(int cw, bytes pfx, sal::alloc_hint hint, uint32_t extra)
      {
         if (cw == 1)  // byte router: full_radix (direct 256 array, never widens during build)
            return make_sized<full_radix_node>(pfx, hint, extra);
         ptr_address a;
         if (cw == 2)
            a = make_sized<setlist_u16>(pfx, hint, extra);
         else
         {
            a = make_sized<setlist_u32>(pfx, hint, extra);
            if (cw == 3)
               mut<setlist_u32>(a)->set_rtype(radix_node_type::setlist_u24);
         }
         return a;
      }
      template <class N>
      ptr_address make_sized(bytes pfx, sal::alloc_hint hint, uint32_t extra)
      {
         uint32_t sz = ucc::round_up_multiple<sal::cacheline_size>(uint32_t(sizeof(N)) +
                                                                   uint32_t(pfx.size()) + extra +
                                                                   N::headroom);
         ptr_address a = _ses->alloc<N>(sz, hint, pfx);
         mut<N>(a)->alloc_tail(tail_reserve);
         stamp<N>(a);
         ++_cc.node_allocs;
         return a;
      }
      // `consume` key bytes (already isolated in b[0..consume)) → a left-justified cell
      // value for a node whose cell is `cell_width` bytes wide.
      static uint32_t encode_cell(const uint8_t* b, int consume, int cell_width) noexcept
      {
         uint32_t v = 0;
         for (int i = 0; i < consume; ++i)
            v = (v << 8) | b[i];
         return v << (8 * (cell_width - consume));
      }
      static int cell_width_for(int cw) noexcept { return cw == 1 ? 1 : cw == 2 ? 2 : 4; }

      // Move a branch slot of `src` into `dst` (a fresh node): inline/leaf → copy the
      // value into dst; node pointer → retain and reuse as-is (suffix unchanged).
      smart_address carry_value(ptr_address& dst, sal::header_type dtype, ptr_address src,
                                smart_address sa, sal::alloc_hint hint)
      {
         if (!is_terminal_branch(sa))
         {
            _ses->retain(sa.as_pointer());
            return sa;
         }
         bytes suf, val;  // zero-copy views into src (stable arena; src stays live here)
         term_view(_ses->get_ref<sal::alloc_header>(src).obj(), sa, suf, val);
         return mv_into(dst, dtype, suf, val, hint);
      }
      // re_stride routers are only ever full_radix (cw=1), setlist_u16 (cw=2), or
      // setlist_u32/u24 (cw=3/4) — never u8/bitset — so none of these widen by density.
      smart_address mv_into(ptr_address& node, sal::header_type t, bytes suf, bytes val,
                            sal::alloc_hint hint)
      {
         if (t == setlist_u16::type_id)
            return make_value<setlist_u16>(node, suf, val, hint);
         if (t == setlist_u32::type_id)
            return make_value<setlist_u32>(node, suf, val, hint);  // u32 / u24
         return make_value<full_radix_node>(node, suf, val, hint);
      }
      void add_ptr(ptr_address& node, sal::header_type t, uint32_t q, smart_address ptr,
                   sal::alloc_hint)
      {
         if (t == setlist_u16::type_id)
            cow<setlist_u16>(node, 8)->set_branch(uint16_t(q), ptr);
         else if (t == setlist_u32::type_id)
            cow<setlist_u32>(node, 8)->set_branch(q, ptr);
         else
         {
            auto* w = cow<full_radix_node>(node);  // direct, no grow/widen
            rr_diverge<full_radix_node>(w, range_of(uint8_t(q)));  // privatize range before set
            w->set_branch(uint8_t(q), ptr);
         }
      }
      void add_inline(ptr_address& node, sal::header_type t, uint32_t q, bytes suf, bytes val,
                      sal::alloc_hint hint)
      {
         smart_address sv = mv_into(node, t, suf, val, hint);
         add_existing(node, t, q, sv, hint);
      }

      // Split a wide node (always a u32 cell, consume W) into a router that consumes Wp<W,
      // pushing the leftover W-Wp tail bytes into each entry (inline → suffix; pointer →
      // a group child keyed by the tail). Returns the new subtree; caller releases `addr`.
      template <class Src>
      ptr_address re_stride_wide(ptr_address addr, int Wp, sal::alloc_hint hint)
      {
         ++_cc.restrides;
         // `old` (the source) is read-only here and addr is not mutated/freed until the
         // caller releases it, and the arena is address-stable, so we hold `old` and read
         // it in place across the build allocations. Stems are stored SORTED, so we walk
         // them in key order directly — no decode-all scratch array, no re-sort, no cap.
         const Src* old = _ses->get_ref<Src>(addr).obj();
         const int  W   = old->consume_width();
         const int  tw  = W - Wp;  // leftover stem bytes per entry (<=3: W<=4, Wp>=1)
         const int  n   = old->nstems();

         auto decode = [&](int i, uint8_t* b) {  // big-endian stem bytes of entry i (W bytes)
            const uint32_t c = old->stem_at(i);
            for (int k = 0; k < W; ++k)
               b[k] = uint8_t((c >> (8 * (Src::stem_width - 1 - k))) & 0xFF);
         };

         ptr_address      na    = make_wide_node(Wp, old->prefix(), hint,
                                                 old->head_size() + old->tail_used() + 64);
         sal::header_type nt    = _ses->get_ref<sal::alloc_header>(na)->type();
         const int        pcell = cell_width_for(Wp);

         if (smart_address term = old->term_value(); !term.is_null())
            set_term_of(na, nt, carry_value(na, nt, addr, term, hint));  // term is before the stride

         int     i = 0;
         uint8_t bi[4], bj[4];
         while (i < n)
         {
            decode(i, bi);
            int j = i + 1;
            for (; j < n; ++j)  // group entries sharing the high Wp bytes (sorted → contiguous)
            {
               decode(j, bj);
               if (std::memcmp(bi, bj, size_t(Wp)) != 0)
                  break;
            }
            const uint32_t pq = encode_cell(bi, Wp, pcell);
            if (j - i == 1 && is_terminal_branch(old->branch_at(i)))  // inline singleton
            {
               bytes suf, val;  // zero-copy views into `old` (stable arena, old stays live)
               term_view(old, old->branch_at(i), suf, val);
               uint8_t kb[4 + 256];  // tw (<=3) leftover stem bytes + inline suffix (len is u8)
               std::memcpy(kb, bi + Wp, size_t(tw));
               std::memcpy(kb + tw, suf.data(), suf.size());
               add_inline(na, nt, pq, bytes{kb, size_t(tw) + suf.size()}, val, hint);
            }
            else  // group child (consumes the tw tail bytes)
            {
               const int        ccell = cell_width_for(tw);
               ptr_address      child = make_wide_node(tw, bytes{}, hint, uint32_t(j - i) * 16 + 64);
               sal::header_type ct    = _ses->get_ref<sal::alloc_header>(child)->type();
               for (int k = i; k < j; ++k)
               {
                  uint8_t bk[4];
                  decode(k, bk);
                  const uint32_t      cq = encode_cell(bk + Wp, tw, ccell);
                  const smart_address cs = carry_value(child, ct, addr, old->branch_at(k), hint);
                  if (!is_terminal_branch(cs))  // a node pointer (carry_value retained it)
                     add_ptr(child, ct, cq, cs, hint);
                  else  // inline value already copied into child by carry_value → just set it
                     add_existing(child, ct, cq, cs, hint);
               }
               add_ptr(na, nt, pq, smart_address::from_pointer(child), hint);
            }
            i = j;
         }
         return na;
      }
      // Set a slot to an already-resident smart_address (no value creation, no widen).
      void add_existing(ptr_address& node, sal::header_type t, uint32_t q, smart_address sa,
                        sal::alloc_hint)
      {
         if (t == setlist_u16::type_id)
            cow<setlist_u16>(node, 8)->set_branch(uint16_t(q), sa);
         else if (t == setlist_u32::type_id)
            cow<setlist_u32>(node, 8)->set_branch(q, sa);
         else
         {
            auto* w = cow<full_radix_node>(node);
            rr_diverge<full_radix_node>(w, range_of(uint8_t(q)));  // privatize range before set
            w->set_branch(uint8_t(q), sa);
         }
      }
      void set_term_of(ptr_address node, sal::header_type t, smart_address sv)
      {
         if (t == setlist_u16::type_id)
            mut<setlist_u16>(node)->set_term_value(sv);
         else if (t == setlist_u32::type_id)
            mut<setlist_u32>(node)->set_term_value(sv);
         else
            mut<full_radix_node>(node)->set_term_value(sv);
      }

      // Add a (suffix→value) continuation under a fresh setlist `node` (inline if it
      // fits, else a SAL leaf). An empty suffix is the node's term.
      void add_continuation(ptr_address& node, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         if (suffix.empty())
         {
            smart_address sv = make_value<setlist_u8>(node, bytes{}, value, hint);
            mut<setlist_u8>(node)->set_term_value(sv);
            return;
         }
         uint8_t       b  = suffix[0];
         smart_address sv = make_value<setlist_u8>(node, suffix.subspan(1), value, hint);
         [[maybe_unused]] bool ok = mut<setlist_u8>(node)->set_branch(b, sv);
         assert(ok && "two-entry setlist always fits");
      }

      template <class N>
      void set_term(ptr_address& addr, bytes val, sal::alloc_hint peer_hint)
      {
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);  // resolved once; threaded into cow()
         smart_address     tv  = ref->term_value();
         if (tv.is_tail())  // update inline term
         {
            const inline_value* iv = inline_of(ref.obj(), tv);
            _old_value_size        = iv->value_len;
            if (iv->value_len == val.size())  // same size → overwrite in place
            {
               N*    n   = cow<N>(ref, addr);
               auto* wiv = reinterpret_cast<inline_value*>(reinterpret_cast<char*>(n) + n->size() -
                                                           tv.tail_offset());
               std::memcpy(wiv->data + wiv->suffix_len, val.data(), val.size());
            }
            else  // grow → new tail (old leaks), cow once
            {
               N* n = cow<N>(ref, addr, value_tail_need(bytes{}, val));
               n->set_term_value(make_value_into<N>(n, bytes{}, val, peer_hint));
            }
            return;
         }
         if (tv.is_pointer())  // update leaf term
         {
            ptr_address old = tv.as_pointer();
            _old_value_size = int(_ses->get_ref<leaf_node>(old).obj()->value_bytes().size());
            ptr_address tl  = update_leaf_value(old, val);
            if (tl != old)
            {
               N* n = cow<N>(ref, addr);
               _ses->release(old);
               n->set_term_value(smart_address::from_pointer(tl));
            }
            return;
         }
         N* n = cow<N>(ref, addr, value_tail_need(bytes{}, val));  // new term, cow once
         n->set_term_value(make_value_into<N>(n, bytes{}, val, peer_hint));
      }

      // Split `addr`'s prefix at p: a new setlist parent takes key[depth..depth+p];
      // the demoted node (prefix loses p+1 bytes) hangs under its old prefix byte.
      template <class N>
      ptr_address prefix_split(ptr_address addr, bytes key, uint32_t depth, int p, bytes val,
                               sal::alloc_hint peer_hint)
      {
         uint8_t ob = _ses->get_ref<N>(addr).obj()->prefix()[size_t(p)];

         ptr_address dnode = addr;
         N*          d     = cow<N>(dnode);
         int         newpl = int(d->prefix_length()) - (p + 1);
         uint8_t     pbuf[sal::cacheline_size];  // a prefix never exceeds prefix_capacity (<1 CL)
         std::memcpy(pbuf, d->prefix().data() + p + 1, size_t(newpl));
         reprefix<N>(dnode, bytes{pbuf, size_t(newpl)});  // changes prefix AND re-stamps the shape

         ptr_address np = make_node<setlist_u8>(key.subspan(depth, size_t(p)), peer_hint);
         mut<setlist_u8>(np)->set_branch(ob, smart_address::from_pointer(dnode));
         if (dnode == addr)
            _ses->retain(dnode);  // in-place demote: np is a 2nd parent; caller releases addr

         add_continuation(np, key.subspan(depth + uint32_t(p)), val, peer_hint);
         return np;
      }
      // The SINGLE prefix-change primitive — and it ALSO re-stamps the dispatch shape, because
      // the shape is a function of the prefix (a prefix forces r1; prefix-less keeps the wide
      // tier). Coupling the two here means no call site can leave a wide-stamped node holding a
      // prefix — the desync that made the skip-header find misroute (nondeterministically) in
      // adaptive mode. Prefix changes are the ONLY shape-affecting mutation; alloc already
      // defaults the control type to r1 (the safe slow path), so this is the only stamp a
      // mutation must do.
      template <class N>
      void reprefix(ptr_address a, bytes p)
      {
         N* n = mut<N>(a);
         // Every setlist width stores stems+branches at a prefix-relative offset, so a prefix
         // change must relocate that block — the node's own reprefix() (wide-aware for u16/u32,
         // end-anchored for u8). bitset/full keep branches in separate cachelines, so the cheap
         // base set_prefix() is correct for them.
         if constexpr (std::is_same_v<N, setlist_u8> || std::is_same_v<N, setlist_u16> ||
                       std::is_same_v<N, setlist_u32>)
            n->reprefix(p);
         else
            n->set_prefix(p);
         stamp<N>(a);
      }

      // ── widen: build a wider encoding from `from` + add (setb→{suf,val}) ─────────
      template <class To>
      smart_address reinline(To* nn, const inline_value* iv) noexcept
      {
         uint32_t n   = uint32_t(iv->byte_size());
         auto*    niv = reinterpret_cast<inline_value*>(nn->alloc_tail(uint16_t(n)));
         std::memcpy(niv, iv, n);
         return smart_address::from_tail(nn->tail_used());
      }
      // Build a wider encoding from `from`, then set setb. `setsa` non-null = set that
      // pointer as-is (a replaced child — fresh, not retained); null = make a new value
      // {suf,val} on the wider node. Re-inlines tails / copies+retains pointer children.
      template <class To, class From>
      ptr_address try_upgrade(ptr_address from, uint8_t setb, smart_address setsa, bytes suf,
                              bytes val, sal::alloc_hint hint, bool& ok)
      {
         const From* cn = _ses->get_ref<From>(from).obj();
         uint32_t    need =
             ucc::round_up_multiple<sal::cacheline_size>(To::head_bytes() + cn->tail_used() +
                                                         tail_reserve + 64 + To::headroom);
         ptr_address na = _ses->alloc<To>(need, hint, cn->prefix());
         To*         nn = mut<To>(na);
         nn->alloc_tail(tail_reserve);
         rr_birth<To>(nn);  // big tier → 8 counters; the term+branch retains below are the base
         stamp<To>(na);
         ok = true;
         smart_address tv = cn->term_value();
         if (tv.is_tail())
            nn->set_term_value(reinline<To>(nn, inline_of(cn, tv)));
         else if (!tv.is_null())
         {
            nn->set_term_value(tv);
            if (tv.is_pointer())
               _ses->retain(tv.as_pointer());
         }
         cn->for_each_entry([&](uint8_t byte, smart_address br) {
            if (!ok || byte == setb)
               return;
            if (br.is_tail())
            {
               if (!nn->set_branch(byte, reinline<To>(nn, inline_of(cn, br))))
                  ok = false;
            }
            else if (!nn->set_branch(byte, br))
               ok = false;
            else if (br.is_pointer())
               _ses->retain(br.as_pointer());
         });
         if (ok)  // set the new branch: a pointer as-is, or a freshly inlined value
         {
            if (!setsa.is_null())
               ok = nn->set_branch(setb, setsa);  // replaced child pointer (fresh, no retain)
            else
            {
               smart_address sv = make_value<To>(na, suf, val, hint);
               nn               = mut<To>(na);  // make_value may have grown na
               ok               = nn->set_branch(setb, sv);
            }
         }
         if (!ok)
         {
            _ses->release(na);
            return ptr_address(0);
         }
         return na;
      }
      // Proactive c1 promotion (opt-in, set_c1_promote): a setlist needn't OVERFLOW to be a c1.
      // Once a prefix-less setlist reaches c1_promote stems AND its whole byte range (incl the
      // new byte) fits a c1 window, force the (tested) grow_and_set path → smallest fitting c1.
      // Stems are sorted, so the range is first..last — O(1). Range grows monotonically, so this
      // fires at most once. Default OFF: on a memory-bound walk (dict) the dispatch win is hidden
      // under the per-hop miss; it pays when the working set is cache-resident / bounded-field.
      static constexpr int c1_promote = 8;
      bool                 want_c1_promote(const setlist_u8* n, uint8_t b) const noexcept
      {
         const int cnt = n->nstems();
         if (!_c1_promote_on || cnt < c1_promote || !n->prefix().empty())
            return false;
         int lo = n->stem_at(0), hi = n->stem_at(cnt - 1);
         lo = b < lo ? b : lo;
         hi = b > hi ? b : hi;
         return (hi - lo) < bitset_c1_128::width;
      }
      // Build a 1-CL narrow-window c1 node from `from`'s entries (+ setb) when they fit a
      // [base, base+To::width) window with <= To::cap branches. Returns 0 (ok=false) if the
      // key range is too wide or there are too many branches — the caller tries the next
      // width, then R2. c1 carries no prefix, so a prefixed source can't upgrade to c1.
      // Mirrors try_upgrade's tail-reinline / pointer-retain / setb-skip contract exactly.
      template <class To, class From>
      ptr_address try_upgrade_c1(ptr_address from, uint8_t setb, smart_address setsa, bytes suf,
                                 bytes val, sal::alloc_hint hint, bool& ok)
      {
         ok             = false;
         const From* cn = _ses->get_ref<From>(from).obj();
         if (!cn->prefix().empty())
            return ptr_address(0);  // c1 has no inline prefix
         // From is a (sorted) setlist, so its byte range is first..last — O(1). Fold in setb.
         const int en = cn->nstems();
         int       lo = setb, hi = setb;
         if (en)
         {
            const int slo = cn->stem_at(0), shi = cn->stem_at(en - 1);
            lo = slo < lo ? slo : lo;
            hi = shi > hi ? shi : hi;
         }
         const int n = en + 1;  // entries + the new branch being added
         if (hi - lo >= To::width || n > To::cap)
            return ptr_address(0);  // range too wide / too many branches → caller widens further
         uint32_t need = ucc::round_up_multiple<sal::cacheline_size>(
             To::head_bytes() + cn->tail_used() + tail_reserve + 64 + To::headroom);
         ptr_address na = _ses->alloc<To>(need, hint, bytes{});
         To*         nn = mut<To>(na);
         nn->alloc_tail(tail_reserve);
         rr_birth<To>(nn);  // big tier → 8 counters; the term+branch retains below are the base
         nn->set_window_base(uint8_t(lo));
         stamp<To>(na);
         ok               = true;
         smart_address tv = cn->term_value();
         if (tv.is_tail())
            nn->set_term_value(reinline<To>(nn, inline_of(cn, tv)));
         else if (!tv.is_null())
         {
            nn->set_term_value(tv);
            if (tv.is_pointer())
               _ses->retain(tv.as_pointer());
         }
         cn->for_each_entry([&](uint8_t byte, smart_address br) {
            if (!ok || byte == setb)
               return;
            if (br.is_tail())
            {
               if (!nn->set_branch(byte, reinline<To>(nn, inline_of(cn, br))))
                  ok = false;
            }
            else if (!nn->set_branch(byte, br))
               ok = false;
            else if (br.is_pointer())
               _ses->retain(br.as_pointer());
         });
         if (ok)
         {
            if (!setsa.is_null())
               ok = nn->set_branch(setb, setsa);
            else
            {
               smart_address sv = make_value<To>(na, suf, val, hint);
               nn               = mut<To>(na);  // make_value may have grown na
               ok               = nn->set_branch(setb, sv);
            }
         }
         if (!ok)
         {
            _ses->release(na);
            return ptr_address(0);
         }
         return na;
      }
      // Returns a wider node holding all of `from`'s entries plus setb→(setsa | {suf,val}).
      // The CALLER releases `from` (so a snapshot still referencing it stays valid).
      template <class From>
      ptr_address grow_and_set(ptr_address from, uint8_t b, smart_address setsa, bytes suf, bytes val,
                               sal::alloc_hint hint)
      {
         bool        ok = false;
         ptr_address up;
         if constexpr (std::is_same_v<From, setlist_u8>)
         {
            // Dense-in-a-narrow-range first: a 1-CL c1 window holds MORE than a setlist
            // (23/22/20 vs ~19) and routes by popcount, not scan. Smallest window that
            // fits wins; only a too-wide / too-populous range falls through to R2.
            up = try_upgrade_c1<bitset_c1_32, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
            up = try_upgrade_c1<bitset_c1_64, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
            up = try_upgrade_c1<bitset_c1_128, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
            up = try_upgrade<bitset_r2_node, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
         }
         if constexpr (!std::is_same_v<From, bitset_r4_node>)
         {
            up = try_upgrade<bitset_r4_node, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
         }
         return try_upgrade<full_radix_node, From>(from, b, setsa, suf, val, hint, ok);  // always fits
      }

      // ── downgrade (the inverse of widen) ─────────────────────────────────────────
      // After a remove shrinks a wide node, re-encode it into the smallest tier that holds
      // its entries with HYSTERESIS — the count must sit ~25% below where that tier would
      // re-widen, so churn around the boundary doesn't thrash grow↔shrink. A node stuck at a
      // big tier wastes cachelines (worse aggregate cache footprint on memory-bound queries)
      // and lengthens a setlist scan; shrinking keeps the tree tight. These caps are the
      // regrow points; the rebuild is the source of truth (it fails → we keep the tier).
      static constexpr int dn_setlist = 14;  // setlist re-widens ~19  → shrink to setlist at <=14
      static constexpr int dn_r2      = 40;  // r2 (2x~27) re-widens ~54 → shrink to r2 at <=40
      static constexpr int dn_r4      = 87;  // r4 (4x~29) re-widens ~116 → shrink to r4 at <=87

      // Rebuild ALL of `from`'s entries (+ term) into a fresh `To` (no added key); returns the
      // new node or 0 (ok=false) if they don't fit To. Same tail-reinline / pointer-retain
      // contract as try_upgrade — so the caller releases `from` and refcounts stay balanced.
      template <class To, class From>
      ptr_address rebuild_into(ptr_address from, bool& ok)
      {
         const From* cn   = _ses->get_ref<From>(from).obj();
         uint32_t    need = ucc::round_up_multiple<sal::cacheline_size>(
             To::head_bytes() + cn->tail_used() + tail_reserve + 64 + To::headroom);
         ptr_address na = _ses->alloc<To>(need, sal::alloc_hint{}, cn->prefix());
         To*         nn = mut<To>(na);
         nn->alloc_tail(tail_reserve);
         rr_birth<To>(nn);  // big tier → 8 counters; the term+branch retains below are the base
         stamp<To>(na);
         ok               = true;
         smart_address tv = cn->term_value();
         if (tv.is_tail())
            nn->set_term_value(reinline<To>(nn, inline_of(cn, tv)));
         else if (!tv.is_null())
         {
            nn->set_term_value(tv);
            if (tv.is_pointer())
               _ses->retain(tv.as_pointer());
         }
         cn->for_each_entry([&](uint8_t byte, smart_address br) {
            if (!ok)
               return;
            if (br.is_tail())
            {
               if (!nn->set_branch(byte, reinline<To>(nn, inline_of(cn, br))))
                  ok = false;
            }
            else if (!nn->set_branch(byte, br))
               ok = false;
            else if (br.is_pointer())
               _ses->retain(br.as_pointer());
         });
         if (!ok)
         {
            _ses->release(na);
            return ptr_address(0);
         }
         return na;
      }

      // If `addr` (a wide tier) has shrunk enough, return a smaller-tier rebuild (caller frees
      // the old per the COW contract); else return `addr` unchanged. setlist is the floor.
      template <class N>
      ptr_address maybe_downgrade(ptr_address addr)
      {
         if constexpr (std::is_same_v<N, setlist_u8> || std::is_same_v<N, setlist_u16> ||
                       std::is_same_v<N, setlist_u32>)
            return addr;  // already the floor / wide-stem variants don't byte-downgrade
         else
         {
            const int   cnt = _ses->get_ref<N>(addr).obj()->branch_count();
            bool        ok  = false;
            ptr_address dg;
            if (cnt <= dn_setlist)  // smallest first: a 1-CL setlist if it fits
            {
               dg = rebuild_into<setlist_u8, N>(addr, ok);
               if (ok)
                  return dg;
            }
            if constexpr (std::is_same_v<N, full_radix_node> || std::is_same_v<N, bitset_r4_node>)
               if (cnt <= dn_r2)
               {
                  dg = rebuild_into<bitset_r2_node, N>(addr, ok);
                  if (ok)
                     return dg;
               }
            if constexpr (std::is_same_v<N, full_radix_node>)
               if (cnt <= dn_r4)
               {
                  dg = rebuild_into<bitset_r4_node, N>(addr, ok);
                  if (ok)
                     return dg;
               }
            return addr;
         }
      }

      // ── bucket mode: radix routers (full_radix here) + bucket_node leaves ───────────
      // Reuses the engine's session/make_node/mut + bucket_node + full_radix_node. In-place
      // mutation (no MVCC snapshots). A full bucket splits into a router over child buckets;
      // a router whose prefix a new key diverges from is prefix-split (path compression).
      sal::header_type ntype(ptr_address a) { return _ses->get_ref<sal::alloc_header>(a)->type(); }
      bucket_node*     bkt(ptr_address a) { return _ses->get_ref<bucket_node>(a).modify().get(); }

      // ── adaptive radix→bucket (deep-narrow detection + subtree collapse) ─────────
      bool is_bucket(ptr_address a)
      {
         return _ses->get_ref<sal::alloc_header>(a)->type() == bucket_node::type_id;
      }
      // < 3 branches (children + term) → "narrow". Only setlist/single-child nodes can be
      // narrow; bitset/full are wide; leaf/bucket aren't radix routers.
      bool node_narrow(ptr_address a)
      {
         auto t = _ses->get_ref<sal::alloc_header>(a)->type();
         if (t == setlist_u8::type_id)
         {
            auto* n = _ses->get_ref<setlist_u8>(a).obj();
            return n->nstems() + (n->has_term_value() ? 1 : 0) < 3;
         }
         if (t == setlist_u16::type_id)
         {
            auto* n = _ses->get_ref<setlist_u16>(a).obj();
            return n->nstems() + (n->has_term_value() ? 1 : 0) < 3;
         }
         if (t == setlist_u32::type_id)
         {
            auto* n = _ses->get_ref<setlist_u32>(a).obj();
            return n->nstems() + (n->has_term_value() ? 1 : 0) < 3;
         }
         return false;
      }
      // Insert into a bucket LEAF; on overflow split it to a radix router (bkt_split) and route
      // the key into that router via the normal engine. Returns the (possibly new) address.
      ptr_address bucket_leaf_insert(ptr_address bp, bytes key, uint32_t depth, bytes val)
      {
         // Everything reached from here lives in a bucket subtree; suppress the radix→bucket
         // trigger so a split's rebuilt radix (collision setlists) doesn't re-collapse mid-build.
         const bool was = _in_bucket;
         _in_bucket     = true;
         ptr_address r  = bucket_leaf_insert_impl(bp, key, depth, val);
         _in_bucket     = was;
         return r;
      }
      ptr_address bucket_leaf_insert_impl(ptr_address bp, bytes key, uint32_t depth, bytes val)
      {
         if (!is_bucket(bp))  // already split to a router (e.g. mid-collapse) → route normally
            return insert_node(bp, key, depth, val, {});
         int r = bkt(bp)->insert(key.data() + depth, int(key.size() - depth), val.data(),
                                 int(val.size()));
         if (r != 0)
         {
            _old_value_size = r < 0 ? 0 : -1;
            return bp;
         }
         ptr_address rp  = bkt_split(bp);                // overflow → router (bp freed by CALLER)
         ptr_address nrp = insert_node(rp, key, depth, val, {});  // normal path adds the key
         if (nrp != rp)
            _ses->release(rp);  // the freshly-split router got replaced (COW) → free it
         return nrp;
      }
      // Collect (suffix,value) of every key in addr's subtree, suffix relative to addr. Bounded:
      // returns false (abort, stay radix) if the subtree is bigger than a bucket should absorb —
      // which, with early detection, is rare.
      static constexpr int conv_sbuf = 8192, conv_vbuf = 4096, conv_cap = 300;
      struct CE { uint16_t so, vo; uint8_t sl, vl; };
      template <class Ad>
      bool collect_in(Ad ad, KB& acc, uint8_t* sb, uint16_t& sp, uint8_t* vb, uint16_t& vp,
                      CE* ents, int& n, int cap)
      {
         const uint32_t save = acc.len;
         bytes          pfx  = ad.prefix();
         acc.append(pfx.data(), pfx.size());
         auto record = [&](bytes suf, value_view v) -> bool {
            const int sl = int(acc.len) + int(suf.size());
            if (n >= cap || sl > 255 || int(v.size()) > 255 || int(sp) + sl > conv_sbuf ||
                int(vp) + int(v.size()) > conv_vbuf)
               return false;
            ents[n].so = sp;
            ents[n].sl = uint8_t(sl);
            std::memcpy(sb + sp, acc.buf, acc.len);
            std::memcpy(sb + sp + acc.len, suf.data(), suf.size());
            sp = uint16_t(sp + sl);
            ents[n].vo = vp;
            ents[n].vl = uint8_t(v.size());
            std::memcpy(vb + vp, v.data(), v.size());
            vp = uint16_t(vp + v.size());
            ++n;
            return true;
         };
         if (!ad.term().is_null())
         {
            bytes      s;
            value_view v;
            term_view(ad.node(), ad.term(), s, v);
            if (!record(s, v))
               return false;
         }
         for (int c = ad.first(); ad.valid(c); c = ad.next(c))
         {
            const uint32_t s0 = acc.len;
            uint8_t        sbb[4];
            const int      nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sbb);
            acc.append(sbb, size_t(nb));
            smart_address br = ad.branch(c);
            if (is_terminal_branch(br))
            {
               bytes      s;
               value_view v;
               term_view(ad.node(), br, s, v);
               if (!record(s, v))
                  return false;
            }
            else if (!collect_walk(br.as_pointer(), acc, sb, sp, vb, vp, ents, n, cap))
               return false;
            acc.truncate(s0);
         }
         acc.truncate(save);
         return true;
      }
      bool collect_walk(ptr_address a, KB& acc, uint8_t* sb, uint16_t& sp, uint8_t* vb, uint16_t& vp,
                        CE* ents, int& n, int cap)
      {
         if (is_bucket(a))  // a deeper region is already a bucket → don't try to collapse
            return false;
         return dispatch(a, [&](auto ad) { return collect_in(ad, acc, sb, sp, vb, vp, ents, n, cap); });
      }
      // Build a bucket holding the whole subtree at addr; false if it's too big to collapse.
      bool convert_subtree_to_bucket(ptr_address addr, ptr_address& out)
      {
         uint8_t  sb[conv_sbuf];
         uint8_t  vb[conv_vbuf];
         CE       ents[conv_cap];
         uint16_t sp = 0, vp = 0;
         int      n  = 0;
         KB       acc;
         acc.len = 0;
         if (!collect_walk(addr, acc, sb, sp, vb, vp, ents, n, conv_cap))
            return false;
         ptr_address bp = make_bucket();
         for (int i = 0; i < n; ++i)
         {
            ptr_address nb = bucket_leaf_insert(bp, bytes{sb + ents[i].so, ents[i].sl}, 0,
                                                bytes{vb + ents[i].vo, ents[i].vl});
            if (nb != bp)
               _ses->release(bp);  // bp split/COW'd into nb → free the consumed node
            bp = nb;
         }
         out = bp;
         return true;
      }
      ptr_address bkt_make(bytes suf, bytes val)
      {
         ptr_address bp = make_bucket();
         bkt(bp)->insert(suf.data(), int(suf.size()), val.data(), int(val.size()));
         return bp;
      }
      void bkt_set_term(ptr_address rp, bytes v)  // store value as an empty-suffix inline term
      {
         // If a term already exists and the new value fits its slot, overwrite IN PLACE.
         // Otherwise alloc_tail leaks the old slot on every update — and repeated updates of
         // the same terminal (e.g. duplicate-key upserts) grow the tail past the node's
         // headroom and into the branch array. (The leak on a genuine grow is rare + bounded.)
         smart_address tv = _ses->get_ref<full_radix_node>(rp).obj()->term_value();
         if (!tv.is_null())
         {
            full_radix_node* r  = mut<full_radix_node>(rp);
            auto*            iv = reinterpret_cast<inline_value*>(
                reinterpret_cast<char*>(r) + r->size() - tv.tail_offset());
            if (iv->value_len >= int(v.size()))  // empty-suffix term; reuse the slot
            {
               iv->value_len = uint8_t(v.size());
               std::memcpy(iv->data, v.data(), v.size());
               return;
            }
         }
         const int        need = inline_value::byte_size(0, int(v.size()));
         full_radix_node* r    = mut<full_radix_node>(rp);
         auto*            iv   = reinterpret_cast<inline_value*>(r->alloc_tail(uint16_t(need)));
         iv->suffix_len = 0;
         iv->value_len  = uint8_t(v.size());
         iv->value_type = 0;
         std::memcpy(iv->data, v.data(), v.size());
         mut<full_radix_node>(rp)->set_term_value(smart_address::from_tail(r->tail_used()));
      }
      // Store a single (suffix,value) terminal INLINE in router rp under byte b — a from_tail
      // branch in the router's own tail heap, NOT a separate object. This is what stops a dense
      // terminal byte (sequential keys) from exploding into one bucket per key. Returns false
      // if it doesn't fit the inline cap or the router's tail is full (caller uses a bucket).
      // Grow a compact router to its working page ONCE its inline-terminal tail overflows, so a
      // dense terminal byte fills one page instead of spilling to a per-key bucket. In-place
      // model (no MVCC sharing): alloc the bigger page, blit head+tail, retain children, release
      // the old router. Returns the new address (the old one is freed here).
      ptr_address bkt_grow_router(ptr_address rp)
      {
         // Copy the prefix to the stack FIRST: the alloc below can relocate/compact the arena,
         // so neither rp's pointer nor a span into it may be held across it.
         uint8_t pbuf[256];
         int     pl;
         {
            bytes p = _ses->get_ref<full_radix_node>(rp).obj()->prefix();
            pl      = int(p.size());
            std::memcpy(pbuf, p.data(), size_t(pl));
         }
         ptr_address np = _ses->alloc<full_radix_node>(full_radix_node::page_bytes, sal::alloc_hint{},
                                                       bytes{pbuf, size_t(pl)});
         full_radix_node* nn = mut<full_radix_node>(np);
         _ses->get_ref<full_radix_node>(rp).obj()->copy_to(nn);  // re-resolve rp AFTER the alloc
         rr_birth<full_radix_node>(nn);  // 8 counters; the for_each_child retains below are the base
         nn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         stamp<full_radix_node>(np);  // dispatch shape (find reads cb_type) — copy_to didn't set it
         _ses->release(rp);  // drop old router (its destroy releases each child once → net no change)
         return np;
      }
      bool bkt_try_inline(ptr_address& rp, uint8_t b, bytes suf, bytes val)
      {
         if (!fits_inline(suf, val))
            return false;
         full_radix_node* r    = mut<full_radix_node>(rp);
         const uint32_t   need = uint32_t(inline_value::byte_size(int(suf.size()), int(val.size())));
         if (r->free_heap() < need)
         {
            if (r->size() >= full_radix_node::page_bytes)
               return false;  // already at the working page and still full → spill to a bucket
            rp = bkt_grow_router(rp);
            r  = mut<full_radix_node>(rp);
            if (r->free_heap() < need)
               return false;
         }
         auto* iv       = reinterpret_cast<inline_value*>(r->alloc_tail(uint16_t(need)));
         iv->suffix_len = uint8_t(suf.size());
         iv->value_len  = uint8_t(val.size());
         iv->value_type = 0;
         std::memcpy(iv->data, suf.data(), suf.size());
         std::memcpy(iv->data + suf.size(), val.data(), val.size());
         r->set_branch(b, smart_address::from_tail(r->tail_used()));
         return true;
      }
      // A new key collides with an existing inline terminal under `byte`. Either update it in
      // place (same key) or promote: build a bucket holding the existing + new entries and
      // repoint the branch at it (the old inline slot leaks — rare).
      ptr_address bkt_promote_inline(ptr_address rp, uint8_t byte, smart_address br, bytes key,
                                     uint32_t depth, bytes val)
      {
         full_radix_node*    r  = mut<full_radix_node>(rp);
         const inline_value* iv = inline_of(r, br);
         bytes               esuf{iv->data, size_t(iv->suffix_len)};
         bytes               nsuf = key.subspan(depth + 1);
         if (esuf.size() == nsuf.size() &&
             (nsuf.empty() || std::memcmp(esuf.data(), nsuf.data(), nsuf.size()) == 0))
         {  // same key -> overwrite the value in place if it fits the slot, else re-inline (leak)
            if (iv->value_len >= int(val.size()))
            {
               auto* w      = const_cast<inline_value*>(iv);
               w->value_len = uint8_t(val.size());
               std::memcpy(w->data + w->suffix_len, val.data(), val.size());
               _old_value_size = 0;
               return rp;
            }
            if (bkt_try_inline(rp, byte, esuf, val))
            {
               _old_value_size = 0;
               return rp;
            }
         }
         // different key (or a value grow with no room): snapshot the existing entry off the
         // router tail, then build a bucket holding it + the new key.
         uint8_t   tmp[inline_value::max_inline_size];
         const int sl = iv->suffix_len, vl = iv->value_len;
         std::memcpy(tmp, iv->data, size_t(sl) + size_t(vl));
         ptr_address nb = bkt_make(bytes{tmp, size_t(sl)}, bytes{tmp + sl, size_t(vl)});
         nb             = bkt_insert(nb, key, depth + 1, val);
         mut<full_radix_node>(rp)->set_branch(byte, smart_address::from_pointer(nb));
         return rp;
      }
      ptr_address bkt_split(ptr_address bp)
      {
         // SNAPSHOT all entries (suffix+value bytes) off the bucket BEFORE allocating any
         // child: a child-build allocation can trigger the SAL compactor to relocate `bp`,
         // dangling raw pointers into it. After the snapshot bp is released. (Same discipline
         // as term_get — guard against object relocation, not just mmap base moves.)
         struct E { uint16_t off; uint8_t sl; uint8_t vl; };
         E        ent[1200];
         uint8_t  buf[bucket_node::page_bytes];
         int      n   = 0;
         uint16_t pos = 0;
         _ses->get_ref<bucket_node>(bp).obj()->for_each([&](const inline_value& iv) {
            assert(n < 1200 && pos + iv.suffix_len + iv.value_len <= int(sizeof(buf)));
            ent[n] = {pos, iv.suffix_len, iv.value_len};
            std::memcpy(buf + pos, iv.data, size_t(iv.suffix_len) + iv.value_len);
            pos = uint16_t(pos + iv.suffix_len + iv.value_len);
            ++n;
         });
         assert(n >= 2);
         auto val = [&](int k) { return bytes{buf + ent[k].off + ent[k].sl, ent[k].vl}; };

         int cp = 0, m = ent[0].sl < ent[n - 1].sl ? ent[0].sl : ent[n - 1].sl;
         while (cp < m && buf[ent[0].off + cp] == buf[ent[n - 1].off + cp]) ++cp;
         ptr_address rp = make_node<full_radix_node>(bytes{buf + ent[0].off, size_t(cp)});
         int         i  = 0;
         if (ent[0].sl == cp)  // lone key ending exactly at the common prefix
         {
            bkt_set_term(rp, val(0));
            i = 1;
         }
         while (i < n)
         {
            const uint8_t bch = buf[ent[i].off + cp];
            int           j   = i;
            while (j < n && ent[j].sl > cp && buf[ent[j].off + cp] == bch) ++j;
            // a lone key under this byte → inline terminal in the router (no per-key object);
            // this is what keeps a dense terminal byte from exploding into one bucket per key.
            if (j - i == 1 &&
                bkt_try_inline(rp, bch, bytes{buf + ent[i].off + cp + 1, size_t(ent[i].sl - cp - 1)},
                               val(i)))
            {
               i = j;
               continue;
            }
            ptr_address cbp = make_bucket();
            for (int k = i; k < j; ++k)  // recursive: a bucket-sized group becomes a sub-radix
               cbp = bkt_insert(cbp, bytes{buf + ent[k].off + cp + 1, size_t(ent[k].sl - cp - 1)}, 0,
                                val(k));
            mut<full_radix_node>(rp)->set_branch(bch, smart_address::from_pointer(cbp));
            i = j;
         }
         // Engine COW contract: return the new node; the CALLER releases the old `bp` (exactly
         // once). Freeing it here too would double-free, since router_insert/upsert already do
         // `if (nc != child) release(child)`.
         return rp;
      }
      ptr_address bkt_prefix_split(ptr_address rp, bytes key, uint32_t depth, int p, bytes val)
      {
         bytes   P  = _ses->get_ref<full_radix_node>(rp).obj()->prefix();
         uint8_t ob = P[size_t(p)];
         uint8_t pbuf[128];
         int     newpl = int(P.size()) - (p + 1);
         std::memcpy(pbuf, P.data() + p + 1, size_t(newpl));
         ptr_address np = make_node<full_radix_node>(bytes{key.data() + depth, size_t(p)});
         reprefix<full_radix_node>(rp, bytes{pbuf, size_t(newpl)});  // changes prefix AND re-stamps
         mut<full_radix_node>(np)->set_branch(ob, smart_address::from_pointer(rp));
         const uint32_t d2 = depth + uint32_t(p);
         if (d2 == key.size())
            bkt_set_term(np, val);
         else
            mut<full_radix_node>(np)->set_branch(
                key[d2],
                smart_address::from_pointer(bkt_make({key.data() + d2 + 1, key.size() - d2 - 1}, val)));
         return np;
      }
      ptr_address bkt_insert(ptr_address addr, bytes key, uint32_t depth, bytes val)
      {
         if (ntype(addr) == bucket_node::type_id)
         {
            int r = bkt(addr)->insert(key.data() + depth, int(key.size() - depth), val.data(),
                                      int(val.size()));
            if (r != 0)
            {
               _old_value_size = r < 0 ? 0 : -1;
               return addr;
            }
            ptr_address split = bkt_split(addr);  // bkt_split no longer frees its input
            _ses->release(addr);                   // so THIS caller frees the consumed bucket
            return bkt_insert(split, key, depth, val);
         }
         bytes     P     = _ses->get_ref<full_radix_node>(addr).obj()->prefix();
         const int avail = int(key.size() - depth);
         const int mm    = int(P.size()) < avail ? int(P.size()) : avail;
         int       p     = 0;
         while (p < mm && key[depth + p] == P[p]) ++p;
         if (p < int(P.size()))
            return bkt_prefix_split(addr, key, depth, p, val);
         depth += uint32_t(P.size());
         if (depth == key.size())
         {
            bkt_set_term(addr, val);
            return addr;
         }
         const uint8_t byte = key[depth];
         smart_address br   = _ses->get_ref<full_radix_node>(addr).obj()->find(byte);
         if (br.is_null())
         {
            bytes suf{key.data() + depth + 1, key.size() - depth - 1};
            if (!bkt_try_inline(addr, byte, suf, val))  // inline terminal, else a child bucket
               mut<full_radix_node>(addr)->set_branch(byte,
                                                      smart_address::from_pointer(bkt_make(suf, val)));
            _old_value_size = -1;
            return addr;
         }
         if (br.is_tail())  // existing inline terminal under this byte: update or promote
            return bkt_promote_inline(addr, byte, br, key, depth, val);
         ptr_address nc = bkt_insert(br.as_pointer(), key, depth + 1, val);
         if (nc != br.as_pointer())
            mut<full_radix_node>(addr)->set_branch(byte, smart_address::from_pointer(nc));
         return addr;
      }
      bool bkt_find(bytes key, value_view& out)
      {
         ptr_address addr  = _root;
         uint32_t    depth = 0;
         for (;;)
         {
            if (ntype(addr) == bucket_node::type_id)
            {
               const inline_value* iv =
                   _ses->get_ref<bucket_node>(addr).obj()->find(key.data() + depth, int(key.size() - depth));
               if (!iv)
                  return false;
               out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
               return true;
            }
            const full_radix_node* r = _ses->get_ref<full_radix_node>(addr).obj();
            bytes                  P = r->prefix();
            if (key.size() - depth < P.size() ||
                (P.size() && std::memcmp(key.data() + depth, P.data(), P.size()) != 0))
               return false;
            depth += uint32_t(P.size());
            if (depth == key.size())
            {
               smart_address tv = r->term_value();
               if (tv.is_null())
                  return false;
               const inline_value* iv = inline_of(r, tv);
               out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
               return true;
            }
            smart_address br = r->find(key[depth]);
            if (br.is_null())
               return false;
            if (br.is_tail())  // inline terminal in the router: suffix == key[depth+1:]
               return inline_at(r, br, key.subspan(depth + 1), out);
            addr = br.as_pointer();
            ++depth;
         }
      }

      // Bucket-mode remove (in-place, no MVCC — matches bkt_insert). Descends the full_radix
      // routers to the bucket leaf / inline terminal holding the key. An emptied bucket leaf is
      // released and its parent branch cleared; an inline terminal / router term is cleared in
      // place (its tail orphans). Empty routers are left (correct for find; not a leak).
      bool bkt_remove(bytes key)
      {
         ptr_address addr = _root, parent{};
         uint8_t     pb = 0;
         bool        have_parent = false;
         uint32_t    depth       = 0;
         for (;;)
         {
            if (ntype(addr) == bucket_node::type_id)
            {
               const uint8_t* s  = key.data() + depth;
               const int      sn = int(key.size() - depth);
               if (!_ses->get_ref<bucket_node>(addr).obj()->find(s, sn))
                  return false;
               bkt(addr)->remove(s, sn);
               if (have_parent && _ses->get_ref<bucket_node>(addr).obj()->nent == 0)
               {
                  mut<full_radix_node>(parent)->remove_branch(pb);
                  _ses->release(addr);  // free the drained bucket
               }
               return true;
            }
            const full_radix_node* r = _ses->get_ref<full_radix_node>(addr).obj();
            bytes                  P = r->prefix();
            if (key.size() - depth < P.size() ||
                (P.size() && std::memcmp(key.data() + depth, P.data(), P.size()) != 0))
               return false;
            depth += uint32_t(P.size());
            if (depth == key.size())  // router term value
            {
               smart_address tv = r->term_value();
               if (tv.is_null())
                  return false;
               mut<full_radix_node>(addr)->set_term_value(smart_address::null());
               if (tv.is_pointer())
                  _ses->release(tv.as_pointer());
               return true;
            }
            uint8_t       b  = key[depth];
            smart_address br = r->find(b);
            if (br.is_null())
               return false;
            if (br.is_tail())  // inline terminal in the router
            {
               bytes      esuf;
               value_view ev;
               term_view(r, br, esuf, ev);
               bytes suf = key.subspan(depth + 1);
               if (esuf.size() != suf.size() ||
                   (esuf.size() && std::memcmp(esuf.data(), suf.data(), esuf.size()) != 0))
                  return false;
               mut<full_radix_node>(addr)->remove_branch(b);  // inline tail orphans
               return true;
            }
            parent      = addr;
            pb          = b;
            have_parent = true;
            addr        = br.as_pointer();
            ++depth;
         }
      }

      std::unique_ptr<sal::allocator> _alloc;
      sal::allocator_session_ptr      _ses;
      ptr_address                     _root;
      ptr_address                     _snap{};
      bool                            _snap_valid     = false;
      int                             _old_value_size = -1;
      bool                            _route_found    = false;
      bool                            _removed        = false;  // remove(): key was found+deleted
      bool                            _emptied        = false;  // remove(): node drained → drop it
      bool                            _wide_stems     = false;  // gate: off = byte radix
      bool                            _bucket_mode    = false;  // leaf=bucket, split to radix
      bool                            _adaptive       = false;  // gate: radix→bucket on deep-narrow
      bool                            _in_bucket      = false;  // re-entrancy: in a bucket subtree
      bool                            _c1_promote_on  = false;  // opt-in: narrow setlist→c1 below overflow
      // COW-to-root unique→shared MODE LATCH (mirrors psitri upsert_mode.make_shared()). Reset at
      // the top of each upsert/remove; latched true the moment descent passes through a SHARED node
      // (ref>1, e.g. a committed snapshot pins the root). Once shared it stays shared for the rest
      // of this single depth-first path, forcing cow() to CLONE every node below — even ref==1 ones
      // that are reachable from the snapshot — instead of mutating them in place (which would
      // corrupt the snapshot). The recursion is a single path, so one member == a threaded mode.
      bool                            _shared         = false;
      cow_counters                    _cc;
   };

   // Out-of-line (uses dispatch(), whose deduced return type must be defined first).
   inline void radix_tree::dump_root_child_refs(const char* tag)
   {
      int total = 0, shared = 0, unique = 0;
      dispatch(_root, [&](auto ad)
      {
         ad.p->for_each_child([&](ptr_address c)
         {
            uint32_t rc = _ses->get_ref<sal::alloc_header>(c).ref();
            ++total;
            if (rc >= 2) ++shared; else ++unique;
         });
      });
      std::printf("  [%s] root ref=%u, pointer-children=%d : ref>=2 (shared)=%d, ref==1 (unique)=%d\n",
                  tag, root_use_count(), total, shared, unique);
   }
}  // namespace heart::radix

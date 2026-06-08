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
            // R1: read CL0, dispatch by format. The SAL type ids are contiguous, so this
            // switch lowers to a JUMP TABLE (one indirect jump) rather than a compare chain
            // — the data-dependent, mispredict-prone branch the per-hop dispatch turns on.
            bool cont;
            switch (unsigned(_ses->get_ref<sal::alloc_header>(addr)->type()))
            {
               case unsigned(leaf_node::type_id):
                  return leaf_read(addr, kb, depth, out);
               case unsigned(bucket_node::type_id):  // a bucket LEAF: search its key suffix
               {
                  const inline_value* iv = _ses->get_ref<bucket_node>(addr).obj()->find(
                      kb.data() + depth, int(kb.size() - depth));
                  if (!iv)
                     return false;
                  out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                   iv->value_len);
                  return true;
               }
               case unsigned(setlist_u8::type_id):
                  cont = route_r1(_ses->get_ref<setlist_u8>(addr).obj(), kb, depth, out, addr);
                  break;
               case unsigned(setlist_u16::type_id):
                  cont = route_r1(_ses->get_ref<setlist_u16>(addr).obj(), kb, depth, out, addr);
                  break;
               case unsigned(setlist_u32::type_id):
                  cont = route_r1(_ses->get_ref<setlist_u32>(addr).obj(), kb, depth, out, addr);
                  break;
               case unsigned(bitset_c1_32::type_id):
                  cont = route_r1(_ses->get_ref<bitset_c1_32>(addr).obj(), kb, depth, out, addr);
                  break;
               case unsigned(bitset_c1_64::type_id):
                  cont = route_r1(_ses->get_ref<bitset_c1_64>(addr).obj(), kb, depth, out, addr);
                  break;
               case unsigned(bitset_c1_128::type_id):
                  cont = route_r1(_ses->get_ref<bitset_c1_128>(addr).obj(), kb, depth, out, addr);
                  break;
               case unsigned(bitset_r2_node::type_id):
                  cont = route_r1(_ses->get_ref<bitset_r2_node>(addr).obj(), kb, depth, out, addr);
                  break;
               case unsigned(bitset_r4_node::type_id):
                  cont = route_r1(_ses->get_ref<bitset_r4_node>(addr).obj(), kb, depth, out, addr);
                  break;
               default:  // full_radix
                  cont = route_r1(_ses->get_ref<full_radix_node>(addr).obj(), kb, depth, out, addr);
                  break;
            }
            if (!cont)
               return _route_found;
         }
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

      void commit()
      {
         auto lock = _ses->lock();
         if (_snap_valid)
            _ses->release(_snap);
         _snap = _root;
         _ses->retain(_root);
         _snap_valid = true;
      }

      uint64_t object_count() const { return _alloc->total_allocated_objects(); }

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
         auto t = _ses->get_ref<sal::alloc_header>(a)->type();
         if (t == leaf_node::type_id)
            ++tc.leaf;
         else if (t == setlist_u8::type_id)
         {
            ++tc.setlist;
            accum_setlist<setlist_u8>(a, tc);
            walk_kids<setlist_u8>(a, tc);
         }
         else if (t == setlist_u16::type_id)
         {
            ++tc.setlist_w16;
            walk_kids<setlist_u16>(a, tc);
         }
         else if (t == setlist_u32::type_id)
         {
            ++tc.setlist_w32;
            walk_kids<setlist_u32>(a, tc);
         }
         else if (t == bitset_c1_32::type_id)
         {
            ++tc.bitset_c1;
            walk_kids<bitset_c1_32>(a, tc);
         }
         else if (t == bitset_c1_64::type_id)
         {
            ++tc.bitset_c1;
            walk_kids<bitset_c1_64>(a, tc);
         }
         else if (t == bitset_c1_128::type_id)
         {
            ++tc.bitset_c1;
            walk_kids<bitset_c1_128>(a, tc);
         }
         else if (t == bitset_r2_node::type_id)
         {
            ++tc.bitset_r2;
            accum_compact<bitset_r2_node>(a, tc);
            walk_kids<bitset_r2_node>(a, tc);
         }
         else if (t == bitset_r4_node::type_id)
         {
            ++tc.bitset_r4;
            accum_compact<bitset_r4_node>(a, tc);
            walk_kids<bitset_r4_node>(a, tc);
         }
         else
         {
            ++tc.full;
            uint8_t q = 0;  // which 64-wide quarters have any branch
            _ses->get_ref<full_radix_node>(a).obj()->for_each_entry(
                [&](uint8_t byte, smart_address) { q |= uint8_t(1u << (byte >> 6)); });
            tc.full_quarters += uint64_t(std::popcount(unsigned(q)));
            walk_kids<full_radix_node>(a, tc);
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
         _ses->get_ref<N>(a).modify().get()->alloc_tail(tail_reserve);  // tail offsets >=16
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
      template <class N>
      ptr_address clone(ptr_address a, uint32_t extra = 0)
      {
         auto     ref  = _ses->get_ref<N>(a);
         const N* src  = ref.obj();
         uint32_t need = ucc::round_up_multiple<sal::cacheline_size>(
             src->head_size() + src->tail_used() + extra + N::headroom);
         ptr_address na = _ses->alloc<N>(need, sal::alloc_hint{}, src->prefix());
         N*          nn = mut<N>(na);
         blit<N>(nn, src);
         nn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         stamp<N>(na);
         ++_cc.clones;
         return na;
      }
      // Make `addr` writable for a mutation needing `extra` tail bytes. Shared → COW;
      // unique with room → in place; unique heap-full → grow. The old node is released
      // by the caller as the replaced child (nc != child) — clone pre-sizes for extra,
      // so we never release here (that would double-free).
      template <class N>
      N* cow(ptr_address& addr, uint32_t extra = 0)
      {
         auto ref = _ses->get_ref<N>(addr);
         if (ref.ref() > 1)
         {
            addr = clone<N>(addr, extra);
            return mut<N>(addr);
         }
         if (ref->free_heap() >= extra)
         {
            ++_cc.in_place;
            return ref.modify().get();
         }
         ++_cc.grows;
         addr = clone<N>(addr, extra);
         return mut<N>(addr);
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
      // Allocate a terminal {suffix,value} attached to `addr`: a tail leaf in addr's
      // heap (small; grows addr), else a SAL leaf pointer. Returns its smart_address.
      template <class N>
      smart_address make_value(ptr_address& addr, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         if (fits_inline(suffix, value))
         {
            uint32_t need = uint32_t(inline_value::byte_size(int(suffix.size()), int(value.size())));
            N*       n    = cow<N>(addr, need);
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
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
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
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
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
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
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
      template <class F>
      auto dispatch(ptr_address addr, F&& f)
      {
         const auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
         if (t == bitset_c1_32::type_id)
            return f(C1Ad<bitset_c1_32>{_ses->get_ref<bitset_c1_32>(addr).obj()});
         if (t == bitset_c1_64::type_id)
            return f(C1Ad<bitset_c1_64>{_ses->get_ref<bitset_c1_64>(addr).obj()});
         if (t == bitset_c1_128::type_id)
            return f(C1Ad<bitset_c1_128>{_ses->get_ref<bitset_c1_128>(addr).obj()});
         if (t == setlist_u8::type_id)
            return f(SLAd<setlist_u8>{_ses->get_ref<setlist_u8>(addr).obj()});
         if (t == setlist_u16::type_id)
            return f(SLAd<setlist_u16>{_ses->get_ref<setlist_u16>(addr).obj()});
         if (t == setlist_u32::type_id)
            return f(SLAd<setlist_u32>{_ses->get_ref<setlist_u32>(addr).obj()});
         if (t == bitset_r2_node::type_id)
            return f(BSAd<bitset_r2_node, 7>{_ses->get_ref<bitset_r2_node>(addr).obj()});
         if (t == bitset_r4_node::type_id)
            return f(BSAd<bitset_r4_node, 6>{_ses->get_ref<bitset_r4_node>(addr).obj()});
         return f(FullAd{_ses->get_ref<full_radix_node>(addr).obj()});
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
         switch (unsigned(_ses->get_ref<sal::alloc_header>(addr)->type()))
         {
            case unsigned(bucket_node::type_id):  // bucket LEAF: insert (split on overflow)
               return bucket_leaf_insert(addr, key, depth, val);
            case unsigned(setlist_u8::type_id):
               return router_insert<setlist_u8>(addr, key, depth, val, peer_hint);
            case unsigned(setlist_u16::type_id):
               return router_insert_wide<setlist_u16>(addr, key, depth, val, peer_hint);
            case unsigned(setlist_u32::type_id):
               return router_insert_wide<setlist_u32>(addr, key, depth, val, peer_hint);
            case unsigned(bitset_c1_32::type_id):
               return router_insert<bitset_c1_32>(addr, key, depth, val, peer_hint);
            case unsigned(bitset_c1_64::type_id):
               return router_insert<bitset_c1_64>(addr, key, depth, val, peer_hint);
            case unsigned(bitset_c1_128::type_id):
               return router_insert<bitset_c1_128>(addr, key, depth, val, peer_hint);
            case unsigned(bitset_r2_node::type_id):
               return router_insert<bitset_r2_node>(addr, key, depth, val, peer_hint);
            case unsigned(bitset_r4_node::type_id):
               return router_insert<bitset_r4_node>(addr, key, depth, val, peer_hint);
            default:  // full_radix
               return router_insert<full_radix_node>(addr, key, depth, val, peer_hint);
         }
      }

      template <class N>
      ptr_address router_insert(ptr_address addr, bytes key, uint32_t depth, bytes val,
                                sal::alloc_hint peer_hint)
      {
         const ptr_address addr_orig = addr;
         const uint32_t    depth0     = depth;  // where addr's region begins (for bucket collapse)
         const N*          cn        = _ses->get_ref<N>(addr).obj();
         bytes             pfx       = cn->prefix();
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
            // set_branch is the source of truth (a tail's compact base depends on its
            // exact offset, so a probe can't predict it). Make the value, try to set;
            // on failure widen (the value is re-made on the wider node). make_value
            // may have COW'd addr to a clone — release that clone here (the caller
            // releases the original), else it orphans.
            const ptr_address pre = addr;
            smart_address      sab = make_value<N>(addr, suf, val, chint);
            N*                 n   = cow<N>(addr);
            if (n->set_branch(b, sab))
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
            bytes esuf, eval;  // zero-copy views into addr (stable arena, consumed pre-COW)
            term_view(_ses->get_ref<N>(addr).obj(), sa, esuf, eval);
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
            N*            n   = cow<N>(addr);
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
         ptr_address nc = insert_node(child, key, depth + 1, val, chint);
         if (nc != child)
         {
            smart_address nsab = smart_address::from_pointer(nc);
            N*            n    = cow<N>(addr);
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
         const uint32_t depth0 = depth;
         const N*       cn     = _ses->get_ref<N>(addr).obj();
         bytes          pfx    = cn->prefix();
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
            bytes esuf, eval;  // zero-copy views into addr (stable arena, consumed pre-COW)
            term_view(_ses->get_ref<N>(addr).obj(), sa, esuf, eval);
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
            N*            w     = cow<N>(addr);
            if (sa.is_pointer())
               _ses->release(sa.as_pointer());
            w->set_branch(q, smart_address::from_pointer(child));
            return addr;
         }

         // pointer to an inner node → recurse
         ptr_address child = sa.as_pointer();
         ptr_address nc    = insert_node(child, key, depth + uint32_t(cw), val, chint);
         if (nc != child)
         {
            N* w = cow<N>(addr);
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
            cow<full_radix_node>(node)->set_branch(uint8_t(q), ptr);  // direct, no grow/widen
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
            cow<full_radix_node>(node)->set_branch(uint8_t(q), sa);
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
         smart_address tv = _ses->get_ref<N>(addr).obj()->term_value();
         if (tv.is_tail())  // update inline term
         {
            const inline_value* iv = inline_of(_ses->get_ref<N>(addr).obj(), tv);
            _old_value_size        = iv->value_len;
            if (iv->value_len == val.size())
            {
               N*    n   = cow<N>(addr);
               auto* wiv = reinterpret_cast<inline_value*>(reinterpret_cast<char*>(n) + n->size() -
                                                           tv.tail_offset());
               std::memcpy(wiv->data + wiv->suffix_len, val.data(), val.size());
            }
            else
            {
               smart_address sv = make_value<N>(addr, bytes{}, val, peer_hint);  // old leaks
               mut<N>(addr)->set_term_value(sv);
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
               N* n = cow<N>(addr);
               _ses->release(old);
               n->set_term_value(smart_address::from_pointer(tl));
            }
            return;
         }
         smart_address sv = make_value<N>(addr, bytes{}, val, peer_hint);  // new term
         mut<N>(addr)->set_term_value(sv);
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
         int lo = setb, hi = setb, n = 1;  // count the new branch being added
         cn->for_each_entry([&](uint8_t byte, smart_address) {
            lo = byte < lo ? byte : lo;
            hi = byte > hi ? byte : hi;
            ++n;
         });
         if (hi - lo >= To::width || n > To::cap)
            return ptr_address(0);  // range too wide / too many branches → caller widens further
         uint32_t need = ucc::round_up_multiple<sal::cacheline_size>(
             To::head_bytes() + cn->tail_used() + tail_reserve + 64 + To::headroom);
         ptr_address na = _ses->alloc<To>(need, hint, bytes{});
         To*         nn = mut<To>(na);
         nn->alloc_tail(tail_reserve);
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

      std::unique_ptr<sal::allocator> _alloc;
      sal::allocator_session_ptr      _ses;
      ptr_address                     _root;
      ptr_address                     _snap{};
      bool                            _snap_valid     = false;
      int                             _old_value_size = -1;
      bool                            _route_found    = false;
      bool                            _wide_stems     = false;  // gate: off = byte radix
      bool                            _bucket_mode    = false;  // leaf=bucket, split to radix
      bool                            _adaptive       = false;  // gate: radix→bucket on deep-narrow
      bool                            _in_bucket      = false;  // re-entrancy: in a bucket subtree
      cow_counters                    _cc;
   };
}  // namespace heart::radix

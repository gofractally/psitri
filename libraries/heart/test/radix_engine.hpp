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

#include <heart/leaf_node.hpp>
#include <heart/radix_nodes.hpp>

namespace heart::radix
{
   using sal::ptr_address;
   using bytes      = std::span<const uint8_t>;
   using key_view   = std::string_view;
   using value_view = std::string_view;
   using setlist_u8 = setlist_node<uint8_t>;

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
         _alloc->register_type_ops<bitset_r2_node>();
         _alloc->register_type_ops<bitset_r4_node>();
         _alloc->register_type_ops<full_radix_node>();
         _root = make_node<setlist_u8>(bytes{});
      }

      // ── public API (psitri key_view/value_view; values are arbitrary bytes) ───
      int upsert(key_view key, value_view val)
      {
         auto        lock = _ses->lock();
         _old_value_size  = -1;
         bytes       kb   = as_bytes(key);
         ptr_address nr   = insert_node(_root, kb, 0, as_bytes(val), {});
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
                  uint8_t       byte = kb[depth];
                  smart_address sa;
                  if (shape == uint8_t(dispatch_shape::r8))
                     std::memcpy(&sa, base + full_radix_node::branch_offset + 4u * byte, sizeof(sa));
                  else if (shape == uint8_t(dispatch_shape::r2))
                     sa = reinterpret_cast<const bitset_r2_node::segment*>(base + 128 +
                                                                          (byte >> 7) * 128)
                              ->find(byte & 127);
                  else
                     sa = reinterpret_cast<const bitset_r4_node::segment*>(base + 128 +
                                                                          (byte >> 6) * 128)
                              ->find(byte & 63);
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
            // R1: read CL0, dispatch by format (leaf value-holder / prefixed router).
            auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
            if (t == leaf_node::type_id)
               return leaf_read(addr, kb, depth, out);
            bool cont;
            if (t == setlist_u8::type_id)
               cont = route_r1(_ses->get_ref<setlist_u8>(addr).obj(), kb, depth, out, addr);
            else if (t == bitset_r2_node::type_id)
               cont = route_r1(_ses->get_ref<bitset_r2_node>(addr).obj(), kb, depth, out, addr);
            else if (t == bitset_r4_node::type_id)
               cont = route_r1(_ses->get_ref<bitset_r4_node>(addr).obj(), kb, depth, out, addr);
            else
               cont = route_r1(_ses->get_ref<full_radix_node>(addr).obj(), kb, depth, out, addr);
            if (!cont)
               return _route_found;
         }
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

      struct tier_counts
      {
         uint64_t leaf = 0, setlist = 0, bitset_r2 = 0, bitset_r4 = 0, full = 0;
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
      void walk(ptr_address a, tier_counts& tc)
      {
         auto t = _ses->get_ref<sal::alloc_header>(a)->type();
         if (t == leaf_node::type_id)
            ++tc.leaf;
         else if (t == setlist_u8::type_id)
         {
            ++tc.setlist;
            walk_kids<setlist_u8>(a, tc);
         }
         else if (t == bitset_r2_node::type_id)
         {
            ++tc.bitset_r2;
            walk_kids<bitset_r2_node>(a, tc);
         }
         else if (t == bitset_r4_node::type_id)
         {
            ++tc.bitset_r4;
            walk_kids<bitset_r4_node>(a, tc);
         }
         else
         {
            ++tc.full;
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
         return std::make_unique<sal::allocator>(dir, cfg, true);
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
      template <class N>
      ptr_address make_node(bytes pfx, sal::alloc_hint hint = {})
      {
         ptr_address a = _ses->alloc<N>(hint, pfx);
         _ses->get_ref<N>(a).modify().get()->alloc_tail(tail_reserve);  // tail offsets >=16
         stamp<N>(a);
         return a;
      }
      ptr_address make_leaf(bytes suffix, bytes val, sal::alloc_hint hint = {})
      {
         return _ses->alloc<leaf_node>(hint, suffix, val);
      }

      // Copy src's payload into dst (same N): head [alloc_header, head_bytes()) and the
      // tail heap [tail()-tail_used, tail()) → dst's tail (which may be at a different
      // offset when dst is larger), so tail-relative smart_addresses stay valid.
      template <class N>
      static void blit(N* dst, const N* src) noexcept
      {
         constexpr uint32_t hdr = uint32_t(sizeof(sal::alloc_header));
         std::memcpy(reinterpret_cast<char*>(dst) + hdr,
                     reinterpret_cast<const char*>(src) + hdr, N::head_bytes() - hdr);
         uint16_t hu = src->tail_used();
         std::memcpy(dst->tail() - hu, src->tail() - hu, hu);
      }
      template <class N>
      ptr_address clone(ptr_address a, uint32_t extra = 0)
      {
         auto     ref  = _ses->get_ref<N>(a);
         const N* src  = ref.obj();
         uint32_t need = ucc::round_up_multiple<sal::cacheline_size>(
             N::head_bytes() + src->tail_used() + extra + N::headroom);
         ptr_address na = _ses->alloc<N>(need, sal::alloc_hint{}, src->prefix());
         N*          nn = mut<N>(na);
         blit<N>(nn, src);
         nn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         stamp<N>(na);
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
            return ref.modify().get();
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
            return smart_address::from_tail(n->tail_used());
         }
         return smart_address::from_pointer(make_leaf(suffix, value, hint));
      }
      // Copy a terminal's {suffix,value} (tail or SAL leaf) into `buf` (<=510 bytes).
      void term_get(const sal::alloc_header* node, smart_address sa, uint8_t* buf, bytes& suf,
                    bytes& val) const
      {
         const inline_value* iv = sa.is_tail() ? inline_of(node, sa)
                                               : &_ses->get_ref<leaf_node>(sa.as_pointer()).obj()->value;
         int s = iv->suffix_len, v = iv->value_len;
         std::memcpy(buf, iv->data, size_t(s + v));
         suf = {buf, size_t(s)};
         val = {buf + s, size_t(v)};
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
         smart_address sa = n->find(key[depth]);
         if (sa.is_null())
         {
            _route_found = false;
            return false;
         }
         if (sa.is_tail())
         {
            _route_found = inline_at(n, sa, key.subspan(depth + 1), out);
            return false;
         }
         addr = sa.as_pointer();
         ++depth;
         return true;
      }

      // ── insert (tier dispatch is the ONLY place type matters) ───────────────────
      ptr_address insert_node(ptr_address addr, bytes key, uint32_t depth, bytes val,
                              sal::alloc_hint peer_hint)
      {
         auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
         if (t == setlist_u8::type_id)
            return router_insert<setlist_u8>(addr, key, depth, val, peer_hint);
         if (t == bitset_r2_node::type_id)
            return router_insert<bitset_r2_node>(addr, key, depth, val, peer_hint);
         if (t == bitset_r4_node::type_id)
            return router_insert<bitset_r4_node>(addr, key, depth, val, peer_hint);
         return router_insert<full_radix_node>(addr, key, depth, val, peer_hint);
      }

      template <class N>
      ptr_address router_insert(ptr_address addr, bytes key, uint32_t depth, bytes val,
                                sal::alloc_hint peer_hint)
      {
         const ptr_address addr_orig = addr;
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
         bool is_term = sa.is_tail();
         if (sa.is_pointer())
            is_term = _ses->get_ref<sal::alloc_header>(sa.as_pointer())->type() == leaf_node::type_id;
         if (is_term)
         {
            uint8_t tbuf[512];
            bytes   esuf, eval;
            term_get(_ses->get_ref<N>(addr).obj(), sa, tbuf, esuf, eval);
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
            // diverge → child setlist holding both continuations (inline if small)
            int           cp    = common_prefix_len(esuf, suf);
            ptr_address   child = make_node<setlist_u8>(esuf.subspan(0, size_t(cp)), chint);
            add_continuation(child, esuf.subspan(size_t(cp)), eval, chint);
            add_continuation(child, suf.subspan(size_t(cp)), val, chint);
            smart_address csa = smart_address::from_pointer(child);
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
         ptr_address nc    = insert_node(child, key, depth + 1, val, chint);
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
         uint8_t     pbuf[256];
         std::memcpy(pbuf, d->prefix().data() + p + 1, size_t(newpl));
         reprefix<N>(d, bytes{pbuf, size_t(newpl)});
         stamp<N>(dnode);

         ptr_address np = make_node<setlist_u8>(key.subspan(depth, size_t(p)), peer_hint);
         mut<setlist_u8>(np)->set_branch(ob, smart_address::from_pointer(dnode));
         if (dnode == addr)
            _ses->retain(dnode);  // in-place demote: np is a 2nd parent; caller releases addr

         add_continuation(np, key.subspan(depth + uint32_t(p)), val, peer_hint);
         return np;
      }
      template <class N>
      void reprefix(N* n, bytes p)
      {
         if constexpr (std::is_same_v<N, setlist_u8>)
            n->reprefix(p);
         else
            n->set_prefix(p);
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

      std::unique_ptr<sal::allocator> _alloc;
      sal::allocator_session_ptr      _ses;
      ptr_address                     _root;
      ptr_address                     _snap{};
      bool                            _snap_valid     = false;
      int                             _old_value_size = -1;
      bool                            _route_found    = false;
   };
}  // namespace heart::radix

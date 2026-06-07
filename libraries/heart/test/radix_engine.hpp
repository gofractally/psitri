// The real engine, rebuilt on the DESIGNED segmented node taxonomy
// (heart/radix_nodes.hpp) — not the hand-rolled prototype (sal_radix_tree.hpp,
// now reference). Routing is by the control-block dispatch_shape: r8/r4/r2 jump
// straight to the segment cacheline and SKIP the node header (CL0); only r1
// (single-cacheline tiers, or a node carrying a prefix) reads CL0.
//
// Mutation model: COW-to-root (ref-based unique|shared). The nodes derive
// `versioned`, but the MVCC-in-place + version_history protocol is NOT used yet.
//
// PHASE 2a: leaf_node (terminal) + setlist_node<u8> (sparse 1-cacheline inner,
// ART Node16) + full_radix_node (R8 dense), with prefix compression and a
// setlist→full upgrade. Branches/term are pure pointers (a terminal is a leaf).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

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
         _alloc->register_type_ops<full_radix_node>();
         _root = make_node<setlist_u8>(bytes{});  // empty root: a sparse setlist
      }

      // ── public API (psitri key_view/value_view; values are arbitrary bytes) ───
      int upsert(key_view key, value_view val)
      {
         auto        lock = _ses->lock();
         _old_value_size  = -1;
         bytes       kb   = as_bytes(key);
         ptr_address nr   = insert_node(_root, kb, 0, as_bytes(val));
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
            auto ref = _ses->get_ref<sal::alloc_header>(addr);
            // SKIP-HEADER FAST PATH: dispatch shape came from the control block, so
            // route from the segment cacheline without touching CL0.
            if (ref.cb_type() == uint8_t(dispatch_shape::r8))
            {
               if (depth < kb.size())
               {
                  const char*   base = reinterpret_cast<const char*>(ref.obj());
                  smart_address sa;
                  std::memcpy(&sa, base + full_radix_node::branch_offset + 4u * kb[depth],
                              sizeof(sa));  // branches[byte] — one segment line, no CL0
                  if (sa.is_null())
                     return false;
                  addr = sa.as_pointer();
                  ++depth;
                  continue;
               }
               const full_radix_node* n = ref.as<full_radix_node>().obj();
               if (n->term_value().is_null())
                  return false;
               return leaf_read(n->term_value().as_pointer(), kb, depth, out);
            }
            // R1: read the header and dispatch by format.
            auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
            if (t == leaf_node::type_id)
               return leaf_read(addr, kb, depth, out);
            if (t == setlist_u8::type_id)
            {
               if (!route_r1(_ses->get_ref<setlist_u8>(addr).obj(), kb, depth, out, addr))
                  return _route_done ? _route_found : false;
               continue;
            }
            // full_radix carrying a prefix (stamped r1)
            if (!route_r1(_ses->get_ref<full_radix_node>(addr).obj(), kb, depth, out, addr))
               return _route_done ? _route_found : false;
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

     private:
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

      // ── generic node helpers ──────────────────────────────────────────────────
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
      ptr_address make_node(bytes pfx)
      {
         ptr_address a = _ses->alloc<N>(sal::alloc_hint{}, pfx);
         stamp<N>(a);
         return a;
      }
      ptr_address make_leaf(bytes suffix, bytes val)
      {
         return _ses->alloc<leaf_node>(sal::alloc_hint{}, suffix, val);
      }

      // SHARED-mode COW: fresh same-size node, byte-copy the payload (everything
      // after the alloc_header — nodes are fixed-layout with no tail heap), RETAIN
      // every pointer child (the clone is a 2nd parent).
      template <class N>
      ptr_address clone(ptr_address a)
      {
         auto        ref = _ses->get_ref<N>(a);
         const N*    src = ref.obj();
         ptr_address na  = make_node<N>(src->prefix());
         N*          nn  = mut<N>(na);
         constexpr uint32_t hdr = uint32_t(sizeof(sal::alloc_header));
         std::memcpy(reinterpret_cast<char*>(nn) + hdr,
                     reinterpret_cast<const char*>(src) + hdr, src->size() - hdr);
         nn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         stamp<N>(na);
         return na;
      }
      template <class N>
      N* cow(ptr_address& addr)
      {
         auto ref = _ses->get_ref<N>(addr);
         if (ref.ref() > 1)
         {
            addr = clone<N>(addr);
            return mut<N>(addr);
         }
         return ref.modify().get();
      }

      // ── insert (tier dispatch) ─────────────────────────────────────────────────
      ptr_address insert_node(ptr_address addr, bytes key, uint32_t depth, bytes val)
      {
         auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
         if (t == leaf_node::type_id)
            return leaf_insert(addr, key, depth, val);
         if (t == setlist_u8::type_id)
            return router_insert<setlist_u8>(addr, key, depth, val);
         return router_insert<full_radix_node>(addr, key, depth, val);
      }

      template <class N>
      ptr_address router_insert(ptr_address addr, bytes key, uint32_t depth, bytes val)
      {
         const uint32_t depth0 = depth;
         const N*       cn     = _ses->get_ref<N>(addr).obj();
         bytes          pfx    = cn->prefix();
         int avail = int(key.size() - depth);
         int maxp  = std::min(int(pfx.size()), avail);
         int p     = common_prefix_len(key.subspan(depth, size_t(maxp)), pfx.subspan(0, size_t(maxp)));
         if (p < int(pfx.size()))
            return prefix_split<N>(addr, key, depth, p, val);

         depth += uint32_t(pfx.size());
         if (depth == key.size())  // key ends at this node → term
         {
            set_term<N>(addr, val);
            return addr;
         }
         uint8_t       b  = key[depth];
         smart_address sa = cn->find(b);
         if (sa.is_null())  // new branch
         {
            if constexpr (!std::is_same_v<N, full_radix_node>)
            {
               if (!cn->branch_has_room(b))  // grow this tier, then retry
               {
                  ptr_address up = upgrade<full_radix_node, N>(addr);
                  return insert_node(up, key, depth0, val);  // caller releases old `addr`
               }
            }
            ptr_address lf = make_leaf(key.subspan(depth + 1), val);
            N*          n  = cow<N>(addr);
            n->set_branch(b, smart_address::from_pointer(lf));
            return addr;
         }
         ptr_address child = sa.as_pointer();
         ptr_address nc    = insert_node(child, key, depth + 1, val);
         if (nc != child)
         {
            N* n = cow<N>(addr);
            _ses->release(child);
            n->set_branch(b, smart_address::from_pointer(nc));
         }
         return addr;
      }

      // Build a wider tier (To) from From's entries; To co-owns From's children via
      // retain. Does NOT release `from_addr` — the caller releases it as the replaced
      // node (so a snapshot still referencing it stays valid).
      template <class To, class From>
      ptr_address upgrade(ptr_address from_addr)
      {
         const From* cn = _ses->get_ref<From>(from_addr).obj();
         ptr_address na = make_node<To>(cn->prefix());
         To*         nn = mut<To>(na);
         nn->set_term_value(cn->term_value());
         cn->for_each_entry([&](uint8_t byte, smart_address br) { nn->set_branch(byte, br); });
         cn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         return na;
      }

      template <class N>
      void set_term(ptr_address& addr, bytes val)
      {
         const N* cn = _ses->get_ref<N>(addr).obj();
         if (!cn->term_value().is_null())
         {
            ptr_address old = cn->term_value().as_pointer();
            ptr_address tl  = update_leaf_value(old, val);
            if (tl != old)
            {
               N* n = cow<N>(addr);
               _ses->release(old);
               n->set_term_value(smart_address::from_pointer(tl));
            }
            return;
         }
         ptr_address lf = make_leaf(bytes{}, val);
         N*          n  = cow<N>(addr);
         n->set_term_value(smart_address::from_pointer(lf));
      }

      // Split `addr`'s prefix at p: a new setlist parent takes key[depth..depth+p];
      // the demoted node (prefix loses p+1 bytes) hangs under its old prefix byte.
      template <class N>
      ptr_address prefix_split(ptr_address addr, bytes key, uint32_t depth, int p, bytes val)
      {
         uint8_t ob = _ses->get_ref<N>(addr).obj()->prefix()[size_t(p)];

         ptr_address dnode = addr;
         N*          d     = cow<N>(dnode);
         int         newpl = int(d->prefix_length()) - (p + 1);
         uint8_t     pbuf[256];
         std::memcpy(pbuf, d->prefix().data() + p + 1, size_t(newpl));
         reprefix<N>(d, bytes{pbuf, size_t(newpl)});
         stamp<N>(dnode);  // prefix changed → r8/r1 may flip

         ptr_address np = make_node<setlist_u8>(key.subspan(depth, size_t(p)));
         mut<setlist_u8>(np)->set_branch(ob, smart_address::from_pointer(dnode));
         if (dnode == addr)
            _ses->retain(dnode);  // in-place demote: np is a 2nd parent; caller releases addr

         uint32_t d2 = depth + uint32_t(p);
         if (d2 == key.size())
         {
            ptr_address lf = make_leaf(bytes{}, val);
            mut<setlist_u8>(np)->set_term_value(smart_address::from_pointer(lf));
         }
         else
         {
            ptr_address lf = make_leaf(key.subspan(d2 + 1), val);
            mut<setlist_u8>(np)->set_branch(key[d2], smart_address::from_pointer(lf));
         }
         return np;
      }

      // Replace a node's prefix in place (the two tiers store prefixes differently).
      template <class N>
      void reprefix(N* n, bytes p)
      {
         if constexpr (std::is_same_v<N, setlist_u8>)
            n->reprefix(p);  // setlist must restow stems after the prefix
         else
            n->set_prefix(p);  // full_radix: prefix is independent of branches
      }

      // ── descent helper for r1 nodes (setlist / prefixed full_radix) ────────────
      // Returns true to continue the loop with `addr` advanced to a child; false if
      // the lookup is resolved (sets _route_done/_route_found and `out`).
      template <class N>
      bool route_r1(const N* n, bytes key, uint32_t& depth, value_view& out, ptr_address& addr)
      {
         bytes pfx = n->prefix();
         if (pfx.size() > key.size() - depth ||
             (pfx.size() && std::memcmp(pfx.data(), key.data() + depth, pfx.size()) != 0))
         {
            _route_done = true;
            _route_found = false;
            return false;
         }
         depth += uint32_t(pfx.size());
         if (depth == key.size())
         {
            _route_done = true;
            if (n->term_value().is_null())
            {
               _route_found = false;
               return false;
            }
            _route_found = leaf_read(n->term_value().as_pointer(), key, depth, out);
            return false;
         }
         smart_address sa = n->find(key[depth]);
         if (sa.is_null())
         {
            _route_done  = true;
            _route_found = false;
            return false;
         }
         addr = sa.as_pointer();
         ++depth;
         return true;
      }

      // ── leaf ────────────────────────────────────────────────────────────────
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

      ptr_address leaf_insert(ptr_address addr, bytes key, uint32_t depth, bytes val)
      {
         const leaf_node* ln  = _ses->get_ref<leaf_node>(addr).obj();
         bytes            suf = ln->suffix();
         bytes            rem = key.subspan(depth);
         int              cp  = common_prefix_len(rem, suf);
         if (cp == int(rem.size()) && cp == int(suf.size()))  // exact → update value
         {
            _old_value_size = int(ln->value_bytes().size());
            return update_leaf_value(addr, val);
         }

         uint8_t lval[inline_value::max_inline_size];  // stash old value
         bytes   ov = ln->value_bytes();
         std::memcpy(lval, ov.data(), ov.size());
         bytes oldval{lval, ov.size()};

         ptr_address node = make_node<setlist_u8>(rem.subspan(0, size_t(cp)));
         if (cp == int(suf.size()))  // old key ends at node → term
         {
            ptr_address tl = make_leaf(bytes{}, oldval);
            mut<setlist_u8>(node)->set_term_value(smart_address::from_pointer(tl));
         }
         else
         {
            ptr_address cl = make_leaf(suf.subspan(size_t(cp + 1)), oldval);
            mut<setlist_u8>(node)->set_branch(suf[size_t(cp)], smart_address::from_pointer(cl));
         }
         // old leaf at `addr` is released by the caller (replaced child)

         if (cp == int(rem.size()))  // new key ends at node → term
         {
            ptr_address tl = make_leaf(bytes{}, val);
            mut<setlist_u8>(node)->set_term_value(smart_address::from_pointer(tl));
         }
         else
         {
            ptr_address nl = make_leaf(rem.subspan(size_t(cp + 1)), val);
            mut<setlist_u8>(node)->set_branch(rem[size_t(cp)], smart_address::from_pointer(nl));
         }
         return node;
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

      std::unique_ptr<sal::allocator> _alloc;
      sal::allocator_session_ptr      _ses;
      ptr_address                     _root;
      ptr_address                     _snap{};
      bool                            _snap_valid     = false;
      int                             _old_value_size = -1;
      bool                            _route_done     = false;
      bool                            _route_found    = false;
   };
}  // namespace heart::radix

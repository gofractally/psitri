// The real engine, rebuilt on the DESIGNED segmented node taxonomy
// (heart/radix_nodes.hpp) — not the hand-rolled prototype (sal_radix_tree.hpp,
// now reference). Routing is by the control-block dispatch_shape: r8/r4/r2 jump
// straight to the segment cacheline and SKIP the node header (CL0); only r1
// (small single-cacheline tiers, or a node carrying a prefix) reads CL0.
//
// Mutation model: COW-to-root (ref-based unique|shared). The nodes derive
// `versioned`, but the MVCC-in-place + version_history protocol is NOT used yet.
//
// PHASE 1: leaf_node (terminal, one inline suffix+value) + full_radix_node (R8
// inner) with prefix compression. Branches and term are pure pointers (a terminal
// is always a leaf_node), matching design3.md's "R8 = 32 consecutive ptr_addresses".
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
         _alloc->register_type_ops<full_radix_node>();
         _root = make_full(bytes{});  // empty root (prefix-less R8)
      }

      // ── public API (mirrors psitri key_view/value_view) ──────────────────────
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
            // SKIP-HEADER FAST PATH: the dispatch shape came from the control block
            // (loaded to resolve the location), so route from the segment cacheline
            // without touching CL0.
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
                  addr = sa.as_pointer();  // pure pointer (leaf or inner)
                  ++depth;
                  continue;
               }
               // key ends here → term (in the header)
               const full_radix_node* n = ref.as<full_radix_node>().obj();
               if (n->term_value().is_null())
                  return false;
               return leaf_read(n->term_value().as_pointer(), kb, depth, out);
            }
            // R1: read the header and dispatch by format.
            auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
            if (t == leaf_node::type_id)
               return leaf_read(addr, kb, depth, out);
            // full_radix carrying a prefix (stamped r1)
            const full_radix_node* n   = _ses->get_ref<full_radix_node>(addr).obj();
            bytes                  pfx = n->prefix();
            if (pfx.size() > kb.size() - depth ||
                std::memcmp(pfx.data(), kb.data() + depth, pfx.size()) != 0)
               return false;
            depth += uint32_t(pfx.size());
            if (depth == kb.size())
            {
               if (n->term_value().is_null())
                  return false;
               return leaf_read(n->term_value().as_pointer(), kb, depth, out);
            }
            smart_address sa = n->find(kb[depth]);
            if (sa.is_null())
               return false;
            addr = sa.as_pointer();
            ++depth;
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

      // ── node construction ────────────────────────────────────────────────────
      ptr_address make_leaf(bytes suffix, bytes val)
      {
         return _ses->alloc<leaf_node>(sal::alloc_hint{}, suffix, val);
      }
      ptr_address make_full(bytes pfx)
      {
         ptr_address a = _ses->alloc<full_radix_node>(sal::alloc_hint{}, pfx);
         stamp_full(a);
         return a;
      }
      void stamp_full(ptr_address a)
      {
         auto ref = _ses->get_ref<full_radix_node>(a);
         ref.control().set_type(uint8_t(ref->stamp_shape()));  // r8 if prefix-less else r1
      }

      template <class N>
      N* mut(ptr_address a)
      {
         return _ses->get_ref<N>(a).modify().get();
      }

      // SHARED-mode COW of a full_radix: fresh node, copy branches+term+prefix,
      // RETAIN every pointer child (the clone is a 2nd parent).
      ptr_address clone_full(ptr_address a)
      {
         auto             ref = _ses->get_ref<full_radix_node>(a);
         ptr_address      na  = make_full(ref->prefix());
         full_radix_node* nn  = mut<full_radix_node>(na);
         nn->branches         = ref->branches;  // POD copy of the 8 segment cachelines
         nn->set_term_value(ref->term_value());
         nn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         return na;
      }
      // Make the full_radix at `addr` writable (COW if shared). Fixed size → no grow.
      full_radix_node* cow_full(ptr_address& addr)
      {
         auto ref = _ses->get_ref<full_radix_node>(addr);
         if (ref.ref() > 1)
         {
            addr = clone_full(addr);
            return mut<full_radix_node>(addr);
         }
         return ref.modify().get();
      }

      // ── insert ────────────────────────────────────────────────────────────────
      ptr_address insert_node(ptr_address addr, bytes key, uint32_t depth, bytes val)
      {
         auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
         if (t == leaf_node::type_id)
            return leaf_insert(addr, key, depth, val);
         return full_insert(addr, key, depth, val);
      }

      ptr_address full_insert(ptr_address addr, bytes key, uint32_t depth, bytes val)
      {
         const full_radix_node* cn  = _ses->get_ref<full_radix_node>(addr).obj();
         bytes                  pfx = cn->prefix();
         int avail = int(key.size() - depth);
         int maxp  = std::min(int(pfx.size()), avail);
         int p     = common_prefix_len(key.subspan(depth, size_t(maxp)), pfx.subspan(0, size_t(maxp)));
         if (p < int(pfx.size()))
            return prefix_split(addr, key, depth, p, val);

         depth += uint32_t(pfx.size());
         if (depth == key.size())  // key ends at this node → term
         {
            set_node_term(addr, val);
            return addr;
         }
         uint8_t       b  = key[depth];
         smart_address sa = cn->find(b);
         if (sa.is_null())  // brand-new branch → a leaf
         {
            ptr_address      lf = make_leaf(key.subspan(depth + 1), val);
            full_radix_node* n  = cow_full(addr);
            n->set_branch(b, smart_address::from_pointer(lf));
            return addr;
         }
         ptr_address child = sa.as_pointer();
         ptr_address nc    = insert_node(child, key, depth + 1, val);
         if (nc != child)
         {
            full_radix_node* n = cow_full(addr);
            _ses->release(child);
            n->set_branch(b, smart_address::from_pointer(nc));
         }
         return addr;
      }

      // Key ends at a full_radix node: point term at a value leaf (empty suffix).
      void set_node_term(ptr_address& addr, bytes val)
      {
         const full_radix_node* cn = _ses->get_ref<full_radix_node>(addr).obj();
         if (!cn->term_value().is_null())  // update existing term value
         {
            ptr_address old = cn->term_value().as_pointer();
            ptr_address tl  = update_leaf_value(old, val);
            if (tl != old)
            {
               full_radix_node* n = cow_full(addr);
               _ses->release(old);
               n->set_term_value(smart_address::from_pointer(tl));
            }
            return;
         }
         ptr_address      lf = make_leaf(bytes{}, val);
         full_radix_node* n  = cow_full(addr);
         n->set_term_value(smart_address::from_pointer(lf));
      }

      // Split `addr`'s prefix at p: a new parent full_radix takes key[depth..depth+p];
      // the demoted node (prefix loses p+1 bytes) hangs under its old prefix byte.
      ptr_address prefix_split(ptr_address addr, bytes key, uint32_t depth, int p, bytes val)
      {
         uint8_t ob;
         {
            const full_radix_node* cn = _ses->get_ref<full_radix_node>(addr).obj();
            ob                        = cn->prefix()[size_t(p)];  // old node's diverging byte
         }
         // demote: COW the node and shorten its prefix to prefix[p+1..]
         ptr_address      dnode = addr;
         full_radix_node* d     = cow_full(dnode);
         int              newpl = int(d->prefix_length()) - (p + 1);
         uint8_t          pbuf[256];
         std::memcpy(pbuf, d->prefix().data() + p + 1, size_t(newpl));
         d->set_prefix(bytes{pbuf, size_t(newpl)});
         stamp_full(dnode);  // prefix changed → r8/r1 may flip

         // new parent over key[depth..depth+p]
         ptr_address np = make_full(key.subspan(depth, size_t(p)));
         mut<full_radix_node>(np)->set_branch(ob, smart_address::from_pointer(dnode));
         if (dnode == addr)
            _ses->retain(dnode);  // in-place demote: np is a 2nd parent; the caller
                                  // will release `addr` (== child) as the replaced node

         uint32_t d2 = depth + uint32_t(p);
         if (d2 == key.size())  // new key ends at the parent → term
         {
            ptr_address lf = make_leaf(bytes{}, val);
            mut<full_radix_node>(np)->set_term_value(smart_address::from_pointer(lf));
         }
         else
         {
            ptr_address lf = make_leaf(key.subspan(d2 + 1), val);
            mut<full_radix_node>(np)->set_branch(key[d2], smart_address::from_pointer(lf));
         }
         return np;
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

      // Insert into / split a leaf. Returns the (possibly new) subtree root for addr.
      ptr_address leaf_insert(ptr_address addr, bytes key, uint32_t depth, bytes val)
      {
         const leaf_node* ln  = _ses->get_ref<leaf_node>(addr).obj();
         bytes            suf = ln->suffix();
         bytes            rem = key.subspan(depth);
         int              cp  = common_prefix_len(rem, suf);
         if (cp == int(rem.size()) && cp == int(suf.size()))  // exact match → update value
         {
            _old_value_size = int(ln->value_bytes().size());
            return update_leaf_value(addr, val);
         }

         // stash the old leaf's value, then split into a full_radix over the common bytes
         uint8_t lval[inline_value::max_inline_size];
         bytes   ov = ln->value_bytes();
         std::memcpy(lval, ov.data(), ov.size());
         bytes oldval{lval, ov.size()};

         ptr_address node = make_full(rem.subspan(0, size_t(cp)));
         if (cp == int(suf.size()))  // old key ends at node → term
         {
            ptr_address tl = make_leaf(bytes{}, oldval);
            mut<full_radix_node>(node)->set_term_value(smart_address::from_pointer(tl));
         }
         else
         {
            ptr_address cl = make_leaf(suf.subspan(size_t(cp + 1)), oldval);
            mut<full_radix_node>(node)->set_branch(suf[size_t(cp)], smart_address::from_pointer(cl));
         }
         // NOTE: the old leaf at `addr` is NOT released here — the caller releases
         // the old child when it sees the returned subtree differs (nc != child).

         if (cp == int(rem.size()))  // new key ends at node → term
         {
            ptr_address tl = make_leaf(bytes{}, val);
            mut<full_radix_node>(node)->set_term_value(smart_address::from_pointer(tl));
         }
         else
         {
            ptr_address nl = make_leaf(rem.subspan(size_t(cp + 1)), val);
            mut<full_radix_node>(node)->set_branch(rem[size_t(cp)], smart_address::from_pointer(nl));
         }
         return node;
      }

      // Update a leaf's value in place if unique+same-size; else a fresh leaf
      // (caller swaps the pointer + releases the old addr).
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
   };
}  // namespace heart::radix

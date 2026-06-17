// gcart::tree — adaptive radix tree on the sal2 FIX region.
//
// Reads: direct line-index descent (no control block), prefix-compressed,
// SWAR setlist scan / direct node256 index. Writes: COW via per-node refcount —
// privatize-on-descent (clone-if-shared, retaining children; then mutate the
// private node in place). A private node grows by COPY-CONSTRUCT-INSERT into a
// bigger node (one pass from the stable husk) then frees the husk's lines, the
// child pointers transferred (the design's "no copy-then-modify, no temporaries,
// stable source"). Reclamation is recursive refcount release (v1; version-GC is
// the design's later phase).
#pragma once
#include <gcart/node.hpp>

#include <cassert>
#include <optional>
#include <string>
#include <vector>

namespace gcart
{
   using sv = std::string_view;

   inline uint32_t lcp(sv a, sv b) noexcept
   {
      const uint32_t n = uint32_t(a.size() < b.size() ? a.size() : b.size());
      uint32_t       i = 0;
      while (i < n && a[i] == b[i])
         ++i;
      return i;
   }

   // recursive refcount release of a subtree, using the arena alone (no session)
   // — reusable by snapshot holders. Frees a node's lines once its ref hits 0,
   // recursing into children + term.
   inline void release_subtree(sal2::arena& a, nptr n)
   {
      if (n == null_ptr || !a.release_one(n))
         return;
      void* p = a.get(n);
      if (is_leaf(p)) { a.free_lines(n, static_cast<leaf*>(p)->nlines); return; }
      if (type_of(p) == uint8_t(ntype::setlist))
      {
         auto* s = static_cast<setlist*>(p);
         for (uint32_t i = 0; i < s->h.n; ++i) release_subtree(a, s->children()[i]);
         if (s->h.term != null_ptr) release_subtree(a, s->h.term);
         a.free_lines(n, s->h.nlines);
      }
      else
      {
         auto* nd = static_cast<node256*>(p);
         const nptr* c = nd->children();
         for (uint32_t b = 0; b < 256; ++b) if (c[b] != null_ptr) release_subtree(a, c[b]);
         if (nd->h.term != null_ptr) release_subtree(a, nd->h.term);
         a.free_lines(n, nd->h.nlines);
      }
   }

   // ── smart_ptr: a held root IS the pin (no explicit pin/unpin) ──────────────
   // An owning handle to a tree root. Holding one keeps that exact version
   // readable under MVCC — a concurrent writer COWs and frees OTHER versions
   // around it. The handle's lifetime is the pin: ctor acquires, dtor releases,
   // copy shares (retain), move transfers. acquire() reads a root slot
   // LOCK-FREE (try_retain + re-verify) so concurrent readers never block each
   // other or a writer; give() relinquishes ownership without releasing (the
   // commit/abort root hand-off, never take()/re-adopt a raw address).
   class smart_ptr
   {
     public:
      smart_ptr() = default;

      // Lock-free acquire of root slot `slot`'s currently-committed snapshot.
      static smart_ptr acquire(sal2::arena& a, uint32_t slot)
      {
         for (;;)
         {
            const nptr r = a.get_root(slot);
            if (r == null_ptr)
               return smart_ptr(&a, null_ptr);
            if (!a.try_retain(r))
               continue;                          // lost the race to a freeing writer — retry
            if (r == a.get_root(slot))
               return smart_ptr(&a, r);           // pinned and still the published root
            release_subtree(a, r);                // root moved under us — undo this pin, retry
         }
      }

      // Adopt a ref the caller already owns (a freshly built / transferred root).
      static smart_ptr adopt(sal2::arena& a, nptr root) { return smart_ptr(&a, root); }

      smart_ptr(const smart_ptr& o) : _a(o._a), _root(o._root)
      {
         if (_a && _root != null_ptr) _a->retain(_root);
      }
      smart_ptr& operator=(const smart_ptr& o)
      {
         if (this != &o) { reset(); _a = o._a; _root = o._root; if (_a && _root != null_ptr) _a->retain(_root); }
         return *this;
      }
      smart_ptr(smart_ptr&& o) noexcept : _a(o._a), _root(o._root) { o._root = null_ptr; }
      smart_ptr& operator=(smart_ptr&& o) noexcept
      {
         if (this != &o) { reset(); _a = o._a; _root = o._root; o._root = null_ptr; }
         return *this;
      }
      ~smart_ptr() { reset(); }

      nptr         root() const noexcept { return _root; }
      sal2::arena* arena() const noexcept { return _a; }
      explicit operator bool() const noexcept { return _root != null_ptr; }

      // Relinquish ownership WITHOUT releasing — the new owner inherits the ref
      // (commit publishes / abort restores by handing the pin to the live tree).
      nptr give() noexcept { const nptr r = _root; _root = null_ptr; return r; }

      void reset() noexcept
      {
         if (_a && _root != null_ptr) release_subtree(*_a, _root);
         _root = null_ptr;
      }

     private:
      smart_ptr(sal2::arena* a, nptr root) : _a(a), _root(root) {}
      sal2::arena* _a    = nullptr;
      nptr         _root = null_ptr;
   };

   class tree
   {
     public:
      explicit tree(sal2::arena& a) : _a(a), _ses(a.start_session()) {}

      nptr     root() const noexcept { return _root; }
      void     set_root(nptr r) noexcept { _root = r; }
      uint64_t size() const noexcept { return _size; }

      // ── read path ────────────────────────────────────────────────────────
      const leaf* find(sv key) const noexcept { return find_in(_root, key); }

      const leaf* find_in(nptr root, sv key) const noexcept
      {
         nptr     cur   = root;
         uint32_t depth = 0;
         while (cur != null_ptr)
         {
            const void* p = _a.get(cur);
            if (is_leaf(p))
            {
               auto* lf = static_cast<const leaf*>(p);
               return lf->key() == key ? lf : nullptr;
            }
            const sv pfx = node_prefix(cur);
            if (key.size() - depth < pfx.size())
               return nullptr;
            if (!pfx.empty() &&
                std::memcmp(key.data() + depth, pfx.data(), pfx.size()) != 0)
               return nullptr;
            depth += uint32_t(pfx.size());
            const auto* h = static_cast<const inner_hdr*>(p);
            if (depth == key.size())
            {
               if (h->term == null_ptr)
                  return nullptr;
               auto* t = static_cast<const leaf*>(_a.get(h->term));
               return t->key() == key ? t : nullptr;
            }
            const uint8_t byte = uint8_t(key[depth++]);
            cur                = node_child(cur, byte);
         }
         return nullptr;
      }

      // ── upsert ───────────────────────────────────────────────────────────
      // returns true if a new key was inserted (false = updated existing).
      bool upsert(sv key, sv val)
      {
         bool inserted = false;
         _root         = upsert_rec(_root, key, 0, val, inserted);
         if (inserted)
            ++_size;
         return inserted;
      }

      // ── erase ──────────────────────────────────────────────────────────────
      // erase_rec privatizes-on-descent (writable() before mutating), the same
      // top-down COW as upsert_rec. That is only balanced when EVERY node on the
      // descent is actually modified: writable() clones a shared node and drops
      // the original's ref on the contract that the parent will reinstall the
      // clone. upsert always modifies, so it is safe. erase, however, can bail
      // mid-descent when the key turns out to be absent below — leaving a node
      // privatized (original released to ref 1) but NOT reinstalled, so the
      // parent keeps pointing at the released original and the clone is orphaned.
      // That under-counts a node shared with a snapshot → premature free on the
      // snapshot's release. A node's refcount reflects only its parent-pointer
      // count, never whether an ANCESTOR is shared, so we cannot detect "must
      // clone" lazily on the way up. The clean fix: a read-only existence check
      // first (find_in handles term-keys too); only then run the destructive
      // descent, where removal — and therefore reinstallation at every level —
      // is guaranteed.
      bool erase(sv key)
      {
         if (find_in(_root, key) == nullptr)
            return false;  // absent → no mutation, no COW, nothing to unwind
         bool removed = false;
         _root        = erase_rec(_root, key, 0, removed);
         if (removed)
            --_size;
         return removed;
      }

      // release a subtree (e.g. an abandoned root) through refcount.
      void release_root(nptr r) { release_node(r); }

      // in-order walk (key ascending) over a subtree: fn(leaf::key, leaf::value).
      template <class Fn>
      void for_each(Fn&& fn) const { for_each_in(_root, fn); }
      template <class Fn>
      void for_each_in(nptr cur, Fn& fn) const
      {
         if (cur == null_ptr)
            return;
         const void* p = _a.get(cur);
         if (is_leaf(p))
         {
            auto* lf = static_cast<const leaf*>(p);
            fn(lf->key(), lf->value());
            return;
         }
         if (type_of(p) == uint8_t(ntype::setlist))
         {
            auto* s = static_cast<const setlist*>(p);
            if (s->h.term != null_ptr)
               for_each_in(s->h.term, fn);  // key ending here sorts first
            for (uint32_t i = 0; i < s->h.n; ++i)
               for_each_in(s->children()[i], fn);  // keys are sorted in setlist
         }
         else
         {
            auto* nd = static_cast<const node256*>(p);
            if (nd->h.term != null_ptr)
               for_each_in(nd->h.term, fn);
            const nptr* c = nd->children();
            for (uint32_t b = 0; b < 256; ++b)
               if (c[b] != null_ptr)
                  for_each_in(c[b], fn);
         }
      }

      sal2::arena& arena() noexcept { return _a; }

     private:
      // ── small accessors ────────────────────────────────────────────────────
      sv node_prefix(nptr n) const noexcept
      {
         const void* p = _a.get(n);
         return type_of(p) == uint8_t(ntype::setlist)
                    ? static_cast<const setlist*>(p)->prefix()
                    : static_cast<const node256*>(p)->prefix();
      }
      nptr node_child(nptr n, uint8_t byte) const noexcept
      {
         const void* p = _a.get(n);
         if (type_of(p) == uint8_t(ntype::setlist))
         {
            const auto* s   = static_cast<const setlist*>(p);
            const int   idx = s->find_index(byte);
            return idx < 0 ? null_ptr : s->children()[idx];
         }
         return static_cast<const node256*>(p)->child(byte);
      }
      uint32_t node_lines(const void* p) const noexcept
      {
         return static_cast<const uint8_t*>(p)[1];  // nlines at offset 1 (all kinds)
      }

      template <class T, class Op>
      nptr make(const Op& op)
      {
         const uint32_t nl = T::alloc_size(op);
         const nptr     ln = _ses.alloc_lines(nl);
         new (_a.get(ln)) T(nl, op);
         return ln;
      }
      nptr make_leaf(sv key, sv val)
      {
         const uint32_t nl = leaf::alloc_size(key, val);
         const nptr     ln = _ses.alloc_lines(nl);
         new (_a.get(ln)) leaf(nl, key, val);
         return ln;
      }
      nptr make_setlist_fresh(uint32_t cap, sv prefix, nptr term)
      {
         return make<setlist>(setlist::fresh_op{cap, prefix, term});
      }

      template <class Fn>
      void visit_kids(nptr n, Fn&& fn)
      {
         void* p = _a.get(n);
         if (type_of(p) == uint8_t(ntype::setlist))
         {
            auto* s = static_cast<setlist*>(p);
            for (uint32_t i = 0; i < s->h.n; ++i)
               fn(s->children()[i]);
            if (s->h.term != null_ptr)
               fn(s->h.term);
         }
         else
         {
            auto* nd = static_cast<node256*>(p);
            const nptr* c = nd->children();
            for (uint32_t b = 0; b < 256; ++b)
               if (c[b] != null_ptr)
                  fn(c[b]);
            if (nd->h.term != null_ptr)
               fn(nd->h.term);
         }
      }

      nptr clone_verbatim(nptr cur)
      {
         void*          p  = _a.get(cur);
         const uint32_t nl = node_lines(p);
         const nptr     w  = _ses.alloc_lines(nl);
         std::memcpy(_a.get(w), p, size_t(nl) * LINE);
         return w;  // ref == 1
      }

      // privatize a node for mutation: if shared, clone verbatim (retaining all
      // children + term — they are now referenced by both the snapshot and the
      // clone) and release the shared original. Returns a ref==1 node.
      nptr writable(nptr cur)
      {
         if (_a.unique(cur))
            return cur;
         const nptr w = clone_verbatim(cur);
         visit_kids(w, [&](nptr c) { _a.retain(c); });
         (void)_a.release_one(cur);  // cur had ref>1 → never hits zero here
         return w;
      }

      void release_node(nptr n)
      {
         if (n == null_ptr)
            return;
         if (!_a.release_one(n))
            return;  // still referenced elsewhere (snapshot)
         void* p = _a.get(n);
         if (is_leaf(p))
         {
            _a.free_lines(n, static_cast<leaf*>(p)->nlines);
            return;
         }
         visit_kids(n, [&](nptr c) { release_node(c); });
         _a.free_lines(n, node_lines(p));
      }

      // reprefix a PRIVATE node: new node with prefix shortened to [start:],
      // children/term transferred; the husk's lines are freed (not released).
      nptr reprefix_private(nptr cur, uint16_t start)
      {
         void*          p   = _a.get(cur);
         const uint32_t old = node_lines(p);
         nptr           out;
         if (type_of(p) == uint8_t(ntype::setlist))
         {
            auto* s = static_cast<setlist*>(p);
            out     = make<setlist>(setlist::clone_reprefix_op{s, start});
         }
         else
         {
            auto* nd = static_cast<node256*>(p);
            out      = make<node256>(node256::clone_reprefix_op{nd, start});
         }
         _a.free_lines(cur, old);  // husk; children transferred to `out`
         return out;
      }

      // install child for `byte` into PRIVATE inner node `cur`. Returns the
      // (possibly grown) node. Children of `cur` are transferred on a grow.
      nptr put_child_private(nptr cur, uint8_t byte, nptr nc, bool replacing)
      {
         void* p = _a.get(cur);
         if (type_of(p) == uint8_t(ntype::node256))
         {
            auto* nd = static_cast<node256*>(p);
            if (replacing)
               nd->replace_inplace(byte, nc);
            else
               nd->insert_inplace(byte, nc);
            return cur;
         }
         auto* s = static_cast<setlist*>(p);
         if (replacing)
         {
            s->replace_inplace(s->find_index(byte), nc);
            return cur;
         }
         if (s->can_insert())
         {
            s->insert_inplace(byte, nc);
            return cur;
         }
         const uint32_t old = s->h.nlines;
         nptr           out;
         if (s->h.n < setlist::setlist_max)
            out = make<setlist>(setlist::clone_insert_op{s, ins_branch{byte, nc}});
         else
            out = make<node256>(node256::from_setlist_op{s, ins_branch{byte, nc}});
         _a.free_lines(cur, old);  // husk; existing children transferred to `out`
         return out;
      }

      // Returns the new subtree root (ref==1, owned by caller); consumes cur's
      // input ref.
      nptr upsert_rec(nptr cur, sv key, uint32_t depth, sv val, bool& inserted)
      {
         if (cur == null_ptr)
         {
            inserted = true;
            return make_leaf(key, val);
         }
         void* p = _a.get(cur);
         if (is_leaf(p))
         {
            auto* lf = static_cast<leaf*>(p);
            if (lf->key() == key)
            {
               const nptr nl = make_leaf(key, val);  // new value
               release_node(cur);                     // consume input ref
               inserted = false;
               return nl;
            }
            // split: two leaves diverge after `depth`.
            const sv       lsuf = lf->key().substr(depth);
            const sv       ksuf = key.substr(depth);
            const uint32_t cp   = lcp(lsuf, ksuf);
            const sv       common = ksuf.substr(0, cp);
            const bool     key_ends  = (cp == ksuf.size());
            const bool     leaf_ends = (cp == lsuf.size());
            assert(!(key_ends && leaf_ends) && "distinct keys cannot both end here");
            const nptr     nl     = make_leaf(key, val);
            inserted              = true;
            const nptr inner = make_setlist_fresh(2, common, null_ptr);
            auto*      sn    = static_cast<setlist*>(_a.get(inner));
            if (key_ends)
               sn->h.term = nl;
            else
               sn->insert_inplace(uint8_t(ksuf[cp]), nl);
            if (leaf_ends)
               sn->h.term = cur;  // transfer cur's input ref to the term slot
            else
               sn->insert_inplace(uint8_t(lsuf[cp]), cur);  // transfer to a child slot
            return inner;
         }

         // inner node: privatize before any mutation.
         cur       = writable(cur);
         auto*  h  = static_cast<inner_hdr*>(_a.get(cur));
         const sv pfx  = node_prefix(cur);
         const sv ksuf = key.substr(depth);
         const uint32_t cp = lcp(pfx, ksuf);
         if (cp < pfx.size())
         {
            // prefix split. common = pfx[0:cp] == ksuf[0:cp]; take it from ksuf
            // (a view into the stable `key` arg) — `pfx` views `cur`, which
            // reprefix_private frees below, so a pfx-view would dangle.
            const uint8_t curbyte = uint8_t(pfx[cp]);  // read before cur is freed
            const sv      common  = ksuf.substr(0, cp);
            const nptr    cur2    = reprefix_private(cur, uint16_t(cp + 1));
            const bool    key_ends = (cp == ksuf.size());
            const nptr    nl      = key_ends ? null_ptr : make_leaf(key, val);
            inserted              = true;
            const nptr parent = make_setlist_fresh(2, common, key_ends ? make_leaf(key, val)
                                                                       : null_ptr);
            auto*      ps     = static_cast<setlist*>(_a.get(parent));
            ps->insert_inplace(curbyte, cur2);
            if (!key_ends)
               ps->insert_inplace(uint8_t(ksuf[cp]), nl);
            return parent;
         }
         depth += uint32_t(pfx.size());
         if (depth == key.size())
         {
            // key ends at this node → term (two distinct keys can't both end at
            // the same prefix-path, so an existing term has the same key).
            if (h->term != null_ptr)
            {
               const nptr nl = make_leaf(key, val);
               release_node(h->term);
               h->term  = nl;
               inserted = false;
            }
            else
            {
               h->term  = make_leaf(key, val);
               inserted = true;
            }
            return cur;
         }
         const uint8_t byte  = uint8_t(key[depth]);
         const nptr    child = node_child(cur, byte);
         const nptr    nc    = upsert_rec(child, key, depth + 1, val, inserted);
         return put_child_private(cur, byte, nc, child != null_ptr);
      }

      // ── erase (recursive; collapse single-child setlists later) ────────────
      nptr erase_rec(nptr cur, sv key, uint32_t depth, bool& removed)
      {
         if (cur == null_ptr)
            return null_ptr;
         void* p = _a.get(cur);
         if (is_leaf(p))
         {
            if (static_cast<leaf*>(p)->key() == key)
            {
               release_node(cur);
               removed = true;
               return null_ptr;  // parent drops this branch
            }
            return cur;  // not found; keep
         }
         const sv pfx  = node_prefix(cur);
         const sv ksuf = key.substr(depth);
         if (ksuf.size() < pfx.size() ||
             std::memcmp(ksuf.data(), pfx.data(), pfx.size()) != 0)
            return cur;  // prefix mismatch → not present
         cur      = writable(cur);
         auto* h  = static_cast<inner_hdr*>(_a.get(cur));
         depth += uint32_t(pfx.size());
         if (depth == key.size())
         {
            if (h->term == null_ptr)
               return cur;
            release_node(h->term);
            h->term = null_ptr;
            removed = true;
            return collapse(cur);
         }
         const uint8_t byte  = uint8_t(key[depth]);
         const nptr    child = node_child(cur, byte);
         if (child == null_ptr)
            return cur;
         const nptr nc = erase_rec(child, key, depth + 1, removed);
         if (!removed)
            return cur;
         if (nc == null_ptr)
            remove_branch_private(cur, byte);
         else
            put_child_private(cur, byte, nc, true);
         return collapse(cur);
      }

      void remove_branch_private(nptr cur, uint8_t byte)
      {
         void* p = _a.get(cur);
         if (type_of(p) == uint8_t(ntype::setlist))
         {
            auto*    s   = static_cast<setlist*>(p);
            const int idx = s->find_index(byte);
            uint8_t* k   = s->keys();
            nptr*    c   = s->children();
            for (uint32_t i = uint32_t(idx); i + 1 < s->h.n; ++i)
            {
               k[i] = k[i + 1];
               c[i] = c[i + 1];
            }
            --s->h.n;
            k[s->h.n] = 0;
         }
         else
         {
            auto* nd = static_cast<node256*>(p);
            nd->children()[byte] = null_ptr;
            --nd->h.n16;
         }
      }

      // collapse a node that now has 0 children: → its term leaf if it has one,
      // else → null (the parent drops the branch and frees this husk). Handles
      // both node kinds — a node256 that loses its last child must drain too,
      // else empty husks survive a full removal and never return to the freeset.
      // (Single-child merge — n==1, no term → fold prefix into the lone child —
      // is a separate minimality optimization, not required for correctness.)
      nptr collapse(nptr cur)
      {
         void*          p = _a.get(cur);
         const bool     sl = type_of(p) == uint8_t(ntype::setlist);
         const uint32_t n  = sl ? static_cast<setlist*>(p)->h.n
                                : static_cast<node256*>(p)->h.n16;
         if (n != 0)
            return cur;  // still has children — keep
         const inner_hdr* h = static_cast<inner_hdr*>(p);
         const nptr       t = h->term;          // the surviving leaf, or null_ptr
         _a.free_lines(cur, h->nlines);         // husk; term (if any) transferred out
         return t;
      }

      sal2::arena&  _a;
      sal2::session _ses;
      nptr          _root = null_ptr;
      uint64_t      _size = 0;
   };
}  // namespace gcart

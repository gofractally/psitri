// gcart::kv — adb-style adapter over the gcart engine, with lock-free MVCC.
// Follows the psitri smart_ptr discipline (smart_ptr IS the pin, try_retain+verify
// acquire, COW-to-root, give() hand-off) but the root is a (branch_type tag, u32
// ptr) pair packed into an atomic u64 published slot.
#pragma once
#include <gcart/tree.hpp>

#include <sal/config.hpp>   // sal::open_mode (embedder open factory)
#include <sal/mapping.hpp>  // sal::mapping — mmap-backed durable root table (SAL roots pattern)

#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gcart::kv
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   // Runtime config (psitri::dwal::kv_engine contract). `sync_mode` is the commit
   // durability the embedder/tier fans out (delta_art sets bcfg.sync_mode); gcart
   // currently commits to the mmap page cache + syncs the root table on clean close,
   // so sync_mode is accepted and recorded but not yet a per-commit msync (a
   // durability follow-up). A delta_art COLD base only needs the config to exist +
   // be accepted by open(path, mode, cfg).
   struct config
   {
      sal::sync_type sync_mode = sal::sync_type::none;
   };

   inline uint64_t pack(bt t, uint32_t p) noexcept
   {
      return (uint64_t(uint8_t(t)) << 32) | p;
   }
   inline bt rtag(uint64_t r) noexcept
   {
      return bt(uint8_t(r >> 32));
   }
   inline uint32_t rptr(uint64_t r) noexcept
   {
      return uint32_t(r);
   }
   inline constexpr uint64_t null_root = (uint64_t(uint8_t(bt::null)) << 32);

   // ── smart_ptr: a held (tag,ptr) root view — Phase C: ALL version-pin, NO node
   //    refcount anywhere. The snap is a non-owning READ VIEW (e, tag, ptr, session).
   //   HAZARD-PIN (the hot reader cursor AND the owned snapshot): publish() stores the
   //     live (version, root) into a hazard slot (a std::atomic<uint64_t>* the holder
   //     owns: the read_session's, or the owned-snapshot's owned_pin). reset() just
   //     CLEARS the slot. Reclamation is the writer's floor-gated drain (a retired
   //     root's spine is freed once the version floor passes its death) — NO pin ever
   //     touches a node refcount or lock. This is the whole point of Phase C: kill the
   //     writer<->reader/snapshot refcount cacheline ping-pong.
   //   The OWNED snapshot's slot LIFETIME is governed by snapshot's shared_ptr<owned_pin>
   //     (a per-HANDLE count, NOT a per-node count): copies share one published slot;
   //     the last copy releases it. _hp != nullptr ⇒ this snap clears a slot on reset.
   class snap
   {
     public:
      snap() = default;
      // `ses` is the holder's OWN leaf session (a read_session's, a work tree's, or
      // the engine admin session) — BORROWED. A snap lives within one thread's use of
      // its holder, so the borrowed session is single-thread-correct. For a
      // routing-only root the session is unused but still carried.
      // HAZARD publish (Part B): publish the live root word into this reader's hazard
      // slot with the start_ref re-verify loop, take NO refcount. The published word
      // is guaranteed live-or-retired-but-not-yet-drained: the writer's drain (run
      // AFTER it stores the new live root, seq_cst) scans this slot (seq_cst), so a
      // root we publish here is never reclaimed out from under the cursor. (See the
      // create_cursor publish loop comment.)
      static snap publish(gcart::engine& e, std::atomic<uint64_t>& live,
                          std::atomic<uint64_t>* hp, sal2::allocator_session* ses)
      {
         for (;;)
         {
            const uint64_t r = live.load(std::memory_order_seq_cst);
            hp->store(r, std::memory_order_seq_cst);              // publish FIRST...
            if (live.load(std::memory_order_seq_cst) == r)       // ...then re-verify
               return snap(&e, rtag(r), rptr(r), ses, hp);       // r was live after publish
            // root moved between our load and store: retry so we never pin a root that
            // was retired-and-drained before our publish became visible.
         }
      }
      // VERSION-AWARE publish (Phase A item 2): mirrors start_ver create_cursor — the
      // reader publishes (version, root) so floor_() can read the pinned version. gcart's
      // packed root fills its u64, so the version rides a PARALLEL slot (hv) next to the
      // root hazard slot (hp): publish version FIRST, then the root, then re-verify the
      // root is still live (the seq_cst publish/scan race-close is on the ROOT word, the
      // one a retiring writer republishes — the version is its companion). If the root
      // moved, retry, re-reading the matching committed version. hv = _hz_version[slot],
      // vsrc = root_version(slot) the commit wrote alongside the live root.
      static snap publish(gcart::engine& e, std::atomic<uint64_t>& live,
                          std::atomic<uint64_t>& vsrc, std::atomic<uint64_t>* hp,
                          std::atomic<uint64_t>* hv, sal2::allocator_session* ses)
      {
         for (;;)
         {
            const uint64_t r = live.load(std::memory_order_seq_cst);
            const uint64_t v = vsrc.load(std::memory_order_seq_cst);  // the root's commit version
            hv->store(v, std::memory_order_seq_cst);              // publish version FIRST...
            hp->store(r, std::memory_order_seq_cst);              // ...then the root...
            if (live.load(std::memory_order_seq_cst) == r)       // ...then re-verify root
               return snap(&e, rtag(r), rptr(r), ses, hp, hv);   // r was live after publish
            // root moved between our load and store: retry (re-read the live version too)
         }
      }
      // ADOPT an existing owned (tag,ptr) WITHOUT retaining — the caller transfers a
      // ref it already holds (the set_root exchange: a work tree's root → a snapshot).
      static snap adopt(gcart::engine& e, bt t, uint32_t p, sal2::allocator_session* ses) noexcept
      {
         return snap(&e, t, p, ses);
      }
      // Copy = a non-owning read view (Phase C: no node refcount to bump). A copy
      // never inherits the hazard slot (only ONE snap clears a given _hp); for the
      // OWNED snapshot the SLOT lifetime is governed by snapshot's shared_ptr<owned_pin>,
      // so a plain (tag,ptr) view copy is correct — the pin stays held by the handle.
      snap(const snap& o) : _e(o._e), _t(o._t), _p(o._p), _ses(o._ses) {}
      snap& operator=(const snap& o)
      {
         if (this != &o)
         {
            reset();
            _e   = o._e;
            _t   = o._t;
            _p   = o._p;
            _ses = o._ses;
            _hp  = nullptr;  // never inherit a slot
            _hv  = nullptr;
         }
         return *this;
      }
      snap(snap&& o) noexcept
          : _e(o._e), _t(o._t), _p(o._p), _ses(o._ses), _hp(o._hp), _hv(o._hv)
      {
         o._t  = bt::null;
         o._hp = nullptr;
         o._hv = nullptr;
      }
      snap& operator=(snap&& o) noexcept
      {
         if (this != &o)
         {
            reset();
            _e    = o._e;
            _t    = o._t;
            _p    = o._p;
            _ses  = o._ses;
            _hp   = o._hp;
            _hv   = o._hv;
            o._t  = bt::null;
            o._hp = nullptr;
            o._hv = nullptr;
         }
         return *this;
      }
      ~snap() { reset(); }

      bt       tag() const noexcept { return _t; }
      uint32_t ptr() const noexcept { return _p; }
      gcart::engine*           eng() const noexcept { return _e; }
      sal2::allocator_session* session() const noexcept { return _ses; }  // borrowed
      void     give() noexcept { _t = bt::null; }  // relinquish the view
      // Phase C: a snap NEVER frees a node. A reader-cursor snap owns its hazard slot
      // (_hp): reset CLEARS it (version slot first, root last — mirror of publish so a
      // writer's floor_/drain scan keying off the root word never sees a stale version).
      // An owned-snapshot view carries no slot here (the handle's owned_pin does).
      void reset() noexcept
      {
         if (_hp)
         {
            if (_hv)
               _hv->store(0, std::memory_order_seq_cst);
            _hp->store(null_root, std::memory_order_seq_cst);
            _hp = nullptr;
            _hv = nullptr;
         }
         _t = bt::null;
      }

     private:
      snap(gcart::engine* e, bt t, uint32_t p, sal2::allocator_session* ses,
           std::atomic<uint64_t>* hp = nullptr, std::atomic<uint64_t>* hv = nullptr)
          : _e(e), _t(t), _p(p), _ses(ses), _hp(hp), _hv(hv)
      {
      }
      gcart::engine*           _e   = nullptr;
      bt                       _t   = bt::null;
      uint32_t                 _p   = 0;
      sal2::allocator_session* _ses = nullptr;  // BORROWED holder session (not owning)
      std::atomic<uint64_t>*   _hp  = nullptr;  // reader-cursor HAZARD slot; reset clears it
      std::atomic<uint64_t>*   _hv  = nullptr;  // parallel VERSION slot (floor_ reads it)
   };

   class database;

   // ── cursor: a pinned snapshot + point get + ORDERED iteration (adb::kv_cursor)
   //    STATEFUL FRAME-STACK cursor (mirrors the hkv cursor): seek/lower_bound
   //    descend ONCE building a path stack; next()/prev() WALK the stack (advance
   //    the top frame, pop when exhausted, descend the new child) — never re-descend
   //    from root. The key is NOT built during iteration: it is materialized ON
   //    DEMAND in key() by replaying the frame prefixes into a reusable buffer.
   //    No std::string anywhere in the hot path. Each frame's `cur` is the slot we
   //    descended at this node: -1 = the node's TERM (sorts before all byte-children),
   //    else the ordered child slot (setlist/nested: key index; full: byte value).
   class cursor
   {
     public:
      cursor() = default;
      // The cursor does NOT hold a lifetime leaf read_lock — that would block the
      // SAL 1.0 leaf compactor from recycling segments for the cursor's whole life
      // (4 readers each pinning forever stalls compaction → garbage bloat → page
      // faults → reads collapse). Instead each leaf-touching operation takes a SHORT,
      // LOCAL read_lock (the reference engine's value_ref discipline): the lock spans
      // exactly the resolve + value read/copy, then drops so the compactor gets a
      // window. gcart leaves are immutable, so even under relocation the lock window
      // keeps the resolved bytes valid; outside it we re-resolve via the stored
      // (tag, ptr). Reads route through get_ref (resolve/leaf_value) → MFU-sampled.
      // The cursor CO-OWNS its leaf session (an allocator_session_ptr is refcounted):
      // it holds a ref to the holder's session so the session outlives the cursor even
      // if the read_session/snapshot that created it is destroyed first (the test
      // pattern `db->start_read_session()->create_cursor(0)` makes the read_session a
      // temporary). All of a reader thread's cursors drive that one session — no
      // global map, no lock. _ses caches the raw pointer for the hot read path.
      explicit cursor(snap s)
          : _s(std::move(s)),
            _lses(_s.session() ? sal2::allocator_session_ptr(_s.session(), /*retain=*/true)
                               : sal2::allocator_session_ptr(nullptr)),
            _ses(_lses.get())
      {
      }

      // point get (no positioning) — find_in is a direct descent, no key build.
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return std::nullopt;
         std::string_view out;  // snapshot pin (refcount) keeps the leaf alive; read = get_ref->value
         if (!find_in(*_s.eng(), ses_(), _s.tag(), _s.ptr(), k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return false;
         std::string_view out;  // zero-copy: hand the value view straight to f (snapshot-pinned)
         if (!find_in(*_s.eng(), ses_(), _s.tag(), _s.ptr(), k, out))
            return false;
         f(value_view(out.data(), out.size()));
         return true;
      }

      // ── ordered forward iteration ───────────────────────────────────────────
      bool seek_begin()
      {
         if (!root_ok_())
            return false;
         _n = 0;
         go_left_(_s.tag(), root_(), _s.ptr());
         _end  = false;
         _rend = false;
         _kv   = false;
         return true;
      }
      bool lower_bound(key_view k)
      {
         if (!root_ok_())
            return false;
         _n  = 0;
         _kv = false;
         if (seek_ge_(_s.tag(), root_(), _s.ptr(), k, 0))
         {
            _end  = false;
            _rend = false;
            return true;
         }
         _end = true;
         return false;
      }
      bool next()
      {
         if (_end)
            return false;
         if (_rend)
            return seek_begin();
         _kv = false;
         return advance_();
      }
      bool     is_end() const { return _end; }
      key_view key() const
      {
         if (_end || _rend)
            return {};
         if (!_kv)
            materialize_key_();
         return key_view(_kb, _klen);
      }
      template <class S = std::string>
      std::optional<S> value() const
      {
         if (_end || _rend)
            return std::nullopt;
         const sv   v  = cur_value_();  // snapshot-pinned; read = get_ref->value
         return S(v.data(), v.size());
      }
      template <class F>
      bool get_value(F&& f) const
      {
         if (_end || _rend)
            return false;
         f(cur_value_());  // snapshot-pinned; read = get_ref->value
         return true;
      }
      bool is_subtree() const { return false; }

      // ── ordered reverse iteration ───────────────────────────────────────────
      bool seek_last()
      {
         if (!root_ok_())
            return false;
         _n = 0;
         go_right_(_s.tag(), root_(), _s.ptr());
         _end  = false;
         _rend = false;
         _kv   = false;
         return true;
      }
      // Mathematical upper bound: the LARGEST key v <= k (the supremum of {v : v <= k};
      // the dual of lower_bound = smallest v >= k). k stays a VIEW — no copy.
      bool upper_bound(key_view k)
      {
         if (lower_bound(k))    // smallest v >= k
         {
            if (key() == k)     // k present → k is its own floor
               return true;
            return prev();      // landed strictly > k → predecessor is the largest v < k
         }
         return seek_last();    // no key >= k → every key < k → the last key is the floor
      }
      bool prev()
      {
         if (_rend)
            return false;
         if (_end)
            return seek_last();
         _kv = false;
         return retreat_();
      }
      bool is_rend() const { return _rend; }

     private:
      // ── geometry ────────────────────────────────────────────────────────────
      static constexpr int MAXF = 160;
      struct frame
      {
         bt          t;
         const void* node;
         int         cur;  // -1 = term; else ordered child slot (idx / byte)
      };

      bool        root_ok_() const { return _s.eng() && _s.tag() != bt::null; }
      const void*  root_() const { return resolve(*_s.eng(), ses_(), _s.tag(), _s.ptr()); }
      gcart::engine& eng_() const { return *_s.eng(); }
      // This cursor's leaf session: BORROWED from its snap (the read_session's owned
      // session), set at construction. A cursor is single-thread, so every leaf touch
      // (lock + get_ref) drives this one session — no map, no lock.
      sal2::allocator_session& ses_() const { return *_ses; }

      // EXTERNAL leaf: store the (tag, ptr) — NOT a resolved pointer. The body is
      // re-resolved (get_ref, MFU-sampled) under a SHORT lock at value/key time, so
      // the cursor holds no lock between positioning and the value read (the leaf
      // compactor stays unblocked). INLINE leaf bodies live in the FIX nested node
      // (address-stable), so those views are cached directly.
      void set_leaf_ext_(bt t, uint32_t p)
      {
         _inl = false;
         _lt  = t;
         _lp  = p;
      }
      void set_leaf_inl_(sv suf, sv val)
      {
         _inl  = true;
         _isuf = suf;
         _ival = val;
      }
      // value of the current position; caller must hold a leaf read_lock for the
      // EXTERNAL case (the returned view points into leaf memory).
      sv cur_value_() const { return _inl ? _ival : leaf_value(ses_(), _lt, _lp); }

      void push_(bt t, const void* node, int cur)
      {
         _f[_n].t    = t;
         _f[_n].node = node;
         _f[_n].cur  = cur;
         ++_n;
      }

      // Descend to the LEFTMOST leaf of subtree (t,node), pushing a frame per
      // routing node (cur = the slot taken: -1 term, else first child). `ptr` tracks
      // the ptr_address that produced `node` (only meaningful for an EXTERNAL leaf —
      // routing nodes are FIX-resolved into frames; an inline setlist has no ptr).
      // Stored as (_lt,_lp) so value/key re-resolve the leaf under a short lock.
      void go_left_(bt t, const void* node, uint32_t ptr)
      {
         for (;;)
         {
            if (is_leaf_tag(t))
            {
               set_leaf_ext_(t, ptr);
               return;
            }
            if (t == bt::full)
            {
               const full* n = static_cast<const full*>(node);
               if (n->has_term())
               {
                  push_(t, node, -1);
                  t    = bt(n->term_tag);
                  ptr  = n->term_ptr;
                  node = resolve(eng_(), ses_(),t, ptr);
                  continue;
               }
               int b = 0;
               while (b < 256 && n->btype(b) == bt::null)
                  ++b;
               push_(t, node, b);
               bt ct = n->btype(b);
               ptr   = n->branch(b);
               node  = resolve(eng_(), ses_(),ct, ptr);
               t     = ct;
               continue;
            }
            if (t == bt::nested_setlist)
            {
               const nested_setlist* n = static_cast<const nested_setlist*>(node);
               if (n->has_term())
               {
                  const uint32_t ti = n->nbranch;
                  push_(t, node, -1);
                  if (n->is_inline(ti))
                  {
                     set_leaf_inl_(sv{}, n->in_value(ti));
                     return;
                  }
                  bt tt = n->otype(ti);
                  ptr   = n->ext_ptr(ti);
                  node  = resolve(eng_(), ses_(),tt, ptr);
                  t     = tt;
                  continue;
               }
               push_(t, node, 0);
               if (n->is_inline(0))
               {
                  if (n->otype(0) == bt::suffix_value)
                  {
                     set_leaf_inl_(n->in_suffix(0), n->in_value(0));
                     return;
                  }
                  t    = bt::setlist;
                  node = n->in_setlist(0);  // inline routing body (FIX): no ptr
                  continue;
               }
               bt ct = n->otype(0);
               ptr   = n->ext_ptr(0);
               node  = resolve(eng_(), ses_(),ct, ptr);
               t     = ct;
               continue;
            }
            const setlist* n = static_cast<const setlist*>(node);
            if (n->has_term())
            {
               push_(t, node, -1);
               bt tt = n->term_type();
               ptr   = n->term_branch();
               node  = resolve(eng_(), ses_(),tt, ptr);
               t     = tt;
               continue;
            }
            push_(t, node, 0);
            bt ct = n->btype(0);
            ptr   = n->branch(0);
            node  = resolve(eng_(), ses_(),ct, ptr);
            t     = ct;
         }
      }

      // Descend to the RIGHTMOST leaf (largest key) — children last, term first
      // (so the rightmost is the last byte-child, or the term iff no children).
      void go_right_(bt t, const void* node, uint32_t ptr)
      {
         for (;;)
         {
            if (is_leaf_tag(t))
            {
               set_leaf_ext_(t, ptr);
               return;
            }
            if (t == bt::full)
            {
               const full* n = static_cast<const full*>(node);
               int         b  = 255;
               while (b >= 0 && n->btype(b) == bt::null)
                  --b;
               if (b < 0)  // term-only
               {
                  push_(t, node, -1);
                  t    = bt(n->term_tag);
                  ptr  = n->term_ptr;
                  node = resolve(eng_(), ses_(),t, ptr);
                  continue;
               }
               push_(t, node, b);
               bt ct = n->btype(b);
               ptr   = n->branch(b);
               node  = resolve(eng_(), ses_(),ct, ptr);
               t     = ct;
               continue;
            }
            if (t == bt::nested_setlist)
            {
               const nested_setlist* n = static_cast<const nested_setlist*>(node);
               if (n->nbranch == 0)  // term-only
               {
                  const uint32_t ti = n->nbranch;
                  push_(t, node, -1);
                  if (n->is_inline(ti))
                  {
                     set_leaf_inl_(sv{}, n->in_value(ti));
                     return;
                  }
                  bt tt = n->otype(ti);
                  ptr   = n->ext_ptr(ti);
                  node  = resolve(eng_(), ses_(),tt, ptr);
                  t     = tt;
                  continue;
               }
               const uint32_t last = n->nbranch - 1u;
               push_(t, node, int(last));
               if (n->is_inline(last))
               {
                  if (n->otype(last) == bt::suffix_value)
                  {
                     set_leaf_inl_(n->in_suffix(last), n->in_value(last));
                     return;
                  }
                  t    = bt::setlist;
                  node = n->in_setlist(last);  // inline routing body (FIX): no ptr
                  continue;
               }
               bt ct = n->otype(last);
               ptr   = n->ext_ptr(last);
               node  = resolve(eng_(), ses_(),ct, ptr);
               t     = ct;
               continue;
            }
            const setlist* n = static_cast<const setlist*>(node);
            if (n->nbranch == 0)  // term-only
            {
               push_(t, node, -1);
               bt tt = n->term_type();
               ptr   = n->term_branch();
               node  = resolve(eng_(), ses_(),tt, ptr);
               t     = tt;
               continue;
            }
            const uint32_t last = n->nbranch - 1u;
            push_(t, node, int(last));
            bt ct = n->btype(last);
            ptr   = n->branch(last);
            node  = resolve(eng_(), ses_(),ct, ptr);
            t     = ct;
         }
      }

      // Forward: advance to the NEXT leaf by walking the stack.
      bool advance_()
      {
         while (_n > 0)
         {
            frame& f = _f[_n - 1];
            if (step_fwd_(f))
               return true;
            --_n;  // node exhausted → pop and continue at the parent
         }
         _end = true;
         return false;
      }
      // Reverse: retreat to the PREVIOUS leaf.
      bool retreat_()
      {
         while (_n > 0)
         {
            frame& f = _f[_n - 1];
            if (step_bwd_(f))
               return true;
            --_n;
         }
         _rend = true;
         return false;
      }

      // Advance frame `f` to its next slot (after f.cur) and descend-left into it;
      // returns false iff this node has no slot after f.cur (caller pops).
      bool step_fwd_(frame& f)
      {
         if (f.t == bt::full)
         {
            const full* n = static_cast<const full*>(f.node);
            int         b = (f.cur < 0) ? 0 : f.cur + 1;  // term was -1 → first child is 0
            while (b < 256 && n->btype(b) == bt::null)
               ++b;
            if (b >= 256)
               return false;
            f.cur = b;
            bt ct = n->btype(b);
            go_left_(ct, resolve(eng_(), ses_(),ct, n->branch(b)), n->branch(b));
            return true;
         }
         if (f.t == bt::nested_setlist)
         {
            const nested_setlist* n = static_cast<const nested_setlist*>(f.node);
            const uint32_t        i = (f.cur < 0) ? 0u : uint32_t(f.cur) + 1u;
            if (i >= n->nbranch)
               return false;
            f.cur = int(i);
            if (n->is_inline(i))
            {
               if (n->otype(i) == bt::suffix_value)
               {
                  set_leaf_inl_(n->in_suffix(i), n->in_value(i));
                  return true;
               }
               go_left_(bt::setlist, n->in_setlist(i), 0);
               return true;
            }
            bt ct = n->otype(i);
            go_left_(ct, resolve(eng_(), ses_(),ct, n->ext_ptr(i)), n->ext_ptr(i));
            return true;
         }
         const setlist* n = static_cast<const setlist*>(f.node);
         const uint32_t i = (f.cur < 0) ? 0u : uint32_t(f.cur) + 1u;
         if (i >= n->nbranch)
            return false;
         f.cur = int(i);
         bt ct = n->btype(i);
         go_left_(ct, resolve(eng_(), ses_(),ct, n->branch(i)), n->branch(i));
         return true;
      }

      // Retreat frame `f` to its previous slot (before f.cur) and descend-right;
      // the term (slot -1) is the smallest, visited last on the way down.
      bool step_bwd_(frame& f)
      {
         if (f.t == bt::full)
         {
            const full* n = static_cast<const full*>(f.node);
            if (f.cur < 0)
               return false;  // already at term (smallest) → pop
            int b = f.cur - 1;
            while (b >= 0 && n->btype(b) == bt::null)
               --b;
            if (b < 0)
            {
               if (!n->has_term())
                  return false;
               f.cur = -1;
               bt tt = bt(n->term_tag);
               go_right_(tt, resolve(eng_(), ses_(),tt, n->term_ptr), n->term_ptr);
               return true;
            }
            f.cur = b;
            bt ct = n->btype(b);
            go_right_(ct, resolve(eng_(), ses_(),ct, n->branch(b)), n->branch(b));
            return true;
         }
         if (f.t == bt::nested_setlist)
         {
            const nested_setlist* n = static_cast<const nested_setlist*>(f.node);
            if (f.cur < 0)
               return false;
            if (f.cur == 0)
            {
               if (!n->has_term())
                  return false;
               const uint32_t ti = n->nbranch;
               f.cur             = -1;
               if (n->is_inline(ti))
               {
                  set_leaf_inl_(sv{}, n->in_value(ti));
                  return true;
               }
               bt tt = n->otype(ti);
               go_right_(tt, resolve(eng_(), ses_(),tt, n->ext_ptr(ti)), n->ext_ptr(ti));
               return true;
            }
            const uint32_t i = uint32_t(f.cur) - 1u;
            f.cur            = int(i);
            if (n->is_inline(i))
            {
               if (n->otype(i) == bt::suffix_value)
               {
                  set_leaf_inl_(n->in_suffix(i), n->in_value(i));
                  return true;
               }
               go_right_(bt::setlist, n->in_setlist(i), 0);
               return true;
            }
            bt ct = n->otype(i);
            go_right_(ct, resolve(eng_(), ses_(),ct, n->ext_ptr(i)), n->ext_ptr(i));
            return true;
         }
         const setlist* n = static_cast<const setlist*>(f.node);
         if (f.cur < 0)
            return false;
         if (f.cur == 0)
         {
            if (!n->has_term())
               return false;
            f.cur = -1;
            bt tt = n->term_type();
            go_right_(tt, resolve(eng_(), ses_(),tt, n->term_branch()), n->term_branch());
            return true;
         }
         const uint32_t i = uint32_t(f.cur) - 1u;
         f.cur            = int(i);
         bt ct            = n->btype(i);
         go_right_(ct, resolve(eng_(), ses_(),ct, n->branch(i)), n->branch(i));
         return true;
      }

      // Directed descent: position at the smallest key >= target, pushing frames.
      // Returns true if such a key exists in this subtree. On a "diverge / target is
      // a prefix" the whole subtree is > target → its leftmost is the answer.
      bool seek_ge_(bt t, const void* node, uint32_t ptr, sv target, uint32_t depth)
      {
         for (;;)
         {
            if (is_leaf_tag(t))
            {
               // compare full key (acc-bytes-above already matched target[0:depth])
               // against the remaining target via the leaf suffix.
               sv suf = leaf_suffix_body(t, node);
               if (!suffix_ge_(suf, target, depth))
                  return false;
               set_leaf_ext_(t, ptr);
               return true;
            }
            // full: prefetch the offset CL for the speculative routing byte
            // (target[depth]) before node_prefix_ reads the header (overlap, §7).
            if (t == bt::full && depth < target.size())
               prefetch(static_cast<const char*>(node) + full::branches_off +
                        (uint32_t(uint8_t(target[depth])) << 2));
            const sv P = node_prefix_(t, node);
            const uint32_t rem = uint32_t(target.size()) - depth;
            const uint32_t pl  = uint32_t(P.size());
            const uint32_t cl  = pl < rem ? pl : rem;
            uint32_t       i   = 0;
            while (i < cl && uint8_t(P[i]) == uint8_t(target[depth + i]))
               ++i;
            if (i < cl)  // prefix diverged in the compared region
            {
               if (uint8_t(P[i]) > uint8_t(target[depth + i]))
               {
                  go_left_(t, node, ptr);
                  return true;
               }
               return false;  // subtree entirely < target
            }
            if (pl >= rem)  // target is a (proper or full) prefix of P → subtree >= target
            {
               go_left_(t, node, ptr);
               return true;
            }
            depth += pl;  // P matched; descend at target[depth]
            const uint8_t tb = uint8_t(target[depth]);
            if (seek_ge_descend_(t, node, target, depth, tb))
               return true;
            return false;
         }
      }

      // At routing node (t,node) whose prefix matched, descend toward byte tb at
      // target[depth]; the matching child is tried for an exact >= continuation, and
      // if it yields nothing, the first child byte > tb gives the subtree's leftmost.
      bool seek_ge_descend_(bt t, const void* node, sv target, uint32_t depth, uint8_t tb)
      {
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(node);
            bt          ct = n->btype(tb);
            if (ct != bt::null)
            {
               push_(t, node, tb);
               if (seek_ge_(ct, resolve(eng_(), ses_(),ct, n->branch(tb)), n->branch(tb), target, depth + 1))
                  return true;
               --_n;  // matching child had nothing >= target; fall through to next byte
            }
            for (int b = tb + 1; b < 256; ++b)
               if (n->btype(b) != bt::null)
               {
                  push_(t, node, b);
                  bt c2 = n->btype(b);
                  go_left_(c2, resolve(eng_(), ses_(),c2, n->branch(b)), n->branch(b));
                  return true;
               }
            return false;
         }
         if (t == bt::nested_setlist)
         {
            const nested_setlist* n   = static_cast<const nested_setlist*>(node);
            int                   idx = int(ucc::lower_bound_padded(n->okeys(), n->nbranch, tb));
            uint32_t              start = uint32_t(idx);
            if (idx < int(n->nbranch) && n->okeys()[idx] == tb)
            {
               const uint32_t ui = uint32_t(idx);
               push_(t, node, int(ui));
               if (n->is_inline(ui))
               {
                  if (n->otype(ui) == bt::suffix_value)
                  {
                     if (suffix_ge_(n->in_suffix(ui), target, depth + 1))
                     {
                        set_leaf_inl_(n->in_suffix(ui), n->in_value(ui));
                        return true;
                     }
                  }
                  else if (seek_ge_(bt::setlist, n->in_setlist(ui), 0, target, depth + 1))
                     return true;
               }
               else
               {
                  bt ct = n->otype(ui);
                  if (seek_ge_(ct, resolve(eng_(), ses_(),ct, n->ext_ptr(ui)), n->ext_ptr(ui), target, depth + 1))
                     return true;
               }
               --_n;
               start = ui + 1;
            }
            if (start < n->nbranch)
            {
               push_(t, node, int(start));
               if (n->is_inline(start))
               {
                  if (n->otype(start) == bt::suffix_value)
                  {
                     set_leaf_inl_(n->in_suffix(start), n->in_value(start));
                     return true;
                  }
                  go_left_(bt::setlist, n->in_setlist(start), 0);
                  return true;
               }
               bt ct = n->otype(start);
               go_left_(ct, resolve(eng_(), ses_(),ct, n->ext_ptr(start)), n->ext_ptr(start));
               return true;
            }
            return false;
         }
         const setlist* n     = static_cast<const setlist*>(node);
         int            idx   = int(ucc::lower_bound_padded(n->keys(), n->nbranch, tb));
         uint32_t       start = uint32_t(idx);
         if (idx < int(n->nbranch) && n->keys()[idx] == tb)
         {
            const uint32_t ui = uint32_t(idx);
            push_(t, node, int(ui));
            bt ct = n->btype(ui);
            if (seek_ge_(ct, resolve(eng_(), ses_(),ct, n->branch(ui)), n->branch(ui), target, depth + 1))
               return true;
            --_n;
            start = ui + 1;
         }
         if (start < n->nbranch)
         {
            push_(t, node, int(start));
            bt ct = n->btype(start);
            go_left_(ct, resolve(eng_(), ses_(),ct, n->branch(start)), n->branch(start));
            return true;
         }
         return false;
      }

      // Is the leaf whose suffix is `suf` (key = matched-bytes-above ++ suf) >= target?
      static bool suffix_ge_(sv suf, sv target, uint32_t depth)
      {
         const sv tail = target.substr(depth <= target.size() ? depth : target.size());
         return suf >= tail;
      }
      static sv node_prefix_(bt t, const void* node)
      {
         return (t == bt::full)             ? static_cast<const full*>(node)->prefix()
                : (t == bt::nested_setlist) ? static_cast<const nested_setlist*>(node)->prefix()
                                            : static_cast<const setlist*>(node)->prefix();
      }

      // Replay the frame stack into the reusable key buffer (NO alloc, NO std::string).
      void materialize_key_() const
      {
         uint32_t len = 0;
         auto     app = [&](const char* p, uint32_t k)
         {
            if (len + k > sizeof(_kb))
               k = uint32_t(sizeof(_kb)) - len;  // adb caps keys; never overflow
            std::memcpy(_kb + len, p, k);
            len += k;
         };
         for (int i = 0; i < _n; ++i)
         {
            const frame& f = _f[i];
            const sv     P = node_prefix_(f.t, f.node);
            app(P.data(), uint32_t(P.size()));
            if (f.cur < 0)
               continue;  // term: no byte
            if (f.t == bt::full)
               _kb[len++] = char(uint8_t(f.cur));
            else if (f.t == bt::nested_setlist)
               _kb[len++] = char(static_cast<const nested_setlist*>(f.node)->okeys()[f.cur]);
            else
               _kb[len++] = char(static_cast<const setlist*>(f.node)->keys()[f.cur]);
         }
         if (_inl)
            app(_isuf.data(), uint32_t(_isuf.size()));
         else
         {
            // EXTERNAL leaf suffix: re-resolve via get_ref (MFU-sampled), copy into the key
            // buffer. Snapshot pin (refcount) keeps the leaf alive — no per-read lock.
            const sv suf = leaf_suffix(ses_(), _lt, _lp);
            app(suf.data(), uint32_t(suf.size()));
         }
         _klen = len;
         _kv   = true;
      }

      snap        _s;
      // CO-OWNED leaf session (refcounted): keeps the holder's session alive for the
      // cursor's whole lifetime. _ses is the cached raw pointer (hot path). Declared
      // before _ses so the ctor's _ses(_lses.get()) init runs after _lses.
      sal2::allocator_session_ptr _lses{nullptr};
      sal2::allocator_session*    _ses = nullptr;
      frame       _f[MAXF];
      int         _n  = 0;       // frames in use
      bt          _lt = bt::null;
      uint32_t    _lp = 0;        // current EXTERNAL leaf ptr (re-resolved under a short lock)
      bool        _inl = false;
      sv          _isuf, _ival;   // current INLINE leaf (nested suffix_value)
      bool        _end = true, _rend = true;
      mutable char     _kb[2048];  // reused key buffer (adb keys are bounded)
      mutable uint32_t _klen = 0;
      mutable bool     _kv   = false;  // key materialized for the current position?
   };

   // ── snapshot: an owned, FROZEN root (adb kv_tree + copyable shared_snapshot).
   //    get_root(i) returns one; it keeps its version alive via the pin, and
   //    snapshot_cursor() walks it. Named `snapshot` (not `tree`) — `tree` is the
   //    mutable engine work-tree used by transactions. ───────────────────────────
   // Phase C: an owned snapshot's version pin (claims a hazard slot, publishes
   // (version, root), releases on the LAST handle copy). Defined after database
   // (it needs claim/release/publish). The snapshot holds a shared_ptr to it — copies
   // share ONE published slot (a per-HANDLE count, NOT a per-node refcount).
   struct owned_pin;

   class snapshot
   {
     public:
      snapshot() = default;
      explicit snapshot(snap s) : _s(std::move(s)) {}
      snapshot(snap s, std::shared_ptr<owned_pin> pin) : _s(std::move(s)), _pin(std::move(pin)) {}
      cursor   snapshot_cursor() const { return cursor(_s); }  // copies the (non-owning) read view
      explicit operator bool() const noexcept { return _s.eng() != nullptr; }

      // point read of the frozen snapshot (used by the snapshot-isolation battery)
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return std::nullopt;
         std::string_view out;  // version pin keeps the spine alive; read = get_ref->value
         // The snapshot carries the engine ADMIN session (cold main-thread path);
         // snapshot::get is the snapshot-isolation battery's reader (single thread).
         if (!find_in(*_s.eng(), *_s.session(), _s.tag(), _s.ptr(), k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      // root-exchange primitives (set_root): the snapshot IS the owned tree.
      bt       tag() const noexcept { return _s.tag(); }
      uint32_t ptr() const noexcept { return _s.ptr(); }
      void     give() noexcept { _s.give(); _pin.reset(); }  // relinquish (moved into a work tree)
      static snapshot adopt(gcart::engine& e, bt t, uint32_t p, sal2::allocator_session* ses)
      {
         return snapshot(snap::adopt(e, t, p, ses));  // unowned view (no version pin)
      }

     private:
      snap                       _s;
      std::shared_ptr<owned_pin> _pin;  // version-pin lifetime (per-handle; null for adopt/empty)
   };

   class transaction;

   // ── root_handle: a view onto the transaction's root work tree. primary()
   //    returns it; set_root is the owned-tree exchange (adb::root_handle). ────────
   class root_handle
   {
     public:
      root_handle() = default;
      root_handle(gcart::engine* e, tree* w) : _e(e), _w(w) {}
      // (node engine, not the kv::engine bundle)
      void insert(key_view k, value_view v) { _w->upsert(k, v); }
      void upsert(key_view k, value_view v) { _w->upsert(k, v); }
      int  remove(key_view k) { return _w->remove(k); }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         sv   v;  // work root pin keeps the leaf alive; read = get_ref->value
         if (!_w->find(k, v))
            return std::nullopt;
         return S(v.data(), v.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         sv   v;  // zero-copy: hand the value view straight to f (work-root-pinned)
         if (!_w->find(k, v))
            return false;
         f(value_view(v.data(), v.size()));
         return true;
      }

      // EXCHANGE: adopt `incoming`'s root into this work tree (transferring its ref),
      // and return the displaced root as an owned snapshot. set_root({}) clears this
      // root and hands back its old contents; the rotation composes from this.
      snapshot set_root(snapshot incoming)
      {
         const bt       dt = _w->root_tag();
         const uint32_t dp = _w->root_ptr();            // displaced
         _w->set_root(incoming.tag(), incoming.ptr());  // adopt incoming's root
         incoming.give();                               // its ref moved into _w
         // The displaced snapshot is owned by the writer; carry the work tree's
         // session so its eventual release routes through the writer session.
         return snapshot::adopt(*_e, dt, dp, _w->leaf_session());  // own the displaced
      }

     private:
      gcart::engine* _e = nullptr;
      tree*   _w = nullptr;
   };

   // ── transaction: COW over ONE root (single-root). gcart is used single-root
   //    everywhere — the psitrimdbx shim (start_transaction(slot)) and delta_art's
   //    cold base B (start_transaction(0)); delta_art requires only W to be
   //    multi-root, and gcart is always B. No extra roots ⟹ no heap. ─────────────
   class transaction
   {
     public:
      transaction(transaction&& o) noexcept
          : _db(o._db),
            _t(o._t),
            _slot(o._slot),
            _lock(std::move(o._lock)),
            _start_root(o._start_root),
            _ver(o._ver),
            _done(o._done)
      {
         o._db   = nullptr;
         o._done = true;  // moved-from: nothing to roll back / no lock to drop
      }
      transaction(const transaction&)            = delete;
      transaction& operator=(const transaction&) = delete;
      ~transaction();

      // the single root this transaction COWs (adb::kv_transaction).
      // (body out-of-line — it touches database, defined below.)
      root_handle primary();

      void insert(key_view k, value_view v) { _t->upsert(k, v); }
      void upsert(key_view k, value_view v) { _t->upsert(k, v); }
      int  remove(key_view k) { return _t->remove(k); }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         sv   v;  // txn work root pin keeps the leaf alive; read = get_ref->value
         if (!_t->find(k, v))
            return std::nullopt;
         return S(v.data(), v.size());
      }
      // lambda form (adb::kv_transaction): invoke f(value) on a hit, return found.
      template <class F>
      bool get(key_view k, F&& f) const
      {
         sv   v;  // zero-copy: hand the value view straight to f (txn-work-root-pinned)
         if (!_t->find(k, v))
            return false;
         f(value_view(v.data(), v.size()));
         return true;
      }
      void commit();
      void abort() noexcept;

     private:
      friend class write_session;
      transaction(database* db, tree* t, uint32_t slot);
      database*                    _db   = nullptr;
      tree*                        _t    = nullptr;
      uint32_t                     _slot = 0;
      // Part B base pin: the per-root WRITER LOCK (held for the txn lifetime) IS the
      // base pin — NO refcount retain on the committed base. start_root = the live
      // root at txn start; the work tree COWs from it (Part A's birth==V latch keeps
      // committed nodes immutable, so they clone). commit() retires start_root.
      std::unique_lock<std::mutex> _lock;
      uint64_t                     _start_root = null_root;  // committed root at txn start
      uint32_t                     _ver  = 0;  // this txn's commit version (birth of new nodes)
      bool                         _done = false;
   };

   class write_session
   {
     public:
      explicit write_session(database* db) : _db(db) {}
      transaction start_transaction(uint32_t idx = 0);
      snapshot    get_root(uint32_t idx = 0);  // owned, FROZEN snapshot of root idx
     private:
      database* _db;
   };

   class read_session
   {
     public:
      // Each read_session OWNS one leaf session (created here), used only by THIS
      // reader thread's cursors — the per-reader session that makes concurrent reads
      // contention-free (no global session mutex). Body out-of-line (needs database).
      // It ALSO claims one HAZARD SLOT for its lifetime (Part B): create_cursor
      // publishes the live root there with no refcount touch; the dtor frees the slot.
      explicit read_session(database* db);
      ~read_session();
      cursor create_cursor(uint32_t idx);

     private:
      database*                   _db;
      sal2::allocator_session_ptr _lses;       // this reader's owned leaf session
      uint32_t                    _hz_slot;     // claimed hazard slot index (Part B)
   };

   // ── database ───────────────────────────────────────────────────────────────
   class database
   {
     public:
      static std::shared_ptr<database> create(std::filesystem::path dir)
      {
         return std::shared_ptr<database>(new database(std::move(dir)));
      }
      static std::shared_ptr<database> open(std::filesystem::path dir)
      {
         return create(std::move(dir));
      }
      // Unified open semantics (adb::openable_database — the embedder contract the
      // MDBX shim / delta_art tier binds to). The spec engine's segment files live
      // under `dir` (cl1.dat is the first); mode gates create-vs-open.
      static std::shared_ptr<database> open(std::filesystem::path dir, sal::open_mode mode)
      {
         const bool exists = std::filesystem::exists(dir / "cl1.dat");
         switch (mode)
         {
            case sal::open_mode::create_only:
               if (exists)
                  throw std::runtime_error("database already exists: " + dir.generic_string());
               break;
            case sal::open_mode::open_existing:
               if (!exists)
                  throw std::runtime_error("database does not exist: " + dir.generic_string());
               break;
            case sal::open_mode::read_only:
               throw std::runtime_error("gcart: read-only opens not implemented yet");
            case sal::open_mode::create_or_open:
               break;
         }
         return create(std::move(dir));
      }
      // 3-arg open (psitri::dwal::kv_engine contract — delta_art cold-base tier). cfg
      // carries the commit-sync mode; gcart records it (see config) and otherwise
      // opens identically (mode gates create-vs-open).
      static std::shared_ptr<database> open(std::filesystem::path dir, sal::open_mode mode,
                                            const config&)
      {
         return open(std::move(dir), mode);
      }

      std::shared_ptr<write_session> start_write_session()
      {
         return std::make_shared<write_session>(this);
      }
      std::shared_ptr<read_session> start_read_session()
      {
         return std::make_shared<read_session>(this);
      }
      void sync(uint32_t = 0) noexcept {}

      gcart::engine&         eng() noexcept { return _e; }
      // The published roots live in an mmap'd file (SAL's "roots" pattern): a
      // commit's store into this slot is durable via the page cache, so a
      // clean close+reopen restores every root for free — the engine's segment
      // files reopen with the same base-relative slot indices, so a restored
      // (tag, ptr) root resolves correctly in the new run.
      std::atomic<uint64_t>& root_slot(uint32_t i) noexcept
      {
         return _roots_map.as<std::atomic<uint64_t>>()[i];
      }
      tree& work_tree(uint32_t i)
      {
         if (!_work[i])
            _work[i] = std::make_unique<tree>(_e);
         return *_work[i];
      }
      // Phase C owned snapshot (get_root): a version pin. Out-of-line (needs owned_pin).
      snapshot make_owned_snapshot(uint32_t idx);
      // Monotonic commit-version counter (design §GC-1 ".meta counter"), persisted in
      // roots.dat just past the root slots so births never collide across a reopen.
      // Each transaction takes a fresh version = the birth stamped on its new nodes;
      // committed nodes read birth<V on the next txn → the COW latch clones them.
      uint32_t next_write_version() noexcept
      {
         return uint32_t(_roots_map.as<std::atomic<uint64_t>>()[ver_index].fetch_add(
                    1, std::memory_order_relaxed)) +
                1;
      }
      // settled once the floor-gated drain has caught up AND the engine has too.
      uint64_t live_bytes() noexcept { return _e.live_bytes(); }
      // Measurement barrier: flush the floor-gated DRAIN FIRST (it frees every committed
      // supersession-log entry), then settle the engine. NB: this completes only when the
      // floor has advanced past every batch's death version — i.e. when no reader/snapshot
      // pins an old version. Tests call drain()/live_bytes() with readers joined (floor =
      // committed), so every batch is eligible and the drain finishes.
      void     drain() noexcept
      {
         flush_reaper();
         _e.drain();
      }
      // Wait until every enqueued supersession batch has been drained (start_ver
      // flush_reaper): queue-empty alone races the free cascade running outside the lock.
      void flush_reaper() noexcept
      {
         while (_reap_done.load(std::memory_order_acquire) !=
                _reap_enq.load(std::memory_order_acquire))
         {
            _dead_cv.notify_one();
            std::this_thread::yield();
         }
      }
      ~database()
      {
         // Stop the drain (force-drain any remaining batches: no readers at teardown),
         // then sync the durable root table. The drain frees the displaced supersession
         // logs — without this they'd leak (live != 0 on the next open's census).
         {
            std::lock_guard<std::mutex> g(_dead_mx);
            _reaper_stop = true;
         }
         _dead_cv.notify_all();
         if (_reaper.joinable())
            _reaper.join();
         _roots_map.sync(sal::sync_type::full);  // clean close
      }

      // ── hazard-slot registry (Part B reader-cursor pin) ────────────────────────
      // A fixed array of reader hazard slots (start_ref's _reader_roots). Each holds a
      // PUBLISHED packed root word, or null_root when free/cleared. A reader claims a
      // slot for its lifetime; its cursors publish the live root into it (no refcount).
      static constexpr uint32_t kHazardSlots = 256;
      std::atomic<uint64_t>&    hazard_slot(uint32_t i) noexcept { return _hazard[i]; }
      // Parallel reader-pin slots (Phase A item 2): the version the cursor pinned (floor_
      // reads it) and the root index the pin is attributed to (per-root floor).
      std::atomic<uint64_t>&    hz_version(uint32_t i) noexcept { return _hz_version[i]; }
      std::atomic<uint32_t>&    hz_root_idx(uint32_t i) noexcept { return _hz_root_idx[i]; }
      // Claim a free hazard slot (lock-free word-bitmap), returns its index. Cleared
      // to null_root before publication so a stale word can't pin a stale root.
      uint32_t claim_hazard_slot()
      {
         static constexpr uint32_t W = kHazardSlots / 64;
         for (uint32_t w = 0; w < W; ++w)
         {
            uint64_t cur = _hz_used[w].load(std::memory_order_relaxed);
            while (cur != ~0ull)
            {
               const uint32_t b   = uint32_t(std::countr_one(cur));
               const uint64_t bit = uint64_t(1) << b;
               if (_hz_used[w].compare_exchange_weak(cur, cur | bit, std::memory_order_acquire))
               {
                  const uint32_t idx = w * 64 + b;
                  _hazard[idx].store(null_root, std::memory_order_seq_cst);
                  return idx;
               }
            }
         }
         throw std::runtime_error("gcart::kv: all hazard reader slots in use");
      }
      void release_hazard_slot(uint32_t idx) noexcept
      {
         _hz_version[idx].store(0, std::memory_order_seq_cst);          // drop the version pin
         _hz_root_idx[idx].store(UINT32_MAX, std::memory_order_relaxed);
         _hazard[idx].store(null_root, std::memory_order_seq_cst);      // unpublish (last)
         _hz_used[idx / 64].fetch_and(~(uint64_t(1) << (idx % 64)), std::memory_order_release);
      }
      // Per-root writer lock (start_ref root_state::lock): the writer holds it for the
      // txn lifetime = the base pin. Single-writer per root (the bank has one).
      std::mutex& root_mx(uint32_t i) noexcept { return _root_mx[i]; }

      // ── version-in-root (Phase A item 1, start_ver _live_arr intent) ─────────────
      // start_ver packs (version<<32 | root32) into ONE atomic per slot because its
      // root is a 32-bit line index. gcart's published root is a 40-bit packed
      // (tag<<32 | ptr) word — no room for a full version in the same u64. So we mirror
      // start_ver's INTENT (the commit version travels WITH the published root) via a
      // PARALLEL per-slot atomic _root_version[slot]: root_slot(slot) keeps pack(tag,ptr)
      // (unchanged for acquire/publish/COW/root_hazarded_), and the commit stores the
      // committing version here ALONGSIDE the root store. A reader recovers BOTH by
      // loading root_slot(slot) AND root_version(slot). This is task option (b), the
      // lower-risk choice (the packed-root format and the existing reclamation paths are
      // byte-unchanged). _committed tracks the latest committed version (floor_ default).
      std::atomic<uint64_t>& root_version(uint32_t i) noexcept { return _root_version[i]; }
      uint64_t committed() const noexcept { return _committed.load(std::memory_order_acquire); }

      // floor = oldest version pinned by ANY reader hazard slot (committed when nothing
      // is pinned). VERBATIM port of start_ver::floor_for_ (kv.hpp ~:887): start at
      // _committed, then min over every published reader slot's pinned version. A pinned
      // version is read from the parallel _hz_version[i] (start_ver reads it as s>>32 of
      // its single combined slot; gcart's slot already carries the packed root, so the
      // version rides the parallel per-slot atomic — same intent, separate word). A slot
      // is "published" iff _hazard[i] != null_root (start_ver's `if (!s) continue`).
      // `root` is the root SLOT INDEX to gate (UINT32_MAX = conservative global floor):
      // a pin on a DIFFERENT specific root does not gate this root.
      uint32_t floor_for_(uint32_t root) noexcept
      {
         uint32_t f = uint32_t(_committed.load(std::memory_order_acquire));
         for (uint32_t i = 0; i < kHazardSlots; ++i)
         {
            const uint64_t s = _hazard[i].load(std::memory_order_seq_cst);
            if (s == null_root)
               continue;  // slot not published
            const uint32_t pidx = _hz_root_idx[i].load(std::memory_order_relaxed);
            if (root != UINT32_MAX && pidx != root && pidx != UINT32_MAX)
               continue;  // pin on a different specific root: doesn't gate `root`
            const uint32_t v = uint32_t(_hz_version[i].load(std::memory_order_seq_cst));
            if (v && v < f)  // pin at V sees state V; deaths <= V stay eligible
               f = v;
         }
         return f;
      }
      uint32_t floor_() noexcept { return floor_for_(UINT32_MAX); }  // global

      // Phase C: publish a new committed root + hand the txn's SUPERSESSION LOG to the
      // floor-gated drain. Under root_mx, store the new root + commit version to the
      // durable+live slots (the seq_cst store + the drain's seq_cst hazard scan close
      // the reader race), advance _committed, then enqueue the log as ONE batch tagged
      // (ver, slot). The drain frees each entry (SHALLOW = node line; DEEP = cascade)
      // only once floor_for_(slot) >= ver — no reader pinned at version < ver can still
      // reach any displaced node. NO old-root deep-reap, NO node refcount: the displaced
      // spine IS the log (every COW clone logged its old node).
      void publish_and_drain(uint32_t slot, uint64_t new_root, uint32_t ver,
                             std::vector<tree::dead_entry>& log) noexcept
      {
         root_version(slot).store(ver, std::memory_order_seq_cst);    // version travels w/ root
         root_slot(slot).store(new_root, std::memory_order_seq_cst);  // durable + live
         _committed.store(ver, std::memory_order_release);            // latest committed version
         enqueue_batch_(ver, slot, log);
      }

      // ── floor-gated reaper (Phase B; mirrors start_ver reap_root_/reaper_loop_) ──
      // Enqueue one committed txn's SUPERSESSION LOG as a drain batch tagged
      // (death=ver, slot). The drain frees each entry once floor_for_(slot) >= ver — a
      // reader pinned at version < ver can still reach it. An empty log (a no-op commit:
      // republish-only) enqueues nothing. MOVES the log in (the writer clears its own).
      void enqueue_batch_(uint32_t ver, uint32_t slot,
                          std::vector<tree::dead_entry>& log) noexcept
      {
         if (log.empty())
            return;
         {
            std::lock_guard<std::mutex> g(_dead_mx);
            _dead.push_back({ver, slot, std::move(log)});
         }
         _reap_enq.fetch_add(1, std::memory_order_release);
         _dead_cv.notify_one();
      }

     private:
      static constexpr uint32_t max_roots   = 64;
      static constexpr uint32_t ver_index   = max_roots;  // commit-version slot (past the roots)
      static constexpr size_t   roots_bytes = max_roots * sizeof(uint64_t);

      // Free one supersession-log entry on the drain's OWN leaf session (Phase C: NO
      // refcount). SHALLOW = free the node's slot line only (its children are carried by
      // the clone that superseded it). DEEP = cascade-free the dropped subtree.
      void free_entry_(sal2::allocator_session& ses, const tree::dead_entry& e) noexcept
      {
         if (e.deep)
            gcart::free_subtree_deep_(_e, ses, e.tag, e.ptr);
         else
            gcart::free_node_shallow_(_e, ses, e.tag, e.ptr);
      }

      // Phase C floor-gated DRAIN loop (start_ver reaper_loop_ ~:813, generalized to
      // per-txn supersession batches). A batch's entries are freed ONLY once
      // floor_for_(its root) >= its death version — a reader pinned at version < death
      // can still reach them; else DEFER and retry as the floor advances. Each entry is
      // a SHALLOW (node-line) or DEEP (cascade) version-liveness free — NO refcount.
      // Runs on a dedicated thread so the frees are off the writer's critical path.
      void drain_loop_() noexcept
      {
         // The drain OWNS its leaf session (start_ver reaper: engine_t v{...}). Frees
         // SYNCHRONOUSLY on THIS thread's session — a SAL session is single-thread-owned.
         sal2::allocator_session_ptr rses = _e.leaf_create_session();
         std::vector<dead_batch>      deferred;
         std::unique_lock<std::mutex> lk(_dead_mx);
         for (;;)
         {
            if (_dead.empty() && !_reaper_stop)
            {
               if (deferred.empty())
                  _dead_cv.wait(lk, [&] { return _reaper_stop || !_dead.empty(); });
               else
                  _dead_cv.wait_for(lk, std::chrono::milliseconds(1));  // retry gated
            }
            if (_reaper_stop && _dead.empty() && deferred.empty())
               return;
            std::vector<dead_batch> work;
            work.swap(_dead);
            lk.unlock();
            for (auto& d : deferred)
               work.push_back(std::move(d));
            deferred.clear();
            for (auto& d : work)
            {
               // shutdown: no readers remain → force-drain (the start_ver pre-gate fallback).
               const uint32_t fl = _reaper_stop ? UINT32_MAX : floor_for_(d.slot);
               if (d.death <= fl)  // floor passed: no reader pinned at version < death
               {
                  for (const auto& e : d.entries)
                     free_entry_(*rses, e);
                  _reap_done.fetch_add(1, std::memory_order_release);
               }
               else
                  deferred.push_back(std::move(d));  // a reader still pins a version that sees it
            }
            lk.lock();
         }
      }

      explicit database(std::filesystem::path dir)
          : _e(dir), _roots_map(dir / "roots.dat", sal::access_mode::read_write)
      {
         for (auto& h : _hazard)
            h.store(null_root, std::memory_order_relaxed);
         for (auto& u : _hz_used)
            u.store(0, std::memory_order_relaxed);
         for (auto& v : _hz_version)
            v.store(0, std::memory_order_relaxed);
         for (auto& ri : _hz_root_idx)
            ri.store(UINT32_MAX, std::memory_order_relaxed);
         for (auto& rv : _root_version)
            rv.store(0, std::memory_order_relaxed);
         if (_roots_map.size() < roots_bytes)  // fresh file → init all slots to null_root
         {                                      // (null_root = bt::null<<32 is NON-zero, so a
            _roots_map.resize(4096);            //  zero-filled mmap would NOT be null)
            auto* r = _roots_map.as<std::atomic<uint64_t>>();
            for (uint32_t i = 0; i < max_roots; ++i)
               r[i].store(null_root, std::memory_order_relaxed);
            r[ver_index].store(1, std::memory_order_relaxed);  // first txn gets version 2
         }
         // Rebuild _committed from the durable version counter (start_ver resumes
         // _committed from get_root(kMaxRoots)): the last-allocated version is the
         // committed boundary. Re-expose each published durable root at that version so
         // a reader pinning before any new commit floors at the right point.
         auto* r = _roots_map.as<std::atomic<uint64_t>>();
         const uint64_t cv = r[ver_index].load(std::memory_order_relaxed);
         _committed.store(cv, std::memory_order_relaxed);
         for (uint32_t i = 0; i < max_roots; ++i)
            if (rtag(r[i].load(std::memory_order_relaxed)) != bt::null)
               _root_version[i].store(cv, std::memory_order_relaxed);
         // Start the floor-gated DRAIN LAST — after every member is initialized (it reads
         // _committed/_hazard/_hz_version via floor_for_ and frees on its own session).
         _reaper = std::thread([this] { drain_loop_(); });
      }

      gcart::engine                         _e;
      sal::mapping                          _roots_map;
      std::array<std::unique_ptr<tree>, 64> _work{};
      // Version-in-root (Phase A item 1): parallel per-root committed version, written
      // by publish_and_drain alongside root_slot. Mirrors start_ver's _live_arr version
      // half (which it packs into the same word; gcart's packed root needs a parallel
      // word). _committed = latest committed version (start_ver db->_committed), the
      // floor_ default. NOT persisted in roots.dat yet (the durable counter at ver_index
      // resumes births on reopen; this in-memory _committed is rebuilt from it at open).
      std::array<std::atomic<uint64_t>, max_roots> _root_version{};
      std::atomic<uint64_t>                 _committed{0};
      // Hazard-slot registry: published reader/snapshot roots + a word-bitmap claim.
      // Separate cachelines from the durable root slots so a pin's publish never false-
      // shares the writer's live-root store. _hz_version = the version the pin holds
      // (floor_ reads it); _hz_root_idx = which root slot the pin is attributed to
      // (start_ver reader_root_idx, per-root floor). Both parallel to _hazard.
      std::array<std::atomic<uint64_t>, kHazardSlots> _hazard{};
      std::array<std::atomic<uint64_t>, kHazardSlots> _hz_version{};
      std::array<std::atomic<uint32_t>, kHazardSlots> _hz_root_idx{};
      std::array<std::atomic<uint64_t>, kHazardSlots / 64> _hz_used{};
      std::array<std::mutex, max_roots>     _root_mx{};   // per-root writer lock
      // ── Phase C floor-gated DRAIN ─────────────────────────────────────────────────
      // One committed txn's SUPERSESSION LOG (the nodes it displaced) + its death version
      // + the root slot it belongs to. The drain frees each entry once
      // floor_for_(slot) >= death (version-liveness, NO refcount). The log entries are
      // moved in (heap vector per batch — the writer's path is alloc-free; the per-commit
      // batch vector amortizes over the txn's writes).
      struct dead_batch
      {
         uint32_t                     death;  // displacing-commit version
         uint32_t                     slot;   // root slot this garbage belongs to (per-root floor)
         std::vector<tree::dead_entry> entries;  // SHALLOW/DEEP supersession log
      };
      std::mutex              _dead_mx;
      std::condition_variable _dead_cv;
      std::vector<dead_batch> _dead;                   // committed batches awaiting the floor
      std::atomic<uint64_t>   _reap_enq{0}, _reap_done{0};
      bool                    _reaper_stop = false;    // guarded by _dead_mx
      std::thread             _reaper;                 // started LAST (after all state inited)
   };

   // ── owned_pin: an owned snapshot's VERSION PIN (Phase C, NO node refcount) ──────
   // Claims a database hazard slot and publishes (committed version, live root) into it
   // — exactly the reader-cursor pin — so floor_for_() gates the displaced spine. The
   // snapshot holds a std::shared_ptr<owned_pin>: copies SHARE one published slot (a
   // per-HANDLE count, not per-node), and the LAST copy's dtor clears + releases it.
   struct owned_pin
   {
      database* db   = nullptr;
      uint32_t  slot = UINT32_MAX;  // the claimed hazard slot index
      owned_pin(database* d, uint32_t s) noexcept : db(d), slot(s) {}
      owned_pin(const owned_pin&)            = delete;
      owned_pin& operator=(const owned_pin&) = delete;
      ~owned_pin()
      {
         if (db && slot != UINT32_MAX)
            db->release_hazard_slot(slot);  // clears version+root, frees the slot
      }
   };

   inline snapshot database::make_owned_snapshot(uint32_t idx)
   {
      // Under the per-root writer/publish mutex so the (root, version) we read+publish is
      // atomic w.r.t. a concurrent commit's publish+drain (the root can't be retired-and-
      // drained between our load and our hazard publish). Empty root ⟹ no pin needed.
      std::lock_guard<std::mutex> g(root_mx(idx));
      const uint64_t r = root_slot(idx).load(std::memory_order_seq_cst);
      if (rtag(r) == bt::null)
         return snapshot(snap::adopt(_e, bt::null, 0, &_e.admin_ses()));
      const uint32_t s = claim_hazard_slot();
      hz_root_idx(s).store(idx, std::memory_order_relaxed);  // per-root floor attribution
      // publish (version, root) into the claimed slot (seq_cst), mirroring the cursor pin.
      hz_version(s).store(root_version(idx).load(std::memory_order_seq_cst),
                          std::memory_order_seq_cst);
      hazard_slot(s).store(r, std::memory_order_seq_cst);
      // the read VIEW carries the admin session (cold main-thread reads); the owned_pin
      // (shared) governs the slot lifetime — the view itself holds no slot.
      return snapshot(snap::adopt(_e, rtag(r), rptr(r), &_e.admin_ses()),
                      std::make_shared<owned_pin>(this, s));
   }

   // ── bodies ─────────────────────────────────────────────────────────────────
   // Phase C base pin: the per-root WRITER LOCK (held for the txn lifetime) IS the
   // base — NO node refcount. The committed root stays alive across the txn because the
   // writer lock serializes commits (so it can't be retired mid-txn), the COW logs its
   // displaced spine into the per-txn supersession log (drained only after the floor
   // passes this txn's version), and concurrent readers pin it via hazard slots.
   inline transaction::transaction(database* db, tree* t, uint32_t slot)
       : _db(db),
         _t(t),
         _slot(slot),
         _lock(db->root_mx(slot)),  // base pin: the per-root writer lock (uncontended single-writer)
         _ver(db->next_write_version())
   {
      _start_root = db->root_slot(slot).load(std::memory_order_acquire);
      const bt       rt = rtag(_start_root);
      const uint32_t rp = rptr(_start_root);
      // Phase C — NO working ref (no node refcount anywhere). The committed root stays
      // alive across the txn by: (a) the per-root writer lock (held below) serializes
      // commits, so this root cannot be retired mid-txn; (b) the COW's first reprefix
      // SHALLOW-logs the old root spine (drained only after the floor passes _ver); (c)
      // concurrent readers pin it via hazard slots. The work tree starts from it and
      // COWs (Part A's birth<_V latch clones every committed node it touches).
      _t->set_root(rt, rp);          // working tree starts from the committed root
      _t->set_write_version(_ver);   // nodes born this txn get birth=_ver (latch: in place)
      _t->begin_txn();               // latch the FIX-allocator txn floor + clear clone-log
   }
   inline transaction::~transaction()
   {
      if (_db && !_done)
         abort();
   }

   inline void transaction::commit()
   {
      if (_done)
         return;
      const uint64_t new_root = pack(_t->root_tag(), _t->root_ptr());
      // Phase C — version-liveness, NO refcount. The COW descent already SUPERSEDED
      // every displaced node: a born-this-txn husk freed inline; a COMMITTED (snapshot-
      // shared) node was logged into the tree's per-txn SUPERSESSION LOG (SHALLOW for a
      // cloned routing spine node — its children carried by the clone; DEEP for a
      // dropped subtree). The old root's whole displaced spine is in that log (the first
      // reprefix logged the old root top node, then each hop down). So commit:
      //   1) publish the new root + commit version (the floor's _committed advances),
      //   2) hand the log to the floor-gated drain — each entry is freed only once the
      //      version floor passes _ver (no reader pinned at version < _ver can reach it).
      // There is NO separate old-root retire/deep-reap: the spine IS the log. NO node
      // refcount is touched here (this is the lever that killed the per-txn root-line
      // contention — the commit no longer CASes any node refcount).
      _db->publish_and_drain(_slot, new_root, _ver, _t->clone_log());
      _t->clear_clone_log();
      _lock.unlock();  // release the per-root writer lock (base pin)
      _done = true;
   }
   inline void transaction::abort() noexcept
   {
      if (_done)
         return;
      const bt       wt = _t->root_tag();
      const uint32_t wp = _t->root_ptr();
      const bt       st = rtag(_start_root);
      const uint32_t sp = rptr(_start_root);
      // Phase C abort (start_ver: "drop the chain unpublished" + walk the diverged
      // window). The COMMITTED nodes this txn displaced are in the supersession log;
      // they belong to the RESTORED root and must SURVIVE — so the log is dropped
      // UN-drained. The DIVERGED work spine (the COW clones + fresh leaves, all born
      // THIS txn) is garbage that was never published — free it by WALKING the work
      // root, recursing only into born-this-txn nodes (a committed node is shared with
      // the restored root and stops the walk). NO refcount.
      _t->clear_clone_log();
      if (wt != st || wp != sp)        // any real write diverged the root
         _t->vabort_release(wt, wp);   // free the born-this-txn diverged spine
      _t->set_root(st, sp);            // restore committed
      _lock.unlock();  // release the per-root writer lock (base pin)
      _done = true;
   }

   inline root_handle transaction::primary()
   {
      return root_handle(&_db->eng(), _t);
   }
   inline transaction write_session::start_transaction(uint32_t idx)
   {
      return transaction(_db, &_db->work_tree(idx), idx);
   }

   inline snapshot write_session::get_root(uint32_t idx)
   {
      // Phase C OWNED snapshot — a VERSION PIN, NO node refcount. Claim a hazard slot
      // and publish (committed version, live root) so floor_() gates the displaced spine
      // exactly as a reader cursor does; the slot lifetime is governed by the snapshot's
      // shared_ptr<owned_pin> (per-handle, copyable). Under the per-root writer/publish
      // mutex so the (root, version) pair we read+publish is atomic w.r.t. a concurrent
      // commit's publish+drain. Cold main/admin-thread path (the engine admin session
      // resolves leaf reads). An EMPTY (null) root publishes no pin (nothing to gate).
      return _db->make_owned_snapshot(idx);
   }

   inline read_session::read_session(database* db)
       : _db(db), _lses(db->eng().leaf_create_session()), _hz_slot(db->claim_hazard_slot())
   {
   }
   inline read_session::~read_session() { _db->release_hazard_slot(_hz_slot); }

   inline cursor read_session::create_cursor(uint32_t idx)
   {
      // Part B HAZARD publish + Phase A version pin: store the live root into THIS
      // reader's hazard slot with the start_ref re-verify loop, take NO root refcount,
      // AND publish the root's commit version into the parallel version slot so floor_()
      // can read what this reader pinned (mirrors start_ver create_cursor). Attribute
      // the pin to its root index BEFORE publishing (per-root floor: a pin on root R
      // must not gate a different root's garbage). The published word is guaranteed
      // not-yet-reclaimed (the writer drains AFTER its seq_cst live-root store; the
      // drain's seq_cst slot scan then sees our slot). The cursor's dtor (snap::reset
      // under _hp) clears both slots — readers never touch a root refcount or lock.
      _db->hz_root_idx(_hz_slot).store(idx, std::memory_order_relaxed);
      return cursor(snap::publish(_db->eng(), _db->root_slot(idx), _db->root_version(idx),
                                  &_db->hazard_slot(_hz_slot), &_db->hz_version(_hz_slot),
                                  _lses.get()));
   }

   // ── adb engine bundle (adb::kv_engine) ───────────────────────────────────────
   // The five role types as nested typedefs — the shape a tiered engine
   // (delta_art) or a drop-in embedder (psitrimdbx: `using TK = engine;`) binds
   // to. `engine` here is the BUNDLE; the actual node/storage engine is
   // gcart::engine (the node/storage engine, written qualified throughout this
   // file to keep the bare name `engine` free for this bundle — the X::kv::engine
   // convention every engine uses).
   struct engine
   {
      using database      = gcart::kv::database;
      using write_session = gcart::kv::write_session;
      using read_session  = gcart::kv::read_session;
      using transaction   = gcart::kv::transaction;
      using cursor        = gcart::kv::cursor;
      using config        = gcart::kv::config;  // delta_art cold-base tier (dwal::kv_engine)
   };
}  // namespace gcart::kv

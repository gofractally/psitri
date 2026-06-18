// gcart::kv — adb-style adapter over the spec engine (gcart), with
// lock-free MVCC. Mirrors the v1 gcart::legacy::kv discipline (smart_ptr IS the pin,
// try_retain+verify acquire, COW-to-root, give() hand-off) but the root is a
// (branch_type tag, u32 ptr) pair packed into an atomic u64 published slot.
#pragma once
#include <gcart/spec_tree.hpp>

#include <sal/config.hpp>   // sal::open_mode (embedder open factory)
#include <sal/mapping.hpp>  // sal::mapping — mmap-backed durable root table (SAL roots pattern)

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

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

   // ── smart_ptr: a held (tag,ptr) root IS the snapshot pin ───────────────────
   class snap
   {
     public:
      snap() = default;
      static snap acquire(gcart::engine& e, std::atomic<uint64_t>& slot)
      {
         for (;;)
         {
            const uint64_t r = slot.load(std::memory_order_acquire);
            const bt       t = rtag(r);
            const uint32_t p = rptr(r);
            if (t == bt::null)
               return snap(&e, bt::null, 0);
            if (!try_retain(e, t, p))
               continue;  // lost race with a freeing writer
            if (r == slot.load(std::memory_order_acquire))
               return snap(&e, t, p);
            release_subtree(e, t, p);  // root moved — undo, retry
         }
      }
      // ADOPT an existing owned (tag,ptr) WITHOUT retaining — the caller transfers a
      // ref it already holds (the set_root exchange: a work tree's root → a snapshot).
      static snap adopt(gcart::engine& e, bt t, uint32_t p) noexcept { return snap(&e, t, p); }
      snap(const snap& o) : _e(o._e), _t(o._t), _p(o._p)
      {
         if (_e && _t != bt::null)
            retain(*_e, _t, _p);
      }
      snap& operator=(const snap& o)
      {
         if (this != &o)
         {
            reset();
            _e = o._e;
            _t = o._t;
            _p = o._p;
            if (_e && _t != bt::null)
               retain(*_e, _t, _p);
         }
         return *this;
      }
      snap(snap&& o) noexcept : _e(o._e), _t(o._t), _p(o._p) { o._t = bt::null; }
      snap& operator=(snap&& o) noexcept
      {
         if (this != &o)
         {
            reset();
            _e   = o._e;
            _t   = o._t;
            _p   = o._p;
            o._t = bt::null;
         }
         return *this;
      }
      ~snap() { reset(); }

      bt       tag() const noexcept { return _t; }
      uint32_t ptr() const noexcept { return _p; }
      gcart::engine* eng() const noexcept { return _e; }
      void     give() noexcept { _t = bt::null; }  // relinquish without releasing
      // the pin OWNS its release. (A/B: SYNC release to isolate the bump allocator
      // from the async bg cascade.)
      void reset() noexcept
      {
         if (_e && _t != bt::null)
            release_subtree(*_e, _t, _p);
         _t = bt::null;
      }

     private:
      snap(gcart::engine* e, bt t, uint32_t p) : _e(e), _t(t), _p(p) {}
      gcart::engine* _e = nullptr;
      bt       _t = bt::null;
      uint32_t _p = 0;
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
      explicit cursor(snap s) : _s(std::move(s)) {}

      // point get (no positioning) — find_in is a direct descent, no key build.
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return std::nullopt;
         std::string_view out;  // view into the value (snapshot pins it); construct once
         if (!find_in(*_s.eng(), _s.tag(), _s.ptr(), k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return false;
         std::string_view out;  // zero-copy: hand the value view straight to f
         if (!find_in(*_s.eng(), _s.tag(), _s.ptr(), k, out))
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
         go_left_(_s.tag(), root_());
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
         if (seek_ge_(_s.tag(), root_(), k, 0))
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
         const sv v = cur_value_();
         return S(v.data(), v.size());
      }
      template <class F>
      bool get_value(F&& f) const
      {
         if (_end || _rend)
            return false;
         f(cur_value_());
         return true;
      }
      bool is_subtree() const { return false; }

      // ── ordered reverse iteration ───────────────────────────────────────────
      bool seek_last()
      {
         if (!root_ok_())
            return false;
         _n = 0;
         go_right_(_s.tag(), root_());
         _end  = false;
         _rend = false;
         _kv   = false;
         return true;
      }
      bool upper_bound(key_view k)  // smallest key STRICTLY > k — no copy, no '\0' synth
      {
         if (!lower_bound(k))   // smallest key >= k (k stays a VIEW)
            return false;
         if (key() == k)        // landed exactly on k → the answer is the next key
            return next();
         return true;           // landed strictly > k already
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
      const void*  root_() const { return resolve(*_s.eng(), _s.tag(), _s.ptr()); }
      gcart::engine& eng_() const { return *_s.eng(); }

      void set_leaf_ext_(bt t, const void* n)
      {
         _inl = false;
         _lt  = t;
         _ln  = n;
      }
      void set_leaf_inl_(sv suf, sv val)
      {
         _inl  = true;
         _isuf = suf;
         _ival = val;
      }
      sv cur_value_() const { return _inl ? _ival : leaf_value_body(_lt, _ln); }

      void push_(bt t, const void* node, int cur)
      {
         _f[_n].t    = t;
         _f[_n].node = node;
         _f[_n].cur  = cur;
         ++_n;
      }

      // Descend to the LEFTMOST leaf of subtree (t,node), pushing a frame per
      // routing node (cur = the slot taken: -1 term, else first child).
      void go_left_(bt t, const void* node)
      {
         for (;;)
         {
            if (is_leaf_tag(t))
            {
               set_leaf_ext_(t, node);
               return;
            }
            if (t == bt::full)
            {
               const full* n = static_cast<const full*>(node);
               if (n->has_term())
               {
                  push_(t, node, -1);
                  t    = bt(n->term_tag);
                  node = resolve(eng_(), t, n->term_ptr);
                  continue;
               }
               int b = 0;
               while (b < 256 && n->btype(b) == bt::null)
                  ++b;
               push_(t, node, b);
               bt ct = n->btype(b);
               node  = resolve(eng_(), ct, n->branch(b));
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
                  node  = resolve(eng_(), tt, n->ext_ptr(ti));
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
                  node = n->in_setlist(0);
                  continue;
               }
               bt ct = n->otype(0);
               node  = resolve(eng_(), ct, n->ext_ptr(0));
               t     = ct;
               continue;
            }
            const setlist* n = static_cast<const setlist*>(node);
            if (n->has_term())
            {
               push_(t, node, -1);
               bt tt = n->term_type();
               node  = resolve(eng_(), tt, n->term_branch());
               t     = tt;
               continue;
            }
            push_(t, node, 0);
            bt ct = n->btype(0);
            node  = resolve(eng_(), ct, n->branch(0));
            t     = ct;
         }
      }

      // Descend to the RIGHTMOST leaf (largest key) — children last, term first
      // (so the rightmost is the last byte-child, or the term iff no children).
      void go_right_(bt t, const void* node)
      {
         for (;;)
         {
            if (is_leaf_tag(t))
            {
               set_leaf_ext_(t, node);
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
                  node = resolve(eng_(), t, n->term_ptr);
                  continue;
               }
               push_(t, node, b);
               bt ct = n->btype(b);
               node  = resolve(eng_(), ct, n->branch(b));
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
                  node  = resolve(eng_(), tt, n->ext_ptr(ti));
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
                  node = n->in_setlist(last);
                  continue;
               }
               bt ct = n->otype(last);
               node  = resolve(eng_(), ct, n->ext_ptr(last));
               t     = ct;
               continue;
            }
            const setlist* n = static_cast<const setlist*>(node);
            if (n->nbranch == 0)  // term-only
            {
               push_(t, node, -1);
               bt tt = n->term_type();
               node  = resolve(eng_(), tt, n->term_branch());
               t     = tt;
               continue;
            }
            const uint32_t last = n->nbranch - 1u;
            push_(t, node, int(last));
            bt ct = n->btype(last);
            node  = resolve(eng_(), ct, n->branch(last));
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
            go_left_(ct, resolve(eng_(), ct, n->branch(b)));
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
               go_left_(bt::setlist, n->in_setlist(i));
               return true;
            }
            bt ct = n->otype(i);
            go_left_(ct, resolve(eng_(), ct, n->ext_ptr(i)));
            return true;
         }
         const setlist* n = static_cast<const setlist*>(f.node);
         const uint32_t i = (f.cur < 0) ? 0u : uint32_t(f.cur) + 1u;
         if (i >= n->nbranch)
            return false;
         f.cur = int(i);
         bt ct = n->btype(i);
         go_left_(ct, resolve(eng_(), ct, n->branch(i)));
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
               go_right_(tt, resolve(eng_(), tt, n->term_ptr));
               return true;
            }
            f.cur = b;
            bt ct = n->btype(b);
            go_right_(ct, resolve(eng_(), ct, n->branch(b)));
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
               go_right_(tt, resolve(eng_(), tt, n->ext_ptr(ti)));
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
               go_right_(bt::setlist, n->in_setlist(i));
               return true;
            }
            bt ct = n->otype(i);
            go_right_(ct, resolve(eng_(), ct, n->ext_ptr(i)));
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
            go_right_(tt, resolve(eng_(), tt, n->term_branch()));
            return true;
         }
         const uint32_t i = uint32_t(f.cur) - 1u;
         f.cur            = int(i);
         bt ct            = n->btype(i);
         go_right_(ct, resolve(eng_(), ct, n->branch(i)));
         return true;
      }

      // Directed descent: position at the smallest key >= target, pushing frames.
      // Returns true if such a key exists in this subtree. On a "diverge / target is
      // a prefix" the whole subtree is > target → its leftmost is the answer.
      bool seek_ge_(bt t, const void* node, sv target, uint32_t depth)
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
               set_leaf_ext_(t, node);
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
                  go_left_(t, node);
                  return true;
               }
               return false;  // subtree entirely < target
            }
            if (pl >= rem)  // target is a (proper or full) prefix of P → subtree >= target
            {
               go_left_(t, node);
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
               if (seek_ge_(ct, resolve(eng_(), ct, n->branch(tb)), target, depth + 1))
                  return true;
               --_n;  // matching child had nothing >= target; fall through to next byte
            }
            for (int b = tb + 1; b < 256; ++b)
               if (n->btype(b) != bt::null)
               {
                  push_(t, node, b);
                  bt c2 = n->btype(b);
                  go_left_(c2, resolve(eng_(), c2, n->branch(b)));
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
                  else if (seek_ge_(bt::setlist, n->in_setlist(ui), target, depth + 1))
                     return true;
               }
               else
               {
                  bt ct = n->otype(ui);
                  if (seek_ge_(ct, resolve(eng_(), ct, n->ext_ptr(ui)), target, depth + 1))
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
                  go_left_(bt::setlist, n->in_setlist(start));
                  return true;
               }
               bt ct = n->otype(start);
               go_left_(ct, resolve(eng_(), ct, n->ext_ptr(start)));
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
            if (seek_ge_(ct, resolve(eng_(), ct, n->branch(ui)), target, depth + 1))
               return true;
            --_n;
            start = ui + 1;
         }
         if (start < n->nbranch)
         {
            push_(t, node, int(start));
            bt ct = n->btype(start);
            go_left_(ct, resolve(eng_(), ct, n->branch(start)));
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
            const sv suf = leaf_suffix_body(_lt, _ln);
            app(suf.data(), uint32_t(suf.size()));
         }
         _klen = len;
         _kv   = true;
      }

      snap        _s;
      frame       _f[MAXF];
      int         _n  = 0;       // frames in use
      bt          _lt = bt::null;
      const void* _ln = nullptr;  // current EXTERNAL leaf
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
   class snapshot
   {
     public:
      snapshot() = default;
      explicit snapshot(snap s) : _s(std::move(s)) {}
      cursor   snapshot_cursor() const { return cursor(_s); }  // copies the pin (retains)
      explicit operator bool() const noexcept { return _s.eng() != nullptr; }

      // point read of the frozen snapshot (used by the snapshot-isolation battery)
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if (!_s.eng() || _s.tag() == bt::null)
            return std::nullopt;
         std::string_view out;  // view into the value (snapshot pins it); construct once
         if (!find_in(*_s.eng(), _s.tag(), _s.ptr(), k, out))
            return std::nullopt;
         return S(out.data(), out.size());
      }
      // root-exchange primitives (set_root): the snapshot IS the owned tree.
      bt       tag() const noexcept { return _s.tag(); }
      uint32_t ptr() const noexcept { return _s.ptr(); }
      void     give() noexcept { _s.give(); }  // relinquish the ref (it moved into a work tree)
      static snapshot adopt(gcart::engine& e, bt t, uint32_t p) { return snapshot(snap::adopt(e, t, p)); }

     private:
      snap _s;
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
         sv v;  // view into the value (work root pins it); construct once
         if (!_w->find(k, v))
            return std::nullopt;
         return S(v.data(), v.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         sv v;  // zero-copy: hand the value view straight to f
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
         return snapshot::adopt(*_e, dt, dp);           // own the displaced
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
            _snap(std::move(o._snap)),
            _ver(o._ver),
            _done(o._done)
      {
         o._db = nullptr;
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
         sv v;  // view into the value (txn work root pins it); construct once
         if (!_t->find(k, v))
            return std::nullopt;
         return S(v.data(), v.size());
      }
      // lambda form (adb::kv_transaction): invoke f(value) on a hit, return found.
      template <class F>
      bool get(key_view k, F&& f) const
      {
         sv v;  // zero-copy: hand the value view straight to f
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
      database* _db   = nullptr;
      tree*     _t    = nullptr;
      uint32_t  _slot = 0;
      snap      _snap;
      uint32_t  _ver  = 0;  // this txn's commit version (birth of new nodes)
      bool      _done = false;
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
      explicit read_session(database* db) : _db(db) {}
      cursor create_cursor(uint32_t idx);

     private:
      database* _db;
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
      // settled once the engine's background cascade has caught up.
      uint64_t live_bytes() noexcept { return _e.live_bytes(); }
      void     drain() noexcept { _e.drain(); }  // measurement barrier
      ~database() { _roots_map.sync(sal::sync_type::full); }  // clean close

     private:
      static constexpr uint32_t max_roots   = 64;
      static constexpr uint32_t ver_index   = max_roots;  // commit-version slot (past the roots)
      static constexpr size_t   roots_bytes = max_roots * sizeof(uint64_t);

      explicit database(std::filesystem::path dir)
          : _e(dir), _roots_map(dir / "roots.dat", sal::access_mode::read_write)
      {
         if (_roots_map.size() < roots_bytes)  // fresh file → init all slots to null_root
         {                                      // (null_root = bt::null<<32 is NON-zero, so a
            _roots_map.resize(4096);            //  zero-filled mmap would NOT be null)
            auto* r = _roots_map.as<std::atomic<uint64_t>>();
            for (uint32_t i = 0; i < max_roots; ++i)
               r[i].store(null_root, std::memory_order_relaxed);
            r[ver_index].store(1, std::memory_order_relaxed);  // first txn gets version 2
         }
      }

      gcart::engine                         _e;
      sal::mapping                          _roots_map;
      std::array<std::unique_ptr<tree>, 64> _work{};
   };

   // ── bodies ─────────────────────────────────────────────────────────────────
   inline transaction::transaction(database* db, tree* t, uint32_t slot)
       : _db(db),
         _t(t),
         _slot(slot),
         _snap(snap::acquire(db->eng(), db->root_slot(slot))),
         _ver(db->next_write_version())
   {
      _t->set_root(_snap.tag(), _snap.ptr());  // working tree starts from the committed root
      _t->set_write_version(_ver);  // nodes born this txn get birth=_ver (latch: in place)
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
      // publish the new working root atomically, then drop the snapshot pin on the
      // old committed root. _snap.reset() routes through the engine's background
      // reclaimer (the cascade runs off the writer's critical path); the smart_ptr
      // owns its own release.
      _db->root_slot(_slot).store(pack(_t->root_tag(), _t->root_ptr()), std::memory_order_release);
      _snap.reset();
      _done = true;
   }
   inline void transaction::abort() noexcept
   {
      if (_done)
         return;
      const bt       wt = _t->root_tag();
      const uint32_t wp = _t->root_ptr();
      _t->set_root(_snap.tag(), _snap.ptr());  // restore committed
      _snap.give();                            // the snap pin becomes the live tree's ownership ref
      release_subtree(_db->eng(), wt, wp);     // free the discarded work (no-op-safe if ==committed)
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
      return snapshot(snap::acquire(_db->eng(), _db->root_slot(idx)));
   }

   inline cursor read_session::create_cursor(uint32_t idx)
   {
      return cursor(snap::acquire(_db->eng(), _db->root_slot(idx)));
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

// gcart::st — the spec node taxonomy over the three regions (refcount era).
//
// Routing lives in FIX (cl1 setlist, cl2 double, cl9 full); terminal (suffix,
// value) leaves live in SAL (32B); large values in BIG. A branch is a 4-byte
// region-typed pointer: its 3-bit `branch_type` tag (in the parent's packed
// types[] array) selects the region AND lets dispatch PREFETCH the child's
// cacheline before any load that depends on it. There is no leaf node — a leaf
// is a suffix_value with a 0-length suffix (the term).
//
// Liveness = per-slot refcount in each region (Dan: refcount for now). Mutation
// is copy-construct-rebuild from the stable source into a right-sized slot (the
// new slot bubbles up; COW-to-root); pointer children are retained per region,
// inline-none (suffix_value is its own SAL object, refcounted there).
#pragma once
#include <gcart/node.hpp>           // branch_type, tag3_get/set, tag3_bytes
#include <sal2/fixed_allocator.hpp>
#include <sal2/var_allocator.hpp>

#include <cstring>
#include <string>
#include <string_view>

namespace gcart::st
{
   using sv  = std::string_view;
   using bt  = branch_type;

   // COW descent mode latch (Dan §5.2): a scoped enum, not a bool. The recursion
   // CACHES whether the subtree is private or shared so it does not re-check every
   // node once committed to <shared>. <unique> reads each routing node's refcount
   // (==1 ⟹ mutate in place); the FIRST shared node clones (retaining children →
   // they become shared) and latches the rest of the descent to <shared>, which
   // clones without re-checking. (Refcount stands in for the spec's birth==V.)
   enum class cow_mode { shared, unique };

   // A node reference: which region (by tag) + the 4-byte slot index.
   inline uint32_t lcp(sv a, sv b) noexcept
   {
      const uint32_t n = uint32_t(a.size() < b.size() ? a.size() : b.size());
      uint32_t i = 0; while (i < n && a[i] == b[i]) ++i; return i;
   }

   // ── engine: the four region allocators ────────────────────────────────────
   struct engine
   {
      sal2::fixed_allocator<128>  cl1;   // setlist / nested_setlist
      sal2::fixed_allocator<256>  cl2;   // double_setlist / big_nested
      sal2::fixed_allocator<1152> cl9;   // full / node256
      sal2::var_allocator<32>     sal;   // suffix_value leaves

      explicit engine(std::filesystem::path dir)
          : cl1({.dir = dir, .name = "cl1"}),
            cl2({.dir = dir, .name = "cl2"}),
            cl9({.dir = dir, .name = "cl9"}),
            sal({.dir = dir, .name = "sal"}) {}

      uint64_t live_bytes() noexcept { return cl1.live_bytes() + cl2.live_bytes() + cl9.live_bytes() + sal.live_bytes(); }
   };

   // ── suffix_value (SAL leaf): [u16 vlen][u8 slen][value][suffix] ───────────
   // slen==0 => the term (a "leaf" is suffix_value with a 0-length suffix).
   struct sv_leaf
   {
      static uint32_t bytes_for(uint32_t vlen, uint32_t slen) noexcept { return 3 + vlen + slen; }
      static uint32_t make(engine& e, sal2::var_allocator<32>::session& ses, sv suffix, sv value)
      {
         const uint32_t vl = uint32_t(value.size()), sl = uint32_t(suffix.size());
         const uint32_t slot = ses.alloc(bytes_for(vl, sl));
         char* p = static_cast<char*>(e.sal.get(slot));
         const uint16_t v16 = uint16_t(vl);
         std::memcpy(p, &v16, 2); p[2] = char(uint8_t(sl));
         std::memcpy(p + 3, value.data(), vl);
         std::memcpy(p + 3 + vl, suffix.data(), sl);
         return slot;
      }
      static sv value(const engine& e, uint32_t slot) noexcept
      {
         const char* p = static_cast<const char*>(e.sal.get(slot));
         uint16_t v16; std::memcpy(&v16, p, 2);
         return {p + 3, v16};
      }
      static sv suffix(const engine& e, uint32_t slot) noexcept
      {
         const char* p = static_cast<const char*>(e.sal.get(slot));
         uint16_t v16; std::memcpy(&v16, p, 2);
         const uint8_t sl = uint8_t(p[2]);
         return {p + 3 + v16, sl};
      }
      static uint32_t total_bytes(const engine& e, uint32_t slot) noexcept
      {
         const char* p = static_cast<const char*>(e.sal.get(slot));
         uint16_t v16; std::memcpy(&v16, p, 2);
         return bytes_for(v16, uint8_t(p[2]));
      }
   };

   // ── setlist (cl1, 128 B): the sparse routing node ─────────────────────────
   // layout: [u8 meta(terminal:1,prefixlen:7)] [u8 nbranch]
   //         prefix[prefixlen]  keys[nbranch]  types[nbranch+term]:3bit  branches[nbranch+term]:u32
   // branches[] is region-typed by types[]; the term branch (if any) is the LAST
   // slot (no key) with type suffix_value/big_suffix_value, slen 0.
   struct setlist
   {
      uint8_t meta;     // terminal:1 (bit7), prefixlen:7 (bits0-6)
      uint8_t nbranch;

      static constexpr uint32_t slot_bytes = 128;
      bool     has_term() const noexcept { return meta & 0x80; }
      uint32_t prefixlen() const noexcept { return meta & 0x7f; }
      uint32_t total() const noexcept { return nbranch + (has_term() ? 1u : 0u); }

      static uint32_t keys_off() noexcept { return 2; }  // after meta+nbranch+prefix; prefix offset added by caller
      // offsets are computed from (nbranch, prefixlen, total):
      char*       base() noexcept { return reinterpret_cast<char*>(this); }
      const char* base() const noexcept { return reinterpret_cast<const char*>(this); }
      char*       prefix_ptr() noexcept { return base() + 2; }
      const char* prefix_ptr() const noexcept { return base() + 2; }
      sv          prefix() const noexcept { return {prefix_ptr(), prefixlen()}; }
      uint8_t*    keys() noexcept { return reinterpret_cast<uint8_t*>(base() + 2 + prefixlen()); }
      const uint8_t* keys() const noexcept { return reinterpret_cast<const uint8_t*>(base() + 2 + prefixlen()); }
      uint8_t*    types() noexcept { return reinterpret_cast<uint8_t*>(keys() + nbranch); }
      const uint8_t* types() const noexcept { return reinterpret_cast<const uint8_t*>(keys() + nbranch); }
      uint32_t    branches_off() const noexcept
      {
         const uint32_t o = 2 + prefixlen() + nbranch + tag3_bytes(total());
         return (o + 3u) & ~3u;  // 4-byte align
      }
      uint32_t*   branches() noexcept { return reinterpret_cast<uint32_t*>(base() + branches_off()); }
      const uint32_t* branches() const noexcept { return reinterpret_cast<const uint32_t*>(base() + branches_off()); }

      static uint32_t bytes_for(uint32_t nbranch, uint32_t prefixlen, bool term) noexcept
      {
         const uint32_t total = nbranch + (term ? 1u : 0u);
         const uint32_t o = 2 + prefixlen + nbranch + tag3_bytes(total);
         return ((o + 3u) & ~3u) + total * 4u;
      }
      static bool fits(uint32_t nbranch, uint32_t prefixlen, bool term) noexcept
      {
         return bytes_for(nbranch, prefixlen, term) <= slot_bytes;
      }

      branch_type btype(uint32_t i) const noexcept { return tag3_get(types(), i); }
      uint32_t    branch(uint32_t i) const noexcept { return branches()[i]; }
      branch_type term_type() const noexcept { return has_term() ? tag3_get(types(), nbranch) : bt::null; }
      uint32_t    term_branch() const noexcept { return branches()[nbranch]; }
      // index of byte among keys[0..nbranch), or -1.
      int find_index(uint8_t byte) const noexcept
      {
         int pos = int(ucc::lower_bound_padded(keys(), nbranch, byte));
         return (pos < int(nbranch) && keys()[pos] == byte) ? pos : -1;
      }
   };

   // ── full (cl9, 1152 B): the dense routing node (direct byte index) ─────────
   // [u8 meta(term:1,prefixlen:7)] [u8 term_tag] [u16 nbranch] [u32 term_ptr]
   //   types[256]:3bit(96B) @8  branches[256]:u32(1024B) @104  prefix[prefixlen] @1128
   struct full
   {
      uint8_t  meta;
      uint8_t  term_tag;   // branch_type of the term (if terminal)
      uint16_t nbranch;
      uint32_t term_ptr;

      static constexpr uint32_t slot_bytes  = 1152;
      static constexpr uint32_t types_off   = 8;
      static constexpr uint32_t branches_off = 104;     // 8 + 96
      static constexpr uint32_t prefix_off  = 1128;     // 104 + 1024
      static constexpr uint32_t max_prefix  = slot_bytes - prefix_off;  // 24

      bool     has_term() const noexcept { return meta & 0x80; }
      uint32_t prefixlen() const noexcept { return meta & 0x7f; }
      char*       base() noexcept { return reinterpret_cast<char*>(this); }
      const char* base() const noexcept { return reinterpret_cast<const char*>(this); }
      uint8_t*    types() noexcept { return reinterpret_cast<uint8_t*>(base() + types_off); }
      const uint8_t* types() const noexcept { return reinterpret_cast<const uint8_t*>(base() + types_off); }
      uint32_t*   branches() noexcept { return reinterpret_cast<uint32_t*>(base() + branches_off); }
      const uint32_t* branches() const noexcept { return reinterpret_cast<const uint32_t*>(base() + branches_off); }
      char*       prefix_ptr() noexcept { return base() + prefix_off; }
      const char* prefix_ptr() const noexcept { return base() + prefix_off; }
      sv          prefix() const noexcept { return {prefix_ptr(), prefixlen()}; }
      branch_type btype(uint32_t byte) const noexcept { return tag3_get(types(), byte); }
      uint32_t    branch(uint32_t byte) const noexcept { return branches()[byte]; }
   };

   // ── region dispatch: a 3-bit tag picks the region; resolve/retain/release ──
   inline bool is_routing(bt t) noexcept
   {
      return t == bt::setlist || t == bt::nested_setlist || t == bt::double_setlist ||
             t == bt::big_nested || t == bt::full;
   }
   // Which FIX class a routing tag lives in (cl1/cl2/cl9). suffix_value → SAL.
   inline void* resolve(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full:           return e.cl9.get(p);
         case bt::double_setlist:
         case bt::big_nested:     return e.cl2.get(p);
         case bt::setlist:
         case bt::nested_setlist: return e.cl1.get(p);
         case bt::suffix_value:   return e.sal.get(p);
         default:                 return nullptr;  // big_suffix_value (BIG) / null
      }
   }
   inline void prefetch(const void* p) noexcept
   {
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(p, 0, 3);
#else
      (void)p;
#endif
   }
   // a routing node is UNIQUE (ref==1) iff no snapshot pins it → safe to mutate
   // in place (no clone). Only routing nodes are mutated in place.
   inline bool unique(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full: return e.cl9.unique(p);
         case bt::double_setlist: case bt::big_nested: return e.cl2.unique(p);
         case bt::setlist: case bt::nested_setlist: return e.cl1.unique(p);
         default: return false;
      }
   }
   // lock-free pin attempt: bump only if alive (snapshot acquire). false = dead.
   [[nodiscard]] inline bool try_retain(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full: return e.cl9.try_retain(p);
         case bt::double_setlist: case bt::big_nested: return e.cl2.try_retain(p);
         case bt::setlist: case bt::nested_setlist: return e.cl1.try_retain(p);
         case bt::suffix_value: return e.sal.try_retain(p);
         default: return false;
      }
   }
   inline void retain(engine& e, bt t, uint32_t p)
   {
      switch (t)
      {
         case bt::full: e.cl9.retain(p); break;
         case bt::double_setlist: case bt::big_nested: e.cl2.retain(p); break;
         case bt::setlist: case bt::nested_setlist: e.cl1.retain(p); break;
         case bt::suffix_value: e.sal.retain(p); break;
         default: break;
      }
   }
   // returns true iff the slot hit zero (caller recurses into children + frees).
   [[nodiscard]] inline bool release_one(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full: return e.cl9.release_one(p);
         case bt::double_setlist: case bt::big_nested: return e.cl2.release_one(p);
         case bt::setlist: case bt::nested_setlist: return e.cl1.release_one(p);
         case bt::suffix_value: return e.sal.release_one(p);
         default: return false;
      }
   }
}  // namespace gcart::st

namespace gcart::st
{
   // a decoded branch (scratch, for copy-construct-rebuild of a routing node)
   struct br { uint8_t byte; bt tag; uint32_t ptr; };

   inline bool pfx_ok_(sv pfx, sv key, uint32_t depth) noexcept
   {
      return key.size() - depth >= pfx.size() &&
             (pfx.empty() || std::memcmp(key.data() + depth, pfx.data(), pfx.size()) == 0);
   }

   // Read-path find from a (tag,ptr) root — usable by readers/cursors with NO
   // write sessions. Dispatch on the branch tag, PREFETCH the child before the
   // dependent load (full prefetches the offset cacheline before the type hdr).
   inline bool find_in(engine& e, bt t, uint32_t p, sv key, std::string& out)
   {
      uint32_t depth = 0;
      for (;;)
      {
         if (t == bt::null) return false;
         void* node = resolve(e, t, p);
         if (t == bt::suffix_value)
         {
            if (sv_leaf::suffix(e, p) == key.substr(depth))
            { sv v = sv_leaf::value(e, p); out.assign(v.data(), v.size()); return true; }
            return false;
         }
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(node);
            if (!pfx_ok_(n->prefix(), key, depth)) return false;
            depth += n->prefixlen();
            if (depth == key.size())
            { if (!n->has_term()) return false; t = bt(n->term_tag); p = n->term_ptr; prefetch(resolve(e, t, p)); continue; }
            const uint8_t byte = uint8_t(key[depth]);
            prefetch(n->branches() + byte);          // offset cacheline before the type header
            const bt ct = n->btype(byte);
            if (ct == bt::null) return false;
            t = ct; p = n->branch(byte); ++depth; prefetch(resolve(e, t, p)); continue;
         }
         const setlist* n = static_cast<const setlist*>(node);
         if (!pfx_ok_(n->prefix(), key, depth)) return false;
         depth += n->prefixlen();
         if (depth == key.size())
         { if (!n->has_term()) return false; t = n->term_type(); p = n->term_branch(); prefetch(resolve(e, t, p)); continue; }
         const int idx = n->find_index(uint8_t(key[depth]));
         if (idx < 0) return false;
         t = n->btype(uint32_t(idx)); p = n->branch(uint32_t(idx)); ++depth;
         prefetch(resolve(e, t, p)); continue;
      }
   }

   // ── free reclamation (snapshots / readers; no write sessions needed) ───────
   inline void release_subtree(engine& e, bt t, uint32_t p)
   {
      if (t == bt::null) return;
      if (t == bt::suffix_value)
      { if (e.sal.release_one(p)) e.sal.free(p, sv_leaf::total_bytes(e, p)); return; }
      if (!is_routing(t)) return;
      if (!release_one(e, t, p)) return;   // still shared (another snapshot pins it)
      // hit zero: recurse children + term, then free the node's slot.
      if (t == bt::full)
      {
         full* n = static_cast<full*>(resolve(e, t, p));
         for (uint32_t b = 0; b < 256; ++b)
         { const bt ct = n->btype(b); if (ct != bt::null) release_subtree(e, ct, n->branch(b)); }
         if (n->has_term()) release_subtree(e, bt(n->term_tag), n->term_ptr);
         e.cl9.free(p);
      }
      else
      {
         setlist* n = static_cast<setlist*>(resolve(e, t, p));
         for (uint32_t i = 0; i < n->nbranch; ++i) release_subtree(e, n->btype(i), n->branch(i));
         if (n->has_term()) release_subtree(e, n->term_type(), n->term_branch());
         (t == bt::double_setlist || t == bt::big_nested) ? e.cl2.free(p) : e.cl1.free(p);
      }
   }

   class tree
   {
     public:
      explicit tree(engine& e)
          : _e(e), _s1(e.cl1.start_session()), _s2(e.cl2.start_session()),
            _s9(e.cl9.start_session()), _ssal(e.sal.start_session()) {}

      uint64_t size() const noexcept { return _size; }

      // ── read path: dispatch on the branch tag, prefetch the child BEFORE the
      //    dependent load (Dan's hard rule; full node prefetches the offset
      //    cacheline before reading the 3-bit type header). ────────────────────
      bool find(sv key, std::string& out) const { return find_in(const_cast<engine&>(_e), _rt, _rp, key, out); }

      bt       root_tag() const noexcept { return _rt; }
      uint32_t root_ptr() const noexcept { return _rp; }
      void     set_root(bt t, uint32_t p) noexcept { _rt = t; _rp = p; }
      engine&  eng() noexcept { return _e; }
      // release a (tag,ptr) subtree through refcount (snapshot drop / abort).
      void     release(bt t, uint32_t p) { release_ref(t, p); }

      // ── write path ─────────────────────────────────────────────────────────
      // Public entry descends in <unique> and latches to <shared> on the first
      // snapshot-pinned node (doc §5.2). One refcount read at the root decides
      // in-place vs clone; the cost is cached down the descent.
      void upsert(sv key, sv val)
      {
         auto [t, p] = upsert_rec<cow_mode::unique>(_rt, _rp, key, 0, val);
         _rt = t; _rp = p;
      }

     private:
      static bool pfx_ok(sv pfx, sv key, uint32_t depth) noexcept
      {
         return key.size() - depth >= pfx.size() &&
                (pfx.empty() || std::memcmp(key.data() + depth, pfx.data(), pfx.size()) == 0);
      }

      std::pair<bt, uint32_t> make_sv(sv suffix, sv val)
      {
         return {bt::suffix_value, sv_leaf::make(_e, _ssal, suffix, val)};
      }
      void release_ref(bt t, uint32_t p)
      {
         if (t == bt::null) return;
         if (t == bt::suffix_value)
         {
            if (_e.sal.release_one(p))
            { /* free its grains */ _e.sal.free(p, sv_leaf::total_bytes(_e, p)); }
            return;
         }
         if (!is_routing(t)) return;
         if (!release_one(_e, t, p)) return;          // still shared (snapshot)
         // hit zero: release children, then free the node's slot
         read_branches(t, p, [&](const br& b) { release_ref(b.tag, b.ptr); });
         bt tt; uint32_t tp;
         if (node_term(t, p, tt, tp)) release_ref(tt, tp);
         free_node(t, p);
      }

      // visit a routing node's byte-branches.
      template <class F>
      void read_branches(bt t, uint32_t p, F&& f) const
      {
         engine& e = const_cast<engine&>(_e);
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(resolve(e, t, p));
            for (uint32_t b = 0; b < 256; ++b)
            { const bt ct = n->btype(b); if (ct != bt::null) f(br{uint8_t(b), ct, n->branch(b)}); }
         }
         else
         {
            const setlist* n = static_cast<const setlist*>(resolve(e, t, p));
            for (uint32_t i = 0; i < n->nbranch; ++i) f(br{n->keys()[i], n->btype(i), n->branch(i)});
         }
      }
      bool node_term(bt t, uint32_t p, bt& tt, uint32_t& tp) const
      {
         engine& e = const_cast<engine&>(_e);
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(resolve(e, t, p));
            if (!n->has_term()) return false; tt = bt(n->term_tag); tp = n->term_ptr; return true;
         }
         const setlist* n = static_cast<const setlist*>(resolve(e, t, p));
         if (!n->has_term()) return false; tt = n->term_type(); tp = n->term_branch(); return true;
      }
      // O(1) child lookup for a byte (setlist: SWAR find_index; full: direct).
      bool node_child(bt t, uint32_t p, uint8_t byte, bt& ct, uint32_t& cp) const
      {
         engine& e = const_cast<engine&>(_e);
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(resolve(e, t, p));
            ct = n->btype(byte); if (ct == bt::null) return false; cp = n->branch(byte); return true;
         }
         const setlist* n = static_cast<const setlist*>(resolve(e, t, p));
         const int idx = n->find_index(byte); if (idx < 0) return false;
         ct = n->btype(uint32_t(idx)); cp = n->branch(uint32_t(idx)); return true;
      }
      sv node_prefix(bt t, uint32_t p) const
      {
         engine& e = const_cast<engine&>(_e);
         return t == bt::full ? static_cast<const full*>(resolve(e, t, p))->prefix()
                              : static_cast<const setlist*>(resolve(e, t, p))->prefix();
      }
      void free_node(bt t, uint32_t p)
      {
         if (t == bt::full) _e.cl9.free(p);
         else if (t == bt::double_setlist || t == bt::big_nested) _e.cl2.free(p);
         else _e.cl1.free(p);
      }

      // Build a routing node from a branch list (sorted by byte) + optional term
      // + prefix; pick cl1/cl2/full by size. Branch ptrs are TAKEN (not retained).
      std::pair<bt, uint32_t> build_node(sv prefix, const br* brs, uint32_t nb,
                                         bool term, bt tt, uint32_t tp)
      {
         // does it fit a setlist (cl1 then cl2)?
         if (setlist::bytes_for(nb, uint32_t(prefix.size()), term) <= 128)
            return {bt::setlist, pack_setlist<128>(_s1, bt::setlist, prefix, brs, nb, term, tt, tp)};
         if (setlist::bytes_for(nb, uint32_t(prefix.size()), term) <= 256)
            return {bt::double_setlist, pack_setlist<256>(_s2, bt::double_setlist, prefix, brs, nb, term, tt, tp)};
         return {bt::full, pack_full(prefix, brs, nb, term, tt, tp)};
      }

      template <uint32_t SB, class Ses>
      uint32_t pack_setlist(Ses& ses, bt /*kind*/, sv prefix, const br* brs, uint32_t nb,
                            bool term, bt tt, uint32_t tp)
      {
         const uint32_t slot = ses.alloc();
         auto* n = static_cast<setlist*>(SB == 128 ? _e.cl1.get(slot) : _e.cl2.get(slot));
         n->meta    = uint8_t((term ? 0x80 : 0) | (prefix.size() & 0x7f));
         n->nbranch = uint8_t(nb);
         std::memcpy(n->prefix_ptr(), prefix.data(), prefix.size());
         for (uint32_t i = 0; i < nb; ++i) n->keys()[i] = brs[i].byte;
         std::memset(n->types(), 0, tag3_bytes(nb + (term ? 1u : 0u)));
         for (uint32_t i = 0; i < nb; ++i) { tag3_set(n->types(), i, brs[i].tag); n->branches()[i] = brs[i].ptr; }
         if (term) { tag3_set(n->types(), nb, tt); n->branches()[nb] = tp; }
         return slot;
      }
      uint32_t pack_full(sv prefix, const br* brs, uint32_t nb, bool term, bt tt, uint32_t tp)
      {
         const uint32_t slot = _s9.alloc();
         auto* n = static_cast<full*>(_e.cl9.get(slot));
         n->meta = uint8_t((term ? 0x80 : 0) | (prefix.size() & 0x7f));
         n->term_tag = uint8_t(term ? uint8_t(tt) : uint8_t(bt::null));
         n->nbranch  = uint16_t(nb);
         n->term_ptr = term ? tp : 0;
         std::memset(n->types(), 0xFF, 96);               // 0xFF bytes => every 3-bit tag = 7 = null
         std::memset(n->branches(), 0, 256 * 4);
         std::memcpy(n->prefix_ptr(), prefix.data(), prefix.size());
         for (uint32_t i = 0; i < nb; ++i) { tag3_set(n->types(), brs[i].byte, brs[i].tag); n->branches()[brs[i].byte] = brs[i].ptr; }
         return slot;
      }

      // COW edit of a routing node: read its branches into scratch, apply (insert
      // / replace one byte / set term), build a new node retaining the COPIED
      // children, release the old. Returns the new (tag, ptr).
      struct edit { enum K { ins, rep, set_term } k; uint8_t byte; bt tag; uint32_t ptr; };
      std::pair<bt, uint32_t> rebuild(bool is_unique, bt t, uint32_t p, edit ed)
      {
         br       scratch[257];
         uint32_t nb = 0;
         bool     term; bt tt = bt::null; uint32_t tp = 0;
         term = node_term(t, p, tt, tp);
         read_branches(t, p, [&](const br& b) { scratch[nb++] = b; });
         // apply edit (keep keys sorted)
         bt new_tt = tt; uint32_t new_tp = tp; bool new_term = term;
         if (ed.k == edit::set_term) { new_term = true; new_tt = ed.tag; new_tp = ed.ptr; }
         else if (ed.k == edit::rep)
         { for (uint32_t i = 0; i < nb; ++i) if (scratch[i].byte == ed.byte) { scratch[i].tag = ed.tag; scratch[i].ptr = ed.ptr; break; } }
         else  // ins (byte absent) — insert sorted
         {
            uint32_t pos = 0; while (pos < nb && scratch[pos].byte < ed.byte) ++pos;
            for (uint32_t i = nb; i > pos; --i) scratch[i] = scratch[i - 1];
            scratch[pos] = br{ed.byte, ed.tag, ed.ptr}; ++nb;
         }
         auto out = build_node(node_prefix(t, p), scratch, nb, new_term, new_tt, new_tp);
         if (is_unique)
         {
            // PRIVATE: every child/term is transferred to `out` (no snapshot holds
            // the old node) — just free the old husk slot. No O(n) retain/release.
            free_node(t, p);
            return out;
         }
         // SHARED (a snapshot pins the old node): retain the COPIED children +
         // carried-over term (now referenced by both old and out), then release
         // the old node (snapshot keeps its copy). The edited/new branch came with
         // ref 1 (owned), so it is NOT retained here.
         read_branches(out.first, out.second, [&](const br& b)
         {
            const bool is_new = ((ed.k == edit::ins || ed.k == edit::rep) && b.byte == ed.byte);
            if (!is_new) retain(_e, b.tag, b.ptr);
         });
         if (new_term)
         {
            bt ot; uint32_t op2;
            const bool had = node_term(t, p, ot, op2);
            const bool term_is_new = (ed.k == edit::set_term) || !had;
            if (!term_is_new) retain(_e, new_tt, new_tp);  // carried over unchanged
         }
         release_ref(t, p);   // drops the old node (+ its children if it hits 0)
         return out;
      }

      template <cow_mode M>
      std::pair<bt, uint32_t> upsert_rec(bt t, uint32_t p, sv key, uint32_t depth, sv val)
      {
         if (t == bt::null) { ++_size; return make_sv(key.substr(depth), val); }
         if (t == bt::suffix_value)
         {
            const sv esuf = sv_leaf::suffix(_e, p);
            const sv nsuf = key.substr(depth);
            if (esuf == nsuf) { auto nv = make_sv(nsuf, val); release_ref(bt::suffix_value, p); return nv; }
            // split two leaves into a setlist
            std::string es(esuf), ns(nsuf);                       // stable copies (esuf views the old SAL obj we free)
            std::string eval_s(sv_leaf::value(_e, p));            // existing value
            release_ref(bt::suffix_value, p);
            const uint32_t cp = lcp(es, ns);
            br       brs[2]; uint32_t nb = 0; bool term = false; bt tt = bt::null; uint32_t tp = 0;
            auto place = [&](const std::string& s, sv v)
            {
               if (cp == s.size()) { term = true; auto t2 = make_sv(sv{}, v); tt = t2.first; tp = t2.second; }
               else { auto t2 = make_sv(sv(s).substr(cp + 1), v); brs[nb++] = br{uint8_t(s[cp]), t2.first, t2.second}; }
            };
            place(es, eval_s); place(ns, val);
            if (nb == 2 && brs[0].byte > brs[1].byte) std::swap(brs[0], brs[1]);
            ++_size;
            return build_node(sv(es).substr(0, cp), brs, nb, term, tt, tp);
         }
         // routing node
         const sv pfx = node_prefix(t, p);
         const sv ksuf = key.substr(depth);
         const uint32_t cp = lcp(pfx, ksuf);
         if (cp < pfx.size())
         {
            // prefix split: new parent over [reprefixed old node] + [new key]
            std::string common(pfx.substr(0, cp));
            const uint8_t curbyte = uint8_t(pfx[cp]);
            auto cur2 = reprefix(t, p, cp + 1);   // old node with prefix pfx[cp+1:]
            br brs[2]; uint32_t nb = 0; bool term = false; bt tt = bt::null; uint32_t tp = 0;
            brs[nb++] = br{curbyte, cur2.first, cur2.second};
            if (cp == ksuf.size()) { term = true; auto nv = make_sv(sv{}, val); tt = nv.first; tp = nv.second; }
            else { auto nv = make_sv(ksuf.substr(cp + 1), val); brs[nb++] = br{uint8_t(ksuf[cp]), nv.first, nv.second}; }
            if (nb == 2 && brs[0].byte > brs[1].byte) std::swap(brs[0], brs[1]);
            ++_size;
            return build_node(common, brs, nb, term, tt, tp);
         }
         depth += uint32_t(pfx.size());
         // The latch, compile-time: in <unique> read the refcount once — a private
         // node mutates in place (children stay <unique>); a shared one clones and
         // its children recurse <shared>. In <shared> there is NO refcount read at
         // all — the instantiation is the clone path only.
         if constexpr (M == cow_mode::unique)
            if (unique(_e, t, p))
               return edit_node<cow_mode::unique>(t, p, key, depth, val);
         return edit_node<cow_mode::shared>(t, p, key, depth, val);
      }

      // Mutate a routing node for the key. CM == unique ⟹ the node is private,
      // mutate (t,p) in place and recurse children <unique>. CM == shared ⟹ the
      // node is snapshot-pinned, clone it (now private), recurse children <shared>.
      // Either way the working node W is private, so all edits free-husk / in-place.
      template <cow_mode CM>
      std::pair<bt, uint32_t> edit_node(bt t, uint32_t p, sv key, uint32_t depth, sv val)
      {
         std::pair<bt, uint32_t> W;
         if constexpr (CM == cow_mode::unique) W = {t, p};
         else W = reprefix(t, p, 0);            // clone the shared node (retain children, drop one old ref)
         const bt wt = W.first; const uint32_t wp = W.second;

         if (depth == key.size())               // term
         {
            bt tt; uint32_t tp;
            if (node_term(wt, wp, tt, tp))       // REPLACE in place
            { release_ref(tt, tp); set_term_inplace(wt, wp, make_sv(sv{}, val)); return W; }
            auto nv = make_sv(sv{}, val); ++_size;   // ADD a term → grow → rebuild (W private)
            return rebuild(true, wt, wp, edit{edit::set_term, 0, nv.first, nv.second});
         }
         const uint8_t byte = uint8_t(key[depth]);
         bt ct; uint32_t cpp;
         if (!node_child(wt, wp, byte, ct, cpp))   // INS: byte absent
         {
            auto nv = make_sv(key.substr(depth + 1), val); ++_size;
            if (wt == bt::full)                   // O(1) direct-index insert (W is private)
            {
               full* n = static_cast<full*>(resolve(_e, wt, wp));
               tag3_set(n->types(), byte, nv.first); n->branches()[byte] = nv.second; ++n->nbranch;
               return W;
            }
            return rebuild(true, wt, wp, edit{edit::ins, byte, nv.first, nv.second});
         }
         // REP: recurse the byte-child in the latched mode, overwrite the slot.
         auto nc = upsert_rec<CM>(ct, cpp, key, depth + 1, val);
         set_branch_inplace(wt, wp, byte, nc);
         return W;
      }

      void set_term_inplace(bt t, uint32_t p, std::pair<bt, uint32_t> nv)
      {
         if (t == bt::full)
         { auto* n = static_cast<full*>(resolve(_e, t, p)); n->term_tag = uint8_t(nv.first); n->term_ptr = nv.second; }
         else
         { auto* n = static_cast<setlist*>(resolve(_e, t, p)); tag3_set(n->types(), n->nbranch, nv.first); n->branches()[n->nbranch] = nv.second; }
      }
      void set_branch_inplace(bt t, uint32_t p, uint8_t byte, std::pair<bt, uint32_t> nc)
      {
         if (t == bt::full)
         {
            auto* n = static_cast<full*>(resolve(_e, t, p));
            tag3_set(n->types(), byte, nc.first); n->branches()[byte] = nc.second;
         }
         else
         {
            auto* n = static_cast<setlist*>(resolve(_e, t, p));
            const int idx = n->find_index(byte);
            tag3_set(n->types(), uint32_t(idx), nc.first); n->branches()[uint32_t(idx)] = nc.second;
         }
      }

      // Reprefix a routing node to prefix[drop:] (build a copy, retain children,
      // release old). Returns the new (tag, ptr).
      std::pair<bt, uint32_t> reprefix(bt t, uint32_t p, uint32_t drop)
      {
         br scratch[257]; uint32_t nb = 0; bt tt = bt::null; uint32_t tp = 0;
         const bool term = node_term(t, p, tt, tp);
         read_branches(t, p, [&](const br& b) { scratch[nb++] = b; });
         std::string np(node_prefix(t, p).substr(drop));
         auto out = build_node(np, scratch, nb, term, tt, tp);
         read_branches(out.first, out.second, [&](const br& b) { retain(_e, b.tag, b.ptr); });
         if (term) retain(_e, tt, tp);
         release_ref(t, p);
         return out;
      }

      engine&                              _e;
      sal2::fixed_allocator<128>::session  _s1;
      sal2::fixed_allocator<256>::session  _s2;
      sal2::fixed_allocator<1152>::session _s9;
      sal2::var_allocator<32>::session     _ssal;
      bt       _rt = bt::null;
      uint32_t _rp = 0;
      uint64_t _size = 0;
   };
}  // namespace gcart::st

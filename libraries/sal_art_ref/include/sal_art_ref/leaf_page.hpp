#pragma once
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>

#include <sal_art_ref/env_config.hpp>  // single-init HEART_PAGE_BYTES (no scattered getenv)
#include <sal_art_ref/inline_value.hpp>
#include <sal_art_ref/paged.hpp>
#include <sal_art_ref/radix_type_ids.hpp>
#include <sal_art_ref/smart_address.hpp>  // tail_ptr
#include <sal_art_ref/value_hooks.hpp>    // typed-value retain/release registry
#include <ucc/lower_bound.hpp>      // lower_bound_u16_padded, find_byte, find_u32x16
#include <ucc/round.hpp>

namespace sal_art_ref
{
   /**
    * The 2KB COW leaf page — the sole terminal tier of the page-leaf design. 2KB is
    * the COW minimum unit: shared-mode write throughput craters if the page is much
    * larger (every shared write clones a whole page). Splits emit SHORT dividers
    * (shortest-near-median, power-of-2 snapped last byte) that become the lower-bound
    * radix above; the page must be reached even on a miss (miss perf is the design's
    * sacrifice), so point-search advantage comes entirely from the fingerprint column.
    *
    * Layout (SoA; columns at fixed offsets given _cap, capacities a multiple of 4 so
    * unique-mode inserts usually shift within slack instead of rebuilding):
    *
    *   [ header | cbases[16] | fp[cap] u8 | pfx[cap] u16 | vals[cap] u16 | free | tail heap ]
    *
    *   fp[i]   = 1-byte fingerprint of the FULL key (stable across reprefix/split —
    *             copied verbatim by every rebuild). Point lookups SIMD-scan this.
    *   pfx[i]  = big-endian first 2 bytes of entry i's stored suffix (zero-padded).
    *             Sorted; lower_bound is one SIMD scan + a short equal-run compare.
    *   vals[i] = tail_ptr offset of entry i's inline_value record in the tail heap.
    *
    * Every record keeps its SUFFIX in-page (compares never dereference). The record's
    * value payload is tagged by inline_value::value_type:
    *   vt_bytes   — value bytes inline (suffix+value <= max_inline_entry).
    *   vt_typed   — a typed value inline: [u16 typenum][token], same size rule as vt_bytes.
    *   vt_subtree — 4-byte nested-tree root.
    *   vt_extern  — suffix+value > max_inline_entry: the value lives in an external
    *                SAL object; payload is a 2-byte code 0xFF00 | cbase<<4 | slot.
    *                cbases[] holds <=16 cacheline base addresses, so cloning retains
    *                children on at most 16 control-block cachelines. When a 17th base
    *                would be needed the page reports FULL and the caller splits early.
    *                (The code range is the top 256 values of the u16 space — a tail
    *                offset in a 2KB page can never reach it.)
    *
    * A common prefix of all keys is factored out ONLY by the rebuild ctors (split /
    * clone — never by in-place inserts): full key = common_prefix() ++ suffix(i).
    * An insert whose key diverges from the common prefix forces a rebuilding clone
    * that re-expands suffixes (psitri's leaf_prepend_prefix analogue).
    *
    * COW allocs the full 2KB (room to grow, one grow per txn chain); the compactor
    * shrinks via compact_size()/compact_to(), which REBUILDS — dropping dead tail
    * bytes, excess capacity, and dead cbases. Same lifecycle as bucket_node/psitri.
    */
   /// Type-erased fetch of a key_val_node's FULL KEY by address — threaded into page
   /// compares that may need to extend an extern record's fragment. Null is valid when
   /// no big items can be involved (asserts if a resolve is ever needed).
   struct kv_resolver
   {
      const void* ctx                                               = nullptr;
      std::span<const uint8_t> (*fn)(const void*, sal::ptr_address) = nullptr;
      std::span<const uint8_t> operator()(sal::ptr_address a) const
      {
         assert(fn && "extern-fragment compare needs a kv_resolver");
         return fn(ctx, a);
      }
      explicit operator bool() const noexcept { return fn != nullptr; }
   };

   struct leaf_page : paged
   {
      static constexpr sal::header_type type_id = radix_type_ids::leaf_page;
      /// COW-unit page size (default 2KB; the design's chosen point). HEART_PAGE_BYTES
      /// overrides it (512..8192) for size experiments — read once, constant per process.
      /// u16 tail offsets and the 0xFF00 extern-code floor cap it well below 64KB.
      static uint32_t page_bytes() noexcept { return env_config::get().page_bytes; }
      static constexpr uint32_t max_cbases       = 16;
      static constexpr int      max_inline_entry = 128;  // suffix+value > this => key_val_node
      static constexpr uint16_t ext_code_base    = 0xFF00;
      static constexpr uint16_t max_suffix_len   = 255;  // inline_value u8 field (inline records)
      // A full-key external record (vt_extern) keeps a comparison FRAGMENT of the suffix,
      // capped so the record is at most 128 bytes: 3 (header) + fragment + 2 (cbase code).
      // A value-only external record (vt_extern_val) keeps its whole suffix, up to
      // max_suffix_len, so it can reach 3 + 255 + 2 = 260 bytes: the largest page record.
      static constexpr uint16_t ext_frag_cap = 123;
      // The key limit (key_val_node::max_key_len; radix_engine.hpp asserts they agree). A
      // page's common prefix and divider together never take more tail than one key.
      static constexpr uint32_t max_key_bytes = 1024;

      uint16_t         _nkeys;
      uint16_t         _cap;        // column capacity, always a multiple of 4
      uint8_t          _ncb : 5;    // used cbases (<= 16)
      uint8_t          _pfxw : 3;   // prefix-column width (1, 2 or 4 bytes per entry)
      uint8_t          _rsv0;       // (was the u8 _cpfx_len — see below)
      uint16_t         _dead;       // dead tail-heap bytes (orphaned records)
      // u16: the key contract is <=1024 bytes and a deep-nested key set
      // legitimately factors a >255-byte common prefix; the old u8 WRAPPED and
      // silently lost every key whose length crossed 256 (contract test 4.7a)
      uint16_t         _cpfx_len;   // factored common prefix length
      tail_ptr<>       _cpfx_pos;   // common prefix bytes in the tail heap (0 = none)
      // The page DIVIDER: this page's lower-bound separator (the routing key the radix uses to
      // reach it). Set ONCE at creation (split right-half = the split divider; left-half / leftmost
      // = inherited / empty), copied verbatim by every rebuild, never touched by in-place ops.
      // Lets a page attach DIRECTLY under a branch byte (no single-child setlist): route_to_page
      // re-screens key[depth..] against _div[depth..] on arrival. Mirrors _cpfx_len/_cpfx_pos.
      uint16_t         _div_len;    // page divider (lower-bound separator) length
      tail_ptr<>       _div_pos;    // divider bytes in the tail heap (0 = none / leftmost)
      // CBASES are VARIABLE: only ~0.9% of Eth keys are external, so ~99% of pages have _ncb==0 and
      // need NO cbase storage. The 16-slot (64B) base-address block lives in the TAIL, allocated
      // lazily on the first external (0 = none). find_u32x16 reads a full 64B regardless of _ncb, so
      // the block is always the full 16 slots (0xFFFFFFFF-filled) — no over-read past it.
      tail_ptr<>       _cbases_pos;  // 16-slot ptr_address block in the tail (0 = no externs)
      // CACHELINE INDEX (the wtree lesson): the FIRST stem of every (stride-th) prefix-column
      // cacheline, sampled HERE in the header — resident from reading the fields above, so the
      // probe is free — letting lower_bound scan ONE column cacheline instead of all of them.
      // Width-W samples, unused slots 0xFF-filled (compare >= every real stem). Rebuilt after
      // any column splice (the touched cachelines are still hot).
      uint8_t _clidx[64];

      // ── op tickets ───────────────────────────────────────────────────────────────
      struct insert_op
      {
         std::span<const uint8_t> key;    // FULL key as the page sees it
         std::span<const uint8_t> value;  // vt_bytes payload / vt_subtree 4-byte root
         uint8_t                  vtype = inline_value::vt_bytes;
         sal::ptr_address         ext   = sal::null_ptr_address;  // vt_extern key_val_node
         kv_resolver              res{};  // fragment extension for position compares
      };
      struct update_op  // value rewrite at an existing index, key unchanged
      {
         uint16_t                 index;
         std::span<const uint8_t> value;
         uint8_t                  vtype = inline_value::vt_bytes;
         sal::ptr_address         ext   = sal::null_ptr_address;
      };
      struct remove_op
      {
         uint16_t index;
      };
      // verbatim-clone ticket: 2 SIMD copies (head + used tail), keeping src's layout/capacity/
      // dead bytes. The fast COW path: blit, then apply() the one mutation in place — never the
      // per-record rebuild — whenever can_apply() says the (copied) layout has room.
      struct clone_op
      {
         const leaf_page* src;
      };
      // rebuild-clone tickets (shared mode / reprefix / shrinking remove)
      struct clone_insert_op
      {
         const leaf_page* src;
         insert_op        ins;
      };
      struct clone_update_op
      {
         const leaf_page* src;
         update_op        upd;
         kv_resolver      res{};
      };
      struct clone_remove_op
      {
         const leaf_page* src;
         uint16_t         index;
         kv_resolver      res{};
      };
      struct split_op  // one half of a split: src entries [lo,hi)
      {
         const leaf_page*         src;
         uint16_t                 lo, hi;
         kv_resolver              res{};
         std::span<const uint8_t> div{};  // this half's divider: left = src's, right = split divider
         // The half's common prefix length and the widest prefix column it may pick, as
         // chosen by plan_split (which checked the half fits a page with them).
         uint16_t ncp   = 0;
         uint8_t  max_w = 4;
      };
      // An empty page reached through `div`: the page a key gets to itself when no page
      // that already holds keys can take it (see radix_tree::page_edge_split).
      struct empty_op
      {
         std::span<const uint8_t> div{};
      };
      // The same entries under a different divider (the page's lower bound moved up because
      // a new page was put in front of it).
      struct redivide_op
      {
         const leaf_page*         src;
         std::span<const uint8_t> div{};
      };

      // ── SAL lifecycle ────────────────────────────────────────────────────────────
      static constexpr uint32_t head_bytes() noexcept { return uint32_t(sizeof(leaf_page)); }
      uint32_t head_size() const noexcept { return head_bytes() + (3u + _pfxw) * _cap; }
      static uint32_t alloc_size() noexcept { return page_bytes(); }
      static uint32_t alloc_size(const clone_op&) noexcept { return page_bytes(); }
      static uint32_t alloc_size(const clone_insert_op&) noexcept { return page_bytes(); }
      static uint32_t alloc_size(const clone_update_op&) noexcept { return page_bytes(); }
      static uint32_t alloc_size(const clone_remove_op&) noexcept { return page_bytes(); }
      static uint32_t alloc_size(const split_op&) noexcept { return page_bytes(); }
      static uint32_t alloc_size(const empty_op&) noexcept { return page_bytes(); }
      static uint32_t alloc_size(const redivide_op&) noexcept { return page_bytes(); }
      uint32_t        cow_size() const noexcept { return size(); }

      /// A page about to be built or changed beyond what it can hold, or from inputs that
      /// cannot be right. Callers decide fit beforehand (can_apply, rebuilt_fits, plan_split);
      /// arriving here means one of them is wrong. Stopping is the only safe answer: carrying
      /// on would write over the page's own columns or the next object.
      [[noreturn]] static void page_fault(const char* what, uint32_t detail) noexcept
      {
         std::fprintf(stderr, "sal_art_ref leaf_page: %s (%u); stopping\n", what, detail);
         std::abort();
      }
      /// Every tail allocation of a page is checked against the page's size in every build.
      char* alloc_tail(uint16_t n) noexcept
      {
         if (head_size() + uint32_t(tail_used()) + n > size()) [[unlikely]]
            page_fault("content does not fit the page", n);
         return paged::alloc_tail(n);
      }

      // ── the common prefix and the divider share their bytes ──────────────────────
      // Both are prefixes of keys near this page, and after a split one is usually a prefix
      // of the other (the divider is the keys' shared bytes plus one). When that holds the
      // longer one is stored once and both views start at it, so a long shared prefix costs
      // its length once, not twice. The rule is a pure function of the two byte strings, so
      // every rebuild of the same content makes the same choice.
      static bool meta_aliases(const uint8_t* cp, uint32_t ncp, const uint8_t* dv,
                               uint32_t dn) noexcept
      {
         return ncp && dn && std::memcmp(cp, dv, ncp < dn ? ncp : dn) == 0;
      }
      static uint32_t meta_bytes(const uint8_t* cp, uint32_t ncp, const uint8_t* dv,
                                 uint32_t dn) noexcept
      {
         return meta_aliases(cp, ncp, dv, dn) ? (ncp > dn ? ncp : dn) : ncp + dn;
      }
      /// Tail bytes this page spends on its common prefix and divider.
      uint32_t meta_live() const noexcept
      {
         if (_cpfx_len && _div_len && _cpfx_pos.offset_from_tail == _div_pos.offset_from_tail)
            return _cpfx_len > _div_len ? _cpfx_len : _div_len;
         return uint32_t(_cpfx_len) + _div_len;
      }
      static uint16_t round_cap(uint32_t n) noexcept
      {
         return uint16_t(ucc::round_up_multiple<4>(n < 8 ? 8u : n));
      }
      uint32_t compact_size() const noexcept
      {
         // must PREDICT the same prefix width compact_to's rebuild will choose: a compact
         // rebuild is transform-free and patch-free, so this walk computes the identical
         // (min suffix, max u16-stem run) inputs.
         uint32_t min_sfx = UINT32_MAX, run = 1, max_run = 0, prev = 0;
         for (uint32_t i = 0; i < _nkeys; ++i)
         {
            const inline_value* iv = record(i);
            if (iv->suffix_len < min_sfx)
               min_sfx = iv->suffix_len;
            const uint32_t s2 = stemN(iv->data, iv->suffix_len, 2);
            if (i && s2 == prev)
            {
               if (++run > max_run)
                  max_run = run;
            }
            else
               run = 1;
            prev = s2;
         }
         // CLAMP to the current width: compaction must only ever SHRINK a page (a wider pick
         // could exceed the page's birth size and overrun the next blit-clone's allocation).
         uint32_t w = width_from(min_sfx == UINT32_MAX ? 2u : min_sfx, max_run);
         if (w > _pfxw)
            w = _pfxw;
         return ucc::round_up_multiple<sal::cacheline_size>(
             head_bytes() + (3u + w) * round_cap(_nkeys) + live_tail());
      }
      void compact_to(sal::alloc_header* dst) const noexcept
      {
         // compaction never changes cpfx (prepend == 0): fragments copy verbatim, already
         // maximal — no resolver needed (SAL's passive compactor has no session).
         static_cast<leaf_page*>(dst)->init_from(this, 0, _nkeys, _cpfx_len, nullptr, 0, nullptr,
                                                 nullptr, /*drop=*/-1, /*max_w=*/_pfxw,
                                                 kv_resolver{}, {div_data(), _div_len});
      }
      void copy_to(sal::alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }

      /// SAL destroy hook: a dying page releases its children (external value objects
      /// and nested subtree roots) — the cascade that frees what only this page owned.
      /// Inline typed values (vt_typed) fire the type's release hook here — this is
      /// the page's physical-death point. Plain bytes (vt_bytes) are never inspected.
      template <typename Session>
      void destroy(const Session& s) const
      {
         for_each_child([&](sal::ptr_address c) { s->release(c); });
         for (uint32_t i = 0; i < _nkeys; ++i)
         {
            const inline_value* iv = record(i);
            if (iv->value_type == inline_value::vt_typed) [[unlikely]]
               fire_typed_release({reinterpret_cast<const char*>(iv->value().data()),
                                   iv->value().size()});
         }
      }
      /// Visit the stored bytes ([u16 typenum][token]) of every inline typed value: the
      /// values a copy of this page duplicates.
      template <typename F>
      void for_each_inline_typed(F&& f) const
      {
         for (uint32_t i = 0; i < _nkeys; ++i)
         {
            const inline_value* iv = record(i);
            if (iv->value_type == inline_value::vt_typed) [[unlikely]]
               f(i, std::string_view{reinterpret_cast<const char*>(iv->value().data()),
                                     iv->value().size()});
         }
      }
      // SAL reachability/refcount-rebuild hook — same external-value + subtree-root set destroy()
      // releases (default alloc_header::visit_children is empty → these would look unreachable).
      void visit_children(const std::function<void(sal::ptr_address)>& v) const
      {
         for_each_child([&](sal::ptr_address c) { v(c); });
      }

      /// Visit SAL children (external values + nested subtree roots) — the release/
      /// retain cascade and ref audits walk this.
      template <typename F>
      void for_each_child(F&& f) const
      {
         for (uint32_t i = 0; i < _nkeys; ++i)
         {
            const inline_value* iv = record(i);
            if (inline_value::external_value(iv->value_type))
               f(decode_extern(iv));
            else if (iv->is_subtree())
            {
               if (iv->subtree_root() != sal::null_ptr_address)
                  f(iv->subtree_root());
            }
         }
      }

      // ── construction ─────────────────────────────────────────────────────────────
      leaf_page(uint32_t asize, sal::ptr_address_seq seq) noexcept
      {
         this->init_page(asize, type_id, seq);
         init_empty();
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const clone_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         assert(asize >= op.src->size());
         op.src->blit_head_tail(this, op.src->head_size());  // verbatim: layout/cap/dead kept
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const clone_insert_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         const leaf_page* s = op.src;
         // factoring only ever LOOSENS here: cpfx may shrink to admit a diverging key
         const uint16_t ncp = uint16_t(
             shared_prefix(op.ins.key.data(), op.ins.key.size(), s->cpfx_data(), s->_cpfx_len));
         init_from(s, 0, s->_nkeys, ncp, nullptr, 0, &op.ins, nullptr, -1, 4,
                   op.ins.res, {s->div_data(), s->_div_len});  // divider unchanged by insert
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const clone_update_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         init_from(op.src, 0, op.src->_nkeys, op.src->_cpfx_len, nullptr, 0, nullptr, &op.upd,
                   -1, 4, op.res, {op.src->div_data(), op.src->_div_len});  // divider unchanged
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const clone_remove_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         init_from(op.src, 0, op.src->_nkeys, op.src->_cpfx_len, nullptr, 0, nullptr, nullptr,
                   int(op.index), 4, op.res, {op.src->div_data(), op.src->_div_len});  // unchanged
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const split_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         const leaf_page* s = op.src;
         assert(op.lo < op.hi && op.hi <= s->_nkeys);
         // Splits are the only place factoring tightens. How far is plan_split's decision
         // (at most the half's natural common prefix; less when that would not fit).
         if (op.ncp < s->_cpfx_len || op.ncp > s->natural_cpfx(op.lo, op.hi)) [[unlikely]]
            page_fault("split with a common prefix the half does not have", op.ncp);
         init_from(s, op.lo, op.hi, op.ncp, s->suffix(op.lo).data(),
                   uint32_t(op.ncp - s->_cpfx_len), nullptr, nullptr, -1, op.max_w, op.res,
                   op.div);
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const empty_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         init_empty();
         _div_len = uint16_t(op.div.size());
         if (!op.div.empty())
         {
            std::memcpy(alloc_tail(uint16_t(op.div.size())), op.div.data(), op.div.size());
            _div_pos = tail_pos();
         }
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const redivide_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         // Same prefix, same records, same width: only the divider differs, and the caller
         // checked redivided_fits().
         init_from(op.src, 0, op.src->_nkeys, op.src->_cpfx_len, nullptr, 0, nullptr, nullptr,
                   -1, op.src->_pfxw, kv_resolver{}, op.div);
      }

      // ── accessors ────────────────────────────────────────────────────────────────
      uint16_t num_keys() const noexcept { return _nkeys; }
      uint16_t capacity() const noexcept { return _cap; }
      uint8_t  num_cbases() const noexcept { return _ncb; }
      uint8_t  prefix_width() const noexcept { return _pfxw; }
      std::span<const uint8_t> common_prefix() const noexcept
      {
         return {cpfx_data(), _cpfx_len};
      }
      std::span<const uint8_t> divider() const noexcept { return {div_data(), _div_len}; }
      std::span<const uint8_t> suffix(uint32_t i) const noexcept { return record(i)->suffix(); }
      uint8_t                  value_type(uint32_t i) const noexcept
      {
         return record(i)->value_type;
      }
      /// Inline payload: the caller's bytes (vt_bytes) or a typed value's stored
      /// [typenum][token] (vt_typed). Empty span for other types — deref the external instead.
      std::span<const uint8_t> value(uint32_t i) const noexcept
      {
         const inline_value* iv = record(i);
         return inline_value::inline_payload(iv->value_type) ? iv->value()
                                                             : std::span<const uint8_t>{};
      }
      bool is_external(uint32_t i) const noexcept
      {
         return inline_value::external_value(record(i)->value_type);
      }
      sal::ptr_address external_addr(uint32_t i) const noexcept
      {
         return decode_extern(record(i));
      }
      sal::ptr_address subtree_root(uint32_t i) const noexcept { return record(i)->subtree_root(); }
      const inline_value* record(uint32_t i) const noexcept
      {
         assert(i < _nkeys);
         return resolve(tail_ptr<inline_value>{vals()[i]});
      }

      /// Reconstruct entry i's full key into `out` (sized >= 255 + cpfx); returns length.
      uint32_t full_key(uint32_t i, uint8_t* out) const noexcept
      {
         std::memcpy(out, cpfx_data(), _cpfx_len);
         auto s = suffix(i);
         std::memcpy(out + _cpfx_len, s.data(), s.size());
         return _cpfx_len + uint32_t(s.size());
      }

      /// Fingerprint of a FULL key — a word-at-a-time multiply-mix folded to one byte
      /// (~2 multiplies for an 8-byte key vs 8 serial FNV rounds). Full-key (not suffix)
      /// so the column survives reprefix and split verbatim.
      static uint8_t fingerprint_of(const uint8_t* k, size_t n) noexcept
      {
         uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t(n) * 0xff51afd7ed558ccdull);
         while (n >= 8)
         {
            uint64_t w;
            std::memcpy(&w, k, 8);
            h = (h ^ w) * 0x9E3779B97F4A7C15ull;
            h ^= h >> 29;
            k += 8;
            n -= 8;
         }
         if (n)
         {
            uint64_t w = 0;
            std::memcpy(&w, k, n);
            h = (h ^ w) * 0x9E3779B97F4A7C15ull;
            h ^= h >> 29;
         }
         return uint8_t(h >> 56);
      }

      // ── point find: fingerprint column scan, pfx-column pre-screen, verify by suffix ─────
      /// @return entry index, or -1 if absent. TOUCH ORDER is the design: the fp scan and the
      /// u16 prefix pre-screen read only the (adjacent) columns; the tail heap — the far
      /// cachelines — is touched only for a candidate that survives both, and the page cpfx
      /// is compared exactly once, inside that verify (a fingerprint-less miss never touches
      /// the tail at all).
      int find(const uint8_t* k, uint32_t n, const kv_resolver& R = {}) const noexcept
      {
         if (n < _cpfx_len)
            return -1;
         const uint8_t* s   = k + _cpfx_len;
         const uint32_t sn  = n - _cpfx_len;
         const uint8_t  f   = fingerprint_of(k, n);
         const uint32_t q2  = stem(s, sn);
         const uint8_t* fpc = fp();
         uint32_t       i   = 0;
         while (i < _nkeys)
         {
            const int rel = ucc::find_byte(fpc + i, _nkeys - i, f);
            if (uint32_t(rel) == _nkeys - i)
               return -1;
            i += uint32_t(rel);
            if (pfx_get(i) == q2)  // column pre-screen: kills fp collisions without a tail touch
            {
               const inline_value* iv = record(i);
               if (iv->value_type == inline_value::vt_extern)
               {
                  // the fragment must prefix-match before paying the dereference; the
                  // key_val_node stores the FULL key, so the final compare covers cpfx too
                  const uint32_t fl = iv->suffix_len;
                  if (fl <= sn && (fl == 0 || std::memcmp(iv->data, s, fl) == 0))
                  {
                     auto kk = R(decode_extern(iv));
                     if (kk.size() == n && (n == 0 || std::memcmp(kk.data(), k, n) == 0))
                        return int(i);
                  }
               }
               else if (iv->suffix_len == sn && (sn == 0 || std::memcmp(iv->data, s, sn) == 0))
               {
                  // suffix matched → the only possible entry; settle the cpfx part once
                  if (_cpfx_len && std::memcmp(k, cpfx_data(), _cpfx_len) != 0)
                     return -1;  // key diverges from the page prefix — nothing here matches
                  return int(i);
               }
            }
            ++i;
         }
         return -1;
      }

      // ── lower_bound: SIMD prefix-column scan + equal-run compare ─────────────────
      /// @return smallest index whose full key >= (k,n); _nkeys if none.
      int lower_bound(const uint8_t* k, uint32_t n, const kv_resolver& R = {}) const noexcept
      {
         if (_cpfx_len)
         {
            const uint32_t m = n < _cpfx_len ? n : _cpfx_len;
            const int      c = m ? std::memcmp(k, cpfx_data(), m) : 0;
            if (c < 0 || (c == 0 && n <= _cpfx_len))
               return 0;  // key <= cpfx => before every entry (or equal-to-prefix run start)
            if (c > 0)
               return _nkeys;
         }
         const uint8_t* s  = k + _cpfx_len;
         const uint32_t sn = n - _cpfx_len;
         int i = pfx_lower_bound(stem(s, sn));
         while (i < int(_nkeys) && cmp_suffix(uint32_t(i), s, sn, R) < 0)
            ++i;
         return i;
      }

      // ── in-place mutation (unique mode) ──────────────────────────────────────────
      enum class can_apply_mode
      {
         none,
         modify
      };

      can_apply_mode can_apply(const insert_op& ins) const noexcept
      {
         const uint32_t n = uint32_t(ins.key.size());
         if (n < _cpfx_len || (_cpfx_len && std::memcmp(ins.key.data(), cpfx_data(), _cpfx_len)))
            return can_apply_mode::none;  // diverges from cpfx => rebuilding clone
         if (ins.vtype != inline_value::vt_extern && n - _cpfx_len > max_suffix_len)
            return can_apply_mode::none;  // big items must arrive classified vt_extern
         if (inline_value::inline_payload(ins.vtype) &&
             int(n - _cpfx_len + ins.value.size()) > max_inline_entry)
            return can_apply_mode::none;  // engine should have externalized this entry
         if (inline_value::external_value(ins.vtype) && !can_add_address(ins.ext))
            return can_apply_mode::none;  // 17th cline => split early
         // first extern on a page with no cbase block → encode_extern lazily alloc_tail(64); reserve it
         const uint32_t cb_need = (inline_value::external_value(ins.vtype) &&
                                   _cbases_pos.offset_from_tail == 0) ? 64u : 0u;
         const uint32_t need = uint32_t(record_bytes(n - _cpfx_len, ins)) +
                               (_nkeys == _cap ? (3u + _pfxw) * 4u : 0u) + cb_need;
         return free_bytes(cols_end()) >= need ? can_apply_mode::modify : can_apply_mode::none;
      }

      /// @pre can_apply(ins) == modify, and the key is NOT present.
      /// @return the index the entry landed at.
      uint16_t apply(const insert_op& ins) noexcept
      {
         assert(can_apply(ins) == can_apply_mode::modify);
         assert(find(ins.key.data(), uint32_t(ins.key.size()), ins.res) < 0);
         if (_nkeys == _cap)
            grow_cap();
         const uint8_t* s   = ins.key.data() + _cpfx_len;
         const uint32_t sn  = uint32_t(ins.key.size()) - _cpfx_len;
         const uint16_t off = alloc_record(s, sn, ins);
         const uint16_t pos =
             uint16_t(lower_bound(ins.key.data(), uint32_t(ins.key.size()), ins.res));
         open_gap(pos);
         fp()[pos] = fingerprint_of(ins.key.data(), ins.key.size());
         pfx_set(pos, stem(s, sn));
         vals()[pos] = off;
         ++_nkeys;
         rebuild_clidx();  // the spliced column cachelines are still hot
         return pos;
      }

      can_apply_mode can_apply(const update_op& upd) const noexcept
      {
         assert(upd.index < _nkeys);
         const inline_value* iv = record(upd.index);
         if (inline_value::inline_payload(upd.vtype) &&
             int(iv->suffix_len + upd.value.size()) > max_inline_entry)
            return can_apply_mode::none;  // engine should have externalized this value
         if (inline_value::external_value(upd.vtype) && !can_add_address(upd.ext))
            return can_apply_mode::none;
         // first extern on a page with no cbase block → encode_extern lazily alloc_tail(64); reserve it
         const uint32_t cb_need = (inline_value::external_value(upd.vtype) &&
                                   _cbases_pos.offset_from_tail == 0) ? 64u : 0u;
         const uint32_t npay = payload_bytes(upd.value.size(), upd.vtype);
         if (npay <= iv->value_len)  // rewrite in the existing record (+ maybe the lazy cbase block)
            return free_bytes(cols_end()) >= cb_need ? can_apply_mode::modify : can_apply_mode::none;
         const uint32_t need = 3u + iv->suffix_len + npay + cb_need;
         return free_bytes(cols_end()) >= need ? can_apply_mode::modify : can_apply_mode::none;
      }

      /// @pre can_apply(upd) == modify
      void apply(const update_op& upd) noexcept
      {
         assert(can_apply(upd) == can_apply_mode::modify);
         inline_value*  iv   = const_cast<inline_value*>(record(upd.index));
         const uint32_t npay = payload_bytes(upd.value.size(), upd.vtype);
         if (npay <= iv->value_len)
         {
            _dead += uint16_t(iv->value_len - npay);
            iv->value_len  = uint8_t(npay);
            iv->value_type = upd.vtype;
            write_payload(iv->data + iv->suffix_len, upd);
            return;
         }
         _dead += uint16_t(iv->byte_size());
         auto* nv = reinterpret_cast<inline_value*>(
             alloc_tail(uint16_t(3u + iv->suffix_len + npay)));
         const uint16_t off = tail_used();  // BEFORE write_payload (which may alloc the cbase block)
         nv->suffix_len = iv->suffix_len;
         nv->value_len  = uint8_t(npay);
         nv->value_type = upd.vtype;
         std::memcpy(nv->data, iv->data, iv->suffix_len);
         write_payload(nv->data + nv->suffix_len, upd);
         vals()[upd.index] = off;
      }

      /// In-place remove (always possible). The record's tail bytes orphan (reclaimed
      /// by compact/rebuild); an external child must be released by the CALLER, which
      /// reads external_addr(index) before calling.
      void apply(const remove_op& rm) noexcept
      {
         assert(rm.index < _nkeys);
         _dead += uint16_t(record(rm.index)->byte_size());
         close_gap(rm.index);
         --_nkeys;
         rebuild_clidx();
      }

      // ── rebuild-fit checks (clone vs split decision) ─────────────────────────────
      bool rebuilt_fits(const insert_op& ins) const noexcept
      {
         const uint32_t n   = uint32_t(ins.key.size());
         const uint32_t ncp = shared_prefix(ins.key.data(), n, cpfx_data(), _cpfx_len);
         const uint32_t sn  = n - ncp;
         if (ins.vtype != inline_value::vt_extern && sn > max_suffix_len)
            return false;  // big items must arrive classified vt_extern
         // Re-expanding the suffixes kept whole in the page must stay within their u8
         // length. Full-key external records keep a re-capped fragment and an empty page
         // has nothing to re-expand, so neither constrains.
         if (const int mls = max_live_suffix();
             mls >= 0 && uint32_t(_cpfx_len - ncp) + uint32_t(mls) > max_suffix_len)
            return false;
         uint32_t cb = live_cbase_count();
         if (inline_value::external_value(ins.vtype))
            cb += 1;  // conservative: may need a fresh base
         if (cb > max_cbases)
            return false;
         // width upper bound: the rebuild picks pick(min suffix), and `ins` is in the set,
         // so its (post-ncp) suffix bounds the chosen width from above.
         const uint32_t wb      = pick_pfx_width(sn);
         // the rebuilt page keeps cpfx[0..ncp) and the divider, sharing bytes where it can
         const uint32_t new_meta = meta_bytes(cpfx_data(), ncp, div_data(), _div_len);
         const uint32_t content = head_bytes() + (3u + wb) * round_cap(_nkeys + 1u) + new_meta +
                                  (live_tail() - meta_live()) +
                                  uint32_t(_cpfx_len - ncp) * _nkeys + record_bytes(sn, ins) +
                                  // a FRESH 16-slot cbase block only if the rebuild gains externs and
                                  // this page has none yet (an existing block is already in live_tail)
                                  ((cb > 0 && _cbases_pos.offset_from_tail == 0) ? 64u : 0u);
         return content <= page_bytes();
      }

      bool rebuilt_fits(const update_op& upd) const noexcept
      {
         const inline_value* iv   = record(upd.index);
         const uint32_t      npay = payload_bytes(upd.value.size(), upd.vtype);
         if (inline_value::inline_payload(upd.vtype) &&
             int(iv->suffix_len + npay) > max_inline_entry)
            return false;
         if (inline_value::external_value(upd.vtype) && live_cbase_count() + 1 > max_cbases)
            return false;
         // width upper bound for the rebuild (cold path: one walk) — the run rule can widen
         // past the current _pfxw when in-place inserts have lengthened the u16 runs.
         const uint32_t wb      = pick_pfx_width(live_min_suffix());
         const uint32_t content = head_bytes() + (3u + wb) * round_cap(_nkeys) + live_tail() +
                                  npay - iv->value_len +
                                  // fresh cbase block only if this update introduces the first extern
                                  ((inline_value::external_value(upd.vtype) &&
                                    _cbases_pos.offset_from_tail == 0) ? 64u : 0u);
         return content <= page_bytes();
      }

      bool can_add_address(sal::ptr_address addr) const noexcept
      {
         if (_cbases_pos.offset_from_tail == 0)
            return _ncb < max_cbases;  // no block yet → never deref it (find_u32x16 reads 64B)
         const uint32_t base = *addr & ~0xFu;
         if (ucc::find_u32x16(reinterpret_cast<const uint32_t*>(cbase_ptr()), max_cbases, base) <
             int(max_cbases))
            return true;
         return _ncb < max_cbases;
      }

      // ── split-divider selection ──────────────────────────────────────────────────
      // TWO probes (part's divider rule, b11dc1a5): the keys at 40% and 60% of the
      // page (median ∓10%; 60 keys → 24 and 36). The divider is the shortest
      // separator between THOSE two keys — their shared prefix + one byte snapped
      // toward trailing zeros (power-of-2 friendly) — and the split index is its
      // lower_bound, which lands in (40%, 60%] by construction so both halves are
      // non-empty. Dividers feed the lower-bound radix above, so shorter + rounder
      // is structurally cheaper up there. Cost: at most two extern resolutions and
      // one in-page search, instead of one resolution per adjacent pair across a
      // middle window.
      struct split_pos
      {
         uint16_t index;    // right half = [index, nkeys)
         uint16_t div_len;  // full-key-space divider length
         uint8_t  div[1056];  // a divider never exceeds a max-length key + 1
         std::span<const uint8_t> divider() const noexcept { return {div, div_len}; }
      };
      /// The shortest separator between entries ia < ib: their shared prefix plus one
      /// byte v with key[ia] < divider <= key[ib], v snapped toward trailing zeros when
      /// `snap` (rounder dividers are cheaper in the radix above). The split index is the
      /// divider's lower bound, in (ia, ib].
      split_pos separator(uint32_t ia, uint32_t ib, const kv_resolver& R, bool snap = true) const
          noexcept
      {
         assert(ia < ib && ib < _nkeys);
         // Separator math needs the REAL suffix of an extern record (its in-page fragment can
         // tie); splits are cold, so resolving the two probes through the key_val_node is fine.
         auto real_suffix = [&](uint32_t i) -> std::span<const uint8_t> {
            const inline_value* iv = record(i);
            if (iv->value_type != inline_value::vt_extern)
               return {iv->data, iv->suffix_len};
            auto kk = R(decode_extern(iv));
            return {kk.data() + _cpfx_len, kk.size() - _cpfx_len};
         };
         auto     a = real_suffix(ia), b = real_suffix(ib);
         uint32_t p = shared_prefix(a.data(), a.size(), b.data(), b.size());
         if (p >= b.size()) [[unlikely]]
            page_fault("entries out of order", ib);  // strict order: b can't be a prefix of a
         // separator = b[0..p) + v, valid v in (a[p], b[p]] — or [0, b[p]] when a
         // is a prefix of b. Maximize trailing zeros of v (simple bit math, no scans).
         const int     lo_excl = p < a.size() ? int(a[p]) : -1;
         const uint8_t hi_v    = b[p];
         uint8_t       v       = hi_v;
         for (int z = snap ? 8 : 0; z >= 1; --z)
         {
            const uint8_t cand = uint8_t(hi_v & ~((1u << z) - 1u));
            if (int(cand) > lo_excl)
            {
               v = cand;
               break;
            }
         }
         split_pos r;
         std::memcpy(r.div, cpfx_data(), _cpfx_len);
         std::memcpy(r.div + _cpfx_len, b.data(), p);
         r.div[_cpfx_len + p] = v;
         r.div_len            = uint16_t(_cpfx_len + p + 1);
         const int idx        = lower_bound(r.div, r.div_len, R);
         if (idx <= int(ia) || idx > int(ib)) [[unlikely]]
            page_fault("divider does not separate its entries", uint32_t(idx));
         r.index = uint16_t(idx);
         return r;
      }
      /// The balanced choice: probes at 40% and 60% of the page.
      split_pos get_split_pos(const kv_resolver& R = {}) const noexcept
      {
         assert(_nkeys >= 2);
         const uint32_t n  = _nkeys;
         uint32_t       ia = (n * 2) / 5, ib = (n * 3) / 5;
         if (ib >= n)
            ib = n - 1;
         if (ib < 1)
            ib = 1;
         if (ia >= ib)
            ia = ib - 1;
         return separator(ia, ib, R);
      }

      // ── split planning: both halves are known to fit before either is built ──────
      // A half carries its divider and its common prefix in its tail as well as its
      // records, and a divider is as long as the bytes its neighbours share. With keys of
      // up to 1,024 bytes that is most of a page, so the balanced split is not always
      // possible: a half's prefix is then tightened less (its records stay as they are),
      // and if the right half still does not fit, the split point moves right until it
      // does. Splitting off the last entry alone always fits (one record, its divider and a
      // prefix that shares the divider's bytes: at most 1,508 bytes of a 2,048-byte page),
      // and the left half then holds what the page held before, less one record.
      struct split_plan
      {
         bool      ok = false;
         split_pos pos;                   // right half = [pos.index, nkeys), reached by pos.divider()
         uint16_t  left_cp = 0, right_cp = 0;  // each half's common prefix length
         uint8_t   left_w = 4, right_w = 4;    // each half's widest prefix column
      };
      /// Longest common prefix a rebuild of entries [lo,hi) can factor out, from the bytes
      /// the page holds (a sorted range's common prefix is that of its first and last).
      uint16_t natural_cpfx(uint32_t lo, uint32_t hi) const noexcept
      {
         auto a = suffix(lo);
         auto b = suffix(hi - 1);
         return uint16_t(_cpfx_len + shared_prefix(a.data(), a.size(), b.data(), b.size()));
      }
      /// Upper bound on the bytes a page built from entries [lo,hi) needs under common
      /// prefix length `ncp` (>= this page's), divider `div` and prefix columns of at most
      /// `max_w` bytes. Mirrors init_from's record sizes; the column width is the bound.
      /// Does the half fit a page? With `bound_meta`, its prefix and divider together must
      /// also stay within one key's length: the rule that leaves every page, down to a
      /// single record, room to take an update of that record or one more key.
      bool half_fits(uint32_t lo, uint32_t hi, uint32_t ncp, std::span<const uint8_t> div,
                     uint32_t max_w, const kv_resolver& R, bool bound_meta = true) const noexcept
      {
         uint32_t meta = 0;
         return half_bytes(lo, hi, ncp, div, max_w, R, &meta) <= page_bytes() &&
                (!bound_meta || meta <= max_key_bytes);
      }
      uint32_t half_bytes(uint32_t lo, uint32_t hi, uint32_t ncp, std::span<const uint8_t> div,
                          uint32_t max_w, const kv_resolver& R, uint32_t* meta_out = nullptr) const
          noexcept
      {
         uint8_t cp[sizeof(split_pos::div)];
         std::memcpy(cp, cpfx_data(), _cpfx_len);
         std::memcpy(cp + _cpfx_len, suffix(lo).data(), ncp - _cpfx_len);
         const uint32_t strip = ncp - _cpfx_len;
         const uint32_t meta  = meta_bytes(cp, ncp, div.data(), uint32_t(div.size()));
         uint32_t       total = head_bytes() + (3u + max_w) * round_cap(hi - lo) + meta;
         if (meta_out)
            *meta_out = meta;
         bool ext = false;
         for (uint32_t si = lo; si < hi; ++si)
         {
            const inline_value* iv = record(si);
            ext |= inline_value::external_value(iv->value_type);
            uint32_t sn;
            if (iv->value_type == inline_value::vt_extern && strip)
            {
               const uint32_t real = uint32_t(R(decode_extern(iv)).size()) - ncp;
               sn                  = real < ext_frag_cap ? real : uint32_t(ext_frag_cap);
            }
            else if (iv->value_type == inline_value::vt_extern)
               sn = iv->suffix_len;
            else
               sn = iv->suffix_len - strip;
            total += 3u + sn + iv->value_len;
         }
         return total + (ext ? 64u : 0u);
      }
      split_plan plan_split(const kv_resolver& R) const noexcept
      {
         split_plan     pl;
         const uint32_t n = _nkeys;
         if (n < 2)
            return pl;
         pl.pos = get_split_pos(R);
         for (;;)
         {
            const uint32_t i = pl.pos.index;
            // left half: this page's divider; tighten the prefix if the result fits
            const uint16_t nl = natural_cpfx(0, i);
            bool           l_ok = true;
            if (half_fits(0, i, nl, divider(), 4, R))
               pl.left_cp = nl, pl.left_w = 4;
            else if (half_fits(0, i, _cpfx_len, divider(), _pfxw, R, /*bound_meta=*/false))
               pl.left_cp = _cpfx_len, pl.left_w = _pfxw;  // what the page already holds
            else
               l_ok = false;
            // right half: the new divider. A prefix no longer than the divider's shared
            // part lies inside the divider's bytes and costs nothing.
            const uint16_t nr     = natural_cpfx(i, n);
            const uint16_t within = uint16_t(pl.pos.div_len - 1) < nr ? uint16_t(pl.pos.div_len - 1)
                                                                      : nr;
            bool r_ok = false;
            for (const uint8_t w : {uint8_t(4), uint8_t(_pfxw)})
            {
               for (const uint16_t cp : {nr, within})
                  if (half_fits(i, n, cp, pl.pos.divider(), w, R))
                  {
                     pl.right_cp = cp, pl.right_w = w, r_ok = true;
                     break;
                  }
               if (r_ok)
                  break;
            }
            if (l_ok && r_ok)
            {
               pl.ok = true;
               return pl;
            }
            if (i + 1 >= n)
               return pl;  // not even the last entry alone: the caller reports it
            pl.pos = separator(i, i + 1, R);
         }
      }

      // ── a page for one key that no existing page can take ────────────────────────
      /// Entry i's whole key into `out` (>= 1,024 bytes); returns its length.
      uint32_t real_key(uint32_t i, uint8_t* out, const kv_resolver& R) const noexcept
      {
         const inline_value* iv = record(i);
         if (iv->value_type == inline_value::vt_extern)
         {
            auto kk = R(decode_extern(iv));
            std::memcpy(out, kk.data(), kk.size());
            return uint32_t(kk.size());
         }
         return full_key(i, out);
      }
      /// The divider of a new page that holds only (k,n), a key above every entry here:
      /// the shortest separator between the last entry and the key.
      split_pos divider_above(const uint8_t* k, uint32_t n, const kv_resolver& R) const noexcept
      {
         uint8_t        last[sizeof(split_pos::div)];
         const uint32_t ln = real_key(_nkeys - 1u, last, R);
         const uint32_t p  = shared_prefix(last, ln, k, n);
         if (p >= n) [[unlikely]]
            page_fault("key is not above the page", n);
         const int lo_excl = p < ln ? int(last[p]) : -1;
         uint8_t   v       = k[p];
         for (int z = 8; z >= 1; --z)
         {
            const uint8_t cand = uint8_t(k[p] & ~((1u << z) - 1u));
            if (int(cand) > lo_excl)
            {
               v = cand;
               break;
            }
         }
         split_pos r;
         std::memcpy(r.div, k, p);
         r.div[p]  = v;
         r.div_len = uint16_t(p + 1);
         r.index   = _nkeys;
         return r;
      }
      /// This page's divider once a new page holding only (k,n), a key below every entry
      /// here, is put in front of it: the first entry's bytes up to and including the one
      /// where it leaves the key. A prefix of this page's keys, so it shares the common
      /// prefix's bytes.
      split_pos divider_below(const uint8_t* k, uint32_t n, const kv_resolver& R) const noexcept
      {
         split_pos      r;
         const uint32_t fn = real_key(0, r.div, R);
         const uint32_t p  = shared_prefix(k, n, r.div, fn);
         if (p >= fn) [[unlikely]]
            page_fault("key is not below the page", n);
         r.div_len = uint16_t(p + 1);
         r.index   = 0;
         return r;
      }
      /// Would this page's entries fit under `div` in place of its divider?
      bool redivided_fits(std::span<const uint8_t> div) const noexcept
      {
         const uint32_t meta =
             meta_bytes(cpfx_data(), _cpfx_len, div.data(), uint32_t(div.size()));
         return head_bytes() + (3u + _pfxw) * round_cap(_nkeys) + meta +
                    (live_tail() - meta_live()) <=
                page_bytes();
      }

      // ── diagnostics ──────────────────────────────────────────────────────────────
      [[nodiscard]] bool validate_invariants() const noexcept
      {
         if (_cap % 4 || _nkeys > _cap || _ncb > max_cbases)
            return false;
         if (head_size() + tail_used() > size())
            return false;
         // cpfx + divider + (the 16-slot cbase block, when present) are all live tail content
         uint32_t live = meta_live() + (_cbases_pos.offset_from_tail ? 64u : 0u);
         uint8_t  kbuf[1056];  // full key <= 1024 (+slack)
         uint8_t  pbuf[1056];
         uint32_t plen     = 0;
         bool     prev_ext = false;
         for (uint32_t i = 0; i < _nkeys; ++i)
         {
            const uint16_t v = vals()[i];
            if (v == 0 || v > tail_used() || v >= ext_code_base)
               return false;
            const inline_value* iv = record(i);
            live += uint32_t(iv->byte_size());
            if (_pfxw != 1 && _pfxw != 2 && _pfxw != 4)
               return false;
            if (pfx_get(i) != stem(iv->data, iv->suffix_len))
               return false;
            if (use_clidx())  // every sampled block start must match its header sample
            {
               const uint32_t block = cl_stride() * cl_entries();
               if (i % block == 0)
               {
                  uint32_t sv;
                  switch (_pfxw)
                  {
                     case 1: sv = _clidx[i / block]; break;
                     case 2: sv = reinterpret_cast<const uint16_t*>(_clidx)[i / block]; break;
                     default: sv = reinterpret_cast<const uint32_t*>(_clidx)[i / block]; break;
                  }
                  if (sv != pfx_get(i))
                     return false;
               }
            }
            const uint32_t klen = full_key(i, kbuf);
            const bool     ext  = iv->value_type == inline_value::vt_extern;
            // extern records: full_key() yields cpfx + FRAGMENT (the real key lives in the
            // key_val_node, unreachable without a session) — the fingerprint hashes the real
            // key and adjacent fragments may tie, so both checks weaken to what is provable.
            if (!ext && fp()[i] != fingerprint_of(kbuf, klen))
               return false;
            if (i)
            {
               const int c = lex_cmp(pbuf, plen, kbuf, klen);
               if ((ext || prev_ext) ? c > 0 : c >= 0)
                  return false;  // strict ascending for inline pairs; fragments may tie
            }
            std::memcpy(pbuf, kbuf, klen);
            plen     = klen;
            prev_ext = ext;
            if (inline_value::external_value(iv->value_type))
            {
               if (iv->value_len != 2)
                  return false;
               uint16_t code;
               std::memcpy(&code, iv->data + iv->suffix_len, 2);
               if (code < ext_code_base || ((code >> 4) & 0xF) >= _ncb)
            return false;
            }
         }
         return live + _dead == tail_used();
      }

     private:
      // ── column geometry ──────────────────────────────────────────────────────────
      char*       cols() noexcept { return reinterpret_cast<char*>(this) + sizeof(leaf_page); }
      const char* cols() const noexcept
      {
         return reinterpret_cast<const char*>(this) + sizeof(leaf_page);
      }
      uint8_t*        fp() noexcept { return reinterpret_cast<uint8_t*>(cols()); }
      const uint8_t*  fp() const noexcept { return reinterpret_cast<const uint8_t*>(cols()); }
      // prefix column: _pfxw native-endian unsigned values per entry (big-endian key bytes
      // packed into an integer, so numeric order == lexicographic order). cap is a multiple
      // of 4 and head_bytes() is a multiple of 4, so every width is naturally aligned.
      uint8_t*       pfxb() noexcept { return reinterpret_cast<uint8_t*>(cols() + _cap); }
      const uint8_t* pfxb() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(cols() + _cap);
      }
      uint32_t pfx_get(uint32_t i) const noexcept
      {
         switch (_pfxw)
         {
            case 1: return pfxb()[i];
            case 2: return reinterpret_cast<const uint16_t*>(pfxb())[i];
            default: return reinterpret_cast<const uint32_t*>(pfxb())[i];
         }
      }
      void pfx_set(uint32_t i, uint32_t v) noexcept
      {
         switch (_pfxw)
         {
            case 1: pfxb()[i] = uint8_t(v); break;
            case 2: reinterpret_cast<uint16_t*>(pfxb())[i] = uint16_t(v); break;
            default: reinterpret_cast<uint32_t*>(pfxb())[i] = v; break;
         }
      }
      // ── cacheline index geometry ─────────────────────────────────────────────────
      // entries per column cacheline / samples the 64B header row can hold at width W
      uint32_t cl_entries() const noexcept { return 64u / _pfxw; }
      // sample stride in CLs: 1 until the column outgrows the 64B sample row (W=4 + big caps)
      uint32_t cl_stride() const noexcept
      {
         const uint32_t col_cls = (uint32_t(_cap) * _pfxw + 63u) / 64u;  // column footprint
         const uint32_t avail   = cl_entries();                          // row holds 64/W samples
         return (col_cls + avail - 1u) / avail;
      }
      bool use_clidx() const noexcept { return uint32_t(_nkeys) * _pfxw > 128u; }  // column > 2 CLs

      /// Rebuild the header cacheline-index: sample the FIRST stem of every stride-th column
      /// cacheline (the blocks the query scans). Called after any column splice — those
      /// cachelines are still hot, so this is ~a dozen resident reads.
      void rebuild_clidx() noexcept
      {
         std::memset(_clidx, 0xFF, sizeof(_clidx));  // unused slots compare >= every real stem
         if (!use_clidx())
            return;
         const uint32_t block = cl_stride() * cl_entries();  // entries per sampled block
         const uint32_t nsamp = (uint32_t(_nkeys) + block - 1u) / block;
         for (uint32_t j = 0; j < nsamp; ++j)
         {
            const uint32_t v = pfx_get(j * block);
            switch (_pfxw)
            {
               case 1: _clidx[j] = uint8_t(v); break;
               case 2: reinterpret_cast<uint16_t*>(_clidx)[j] = uint16_t(v); break;
               default: reinterpret_cast<uint32_t*>(_clidx)[j] = v; break;
            }
         }
      }

      /// SIMD lower-bound over the prefix column: first index whose stem >= v. When the
      /// column exceeds 2 cachelines, probe the header-resident cacheline index first and
      /// scan ONE sampled block instead of the whole column (the wtree lesson).
      int pfx_lower_bound(uint32_t v) const noexcept
      {
         uint32_t start = 0, len = _nkeys;
         if (use_clidx())
         {
            const uint32_t block = cl_stride() * cl_entries();
            const uint32_t nsamp = (uint32_t(_nkeys) + block - 1u) / block;
            uint32_t       cnt;  // samples STRICTLY below v (a block starting == v may have
                                 // equal stems spilling back into the previous block)
            switch (_pfxw)
            {
               case 1: cnt = uint32_t(ucc::lower_bound_padded(_clidx, nsamp, uint8_t(v))); break;
               case 2:
                  cnt = uint32_t(ucc::lower_bound_u16_padded(
                      reinterpret_cast<const uint16_t*>(_clidx), int(nsamp), uint16_t(v)));
                  break;
               default:
                  cnt = uint32_t(ucc::lower_bound_u32_padded(
                      reinterpret_cast<const uint32_t*>(_clidx), int(nsamp), v));
                  break;
            }
            const uint32_t cl = cnt ? cnt - 1u : 0u;
            start             = cl * block;
            len               = uint32_t(_nkeys) - start;
            if (len > block)
               len = block;  // landing at block-end == the next block's first entry: exact
         }
         int r;
         switch (_pfxw)
         {
            case 1:
               r = int(ucc::lower_bound_padded(pfxb() + start, len, uint8_t(v)));
               break;
            case 2:
               r = ucc::lower_bound_u16_padded(reinterpret_cast<const uint16_t*>(pfxb()) + start,
                                               int(len), uint16_t(v));
               break;
            default:
               r = ucc::lower_bound_u32_padded(reinterpret_cast<const uint32_t*>(pfxb()) + start,
                                               int(len), v);
               break;
         }
         return int(start) + r;
      }
      uint16_t*       vals() noexcept
      {
         return reinterpret_cast<uint16_t*>(cols() + (1u + _pfxw) * _cap);
      }
      const uint16_t* vals() const noexcept
      {
         return reinterpret_cast<const uint16_t*>(cols() + (1u + _pfxw) * _cap);
      }
      const void* cols_end() const noexcept { return cols() + (3u + _pfxw) * _cap; }

      const uint8_t* cpfx_data() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(resolve(tail_ptr<const uint8_t>{
             _cpfx_pos.offset_from_tail}));
      }
      const uint8_t* div_data() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(resolve(tail_ptr<const uint8_t>{
             _div_pos.offset_from_tail}));
      }
      // The 16-slot cbase block in the tail (valid only when _cbases_pos != 0, i.e. _ncb>0).
      sal::ptr_address* cbase_ptr() noexcept
      {
         return reinterpret_cast<sal::ptr_address*>(resolve(tail_ptr<>{_cbases_pos.offset_from_tail}));
      }
      const sal::ptr_address* cbase_ptr() const noexcept
      {
         return reinterpret_cast<const sal::ptr_address*>(
             resolve(tail_ptr<const uint8_t>{_cbases_pos.offset_from_tail}));
      }

      uint32_t live_tail() const noexcept { return tail_used() - _dead; }

      // ── small helpers ────────────────────────────────────────────────────────────
      /// Big-endian first `w` suffix bytes packed into an integer (zero-padded when the
      /// suffix is shorter) — numeric compare == lexicographic compare on those bytes.
      static uint32_t stemN(const uint8_t* s, size_t n, uint32_t w) noexcept
      {
         uint32_t v = 0;
         for (uint32_t i = 0; i < w; ++i)
            v = (v << 8) | (i < n ? uint32_t(s[i]) : 0u);
         return v;
      }
      uint32_t stem(const uint8_t* s, size_t n) const noexcept { return stemN(s, n, _pfxw); }
      /// UPPER BOUND on the rebuild width: the MIN post-cpfx suffix length, capped at 4
      /// (rounded down to a SIMD-friendly 1/2/4). The actual pick (width_from) may choose
      /// narrower — used by the fits checks, which need only an upper bound.
      static uint32_t pick_pfx_width(uint32_t min_suffix) noexcept
      {
         return min_suffix >= 4 ? 4u : (min_suffix >= 2 ? 2u : 1u);
      }
      /// The actual width rule: 1 only when forced (a <2-byte suffix exists); widen 2 → 4
      /// only when the u16 stems leave LONG equal runs (the run walk would out-cost the
      /// doubled column scan) — uniform keysets stay at u16, prefix-dense text gets u32.
      static uint32_t width_from(uint32_t min_suffix, uint32_t max_u16_run) noexcept
      {
         if (min_suffix < 2)
            return 1;
         if (min_suffix < 4 || max_u16_run <= 3)
            return 2;
         return 4;
      }
      /// Smallest live suffix length (UINT32_MAX when empty) — the width input.
      uint32_t live_min_suffix() const noexcept
      {
         uint32_t m = UINT32_MAX;
         for (uint32_t i = 0; i < _nkeys; ++i)
            if (record(i)->suffix_len < m)
               m = record(i)->suffix_len;
         return m;
      }
      static uint32_t shared_prefix(const uint8_t* a, size_t an, const uint8_t* b,
                                    size_t bn) noexcept
      {
         const size_t m = an < bn ? an : bn;
         size_t       i = 0;
         while (i < m && a[i] == b[i])
            ++i;
         return uint32_t(i);
      }
      static int lex_cmp(const uint8_t* a, size_t an, const uint8_t* b, size_t bn) noexcept
      {
         const size_t m = an < bn ? an : bn;
         const int    c = m ? std::memcmp(a, b, m) : 0;
         if (c)
            return c;
         return an == bn ? 0 : (an < bn ? -1 : 1);
      }
      /// Suffix order of entry i vs (s,sn). An extern record holds only a FRAGMENT (a
      /// prefix of the real suffix): any comparison decided strictly within the fragment
      /// bytes is decided for the real suffix too; a prefix-tie (the fragment ran out at
      /// or before s's end) resolves through the key_val_node's full key.
      int cmp_suffix(uint32_t i, const uint8_t* s, uint32_t sn, const kv_resolver& R) const
          noexcept
      {
         const inline_value* iv = record(i);
         const uint32_t      fl = iv->suffix_len;
         const uint32_t      m  = fl < sn ? fl : sn;
         const int           c  = m ? std::memcmp(iv->data, s, m) : 0;
         if (c)
            return c < 0 ? -1 : 1;
         if (iv->value_type != inline_value::vt_extern)
            return fl == sn ? 0 : (fl < sn ? -1 : 1);
         if (sn < fl)
            return 1;  // s is a proper prefix of the fragment → s < fragment <= real suffix
         auto kk = R(decode_extern(iv));  // fragment exhausted → extend via the full key
         return lex_cmp(kk.data() + _cpfx_len, kk.size() - _cpfx_len, s, sn);
      }
      static uint32_t payload_bytes(size_t vlen, uint8_t vtype) noexcept
      {
         return inline_value::external_value(vtype) ? 2u : uint32_t(vlen);  // cbase code
      }
      static uint32_t record_bytes(uint32_t sn, const insert_op& ins) noexcept
      {
         if (ins.vtype == inline_value::vt_extern)  // FRAGMENT + cbase code (full key external)
            return 3u + (sn < ext_frag_cap ? sn : uint32_t(ext_frag_cap)) + 2u;
         if (ins.vtype == inline_value::vt_extern_val)  // FULL suffix in-page + cbase code (value-only)
            return 3u + sn + 2u;
         return 3u + sn + payload_bytes(ins.value.size(), ins.vtype);
      }
      /// Largest live INLINE suffix length (rebuild prepend must keep them <= 255; extern
      /// fragments re-cap at ext_frag_cap during the rebuild, so they never constrain).
      /// -1 when no record keeps its whole suffix in the page.
      int max_live_suffix() const noexcept
      {
         int m = -1;
         for (uint32_t i = 0; i < _nkeys; ++i)
         {
            const inline_value* iv = record(i);
            // inline + vt_extern_val carry the full post-cpfx suffix (re-expands on rebuild);
            // vt_extern keeps only a re-capped fragment so it never constrains.
            if (inline_value::suffix_complete(iv->value_type) && int(iv->suffix_len) > m)
               m = int(iv->suffix_len);
         }
         return m;
      }
      uint32_t live_cbase_count() const noexcept
      {
         uint32_t bases[max_cbases];
         std::memset(bases, 0xFF, sizeof(bases));  // 0xFFFFFFFF never equals a base (low 4 = 0)
         uint32_t n = 0;
         for (uint32_t i = 0; i < _nkeys && n < max_cbases; ++i)
         {
            const inline_value* iv = record(i);
            if (!inline_value::external_value(iv->value_type))
               continue;
            const uint32_t base = *decode_extern(iv) & ~0xFu;
            if (ucc::find_u32x16(bases, max_cbases, base) == int(max_cbases))
               bases[n++] = base;
         }
         return n;
      }

      // ── extern coding ────────────────────────────────────────────────────────────
      sal::ptr_address decode_extern(const inline_value* iv) const noexcept
      {
         assert(inline_value::external_value(iv->value_type) && iv->value_len == 2);
         uint16_t code;
         std::memcpy(&code, iv->data + iv->suffix_len, 2);
         assert(code >= ext_code_base);
         assert(_cbases_pos.offset_from_tail != 0 && "extern record but no cbase block");
         return sal::ptr_address(*cbase_ptr()[(code >> 4) & 0xF] + (code & 0xF));
      }
      /// @pre can_add_address(addr) AND caller reserved the 64B block via can_apply/rebuilt_fits
      uint16_t encode_extern(sal::ptr_address addr) noexcept
      {
         if (_cbases_pos.offset_from_tail == 0)  // lazy: 16-slot block on the FIRST extern
         {
            constexpr uint16_t blk = uint16_t(sizeof(sal::ptr_address) * max_cbases);  // 64B
            uint8_t*           d   = reinterpret_cast<uint8_t*>(alloc_tail(blk));
            std::memset(d, 0xFF, blk);  // null_ptr_address sentinels (find_u32x16 reads all 16)
            _cbases_pos = tail_pos();
         }
         const uint32_t base = *addr & ~0xFu;
         int idx = ucc::find_u32x16(reinterpret_cast<const uint32_t*>(cbase_ptr()), max_cbases, base);
         if (idx == int(max_cbases))
         {
            assert(_ncb < max_cbases);
            idx                 = _ncb;
            cbase_ptr()[_ncb++] = sal::ptr_address(base);
         }
         return uint16_t(ext_code_base | (uint32_t(idx) << 4) | (*addr & 0xFu));
      }
      void write_payload(uint8_t* dst, const auto& op) noexcept  // insert_op or update_op
      {
         if (inline_value::external_value(op.vtype))
         {
            const uint16_t code = encode_extern(op.ext);
            std::memcpy(dst, &code, 2);
         }
         else
            std::memcpy(dst, op.value.data(), op.value.size());
      }
      uint16_t alloc_record(const uint8_t* s, uint32_t sn, const insert_op& ins) noexcept
      {
         assert(!inline_value::inline_payload(ins.vtype) || int(sn + ins.value.size()) <=
                    max_inline_entry);
         if (ins.vtype == inline_value::vt_extern && sn > ext_frag_cap)
            sn = ext_frag_cap;  // store the comparison fragment; the key_val_node has the rest
         auto* iv = reinterpret_cast<inline_value*>(
             alloc_tail(uint16_t(record_bytes(sn, ins))));
         const uint16_t off = tail_used();  // record offset — capture BEFORE write_payload, which
                                            // may lazily alloc_tail the cbase block (moving tail_used)
         iv->suffix_len = uint8_t(sn);
         iv->value_len  = uint8_t(payload_bytes(ins.value.size(), ins.vtype));
         iv->value_type = ins.vtype;
         std::memcpy(iv->data, s, sn);
         write_payload(iv->data + sn, ins);
         return off;
      }

      // ── column splice (in-place) ─────────────────────────────────────────────────
      void open_gap(uint32_t pos) noexcept
      {
         assert(_nkeys < _cap);
         const uint32_t w = _pfxw;
         std::memmove(fp() + pos + 1, fp() + pos, _nkeys - pos);
         std::memmove(pfxb() + (pos + 1) * w, pfxb() + pos * w, w * (_nkeys - pos));
         std::memmove(vals() + pos + 1, vals() + pos, 2u * (_nkeys - pos));
      }
      void close_gap(uint32_t pos) noexcept
      {
         const uint32_t w = _pfxw;
         std::memmove(fp() + pos, fp() + pos + 1, _nkeys - pos - 1);
         std::memmove(pfxb() + pos * w, pfxb() + (pos + 1) * w, w * (_nkeys - pos - 1));
         std::memmove(vals() + pos, vals() + pos + 1, 2u * (_nkeys - pos - 1));
      }
      /// +4 column capacity within the free middle; rightmost column moves first.
      void grow_cap() noexcept
      {
         const uint32_t w  = _pfxw;
         const uint16_t nc = _cap + 4;
         assert(head_bytes() + (3u + w) * nc <= size() - tail_used());
         char* base = cols();
         std::memmove(base + (1u + w) * nc, base + (1u + w) * _cap, 2u * _nkeys);  // vals
         std::memmove(base + nc, base + _cap, w * _nkeys);                          // pfx
         _cap = nc;
         // capacity feeds the sample-block geometry (stride can cross at W=4) and the column
         // base moved — the header index must match before the caller's lower_bound runs.
         rebuild_clidx();
      }

      void init_empty() noexcept
      {
         heap_alloc_pos = tail_ptr<>{0};  // compact_to enters here without init_page
         _nkeys         = 0;
         _cap           = round_cap(0);
         _ncb           = 0;
         _pfxw          = 2;  // until a rebuild picks the width from the live suffixes
         _cpfx_len      = 0;
         _dead          = 0;
         _cpfx_pos      = tail_ptr<>{0};
         _div_len       = 0;
         _div_pos       = tail_ptr<>{0};
         _cbases_pos    = tail_ptr<>{0};  // no cbase block until the first extern (lazy)
         std::memset(_clidx, 0xFF, sizeof(_clidx));
      }

      // ── the ONE rebuild engine (shared splice) ───────────────────────────────────
      // Every rebuilding ctor and compact_to funnels here: copy src entries [lo,hi)
      // under a new common prefix (extra_src/extra_n = bytes appended to src's cpfx
      // when tightening; ncp < src->_cpfx_len means loosening and suffixes re-expand),
      // optionally splicing one insert, one update, or one removal. Dead tail bytes
      // and dead cbases never survive a rebuild.
      void init_from(const leaf_page* src, uint32_t lo, uint32_t hi, uint16_t ncp,
                     const uint8_t* extra_src, uint32_t extra_n, const insert_op* ins,
                     const update_op* upd, int drop = -1, uint32_t max_w = 4,
                     const kv_resolver& R = {}, std::span<const uint8_t> div = {}) noexcept
      {
         init_empty();
         _version_history = src->_version_history;
         const uint32_t cnt = (hi - lo) + (ins ? 1u : 0u) - (drop >= 0 ? 1u : 0u);
         _cap               = round_cap(cnt);
         // DYNAMIC PREFIX WIDTH: a rebuild is the only place the column layout changes, so
         // it decides the width — 1 when a short suffix forces it; 4 only when the u16 stems
         // of the (post-transform, sorted) suffixes leave runs long enough that the run walk
         // would out-cost the doubled column scan; else the u16 default.
         {
            const int prepend_d = int(src->_cpfx_len) - int(ncp);
            // first two bytes of entry si's POST-transform suffix as a u16 stem
            auto tstem = [&](const inline_value* iv) -> uint32_t {
               uint8_t b[2];
               int     got = 0;
               if (prepend_d > 0)
               {
                  const uint8_t* pre = src->cpfx_data() + ncp;
                  for (; got < 2 && got < prepend_d; ++got)
                     b[got] = pre[got];
                  for (int j = 0; got < 2 && j < int(iv->suffix_len); ++j, ++got)
                     b[got] = iv->data[j];
               }
               else
               {
                  const uint8_t* p   = iv->data - prepend_d;
                  const int      len = int(iv->suffix_len) + prepend_d;
                  for (; got < 2 && got < len; ++got)
                     b[got] = p[got];
               }
               return stemN(b, size_t(got), 2);
            };
            uint32_t min_sfx = UINT32_MAX, run = 1, max_run = 0, prev = 0;
            bool     have_prev = false;
            for (uint32_t si = lo; si < hi; ++si)
            {
               if (int(si) == drop)
                  continue;
               const inline_value* iv = src->record(si);
               const uint32_t      sl = uint32_t(int(iv->suffix_len) + prepend_d);
               if (sl < min_sfx)
                  min_sfx = sl;
               const uint32_t s2 = tstem(iv);
               if (have_prev && s2 == prev)
               {
                  if (++run > max_run)
                     max_run = run;
               }
               else
                  run = 1;
               prev      = s2;
               have_prev = true;
            }
            if (ins && uint32_t(ins->key.size()) - ncp < min_sfx)
               min_sfx = uint32_t(ins->key.size()) - ncp;
            uint32_t w = width_from(min_sfx == UINT32_MAX ? 2u : min_sfx, max_run);
            if (w > max_w)
               w = max_w;  // compaction clamps to the source width (shrink-only)
            _pfxw = uint8_t(w);
         }
         if (head_bytes() + (3u + _pfxw) * _cap > size()) [[unlikely]]
            page_fault("columns do not fit the page", cnt);
         if (ncp > sizeof(split_pos::div) || div.size() > sizeof(split_pos::div)) [[unlikely]]
            page_fault("prefix or divider longer than a key", ncp);

         // new common prefix bytes: src's cpfx truncated-or-extended
         uint8_t cp[sizeof(split_pos::div)];
         {
            const uint32_t keep = ncp < src->_cpfx_len ? ncp : src->_cpfx_len;
            std::memcpy(cp, src->cpfx_data(), keep);
            if (ncp > src->_cpfx_len)
            {
               assert(extra_n == uint32_t(ncp - src->_cpfx_len) && extra_src);
               std::memcpy(cp + keep, extra_src, extra_n);
            }
         }
         // The common prefix and the page divider (copied verbatim, or set for a split's right
         // half) are live tail content, allocated here so the blit/compact used-tail region
         // carries them: stored once when one is a prefix of the other (see meta_aliases).
         _cpfx_len = uint16_t(ncp);
         _div_len  = uint16_t(div.size());
         if (meta_aliases(cp, ncp, div.data(), uint32_t(div.size())))
         {
            const bool     div_longer = div.size() > ncp;
            const uint32_t len        = div_longer ? uint32_t(div.size()) : uint32_t(ncp);
            std::memcpy(alloc_tail(uint16_t(len)), div_longer ? div.data() : cp, len);
            _cpfx_pos = _div_pos = tail_pos();
         }
         else
         {
            if (ncp)
            {
               std::memcpy(alloc_tail(uint16_t(ncp)), cp, ncp);
               _cpfx_pos = tail_pos();
            }
            if (!div.empty())
            {
               std::memcpy(alloc_tail(uint16_t(div.size())), div.data(), div.size());
               _div_pos = tail_pos();
            }
         }
         const int prepend = int(src->_cpfx_len) - int(ncp);  // >0: suffixes re-expand

         uint32_t w = 0;
         auto     emit_src = [&](uint32_t si)
         {
            const inline_value* iv = src->record(si);
            uint32_t            sn = (prepend >= 0) ? iv->suffix_len + uint32_t(prepend)
                                                    : iv->suffix_len - uint32_t(-prepend);
            const bool is_ext      = (upd && si == upd->index)
                                         ? (upd->vtype == inline_value::vt_extern)
                                         : (iv->value_type == inline_value::vt_extern);
            // INVARIANT: an extern record's fragment is always MAXIMAL — min(real suffix,
            // ext_frag_cap) bytes of the real key in the CURRENT cpfx space — so the pfx
            // stem column always reflects the real suffix's first bytes (a short fragment
            // would zero-pad its stem and break the find pre-screen). When the transform
            // changes the suffix space (prepend != 0), re-derive from the full key.
            const uint8_t* ext_src = nullptr;  // fragment source when re-deriving
            if (is_ext)
            {
               if (prepend != 0 && iv->value_type == inline_value::vt_extern)
               {
                  auto kk = R(src->decode_extern(iv));
                  assert(kk.size() >= ncp);
                  ext_src = kk.data() + ncp;
                  sn      = uint32_t(kk.size()) - ncp;
               }
               if (sn > ext_frag_cap)
                  sn = ext_frag_cap;
            }
            uint8_t pay = iv->value_len;
            if (upd && si == upd->index)
               pay = uint8_t(payload_bytes(upd->value.size(), upd->vtype));
            if (sn > max_suffix_len) [[unlikely]]
               page_fault("suffix too long for a page record", sn);
            auto* nv = reinterpret_cast<inline_value*>(alloc_tail(uint16_t(3u + sn + pay)));
            const uint16_t off = tail_used();  // capture BEFORE encode_extern may alloc the cbase block
            nv->suffix_len = uint8_t(sn);
            nv->value_len  = pay;
            if (ext_src)
               std::memcpy(nv->data, ext_src, sn);  // maximal fragment off the real key
            else if (prepend > 0)
            {
               const uint32_t a = uint32_t(prepend) < sn ? uint32_t(prepend) : sn;
               std::memcpy(nv->data, src->cpfx_data() + ncp, a);
               if (sn > a)
                  std::memcpy(nv->data + a, iv->data, sn - a);
            }
            else
               std::memcpy(nv->data, iv->data - prepend, sn);  // -prepend = strip count
            if (upd && si == upd->index)
            {
               nv->value_type = upd->vtype;
               write_payload(nv->data + sn, *upd);
            }
            else if (inline_value::external_value(iv->value_type))
            {
               nv->value_type      = iv->value_type;  // preserve vt_extern vs vt_extern_val
               const uint16_t code = encode_extern(src->decode_extern(iv));  // re-base compactly
               std::memcpy(nv->data + sn, &code, 2);
            }
            else
            {
               nv->value_type = iv->value_type;
               std::memcpy(nv->data + sn, iv->data + iv->suffix_len, iv->value_len);
            }
            fp()[w] = src->fp()[si];  // full-key fingerprints survive any refactoring
            pfx_set(w, stem(nv->data, sn));
            vals()[w] = off;
            ++w;
         };

         // splice position for the insert (in src index space)
         uint32_t ipos = hi;
         if (ins)
         {
            ipos = uint32_t(
                src->lower_bound(ins->key.data(), uint32_t(ins->key.size()), ins->res));
            assert(src->find(ins->key.data(), uint32_t(ins->key.size()), ins->res) < 0);
            if (ipos < lo)
               ipos = lo;
            if (ipos > hi)
               ipos = hi;
         }
         for (uint32_t si = lo; si < hi; ++si)
         {
            if (ins && si == ipos)
            {
               emit_insert(*ins, w);
               ++w;
            }
            if (int(si) == drop)
               continue;
            emit_src(si);
         }
         if (ins && ipos == hi)
         {
            emit_insert(*ins, w);
            ++w;
         }
         _nkeys = uint16_t(w);
         if (w != cnt || head_size() + uint32_t(tail_used()) > size()) [[unlikely]]
            page_fault("rebuilt page is inconsistent", w);
         rebuild_clidx();
         clear_checksum();  // compact path: dst keeps its own SAL header, content is fresh
      }

      void emit_insert(const insert_op& ins, uint32_t w) noexcept
      {
         const uint8_t* k  = ins.key.data();
         const uint32_t n  = uint32_t(ins.key.size());
         assert(n >= _cpfx_len && (!_cpfx_len || std::memcmp(k, cpfx_data(), _cpfx_len) == 0));
         const uint32_t sn = n - _cpfx_len;
         vals()[w]         = alloc_record(k + _cpfx_len, sn, ins);
         fp()[w]           = fingerprint_of(k, n);
         pfx_set(w, stem(k + _cpfx_len, sn));
      }
   } __attribute__((packed));

   static_assert(sizeof(leaf_page) == 104, "paged(22) + fields(18, cbases now variable) + clidx(64)");
}  // namespace sal_art_ref

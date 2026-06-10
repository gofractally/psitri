#pragma once
#include <cassert>
#include <cstring>
#include <span>

#include <heart/inline_value.hpp>
#include <heart/paged.hpp>
#include <heart/radix_type_ids.hpp>
#include <heart/smart_address.hpp>  // tail_ptr
#include <ucc/lower_bound.hpp>      // lower_bound_u16_padded, find_byte, find_u32x16
#include <ucc/round.hpp>

namespace heart
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
   struct leaf_page : paged
   {
      static constexpr sal::header_type type_id          = radix_type_ids::leaf_page;
      static constexpr uint32_t         page_bytes       = 2048;
      static constexpr uint32_t         max_cbases       = 16;
      static constexpr int              max_inline_entry = 128;  // suffix+value > this => extern
      static constexpr uint16_t         ext_code_base    = 0xFF00;
      static constexpr uint16_t         max_suffix_len   = 255;  // inline_value u8 field

      uint16_t         _nkeys;
      uint16_t         _cap;       // column capacity, always a multiple of 4
      uint8_t          _ncb;       // used cbases
      uint8_t          _cpfx_len;  // factored common prefix length
      uint16_t         _dead;      // dead tail-heap bytes (orphaned records)
      tail_ptr<>       _cpfx_pos;  // common prefix bytes in the tail heap (0 = none)
      sal::ptr_address _cbases[max_cbases];

      // ── op tickets ───────────────────────────────────────────────────────────────
      struct insert_op
      {
         std::span<const uint8_t> key;    // FULL key as the page sees it
         std::span<const uint8_t> value;  // vt_bytes payload / vt_subtree 4-byte root
         uint8_t                  vtype = inline_value::vt_bytes;
         sal::ptr_address         ext   = sal::null_ptr_address;  // vt_extern target
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
      };
      struct clone_remove_op
      {
         const leaf_page* src;
         uint16_t         index;
      };
      struct split_op  // one half of a split: src entries [lo,hi)
      {
         const leaf_page* src;
         uint16_t         lo, hi;
      };

      // ── SAL lifecycle ────────────────────────────────────────────────────────────
      static constexpr uint32_t head_bytes() noexcept { return uint32_t(sizeof(leaf_page)); }
      uint32_t head_size() const noexcept { return head_bytes() + 5u * _cap; }
      static uint32_t alloc_size() noexcept { return page_bytes; }
      static uint32_t alloc_size(const clone_insert_op&) noexcept { return page_bytes; }
      static uint32_t alloc_size(const clone_update_op&) noexcept { return page_bytes; }
      static uint32_t alloc_size(const clone_remove_op&) noexcept { return page_bytes; }
      static uint32_t alloc_size(const split_op&) noexcept { return page_bytes; }
      uint32_t        cow_size() const noexcept { return page_bytes; }
      static uint16_t round_cap(uint32_t n) noexcept
      {
         return uint16_t(ucc::round_up_multiple<4>(n < 8 ? 8u : n));
      }
      uint32_t compact_size() const noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(head_bytes() +
                                                            5u * round_cap(_nkeys) + live_tail());
      }
      void compact_to(sal::alloc_header* dst) const noexcept
      {
         static_cast<leaf_page*>(dst)->init_from(this, 0, _nkeys, _cpfx_len, nullptr, 0, nullptr,
                                                 nullptr);
      }
      void copy_to(sal::alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }

      /// SAL destroy hook: a dying page releases its children (external value objects
      /// and nested subtree roots) — the cascade that frees what only this page owned.
      template <typename Session>
      void destroy(const Session& s) const
      {
         for_each_child([&](sal::ptr_address c) { s->release(c); });
      }

      /// Visit SAL children (external values + nested subtree roots) — the release/
      /// retain cascade and ref audits walk this.
      template <typename F>
      void for_each_child(F&& f) const
      {
         for (uint32_t i = 0; i < _nkeys; ++i)
         {
            const inline_value* iv = record(i);
            if (iv->value_type == inline_value::vt_extern)
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
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const clone_insert_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         const leaf_page* s = op.src;
         // factoring only ever LOOSENS here: cpfx may shrink to admit a diverging key
         const uint16_t ncp = uint16_t(
             shared_prefix(op.ins.key.data(), op.ins.key.size(), s->cpfx_data(), s->_cpfx_len));
         init_from(s, 0, s->_nkeys, ncp, nullptr, 0, &op.ins, nullptr);
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const clone_update_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         init_from(op.src, 0, op.src->_nkeys, op.src->_cpfx_len, nullptr, 0, nullptr, &op.upd);
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const clone_remove_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         init_from(op.src, 0, op.src->_nkeys, op.src->_cpfx_len, nullptr, 0, nullptr, nullptr,
                   int(op.index));
      }
      leaf_page(uint32_t asize, sal::ptr_address_seq seq, const split_op& op) noexcept
      {
         this->init_page(asize, type_id, seq);
         const leaf_page* s = op.src;
         assert(op.lo < op.hi && op.hi <= s->_nkeys);
         // common prefix of a SORTED range == common prefix of its first & last suffix;
         // splits are the only place factoring tightens.
         auto       a     = s->suffix(op.lo);
         auto       b     = s->suffix(op.hi - 1);
         const auto extra = shared_prefix(a.data(), a.size(), b.data(), b.size());
         init_from(s, op.lo, op.hi, uint16_t(s->_cpfx_len + extra), a.data(), uint32_t(extra),
                   nullptr, nullptr);
      }

      // ── accessors ────────────────────────────────────────────────────────────────
      uint16_t num_keys() const noexcept { return _nkeys; }
      uint16_t capacity() const noexcept { return _cap; }
      uint8_t  num_cbases() const noexcept { return _ncb; }
      std::span<const uint8_t> common_prefix() const noexcept
      {
         return {cpfx_data(), _cpfx_len};
      }
      std::span<const uint8_t> suffix(uint32_t i) const noexcept { return record(i)->suffix(); }
      uint8_t                  value_type(uint32_t i) const noexcept
      {
         return record(i)->value_type;
      }
      /// vt_bytes payload (empty span for other types — deref the external instead).
      std::span<const uint8_t> value(uint32_t i) const noexcept
      {
         const inline_value* iv = record(i);
         return iv->value_type == inline_value::vt_bytes ? iv->value()
                                                         : std::span<const uint8_t>{};
      }
      bool is_external(uint32_t i) const noexcept
      {
         return record(i)->value_type == inline_value::vt_extern;
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

      /// Fingerprint of a FULL key — FNV-1a folded to one byte. Full-key (not suffix)
      /// so the column survives reprefix and split verbatim.
      static uint8_t fingerprint_of(const uint8_t* k, size_t n) noexcept
      {
         uint32_t h = 2166136261u;
         for (size_t i = 0; i < n; ++i)
            h = (h ^ k[i]) * 16777619u;
         return uint8_t(h ^ (h >> 8) ^ (h >> 16) ^ (h >> 24));
      }

      // ── point find: fingerprint column scan, verify by suffix compare ────────────
      /// @return entry index, or -1 if absent.
      int find(const uint8_t* k, uint32_t n) const noexcept
      {
         if (n < _cpfx_len || (_cpfx_len && std::memcmp(k, cpfx_data(), _cpfx_len) != 0))
            return -1;
         const uint8_t* s  = k + _cpfx_len;
         const uint32_t sn = n - _cpfx_len;
         const uint8_t  f  = fingerprint_of(k, n);
         const uint8_t* fpc = fp();
         uint32_t       i   = 0;
         while (i < _nkeys)
         {
            const int rel = ucc::find_byte(fpc + i, _nkeys - i, f);
            if (uint32_t(rel) == _nkeys - i)
               return -1;
            i += uint32_t(rel);
            const inline_value* iv = record(i);
            if (iv->suffix_len == sn && (sn == 0 || std::memcmp(iv->data, s, sn) == 0))
               return int(i);
            ++i;
         }
         return -1;
      }

      // ── lower_bound: SIMD prefix-column scan + equal-run compare ─────────────────
      /// @return smallest index whose full key >= (k,n); _nkeys if none.
      int lower_bound(const uint8_t* k, uint32_t n) const noexcept
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
         int i = ucc::lower_bound_u16_padded(pfx(), _nkeys, stem2(s, sn));
         while (i < int(_nkeys) && cmp_suffix(uint32_t(i), s, sn) < 0)
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
         if (n - _cpfx_len > max_suffix_len)
            return can_apply_mode::none;
         if (ins.vtype == inline_value::vt_bytes &&
             int(n - _cpfx_len + ins.value.size()) > max_inline_entry)
            return can_apply_mode::none;  // engine should have externalized this entry
         if (ins.vtype == inline_value::vt_extern && !can_add_address(ins.ext))
            return can_apply_mode::none;  // 17th cline => split early
         const uint32_t need = uint32_t(record_bytes(n - _cpfx_len, ins)) +
                               (_nkeys == _cap ? 5u * 4u : 0u);
         return free_bytes(cols_end()) >= need ? can_apply_mode::modify : can_apply_mode::none;
      }

      /// @pre can_apply(ins) == modify, and the key is NOT present.
      /// @return the index the entry landed at.
      uint16_t apply(const insert_op& ins) noexcept
      {
         assert(can_apply(ins) == can_apply_mode::modify);
         assert(find(ins.key.data(), uint32_t(ins.key.size())) < 0);
         if (_nkeys == _cap)
            grow_cap();
         const uint8_t* s   = ins.key.data() + _cpfx_len;
         const uint32_t sn  = uint32_t(ins.key.size()) - _cpfx_len;
         const uint16_t off = alloc_record(s, sn, ins);
         const uint16_t pos = uint16_t(lower_bound(ins.key.data(), uint32_t(ins.key.size())));
         open_gap(pos);
         fp()[pos]   = fingerprint_of(ins.key.data(), ins.key.size());
         pfx()[pos]  = stem2(s, sn);
         vals()[pos] = off;
         ++_nkeys;
         return pos;
      }

      can_apply_mode can_apply(const update_op& upd) const noexcept
      {
         assert(upd.index < _nkeys);
         const inline_value* iv = record(upd.index);
         if (upd.vtype == inline_value::vt_bytes &&
             int(iv->suffix_len + upd.value.size()) > max_inline_entry)
            return can_apply_mode::none;  // engine should have externalized this value
         if (upd.vtype == inline_value::vt_extern && !can_add_address(upd.ext))
            return can_apply_mode::none;
         const uint32_t npay = payload_bytes(upd.value.size(), upd.vtype);
         if (npay <= iv->value_len)
            return can_apply_mode::modify;  // rewrite in the existing record
         const uint32_t need = 3u + iv->suffix_len + npay;
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
         nv->suffix_len = iv->suffix_len;
         nv->value_len  = uint8_t(npay);
         nv->value_type = upd.vtype;
         std::memcpy(nv->data, iv->data, iv->suffix_len);
         write_payload(nv->data + nv->suffix_len, upd);
         vals()[upd.index] = tail_used();
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
      }

      // ── rebuild-fit checks (clone vs split decision) ─────────────────────────────
      bool rebuilt_fits(const insert_op& ins) const noexcept
      {
         const uint32_t n   = uint32_t(ins.key.size());
         const uint32_t ncp = shared_prefix(ins.key.data(), n, cpfx_data(), _cpfx_len);
         const uint32_t sn  = n - ncp;
         if (sn > max_suffix_len || uint32_t(_cpfx_len - ncp) + max_live_suffix() > max_suffix_len)
            return false;
         uint32_t cb = live_cbase_count();
         if (ins.vtype == inline_value::vt_extern)
            cb += 1;  // conservative: may need a fresh base
         if (cb > max_cbases)
            return false;
         const uint32_t content = head_bytes() + 5u * round_cap(_nkeys + 1u) + ncp +
                                  (live_tail() - _cpfx_len) +
                                  uint32_t(_cpfx_len - ncp) * _nkeys + record_bytes(sn, ins);
         return content <= page_bytes;
      }

      bool rebuilt_fits(const update_op& upd) const noexcept
      {
         const inline_value* iv   = record(upd.index);
         const uint32_t      npay = payload_bytes(upd.value.size(), upd.vtype);
         if (upd.vtype == inline_value::vt_bytes && int(iv->suffix_len + npay) > max_inline_entry)
            return false;
         if (upd.vtype == inline_value::vt_extern && live_cbase_count() + 1 > max_cbases)
            return false;
         const uint32_t content = head_bytes() + 5u * round_cap(_nkeys) + live_tail() + npay -
                                  iv->value_len;
         return content <= page_bytes;
      }

      bool can_add_address(sal::ptr_address addr) const noexcept
      {
         const uint32_t base = *addr & ~0xFu;
         if (ucc::find_u32x16(reinterpret_cast<const uint32_t*>(_cbases), max_cbases, base) <
             int(max_cbases))
            return true;
         return _ncb < max_cbases;
      }

      // ── split-divider selection ──────────────────────────────────────────────────
      // The shortest separator near the median, last byte snapped toward trailing
      // zeros (power-of-2 friendly) — dividers feed the lower-bound radix above, so
      // shorter + rounder is structurally cheaper up there. Bounded skew: candidates
      // come from the middle window [3n/8, 5n/8].
      struct split_pos
      {
         uint16_t index;    // right half = [index, nkeys)
         uint16_t div_len;  // full-key-space divider length
         uint8_t  div[520];  // cpfx(<=255) + separator(<=256)
         std::span<const uint8_t> divider() const noexcept { return {div, div_len}; }
      };
      split_pos get_split_pos() const noexcept
      {
         assert(_nkeys >= 2);
         const uint32_t n  = _nkeys;
         uint32_t       lo = (n * 3) / 8, hi = (n * 5) / 8;
         if (lo < 1)
            lo = 1;
         if (hi >= n)
            hi = n - 1;
         if (hi < lo)
            hi = lo;
         uint32_t best_i = lo, best_len = ~0u, best_tz = 0, best_dist = ~0u;
         uint8_t  best_byte = 0;
         for (uint32_t i = lo; i <= hi; ++i)
         {
            auto     a = suffix(i - 1), b = suffix(i);
            uint32_t p = shared_prefix(a.data(), a.size(), b.data(), b.size());
            assert(p < b.size());  // strict order: b can't be a prefix of a
            // separator = b[0..p] + v, valid v in (a[p], b[p]] — or [0, b[p]] when a
            // is a prefix of b. Maximize trailing zeros of v.
            const int     lo_excl = p < a.size() ? int(a[p]) : -1;
            const uint8_t hi_v    = b[p];
            uint8_t       v       = hi_v;
            uint32_t      tz      = 0;
            for (int z = 8; z >= 1; --z)
            {
               const uint8_t cand = uint8_t(hi_v & ~((1u << z) - 1u));
               if (int(cand) > lo_excl)
               {
                  v  = cand;
                  tz = uint32_t(z);
                  break;
               }
            }
            const uint32_t len  = p + 1;
            const uint32_t dist = i > n / 2 ? i - n / 2 : n / 2 - i;
            if (len < best_len || (len == best_len && tz > best_tz) ||
                (len == best_len && tz == best_tz && dist < best_dist))
            {
               best_i    = i;
               best_len  = len;
               best_tz   = tz;
               best_dist = dist;
               best_byte = v;
            }
         }
         split_pos r;
         r.index = uint16_t(best_i);
         auto b  = suffix(best_i);
         std::memcpy(r.div, cpfx_data(), _cpfx_len);
         std::memcpy(r.div + _cpfx_len, b.data(), best_len - 1);
         r.div[_cpfx_len + best_len - 1] = best_byte;
         r.div_len                       = uint16_t(_cpfx_len + best_len);
         return r;
      }

      // ── diagnostics ──────────────────────────────────────────────────────────────
      [[nodiscard]] bool validate_invariants() const noexcept
      {
         if (_cap % 4 || _nkeys > _cap || _ncb > max_cbases)
            return false;
         if (head_size() + tail_used() > size())
            return false;
         uint32_t live = _cpfx_len;
         uint8_t  kbuf[520];
         uint8_t  pbuf[520];
         uint32_t plen = 0;
         for (uint32_t i = 0; i < _nkeys; ++i)
         {
            const uint16_t v = vals()[i];
            if (v == 0 || v > tail_used() || v >= ext_code_base)
               return false;
            const inline_value* iv = record(i);
            live += uint32_t(iv->byte_size());
            if (pfx()[i] != stem2(iv->data, iv->suffix_len))
               return false;
            const uint32_t klen = full_key(i, kbuf);
            if (fp()[i] != fingerprint_of(kbuf, klen))
               return false;
            if (i && lex_cmp(pbuf, plen, kbuf, klen) >= 0)
               return false;  // strict ascending, no dups
            std::memcpy(pbuf, kbuf, klen);
            plen = klen;
            if (iv->value_type == inline_value::vt_extern)
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
      uint16_t*       pfx() noexcept { return reinterpret_cast<uint16_t*>(cols() + _cap); }
      const uint16_t* pfx() const noexcept
      {
         return reinterpret_cast<const uint16_t*>(cols() + _cap);
      }
      uint16_t*       vals() noexcept { return reinterpret_cast<uint16_t*>(cols() + 3u * _cap); }
      const uint16_t* vals() const noexcept
      {
         return reinterpret_cast<const uint16_t*>(cols() + 3u * _cap);
      }
      const void* cols_end() const noexcept { return cols() + 5u * _cap; }

      const uint8_t* cpfx_data() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(resolve(tail_ptr<const uint8_t>{
             _cpfx_pos.offset_from_tail}));
      }

      uint32_t live_tail() const noexcept { return tail_used() - _dead; }

      // ── small helpers ────────────────────────────────────────────────────────────
      static uint16_t stem2(const uint8_t* s, size_t n) noexcept
      {
         return uint16_t((n > 0 ? uint16_t(s[0]) << 8 : 0) | (n > 1 ? uint16_t(s[1]) : 0));
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
      int cmp_suffix(uint32_t i, const uint8_t* s, uint32_t sn) const noexcept
      {
         const inline_value* iv = record(i);
         return lex_cmp(iv->data, iv->suffix_len, s, sn);
      }
      static uint32_t payload_bytes(size_t vlen, uint8_t vtype) noexcept
      {
         return vtype == inline_value::vt_extern ? 2u : uint32_t(vlen);
      }
      static uint32_t record_bytes(uint32_t sn, const insert_op& ins) noexcept
      {
         return 3u + sn + payload_bytes(ins.value.size(), ins.vtype);
      }
      /// Largest live suffix length (rebuild prepend must keep every suffix <= 255).
      uint32_t max_live_suffix() const noexcept
      {
         uint32_t m = 0;
         for (uint32_t i = 0; i < _nkeys; ++i)
            if (record(i)->suffix_len > m)
               m = record(i)->suffix_len;
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
            if (iv->value_type != inline_value::vt_extern)
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
         assert(iv->value_type == inline_value::vt_extern && iv->value_len == 2);
         uint16_t code;
         std::memcpy(&code, iv->data + iv->suffix_len, 2);
         assert(code >= ext_code_base);
         return sal::ptr_address(*_cbases[(code >> 4) & 0xF] + (code & 0xF));
      }
      /// @pre can_add_address(addr)
      uint16_t encode_extern(sal::ptr_address addr) noexcept
      {
         const uint32_t base = *addr & ~0xFu;
         int idx = ucc::find_u32x16(reinterpret_cast<const uint32_t*>(_cbases), max_cbases, base);
         if (idx == int(max_cbases))
         {
            assert(_ncb < max_cbases);
            idx                  = _ncb;
            _cbases[_ncb++]      = sal::ptr_address(base);
         }
         return uint16_t(ext_code_base | (uint32_t(idx) << 4) | (*addr & 0xFu));
      }
      void write_payload(uint8_t* dst, const auto& op) noexcept  // insert_op or update_op
      {
         if (op.vtype == inline_value::vt_extern)
         {
            const uint16_t code = encode_extern(op.ext);
            std::memcpy(dst, &code, 2);
         }
         else
            std::memcpy(dst, op.value.data(), op.value.size());
      }
      uint16_t alloc_record(const uint8_t* s, uint32_t sn, const insert_op& ins) noexcept
      {
         assert(ins.vtype != inline_value::vt_bytes || int(sn + ins.value.size()) <=
                    max_inline_entry);
         auto* iv = reinterpret_cast<inline_value*>(
             alloc_tail(uint16_t(record_bytes(sn, ins))));
         iv->suffix_len = uint8_t(sn);
         iv->value_len  = uint8_t(payload_bytes(ins.value.size(), ins.vtype));
         iv->value_type = ins.vtype;
         std::memcpy(iv->data, s, sn);
         write_payload(iv->data + sn, ins);
         return tail_used();
      }

      // ── column splice (in-place) ─────────────────────────────────────────────────
      void open_gap(uint32_t pos) noexcept
      {
         assert(_nkeys < _cap);
         std::memmove(fp() + pos + 1, fp() + pos, _nkeys - pos);
         std::memmove(pfx() + pos + 1, pfx() + pos, 2u * (_nkeys - pos));
         std::memmove(vals() + pos + 1, vals() + pos, 2u * (_nkeys - pos));
      }
      void close_gap(uint32_t pos) noexcept
      {
         std::memmove(fp() + pos, fp() + pos + 1, _nkeys - pos - 1);
         std::memmove(pfx() + pos, pfx() + pos + 1, 2u * (_nkeys - pos - 1));
         std::memmove(vals() + pos, vals() + pos + 1, 2u * (_nkeys - pos - 1));
      }
      /// +4 column capacity within the free middle; rightmost column moves first.
      void grow_cap() noexcept
      {
         const uint16_t nc = _cap + 4;
         assert(head_bytes() + 5u * nc <= size() - tail_used());
         char* base = cols();
         std::memmove(base + 3u * nc, base + 3u * _cap, 2u * _nkeys);  // vals
         std::memmove(base + nc, base + _cap, 2u * _nkeys);            // pfx
         _cap = nc;
      }

      void init_empty() noexcept
      {
         heap_alloc_pos = tail_ptr<>{0};  // compact_to enters here without init_page
         _nkeys         = 0;
         _cap           = round_cap(0);
         _ncb           = 0;
         _cpfx_len      = 0;
         _dead          = 0;
         _cpfx_pos      = tail_ptr<>{0};
         for (auto& b : _cbases)
            b = sal::null_ptr_address;
      }

      // ── the ONE rebuild engine (shared splice) ───────────────────────────────────
      // Every rebuilding ctor and compact_to funnels here: copy src entries [lo,hi)
      // under a new common prefix (extra_src/extra_n = bytes appended to src's cpfx
      // when tightening; ncp < src->_cpfx_len means loosening and suffixes re-expand),
      // optionally splicing one insert, one update, or one removal. Dead tail bytes
      // and dead cbases never survive a rebuild.
      void init_from(const leaf_page* src, uint32_t lo, uint32_t hi, uint16_t ncp,
                     const uint8_t* extra_src, uint32_t extra_n, const insert_op* ins,
                     const update_op* upd, int drop = -1) noexcept
      {
         init_empty();
         _version_history = src->_version_history;
         const uint32_t cnt = (hi - lo) + (ins ? 1u : 0u) - (drop >= 0 ? 1u : 0u);
         _cap               = round_cap(cnt);
         assert(head_bytes() + 5u * _cap <= size());

         // new common prefix bytes: src's cpfx truncated-or-extended
         _cpfx_len = uint8_t(ncp);
         if (ncp)
         {
            uint8_t* d = reinterpret_cast<uint8_t*>(alloc_tail(uint16_t(ncp)));
            const uint32_t keep = ncp < src->_cpfx_len ? ncp : src->_cpfx_len;
            std::memcpy(d, src->cpfx_data(), keep);
            if (ncp > src->_cpfx_len)
            {
               assert(extra_n == uint32_t(ncp - src->_cpfx_len) && extra_src);
               std::memcpy(d + keep, extra_src, extra_n);
            }
            _cpfx_pos = tail_pos();
         }
         const int prepend = int(src->_cpfx_len) - int(ncp);  // >0: suffixes re-expand

         uint32_t w = 0;
         auto     emit_src = [&](uint32_t si)
         {
            const inline_value* iv  = src->record(si);
            const uint32_t      sn  = (prepend >= 0) ? iv->suffix_len + uint32_t(prepend)
                                                     : iv->suffix_len - uint32_t(-prepend);
            uint8_t pay = iv->value_len;
            if (upd && si == upd->index)
               pay = uint8_t(payload_bytes(upd->value.size(), upd->vtype));
            assert(sn <= max_suffix_len);
            auto* nv = reinterpret_cast<inline_value*>(alloc_tail(uint16_t(3u + sn + pay)));
            nv->suffix_len = uint8_t(sn);
            nv->value_len  = pay;
            if (prepend > 0)
            {
               std::memcpy(nv->data, src->cpfx_data() + ncp, uint32_t(prepend));
               std::memcpy(nv->data + prepend, iv->data, iv->suffix_len);
            }
            else
               std::memcpy(nv->data, iv->data - prepend, sn);  // -prepend = strip count
            if (upd && si == upd->index)
            {
               nv->value_type = upd->vtype;
               write_payload(nv->data + sn, *upd);
            }
            else if (iv->value_type == inline_value::vt_extern)
            {
               nv->value_type      = inline_value::vt_extern;
               const uint16_t code = encode_extern(src->decode_extern(iv));  // re-base compactly
               std::memcpy(nv->data + sn, &code, 2);
            }
            else
            {
               nv->value_type = iv->value_type;
               std::memcpy(nv->data + sn, iv->data + iv->suffix_len, iv->value_len);
            }
            fp()[w]   = src->fp()[si];  // full-key fingerprints survive any refactoring
            pfx()[w]  = stem2(nv->data, sn);
            vals()[w] = tail_used();
            ++w;
         };

         // splice position for the insert (in src index space)
         uint32_t ipos = hi;
         if (ins)
         {
            ipos = uint32_t(src->lower_bound(ins->key.data(), uint32_t(ins->key.size())));
            assert(src->find(ins->key.data(), uint32_t(ins->key.size())) < 0);
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
         assert(w == cnt);
         assert(head_size() + tail_used() <= size());
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
         pfx()[w]          = stem2(k + _cpfx_len, sn);
      }
   } __attribute__((packed));

   static_assert(sizeof(leaf_page) == 96, "paged(22) + fields(10) + cbases(64)");
}  // namespace heart

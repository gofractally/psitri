#pragma once
#include <cstring>

#include <heart/inline_value.hpp>
#include <heart/paged.hpp>
#include <heart/radix_type_ids.hpp>
#include <heart/smart_address.hpp>  // tail_ptr
#include <ucc/fast_memcpy.hpp>      // memcpy_aligned_64byte
#include <ucc/lower_bound.hpp>      // lower_bound_u16_padded (SIMD), find
#include <ucc/round.hpp>

namespace heart
{
   /**
    * The "last-mile" terminal: a bucket of variable-length (suffix, value) entries kept as
    * a FLAT, lexicographically-sorted SoA inside this node's own tail heap. A plain `paged`
    * node — a page like any other; NOT a radix node (it terminates the radix, doesn't
    * route). It collapses the sparse, early-diverging terminal region that a pure byte-radix
    * over-nodes.
    *
    * Layout (one cache-friendly, COW-as-a-unit object):
    *   [ bucket_node header | div[n] (u16) | slot[n] (u16 tail_ptr) | free | ...entries... ]
    * div[i] = the big-endian first 2 bytes of entry i's suffix (zero-padded). The dividers
    * are a contiguous u16 array searched BRANCHLESSLY with SIMD (ucc::lower_bound_u16) — the
    * fast scan for the common small/spread bucket. Because div is the first 2 bytes, sorting
    * by (div, then full suffix) IS lexicographic, so the array is lex-sorted: in-order walk,
    * lower_bound, and full-suffix-compare on the (usually short) equal-divider run all fall
    * out. Entries (inline_value: full suffix + value) live in the tail heap and never move;
    * insert shifts only the small u16 index arrays. Overflow (no room) -> caller splits to
    * radix. (v1 stored full suffixes inline via inline_value, u8 lengths; oversized values
    * would spill to an external leaf — later.)
    */
   struct bucket_node : paged
   {
      static constexpr sal::header_type type_id    = radix_type_ids::bucket;
      static constexpr uint32_t         page_bytes = 4096;  // target page; fills then splits

      uint16_t nent;  // entry count

      // ── page lifecycle (concrete) ───────────────────────────────────────────────────
      // The head region is the fixed struct + the two forward index arrays (div[], slot[]);
      // the tail region is the (used) entry heap. Everything in between is free space we never
      // touch — COW and compaction copy ONLY head + used-tail (two fast memcpys), never the hole.
      static constexpr uint32_t head_bytes() noexcept { return uint32_t(sizeof(bucket_node)); }
      uint32_t head_size() const noexcept { return uint32_t(sizeof(bucket_node)) + 4u * nent; }
      static uint32_t alloc_size() noexcept  // the 4 KB working page (room to grow → 1 grow/txn)
      {
         return ucc::round_up_multiple<sal::cacheline_size>(page_bytes);
      }
      // COW always expands back to the full working page so a unique-mode insert run mutates in
      // place afterward (no repeated grow phases); the compactor later shrinks it to fit.
      uint32_t cow_size() const noexcept { return alloc_size(); }
      uint32_t compact_size() const noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(head_size() + tail_used());
      }
      void copy_to(sal::alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
      void compact_to(sal::alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }

      template <typename F>
      void for_each_child(F&&) const  // no SAL children (entries are inline) -> destroy is no-op
      {
      }

      bucket_node(uint32_t asize, sal::ptr_address_seq seq) noexcept
      {
         this->init_page(asize, type_id, seq);
         nent = 0;
      }

      // ── index accessors ─────────────────────────────────────────────────────────────
      uint16_t*       div() noexcept { return reinterpret_cast<uint16_t*>(idx_base()); }
      const uint16_t* div() const noexcept { return reinterpret_cast<const uint16_t*>(idx_base()); }
      uint16_t*       slot() noexcept { return div() + nent; }
      const uint16_t* slot() const noexcept { return div() + nent; }
      const inline_value* entry(int i) const noexcept
      {
         return reinterpret_cast<const inline_value*>(reinterpret_cast<const char*>(this) +
                                                      this->size() - slot()[i]);
      }
      static uint16_t stem2(const uint8_t* s, int n) noexcept  // big-endian first 2 bytes
      {
         return uint16_t((n > 0 ? uint16_t(s[0]) << 8 : 0) | (n > 1 ? uint16_t(s[1]) : 0));
      }
      int cmp(int i, const uint8_t* s, int sn) const noexcept  // entry i's suffix vs (s,sn)
      {
         const inline_value* iv = entry(i);
         const int           el = iv->suffix_len;
         const int           m  = el < sn ? el : sn;
         const int           c  = m ? std::memcmp(iv->data, s, size_t(m)) : 0;
         if (c)
            return c < 0 ? -1 : 1;
         return el == sn ? 0 : (el < sn ? -1 : 1);
      }

      // ── find ───────────────────────────────────────────────────────────────────────
      const inline_value* find(const uint8_t* s, int sn) const noexcept
      {
         const uint16_t  q2 = stem2(s, sn);
         const uint16_t* d  = div();
         int             i  = ucc::lower_bound_u16_padded(d, nent, q2);  // first div >= q2
         for (; i < nent && d[i] == q2; ++i)
         {
            const inline_value* iv = entry(i);
            if (iv->suffix_len == sn && (sn == 0 || std::memcmp(iv->data, s, size_t(sn)) == 0))
               return iv;
         }
         return nullptr;
      }

      // ── lower_bound: smallest entry index whose suffix is >= (s,sn); nent if none ─────
      int lower_bound(const uint8_t* s, int sn) const noexcept
      {
         const uint16_t* d = div();
         int             i = ucc::lower_bound_u16_padded(d, nent, stem2(s, sn));  // skip div < q2
         while (i < nent && cmp(i, s, sn) < 0)  // then full-compare within/after the run
            ++i;
         return i;
      }

      // ── insert: 1 inserted, -1 updated, 0 OVERFLOW (caller splits) ────────────────────
      int insert(const uint8_t* s, int sn, const uint8_t* v, int vn) noexcept
      {
         const uint16_t q2  = stem2(s, sn);
         uint16_t*      d   = div();
         int            pos = ucc::lower_bound_u16_padded(d, nent, q2);
         for (; pos < nent && d[pos] == q2; ++pos)  // walk the equal-divider run, full-compare
         {
            const int c = cmp(pos, s, sn);
            if (c == 0)  // update in place if the value fits the existing slot, else re-alloc
            {
               const inline_value* iv = entry(pos);
               if (iv->value_len >= vn)
               {
                  auto* w     = const_cast<inline_value*>(iv);
                  w->value_len = uint8_t(vn);
                  std::memcpy(w->data + w->suffix_len, v, size_t(vn));
                  return -1;
               }
               const uint16_t off = alloc_entry(s, sn, v, vn);
               if (!off)
                  return 0;
               slot()[pos] = off;  // old slot leaks (rare: grow-update)
               return -1;
            }
            if (c > 0)
               break;
         }
         // new entry at `pos` (keeps the array lex-sorted)
         const int need = 4 + inline_value::byte_size(sn, vn);  // new div+slot + entry
         if (int(free_index_bytes()) < need)
            return 0;  // page full
         const uint16_t off = alloc_entry(s, sn, v, vn);
         // grow the index: slot block shifts right by one u16 (the new div), then open a gap
         // at `pos` in both arrays (move highest addresses first).
         d                 = div();  // (alloc_entry didn't move the head arrays)
         uint16_t* s_old   = d + nent;
         uint16_t* s_new   = d + nent + 1;
         std::memmove(s_new + pos + 1, s_old + pos, size_t(nent - pos) * 2);  // slot upper
         std::memmove(s_new, s_old, size_t(pos) * 2);                          // slot lower
         std::memmove(d + pos + 1, d + pos, size_t(nent - pos) * 2);           // div upper
         d[pos]     = q2;
         s_new[pos] = off;
         ++nent;
         return 1;
      }

      // ── remove: true if (s,sn) was found+deleted ─────────────────────────────────────
      // Drops the entry's div+slot index pair (keeping both arrays sorted+aligned); the
      // inline entry's tail bytes orphan (reclaimed on the next compact/COW). No SAL
      // children, so nothing to release.
      bool remove(const uint8_t* s, int sn) noexcept
      {
         const uint16_t q2  = stem2(s, sn);
         uint16_t*      d   = div();
         int            pos = ucc::lower_bound_u16_padded(d, nent, q2);
         for (; pos < nent && d[pos] == q2; ++pos)
         {
            const int c = cmp(pos, s, sn);
            if (c == 0)
            {
               // close the gap at pos in div[], then restow slot[] one u16 to the left
               // (div shrank by one) dropping slot[pos]. Old slot block is at d+nent.
               std::memmove(d + pos, d + pos + 1, size_t(nent - 1 - pos) * 2);
               std::memmove(d + nent - 1, d + nent, size_t(pos) * 2);
               std::memmove(d + nent - 1 + pos, d + nent + pos + 1, size_t(nent - 1 - pos) * 2);
               --nent;
               return true;
            }
            if (c > 0)
               break;
         }
         return false;
      }

      // ── in-order visit: f(const inline_value&) ──────────────────────────────────────
      template <typename F>
      void for_each(F&& f) const
      {
         for (int i = 0; i < nent; ++i)
            f(*entry(i));
      }

     private:
      char*       idx_base() noexcept { return reinterpret_cast<char*>(this) + sizeof(bucket_node); }
      const char* idx_base() const noexcept
      {
         return reinterpret_cast<const char*>(this) + sizeof(bucket_node);
      }
      // bytes free between the forward index arrays (div+slot = 4*nent) and the tail heap.
      uint32_t free_index_bytes() const noexcept
      {
         return (this->size() - this->tail_used()) - (uint32_t(sizeof(bucket_node)) + 4u * nent);
      }
      uint16_t alloc_entry(const uint8_t* s, int sn, const uint8_t* v, int vn) noexcept
      {
         const int need = inline_value::byte_size(sn, vn);
         auto*     iv   = reinterpret_cast<inline_value*>(this->alloc_tail(uint16_t(need)));
         iv->suffix_len = uint8_t(sn);
         iv->value_len  = uint8_t(vn);
         iv->value_type = 0;
         std::memcpy(iv->data, s, size_t(sn));
         std::memcpy(iv->data + sn, v, size_t(vn));
         return this->tail_used();  // offset-from-tail (== a tail_ptr)
      }
   } __attribute__((packed));
}  // namespace heart

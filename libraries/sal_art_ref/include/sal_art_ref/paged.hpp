#pragma once
#include <cstdint>
#include <cstring>

#include <sal_art_ref/smart_address.hpp>  // tail_ptr
#include <sal_art_ref/versioned.hpp>
#include <ucc/fast_memcpy.hpp>  // memcpy_aligned_64byte

namespace sal_art_ref
{
   /**
    * The arena layer: treats the SAL allocation as a memory arena on top of the
    * MVCC `versioned` header. A "heap" grows DOWNWARD from the tail (one-past-the-
    * end, == alloc_header::tail()); tail blocks are addressed by tail_ptr (an
    * offset from the tail) so the reference survives relocation. This is what lets
    * a node store inline values/suffixes on its own page and reference them with a
    * smart_address.
    *
    *   [ fixed header | head-growing arrays | free | ... tail heap ... | tail() ]
    *                                          ^ tail_top()              ^
    *
    * Layout: versioned (20 B) + heap_alloc_pos (2 B) = 22 B, packed.
    */
   struct paged : versioned
   {
      tail_ptr<> heap_alloc_pos;  // bytes consumed from the tail so far (heap depth)

      /// Initialize as an empty versioned arena of `asize` bytes (cacheline multiple).
      void init_page(uint32_t asize, sal::header_type t, sal::ptr_address_seq seq) noexcept
      {
         init_versioned(asize, t, seq);
         heap_alloc_pos = tail_ptr<>{0};
      }

      using sal::alloc_header::tail;  // char* tail() == (char*)this + size()

      char*       tail_top() noexcept { return tail() - heap_alloc_pos.offset_from_tail; }
      const char* tail_top() const noexcept { return tail() - heap_alloc_pos.offset_from_tail; }

      uint16_t tail_used() const noexcept { return heap_alloc_pos.offset_from_tail; }

      /// Free bytes between the end of the head region (`head_end`) and the tail heap.
      uint32_t free_bytes(const void* head_end) const noexcept
      {
         return uint32_t(tail_top() - static_cast<const char*>(head_end));
      }
      bool can_alloc_tail(uint16_t n, const void* head_end) const noexcept
      {
         return free_bytes(head_end) >= n;
      }

      /// Reserve `n` bytes from the tail; returns the block start. Caller ensures room.
      char* alloc_tail(uint16_t n) noexcept
      {
         heap_alloc_pos.offset_from_tail += n;
         return tail_top();
      }

      /// A tail_ptr to the current tail frontier (the block just alloc_tail'd).
      template <typename T = void>
      tail_ptr<T> tail_pos() const noexcept
      {
         return tail_ptr<T>{heap_alloc_pos.offset_from_tail};
      }

      /// Materialize a tail_ptr to a live pointer within this page.
      template <typename T>
      T* resolve(tail_ptr<T> tp) noexcept
      {
         return reinterpret_cast<T*>(tail() - tp.offset_from_tail);
      }
      template <typename T>
      const T* resolve(tail_ptr<T> tp) const noexcept
      {
         return reinterpret_cast<const T*>(tail() - tp.offset_from_tail);
      }

      /// Resolve a LOCAL (inline) branch to a live pointer in this page's tail heap.
      /// Precondition: sa.is_tail() (a node-pointer branch resolves via as_pointer()
      /// through the allocator, not the page).
      char* resolve(smart_address sa) noexcept
      {
         assert(sa.is_tail());
         return tail() - sa.tail_offset();
      }
      const char* resolve(smart_address sa) const noexcept
      {
         assert(sa.is_tail());
         return tail() - sa.tail_offset();
      }

      // ── COW / compaction: copy a page's LIVE contents, skipping the free hole ────────
      // SIMD copy for an arbitrary length: 64-byte aligned bulk + scalar remainder. Never
      // over-reads/over-writes, so it is safe even against a tightly-sized compaction dst.
      static void fast_copy(void* dst, const void* src, uint32_t n) noexcept
      {
         const uint32_t bulk = n & ~uint32_t(63);
         ucc::memcpy_aligned_64byte(dst, src, bulk);
         if (n & 63u)
            std::memcpy(static_cast<char*>(dst) + bulk, static_cast<const char*>(src) + bulk,
                        n & 63u);
      }
      // Clone this page's live contents into `dst` (sized cow_size for COW, or compact_size
      // for compaction) with TWO copies — the head [0,hsize) and the used tail heap — skipping
      // the free middle. `dst` keeps its OWN SAL header (checksum/address/size); version,
      // heap_alloc_pos, index and entries come from this page. Tail offsets are tail()-relative
      // so they stay valid even though dst->tail() lands at a different (smaller) address.
      void blit_head_tail(sal::alloc_header* dst, uint32_t hsize) const noexcept
      {
         const sal::alloc_header saved = *dst;                 // dst's own checksum/address/size
         fast_copy(dst, this, hsize);                          // head: fixed struct + index
         std::memcpy(dst, &saved, sizeof(sal::alloc_header));  // restore SAL header
         auto*          out = static_cast<paged*>(dst);
         const uint16_t tu  = tail_used();
         fast_copy(out->tail() - tu, this->tail() - tu, tu);   // tail: used heap only
         out->clear_checksum();
      }
   } __attribute__((packed));

   static_assert(sizeof(paged) == 22, "versioned(20) + heap_alloc_pos(2)");
}  // namespace sal_art_ref

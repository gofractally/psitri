// part_alloc — plugs the part arena into hkv::kv_tree via hkv's allocator
// protocol (hkv_handle + hkv_base + allocate/deallocate). The handle is the
// 4-byte indexed line_ptr from hkv's Stage C pool work: [28-bit line index]
// [4-bit kind tag], little-endian — bit-identical to part's ptr design, so the
// tree's deref is base + (idx << 7) and descent dispatch never loads a header
// (SPEC §2.7). The node layout and descent path are hkv's, verbatim: that is
// the read-performance contract.
//
// Allocation routes through a part::session (first-fit-forward from the txn
// floor); deallocation releases each line's refcount (1 -> 0 frees). Under
// exclusive ownership that is exactly hkv's free; under COW the same call
// safely just drops a reference when a line is shared.
#pragma once
#include <part/arena.hpp>
#include <part/hkv/tagged_ptr.hpp>

namespace part
{
   namespace detail
   {
      // 4-byte indexed branch handle (hkv line_ptr): same byte-array shape as
      // packed_ptr_t (memcpy load/store, unaligned-safe in packed node arrays).
      struct line_ptr
      {
         static constexpr bool indexed = true;
         uint8_t               b[4];

         static line_ptr null() noexcept
         {
            line_ptr p;
            std::memset(p.b, 0xFF, 4);  // tag nibble becomes 0xF == K::null
            return p;
         }
         static line_ptr from_off(size_t byte_off, hkv::detail::K k) noexcept
         {
            assert((byte_off & 0x7f) == 0 && "part: node offsets are line-aligned");
            assert(byte_off >> 35 == 0 && "part: offset exceeds the 28-bit line index");
            line_ptr       p;
            const uint32_t raw = uint32_t(byte_off >> 3) | uint8_t(k);  // (off>>7)<<4 == off>>3
            std::memcpy(p.b, &raw, 4);
            return p;
         }
         uint64_t raw() const noexcept
         {
            uint32_t v;
            std::memcpy(&v, b, 4);
            return v;
         }
         hkv::detail::K tag() const noexcept { return hkv::detail::K(b[0] & 0xF); }
         bool           is_null() const noexcept { return tag() == hkv::detail::K::null; }

         line_idx index() const noexcept { return line_idx(raw() >> 4); }
      };
      static_assert(sizeof(line_ptr) == 4, "line_ptr is exactly 4 bytes");
   }  // namespace detail

   template <class T>
   struct part_alloc
   {
      using value_type = T;
      using hkv_handle = detail::line_ptr;

      arena*   a   = nullptr;  // deref base + refcounts (any thread)
      session* ses = nullptr;  // allocation context (writer thread only);
                               // null for read-only snapshot views
      bool defer = false;      // writer-side trees may hand release-to-zero
                               // cascades to the background releaser

      part_alloc() = default;
      explicit part_alloc(session* s) noexcept : a(&s->store()), ses(s), defer(true) {}
      explicit part_alloc(session& s) noexcept : a(&s.store()), ses(&s), defer(true) {}
      explicit part_alloc(arena* ar) noexcept : a(ar) {}  // read-only views
      part_alloc(arena* ar, bool d) noexcept : a(ar), defer(d) {}  // writer temp views
      template <class U>
      part_alloc(const part_alloc<U>& o) noexcept : a(o.a), ses(o.ses), defer(o.defer)
      {
      }

      const std::byte* hkv_base() const noexcept { return a ? a->base() : nullptr; }

      static constexpr size_t lines_(size_t bytes) noexcept
      {
         return (bytes + line_bytes - 1) / line_bytes;
      }
      T* allocate(size_t n)
      {
         static_assert(alignof(T) <= line_bytes, "part lines are 128-byte aligned");
         assert(ses && "read-only view allocators cannot allocate");
         const line_idx i = ses->alloc_lines(uint32_t(lines_(n * sizeof(T))));
         return static_cast<T*>(a->get(i));
      }
      void deallocate(T* p, size_t n) noexcept
      {
         const line_idx i =
             line_idx((reinterpret_cast<const std::byte*>(p) - a->base()) / line_bytes);
         if (ses)
            ses->release_run(i, uint32_t(lines_(n * sizeof(T))));
         else
            a->release_run(i, uint32_t(lines_(n * sizeof(T))));
      }

      template <class U>
      bool operator==(const part_alloc<U>& o) const noexcept
      {
         return a == o.a && ses == o.ses;
      }

      // ── COW refcount hooks (consumed by the tree's part-COW extension) ─────
      // All arena-level: safe from any thread (cascades often run on a reader
      // dropping the last snapshot reference).
      static constexpr bool hkv_cow = true;
      uint8_t hkv_ref(uint32_t line) const noexcept { return a->ref(line); }
      bool hkv_defer(uint32_t raw) noexcept { return defer && a->try_defer(raw); }
      bool    hkv_retain(uint32_t line) noexcept { return a->retain(line); }
      template <class D>
      bool hkv_release(uint32_t line, D&& d) noexcept(noexcept(d(line)))
      {
         return a->release(line, static_cast<D&&>(d));
      }
      void hkv_free_run(uint32_t line, uint32_t n) noexcept { a->release_run(line, n); }
   };
}  // namespace part

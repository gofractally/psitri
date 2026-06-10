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

      session* ses = nullptr;

      part_alloc() = default;
      explicit part_alloc(session* s) noexcept : ses(s) {}
      explicit part_alloc(session& s) noexcept : ses(&s) {}
      template <class U>
      part_alloc(const part_alloc<U>& o) noexcept : ses(o.ses)
      {
      }

      const std::byte* hkv_base() const noexcept { return ses ? ses->store().base() : nullptr; }

      static constexpr size_t lines_(size_t bytes) noexcept
      {
         return (bytes + line_bytes - 1) / line_bytes;
      }
      T* allocate(size_t n)
      {
         static_assert(alignof(T) <= line_bytes, "part lines are 128-byte aligned");
         const line_idx i = ses->alloc_lines(uint32_t(lines_(n * sizeof(T))));
         return static_cast<T*>(ses->store().get(i));
      }
      void deallocate(T* p, size_t n) noexcept
      {
         arena&         a = ses->store();
         const line_idx i =
             line_idx((reinterpret_cast<const std::byte*>(p) - a.base()) / line_bytes);
         const uint32_t nl = uint32_t(lines_(n * sizeof(T)));
         for (uint32_t k = 0; k < nl; ++k)
            a.release(i + k);
      }

      template <class U>
      bool operator==(const part_alloc<U>& o) const noexcept
      {
         return ses == o.ses;
      }
   };
}  // namespace part

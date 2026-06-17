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
      // index of byte among keys[0..nbranch), or -1.
      int find_index(uint8_t byte) const noexcept
      {
         int pos = int(ucc::lower_bound_padded(keys(), nbranch, byte));
         return (pos < int(nbranch) && keys()[pos] == byte) ? pos : -1;
      }
   };
}  // namespace gcart::st

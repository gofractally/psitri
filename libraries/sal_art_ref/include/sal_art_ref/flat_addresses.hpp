#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

#include <sal_art_ref/smart_address.hpp>
#include <sal/numbers.hpp>

namespace sal_art_ref
{
   /**
    * A rank-dense array of `smart_address`es stored FLAT (4 bytes each) — a drop-in
    * replacement for `compact_addresses<ByteCapacity>` with the IDENTICAL public API
    * (init/size/can_insert/insert/remove/at/set/get_hint/canonicalize/…) and the
    * same byte footprint, but WITHOUT cacheline-base deduplication.
    *
    * Why flat: measured pointer-child control-block sharing is only ~1.3–1.55
    * (below the ~1.7 byte break-even), and the overwhelming majority of branches in
    * real nodes are inline-tail terminals that live in the node's own heap and can
    * NEVER share a control-block cacheline. So compact's per-base overhead (a
    * scattered branch costs 5 B = 4-byte base + 1-byte branch) is a net loss vs a
    * flat 4 B, it caps a node at 16 distinct cachelines (forcing premature upgrades
    * to the dense tier), and it adds encode/decode (>>4 / <<4, base lookups, SIMD
    * free-slot scans) on every descend. Flat is 4 B/branch, uncapped to data_size/4,
    * O(1) at/set with no decode.
    *
    * Layout (sizeof == ByteCapacity, matching compact_addresses so it swaps into the
    * same space inside a radix_bitset with no size change):
    *
    *   [ _naddr | _pad | addr[0] addr[1] ... addr[n-1] | free ]
    *
    * Addresses are kept rank-dense (position i == the i-th logical entry, matching
    * bitset::rank in a radix node); insert/remove shift the small 4-byte array.
    *
    * Trivially copyable — no constructors; call init() after placement.
    */
   template <int ByteCapacity>
   struct flat_addresses
   {
      static_assert(ByteCapacity >= 6, "flat_addresses needs a 2-byte header + one 4-byte address");
      static constexpr int data_size = ByteCapacity - 2;
      static constexpr int capacity  = data_size / 4;  // max branches in this footprint

      void init() noexcept { _naddr = 0; }

      int size() const noexcept { return _naddr; }
      // ── compact_addresses API parity (flat has no bases: each address is its own) ──
      int  base_count() const noexcept { return _naddr; }
      int  live_base_count() const noexcept { return _naddr; }
      int  free_bytes() const noexcept { return data_size - _naddr * 4; }
      void canonicalize() noexcept {}  // nothing to compact — already dense

      /// Would adding one more address fit? (No sharing/canonicalize tricks — pure room.)
      bool can_insert(smart_address) const noexcept { return _naddr < capacity; }

      /// Insert `a` at logical position `pos` (0..size()). Caller invariants asserted.
      void insert(smart_address a, int pos) noexcept
      {
         assert(pos >= 0 && pos <= _naddr && _naddr < capacity);
         std::memmove(_data + (pos + 1) * 4, _data + pos * 4, size_t(_naddr - pos) * 4);
         store(pos, a);
         ++_naddr;
      }

      /// Remove the address at logical position `pos`, returning what was stored there.
      smart_address remove(int pos) noexcept
      {
         assert(pos >= 0 && pos < _naddr);
         const smart_address r = load(pos);
         std::memmove(_data + pos * 4, _data + (pos + 1) * 4, size_t(_naddr - pos - 1) * 4);
         --_naddr;
         return r;
      }

      smart_address at(int pos) const noexcept
      {
         assert(pos >= 0 && pos < _naddr);
         return load(pos);
      }

      /// Replace the address at `pos`. Flat never re-homes, so this always succeeds.
      bool set(int pos, smart_address a) noexcept
      {
         assert(pos >= 0 && pos < _naddr);
         store(pos, a);
         return true;
      }

      /**
       * Alloc hint: the branch addresses themselves (a pointer branch resolves to its
       * control-block cacheline via SAL's `>> 4`; an inline-tail branch falls in SAL's
       * reserved range so the hint path skips it). Same shape setlist nodes already use.
       * No dedup — duplicate cachelines are harmless (the hint scan rotates and wraps).
       */
      std::span<const sal::ptr_address> get_hint() const noexcept
      {
         return {reinterpret_cast<const sal::ptr_address*>(_data), static_cast<size_t>(_naddr)};
      }

     private:
      // 4-byte smart_address at an arbitrary byte offset → go through memcpy
      // (unaligned-safe; zero cost on x86-64/ARM64), exactly like compact_addresses.
      smart_address load(int i) const noexcept
      {
         smart_address a;
         std::memcpy(&a, _data + i * 4, 4);
         return a;
      }
      void store(int i, smart_address a) noexcept { std::memcpy(_data + i * 4, &a, 4); }

      uint8_t _naddr;
      uint8_t _pad;
      uint8_t _data[data_size];
   };

   static_assert(sizeof(flat_addresses<128>) == 128, "flat_addresses must be byte-exact");
   static_assert(sizeof(flat_addresses<112>) == 112);
   static_assert(sizeof(flat_addresses<120>) == 120);
}  // namespace sal_art_ref

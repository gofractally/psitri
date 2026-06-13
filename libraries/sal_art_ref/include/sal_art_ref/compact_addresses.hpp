#pragma once
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

#include <sal_art_ref/smart_address.hpp>
#include <sal/numbers.hpp>
#include <ucc/lower_bound.hpp>

namespace sal_art_ref
{
   /**
    * A dense, cacheline-deduplicating array of `ptr_address`es.
    *
    * Rationale (design3.md): a SAL control_block is 8 bytes and a cacheline is
    * 128 bytes, so 16 control blocks share one cacheline. A ptr_address therefore
    * decomposes into a 28-bit cacheline index (the high bits) and a 4-bit slot
    * (the low bits). When a radix node's children were allocated near each other
    * — which the SAL hint path actively encourages — many branches land on the
    * same cacheline, so storing the 28-bit cacheline base once and only a 4-bit
    * slot per branch packs far more pointers into a single node cacheline.
    *
    * Layout inside `_data` (data_size = ByteCapacity - 2 bytes):
    *
    *   [ base[0] base[1] ... base[nbase-1] | free | ... branch[1] branch[0] ]
    *     ^ grows toward the tail                          ^ block at the tail,
    *                                                        grows toward the head
    *
    *   base   = base_entry  { uint32_t share:4; uint32_t cacheline:28; } [4 bytes]
    *   branch = branch_entry{ uint8_t  base:4;  uint8_t  slot:4;       } [1 byte]
    *
    * `share` is the per-base reference count minus one (so 0 == one branch, 15 ==
    * a full 16-slot cacheline). It is the key to O(1) maintenance (the psitri
    * trick):
    *   - remove() reads the branch's base, and if that base had one reference left
    *     (share == 0) it stamps the whole slot `null_ptr_address`; otherwise it
    *     just decrements share. Either way it never drops/renumbers bases.
    *   - a free slot is therefore exactly the value `null_ptr_address`. insert()
    *     finds one with a single SIMD scan (ucc::find_u32x16) for that sentinel.
    *
    * Two facts make the sentinel safe. (1) A base word reinterpreted as a u32 is
    * `(cacheline << 4) | share`; a *live* base can never equal 0xFFFFFFFF because
    * that would require cacheline 0x0FFFFFFF, which is a SAL-reserved address, not
    * a real child. (2) get_hint() returns a span straight over the base block:
    * live entries resolve to their cacheline via `>> 4` (share bits ignored by
    * SAL), and free (null) entries are in SAL's reserved range, so the hint path
    * skips them — no scratch buffer, no filtering.
    *
    * The branch block stays rank-dense (position i == the i-th logical entry,
    * matching bitset::rank in a radix node), so insert/remove shift it — one cheap
    * in-cacheline memmove. Bases are only ever compacted by canonicalize(), called
    * when out of room.
    *
    * Trivially copyable — no constructors; call init() after placement.
    */
   template <int ByteCapacity>
   struct compact_addresses
   {
      static_assert(ByteCapacity >= 7,
                    "compact_addresses needs room for 2-byte header + a base + a branch");
      static constexpr int data_size = ByteCapacity - 2;
      static constexpr int max_bases = 16;  // base index is 4 bits
      static constexpr int max_share = 16;  // slot is 4 bits -> 16 sharers per base

      /// A distinct cacheline base, or `null_ptr_address` when the slot is free.
      struct base_entry
      {
         uint32_t share : 4;       // sharers - 1 (low nibble; harmless in hints)
         uint32_t cacheline : 28;  // shared cacheline index = ptr_address >> 4
      };
      static_assert(sizeof(base_entry) == 4);

      /// One logical branch: which base it belongs to plus its 4-bit slot (the low
      /// 4 bits of the ptr_address = position within the base's cacheline).
      struct branch_entry
      {
         uint8_t base : 4;
         uint8_t slot : 4;
      };
      static_assert(sizeof(branch_entry) == 1);

      void init() noexcept
      {
         _nbase = 0;
         _naddr = 0;
      }

      int size() const noexcept { return _naddr; }
      /// Number of base slots, including free (null) ones.
      int base_count() const noexcept { return _nbase; }
      /// Number of base slots still in use (not null).
      int live_base_count() const noexcept
      {
         int n = 0;
         for (int i = 0; i < _nbase; ++i)
            n += !is_free(i);
         return n;
      }
      int free_bytes() const noexcept { return data_size - _nbase * 4 - _naddr; }

      /// Would inserting `a` succeed, possibly after one canonicalize()? Only the
      /// tight path (no slack to grow into) pays for the scans.
      bool can_insert(smart_address a) const noexcept
      {
         const int free = free_bytes();
         if (free >= 5 && _nbase < max_bases)
            return true;  // room to just grow a fresh base+branch

         const uint32_t cl   = a.raw >> 4;
         const bool     same = find_base_with_room(cl) >= 0;
         if (free >= 1)
         {
            if (same)
               return true;  // reuse a same-cacheline base, only a branch byte
            if (find_free_slot() >= 0)
               return true;  // reuse a null base slot, only a branch byte
         }
         // Last resort: would dropping null slots free enough room?
         const int live = live_base_count();
         const int dead = _nbase - live;
         if (dead == 0)
            return false;  // nothing reclaimable — genuinely full
         const int free2 = data_size - live * 4 - _naddr;
         return same ? (free2 >= 1) : (live < max_bases && free2 >= 5);
      }

      /// Insert `a` at logical position `pos` (0..size()). Preconditions — a valid
      /// `pos` and prior can_insert(a) — are caller invariants, asserted (and
      /// compiled out in release), not re-checked on the hot path.
      void insert(smart_address a, int pos) noexcept
      {
         assert(pos >= 0 && pos <= _naddr);
         assert(can_insert(a));

         // can_insert's last-resort feasibility assumes dead base slots get reclaimed,
         // but the same-cacheline path below (find_base_with_room → ++share + one
         // branch byte) does NOT canonicalize. Reclaim here when tight so that path
         // has room for the branch byte (and the new-base path has room for both).
         if (free_bytes() < 5 && live_base_count() < _nbase)
            canonicalize();

         const uint32_t cl   = a.raw >> 4;
         const uint32_t slot = a.raw & 0xF;

         int bi = find_base_with_room(cl);
         if (bi >= 0)
         {
            base_entry be = load_base(bi);
            ++be.share;  // one more sharer of this cacheline
            store_base(bi, be);
         }
         else
         {
            bi = acquire_base_slot();
            store_base(bi, base_entry{0, cl});  // first sharer
         }

         // Splice the new branch byte into the tail-anchored block at `pos`. The
         // tail block has not shifted yet, so write at old_start - 1.
         const int old_start = branch_start();
         std::memmove(_data + old_start - 1, _data + old_start, pos);
         _data[old_start - 1 + pos] =
             std::bit_cast<uint8_t>(branch_entry{uint8_t(bi), uint8_t(slot)});
         ++_naddr;
      }

      /// Remove the address at logical position `pos`, returning the address that
      /// was stored there (which is null_ptr_address if a null was stored, so the
      /// caller can release it). `pos` in [0, size()) is a caller invariant.
      ///
      /// O(1) base bookkeeping: decrement the base's share, or stamp the slot null
      /// when its last reference leaves. Only the branch block is shifted.
      smart_address remove(int pos) noexcept
      {
         assert(pos >= 0 && pos < _naddr);

         const branch_entry  br = load_branch(pos);
         const int           bi = br.base;
         base_entry          be = load_base(bi);
         const smart_address removed{uint32_t((be.cacheline << 4) | br.slot)};

         if (be.share == 0)
            set_free(bi);  // last reference -> mark the slot free
         else
         {
            --be.share;
            store_base(bi, be);
         }

         const int old_start = branch_start();
         std::memmove(_data + old_start + 1, _data + old_start, pos);
         --_naddr;
         return removed;
      }

      smart_address at(int pos) const noexcept
      {
         assert(pos >= 0 && pos < _naddr);
         const branch_entry br = load_branch(pos);
         const base_entry   be = load_base(br.base);
         return smart_address{uint32_t((be.cacheline << 4) | br.slot)};
      }

      /// Replace the address at logical position `pos` with `a`. The common case —
      /// `a` on the same cacheline as the old entry — just rewrites the 4-bit slot
      /// in place (no shift, no base churn). Otherwise it re-homes the entry; that
      /// can fail (returning false, leaving the entry unchanged) if `a` needs a
      /// base slot that cannot be made available. `pos` in [0, size()) is a
      /// caller invariant.
      bool set(int pos, smart_address a) noexcept
      {
         assert(pos >= 0 && pos < _naddr);
         const branch_entry obr = load_branch(pos);
         const base_entry   obe = load_base(obr.base);
         const uint32_t     cl  = a.raw >> 4;
         if (obe.cacheline == cl)  // same cacheline: just rewrite the slot
         {
            store_branch(pos, branch_entry{obr.base, uint8_t(a.raw & 0xF)});
            return true;
         }
         // Different cacheline: drop the old entry and re-add it at the same pos.
         const smart_address old = remove(pos);
         if (!can_insert(a))
         {
            insert(old, pos);  // undo — guaranteed to fit, we just freed it
            return false;
         }
         insert(a, pos);
         return true;
      }

      /// Drop every null base slot and renumber the survivors, packing the base
      /// block. The only O(nbase + naddr) pass; called by insert() when out of
      /// room, but exposed for tests / explicit defragmentation.
      void canonicalize() noexcept
      {
         uint8_t remap[max_bases];
         int     w = 0;
         for (int i = 0; i < _nbase; ++i)
            if (!is_free(i))
            {
               if (w != i)
                  set_base_word(w, base_word(i));
               remap[i] = uint8_t(w);
               ++w;
            }
         _nbase = uint8_t(w);

         for (int i = 0; i < _naddr; ++i)
         {
            branch_entry b = load_branch(i);
            if (remap[b.base] != b.base)
            {
               b.base = remap[b.base];
               store_branch(i, b);
            }
         }
      }

      /**
       * Cacheline-base addresses, suitable as a sal::alloc_hint. Returns a span
       * directly over the stored base block — live entries resolve to their
       * cacheline via SAL's `>> 4`, and free (null) entries fall in SAL's reserved
       * range so the hint path skips them. No scratch buffer. Valid while `*this`.
       */
      std::span<const sal::ptr_address> get_hint() const noexcept
      {
         return {reinterpret_cast<const sal::ptr_address*>(_data), static_cast<size_t>(_nbase)};
      }

     private:
      // base_entry is 4 bytes wide at an arbitrary offset in a byte array, so go
      // through memcpy (unaligned-safe; zero cost on x86-64/ARM64).
      uint32_t base_word(int i) const noexcept
      {
         uint32_t w;
         std::memcpy(&w, _data + i * 4, 4);
         return w;
      }
      void       set_base_word(int i, uint32_t w) noexcept { std::memcpy(_data + i * 4, &w, 4); }
      base_entry load_base(int i) const noexcept { return std::bit_cast<base_entry>(base_word(i)); }
      void       store_base(int i, base_entry b) noexcept { set_base_word(i, std::bit_cast<uint32_t>(b)); }
      bool       is_free(int i) const noexcept { return base_word(i) == *sal::null_ptr_address; }
      void       set_free(int i) noexcept { set_base_word(i, *sal::null_ptr_address); }

      int branch_start() const noexcept { return data_size - _naddr; }

      branch_entry load_branch(int pos) const noexcept
      {
         return std::bit_cast<branch_entry>(_data[branch_start() + pos]);
      }
      void store_branch(int pos, branch_entry b) noexcept
      {
         _data[branch_start() + pos] = std::bit_cast<uint8_t>(b);
      }

      /// Index of a live base for cacheline `cl` with room for one more branch, or
      /// -1. (A null slot has share == 15 and a reserved cacheline, so it never
      /// matches the share < max_share-1 test.)
      int find_base_with_room(uint32_t cl) const noexcept
      {
         for (int i = 0; i < _nbase; ++i)
         {
            const base_entry be = load_base(i);
            if (be.cacheline == cl && be.share + 1 < max_share && !is_free(i))
               return i;
         }
         return -1;
      }

      /// First free (null) base slot, via a single SIMD scan for the sentinel on
      /// real-sized nodes; scalar for sub-cacheline test instances (the SIMD path
      /// reads a full 64-byte block from the base array).
      int find_free_slot() const noexcept
      {
         if (_nbase == 0)
            return -1;
         if constexpr (data_size >= 64)
         {
            const int i = ucc::find_u32x16(reinterpret_cast<const uint32_t*>(_data),
                                           static_cast<size_t>(_nbase), *sal::null_ptr_address);
            return i < _nbase ? i : -1;
         }
         else
         {
            for (int i = 0; i < _nbase; ++i)
               if (is_free(i))
                  return i;
            return -1;
         }
      }

      /// Obtain a base slot for a brand-new cacheline. While there is slack, just
      /// grows; only when out of room does it reuse a free slot, then (last resort)
      /// canonicalize. Feasibility is guaranteed by the can_insert ticket.
      int acquire_base_slot() noexcept
      {
         if (_nbase < max_bases && free_bytes() >= 5)
            return _nbase++;  // free space: just grow
         const int f = find_free_slot();
         if (free_bytes() >= 1 && f >= 0)
            return f;  // out of grow room: reuse a null slot in place
         canonicalize();
         assert(_nbase < max_bases && free_bytes() >= 5);
         return _nbase++;
      }

      uint8_t _nbase;
      uint8_t _naddr;
      uint8_t _data[data_size];
   };

   static_assert(sizeof(compact_addresses<128>) == 128, "compact_addresses must be byte-exact");
}  // namespace sal_art_ref

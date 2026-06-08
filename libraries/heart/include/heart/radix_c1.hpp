#pragma once
#include <cstdint>
#include <cstring>
#include <span>

#include <heart/radix_node.hpp>
#include <heart/radix_type_ids.hpp>
#include <heart/smart_address.hpp>

namespace heart
{
   // ── single-cacheline narrow-window bitset node (bitset_c1_{32,64,128}) ──────────
   //
   // A 1-CL router for a DENSE-in-a-NARROW-RANGE byte fan-out — the tier between a
   // setlist (explicit stems, ~19 branches) and the multi-CL R2/R4. Instead of an
   // explicit sorted stem array it keeps a Width-bit presence mask over a window
   // [base, base+Width) (base in the fixed header, the reclaimed-byte slot), and a
   // rank-dense branch array: a present byte `b` maps to branch slot
   // popcount(mask below b-base). Routing is a subtract + range-check + popcount —
   // constant time, no scan — and packs MORE branches than a setlist in the same
   // cacheline (Width 32/64/128 → 23/22/20 vs setlist's ~19):
   //
   //   radix_base(29) + base(1) + mask(Width/8) + branch[cap]*4  ==  126 B (+pcap 2)
   //
   // It is r1-tier: a 1-CL node has no header to skip, so descent reads CL0 and
   // dispatches by SAL type (route_r1 handles it uniformly — stem_width==1, find(byte)).
   // It carries NO inline prefix (a prefixed dense node goes to R2/R4 instead), so all
   // of CL0's payload is mask+branches. Inline terminal values live in the tail heap
   // (smart_address::from_tail), exactly as setlist — the radix_impl lifecycle is
   // inherited unchanged. On overflow (window-miss OR cap-full) it widens straight to
   // R4 (the generic try_upgrade: c1 is a `From`, R4.set_branch takes the absolute byte).

   // cap = branches that fit a 128-B node after radix_base(29) + base(1) + mask(Width/8),
   // leaving >=2 B so radix_impl's prefix_capacity (which must be >0) is satisfied.
   inline constexpr int c1_cap(int Width) noexcept { return (96 - Width / 8) / 4; }

   template <int Width, sal::header_type TypeId>
   struct c1_fixed
   {
      static constexpr int cap = c1_cap(Width);
      uint8_t              base;             // window start: covers [base, base+Width)
      uint8_t              mask[Width / 8];  // presence bits over the window (LSB = base+0)
      uint8_t              branches[cap * 4];  // rank-dense smart_address array (memcpy access)
   };

   template <int Width, sal::header_type TypeId>
   struct bitset_c1_node : radix_impl<bitset_c1_node<Width, TypeId>, c1_fixed<Width, TypeId>>
   {
      using base_t = radix_impl<bitset_c1_node<Width, TypeId>, c1_fixed<Width, TypeId>>;
      static constexpr dispatch_shape   tier       = dispatch_shape::r1;  // 1 CL → read CL0
      static constexpr int              stem_width = 1;                   // 1 key byte / hop
      using stem_type                              = uint8_t;             // byte-indexed (engine generics)
      static constexpr int              width      = Width;
      static constexpr int              cap        = c1_cap(Width);
      static constexpr sal::header_type type_id    = TypeId;

      bitset_c1_node() noexcept = default;

      // ── window base ─────────────────────────────────────────────────────────────
      uint8_t window_base() const noexcept { return this->fixed.base; }
      void    set_window_base(uint8_t b) noexcept { this->fixed.base = b; }

      // ── presence mask: NATIVE-WORD popcount, not a byte loop ─────────────────────
      // rank()/count() are the dispatch hot path, so they must be the single-instruction
      // popcount the design intends (a byte-wise loop throws that away). The mask lives at
      // a packed/unaligned offset, so the word is loaded with memcpy (an unaligned load).
      bool present(int d) const noexcept { return (this->fixed.mask[d >> 3] >> (d & 7)) & 1; }
      void set_present(int d) noexcept { this->fixed.mask[d >> 3] |= uint8_t(1u << (d & 7)); }
      void clear_present(int d) noexcept { this->fixed.mask[d >> 3] &= uint8_t(~(1u << (d & 7))); }
      static uint64_t load64(const uint8_t* p) noexcept
      {
         uint64_t w;
         std::memcpy(&w, p, 8);
         return w;
      }
      int count() const noexcept
      {
         if constexpr (Width == 32)
         {
            uint32_t m;
            std::memcpy(&m, this->fixed.mask, 4);
            return __builtin_popcount(m);
         }
         else if constexpr (Width == 64)
            return __builtin_popcountll(load64(this->fixed.mask));
         else
            return __builtin_popcountll(load64(this->fixed.mask)) +
                   __builtin_popcountll(load64(this->fixed.mask + 8));
      }
      // rank(d) = present bits strictly below window-index d (the dense branch slot).
      // One masked popcount (two for the 128-window's high half) — no loop.
      int rank(int d) const noexcept
      {
         if constexpr (Width == 32)
         {
            uint32_t m;
            std::memcpy(&m, this->fixed.mask, 4);
            return __builtin_popcount(m & ((1u << d) - 1u));
         }
         else if constexpr (Width == 64)
            return __builtin_popcountll(load64(this->fixed.mask) & ((uint64_t(1) << d) - 1));
         else  // 128: low word fully below 64; high word masked
            return d < 64 ? __builtin_popcountll(load64(this->fixed.mask) & ((uint64_t(1) << d) - 1))
                          : __builtin_popcountll(load64(this->fixed.mask)) +
                                __builtin_popcountll(load64(this->fixed.mask + 8) &
                                                     ((uint64_t(1) << (d - 64)) - 1));
      }
      // Smallest present window-index >= d, or -1 if none (ordered-walk helper).
      int next_present(int d) const noexcept
      {
         for (int i = d < 0 ? 0 : d; i < Width; ++i)
            if (present(i))
               return i;
         return -1;
      }

      // ── rank-dense branch array (memcpy; the fixed header is packed/unaligned) ────
      smart_address branch_at(int pos) const noexcept
      {
         smart_address sa;
         std::memcpy(&sa, this->fixed.branches + size_t(pos) * 4, 4);
         return sa;
      }
      void set_branch_slot(int pos, smart_address sa) noexcept
      {
         std::memcpy(this->fixed.branches + size_t(pos) * 4, &sa, 4);
      }

      // ── uniform node interface (matches setlist/bitset, used by find/insert) ─────
      // find: subtract base, range-check the window, then present-gated rank lookup.
      smart_address find(uint8_t b) const noexcept
      {
         const unsigned d = unsigned(b) - unsigned(this->fixed.base);
         if (d >= unsigned(Width) || !present(int(d)))
            return smart_address::null();
         return branch_at(rank(int(d)));
      }
      // Can this byte be stored here? In-window replace OR (in-window new with room).
      // Out-of-window bytes can NEVER be held → false → caller widens to R4.
      bool can_set(uint8_t b, smart_address) const noexcept
      {
         const unsigned d = unsigned(b) - unsigned(this->fixed.base);
         if (d >= unsigned(Width))
            return false;
         return present(int(d)) || count() < cap;
      }
      // Sorted (by window-index) update-or-insert. Returns false (no change) if the
      // byte is out of window or the array is full — the widen signal.
      bool set_branch(uint8_t b, smart_address sa) noexcept
      {
         const unsigned d = unsigned(b) - unsigned(this->fixed.base);
         if (d >= unsigned(Width))
            return false;
         const int pos = rank(int(d));
         if (present(int(d)))
         {
            set_branch_slot(pos, sa);  // replace in place
            return true;
         }
         const int n = count();
         if (n >= cap)
            return false;
         // open a gap at pos (ranks at/after d shift up by one), then set the bit.
         std::memmove(this->fixed.branches + size_t(pos + 1) * 4,
                      this->fixed.branches + size_t(pos) * 4, size_t(n - pos) * 4);
         set_branch_slot(pos, sa);
         set_present(int(d));
         return true;
      }
      // Remove byte `b` from the window: close the rank gap, clear the presence bit.
      // No-op if absent. (The removed branch's value is the caller's to release.)
      void remove_branch(uint8_t b) noexcept
      {
         const unsigned d = unsigned(b) - unsigned(this->fixed.base);
         if (d >= unsigned(Width) || !present(int(d)))
            return;
         const int n   = count();
         const int pos = rank(int(d));
         std::memmove(this->fixed.branches + size_t(pos) * 4,
                      this->fixed.branches + size_t(pos + 1) * 4, size_t(n - 1 - pos) * 4);
         clear_present(int(d));
      }
      int branch_count() const noexcept { return count(); }
      // Alloc-hint: the branch slots (contiguous) so a new child clusters with siblings.
      std::span<const sal::ptr_address> branch_hint(uint8_t = 0) const noexcept
      {
         return {reinterpret_cast<const sal::ptr_address*>(this->fixed.branches), size_t(count())};
      }

      template <typename F>
      void for_each_entry(F&& f) const  // (absolute byte, branch) in key order
      {
         int idx = 0;
         for (int d = next_present(0); d >= 0; d = next_present(d + 1))
            f(uint8_t(this->fixed.base + d), branch_at(idx++));
      }
      template <typename F>
      void for_each_child(F&& f) const
      {
         if (this->_term_value.is_pointer())
            f(this->_term_value.as_pointer());
         for_each_entry([&](uint8_t, smart_address br) {
            if (br.is_pointer())
               f(br.as_pointer());
         });
      }

      // alloc_size / head_bytes / head_size / free_heap / destroy: inherited from radix_impl
      // (fixed-size head + a trailing tail heap for inline terminals, exactly like setlist_u8).
      bitset_c1_node(uint32_t asize, sal::ptr_address_seq seq, std::span<const uint8_t> prefix) noexcept
      {
         this->init_page(asize, type_id, seq);
         this->_rtype         = radix_node_type::bitset64;  // format tag (unused on this path)
         this->_term_value    = smart_address::null();
         this->_prefix_length = 0;
         this->_seg_presence  = 0;
         this->fixed.base     = 0;
         std::memset(this->fixed.mask, 0, sizeof(this->fixed.mask));
         this->set_prefix(prefix);  // c1 carries no prefix; asserts empty
      }
   };

   using bitset_c1_32  = bitset_c1_node<32, radix_type_ids::bitset_c1_32>;
   using bitset_c1_64  = bitset_c1_node<64, radix_type_ids::bitset_c1_64>;
   using bitset_c1_128 = bitset_c1_node<128, radix_type_ids::bitset_c1_128>;

   static_assert(sizeof(bitset_c1_32) == 128, "bitset_c1_32 must be one cacheline");
   static_assert(sizeof(bitset_c1_64) == 128, "bitset_c1_64 must be one cacheline");
   static_assert(sizeof(bitset_c1_128) == 128, "bitset_c1_128 must be one cacheline");
   static_assert(c1_cap(32) == 23 && c1_cap(64) == 22 && c1_cap(128) == 20, "branch caps");
}  // namespace heart

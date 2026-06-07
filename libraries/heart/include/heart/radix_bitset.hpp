#pragma once
#include <cassert>
#include <span>

#include <heart/bitset.hpp>
#include <heart/compact_addresses.hpp>
#include <heart/radix_full.hpp>
#include <sal/numbers.hpp>

namespace heart
{
   /**
    * A presence bitset paired with a dense, cacheline-deduplicated branch array —
    * the building block a radix node uses to map a key byte to a child address.
    *
    * It is a pure composition of two already-vetted utilities, with exactly two
    * fields:
    *   - `present`  : bitset<Bits>                       (Bits in {64,128,256})
    *   - `branches` : compact_addresses<Size - sizeof(present)>
    *
    * The whole structure occupies `Size` bytes; `branches` gets whatever is left
    * after `present`. A present byte `b` maps to branch slot `present.rank(b)` (the
    * count of set bits below b) — that rank is exactly the dense index into
    * `branches`, so the two stay aligned with no extra indirection.
    *
    * This is a generic container, NOT a node: it carries no version/refcount/COW
    * logic and no upsert<mode> — those live on the node types that embed it. The
    * fields are public so callers can iterate (e.g. present.select(i) /
    * branches.at(i)) or compose richer operations directly.
    *
    * Trivially copyable — no constructors; call init() after placement.
    */
   template <int Bits, int Size>
   struct radix_bitset
   {
      static constexpr int range = Bits;

      using present_type  = bitset<Bits>;
      using branches_type = compact_addresses<Size - int(sizeof(bitset<Bits>))>;

      present_type  present;
      branches_type branches;

      void init() noexcept
      {
         present.clear();
         branches.init();
      }

      /// Number of present bytes (== present.count() == branches.size()).
      int  size() const noexcept { return branches.size(); }
      bool empty() const noexcept { return size() == 0; }

      bool contains(int byte) const noexcept { return present.test(byte); }

      /// Mathematical lower/upper bound (NOT STL): bounding *keys*. lower_bound(byte)
      /// = greatest present byte at or below (the floor); upper_bound(byte) = least
      /// present byte at or above (the ceiling); -1 if none. (For the dense branch
      /// index of a present byte, use present.rank(byte).)
      int lower_bound(int byte) const noexcept { return present.lower_bound(byte); }
      int upper_bound(int byte) const noexcept { return present.upper_bound(byte); }

      /// Branch mapped to `byte` — a smart_address that is either a child node
      /// pointer or a local (inline) tail allocation — or a null smart_address if
      /// `byte` is absent. (If you store null branches as values, gate on contains.)
      smart_address find(int byte) const noexcept
      {
         assert(byte >= 0 && byte < Bits);
         if (!present.test(byte))
            return smart_address::null();
         return branches.at(present.rank(byte));
      }

      /// Op-ticket: would adding one branch `a` fit?
      bool can_insert(smart_address a) const noexcept { return branches.can_insert(a); }

      /**
       * Set bit `i`'s branch to `a`. If `i` was absent this inserts a new branch
       * (and sets the present bit); if `i` was already present it replaces the
       * existing branch. The branch index is `present.rank(i)` (the count of set
       * bits below i). Returns false — changing nothing — if there is not enough
       * room for a new/relocated branch.
       */
      bool set_branch(int i, smart_address a) noexcept
      {
         assert(i >= 0 && i < Bits);
         const int pos = present.rank(i);
         if (present.test(i))
            return branches.set(pos, a);  // replace existing branch
         if (!branches.can_insert(a))
            return false;  // not enough room for a new branch
         branches.insert(a, pos);
         present.set(i);
         return true;
      }

      /**
       * Clear bit `i`, removing its branch and writing the address that was stored
       * there to `removed` (so the caller can release it). Returns false — leaving
       * `removed` untouched — if `i` was not present.
       */
      bool clear_branch(int i, smart_address& removed) noexcept
      {
         assert(i >= 0 && i < Bits);
         if (!present.test(i))
            return false;
         const int pos = present.rank(i);  // i excludes itself from its rank
         present.reset(i);
         removed = branches.remove(pos);
         return true;
      }

      /// Pass-through alloc hint (the branch cacheline bases).
      std::span<const sal::ptr_address> get_hint() const noexcept { return branches.get_hint(); }

      /**
       * Build this (which must be empty) from a full array, skipping null slots —
       * the inverse of expand(), used when a full node shrinks enough to re-compact.
       * Returns false (partial result) if the live entries don't fit a single
       * radix_bitset<Bits, Size>. Walks high byte -> low for O(1) front inserts.
       */
      bool contract_from(const radix_full<Bits>& full) noexcept
      {
         assert(empty());
         for (int b = Bits - 1; b >= 0; --b)
         {
            const smart_address a = full.find(b);
            if (!a.is_null() && !set_branch(b, a))
               return false;
         }
         return true;
      }

      static constexpr int half_bits = Bits / 2;

      /**
       * Split this Bits-wide range into two half-width halves: `lo` takes bytes
       * [0, half_bits), `hi` takes [half_bits, Bits) re-based to [0, half_bits).
       * Both must be freshly init()'d.
       *
       * This is the fast path a node takes when an insert no longer fits in a
       * single radix_bitset<Bits, Size>: each half gets its own cacheline, so total
       * branch capacity grows. Returns false (halves left partial) only if a half
       * lacks capacity — which cannot happen when each half's Size >= this Size.
       *
       * Linear and shift-free: entries are visited high byte -> low byte, so each
       * set_branch lands at rank 0 of its half (an O(1) front insert) — no
       * branch-block memmove, no rank recomputation.
       */
      template <int LoSize, int HiSize>
      bool split(radix_bitset<half_bits, LoSize>& lo, radix_bitset<half_bits, HiSize>& hi) const noexcept
      {
         static_assert(Bits >= 128, "split halves a >=128-wide range into two >=64 halves");
         // Walk present bits high -> low (prev_set, word-skipping) with a running
         // rank index, so each set_branch is an O(1) front insert.
         int idx = size() - 1;
         for (int byte = present.last_set(); byte >= 0; byte = present.prev_set(byte - 1))
         {
            const smart_address a = branches.at(idx--);
            if (byte >= half_bits)
            {
               if (!hi.set_branch(byte - half_bits, a))
                  return false;
            }
            else if (!lo.set_branch(byte, a))
               return false;
         }
         return true;
      }

      /**
       * Compose this (which must be empty) from two half-width halves — the inverse
       * of split(). Returns false, leaving a partial result, when the combined
       * content does not fit in a single radix_bitset<Bits, Size>; that false is
       * exactly the "won't fit as a radix_bitset<Bits, Size>" signal telling a node
       * to keep the two halves split. Linear and shift-free like split().
       */
      template <int LoSize, int HiSize>
      bool merge(const radix_bitset<half_bits, LoSize>& lo,
                 const radix_bitset<half_bits, HiSize>& hi) noexcept
      {
         static_assert(Bits >= 128, "merge composes a >=128-wide range from two >=64 halves");
         assert(empty());
         // high bytes first, each half walked high -> low -> every set_branch a
         // front insert. Running rank index avoids a rank()/select() per entry.
         int idx = hi.size() - 1;
         for (int b = hi.present.last_set(); b >= 0; b = hi.present.prev_set(b - 1))
            if (!set_branch(b + half_bits, hi.branches.at(idx--)))
               return false;
         idx = lo.size() - 1;
         for (int b = lo.present.last_set(); b >= 0; b = lo.present.prev_set(b - 1))
            if (!set_branch(b, lo.branches.at(idx--)))
               return false;
         return true;
      }
   };

   static_assert(sizeof(radix_bitset<256, 128>) == 128, "radix_bitset must be byte-exact");
   static_assert(sizeof(radix_bitset<128, 128>) == 128, "radix_bitset must be byte-exact");
   static_assert(sizeof(radix_bitset<64, 128>) == 128, "radix_bitset must be byte-exact");

   namespace detail
   {
      // Scatter one source set's branches into `out` at [base, base+W), in rank
      // order. Walks present bits ascending; absent bytes keep `out`'s null fill.
      template <int Bits, int W, int Sz>
      void scatter_into_full(radix_full<Bits>& out, int base, const radix_bitset<W, Sz>& src) noexcept
      {
         int idx = 0;
         for (int b = src.present.first_set(); b >= 0; b = src.present.next_set(b + 1))
            out.branches[base + b] = src.branches.at(idx++);
      }
   }  // namespace detail

   /**
    * Expand into a full byte-indexed array of `Bits` ptr_addresses (the dense
    * 8-cacheline tier for Bits=256), with null_ptr_address wherever no source has a
    * branch. `out` is fully overwritten. The dense tier can be assembled from the
    * compact tier in whatever shape it currently has:
    *
    *   - one radix_bitset<Bits>            (whole range in one compact set)
    *   - two radix_bitset<Bits/2>          (halves: [0,Bits/2), [Bits/2,Bits))
    *   - four radix_bitset<Bits/4>         (quarters; Bits/4 must be >= 64, i.e. Bits=256)
    *
    * O(Bits) to null-fill + O(total live) to scatter (each source walked in rank
    * order).
    */
   template <int Bits, int Sz>
   void expand(radix_full<Bits>& out, const radix_bitset<Bits, Sz>& whole) noexcept
   {
      out.init();
      detail::scatter_into_full(out, 0, whole);
   }

   template <int Bits, int Half, int LoSz, int HiSz>
   void expand(radix_full<Bits>&                out,
               const radix_bitset<Half, LoSz>&  lo,
               const radix_bitset<Half, HiSz>&  hi) noexcept
   {
      static_assert(Half * 2 == Bits, "two halves must cover the full range");
      out.init();
      detail::scatter_into_full(out, 0, lo);
      detail::scatter_into_full(out, Half, hi);
   }

   template <int Bits, int Q, int S0, int S1, int S2, int S3>
   void expand(radix_full<Bits>&             out,
               const radix_bitset<Q, S0>&    q0,
               const radix_bitset<Q, S1>&    q1,
               const radix_bitset<Q, S2>&    q2,
               const radix_bitset<Q, S3>&    q3) noexcept
   {
      static_assert(Q * 4 == Bits, "four quarters must cover the full range");
      out.init();
      detail::scatter_into_full(out, 0, q0);
      detail::scatter_into_full(out, Q, q1);
      detail::scatter_into_full(out, 2 * Q, q2);
      detail::scatter_into_full(out, 3 * Q, q3);
   }
}  // namespace heart

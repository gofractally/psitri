#pragma once
#include <cassert>
#include <cstdint>

#include <ucc/find_nth_set_bit.hpp>

namespace heart
{
   /**
    * Fixed-width bitset of `N` bits (N a multiple of 64), laid out as a plain
    * array of `uint64_t` words. Trivially copyable so it can live in-place inside
    * an mmap-backed node; there are no constructors — call clear() after placement.
    *
    * The hot operation for radix nodes is rank(): given a present key byte `b`,
    * `rank(b)` is the count of set bits strictly below `b`, which is exactly the
    * dense slot index of `b` in a parallel compact branch array. design3.md calls
    * this `lower_bound`; the alias is provided.
    */
   template <int N>
   struct bitset
   {
      static_assert(N > 0 && N % 64 == 0, "bitset<N>: N must be a positive multiple of 64");
      static constexpr int word_count = N / 64;
      static constexpr int size_bits  = N;

      uint64_t words[word_count];

      void clear() noexcept
      {
         for (int i = 0; i < word_count; ++i)
            words[i] = 0;
      }

      bool test(int bit) const noexcept
      {
         assert(bit >= 0 && bit < N);
         return (words[bit >> 6] >> (bit & 63)) & 1ull;
      }

      void set(int bit) noexcept
      {
         assert(bit >= 0 && bit < N);
         words[bit >> 6] |= (1ull << (bit & 63));
      }
      void reset(int bit) noexcept
      {
         assert(bit >= 0 && bit < N);
         words[bit >> 6] &= ~(1ull << (bit & 63));
      }

      /// Set `bit`, returning true if it was previously unset (a genuine insert).
      bool set_if_unset(int bit) noexcept
      {
         assert(bit >= 0 && bit < N);
         const uint64_t mask = 1ull << (bit & 63);
         uint64_t&      w    = words[bit >> 6];
         const bool     was  = w & mask;
         w |= mask;
         return !was;
      }

      /// Total number of set bits.
      int count() const noexcept
      {
         int c = 0;
         for (int i = 0; i < word_count; ++i)
            c += __builtin_popcountll(words[i]);
         return c;
      }

      bool any() const noexcept
      {
         for (int i = 0; i < word_count; ++i)
            if (words[i])
               return true;
         return false;
      }
      bool none() const noexcept { return !any(); }

      /// Number of set bits strictly below `bit` (rank). `bit` may equal N.
      int rank(int bit) const noexcept
      {
         assert(bit >= 0 && bit <= N);
         const int w = bit >> 6;
         int       r = 0;
         for (int i = 0; i < w; ++i)
            r += __builtin_popcountll(words[i]);
         const int rem = bit & 63;
         if (rem)
            r += __builtin_popcountll(words[w] & ((1ull << rem) - 1));
         return r;
      }

      /// Mathematical lower/upper bound (NOT STL): bounding *keys*, not counts.
      /// lower_bound(bit) = greatest set bit at or below `bit` (the floor) == prev_set;
      /// upper_bound(bit) = least set bit at or above `bit` (the ceiling) == next_set;
      /// -1 if none. (For the count/rank, use rank().)
      int lower_bound(int bit) const noexcept { return prev_set(bit); }
      int upper_bound(int bit) const noexcept { return next_set(bit); }

      /**
       * Position of the `n`-th set bit (0-indexed), or -1 if there are <= n set
       * bits. This is the inverse of rank(): rank(select(n)) == n, and
       * select(rank(b)) == b for any set bit b. Walks word popcounts to find the
       * owning word, then uses ucc::find_nth_set_bit (BMI2 PDEP+TZCNT on x86).
       */
      int select(int n) const noexcept
      {
         assert(n >= 0);  // n >= count() legitimately returns -1; n < 0 is a caller error
         for (int i = 0; i < word_count; ++i)
         {
            const int pc = __builtin_popcountll(words[i]);
            if (n < pc)
               return (i << 6) + (int)ucc::find_nth_set_bit(words[i], (unsigned)n);
            n -= pc;
         }
         return -1;
      }

      /// Lowest set bit >= `from`, or -1 if none. `from` may be 0..N.
      int next_set(int from) const noexcept
      {
         assert(from >= 0);  // lower bound is an invariant; the upper end is clamped below
         if (from >= N)
            return -1;
         int      w = from >> 6;
         uint64_t m = words[w] & (~0ull << (from & 63));
         for (;;)
         {
            if (m)
               return (w << 6) + __builtin_ctzll(m);
            if (++w == word_count)
               return -1;
            m = words[w];
         }
      }

      /// Lowest set bit, or -1 if empty.
      int first_set() const noexcept { return next_set(0); }

      /// Highest set bit at or below `from` (the inclusive descending mirror of
      /// next_set), or -1 if none. `from` is clamped to [.., N-1]; from < 0 -> -1.
      /// Skips empty words; uses clz, never a per-bit scan. Iterate downward with
      /// prev_set(b - 1).
      int prev_set(int from) const noexcept
      {
         if (from < 0)
            return -1;
         if (from >= N)
            from = N - 1;
         int            w = from >> 6;
         const int      r = from & 63;                                     // include bit r
         uint64_t       m = words[w] & ((r == 63) ? ~0ull : ((1ull << (r + 1)) - 1));
         for (;;)
         {
            if (m)
               return (w << 6) + 63 - __builtin_clzll(m);
            if (w == 0)
               return -1;
            m = words[--w];
         }
      }

      /// Highest set bit, or -1 if empty.
      int last_set() const noexcept { return prev_set(N - 1); }
   };
}  // namespace heart

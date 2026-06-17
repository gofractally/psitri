// start_ref::seg_freeset — per-segment hierarchical free bitmap (SPEC §6).
//
// Tier 0 (l0) is the single authority: one bit per 128-byte line, 1 = free;
// every alloc/free is one CAS/fetch_or on one u64 word, so an allocation can
// never cross a word (= 8 KB region) and is at most 64 lines.
//
// Everything above l0 is derived navigation, maintained EXACTLY by the
// "responsible toggler" rule: the thread whose atomic op on l0 causes a
// group/word boundary transition learns it from the op's prior value and is
// the one thread that XOR-toggles the level above (and adjusts counters).
// Toggles commute, every transition has exactly one toggler, so hints and
// counters track truth without fixer races; readers may still observe a
// stale hint mid-flight and simply re-descend after the authoritative CAS
// fails.
//
//   g2/g4/g8   bit g = aligned group of 2/4/8 lines entirely free
//   meta_k     bit w = word w of array k is nonzero (worth scanning)
//   top_k      bit m = word m of meta_k is nonzero (only when meta_k > 1 word)
//
// Class allocs {1,2,4,8} descend their own array; 16 = an adjacent pair of
// 8-groups inside one byte-octet; arbitrary 3..64 navigate the largest class
// array whose full group any such run must contain (n >= 2^(k+1)-1), then
// verify in-word with shift-AND runs. All descents are masked
// find-first-set-at-or-after(floor) — first-fit-forward falls out of the
// bitmap ordering.
#pragma once
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <sal2/config.hpp>

namespace sal2
{
   namespace detail
   {
      // Pack the bits at even positions of x into the low half (Morton one-axis).
      constexpr uint64_t compress2(uint64_t x) noexcept
      {
         x &= 0x5555555555555555ull;
         x = (x | (x >> 1)) & 0x3333333333333333ull;
         x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0Full;
         x = (x | (x >> 4)) & 0x00FF00FF00FF00FFull;
         x = (x | (x >> 8)) & 0x0000FFFF0000FFFFull;
         x = (x | (x >> 16)) & 0x00000000FFFFFFFFull;
         return x;
      }
      // Dense masks of fully-free aligned groups within one l0 word.
      struct full_masks
      {
         uint32_t f2;  // bit i = lines 2i..2i+1 free      (32 groups/word)
         uint16_t f4;  // bit i = lines 4i..4i+3 free      (16 groups/word)
         uint8_t  f8;  // bit i = lines 8i..8i+7 free      ( 8 groups/word)
      };
      inline full_masks group_masks(uint64_t v) noexcept
      {
         const uint64_t p2 = v & (v >> 1) & 0x5555555555555555ull;
         const uint32_t c2 = uint32_t(compress2(p2));
         const uint32_t p4 = c2 & (c2 >> 1) & 0x55555555u;
         const uint16_t c4 = uint16_t(compress2(p4));
         const uint16_t p8 = uint16_t(c4 & (c4 >> 1) & 0x5555u);
         const uint8_t  c8 = uint8_t(compress2(p8));
         return {c2, c4, c8};
      }
      // Expand a dense group mask back to pair positions: bit i -> bits 2i,2i+1.
      constexpr uint64_t expand2(uint64_t x) noexcept
      {
         x &= 0x00000000FFFFFFFFull;
         x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
         x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
         x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
         x = (x | (x << 2)) & 0x3333333333333333ull;
         x = (x | (x << 1)) & 0x5555555555555555ull;
         return x | (x << 1);  // each kept bit covers its pair
      }
   }  // namespace detail

   // Probe counter for tests: descent iterations of the most recent alloc on
   // this thread. Proves O(1) behavior without wall clocks.
   inline thread_local uint32_t g_freeset_probes = 0;

   // Grain-AGNOSTIC: templated on UNITS-per-segment (bits in l0), never on bytes.
   // The line arena and the large-value arena instantiate the SAME freeset (both
   // have units_per_segment == lines_per_segment by the large_geometry choice),
   // so the default instantiation is byte-identical to the pre-template struct.
   // Whether a freeset op maintains the g2/g4/g8 run tiers. A single-size slab
   // (FixedAllocator) only ever allocates one slot, so it never reads the run
   // tiers — `skip` makes its alloc/free O(1) on l0 + the find-index alone.
   enum class run_tiers { keep, skip };

   template <uint32_t UnitsPerSegment = lines_per_segment>
   struct seg_freeset_t
   {
      static constexpr uint32_t lines_per_segment = UnitsPerSegment;
      static constexpr uint32_t W0 = lines_per_segment / 64;  // l0 words
      static constexpr uint32_t W1 = (lines_per_segment / 2) / 64;
      static constexpr uint32_t W2 = (lines_per_segment / 4) / 64;
      static constexpr uint32_t W3 = (lines_per_segment / 8) / 64;
      static constexpr uint32_t MW0 = (W0 + 63) / 64, MW1 = (W1 + 63) / 64,
                                MW2 = (W2 + 63) / 64, MW3 = (W3 + 63) / 64;
      static_assert(W0 <= 64 * 64, "one top word per tier caps segments at 32 MB");

      std::atomic<uint64_t> l0[W0];  // AUTHORITY: 1 bit per line, 1 = free
      std::atomic<uint64_t> g2[W1];
      std::atomic<uint64_t> g4[W2];
      std::atomic<uint64_t> g8[W3];
      std::atomic<uint64_t> meta[4][MW0 > 1 ? MW0 : 1];  // sized for the largest tier
      std::atomic<uint64_t> top[4];
      std::atomic<uint32_t> free_count[4];  // free lines / 2-groups / 4-groups / 8-groups
      // Lowest line freed since the owner last consumed it: lets the owning
      // session start its first-fit scan at min(cursor, low_water) instead of
      // rescanning from the floor — exact first-fit placements, bump-like cost.
      std::atomic<uint32_t> low_water{UINT32_MAX};

      static constexpr uint32_t words_of(uint32_t k) noexcept
      {
         return k == 0 ? W0 : k == 1 ? W1 : k == 2 ? W2 : W3;
      }
      static constexpr uint32_t meta_words_of(uint32_t k) noexcept
      {
         return (words_of(k) + 63) / 64;
      }
      std::atomic<uint64_t>* array_of(uint32_t k) noexcept
      {
         return k == 0 ? l0 : k == 1 ? g2 : k == 2 ? g4 : g8;
      }

      // ── init (single-threaded placement, plain stores) ─────────────────────
      void init(bool all_free) noexcept
      {
         low_water.store(UINT32_MAX, std::memory_order_relaxed);
         for (auto& w : l0)
            w.store(all_free ? ~0ull : 0ull, std::memory_order_relaxed);
         for (auto& w : g2)
            w.store(all_free ? ~0ull : 0ull, std::memory_order_relaxed);
         for (auto& w : g4)
            w.store(all_free ? ~0ull : 0ull, std::memory_order_relaxed);
         for (auto& w : g8)
            w.store(all_free ? ~0ull : 0ull, std::memory_order_relaxed);
         for (uint32_t k = 0; k < 4; ++k)
         {
            for (uint32_t m = 0; m < (MW0 > 1 ? MW0 : 1); ++m)
               meta[k][m].store(0, std::memory_order_relaxed);
            const uint32_t mw = meta_words_of(k);
            if (all_free)
            {
               for (uint32_t w = 0; w < words_of(k); ++w)
                  meta[k][w / 64].fetch_or(1ull << (w % 64), std::memory_order_relaxed);
            }
            uint64_t t = 0;
            for (uint32_t m = 0; m < mw; ++m)
               if (meta[k][m].load(std::memory_order_relaxed))
                  t |= 1ull << m;
            top[k].store(t, std::memory_order_relaxed);
            free_count[k].store(all_free ? (lines_per_segment >> k) : 0,
                                std::memory_order_relaxed);
         }
      }

      // Rebuild every derived tier from l0 (recovery / reopen). Single-threaded.
      void rebuild_from_l0() noexcept
      {
         low_water.store(UINT32_MAX, std::memory_order_relaxed);
         uint32_t cnt[4] = {0, 0, 0, 0};
         for (uint32_t k = 1; k < 4; ++k)
            for (uint32_t w = 0; w < words_of(k); ++w)
               array_of(k)[w].store(0, std::memory_order_relaxed);
         for (uint32_t w = 0; w < W0; ++w)
         {
            const uint64_t      v = l0[w].load(std::memory_order_relaxed);
            const detail::full_masks fm = detail::group_masks(v);
            cnt[0] += uint32_t(std::popcount(v));
            cnt[1] += uint32_t(std::popcount(uint64_t(fm.f2)));
            cnt[2] += uint32_t(std::popcount(uint64_t(fm.f4)));
            cnt[3] += uint32_t(std::popcount(uint64_t(fm.f8)));
            hint_word_or_(1, w, fm.f2);
            hint_word_or_(2, w, fm.f4);
            hint_word_or_(3, w, fm.f8);
         }
         for (uint32_t k = 0; k < 4; ++k)
         {
            const uint32_t mw = meta_words_of(k);
            for (uint32_t m = 0; m < mw; ++m)
               meta[k][m].store(0, std::memory_order_relaxed);
            for (uint32_t w = 0; w < words_of(k); ++w)
               if (array_of(k)[w].load(std::memory_order_relaxed))
                  meta[k][w / 64].fetch_or(1ull << (w % 64), std::memory_order_relaxed);
            uint64_t t = 0;
            for (uint32_t m = 0; m < mw; ++m)
               if (meta[k][m].load(std::memory_order_relaxed))
                  t |= 1ull << m;
            top[k].store(t, std::memory_order_relaxed);
            free_count[k].store(cnt[k], std::memory_order_relaxed);
         }
      }

      // ── public ops ─────────────────────────────────────────────────────────
      // Allocate n consecutive lines (1..64) starting at-or-after line `from`
      // within this segment. Returns the line offset, or UINT32_MAX.
      uint32_t alloc(uint32_t n, uint32_t from) noexcept
      {
         assert(n >= 1 && n <= max_alloc_lines);
         g_freeset_probes = 0;
         // (ported from vpart, Dan: apples-to-apples) the per-case constant IS
         // ctz(n); callers with COMPILE-TIME n use alloc_c<N> below.
         if (n <= 8 && (n & (n - 1)) == 0)
            return alloc_class_(uint32_t(std::countr_zero(n)), from);
         if (n == 16)
            return alloc_16_(from);
         return alloc_run_(n, from);
      }

      template <uint32_t N>
      uint32_t alloc_c(uint32_t from) noexcept
      {
         static_assert(N >= 1 && N <= max_alloc_lines);
         if constexpr (N <= 8 && (N & (N - 1)) == 0)
            return alloc_class_(uint32_t(std::countr_zero(N)), from);
         else if constexpr (N == 16)
            return alloc_16_(from);
         else
            return alloc_run_(N, from);
      }

      // Free n lines at `line` (previously allocated together or line-by-line —
      // bits are bits; assert catches double frees).
      void free_run(uint32_t line, uint32_t n) noexcept
      {
         assert(line + n <= lines_per_segment && n >= 1 && n <= max_alloc_lines);
         const uint32_t w    = line / 64;
         const uint64_t mask = mask_(n) << (line % 64);
         assert((line % 64) + n <= 64 && "allocation never crosses an l0 word");
         const uint64_t prior = l0[w].fetch_or(mask, std::memory_order_release);
         assert((prior & mask) == 0 && "double free");
         propagate_(w, prior, prior | mask);
         auto lw = low_water.load(std::memory_order_relaxed);
         while (line < lw &&
                !low_water.compare_exchange_weak(lw, line, std::memory_order_relaxed))
            ;
      }

      // ── single-slot fast path (FixedAllocator slab; skips the run tiers) ──────
      // Find/take the first free slot at-or-after `from`; no g2/g4/g8 update.
      uint32_t alloc1(uint32_t from) noexcept
      {
         uint32_t gfrom = from;
         for (;;)
         {
            const uint32_t hw = find_word_(0, gfrom / 64);
            if (hw == UINT32_MAX) return UINT32_MAX;
            uint64_t cand = l0[hw].load(std::memory_order_relaxed);
            if (hw == gfrom / 64) cand &= ~0ull << (gfrom % 64);
            while (cand)
            {
               const uint32_t bit  = uint32_t(std::countr_zero(cand));
               const uint32_t line = hw * 64 + bit;
               const uint64_t mask = 1ull << bit;
               uint64_t       cur  = l0[hw].load(std::memory_order_relaxed);
               while (cur & mask)
               {
                  if (l0[hw].compare_exchange_weak(cur, cur & ~mask, std::memory_order_acquire,
                                                   std::memory_order_relaxed))
                  { propagate_<run_tiers::skip>(hw, cur, cur & ~mask); return line; }
               }
               cand &= ~mask;  // taken under us — next candidate
            }
            gfrom = (hw + 1) * 64;
         }
      }
      void free1(uint32_t line) noexcept
      {
         const uint32_t w    = line / 64;
         const uint64_t mask = 1ull << (line % 64);
         const uint64_t prior = l0[w].fetch_or(mask, std::memory_order_release);
         assert((prior & mask) == 0 && "double free");
         propagate_<run_tiers::skip>(w, prior, prior | mask);
         auto lw = low_water.load(std::memory_order_relaxed);
         while (line < lw && !low_water.compare_exchange_weak(lw, line, std::memory_order_relaxed)) ;
      }

      bool     is_free(uint32_t line) const noexcept
      {
         return (l0[line / 64].load(std::memory_order_relaxed) >> (line % 64)) & 1;
      }
      uint32_t free_lines() const noexcept
      {
         return free_count[0].load(std::memory_order_relaxed);
      }
      uint32_t free_groups8() const noexcept
      {
         return free_count[3].load(std::memory_order_relaxed);
      }

      // For the session word-window (arena.hpp): the owner moved word `w` of l0
      // prior->now itself and is the responsible toggler for the transition.
      void apply_word_transition(uint32_t w, uint64_t prior, uint64_t now) noexcept
      {
         propagate_(w, prior, now);
      }

      // Claim specific lines during recovery (single-threaded; bits must be free).
      void claim_exact(uint32_t line, uint32_t n) noexcept
      {
         const uint32_t w     = line / 64;
         const uint64_t mask  = mask_(n) << (line % 64);
         const uint64_t prior = l0[w].load(std::memory_order_relaxed);
         assert((prior & mask) == mask && "claim of non-free lines");
         l0[w].store(prior & ~mask, std::memory_order_relaxed);
      }

     private:
      static constexpr uint64_t mask_(uint32_t n) noexcept
      {
         return n >= 64 ? ~0ull : ((1ull << n) - 1);
      }

      void hint_word_or_(uint32_t k, uint32_t w0_word, uint64_t dense) noexcept
      {
         const uint32_t per  = 64u >> k;                       // group bits per l0 word
         const uint32_t hw   = w0_word >> k;                   // hint word
         const uint32_t off  = (w0_word & ((1u << k) - 1)) * per;
         array_of(k)[hw].fetch_or(uint64_t(dense) << off, std::memory_order_relaxed);
      }

      // Toggle meta/top after the hint word `w` of tier k transitioned 0<->nonzero.
      void toggle_meta_(uint32_t k, uint32_t w) noexcept
      {
         const uint64_t bit   = 1ull << (w % 64);
         const uint64_t prior = meta[k][w / 64].fetch_xor(bit, std::memory_order_relaxed);
         const uint64_t now   = prior ^ bit;
         if (meta_words_of(k) > 1 && ((prior == 0) != (now == 0)))
            top[k].fetch_xor(1ull << (w / 64), std::memory_order_relaxed);
      }

      // The responsible-toggler propagation: this thread's atomic op moved l0
      // word `w` prior->now; reflect every boundary transition exactly once.
      // run_tiers::skip omits the g2/g4/g8 run-tier maintenance — for a single-
      // size slab (FixedAllocator) that never allocates a multi-slot run, those
      // tiers are never consulted, so updating them is pure overhead (spec §6.1).
      template <run_tiers RT = run_tiers::keep>
      void propagate_(uint32_t w, uint64_t prior, uint64_t now) noexcept
      {
         const int dl = std::popcount(now) - std::popcount(prior);
         if (dl > 0)
            free_count[0].fetch_add(uint32_t(dl), std::memory_order_relaxed);
         else if (dl < 0)
            free_count[0].fetch_sub(uint32_t(-dl), std::memory_order_relaxed);
         if ((prior == 0) != (now == 0))
            toggle_meta_(0, w);
         if constexpr (RT == run_tiers::skip) return;

         const detail::full_masks pm = detail::group_masks(prior);
         const detail::full_masks nm = detail::group_masks(now);
         const uint64_t           d2 = uint64_t(pm.f2 ^ nm.f2);
         const uint64_t           d4 = uint64_t(pm.f4 ^ nm.f4);
         const uint64_t           d8 = uint64_t(pm.f8 ^ nm.f8);
         if (d2)
            toggle_hint_(1, w, d2, std::popcount(uint64_t(nm.f2)) - std::popcount(uint64_t(pm.f2)));
         if (d4)
            toggle_hint_(2, w, d4, std::popcount(uint64_t(nm.f4)) - std::popcount(uint64_t(pm.f4)));
         if (d8)
            toggle_hint_(3, w, d8, std::popcount(uint64_t(nm.f8)) - std::popcount(uint64_t(pm.f8)));
      }

      void toggle_hint_(uint32_t k, uint32_t w0_word, uint64_t diff_dense, int dcount) noexcept
      {
         const uint32_t per   = 64u >> k;
         const uint32_t hw    = w0_word >> k;
         const uint32_t off   = (w0_word & ((1u << k) - 1)) * per;
         const uint64_t bits  = diff_dense << off;
         const uint64_t prior = array_of(k)[hw].fetch_xor(bits, std::memory_order_relaxed);
         const uint64_t now   = prior ^ bits;
         if ((prior == 0) != (now == 0))
            toggle_meta_(k, hw);
         if (dcount > 0)
            free_count[k].fetch_add(uint32_t(dcount), std::memory_order_relaxed);
         else if (dcount < 0)
            free_count[k].fetch_sub(uint32_t(-dcount), std::memory_order_relaxed);
      }

      // Find the first word >= `from_word` of tier k whose array word is nonzero,
      // masking bits below `from_bit_in_first` when landing on from_word itself.
      // Returns word index or UINT32_MAX. Pure navigation — caller re-verifies.
      uint32_t find_word_(uint32_t k, uint32_t from_word) noexcept
      {
         const uint32_t words = words_of(k);
         if (from_word >= words)
            return UINT32_MAX;
         const uint32_t mw = meta_words_of(k);
         uint32_t       m  = from_word / 64;
         uint64_t mword = meta[k][m].load(std::memory_order_relaxed) & (~0ull << (from_word % 64));
         while (true)
         {
            ++g_freeset_probes;
            if (mword)
               return m * 64 + uint32_t(std::countr_zero(mword));
            if (mw == 1)
               return UINT32_MAX;
            uint64_t t = top[k].load(std::memory_order_relaxed);
            t &= (m + 1 >= 64) ? 0 : (~0ull << (m + 1));
            if (!t)
               return UINT32_MAX;
            m     = uint32_t(std::countr_zero(t));
            mword = meta[k][m].load(std::memory_order_relaxed);
         }
      }

      // Class alloc: group size 2^k, aligned. Anti-split: within the chosen word
      // prefer a group whose parent group is NOT fully free (SPEC §6).
      uint32_t alloc_class_(uint32_t k, uint32_t from) noexcept
      {
         const uint32_t n     = 1u << k;
         uint32_t       gfrom = (from + n - 1) >> k;  // first aligned group >= from
         while (true)
         {
            const uint32_t hw = find_word_(k, gfrom / 64);
            if (hw == UINT32_MAX)
               return UINT32_MAX;
            uint64_t cand = array_of(k)[hw].load(std::memory_order_relaxed);
            if (hw == gfrom / 64)
               cand &= ~0ull << (gfrom % 64);
            while (cand)
            {
               ++g_freeset_probes;
               uint64_t pick = cand & ~parent_full_(k, hw);
               if (!pick)
                  pick = cand;
               const uint32_t bit  = uint32_t(std::countr_zero(pick));
               const uint32_t g    = hw * 64 + bit;
               const uint32_t line = g << k;
               if (try_take_(line, n))
                  return line;
               cand &= ~(1ull << bit);  // taken under us — try the next candidate
            }
            gfrom = (hw + 1) * 64;  // word exhausted (stale meta settles via toggler)
         }
      }

      // 16 lines = an adjacent pair of free 8-groups within one byte-octet of g8
      // (pairs crossing an l0 word are excluded by the 0x7f byte mask).
      uint32_t alloc_16_(uint32_t from) noexcept
      {
         uint32_t gfrom = (from + 15) >> 3;  // in 8-group units (start group, 16-aligned below)
         gfrom &= ~1u;
         while (true)
         {
            const uint32_t hw = find_word_(3, gfrom / 64);
            if (hw == UINT32_MAX)
               return UINT32_MAX;
            uint64_t c8 = g8[hw].load(std::memory_order_relaxed);
            if (hw == gfrom / 64)
               c8 &= ~0ull << (gfrom % 64);
            uint64_t cand = c8 & (c8 >> 1) & 0x5555555555555555ull & 0x7f7f7f7f7f7f7f7full;
            while (cand)
            {
               ++g_freeset_probes;
               const uint32_t bit  = uint32_t(std::countr_zero(cand));
               const uint32_t line = (hw * 64 + bit) << 3;
               if (try_take_(line, 16))
                  return line;
               cand &= cand - 1;
            }
            gfrom = (hw + 1) * 64;
         }
      }

      // Arbitrary n: navigate the largest class array a run of n must fully
      // contain (n >= 2^(k+1)-1), then shift-AND inside the l0 word.
      uint32_t alloc_run_(uint32_t n, uint32_t from) noexcept
      {
         // Largest k (<=3) with n >= 2^(k+1)-1: every n-run fully contains an
         // aligned 2^k group, so tier k's index can navigate without misses.
         uint32_t k = 0;
         while (k < 3 && n >= (2u << (k + 1)) - 1)
            ++k;
         uint32_t wfrom = from / 64;
         while (true)
         {
            const uint32_t hw = find_word_(k, wfrom >> k);
            if (hw == UINT32_MAX)
               return UINT32_MAX;
            // hint word hw of tier k spans l0 words [hw<<k, (hw+1)<<k)
            const uint32_t w_lo = hw << k, w_hi = ((hw + 1) << k);
            for (uint32_t w = (w_lo > wfrom ? w_lo : wfrom); w < w_hi && w < W0; ++w)
            {
               ++g_freeset_probes;
               const uint64_t v = l0[w].load(std::memory_order_relaxed);
               uint64_t       r = runs_(v, n);
               if (w == from / 64)
                  r &= ~0ull << (from % 64);
               while (r)
               {
                  const uint32_t pos  = uint32_t(std::countr_zero(r));
                  const uint32_t line = w * 64 + pos;
                  if (try_take_(line, n))
                     return line;
                  r &= r - 1;
               }
            }
            wfrom = w_hi;
            if (wfrom >= W0)
               return UINT32_MAX;
         }
      }

      static constexpr uint64_t runs_(uint64_t v, uint32_t n) noexcept
      {
         // r proves runs of `got`; double until >= n (binary, <= 6 steps for n <= 64)
         uint64_t r   = v;
         uint32_t got = 1;
         while (got * 2 <= n)
         {
            r &= r >> got;
            got *= 2;
         }
         if (got < n)
            r &= r >> (n - got);
         return r;
      }

      // Dense mask (in tier-k group positions of hint word hw) of groups whose
      // PARENT group is entirely free — used to prefer not splitting. For k=3
      // the parent is the implicit 16-pair (adjacent buddy within a byte-octet).
      uint64_t parent_full_(uint32_t k, uint32_t hw) noexcept
      {
         if (k == 3)
         {
            const uint64_t c = g8[hw].load(std::memory_order_relaxed);
            const uint64_t b = c & (c >> 1) & 0x5555555555555555ull & 0x7f7f7f7f7f7f7f7full;
            return b | (b << 1);
         }
         const uint64_t p = array_of(k + 1)[hw / 2].load(std::memory_order_relaxed);
         const uint32_t half = (hw & 1) * 32;
         return detail::expand2((p >> half) & 0xFFFFFFFFull);
      }

      // The single authoritative step: clear `n` bits at `line` if all free.
      bool try_take_(uint32_t line, uint32_t n) noexcept
      {
         const uint32_t w    = line / 64;
         const uint64_t mask = mask_(n) << (line % 64);
         uint64_t       cur  = l0[w].load(std::memory_order_relaxed);
         while ((cur & mask) == mask)
         {
            if (l0[w].compare_exchange_weak(cur, cur & ~mask, std::memory_order_acquire,
                                            std::memory_order_relaxed))
            {
               propagate_(w, cur, cur & ~mask);
               return true;
            }
         }
         return false;
      }
   };

   // Default (128 B grain) alias — byte-identical to the pre-template struct.
   using seg_freeset = seg_freeset_t<lines_per_segment>;
}  // namespace sal2

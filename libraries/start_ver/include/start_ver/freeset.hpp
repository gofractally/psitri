// start_ver::seg_freeset — per-segment free bitmap, hybrid design (SPEC §6 rev 2).
//
// Replaces the 4-tier hierarchical index (g2/g4/g8 + meta/top + counters)
// after the standalone A/B in myartlib/bench/freeset_ab (2026-06-12): the
// tiers made every FREE pay full propagation (~17 ns: two dense group-mask
// computes, popcounts, up to 5 RMWs) to keep alloc navigation O(1), yet odd
// run sizes (c8=9, full=10) still ground through per-word verification. The
// hybrid keeps one authority and one tiny derived layer:
//
//   l0       AUTHORITY: 1 bit per line, 1 = free. Every alloc/free is one
//            CAS/fetch_or on one word — unchanged contract, allocations are
//            word-confined (<= 64 lines, never cross an 8 KB region).
//   pres[c]  PRESENCE: one bit per l0 word per class c in {1,2,4,8,16-pair} =
//            "this word contains a free aligned group of that class".
//            5 x W0/64 words (640 B at 8 MB segments vs 7 KB of tiers).
//            Maintained by the responsible-toggler rule with fetch_xor —
//            toggles commute, so opposite in-flight transitions land exact;
//            readers may see a stale bit mid-flight and simply re-descend
//            (same contract as the old hints).
//   free8_count  Exact count of free aligned 8-groups (the claim/qualify
//            currency). Delta-maintained: each l0 RMW winner adds its own
//            pos8 popcount diff — deltas commute, exact at rest.
//
// Measured (M5, myartlib freeset_ab record): free 15 -> ~5 ns; steady churn
// 23 -> 13 ns/op; alloc(9/10, from=0) on fragmented segments 150-500 ->
// 52-100 ns; needle/fail-fast within 2x of the tiers (a pure bitmap scan is
// 8-300x); writer-vs-cascading-freer interference ~3.5x faster both sides.
//
// Search: classes {1,2,4,8} walk pres[k] to the first flagged word, then
// resolve in-word with positional masks, keeping the span-scoped anti-split
// preference (prefer a group whose PARENT group is not fully free; measured:
// dropping it saves 0.2 ns/op and costs 15% of the free 8-groups). 16 = an
// even-aligned pair of 8-groups via pres[4]. Odd runs navigate the guarantee
// class (every n-run contains an aligned 2^k group for n >= 2^(k+1)-1) and
// verify whole 8-word blocks with NEON; with a compile-time N (alloc_t<N> —
// the tree's node sizes are all static) the chains unroll to immediate
// shifts.
//
// Placement: first-fit-forward, classes aligned. VPART_FS_XFIT additionally
// enables exact-fit-ahead for runs/16: a maximal free run of exactly n (both
// neighbors allocated / word edge — 2 ops to detect) is preferred over
// carving n out of a bigger run, looking up to one word (64 bits) ahead
// before carving a CLOSED run (open-tail words are the bump frontier and are
// exempt). Standalone churn: +5%/+18%/+7% free 8-groups at 50/75/85%
// occupancy — the qualifies_ currency. DEFAULT OFF (2026-06-12 bank A/B):
// hole-filling disperses sibling nodes off the frontier and cost the PURE
// READ phases 6-9% (solo reader 5.65->5.12M/s, validation scan -6%), while
// the file size at bank scale was growth-dominated either way. Flip it when
// long-horizon segment-reuse pressure is the bottleneck being tuned.
#pragma once
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <start_ver/config.hpp>

#ifndef VPART_FS_XFIT
#define VPART_FS_XFIT 0
#endif

// NEON block verification uses wide non-atomic loads of l0 (a stale/torn read
// only costs a wasted probe — try_take_ re-verifies atomically). TSan flags
// them, so fall back to the scalar word loop under it.
#if defined(__aarch64__) && !defined(VPART_FS_NO_NEON)
#if defined(__has_feature)
#if !__has_feature(thread_sanitizer)
#define VPART_FS_NEON 1
#endif
#else
#define VPART_FS_NEON 1
#endif
#endif
#ifdef VPART_FS_NEON
#include <arm_neon.h>
#endif

namespace start_ver
{
   // Descent iterations of the most recent alloc on this thread (tests: the
   // O(1) claim — bounded by presence words + one span, never a bitmap sweep).
   inline thread_local uint32_t g_freeset_probes = 0;

   namespace detail
   {
      inline constexpr uint64_t mask_n(uint32_t n) noexcept
      {
         return n >= 64 ? ~0ull : ((1ull << n) - 1);
      }
      // bit i set iff v has n consecutive set bits starting at i (in-word);
      // with a compile-time n the chain unrolls to constant shifts.
      inline constexpr uint64_t runs_in_word(uint64_t v, uint32_t n) noexcept
      {
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
      // Positional masks: bit i set iff the ALIGNED group starting at i is free.
      inline constexpr uint64_t PM2  = 0x5555555555555555ull;
      inline constexpr uint64_t PM4  = 0x1111111111111111ull;
      inline constexpr uint64_t PM8  = 0x0101010101010101ull;
      inline constexpr uint64_t PM16 = 0x0001000100010001ull;
      inline constexpr uint64_t pos2(uint64_t v) noexcept { return v & (v >> 1) & PM2; }
      inline constexpr uint64_t pos4(uint64_t v) noexcept
      {
         const uint64_t p = pos2(v);
         return p & (p >> 2) & PM4;
      }
      inline constexpr uint64_t pos8(uint64_t v) noexcept
      {
         const uint64_t p = pos4(v);
         return p & (p >> 4) & PM8;
      }
      inline constexpr uint64_t pos16(uint64_t v) noexcept
      {
         const uint64_t p = pos8(v);
         return p & (p >> 8) & PM16;
      }
      template <int K>
      inline constexpr uint64_t posk(uint64_t v) noexcept
      {
         if constexpr (K == 0)
            return v;
         else if constexpr (K == 1)
            return pos2(v);
         else if constexpr (K == 2)
            return pos4(v);
         else
            return pos8(v);
      }
      // Start positions (class K) covered by a fully-free PARENT (2^(K+1))
      // group — the anti-split preference avoids these when it can.
      template <int K>
      inline constexpr uint64_t parent_cover(uint64_t v) noexcept
      {
         if constexpr (K == 0)
         {
            const uint64_t p = pos2(v);
            return p | (p << 1);
         }
         else if constexpr (K == 1)
         {
            const uint64_t p = pos4(v);
            return p | (p << 2);
         }
         else if constexpr (K == 2)
         {
            const uint64_t p = pos8(v);
            return p | (p << 4);
         }
         else
         {
            const uint64_t p = pos16(v);
            return p | (p << 8);
         }
      }
      // guarantee class for an n-run: largest k with n >= 2^(k+1)-1 — every
      // n-run fully contains an aligned 2^k group, so pres[k] cannot miss.
      inline constexpr uint32_t guar_k(uint32_t n) noexcept
      {
         return n >= 15 ? 3u : n >= 7 ? 2u : n >= 3 ? 1u : 0u;
      }
#ifdef VPART_FS_NEON
      // compile-time-N run chain with IMMEDIATE shifts
      template <uint32_t GOT, uint32_t N>
      inline uint64x2_t runs_step_q(uint64x2_t r) noexcept
      {
         if constexpr (GOT * 2 <= N)
            return runs_step_q<GOT * 2, N>(vandq_u64(r, vshrq_n_u64(r, GOT)));
         else if constexpr (GOT < N)
            return vandq_u64(r, vshrq_n_u64(r, N - GOT));
         else
            return r;
      }
#endif
   }  // namespace detail

   struct seg_freeset
   {
      static constexpr uint32_t W0 = lines_per_segment / 64;  // l0 words
      static constexpr uint32_t PW = W0 / 64;                 // presence words/class
      static_assert(W0 % 64 == 0, "presence words are exact (segment >= 512 KB)");
      static constexpr bool xfit = VPART_FS_XFIT != 0;

      std::atomic<uint64_t> l0[W0];       // AUTHORITY: 1 bit per line, 1 = free
      std::atomic<uint64_t> pres[5][PW];  // c: 0..3 = class 2^c, 4 = 16-pair
      std::atomic<uint32_t> free8_count{0};
      // Lowest line freed since the owner last consumed it (arena first-fit
      // cursor rescue) — semantics unchanged from rev 1.
      std::atomic<uint32_t> low_water{UINT32_MAX};

      // ── init / rebuild (single-threaded placement, plain stores) ───────────
      void init(bool all_free) noexcept
      {
         low_water.store(UINT32_MAX, std::memory_order_relaxed);
         for (auto& w : l0)
            w.store(all_free ? ~0ull : 0ull, std::memory_order_relaxed);
         for (auto& c : pres)
            for (auto& w : c)
               w.store(all_free ? ~0ull : 0ull, std::memory_order_relaxed);
         free8_count.store(all_free ? lines_per_segment / 8 : 0,
                           std::memory_order_relaxed);
      }

      // Recompute every derived field from l0 (recovery / reopen / attach).
      void rebuild_from_l0() noexcept
      {
         low_water.store(UINT32_MAX, std::memory_order_relaxed);
         uint32_t f8 = 0;
         for (uint32_t c = 0; c < 5; ++c)
            for (uint32_t m = 0; m < PW; ++m)
               pres[c][m].store(0, std::memory_order_relaxed);
         for (uint32_t w = 0; w < W0; ++w)
         {
            const uint64_t v   = l0[w].load(std::memory_order_relaxed);
            const uint64_t bit = 1ull << (w % 64);
            if (v)
               pres[0][w / 64].fetch_or(bit, std::memory_order_relaxed);
            if (detail::pos2(v))
               pres[1][w / 64].fetch_or(bit, std::memory_order_relaxed);
            if (detail::pos4(v))
               pres[2][w / 64].fetch_or(bit, std::memory_order_relaxed);
            const uint64_t p8 = detail::pos8(v);
            if (p8)
               pres[3][w / 64].fetch_or(bit, std::memory_order_relaxed);
            if (detail::pos16(v))
               pres[4][w / 64].fetch_or(bit, std::memory_order_relaxed);
            f8 += uint32_t(std::popcount(p8));
         }
         free8_count.store(f8, std::memory_order_relaxed);
      }

      // ── public ops ─────────────────────────────────────────────────────────
      // Allocate n consecutive lines (1..64) at-or-after line `from`. Returns
      // the line offset or UINT32_MAX. Prefer alloc_t<N>() when N is static.
      uint32_t alloc(uint32_t n, uint32_t from) noexcept
      {
         assert(n >= 1 && n <= max_alloc_lines);
         g_freeset_probes = 0;
         switch (n)
         {
            case 1:
               return cls_<0>(from);
            case 2:
               return cls_<1>(from);
            case 4:
               return cls_<2>(from);
            case 8:
               return cls_<3>(from);
            case 16:
               return pair16_(from);
            default:
               return run_(n, from);
         }
      }

      // Compile-time-N entry: same placements, dispatch resolved statically
      // and the verify chains unrolled with constant shifts. The tree layer's
      // node sizes are all static — this is its entry point.
      template <uint32_t N>
      uint32_t alloc_t(uint32_t from) noexcept
      {
         static_assert(N >= 1 && N <= max_alloc_lines);
         g_freeset_probes = 0;
         if constexpr (N == 1)
            return cls_<0>(from);
         else if constexpr (N == 2)
            return cls_<1>(from);
         else if constexpr (N == 4)
            return cls_<2>(from);
         else if constexpr (N == 8)
            return cls_<3>(from);
         else if constexpr (N == 16)
            return pair16_(from);
         else
            return run_t<N>(from);
      }

      // Free n lines at `line`. ONE fetch_or + presence/count transition work
      // (usually zero extra RMWs) — no tier propagation.
      void free_run(uint32_t line, uint32_t n) noexcept
      {
         assert(line + n <= lines_per_segment && n >= 1 && n <= max_alloc_lines);
         const uint32_t w    = line / 64;
         const uint64_t mask = detail::mask_n(n) << (line % 64);
         assert((line % 64) + n <= 64 && "allocation never crosses an l0 word");
         const uint64_t prior = l0[w].fetch_or(mask, std::memory_order_release);
         assert((prior & mask) == 0 && "double free");
         update_derived_(w, prior, prior | mask);
         auto lw = low_water.load(std::memory_order_relaxed);
         while (line < lw &&
                !low_water.compare_exchange_weak(lw, line, std::memory_order_relaxed))
            ;
      }

      bool is_free(uint32_t line) const noexcept
      {
         return (l0[line / 64].load(std::memory_order_relaxed) >> (line % 64)) & 1;
      }
      // Exact at quiescence; tests/census only — hot paths never sum this.
      uint32_t free_lines() const noexcept
      {
         uint32_t c = 0;
         for (uint32_t w = 0; w < W0; ++w)
            c += uint32_t(std::popcount(l0[w].load(std::memory_order_relaxed)));
         return c;
      }
      // O(1) exact counter — the segment-qualify currency (arena::qualifies_).
      uint32_t free_groups8() const noexcept
      {
         return free8_count.load(std::memory_order_relaxed);
      }

      // The owner moved word `w` of l0 prior->now itself (session word-window
      // reserve/flush) and owns the derived transitions for it.
      void apply_word_transition(uint32_t w, uint64_t prior, uint64_t now) noexcept
      {
         update_derived_(w, prior, now);
      }

      // Claim specific lines during recovery (single-threaded; bits must be
      // free). Derived state is NOT touched — rebuild_end runs
      // rebuild_from_l0() once the reachability marks are in.
      void claim_exact(uint32_t line, uint32_t n) noexcept
      {
         const uint32_t w     = line / 64;
         const uint64_t mask  = detail::mask_n(n) << (line % 64);
         const uint64_t prior = l0[w].load(std::memory_order_relaxed);
         assert((prior & mask) == mask && "claim of non-free lines");
         l0[w].store(prior & ~mask, std::memory_order_relaxed);
      }

     private:
      // ── derived maintenance (the whole replacement for propagate_) ─────────
      // The thread whose l0 RMW moved word w prior->now owns every presence
      // 0<->nonzero transition (XOR — commutes) and adds its own pos8 delta
      // (sums commute). Early-out: once a class is absent on BOTH sides,
      // nothing larger can exist on either.
      void update_derived_(uint32_t w, uint64_t prior, uint64_t now) noexcept
      {
         const uint64_t bit = 1ull << (w % 64);
         if ((prior == 0) != (now == 0))
            pres[0][w / 64].fetch_xor(bit, std::memory_order_relaxed);
         if (!prior && !now)
            return;
         const uint64_t p2p = detail::pos2(prior), p2n = detail::pos2(now);
         if ((p2p == 0) != (p2n == 0))
            pres[1][w / 64].fetch_xor(bit, std::memory_order_relaxed);
         if (!p2p && !p2n)
            return;
         const uint64_t p4p = p2p & (p2p >> 2) & detail::PM4;
         const uint64_t p4n = p2n & (p2n >> 2) & detail::PM4;
         if ((p4p == 0) != (p4n == 0))
            pres[2][w / 64].fetch_xor(bit, std::memory_order_relaxed);
         if (!p4p && !p4n)
            return;
         const uint64_t p8p = p4p & (p4p >> 4) & detail::PM8;
         const uint64_t p8n = p4n & (p4n >> 4) & detail::PM8;
         if ((p8p == 0) != (p8n == 0))
            pres[3][w / 64].fetch_xor(bit, std::memory_order_relaxed);
         const int d8 = std::popcount(p8n) - std::popcount(p8p);
         if (d8 > 0)
            free8_count.fetch_add(uint32_t(d8), std::memory_order_relaxed);
         else if (d8 < 0)
            free8_count.fetch_sub(uint32_t(-d8), std::memory_order_relaxed);
         if (!p8p && !p8n)
            return;
         const uint64_t pgp = p8p & (p8p >> 8) & detail::PM16;
         const uint64_t pgn = p8n & (p8n >> 8) & detail::PM16;
         if ((pgp == 0) != (pgn == 0))
            pres[4][w / 64].fetch_xor(bit, std::memory_order_relaxed);
      }

      // The single authoritative step: clear `n` bits at `line` if all free.
      bool try_take_(uint32_t line, uint32_t n) noexcept
      {
         const uint32_t w    = line / 64;
         const uint64_t mask = detail::mask_n(n) << (line % 64);
         uint64_t       cur  = l0[w].load(std::memory_order_relaxed);
         while ((cur & mask) == mask)
         {
            if (l0[w].compare_exchange_weak(cur, cur & ~mask, std::memory_order_acquire,
                                            std::memory_order_relaxed))
            {
               update_derived_(w, cur, cur & ~mask);
               return true;
            }
         }
         return false;
      }

      // First presence-flagged word >= wfrom for class c, or UINT32_MAX.
      uint32_t next_flagged_(uint32_t c, uint32_t wfrom) noexcept
      {
         if (wfrom >= W0)
            return UINT32_MAX;
         uint32_t m  = wfrom / 64;
         uint64_t pw = pres[c][m].load(std::memory_order_relaxed) & (~0ull << (wfrom % 64));
         while (true)
         {
            ++g_freeset_probes;
            if (pw)
               return m * 64 + uint32_t(std::countr_zero(pw));
            if (++m >= PW)
               return UINT32_MAX;
            pw = pres[c][m].load(std::memory_order_relaxed);
         }
      }

      // Scan one anti-split SPAN (the 2^K words the parent group covers) for
      // the lowest non-splitting candidate, falling back to the lowest
      // candidate; `first_line` masks starts below the floor.
      template <int K>
      uint32_t span_pick_(uint32_t w0, uint32_t first_line) noexcept
      {
         const uint32_t SPAN     = 1u << K;
         uint32_t       any_line = UINT32_MAX;
         for (uint32_t w = w0; w < w0 + SPAN; ++w)
         {
            ++g_freeset_probes;
            const uint64_t v    = l0[w].load(std::memory_order_relaxed);
            uint64_t       cand = detail::posk<K>(v);
            if (uint64_t(w) * 64 < first_line)
            {
               const uint32_t cut = first_line - w * 64;
               if (cut >= 64)
                  continue;
               cand &= ~0ull << cut;
            }
            if (!cand)
               continue;
            if (any_line == UINT32_MAX)
               any_line = w * 64 + uint32_t(std::countr_zero(cand));
            const uint64_t ns = cand & ~detail::parent_cover<K>(v);
            if (ns)
               return w * 64 + uint32_t(std::countr_zero(ns));
         }
         return any_line;
      }

      template <int K>
      uint32_t cls_(uint32_t from) noexcept
      {
         const uint32_t n          = 1u << K;
         const uint32_t first_line = ((from + n - 1) >> K) << K;
         if (first_line >= lines_per_segment)
            return UINT32_MAX;
         uint32_t wfrom = first_line / 64;
         while (true)
         {
            const uint32_t w = next_flagged_(K, wfrom);
            if (w == UINT32_MAX)
               return UINT32_MAX;
            const uint32_t sp   = w >> K;
            const uint32_t pick = span_pick_<K>(sp << K, first_line);
            if (pick == UINT32_MAX)
            {
               wfrom = (sp + 1) << K;  // span exhausted under the floor mask
               continue;
            }
            if (try_take_(pick, n))
               return pick;
            wfrom = sp << K;  // CAS lost — rescan the same span
         }
      }

      uint32_t pair16_(uint32_t from) noexcept
      {
         const uint32_t first_line = ((from + 15) >> 4) << 4;
         if (first_line >= lines_per_segment)
            return UINT32_MAX;
         uint32_t wfrom = first_line / 64;
         while (true)
         {
            const uint32_t w = next_flagged_(4, wfrom);
            if (w == UINT32_MAX)
               return UINT32_MAX;
            ++g_freeset_probes;
            const uint64_t v    = l0[w].load(std::memory_order_relaxed);
            uint64_t       cand = detail::pos16(v);
            if (uint64_t(w) * 64 < first_line)
            {
               const uint32_t cut = first_line - w * 64;
               cand &= (cut >= 64) ? 0 : (~0ull << cut);
            }
            if constexpr (xfit)
            {
               if (cand && v != ~0ull)
               {
                  const uint64_t ex = cand & ~(v << 1) & ~(v >> 16);
                  if (!ex && ((~v) >> std::countr_zero(cand)) != 0 && w + 1 < W0)
                  {  // only closed-run carves here: peek 64 bits ahead
                     const uint64_t v2 = l0[w + 1].load(std::memory_order_relaxed);
                     const uint64_t e2 = detail::pos16(v2) & ~(v2 << 1) & ~(v2 >> 16);
                     if (e2)
                     {
                        const uint32_t pos = uint32_t(std::countr_zero(e2));
                        if (try_take_((w + 1) * 64 + pos, 16))
                           return (w + 1) * 64 + pos;
                     }
                  }
                  if (ex)
                     cand = ex;  // exact pairs first; carve only if none take
               }
            }
            while (cand)
            {
               const uint32_t pos = uint32_t(std::countr_zero(cand));
               if (try_take_(w * 64 + pos, 16))
                  return w * 64 + pos;
               cand &= cand - 1;
            }
            wfrom = w + 1;
         }
      }

      // In-word resolution for runs: exact-fit-ahead policy, then lowest-first
      // take attempts. r is the (floor-masked) run-start mask of word w.
      uint32_t run_pick_(uint32_t w, uint64_t v, uint64_t r, uint32_t n) noexcept
      {
         if constexpr (xfit)
         {
            if (r && v != ~0ull)  // fully-free word = frontier; no exact fit
            {
               const uint64_t ex = r & ~(v << 1) & ~(v >> n);
               if (!ex && ((~v) >> std::countr_zero(r)) != 0 && w + 1 < W0)
               {  // carving a CLOSED run: look one word ahead for an exact fit
                  const uint64_t v2 = l0[w + 1].load(std::memory_order_relaxed);
                  const uint64_t e2 =
                      detail::runs_in_word(v2, n) & ~(v2 << 1) & ~(v2 >> n);
                  if (e2)
                  {
                     const uint32_t pos = uint32_t(std::countr_zero(e2));
                     if (try_take_((w + 1) * 64 + pos, n))
                        return (w + 1) * 64 + pos;
                  }
               }
               if (ex)
                  r = ex;  // exact fits first
            }
         }
         while (r)
         {
            const uint32_t pos = uint32_t(std::countr_zero(r));
            if (try_take_(w * 64 + pos, n))
               return w * 64 + pos;
            r &= r - 1;
         }
         return UINT32_MAX;
      }

#ifdef VPART_FS_NEON
      const uint64_t* raw_() const noexcept
      {
         return reinterpret_cast<const uint64_t*>(&l0[0]);
      }
      // does any of the 8 words starting at w (one cacheline of bitmap, 512
      // lines) contain an n-run?
      bool blk_run_(uint32_t w, uint32_t n) const noexcept
      {
         const uint64_t* p = raw_() + w;
         uint64x2_t      a = vld1q_u64(p), b = vld1q_u64(p + 2);
         uint64x2_t      c = vld1q_u64(p + 4), d = vld1q_u64(p + 6);
         uint32_t        got = 1;
         while (got * 2 <= n)
         {
            const int64x2_t s = vdupq_n_s64(-int64_t(got));
            a                 = vandq_u64(a, vshlq_u64(a, s));
            b                 = vandq_u64(b, vshlq_u64(b, s));
            c                 = vandq_u64(c, vshlq_u64(c, s));
            d                 = vandq_u64(d, vshlq_u64(d, s));
            got *= 2;
         }
         if (got < n)
         {
            const int64x2_t s = vdupq_n_s64(-int64_t(n - got));
            a                 = vandq_u64(a, vshlq_u64(a, s));
            b                 = vandq_u64(b, vshlq_u64(b, s));
            c                 = vandq_u64(c, vshlq_u64(c, s));
            d                 = vandq_u64(d, vshlq_u64(d, s));
         }
         const uint64x2_t o = vorrq_u64(vorrq_u64(a, b), vorrq_u64(c, d));
         return vmaxvq_u32(vreinterpretq_u32_u64(o)) != 0;
      }
      template <uint32_t N>
      bool blk_run_t_(uint32_t w) const noexcept
      {
         const uint64_t*  p = raw_() + w;
         const uint64x2_t a = detail::runs_step_q<1, N>(vld1q_u64(p));
         const uint64x2_t b = detail::runs_step_q<1, N>(vld1q_u64(p + 2));
         const uint64x2_t c = detail::runs_step_q<1, N>(vld1q_u64(p + 4));
         const uint64x2_t d = detail::runs_step_q<1, N>(vld1q_u64(p + 6));
         const uint64x2_t o = vorrq_u64(vorrq_u64(a, b), vorrq_u64(c, d));
         return vmaxvq_u32(vreinterpretq_u32_u64(o)) != 0;
      }
#endif

      // Walk presence-flagged words for the guarantee class; on block-aligned
      // boundaries verify whole 8-word blocks at once (presence byte first).
      // BlkFn(w) -> bool: may block [w, w+8) contain an n-run?
      template <class BlkFn>
      uint32_t run_walk_(uint32_t n, uint32_t from, uint32_t k, BlkFn&& blk) noexcept
      {
         const uint32_t wfirst = from / 64;
         if (wfirst >= W0)
            return UINT32_MAX;
         uint32_t w = wfirst;
         while (w < W0)
         {
            w = next_flagged_(k, w);
            if (w == UINT32_MAX)
               return UINT32_MAX;
#ifdef VPART_FS_NEON
            if ((w & 7u) == 0 && w != wfirst)
            {
               while (w < W0)
               {
                  const uint64_t pw   = pres[k][w / 64].load(std::memory_order_relaxed);
                  const uint32_t byte = (w % 64) / 8;
                  if (((pw >> (byte * 8)) & 0xFF) == 0 || !blk(w))
                  {
                     ++g_freeset_probes;
                     w += 8;
                     continue;
                  }
                  break;
               }
               if (w >= W0)
                  return UINT32_MAX;
            }
#endif
            ++g_freeset_probes;
            const uint64_t v = l0[w].load(std::memory_order_relaxed);
            uint64_t       r = detail::runs_in_word(v, n);
            if (w == wfirst)
               r &= ~0ull << (from % 64);
            const uint32_t got = run_pick_(w, v, r, n);
            if (got != UINT32_MAX)
               return got;
            ++w;
         }
         return UINT32_MAX;
      }

      uint32_t run_(uint32_t n, uint32_t from) noexcept
      {
         return run_walk_(n, from, detail::guar_k(n),
                          [&](uint32_t w)
                          {
#ifdef VPART_FS_NEON
                             return blk_run_(w, n);
#else
                             (void)w;
                             return true;
#endif
                          });
      }
      template <uint32_t N>
      uint32_t run_t(uint32_t from) noexcept
      {
         return run_walk_(N, from, detail::guar_k(N),
                          [&](uint32_t w)
                          {
#ifdef VPART_FS_NEON
                             return blk_run_t_<N>(w);
#else
                             (void)w;
                             return true;
#endif
                          });
      }
   };
}  // namespace start_ver

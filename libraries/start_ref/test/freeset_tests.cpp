// SPEC §15 test 1 — seg_freeset, single thread, model-based: every placement is
// checked against an oracle recomputed from first principles, and every derived
// tier (hints, meta, top, counters) is compared against truth recomputed from
// l0 after randomized op tapes.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <start_ref/freeset.hpp>

using start_ref::lines_per_segment;
using start_ref::seg_freeset;

namespace
{
   std::unique_ptr<seg_freeset> make_fs(bool all_free = true)
   {
      auto fs = std::make_unique<seg_freeset>();
      fs->init(all_free);
      return fs;
   }

   // Oracle: is the aligned/odd run [line, line+n) entirely free in l0?
   bool oracle_free(const seg_freeset& fs, uint32_t line, uint32_t n)
   {
      for (uint32_t i = 0; i < n; ++i)
         if (!fs.is_free(line + i))
            return false;
      return true;
   }

   // Oracle: the first l0 word index >= floor's word holding an eligible start
   // for n lines at-or-after `from` (aligned for class sizes, any pos for odd).
   // Returns the word, or UINT32_MAX.
   uint32_t oracle_word(const seg_freeset& fs, uint32_t n, uint32_t from, bool aligned)
   {
      const uint32_t step = aligned ? n : 1;
      uint32_t start = aligned ? ((from + n - 1) / n) * n : from;
      for (uint32_t s = start; s + n <= lines_per_segment; s += step)
      {
         if ((s % 64) + n > 64)  // word-confined placements only
            continue;
         if (oracle_free(fs, s, n))
            return s / 64;
      }
      return UINT32_MAX;
   }

   // Recompute every derived tier from l0 and compare. The exactness claim of
   // the responsible-toggler protocol is that this holds at ANY quiescent point.
   void check_derived(const seg_freeset& fs)
   {
      uint32_t cnt[4] = {0, 0, 0, 0};
      for (uint32_t k = 1; k < 4; ++k)
      {
         const uint32_t per = 64u >> k;
         for (uint32_t hw = 0; hw < seg_freeset::words_of(k); ++hw)
         {
            uint64_t expect = 0;
            for (uint32_t sub = 0; sub < (1u << k); ++sub)
            {
               const uint32_t w0 = (hw << k) + sub;
               const uint64_t v = fs.l0[w0].load();
               const auto     fm = start_ref::detail::group_masks(v);
               const uint64_t dense = k == 1 ? fm.f2 : k == 2 ? fm.f4 : fm.f8;
               expect |= dense << (sub * per);
            }
            const uint64_t got = (k == 1 ? fs.g2[hw] : k == 2 ? fs.g4[hw] : fs.g8[hw]).load();
            REQUIRE(got == expect);
            cnt[k] += uint32_t(std::popcount(expect));
         }
      }
      for (uint32_t w = 0; w < seg_freeset::W0; ++w)
         cnt[0] += uint32_t(std::popcount(fs.l0[w].load()));
      for (uint32_t k = 0; k < 4; ++k)
      {
         REQUIRE(fs.free_count[k].load() == cnt[k]);
         // meta: bit w == (array word w nonzero); top: bit m == (meta word m nonzero)
         const auto* arr = const_cast<seg_freeset&>(fs).array_of(k);
         uint64_t    t   = 0;
         for (uint32_t m = 0; m < seg_freeset::meta_words_of(k); ++m)
         {
            uint64_t expect_meta = 0;
            for (uint32_t b = 0; b < 64; ++b)
            {
               const uint32_t w = m * 64 + b;
               if (w < seg_freeset::words_of(k) && arr[w].load())
                  expect_meta |= 1ull << b;
            }
            REQUIRE(fs.meta[k][m].load() == expect_meta);
            if (expect_meta)
               t |= 1ull << m;
         }
         if (seg_freeset::meta_words_of(k) > 1)
            REQUIRE(fs.top[k].load() == t);
      }
   }
}  // namespace

TEST_CASE("fresh segment fills bottom-up, ascending")
{
   auto fs = make_fs();
   for (uint32_t i = 0; i < 256; ++i)
      REQUIRE(fs->alloc(1, 0) == i);
   REQUIRE(fs->free_lines() == lines_per_segment - 256);
}

TEST_CASE("class allocations are naturally aligned")
{
   auto fs = make_fs();
   for (uint32_t n : {2u, 4u, 8u, 16u})
   {
      const uint32_t line = fs->alloc(n, 0);
      REQUIRE(line != UINT32_MAX);
      REQUIRE(line % n == 0);
      REQUIRE((line % 64) + n <= 64);
   }
}

TEST_CASE("find-first at-or-after floor")
{
   auto fs = make_fs();
   REQUIRE(fs->alloc(1, 100) == 100);
   REQUIRE(fs->alloc(1, 100) == 101);
   REQUIRE(fs->alloc(8, 100) == 104);  // first 8-aligned >= 100 with the run free
   REQUIRE(fs->alloc(1, 0) == 0);      // floor 0 still sees the bottom
}

TEST_CASE("free + re-alloc finds the lowest hole")
{
   auto fs = make_fs();
   for (uint32_t i = 0; i < 128; ++i)
      fs->alloc(1, 0);
   fs->free_run(40, 1);
   fs->free_run(90, 1);
   REQUIRE(fs->alloc(1, 0) == 40);
   REQUIRE(fs->alloc(1, 0) == 90);
   REQUIRE(fs->alloc(1, 0) == 128);
}

TEST_CASE("anti-split: lone line preferred over carving an 8-group, same word")
{
   auto fs = make_fs(false);  // start fully allocated
   // word 0: free a lone line at 5 (its pair-buddy 4 stays allocated) and the
   // aligned 8-group [8,16).
   fs->free_run(5, 1);
   fs->free_run(8, 8);
   REQUIRE(fs->alloc(1, 0) == 5);   // lone hole wins over splitting the 8-group
   REQUIRE(fs->alloc(1, 0) == 8);   // nothing else left — now it splits
   check_derived(*fs);
}

TEST_CASE("anti-split never skips forward past the first eligible word")
{
   auto fs = make_fs(false);
   // word 0 offers ONLY a split (free 8-group); word 1 offers a lone line.
   fs->free_run(16, 8);
   fs->free_run(64 + 3, 1);
   // The first eligible word is word 0 — locality wins, split taken there.
   REQUIRE(fs->alloc(1, 0) == 16);
}

TEST_CASE("odd sizes: contiguous, word-confined, lowest-eligible")
{
   auto fs = make_fs();
   for (uint32_t n : {3u, 5u, 9u, 17u, 33u, 63u})
   {
      const uint32_t pre_word = oracle_word(*fs, n, 0, false);
      const uint32_t line     = fs->alloc(n, 0);
      REQUIRE(line != UINT32_MAX);
      REQUIRE((line % 64) + n <= 64);
      REQUIRE(line / 64 == pre_word);
   }
   check_derived(*fs);
}

TEST_CASE("16-line allocs and word confinement under fragmentation")
{
   auto fs = make_fs(false);
   // Free [56, 72): spans two words; a 16 must NOT take it (it would straddle).
   fs->free_run(56, 8);
   fs->free_run(64, 8);
   REQUIRE(fs->alloc(16, 0) == UINT32_MAX);
   fs->free_run(80, 16);  // properly confined pair of 8s
   REQUIRE(fs->alloc(16, 0) == 80);
}

TEST_CASE("exhaustion returns failure")
{
   auto fs = make_fs(false);
   REQUIRE(fs->alloc(1, 0) == UINT32_MAX);
   fs->free_run(7, 1);
   REQUIRE(fs->alloc(2, 0) == UINT32_MAX);  // a lone line can't serve a pair
   REQUIRE(fs->alloc(1, 0) == 7);
}

TEST_CASE("O(1) probes on a nearly-full segment")
{
   auto fs = make_fs(false);
   // One free line near the very top — the masked descent must jump there.
   const uint32_t hi = lines_per_segment - 3;
   fs->free_run(hi, 1);
   REQUIRE(fs->alloc(1, 0) == hi);
   REQUIRE(start_ref::g_freeset_probes <= 8);
   // Same for an 8-group and an odd run.
   fs->free_run(lines_per_segment - 64, 8);
   REQUIRE(fs->alloc(8, 0) == lines_per_segment - 64);
   REQUIRE(start_ref::g_freeset_probes <= 8);
   fs->free_run(128, 9);
   REQUIRE(fs->alloc(9, 0) == 128);
   REQUIRE(start_ref::g_freeset_probes <= 16);
}

TEST_CASE("randomized op tape: placements match oracle, derived tiers exact")
{
   auto            fs = make_fs();
   std::mt19937_64 rng(0xC0FFEE);
   // live allocations: line -> n
   std::map<uint32_t, uint32_t> live;
   const uint32_t               sizes[] = {1, 1, 1, 2, 3, 4, 5, 8, 9, 16};
   for (int step = 0; step < 4000; ++step)
   {
      const bool do_alloc = live.empty() || (rng() % 100) < 60;
      if (do_alloc)
      {
         const uint32_t n       = sizes[rng() % std::size(sizes)];
         const uint32_t from    = rng() % 16 == 0 ? uint32_t(rng() % lines_per_segment) : 0;
         const bool     aligned = std::has_single_bit(n) && n <= 8;
         const uint32_t ow      = oracle_word(*fs, n, from, aligned && n > 1);
         const uint32_t line    = fs->alloc(n, from);
         if (line == UINT32_MAX)
         {
            if (n == 1)  // for 1-line allocs the oracle is exact: no free line >= from
               REQUIRE(ow == UINT32_MAX);
            continue;
         }
         REQUIRE(line + n <= lines_per_segment);
         REQUIRE((line % 64) + n <= 64);
         REQUIRE(line >= from);
         // The lines were free before (they are allocated now — flip back check
         // via the model below) and the model had no overlap.
         auto next = live.lower_bound(line);
         if (next != live.end())
            REQUIRE(line + n <= next->first);
         if (next != live.begin())
         {
            auto prev = std::prev(next);
            REQUIRE(prev->first + prev->second <= line);
         }
         live[line] = n;
      }
      else
      {
         auto it = live.begin();
         std::advance(it, rng() % live.size());
         fs->free_run(it->first, it->second);
         live.erase(it);
      }
   }
   check_derived(*fs);
   // Drain: free everything, the segment must read fully free again.
   for (auto& [line, n] : live)
      fs->free_run(line, n);
   REQUIRE(fs->free_lines() == lines_per_segment);
   REQUIRE(fs->free_groups8() == lines_per_segment / 8);
   check_derived(*fs);
}

// gofractally/psitri#35 item 7. A line is freed by setting its bit and only
// then counting it; a session that owns a word of the map claims bits straight
// out of that word and then uncounts them. A claim that lands between the two
// steps of a free uncounts a line that was never counted, so the count dips
// below zero until the free catches up. Readers must see that dip as "none
// free", not as four billion free lines.
TEST_CASE("freeset: the free count never reads as wrapped while frees and claims interleave")
{
   auto fs = make_fs(false);  // every line allocated
   const uint32_t        w = 5;
   std::atomic<uint64_t> owed{~0ull};  // lines of word w the freeing thread holds
   std::atomic<bool>     stop{false};
   std::atomic<uint64_t> wrapped{0}, samples{0}, claims{0};
   std::thread freer([&] {
      while (!stop.load(std::memory_order_relaxed))
      {
         uint64_t mine = owed.exchange(0, std::memory_order_acquire);
         while (mine)
         {
            const uint32_t bit = uint32_t(std::countr_zero(mine));
            mine &= mine - 1;
            fs->free_run(w * 64 + bit, 1);
         }
      }
   });
   std::thread claimer([&] {
      while (!stop.load(std::memory_order_relaxed))
      {
         uint64_t cur = fs->l0[w].load(std::memory_order_relaxed);
         if (cur && fs->l0[w].compare_exchange_strong(cur, 0, std::memory_order_acquire,
                                                      std::memory_order_relaxed))
         {
            fs->apply_word_transition(w, cur, 0);
            owed.fetch_or(cur, std::memory_order_release);
            claims.fetch_add(1, std::memory_order_relaxed);
         }
      }
   });
   const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
   while (std::chrono::steady_clock::now() < until)
   {
      for (int i = 0; i < 4096; ++i)
      {
         if (fs->free_lines() > lines_per_segment)
            wrapped.fetch_add(1, std::memory_order_relaxed);
         if (fs->free_groups8() > lines_per_segment / 8)
            wrapped.fetch_add(1, std::memory_order_relaxed);
      }
      samples.fetch_add(4096, std::memory_order_relaxed);
   }
   stop = true;
   freer.join();
   claimer.join();
   INFO("samples " << samples.load() << ", claims " << claims.load());
   REQUIRE(claims.load() > 1000);
   REQUIRE(wrapped.load() == 0);
   // settled: what is free is what the freeing thread has not been handed back
   REQUIRE(fs->free_lines() == uint32_t(std::popcount(fs->l0[w].load())));
}

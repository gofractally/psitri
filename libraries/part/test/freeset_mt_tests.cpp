// SPEC §15 test 2 — seg_freeset under ThreadSanitizer: N threads hammer
// alloc/free on one segment. Invariants: no line is ever owned twice (per-thread
// ledgers are disjoint), l0 popcount reconciles with the merged ledger, and all
// derived tiers are exact at quiescence.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <part/freeset.hpp>

using part::lines_per_segment;
using part::seg_freeset;

namespace
{
   void check_derived_exact(seg_freeset& fs)
   {
      uint32_t cnt[4] = {0, 0, 0, 0};
      for (uint32_t w = 0; w < seg_freeset::W0; ++w)
         cnt[0] += uint32_t(std::popcount(fs.l0[w].load()));
      for (uint32_t k = 1; k < 4; ++k)
      {
         const uint32_t per = 64u >> k;
         for (uint32_t hw = 0; hw < seg_freeset::words_of(k); ++hw)
         {
            uint64_t expect = 0;
            for (uint32_t sub = 0; sub < (1u << k); ++sub)
            {
               const auto fm = part::detail::group_masks(fs.l0[(hw << k) + sub].load());
               expect |= uint64_t(k == 1 ? fm.f2 : k == 2 ? fm.f4 : fm.f8) << (sub * per);
            }
            REQUIRE(fs.array_of(k)[hw].load() == expect);
            cnt[k] += uint32_t(std::popcount(expect));
         }
      }
      for (uint32_t k = 0; k < 4; ++k)
         REQUIRE(fs.free_count[k].load() == cnt[k]);
   }
}  // namespace

TEST_CASE("concurrent alloc/free: disjoint ownership, exact reconciliation")
{
   auto fs = std::make_unique<seg_freeset>();
   fs->init(true);

   constexpr int                          NT = 8;
   constexpr int                          OPS = 20000;
   std::vector<std::vector<std::pair<uint32_t, uint32_t>>> held(NT);
   std::atomic<uint64_t>                  total_allocs{0};

   auto worker = [&](int tid)
   {
      std::mt19937_64 rng(uint64_t(tid) * 7919 + 1);
      auto&           mine    = held[tid];
      const uint32_t  sizes[] = {1, 1, 2, 3, 4, 8, 9, 16};
      for (int i = 0; i < OPS; ++i)
      {
         if (mine.empty() || (rng() % 100) < 55)
         {
            const uint32_t n    = sizes[rng() % std::size(sizes)];
            const uint32_t from = uint32_t(rng() % lines_per_segment);
            uint32_t       line = fs->alloc(n, from);
            if (line == UINT32_MAX)
               line = fs->alloc(n, 0);
            if (line != UINT32_MAX)
            {
               mine.emplace_back(line, n);
               total_allocs.fetch_add(1, std::memory_order_relaxed);
            }
         }
         else
         {
            const size_t j = rng() % mine.size();
            fs->free_run(mine[j].first, mine[j].second);
            mine[j] = mine.back();
            mine.pop_back();
         }
      }
   };
   std::vector<std::thread> ts;
   for (int t = 0; t < NT; ++t)
      ts.emplace_back(worker, t);
   for (auto& t : ts)
      t.join();

   // Disjointness: collect every held line into a bitmap; duplicates = overlap.
   std::vector<uint8_t> owned(lines_per_segment, 0);
   uint64_t             held_lines = 0;
   for (auto& v : held)
      for (auto [line, n] : v)
         for (uint32_t i = 0; i < n; ++i)
         {
            REQUIRE(owned[line + i] == 0);
            owned[line + i] = 1;
            ++held_lines;
         }
   REQUIRE(fs->free_lines() == lines_per_segment - held_lines);
   check_derived_exact(*fs);

   for (auto& v : held)
      for (auto [line, n] : v)
         fs->free_run(line, n);
   REQUIRE(fs->free_lines() == lines_per_segment);
   check_derived_exact(*fs);
}

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

#include <start_ver/freeset.hpp>

using start_ver::lines_per_segment;
using start_ver::seg_freeset;

namespace
{
   void check_derived_exact(seg_freeset& fs)
   {
      uint32_t f8 = 0;
      for (uint32_t m = 0; m < seg_freeset::PW; ++m)
      {
         uint64_t expect[5] = {0, 0, 0, 0, 0};
         for (uint32_t b = 0; b < 64; ++b)
         {
            const uint64_t v = fs.l0[m * 64 + b].load();
            if (v)
               expect[0] |= 1ull << b;
            if (start_ver::detail::pos2(v))
               expect[1] |= 1ull << b;
            if (start_ver::detail::pos4(v))
               expect[2] |= 1ull << b;
            const uint64_t p8 = start_ver::detail::pos8(v);
            if (p8)
               expect[3] |= 1ull << b;
            if (start_ver::detail::pos16(v))
               expect[4] |= 1ull << b;
            f8 += uint32_t(std::popcount(p8));
         }
         for (uint32_t c = 0; c < 5; ++c)
            REQUIRE(fs.pres[c][m].load() == expect[c]);
      }
      REQUIRE(fs.free_groups8() == f8);
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

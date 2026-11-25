// implicit_ref_pool — self-freeing 1-byte per-range refcounts on colo_alloc<u8>.
// Validates the lifetime model (alloc=1, retain++, release--→was_last frees the slot),
// conservation under churn, concurrent retain/release safety, and the recovery re-tally
// (counts reconstructed from replayed handles into a freshly-cleared pool).
#include <catch2/catch_test_macros.hpp>

#include <sal/implicit_ref_pool.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace
{
   fs::path fresh(const char* n)
   {
      auto p = fs::temp_directory_path() / n;
      fs::remove_all(p);
      return p;
   }
}  // namespace

TEST_CASE("implicit_ref lifetime: alloc=1, retain, release, last-frees", "[sal][implicit_ref]")
{
   auto                  dir = fresh("iref_life");
   sal::implicit_ref_pool p(dir);

   auto a = p.alloc_fresh();
   REQUIRE(p.count(a) == 1);
   REQUIRE(p.try_retain(a));
   REQUIRE(p.try_retain(a));
   REQUIRE(p.count(a) == 3);          // 1 + 2 clones share this range
   REQUIRE_FALSE(p.release(a));       // 3 -> 2, not last
   REQUIRE_FALSE(p.release(a));       // 2 -> 1, not last
   REQUIRE(p.used() == 1);
   REQUIRE(p.release(a));             // 1 -> 0, WAS LAST → slot freed
   REQUIRE(p.used() == 0);
   fs::remove_all(dir);
}

TEST_CASE("implicit_ref conservation: random retain/release churn", "[sal][implicit_ref]")
{
   auto                  dir = fresh("iref_churn");
   sal::implicit_ref_pool p(dir);
   std::mt19937          rng(12345);

   // Model: address -> live share count. Mirror every pool op and assert agreement.
   std::unordered_map<uint32_t, int> model;
   std::vector<uint32_t>             live;

   for (int step = 0; step < 200000; ++step)
   {
      bool do_alloc = live.empty() || (rng() % 3 != 0);
      if (do_alloc)
      {
         auto a       = p.alloc_fresh();
         model[*a]    = 1;
         live.push_back(*a);
      }
      else
      {
         uint32_t a = live[rng() % live.size()];
         if (rng() & 1)  // retain
         {
            if (model[a] < 250) { REQUIRE(p.try_retain(sal::ptr_address(a))); ++model[a]; }
         }
         else  // release
         {
            bool last = p.release(sal::ptr_address(a));
            REQUIRE(last == (model[a] == 1));   // pool agrees with the model on last-ref
            if (--model[a] == 0)
            {
               model.erase(a);
               live.erase(std::find(live.begin(), live.end(), a));
            }
         }
      }
   }
   REQUIRE(p.used() == model.size());            // conservation: live slots == model entries
   for (auto& [addr, cnt] : model)
      REQUIRE(int(p.count(sal::ptr_address(addr))) == cnt);  // every live count matches
   fs::remove_all(dir);
}

TEST_CASE("implicit_ref concurrent retain/release is conservative", "[sal][implicit_ref]")
{
   auto                  dir = fresh("iref_mt");
   sal::implicit_ref_pool p(dir);

   // Pre-allocate K slots each with a large refcount; N threads each do one release per slot.
   // Exactly one thread per slot should observe `was_last`, and the slot count must end at 0.
   constexpr int            K = 20000, threads = 8;
   std::vector<uint32_t>    slots(K);
   for (int i = 0; i < K; ++i)
   {
      auto a   = p.alloc_fresh();        // count = 1
      slots[i] = *a;
      for (int t = 1; t < threads; ++t)  // bring to `threads` so each thread releases once
         REQUIRE(p.try_retain(sal::ptr_address(slots[i])));
      REQUIRE(p.count(sal::ptr_address(slots[i])) == threads);
   }

   std::vector<std::atomic<int>> last_count(K);
   for (auto& x : last_count) x.store(0);
   std::vector<std::thread> ts;
   for (int t = 0; t < threads; ++t)
      ts.emplace_back([&] {
         for (int i = 0; i < K; ++i)
            if (p.release(sal::ptr_address(slots[i])))
               last_count[i].fetch_add(1, std::memory_order_relaxed);
      });
   for (auto& th : ts) th.join();

   for (int i = 0; i < K; ++i)
      REQUIRE(last_count[i].load() == 1);  // exactly one releaser saw last-ref per slot
   REQUIRE(p.used() == 0);                 // all slots freed
   fs::remove_all(dir);
}

// A count is one byte. The 256th sharer is refused instead of wrapping the count to
// zero, which would free the slot (and the range's branches) under 255 live sharers.
TEST_CASE("implicit_ref ceiling: the 256th sharer is refused and nothing wraps", "[sal][implicit_ref]")
{
   auto                   dir = fresh("iref_ceiling");
   sal::implicit_ref_pool p(dir);

   auto a = p.alloc_fresh();  // count = 1
   for (int i = 1; i < sal::implicit_ref_max; ++i)
      REQUIRE(p.try_retain(a));
   REQUIRE(p.count(a) == sal::implicit_ref_max);
   REQUIRE_FALSE(p.try_retain(a));
   REQUIRE(p.count(a) == sal::implicit_ref_max);

   // Many threads at the ceiling together: exactly as many succeed as there is room for.
   auto b = p.alloc_fresh();
   for (int i = 1; i < 200; ++i)
      REQUIRE(p.try_retain(b));
   std::atomic<int>         taken{0};
   std::vector<std::thread> ts;
   for (int t = 0; t < 8; ++t)
      ts.emplace_back([&] {
         for (int i = 0; i < 40; ++i)
            if (p.try_retain(b))
               taken.fetch_add(1, std::memory_order_relaxed);
      });
   for (auto& th : ts) th.join();
   REQUIRE(taken.load() == sal::implicit_ref_max - 200);
   REQUIRE(p.count(b) == sal::implicit_ref_max);

   // Every share is given back once; only the last one frees the slot.
   for (auto slot : {a, b})
   {
      for (int i = 1; i < sal::implicit_ref_max; ++i)
         REQUIRE_FALSE(p.release(slot));
      REQUIRE(p.release(slot));
   }
   REQUIRE(p.used() == 0);
   fs::remove_all(dir);
}

TEST_CASE("implicit_ref recovery: counts re-tallied from replayed handles", "[sal][implicit_ref]")
{
   auto                  dir = fresh("iref_recover");
   sal::implicit_ref_pool p(dir);
   std::mt19937          rng(777);

   // Build a population of slots with assorted share counts (simulating live versions sharing
   // ranges). Record the (handle -> count) truth — this is what a recovery walk must rebuild.
   std::unordered_map<uint32_t, int> truth;
   for (int i = 0; i < 30000; ++i)
   {
      auto    a   = p.alloc_fresh();   // 1
      int     extra = int(rng() % 5);  // 0..4 more versions share it
      for (int k = 0; k < extra; ++k) REQUIRE(p.try_retain(sal::ptr_address(a)));
      truth[*a] = 1 + extra;
   }
   REQUIRE(p.used() == truth.size());

   // "Crash": counts are ephemeral. Clear the pool, then replay every reference the live nodes
   // would carry as persistent handles — for slot s with count c, c (node,range) handles name s.
   p.clear_all();
   REQUIRE(p.used() == 0);
   // replay in shuffled order to prove order-independence
   std::vector<uint32_t> replay;
   for (auto& [addr, cnt] : truth)
      for (int k = 0; k < cnt; ++k) replay.push_back(addr);
   std::shuffle(replay.begin(), replay.end(), rng);
   for (auto a : replay) p.retally(sal::ptr_address(a));

   // After the walk: allocation set + every count reconstructed exactly.
   REQUIRE(p.used() == truth.size());
   size_t bad = 0;
   for (auto& [addr, cnt] : truth)
      if (int(p.count(sal::ptr_address(addr))) != cnt) ++bad;
   REQUIRE(bad == 0);
   fs::remove_all(dir);
}

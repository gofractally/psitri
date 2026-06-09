// colo_alloc<T> — the generic co-locating slab allocator (control_block_alloc generalized).
// Validates the templated slot machinery for both extremes (control_block = 16 slots/line,
// u8 = 128 slots/line), co-location via hints, concurrent conservation (no slot double-handed,
// no lost free), and a DIFFERENTIAL gate against the original control_block_alloc.
#include <catch2/catch_test_macros.hpp>

#include <sal/colo_alloc.hpp>
#include <sal/control_block.hpp>
#include <sal/control_block_alloc.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace
{
   fs::path fresh(const char* name)
   {
      auto p = fs::temp_directory_path() / name;
      fs::remove_all(p);
      return p;
   }
   // slots-per-cacheline for an address: which 128B line it lands in.
   template <class A>
   uint32_t line_of(uint32_t addr) { return addr / A::slots_per_cacheline; }
}  // namespace

TEST_CASE("colo_alloc u8 basic alloc/get/free/conservation", "[sal][colo_alloc]")
{
   auto                     dir = fresh("colo_u8_basic");
   sal::colo_alloc<uint8_t> pool(dir);
   REQUIRE(sal::colo_alloc<uint8_t>::slots_per_cacheline == 128);  // 128B line / 1B

   constexpr int                 N = 200000;
   std::vector<sal::ptr_address> hs;
   hs.reserve(N);
   for (int i = 0; i < N; ++i)
   {
      auto a   = pool.alloc();
      *a.ptr   = uint8_t(i & 0xff);  // write through the returned slot pointer
      hs.push_back(a.addr_seq.address);
   }
   REQUIRE(pool.used() == N);

   // every address distinct (no slot double-handed) and value round-trips
   std::unordered_set<uint32_t> seen;
   size_t                       bad = 0;
   for (int i = 0; i < N; ++i)
   {
      REQUIRE(seen.insert(*hs[i]).second);
      if (pool.get(hs[i]) != uint8_t(i & 0xff)) ++bad;
   }
   REQUIRE(bad == 0);

   for (int i = 0; i < N; i += 2) pool.free(hs[i]);
   REQUIRE(pool.used() == N - N / 2);
   for (int i = 1; i < N; i += 2) pool.free(hs[i]);
   REQUIRE(pool.used() == 0);
   fs::remove_all(dir);
}

TEST_CASE("colo_alloc control_block basic", "[sal][colo_alloc]")
{
   auto                            dir = fresh("colo_cb_basic");
   sal::colo_alloc<sal::control_block> pool(dir);
   REQUIRE(sal::colo_alloc<sal::control_block>::slots_per_cacheline == 16);  // 128B / 8B

   std::vector<sal::ptr_address> hs;
   for (int i = 0; i < 50000; ++i) hs.push_back(pool.alloc().addr_seq.address);
   REQUIRE(pool.used() == 50000);
   for (auto a : hs) pool.free(a);
   REQUIRE(pool.used() == 0);
   fs::remove_all(dir);
}

TEST_CASE("colo_alloc hint co-locates into the same cacheline", "[sal][colo_alloc]")
{
   auto                     dir = fresh("colo_u8_hint");
   sal::colo_alloc<uint8_t> pool(dir);

   // Seed one slot, then allocate a cluster hinting at it; they should mostly land in its line.
   auto         seed = pool.alloc().addr_seq.address;
   uint32_t     line = line_of<sal::colo_alloc<uint8_t>>(*seed);
   sal::ptr_address hintbuf[1] = {seed};
   sal::alloc_hint  hint(hintbuf, 1);

   int same = 0, total = 100;  // a 128-slot line has room
   for (int i = 0; i < total; ++i)
   {
      auto a = pool.alloc(hint);
      if (line_of<sal::colo_alloc<uint8_t>>(*a.addr_seq.address) == line) ++same;
   }
   // co-location is the whole point: the large majority must share the hint's line.
   REQUIRE(same > total * 3 / 4);
   fs::remove_all(dir);
}

TEST_CASE("colo_alloc concurrent conservation (no double-hand, no lost free)", "[sal][colo_alloc]")
{
   auto                     dir = fresh("colo_u8_mt");
   sal::colo_alloc<uint8_t> pool(dir);

   constexpr int            kThreads = 8, kPer = 50000;
   std::vector<std::vector<uint32_t>> got(kThreads);
   std::vector<std::thread>           ts;
   for (int t = 0; t < kThreads; ++t)
      ts.emplace_back([&, t] {
         std::mt19937 rng(uint32_t(t * 2654435761u));
         auto&        mine = got[t];
         mine.reserve(kPer);
         for (int i = 0; i < kPer; ++i) mine.push_back(*pool.alloc().addr_seq.address);
         // free a random half back, keep the rest
         std::shuffle(mine.begin(), mine.end(), rng);
         for (int i = 0; i < kPer / 2; ++i) pool.free(sal::ptr_address(mine[i]));
         mine.erase(mine.begin(), mine.begin() + kPer / 2);
      });
   for (auto& th : ts) th.join();

   // Every still-live address is unique across all threads (no slot handed to two owners).
   std::unordered_set<uint32_t> live;
   size_t                       live_count = 0;
   for (auto& v : got)
      for (auto a : v)
      {
         REQUIRE(live.insert(a).second);
         ++live_count;
      }
   REQUIRE(pool.used() == live_count);  // conservation: header count == actually-live slots
   fs::remove_all(dir);
}

TEST_CASE("implicit_ref_addr compact handle round-trips", "[sal][colo_alloc]")
{
   using namespace sal;
   for (uint32_t a : {0u, 1u, 0xFFFFu, 0x10000u, 0xABCD1234u, 0xFFFFFFFFu})
   {
      ptr_address full(a);
      REQUIRE(*combine(region_of(full), offset_of(full)) == a);
   }
   // a region is the high 16 bits; offset the low 16
   REQUIRE(region_of(ptr_address(0xABCD1234u)) == 0xABCD);
   REQUIRE(*offset_of(ptr_address(0xABCD1234u)) == 0x1234);
}

TEST_CASE("colo_alloc region hint co-locates + falls back to the region", "[sal][colo_alloc]")
{
   using namespace sal;
   auto                     dir = fresh("colo_u8_region");
   sal::colo_alloc<uint8_t> pool(dir);

   // Seed a slot; its region is where the node's whole counter-set will live.
   auto                seed   = pool.alloc().addr_seq.address;
   implicit_ref_region region = region_of(seed);
   uint32_t            line   = line_of<sal::colo_alloc<uint8_t>>(*seed);

   // Allocate a cluster hinting at the seed: every one must stay IN THE REGION, and the
   // large majority should share the seed's cacheline.
   implicit_ref_addr sibs[1] = {offset_of(seed)};
   // A u8 cacheline holds 128 slots; with the seed taking 1, up to 127 more pack into it before
   // spill. Stay within line capacity so the assertion measures co-location, not the line filling.
   int               same_line = 0, in_region = 0, total = 100;
   for (int i = 0; i < total; ++i)
   {
      auto a = pool.try_alloc(implicit_ref_hint{region, std::span<const implicit_ref_addr>(sibs, 1)});
      REQUIRE(a.has_value());
      if (region_of(a->addr_seq.address) == region) ++in_region;
      if (line_of<sal::colo_alloc<uint8_t>>(*a->addr_seq.address) == line) ++same_line;
   }
   REQUIRE(in_region == total);          // never spills outside the region
   REQUIRE(same_line == total);          // all pack into the seed's cacheline (room for 127)
   fs::remove_all(dir);
}

TEST_CASE("colo_alloc region fills then returns nullopt (caller re-homes)", "[sal][colo_alloc]")
{
   using namespace sal;
   auto                     dir = fresh("colo_u8_regionfull");
   sal::colo_alloc<uint8_t> pool(dir);

   auto                seed   = pool.alloc().addr_seq.address;
   implicit_ref_region region = region_of(seed);

   // Drain the region (65536 slots, one already taken by seed) via the region path only.
   uint32_t got = 1;  // seed
   while (auto a = pool.try_alloc_in_region(region))
   {
      REQUIRE(region_of(a->addr_seq.address) == region);
      ++got;
   }
   REQUIRE(got == implicit_ref_region_slots);                 // exactly the whole region
   REQUIRE(!pool.try_alloc_in_region(region).has_value());    // full → nullopt, no spill
   fs::remove_all(dir);
}

// THE GATE: same op-stream through colo_alloc<control_block> and the original
// control_block_alloc — assert identical conservation at every checkpoint.
TEST_CASE("colo_alloc<control_block> matches control_block_alloc (differential)", "[sal][colo_alloc]")
{
   auto dirA = fresh("colo_diff_new");
   auto dirB = fresh("colo_diff_old");
   sal::colo_alloc<sal::control_block> A(dirA);
   sal::control_block_alloc            B(dirB);

   std::mt19937                  rng(0xABCDEF);
   std::vector<sal::ptr_address> liveA, liveB;
   for (int round = 0; round < 40; ++round)
   {
      int n = 1000 + int(rng() % 4000);
      for (int i = 0; i < n; ++i)
      {
         liveA.push_back(A.alloc().addr_seq.address);
         liveB.push_back(B.alloc().addr_seq.address);
      }
      // free a random fraction from each (independent addresses, but equal COUNTS)
      int f = int(rng() % (liveA.size() / 2 + 1));
      for (int i = 0; i < f; ++i)
      {
         std::swap(liveA[rng() % liveA.size()], liveA.back());
         std::swap(liveB[rng() % liveB.size()], liveB.back());
         A.free(liveA.back()); liveA.pop_back();
         // control_block_alloc::free asserts ref()==0; the slots were never retained (ref starts
         // at 1 on alloc), so drop it to 0 first to satisfy the precondition.
         B.get(liveB.back()).set_ref(0);
         B.free(liveB.back()); liveB.pop_back();
      }
      REQUIRE(A.used() == B.used());  // conservation tracks identically
      REQUIRE(A.used() == liveA.size());
   }
   fs::remove_all(dirA);
   fs::remove_all(dirB);
}

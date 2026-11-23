// sal_hybrid var_allocator<32> RECYCLE / high-water test — reproduces the Silkworm
// "var_allocator full" crash (sal.dat 137G committed, only 35.5% live, 11875
// segments >50% free but never recycled). The existing fuzz checks correctness
// (no overlap, drains to 0) but NOT that committed segments stay bounded under
// steady-state churn. This isolates WHERE the recycle fails:
//   A) single-class steady churn  — if segs grow unbounded, take_recyclable_/seal is broken
//   B) drifting size distribution  — if A is bounded but B grows, it's per-class stranding
// Bound check: at constant live, committed segments must stabilize (recycle reclaims
// the >50%-free segments), not march upward.
#include <sal_hybrid/var_allocator.hpp>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace sal_hybrid;
using VA = var_allocator<32>;

// Run a steady-state churn keeping ~target grains live; `size_at(op)` picks the
// alloc size per op. Returns max segments observed after warmup; prints the curve.
static uint32_t churn(const char* label, uint64_t ops, uint64_t target_grains,
                      std::mt19937& r, std::vector<std::pair<uint32_t,uint32_t>>& live,
                      VA& va, VA::session& ses, uint32_t (*size_at)(uint64_t))
{
   uint64_t live_grains = 0;
   for (auto [slot, bytes] : live)  // objects carried over from an earlier call
      live_grains += VA::grains_for(bytes);
   uint32_t warmup_seg = 0, maxseg = 0;
   const uint64_t warmup = ops / 4;
   std::printf("  [%s] target_live=%lluG ops=%lluM\n", label,
               (unsigned long long)(target_grains*32/1000000000ull?target_grains*32/1000000000ull:0),
               (unsigned long long)(ops/1000000));
   for (uint64_t i = 0; i < ops; ++i)
   {
      // free until under target, then alloc one (keeps live ~constant after warmup)
      while (live_grains >= target_grains && !live.empty())
      {
         size_t k = r() % live.size();
         auto [s, b] = live[k]; live[k] = live.back(); live.pop_back();
         uint32_t n = VA::grains_for(b);
         if (va.release_one(s)) va.free(s, b);
         live_grains -= n;
      }
      uint32_t bytes = size_at(i);
      uint32_t s = ses.alloc(bytes);
      live.push_back({s, bytes});
      live_grains += VA::grains_for(bytes);
      uint32_t segs = va.num_segments();
      if (segs > maxseg) maxseg = segs;
      if (i == warmup) warmup_seg = segs;
      if (i % (ops/8) == 0)
         std::printf("    op=%6lluk  live=%5lluk grains  segs=%u\n",
                     (unsigned long long)(i/1000), (unsigned long long)(live_grains/1000), segs);
   }
   uint32_t finalseg = va.num_segments();
   uint64_t need = (target_grains + VA::slots_per_segment - 1) / VA::slots_per_segment;
   std::printf("  [%s] live-need=%llu segs, final=%u, max=%u, warmup=%u  (ideal recycle <= ~2-3x need)\n",
               label, (unsigned long long)need, finalseg, maxseg, warmup_seg);
   return finalseg;
}

int main()
{
   namespace fs = std::filesystem;
   auto dir = fs::temp_directory_path() / "va32_recycle"; fs::remove_all(dir);
   VA va({.dir = dir, .name = "sal"});
   auto ses = va.start_session();
   std::mt19937 r(11);
   std::vector<std::pair<uint32_t,uint32_t>> live;
   const uint64_t TARGET = 4ull * VA::slots_per_segment;  // ~4 segments of live data
   int fails = 0;

   // ── A: SINGLE class (all 100B = 4 grains). Steady churn. Segs must bound. ──
   std::printf("=== A: single-class steady churn (100B) ===\n");
   uint32_t a = churn("A", 4'000'000, TARGET, r, live, va, ses,
                      [](uint64_t){ return 100u; });
   uint64_t need = (TARGET + VA::slots_per_segment - 1) / VA::slots_per_segment;
   if (a > need * 4) { ++fails; std::printf("  A FAIL: %u segs >> %llu need (single-class recycle broken)\n", a, (unsigned long long)(need*4)); }
   else std::printf("  A OK: single-class recycle bounded\n");

   // drain A
   for (auto [s,b] : live) if (va.release_one(s)) va.free(s,b);
   live.clear();

   // ── B: DRIFTING size distribution (32B..2048B sweeping). Steady churn. ──
   // Simulates Ethereum leaf-size drift across blocks. If A bounds but B does not,
   // the failure is per-class segment stranding.
   //
   // A segment holds one size class, so every class with a live object pins a
   // whole segment however little of it is used. The floor for this design is
   // therefore the live data plus one segment per class, not a multiple of the
   // live data. What must hold is (1) the count stays under that floor and
   // (2) it stops rising: more sweeps at the same live size add nothing.
   // Getting below the floor needs survivors to be moved, which this allocator
   // does not do.
   std::printf("=== B: drifting multi-class steady churn (32B..2KB sweep) ===\n");
   auto drift = [](uint64_t i){ return uint32_t(32 + (i / 2000) % 2016); };
   const uint64_t sweep = 2000ull * 2016;  // operations for one pass over every size
   uint32_t b1 = churn("B sweep 1", sweep, TARGET, r, live, va, ses, drift);
   uint32_t b3 = churn("B sweeps 2-3", 2 * sweep, TARGET, r, live, va, ses, drift);
   const uint64_t floor = need + VA::max_run;
   if (b3 > floor) { ++fails; std::printf("  B FAIL: %u segs > %llu (live need + one per size class)\n", b3, (unsigned long long)floor); }
   else if (b3 > b1 + 2) { ++fails; std::printf("  B FAIL: %u segs after one sweep, %u after three (size-drift stranding: still growing)\n", b1, b3); }
   else std::printf("  B OK: drift recycle bounded and flat (%u after one sweep, %u after three, floor %llu)\n", b1, b3, (unsigned long long)floor);

   for (auto [s,bb] : live) if (va.release_one(s)) va.free(s,bb);
   live.clear();
   std::printf("[var32-recycle] final_segs=%u fails=%d %s\n", va.num_segments(), fails, fails?"FAIL":"PASS");
   return fails ? 1 : 0;
}

// sal_gc allocator robustness gate (design §8): random alloc/free, verify no
// overlap, full reclaim, refcount lifecycle. Plain main (no Catch2 dep).
#include <sal_gc/arena.hpp>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

namespace
{
   int g_checks = 0, g_fails = 0;
   void ok(bool c, const char* what)
   {
      ++g_checks;
      if (!c)
      {
         ++g_fails;
         if (g_fails <= 16)
            std::fprintf(stderr, "FAIL: %s\n", what);
      }
   }
}  // namespace

int main()
{
   namespace fs = std::filesystem;
   const fs::path dir = fs::temp_directory_path() / "sal_gc_fuzz";
   fs::remove_all(dir);

   sal_gc::arena a(sal_gc::arena_config{.dir = dir});
   auto        ses = a.start_session();

   std::mt19937                            rng(1337);
   std::uniform_int_distribution<uint32_t> nlines(1, 9);   // cl1..cl9 span
   std::uniform_int_distribution<int>      coin(0, 1);

   // live[head] = nlines; a byte-occupancy map detects any overlap.
   std::map<sal_gc::line_idx, uint32_t> live;
   std::vector<sal_gc::line_idx>        keys;
   std::map<uint64_t, sal_gc::line_idx> occupied;  // line -> owning head (overlap check)

   const int OPS = 2'000'000;
   for (int op = 0; op < OPS; ++op)
   {
      const bool do_alloc = keys.empty() || (live.size() < 200'000 && coin(rng));
      if (do_alloc)
      {
         const uint32_t      n = nlines(rng);
         const sal_gc::line_idx g = ses.alloc_lines(n);
         ok(!sal_gc::is_sentinel(g), "alloc returned a real index");
         ok(a.ref(g) == 1, "fresh node ref==1");
         // stamp occupancy; any collision = overlap bug.
         for (uint32_t k = 0; k < n; ++k)
         {
            auto [it, ins] = occupied.emplace(uint64_t(g) + k, g);
            ok(ins, "no overlap with a live allocation");
         }
         // touch the memory (catch bad addressing / unmapped pages).
         std::memset(a.get(g), 0xAB, size_t(n) * sal_gc::line_bytes);
         live.emplace(g, n);
         keys.push_back(g);
      }
      else
      {
         // free a random live node.
         std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);
         const size_t        idx = pick(rng);
         const sal_gc::line_idx g  = keys[idx];
         const uint32_t       n  = live[g];
         ok(a.release_one(g), "release of unique node hits zero");
         a.free_lines(g, n);
         for (uint32_t k = 0; k < n; ++k)
            occupied.erase(uint64_t(g) + k);
         live.erase(g);
         keys[idx] = keys.back();
         keys.pop_back();
      }

      // exercise refcount retain/release on a random live node periodically.
      if (!keys.empty() && (op & 0x3FF) == 0)
      {
         const sal_gc::line_idx g = keys[rng() % keys.size()];
         a.retain(g);
         ok(a.ref(g) == 2, "retain -> 2");
         ok(!a.release_one(g), "release from 2 -> 1, not zero");
         ok(a.ref(g) == 1, "back to 1");
      }
   }

   // drain everything; the freeset must then report a fully-free arena.
   for (auto [g, n] : live)
   {
      (void)a.release_one(g);
      a.free_lines(g, n);
   }
   live.clear();
   const uint64_t after = a.live_lines();
   ok(after == 0, "all lines reclaimed after draining");

   std::printf("sal_gc-tests: %d checks, %d fails (segments grown=%u)\n", g_checks, g_fails,
               a.num_segments());
   return g_fails == 0 ? 0 : 1;
}

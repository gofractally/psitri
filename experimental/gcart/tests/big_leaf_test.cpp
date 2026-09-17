// gcart big_suffix_value round-trip — exercises big-leaf alloc + FREE. Phase 2:
// big leaves are SAL big_leaf objects (alloc<big_leaf>) in the SAME leaf
// allocator as small sv_leaf (the SAL large-object path takes the bigger size),
// so this gate confirms big insert / overwrite / remove reclaim correctly via the
// SAL control_block refcount (don't assert/leak), and the leaf allocator
// drains to 0 live bytes after remove-all.
#include <gcart/tree.hpp>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>
#include <string>
using namespace gcart;

int main()
{
   namespace fs = std::filesystem;
   auto dir     = fs::temp_directory_path() / "gcart_bigleaf";
   fs::remove_all(dir);
   int  fails = 0;
   auto ck    = [&](bool c, const char* m) { if (!c) { ++fails; std::printf("FAIL %s\n", m); } };

   engine        e{dir};
   tree          t(e);
   std::map<std::string, std::string> oracle;
   std::mt19937  r(99);
   const int     N = 3000;
   auto          bigval = [&](uint32_t seed, uint32_t len)
   { std::string s(len, char('A' + (seed & 15))); for (uint32_t i = 0; i < len; i += 64) s[i] = char(seed); return s; };
   std::string out;

   // insert N big values (4–9 KB each → BIG region, since > sal_max_body ~2KB)
   for (int i = 0; i < N; ++i)
   {
      char k[16];
      std::memcpy(k, &i, 4); std::memset(k + 4, 0, 12);
      std::string v = bigval(uint32_t(i), 4096 + (r() % 5000));
      t.upsert(std::string_view(k, 16), v);
      oracle[std::string(k, 16)] = v;
   }
   for (auto& [k, v] : oracle)
      ck(t.find(k, out) && out == v, "find after insert");

   // OVERWRITE each with a different big value → frees the OLD big leaf (the #3 path)
   for (int i = 0; i < N; ++i)
   {
      char k[16];
      std::memcpy(k, &i, 4); std::memset(k + 4, 0, 12);
      std::string v = bigval(uint32_t(i) ^ 0xABCD, 4096 + (r() % 5000));
      t.upsert(std::string_view(k, 16), v);
      oracle[std::string(k, 16)] = v;
   }
   for (auto& [k, v] : oracle)
      ck(t.find(k, out) && out == v, "find after overwrite");

   // remove all → frees every live big leaf
   for (auto& [k, v] : oracle)
      ck(t.remove(k) == 1, "remove present");
   for (auto& [k, v] : oracle)
      ck(!t.find(k, out), "gone after remove");

   e.drain();
   const uint64_t live = e.live_bytes();  // total (FIX routing + SAL leaf)
   ck(live == 0, "engine drained to 0 (leaf big objects freed) after remove-all");
   std::printf("big-leaf-test: %s (fails=%d, total live=%lluB after drain)\n",
               fails ? "FAIL" : "PASS", fails, (unsigned long long)live);
   return fails ? 1 : 0;
}

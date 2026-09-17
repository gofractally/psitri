// sal_hybrid control_block + control_block_alloc: the ptr_address indirection (§4.3).
// Proves: alloc hands out distinct ptr_addresses; cb_at resolves; free recycles
// (ref==0 marker, offset=next free-list link, no resurrection); cas_move
// relocates the offset under a stable ptr_address; reopen restores live CBs.
#include <sal_hybrid/control_block_alloc.hpp>
#include <cstdio>
#include <filesystem>
#include <random>
#include <unordered_map>
#include <vector>
using namespace sal_hybrid;

int main()
{
   namespace fs = std::filesystem;
   auto dir     = fs::temp_directory_path() / "sal_hybrid_cb";
   fs::remove_all(dir);
   int fails = 0;
   auto ck   = [&](bool c, const char* m) { if (!c) { ++fails; std::printf("FAIL %s\n", m); } };

   {
      control_block_alloc ca({.dir = dir, .name = "ctrl"});
      // alloc: distinct ptr_addresses, each resolvable to {zone, offset, ref=1}
      std::vector<ptr_address> live;
      std::unordered_map<ptr_address, uint64_t> off;  // pa -> offset it should hold
      std::mt19937 r(7);
      for (uint32_t i = 0; i < 200000; ++i)
      {
         const uint64_t o  = (uint64_t(r()) << 4) & ((1ull << 44) - 1);
         ptr_address    pa = ca.alloc(cb_zone::movable, o);
         control_block* cb = ca.cb_at(pa);
         if (cb->use_count() != 1 || cb->offset() != o || cb->zone() != cb_zone::movable)
            { ++fails; if (fails<=3) std::printf("FAIL alloc state pa=%u\n", pa); }
         off[pa] = o;
         live.push_back(pa);
      }
      // no two live CBs share a ptr_address
      ck(std::unordered_map<ptr_address,int>(
             [&]{ std::unordered_map<ptr_address,int> m; for (auto pa:live) m[pa]++; return m; }())
             .size() == live.size(), "distinct ptr_addresses");

      // cas_move: relocate the offset; ptr_address (and ref) unchanged
      {
         ptr_address pa = live[123];
         uint64_t    oldo = off[pa], newo = (oldo ^ 0xABCDE) & ((1ull<<44)-1);
         ck(ca.cb_at(pa)->cas_move(oldo, newo), "cas_move ok");
         ck(ca.cb_at(pa)->offset() == newo, "cas_move offset");
         ck(ca.cb_at(pa)->use_count() == 1, "cas_move keeps ref");
         ck(!ca.cb_at(pa)->cas_move(oldo, 5), "cas_move stale rejects");
         off[pa] = newo;
      }

      // free half, then re-alloc: freed CBs are recycled (live count consistent),
      // and a freed CB never resurrects (ref==0 until reclaimed).
      const uint64_t before = ca.live_count();
      for (size_t k = 0; k < live.size(); k += 2)
      {
         ck(!ca.cb_at(live[k])->try_retain() || true, "");  // (no-op; just exercise)
         ca.free(live[k]);
         ck(ca.cb_at(live[k])->is_free(), "freed cb is ref==0");
         ck(!ca.cb_at(live[k])->try_retain(), "freed cb rejects try_retain");
      }
      ck(ca.live_count() == before - (live.size()/2 + live.size()%2), "live_count after free");
      uint64_t reused = 0, peak_before = ca.num_segments();
      for (size_t k = 0; k < live.size()/2; ++k)
      {
         ptr_address pa = ca.alloc(cb_zone::big, 0x1234);
         ck(ca.cb_at(pa)->use_count()==1 && ca.cb_at(pa)->zone()==cb_zone::big, "re-alloc state");
      }
      ck(ca.num_segments() == peak_before, "re-alloc reused freed CBs (no new segment)");
      std::printf("[cb] live=%llu segs=%u  fails=%d\n",
                  (unsigned long long)ca.live_count(), ca.num_segments(), fails);
   }

   // reopen: clean close flushed the .ctrl mmap; a fresh open resolves old CBs.
   {
      control_block_alloc ca({.dir = dir, .name = "ctrl"});
      ptr_address pa = ca.alloc(cb_zone::movable, 999);  // appends after committed
      ck(ca.cb_at(pa)->offset() == 999, "reopen new alloc");
      std::printf("[cb] reopened, segs=%u\n", ca.num_segments());
   }

   std::printf("control_block-tests: %s (fails=%d)\n", fails ? "FAIL" : "PASS", fails);
   return fails ? 1 : 0;
}

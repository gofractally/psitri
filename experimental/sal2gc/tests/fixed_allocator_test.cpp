// sal2gc FixedAllocator<N> fuzz — the spec's dev-plan step 1: the per-size-class
// slab (cl1=128/cl2=256/cl9=1152) must be robust against random alloc/free.
// Verifies: no slot double-allocated while live, born `death_alive` on alloc,
// stamp_death→reclaim returns the slot to free, and the arena fully drains.
// ~6M ops across classes. (Version-liveness model: "delete" = stamp a death
// version, then the GC reclaims; here the test reclaims immediately.)
#include <sal2gc/fixed_allocator.hpp>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>
using namespace sal2gc;

template<uint32_t SB>
int fuzz(const char* name, int ops){
   namespace fs=std::filesystem; auto dir=fs::temp_directory_path()/("fa_"+std::string(name)); fs::remove_all(dir);
   fixed_allocator<SB> fa(typename fixed_allocator<SB>::config{.dir=dir, .name=name});
   auto ses=fa.start_session();
   std::mt19937 r(99);
   std::vector<uint32_t> live;
   std::unordered_set<uint32_t> set;   // detect double-alloc of the same slot
   int fails=0;
   version_t ver=version_first;        // monotonic commit version for death stamps
   auto del=[&](uint32_t s){ fa.stamp_death(s, ver++); fa.reclaim(s); };  // delete = stamp+GC
   for(int i=0;i<ops;++i){
      bool doAlloc = live.empty() || (r()&1);
      if(doAlloc){
         uint32_t s=ses.alloc();
         if(!set.insert(s).second){ if(++fails<=5) std::printf("  DOUBLE-ALLOC slot %u\n",s); }
         if(!fa.alive(s)){ if(++fails<=5) std::printf("  not alive on alloc, death=%u\n",fa.death(s)); }
         // touch the slot (ensure addressable)
         std::memset(fa.get(s),0xAB,SB);
         live.push_back(s);
      } else {
         size_t k=r()%live.size(); uint32_t s=live[k]; live[k]=live.back(); live.pop_back();
         set.erase(s);
         del(s);
      }
   }
   // drain
   for(uint32_t s: live){ set.erase(s); del(s); }
   live.clear();
   uint64_t after = fa.live_slots();
   if(after!=0){ ++fails; std::printf("  NOT drained: %llu live slots\n",(unsigned long long)after); }
   std::printf("[%s] slot=%uB slots/seg=%u ops=%d segs=%u fails=%d  %s\n",
      name, SB, fixed_allocator<SB>::slots_per_segment, ops, fa.num_segments(), fails,
      fails?"FAIL":"PASS");
   return fails;
}
int main(){
   int f=0;
   f+=fuzz<128>("cl1", 3000000);
   f+=fuzz<256>("cl2", 2000000);
   f+=fuzz<1152>("cl9", 1000000);
   std::printf("TOTAL fails=%d\n",f);
   return f?1:0;
}

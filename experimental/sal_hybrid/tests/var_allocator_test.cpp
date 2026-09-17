// sal_hybrid var_allocator<Grain> fuzz — the SAL/movable region (suffix_value leaves,
// 32B grain). Variable-size alloc/free robust: no live runs overlap, ref==1 on
// alloc, release_one->free reclaims, arena drains. 2M ops across many segments.
#include <sal_hybrid/var_allocator.hpp>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <unordered_map>
using namespace sal_hybrid;
int main(){
   namespace fs=std::filesystem; auto dir=fs::temp_directory_path()/"va32"; fs::remove_all(dir);
   var_allocator<32> va({.dir=dir,.name="sal"});
   auto ses=va.start_session();
   std::mt19937 r(7);
   struct Obj{uint32_t slot,bytes;};
   std::vector<Obj> live;
   std::unordered_map<uint32_t,int> occ;  // global slot -> times occupied (must stay 0/1)
   int fails=0;
   auto mark=[&](uint32_t s,uint32_t n,int d){ for(uint32_t i=0;i<n;++i){ occ[s+i]+=d; if(occ[s+i]>1){if(++fails<=5)std::printf(" OVERLAP slot %u\n",s+i);} if(occ[s+i]<0){if(++fails<=5)std::printf(" UNDERFREE %u\n",s+i);} } };
   for(int i=0;i<2000000;++i){
      bool doAlloc = live.empty() || (r()%3);
      if(doAlloc){
         uint32_t bytes = 1 + r()%2048;
         uint32_t s = ses.alloc(bytes);
         uint32_t n = var_allocator<32>::grains_for(bytes);
         mark(s,n,+1);
         if(va.ref(s)!=1){if(++fails<=5)std::printf(" bad ref\n");}
         std::memset(va.get(s),0xCD,bytes);  // touch
         live.push_back({s,bytes});
      } else {
         size_t k=r()%live.size(); Obj o=live[k]; live[k]=live.back(); live.pop_back();
         uint32_t n=var_allocator<32>::grains_for(o.bytes);
         mark(o.slot,n,-1);
         if(va.release_one(o.slot)) va.free(o.slot,o.bytes);
      }
   }
   for(Obj o:live){ uint32_t n=var_allocator<32>::grains_for(o.bytes); mark(o.slot,n,-1); if(va.release_one(o.slot)) va.free(o.slot,o.bytes); }
   uint64_t after=va.live_slots();
   if(after!=0){++fails; std::printf(" NOT drained: %llu\n",(unsigned long long)after);}
   std::printf("[var32] segs=%u slots/seg=%u max_obj=%uB fails=%d %s\n",
      va.num_segments(), var_allocator<32>::slots_per_segment, var_allocator<32>::max_object_bytes, fails, fails?"FAIL":"PASS");
   return fails?1:0;
}

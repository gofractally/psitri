// A/B: gcart (CB-movable leaves) vs rcart (direct-grain leaves), spec engine.
#include <gcart/spec_tree.hpp>
#include <rcart/spec_tree.hpp>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <filesystem>
#include <sys/resource.h>
using clk = std::chrono::steady_clock;
static double nsop(clk::time_point a, clk::time_point b, size_t n){ return double(std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count())/double(n); }
static long rss_mb(){ struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1048576; }

template<class Engine, class Tree>
void run(const char* name, const char* dir, size_t N, bool seq){
   std::filesystem::remove_all(dir);
   Engine e{std::filesystem::path(dir)};
   Tree t(e);
   std::mt19937_64 r(12345);
   std::vector<uint64_t> ks(N);
   // seq=true → dense sequential keys (256-way FULL nodes on the low bytes,
   // the full-node-prefetch path); false → random (sparse, mostly setlists).
   for(size_t i=0;i<N;i++) ks[i] = seq ? __builtin_bswap64(uint64_t(i)) : r();
   char kb[16], vb[100]; for(int i=0;i<100;i++) vb[i]='v';
   auto a=clk::now();
   for(size_t i=0;i<N;i++){ std::memcpy(kb,&ks[i],8); std::memcpy(kb+8,&ks[i],8);
      t.upsert(std::string_view(kb,16), std::string_view(vb,100)); }
   auto b=clk::now();
   std::string out; size_t hit=0;
   auto c=clk::now();
   for(size_t i=0;i<N;i++){ std::memcpy(kb,&ks[i],8); std::memcpy(kb+8,&ks[i],8);
      if(t.find(std::string_view(kb,16), out)) hit++; }
   auto d=clk::now();
   std::printf("%-7s %-4s N=%zu  insert=%6.1f ns/op  find=%6.1f ns/op  hits=%zu  rss=%ldMB\n",
      name, seq?"seq":"rand", N, nsop(a,b,N), nsop(c,d,N), hit, rss_mb());
}
int main(int argc, char**argv){
   size_t N = argc>1? std::strtoull(argv[1],0,10):1000000;
   run<gcart::engine, gcart::tree>("gcart", "/tmp/sb_gcart", N, false);
   run<rcart::engine, rcart::tree>("rcart", "/tmp/sb_rcart", N, false);
   return 0;
}

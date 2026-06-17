#include <gcart/tree.hpp>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <sys/resource.h>
using Clock = std::chrono::steady_clock;
static double nsop(Clock::time_point a, Clock::time_point b, size_t n)
{ return double(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()) / double(n); }
static long rss_mb() { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / 1048576; }

int main(int argc, char** argv)
{
   const size_t N    = argc > 1 ? std::strtoull(argv[1], 0, 10) : 1000000;
   const int    mode = argc > 2 ? atoi(argv[2]) : 1;  // 0=dict-ish,1=random,2=seq
   namespace fs = std::filesystem;
   const fs::path dir = fs::temp_directory_path() / "gcart_bench";
   fs::remove_all(dir);

   std::vector<std::string> keys;
   keys.reserve(N);
   std::mt19937 r(99);
   for (size_t i = 0; i < N; ++i)
   {
      char b[40];
      if (mode == 0) { static const char* p[]={"user:","user:profile:","acct:bal:","session:tok:"}; std::string s=p[r()%4]; std::snprintf(b,sizeof b,"%u",unsigned(r())); keys.push_back(s+b); }
      else if (mode == 2) { std::snprintf(b,sizeof b,"key%012zu",i); keys.push_back(b); }
      else { int n=4+int(r()%20); std::string s; for(int j=0;j<n;++j) s.push_back(char('a'+r()%26)); keys.push_back(std::move(s)); }
   }

   sal2::arena a(sal2::arena_config{.dir = dir});
   gcart::legacy::tree t(a);
   const std::string val = "value-payload-100b........................................................................xx";

   auto t0 = Clock::now();
   for (auto& k : keys) t.upsert(k, val);
   auto t1 = Clock::now();
   // get hits (shuffled order to defeat any locality from insert order)
   std::vector<uint32_t> idx(N); for (size_t i=0;i<N;++i) idx[i]=uint32_t(i);
   std::shuffle(idx.begin(), idx.end(), r);
   volatile uint64_t sink = 0;
   auto t2 = Clock::now();
   for (uint32_t i : idx) { const gcart::legacy::leaf* lf = t.find(keys[i]); sink += lf ? lf->vlen : 0; }
   auto t3 = Clock::now();
   // get misses
   size_t misses = 0;
   auto t4 = Clock::now();
   for (size_t i = 0; i < N; ++i) { std::string m = keys[i] + "\x01zz"; if (!t.find(m)) ++misses; }
   auto t5 = Clock::now();

   const uint64_t live = a.live_bytes();
   std::printf("gcart mode=%d N=%zu  insert=%.1f ns  hit=%.1f ns  miss=%.1f ns  live=%.1f MB  bytes/key=%.1f  rss=%ld MB  (sink=%llu miss=%zu)\n",
               mode, N, nsop(t0,t1,N), nsop(t2,t3,N), nsop(t4,t5,N),
               live/1048576.0, double(live)/N, rss_mb(), (unsigned long long)sink, misses);
   return 0;
}

// Same-machine, same-keys read-perf + memory comparison: start_ver vs
// sal_art_ref (the engines gcart must beat / approach), pure get(hit).
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <sys/resource.h>

#include <start_ver/kv.hpp>
#include "heart_kv.hpp"  // sal_art_ref::kv::database

using Clock = std::chrono::steady_clock;
static double nsop(Clock::time_point a, Clock::time_point b, size_t n)
{ return double(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()) / double(n); }
static long rss_mb() { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / 1048576; }

template <class DB>
void run(const char* name, const std::vector<std::string>& keys, const std::string& val,
         const std::filesystem::path& dir)
{
   std::filesystem::remove_all(dir);
   auto db  = DB::create(dir);
   auto ws  = db->start_write_session();
   auto t0  = Clock::now();
   const size_t B = 4096;
   for (size_t i = 0; i < keys.size(); i += B)
   {
      auto tx = ws->start_transaction(0);
      for (size_t j = i; j < std::min(keys.size(), i + B); ++j)
         tx.upsert(keys[j], val);
      tx.commit();
   }
   auto t1 = Clock::now();
   auto rs = db->start_read_session();
   auto cur = rs->create_cursor(0);
   std::vector<uint32_t> idx(keys.size()); for (size_t i=0;i<keys.size();++i) idx[i]=uint32_t(i);
   std::mt19937 r(7); std::shuffle(idx.begin(), idx.end(), r);
   volatile uint64_t sink = 0;
   auto t2 = Clock::now();
   for (uint32_t i : idx) cur.get(keys[i], [&](std::string_view v){ sink += v.size(); });
   auto t3 = Clock::now();
   std::printf("%-12s insert=%.1f ns  hit=%.1f ns  rss=%ld MB  (sink=%llu)\n", name,
               nsop(t0,t1,keys.size()), nsop(t2,t3,keys.size()), rss_mb(), (unsigned long long)sink);
}

int main(int argc, char** argv)
{
   const size_t N    = argc > 1 ? std::strtoull(argv[1], 0, 10) : 1000000;
   const char*  only = argc > 2 ? argv[2] : nullptr;
   std::vector<std::string> keys; keys.reserve(N);
   std::mt19937 r(99);
   for (size_t i = 0; i < N; ++i)
   { int n=4+int(r()%20); std::string s; for(int j=0;j<n;++j) s.push_back(char('a'+r()%26)); keys.push_back(std::move(s)); }
   const std::string val = "value-payload-100b........................................................................xx";
   namespace fs = std::filesystem;
   if (!only || std::string(only)=="start_ver")
      run<start_ver::kv::database>("start_ver", keys, val, fs::temp_directory_path()/"cmp_sv");
   if (!only || std::string(only)=="sal_art_ref")
      run<sal_art_ref::kv::database>("sal_art_ref", keys, val, fs::temp_directory_path()/"cmp_sar");
   return 0;
}

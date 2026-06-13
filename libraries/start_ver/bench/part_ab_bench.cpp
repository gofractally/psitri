// part vs hkv vs libart — the two read/write gates (SPEC perf goals):
//   * dictionary-style read-only: part must MATCH hkv (identical layout+descent;
//     libart loads a header per hop even for full nodes)
//   * uniform u64 (hash24-style) reads + unique writes: near libart
//
// Methodology per the house perf hygiene: contestants run BACK-TO-BACK inside
// each rep (thermal drift cancels in the ratio), one process at a time, median
// of reps reported.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <start_ver/hkv/tree.hpp>
#include <start_ver/tree_alloc.hpp>

extern "C"
{
#include <art.h>
}

using Clock = std::chrono::steady_clock;

static double now_ns()
{
   return double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     Clock::now().time_since_epoch())
                     .count());
}

struct workload
{
   const char*              name;
   std::vector<std::string> keys;     // insert order
   std::vector<uint32_t>    qorder;   // query order (indexes into keys)
   std::vector<std::string> misses;   // absent keys
};

static workload make_dict()
{
   workload w;
   w.name = "dict";
   std::ifstream f("/usr/share/dict/words");
   std::string   line;
   while (std::getline(f, line))
      if (!line.empty())
         w.keys.push_back(line);
   std::mt19937_64 rng(7);
   w.qorder.resize(w.keys.size());
   for (uint32_t i = 0; i < w.qorder.size(); ++i)
      w.qorder[i] = i;
   std::shuffle(w.qorder.begin(), w.qorder.end(), rng);
   for (size_t i = 0; i < w.keys.size(); ++i)
      w.misses.push_back(w.keys[i] + "~");
   return w;
}

static workload make_u64(size_t n)
{
   workload w;
   w.name = "u64";
   std::mt19937_64 rng(9);
   w.keys.reserve(n);
   for (size_t i = 0; i < n; ++i)
   {
      uint64_t r = rng();
      w.keys.emplace_back(reinterpret_cast<const char*>(&r), 8);
   }
   w.qorder.resize(n);
   for (uint32_t i = 0; i < n; ++i)
      w.qorder[i] = i;
   std::shuffle(w.qorder.begin(), w.qorder.end(), rng);
   for (size_t i = 0; i < n; ++i)
   {
      uint64_t r = rng() | (1ull << 63);
      w.misses.emplace_back(reinterpret_cast<const char*>(&r), 8);
   }
   return w;
}

struct timings
{
   double insert_ns = 0, hit_ns = 0, miss_ns = 0;
};

template <class TreeFactory>
static timings run_tree(const workload& w, TreeFactory&& make)
{
   auto    t = make();
   timings r;
   uint64_t sink = 0;
   {
      const double t0 = now_ns();
      uint64_t     v  = 1;
      for (auto& k : w.keys)
         t.insert(std::string_view(k), v++);
      r.insert_ns = (now_ns() - t0) / double(w.keys.size());
   }
   {
      const double t0 = now_ns();
      uint64_t     out;
      for (uint32_t i : w.qorder)
         sink += t.find(std::string_view(w.keys[i]), out) ? out : 0;
      r.hit_ns = (now_ns() - t0) / double(w.qorder.size());
   }
   {
      const double t0 = now_ns();
      for (auto& k : w.misses)
         sink += t.contains(std::string_view(k));
      r.miss_ns = (now_ns() - t0) / double(w.misses.size());
   }
   if (sink == 0xdeadbeef)
      std::printf("impossible\n");
   return r;
}

static timings run_libart(const workload& w)
{
   art_tree t;
   art_tree_init(&t);
   timings  r;
   uint64_t sink = 0;
   {
      const double t0 = now_ns();
      uintptr_t    v  = 1;
      for (auto& k : w.keys)
         art_insert(&t, reinterpret_cast<const unsigned char*>(k.data()), int(k.size()),
                    reinterpret_cast<void*>(v++));
      r.insert_ns = (now_ns() - t0) / double(w.keys.size());
   }
   {
      const double t0 = now_ns();
      for (uint32_t i : w.qorder)
         sink += reinterpret_cast<uintptr_t>(
             art_search(&t, reinterpret_cast<const unsigned char*>(w.keys[i].data()),
                        int(w.keys[i].size())));
      r.hit_ns = (now_ns() - t0) / double(w.qorder.size());
   }
   {
      const double t0 = now_ns();
      for (auto& k : w.misses)
         sink += art_search(&t, reinterpret_cast<const unsigned char*>(k.data()),
                            int(k.size())) != nullptr;
      r.miss_ns = (now_ns() - t0) / double(w.misses.size());
   }
   art_tree_destroy(&t);
   if (sink == 0xdeadbeef)
      std::printf("impossible\n");
   return r;
}

int main(int argc, char** argv)
{
   const size_t nu64 = argc > 1 ? std::stoull(argv[1]) : 2'000'000;
   const int    reps = argc > 2 ? std::stoi(argv[2]) : 5;

   for (auto& w : {make_dict(), make_u64(nu64)})
   {
      std::printf("== %s: %zu keys, %d reps (median), back-to-back per rep ==\n", w.name,
                  w.keys.size(), reps);
      std::vector<timings> part_r, hkv_r, art_r;
      for (int rep = 0; rep < reps; ++rep)
      {
         // part — arena-backed, volatile mode
         {
            auto dir = std::filesystem::temp_directory_path() /
                       ("part_ab_" + std::to_string(::getpid()) + "_" + std::to_string(rep));
            std::filesystem::remove_all(dir);
            start_ver::arena_config c;
            c.dir = dir;
            start_ver::arena a(c);
            auto        ses = a.start_session();
            auto        tx  = ses.start_transaction(0);
            part_r.push_back(run_tree(w, [&] {
               return start_ver::hkv::kv_tree<std::string_view, uint64_t, start_ver::hkv::mode::none,
                                   start_ver::part_alloc<uint64_t>>{
                   start_ver::part_alloc<uint64_t>(&ses)};
            }));
            std::filesystem::remove_all(dir);
         }
         // hkv gold standard — std::allocator, 6-byte direct handles
         hkv_r.push_back(
             run_tree(w, [] { return start_ver::hkv::kv_tree<std::string_view, uint64_t>{}; }));
         // libart
         art_r.push_back(run_libart(w));
      }
      auto med = [](std::vector<timings>& v, double timings::* f) {
         std::vector<double> x;
         for (auto& t : v)
            x.push_back(t.*f);
         std::sort(x.begin(), x.end());
         return x[x.size() / 2];
      };
      auto row = [&](const char* phase, double timings::* f) {
         const double p = med(part_r, f), h = med(hkv_r, f), l = med(art_r, f);
         std::printf("  %-8s part %7.1f ns  hkv %7.1f ns  libart %7.1f ns   part/hkv %5.2fx  part/libart %5.2fx\n",
                     phase, p, h, l, p / h, p / l);
      };
      row("insert", &timings::insert_ns);
      row("hit", &timings::hit_ns);
      row("miss", &timings::miss_ns);
   }
   return 0;
}

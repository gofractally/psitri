// Head-to-head: heart radix_engine (CRTP byte ART) vs libart, on two workloads:
//   - dict:  /usr/share/dict/words (variable-length, prefix-dense)
//   - rand:  N random u64 keys (8 bytes, fixed length)
// Times insert and random-order lookup (ns/op) for each. Keys are NUL-terminated for
// both engines so the trees are structurally identical (fair comparison; libart needs
// it anyway). Usage: art-vs-heart-bench [dict] [rand_n]
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

extern "C" {
#include "art.h"
}
#include <radix_engine.hpp>

using clk = std::chrono::steady_clock;
static double nsop(clk::time_point a, clk::time_point b, size_t n)
{
   return double(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()) / double(n);
}

namespace
{
   std::vector<std::string> load_dict(const char* p)
   {
      std::vector<std::string> w;
      std::ifstream            f(p);
      std::string              s;
      while (std::getline(f, s))
         if (!s.empty())
         {
            s.push_back('\0');  // NUL-terminate: no key a prefix of another (libart needs it)
            w.push_back(std::move(s));
         }
      return w;
   }

   // run both engines over a key set, return {heart_ins, heart_look, art_ins, art_look} ns/op
   struct Row
   {
      double hi, hl, ai, al;
      bool   ok;
   };
   Row run(const std::vector<std::string>& keys, const std::vector<uint32_t>& order, const char* tag)
   {
      Row  r{};
      auto dir = std::filesystem::temp_directory_path() / (std::string("avh_") + tag);

      // ---- heart radix_engine ----
      heart::radix::radix_tree t(dir);
      if (std::getenv("WIDE"))
         t.set_wide_stems(true);
      auto                     vv = [](const uint64_t& v)
      { return std::string_view(reinterpret_cast<const char*>(&v), 8); };
      auto t0 = clk::now();
      for (size_t i = 0; i < keys.size(); ++i)
         t.upsert(keys[i], vv(uint64_t(i)));
      auto     t1  = clk::now();
      uint64_t acc = 0;
      auto     l0  = clk::now();
      for (uint32_t idx : order)
      {
         std::string_view v;
         if (t.find(keys[idx], v))
         {
            uint64_t g = 0;
            std::memcpy(&g, v.data(), 8);
            acc += g;
         }
      }
      auto l1 = clk::now();
      r.hi    = nsop(t0, t1, keys.size());
      r.hl    = nsop(l0, l1, order.size());

      // ---- libart ----
      art_tree at;
      art_tree_init(&at);
      auto a0 = clk::now();
      for (size_t i = 0; i < keys.size(); ++i)
         art_insert(&at, (const unsigned char*)keys[i].data(), int(keys[i].size()), (void*)(i + 1));
      auto     a1   = clk::now();
      uint64_t acc2 = 0;
      auto     al0  = clk::now();
      for (uint32_t idx : order)
      {
         void* v = art_search(&at, (const unsigned char*)keys[idx].data(), int(keys[idx].size()));
         acc2 += (uintptr_t)v;
      }
      auto al1 = clk::now();
      r.ai     = nsop(a0, a1, keys.size());
      r.al     = nsop(al0, al1, order.size());
      r.ok     = (acc == uint64_t(keys.size()) * (keys.size() - 1) / 2) && (acc2 != 0);
      art_tree_destroy(&at);
      return r;
   }
}  // namespace

int main(int argc, char** argv)
{
   const char* dict   = (argc > 1) ? argv[1] : "/usr/share/dict/words";
   size_t      rand_n = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 1000000;

   std::printf("%-6s %-8s  %10s  %10s\n", "work", "engine", "insert ns", "lookup ns");
   std::printf("------------------------------------------------\n");

   // dict
   {
      auto                  keys = load_dict(dict);
      std::vector<uint32_t> order(keys.size());
      for (size_t i = 0; i < keys.size(); ++i)
         order[i] = uint32_t(i);
      std::shuffle(order.begin(), order.end(), std::mt19937(12345));
      Row r = run(keys, order, "dict");
      std::printf("dict   heart     %10.1f  %10.1f   [%zu keys]\n", r.hi, r.hl, keys.size());
      std::printf("dict   libart    %10.1f  %10.1f   [%s]\n", r.ai, r.al, r.ok ? "ok" : "MISMATCH");
   }
   // rand: N distinct 8-byte (u64) keys, NUL-padded length 9 so libart is happy
   {
      std::vector<std::string> keys(rand_n);
      for (size_t i = 0; i < rand_n; ++i)
      {
         uint64_t k = (uint64_t(i) * 0x9E3779B97F4A7C15ull) ^ (i >> 29);
         keys[i].assign(reinterpret_cast<const char*>(&k), 8);
         keys[i].push_back('\0');
      }
      std::vector<uint32_t> order(rand_n);
      for (size_t i = 0; i < rand_n; ++i)
         order[i] = uint32_t(i);
      std::shuffle(order.begin(), order.end(), std::mt19937(999));
      Row r = run(keys, order, "rand");
      std::printf("rand   heart     %10.1f  %10.1f   [%zu keys]\n", r.hi, r.hl, rand_n);
      std::printf("rand   libart    %10.1f  %10.1f   [%s]\n", r.ai, r.al, r.ok ? "ok" : "MISMATCH");
   }
   return 0;
}

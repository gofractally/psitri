// dwal delta-map head-to-head: art::art_map<btree_value> vs hkv on the EXACT
// dwal RW-layer op mix — bank-shaped fill (2 overwrites + 1 fresh log insert
// per tx, ~7% tombstone erases), then the sorted merge-drain, cycled like
// RW->RO swaps at max_rw_entries=100k. Both store btree_value (16B view +
// tag) with payload bytes in an identical side arena.
#include <art/art_map.hpp>
#include <psitri/dwal/btree_value.hpp>
#include "hkv/tree.hpp"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
using psitri::dwal::btree_value;
static uint64_t sm(uint64_t x){x+=0x9E3779B97F4A7C15ull;x=(x^(x>>30))*0xBF58476D1CE4E5B9ull;x=(x^(x>>27))*0x94D049BB133111EBull;return x^(x>>31);}
static double ns(Clock::time_point a, Clock::time_point b, uint64_t n)
{ return double(std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count())/double(n?n:1); }

struct payload_arena {  // identical value-byte storage for both contestants
   std::vector<std::unique_ptr<char[]>> blocks; size_t off = 1 << 20, cap = 1 << 20;
   std::string_view copy(std::string_view v) {
      if (off + v.size() > cap) { blocks.push_back(std::make_unique<char[]>(cap)); off = 0; }
      char* p = blocks.back().get() + off; memcpy(p, v.data(), v.size()); off += v.size();
      return {p, v.size()};
   }
   void clear() { blocks.clear(); off = cap; }
};
static std::string name_key(uint64_t i){ char b[40]; int n=snprintf(b,sizeof b,"acct_%llu_%08llx",(unsigned long long)(i%100000),(unsigned long long)(sm(i)&0xFFFFFFFF)); return std::string(b,n);}
static std::string log_key(uint64_t s){ char b[32]; int n=snprintf(b,sizeof b,"log_%016llx",(unsigned long long)s); return std::string(b,n);}

template <class Map, class Up, class Er, class Find, class Drain>
void run(const char* label, int cycles, Up up, Er er, Find fnd, Drain drain)
{
   payload_arena pa;
   double fill_ns = 0, drain_ns = 0; uint64_t fills = 0, drained = 0, hits = 0;
   uint64_t seq = 0;
   for (int c = 0; c < cycles; ++c)
   {
      Map m;
      auto t0 = Clock::now();
      uint64_t ops = 0;
      while (ops < 100000)  // one RW generation (max_rw_entries)
      {
         uint64_t t = seq++;
         char vb[64]; int vn = snprintf(vb, sizeof vb, "balance_%llu_xxxxxxxxxxxxxxxx", (unsigned long long)t);
         auto v = pa.copy(std::string_view(vb, vn));
         hits += fnd(m, name_key(sm(t*3+0)));        // miss probe (cold key)
         if (t) hits += fnd(m, log_key(t - 1));      // hit probe (last tx's log)
         up(m, name_key(sm(t*3+0)), v);          // overwrite-ish (hot keys)
         up(m, name_key(sm(t*3+1)), v);
         up(m, log_key(t), v);                   // fresh insert
         ops += 5;
         if (t % 14 == 0) { er(m, name_key(sm(t*3+0))); ++ops; }  // ~7% tombstone-as-erase
      }
      auto t1 = Clock::now();
      fill_ns += ns(t0, t1, ops) * ops; fills += ops;
      auto t2 = Clock::now();
      uint64_t n = drain(m);
      auto t3 = Clock::now();
      drain_ns += ns(t2, t3, n) * n; drained += n;
      pa.clear();
   }
   printf("  %-10s mixed %6.1f ns/op   drain %6.1f ns/entry   (%llu ops, %llu hits, %llu drained)\n",
          label, fill_ns / fills, drain_ns / drained,
          (unsigned long long)fills, (unsigned long long)hits, (unsigned long long)drained);
}
int main()
{
   const int CYC = 20;
   printf("dwal delta-map A/B — bank op mix, 100k-entry generations, %d cycles\n", CYC);
   run<art::art_map<btree_value>>("art_map", CYC,
      [](auto& m, const std::string& k, std::string_view v){ m.upsert(k, btree_value::make_data(v)); },
      [](auto& m, const std::string& k){ m.erase(k); },
      [](auto& m, const std::string& k){ auto it = m.find(k); return it != m.end() ? 1 : 0; },
      [](auto& m){ uint64_t n=0, b=0; for (auto it=m.begin(); it!=m.end(); ++it){ b+=it.key().size()+it.value().data.size(); ++n; } return n; });
   run<hkv::kv_tree<std::string_view, btree_value>>("hkv", CYC,
      [](auto& m, const std::string& k, std::string_view v){ m.insert_or_assign(k, btree_value::make_data(v)); },
      [](auto& m, const std::string& k){ m.erase(k); },
      [](auto& m, const std::string& k){ btree_value out; return m.find(k, out) ? 1 : 0; },
      [](auto& m){ uint64_t n=0, b=0; for (auto it=m.begin(); it!=m.end(); ++it){ b+=it.key().size()+it.value().data.size(); ++n; } return n; });
   return 0;
}

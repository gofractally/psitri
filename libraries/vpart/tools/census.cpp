#include <vpart/kv.hpp>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <vector>
using engine = vpart::kv::engine_t;
using K      = vhkv::detail::K;
static const char* kname(K k){
  switch(k){ case K::value_ptr: return "leaf"; case K::value_inline: return "inline";
    case K::prefix_node: return "prefix"; case K::setlist_u8: return "setlist8";
    case K::setlist_u16: return "setlist16"; case K::c2: return "c2"; case K::c4: return "c4";
    case K::c8: return "c8"; case K::node_full: return "full"; case K::full_x: return "full_x";
    case K::cx: return "cx"; case K::bucket: return "bucket"; case K::cbucket: return "cbucket";
    default: return "?"; } }
struct agg { uint64_t n=0, bytes=0, nbr=0, lu=0, la=0; };
static void run(const char* label, std::vector<std::pair<std::string,std::string>>& kv){
  auto dir=std::filesystem::temp_directory_path()/"census_db"; std::filesystem::remove_all(dir);
  auto db=vpart::kv::database::create(dir); auto ses=db->start_write_session();
  { auto tx=ses->start_transaction(0); for(auto&[k,v]:kv) tx.upsert(k,v); tx.commit(); }
  engine view{vpart::part_alloc<vpart::kv::pval>(&db->store())};
  view.adopt_root(db->live_root_dbg());
  std::map<K,agg> by; uint64_t fullhist[6]={0,0,0,0,0,0}; uint64_t full_lu=0, full_la=0;
  uint64_t user=0; for(auto&[k,v]:kv) user+=k.size()+v.size();
  view.census([&](engine::census_row r){
    auto& a=by[r.kind]; ++a.n; a.bytes+=r.bytes; a.nbr+=r.nbranch; a.lu+=r.lines_used; a.la+=r.lines_alloc;
    if(r.kind==K::node_full||r.kind==K::full_x){
      int b = r.nbranch<=12?0: r.nbranch<=31?1: r.nbranch<=62?2: r.nbranch<=124?3: r.nbranch<=248?4:5;
      ++fullhist[b]; full_lu+=r.lines_used; full_la+=r.lines_alloc; } });
  view.adopt_root(vpart::kv::root_ptr::null());
  printf("\n== %s: %zu keys, user bytes=%.1fMB ==\n", label, kv.size(), user/1048576.0);
  printf("%-10s %10s %12s %10s %8s %8s %8s\n","kind","count","bytes","MB","avg nbr","lines u","lines a");
  uint64_t tot=0;
  for(auto&[k,a]:by){ tot+=a.bytes;
    printf("%-10s %10llu %12llu %10.1f %8.1f %8llu %8llu\n", kname(k),
      (unsigned long long)a.n,(unsigned long long)a.bytes,a.bytes/1048576.0,
      a.n?double(a.nbr)/a.n:0.0,(unsigned long long)a.lu,(unsigned long long)a.la); }
  printf("TOTAL %.1fMB  = %.1fx user bytes\n", tot/1048576.0, double(tot)/double(user));
  printf("full/full_x nbranch histogram: <=12:%llu 13-31:%llu 32-62:%llu 63-124:%llu 125-248:%llu 249+:%llu  (lines used %llu / alloc %llu)\n",
    (unsigned long long)fullhist[0],(unsigned long long)fullhist[1],(unsigned long long)fullhist[2],
    (unsigned long long)fullhist[3],(unsigned long long)fullhist[4],(unsigned long long)fullhist[5],
    (unsigned long long)full_lu,(unsigned long long)full_la);
}
static uint64_t be64(uint64_t v){ return __builtin_bswap64(v); }
int main(){
  // dataset 1: dictionary
  { std::vector<std::pair<std::string,std::string>> kv; std::ifstream d("/usr/share/dict/words"); std::string w;
    while(std::getline(d,w)) if(!w.empty()) kv.push_back({w,w});
    run("dict (value=word)", kv); }
  // dataset 2: bank shape — 1M account names + 2M sequential logs
  { std::vector<std::pair<std::string,std::string>> kv; std::ifstream d("/usr/share/dict/words"); std::string w;
    std::vector<std::string> names;
    while(std::getline(d,w)&&names.size()<1000000) if(!w.empty()) names.push_back(w);
    uint64_t counter=0;
    while(names.size()<1000000){ int t=counter%3; uint64_t idx=counter/3+1;
      if(t==0){ uint64_t b=be64(idx); names.push_back(std::string((char*)&b,8)); }
      else if(t==1){ names.push_back(std::string((char*)&idx,8)); }
      else { char buf[16]; snprintf(buf,sizeof buf,"%08lu",(unsigned long)idx); names.push_back(buf); }
      ++counter; }
    std::mt19937_64 rng(12345); std::shuffle(names.begin(),names.end(),rng);
    std::string bal("\x40\x42\x0f\x00\x00\x00\x00\x00",8);
    for(auto&n:names) kv.push_back({n,bal});
    std::string lp("\x00TX",3);
    for(uint64_t s2=0;s2<2000000;++s2){ uint64_t be=be64(s2);
      kv.push_back({lp+std::string((char*)&be,8), std::string(34,'L')}); }
    run("bank shape (1M accts + 2M logs)", kv); }
  return 0;
}

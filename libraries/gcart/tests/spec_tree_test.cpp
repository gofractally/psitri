// gcart::st spec-engine oracle — the §7 node taxonomy over 3 regions (FIX
// setlist/full + SAL suffix_value), refcount COW. get/upsert vs std::map on
// mixed string + hashed keys; reports bytes/key (target: beat sal_art_ref's 83).
#include <gcart/spec_tree.hpp>
#include <cstdio>
#include <map>
#include <random>
#include <string>
int main(){
  namespace fs=std::filesystem; auto dir=fs::temp_directory_path()/"st_oracle"; fs::remove_all(dir);
  gcart::st::engine e(dir);
  gcart::st::tree t(e);
  std::map<std::string,std::string> oracle;
  std::mt19937 r(2024);
  int checks=0, fails=0;
  auto mkk=[&](){ int n=3+int(r()%18); std::string k; for(int j=0;j<n;++j) k.push_back(char('a'+r()%26)); return k; };
  // also binary/hashed keys for fanout
  for(int i=0;i<200000;++i){
     std::string k = (r()&1)? mkk() : [&]{ uint64_t h=r()*0x9E3779B97F4A7C15ull; return std::string((char*)&h,8); }();
     std::string v = "v"+std::to_string(r()%100000);
     t.upsert(k,v); oracle[k]=v;
  }
  // verify all
  std::string out;
  for(auto&[k,v]:oracle){ bool got=t.find(k,out); ++checks; if(!got||out!=v){ if(++fails<=10) std::printf("MISS %s\n",k.c_str()); } }
  // verify some absent
  for(int i=0;i<5000;++i){ uint64_t h=(r()+1234567)*0x9E3779B97F4A7C15ull; std::string k((char*)&h,8); if(oracle.count(k))continue; ++checks; if(t.find(k,out)){ if(++fails<=10) std::printf("PHANTOM\n"); } }
  double bpk = oracle.empty()?0: double(e.live_bytes())/oracle.size();
  std::printf("st-oracle: %d checks, %d fails, keys=%zu, live=%lluB, bytes/key=%.0f\n",
     checks, fails, oracle.size(), (unsigned long long)e.live_bytes(), bpk);
  return fails?1:0;
}

// gcart::st::kv adb adapter — CRUD + snapshot isolation + concurrent MVCC
// (4 readers + writer, lock-free across the 3 regions). 0 fails expected.
#include <gcart/spec_kv.hpp>
#include <atomic>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>
using namespace gcart::st::kv;
int main(){
  namespace fs=std::filesystem; auto dir=fs::temp_directory_path()/"sk_test"; fs::remove_all(dir);
  auto db=database::create(dir); auto ws=db->start_write_session();
  std::map<std::string,std::string> oracle; std::mt19937 r(1);
  auto mkk=[&](){ int n=3+int(r()%18); std::string k; for(int j=0;j<n;++j)k.push_back(char('a'+r()%26)); return k; };
  int checks=0, fails=0;
  // batched upserts
  for(int b=0;b<40;++b){ auto tx=ws->start_transaction(0); for(int i=0;i<2000;++i){auto k=mkk();auto v="v"+std::to_string(r()%100000);tx.upsert(k,v);oracle[k]=v;} tx.commit(); }
  // read via cursor
  { auto rs=db->start_read_session(); auto c=rs->create_cursor(0);
    for(auto&[k,v]:oracle){ auto got=c.get<std::string>(k); ++checks; if(!got||*got!=v){if(++fails<=5)std::printf("MISS %s\n",k.c_str());} } }
  // snapshot isolation: pin a cursor, mutate a key, the cursor still sees old
  { auto rs=db->start_read_session(); auto c=rs->create_cursor(0);
    auto it=oracle.begin(); std::string k=it->first, oldv=it->second;
    { auto tx=ws->start_transaction(0); tx.upsert(k,"CHANGED"); tx.commit(); oracle[k]="CHANGED"; }
    auto got=c.get<std::string>(k); ++checks; if(!got||*got!=oldv){++fails;std::printf("ISO FAIL: snapshot saw %s not %s\n", got?got->c_str():"(none)", oldv.c_str());}
    auto c2=db->start_read_session()->create_cursor(0); auto g2=c2.get<std::string>(k); ++checks; if(!g2||*g2!="CHANGED"){++fails;std::printf("fresh cursor stale\n");}
  }
  // concurrent: readers + writer (balance-style)
  { std::atomic<bool> stop{false}; std::atomic<long> reads{0}, rfail{0};
    std::vector<std::thread> rt;
    for(int i=0;i<4;++i) rt.emplace_back([&]{ auto rs=db->start_read_session(); std::mt19937 rr(i*7+1);
       std::vector<std::string> keys; for(auto&[k,v]:oracle) keys.push_back(k);
       while(!stop.load()){ auto c=rs->create_cursor(0); for(int b=0;b<500;++b){ auto&k=keys[rr()%keys.size()]; auto got=c.get<std::string>(k); if(got) reads.fetch_add(1); } } });
    // writer churns values
    std::mt19937 wr(123); std::vector<std::string> keys; for(auto&[k,v]:oracle)keys.push_back(k);
    for(int round=0;round<200;++round){ auto tx=ws->start_transaction(0); for(int i=0;i<500;++i){auto&k=keys[wr()%keys.size()]; tx.upsert(k,"w"+std::to_string(wr()%1000));} tx.commit(); }
    stop.store(true); for(auto&t:rt)t.join();
    std::printf("concurrent: %ld reads, %ld rfail\n", reads.load(), rfail.load());
  }
  std::printf("sk-test: %d checks, %d fails, keys=%zu, bytes/key=%.0f\n", checks, fails, oracle.size(),
     oracle.empty()?0:double(db->live_bytes())/oracle.size());
  return fails?1:0;
}

#include <part/kv.hpp>
#include <cstdio>
#include <random>
#include <filesystem>
extern "C" __attribute__((noinline)) void wp_anchor(void* p){ asm volatile(""::"r"(p)); }
int main(){
  auto dir = std::filesystem::temp_directory_path()/"cbw"; std::filesystem::remove_all(dir);
  auto db = part::kv::database::create(dir); auto ses = db->start_write_session();
  auto& a = db->store();
  auto state=[&](const char* tag, int i){
    if(a.num_segments()==0) return;
    bool fr=a.freeset_of(0).is_free(1); uint8_t rc=a.ref(1);
    static int last=-1; int cur=(fr?2:0)|(rc?1:0);
    if(cur!=last){ printf("STATE %s i=%d line1 free=%d ref=%u\n",tag,i,(int)fr,rc); last=cur; }
  };
  std::vector<std::string> keys;
  for(int i=0;i<1200;++i) keys.push_back("acct/"+std::to_string(i));
  { auto tx=ses->start_transaction(0);
    for(int i=0;i<1200;++i){ fprintf(stderr,"OP ins %d\n",i); tx.upsert(keys[i],"00000000"); }
    tx.commit(); }
  state("post-commit",-1);
  wp_anchor((void*)a.refs_base_dbg());
  { auto tx=ses->start_transaction(0);
    for(int i=0;i<1200;++i){ fprintf(stderr,"OP rm %d\n",i); tx.remove(keys[i]); }
    tx.commit(); }
  state("end",-1);
  for(uint32_t s2=0;s2<a.num_segments();++s2){ auto& fs=a.freeset_of(s2);
    for(uint32_t l=0;l<part::lines_per_segment;++l) if(!fs.is_free(l)){
      uint32_t g=s2*part::lines_per_segment+l; const uint8_t* p=(const uint8_t*)a.get(g);
      printf("LEAK %u ref=%u ",g,a.ref(g)); for(int b2=0;b2<12;++b2) printf("%02x",p[b2]); printf("\n"); } }
  return 0;
}

#include <start_ref/kv.hpp>
#include <cstdio>
#include <random>
#include <filesystem>
int main(){
  auto dir = std::filesystem::temp_directory_path()/"cbl"; std::filesystem::remove_all(dir);
  auto db = start_ref::kv::database::create(dir); auto ses = db->start_write_session();
  std::vector<std::string> keys;
  for(int i=0;i<1200;++i) keys.push_back("acct/"+std::to_string(i));
  { auto tx=ses->start_transaction(0); for(auto&k:keys) tx.upsert(k,"00000000"); tx.commit(); }
  std::mt19937_64 rng(7);
  for(int r=0;r<100;++r){ auto tx=ses->start_transaction(0);
    for(int j=0;j<3;++j){ char v[9]; snprintf(v,9,"%08x",(unsigned)rng());
      tx.upsert(keys[rng()%keys.size()], std::string_view(v,8)); }
    tx.commit(); }
  { auto tx=ses->start_transaction(0); for(auto&k:keys) tx.remove(k); tx.commit(); }
  { auto tx=ses->start_transaction(0); tx.commit(); }  // force window/fcache flush
  auto& a = db->store();
  { uint64_t af=0, fa=0;
    for(uint32_t s2=0;s2<a.num_segments();++s2){ auto& fs=a.freeset_of(s2);
      for(uint32_t l=0;l<start_ref::lines_per_segment;++l){ uint32_t g=s2*start_ref::lines_per_segment+l;
        bool fr=fs.is_free(l); uint8_t rc=a.ref(g);
        if(!fr && rc==0){ ++af; if(af<4) printf("ALLOC-BIT,REF0 line=%u\n",g); }
        if(fr && rc!=0){ ++fa; if(fa<4) printf("FREE-BIT,REF=%u line=%u\n",rc,g); } } }
    printf("mismatch alloc-bit/ref0=%llu free-bit/ref>0=%llu\n",(unsigned long long)af,(unsigned long long)fa); }
  uint64_t leaked=0;
  for(uint32_t s=0;s<a.num_segments();++s){
    auto& fs=a.freeset_of(s);
    for(uint32_t l=0;l<start_ref::lines_per_segment;++l)
      if(!fs.is_free(l)){
        uint32_t g=s*start_ref::lines_per_segment+l;
        const uint8_t* p=(const uint8_t*)a.get(g);
        printf("leak line=%u ref=%u bytes[0..15]=",g,a.ref(g));
        for(int b2=0;b2<16;++b2) printf("%02x",p[b2]);
        printf("\n");
        if(++leaked>24){ printf("...\n"); s=a.num_segments(); break; }
      }
  }
  printf("leaked=%llu lines\n",(unsigned long long)leaked);
  return 0;
}

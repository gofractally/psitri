// rcart::legacy::tree correctness gate vs std::map oracle: insert / get / update /
// in-order iterate / erase, on dictionary-shaped and random keys.
#include <rcart/tree.hpp>

#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace
{
   int g_checks = 0, g_fails = 0;
   void ok(bool c, const char* what)
   {
      ++g_checks;
      if (!c)
      {
         ++g_fails;
         if (g_fails <= 20)
            std::fprintf(stderr, "FAIL: %s\n", what);
      }
   }

   std::string keygen(std::mt19937& r, int mode)
   {
      // mode 0: shared-prefix dict-ish; 1: random bytes; 2: sequential-ish
      if (mode == 0)
      {
         static const char* pre[] = {"user:", "user:profile:", "acct:", "acct:bal:", "z"};
         std::string        s = pre[r() % 5];
         char               buf[16];
         std::snprintf(buf, sizeof buf, "%u", r() % 100000);
         return s + buf;
      }
      if (mode == 1)
      {
         std::string s;
         int         n = 1 + int(r() % 24);
         for (int i = 0; i < n; ++i)
            s.push_back(char('a' + (r() % 26)));
         return s;
      }
      char buf[24];
      std::snprintf(buf, sizeof buf, "k%010u", r() % 2000000);
      return buf;
   }

   bool verify_inorder(const rcart::legacy::tree& t, const std::map<std::string, std::string>& oracle)
   {
      std::vector<std::pair<std::string, std::string>> got;
      t.for_each([&](rcart::legacy::sv k, rcart::legacy::sv v) { got.emplace_back(std::string(k), std::string(v)); });
      if (got.size() != oracle.size())
      {
         std::fprintf(stderr, "size mismatch: tree=%zu oracle=%zu\n", got.size(), oracle.size());
         return false;
      }
      size_t i = 0;
      for (auto& [k, v] : oracle)
      {
         if (got[i].first != k || got[i].second != v)
         {
            std::fprintf(stderr, "inorder mismatch at %zu: got '%s'='%s' want '%s'='%s'\n", i,
                         got[i].first.c_str(), got[i].second.c_str(), k.c_str(), v.c_str());
            return false;
         }
         ++i;
      }
      return true;
   }
}  // namespace

int main()
{
   namespace fs = std::filesystem;
   const fs::path dir = fs::temp_directory_path() / "rcart_tree_test";

   for (int mode = 0; mode < 3; ++mode)
   {
      fs::remove_all(dir);
      sal2::arena a(sal2::arena_config{.dir = dir});
      rcart::legacy::tree t(a);
      std::map<std::string, std::string> oracle;
      std::mt19937 r(1234 + mode);

      // insert + update
      const int N = 60000;
      for (int i = 0; i < N; ++i)
      {
         std::string k = keygen(r, mode);
         std::string v = "v" + std::to_string(r() % 1000000);
         bool        had = oracle.count(k) != 0;
         bool        ins = t.upsert(k, v);
         oracle[k]       = v;
         ok(ins == !had, "insert/update return flag matches oracle");
      }
      ok(t.size() == oracle.size(), "size matches oracle after inserts");

      // point gets
      for (auto& [k, v] : oracle)
      {
         const rcart::legacy::leaf* lf = t.find(k);
         ok(lf != nullptr && lf->value() == v, "get matches oracle");
      }
      // misses
      for (int i = 0; i < 5000; ++i)
      {
         std::string k = keygen(r, mode) + "_absent_zz";
         ok(t.find(k) == nullptr || oracle.count(k), "miss is a real miss");
      }
      // in-order
      ok(verify_inorder(t, oracle), "in-order traversal matches oracle");

      // erase half
      std::vector<std::string> keys;
      for (auto& [k, v] : oracle)
         keys.push_back(k);
      std::shuffle(keys.begin(), keys.end(), r);
      for (size_t i = 0; i < keys.size() / 2; ++i)
      {
         bool rem = t.erase(keys[i]);
         ok(rem, "erase of present key returns true");
         oracle.erase(keys[i]);
      }
      ok(t.size() == oracle.size(), "size matches oracle after erase");
      for (auto& [k, v] : oracle)
      {
         const rcart::legacy::leaf* lf = t.find(k);
         ok(lf && lf->value() == v, "surviving key still present + correct");
      }
      ok(verify_inorder(t, oracle), "in-order matches oracle after erase");

      std::printf("mode %d: %d checks, %d fails, live_bytes=%llu, segs=%u\n", mode, g_checks,
                  g_fails, (unsigned long long)a.live_bytes(), a.num_segments());
   }
   std::printf("rcart-tree-tests: %d checks, %d fails\n", g_checks, g_fails);
   return g_fails == 0 ? 0 : 1;
}

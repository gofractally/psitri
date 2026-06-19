// rcart spec-engine oracle — the §7 node taxonomy over 3 regions (FIX
// setlist/full + SAL suffix_value), refcount COW. get/upsert vs std::map on
// mixed string + hashed keys; reports bytes/key (target: beat sal_art_ref's 83).
#include <algorithm>
#include <cstdio>
#include <rcart/tree.hpp>
#include <map>
#include <random>
#include <string>
#include <vector>
int main()
{
   namespace fs = std::filesystem;
   auto dir     = fs::temp_directory_path() / "st_oracle";
   fs::remove_all(dir);
   rcart::engine                  e(dir);
   rcart::tree                    t(e);
   std::map<std::string, std::string> oracle;
   std::mt19937                       r(2024);
   int                                checks = 0, fails = 0;
   auto                               mkk = [&]()
   {
      int         n = 3 + int(r() % 18);
      std::string k;
      for (int j = 0; j < n; ++j)
         k.push_back(char('a' + r() % 26));
      return k;
   };
   // also binary/hashed keys for fanout
   for (int i = 0; i < 200000; ++i)
   {
      std::string k = (r() & 1) ? mkk() : [&]
      {
         uint64_t h = r() * 0x9E3779B97F4A7C15ull;
         return std::string((char*)&h, 8);
      }();
      std::string v = "v" + std::to_string(r() % 100000);
      t.upsert(k, v);
      oracle[k] = v;
   }
   // verify all
   std::string out;
   for (auto& [k, v] : oracle)
   {
      bool got = t.find(k, out);
      ++checks;
      if (!got || out != v)
      {
         if (++fails <= 10)
            std::printf("MISS %s\n", k.c_str());
      }
   }
   // verify some absent
   for (int i = 0; i < 5000; ++i)
   {
      uint64_t    h = (r() + 1234567) * 0x9E3779B97F4A7C15ull;
      std::string k((char*)&h, 8);
      if (oracle.count(k))
         continue;
      ++checks;
      if (t.find(k, out))
      {
         if (++fails <= 10)
            std::printf("PHANTOM\n");
      }
   }

   // ── density vs theoretical minimum (Dan): min B/key = key.len + val.len +
   //    sizeof(key.len) + sizeof(val.len). rcart's length fields are slen:u8 +
   //    vlen:u16 = 3 bytes. The min is the flat-store baseline (no structural
   //    overhead, no prefix amortization); a radix trie can beat it on shared
   //    prefixes and lose to it on random keys + pointers/inner nodes. ─────────
   {
      uint64_t kb = 0, vb = 0;
      for (auto& [k, v] : oracle)
      {
         kb += k.size();
         vb += v.size();
      }
      const double minbpk = double(kb + vb + oracle.size() * 3) / double(oracle.size());
      const double actbpk = double(e.live_bytes()) / double(oracle.size());
      std::printf(
          "density@load: keys=%zu avgkey=%.1f avgval=%.1f | actual=%.1f B/key  min=%.1f (k+v+3)  "
          "ratio=%.2fx  overhead=%.1f B/key\n",
          oracle.size(), double(kb) / oracle.size(), double(vb) / oracle.size(), actbpk, minbpk,
          actbpk / minbpk, actbpk - minbpk);
   }

   // ── large values (> SAL 2KB cap → BIG region, bt::big_suffix_value) ────────
   {
      auto bigval = [&](int i, size_t n)
      {
         std::string s(n, char('A' + i % 26));
         std::memcpy(s.data(), &i, std::min<size_t>(4, n));
         return s;
      };
      std::vector<std::string> bk;
      for (int i = 0; i < 500; ++i)
      {
         std::string k = "BIGKEY" + std::to_string(i);
         std::string v = bigval(i, 2100 + (r() % 7000));
         t.upsert(k, v);
         oracle[k] = v;
         bk.push_back(k);
      }
      for (auto& k : bk)
      {
         ++checks;
         if (!t.find(k, out) || out != oracle[k])
         {
            if (++fails <= 10)
               std::printf("BIG-MISS %s\n", k.c_str());
         }
      }
      // overwrite: big->big(diff size), big->small (SAL), small->big
      for (size_t i = 0; i < bk.size(); ++i)
      {
         const std::string& k  = bk[i];
         std::string        nv = (i % 3 == 0) ? bigval(int(i), 2100 + (r() % 9000))
                                              : (i % 3 == 1 ? std::string("small-now")
                                                            : bigval(int(i), 3000 + (r() % 3000)));
         t.upsert(k, nv);
         oracle[k] = nv;
      }
      for (auto& k : bk)
      {
         ++checks;
         if (!t.find(k, out) || out != oracle[k])
         {
            if (++fails <= 10)
               std::printf("BIG-UPD-MISS %s\n", k.c_str());
         }
      }
      // remove half (frees BIG objects); rest ride through churn + drain-to-empty
      for (size_t i = 0; i < bk.size() / 2; ++i)
      {
         ++checks;
         if (t.remove(bk[i]) != 1)
         {
            if (++fails <= 10)
               std::printf("BIG-RM-MISS\n");
         }
         oracle.erase(bk[i]);
         ++checks;
         if (t.find(bk[i], out))
         {
            if (++fails <= 10)
               std::printf("BIG-RESIDUAL\n");
         }
      }
   }

   // ── long shared prefix before a 256-way fan-out → a full node whose prefix
   //    exceeds full::max_prefix (24B): exercises build_node's prefix-split (the
   //    Silkworm storage-key shape; key.len here > the oracle's usual ≤21). ─────
   {
      for (int rep = 0; rep < 3; ++rep)
      {
         std::string cp(28 + rep * 5, char('P' + rep));  // 28/33/38-byte common prefix (>24)
         for (int b = 0; b < 256; ++b)
         {
            std::string k = cp;
            k.push_back(char(b));
            k += "LPtail" + std::to_string(rep);
            std::string v = "lp" + std::to_string(rep * 1000 + b);
            t.upsert(k, v);
            oracle[k] = v;
         }
         for (int b = 0; b < 256; ++b)
         {
            std::string k = cp;
            k.push_back(char(b));
            k += "LPtail" + std::to_string(rep);
            ++checks;
            if (!t.find(k, out) || out != oracle[k])
            {
               if (++fails <= 10)
                  std::printf("LONGPFX-MISS rep=%d b=%d\n", rep, b);
            }
         }
      }
   }

   // ── remove: return-value contract (1=was present, 0=absent) ────────────────
   for (int i = 0; i < 5000; ++i)
   {
      uint64_t    h = (r() + 999) * 0x9E3779B97F4A7C15ull;
      std::string k((char*)&h, 8);
      if (oracle.count(k))
         continue;
      ++checks;
      if (t.remove(k) != 0)
      {
         if (++fails <= 10)
            std::printf("RM-PHANTOM (removed absent key)\n");
      }
   }

   // ── remove ~40% of the keys, cross-check return value + post-state ─────────
   {
      std::vector<std::string> keys;
      keys.reserve(oracle.size());
      for (auto& [k, v] : oracle)
         keys.push_back(k);
      std::shuffle(keys.begin(), keys.end(), r);
      size_t ndel = keys.size() * 4 / 10;
      for (size_t i = 0; i < ndel; ++i)
      {
         const std::string& k = keys[i];
         ++checks;
         int rc = t.remove(k);
         if (rc != 1)
         {
            if (++fails <= 10)
               std::printf("RM-MISS %s\n", k.c_str());
         }
         oracle.erase(k);
         ++checks;
         if (t.find(k, out))
         {
            if (++fails <= 10)
               std::printf("RM-RESIDUAL %s\n", k.c_str());
         }
      }
   }
   // full re-verify after the bulk delete (survivors intact, re-nesting correct)
   for (auto& [k, v] : oracle)
   {
      bool got = t.find(k, out);
      ++checks;
      if (!got || out != v)
      {
         if (++fails <= 10)
            std::printf("POST-DEL MISS %s\n", k.c_str());
      }
   }

   // ── churn: interleave upsert + remove, periodic full verify ────────────────
   for (int round = 0; round < 60000; ++round)
   {
      std::string k = (r() & 1) ? mkk() : [&]
      {
         uint64_t h = r() * 0x9E3779B97F4A7C15ull;
         return std::string((char*)&h, 8);
      }();
      if (r() % 3 == 0)
      {
         int rc   = t.remove(k);
         int want = oracle.count(k) ? 1 : 0;
         ++checks;
         if (rc != want)
         {
            if (++fails <= 10)
               std::printf("CHURN-RM rc=%d want=%d\n", rc, want);
         }
         oracle.erase(k);
      }
      else
      {
         std::string v = "w" + std::to_string(r() % 100000);
         t.upsert(k, v);
         oracle[k] = v;
      }
      if (round % 10000 == 0)
      {
         for (auto& [kk, vv] : oracle)
         {
            if (!t.find(kk, out) || out != vv)
            {
               ++checks;
               if (++fails <= 10)
                  std::printf("CHURN MISS %s\n", kk.c_str());
            }
         }
      }
   }
   for (auto& [k, v] : oracle)
   {
      bool got = t.find(k, out);
      ++checks;
      if (!got || out != v)
      {
         if (++fails <= 10)
            std::printf("CHURN-FINAL MISS %s\n", k.c_str());
      }
   }
   double bpk = oracle.empty() ? 0 : double(e.live_bytes()) / oracle.size();

   // ── drain to empty: remove EVERY remaining key, tree must end empty ─────────
   {
      std::vector<std::string> keys;
      for (auto& [k, v] : oracle)
         keys.push_back(k);
      for (auto& k : keys)
      {
         ++checks;
         if (t.remove(k) != 1)
         {
            if (++fails <= 10)
               std::printf("DRAIN-MISS %s\n", k.c_str());
         }
      }
      oracle.clear();
      for (int i = 0; i < 2000; ++i)
      {
         uint64_t    h = r() * 0x9E3779B97F4A7C15ull;
         std::string k((char*)&h, 8);
         ++checks;
         if (t.find(k, out))
         {
            if (++fails <= 10)
               std::printf("DRAIN-RESIDUAL\n");
         }
      }
      e.drain();
      uint64_t lb = e.live_bytes();
      ++checks;
      if (lb != 0)
      {
         if (++fails <= 10)
            std::printf("DRAIN-LEAK live=%lluB after removing all keys\n", (unsigned long long)lb);
      }
   }

   std::printf("st-oracle: %d checks, %d fails, keys-at-bpk=(churn), live=%lluB, bytes/key=%.0f\n",
               checks, fails, (unsigned long long)0, bpk);
   return fails ? 1 : 0;
}

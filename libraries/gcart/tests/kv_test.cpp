// gcart::kv adb-conformance + functional gate.
#include <gcart/kv.hpp>
#include <adb/concepts.hpp>

#include <cstdio>
#include <map>
#include <random>
#include <string>

static_assert(adb::kv_database<gcart::kv::database>, "gcart::kv::database must model adb::kv_database");
static_assert(adb::kv_cursor<gcart::kv::cursor>, "gcart cursor must model adb::kv_cursor");
static_assert(adb::kv_tree<gcart::kv::tree>, "gcart snapshot tree must model adb::kv_tree");

namespace
{
   int g_checks = 0, g_fails = 0;
   void ok(bool c, const char* w) { ++g_checks; if (!c) { ++g_fails; if (g_fails <= 20) std::fprintf(stderr, "FAIL: %s\n", w); } }
}

int main()
{
   namespace fs = std::filesystem;
   const fs::path dir = fs::temp_directory_path() / "gcart_kv_test";
   fs::remove_all(dir);

   auto db  = gcart::kv::database::create(dir);
   auto ws  = db->start_write_session();
   std::map<std::string, std::string> oracle;
   std::mt19937 r(2024);

   // batched transactions of CRUD
   for (int batch = 0; batch < 40; ++batch)
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 2000; ++i)
      {
         int n = 3 + int(r() % 18);
         std::string k; for (int j = 0; j < n; ++j) k.push_back(char('a' + r() % 26));
         std::string v = "v" + std::to_string(r() % 100000);
         tx.upsert(k, v); oracle[k] = v;
      }
      // a few removes
      if (!oracle.empty())
         for (int i = 0; i < 200; ++i)
         {
            auto it = oracle.begin(); std::advance(it, r() % oracle.size());
            tx.remove(it->first); oracle.erase(it);
         }
      tx.commit();
   }

   // point reads via a fresh transaction
   {
      auto tx = ws->start_transaction(0);
      for (auto& [k, v] : oracle)
      {
         auto got = tx.get<std::string>(k);
         ok(got.has_value() && *got == v, "tx.get matches oracle");
      }
      bool miss = tx.get("____absent____", [](std::string_view) {});
      ok(!miss, "absent key misses");
      tx.abort();
   }

   // snapshot + ordered cursor scan == oracle order
   {
      auto snap = ws->get_root(0);
      auto cur  = snap.snapshot_cursor();
      size_t i  = 0;
      bool   ord = true;
      auto   it  = oracle.begin();
      for (bool ok2 = cur.seek_begin(); ok2; ok2 = cur.next(), ++it, ++i)
      {
         if (it == oracle.end() || cur.key() != it->first || *cur.value() != it->second) { ord = false; break; }
      }
      ok(ord && i == oracle.size() && it == oracle.end(), "snapshot cursor in-order == oracle");

      // lower_bound
      auto lb = oracle.lower_bound("m");
      cur.lower_bound("m");
      ok((lb == oracle.end()) == cur.is_end(), "lower_bound end-ness matches");
      if (lb != oracle.end()) ok(cur.key() == lb->first, "lower_bound key matches");

      // reverse scan
      i = 0; bool rord = true; auto rit = oracle.rbegin();
      for (bool ok2 = cur.seek_last(); ok2; ok2 = cur.prev(), ++rit, ++i)
         if (rit == oracle.rend() || cur.key() != rit->first) { rord = false; break; }
      ok(rord && i == oracle.size(), "reverse cursor == oracle (descending)");
   }

   // snapshot isolation: snapshot taken, then mutate, snapshot unchanged
   {
      auto snap = ws->get_root(0);
      std::string probe; { auto c = snap.snapshot_cursor(); c.seek_begin(); probe = std::string(c.key()); }
      auto sval = snap.get<std::string>(probe);
      { auto tx = ws->start_transaction(0); tx.remove(probe); tx.commit(); }
      auto still = snap.get<std::string>(probe);  // snapshot must still see it
      ok(still.has_value() && still == sval, "snapshot isolation: removed key still in snapshot");
      ok(!ws->start_transaction(0).get<std::string>(probe).has_value(), "live tree no longer has it");
   }

   std::printf("gcart-kv-tests: %d checks, %d fails (live=%.1f MB)\n", g_checks, g_fails,
               db->store().live_bytes() / 1048576.0);
   return g_fails == 0 ? 0 : 1;
}

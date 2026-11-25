// Defects found by the review of the Large Engine (gofractally/psitri#19).
// Each case is the reproduction of one issue and fails on the code before
// its fix.
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include <sal_art_ref/engine.hpp>

namespace
{
   using engine = sal_art_ref::radix::engine;
   using frame  = sal_art_ref::radix::radix_tree::frame;

   struct temp_dir
   {
      std::filesystem::path path;
      explicit temp_dir(const char* name)
          : path(std::filesystem::temp_directory_path() /
                 (std::string("psitri-review-") + name + "-" + std::to_string(::getpid())))
      {
         std::filesystem::remove_all(path);
      }
      ~temp_dir() { std::filesystem::remove_all(path); }
   };

   using model_t = std::map<std::string, std::string>;

   // Every key of the model reads back by key, and a forward and a backward walk
   // of the committed root return exactly the model's keys and values in order.
   void check_against(engine::write_session& ws, const model_t& model)
   {
      auto tree = ws.get_root(0);
      for (const auto& [k, v] : model)
      {
         auto got = tree.get<std::string>(k);
         INFO("key of " << k.size() << " bytes");
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      auto   c  = tree.snapshot_cursor();
      auto   it = model.begin();
      size_t n  = 0;
      for (bool ok = c.seek_begin(); ok; ok = c.next(), ++it, ++n)
      {
         REQUIRE(it != model.end());
         REQUIRE(std::string(c.key()) == it->first);
         REQUIRE(*c.value<std::string>() == it->second);
      }
      REQUIRE(n == model.size());
      auto rit = model.rbegin();
      n        = 0;
      for (bool ok = c.seek_last(); ok; ok = c.prev(), ++rit, ++n)
      {
         REQUIRE(rit != model.rend());
         REQUIRE(std::string(c.key()) == rit->first);
      }
      REQUIRE(n == model.size());
   }
   void put(engine::write_session& ws, model_t& model, const std::string& k, const std::string& v)
   {
      auto tx = ws.start_transaction(0);
      tx.upsert(k, v);
      tx.commit();
      model[k] = v;
   }
   void del(engine::write_session& ws, model_t& model, const std::string& k)
   {
      auto tx = ws.start_transaction(0);
      CHECK(tx.remove(k) == (model.count(k) ? 0 : -1));
      tx.commit();
      model.erase(k);
   }
}  // namespace

// gofractally/psitri#26: a moved-from savepoint stayed active, and its destructor
// put its empty saved root in place of the transaction's working root.
TEST_CASE("review: a savepoint that was moved does not wipe the working root", "[review]")
{
   temp_dir dir("frame-move");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 100; ++i)
         tx.upsert("base" + std::to_string(i), "v");
      tx.commit();
   }
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("in-tx", "v");
      {
         std::optional<frame> sp;
         sp.emplace(tx.sub_transaction());  // move-constructed from the temporary
         sp->upsert("in-frame", "v");
         sp->commit();
      }
      CHECK(tx.snapshot_cursor().count_keys() == 102);
      {
         // A moved savepoint that is left to abort restores what it saved.
         std::optional<frame> sp;
         sp.emplace(tx.sub_transaction());
         sp->upsert("dropped", "v");
      }
      CHECK(tx.snapshot_cursor().count_keys() == 102);
      CHECK_FALSE(tx.get<std::string>("dropped"));
      tx.commit();
   }
   CHECK(ws->get_root(0).count_keys() == 102);
}

// The same for a transaction: the moved-from object is closed and the
// destination commits what was written.
TEST_CASE("review: a transaction that was moved commits once", "[review]")
{
   temp_dir dir("tx-move");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      std::optional<sal_art_ref::radix::radix_tree::transaction> held;
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "1");
         held.emplace(std::move(tx));
      }  // the moved-from transaction dies here
      held->upsert("b", "2");
      held->commit();
   }
   CHECK(ws->get_root(0).count_keys() == 2);
   auto again = ws->start_transaction(0);  // the root's writer lock was released
   again.upsert("c", "3");
   again.commit();
   CHECK(ws->get_root(0).count_keys() == 3);
}

// gofractally/psitri#28: a key stored with its full bytes in an external record
// (its suffix was over 255 bytes when it arrived) kept only a 123-byte fragment
// in the page. Once splits had tightened the page's common prefix, an update
// replaced the external record by one without the key.
TEST_CASE("review: updating a long key keeps the key", "[review]")
{
   temp_dir dir("long-key-update");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   model_t  model;

   const std::string prefix(200, 'p');
   const std::string first = prefix + 'm' + std::string(199, 'q');  // 400 bytes
   put(*ws, model, first, "first-value");
   for (int i = 0; i < 300; ++i)
   {
      std::string k = prefix;
      k += char('a' + i / 26 % 26);
      k += char('a' + i % 26);
      k += char('0' + i / 676);
      put(*ws, model, k, "0123456789");
   }
   check_against(*ws, model);
   put(*ws, model, first, "second-value");
   check_against(*ws, model);
   // Every storage kind, updated to a small value, a value too large to stay in
   // the page, and back.
   const std::string big(600, 'B');
   for (const auto& [k, v] : model_t(model))
   {
      put(*ws, model, k, big);
      put(*ws, model, k, "s");
   }
   check_against(*ws, model);
   for (int len : {130, 254, 255, 256, 257, 400, 1024})
   {
      std::string k = len < int(prefix.size())
                          ? std::string(size_t(len), 'k')
                          : prefix + std::string(size_t(len) - prefix.size(), char('A' + len % 26));
      put(*ws, model, k, "v1");
      put(*ws, model, k, big);
      put(*ws, model, k, "v3");
   }
   check_against(*ws, model);
}

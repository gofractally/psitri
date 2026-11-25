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

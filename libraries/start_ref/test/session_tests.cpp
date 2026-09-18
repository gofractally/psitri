#include <catch2/catch_test_macros.hpp>
#include <start_ref/kv.hpp>
#include <unistd.h>

TEST_CASE("Small Engine write sessions begin from the latest committed roots", "[session-refresh]")
{
   auto path = std::filesystem::temp_directory_path() / ("small_sessions_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   {
      auto db = start_ref::kv::database::create(path);
      auto first = db->start_write_session();
      auto second = db->start_write_session();
      // Exercise both root 0's eager working tree and lazy auxiliary roots.
      for (uint32_t root : {0u, 3u})
      {
         { auto tx = second->start_transaction(root); tx.abort(); }
         { auto tx = first->start_transaction(root); tx.upsert("a", "one"); tx.commit(); }
         auto held = db->get_root_owned(root);
         { auto tx = second->start_transaction(root);
           REQUIRE(tx.get<std::string>("a") == "one");
           tx.upsert("b", "two"); tx.commit(); }
         { auto tx = first->start_transaction(root);
           REQUIRE(tx.get<std::string>("b") == "two");
           tx.upsert("a", "aborted"); tx.remove("b"); tx.abort(); }
         { auto tx = first->start_transaction(root);
           CHECK(tx.get<std::string>("a") == "one");
           CHECK(tx.get<std::string>("b") == "two"); tx.commit(); }
         CHECK(held.get<std::string>("a") == "one");
         CHECK_FALSE(held.get<std::string>("b"));
      }
      { auto tx = second->start_transaction(0);
        auto aux = tx.open_root(3); aux.upsert("c", "three"); tx.commit(); }
      { auto tx = first->start_transaction(0); auto aux = tx.open_root(3);
        REQUIRE(aux.get<std::string>("c") == "three");
        aux.upsert("d", "four"); tx.commit(); }
      { auto tx = second->start_transaction(3);
        CHECK(tx.get<std::string>("d") == "four"); tx.remove("a"); tx.commit(); }
      { auto tx = first->start_transaction(3);
        CHECK_FALSE(tx.get<std::string>("a")); tx.upsert("e", "five"); tx.commit(); }
   }
   { auto db = start_ref::kv::database::open(path, sal::open_mode::open_existing, {});
     auto root = db->get_root_owned(3);
     CHECK_FALSE(root.get<std::string>("a"));
     CHECK(root.get<std::string>("b") == "two");
     CHECK(root.get<std::string>("c") == "three");
     CHECK(root.get<std::string>("d") == "four");
     CHECK(root.get<std::string>("e") == "five"); }
   std::filesystem::remove_all(path);
}

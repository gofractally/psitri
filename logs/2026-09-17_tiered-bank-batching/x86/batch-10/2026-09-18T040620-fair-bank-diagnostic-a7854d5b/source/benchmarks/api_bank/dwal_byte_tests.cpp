#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <psitri/dwal/byte_engine.hpp>
#include <sal_art_ref/engine.hpp>
#include <start_ref/kv.hpp>
#include <unistd.h>

TEMPLATE_TEST_CASE("DWAL byte snapshots survive rollback and repeated merges", "[dwal-byte]",
                   sal_art_ref::radix::engine, start_ref::kv::engine)
{
   using Engine = psitri::dwal::byte_engine<TestType>;
   auto path = std::filesystem::temp_directory_path() / ("dwal_byte_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   typename Engine::config cfg;
   if constexpr (requires { cfg.base.enable_read_cache; })
      cfg.base.enable_read_cache = false;
   cfg.buffering.max_rw_entries = 16;
   {
      auto db = Engine::database::open(path, sal::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "original"); tx.upsert("b", "keep"); tx.commit();
      }
   }
   {
      // Reopen so held cursors pin a populated base, not only the DWAL overlay.
      auto db = Engine::database::open(path, sal::open_mode::open_existing, cfg);
      auto ws = db->start_write_session();
      auto rs = db->start_read_session();
      auto old = rs->create_cursor(0);
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "aborted");
         auto cur = tx.create_cursor();
         REQUIRE(cur.seek_begin()); CHECK(cur.key() == "a");
         CHECK(cur.template value<std::string>() == "aborted");
         tx.abort();
         CHECK(cur.template get<std::string>("a") == "aborted");
      }
      for (int round = 0; round < 30; ++round)
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "new");
         tx.remove("b");
         for (int i = 0; i < 32; ++i)
            tx.upsert("extra-" + std::to_string(round * 32 + i), "value");
         tx.commit();
      }
      REQUIRE(db->dwal.root(0).generation.load() > 0);
      CHECK(old.template get<std::string>("a") == "original");
      CHECK(old.template get<std::string>("b") == "keep");
      CHECK_FALSE(old.template get<std::string>("extra-0"));
      // A second cursor in the same read transaction must retain the same view.
      REQUIRE(old.seek_begin());
      auto second = rs->create_cursor(0);
      REQUIRE(second.seek_begin()); CHECK(second.key() == "a");
      CHECK(second.template value<std::string>() == "original");
      REQUIRE(second.next()); CHECK(second.key() == "b");
      CHECK_FALSE(second.next());
      CHECK(old.key() == "a");
      REQUIRE(old.next()); CHECK(old.key() == "b");
      auto fresh = db->start_read_session()->create_cursor(0);
      CHECK(fresh.template get<std::string>("a") == "new");
      CHECK_FALSE(fresh.template get<std::string>("b"));
      CHECK(fresh.template get<std::string>("extra-959") == "value");
   }
   {
      auto db = Engine::database::open(path, sal::open_mode::open_existing, cfg);
      auto fresh = db->start_read_session()->create_cursor(0);
      CHECK(fresh.template get<std::string>("a") == "new");
      CHECK_FALSE(fresh.template get<std::string>("b"));
      CHECK(fresh.template get<std::string>("extra-959") == "value");
   }
   std::filesystem::remove_all(path);
}

TEST_CASE("Small Engine range removal includes working changes and preserves snapshots", "[dwal-byte]")
{
   using Base = start_ref::kv::engine;
   auto path = std::filesystem::temp_directory_path() / ("small_range_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   {
      auto db = Base::database::open(path, sal::open_mode::create_only, {});
      auto ws = db->start_write_session();
      { auto tx = ws->start_transaction(0);
        tx.upsert("a", "a"); tx.upsert("c", "c"); tx.upsert("d", "d"); tx.commit(); }
      auto old = db->get_root_owned(0);
      { auto tx = ws->start_transaction(0);
        tx.upsert("b", "b"); CHECK(tx.remove_range_counted("a", "d") == 3);
        CHECK_FALSE(tx.get<std::string>("a")); CHECK_FALSE(tx.get<std::string>("b"));
        CHECK(tx.get<std::string>("d") == "d"); tx.abort(); }
      { auto tx = ws->start_transaction(0);
        CHECK(tx.get<std::string>("a") == "a"); CHECK_FALSE(tx.get<std::string>("b"));
        CHECK(tx.remove_range_counted("z", "a") == 0);
        CHECK(tx.remove_range_counted("a", "a") == 0);
        CHECK(tx.remove_range_counted("a", "d") == 2); tx.commit(); }
      CHECK(old.get<std::string>("a") == "a"); CHECK(old.get<std::string>("c") == "c");
   }
   { auto db = Base::database::open(path, sal::open_mode::open_existing, {});
     auto cur = db->get_root_owned(0).snapshot_cursor();
     REQUIRE(cur.seek_begin()); CHECK(cur.key() == "d"); CHECK_FALSE(cur.next()); }
   std::filesystem::remove_all(path);
}

TEMPLATE_TEST_CASE("DWAL drains range-only buffers into its base", "[dwal-byte]",
                   sal_art_ref::radix::engine, start_ref::kv::engine)
{
   using Engine = psitri::dwal::byte_engine<TestType>;
   auto path = std::filesystem::temp_directory_path() / ("dwal_small_range_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   typename Engine::config cfg;
   cfg.buffering.merge_threads = 0;
   { auto db = Engine::database::open(path, sal::open_mode::create_only, cfg);
     auto tx = db->dwal.start_write_transaction(0);
     tx.upsert("a", "a"); tx.upsert("b", "b"); tx.upsert("c", "c"); tx.commit(); }
   { auto db = Engine::database::open(path, sal::open_mode::open_existing, cfg);
     auto tx = db->dwal.start_write_transaction(0);
     tx.remove_range("a", "c"); tx.commit(); }
   { auto db = Engine::database::open(path, sal::open_mode::open_existing, cfg);
     auto cur = db->start_read_session()->create_cursor(0);
     CHECK_FALSE(cur.template get<std::string>("a")); CHECK_FALSE(cur.template get<std::string>("b"));
     REQUIRE(cur.seek_begin()); CHECK(cur.key() == "c"); CHECK_FALSE(cur.next()); }
   std::filesystem::remove_all(path);
}

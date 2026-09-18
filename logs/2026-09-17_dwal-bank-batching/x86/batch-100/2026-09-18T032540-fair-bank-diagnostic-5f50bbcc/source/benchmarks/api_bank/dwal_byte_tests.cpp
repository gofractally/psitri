#include <catch2/catch_test_macros.hpp>
#include <psitri/dwal/byte_engine.hpp>
#include <sal_art_ref/engine.hpp>
#include <unistd.h>

TEST_CASE("DWAL byte snapshots survive rollback and repeated Large Engine merges", "[dwal-byte]")
{
   using Engine = psitri::dwal::byte_engine<sal_art_ref::radix::engine>;
   auto path = std::filesystem::temp_directory_path() / ("dwal_byte_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   Engine::config cfg;
   cfg.base.enable_read_cache = false;
   cfg.buffering.max_rw_entries = 16;
   {
      auto db = Engine::database::open(path, sal::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "original"); tx.upsert("b", "keep"); tx.commit();
      }
      auto rs = db->start_read_session();
      auto old = rs->create_cursor(0);
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "aborted");
         auto cur = tx.create_cursor();
         REQUIRE(cur.seek_begin()); CHECK(cur.key() == "a");
         CHECK(cur.value<std::string>() == "aborted");
         tx.abort();
         CHECK(cur.get<std::string>("a") == "aborted");
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
      CHECK(old.get<std::string>("a") == "original");
      CHECK(old.get<std::string>("b") == "keep");
      CHECK_FALSE(old.get<std::string>("extra-0"));
      // A second cursor in the same read transaction must retain the same view.
      auto second = rs->create_cursor(0);
      REQUIRE(second.seek_begin()); CHECK(second.key() == "a");
      CHECK(second.value<std::string>() == "original");
      REQUIRE(second.next()); CHECK(second.key() == "b");
      CHECK_FALSE(second.next());
      auto fresh = db->start_read_session()->create_cursor(0);
      CHECK(fresh.get<std::string>("a") == "new");
      CHECK_FALSE(fresh.get<std::string>("b"));
      CHECK(fresh.get<std::string>("extra-959") == "value");
   }
   {
      auto db = Engine::database::open(path, sal::open_mode::open_existing, cfg);
      auto fresh = db->start_read_session()->create_cursor(0);
      CHECK(fresh.get<std::string>("a") == "new");
      CHECK_FALSE(fresh.get<std::string>("b"));
      CHECK(fresh.get<std::string>("extra-959") == "value");
   }
   std::filesystem::remove_all(path);
}

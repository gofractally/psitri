#include <catch2/catch_test_macros.hpp>
#include <psitri/experimental/tiered.hpp>
#include <atomic>
#include <thread>
#include <unistd.h>

using Engine = psitri::experimental::tiered_engine<>;

TEST_CASE("Small and Large Engine tiering preserves snapshots, tombstones and reopen", "[tiered-pair]")
{
   auto path = std::filesystem::temp_directory_path() / ("tiered_pair_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   Engine::config cfg;
   cfg.commit_sync = sal::sync_type::none;
   cfg.bcfg.enable_read_cache = false;
   cfg.knobs.background = false;
   cfg.knobs.flush_chunk = 1;
   {
      auto db = Engine::database::open(path, sal::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      { auto tx = ws->start_transaction(0);
        tx.upsert("a", "one"); tx.upsert("b", "two"); tx.commit(); }
      auto old_rs = db->start_read_session(); auto old = old_rs->create_cursor(0);
      CHECK(db->flush_now() == 2); CHECK(db->base_commit_count() == 2);
      auto base_rs = db->start_read_session(); auto base = base_rs->create_cursor(1);
      CHECK(base.get<std::string>("a") == "one");
      { auto tx = ws->start_transaction(0);
        tx.upsert("a", "aborted"); tx.remove("b"); tx.abort(); }
      { auto tx = ws->start_transaction(0);
        CHECK(tx.get<std::string>("a") == "one"); CHECK(tx.get<std::string>("b") == "two");
        tx.upsert("a", "new"); tx.remove("b"); tx.upsert("c", "three"); tx.commit(); }
      CHECK(db->flush_now() == 3);
      CHECK(old.get<std::string>("a") == "one"); CHECK(old.get<std::string>("b") == "two");
      CHECK(base.get<std::string>("a") == "one"); CHECK(base.get<std::string>("b") == "two");
      auto fresh_rs = db->start_read_session(); auto fresh = fresh_rs->create_cursor(0);
      REQUIRE(fresh.seek_begin()); CHECK(fresh.key() == "a"); CHECK(fresh.value<std::string>() == "new");
      REQUIRE(fresh.next()); CHECK(fresh.key() == "c"); CHECK_FALSE(fresh.next());
      CHECK(db->flush_now() == 0);
      // Leave a committed hot value in Small Engine across close/reopen.
      { auto tx = ws->start_transaction(0); tx.upsert("d", "hot"); tx.commit(); }
   }
   {
      auto db = Engine::database::open(path, sal::open_mode::open_existing, cfg);
      auto rs = db->start_read_session(); auto cur = rs->create_cursor(0);
      CHECK(cur.get<std::string>("a") == "new"); CHECK_FALSE(cur.get<std::string>("b"));
      CHECK(cur.get<std::string>("c") == "three"); CHECK(cur.get<std::string>("d") == "hot");
   }
   std::filesystem::remove_all(path);
}

TEST_CASE("Tiered pair readers see complete transactions during background migration", "[tiered-pair]")
{
   auto path = std::filesystem::temp_directory_path() / ("tiered_pair_bg_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   Engine::config cfg;
   cfg.commit_sync = sal::sync_type::none;
   cfg.bcfg.enable_read_cache = false;
   cfg.knobs.hi_water_bytes = 256;
   cfg.knobs.flush_chunk = 1;
   {
      auto db = Engine::database::open(path, sal::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      { auto tx = ws->start_transaction(0); tx.upsert("a", "0"); tx.upsert("b", "10000"); tx.commit(); }
      std::atomic<bool> stop{false};
      std::atomic<uint64_t> errors{0}, reads{0};
      std::jthread reader([&] {
         auto rs = db->start_read_session();
         while (!stop.load())
         {
            auto cur = rs->create_cursor(0);
            auto a=cur.get<std::string>("a"), b=cur.get<std::string>("b");
            if (!a || !b || std::stoul(*a)+std::stoul(*b)!=10000) ++errors;
            ++reads;
         }
      });
      for (int n=1;n<=2000;++n)
      { auto tx = ws->start_transaction(0);
        tx.upsert("a", std::to_string(n)); tx.upsert("b", std::to_string(10000-n)); tx.commit(); }
      stop=true; reader.join();
      REQUIRE(reads.load()>0); CHECK(errors.load()==0);
      CHECK(db->base_commit_count()>0);
      // Wait for the background worker to finish any in-flight rotation.
      // Close joins it; persisted hot + base must reopen to the final state.
   }
   {
      auto db = Engine::database::open(path, sal::open_mode::open_existing, cfg);
      auto rs = db->start_read_session(); auto cur = rs->create_cursor(0);
      CHECK(cur.get<std::string>("a") == "2000"); CHECK(cur.get<std::string>("b") == "8000");
   }
   std::filesystem::remove_all(path);
}

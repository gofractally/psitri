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

#include <signal.h>
#include <sys/wait.h>
#include <atomic>
#include <thread>

// This suite is built with assertions on, so the detector is active here.
TEST_CASE("Small Engine: two threads active on one write session are detected", "[session-threads]")
{
   auto path = std::filesystem::temp_directory_path() / ("small_two_threads_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   // Serialized use from several threads is the supported pattern: no report.
   {
      auto db = start_ref::kv::database::create(path);
      auto ws = db->start_write_session();
      for (int i = 0; i < 20; ++i)
      {
         std::thread t([&] {
            auto tx = ws->start_transaction(0);
            tx.upsert("k", std::to_string(i));
            // a read callback that reads again on its own thread is re-entry, not a second user
            REQUIRE(tx.get("k", [&](std::string_view) { CHECK(tx.get<std::string>("k")); }));
            tx.commit();
         });
         t.join();
      }
   }
   // Overlapping use: one thread is inside a read callback while another writes.
   std::fflush(nullptr);
   const pid_t pid = fork();
   REQUIRE(pid >= 0);
   if (pid == 0)
   {
      (void)!freopen("/dev/null", "w", stderr);
      (void)!freopen("/dev/null", "w", stdout);  // and the test framework's own report of the abort
      auto db = start_ref::kv::database::open(path, sal::open_mode::open_existing, {});
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      std::atomic<int> stage{0};
      std::thread      intruder([&] {
         while (stage.load() != 1)
            std::this_thread::yield();
         tx.upsert("other", "x");  // the detector stops the process here
         stage = 2;
      });
      tx.get("k", [&](std::string_view) {
         stage = 1;
         while (stage.load() != 2)
            std::this_thread::yield();
      });
      intruder.join();
      _exit(0);  // not detected
   }
   int status = 0;
   waitpid(pid, &status, 0);
   REQUIRE(WIFSIGNALED(status));
   REQUIRE(WTERMSIG(status) == SIGABRT);
   std::filesystem::remove_all(path);
}

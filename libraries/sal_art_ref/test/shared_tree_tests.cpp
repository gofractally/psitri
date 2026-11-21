// shared_tree: the handle that carries a tree between the single writer and reader threads.
// Readers open their own cursors on snapshots the writer shared, concurrently with further
// publication, and drop the references on their own threads.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sal_art_ref/engine.hpp>

namespace
{
   using engine = sal_art_ref::radix::engine;
   constexpr unsigned key_count = 400;

   std::string key(unsigned i)
   {
      std::string k = "key-000000";
      for (int p = 9; i; i /= 10, --p)
         k[size_t(p)] = char('0' + i % 10);
      return k;
   }
   std::string value(unsigned generation, unsigned i)
   {
      return std::to_string(generation) + ":" + std::string(40 + i % 7, char('a' + generation % 26));
   }
   // Every key present, all from one generation: a snapshot is never a mix.
   unsigned check(engine::cursor c)
   {
      unsigned generation = 0, seen = 0;
      for (bool ok = c.lower_bound(std::string_view{}); ok; ok = c.next(), ++seen)
      {
         auto v = c.value<std::string>();
         REQUIRE(v);
         const unsigned g = unsigned(std::stoul(*v));
         if (seen == 0)
            generation = g;
         REQUIRE(c.key() == key(seen));
         REQUIRE(*v == value(generation, seen));
      }
      REQUIRE(seen == key_count);
      return generation;
   }
}  // namespace

TEST_CASE("shared_tree: readers on shared snapshots run beside the writer", "[shared_tree]")
{
   const auto dir = std::filesystem::temp_directory_path() / "heart_shared_tree";
   std::filesystem::remove_all(dir);
   uint64_t baseline = 0;
   {
      auto db     = engine::database::open(dir, sal::open_mode::create_only);
      auto writer = db->start_write_session();
      auto build  = [&](unsigned generation, engine::shared_tree base)
      {
         auto edit = writer->start_write_transaction(base ? writer->adopt(base)
                                                          : writer->create_temporary_tree());
         for (unsigned i = 0; i < key_count; ++i)
            edit.upsert(key(i), value(generation, i));
         return writer->share(edit.get_tree());
      };

      std::mutex          mutex;
      engine::shared_tree latest = build(1, {});
      std::atomic<bool>   done{false};
      std::atomic<unsigned> reads{0};

      std::vector<std::thread> readers;
      for (unsigned r = 0; r < 4; ++r)
         readers.emplace_back(
             [&]
             {
                auto     session = db->start_read_session();
                unsigned last    = 0;
                while (!done.load(std::memory_order_acquire))
                {
                   engine::shared_tree mine;
                   {
                      std::lock_guard l(mutex);
                      mine = latest;
                   }
                   const unsigned generation = check(session->create_cursor(mine));
                   REQUIRE(generation >= last);  // never goes back in time
                   last = generation;
                   ++reads;
                   // `mine` is dropped here, on the reader thread.
                }
             });

      for (unsigned generation = 2; generation <= 300; ++generation)
      {
         engine::shared_tree base;
         {
            std::lock_guard l(mutex);
            base = latest;
         }
         auto next = build(generation, base);
         // Unpublished trees are readable too; publish every tenth one.
         if (generation % 10 == 0)
         {
            auto tx = writer->start_transaction(0);
            tx.primary().set_root(writer->adopt(next));
            tx.commit();
         }
         std::lock_guard l(mutex);
         latest = std::move(next);
      }
      while (reads.load() < 50)
         std::this_thread::yield();
      done = true;
      for (auto& t : readers)
         t.join();

      // A reader's view of the published root can leave its thread.
      auto reader    = db->start_read_session();
      auto published = reader->get_root(0);
      std::thread([&] {
         auto session = db->start_read_session();
         REQUIRE(check(session->create_cursor(published)) == 300);
      }).join();
      published.reset();

      // Only the published root keeps anything alive once the shared handles are gone.
      latest.reset();
      REQUIRE(db->wait_for_compactor(std::chrono::seconds(30)));
      baseline = db->engine_for_testing().object_count();
      REQUIRE(baseline > 0);
      REQUIRE(check(reader->create_cursor(0)) == 300);
   }
   {
      // Handles dropped right before close are still released: nothing leaks across a clean
      // close and reopen.
      auto db = engine::database::open(dir, sal::open_mode::open_existing);
      REQUIRE(db->recovery_performed() == sal::recovery_mode::none);
      REQUIRE(db->wait_for_compactor(std::chrono::seconds(30)));
      REQUIRE(db->engine_for_testing().object_count() == baseline);
      auto writer = db->start_write_session();
      auto shared = writer->share(writer->get_root(0));
      auto edit   = writer->start_write_transaction(writer->adopt(shared));
      for (unsigned i = 0; i < key_count; ++i)
         edit.upsert(key(i), value(301, i));
      auto dropped_at_close = writer->share(edit.get_tree());
   }
   auto db = engine::database::open(dir, sal::open_mode::open_existing);
   REQUIRE(db->recovery_performed() == sal::recovery_mode::none);
   REQUIRE(db->wait_for_compactor(std::chrono::seconds(30)));
   REQUIRE(db->engine_for_testing().object_count() == baseline);
   std::filesystem::remove_all(dir);
}

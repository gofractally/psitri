// shared_tree: the handle that carries a tree between the single writer and reader threads.
// Readers open their own cursors on snapshots the writer shared, concurrently with further
// publication, and drop the references on their own threads.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
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
   // No test macro in here: it runs on reader threads, and the framework's
   // assertions are for one thread at a time. nullopt = the snapshot was wrong.
   std::optional<unsigned> scan(engine::cursor c)
   {
      unsigned generation = 0, seen = 0;
      for (bool ok = c.lower_bound(std::string_view{}); ok; ok = c.next(), ++seen)
      {
         auto v = c.value<std::string>();
         if (!v)
            return std::nullopt;
         const unsigned g = unsigned(std::stoul(*v));
         if (seen == 0)
            generation = g;
         if (c.key() != key(seen) || *v != value(generation, seen))
            return std::nullopt;
      }
      if (seen != key_count)
         return std::nullopt;
      return generation;
   }
   // The same, asserted: for the test's own thread only.
   unsigned check(engine::cursor c)
   {
      auto generation = scan(std::move(c));
      REQUIRE(generation);
      return *generation;
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
      std::atomic<unsigned> wrong{0};

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
                   // Half the readers borrow: no reference taken on the shared root.
                   const auto generation = scan(reads.load() % 2 ? session->borrow_cursor(mine)
                                                                 : session->create_cursor(mine));
                   if (!generation || *generation < last)  // never goes back in time
                      ++wrong;
                   else
                      last = *generation;
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
      CHECK(wrong.load() == 0);

      // A reader's view of the published root can leave its thread.
      auto reader    = db->start_read_session();
      auto published = reader->get_root(0);
      std::optional<unsigned> elsewhere;
      std::thread([&] {
         auto session = db->start_read_session();
         elsewhere    = scan(session->create_cursor(published));
      }).join();
      REQUIRE(elsewhere == 300u);
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

#include <map>
#include <random>

// An embedder that keeps each version as a shared tree edits it one key at a
// time: adopt the tree, change a key, share the result, and keep older
// versions to return to. Every seek into the newest version, from every
// possible key, has to agree with a plain map, and so does every point read.
TEST_CASE("shared_tree: seeks and point reads agree after single-key edits of kept versions",
          "[shared_tree][versions]")
{
   const auto dir = std::filesystem::temp_directory_path() / "heart_shared_tree_versions";
   std::filesystem::remove_all(dir);
   {
      auto db     = engine::database::open(dir, sal::open_mode::create_only);
      auto writer = db->start_write_session();
      auto reader = db->start_read_session();
      using model = std::map<std::string, std::string>;

      std::vector<std::string> all_keys;  // every key of 1..3 letters from a..d
      for (int len = 1; len <= 3; ++len)
         for (int n = 0, end = 1 << (2 * len); n < end; ++n)
         {
            std::string k(size_t(len), 'a');
            for (int p = 0; p < len; ++p)
               k[size_t(p)] = char('a' + ((n >> (2 * p)) & 3));
            all_keys.push_back(k);
         }

      std::mt19937 rng(20261122);
      auto random_key = [&]
      {
         std::string k(1 + rng() % 3, 'a');
         for (auto& c : k)
            c = char('a' + rng() % 4);
         return k;
      };
      auto edit = [&](engine::shared_tree& tree, auto&& change)
      {
         auto tx = writer->start_write_transaction(tree ? writer->adopt(tree)
                                                        : writer->create_temporary_tree());
         change(tx);
         tree = writer->share(tx.get_tree());
      };
      auto verify = [&](const engine::shared_tree& tree, const model& expected, int step)
      {
         INFO("step " << step);
         if (!tree)
         {
            REQUIRE(expected.empty());
            return;
         }
         auto c = reader->borrow_cursor(tree);
         // A full scan.
         model seen;
         for (bool ok = c.lower_bound(std::string_view{}); ok; ok = c.next())
            seen[std::string(c.key())] = *c.value<std::string>();
         REQUIRE(seen == expected);
         // Every seek and point read, from every possible key.
         for (const auto& k : all_keys)
         {
            INFO("key " << k);
            auto it = expected.lower_bound(k);
            REQUIRE(c.lower_bound(k) == (it != expected.end()));
            if (it != expected.end())
               REQUIRE(std::string(c.key()) == it->first);
            // The key just after `k`: how an embedder steps to the next key.
            std::string after = k;
            after.push_back('\0');
            auto next = expected.upper_bound(k);
            REQUIRE(c.lower_bound(after) == (next != expected.end()));
            if (next != expected.end())
               REQUIRE(std::string(c.key()) == next->first);
            auto exact = expected.find(k);
            auto got   = c.get<std::string>(k);
            REQUIRE(bool(got) == (exact != expected.end()));
            if (got)
               REQUIRE(*got == exact->second);
         }
      };

      std::vector<engine::shared_tree> versions(1);
      std::vector<model>               models(1);
      for (int step = 0; step < 4000; ++step)
      {
         const auto roll = rng() % 100;
         if (roll < 55)
         {
            const auto k = random_key();
            const auto v = "s" + std::to_string(step);
            edit(versions.back(), [&](auto& tx) { tx.upsert(k, v); });
            models.back()[k] = v;
         }
         else if (roll < 75)
         {
            const auto k = random_key();
            if (versions.back())
               edit(versions.back(), [&](auto& tx) { tx.remove(k); });
            models.back().erase(k);
         }
         else if (roll < 85 && versions.size() < 5)
         {
            versions.push_back(versions.back());  // keep the version to return to
            models.push_back(models.back());
         }
         else if (roll < 90 && versions.size() > 1)
         {
            versions.erase(versions.end() - 2);  // keep the newest
            models.erase(models.end() - 2);
         }
         else if (roll < 95 && versions.size() > 1)
         {
            versions.pop_back();  // return to the one before
            models.pop_back();
         }
         verify(versions.back(), models.back(), step);
      }
   }
   std::filesystem::remove_all(dir);
}

// gofractally/psitri#24: an embedder that gives each reader thread its own
// session relies on running out of sessions being an exception it can catch.
TEST_CASE("starting more read sessions than the allocator has slots throws", "[shared_tree][limit]")
{
   const auto dir = std::filesystem::temp_directory_path() / "heart_session_limit";
   std::filesystem::remove_all(dir);
   {
      auto db = engine::database::open(dir, sal::open_mode::create_only);
      std::vector<decltype(db->start_read_session())> readers;
      bool                                            refused = false;
      for (int i = 0; i < 80 && !refused; ++i)
      {
         try
         {
            readers.push_back(db->start_read_session());
         }
         catch (const std::runtime_error&)
         {
            refused = true;
         }
      }
      CHECK(refused);
      CHECK(readers.size() < 64);  // the engine's own threads hold sessions too
      CHECK(readers.size() > 32);

      // The ones that were made still work, and a freed one can be replaced.
      readers.pop_back();
      CHECK_NOTHROW(readers.push_back(db->start_read_session()));
      auto cursor = readers.front()->create_cursor(0);
      CHECK_FALSE(cursor.get("absent"));
   }
   std::filesystem::remove_all(dir);
}

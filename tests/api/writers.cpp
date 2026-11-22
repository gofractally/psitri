// Writer and root semantics every engine must provide.
//
//  1. A database has several independent roots. Write sessions working on
//     different roots run at the same time without affecting each other.
//  2. A second writer on a root that already has an open transaction waits for
//     it; neither transaction's committed work is lost.
//  3. A reader sees only committed states of a root, and a cursor it holds does
//     not change when a writer commits.
//
// Each writer creates its write session on its own thread and uses it only
// there. Readers likewise.

#include <psitri/api.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace
{
   struct temporary_directory
   {
      std::filesystem::path path;
      temporary_directory()
      {
         std::string pattern =
             (std::filesystem::temp_directory_path() / "psitri-writers-XXXXXX").string();
         auto* result = ::mkdtemp(pattern.data());
         if (!result)
            throw std::runtime_error("mkdtemp failed");
         path = result;
      }
      ~temporary_directory()
      {
         std::error_code ignored;
         std::filesystem::remove_all(path, ignored);
      }
   };

   template <psitri::engine E>
   typename E::config config()
   {
      typename E::config cfg;
      cfg.sync_mode = psitri::sync_mode::none;
      if constexpr (std::same_as<E, psitri::small_engine>)
         cfg.max_segments = 32;
      else
         cfg.max_pinned_cache_size_mb = 128;
      return cfg;
   }

   std::string key(std::uint32_t root, std::uint32_t round, std::uint32_t i)
   {
      char buffer[40];
      std::snprintf(buffer, sizeof(buffer), "r%02u/%05u/%05u", root, round, i);
      return buffer;
   }

   // Number of keys under `root`, checking that every key belongs to it.
   template <typename Cursor>
   std::size_t count_own_keys(Cursor& cursor, std::uint32_t root, bool& foreign)
   {
      char prefix[8];
      std::snprintf(prefix, sizeof(prefix), "r%02u/", root);
      std::size_t n = 0;
      for (cursor.seek_begin(); !cursor.is_end(); cursor.next())
      {
         if (!cursor.key().starts_with(prefix))
            foreign = true;
         ++n;
      }
      return n;
   }
}  // namespace

TEMPLATE_TEST_CASE("Writers: sessions on different roots commit concurrently",
                   "[api][writers]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());

   constexpr std::uint32_t roots  = 4;
   constexpr std::uint32_t rounds = 40;
   constexpr std::uint32_t batch  = 250;

   std::atomic<bool>        go{false}, done{false}, torn{false}, foreign{false};
   std::atomic<std::size_t> snapshots{0};
   std::vector<std::thread> threads;
   for (std::uint32_t root = 0; root < roots; ++root)
   {
      threads.emplace_back(
          [&, root]
          {
             auto writer = db->start_write_session();
             while (!go)
                std::this_thread::yield();
             for (std::uint32_t round = 0; round < rounds; ++round)
             {
                auto tx = writer->start_transaction(root);
                for (std::uint32_t i = 0; i < batch; ++i)
                   tx.upsert(key(root, round, i), key(root, round, i));
                tx.commit();
             }
          });
   }
   // Readers: every state they see is a whole number of committed batches of
   // that root's own keys.
   for (std::uint32_t root = 0; root < roots; ++root)
   {
      threads.emplace_back(
          [&, root]
          {
             auto reader = db->start_read_session();
             while (!go)
                std::this_thread::yield();
             while (!done)
             {
                auto cursor = reader->create_cursor(root);
                bool other  = false;
                auto n      = count_own_keys(cursor, root, other);
                if (other)
                   foreign = true;
                if (n % batch != 0)
                   torn = true;
                ++snapshots;
             }
          });
   }
   go = true;
   for (std::uint32_t root = 0; root < roots; ++root)
      threads[root].join();
   done = true;
   for (std::uint32_t i = roots; i < threads.size(); ++i)
      threads[i].join();

   REQUIRE_FALSE(torn);
   REQUIRE_FALSE(foreign);
   REQUIRE(snapshots > 0);
   auto reader = db->start_read_session();
   for (std::uint32_t root = 0; root < roots; ++root)
   {
      auto cursor = reader->create_cursor(root);
      bool other  = false;
      REQUIRE(count_own_keys(cursor, root, other) == std::size_t{rounds} * batch);
      REQUIRE_FALSE(other);
   }
}

TEMPLATE_TEST_CASE("Writers: a second writer on a root waits and no commit is lost",
                   "[api][writers]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());

   constexpr std::uint32_t writers = 4;
   constexpr std::uint32_t rounds  = 200;
   std::atomic<bool>        go{false};
   std::atomic<int>         open{0};
   std::atomic<bool>        overlapped{false};
   std::vector<std::thread> threads;
   for (std::uint32_t w = 0; w < writers; ++w)
   {
      threads.emplace_back(
          [&, w]
          {
             auto writer = db->start_write_session();
             while (!go)
                std::this_thread::yield();
             for (std::uint32_t round = 0; round < rounds; ++round)
             {
                auto tx = writer->start_transaction(0);
                // Only one transaction may be open on the root at a time.
                if (open.fetch_add(1) != 0)
                   overlapped = true;
                tx.upsert(key(w, round, 0), "v");
                std::this_thread::yield();
                open.fetch_sub(1);
                tx.commit();
             }
          });
   }
   go = true;
   for (auto& thread : threads)
      thread.join();

   REQUIRE_FALSE(overlapped);
   auto        reader = db->start_read_session();
   auto        cursor = reader->create_cursor(0);
   std::size_t n      = 0;
   for (cursor.seek_begin(); !cursor.is_end(); cursor.next())
      ++n;
   REQUIRE(n == std::size_t{writers} * rounds);
}

TEMPLATE_TEST_CASE("Writers: a held cursor is unchanged by commits and aborts are invisible",
                   "[api][writers]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());

   auto writer = db->start_write_session();
   {
      auto tx = writer->start_transaction(2);
      for (std::uint32_t i = 0; i < 100; ++i)
         tx.upsert(key(2, 0, i), "first");
      tx.commit();
   }
   std::atomic<int>  stage{0};
   std::atomic<bool> wrong{false};
   std::thread       reader_thread(
       [&]
       {
          auto reader = db->start_read_session();
          auto held   = reader->create_cursor(2);
          stage       = 1;
          while (stage != 2)
             std::this_thread::yield();
          // The cursor created before the writer's work still shows that state.
          std::size_t n = 0;
          for (held.seek_begin(); !held.is_end(); held.next())
          {
             if (held.template value<std::string>() != std::optional<std::string>{"first"})
                wrong = true;
             ++n;
          }
          if (n != 100)
             wrong = true;
          // A new cursor shows the committed work and none of the aborted work.
          auto        fresh = reader->create_cursor(2);
          std::size_t m     = 0;
          for (fresh.seek_begin(); !fresh.is_end(); fresh.next())
          {
             if (fresh.template value<std::string>() != std::optional<std::string>{"second"})
                wrong = true;
             ++m;
          }
          if (m != 150)
             wrong = true;
       });
   while (stage != 1)
      std::this_thread::yield();
   {
      auto tx = writer->start_transaction(2);
      for (std::uint32_t i = 0; i < 150; ++i)
         tx.upsert(key(2, 0, i), "second");
      tx.commit();
   }
   {
      auto tx = writer->start_transaction(2);
      for (std::uint32_t i = 0; i < 300; ++i)
         tx.upsert(key(2, 0, i), "aborted");
      tx.abort();
   }
   stage = 2;
   reader_thread.join();
   REQUIRE_FALSE(wrong);
}

TEMPLATE_TEST_CASE("Writers: the eight portable roots are separate keyspaces",
                   "[api][writers]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());

   auto writer = db->start_write_session();
   for (std::uint32_t root = 0; root < 8; ++root)
   {
      auto tx = writer->start_transaction(root);
      for (std::uint32_t i = 0; i <= root; ++i)
         tx.upsert(key(root, 0, i), "v");
      tx.commit();
   }
   auto reader = db->start_read_session();
   for (std::uint32_t root = 0; root < 8; ++root)
   {
      auto cursor = reader->create_cursor(root);
      bool other  = false;
      REQUIRE(count_own_keys(cursor, root, other) == root + 1);
      REQUIRE_FALSE(other);
   }
}

// Large Engine extension: trees that are not installed in any root. Each
// writer edits its own copy of one shared ancestor at the same time; the
// ancestor and the other writers' copies are unaffected.
TEST_CASE("Writers: sessions edit detached trees from a shared ancestor concurrently",
          "[api][writers][detached]")
{
   using E  = psitri::big_engine;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());

   constexpr std::uint32_t writers = 4;
   constexpr std::uint32_t rounds  = 30;
   constexpr std::uint32_t batch   = 200;
   constexpr std::uint32_t common  = 2000;

   auto owner    = db->start_write_session();
   auto ancestor = [&]
   {
      auto edit = owner->start_write_transaction(owner->create_temporary_tree());
      for (std::uint32_t i = 0; i < common; ++i)
         edit.upsert(key(99, 0, i), "common");
      return owner->share(edit.get_tree());
   }();

   std::atomic<bool>        go{false}, wrong{false};
   std::vector<std::thread> threads;
   std::vector<decltype(ancestor)> results(writers);
   for (std::uint32_t w = 0; w < writers; ++w)
   {
      threads.emplace_back(
          [&, w]
          {
             auto writer = db->start_write_session();
             auto mine   = writer->adopt(ancestor);
             while (!go)
                std::this_thread::yield();
             for (std::uint32_t round = 0; round < rounds; ++round)
             {
                auto edit = writer->start_write_transaction(std::move(mine));
                for (std::uint32_t i = 0; i < batch; ++i)
                   edit.upsert(key(w, round, i), "own");
                // Replace and remove some of the shared keys as well.
                edit.upsert(key(99, 0, (round * 7 + w) % common), "changed");
                edit.remove(key(99, 0, (round * 13 + w * 101) % common));
                mine = edit.get_tree();
             }
             results[w] = writer->share(std::move(mine));
          });
   }
   go = true;
   for (auto& thread : threads)
      thread.join();

   auto reader = db->start_read_session();
   // The ancestor is exactly as it was.
   {
      auto        cursor = reader->create_cursor(ancestor);
      std::size_t n      = 0;
      for (bool more = cursor.seek_begin(); more; more = cursor.next(), ++n)
         if (cursor.template value<std::string>() != std::optional<std::string>{"common"})
            wrong = true;
      REQUIRE(n == common);
   }
   // Each result holds its writer's keys and nobody else's.
   for (std::uint32_t w = 0; w < writers; ++w)
   {
      auto        cursor = reader->create_cursor(results[w]);
      char        own[8], shared[8];
      std::snprintf(own, sizeof(own), "r%02u/", w);
      std::snprintf(shared, sizeof(shared), "r%02u/", 99u);
      std::size_t mine = 0;
      for (bool more = cursor.seek_begin(); more; more = cursor.next())
      {
         if (cursor.key().starts_with(own))
            ++mine;
         else if (!cursor.key().starts_with(shared))
            wrong = true;
      }
      REQUIRE(mine == std::size_t{rounds} * batch);
   }
   REQUIRE_FALSE(wrong);
}

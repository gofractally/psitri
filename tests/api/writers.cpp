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

#include "writer_support.hpp"

namespace
{
   template <psitri::engine E>
   typename E::config writer_config(bool small_store = false)
   {
      typename E::config cfg;
      cfg.sync_mode = psitri::sync_mode::none;
      if constexpr (std::same_as<E, psitri::small_engine>)
         cfg.max_segments = small_store ? 3 : 32;
      else
         cfg.max_pinned_cache_size_mb = 128;
      return cfg;
   }
}  // namespace

#define WRITER_ENGINES psitri::small_engine, psitri::big_engine
#include "writer_cases.inc"

TEST_CASE("Writers: sessions edit detached trees from a shared ancestor concurrently",
          "[api][writers][detached]")
{
   using E  = psitri::big_engine;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, writer_config<E>());

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

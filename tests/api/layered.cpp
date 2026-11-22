// The writer, root and reader rules on the engines built from other engines:
// the buffered write-ahead layer over the Large Engine, and the Small Engine
// tiered over the Large Engine. They are held to the same cases as the
// engines themselves.

#include <psitri/api.hpp>
#include <psitri/dwal/byte_engine.hpp>
#include <psitri/experimental/tiered.hpp>

#include "writer_support.hpp"

namespace
{
   using buffered_engine = psitri::dwal::byte_engine<psitri::big_engine>;
   using tiered_engine   = psitri::experimental::write_engine;

   template <class E>
   typename E::config writer_config(bool small_store = false)
   {
      typename E::config cfg;
      if constexpr (std::same_as<E, buffered_engine>)
      {
         cfg.base.sync_mode                = psitri::sync_mode::none;
         cfg.base.max_pinned_cache_size_mb = 128;
      }
      else
      {
         cfg.commit_sync                   = psitri::sync_mode::none;
         cfg.wcfg.sync_mode                = psitri::sync_mode::none;
         cfg.wcfg.max_segments             = small_store ? 3 : 32;
         cfg.bcfg.sync_mode                = psitri::sync_mode::none;
         cfg.bcfg.max_pinned_cache_size_mb = 128;
      }
      return cfg;
   }
}  // namespace

// The tiered engine has one root, so it runs the cases that need only root 0.
#define WRITER_ENGINES buffered_engine
#define WRITER_ROOT0_ENGINES buffered_engine, tiered_engine
#define WRITER_FREE_READER_ENGINES tiered_engine
#include "writer_cases.inc"

// KNOWN GAP. The buffered layer's transaction changes a shared in-memory layer
// in place and undoes it on abort, so a reader that wants only committed data
// waits until the transaction ends. This case records the rule it breaks and
// is expected to fail until that transaction keeps its changes aside and
// applies them at commit.
TEST_CASE("Readers: the buffered layer holds a reader up behind an open transaction",
          "[api][writers][readers][!shouldfail]")
{
   using E = buffered_engine;
   temporary_directory temp;
   auto db = E::database::open(temp.path / "db", psitri::open_mode::create_only, writer_config<E>());
   auto writer = db->start_write_session();
   auto tx     = writer->start_transaction(0);
   tx.upsert("k", "uncommitted");

   std::atomic<bool> done{false};
   std::thread       reader_thread(
       [&]
       {
          auto reader = db->start_read_session();
          auto cursor = reader->create_cursor(0);
          cursor.seek_begin();
          done = true;
       });
   const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
   while (!done && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
   const bool while_open = done;
   tx.abort();
   reader_thread.join();
   REQUIRE(while_open);
}

TEST_CASE("Tiered: roots other than the one keyspace are refused", "[api][writers][tiered]")
{
   temporary_directory temp;
   auto db = tiered_engine::database::open(temp.path / "db", psitri::open_mode::create_only,
                                           writer_config<tiered_engine>());
   auto writer = db->start_write_session();
   auto reader = db->start_read_session();
   REQUIRE_THROWS_AS(writer->start_transaction(1), std::out_of_range);
   REQUIRE_THROWS_AS(writer->start_transaction(2), std::out_of_range);
   REQUIRE_THROWS_AS(reader->create_cursor(2), std::out_of_range);
   {
      auto tx = writer->start_transaction(0);
      tx.upsert("k", "v");
      tx.commit();
   }
   auto cursor = reader->create_cursor(0);
   REQUIRE(cursor.seek_begin());
   REQUIRE(cursor.key() == "k");
}

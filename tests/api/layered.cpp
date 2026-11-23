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

#define WRITER_ENGINES buffered_engine, tiered_engine
#include "writer_cases.inc"

TEST_CASE("Tiered: each root is tiered on its own, and its base can be read alone",
          "[api][writers][tiered]")
{
   temporary_directory temp;
   auto db = tiered_engine::database::open(temp.path / "db", psitri::open_mode::create_only,
                                           writer_config<tiered_engine>());
   auto writer = db->start_write_session();
   auto reader = db->start_read_session();
   const std::uint32_t roots = 8;
   REQUIRE_THROWS_AS(writer->start_transaction(roots), std::out_of_range);
   REQUIRE_THROWS_AS(reader->create_cursor(roots), std::out_of_range);

   auto count = [](auto cursor, std::uint32_t root, bool& foreign)
   { return count_own_keys(cursor, root, foreign); };
   for (int pass = 0; pass < 3; ++pass)
   {
      for (std::uint32_t root = 0; root < roots; ++root)
      {
         auto tx = writer->start_transaction(root);
         for (std::uint32_t i = 0; i < 50; ++i)
            tx.upsert(key(root, pass, i), "v");
         tx.commit();
      }
      // Merged or not, every root shows exactly its own keys.
      for (std::uint32_t root = 0; root < roots; ++root)
      {
         bool foreign = false;
         REQUIRE(count(reader->create_cursor(root), root, foreign) == 50u * (pass + 1));
         REQUIRE_FALSE(foreign);
      }
      // Move everything written so far into the base tier.
      db->flush_now();
      for (std::uint32_t root = 0; root < roots; ++root)
      {
         bool foreign = false;
         REQUIRE(count(reader->create_cursor(root), root, foreign) == 50u * (pass + 1));
         REQUIRE(count(reader->create_base_cursor(root), root, foreign) == 50u * (pass + 1));
         REQUIRE_FALSE(foreign);
      }
   }
   // Removals in one root reach its base and leave the other roots alone.
   {
      auto tx = writer->start_transaction(3);
      for (std::uint32_t i = 0; i < 50; ++i)
         tx.remove(key(3, 0, i));
      tx.commit();
   }
   db->flush_now();
   for (std::uint32_t root = 0; root < roots; ++root)
   {
      bool foreign = false;
      REQUIRE(count(reader->create_base_cursor(root), root, foreign) == (root == 3 ? 100u : 150u));
      REQUIRE_FALSE(foreign);
   }
}

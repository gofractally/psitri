#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include <sal/allocator_session_impl.hpp>
#include <sal/mapping.hpp>
#include <sal/mapped_memory/allocator_state.hpp>

namespace fs = std::filesystem;

namespace
{
   struct tail_test_node : sal::alloc_header
   {
      static constexpr sal::header_type type_id = sal::header_type::start_user_type;
      tail_test_node(uint32_t size, sal::ptr_address_seq seq) noexcept
          : sal::alloc_header(size, type_id, seq)
      {
      }
   };

   fs::path temp_dir(std::string name)
   {
      auto path = fs::temp_directory_path() / name;
      fs::remove_all(path);
      fs::create_directories(path);
      return path;
   }
}  // namespace

TEST_CASE("segment rollover accounts for its tail exactly once", "[sal][session][tail]")
{
   bool sync_before_rollover = false;
   bool fill_segment = false;
   SECTION("rollover with unsynced data") {}
   SECTION("rollover after a page-aligned sync") { sync_before_rollover = true; }
   SECTION("rollover after natural fill already counted terminal padding")
   {
      sync_before_rollover = true;
      fill_segment = true;
   }

   const auto path = temp_dir("sal_segment_tail_accounting_test");
   {
      sal::runtime_config cfg;
      cfg.max_pinned_cache_size_mb = 128;
      // Keep segments stationary while comparing their exact free-space counts.
      cfg.compact_pinned_unused_threshold_mb = 255;
      cfg.compact_unpinned_unused_threshold_mb = 255;
      sal::allocator alloc(path, cfg, false);
      alloc.register_type_ops<tail_test_node>();
      alloc.start_background_threads();
      auto session = alloc.create_session();

      constexpr uint32_t first_size = 16 * 1024 * 1024;
      constexpr uint32_t usable_end = sal::segment_size - sal::mapped_memory::segment_footer_size;
      constexpr uint32_t fresh_capacity =
          (usable_end - sal::cacheline_size) / sal::cacheline_size * sal::cacheline_size;
      const uint32_t second_size = fill_segment ? fresh_capacity - first_size : 4 * 1024 * 1024;
      auto first = session->smart_alloc<tail_test_node>(first_size, sal::alloc_hint{});
      auto second = session->smart_alloc<tail_test_node>(second_size, sal::alloc_hint{});
      const auto segment = alloc.resolve(first.address()).second.segment();
      REQUIRE(alloc.resolve(second.address()).second.segment() == segment);

      auto segment_info = [&] {
         const auto dump = alloc.dump();
         for (const auto& info : dump.segments)
            if (info.segment_num == segment) return info;
         FAIL("allocated segment missing from allocator statistics");
         return dump.segments.front();
      };
      const uint64_t expected_free = usable_end - first_size - second_size;
      if (sync_before_rollover)
      {
         session->sync(sal::sync_type::none);
         const auto before = segment_info();
         if (fill_segment)
         {
            REQUIRE(before.hdr_read_only);
            REQUIRE(before.freed_bytes == expected_free);
         }
         else
         {
            REQUIRE_FALSE(before.hdr_read_only);
            REQUIRE(before.freed_bytes == before.alloc_pos - first_size - second_size);
         }
      }

      // This request cannot fit in the old segment. Normal prepare_alloc()
      // finalizes it; no caller adds the skipped bytes to freed_space.
      auto next = session->smart_alloc<tail_test_node>(first_size, sal::alloc_hint{});
      REQUIRE(alloc.resolve(next.address()).second.segment() != segment);
      session->sync(sal::sync_type::none);
      const auto finalized = segment_info();
      REQUIRE(finalized.is_finalized);
      REQUIRE(finalized.hdr_read_only);
      REQUIRE(finalized.is_read_only);  // published to the compactor
      REQUIRE(finalized.freed_bytes == expected_free);

      // Prior sync padding + terminal marker + trailing gap cover the unused
      // bytes exactly once. Repeated sync must not add them again.
      for (int i = 0; i < 3; ++i)
      {
         session->sync(sal::sync_type::none);
         REQUIRE(segment_info().freed_bytes == expected_free);
      }
   }
   fs::remove_all(path);
}

TEST_CASE("allocator open resets stale session bitmap", "[sal][session][recovery]")
{
   auto path = temp_dir("sal_session_bitmap_recovery_test");

   {
      sal::allocator alloc(path, sal::runtime_config{}, false);
      auto           session = alloc.create_session();
      REQUIRE(session.get() != nullptr);
   }

   {
      sal::mapping header(path / "header", sal::access_mode::read_write);
      auto* state = reinterpret_cast<sal::mapped_memory::allocator_state*>(header.data());
      for (uint32_t i = 0; i < sal::mapped_memory::session_data::session_cap; ++i)
         (void)state->_session_data.alloc_session_num();
      REQUIRE(state->_session_data.free_session_bitmap() == 0);
      header.sync(sal::sync_type::full);
   }

   {
      sal::allocator alloc(path, sal::runtime_config{}, false);
      CHECK(alloc.dump().active_sessions == 0);
      auto session = alloc.create_session();
      REQUIRE(session.get() != nullptr);
      CHECK(alloc.dump().active_sessions == 1);
   }

   fs::remove_all(path);
}

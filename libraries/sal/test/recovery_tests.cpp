#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <vector>

#include <sal/allocator_impl.hpp>
#include <sal/allocator_session_impl.hpp>
#include <sal/smart_ptr_impl.hpp>

namespace fs = std::filesystem;

namespace
{
   struct recovery_test_node : sal::alloc_header
   {
      static constexpr sal::header_type type_id = sal::header_type::start_user_type;
      uint64_t                          magic;
      recovery_test_node(uint32_t size, sal::ptr_address_seq seq, uint64_t m) noexcept
          : sal::alloc_header(size, type_id, seq), magic(m)
      {
      }
   };
}  // namespace

// gofractally/psitri#22. Recovery picks, among the copies of an address it
// finds in the segments, the one with the newest allocation sequence. So the
// sequence has to keep counting across a recovery: an object allocated after
// one must still be the newest when the next one runs.
TEST_CASE("an object allocated after a recovery survives the next recovery", "[sal][recovery]")
{
   const auto path = fs::temp_directory_path() / "sal_recover_sequence_test";
   fs::remove_all(path);
   fs::create_directories(path);
   {
      sal::runtime_config cfg;
      cfg.max_pinned_cache_size_mb             = 64;
      cfg.compact_pinned_unused_threshold_mb   = 255;
      cfg.compact_unpinned_unused_threshold_mb = 255;
      sal::allocator a(path, cfg, false);
      a.register_type_ops<recovery_test_node>();
      a.start_background_threads();
      auto ses = a.create_session();

      std::vector<sal::ptr_address> burned;
      for (int i = 0; i < 100; ++i)
         burned.push_back(
             ses->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(0x1111)));
      sal::ptr_address dead =
          ses->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(0xAAAA));
      const auto dead_sequence = ses->get_ref<recovery_test_node>(dead)->sequence();
      ses->sync(sal::sync_type::none);
      ses->release(dead);  // its bytes stay in the segment
      for (auto adr : burned)
         ses->release(adr);

      a.recover();

      // A new live object at the dead one's address.
      sal::ptr_address              live = sal::null_ptr_address;
      std::vector<sal::ptr_address> extra;
      for (int i = 0; i < 100000 && live == sal::null_ptr_address; ++i)
      {
         auto adr = ses->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint(&dead, 1),
                                                   uint64_t(0xBBBB));
         if (adr == dead)
            live = adr;
         else
            extra.push_back(adr);
      }
      REQUIRE(live != sal::null_ptr_address);
      CHECK(int32_t(uint32_t(ses->get_ref<recovery_test_node>(live)->sequence()) -
                    uint32_t(dead_sequence)) > 0);
      for (auto adr : extra)
         ses->release(adr);
      (void)ses->set_root(sal::root_object_number(0),
                          sal::smart_ptr<recovery_test_node>(ses, live), sal::sync_type::none);

      a.recover();

      auto resolved = a.resolve(dead);
      REQUIRE(resolved.first);
      CHECK(static_cast<const recovery_test_node*>(resolved.first)->magic == 0xBBBB);
      (void)ses->set_root(sal::root_object_number(0), sal::smart_ptr<recovery_test_node>(),
                          sal::sync_type::none);
   }
   fs::remove_all(path);
}

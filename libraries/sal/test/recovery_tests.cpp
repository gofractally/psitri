#include <catch2/catch_test_macros.hpp>
#include <chrono>
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

namespace
{
   sal::runtime_config recovery_test_config()
   {
      sal::runtime_config cfg;
      cfg.max_pinned_cache_size_mb             = 64;
      cfg.compact_pinned_unused_threshold_mb   = 255;
      cfg.compact_unpinned_unused_threshold_mb = 255;
      return cfg;
   }

   uint64_t root_magic(sal::allocator& a, sal::allocator_session& ses)
   {
      auto root = ses.get_root<recovery_test_node>(sal::root_object_number(0));
      if (!root)
         return 0;
      auto resolved = a.resolve(root.address());
      return resolved.first ? static_cast<const recovery_test_node*>(resolved.first)->magic : 0;
   }

   // The sync header that follows `obj` in its segment: the one its commit wrote.
   sal::sync_header* sync_header_after(const sal::alloc_header* obj)
   {
      while (obj->type() != sal::header_type::sync_head)
         obj = obj->next();
      return const_cast<sal::sync_header*>(static_cast<const sal::sync_header*>(obj));
   }
}  // namespace

TEST_CASE("commit stamps follow the wall clock and never repeat", "[sal][recovery][stamp]")
{
   const uint64_t wall = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count());
   uint64_t last = *sal::next_commit_stamp();
   // Time since boot would be smaller than the wall clock by decades.
   CHECK(last + 60'000'000 > wall);
   for (int i = 0; i < 100000; ++i)
   {
      uint64_t next = *sal::next_commit_stamp();
      REQUIRE(next > last);
      last = next;
   }
   sal::commit_stamp_at_least(sal::usec_timestamp(last + 1'000'000'000));
   CHECK(*sal::next_commit_stamp() > last + 1'000'000'000);
}

// gofractally/psitri#23. The newest commit of a root may have cleared it; an
// older commit found in the segments must not bring the root back.
TEST_CASE("power-loss recovery keeps a root cleared by its newest commit", "[sal][recovery]")
{
   const auto path = fs::temp_directory_path() / "sal_recover_cleared_root_test";
   fs::remove_all(path);
   fs::create_directories(path);
   {
      sal::allocator a(path, recovery_test_config(), false);
      a.register_type_ops<recovery_test_node>();
      a.start_background_threads();
      auto ses = a.create_session();
      auto adr =
          ses->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(0xAAAA));
      (void)ses->set_root(sal::root_object_number(0),
                          sal::smart_ptr<recovery_test_node>(ses, adr), sal::sync_type::none);
      CHECK(root_magic(a, *ses) == 0xAAAA);
      (void)ses->set_root(sal::root_object_number(0), sal::smart_ptr<recovery_test_node>(),
                          sal::sync_type::none);
   }
   {
      sal::allocator a(path, recovery_test_config(), false);
      a.register_type_ops<recovery_test_node>();
      a.recover_from_power_loss();
      auto ses = a.create_session();
      CHECK(root_magic(a, *ses) == 0);
   }
   fs::remove_all(path);
}

// gofractally/psitri#23. Commits are ordered by their stamps. An old commit
// can carry a stamp larger than the clock gives now (the clock that stamped
// it restarted, or was set back); once a recovery has seen it, every later
// commit is stamped above it and wins the next recovery.
TEST_CASE("a commit made after a recovery is newer than every commit the recovery found",
          "[sal][recovery]")
{
   const auto path = fs::temp_directory_path() / "sal_recover_stamp_order_test";
   fs::remove_all(path);
   fs::create_directories(path);
   {
      sal::allocator a(path, recovery_test_config(), false);
      a.register_type_ops<recovery_test_node>();
      a.start_background_threads();
      auto ses = a.create_session();
      auto adr =
          ses->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(0xAAAA));
      (void)ses->set_root(sal::root_object_number(0),
                          sal::smart_ptr<recovery_test_node>(ses, adr), sal::sync_type::none);
      // That commit's stamp, as a clock running far ahead of this one would have written
      // it: in its sync header, where a power loss that kept the header and lost the roots
      // file's update would leave it.
      auto* header = sync_header_after(a.resolve(adr).first);
      auto  info   = *header->get_root_info();
      info.root_version += 86'400'000'000ull * 365;
      header->set_user_data(info);
   }
   {
      sal::allocator a(path, recovery_test_config(), false);
      a.register_type_ops<recovery_test_node>();
      a.recover_from_power_loss();
      a.start_background_threads();
      auto ses = a.create_session();
      REQUIRE(root_magic(a, *ses) == 0xAAAA);
      auto adr =
          ses->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(0xBBBB));
      (void)ses->set_root(sal::root_object_number(0),
                          sal::smart_ptr<recovery_test_node>(ses, adr), sal::sync_type::none);
   }
   {
      sal::allocator a(path, recovery_test_config(), false);
      a.register_type_ops<recovery_test_node>();
      a.recover_from_power_loss();
      auto ses = a.create_session();
      CHECK(root_magic(a, *ses) == 0xBBBB);
   }
   fs::remove_all(path);
}

// A commit's stamp is wall-clock time; ages (what the compactor and the read
// cache compare) are measured on the clock that counts from boot. Converting
// one to the other must keep "how long ago", not the number.
TEST_CASE("a commit stamp converts to the age clock by how long ago it was", "[sal][recovery][stamp]")
{
   const uint64_t now_age = *sal::get_current_time_msec();
   const auto     just    = sal::next_commit_stamp();
   const uint64_t fresh   = *sal::age_time_of_commit_stamp(just);
   CHECK(fresh <= now_age + 50);
   CHECK(fresh + 2000 >= now_age);

   const uint64_t hour_ago = *sal::age_time_of_commit_stamp(
       sal::usec_timestamp(*just - 3'600'000'000ull));
   CHECK(fresh - hour_ago >= 3'599'000);
   CHECK(fresh - hour_ago <= 3'601'000);

   // A stamp from the future (a clock that was set back) is "now", not later than now.
   CHECK(*sal::age_time_of_commit_stamp(sal::usec_timestamp(*just + 3'600'000'000ull)) <=
         now_age + 50);
}

// A finalized segment is closed by a sync header that runs to the end of its
// usable space, which is 64 bytes short of a grain boundary, so that header's
// size is not a whole number of grains. Power-loss recovery that keeps only
// what was flushed must still accept it: what lies in front of it was flushed
// with it. (It was rejected, and everything between the last ordinary header
// and the closing one was dropped: after a clean close, the objects the
// compactor had moved last.)
TEST_CASE("recovery keeps what the closing sync header of a segment covers",
          "[sal][recovery][durable_prefix]")
{
   const auto path = fs::temp_directory_path() / "sal_recover_closing_header_test";
   fs::remove_all(path);
   fs::create_directories(path);
   {
      sal::allocator a(path, recovery_test_config(), false);
      a.register_type_ops<recovery_test_node>();
      a.start_background_threads();
      auto committer = a.create_session();
      {
         // Another session writes the object and never syncs: like the compactor's copy of
         // a moved object when the store is closed.
         auto other = a.create_session();
         auto first =
             other->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(0x1111));
         other->sync(sal::sync_type::none);  // an ordinary header, then more data
         auto adr =
             other->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(0xEEEE));
         (void)committer->set_root(sal::root_object_number(0),
                                   sal::smart_ptr<recovery_test_node>(committer, adr),
                                   sal::sync_type::none);
         other->release(first);
      }  // the session ends: its segment is finalized and closed with a header
   }
   {
      sal::allocator             a(path, recovery_test_config(), false);
      sal::allocator::recovery_hooks hooks;
      hooks.truncate_unsynced_tails = true;
      hooks.fail_on_unresolved      = true;
      hooks.restart_threads         = false;
      a.set_recovery_hooks(hooks);
      a.register_type_ops<recovery_test_node>();
      REQUIRE_NOTHROW(a.recover_from_power_loss());
      auto ses = a.create_session();
      CHECK(root_magic(a, *ses) == 0xEEEE);
   }
   fs::remove_all(path);
}

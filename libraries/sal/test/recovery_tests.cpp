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

// The commits of a root are numbered from its own record: each commit is one
// more than the last, whatever the clock does and whatever other roots do.
TEST_CASE("each root numbers its own commits", "[sal][recovery][commit_number]")
{
   const auto path = fs::temp_directory_path() / "sal_commit_number_test";
   fs::remove_all(path);
   fs::create_directories(path);
   {
      sal::allocator a(path, recovery_test_config(), false);
      a.register_type_ops<recovery_test_node>();
      a.start_background_threads();
      auto ses    = a.create_session();
      auto commit = [&](uint32_t root)
      {
         auto adr =
             ses->alloc<recovery_test_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(root));
         (void)ses->set_root(sal::root_object_number(root),
                             sal::smart_ptr<recovery_test_node>(ses, adr), sal::sync_type::none);
      };
      CHECK(ses->next_root_commit(sal::root_object_number(0)) == 1);
      commit(0);
      commit(0);
      commit(0);
      CHECK(ses->next_root_commit(sal::root_object_number(0)) == 4);
      CHECK(ses->next_root_commit(sal::root_object_number(1)) == 1);
      commit(1);
      CHECK(ses->next_root_commit(sal::root_object_number(1)) == 2);
      CHECK(ses->next_root_commit(sal::root_object_number(0)) == 4);
      // Clearing a root is a commit of it too.
      (void)ses->set_root(sal::root_object_number(1), sal::smart_ptr<recovery_test_node>(),
                          sal::sync_type::none);
      CHECK(ses->next_root_commit(sal::root_object_number(1)) == 3);
   }
   {
      // The numbers are in the roots file: they go on after a reopen.
      sal::allocator a(path, recovery_test_config(), false);
      a.register_type_ops<recovery_test_node>();
      auto ses = a.create_session();
      CHECK(ses->next_root_commit(sal::root_object_number(0)) == 4);
      CHECK(ses->next_root_commit(sal::root_object_number(1)) == 3);
   }
   fs::remove_all(path);
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

// gofractally/psitri#23. A sync header can carry a commit number above the
// roots file's (its commit was flushed and the file's update was not). Once a
// recovery has taken that header, the root's record carries its number, so
// every later commit of the root is numbered above it and wins the next
// recovery.
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
      // That commit's number as a later commit's would be, in its sync header: what a
      // power loss that kept a later header and lost the roots file's update leaves.
      auto* header = sync_header_after(a.resolve(adr).first);
      auto  info   = *header->get_root_info();
      info.root_version += 1000;
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

#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <filesystem>
#include <vector>

#include <sal/allocator_impl.hpp>
#include <sal/allocator_session_impl.hpp>

namespace fs = std::filesystem;

namespace
{
   struct release_test_node : sal::alloc_header
   {
      static constexpr sal::header_type type_id = sal::header_type::start_user_type;
      release_test_node(uint32_t size, sal::ptr_address_seq seq) noexcept
          : sal::alloc_header(size, type_id, seq)
      {
      }
   };

   std::atomic<uint32_t> probed_deferred{0};
   std::atomic<uint32_t> probed_inline{0};
   std::atomic<uint32_t> probed_at_zero{0};

   void probe(const sal::control_block& cb, bool deferred) noexcept
   {
      (deferred ? probed_deferred : probed_inline).fetch_add(1, std::memory_order_relaxed);
      if (cb.ref() == 0)
         probed_at_zero.fetch_add(1, std::memory_order_relaxed);
   }

   struct probe_scope
   {
      probe_scope()
      {
         probed_deferred = probed_inline = probed_at_zero = 0;
         sal::detail::release_last_reference_probe.store(&probe);
      }
      ~probe_scope() { sal::detail::release_last_reference_probe.store(nullptr); }
   };
}  // namespace

// gofractally/psitri#20. The compactor treats a reference count of zero as
// "already freed": it leaves such an object behind when it empties a segment,
// and the segment can then be reused. So an object that is still going to be
// destroyed must never be observable at zero before its bytes are protected:
// not while its last reference is on its way to the release queue, and not
// before the destroying thread holds its read lock.
TEST_CASE("the last reference is never dropped to zero before the object is protected",
          "[sal][release]")
{
   const auto path = fs::temp_directory_path() / "sal_release_not_through_zero_test";
   fs::remove_all(path);
   fs::create_directories(path);
   {
      sal::runtime_config cfg;
      cfg.max_pinned_cache_size_mb             = 0;
      cfg.compact_pinned_unused_threshold_mb   = 255;
      cfg.compact_unpinned_unused_threshold_mb = 255;
      sal::allocator a(path, cfg, false);
      a.register_type_ops<release_test_node>();
      a.start_background_threads();

      auto                          ses = a.create_session();
      std::vector<sal::ptr_address> objects;
      for (int i = 0; i < 200; ++i)
         objects.push_back(ses->alloc<release_test_node>(uint32_t(128), sal::alloc_hint{}));
      ses->sync(sal::sync_type::none);

      probe_scope scope;

      SECTION("handed to the release thread (what every read session does)")
      {
         ses->set_defer_destroy(true);
         for (auto adr : objects)
            ses->release(adr);
         CHECK(probed_deferred.load() == objects.size());
         CHECK(probed_at_zero.load() == 0);
      }
      SECTION("destroyed inline")
      {
         for (auto adr : objects)
            ses->release(adr);
         CHECK(probed_inline.load() == objects.size());
         CHECK(probed_at_zero.load() == 0);
      }
      SECTION("a shared object is only decremented")
      {
         for (auto adr : objects)
            ses->retain(adr);
         for (auto adr : objects)
            ses->release(adr);
         CHECK(probed_inline.load() + probed_deferred.load() == 0);
         for (auto adr : objects)
            CHECK(ses->get_ref<release_test_node>(adr).ref() == 1);
         for (auto adr : objects)
            ses->release(adr);
         CHECK(probed_at_zero.load() == 0);
      }
   }
   fs::remove_all(path);
}

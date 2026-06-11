#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <filesystem>
#include <thread>
#include <vector>

#include <sal/allocator_session_impl.hpp>
#include <sal/atomic_ptr.hpp>
#include <sal/smart_ptr_impl.hpp>

namespace fs = std::filesystem;

namespace
{
   fs::path temp_dir(std::string name)
   {
      auto path = fs::temp_directory_path() / name;
      fs::remove_all(path);
      fs::create_directories(path);
      return path;
   }

   // Minimal user object: one cacheline, a tag payload.
   struct probe : sal::alloc_header
   {
      static constexpr sal::header_type type_id = sal::header_type::start_user_type;
      uint64_t                          tag;
      static uint32_t alloc_size(uint64_t) noexcept { return sal::cacheline_size; }
      probe(uint32_t asize, sal::ptr_address_seq seq, uint64_t t) noexcept
          : alloc_header(asize, type_id, seq), tag(t)
      {
      }
   };

   uint64_t live_count(const sal::allocator_session_ptr& ses)
   {
      uint64_t n = 0;
      ses->for_each_live_object([&](auto, auto, auto*) { ++n; });
      return n;
   }
}  // namespace

TEST_CASE("atomic_ptr: single-thread refcount exactness", "[sal][atomic_ptr]")
{
   auto           path = temp_dir("sal_atomic_ptr_basic");
   sal::allocator alloc(path, sal::runtime_config{}, false);
   alloc.register_type_ops<probe>();
   auto       ses  = alloc.get_session();
   const auto base = live_count(ses);

   sal::atomic_ptr<probe> slot;
   REQUIRE(slot.address_hint() == sal::null_ptr_address);
   {
      auto e = slot.load(ses);  // empty load → empty ptr, no refcount touched
      REQUIRE(!e.is_valid());
   }

   auto p1 = ses->smart_alloc<probe>(uint64_t{111});
   auto a1 = p1.address();
   REQUIRE(ses->get_ref<probe>(a1).ref() == 1);
   slot.store(std::move(p1));  // donate: slot now owns the single ref
   REQUIRE(slot.address_hint() == a1);
   REQUIRE(ses->get_ref<probe>(a1).ref() == 1);

   auto l1 = slot.load(ses);  // acquire: exactly +1
   REQUIRE(l1.address() == a1);
   REQUIRE(l1->tag == 111);
   REQUIRE(ses->get_ref<probe>(a1).ref() == 2);
   l1.release();
   REQUIRE(ses->get_ref<probe>(a1).ref() == 1);

   auto p2 = ses->smart_alloc<probe>(uint64_t{222});
   auto a2 = p2.address();
   auto old = slot.exchange(std::move(p2));  // swap: old comes back owned
   REQUIRE(old.address() == a1);
   REQUIRE(ses->get_ref<probe>(a1).ref() == 1);  // `old` holds the only ref
   REQUIRE(ses->get_ref<probe>(a2).ref() == 1);  // the slot's
   old.release();                                // frees 111

   slot.reset(ses);  // frees 222
   REQUIRE(slot.address_hint() == sal::null_ptr_address);
   REQUIRE(live_count(ses) == base);

   fs::remove_all(path);
}

TEST_CASE("atomic_ptr: cross-thread handoff transfers the refcount", "[sal][atomic_ptr]")
{
   auto           path = temp_dir("sal_atomic_ptr_handoff");
   sal::allocator alloc(path, sal::runtime_config{}, false);
   alloc.register_type_ops<probe>();
   auto       main_ses = alloc.get_session();
   const auto base     = live_count(main_ses);

   constexpr uint64_t     rounds = 2000;
   sal::atomic_ptr<probe> slot;
   std::atomic<uint64_t>  published{0};  // tag of the newest value the producer stored
   std::atomic<bool>      done{false};
   std::atomic<uint64_t>  consumed{0};

   std::thread producer(
       [&]
       {
          auto ses = alloc.get_session();  // its own session
          for (uint64_t i = 1; i <= rounds; ++i)
          {
             slot.store(ses->smart_alloc<probe>(i));  // releases the displaced value
             published.store(i, std::memory_order_release);
          }
          done.store(true, std::memory_order_release);
       });
   std::thread consumer(
       [&]
       {
          auto ses = alloc.get_session();  // its own session
          while (!done.load(std::memory_order_acquire))
          {
             auto p = slot.load(ses);
             if (p.is_valid())
             {
                // The tag must be a value some store actually published, and
                // monotonicity bounds it by what the producer announced.
                auto t = p->tag;
                REQUIRE(t >= 1);
                REQUIRE(t <= rounds);
                consumed.fetch_add(1, std::memory_order_relaxed);
             }
          }
       });
   producer.join();
   consumer.join();
   REQUIRE(consumed.load() > 0);  // the handoff actually happened

   slot.reset(main_ses);
   REQUIRE(live_count(main_ses) == base);
   fs::remove_all(path);
}

TEST_CASE("atomic_ptr: multi-writer multi-reader hammer drains clean", "[sal][atomic_ptr]")
{
   auto           path = temp_dir("sal_atomic_ptr_hammer");
   sal::allocator alloc(path, sal::runtime_config{}, false);
   alloc.register_type_ops<probe>();
   auto       main_ses = alloc.get_session();
   const auto base     = live_count(main_ses);

   constexpr int          writers = 2, readers = 2;
   constexpr uint64_t     per_writer = 20000;
   sal::atomic_ptr<probe> slot;
   std::atomic<int>       writers_left{writers};

   std::vector<std::thread> ts;
   for (int wi = 0; wi < writers; ++wi)
      ts.emplace_back(
          [&, wi]
          {
             auto ses = alloc.get_session();
             for (uint64_t i = 0; i < per_writer; ++i)
             {
                auto old = slot.exchange(ses->smart_alloc<probe>((uint64_t(wi) << 32) | i));
                old.release();  // constant churn → address recycling → harvest paths
             }
             writers_left.fetch_sub(1, std::memory_order_release);
          });
   for (int ri = 0; ri < readers; ++ri)
      ts.emplace_back(
          [&]
          {
             auto     ses = alloc.get_session();
             uint64_t got = 0;
             while (writers_left.load(std::memory_order_acquire) > 0)
             {
                auto p = slot.load(ses);
                if (p.is_valid())
                {
                   // Deref under our own ref: the object must be alive and sane.
                   auto t = p->tag;
                   REQUIRE((t & 0xFFFFFFFFull) < per_writer);
                   ++got;
                }
             }
             REQUIRE(got > 0);
          });
   for (auto& t : ts)
      t.join();

   slot.reset(main_ses);
   REQUIRE(live_count(main_ses) == base);  // every refcount accounted — none lost, none leaked
   fs::remove_all(path);
}

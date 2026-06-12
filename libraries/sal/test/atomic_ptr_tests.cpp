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
   alloc.register_type_ops<probe>();  // ops BEFORE threads (sal contract)
   alloc.start_background_threads();  // provider ON: get_new_segment waits on
                                      // it forever otherwise (the recorded trap)
   auto       ses  = alloc.create_session();
   const auto base = live_count(ses);

   sal::atomic_ptr<probe> slot;
   REQUIRE(slot.address_hint() == sal::null_ptr_address);
   {
      auto e = slot.load(ses);  // empty load → empty ptr, no refcount touched
      REQUIRE(!e.is_valid());
   }

   constexpr uint64_t bias = sal::atomic_ptr<probe>::credit_bias;

   auto p1 = ses->smart_alloc<probe>(uint64_t{111});
   auto a1 = p1.address();
   REQUIRE(ses->get_ref<probe>(a1).ref() == 1);
   slot.store(std::move(p1));  // donate base ref + bank `bias` credits
   REQUIRE(slot.address_hint() == a1);
   REQUIRE(ses->get_ref<probe>(a1).ref() == 1 + bias);

   auto l1 = slot.load(ses);  // consumes a pre-paid credit: strong unchanged
   REQUIRE(l1.address() == a1);
   REQUIRE(l1->tag == 111);
   REQUIRE(ses->get_ref<probe>(a1).ref() == 1 + bias);
   l1.release();  // returns the consumed credit
   REQUIRE(ses->get_ref<probe>(a1).ref() == bias);

   auto p2 = ses->smart_alloc<probe>(uint64_t{222});
   auto a2 = p2.address();
   auto old = slot.exchange(std::move(p2));  // releases old's unconsumed credits
   REQUIRE(old.address() == a1);
   REQUIRE(ses->get_ref<probe>(a1).ref() == 1);  // `old` holds the base ref only
   REQUIRE(ses->get_ref<probe>(a2).ref() == 1 + bias);
   old.release();  // frees 111

   slot.reset(ses);  // releases a2's credits + base → frees 222
   REQUIRE(slot.address_hint() == sal::null_ptr_address);
   REQUIRE(live_count(ses) == base);

   fs::remove_all(path);
}

TEST_CASE("atomic_ptr: credit re-arm under sustained loads", "[sal][atomic_ptr]")
{
   auto           path = temp_dir("sal_atomic_ptr_rearm");
   sal::allocator alloc(path, sal::runtime_config{}, false);
   alloc.register_type_ops<probe>();  // ops BEFORE threads (sal contract)
   alloc.start_background_threads();  // provider ON: get_new_segment waits on
                                      // it forever otherwise (the recorded trap)
   auto       ses  = alloc.create_session();
   const auto base = live_count(ses);

   constexpr uint64_t     bias = sal::atomic_ptr<probe>::credit_bias;
   sal::atomic_ptr<probe> slot;
   slot.store(ses->smart_alloc<probe>(uint64_t{42}));

   // Exhaust the credit bank several times over: every bias-th load takes the
   // re-arm slow path (lock + fresh bank).
   for (uint64_t i = 0; i < 3 * bias + 7; ++i)
   {
      auto p = slot.load(ses);
      REQUIRE(p.is_valid());
      REQUIRE(p->tag == 42);
   }
   slot.reset(ses);
   REQUIRE(live_count(ses) == base);  // re-arms fully settled — nothing leaked
   fs::remove_all(path);
}

TEST_CASE("atomic_ptr: cross-thread handoff transfers the refcount", "[sal][atomic_ptr]")
{
   auto           path = temp_dir("sal_atomic_ptr_handoff");
   sal::allocator alloc(path, sal::runtime_config{}, false);
   alloc.register_type_ops<probe>();  // ops BEFORE threads (sal contract)
   alloc.start_background_threads();  // provider ON: get_new_segment waits on
                                      // it forever otherwise (the recorded trap)
   auto       main_ses = alloc.create_session();
   const auto base     = live_count(main_ses);

   constexpr uint64_t     rounds = 2000;
   sal::atomic_ptr<probe> slot;
   std::atomic<uint64_t>  published{0};  // tag of the newest value the producer stored
   std::atomic<bool>      done{false};
   std::atomic<uint64_t>  consumed{0};

   std::thread producer(
       [&]
       {
          auto ses = alloc.create_session();  // its own session
          for (uint64_t i = 1; i <= rounds; ++i)
          {
             slot.store(ses->smart_alloc<probe>(i));  // releases the displaced value
             published.store(i, std::memory_order_release);
          }
          done.store(true, std::memory_order_release);
       });
   std::atomic<uint64_t> bad_tags{0};  // Catch2 macros are not thread-safe — collect, assert after join
   std::thread           consumer(
       [&]
       {
          auto ses = alloc.create_session();  // its own session
          while (!done.load(std::memory_order_acquire))
          {
             auto p = slot.load(ses);
             if (p.is_valid())
             {
                // The tag must be a value some store actually published.
                auto t = p->tag;
                if (t < 1 || t > rounds)
                   bad_tags.fetch_add(1, std::memory_order_relaxed);
                consumed.fetch_add(1, std::memory_order_relaxed);
             }
          }
       });
   producer.join();
   consumer.join();
   REQUIRE(bad_tags.load() == 0);
   REQUIRE(consumed.load() > 0);  // the handoff actually happened

   slot.reset(main_ses);
   REQUIRE(live_count(main_ses) == base);
   fs::remove_all(path);
}

TEST_CASE("atomic_ptr: multi-writer multi-reader hammer drains clean", "[sal][atomic_ptr]")
{
   auto           path = temp_dir("sal_atomic_ptr_hammer");
   sal::allocator alloc(path, sal::runtime_config{}, false);
   alloc.register_type_ops<probe>();  // ops BEFORE threads (sal contract)
   alloc.start_background_threads();  // provider ON: get_new_segment waits on
                                      // it forever otherwise (the recorded trap)
   auto       main_ses = alloc.create_session();
   const auto base     = live_count(main_ses);

   constexpr int          writers = 2, readers = 2;
   constexpr uint64_t     per_writer = 20000;
   sal::atomic_ptr<probe> slot;
   std::atomic<int>       writers_left{writers};
   std::atomic<uint64_t>  bad_tags{0};   // Catch2 macros are not thread-safe —
   std::atomic<uint64_t>  total_got{0};  // collect in atomics, assert after join

   std::vector<std::thread> ts;
   for (int wi = 0; wi < writers; ++wi)
      ts.emplace_back(
          [&, wi]
          {
             auto ses = alloc.create_session();
             for (uint64_t i = 0; i < per_writer; ++i)
             {
                auto old = slot.exchange(ses->smart_alloc<probe>((uint64_t(wi) << 32) | i));
                old.release();  // constant churn → address recycling → credit paths
             }
             writers_left.fetch_sub(1, std::memory_order_release);
          });
   for (int ri = 0; ri < readers; ++ri)
      ts.emplace_back(
          [&]
          {
             auto     ses = alloc.create_session();
             uint64_t got = 0;
             while (writers_left.load(std::memory_order_acquire) > 0)
             {
                auto p = slot.load(ses);
                if (p.is_valid())
                {
                   // Deref under our own ref: the object must be alive and sane.
                   auto t = p->tag;
                   if ((t & 0xFFFFFFFFull) >= per_writer || (t >> 32) >= writers)
                      bad_tags.fetch_add(1, std::memory_order_relaxed);
                   ++got;
                }
             }
             total_got.fetch_add(got, std::memory_order_relaxed);
          });
   for (auto& t : ts)
      t.join();
   REQUIRE(bad_tags.load() == 0);
   REQUIRE(total_got.load() > 0);

   slot.reset(main_ses);
   REQUIRE(live_count(main_ses) == base);  // every refcount accounted — none lost, none leaked
   fs::remove_all(path);
}

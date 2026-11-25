// gofractally/psitri#40. A reader acquires a root without a lock: it names the
// root in its session's slot, looks again, and retains. A commit never waits
// for it: if it displaces a root some slot names, it keeps that root alive
// until the slot lets go.
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

#include <sal/allocator_session_impl.hpp>
#include <sal/smart_ptr_impl.hpp>
#include <sal/transaction.hpp>

namespace fs = std::filesystem;

namespace
{
   fs::path root_test_dir(const char* name)
   {
      auto path = fs::temp_directory_path() / (std::string(name) + "_" + std::to_string(::getpid()));
      fs::remove_all(path);
      fs::create_directories(path);
      return path;
   }

   struct root_node : sal::alloc_header
   {
      static constexpr sal::header_type type_id = sal::header_type::start_user_type;
      uint64_t                          tag;
      uint64_t                          not_tag;
      static uint32_t alloc_size(uint64_t) noexcept { return sal::cacheline_size; }
      root_node(uint32_t asize, sal::ptr_address_seq seq, uint64_t t) noexcept
          : alloc_header(asize, type_id, seq), tag(t), not_tag(~t)
      {
      }
   };

   uint64_t live_count(const sal::allocator_session_ptr& ses)
   {
      uint64_t n = 0;
      ses->for_each_live_object([&](auto, auto, auto*) { ++n; });
      return n;
   }

   void settle(sal::allocator& a)
   {
      for (int i = 0; i < 2000 && a.total_pending_releases() != 0; ++i)
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
   }

   // The reader stops here, between "the root is unchanged" and its retain.
   std::atomic<int>              reader_stage{0};  // 1: reader is waiting, 2: go on
   std::atomic<sal::ptr_address> reader_about_to_retain{sal::null_ptr_address};
   void hold_reader(sal::ptr_address root) noexcept
   {
      if (reader_stage.load() != 0)
         return;  // only the first acquisition is held
      reader_about_to_retain.store(root);
      reader_stage.store(1);
      while (reader_stage.load() != 2)
         std::this_thread::yield();
   }
}  // namespace

TEST_CASE("a root displaced while a reader is about to retain it stays alive for that reader",
          "[sal][root][acquire]")
{
   auto           path = root_test_dir("sal_root_acquire_held");
   {
      sal::allocator alloc(path, sal::runtime_config{}, false);
      alloc.register_type_ops<root_node>();
      alloc.start_background_threads();
      const sal::root_object_number root(5);
      auto                          writer = alloc.create_session();
      const auto                    base   = live_count(writer);

      (void)writer->set_root(root, writer->smart_alloc<root_node>(uint64_t(1)),
                             sal::sync_type::none);

      reader_stage = 0;
      sal::detail::root_acquire_probe.store(&hold_reader);
      uint64_t    seen_tag = 0, seen_not = 0;
      std::thread reader(
          [&]
          {
             auto ses  = alloc.create_session();
             auto rl   = ses->lock();
             auto held = ses->get_root<root_node>(root);
             seen_tag  = held->tag;
             seen_not  = held->not_tag;
          });
      while (reader_stage.load() != 1)
         std::this_thread::yield();
      sal::detail::root_acquire_probe.store(nullptr);

      // The reader has named root 1 and not yet retained it. Commit twice and drop
      // everything the commits hand back; without the reader the first root would be
      // destroyed here.
      const auto first = reader_about_to_retain.load();
      {
         auto old = writer->set_root(root, writer->smart_alloc<root_node>(uint64_t(2)),
                                     sal::sync_type::none);
         REQUIRE(old.address() == first);
      }
      (void)writer->set_root(root, writer->smart_alloc<root_node>(uint64_t(3)),
                             sal::sync_type::none);
      settle(alloc);
      // Still there: the table's root, and the first root kept for the reader.
      CHECK(live_count(writer) == base + 2);

      reader_stage.store(2);
      reader.join();
      CHECK(seen_tag == 1);
      CHECK(seen_not == ~uint64_t(1));

      // The reader has let go; the next commit of the root (or the release thread)
      // releases what was kept for it.
      (void)writer->set_root(root, sal::smart_ptr<root_node>(), sal::sync_type::none);
      settle(alloc);
      CHECK(live_count(writer) == base);
   }
   fs::remove_all(path);
}

TEST_CASE("readers acquire a root that commits keep replacing", "[sal][root][acquire][stress]")
{
   auto path = root_test_dir("sal_root_acquire_stress");
   {
      sal::allocator alloc(path, sal::runtime_config{}, false);
      alloc.register_type_ops<root_node>();
      alloc.start_background_threads();
      const sal::root_object_number root(7);
      auto                          writer = alloc.create_session();
      const auto                    base   = live_count(writer);

      std::atomic<bool>     stop{false};
      std::atomic<uint64_t> published{1}, acquired{0}, wrong{0};
      (void)writer->set_root(root, writer->smart_alloc<root_node>(uint64_t(1)),
                             sal::sync_type::none);
      std::vector<std::thread> readers;
      for (int r = 0; r < 4; ++r)
         readers.emplace_back(
             [&]
             {
                // A session per burst as well, so slots are given out again while
                // commits run.
                while (!stop.load(std::memory_order_relaxed))
                {
                   auto ses = alloc.create_session();
                   for (int i = 0; i < 2000 && !stop.load(std::memory_order_relaxed); ++i)
                   {
                      const uint64_t before = published.load(std::memory_order_acquire);
                      // The read lock keeps the object's bytes where they are read from:
                      // the compactor moves objects and recycles their old segments.
                      auto           rl     = ses->lock();
                      auto           held   = ses->get_root<root_node>(root);
                      const uint64_t after  = published.load(std::memory_order_acquire);
                      // An intact object, and a root that was the root at some time
                      // during the call (the writer publishes the tag after the commit,
                      // so one ahead is possible).
                      if (!held.is_valid() || held->not_tag != ~held->tag ||
                          held->tag + 1 < before || held->tag > after + 1)
                      {
                         if (wrong.fetch_add(1, std::memory_order_relaxed) < 5)
                            std::fprintf(stderr, "WRONG valid=%d tag=%llu not=%llx before=%llu after=%llu\n",
                                         int(held.is_valid()),
                                         (unsigned long long)(held.is_valid() ? held->tag : 0),
                                         (unsigned long long)(held.is_valid() ? ~held->not_tag : 0),
                                         (unsigned long long)before, (unsigned long long)after);
                      }
                      acquired.fetch_add(1, std::memory_order_relaxed);
                   }
                }
             });
      const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
      uint64_t   n   = 1;
      while (std::chrono::steady_clock::now() < end)
      {
         ++n;
         (void)writer->set_root(root, writer->smart_alloc<root_node>(n), sal::sync_type::none);
         published.store(n, std::memory_order_release);
      }
      stop = true;
      for (auto& t : readers)
         t.join();
      CHECK(wrong.load() == 0);
      CHECK(acquired.load() > 0);
      CHECK(n > 100);

      (void)writer->set_root(root, sal::smart_ptr<root_node>(), sal::sync_type::none);
      settle(alloc);
      CHECK(live_count(writer) == base);  // nothing kept for a reader is left behind
   }
   fs::remove_all(path);
}

// A transaction is finished by whichever thread the caller has it on at that moment; the
// root's gate has no owner. It still lets one transaction at a time work on a root.
TEST_CASE("a transaction started on one thread is committed on another, and the next one waits",
          "[sal][root][transaction]")
{
   auto path = root_test_dir("sal_root_transaction_gate");
   {
      sal::allocator alloc(path, sal::runtime_config{}, false);
      alloc.register_type_ops<root_node>();
      alloc.start_background_threads();
      const sal::root_object_number root(11);
      auto                          first_ses  = alloc.create_session();
      auto                          second_ses = alloc.create_session();
      const auto                    base       = live_count(first_ses);

      for (int round = 0; round < 50; ++round)
      {
         auto tx    = first_ses->start_transaction(root);  // this thread
         tx->root() = sal::smart_ptr<sal::alloc_header>(
             first_ses, first_ses->smart_alloc<root_node>(uint64_t(round)).take());

         std::atomic<int> second_state{0};  // 1: asked for the root, 2: has it
         std::thread      second(
             [&]
             {
                second_state.store(1);
                auto tx2 = second_ses->start_transaction(root);  // waits for the first
                second_state.store(2);
                tx2->abort();
             });
         while (second_state.load() == 0)
            std::this_thread::yield();
         std::this_thread::sleep_for(std::chrono::microseconds(200));
         const int seen_while_open = second_state.load();

         // Finish the first transaction on a third thread. The session is handed over,
         // not shared: this thread does not touch it until the join.
         std::thread finisher([&] { (void)tx->commit(sal::sync_type::none); });
         finisher.join();
         second.join();
         REQUIRE(seen_while_open == 1);  // the second did not get in while the first was open
         REQUIRE(second_state.load() == 2);
      }
      (void)first_ses->set_root(root, sal::smart_ptr<root_node>(), sal::sync_type::none);
      settle(alloc);
      CHECK(live_count(first_ses) == base);
   }
   fs::remove_all(path);
}

// Hidden ([.bench]): how acquiring a root scales with readers while one writer commits.
// Uses only the public session API, so the same file builds against older trees.
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

#include <sal/allocator_session_impl.hpp>
#include <sal/smart_ptr_impl.hpp>

namespace
{
   struct bench_node : sal::alloc_header
   {
      static constexpr sal::header_type type_id = sal::header_type::start_user_type;
      uint64_t                          tag;
      static uint32_t alloc_size(uint64_t) noexcept { return sal::cacheline_size; }
      bench_node(uint32_t asize, sal::ptr_address_seq seq, uint64_t t) noexcept
          : alloc_header(asize, type_id, seq), tag(t)
      {
      }
   };
}  // namespace

TEST_CASE("root acquisition: readers against one committing writer", "[.bench][root]")
{
   namespace fs = std::filesystem;
   for (int with_writer = 0; with_writer <= 1; ++with_writer)
      for (int readers : {1, 2, 4, 8})
      {
         auto path = fs::temp_directory_path() /
                     ("sal_root_bench_" + std::to_string(::getpid()));
         fs::remove_all(path);
         fs::create_directories(path);
         {
            sal::allocator alloc(path, sal::runtime_config{}, false);
            alloc.register_type_ops<bench_node>();
            alloc.start_background_threads();
            const sal::root_object_number root(9);
            auto                          writer = alloc.create_session();
            (void)writer->set_root(root, writer->smart_alloc<bench_node>(uint64_t(1)),
                                   sal::sync_type::none);
            std::atomic<bool>        go{false}, stop{false};
            std::atomic<uint64_t>    acquired{0};
            std::vector<std::thread> threads;
            for (int r = 0; r < readers; ++r)
               threads.emplace_back(
                   [&]
                   {
                      auto     ses = alloc.create_session();
                      uint64_t n   = 0;
                      while (!go.load(std::memory_order_acquire))
                         std::this_thread::yield();
                      while (!stop.load(std::memory_order_relaxed))
                      {
                         auto held = ses->get_root<bench_node>(root);
                         ++n;
                      }
                      acquired.fetch_add(n);
                   });
            uint64_t   commits = 0;
            const auto t0      = std::chrono::steady_clock::now();
            go                 = true;
            const auto end     = t0 + std::chrono::milliseconds(1000);
            while (std::chrono::steady_clock::now() < end)
            {
               if (with_writer)
               {
                  (void)writer->set_root(root, writer->smart_alloc<bench_node>(++commits),
                                         sal::sync_type::none);
               }
               else
                  std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            stop = true;
            for (auto& t : threads)
               t.join();
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("ROOTBENCH writer=%d readers=%d acquisitions/s=%.0f commits/s=%.0f\n",
                        with_writer, readers, double(acquired.load()) / secs,
                        double(commits) / secs);
            (void)writer->set_root(root, sal::smart_ptr<bench_node>(), sal::sync_type::none);
         }
         fs::remove_all(path);
      }
}

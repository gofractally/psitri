#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include <sal/allocator_impl.hpp>
#include <sal/allocator_session_impl.hpp>
#include <sal/control_block_alloc.hpp>
#include <sal/smart_ptr_impl.hpp>
#include <sal/transaction.hpp>

namespace fs = std::filesystem;

namespace
{
   struct fail_stop_node : sal::alloc_header
   {
      static constexpr sal::header_type type_id = sal::header_type::start_user_type;
      fail_stop_node(uint32_t size, sal::ptr_address_seq seq) noexcept
          : sal::alloc_header(size, type_id, seq)
      {
      }
   };

   fs::path fresh_dir(const char* name)
   {
      auto path = fs::temp_directory_path() / name;
      fs::remove_all(path);
      fs::create_directories(path);
      return path;
   }

   // Runs `body` in a child process and returns how it ended: 0 for a normal
   // return, otherwise the signal that stopped it.
   template <typename F>
   int signal_from_child(F&& body)
   {
      std::fflush(nullptr);
      pid_t pid = ::fork();
      REQUIRE(pid >= 0);
      if (pid == 0)
      {
         // The test framework's handlers would report the child's signal as a failure.
         for (int s : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE})
            std::signal(s, SIG_DFL);
         if (!std::freopen("/dev/null", "w", stderr))
            ::_exit(3);
         body();
         ::_exit(0);
      }
      int status = 0;
      REQUIRE(::waitpid(pid, &status, 0) == pid);
      if (WIFSIGNALED(status))
         return WTERMSIG(status);
      REQUIRE(WIFEXITED(status));
      REQUIRE(WEXITSTATUS(status) == 0);
      return 0;
   }
}  // namespace

// A reference taken to an object whose count is already zero cannot be made
// right afterwards: the process stops there, in every build.
TEST_CASE("retaining a released object stops the process", "[sal][fail_stop]")
{
   const auto path = fresh_dir("sal_retain_dead_test");
   int        sig  = signal_from_child(
       [&]
       {
          sal::runtime_config cfg;
          cfg.max_pinned_cache_size_mb = 0;
          sal::allocator a(path, cfg, false);
          a.register_type_ops<fail_stop_node>();
          a.start_background_threads();
          auto ses  = a.create_session();
          auto adr  = ses->alloc<fail_stop_node>(uint32_t(128), sal::alloc_hint{});
          auto keep = ses->alloc<fail_stop_node>(uint32_t(128), sal::alloc_hint{});
          (void)keep;
          ses->release(adr);
          ses->retain(adr);
       });
   CHECK(sig == SIGABRT);
   fs::remove_all(path);
}

// Freeing an address twice must not touch the neighbouring address's bit in
// the free map; it stops the process.
TEST_CASE("freeing a control block address twice stops the process", "[sal][fail_stop]")
{
   const auto path = fresh_dir("sal_double_free_test");
   int        sig  = signal_from_child(
       [&]
       {
          sal::control_block_alloc cba(path);
          auto                     a = cba.alloc();
          auto                     b = cba.alloc();
          (void)b;
          cba.get(a.addr_seq.address).store(sal::control_block_data(), std::memory_order_relaxed);
          cba.free(a.addr_seq.address);
          cba.free(a.addr_seq.address);
       });
   CHECK(sig == SIGABRT);
   fs::remove_all(path);
}

// allocator::get loads a root and then retains it, under the root's shared
// lock. A commit that swaps the root has to exclude it, or the old root can be
// released between the two steps and the reader retains a dead object.
TEST_CASE("readers of a root and transactions committing it do not race", "[sal][root][stress]")
{
   const auto path = fresh_dir("sal_root_commit_race_test");
   int        sig  = signal_from_child(
       [&]
       {
          sal::runtime_config cfg;
          cfg.max_pinned_cache_size_mb = 0;
          sal::allocator a(path, cfg, false);
          a.register_type_ops<fail_stop_node>();
          a.start_background_threads();
          const sal::root_object_number root(3);
          std::atomic<bool>             stop{false};
          std::vector<std::thread>      readers;
          for (int r = 0; r < 3; ++r)
             readers.emplace_back(
                 [&]
                 {
                    auto ses = a.create_session();
                    while (!stop.load(std::memory_order_relaxed))
                    {
                       auto held = ses->get_root<fail_stop_node>(root);
                    }
                 });
          {
             auto       ses = a.create_session();
             const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
             while (std::chrono::steady_clock::now() < end)
             {
                auto tx   = ses->start_transaction(root);
                auto next = ses->alloc<fail_stop_node>(uint32_t(64), sal::alloc_hint{});
                tx->root() = sal::smart_ptr<sal::alloc_header>(ses, next);
                (void)tx->commit(sal::sync_type::none);
             }
          }
          stop = true;
          for (auto& t : readers)
             t.join();
       });
   CHECK(sig == 0);
   fs::remove_all(path);
}

#include <sal/segment_thread.hpp>

// A background thread whose loop throws is gone for good: with the segment
// provider gone, writers wait for a segment forever. That must not be a log
// line and a hang; the process stops and says which thread and why.
TEST_CASE("an exception out of a background thread stops the process", "[sal][fail_stop]")
{
   int sig = signal_from_child(
       []
       {
          sal::mapped_memory::segment_thread_state state;
          sal::segment_thread thread(&state, "test_provider",
                                     [](sal::segment_thread&)
                                     { throw std::runtime_error("no space left on device"); });
          thread.start();
          std::this_thread::sleep_for(std::chrono::seconds(5));
       });
   CHECK(sig == SIGABRT);
}

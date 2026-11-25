#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <sal/config.hpp>
#include <sal/mapped_memory/read_lock_queue.hpp>

using namespace sal;
using namespace sal::mapped_memory;

// A read lock keeps a segment the compactor has emptied from being handed out
// again while the session may still be reading it. That has to hold for the
// first segment ever recycled too: on a new database nothing has been pushed
// yet when sessions take their first locks.
TEST_CASE("a read lock taken before the first recycled segment holds it back",
          "[sal][read_lock_queue]")
{
   auto  queue = std::make_unique<read_lock_queue>();
   auto& lock  = queue->get_session_lock(allocator_session_number(3));

   lock.lock();
   queue->push_recycled_segment(segment_number(7));
   CHECK(queue->available_to_pop() == 0);
   queue->push_recycled_segment(segment_number(8));
   CHECK(queue->available_to_pop() == 0);

   lock.unlock();
   CHECK(queue->available_to_pop() == 2);

   // A lock taken now is behind nothing that is queued.
   lock.lock();
   CHECK(queue->available_to_pop() == 2);
   queue->push_recycled_segment(segment_number(9));
   CHECK(queue->available_to_pop() == 2);
   lock.unlock();
   CHECK(queue->available_to_pop() == 3);
}

TEST_CASE("resetting the session locks leaves them at the queue's end position",
          "[sal][read_lock_queue]")
{
   auto queue = std::make_unique<read_lock_queue>();
   for (uint32_t i = 0; i < 5; ++i)
      queue->push_recycled_segment(segment_number(i));
   segment_number out[8];
   REQUIRE(queue->pop_recycled_segments(out, 8) == 5);

   queue->reset_all_session_locks();
   auto& lock = queue->get_session_lock(allocator_session_number(0));
   lock.lock();
   queue->push_recycled_segment(segment_number(5));
   CHECK(queue->available_to_pop() == 0);
   lock.unlock();
   CHECK(queue->available_to_pop() == 1);
}

// gofractally/psitri#21. A session can store a position the queue has already
// passed: lock() reads the end position and stores it in two steps, and a push
// and a pop can come between them. available_to_pop() must then answer "none",
// not the unsigned difference (about 2^32), which the provider used as a count.
TEST_CASE("a session position behind the pop position does not underflow",
          "[sal][read_lock_queue]")
{
   auto  queue = std::make_unique<read_lock_queue>();
   auto& lock  = queue->get_session_lock(allocator_session_number(11));

   // lock() read the end position (0) before this push was broadcast...
   queue->push_recycled_segment(segment_number(1));
   segment_number out[4];
   REQUIRE(queue->available_to_pop() == 1);
   REQUIRE(queue->pop_recycled_segments(out, 4) == 1);
   // ...and stores it after the provider has popped: the session is now at
   // position 0 and the queue's pop position is 1.
   lock.update(0);
   lock.lock();

   queue->push_recycled_segment(segment_number(2));
   CHECK(queue->available_to_pop() == 0);
   lock.unlock();
   CHECK(queue->available_to_pop() == 1);
}

TEST_CASE("the three roles of the read lock queue never see more to pop than was pushed",
          "[sal][read_lock_queue][stress]")
{
   auto                  queue = std::make_unique<read_lock_queue>();
   std::atomic<bool>     stop{false};
   std::atomic<uint32_t> bad{0};
   std::atomic<uint64_t> pops{0};

   std::thread compactor(
       [&]
       {
          uint32_t n = 0;
          while (!stop.load(std::memory_order_relaxed))
             if (queue->available_to_push() > 0)
                queue->push_recycled_segment(segment_number(n++));
       });
   std::thread provider(
       [&]
       {
          segment_number segs[1024];
          while (!stop.load(std::memory_order_relaxed))
          {
             uint32_t available = queue->available_to_pop();
             if (available > queue->recycled_queue_capacity())
             {
                bad.store(available);
                stop = true;
                return;
             }
             if (available)
                pops += queue->pop_recycled_segments(segs, int(available));
          }
       });
   std::vector<std::thread> sessions;
   for (int s = 0; s < 6; ++s)
      sessions.emplace_back(
          [&, s]
          {
             auto&    lock = queue->get_session_lock(allocator_session_number(s + 10));
             uint64_t x    = s * 7919 + 1;
             while (!stop.load(std::memory_order_relaxed))
             {
                lock.lock();
                x = x * 6364136223846793005ull + 1442695040888963407ull;
                for (volatile uint32_t spin = 0; spin < (x >> 54); spin = spin + 1)
                {
                }
                lock.unlock();
             }
          });
   const auto start = std::chrono::steady_clock::now();
   while (!stop && std::chrono::steady_clock::now() - start < std::chrono::seconds(3))
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
   stop = true;
   compactor.join();
   provider.join();
   for (auto& t : sessions)
      t.join();
   CHECK(bad.load() == 0);
   CHECK(pops.load() > 0);
}

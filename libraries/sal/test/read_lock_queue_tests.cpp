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

#include <sal/mapped_memory/segment.hpp>

// A segment's pinned flag is set and cleared by one thread while another moves
// the segment through its life cycle in the same word. Neither may undo the
// other's change.
TEST_CASE("a segment's pinned flag survives concurrent state changes", "[sal][segment_meta]")
{
   auto meta = std::make_unique<sal::mapped_memory::segment_meta>();
   meta->recovered_free();

   std::atomic<int>  round{0};
   std::atomic<int>  done{0};
   std::atomic<bool> stop{false};
   const int         rounds = 200000;
   int               lost   = 0;

   std::thread cycler(
       [&]
       {
          int seen = 0;
          while (!stop.load(std::memory_order_relaxed))
          {
             int r = round.load(std::memory_order_acquire);
             if (r == seen)
                continue;
             seen = r;
             meta->added_to_provider_queue();
             meta->allocated_by_session();
             meta->prepare_for_compaction(1);
             meta->added_to_read_lock_queue();
             meta->added_to_free_list();
             done.store(r, std::memory_order_release);
          }
       });
   for (int r = 1; r <= rounds; ++r)
   {
      const bool pin = r & 1;
      round.store(r, std::memory_order_release);
      meta->set_pinned(pin);
      while (done.load(std::memory_order_acquire) != r)
      {
      }
      if (meta->is_pinned() != pin)
         ++lost;
   }
   stop = true;
   cycler.join();
   CHECK(lost == 0);
}

#include <filesystem>

#include <sal/allocator_impl.hpp>
#include <sal/allocator_session_impl.hpp>
#include <sal/smart_ptr_impl.hpp>

namespace
{
   struct first_recycle_node : sal::alloc_header
   {
      static constexpr sal::header_type type_id = sal::header_type::start_user_type;
      uint64_t                          magic;
      first_recycle_node(uint32_t size, sal::ptr_address_seq seq, uint64_t m) noexcept
          : sal::alloc_header(size, type_id, seq), magic(m)
      {
      }
   };
}  // namespace

// gofractally/psitri#16, end to end. On a NEW database: a session takes its
// read lock and reads an object; the compactor then empties that object's
// segment, the first segment this database ever recycles. Until the session
// drops its lock the bytes it is reading must stay what they were, however
// much is allocated meanwhile: the segment may not be handed out again.
//
// Before the read-lock fix the lock taken before the first recycle counted
// for nothing, the provider gave the segment away at once, and the reader's
// object was overwritten (on Linux, zeroed by the hole punch). A writer that
// copied a page from such a segment built its new tree from those bytes:
// inserts lost for good.
TEST_CASE("a session reading across the first recycled segment of a new database keeps its bytes",
          "[sal][read_lock_queue][first_recycle]")
{
   namespace fs    = std::filesystem;
   const auto path = fs::temp_directory_path() / "sal_first_recycle_test";
   fs::remove_all(path);
   fs::create_directories(path);
   {
      sal::runtime_config cfg;
      cfg.max_pinned_cache_size_mb             = 0;
      cfg.compact_pinned_unused_threshold_mb   = 1;
      cfg.compact_unpinned_unused_threshold_mb = 1;
      cfg.enable_read_cache                    = false;
      sal::allocator a(path, cfg, false);
      a.register_type_ops<first_recycle_node>();
      a.start_background_threads();

      const uint32_t MiB    = 1 << 20;
      auto           writer = a.create_session();
      auto           reader = a.create_session();

      // One small object and enough filler to take its segment and the next.
      sal::ptr_address watched =
          writer->alloc<first_recycle_node>(uint32_t(128), sal::alloc_hint{}, uint64_t(0xAAAA));
      std::vector<sal::ptr_address> filler;
      for (int i = 0; i < 70; ++i)
         filler.push_back(
             writer->alloc<first_recycle_node>(MiB, sal::alloc_hint{}, uint64_t(0x1111)));
      writer->sync(sal::sync_type::none);

      // The reader starts reading: lock, then find the object.
      auto                      lock = reader->lock();
      const first_recycle_node* seen =
          static_cast<const first_recycle_node*>(a.resolve(watched).first);
      REQUIRE(seen->magic == 0xAAAA);

      // The rest of the segment dies; the compactor moves the watched object out and the
      // segment becomes the first one this database recycles.
      for (auto adr : filler)
         writer->release(adr);
      bool moved = false;
      for (int i = 0; i < 6000 && !moved; ++i)
      {
         moved = a.resolve(watched).first != seen;
         if (!moved)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      REQUIRE(moved);

      // Give the provider time to take the segment if it (wrongly) may, then allocate far
      // more than a segment: a segment handed out again is written over.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      std::vector<sal::ptr_address> more;
      for (int i = 0; i < 300; ++i)
         more.push_back(
             writer->alloc<first_recycle_node>(MiB, sal::alloc_hint{}, uint64_t(0x2222)));

      CHECK(seen->magic == 0xAAAA);
      CHECK(*seen->address() == *watched);

      for (auto adr : more)
         writer->release(adr);
      writer->release(watched);
   }
   fs::remove_all(path);
}

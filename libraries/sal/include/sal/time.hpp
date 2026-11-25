#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>
#include <ucc/typed_int.hpp>

namespace sal
{
   /**
    * time_manager - Singleton class that maintains a background thread 
    * for updating the current time at regular intervals.
    * 
    * This reduces the overhead of frequent chrono calls by having a single
    * background thread update a shared atomic value that other threads read,
    * this is useful when we don't need the full precision of a high resolution
    * clock, but still want a monotonic clock that is consistent across threads.
    */
   class time_manager
   {
     private:
      std::atomic<uint64_t> _current_time_ms{0};
      std::thread           _time_thread;
      std::atomic<bool>     _running{true};

      // Singleton instance
      static time_manager& instance()
      {
         static time_manager manager;
         return manager;
      }

      // Private constructor - starts the background thread
      time_manager()
      {
         // Initialize with current time
         _current_time_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());

         // Start background thread
         _time_thread = std::thread(
             [this]()
             {
// Set thread name
#ifdef __APPLE__
                pthread_setname_np("time_updater");
#else
                pthread_setname_np(pthread_self(), "time_updater");
#endif

                while (_running.load(std::memory_order_relaxed))
                {
                   // Update time
                   _current_time_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::steady_clock::now().time_since_epoch())
                                              .count(),
                                          std::memory_order_relaxed);

                   // The value is in milliseconds and its readers age cache
                   // entries by it, so one update per millisecond is all it
                   // can express. This slept 100 microseconds: ten thousand
                   // wakeups a second, in every process, for nothing.
                   std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
             });

         // Register shutdown handler
         std::atexit([]() { time_manager::shutdown(); });
      }

     public:
      ~time_manager() { shutdown(); }

      // Get current time (milliseconds)
      static uint64_t now_ms()
      {
         return instance()._current_time_ms.load(std::memory_order_relaxed);
      }

      // Explicitly shutdown the time manager
      static void shutdown()
      {
         auto& manager = instance();
         if (manager._running.exchange(false) && manager._time_thread.joinable())
         {
            manager._time_thread.join();
         }
      }

      // Prevent copying
      time_manager(const time_manager&)            = delete;
      time_manager& operator=(const time_manager&) = delete;
   };

   using msec_timestamp = ucc::typed_int<uint64_t, struct msec_tag>;
   using usec_timestamp = ucc::typed_int<uint64_t, struct usec_tag>;
   inline msec_timestamp get_current_time_msec()
   {
      return msec_timestamp(time_manager::now_ms());
      /*
      return msec_timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
                                */
   }
   namespace detail
   {
      inline std::atomic<uint64_t> last_commit_stamp{0};
   }

   /// The stamp of a commit of a root. It is written to the root's record in the roots
   /// file and into the sync headers of the commit; power-loss recovery takes, for each
   /// root, the commit with the newest stamp. So it must keep increasing across a restart
   /// of the machine, which a steady clock (time since boot) does not. It is wall-clock
   /// microseconds, never less than one more than the last stamp given out in this
   /// process; opening a store and recovering one raise that floor above every stamp
   /// found in it (commit_stamp_at_least), so a clock set backwards does not reorder
   /// commits either.
   inline usec_timestamp next_commit_stamp() noexcept
   {
      const uint64_t wall = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count());
      uint64_t last = detail::last_commit_stamp.load(std::memory_order_relaxed);
      uint64_t next;
      do
      {
         next = std::max(wall, last + 1);
      } while (!detail::last_commit_stamp.compare_exchange_weak(last, next,
                                                                std::memory_order_relaxed));
      return usec_timestamp(next);
   }
   /// Recovery: no later commit may be stamped at or below `stamp`.
   inline void commit_stamp_at_least(usec_timestamp stamp) noexcept
   {
      uint64_t last = detail::last_commit_stamp.load(std::memory_order_relaxed);
      while (last < *stamp && !detail::last_commit_stamp.compare_exchange_weak(
                                  last, *stamp, std::memory_order_relaxed))
      {
      }
   }

}  // namespace sal

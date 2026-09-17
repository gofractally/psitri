#pragma once
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace psitri::detail
{
   /// FIFO admission for transactions and committed-snapshot capture.
   /// Consecutive shared requests may run together; a queued exclusive request
   /// prevents later readers or writers from overtaking it. No lock is retained
   /// while reading an already captured snapshot.
   class fifo_shared_mutex
   {
     public:
      void lock()
      {
         std::unique_lock lock(_mutex);
         const auto ticket = _next++;
         _changed.wait(lock, [&] { return ticket == _serving && !_writer && !_readers; });
         _writer = true;
         ++_serving;
      }

      bool try_lock()
      {
         std::unique_lock lock(_mutex, std::try_to_lock);
         if (!lock || _writer || _readers || _next != _serving) return false;
         _writer = true;
         ++_next;
         ++_serving;
         return true;
      }

      void unlock()
      {
         std::unique_lock lock(_mutex);
         assert(_writer);
         _writer = false;
         wake_next(lock);
      }

      void lock_shared()
      {
         std::unique_lock lock(_mutex);
         const auto ticket = _next++;
         _changed.wait(lock, [&] { return ticket == _serving && !_writer; });
         ++_readers;
         ++_serving;
         wake_next(lock); // Admit the next shared request without waiting for us.
      }

      bool try_lock_shared()
      {
         std::unique_lock lock(_mutex, std::try_to_lock);
         if (!lock || _writer || _next != _serving) return false;
         ++_readers;
         ++_next;
         ++_serving;
         return true;
      }

      void unlock_shared()
      {
         std::unique_lock lock(_mutex);
         assert(_readers);
         if (--_readers == 0) wake_next(lock);
      }

     private:
      void wake_next(std::unique_lock<std::mutex>& lock)
      {
         const bool waiting = _next != _serving;
         lock.unlock();
         // notify_one could wake a later ticket and leave the next one asleep.
         if (waiting) _changed.notify_all();
      }

      std::mutex _mutex;
      std::condition_variable _changed;
      uint64_t _next = 0, _serving = 0, _readers = 0;
      bool _writer = false;
   };
}

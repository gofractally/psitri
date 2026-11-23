#pragma once
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <bit>
#include <cstdint>
#include <sal/config.hpp>
#include <sal/debug.hpp>
#include <sal/location.hpp>
#include <stdexcept>

#if PSITRI_DEEP_INVARIANTS
#include <execinfo.h>
#include <unistd.h>
#endif

namespace sal
{
   [[noreturn, gnu::cold, gnu::noinline]] inline void sal_release_underflow_abort(
       const void* cb,
       uint64_t    prior_word)
   {
      fprintf(stderr,
              "sal: release of control block %p whose reference count was already zero "
              "(word 0x%llx): a reference was released twice\n",
              cb, (unsigned long long)prior_word);
      ::abort();
   }

#if PSITRI_DEEP_INVARIANTS
   [[noreturn]] inline void sal_debug_resurrection_abort(const void* cb, uint64_t prior_word)
   {
      fprintf(stderr, "RESURRECTION-RETAIN: cb=%p prior_word=0x%llx (ref was 0)\n", cb,
              (unsigned long long)prior_word);
      void* frames[32];
      int   n = ::backtrace(frames, 32);
      ::backtrace_symbols_fd(frames, n, STDERR_FILENO);
      ::abort();
   }
#endif

   /**
    * A control block for a shared pointer to a location in shared memory,
    * modeled after std::control_block but adapted for the needs of the psitri.
    */
   struct control_block
   {
     private:
      std::atomic<uint64_t> _data;

     public:
      // Default constructor to ensure proper initialization of the atomic
      control_block() noexcept : _data(0) {}

      static constexpr uint64_t location_offset      = 24;  // ref(18)+type(2)+presence(4)
      static constexpr uint64_t max_cacheline_offset = (1ULL << 38) - 1;  // 32 TB @ 128 B
      /**
       * The internal structure of the bits stored in the atomic _data
       */
      struct control_block_data
      {
         /// reference count, up to 256K shared references (bits 0-17)
         uint64_t ref : 18;
         /// Opaque 6-bit user/application field (was 3) — SAL never reads or interprets it. The
         /// radix engine splits it as 2 bits dispatch tier (r1/r2/r4/r8) + 4 bits segment-
         /// presence map; both are DERIVED HINTS for skip-header routing (the node is the source
         /// of truth — control blocks are rebuilt on recovery). Sits just above `ref`, so the
         /// optimistic fetch_add/sub refcount path never carries into it (max_ref_count guard
         /// keeps ref + thread overshoot below 2^18).
         uint64_t type : 6;
         /// index to the cacheline of up to 32 TB of memory at 128 bytes per cacheline
         /// (sal::cacheline_size = max_database_size; static_assert below pins the width).
         uint64_t cacheline_offset : 38;

         /// set this bit when object is read, clearered when ref count goes to 0
         uint64_t active : 1;
         /// set this bit when object should be cached, but this gets cleared
         /// when reference count goes to 0 along with the active bit.
         uint64_t pending_cache : 1;

         control_block_data() noexcept
             : ref(0), type(0), cacheline_offset(0), active(0), pending_cache(0)
         {
         }
         control_block_data(uint64_t value) noexcept { from_int(value); }
         uint64_t to_int() const noexcept { return std::bit_cast<uint64_t>(*this); }
         void     from_int(uint64_t value) noexcept
         {
            *this = std::bit_cast<control_block_data>(value);
         }
         location loc() const noexcept { return location::from_cacheline(cacheline_offset); }

         control_block_data& set_ref(uint64_t r) noexcept
         {
            assert(r <= max_ref_count);
            ref = r;
            return *this;
         }
         control_block_data& set_loc(location l) noexcept
         {
            cacheline_offset = l.cacheline();
            return *this;
         }
         control_block_data& set_active(bool a) noexcept
         {
            active = a;
            return *this;
         }
         control_block_data& set_pending_cache(bool p) noexcept
         {
            pending_cache = p;
            return *this;
         }
         /// Set the opaque 6-bit user field (only the low 6 bits are stored).
         control_block_data& set_type(uint8_t t) noexcept
         {
            type = uint64_t(t) & 0x3Fu;
            return *this;
         }
         friend std::ostream& operator<<(std::ostream& os, const control_block_data& data)
         {
            os << "{ref:" << data.ref << " type:" << data.type << " loc:" << data.loc()
               << " active:" << data.active << " pending_cache:" << data.pending_cache << "}";
            return os;
         }
      };

      /** there are 18 bits for the reference count (bits 18-20 reclaimed for the opaque
       * `type` user field), and the algorithm utilizes an
       * optimistic approach to reference counting by using fetch_add instead of
       * the slower compare_exchange, this means that we need handle the overflow case
       * by using fetch_sub to decrement the reference count; therefore, the maximum
       * ref count must allow for a certain number of threads to over-shoot the count
       * without causing problems.  This is why we subtract the number of threads from
       * the maximum possible ref count.
       */
      static constexpr uint64_t max_ref_count = (1ULL << 18) - sal::max_threads;

      int use_count() const noexcept
      {
         return control_block_data(_data.load(std::memory_order_relaxed)).ref;
      }
      bool unique() const noexcept { return use_count() == 1; }
      void reset() noexcept
      {
         _data.store(control_block_data().to_int(), std::memory_order_relaxed);
      }
      control_block_data retain()
      {
         control_block_data prior = _data.fetch_add(1, std::memory_order_relaxed);
         if (prior.ref >= max_ref_count)
         {
            _data.fetch_sub(1, std::memory_order_relaxed);
            throw std::runtime_error("reference count exceeded limits");
         }
#if PSITRI_DEEP_INVARIANTS
         // A retain on ref==0 resurrects a dead control (stale loc, bytes
         // already freed) — the seed of compactor double-free corruption.
         // Catch it AT the caller, with a backtrace.
         if (prior.ref == 0)
            sal_debug_resurrection_abort(this, prior.to_int());
#endif
         assert(prior.ref > 0);
         return prior;
      };

      uint32_t ref() const noexcept { return load(std::memory_order_relaxed).ref; }
      location loc() const noexcept { return load(std::memory_order_acquire).loc(); }
      bool     active() const noexcept { return load(std::memory_order_relaxed).active; }
      bool pending_cache() const noexcept { return load(std::memory_order_relaxed).pending_cache; }

      /// @deprecated dont use this
      uint64_t to_int(std::memory_order order = std::memory_order_relaxed) const noexcept
      {
         return _data.load(order);
      }

      control_block_data load(std::memory_order order = std::memory_order_relaxed) const noexcept
      {
         return control_block_data(_data.load(order));
      }
      void store(control_block_data value,
                 std::memory_order  order = std::memory_order_relaxed) noexcept
      {
         _data.store(value.to_int(), order);
      }
      void reset(location          loc,
                 int               ref   = 1,
                 std::memory_order order = std::memory_order_release) noexcept
      {
         store(control_block_data().set_loc(loc).set_ref(ref), order);
      }
      void set_ref(int ref, std::memory_order order = std::memory_order_relaxed) noexcept
      {
         store(load(order).set_ref(ref), order);
      }

      /**
       * Reclaim exclusive ownership of a just-resurrected control for inline destroy:
       * (ref=1, loc=L) -> (ref=0, loc=L) under CAS, so a concurrent compactor cas_move
       * (the only legal writer of an unreachable ref-1 object) cannot be lost — the
       * object may have relocated since the caller's last read. Returns the
       * pre-transition word: destroy at its loc().
       */
      control_block_data take_for_destroy() noexcept
      {
         uint64_t           expect = _data.load(std::memory_order_acquire);
         control_block_data prior, next;
         do
         {
            prior.from_int(expect);
            assert(prior.ref == 1);
            next = prior;
            next.set_ref(0);
         } while (
             not _data.compare_exchange_weak(expect, next.to_int(), std::memory_order_acq_rel));
         return prior;
      }

      /**
       * @return the state of the control_block_data before decrementing the reference count
       */
      control_block_data release() noexcept
      {
         // if we are not the last reference then relaxed is best, we will
         // load with acquire before returning if we are the last reference.
         // TSAN seems happy with relaxed + acquire and this works because all
         // modifications are either done by the "unique owner" aka (ref 1),
         // or done by a thread that has just copied the data to a new location
         // and we are synchronziing with the cas_move below.
         control_block_data prior(_data.fetch_sub(1, std::memory_order_relaxed));
         //         SAL_WARN("release cb{} : pre ref: {}", this, int(prior.ref));
         if (prior.ref == 0) [[unlikely]]
         {
            // Someone released a reference it did not hold. Left alone, the
            // count wraps to its maximum in the mapped file and the damage
            // shows up much later, as a retain that "exceeds limits" on some
            // other thread or after a restart. Put the count back and stop here,
            // where the caller is still on the stack.
            _data.fetch_add(1, std::memory_order_relaxed);
            sal_release_underflow_abort(this, prior.to_int());
         }
         if (prior.ref == 1)
         {
            if (prior.pending_cache or prior.active)
               clear_pending_cache();
            // make sure that any changes in location and the new memory being
            // pointed at are visible to the releasing thread.
            return load(std::memory_order_acquire);
         }
         return prior;
      };

      void clear_pending_cache() noexcept
      {
         uint64_t           expected = _data.load(std::memory_order_relaxed);
         control_block_data updated;
         do
         {
            updated.from_int(expected);
            // Preserve custom CB marker: {active=0, pending_cache=1}
            if (!updated.active && updated.pending_cache)
               return;
            updated.set_active(false);
            updated.set_pending_cache(false);
         } while (
             !_data.compare_exchange_weak(expected, updated.to_int(), std::memory_order_release));
      }

      /// Opaque 6-bit user/application field — SAL never interprets it (the radix engine
      /// splits it 2 bits dispatch tier + 4 bits segment presence).
      uint8_t get_type(std::memory_order order = std::memory_order_relaxed) const noexcept
      {
         return uint8_t(load(order).type);
      }
      /// Atomically set the user `type` (low 6 bits) without disturbing the other fields.
      void set_type(uint8_t t, std::memory_order order = std::memory_order_relaxed) noexcept
      {
         uint64_t           expected = _data.load(std::memory_order_relaxed);
         control_block_data updated;
         do
         {
            updated.from_int(expected);
            updated.set_type(t);
         } while (!_data.compare_exchange_weak(expected, updated.to_int(), order));
      }

      /**
       * compare and swap move,
       * updates the cacheline_offset to the desired value 
       * if the current value is equal to the expected value 
       * and the ref count is not 0, note that other changes to
       * the control_block_data are alloewd 
       */
      bool cas_move(location expected_loc, location desired_loc) noexcept
      {
         assert(desired_loc.cacheline() != max_cacheline_offset);
         uint64_t           expect_data = _data.load(std::memory_order_relaxed);
         control_block_data prior;
         do
         {
            prior.from_int(expect_data);
            if (prior.loc() != expected_loc or prior.ref == 0) [[unlikely]]
               return false;
            prior.set_loc(desired_loc);
         } while (not _data.compare_exchange_weak(expect_data, prior.to_int(),
                                                  std::memory_order_seq_cst));
         return true;
      }

      // used by the allocator to claim this pointer to use, uses the
      // cacheline_offset max value to indicate the pointer alloc/free.
      /*  [[nodiscard]] bool claim() noexcept
      {
         uint64_t           expect_data = _data.load(std::memory_order_relaxed);
         control_block_data prior;
         do
         {
            prior.from_int(expect_data);
            if (uint64_t(prior.cacheline_offset) != max_cacheline_offset) [[unlikely]]
               return false;
            prior.cacheline_offset = 0;
         } while (not _data.compare_exchange_weak(expect_data, prior.to_int(),
                                                  std::memory_order_seq_cst));
         return true;
      }
      void release_claim() noexcept
      {
         control_block_data prior;
         prior.cacheline_offset = max_cacheline_offset;
         _data.store(prior.to_int(), std::memory_order_relaxed);
      }
*/
      /**
       * Range-refcount block support: the cacheline_offset field doubles as an opaque
       * user value (packed nibble counters) on a custom CB. Atomically replace it iff it
       * still equals `expect_val`, preserving `ref` and every other field (which other
       * threads — retain/release of the block's share count — may be changing). Returns
       * false if the value moved under us (caller re-reads and retries).
       */
      bool cas_user_value(uint64_t expect_val, uint64_t desired_val) noexcept
      {
         uint64_t           expect = _data.load(std::memory_order_relaxed);
         control_block_data prior;
         do
         {
            prior.from_int(expect);
            if (uint64_t(prior.cacheline_offset) != expect_val) [[unlikely]]
               return false;
            prior.cacheline_offset = desired_val;
         } while (
             not _data.compare_exchange_weak(expect, prior.to_int(), std::memory_order_seq_cst));
         return true;
      }

      /**
       * Moves the location without regard to the prior location, but
       * without disrupting any other fields that may be updated by
       * other threads.
       *
       * @return the updated control_block_data
       */
      control_block_data move(location          loc,
                              std::memory_order order = std::memory_order_relaxed) noexcept
      {
         assert(loc.cacheline() != max_cacheline_offset);
         auto               expected = _data.load(order);
         control_block_data updated;
         do
         {
            updated.from_int(expected);
            assert(updated.loc() != loc);
            updated.set_loc(loc);
         } while (not _data.compare_exchange_weak(expected, updated.to_int(), order));
         updated.from_int(expected);
         return updated;
      }

      /**
       * Like move(), but sets the location AND the opaque `type` (6 bits) in the SAME CAS — so a
       * reader's single atomic load never sees a new location paired with a stale type/presence
       * (the segment layout and the routing hint that describes it must change together). Used by
       * the grow path that reallocs a bitset node to splice in a previously-absent segment
       * cacheline (the engine bumps the presence nibble inside `type`).
       * @return the prior control_block_data.
       */
      control_block_data move_with_type(location loc, uint8_t type,
                                        std::memory_order order = std::memory_order_relaxed) noexcept
      {
         assert(loc.cacheline() != max_cacheline_offset);
         auto               expected = _data.load(order);
         control_block_data updated;
         do
         {
            updated.from_int(expected);
            updated.set_loc(loc).set_type(type);
         } while (not _data.compare_exchange_weak(expected, updated.to_int(), order));
         updated.from_int(expected);
         return updated;
      }

      /**
       * Attempts to increment the activity counter in a non-blocking way.
       * If the object is not marked as active, tries to set the active bit.
       * If already active, tries to set the pending_cache bit.
       * May fail if there is contention, which is acceptable since this 
       * simulates random sampling behavior.
       *
       * @return true if successfully incremented activity, false if failed due to contention
       */
      bool try_inc_activity() noexcept
      {
         uint64_t           expected = _data.load(std::memory_order_relaxed);
         control_block_data updated(expected);
         if (updated.pending_cache)
            return false;
         if (updated.active)
            return _data.compare_exchange_weak(expected, updated.set_pending_cache(true).to_int(),
                                               std::memory_order_relaxed);
         return _data.compare_exchange_weak(expected, updated.set_active(true).to_int(),
                                            std::memory_order_relaxed);
      }

      /**
       * Marks a sampled read. The first sampled hit sets the active bit; a
       * later sampled hit within the decay window sets pending_cache and asks
       * the caller to enqueue the object for promotion.
       *
       * @return true only when this call transitioned the object to
       * pending_cache and the read-cache queue should receive the address.
       */
      bool try_mark_read_cache_pending() noexcept
      {
         uint64_t           expected = _data.load(std::memory_order_relaxed);
         control_block_data updated(expected);
         if (updated.pending_cache)
            return false;

         if (updated.active)
         {
            return _data.compare_exchange_weak(expected,
                                               updated.set_pending_cache(true).to_int(),
                                               std::memory_order_relaxed);
         }

         _data.compare_exchange_weak(expected, updated.set_active(true).to_int(),
                                     std::memory_order_relaxed);
         return false;
      }

      /**
       *  Clears the pending cache bit, returns false if it is already cleared
       * 
       * @return true if the pending cache bit was cleared, false otherwise
       */
      bool try_end_pending_cache() noexcept
      {
         uint64_t           expected = _data.load(std::memory_order_relaxed);
         control_block_data updated;
         do
         {
            updated.from_int(expected);
            if (updated.pending_cache == false)
               return false;
            updated.set_pending_cache(false);
         } while (not _data.compare_exchange_weak(expected, updated.to_int(),
                                                  std::memory_order_relaxed));
         return true;
      }

      static_assert(sizeof(control_block_data) == 8, "shared_ptr_data must be 8 bytes");
      // cacheline_offset shrank 41→38 to free 3 bits for `presence`. 38 bits × 128 B = 32 TB,
      // which must still cover the whole addressable database. If max_database_size grows,
      // steal the bits back from presence/type.
      static_assert(max_database_size / cacheline_size <= (1ull << 38),
                    "cacheline_offset (38 bits) can't address max_database_size");
   };
   static_assert(sizeof(control_block) == 8, "shared_ptr must be 8 bytes");

   using control_block_data = control_block::control_block_data;

}  // namespace sal

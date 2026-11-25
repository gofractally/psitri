#pragma once
// implicit_ref_pool — per-range branch-set refcounts on colo_alloc<u8> (doc/micro_refcount.md).
//
// Each slot is a 1-byte atomic refcount that IS its own lifetime: alloc sets it to 1, a node
// clone retain()s it (++), a node destroy release()s it (--), and the releaser that drives it to
// 0 frees the slot (and is told `was_last` so it can release that range's actual branch
// pointers). No "deferral"/settle special case. The one-byte count has a ceiling of 255 sharers:
// try_retain refuses the 256th and the caller forks (takes a counter of its own) — correct and async-safe by construction (clone's +1 and destroy's -1 are a conserved
// pair regardless of when the destructor runs). The "pointee" is implicit: the slot counts the
// branch ptr_addresses of a node's range, which live in the node array, not in the slot.
//
// Counts are EPHEMERAL: rebuilt on recovery by walking the live roots and replaying each node's
// persistent handles through retally(). Storage/co-location/addressing are colo_alloc<u8>'s job;
// the compact handle is region+offset (sal::implicit_ref_addr).
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include <sal/colo_alloc.hpp>

namespace sal
{
   using implicit_ref_count           = std::atomic<uint8_t>;
   inline constexpr uint8_t implicit_ref_max = 255;  // 1-byte ceiling; fork before exceeding

   class implicit_ref_pool
   {
     public:
      explicit implicit_ref_pool(const std::filesystem::path& dir,
                                 access_mode                  mode = access_mode::read_write)
          : _a(dir, mode)
      {
      }

      static constexpr uint32_t slots_per_cacheline =
          colo_alloc<implicit_ref_count>::slots_per_cacheline;  // 128

      /// Birth's first counter (no siblings yet): a fresh slot, refcount = 1. Returns the full
      /// ptr_address; the caller takes region_of()/offset_of() for the compact handle.
      ptr_address alloc_fresh()
      {
         auto a = _a.alloc();
         a.ptr->store(1, std::memory_order_relaxed);
         return a.addr_seq.address;
      }

      /// A sibling counter for the same node: co-locate in the hint's region (cacheline first,
      /// then anywhere in the region). nullopt only if the region is full → caller picks a new
      /// region (alloc_fresh) and re-homes its still-unique counters.
      std::optional<ptr_address> alloc_in_region(const implicit_ref_hint& h)
      {
         auto a = _a.try_alloc(h);
         if (!a)
            return std::nullopt;
         a->ptr->store(1, std::memory_order_relaxed);
         return a->addr_seq.address;
      }

      /// Clone shares the slot: one more version references this range's branch-set.
      /// Returns false, and changes nothing, when the count is already at its one-byte
      /// ceiling: the caller must then give its version a counter of its own for the
      /// range (and retain the range's branches itself). Never wraps. More than one
      /// thread may share a slot at once (writers continuing from versions of one tree),
      /// so the ceiling is tested and the count raised in one atomic step.
      [[nodiscard]] bool try_retain(ptr_address adr)
      {
         auto& count = _a.get(adr);
         auto  prev  = count.load(std::memory_order_relaxed);
         do
         {
            assert(prev >= 1 && "retain of a dead (0) slot");
            if (prev == implicit_ref_max)
               return false;
         } while (not count.compare_exchange_weak(prev, uint8_t(prev + 1),
                                                  std::memory_order_relaxed));
         return true;
      }

      /// Destroy releases the slot. Returns true iff this was the LAST reference — the slot is
      /// then freed and the caller must release that range's branch pointers.
      [[nodiscard]] bool release(ptr_address adr)
      {
         auto prev = _a.get(adr).fetch_sub(1, std::memory_order_acq_rel);
         assert(prev >= 1 && "release of a dead (0) slot");
         if (prev == 1)
         {
            _a.free(adr);  // self-free: slot returns to the pool
            return true;
         }
         return false;
      }

      uint8_t  count(ptr_address adr) { return _a.get(adr).load(std::memory_order_relaxed); }
      uint64_t used() const { return _a.used(); }

      // ── recovery (counts reconstructed from node data alone) ──
      /// Reset: every slot free, every counter 0. Then replay live handles via retally().
      void clear_all() { _a.clear_all(); }
      /// Replay ONE reference to `adr` from the recovery walk: claim the slot on first sight,
      /// then ++ its counter. After walking all live (node,range) handles, each slot's counter
      /// equals its live share-count and the allocation set is exactly the referenced slots.
      /// Returns true on first sight: the caller then owns walking that range's branches,
      /// which are referenced once per counter rather than once per sharing node.
      bool retally(ptr_address adr)
      {
         const auto previous = _a.get_or_claim(adr).fetch_add(1, std::memory_order_relaxed);
         if (previous == implicit_ref_max)
            throw std::runtime_error("implicit_ref_pool: share count overflow during recovery");
         return previous == 0;
      }
      void sync(sync_type type) noexcept { _a.sync(type); }

     private:
      colo_alloc<implicit_ref_count> _a;
   };
}  // namespace sal

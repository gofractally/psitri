#pragma once
// implicit_ref_pool — per-range branch-set refcounts on colo_alloc<u8> (doc/micro_refcount.md).
//
// Each slot is a 1-byte atomic refcount that IS its own lifetime: alloc sets it to 1, a node
// clone retain()s it (++), a node destroy release()s it (--), and the releaser that drives it to
// 0 frees the slot (and is told `was_last` so it can release that range's actual branch
// pointers). No "deferral"/settle special case, no fork-on-overflow except the genuine 255-version
// ceiling — correct and async-safe by construction (clone's +1 and destroy's -1 are a conserved
// pair regardless of when the destructor runs). The "pointee" is implicit: the slot counts the
// branch ptr_addresses of a node's range, which live in the node array, not in the slot.
//
// Counts are EPHEMERAL: rebuilt on recovery by walking the live roots and replaying each node's
// persistent handles through retally(). Storage/co-location/addressing are colo_alloc<u8>'s job;
// the compact handle is region+offset (sal2::implicit_ref_addr).
#include <cassert>
#include <cstdint>
#include <optional>

#include <sal2/alloc/colo_alloc.hpp>

namespace sal2
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
      void retain(ptr_address adr)
      {
         [[maybe_unused]] auto prev = _a.get(adr).fetch_add(1, std::memory_order_relaxed);
         assert(prev >= 1 && "retain of a dead (0) slot");
         assert(prev < implicit_ref_max && "ref overflow — fork before 255 versions");
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
      void retally(ptr_address adr)
      {
         _a.get_or_claim(adr).fetch_add(1, std::memory_order_relaxed);
      }

     private:
      colo_alloc<implicit_ref_count> _a;
   };
}  // namespace sal2

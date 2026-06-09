#pragma once
// colo_alloc<T> — co-locating slab allocator, the generic form of control_block_alloc.
//
// control_block_alloc packs 8-byte control_blocks tightly into mmap'd 4M-slot zones,
// hands out 32-bit indices, and lets callers HINT a cacheline so a node's children's
// refcounts cluster (update 256 refs touching ~32 cachelines, not 256). colo_alloc<T>
// is exactly that machinery parameterized on the element type:
//
//   slots_per_cacheline = cacheline_size / sizeof(T)   (16 for control_block, 128 for u8)
//   zone_bytes          = ptrs_per_zone * sizeof(T)    (32MB for control_block, 4MB for u8)
//
// Zones stay a FIXED ptr-count (detail::ptrs_per_zone) so ptr_address->(zone,offset) math is
// T-independent and the index space is the full 2^32 for every T (a u8 pool therefore has at
// least as many slots as the control-block pool). Only the per-zone mmap BYTES scale with
// sizeof(T). Free state lives in the same per-slot atomic bitmap as control_block_alloc
// (1 bit/slot, scanned via relaxed atomic word loads + CAS-to-claim — TSan-clean; the lazy
// read is a hint, the CAS is the gate), so colo_alloc<control_block> is behaviorally
// comparable to the original for a differential gate. The shared types (version_t,
// ptr_address_seq, alloc_hint, detail::ptr_alloc_header, ptrs_per_zone) are reused verbatim.
//
// This does NOT add element semantics (retain/release/was_last); those layer on top for the
// counter instantiation. It is the slot allocator: alloc / free / get / try_get / hint.
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>

#include <hash/lehmer64.h>
#include <sal/block_allocator.hpp>
#include <sal/config.hpp>             // cacheline_size
#include <sal/control_block_alloc.hpp>  // version_t, ptr_address_seq, alloc_hint, detail::*
#include <sal/debug.hpp>
#include <sal/numbers.hpp>
#include <sal/simd_utils.hpp>

namespace sal
{
   template <typename T>
   struct colo_allocation
   {
      ptr_address_seq addr_seq;
      T*              ptr;
   };

   // ── compact addressing for dense per-node counter sets ────────────────────────────────────
   // A 32-bit ptr_address splits into a 16-bit REGION (high half, 65536 contiguous slots) and a
   // 16-bit OFFSET within it. A node that owns N counters stores ONE region + N offsets
   // (`implicit_ref_addr`, 2 bytes each) instead of N full 4-byte ptr_addresses — the compact
   // handle that fits the leftover header cacheline. "implicit" because the slot is a refcount
   // whose pointee (a range's branch-set) is not stored, only counted.
   using implicit_ref_region                          = uint16_t;
   using implicit_ref_addr                            = ucc::typed_int<uint16_t, struct implicit_ref_addr_tag>;
   inline constexpr uint32_t implicit_ref_region_slots = 1u << 16;  // 65536 slots / region

   inline ptr_address combine(implicit_ref_region r, implicit_ref_addr o) noexcept
   {
      return ptr_address((uint32_t(r) << 16) | uint32_t(*o));
   }
   inline implicit_ref_region region_of(ptr_address a) noexcept
   {
      return implicit_ref_region(*a >> 16);
   }
   inline implicit_ref_addr offset_of(ptr_address a) noexcept
   {
      return implicit_ref_addr(uint16_t(*a));
   }

   // Hint for the compact path: a region + the sibling offsets already placed there. Unlike
   // control_block_alloc's span<ptr_address> hint (cacheline-or-global), this prefers the
   // siblings' CACHELINE, then falls back to anywhere in the SAME REGION (so a node's counters
   // stay together for small-offset addressing); the caller re-homes only if the region fills.
   struct implicit_ref_hint
   {
      implicit_ref_region                region;
      std::span<const implicit_ref_addr> within;  // sibling offsets in `region`
   };

   template <typename T>
   class colo_alloc
   {
     public:
      // 128B cacheline / sizeof(T): the co-location granularity (16 for control_block, 128 u8).
      static constexpr uint32_t slots_per_cacheline = uint32_t(sal::cacheline_size) / sizeof(T);
      static constexpr uint32_t zone_bytes          = detail::ptrs_per_zone * uint32_t(sizeof(T));
      static_assert(slots_per_cacheline >= 1, "T larger than a cacheline");
      static_assert((slots_per_cacheline & (slots_per_cacheline - 1)) == 0,
                    "slots_per_cacheline must be a power of two (sizeof(T) | cacheline_size)");

      struct mlock_stats
      {
         bool                         header_pinned = false;
         block_allocator::mlock_stats zones;
         block_allocator::mlock_stats free_list;
      };

      explicit colo_alloc(const std::filesystem::path& dir) : _dir(dir)
      {
         std::filesystem::create_directories(dir);
         _header = std::make_unique<mapping>(dir / "header.bin", sal::access_mode::read_write, true);
         if (_header->size() != sizeof(detail::ptr_alloc_header))
         {
            if (_header->size() != 0)
               throw std::runtime_error("colo_alloc header file is wrong size");
            _header->resize(sizeof(detail::ptr_alloc_header));
            new (_header->data()) detail::ptr_alloc_header();
         }
         _header_ptr = reinterpret_cast<detail::ptr_alloc_header*>(_header->data());
         if (not _header_ptr)
            throw std::runtime_error("failed to create colo_alloc header");
         _zone_allocator = std::make_unique<block_allocator>(dir / "zone.bin", zone_bytes,
                                                             detail::max_allocated_zones);
         _zone_free_list = std::make_unique<block_allocator>(
             dir / "free_list.bin", detail::ptrs_per_zone / 8, detail::max_allocated_zones);
         _zone_allocator->mlock_mapped_blocks();
         _zone_free_list->mlock_mapped_blocks();
         ensure_capacity(1);
         _ptr_base       = _zone_allocator->template get<T>(block_allocator::offset_ptr(0));
         _free_list_base = _zone_free_list->template get<std::atomic<uint64_t>>(
             block_allocator::offset_ptr(0));
      }
      ~colo_alloc() = default;

      mlock_stats get_mlock_stats() const noexcept
      {
         return {_header->pinned(), _zone_allocator->get_mlock_stats(),
                 _zone_free_list->get_mlock_stats()};
      }

      /// Allocate on one of the least-filled cachelines in the current zone.
      colo_allocation<T> alloc()
      {
         static std::atomic<uint64_t>     seed(0);
         static thread_local lehmer64_rng rnd(seed.fetch_add(1, std::memory_order_relaxed));
         int                              attempts = 0;
         do
         {
            if (_header_ptr->average_allocations() > detail::ptrs_per_zone / 2) [[unlikely]]
               ensure_capacity(num_allocated_zones() + 1);

            auto  min_zone       = _header_ptr->min_alloc_zone.load(std::memory_order_relaxed);
            auto* zone_free_base = _free_list_base + ((detail::ptrs_per_zone / 64) * min_zone);

            static constexpr uint32_t n64_per_zone = detail::ptrs_per_zone / 64;
            uint64_t                  start_index  = rnd.next() % n64_per_zone;
            start_index &= ~7ull;  // 8 uint64 = 64B free-list line = 512 slots scanned

#if defined(SAL_STRICT_ATOMICS) || defined(__SANITIZE_THREAD__) || \
    (defined(__has_feature) && __has_feature(thread_sanitizer))
            alignas(128) uint64_t free_bits[8];
            for (int i = 0; i < 8; ++i)
               free_bits[i] = zone_free_base[start_index + i].load(std::memory_order_relaxed);
            uint8_t* free_bytes = reinterpret_cast<uint8_t*>(free_bits);
#else
            uint8_t* free_bytes = reinterpret_cast<uint8_t*>(zone_free_base + start_index);
#endif
            int most_free_byte = max_pop_cnt8_index64(free_bytes);
            if (free_bytes[most_free_byte] == 0) [[unlikely]]
            {
               if (++attempts >= 1000 && (attempts % 1000) == 0)
               {
                  if (attempts < 10000)
                     std::this_thread::yield();
                  else
                     std::this_thread::sleep_for(
                         std::chrono::microseconds(std::min(attempts / 100, 10000)));
               }
               if (attempts >= 1024 * 1024)
                  throw std::runtime_error("colo_alloc: failed to allocate after 1M attempts");
               continue;
            }
            // The byte gives 8 candidate slots; claim a free one in that 8-slot group.
            if (auto a = try_alloc_at_bit(min_zone * detail::ptrs_per_zone + start_index * 64 +
                                          most_free_byte * 8))
               return *a;
         } while (_header_ptr->total_allocations.load(std::memory_order_acquire) < (1ull << 32));
         throw std::runtime_error("colo_alloc: address space exhausted");
      }

      /// Try the hints (co-location), else fall back to a least-filled cacheline.
      colo_allocation<T> alloc(const alloc_hint& hint)
      {
         if (auto a = try_alloc(hint))
            return *a;
         return alloc();
      }

      /// Claim a free slot in the cacheline that CONTAINS `addr` (the hint's cacheline) — the
      /// co-location primitive. Generalized over slots_per_cacheline, which may span several
      /// 64-bit free-list words when sizeof(T) is small (128 slots/line for u8 = 2 words).
      std::optional<colo_allocation<T>> try_alloc(ptr_address addr)
      {
         if (is_reserved_ptr_address(addr)) [[unlikely]]
            return std::nullopt;
         const uint32_t cl_base = *addr & ~(slots_per_cacheline - 1);  // first slot of the line
         const uint32_t cl_end  = cl_base + slots_per_cacheline;       // exclusive
         for (uint32_t b = cl_base; b < cl_end;)
         {
            const uint32_t word = b / 64;
            const uint32_t off  = b % 64;
            const uint32_t span = std::min<uint32_t>(64 - off, cl_end - b);  // bits this word
            const uint64_t mask = (span == 64) ? ~0ull : (((1ull << span) - 1) << off);
            if (auto a = try_claim_in_word(word, mask))
               return a;
            b += span;
         }
         return std::nullopt;
      }

      /// Try each hint cacheline in a per-thread rotating order (spreads fills evenly).
      std::optional<colo_allocation<T>> try_alloc(const alloc_hint& hint)
      {
         const uint32_t n = uint32_t(hint.size());
         if (n == 0)
            return std::nullopt;
         static thread_local uint32_t s_rot = 0;
         uint32_t                     idx   = s_rot++ % n;
         for (uint32_t k = 0; k < n; ++k)
         {
            const ptr_address a = hint[idx];
            if (++idx == n) idx = 0;
            if (is_reserved_ptr_address(a))
               continue;
            if (auto r = try_alloc(a))
               return r;
         }
         return std::nullopt;
      }

      /// Compact-path alloc: co-locate within a region. Try the sibling offsets' cachelines
      /// first (rotating start), then fall back to ANY free slot in the same region. Returns
      /// nullopt only if the entire region is full — the caller then picks a new region and
      /// re-homes its (still-unique, at birth) counters. Never spills outside the region.
      std::optional<colo_allocation<T>> try_alloc(const implicit_ref_hint& h)
      {
         const uint32_t n = uint32_t(h.within.size());
         if (n)
         {
            static thread_local uint32_t s_rot = 0;
            uint32_t                     idx   = s_rot++ % n;
            for (uint32_t k = 0; k < n; ++k)
            {
               const ptr_address full = combine(h.region, h.within[idx]);
               if (++idx == n) idx = 0;
               if (auto a = try_alloc(full))  // cacheline co-location off this sibling
                  return a;
            }
         }
         return try_alloc_in_region(h.region);  // fall back to anywhere in the region
      }

      /// Claim any free slot in `region` (65536 slots = 1024 free-list words). nullopt if full
      /// or the region's backing zone isn't allocated yet.
      std::optional<colo_allocation<T>> try_alloc_in_region(implicit_ref_region region)
      {
         const uint32_t base_slot = uint32_t(region) * implicit_ref_region_slots;
         if (base_slot / detail::ptrs_per_zone >= num_allocated_zones())
            return std::nullopt;  // region not backed by an allocated zone
         const uint32_t w0 = base_slot / 64;
         const uint32_t nw = implicit_ref_region_slots / 64;  // 1024 words
         // rotating start within the region to spread fills / cut contention
         static thread_local uint32_t s_rot = 0;
         const uint32_t               start = s_rot++ % nw;
         for (uint32_t i = 0; i < nw; ++i)
         {
            const uint32_t w = w0 + ((start + i) % nw);
            if (auto a = try_claim_in_word(w, ~0ull))
               return a;
         }
         return std::nullopt;
      }

      /// @pre slot is allocated; mark it free.
      void free(ptr_address address) noexcept
      {
         assert((*address / detail::ptrs_per_zone) <
                _header_ptr->allocated_zones.load(std::memory_order_relaxed));
         release_address(address);
         _header_ptr->dec_alloc_count(address);
      }

      T&  get(ptr_address address) noexcept
      {
         assert(address != null_ptr_address);
         return _ptr_base[*address];
      }
      T* try_get(ptr_address address) noexcept
      {
         if ((*address / detail::ptrs_per_zone) >=
             _header_ptr->allocated_zones.load(std::memory_order_relaxed))
            return nullptr;
         if (is_free(address))
            return nullptr;
         return &_ptr_base[*address];
      }

      uint64_t used() const { return _header_ptr->total_allocations.load(std::memory_order_relaxed); }
      uint32_t num_allocated_zones() const
      {
         return _header_ptr->allocated_zones.load(std::memory_order_relaxed);
      }
      uint32_t current_max_address_count() const
      {
         return detail::ptrs_per_zone * num_allocated_zones();
      }

      /// DEBUG/recovery: visit every non-free slot as visitor(ptr_address, T&).
      template <typename Visitor>
      void for_each_allocated(Visitor&& visitor)
      {
         auto nz = num_allocated_zones();
         for (uint32_t z = 0; z < nz; ++z)
         {
            uint32_t zone_base = z * detail::ptrs_per_zone;
            for (uint32_t i = 0; i < detail::ptrs_per_zone / 64; ++i)
            {
               uint64_t free_bits =
                   _free_list_base[zone_base / 64 + i].load(std::memory_order_relaxed);
               for (uint32_t bit = 0; bit < 64; ++bit)
               {
                  if (free_bits & (1ULL << bit))
                     continue;
                  ptr_address addr(zone_base + i * 64 + bit);
                  visitor(addr, _ptr_base[*addr]);
               }
            }
         }
      }

      bool is_free(ptr_address address)
      {
         return (_free_list_base[*address / 64].load(std::memory_order_relaxed) &
                 (1ull << (*address % 64)));
      }

     private:
      // Claim one free bit selected by `mask` within free-list word `word`; CAS to take it.
      std::optional<colo_allocation<T>> try_claim_in_word(uint32_t word, uint64_t mask)
      {
         std::atomic<uint64_t>& free_list = _free_list_base[word];
         uint64_t               flist     = free_list.load();
         uint64_t               cand      = flist & mask;
         int                    spins     = 0;
         while (cand)
         {
            int      index = std::countr_zero(cand);
            uint64_t bit   = 1ull << index;
            if (not free_list.compare_exchange_strong(flist, flist ^ bit,
                                                      std::memory_order_acquire)) [[unlikely]]
            {
               cand = flist & mask;
               if (++spins > 100)
                  return std::nullopt;  // heavy contention — let the caller pick elsewhere
               continue;
            }
            ptr_address ptr(word * 64 + index);
            return colo_allocation<T>{{ptr, _header_ptr->inc_alloc_count(ptr)}, &_ptr_base[*ptr]};
         }
         return std::nullopt;
      }
      // Claim any free slot in the 8-slot group beginning at bit `base` (used by alloc()).
      std::optional<colo_allocation<T>> try_alloc_at_bit(uint32_t base)
      {
         const uint32_t word = base / 64;
         const uint32_t off  = base % 64;
         return try_claim_in_word(word, 0xffull << off);
      }

      void ensure_capacity(uint32_t req_zones)
      {
         auto azone = _header_ptr->allocated_zones.load(std::memory_order_relaxed);
         if (azone == detail::max_allocated_zones)
            return;
         std::lock_guard<std::mutex> lock(_grow_mutex);
         azone = _header_ptr->allocated_zones.load(std::memory_order_acquire);
         if (azone == detail::max_allocated_zones)
            return;

         auto num_zones = _zone_allocator->reserve(req_zones, true);
         _zone_free_list->reserve(req_zones, true);
         _zone_free_list->resize(num_zones);
         _zone_allocator->resize(num_zones);

         if (_ptr_base != nullptr)
         {
            auto* nb = _zone_allocator->template get<T>(block_allocator::offset_ptr(0));
            if (nb != _ptr_base)
               throw std::runtime_error("colo_alloc: zone base moved after resize");
         }
         if (_free_list_base != nullptr)
         {
            auto* nf = _zone_free_list->template get<std::atomic<uint64_t>>(
                block_allocator::offset_ptr(0));
            if (nf != _free_list_base)
               throw std::runtime_error("colo_alloc: free-list base moved after resize");
         }

         while (azone < num_zones)
         {
            char* c = _zone_allocator->get(block_allocator::block_number(azone));
            char* f = _zone_free_list->get(block_allocator::block_number(azone));
            std::memset(c, 0x00, _zone_allocator->block_size());  // data slots zeroed (T==0 = clean)
            std::memset(f, 0xff, _zone_free_list->block_size());  // all bits set = all free
            ++azone;
         }
         azone = _header_ptr->allocated_zones.load(std::memory_order_relaxed);
         while (azone < num_zones)
            _header_ptr->allocated_zones.compare_exchange_strong(azone, num_zones,
                                                                 std::memory_order_release);
         _header_ptr->min_alloc_zone.store(azone - 1, std::memory_order_relaxed);
      }

      void release_address(ptr_address address)
      {
         assert(not is_free(address));
         auto prior = _free_list_base[*address / 64].fetch_add(1ull << (*address % 64),
                                                               std::memory_order_release);
         assert(not(prior & (1ull << (*address % 64))));
         (void)prior;
      }

      std::unique_ptr<block_allocator> _zone_allocator;
      std::unique_ptr<block_allocator> _zone_free_list;
      std::unique_ptr<mapping>         _header;
      std::filesystem::path            _dir;
      detail::ptr_alloc_header*        _header_ptr     = nullptr;
      T*                               _ptr_base       = nullptr;
      std::atomic<uint64_t>*           _free_list_base = nullptr;
      std::mutex                       _grow_mutex;
   };
}  // namespace sal

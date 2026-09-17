// sal_hybrid::control_block — the MOVABLE + BIG region indirection (design §4.3).
//
// Copy-and-evolved from sal/control_block.hpp. The spec's four mutations land
// here as ONE struct change: `type:6` is gone (the parent branch tag carries node
// type/region, §7), `zone:2` + a wider `offset:44` take its place. A branch
// pointer into a movable/BIG leaf is a `ptr_address` = an INDEX into the
// control-block array; `resolve(pa)` reads the CB's {zone, offset} and returns
// `zone_base[zone] + offset · zone_stride[zone]`. The compactor relocates an
// object by CAS-rewriting ONLY `offset` (cas_move) — so the u32 ptr_address in
// every parent branch stays valid across relocation. THIS indirection is what
// makes the movable region movable and gives 2^32 objects each addressing a
// 2^44·grain = 512 TB zone (vs the direct-grain-index shortcut's 128 GB cap).
//
// Liveness v1 = refcount (the ONE sanctioned deviation from death+version-GC):
// `ref` is the object's liveness here exactly as in sal — retain/release/
// try_retain stay live. The spec keeps `ref` anyway (for smart_ptr, §4.3), so the
// later version-GC swap only adds a parallel death_ver[] and stops the COW path
// from touching `ref`; the CB struct itself does not change.
#pragma once
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <ostream>
#include <stdexcept>

namespace sal_hybrid
{
   // Concurrent-overshoot headroom for the optimistic fetch_add refcount path
   // (mirrors sal::max_threads): ref + transient over-shoot must stay < 2^16.
   inline constexpr uint32_t cb_max_threads = 256;

   // Which zone an `offset` addresses. The READ path never consults this (the
   // branch tag already routes by region, §7.2) — it rides here for the tag-less
   // consumers (the MFU read-cache promoter and the compactor), §4.3.
   enum class cb_zone : uint8_t
   {
      movable = 0,  // .segs, relocatable (compactor may cas_move)
      big     = 1,  // .big, never relocated
      // 2,3 reserved (second movable tier / stored-subtree-root reference) — not v1
   };

   // An 8-byte packed control block (atomic). One per movable/BIG object; its
   // array index IS the object's ptr_address (stable across relocation).
   struct control_block
   {
     private:
      std::atomic<uint64_t> _data;

     public:
      control_block() noexcept : _data(0) {}

      struct control_block_data
      {
         uint64_t ref           : 16;  // smart_ptr/structural pin count (v1 liveness); baseline 1.
                                       //   Overshoot headroom: max_ref_count guard (cb_max_threads).
         uint64_t active        : 1;   // MFU: object was read (read-cache sampling)
         uint64_t pending_cache : 1;   // MFU: pending promotion to the pinned/hot cache
         uint64_t zone          : 2;   // cb_zone of `offset`: movable | big | 2 reserved
         uint64_t offset        : 44;  // zone-relative location in grain units:
                                       //   addr = zone_base[zone] + offset · zone_stride[zone]
                                       //   2^44 · 32 B = 512 TB per zone (§4.3)

         control_block_data() noexcept : ref(0), active(0), pending_cache(0), zone(0), offset(0) {}
         control_block_data(uint64_t value) noexcept { from_int(value); }
         uint64_t to_int() const noexcept { return std::bit_cast<uint64_t>(*this); }
         void from_int(uint64_t value) noexcept { *this = std::bit_cast<control_block_data>(value); }

         control_block_data& set_ref(uint64_t r) noexcept { ref = r; return *this; }
         control_block_data& set_zone(cb_zone z) noexcept { zone = uint64_t(z); return *this; }
         control_block_data& set_offset(uint64_t o) noexcept { offset = o; return *this; }
         control_block_data& set_active(bool a) noexcept { active = a; return *this; }
         control_block_data& set_pending_cache(bool p) noexcept { pending_cache = p; return *this; }
         cb_zone             zone_of() const noexcept { return cb_zone(zone); }

         friend std::ostream& operator<<(std::ostream& os, const control_block_data& d)
         {
            return os << "{ref:" << d.ref << " zone:" << d.zone << " off:" << d.offset
                      << " active:" << d.active << " pending_cache:" << d.pending_cache << "}";
         }
      };
      static_assert(sizeof(control_block_data) == 8, "control_block_data must be 8 bytes");

      // A FREE/dead CB has ref==0 (retain/try_retain reject it); the CB allocator's
      // free list stashes its next-free ptr_address in `offset` while ref==0.
      static constexpr uint64_t max_ref_count = (1ull << 16) - cb_max_threads;

      // ── reads ──────────────────────────────────────────────────────────────
      control_block_data load(std::memory_order o = std::memory_order_relaxed) const noexcept
      {
         return control_block_data(_data.load(o));
      }
      uint32_t use_count() const noexcept { return load().ref; }
      bool     unique() const noexcept { return use_count() == 1; }
      cb_zone  zone() const noexcept { return load(std::memory_order_relaxed).zone_of(); }
      uint64_t offset() const noexcept { return load(std::memory_order_acquire).offset; }
      bool     active() const noexcept { return load().active; }
      bool     pending_cache() const noexcept { return load().pending_cache; }
      bool     is_free() const noexcept { return load().ref == 0; }  // ref==0 ⇒ dead/free

      // ── writes ─────────────────────────────────────────────────────────────
      void store(control_block_data v, std::memory_order o = std::memory_order_relaxed) noexcept
      {
         _data.store(v.to_int(), o);
      }
      // Claim a free slot for a freshly-allocated object (zone, offset, ref=1).
      void reset(cb_zone z, uint64_t off, int ref = 1,
                 std::memory_order o = std::memory_order_release) noexcept
      {
         control_block_data d;
         d.set_zone(z).set_offset(off).set_ref(ref);
         store(d, o);
      }
      // ── refcount (v1 liveness; identical contract to sal) ────────────────────
      control_block_data retain()
      {
         control_block_data prior(_data.fetch_add(1, std::memory_order_relaxed));
         if (prior.ref >= max_ref_count)
         {
            _data.fetch_sub(1, std::memory_order_relaxed);
            throw std::runtime_error("sal_hybrid: control_block ref overflow");
         }
         assert(prior.ref > 0 && "retain on a dead (ref==0) control block");
         return prior;
      }
      // lock-free pin attempt: bump only if still alive (ref>0). false ⇒ dead.
      [[nodiscard]] bool try_retain() noexcept
      {
         uint64_t           expect = _data.load(std::memory_order_relaxed);
         control_block_data prior, next;
         do
         {
            prior.from_int(expect);
            if (prior.ref == 0 || prior.ref >= max_ref_count)
               return false;
            next = prior;
            next.set_ref(prior.ref + 1);
         } while (!_data.compare_exchange_weak(expect, next.to_int(), std::memory_order_acquire));
         return true;
      }
      // Decrement; returns the state BEFORE the decrement. prior.ref==1 ⇒ this was
      // the last reference: the caller frees the object at cb.offset() (re-read with
      // acquire — ref is now 0 so the compactor won't relocate it). acq_rel gives
      // the freer the up-to-date offset published by any prior cas_move.
      control_block_data release() noexcept
      {
         control_block_data prior(_data.fetch_sub(1, std::memory_order_acq_rel));
         assert(prior.ref > 0 && "release of a dead control block");
         return prior;
      }

      // ── relocation (the whole point) ─────────────────────────────────────────
      // CAS the offset to `desired` iff it still equals `expected` AND ref>0.
      // Other fields (ref, MFU bits — moved by concurrent retain/release) are
      // preserved. This is how the compactor moves an object without touching the
      // ptr_address held by every parent branch.
      bool cas_move(uint64_t expected_off, uint64_t desired_off) noexcept
      {
         uint64_t           expect = _data.load(std::memory_order_relaxed);
         control_block_data prior;
         do
         {
            prior.from_int(expect);
            if (prior.offset != expected_off || prior.ref == 0) [[unlikely]]
               return false;
            prior.set_offset(desired_off);
         } while (!_data.compare_exchange_weak(expect, prior.to_int(), std::memory_order_seq_cst));
         return true;
      }

      // ── MFU read-cache sampling (port of sal's; bits feed the hot/cold promoter) ─
      bool try_mark_read_cache_pending() noexcept
      {
         uint64_t           expected = _data.load(std::memory_order_relaxed);
         control_block_data updated(expected);
         if (updated.pending_cache)
            return false;
         if (updated.active)
            return _data.compare_exchange_weak(expected, updated.set_pending_cache(true).to_int(),
                                               std::memory_order_relaxed);
         _data.compare_exchange_weak(expected, updated.set_active(true).to_int(),
                                     std::memory_order_relaxed);
         return false;
      }
      void clear_mfu() noexcept
      {
         uint64_t           expected = _data.load(std::memory_order_relaxed);
         control_block_data updated;
         do
         {
            updated.from_int(expected);
            updated.set_active(false).set_pending_cache(false);
         } while (!_data.compare_exchange_weak(expected, updated.to_int(), std::memory_order_release));
      }
   };
   static_assert(sizeof(control_block) == 8, "control_block must be 8 bytes");

   using control_block_data = control_block::control_block_data;

}  // namespace sal_hybrid

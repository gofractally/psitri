// sal2::movable_region<Grain> — the MOVABLE leaf region, behind a control block
// (design §6.2 / §7.3). THE fix for the spec deviation that crashed Silkworm: a
// leaf branch pointer is a `ptr_address` = INDEX into a control_block array, not
// a direct grain index. resolve(pa) = grain_base + cb[pa].offset·Grain, so the
// compactor can relocate an object's bytes (cas_move rewrites cb[pa].offset)
// while the ptr_address in every parent branch stays valid — and the addressable
// space is 2^32 objects × (2^44·Grain = 512 TB), not 2^32 grains = 128 GB.
//
// It COMPOSES (does not modify) the proven `var_allocator<Grain>` as a pure grain
// byte-arena: var owns segment bump/recycle/write-around + a per-grain occupancy
// marker (its u8 array, used here as 0/1 occupied — NOT as the liveness count);
// the control_block holds the real refcount (v1 liveness, the sanctioned
// deviation) + the relocatable offset. var_allocator is left untouched so rcart
// (the frozen refcount baseline) keeps using it directly — this is the A/B.
//
// Public surface mirrors var_allocator's (alloc/get/retain/try_retain/
// release_one/free/unique/grains_for/start_session/live_*) but every index is a
// ptr_address, so the gcart engine swaps `var_allocator<32> sal` →
// `movable_region<32> sal` with no call-site churn (the engine treats the u32
// opaquely and always round-trips it through these methods).
#pragma once
#include <cstdint>
#include <utility>

#include <sal2/control_block_alloc.hpp>
#include <sal2/var_allocator.hpp>

namespace sal2
{
   template <uint32_t Grain>
   class movable_region
   {
     public:
      // u64 grain index: the data cap is the segment cap, not 2^32 grains (128 GB).
      // The CB's 44-bit offset holds it. This is what lifts the Silkworm wall.
      using arena_t = var_allocator<Grain, uint64_t>;
      static constexpr uint32_t grain            = Grain;
      static constexpr uint32_t max_object_bytes = arena_t::max_object_bytes;

      struct config
      {
         std::filesystem::path dir;
         std::string           name;          // file stem; arena=.dat/.rc, CBs=<name>c.ctrl
         uint32_t              max_segments = 65536;
      };

      explicit movable_region(config cfg)
          : _arena({.dir = cfg.dir, .name = cfg.name, .max_segments = cfg.max_segments}),
            _cb({.dir = cfg.dir, .name = cfg.name + "c", .max_segments = 4096})
      {
      }

      static uint32_t grains_for(uint32_t bytes) noexcept { return arena_t::grains_for(bytes); }

      // ── resolve: ptr_address → object bytes (THE indirection) ────────────────
      void* get(ptr_address pa) noexcept
      {
         return _arena.get(_cb.cb_at(pa)->offset());  // offset = u64 grain index
      }
      const void* get(ptr_address pa) const noexcept
      {
         return _arena.get(_cb.cb_at(pa)->offset());
      }

      // ── liveness = the control block's refcount (v1) ─────────────────────────
      uint32_t ref(ptr_address pa) const noexcept { return _cb.cb_at(pa)->use_count(); }
      bool     unique(ptr_address pa) const noexcept { return _cb.cb_at(pa)->use_count() == 1; }
      void     retain(ptr_address pa) { _cb.cb_at(pa)->retain(); }
      [[nodiscard]] bool try_retain(ptr_address pa) noexcept { return _cb.cb_at(pa)->try_retain(); }
      // returns true iff this was the LAST reference (caller then calls free).
      [[nodiscard]] bool release_one(ptr_address pa) noexcept
      {
         return _cb.cb_at(pa)->release().ref == 1;
      }

      // Free a dead object (ref already 0 via release_one): release the grain run
      // back to the arena (segment recycle bookkeeping) and recycle the CB. `bytes`
      // is the object body size the caller allocated (== what alloc saw).
      void free(ptr_address pa, uint32_t bytes) noexcept
      {
         _arena.free(_cb.cb_at(pa)->offset(), bytes);  // u64 grain index; clears occupancy + accounting
         _cb.free(pa);
      }

      uint32_t num_segments() const noexcept { return _arena.num_segments(); }
      uint64_t live_slots() noexcept { return _arena.live_slots(); }
      uint64_t live_bytes() noexcept { return _arena.live_bytes(); }
      uint64_t cb_live() const noexcept { return _cb.live_count(); }

      class session;
      session start_session() { return session(this); }

     private:
      friend class session;
      arena_t             _arena;  // grain byte-store (occupancy + recycle); NOT the liveness
      control_block_alloc _cb;     // ptr_address space: refcount (v1 liveness) + relocatable offset
   };

   // A writer session: allocate the grain bytes via the arena session, then bind a
   // control block to that grain offset and return the CB's ptr_address.
   template <uint32_t Grain>
   class movable_region<Grain>::session
   {
     public:
      session(const session&)            = delete;
      session& operator=(const session&) = delete;
      session(session&&)                 = default;

      // allocate an object of `bytes` (≤ max_object_bytes); CB ref stamped to 1.
      ptr_address alloc(uint32_t bytes)
      {
         const uint64_t    g  = _as.alloc(bytes);  // u64 grain slot (arena occupancy = 1)
         const ptr_address pa = _r->_cb.alloc(cb_zone::movable, g);
         return pa;
      }

     private:
      friend class movable_region<Grain>;
      explicit session(movable_region* r) : _r(r), _as(r->_arena.start_session()) {}
      movable_region*                  _r;
      typename arena_t::session        _as;
   };

}  // namespace sal2

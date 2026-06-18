// sal2::big_region<Grain> — the BIG leaf region behind a control block, zone=big
// (design §6.3). Like movable_region, a big leaf's branch pointer is a
// ptr_address (CB index) — giving big objects a stable 4-byte identity and a
// uniform resolve path with movable — but zone=big tells the compactor/MFU
// promoter to SKIP it: .big is never relocated (objects are already ≥ OS page).
//
// Composes the unchanged big_allocator<Grain> as the byte store (its inline
// refcount is used only as occupancy; the control_block holds the v1 liveness).
#pragma once
#include <cstdint>

#include <sal2/big_allocator.hpp>
#include <sal2/control_block_alloc.hpp>

namespace sal2
{
   template <uint32_t Grain>
   class big_region
   {
     public:
      using arena_t = big_allocator<Grain>;
      static constexpr uint32_t grain = Grain;

      struct config
      {
         std::filesystem::path dir;
         std::string           name;          // arena=<name>.big, CBs=<name>c.ctrl
         uint32_t              max_segments = 4096;
      };

      explicit big_region(config cfg)
          : _arena({.dir = cfg.dir, .name = cfg.name, .max_segments = cfg.max_segments}),
            _cb({.dir = cfg.dir, .name = cfg.name + "c", .max_segments = 4096})
      {
      }

      static uint32_t grains_for(uint32_t body_bytes) noexcept { return arena_t::grains_for(body_bytes); }

      ptr_address alloc(uint32_t body_bytes)
      {
         const uint32_t g = _arena.alloc(body_bytes);  // grain handle (arena occupancy=1)
         return _cb.alloc(cb_zone::big, g);
      }
      void*       get(ptr_address pa) noexcept { return _arena.get(uint32_t(_cb.cb_at(pa)->offset())); }
      const void* get(ptr_address pa) const noexcept
      {
         return _arena.get(uint32_t(_cb.cb_at(pa)->offset()));
      }

      uint32_t ref(ptr_address pa) const noexcept { return _cb.cb_at(pa)->use_count(); }
      bool     unique(ptr_address pa) const noexcept { return _cb.cb_at(pa)->use_count() == 1; }
      void     retain(ptr_address pa) { _cb.cb_at(pa)->retain(); }
      [[nodiscard]] bool try_retain(ptr_address pa) noexcept { return _cb.cb_at(pa)->try_retain(); }
      [[nodiscard]] bool release_one(ptr_address pa) noexcept
      {
         return _cb.cb_at(pa)->release().ref == 1;
      }
      void free(ptr_address pa) noexcept
      {
         _arena.free(uint32_t(_cb.cb_at(pa)->offset()));  // big_allocator reads its own ngrains
         _cb.free(pa);
      }

      uint64_t live_bytes() noexcept { return _arena.live_bytes(); }
      uint32_t num_segments() const noexcept { return _arena.num_segments(); }

     private:
      arena_t             _arena;
      control_block_alloc _cb;
   };

}  // namespace sal2

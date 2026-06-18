// sal2::control_block_alloc — the ptr_address space (design §4.3 / §6).
//
// A dense, mmap'd array of control_block (8 B each) in a `.ctrl` file; an
// object's ptr_address is its INDEX here. This is the indirection the movable
// (.segs) and BIG (.big) regions share: a parent branch stores a 4-byte
// ptr_address, the leaf store keeps the bytes, and the CB binds the two via
// {zone, offset} — so the compactor relocates the bytes (cas_move rewrites the
// CB's offset) while the parent's ptr_address never changes.
//
// LOCK-FREE hot path (no-heap/hot-path rule): the .ctrl base is a stable
// PROT_NONE reservation (MAP_FIXED commits, never moves), so alloc reads it
// without a lock — claim a free CB via a tagged Treiber free list (ABA-safe) or
// an atomic bump; free pushes to the same list. Only segment GROWTH takes a mutex
// (rare). Liveness is the CB's own `ref` (refcount v1); death_ver[] is added with
// the version-GC swap.
#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <sal/block_allocator.hpp>

#include <sal2/control_block.hpp>

namespace sal2
{
   using ptr_address = uint32_t;  // index into the control-block array

   class control_block_alloc
   {
     public:
      static constexpr uint64_t    segment_bytes = 8u * 1024u * 1024u;  // 8 MB commit unit
      static constexpr uint32_t    cbs_per_seg   = uint32_t(segment_bytes / sizeof(control_block));
      static constexpr ptr_address null_pa       = 0xFFFFFFFFu;  // free-list terminator / "no CB"
      static_assert(sizeof(control_block) == 8);

      struct config
      {
         std::filesystem::path dir;
         std::string           name;          // file stem, e.g. "ctrl"
         uint32_t              max_segments = 4096;  // 4096·1M = 2^32 CBs = full ptr_address space
      };

      explicit control_block_alloc(config cfg) : _cfg(std::move(cfg))
      {
         std::filesystem::create_directories(_cfg.dir);
         _data = std::make_unique<sal::block_allocator>(_cfg.dir / (_cfg.name + ".ctrl"),
                                                        segment_bytes, _cfg.max_segments,
                                                        sal::access_mode::read_write);
         _base = static_cast<control_block*>(const_cast<void*>(_data->mapped_base()));  // STABLE
         _free_head.store(pack_(0, null_pa), std::memory_order_relaxed);
         adopt_existing_();
      }

      // reopen: the .ctrl bytes (CBs + their {zone,offset,ref}) survive a clean
      // close, so every old ptr_address still resolves. Resume the bump cursor at
      // the end of the committed region; the free list starts empty (old freed CBs
      // aren't re-pooled — a later GC/recovery pass can, §10).
      void adopt_existing_() noexcept
      {
         const uint32_t existing = uint32_t(_data->mapped_blocks());
         if (existing == 0)
            return;
         _data->resize(existing);
         _base = static_cast<control_block*>(const_cast<void*>(_data->mapped_base()));
         _nseg.store(existing, std::memory_order_relaxed);
         _committed_cbs.store(existing * cbs_per_seg, std::memory_order_relaxed);
         _bump.store(existing * cbs_per_seg, std::memory_order_relaxed);  // append after committed
      }

      control_block*       cb_at(ptr_address pa) noexcept { return _base + pa; }
      const control_block* cb_at(ptr_address pa) const noexcept { return _base + pa; }

      // Claim a CB for a new object at (zone, offset). Lock-free (grow is locked).
      ptr_address alloc(cb_zone zone, uint64_t offset)
      {
         // 1) tagged Treiber pop — reuse a freed CB (ABA-safe via the tag).
         uint64_t head = _free_head.load(std::memory_order_acquire);
         while (idx_(head) != null_pa)
         {
            const ptr_address idx  = idx_(head);
            const uint32_t    next = uint32_t(_base[idx].load(std::memory_order_relaxed).offset);
            const uint64_t    nh   = pack_(tag_(head) + 1, next);
            if (_free_head.compare_exchange_weak(head, nh, std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
            {
               _base[idx].reset(zone, offset, 1);
               _live.fetch_add(1, std::memory_order_relaxed);
               return idx;
            }
         }
         // 2) bump a fresh CB (atomic; grow the .ctrl reservation if needed).
         const ptr_address pa = _bump.fetch_add(1, std::memory_order_relaxed);
         if (pa >= _committed_cbs.load(std::memory_order_acquire))
            grow_to_(pa);
         _base[pa].reset(zone, offset, 1);
         _live.fetch_add(1, std::memory_order_relaxed);
         return pa;
      }

      // Recycle a CB (its object is gone). Lock-free Treiber push. ref=0 marks it
      // dead (a stray try_retain/retain rejects it); `offset` holds the next index.
      void free(ptr_address pa) noexcept
      {
         uint64_t                          head = _free_head.load(std::memory_order_relaxed);
         control_block::control_block_data d;  // ref=0 (free marker)
         for (;;)
         {
            d.set_ref(0).set_offset(idx_(head));  // next = current head index
            _base[pa].store(d, std::memory_order_release);
            const uint64_t nh = pack_(tag_(head) + 1, pa);
            if (_free_head.compare_exchange_weak(head, nh, std::memory_order_acq_rel,
                                                 std::memory_order_relaxed))
               break;
         }
         _live.fetch_sub(1, std::memory_order_relaxed);
      }

      uint64_t live_count() const noexcept { return _live.load(std::memory_order_relaxed); }
      uint32_t num_segments() const noexcept { return _nseg.load(std::memory_order_relaxed); }

     private:
      static constexpr uint64_t pack_(uint32_t tag, ptr_address idx) noexcept
      {
         return (uint64_t(tag) << 32) | idx;
      }
      static constexpr ptr_address idx_(uint64_t h) noexcept { return ptr_address(h); }
      static constexpr uint32_t    tag_(uint64_t h) noexcept { return uint32_t(h >> 32); }

      void grow_to_(ptr_address pa)  // commit segments until pa is in-bounds (rare)
      {
         std::lock_guard<std::mutex> lk(_grow_mx);
         while (pa >= _committed_cbs.load(std::memory_order_relaxed))
         {
            if (_nseg.load(std::memory_order_relaxed) >= _cfg.max_segments)
               throw std::runtime_error("sal2: control_block_alloc full (ptr_address space)");
            _data->alloc();  // commit the next contiguous segment (MAP_FIXED in the reservation)
            _committed_cbs.fetch_add(cbs_per_seg, std::memory_order_release);
            _nseg.fetch_add(1, std::memory_order_relaxed);
         }
      }

      config                                _cfg;
      std::unique_ptr<sal::block_allocator> _data;
      control_block*                        _base = nullptr;  // STABLE reservation base
      std::mutex                            _grow_mx;         // segment growth only
      std::atomic<uint32_t>                 _bump{0};         // next fresh CB index
      std::atomic<uint64_t>                 _free_head{0};    // tagged Treiber head (tag<<32 | idx)
      std::atomic<uint32_t>                 _committed_cbs{0};
      std::atomic<uint32_t>                 _nseg{0};
      std::atomic<uint64_t>                 _live{0};
   };

}  // namespace sal2

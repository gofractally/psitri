// sal2::control_block_alloc — the ptr_address space (design §4.3 / §6).
//
// A dense, mmap'd array of control_block (8 B each) in a `.ctrl` file; an
// object's ptr_address is its INDEX here. This is the indirection the movable
// (.segs) and BIG (.big) regions share: a parent branch stores a 4-byte
// ptr_address, the leaf store keeps the bytes, and the CB binds the two via
// {zone, offset} — so the compactor can relocate the bytes (cas_move rewrites
// the CB's offset) while the parent's ptr_address never changes.
//
// alloc(zone, offset) claims a CB (free-list LIFO, else bump) and returns its
// ptr_address; free(pa) recycles it. The CB array is grown by 8 MB segments via
// sal::block_allocator (PROT_NONE reserve + MAP_FIXED commit → cb_at(pa) is a
// stable pointer; only the bytes the CB *points at* relocate). alloc/free are
// mutex-guarded (correctness-first; CB churn is one op per leaf birth/death, and
// release-to-zero can come from any thread) — a lock-free free list is a later
// optimization. Liveness is the CB's own `ref` (refcount v1); death_ver[] is
// added with the version-GC swap.
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
         _base = static_cast<control_block*>(const_cast<void*>(_data->mapped_base()));
         adopt_existing_();
      }

      // reopen: the .ctrl bytes (CBs + their {zone,offset,ref}) survive a clean
      // close, so every old ptr_address still resolves. Resume the bump cursor at
      // the end of the committed region; the free list starts empty (old freed CBs
      // aren't re-pooled — a later GC/recovery pass can, §10). ref/offset are read
      // back as-is on a clean reopen; a crash needs the §5.6 rebuild (later).
      void adopt_existing_() noexcept
      {
         const uint32_t existing = uint32_t(_data->mapped_blocks());
         if (existing == 0)
            return;
         _data->resize(existing);
         _base          = static_cast<control_block*>(const_cast<void*>(_data->mapped_base()));
         _nseg          = existing;
         _committed_cbs = existing * cbs_per_seg;
         _bump          = _committed_cbs;  // append after committed (free holes not re-pooled)
      }

      control_block*       cb_at(ptr_address pa) noexcept { return _base + pa; }
      const control_block* cb_at(ptr_address pa) const noexcept { return _base + pa; }

      // Claim a CB for a new object at (zone, offset). LOCKED.
      ptr_address alloc(cb_zone zone, uint64_t offset)
      {
         std::lock_guard<std::mutex> lk(_mx);
         ptr_address                 pa;
         if (_free_head != null_pa)
         {  // free-list LIFO: a freed CB is {ref=0, offset=next-free ptr_address}
            pa         = _free_head;
            _free_head = ptr_address(_base[pa].load(std::memory_order_relaxed).offset);
         }
         else
         {
            ensure_committed_(uint64_t(_bump) + 1);
            pa = _bump++;
         }
         _base[pa].reset(zone, offset, /*ref=*/1);
         ++_live;
         return pa;
      }

      // Recycle a CB (its object is gone). LOCKED. ref=0 marks it dead (so a stray
      // try_retain/retain rejects it) and `offset` holds the next free ptr_address.
      void free(ptr_address pa) noexcept
      {
         std::lock_guard<std::mutex> lk(_mx);
         control_block::control_block_data d;  // ref=0 (free marker), offset=next
         d.set_ref(0).set_offset(_free_head);
         _base[pa].store(d, std::memory_order_release);
         _free_head = pa;
         --_live;
      }

      uint64_t live_count() const noexcept { return _live; }
      uint32_t num_segments() const noexcept { return _nseg; }

     private:
      void ensure_committed_(uint64_t need)  // caller holds _mx
      {
         while (uint64_t(_committed_cbs) < need)
         {
            if (_nseg >= _cfg.max_segments)
               throw std::runtime_error("sal2: control_block_alloc full (ptr_address space)");
            _data->alloc();  // commit the next contiguous segment (MAP_FIXED in the reservation)
            _base = static_cast<control_block*>(const_cast<void*>(_data->mapped_base()));
            _committed_cbs += cbs_per_seg;
            ++_nseg;
         }
      }

      config                                _cfg;
      std::unique_ptr<sal::block_allocator> _data;
      control_block*                        _base          = nullptr;  // fixed reservation base
      std::mutex                            _mx;
      ptr_address                           _bump          = 0;        // next fresh CB index
      ptr_address                           _free_head     = null_pa;  // LIFO free list head
      uint32_t                              _committed_cbs = 0;
      uint32_t                              _nseg          = 0;
      uint64_t                              _live          = 0;
   };

}  // namespace sal2

// sal2::var_allocator<Grain> — the SAL/movable region (refcount-mode, v1).
//
// Spec: suffix_value leaf objects (<4KB) live in a 32-byte-grain region (denser
// than the 128B FIX line — the leaf-density lever). The spec's movable region is
// bump+compactor; for the refcount era (Dan: refcount for now) this is a
// variable-run refcounted slab over the same seg_freeset the FIX region uses:
// alloc(nbytes) → a run of ceil(nbytes/Grain) grains; free returns the run to the
// bitmap immediately (churn-safe, no compactor). A 4-byte slot index is the SAL
// ptr_address. (Compactor/movability arrives with the version-GC model later.)
//
// One l0 word per segment caps slots at 64·64·64 = 262144, so a Grain-byte
// segment is 262144·Grain bytes (8 MB at Grain=32); a single object is ≤64 grains
// (one freeset word) = 2 KB at Grain=32 — larger values go BIG.
#pragma once
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <sal/block_allocator.hpp>

#include <sal2/config.hpp>
#include <sal2/freeset.hpp>

namespace sal2
{
   template <uint32_t Grain>
   class var_allocator
   {
     public:
      static constexpr uint32_t grain               = Grain;
      static constexpr uint32_t slots_per_segment   = 64u * 64u * 64u;      // freeset cap
      static constexpr uint64_t segment_bytes       = uint64_t(slots_per_segment) * Grain;
      static constexpr uint32_t max_run             = max_alloc_lines;       // 64 grains/object
      static constexpr uint32_t max_object_bytes    = max_run * Grain;       // 2 KB at Grain=32
      using freeset = seg_freeset_t<slots_per_segment>;
      static_assert((segment_bytes & (segment_bytes - 1)) == 0, "segment must be a power of 2");

      struct config { std::filesystem::path dir; std::string name; uint32_t max_segments = 65536; };
      class session;

      explicit var_allocator(config cfg) : _cfg(std::move(cfg))
      {
         std::filesystem::create_directories(_cfg.dir);
         _data = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".dat"), segment_bytes, _cfg.max_segments, sal::access_mode::read_write);
         _free = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".fs"), uint64_t(std::bit_ceil(sizeof(freeset))), _cfg.max_segments, sal::access_mode::read_write);
         _refs = std::make_unique<sal::block_allocator>(
             _cfg.dir / (_cfg.name + ".rc"), std::bit_ceil(uint64_t(slots_per_segment)), _cfg.max_segments, sal::access_mode::read_write);
         grow_();
         cache_bases_();
      }

      void*       get(uint32_t slot) noexcept { return _base + size_t(slot) * Grain; }
      const void* get(uint32_t slot) const noexcept { return _base + size_t(slot) * Grain; }
      static uint32_t grains_for(uint32_t bytes) noexcept { return (bytes + Grain - 1) / Grain; }

      uint8_t ref(uint32_t i) const noexcept { return _ref_base[i].load(std::memory_order_relaxed); }
      bool    unique(uint32_t i) const noexcept { return ref(i) == 1; }
      void    retain(uint32_t i)
      {
         auto& r = _ref_base[i]; auto c = r.load(std::memory_order_relaxed);
         do { if (c >= ref_ceiling) throw std::runtime_error("sal2: var_allocator refcount overflow"); }
         while (!r.compare_exchange_weak(c, uint8_t(c + 1), std::memory_order_relaxed));
      }
      [[nodiscard]] bool try_retain(uint32_t i) noexcept
      {
         auto& r = _ref_base[i]; auto c = r.load(std::memory_order_acquire);
         do { if (c == 0 || c >= ref_ceiling) return false; }
         while (!r.compare_exchange_weak(c, uint8_t(c + 1), std::memory_order_acq_rel, std::memory_order_acquire));
         return true;
      }
      [[nodiscard]] bool release_one(uint32_t i) noexcept
      {
         const uint8_t prior = _ref_base[i].fetch_sub(1, std::memory_order_acq_rel);
         assert(prior >= 1 && "release of a dead object");
         return prior == 1;
      }
      void free(uint32_t i, uint32_t bytes) noexcept
      {
         const uint32_t s = i / slots_per_segment;
         freeset_(s).free_run(i % slots_per_segment, grains_for(bytes));
      }

      session  start_session() { return session(this); }
      uint32_t num_segments() const noexcept { return _nseg.load(std::memory_order_acquire); }
      freeset& freeset_of(uint32_t s) noexcept { return freeset_(s); }
      uint64_t live_slots() noexcept
      {
         const uint32_t ns = num_segments(); uint64_t used = 0;
         for (uint32_t s = 0; s < ns; ++s) used += slots_per_segment - freeset_(s).free_lines();
         return used;
      }
      uint64_t live_bytes() noexcept { return live_slots() * uint64_t(Grain); }

     private:
      friend class session;
      void cache_bases_() noexcept
      {
         _base      = static_cast<std::byte*>(const_cast<void*>(_data->mapped_base()));
         _free_base = static_cast<std::byte*>(const_cast<void*>(_free->mapped_base()));
         _ref_base  = reinterpret_cast<std::atomic<uint8_t>*>(const_cast<void*>(_refs->mapped_base()));
      }
      freeset& freeset_(uint32_t s) noexcept
      {
         return *reinterpret_cast<freeset*>(_free_base + size_t(s) * std::bit_ceil(sizeof(freeset)));
      }
      uint32_t grow_()
      {
         std::lock_guard<std::mutex> g(_grow_mx);
         const uint32_t s = _nseg.load(std::memory_order_relaxed);
         if (s >= _cfg.max_segments) throw std::runtime_error("sal2: var_allocator full");
         _data->alloc(); _free->alloc(); _refs->alloc(); cache_bases_();
         new (&freeset_(s)) freeset(); freeset_(s).init(true);
         _nseg.store(s + 1, std::memory_order_release);
         return s;
      }

      config                                _cfg;
      std::unique_ptr<sal::block_allocator> _data, _free, _refs;
      std::byte*                            _base = nullptr;
      std::byte*                            _free_base = nullptr;
      std::atomic<uint8_t>*                 _ref_base = nullptr;
      std::atomic<uint32_t>                 _nseg{0};
      std::mutex                            _grow_mx;
   };

   template <uint32_t Grain>
   class var_allocator<Grain>::session
   {
     public:
      session(const session&)            = delete;
      session& operator=(const session&) = delete;

      // allocate an object of `bytes` (≤ max_object_bytes); ref stamped to 1.
      uint32_t alloc(uint32_t bytes)
      {
         const uint32_t n = grains_for(bytes);
         assert(n >= 1 && n <= max_run);
         if (_seg != UINT32_MAX)
         {
            const uint32_t local = _a->freeset_(_seg).alloc(n, _cursor);
            if (local != UINT32_MAX) { _cursor = local + n; return finish_(local); }
         }
         return alloc_slow_(n);
      }

     private:
      friend class var_allocator<Grain>;
      explicit session(var_allocator* a) noexcept : _a(a) {}
      uint32_t alloc_slow_(uint32_t n)
      {
         const uint32_t ns = _a->num_segments();
         for (uint32_t i = 0; i < ns; ++i)
         {
            const uint32_t s = (_seg + 1 + i < ns) ? _seg + 1 + i : _seg + 1 + i - ns;
            if (_a->freeset_(s).free_lines() < n) continue;
            const uint32_t local = _a->freeset_(s).alloc(n, 0);
            if (local != UINT32_MAX) { _seg = s; _cursor = local + n; return finish_(local); }
         }
         const uint32_t s     = _a->grow_();
         const uint32_t local = _a->freeset_(s).alloc(n, 0);
         _seg = s; _cursor = local + n;
         return finish_(local);
      }
      uint32_t finish_(uint32_t local)
      {
         const uint32_t g = _seg * slots_per_segment + local;
         _a->_ref_base[g].store(1, std::memory_order_relaxed);
         return g;
      }
      var_allocator* _a;
      uint32_t       _seg    = UINT32_MAX;
      uint32_t       _cursor = 0;
   };
}  // namespace sal2

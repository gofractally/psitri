// rcart — the spec node taxonomy over the three regions (refcount era).
//
// Routing lives in FIX (cl1 setlist, cl2 double, cl9 full); terminal (suffix,
// value) leaves live in SAL (32B); large values in BIG. A branch is a 4-byte
// region-typed pointer: its 3-bit `branch_type` tag (in the parent's packed
// types[] array) selects the region AND lets dispatch PREFETCH the child's
// cacheline before any load that depends on it. There is no leaf node — a leaf
// is a suffix_value with a 0-length suffix (the term).
//
// Liveness = per-slot refcount in each region (Dan: refcount for now). Mutation
// is copy-construct-rebuild from the stable source into a right-sized slot (the
// new slot bubbles up; COW-to-root); pointer children are retained per region,
// inline-none (suffix_value is its own SAL object, refcounted there).
#pragma once
#include <rcart/branch_tag.hpp>  // branch_type, tag3_get/set, tag3_bytes
#include <ucc/lower_bound.hpp>   // ucc::lower_bound_padded (vectorized in-node scan)
#include <ucc/round.hpp>         // ucc::round_up_multiple (cacheline-rounded leaf alloc)
#include <sal_hybrid/fixed_allocator.hpp>  // FIX routing (cl1/cl2/cl9) — UNCHANGED, direct, no CB
// ── the LEAF region rides the copied SAL allocator ───────────────────────
// (experimental/sal_hybrid/include/sal_hybrid/alloc/* — namespace sal_hybrid — the real machinery:
//  background compactor, MFU read-cache (get_ref SAMPLES the active bit),
//  segment threads, control_block refcount). Routing stays on the from-scratch
//  fixed_allocator above; ONLY suffix_value leaves move here. big_suffix_value
//  STAYS on the standalone sal_hybrid::big_allocator (the fixed, non-moving, malloc-
//  style BIG region — Dan's ruling: do NOT merge big onto SAL).
//  NOTE: movable_region.hpp / big_region.hpp are NOT included — they pull in the
//  from-scratch sal_hybrid/control_block_alloc.hpp which redefines sal_hybrid::ptr_address
//  (uint32_t) and would collide with the typed ptr_address the SAL subsystem
//  declares.
#include <sal_hybrid/big_allocator.hpp>  // BIG region (big_suffix_value) — UNCHANGED
#include <sal_hybrid/alloc/allocator.hpp>
#include <sal_hybrid/alloc/allocator_session_impl.hpp>
#include <sal_hybrid/alloc/read_lock.hpp>
#include <sal_hybrid/alloc/smart_ptr.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace rcart
{
   using sv = std::string_view;
   using bt = branch_type;
   // The LEAF region's SAL pointer is a typed_int<uint32_t> — the SAME width
   // as the engine's u32 routing-branch slot, so a suffix_value leaf branch keeps
   // storing a bare u32 (the ptr_address value) and only reconstructs leaf_addr()
   // at the SAL call boundary. (big_suffix_value stores a big_allocator handle,
   // also a bare u32 — unchanged.) ptr_address 0 is VALID; bt::null (not ptr==0) is
   // the only emptiness sentinel.
   using leaf_addr_t = sal_hybrid::ptr_address;
   inline leaf_addr_t leaf_addr(uint32_t p) noexcept { return leaf_addr_t(p); }

   // value_ref — a leaf value VIEW bundled with the leaf read_lock that keeps it
   // safe to read. The SAL leaf region is compactor-MOVABLE: a raw view into a
   // leaf body is valid ONLY while a read_lock pins the page resident (the compactor
   // can't relocate/evict it). Returning this by value hands the caller a self-pinning
   // view — the safe-read window lasts exactly as long as the value_ref is alive
   // (sal_hybrid::read_lock is copyable: its copy retains, its dtor releases the session
   // read-lock counter). Implicitly converts to sv for transient in-place use. Hold it
   // only BRIEFLY (an open read_lock blocks compaction → bloat); for long-lived needs
   // copy the bytes out and let it drop. This is the radix_engine value_ref pattern.
   class value_ref
   {
      sal_hybrid::read_lock _lock;  // residency pin (copyable: retain/release the read-lock counter)
      sv              _v;

     public:
      value_ref(sal_hybrid::read_lock lk, sv v) noexcept : _lock(lk), _v(v) {}
      operator sv() const noexcept { return _v; }
      sv          view() const noexcept { return _v; }
      const char* data() const noexcept { return _v.data(); }
      size_t      size() const noexcept { return _v.size(); }
      bool        empty() const noexcept { return _v.empty(); }
   };

   // Lock-free MPSC ring of packed (tag,ptr) roots awaiting their release
   // cascade: any thread's smart_ptr release is a producer; the engine's single
   // background release thread is the consumer (Vyukov bounded MPMC cells, used
   // single-consumer). The smart_ptr OWNS its release — it spills here, the bg
   // runs the cascade off the critical path (SAL's design).
   template <uint32_t CAP>
   struct release_ring
   {
      static_assert((CAP & (CAP - 1)) == 0, "CAP must be a power of two");
      struct cell
      {
         std::atomic<uint64_t> seq;
         uint64_t              data;
      };
      alignas(64) cell _buf[CAP];
      alignas(64) std::atomic<uint64_t> _enq{0};
      alignas(64) std::atomic<uint64_t> _deq{0};

      release_ring()
      {
         for (uint32_t i = 0; i < CAP; ++i)
            _buf[i].seq.store(i, std::memory_order_relaxed);
      }

      bool try_push(uint64_t v) noexcept
      {
         uint64_t pos = _enq.load(std::memory_order_relaxed);
         for (;;)
         {
            cell&          c   = _buf[pos & (CAP - 1)];
            const uint64_t seq = c.seq.load(std::memory_order_acquire);
            const int64_t  dif = int64_t(seq) - int64_t(pos);
            if (dif == 0)
            {
               if (_enq.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
               {
                  c.data = v;
                  c.seq.store(pos + 1, std::memory_order_release);
                  return true;
               }
            }
            else if (dif < 0)
               return false;  // full
            else
               pos = _enq.load(std::memory_order_relaxed);
         }
      }
      bool try_pop(uint64_t& v) noexcept  // single consumer (bg)
      {
         uint64_t pos = _deq.load(std::memory_order_relaxed);
         for (;;)
         {
            cell&          c   = _buf[pos & (CAP - 1)];
            const uint64_t seq = c.seq.load(std::memory_order_acquire);
            const int64_t  dif = int64_t(seq) - int64_t(pos + 1);
            if (dif == 0)
            {
               if (_deq.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
               {
                  v = c.data;
                  c.seq.store(pos + CAP, std::memory_order_release);
                  return true;
               }
            }
            else if (dif < 0)
               return false;  // empty
            else
               pos = _deq.load(std::memory_order_relaxed);
         }
      }
      uint64_t usage() const noexcept
      {
         return _enq.load(std::memory_order_acquire) - _deq.load(std::memory_order_acquire);
      }
   };

   // A top-level drop destroys at most this many nodes inline on the releasing
   // thread, then spills the rest to the bg release thread (SAL's bounded-inline
   // budget). The common COW-to-root drop is fully inline; a batch's garbage spills.
   inline constexpr int      inline_destroy_budget = 64;
   inline constexpr uint32_t release_ring_cap      = 1u << 18;  // 256K cells (~4 MB)

   // pack a (tag,ptr) reference into one u64 for the release ring.
   inline uint64_t kv_pack_(bt t, uint32_t p) noexcept
   {
      return (uint64_t(uint8_t(t)) << 32) | p;
   }

   // COW descent mode latch (Dan §5.2): a scoped enum, not a bool. The recursion
   // CACHES whether the subtree is private or shared so it does not re-check every
   // node once committed to <shared>. <unique> reads each routing node's refcount
   // (==1 ⟹ mutate in place); the FIRST shared node clones (retaining children →
   // they become shared) and latches the rest of the descent to <shared>, which
   // clones without re-checking. (Refcount stands in for the spec's birth==V.)
   enum class cow_mode
   {
      shared,
      unique
   };

   // A node reference: which region (by tag) + the 4-byte slot index.
   inline uint32_t lcp(sv a, sv b) noexcept
   {
      const uint32_t n = uint32_t(a.size() < b.size() ? a.size() : b.size());
      uint32_t       i = 0;
      while (i < n && a[i] == b[i])
         ++i;
      return i;
   }

   // ── engine: the four region allocators + the background reclaimer ───────────
   // The engine owns the release thread (SAL puts it in the allocator). A
   // smart_ptr release routes through release_async() — the smart_ptr OWNS its
   // own (async) reclamation; no caller does manual accounting.
   struct engine
   {
      sal_hybrid::fixed_allocator<128>  cl1;  // setlist / nested_setlist  (FIX, direct, no CB)
      sal_hybrid::fixed_allocator<256>  cl2;  // double_setlist / big_nested (FIX)
      sal_hybrid::fixed_allocator<1152> cl9;  // full / node256             (FIX)
      sal_hybrid::big_allocator<64>     big;  // big_suffix_value leaves (>2KB, fixed non-moving BIG)

      // The copied SAL allocator drives the SMALL leaf data (suffix_value): its
      // CB region holds small leaves, the background compactor relocates them, and
      // get_ref SAMPLES the MFU read-cache. Opened in the ctor (init_leaf_), threads
      // started AFTER register_type_ops<sv_leaf>() (the SAL contract). A per-HOLDER
      // session (work tree, reader, drain, admin) drives get_ref/retain/release.
      // big_suffix_value STAYS on `big` (the standalone, non-moving BIG region).
      std::unique_ptr<sal_hybrid::allocator> leaf;

      // Leaf body cap below which a (suffix,value) is a suffix_value (SAL,
      // inline-able into a nested_setlist); larger bodies are big_suffix_value (BIG).
      static constexpr uint32_t sal_max_body = 2048;

      // Per-region segment cap = max addressable size before the u32 slot index
      // wraps (each region's branch pointer is a 4-byte intra-region index). A
      // segment is PROT_NONE-reserved up front and committed on demand, so a large
      // cap costs virtual address space, not RSS — sized here for big workloads
      // (Silkworm-scale sync). FIX caps sit at the u32 ceiling (16384·32MB=512GB);
      // big's grain index ceiling is 8192·8MB=64GB. Override per region via env
      // (RCART_CL1_SEGS / _CL2_ / _CL9_ / _BIG_SEGS) for tuning. The SAL leaf
      // allocator derives its own max_segments from runtime_config.max_database_size
      // (the 8 TiB default at the 32-B grain), so it has NO env knob here.
      static uint32_t seg_cap(const char* env, uint32_t def) noexcept
      {
         if (const char* v = std::getenv(env))
            if (const unsigned long n = std::strtoul(v, nullptr, 10))
               return uint32_t(n);
         return def;
      }
      explicit engine(std::filesystem::path dir)
          : cl1({.dir = dir, .name = "cl1", .max_segments = seg_cap("RCART_CL1_SEGS", 4096)}),
            cl2({.dir = dir, .name = "cl2", .max_segments = seg_cap("RCART_CL2_SEGS", 2048)}),
            cl9({.dir = dir, .name = "cl9", .max_segments = seg_cap("RCART_CL9_SEGS", 2048)}),
            big({.dir = dir, .name = "big", .max_segments = seg_cap("RCART_BIG_SEGS", 8192)}),
            _dir(std::move(dir))
      {
         // The SMALL-leaf allocator (the copied SAL) opens its own files under a
         // "leaf" subdir: register sv_leaf type ops BEFORE start_background_threads
         // (the SAL contract — the compactor/release loops read the non-atomic
         // type-ops array on every object), then start the background compactor + MFU
         // promoter. Out-of-line (init_leaf_) because sv_leaf is declared after engine.
         // (A/B: rcart's own bg release thread stays disabled — sync release.)
         init_leaf_();
      }

      ~engine()
      {
         if (_bg.joinable())
         {
            _bg_stop.store(true, std::memory_order_release);
            _bg.join();
         }
         // Drop the engine-owned leaf sessions BEFORE the leaf allocator (they
         // reference it). Assigning a null ptr ends the session before `leaf` dies.
         _admin_ses = sal_hybrid::allocator_session_ptr{nullptr};
         _bg_ses    = sal_hybrid::allocator_session_ptr{nullptr};
         // The leaf allocator destructor stops its own background threads.
      }

      // ── leaf sessions: one per session-HOLDER, never a global map ──────────────
      // SAL sessions are single-thread-owned; each holder (a read_session's
      // cursors, a work tree's writer, the database drain) OWNS its session via
      // leaf_create_session() and uses it only on that one thread. No global map,
      // no mutex; a read's protection is the snapshot pin, not a lock.
      sal_hybrid::allocator_session_ptr leaf_create_session() { return leaf->create_session(); }
      // Cold single-thread admin path (leaf_live_bytes_): an engine-owned session.
      sal_hybrid::allocator_session& admin_ses() noexcept { return *_admin_ses; }

      void init_leaf_();  // out-of-line (needs sv_leaf complete)

      // raw live bytes; drain() first for an accurate (settled) figure.
      uint64_t live_bytes_raw() noexcept
      {
         return cl1.live_bytes() + cl2.live_bytes() + cl9.live_bytes() + leaf_live_bytes_() +
                big.live_bytes();
      }
      uint64_t live_bytes() noexcept
      {
         drain();
         return live_bytes_raw();
      }
      uint64_t leaf_live_bytes_() noexcept;  // out-of-line (SAL live set + big)

      // smart_ptr release entry: bounded-inline cascade, spill the rest to the bg.
      void release_async(bt t, uint32_t p) noexcept;
      // block until every spilled cascade has finished (measurement barrier).
      void drain() noexcept
      {
         while (_rq.usage() != 0 || _bg_inflight.load(std::memory_order_acquire) != 0)
            std::this_thread::yield();
      }

      // Spill functor handed to release_async_: push the un-decremented reference
      // to the bg; if the ring is full, finish inline (backpressure, never block).
      struct spill_fn
      {
         engine* e;
         void    operator()(bt t, uint32_t p) const noexcept;
      };
      spill_fn _spill{this};

     private:
      void bg_loop_() noexcept;

      std::filesystem::path          _dir;  // engine dir (leaf allocator lives under _dir/"leaf")
      // Engine-owned leaf sessions (created in init_leaf_ AFTER `leaf` is up, dropped
      // in ~engine BEFORE `leaf`): the cold admin/main-thread session and (kept for
      // the dormant bg ring) the reclaimer session. Every other holder owns its own.
      sal_hybrid::allocator_session_ptr    _bg_ses{nullptr};
      sal_hybrid::allocator_session_ptr    _admin_ses{nullptr};
      release_ring<release_ring_cap> _rq;
      std::thread                    _bg;
      std::atomic<bool>              _bg_stop{false};
      std::atomic<uint64_t>          _bg_inflight{0};
   };

   // ── suffix_value (SAL leaf): alloc_header + [u16 vlen][u16 slen][value][suffix] ──
   // slen==0 => the term (a "leaf" is suffix_value with a 0-length suffix). slen
   // is u16 (NOT u8): a leaf suffix can be long when the differing key tail is the
   // value itself (e.g. the MDBX DUPSORT composite key — the dup value rides in
   // the trie key, so the leaf below the shared prefix carries a multi-hundred-byte
   // suffix). A u8 slen silently truncated those (slen & 0xff = 0 → the leaf reads
   // back a 0-length suffix → every key diverges → NOTFOUND). This 4-byte body header
   // is the SHARED leaf-body format: it is byte-identical whether the leaf is a
   // standalone SAL object (body() begins immediately after the alloc_header) OR
   // memcpy'd verbatim into a nested_setlist inline line (cap 127 B). rcart is the
   // REFCOUNT fork — there is NO birth field (gcart's version-liveness prefix is
   // absent), so the standalone body() and the inline form stay byte-for-byte equal
   // and the verbatim memcpy in pack_nested_leaves / find_in's hand-read path are
   // valid. The CB refcount IS the leaf's liveness (ref=1 at alloc, freed by release).
   struct sv_leaf : sal_hybrid::alloc_header
   {
      static constexpr sal_hybrid::header_type type_id = sal_hybrid::header_type::start_user_type;  // 3
      static constexpr uint32_t          hdr     = 4;  // [u16 vlen][u16 slen]
      // LOGICAL body size (== the inline-into-nested form): used for inline packing.
      static uint32_t bytes_for(uint32_t vlen, uint32_t slen) noexcept { return hdr + vlen + slen; }

      char*       body() noexcept { return reinterpret_cast<char*>(this) + sizeof(sal_hybrid::alloc_header); }
      const char* body() const noexcept
      {
         return reinterpret_cast<const char*>(this) + sizeof(sal_hybrid::alloc_header);
      }

      // ── SAL construction contract: alloc<sv_leaf>(suffix,value) calls
      //    alloc_size(suffix,value) then constructs sv_leaf(asize, seq, suffix, value).
      //    SAL's alloc<T> asserts (size % 64 == 0), so round to 64 (cacheline_size
      //    is 32 — rounding only to it would fail the assert for non-64-aligned bodies).
      static constexpr uint32_t alloc_grain =
          sal_hybrid::cacheline_size > 64u ? sal_hybrid::cacheline_size : 64u;
      static uint32_t alloc_size(sv suffix, sv value) noexcept
      {
         const uint32_t raw = uint32_t(sizeof(sal_hybrid::alloc_header) +
                                       bytes_for(uint32_t(value.size()), uint32_t(suffix.size())));
         return ucc::round_up_multiple<alloc_grain>(raw);
      }
      sv_leaf(uint32_t asize, sal_hybrid::ptr_address_seq seq, sv suffix, sv value) noexcept
          : sal_hybrid::alloc_header(asize, type_id, seq)
      {
         const uint16_t v16 = uint16_t(value.size());
         const uint16_t s16 = uint16_t(suffix.size());
         char*          d   = body();
         std::memcpy(d, &v16, 2);
         std::memcpy(d + 2, &s16, 2);
         std::memcpy(d + hdr, value.data(), value.size());
         std::memcpy(d + hdr + value.size(), suffix.data(), suffix.size());
      }

      sv value() const noexcept
      {
         const char* p = body();
         uint16_t    v16;
         std::memcpy(&v16, p, 2);
         return {p + hdr, v16};
      }
      sv suffix() const noexcept
      {
         const char* p = body();
         uint16_t    v16, s16;
         std::memcpy(&v16, p, 2);
         std::memcpy(&s16, p + 2, 2);
         return {p + hdr + v16, s16};
      }
   };

   // ── big_leaf (BIG region): [u32 vlen][u16 slen][value][suffix] ─────────────
   // The same (suffix,value) leaf as sv_leaf, but the body is too large for the
   // SAL var-allocator (>2KB) so it lives as ONE contiguous object in the BIG
   // region (address-stable, refcounted). vlen is u32 (big values can exceed 64K);
   // slen is u16 (same reason as sv_leaf — a long key tail / dup-value suffix; a u8
   // would truncate it). Reached as a bt::big_suffix_value branch; no SAL slot.
   struct big_leaf
   {
      static constexpr uint32_t hdr = 6;  // [u32 vlen][u16 slen]
      static uint32_t body_bytes(uint32_t vlen, uint32_t slen) noexcept { return hdr + vlen + slen; }
      static uint32_t make(engine& e, sv suffix, sv value)
      {
         const uint32_t vl = uint32_t(value.size()), sl = uint32_t(suffix.size());
         const uint32_t h  = e.big.alloc(body_bytes(vl, sl));
         char*          p  = static_cast<char*>(e.big.get(h));
         const uint16_t s16 = uint16_t(sl);
         std::memcpy(p, &vl, 4);
         std::memcpy(p + 4, &s16, 2);
         std::memcpy(p + hdr, value.data(), vl);
         std::memcpy(p + hdr + vl, suffix.data(), sl);
         return h;
      }
      static sv value(const engine& e, uint32_t h) noexcept
      {
         const char* p = static_cast<const char*>(e.big.get(h));
         uint32_t    vl;
         std::memcpy(&vl, p, 4);
         return {p + hdr, vl};
      }
      static sv suffix(const engine& e, uint32_t h) noexcept
      {
         const char* p = static_cast<const char*>(e.big.get(h));
         uint32_t    vl;
         uint16_t    s16;
         std::memcpy(&vl, p, 4);
         std::memcpy(&s16, p + 4, 2);
         return {p + hdr + vl, s16};
      }
   };

   // ── engine::init_leaf_ / leaf_live_bytes_ (out-of-line: sv_leaf complete) ──────
   // ORDER IS LOAD-BEARING: construct allocator(start_threads_now=FALSE) →
   // register_type_ops<sv_leaf>() → start_background_threads() → create sessions.
   // (start_threads_now=true would seal the type array in the ctor and a later
   //  register_type_ops would std::abort; registering after start_background_threads
   //  hits the same seal; not starting the segment-provider thread stalls
   //  get_new_segment forever once the initial segments are consumed.)
   inline void engine::init_leaf_()
   {
      std::filesystem::create_directories(_dir / "leaf");
      sal_hybrid::runtime_config cfg;  // MFU read-cache defaults ON (the lever this move earns)
      leaf = std::make_unique<sal_hybrid::allocator>(_dir / "leaf", cfg, /*start_threads_now=*/false);
      leaf->register_type_ops<sv_leaf>();   // ops BEFORE start_background_threads (SAL contract)
      // big_leaf is NOT a SAL type — it lives on the standalone `big` allocator.
      leaf->start_background_threads();      // compactor + MFU promoter + release + provider ON
      // Engine-owned session, created AFTER the leaf allocator is up + threads started:
      // the cold admin/main-thread session (leaf_live_bytes_). _bg_ses is kept for the
      // dormant engine bg ring (sync release today).
      _bg_ses    = leaf->create_session();
      _admin_ses = leaf->create_session();
   }
   inline uint64_t engine::leaf_live_bytes_() noexcept
   {
      uint64_t b = 0;
      _admin_ses->for_each_live_object(
          [&](auto, auto, const sal_hybrid::alloc_header* h) { b += h->size(); });
      return b + big.live_bytes();  // SAL small leaves + standalone BIG region
   }

   // ── leaf dispatch: a terminal is suffix_value (SAL) OR big_suffix_value (BIG);
   //    read its (suffix,value) without the caller caring which region. The sv arm
   //    resolves via the caller's leaf SESSION's get_ref<sv_leaf> (which SAMPLES the
   //    MFU active bit — the whole point of the move); the big arm uses e.big.get
   //    (the non-moving BIG region) UNCHANGED. The returned sv VIEWS leaf memory:
   //    valid while the snapshot pin keeps the leaf alive. ──────────────────────────
   inline sv leaf_suffix(engine& e, sal_hybrid::allocator_session& ses, bt t, uint32_t p) noexcept
   {
      return t == bt::big_suffix_value
                 ? big_leaf::suffix(e, p)
                 : ses.get_ref<sv_leaf>(leaf_addr(p)).obj()->suffix();
   }
   inline sv leaf_value(engine& e, sal_hybrid::allocator_session& ses, bt t, uint32_t p) noexcept
   {
      return t == bt::big_suffix_value
                 ? big_leaf::value(e, p)
                 : ses.get_ref<sv_leaf>(leaf_addr(p)).obj()->value();
   }
   inline bool is_leaf_tag(bt t) noexcept
   {
      return t == bt::suffix_value || t == bt::big_suffix_value;
   }

   // ── setlist (cl1/cl2): the sparse routing node, CAP-based for in-place insert
   // layout: [u8 meta(terminal:1,prefixlen:7)] [u8 nbranch] [u8 cap]
   //         prefix[prefixlen]  keys[cap]  types[cap]:3bit  branches[cap]:u32
   // cap = total branch slots (byte-branches + a term slot) the node's fixed slot
   // can hold; offsets are from cap (NOT nbranch), so an insert shifts within the
   // pre-sized arrays — no realloc until n reaches cap (then grow to cl2/full).
   // Same density as compact (the slot is a fixed size regardless); the slack is
   // reused as growth room. The term (if any) occupies branch slot `nbranch`.
   struct setlist
   {
      uint8_t meta;     // terminal:1 (bit7), prefixlen:7 (bits0-6)
      uint8_t nbranch;  // byte-branch count
      uint8_t cap;      // total branch slots provisioned (byte + term)

      static constexpr uint32_t slot_bytes = 128;
      static constexpr uint32_t hdr        = 3;
      bool                      has_term() const noexcept { return meta & 0x80; }
      uint32_t                  prefixlen() const noexcept { return meta & 0x7f; }
      uint32_t                  total() const noexcept { return nbranch + (has_term() ? 1u : 0u); }
      bool                      full_cap() const noexcept { return total() >= cap; }

      char*       base() noexcept { return reinterpret_cast<char*>(this); }
      const char* base() const noexcept { return reinterpret_cast<const char*>(this); }
      char*       prefix_ptr() noexcept { return base() + hdr; }
      const char* prefix_ptr() const noexcept { return base() + hdr; }
      sv          prefix() const noexcept { return {prefix_ptr(), prefixlen()}; }
      uint8_t*    keys() noexcept { return reinterpret_cast<uint8_t*>(base() + hdr + prefixlen()); }
      const uint8_t* keys() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(base() + hdr + prefixlen());
      }
      uint8_t*        types() noexcept { return keys() + cap; }
      const uint8_t*  types() const noexcept { return keys() + cap; }
      static uint32_t branches_off_(uint32_t prefixlen, uint32_t cap) noexcept
      {
         return ((hdr + prefixlen + cap + tag3_bytes(cap)) + 3u) & ~3u;  // 4-byte align
      }
      uint32_t* branches() noexcept
      {
         return reinterpret_cast<uint32_t*>(base() + branches_off_(prefixlen(), cap));
      }
      const uint32_t* branches() const noexcept
      {
         return reinterpret_cast<const uint32_t*>(base() + branches_off_(prefixlen(), cap));
      }

      // bytes a node with `cap` total slots + `prefixlen` occupies.
      static uint32_t bytes_for_cap(uint32_t cap, uint32_t prefixlen) noexcept
      {
         return branches_off_(prefixlen, cap) + cap * 4u;
      }
      // does the CONTENT (nbranch + term) fit `slot`? (minimum cap = content)
      // prefixlen is stored in a 7-bit meta field, so a prefix longer than 127
      // CANNOT be represented in a setlist/double_setlist — it must NOT "fit" here
      // (else meta = prefixlen & 0x7f silently truncates it and the node loses its
      // prefix → every routed key diverges → NOTFOUND). build_node then takes the
      // full-node prefix-split path (which caps the held prefix at full::max_prefix).
      static constexpr uint32_t max_prefixlen = 0x7f;
      static bool fits(uint32_t nbranch, uint32_t prefixlen, bool term, uint32_t slot) noexcept
      {
         return prefixlen <= max_prefixlen &&
                bytes_for_cap(nbranch + (term ? 1u : 0u), prefixlen) <= slot;
      }
      // largest cap whose node fits `slot` for this prefixlen.
      static uint32_t derive_cap(uint32_t prefixlen, uint32_t slot) noexcept
      {
         uint32_t c = 1;
         while (c < 255 && bytes_for_cap(c + 1, prefixlen) <= slot)
            ++c;
         return c;
      }

      branch_type btype(uint32_t i) const noexcept { return tag3_get(types(), i); }
      uint32_t    branch(uint32_t i) const noexcept { return branches()[i]; }
      branch_type term_type() const noexcept
      {
         return has_term() ? tag3_get(types(), nbranch) : bt::null;
      }
      uint32_t term_branch() const noexcept { return branches()[nbranch]; }
      int      find_index(uint8_t byte) const noexcept
      {
         int pos = int(ucc::lower_bound_padded(keys(), nbranch, byte));
         return (pos < int(nbranch) && keys()[pos] == byte) ? pos : -1;
      }
   };

   // ── nested_setlist (cl1, 128B): an outer setlist with INLINE-or-external ───
   // branches packed in the SAME line (Dan's §7 design). Each outer branch is
   // EITHER inline (a sub-node packed in the tail: a small setlist body, or a
   // suffix_value leaf body — a leaf with NO SAL object at all) OR external (a
   // 4-byte region ptr). Routes the outer byte + (for an inline setlist) the next
   // byte in ONE cacheline, and inlines small children for density.
   //   THE WHOLE NODE IS A SINGLE NODE (one cl1 slot, one refcount). Inline parts
   //   are BODY, freed with the slot — never released/promoted independently. The
   //   only refcounted children are EXTERNAL ptrs + an inline setlist's u32
   //   branches (its grandchildren). An inline leaf is pure data (not a child).
   // layout: [u8 meta(term:1,prefixlen:7)] [u8 nbranch]
   //         prefix[prefixlen]
   //         okeys[nbranch]                 (outer bytes, sorted)
   //         otype[nbranch+has_term]:3bit   (branch_type kind of each entry; term last)
   //         ooff [nbranch+has_term]:u8     (bit7 = INLINE flag; bits0-6 = offset-from-END)
   //         ... free gap ...
   //         [entries packed contiguously, growing DOWN from base()+128]
   // entry i = [base()+128-eoff(i), base()+128-eoff(i-1)); size(i)=eoff(i)-eoff(i-1)
   //   (eoff ascending, eoff(-1):=0, tail_used=eoff(nslot-1)). EXTERNAL entry = a
   //   u32 ptr; INLINE setlist entry = a setlist body; INLINE suffix_value entry =
   //   [u16 vlen][u16 slen][value][suffix] (sv_leaf body — header matches sv_leaf::hdr=4,
   //   in_value/in_suffix below). offsets are from-END so a verbatim memcpy keeps them valid.
   struct nested_setlist
   {
      uint8_t meta;     // outer term:1 (bit7), prefixlen:7 (bits0-6)
      uint8_t nbranch;  // outer byte-branch count

      static constexpr uint32_t slot_bytes  = 128;
      static constexpr uint32_t hdr         = 2;
      static constexpr uint8_t  inline_flag = 0x80;  // ooff bit7
      // Max decoded entries an edit/remove scratch must hold: a 128-byte line costs at
      // least ~3 bytes/entry (1 okey + 1 ooff + 1 tail byte; otypes add 0.375), so it
      // never holds more than ~40; 64 is a safe cap that keeps the recursive descent's
      // per-frame nentry[] scratch small (a fat frame overflows the stack on deep keys).
      static constexpr uint32_t es_cap = 64;

      bool     has_term() const noexcept { return meta & 0x80; }
      uint32_t prefixlen() const noexcept { return meta & 0x7f; }
      uint32_t nslot() const noexcept { return nbranch + (has_term() ? 1u : 0u); }

      char*       base() noexcept { return reinterpret_cast<char*>(this); }
      const char* base() const noexcept { return reinterpret_cast<const char*>(this); }
      sv          prefix() const noexcept { return {base() + hdr, prefixlen()}; }

      uint8_t* okeys() noexcept { return reinterpret_cast<uint8_t*>(base() + hdr + prefixlen()); }
      const uint8_t* okeys() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(base() + hdr + prefixlen());
      }
      uint8_t*       otypes() noexcept { return okeys() + nbranch; }
      const uint8_t* otypes() const noexcept { return okeys() + nbranch; }
      uint8_t*       ooff() noexcept { return otypes() + tag3_bytes(nslot()); }
      const uint8_t* ooff() const noexcept { return otypes() + tag3_bytes(nslot()); }

      branch_type otype(uint32_t i) const noexcept { return tag3_get(otypes(), i); }
      bool        is_inline(uint32_t i) const noexcept { return ooff()[i] & inline_flag; }
      uint32_t    eoff(uint32_t i) const noexcept
      {
         return ooff()[i] & 0x7f;
      }  // offset-from-END of entry start
      char*       entry(uint32_t i) noexcept { return base() + slot_bytes - eoff(i); }
      const char* entry(uint32_t i) const noexcept { return base() + slot_bytes - eoff(i); }
      uint32_t    entry_size(uint32_t i) const noexcept
      {
         return eoff(i) - (i == 0 ? 0u : eoff(i - 1));
      }

      // EXTERNAL entry → a u32 region pointer.
      uint32_t ext_ptr(uint32_t i) const noexcept
      {
         uint32_t v;
         std::memcpy(&v, entry(i), 4);
         return v;
      }
      // INLINE setlist entry → a setlist body (its branches are u32 grandchildren).
      setlist* in_setlist(uint32_t i) noexcept { return reinterpret_cast<setlist*>(entry(i)); }
      const setlist* in_setlist(uint32_t i) const noexcept
      {
         return reinterpret_cast<const setlist*>(entry(i));
      }
      // INLINE suffix_value entry → an sv_leaf body in the line (value + suffix).
      // SHARED sv_leaf header: [u16 vlen][u16 slen]. (Inline entries are ≤127 B so
      // both lengths are small, but the header layout MUST match sv_leaf exactly —
      // inline bodies are memcpy'd verbatim from standalone SAL leaves.)
      sv in_value(uint32_t i) const noexcept
      {
         const char* q = entry(i);
         uint16_t    v;
         std::memcpy(&v, q, 2);
         return {q + sv_leaf::hdr, v};
      }
      sv in_suffix(uint32_t i) const noexcept
      {
         const char* q = entry(i);
         uint16_t    v, s;
         std::memcpy(&v, q, 2);
         std::memcpy(&s, q + 2, 2);
         return {q + sv_leaf::hdr + v, s};
      }

      int find_index(uint8_t byte) const noexcept
      {
         int pos = int(ucc::lower_bound_padded(okeys(), nbranch, byte));
         return (pos < int(nbranch) && okeys()[pos] == byte) ? pos : -1;
      }
   };
   static_assert(sizeof(nested_setlist) == 2);

   // ── full (cl9, 1152 B): the dense routing node (direct byte index) ─────────
   // [u8 meta(term:1,prefixlen:7)] [u8 term_tag] [u16 nbranch] [u32 term_ptr]
   //   types[256]:3bit(96B) @8  branches[256]:u32(1024B) @104  prefix[prefixlen] @1128
   struct full
   {
      uint8_t  meta;
      uint8_t  term_tag;  // branch_type of the term (if terminal)
      uint16_t nbranch;
      uint32_t term_ptr;

      static constexpr uint32_t slot_bytes   = 1152;
      static constexpr uint32_t types_off    = 8;
      static constexpr uint32_t branches_off = 104;                      // 8 + 96
      static constexpr uint32_t prefix_off   = 1128;                     // 104 + 1024
      static constexpr uint32_t max_prefix   = slot_bytes - prefix_off;  // 24

      bool           has_term() const noexcept { return meta & 0x80; }
      uint32_t       prefixlen() const noexcept { return meta & 0x7f; }
      char*          base() noexcept { return reinterpret_cast<char*>(this); }
      const char*    base() const noexcept { return reinterpret_cast<const char*>(this); }
      uint8_t*       types() noexcept { return reinterpret_cast<uint8_t*>(base() + types_off); }
      const uint8_t* types() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(base() + types_off);
      }
      uint32_t* branches() noexcept { return reinterpret_cast<uint32_t*>(base() + branches_off); }
      const uint32_t* branches() const noexcept
      {
         return reinterpret_cast<const uint32_t*>(base() + branches_off);
      }
      char*       prefix_ptr() noexcept { return base() + prefix_off; }
      const char* prefix_ptr() const noexcept { return base() + prefix_off; }
      sv          prefix() const noexcept { return {prefix_ptr(), prefixlen()}; }
      branch_type btype(uint32_t byte) const noexcept { return tag3_get(types(), byte); }
      uint32_t    branch(uint32_t byte) const noexcept { return branches()[byte]; }
   };

   // ── region dispatch: a 3-bit tag picks the region; resolve/retain/release ──
   inline bool is_routing(bt t) noexcept
   {
      return t == bt::setlist || t == bt::nested_setlist || t == bt::double_setlist ||
             t == bt::big_nested || t == bt::full;
   }
   // Resolve a (tag,ptr) to a body pointer. Routing tags use the direct FIX
   // allocators (e.cl*, address-stable); the suffix_value LEAF tag uses the
   // caller-supplied SAL session's get_ref (MFU-sampled, compactor-transparent);
   // big_suffix_value uses the non-moving BIG region (e.big.get). The caller passes
   // its OWN session — no global map, no lock. The sv body pointer is valid while the
   // snapshot pin keeps the leaf alive (the compactor may relocate it otherwise; a
   // fresh get_ref always returns the current location).
   inline void* resolve(engine& e, sal_hybrid::allocator_session& ses, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full:
            return e.cl9.get(p);
         case bt::double_setlist:
         case bt::big_nested:
            return e.cl2.get(p);
         case bt::setlist:
         case bt::nested_setlist:
            return e.cl1.get(p);
         case bt::suffix_value:
            return const_cast<char*>(ses.get_ref<sv_leaf>(leaf_addr(p)).obj()->body());
         case bt::big_suffix_value:
            return e.big.get(p);
         default:
            return nullptr;  // null
      }
   }
   // ROUTING-ONLY resolve (no session): cl1/cl2/cl9 are direct, address-stable FIX
   // allocators that need no leaf session. The write path's routing-only callers
   // (read_branches / gather_children_ / node_* / set_*_inplace / reprefix) use this;
   // a suffix_value leaf tag needs a session (use the session overload above). The
   // big_suffix_value arm stays available (non-moving BIG region, no session needed).
   inline void* resolve(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full:
            return e.cl9.get(p);
         case bt::double_setlist:
         case bt::big_nested:
            return e.cl2.get(p);
         case bt::setlist:
         case bt::nested_setlist:
            return e.cl1.get(p);
         case bt::big_suffix_value:
            return e.big.get(p);
         case bt::suffix_value:
            assert(false && "suffix_value resolve needs a session (use resolve(e, ses, t, p))");
            return nullptr;
         default:
            return nullptr;  // null
      }
   }
   inline void prefetch(const void* p) noexcept
   {
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(p, 0, 3);
#else
      (void)p;
#endif
   }
   // a routing node is UNIQUE (ref==1) iff no snapshot pins it → safe to mutate
   // in place (no clone). Only routing nodes are mutated in place.
   inline bool unique(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full:
            return e.cl9.unique(p);
         case bt::double_setlist:
         case bt::big_nested:
            return e.cl2.unique(p);
         case bt::setlist:
         case bt::nested_setlist:
            return e.cl1.unique(p);
         default:
            return false;
      }
   }
   // lock-free pin attempt: bump only if alive (snapshot acquire). false = dead.
   // rcart RETAINS leaves on a COW share (unlike gcart's version-liveness), so the
   // suffix_value arm routes through the SAL session's try_retain (ref>0 CAS);
   // big stays on the standalone BIG region; routing on the FIX allocators.
   [[nodiscard]] inline bool try_retain(engine& e, sal_hybrid::allocator_session& ses, bt t,
                                        uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full:
            return e.cl9.try_retain(p);
         case bt::double_setlist:
         case bt::big_nested:
            return e.cl2.try_retain(p);
         case bt::setlist:
         case bt::nested_setlist:
            return e.cl1.try_retain(p);
         case bt::suffix_value:
            return ses.try_retain(leaf_addr(p));
         case bt::big_suffix_value:
            return e.big.try_retain(p);
         default:
            return false;
      }
   }
   inline void retain(engine& e, sal_hybrid::allocator_session& ses, bt t, uint32_t p)
   {
      switch (t)
      {
         case bt::full:
            e.cl9.retain(p);
            break;
         case bt::double_setlist:
         case bt::big_nested:
            e.cl2.retain(p);
            break;
         case bt::setlist:
         case bt::nested_setlist:
            e.cl1.retain(p);
            break;
         case bt::suffix_value:
            ses.retain(leaf_addr(p));
            break;
         case bt::big_suffix_value:
            e.big.retain(p);
            break;
         default:
            break;
      }
   }
   // returns true iff the slot hit zero (caller recurses into children + frees).
   // ROUTING + BIG ONLY — the suffix_value leaf release collapses into ONE
   // ses.release(leaf_addr(p)) inside the cascade functions (SAL does the atomic
   // decrement and frees + runs type_ops.destroy at zero), so suffix_value is NOT a
   // case here. big_suffix_value keeps its two-call release_one + free(h) idiom.
   [[nodiscard]] inline bool release_one(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full:
            return e.cl9.release_one(p);
         case bt::double_setlist:
         case bt::big_nested:
            return e.cl2.release_one(p);
         case bt::setlist:
         case bt::nested_setlist:
            return e.cl1.release_one(p);
         case bt::big_suffix_value:
            return e.big.release_one(p);
         default:
            return false;
      }
   }
}  // namespace rcart

namespace rcart
{
   // a decoded branch (scratch, for copy-construct-rebuild of a routing node)
   struct br
   {
      uint8_t  byte;
      bt       tag;
      uint32_t ptr;
   };

   // ── nentry: the decoded, PATH-AGNOSTIC entry the canonical packer (lay_nested_into_)
   //    and the direct-mutable nested editor (nested_edit/nested_remove) both speak.
   //    A nested_setlist's bytes are a PURE FUNCTION of (prefix, sorted nentry set, term);
   //    funnelling initial-pack, unique in-place edit, and shared clone+edit ALL through
   //    lay_nested_into_(nentry*) is what makes the 128 bytes byte-identical across the
   //    three paths (killing promote-on-write).
   //    kind 0 = EXTERNAL  → (otype, ext) is the region-typed 4-byte child ptr.
   //    kind 1 = inline suffix_value → (otype=suffix_value); body/blen point at the
   //             sv_leaf body bytes ([u16 vlen][u16 slen][value][suffix]) to memcpy.
   //    kind 2 = inline setlist → (otype=setlist); body/blen point at a TIGHT setlist
   //             body to memcpy (its u32 grandchildren ride along unchanged).
   //    For kinds 1/2 `body` points at a stack-scratch copy (re-pack of an existing
   //    nested) OR a re-resolved/tight-staged body (initial pack) — lay_nested_into_
   //    ONLY ever memcpy's raw bytes, never refcounts. Inline bodies are PURE DATA.
   struct nentry
   {
      uint8_t     okey;   // outer byte (sorted; unused for the term entry)
      uint8_t     kind;   // 0=ext u32 ptr, 1=inline-sv body, 2=inline-setlist body
      bt          otype;  // resolved branch_type tag of this entry
      uint32_t    ext;    // kind 0: the external 4-byte region ptr
      const char* body;   // kind 1/2: bytes to memcpy into the tail-heap
      uint16_t    blen;   // kind 1/2: byte count of body
   };

   // Forward declarations: find_in reads a RESOLVED leaf body via these (defined just
   // below the cursor descents); the standalone sv_leaf/big_leaf body begins with the
   // SAME [vlen][slen] header so they work on a resolve()'d body verbatim.
   inline sv leaf_suffix_body(bt t, const void* b) noexcept;
   inline sv leaf_value_body(bt t, const void* b) noexcept;

   inline bool pfx_ok_(sv pfx, sv key, uint32_t depth) noexcept
   {
      return key.size() - depth >= pfx.size() &&
             (pfx.empty() || std::memcmp(key.data() + depth, pfx.data(), pfx.size()) == 0);
   }

   // Read-path find from a (tag,ptr) root — usable by readers/cursors. Dispatch on
   // the branch tag, PREFETCH the child before the dependent load (full prefetches
   // the offset cacheline before the type hdr). out is a VIEW into the value bytes
   // (no copy; caller pins the root). Mirrors start_ref's view-based read — the hot
   // path never allocates or copies. The caller passes its OWN leaf session (the
   // sv arm resolves via get_ref, MFU-sampled); the leaf read uses leaf_*_body on
   // the ONE resolved body (no second get_ref).
   inline bool find_in(engine& e, sal_hybrid::allocator_session& ses, bt t, uint32_t p, sv key, sv& out)
   {
      uint32_t depth = 0;
      for (;;)
      {
         if (t == bt::null)
            return false;
         void* node = resolve(e, ses, t, p);
         if (is_leaf_tag(t))  // suffix_value (SAL) or big_suffix_value (BIG)
         {
            if (leaf_suffix_body(t, node) == key.substr(depth))
            {
               out = leaf_value_body(t, node);
               return true;
            }
            return false;
         }
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(node);
            // Prefetch the offset cacheline for the SPECULATIVE routing byte
            // (key[depth] — correct when prefixless, the common case) BEFORE the
            // header read, so the header's cold miss overlaps it (§7). Header NOT
            // prefetched — it IS the overlap window. Cold-scale win only.
            if (depth < key.size())
               prefetch(n->base() + full::branches_off + (uint32_t(uint8_t(key[depth])) << 2));
            if (!pfx_ok_(n->prefix(), key, depth))  // header cold miss — overlaps the prefetch
               return false;
            depth += n->prefixlen();
            if (depth == key.size())
            {
               if (!n->has_term())
                  return false;
               t = bt(n->term_tag);
               p = n->term_ptr;
               continue;
            }
            const uint8_t byte = uint8_t(key[depth]);
            const bt      ct   = n->btype(byte);
            if (ct == bt::null)
               return false;
            t = ct;
            p = n->branch(byte);
            ++depth;
            continue;
         }
         if (t == bt::nested_setlist)
         {
            const nested_setlist* nn = static_cast<const nested_setlist*>(node);
            if (!pfx_ok_(nn->prefix(), key, depth))
               return false;
            depth += nn->prefixlen();
            if (depth == key.size())  // key ends at the outer prefix → outer term
            {
               if (!nn->has_term())
                  return false;
               const uint32_t ti = nn->nbranch;
               if (nn->is_inline(ti))  // inline term = the value in the line (0-suffix)
               {
                  out = nn->in_value(ti);  // view into the line
                  return true;
               }
               t = nn->otype(ti);
               p = nn->ext_ptr(ti);
               continue;
            }
            const int oi = nn->find_index(uint8_t(key[depth]));
            if (oi < 0)
               return false;
            const uint32_t ui = uint32_t(oi);
            if (nn->is_inline(ui))
            {
               if (nn->otype(ui) ==
                   bt::suffix_value)  // inline leaf — compare in the line (resolve once)
               {
                  const char* q = nn->entry(ui);
                  uint16_t    v, s;
                  std::memcpy(&v, q, 2);
                  std::memcpy(&s, q + 2, 2);
                  if (sv{q + sv_leaf::hdr + v, s} != key.substr(depth + 1))
                     return false;
                  out = sv{q + sv_leaf::hdr, v};  // view into the line
                  return true;
               }
               const setlist* in = nn->in_setlist(ui);  // inline setlist — route the next byte
               ++depth;                                 // consumed the outer byte
               if (!pfx_ok_(in->prefix(), key, depth))
                  return false;
               depth += in->prefixlen();
               if (depth == key.size())
               {
                  if (!in->has_term())
                     return false;
                  t = in->term_type();
                  p = in->term_branch();
                  continue;
               }
               const int ii = in->find_index(uint8_t(key[depth]));
               if (ii < 0)
                  return false;
               t = in->btype(uint32_t(ii));
               p = in->branch(uint32_t(ii));
               ++depth;
               continue;
            }
            t = nn->otype(ui);
            p = nn->ext_ptr(ui);
            ++depth;  // external child
            continue;
         }
         const setlist* n = static_cast<const setlist*>(node);
         if (!pfx_ok_(n->prefix(), key, depth))
            return false;
         depth += n->prefixlen();
         if (depth == key.size())
         {
            if (!n->has_term())
               return false;
            t = n->term_type();
            p = n->term_branch();
            continue;
         }
         const int idx = n->find_index(uint8_t(key[depth]));
         if (idx < 0)
            return false;
         t = n->btype(uint32_t(idx));
         p = n->branch(uint32_t(idx));
         ++depth;
         continue;
      }
   }

   // ── ordered cursor descents (read-only; usable by snapshot cursors) ─────────
   // Body-based leaf readers: read (suffix,value) from a RESOLVED leaf body — works
   // for a standalone leaf (resolve(e,t,p)) and an inline-in-nested suffix_value body.
   inline sv leaf_suffix_body(bt t, const void* b) noexcept
   {
      const char* p = static_cast<const char*>(b);
      if (t == bt::big_suffix_value)  // [u32 vlen][u16 slen]
      {
         uint32_t v;
         uint16_t s;
         std::memcpy(&v, p, 4);
         std::memcpy(&s, p + 4, 2);
         return {p + big_leaf::hdr + v, s};
      }
      uint16_t v, s;  // sv_leaf: [u16 vlen][u16 slen]
      std::memcpy(&v, p, 2);
      std::memcpy(&s, p + 2, 2);
      return {p + sv_leaf::hdr + v, s};
   }
   inline sv leaf_value_body(bt t, const void* b) noexcept
   {
      const char* p = static_cast<const char*>(b);
      if (t == bt::big_suffix_value)
      {
         uint32_t v;
         std::memcpy(&v, p, 4);
         return {p + big_leaf::hdr, v};
      }
      uint16_t v;
      std::memcpy(&v, p, 2);
      return {p + sv_leaf::hdr, v};
   }

   // Visit a nested_setlist's REFCOUNTABLE byte-branch children (external ptrs +
   // an inline setlist's u32 grandchildren). Inline suffix_value entries are pure
   // data (no child). The outer term is handled by the caller (node_term parity).
   template <class F>
   inline void for_each_nested_branch_(const nested_setlist* n, F&& f)
   {
      const char*    bp   = n->base();
      const uint8_t* off  = n->ooff();
      const uint8_t* otyp = n->otypes();
      for (uint32_t i = 0; i < n->nbranch; ++i)
      {
         const uint8_t o = off[i];
         if (o & nested_setlist::inline_flag)
         {
            if (tag3_get(otyp, i) == bt::setlist)  // inline setlist body → its u32 grandchildren
            {
               const setlist* in = reinterpret_cast<const setlist*>(bp + 128 - (o & 0x7f));
               for (uint32_t j = 0; j < in->nbranch; ++j)
                  f(in->btype(j), in->branch(j));
               if (in->has_term())
                  f(in->term_type(), in->term_branch());
            }
            // inline suffix_value → no refcountable child
         }
         else  // external child
         {
            uint32_t v;
            std::memcpy(&v, bp + 128 - (o & 0x7f), 4);
            f(tag3_get(otyp, i), v);
         }
      }
   }

   // Free a routing-node slot by region tag (free-function form of free_node).
   inline void free_node_(engine& e, bt t, uint32_t p) noexcept
   {
      if (t == bt::full)
         e.cl9.free(p);
      else if (t == bt::double_setlist || t == bt::big_nested)
         e.cl2.free(p);
      else
         e.cl1.free(p);
   }
   // Copy a routing node's refcountable children + external term into caller-owned
   // locals. The caller invokes this only AFTER its release_one() has put the slot
   // in the DYING sentinel state: dying slots are not reuse-eligible (write-around
   // takes only ref==0), so the content read here cannot race a writer's reuse.
   // Inline nested data rides with the slot (not a child).
   inline void gather_children_(engine&   e,
                                bt        t,
                                uint32_t  p,
                                br*       kids,
                                uint32_t& nb,
                                bt&       tt,
                                uint32_t& tp,
                                bool&     term)
   {
      nb   = 0;
      term = false;
      tt   = bt::null;
      tp   = 0;
      if (t == bt::full)
      {
         full* n = static_cast<full*>(resolve(e, t, p));
         for (uint32_t b = 0; b < 256; ++b)
         {
            const bt ct = n->btype(b);
            if (ct != bt::null)
               kids[nb++] = br{uint8_t(b), ct, n->branch(b)};
         }
         if (n->has_term())
         {
            term = true;
            tt   = bt(n->term_tag);
            tp   = n->term_ptr;
         }
      }
      else if (t == bt::nested_setlist)
      {
         nested_setlist* n = static_cast<nested_setlist*>(resolve(e, t, p));
         for_each_nested_branch_(n, [&](bt ct, uint32_t cp) { kids[nb++] = br{0, ct, cp}; });
         if (n->has_term() && !n->is_inline(n->nbranch))
         {
            term = true;
            tt   = n->otype(n->nbranch);
            tp   = n->ext_ptr(n->nbranch);
         }
      }
      else
      {
         setlist* n = static_cast<setlist*>(resolve(e, t, p));
         for (uint32_t i = 0; i < n->nbranch; ++i)
            kids[nb++] = br{n->keys()[i], n->btype(i), n->branch(i)};
         if (n->has_term())
         {
            term = true;
            tt   = n->term_type();
            tp   = n->term_branch();
         }
      }
   }

   // ── free reclamation (snapshots / readers; no write sessions needed) ───────
   // release_one() drops the last ref to the DYING sentinel (not 0), so the slot's
   // content stays readable AFTER the decrement — a concurrent writer's write-around
   // reuses only ref==0 (free()'d) slots, never a dying one. Hence we read children
   // AFTER release_one and ONLY when we actually own the kill: a still-shared node
   // (the common COW case) returns immediately without touching its content.
   inline void release_subtree(engine& e, sal_hybrid::allocator_session& ses, bt t, uint32_t p)
   {
      if (t == bt::null)
         return;
      if (t == bt::suffix_value)
      {
         // SAL: ONE atomic decrement; at the last ref SAL frees + runs
         // type_ops.destroy (a no-op for a childless byte leaf). No byte count.
         ses.release(leaf_addr(p));
         return;
      }
      if (t == bt::big_suffix_value)
      {
         if (e.big.release_one(p))
            e.big.free(p);  // standalone BIG region: handle-only free, unchanged
         return;
      }
      if (!is_routing(t))
         return;
      if (!release_one(e, t, p))
         return;  // still shared (another snapshot pins it) — never read its content
      br       kids[257];
      uint32_t nb;
      bt       tt;
      uint32_t tp;
      bool     term;
      gather_children_(e, t, p, kids, nb, tt, tp, term);  // p is dying: read race-free
      for (uint32_t i = 0; i < nb; ++i)
         release_subtree(e, ses, kids[i].tag, kids[i].ptr);
      if (term)
         release_subtree(e, ses, tt, tp);
      free_node_(e, t, p);
   }

   // ── bounded-inline / async release (SAL's lock-free reclaim design) ─────────
   // A top-level drop destroys at most `budget` nodes inline on the calling
   // thread, then hands each remaining sub-reference to `spill` (a background
   // queue); the background thread re-enters with an effectively infinite budget
   // so the cascade runs OFF the writer's critical path. The common COW-to-root
   // drop is a handful of nodes (fully inline); only a large drop (a batch's
   // worth of garbage, a snapshot release) spills. `spill` receives an
   // UN-decremented reference — the bg does the release_one, exactly as the
   // foreground would — so foreground and background share one traversal.
   // Mirrors release_subtree's structure verbatim.
   template <class Spill>
   inline void release_async_(engine& e, sal_hybrid::allocator_session& ses, bt t, uint32_t p,
                              int& budget, Spill& spill)
   {
      if (t == bt::null)
         return;
      if (budget <= 0)
      {
         spill(t, p);
         return;
      }  // out of inline budget — hand to the bg
      if (t == bt::suffix_value)
      {
         --budget;
         ses.release(leaf_addr(p));  // SAL: decrement + free at zero (no byte count)
         return;
      }
      if (t == bt::big_suffix_value)
      {
         --budget;
         if (e.big.release_one(p))
            e.big.free(p);  // standalone BIG region: handle-only free, unchanged
         return;
      }
      if (!is_routing(t))
         return;
      if (!release_one(e, t, p))
         return;  // still shared (another snapshot pins it) — never read its content
      --budget;
      br       kids[257];
      uint32_t nb;
      bt       tt;
      uint32_t tp;
      bool     term;
      gather_children_(e, t, p, kids, nb, tt, tp, term);  // p is dying: read race-free
      for (uint32_t i = 0; i < nb; ++i)
         release_async_(e, ses, kids[i].tag, kids[i].ptr, budget, spill);
      if (term)
         release_async_(e, ses, tt, tp, budget, spill);
      free_node_(e, t, p);
   }

   // ── engine reclaimer (out-of-line: depends on release_async_) ───────────────
   // The engine-side async release ring is DORMANT (rcart's _bg thread is never
   // started — sync release today; SAL has its OWN release thread). These pass
   // the engine-owned _bg_ses (single-thread-owned: only the bg thread / inline
   // backpressure on one thread would drive it) so the suffix_value leaf arm has a
   // session. Not on any hot path.
   inline void engine::spill_fn::operator()(bt t, uint32_t p) const noexcept
   {
      if (e->_rq.try_push(kv_pack_(t, p)))
         return;
      int big = 0x7fffffff;  // ring full → finish inline (backpressure)
      release_async_(*e, *e->_bg_ses, t, p, big, e->_spill);
   }
   inline void engine::release_async(bt t, uint32_t p) noexcept
   {
      if (t == bt::null)
         return;
      int budget = inline_destroy_budget;
      release_async_(*this, *_bg_ses, t, p, budget, _spill);
   }
   inline void engine::bg_loop_() noexcept
   {
      for (;;)
      {
         _bg_inflight.store(1, std::memory_order_release);  // claim BEFORE pop (drain predicate)
         uint64_t v;
         if (_rq.try_pop(v))
         {
            int big = 0x7fffffff;  // bg has effectively infinite budget
            release_async_(*this, *_bg_ses, bt(uint8_t(v >> 32)), uint32_t(v), big, _spill);
            _bg_inflight.store(0, std::memory_order_release);
         }
         else
         {
            _bg_inflight.store(0, std::memory_order_release);
            if (_bg_stop.load(std::memory_order_acquire))
               return;  // stopped AND drained
            std::this_thread::yield();
         }
      }
   }

   class tree
   {
     public:
      explicit tree(engine& e)
          : _e(e),
            _s1(e.cl1.start_session()),
            _s2(e.cl2.start_session()),
            _s9(e.cl9.start_session()),
            _lses(e.leaf_create_session())
      {
      }

      uint64_t size() const noexcept { return _size; }
      // The work tree's OWN leaf session (single-thread-owned). The transaction/abort
      // path frees a work-tree husk through this session (the husk being freed is the
      // writer's).
      sal_hybrid::allocator_session* leaf_session() const noexcept { return _lses.get(); }

      // ── read path: dispatch on the branch tag, prefetch the child BEFORE the
      //    dependent load (Dan's hard rule; full node prefetches the offset
      //    cacheline before reading the 3-bit type header). The descent resolves +
      //    copies leaf bodies, so hold a leaf read_lock across it. ────────────────
      bool find(sv key, std::string& out) const
      {
         auto lk = _lses->lock();  // pin leaf pages for the descent + value copy
         sv   v;
         if (!find_in(const_cast<engine&>(_e), *_lses, _rt, _rp, key, v))
            return false;
         out.assign(v.data(), v.size());  // string-out API: one copy
         return true;
      }
      // view-returning find for the hot read path (no copy). Returns a value_ref that
      // BUNDLES the leaf read_lock with the view: the SAL leaf region is compactor-
      // movable, so the view is safe only while the lock is held — the caller keeps it
      // valid simply by holding the returned value_ref (it drops the lock on destroy).
      // nullopt = miss. (Replaces the old out-param form, whose lock dropped on return
      // and left the view dangling for the get callers that read it afterwards.)
      std::optional<value_ref> find(sv key) const
      {
         sal_hybrid::read_lock lk = _lses->lock();  // pin leaf pages for the descent + caller use
         sv              v;
         if (!find_in(const_cast<engine&>(_e), *_lses, _rt, _rp, key, v))
            return std::nullopt;
         return value_ref{lk, v};
      }

      bt       root_tag() const noexcept { return _rt; }
      uint32_t root_ptr() const noexcept { return _rp; }
      void     set_root(bt t, uint32_t p) noexcept
      {
         _rt = t;
         _rp = p;
      }
      engine& eng() noexcept { return _e; }
      // Mark the FIX-allocator transaction boundary (start_ref::session model): the
      // routing-node sessions (cl1/cl2/cl9) latch _floor = _cursor = the current
      // frontier so this txn's at/after-floor frees recycle hot and the freeset scan
      // first-fits forward from here. Called at write-transaction start. The SAL
      // leaf session (_lses) has no begin_txn/floor model (it does not participate).
      void begin_txn() noexcept
      {
         _s1.begin_txn();
         _s2.begin_txn();
         _s9.begin_txn();
      }
      // release a (tag,ptr) subtree through refcount (snapshot drop / abort).
      void release(bt t, uint32_t p) { release_ref(t, p); }

      // ── write path ─────────────────────────────────────────────────────────
      // Public entry descends in <unique> and latches to <shared> on the first
      // snapshot-pinned node (doc §5.2). One refcount read at the root decides
      // in-place vs clone; the cost is cached down the descent. Scope a leaf
      // read_lock over the whole mutation (the descent resolves + copies leaf bodies
      // and may alloc/free leaves through _lses).
      void upsert(sv key, sv val)
      {
         auto lk     = _lses->lock();
         auto [t, p] = upsert_rec<cow_mode::unique>(_rt, _rp, key, 0, val);
         _rt         = t;
         _rp         = p;
      }

      // Remove a key. Returns 1 if it was present (and deleted), 0 otherwise.
      // A read-path pre-check (find_in) confirms presence BEFORE any clone, so
      // remove_rec is guaranteed to reach the leaf — it carries no not-found
      // branch and never wastes a clone on a missing key (a missing key is a
      // pure no-op: the committed tree is unchanged, nothing to COW).
      int remove(sv key)
      {
         auto lk = _lses->lock();  // pin leaf pages for the pre-check + reclaim
         sv   _scanv;              // presence pre-check only — discards the value view
         if (!find_in(_e, *_lses, _rt, _rp, key, _scanv))
            return 0;  // absent — no-op, no clone
         auto [t, p] = remove_rec<cow_mode::unique>(_rt, _rp, key, 0);
         _rt         = t;
         _rp         = p;
         --_size;
         return 1;
      }

     private:
      // Build a terminal leaf: SAL suffix_value if the body fits the size gate,
      // else a BIG big_suffix_value (single contiguous object in the non-moving BIG
      // region). Returns the tag so callers stay region-agnostic. ses.alloc<T>(args)
      // sizes via T::alloc_size, allocs data + CB ref=1, returns a ptr_address; the
      // bare u32 (uint32_t(*a)) is what the branch stores. (NO birth: refcount fork.)
      std::pair<bt, uint32_t> make_sv(sv suffix, sv val)
      {
         if (sv_leaf::bytes_for(uint32_t(val.size()), uint32_t(suffix.size())) <=
             engine::sal_max_body)
         {
            const leaf_addr_t a = _lses->alloc<sv_leaf>(suffix, val);
            return {bt::suffix_value, uint32_t(*a)};
         }
         return {bt::big_suffix_value, big_leaf::make(_e, suffix, val)};
      }
      // Build an INLINE suffix_value (sv_leaf) body — [u16 vlen][u16 slen][value][suffix]
      // — into a caller stack buffer, WITHOUT any SAL alloc (pure carry-over bytes; the
      // ONLY make_sv-free way to mint an inline-sv for a direct nested edit). Returns the
      // total body byte count (== sv_leaf::bytes_for(vlen, slen)). The caller checks the
      // return against the per-line tail budget; if the body is too big to inline it must
      // make_sv an EXTERNAL leaf instead (the only make_sv in the nested write path).
      // dst must have room for sv_leaf::hdr + value.size() + suffix.size() bytes (<=127
      // for an inline entry; the caller's buffer is 127 B).
      // Does an inline-sv body for (suffix,value) fit the per-line inline cap (127 B,
      // dst buffer size)? MUST be checked BEFORE build_sv_body_ — a body larger than the
      // buffer would smash the stack. Too-big values become an EXTERNAL leaf (make_sv).
      static bool sv_body_fits_(sv suffix, sv value) noexcept
      {
         return size_t(sv_leaf::hdr) + value.size() + suffix.size() <= 127;
      }
      static uint16_t build_sv_body_(char* dst, sv suffix, sv value) noexcept
      {
         assert(sv_body_fits_(suffix, value) && "build_sv_body_ overflow — check sv_body_fits_ first");
         const uint16_t v16 = uint16_t(value.size());
         const uint16_t s16 = uint16_t(suffix.size());
         std::memcpy(dst, &v16, 2);
         std::memcpy(dst + 2, &s16, 2);
         std::memcpy(dst + sv_leaf::hdr, value.data(), value.size());
         std::memcpy(dst + sv_leaf::hdr + value.size(), suffix.data(), suffix.size());
         return uint16_t(sv_leaf::hdr + value.size() + suffix.size());
      }
      // WRITER-ONLY COW reclaim (work tree husks). Single-threaded with the writer's
      // alloc, and a release cascade never allocates, so no concurrent slot reuse can
      // race with reading a node's children — unlike the reader path (release_subtree)
      // this needs NO gather-before-release, keeping the COW write path fast. Readers
      // never call this; they release via snap → release_subtree. Runs on the writer
      // thread under the upsert/remove leaf read_lock, so _lses is the right session.
      void release_ref(bt t, uint32_t p)
      {
         if (t == bt::null)
            return;
         if (t == bt::suffix_value)
         {
            _lses->release(leaf_addr(p));  // SAL: decrement + free at zero (no byte count)
            return;
         }
         if (t == bt::big_suffix_value)
         {
            if (_e.big.release_one(p))
               _e.big.free(p);  // standalone BIG region: handle-only free, unchanged
            return;
         }
         if (!is_routing(t))
            return;
         if (!release_one(_e, t, p))
            return;  // still shared (snapshot)
         read_branches(t, p, [&](const br& b) { release_ref(b.tag, b.ptr); });
         bt       tt;
         uint32_t tp;
         if (node_term(t, p, tt, tp))
            release_ref(tt, tp);
         free_node(t, p);
      }

      // visit a routing node's byte-branches.
      template <class F>
      void read_branches(bt t, uint32_t p, F&& f) const
      {
         engine& e = const_cast<engine&>(_e);
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(resolve(e, t, p));
            for (uint32_t b = 0; b < 256; ++b)
            {
               const bt ct = n->btype(b);
               if (ct != bt::null)
                  f(br{uint8_t(b), ct, n->branch(b)});
            }
         }
         else if (t == bt::nested_setlist)
         {
            // refcount-walk contract only (byte cosmetic — rebuild/reprefix PROMOTE
            // nested first). Visits external children + inline-setlist grandchildren;
            // inline leaves are data (not children); the outer term → node_term().
            const nested_setlist* n = static_cast<const nested_setlist*>(resolve(e, t, p));
            for_each_nested_branch_(n, [&](bt ct, uint32_t cp) { f(br{0, ct, cp}); });
         }
         else
         {
            const setlist* n = static_cast<const setlist*>(resolve(e, t, p));
            for (uint32_t i = 0; i < n->nbranch; ++i)
               f(br{n->keys()[i], n->btype(i), n->branch(i)});
         }
      }
      bool node_term(bt t, uint32_t p, bt& tt, uint32_t& tp) const
      {
         engine& e = const_cast<engine&>(_e);
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(resolve(e, t, p));
            if (!n->has_term())
               return false;
            tt = bt(n->term_tag);
            tp = n->term_ptr;
            return true;
         }
         if (t == bt::nested_setlist)
         {
            const nested_setlist* n = static_cast<const nested_setlist*>(resolve(e, t, p));
            if (!n->has_term() || n->is_inline(n->nbranch))
               return false;  // no term, or inline term = data
            tt = n->otype(n->nbranch);
            tp = n->ext_ptr(n->nbranch);
            return true;
         }
         const setlist* n = static_cast<const setlist*>(resolve(e, t, p));
         if (!n->has_term())
            return false;
         tt = n->term_type();
         tp = n->term_branch();
         return true;
      }
      // O(1) child lookup for a byte (setlist: SWAR find_index; full: direct).
      bool node_child(bt t, uint32_t p, uint8_t byte, bt& ct, uint32_t& cp) const
      {
         engine& e = const_cast<engine&>(_e);
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(resolve(e, t, p));
            ct            = n->btype(byte);
            if (ct == bt::null)
               return false;
            cp = n->branch(byte);
            return true;
         }
         // nested_setlist children are inline (no slot) — the write path PROMOTES a
         // nested before editing, so node_child never sees one. Defensive guard.
         assert(t != bt::nested_setlist && "edit a nested_setlist only after promote_nested");
         const setlist* n   = static_cast<const setlist*>(resolve(e, t, p));
         const int      idx = n->find_index(byte);
         if (idx < 0)
            return false;
         ct = n->btype(uint32_t(idx));
         cp = n->branch(uint32_t(idx));
         return true;
      }
      sv node_prefix(bt t, uint32_t p) const
      {
         engine& e = const_cast<engine&>(_e);
         if (t == bt::full)
            return static_cast<const full*>(resolve(e, t, p))->prefix();
         if (t == bt::nested_setlist)
            return static_cast<const nested_setlist*>(resolve(e, t, p))->prefix();
         return static_cast<const setlist*>(resolve(e, t, p))->prefix();
      }
      void free_node(bt t, uint32_t p)
      {
         // Writer-foreground husk free (same thread as this tree's sessions): route
         // through the SESSION so a slot freed at/after the txn floor (born this txn,
         // its page already dirty) recycles HOT via the per-session fcache instead of
         // the freeset round trip (start_ref §7). Cold (behind-floor) frees fall
         // through to the freeset authority inside session::free.
         if (t == bt::full)
            _s9.free(p);
         else if (t == bt::double_setlist || t == bt::big_nested)
            _s2.free(p);
         else
            _s1.free(p);
      }
      // Release a husk whose children were already TRANSFERRED into a freshly
      // built node (rebuild / rebuild_remove / promote, private path). Pure
      // refcount: release_one() decides, free() is reached ONLY at ref 0 — and
      // we do NOT cascade into children (they now live in the new node). The
      // husk is private (ref 1) in every caller, so this is the 1→0→free path.
      // (This replaces a manual free_node(husk): in a refcount system only
      // release may free, and a manual free left the slot ref==1, which the
      // allocator's write-around treats as live and never reuses.)
      void release_husk(bt t, uint32_t p)
      {
         if (release_one(_e, t, p))
            free_node(t, p);
      }

      // Build a routing node from a branch list (sorted by byte) + optional term
      // + prefix; pick cl1/cl2/full by size. Branch ptrs are TAKEN (not retained).
      // may_inline: try a nested_setlist first (inline private small leaf children
      // into the line). Pass true ONLY where the caller does NOT release the child
      // set afterward (rebuild-unique free_node / fresh splits) — inlined leaves are
      // consumed (release_ref'd), which would double-free under a later release_ref(t).
      static bool nested_enabled() noexcept
      {
         static const bool on = (std::getenv("RCART_NO_NESTED") == nullptr);
         return on;
      }
      std::pair<bt, uint32_t> build_node(sv        prefix,
                                         const br* brs,
                                         uint32_t  nb,
                                         bool      term,
                                         bt        tt,
                                         uint32_t  tp,
                                         bool      may_inline = false)
      {
         const uint32_t pl = uint32_t(prefix.size());
         if (may_inline && nested_enabled())
         {
            const uint32_t s = pack_nested_leaves(prefix, brs, nb, term, tt, tp);
            if (s != UINT32_MAX)
               return {bt::nested_setlist, s};
         }
         if (setlist::fits(nb, pl, term, 128))
            return {bt::setlist, pack_setlist<128>(_s1, prefix, brs, nb, term, tt, tp)};
         if (setlist::fits(nb, pl, term, 256))
            return {bt::double_setlist, pack_setlist<256>(_s2, prefix, brs, nb, term, tt, tp)};
         // A full node's prefix slot is only full::max_prefix (24) bytes, but a
         // long shared prefix can precede a high-fanout point (e.g. Silkworm
         // storage: dbi + 32B account hash, then a 256-way slot fan-out). Split
         // the prefix: a wrapper holds the HEAD and routes ONE byte to a full
         // that holds the TAIL (≤24) + the fan-out. (setlist/double already
         // account for the prefix in their fits() check, so only full overflows.)
         if (pl > full::max_prefix)
         {
            const uint32_t h   = pl - full::max_prefix - 1;  // head len; 1 byte links to the full
            const uint8_t  sep = uint8_t(prefix[h]);
            const uint32_t fs =
                pack_full(prefix.substr(h + 1), brs, nb, term, tt, tp);  // tail = 24
            br link{sep, bt::full, fs};
            return build_node(prefix.substr(0, h), &link, 1, false, bt::null, 0,
                              false);  // wrapper (recurses if head>127)
         }
         return {bt::full, pack_full(prefix, brs, nb, term, tt, tp)};
      }

      // Copy setlist `c` into `dst` re-packed TIGHT (cap == content) — the inline
      // form for a nested tail (its u32 grandchildren ride along unchanged).
      void copy_setlist_tight(char* dst, const setlist* c)
      {
         setlist*       d     = reinterpret_cast<setlist*>(dst);
         const uint32_t total = c->total();
         d->meta              = c->meta;
         d->nbranch           = c->nbranch;
         d->cap               = uint8_t(total);
         std::memcpy(d->prefix_ptr(), c->prefix_ptr(), c->prefixlen());
         std::memcpy(d->keys(), c->keys(), c->nbranch);
         for (uint32_t i = 0; i < total; ++i)
         {
            tag3_set(d->types(), i, tag3_get(c->types(), i));
            d->branches()[i] = c->branches()[i];
         }
      }

      // Externalize an inline-setlist body (raw bytes `body`/`blen`, e.g. a
      // nested-entry kind2) into a fresh standalone cl1 slot so it can be recursed
      // into <unique> + re-inlined. PURE COPY — no retain. The grandchildren ptrs are
      // aliases of the inline-setlist's grandchildren; in SHARED mode the nested_edit/
      // nested_remove kind2 arm RETAINS every carry-over grandchild IMMEDIATELY before
      // calling this, so the <unique> recursion's release of a carried-over grandchild
      // brings it to ref 1 (snapshot keeps it), never 0. In UNIQUE mode the inline-setlist
      // was private — grandchildren are private. Either way, externalize itself adds no ref.
      uint32_t externalize_inline_setlist_(const char* body, uint16_t blen)
      {
         uint32_t s = _s1.alloc();
         std::memcpy(_e.cl1.get(s), body, blen);
         return s;
      }

      // Clone a SHARED (snapshot-pinned) nested_setlist into a FRESH private cl1 slot —
      // the nested analog of reprefix(t,p,0): verbatim 128-byte copy, RETAIN every
      // carry-over child (external ptrs) + inline-setlist grandchild (so they are now
      // co-owned by old + clone), then drop ONE ref on the old node (the snapshot keeps
      // its copy). inline-sv bodies are pure data (copied verbatim, no refcount). After
      // this the returned slot is ref 1 (private) and can be edited with the simple
      // <unique> path. Returns {nested_setlist, fresh_slot}.
      std::pair<bt, uint32_t> clone_nested_(bt t, uint32_t p)
      {
         const uint32_t  s   = _s1.alloc();
         std::memcpy(_e.cl1.get(s), resolve(_e, t, p), nested_setlist::slot_bytes);
         const nested_setlist* src = static_cast<const nested_setlist*>(resolve(_e, t, p));
         for_each_nested_branch_(src, [&](bt ct, uint32_t cp) { retain(_e, *_lses, ct, cp); });
         if (src->has_term() && !src->is_inline(src->nbranch))
            retain(_e, *_lses, src->otype(src->nbranch), src->ext_ptr(src->nbranch));
         release_ref(t, p);  // drop one ref on the old node (snapshot keeps its copy)
         return {bt::nested_setlist, s};
      }

      // ── THE CANONICAL nested_setlist byte packer ──────────────────────────────
      // Writes header + okeys + otypes + ooff + the from-END tail-heap of a
      // nested_setlist into slot `n` from a decoded nentry set. PURE BYTES: no
      // refcount side-effects (no retain / release / free / alloc) — the consume /
      // transfer / retain bookkeeping is the CALLER's job. `es[0..ne)` are the
      // entries IN INDEX ORDER (byte-branches sorted ascending, then — if `term` —
      // the term entry LAST at index ne-1; the term's okey is unused). Returns false
      // WITHOUT writing a single byte if the line would overflow (tail>127 or
      // head+tail>128), so a same-slot unique re-pack can fail safely and the caller
      // promotes from the still-intact original. This is the one helper that
      // initial-pack, unique in-place edit, and shared clone+edit all funnel through,
      // making a nested line's bytes a pure function of (prefix, nentry set, term).
      bool lay_nested_into_(nested_setlist* n, sv prefix, const nentry* es, uint32_t ne, bool term)
      {
         const uint32_t pl    = uint32_t(prefix.size());
         const uint32_t nbr   = ne - (term ? 1u : 0u);  // byte-branch count
         const uint32_t nslot = ne;
         // SIZE CHECK BEFORE WRITING (risk: a half-written slot can't be safely
         // promoted). head = hdr + prefix + okeys[nbr] + otypes(3bit×nslot) + ooff[nslot].
         uint32_t tail = 0;
         for (uint32_t i = 0; i < ne; ++i)
            tail += (es[i].kind ? es[i].blen : 4u);
         const uint32_t head = nested_setlist::hdr + pl + nbr + tag3_bytes(nslot) + nslot;
         if (pl > nested_setlist::inline_flag - 1 || tail > 127 || head + tail > 128)
            return false;  // overflow — caller promotes (the original slot is untouched)

         n->meta    = uint8_t((term ? 0x80 : 0) | (pl & 0x7f));
         n->nbranch = uint8_t(nbr);
         std::memcpy(n->base() + nested_setlist::hdr, prefix.data(), pl);
         for (uint32_t i = 0; i < nbr; ++i)
            n->okeys()[i] = es[i].okey;
         uint32_t eoff = 0;
         for (uint32_t idx = 0; idx < ne; ++idx)
         {
            const nentry&  e    = es[idx];
            const uint32_t esz  = (e.kind ? e.blen : 4u);
            eoff += esz;
            n->ooff()[idx] = uint8_t((e.kind ? nested_setlist::inline_flag : 0) | eoff);
            char* dst      = n->base() + 128 - eoff;
            if (e.kind)  // inline (kind 1 sv body OR kind 2 setlist body) — raw bytes
            {
               std::memcpy(dst, e.body, e.blen);
               tag3_set(n->otypes(), idx, e.otype);
            }
            else  // external: 4-byte region ptr
            {
               std::memcpy(dst, &e.ext, 4);
               tag3_set(n->otypes(), idx, e.otype);
            }
         }
         return true;
      }

      // SHARED-emit carry-over retains. After an emitted (fresh) nested line / flat node
      // is built from the edited nentry set, every REFCOUNTABLE child it now holds that
      // is a genuine CARRY-OVER from the old (snapshot-pinned) node must be retained so it
      // is co-owned by both the old node and the fresh one (the subsequent release_ref(old)
      // — net-zero now while a snapshot pins old — leaves the old reference intact, and the
      // old node's eventual drain cascade balances these retains). EXACTLY mirrors
      // rebuild<shared>'s "retain copied children except is_new". The FRESHLY-OWNED branch
      // (have_fresh && idx==fresh_idx: a make_sv leaf, a deeper-split child, or a recursed/
      // re-inlined child — already owned, ref 1) is SKIPPED. Inline-sv entries are pure data
      // (no CB) and are never retained. Walks the SAME es[] set lay_nested_into_ packed, so
      // it is index-aligned with the fresh node's entries: kind0 external → retain the ptr;
      // kind2 inline-setlist → retain its u32 grandchildren (+ external term grandchild).
      void retain_shared_carryovers_(const nentry* es, uint32_t ne, bool have_fresh,
                                     uint32_t fresh_idx)
      {
         for (uint32_t i = 0; i < ne; ++i)
         {
            if (have_fresh && i == fresh_idx)
               continue;  // freshly-owned (recursed / minted) — already ref 1, do NOT retain
            const nentry& e = es[i];
            if (e.kind == 0)  // external child / external term → refcountable ptr
               retain(_e, *_lses, e.otype, e.ext);
            else if (e.kind == 2)  // inline-setlist → refcountable u32 grandchildren
            {
               const setlist* in = reinterpret_cast<const setlist*>(e.body);
               for (uint32_t j = 0; j < in->nbranch; ++j)
                  retain(_e, *_lses, in->btype(j), in->branch(j));
               if (in->has_term())
                  retain(_e, *_lses, in->term_type(), in->term_branch());
            }
            // kind 1 inline-sv → pure data, never retained
         }
      }

      // Decode a nested_setlist into a PATH-AGNOSTIC nentry set. memcpy's the WHOLE
      // 128 B node to `scratch128` FIRST (risk: a same-slot unique re-pack rewrites
      // the live slot, so we must read everything from a private snapshot — the
      // from-END tail-heap means a new entry's start can overlap an old entry's
      // bytes). Inline bodies are kept as POINTERS INTO scratch128 (pure carry-over
      // bytes, NEVER make_sv'd, so NOT refcountable); externals keep their u32 ptr.
      void read_nested_entries_(const nested_setlist* src,
                                char                  scratch128[128],
                                nentry                out[257],
                                uint32_t&             ne,
                                bool&                 term)
      {
         std::memcpy(scratch128, src, 128);
         const nested_setlist* sn = reinterpret_cast<const nested_setlist*>(scratch128);
         ne                       = 0;
         term                     = sn->has_term();
         auto decode_at           = [&](uint32_t i, uint8_t okey)
         {
            nentry& o = out[ne++];
            o.okey    = okey;
            if (!sn->is_inline(i))
            {
               o.kind  = 0;
               o.otype = sn->otype(i);
               o.ext   = sn->ext_ptr(i);
               o.body  = nullptr;
               o.blen  = 0;
            }
            else
            {
               o.kind  = (sn->otype(i) == bt::setlist) ? uint8_t(2) : uint8_t(1);
               o.otype = sn->otype(i);
               o.ext   = 0;
               o.body  = sn->entry(i);  // pointer INTO scratch128 (carry-over bytes)
               o.blen  = uint16_t(sn->entry_size(i));
            }
         };
         for (uint32_t i = 0; i < sn->nbranch; ++i)
            decode_at(i, sn->okeys()[i]);
         if (term)
            decode_at(sn->nbranch, 0);  // term last; okey unused
      }

      // Try to build a nested_setlist that INLINES private small children into the
      // line: a suffix_value LEAF (value+suffix body, no SAL object) or a plain
      // SETLIST (tight body; its u32 grandchildren stay external). shared / non-
      // inlineable / oversize children stay external 4-byte ptrs. Returns the cl1
      // slot, or UINT32_MAX if it doesn't fit one line or nothing would inline.
      // Inlined children are consumed (the source node freed; grandchildren
      // transfer); external children are TAKEN.
      uint32_t pack_nested_leaves(sv        prefix,
                                  const br* brs,
                                  uint32_t  nb,
                                  bool      term,
                                  bt        tt,
                                  uint32_t  tp)
      {
         const uint32_t pl    = uint32_t(prefix.size());
         const uint32_t nslot = nb + (term ? 1u : 0u);
         uint8_t        kind[257];
         uint32_t       esize[257];
         const char*    esrc[257];  // kind: 0=ext,1=leaf,2=setlist
         uint32_t       ninline = 0, tail = 0;
         auto           decide = [&](uint32_t idx, bt tag, uint32_t ptr)
         {
            uint8_t     k   = 0;
            uint32_t    sz  = 4;
            const char* src = nullptr;
            // Privacy gate: a suffix_value leaf with CB ref==1 is reachable only
            // through THIS branch, so it is safe to inline its body + consume the
            // standalone leaf. (rcart's refcount analog of gcart's birth==_V gate.)
            // ses.ref(adr) is a relaxed read of the control_block refcount (no mutate).
            if (tag == bt::suffix_value && _lses->ref(leaf_addr(ptr)) == 1)
            {
               // Read the body to MEASURE the inline size (the leaf may be relocated by
               // the compactor before lay() copies it, so lay() re-resolves; here we
               // only read the [u16 vlen][u16 slen] header under the write-path lock).
               const char* b0 = _lses->get_ref<sv_leaf>(leaf_addr(ptr)).obj()->body();
               uint16_t v16, s16;  // sv_leaf body header: [u16 vlen][u16 slen]
               std::memcpy(&v16, b0, 2);
               std::memcpy(&s16, b0 + 2, 2);
               const uint32_t b = sv_leaf::bytes_for(v16, s16);
               if (b <= 127)
               {
                  k   = 1;
                  sz  = b;
                  src = b0;  // unused for kind==1 (lay re-resolves); set non-null as a marker
               }
            }
            else if (tag == bt::setlist && _e.cl1.unique(ptr))
            {
               const setlist* c = static_cast<const setlist*>(_e.cl1.get(ptr));
               const uint32_t b = setlist::bytes_for_cap(c->total(), c->prefixlen());  // TIGHT
               if (b <= 127)
               {
                  k   = 2;
                  sz  = b;
                  src = reinterpret_cast<const char*>(c);
               }
            }
            kind[idx]  = k;
            esize[idx] = sz;
            esrc[idx]  = src;
            ninline += (k != 0);
            tail += sz;
         };
         for (uint32_t i = 0; i < nb; ++i)
            decide(i, brs[i].tag, brs[i].ptr);
         if (term)
            decide(nb, tt, tp);
         const uint32_t head = nested_setlist::hdr + pl + nb + tag3_bytes(nslot) + nslot;
         if (ninline == 0 || tail > 127 || head + tail > 128)
            return UINT32_MAX;

         // Build the PATH-AGNOSTIC nentry set and stage inline bodies into a scratch
         // (lay_nested_into_ only memcpy's raw bytes — no re-resolve, no tight-pack).
         // kind1: re-resolve the sv_leaf body NOW (the compactor may have moved it
         //   since decide(); get_ref returns the current location) and stage the
         //   verbatim [u16 vlen][u16 slen][value][suffix] bytes (byte-identical to the
         //   inline form — rcart has no birth prefix). kind2: copy_setlist_tight into
         //   the scratch so the cap==total tight body is what gets memcpy'd (byte-
         //   identity with today's output). The CONSUME side-effects (release the
         //   standalone leaf / free the source setlist / take the external) run AFTER
         //   the bytes are laid, in pack order — lay_nested_into_ is pure bytes.
         // inline bodies sum to <=127 (the tail budget, enforced by the reject above),
         // so ONE 128-byte staging buffer holds every inline body contiguously.
         char        stage[128];
         uint32_t    soff = 0;
         nentry      es[257];
         auto        make_entry = [&](uint32_t idx, uint8_t okey, bt tag, uint32_t ptr)
         {
            nentry& o = es[idx];
            o.okey    = okey;
            if (kind[idx] == 1)
            {
               char* dst = stage + soff;
               soff += esize[idx];
               const char* src = _lses->get_ref<sv_leaf>(leaf_addr(ptr)).obj()->body();
               std::memcpy(dst, src, esize[idx]);
               o.kind  = 1;
               o.otype = bt::suffix_value;
               o.ext   = 0;
               o.body  = dst;
               o.blen  = uint16_t(esize[idx]);
            }
            else if (kind[idx] == 2)
            {
               char* dst = stage + soff;
               soff += esize[idx];
               copy_setlist_tight(dst, reinterpret_cast<const setlist*>(esrc[idx]));
               o.kind  = 2;
               o.otype = bt::setlist;
               o.ext   = 0;
               o.body  = dst;
               o.blen  = uint16_t(esize[idx]);
            }
            else
            {
               o.kind  = 0;
               o.otype = tag;
               o.ext   = ptr;
               o.body  = nullptr;
               o.blen  = 0;
            }
         };
         for (uint32_t i = 0; i < nb; ++i)
            make_entry(i, brs[i].byte, brs[i].tag, brs[i].ptr);
         if (term)
            make_entry(nb, 0, tt, tp);

         const uint32_t  slot = _s1.alloc();
         nested_setlist* n    = static_cast<nested_setlist*>(_e.cl1.get(slot));
         lay_nested_into_(n, prefix, es, nslot, term);  // measured to fit (reject @above)

         // CONSUME side-effects, in pack order (NOT inside lay_nested_into_).
         auto consume = [&](uint32_t idx, uint32_t ptr)
         {
            if (kind[idx] == 1)
               release_ref(bt::suffix_value, ptr);  // private (ref==1) → free the standalone leaf
            else if (kind[idx] == 2)
            {
               if (_e.cl1.release_one(ptr))
                  _e.cl1.free(ptr);  // grandchildren transferred into the line
            }
            // kind 0 external: ptr TAKEN, nothing to consume
         };
         for (uint32_t i = 0; i < nb; ++i)
            consume(i, brs[i].ptr);
         if (term)
            consume(nb, tp);
         return slot;
      }

      template <uint32_t SB, class Ses>
      uint32_t
      pack_setlist(Ses& ses, sv prefix, const br* brs, uint32_t nb, bool term, bt tt, uint32_t tp)
      {
         const uint32_t slot = ses.alloc();
         auto*          n = static_cast<setlist*>(SB == 128 ? _e.cl1.get(slot) : _e.cl2.get(slot));
         n->meta          = uint8_t((term ? 0x80 : 0) | (prefix.size() & 0x7f));
         n->nbranch       = uint8_t(nb);
         n->cap = uint8_t(setlist::derive_cap(uint32_t(prefix.size()),
                                              SB));  // fill the slot's slack as growth room
         std::memcpy(n->prefix_ptr(), prefix.data(), prefix.size());
         for (uint32_t i = 0; i < nb; ++i)
         {
            n->keys()[i] = brs[i].byte;
            tag3_set(n->types(), i, brs[i].tag);
            n->branches()[i] = brs[i].ptr;
         }
         if (term)
         {
            tag3_set(n->types(), nb, tt);
            n->branches()[nb] = tp;
         }
         return slot;
      }

      // in-place sorted insert of a byte-branch into a PRIVATE setlist with room
      // (total() < cap): shift keys/types/branches within the pre-sized arrays.
      void setlist_insert_inplace(bt t, uint32_t p, uint8_t byte, bt tag, uint32_t ptr)
      {
         auto*          n     = static_cast<setlist*>(resolve(_e, t, p));
         const uint32_t pos   = uint32_t(ucc::lower_bound_padded(n->keys(), n->nbranch, byte));
         const uint32_t total = n->total();
         for (uint32_t i = total; i > pos; --i)
         {
            n->branches()[i] = n->branches()[i - 1];
            tag3_set(n->types(), i, tag3_get(n->types(), i - 1));
         }
         for (uint32_t i = n->nbranch; i > pos; --i)
            n->keys()[i] = n->keys()[i - 1];
         n->keys()[pos] = byte;
         tag3_set(n->types(), pos, tag);
         n->branches()[pos] = ptr;
         ++n->nbranch;
      }
      uint32_t pack_full(sv prefix, const br* brs, uint32_t nb, bool term, bt tt, uint32_t tp)
      {
         const uint32_t slot = _s9.alloc();
         auto*          n    = static_cast<full*>(_e.cl9.get(slot));
         n->meta             = uint8_t((term ? 0x80 : 0) | (prefix.size() & 0x7f));
         n->term_tag         = uint8_t(term ? uint8_t(tt) : uint8_t(bt::null));
         n->nbranch          = uint16_t(nb);
         n->term_ptr         = term ? tp : 0;
         std::memset(n->types(), 0xFF, 96);  // 0xFF bytes => every 3-bit tag = 7 = null
         std::memset(n->branches(), 0, 256 * 4);
         assert(prefix.size() <= full::max_prefix &&
                "full node prefix overflow (>24B) — must split the prefix");
         std::memcpy(n->prefix_ptr(), prefix.data(), prefix.size());
         for (uint32_t i = 0; i < nb; ++i)
         {
            tag3_set(n->types(), brs[i].byte, brs[i].tag);
            n->branches()[brs[i].byte] = brs[i].ptr;
         }
         return slot;
      }

      // COW edit of a routing node: read its branches into scratch, apply (insert
      // / replace one byte / set term), build a new node retaining the COPIED
      // children, release the old. Returns the new (tag, ptr).
      struct edit
      {
         enum K
         {
            ins,
            rep,
            set_term
         } k;
         uint8_t  byte;
         bt       tag;
         uint32_t ptr;
      };
      std::pair<bt, uint32_t> rebuild(bool is_unique, bt t, uint32_t p, edit ed)
      {
         br       scratch[257];
         uint32_t nb = 0;
         bool     term;
         bt       tt = bt::null;
         uint32_t tp = 0;
         term        = node_term(t, p, tt, tp);
         read_branches(t, p, [&](const br& b) { scratch[nb++] = b; });
         // apply edit (keep keys sorted)
         bt       new_tt   = tt;
         uint32_t new_tp   = tp;
         bool     new_term = term;
         if (ed.k == edit::set_term)
         {
            new_term = true;
            new_tt   = ed.tag;
            new_tp   = ed.ptr;
         }
         else if (ed.k == edit::rep)
         {
            for (uint32_t i = 0; i < nb; ++i)
               if (scratch[i].byte == ed.byte)
               {
                  scratch[i].tag = ed.tag;
                  scratch[i].ptr = ed.ptr;
                  break;
               }
         }
         else  // ins (byte absent) — insert sorted
         {
            uint32_t pos = 0;
            while (pos < nb && scratch[pos].byte < ed.byte)
               ++pos;
            for (uint32_t i = nb; i > pos; --i)
               scratch[i] = scratch[i - 1];
            scratch[pos] = br{ed.byte, ed.tag, ed.ptr};
            ++nb;
         }
         auto out = build_node(node_prefix(t, p), scratch, nb, new_term, new_tt, new_tp, is_unique);
         if (is_unique)
         {
            // PRIVATE: every child/term is transferred to `out` (no snapshot holds
            // the old node) — release the old husk (ref 1→0→free, no child
            // cascade). No O(n) retain/release.
            release_husk(t, p);
            return out;
         }
         // SHARED (a snapshot pins the old node): retain the COPIED children +
         // carried-over term (now referenced by both old and out), then release
         // the old node (snapshot keeps its copy). The edited/new branch came with
         // ref 1 (owned), so it is NOT retained here.
         read_branches(out.first, out.second,
                       [&](const br& b)
                       {
                          const bool is_new =
                              ((ed.k == edit::ins || ed.k == edit::rep) && b.byte == ed.byte);
                          if (!is_new)
                             retain(_e, *_lses, b.tag, b.ptr);
                       });
         if (new_term)
         {
            bt         ot;
            uint32_t   op2;
            const bool had         = node_term(t, p, ot, op2);
            const bool term_is_new = (ed.k == edit::set_term) || !had;
            if (!term_is_new)
               retain(_e, *_lses, new_tt, new_tp);  // carried over unchanged
         }
         release_ref(t, p);  // drops the old node (+ its children if it hits 0)
         return out;
      }

      template <cow_mode M>
      std::pair<bt, uint32_t> upsert_rec(bt t, uint32_t p, sv key, uint32_t depth, sv val)
      {
         if (t == bt::null)
         {
            ++_size;
            return make_sv(key.substr(depth), val);
         }
         if (is_leaf_tag(t))  // suffix_value (SAL) or big_suffix_value (BIG)
         {
            const sv esuf = leaf_suffix(_e, *_lses, t, p);
            const sv nsuf = key.substr(depth);
            if (esuf == nsuf)
            {
               auto nv = make_sv(nsuf, val);
               release_ref(t, p);
               return nv;
            }
            // split two leaves into a setlist
            std::string es(esuf), ns(nsuf);  // stable copies (esuf views the old obj we free)
            std::string eval_s(leaf_value(_e, *_lses, t, p));  // existing value
            release_ref(t, p);
            const uint32_t cp = lcp(es, ns);
            br             brs[2];
            uint32_t       nb    = 0;
            bool           term  = false;
            bt             tt    = bt::null;
            uint32_t       tp    = 0;
            auto           place = [&](const std::string& s, sv v)
            {
               if (cp == s.size())
               {
                  term    = true;
                  auto t2 = make_sv(sv{}, v);
                  tt      = t2.first;
                  tp      = t2.second;
               }
               else
               {
                  auto t2   = make_sv(sv(s).substr(cp + 1), v);
                  brs[nb++] = br{uint8_t(s[cp]), t2.first, t2.second};
               }
            };
            place(es, eval_s);
            place(ns, val);
            if (nb == 2 && brs[0].byte > brs[1].byte)
               std::swap(brs[0], brs[1]);
            ++_size;
            return build_node(sv(es).substr(0, cp), brs, nb, term, tt, tp, /*may_inline=*/true);
         }
         // routing node
         if (t == bt::nested_setlist)  // DIRECTLY MUTABLE — re-pack in place (no promote)
         {
            // The latch (same as the flat path @below): in <unique> read the refcount
            // once — a private node re-packs the SAME slot <unique>; a shared (snapshot-
            // pinned) one clones a fresh slot <shared> and recurses children <shared>.
            // In <shared> there is NO refcount read — it is the clone path only. The
            // nested node's prefix is consumed INSIDE nested_edit (it needs the entry
            // depth for the prefix-split fallback), so the latch keys off `unique()`.
            if constexpr (M == cow_mode::unique)
               if (unique(_e, t, p))
                  return nested_edit<cow_mode::unique>(t, p, key, depth, val);
            return nested_edit<cow_mode::shared>(t, p, key, depth, val);
         }
         // full: prefetch the offset CL for the speculative routing byte before
         // node_prefix touches the header (cold-miss overlap window, §7).
         if (t == bt::full && depth < key.size())
            prefetch(static_cast<const char*>(resolve(_e, t, p)) + full::branches_off +
                     (uint32_t(uint8_t(key[depth])) << 2));
         const sv       pfx  = node_prefix(t, p);
         const sv       ksuf = key.substr(depth);
         const uint32_t cp   = lcp(pfx, ksuf);
         if (cp < pfx.size())
         {
            // prefix split: new parent over [reprefixed old node] + [new key]
            std::string   common(pfx.substr(0, cp));
            const uint8_t curbyte = uint8_t(pfx[cp]);
            auto          cur2    = reprefix(t, p, cp + 1);  // old node with prefix pfx[cp+1:]
            br            brs[2];
            uint32_t      nb   = 0;
            bool          term = false;
            bt            tt   = bt::null;
            uint32_t      tp   = 0;
            brs[nb++]          = br{curbyte, cur2.first, cur2.second};
            if (cp == ksuf.size())
            {
               term    = true;
               auto nv = make_sv(sv{}, val);
               tt      = nv.first;
               tp      = nv.second;
            }
            else
            {
               auto nv   = make_sv(ksuf.substr(cp + 1), val);
               brs[nb++] = br{uint8_t(ksuf[cp]), nv.first, nv.second};
            }
            if (nb == 2 && brs[0].byte > brs[1].byte)
               std::swap(brs[0], brs[1]);
            ++_size;
            return build_node(common, brs, nb, term, tt, tp, /*may_inline=*/true);
         }
         depth += uint32_t(pfx.size());
         // The latch, compile-time: in <unique> read the refcount once — a private
         // node mutates in place (children stay <unique>); a shared one clones and
         // its children recurse <shared>. In <shared> there is NO refcount read at
         // all — the instantiation is the clone path only.
         // Overlap the cold refcount load (.rc array) with the node-body load:
         // resolve() is pure address arithmetic (base+p*size, no deref), so issue
         // the body prefetch BEFORE unique() so the body is in flight while the
         // refcount stalls — edit_node then dereferences it resident (one memory
         // latency per level, not two). Pure hint; correctness-neutral.
         if constexpr (M == cow_mode::unique)
         {
            prefetch(resolve(_e, t, p));
            if (unique(_e, t, p))
               return edit_node<cow_mode::unique>(t, p, key, depth, val);
         }
         return edit_node<cow_mode::shared>(t, p, key, depth, val);
      }

      // Mutate a routing node for the key. CM == unique ⟹ the node is private,
      // mutate (t,p) in place and recurse children <unique>. CM == shared ⟹ the
      // node is snapshot-pinned, clone it (now private), recurse children <shared>.
      // Either way the working node W is private, so all edits free-husk / in-place.
      template <cow_mode CM>
      std::pair<bt, uint32_t> edit_node(bt t, uint32_t p, sv key, uint32_t depth, sv val)
      {
         std::pair<bt, uint32_t> W;
         if constexpr (CM == cow_mode::unique)
            W = {t, p};
         else
            W = reprefix(t, p, 0);  // clone the shared node (retain children, drop one old ref)
         const bt       wt = W.first;
         const uint32_t wp = W.second;

         if (depth == key.size())  // term
         {
            bt       tt;
            uint32_t tp;
            if (node_term(wt, wp, tt, tp))  // REPLACE in place
            {
               release_ref(tt, tp);
               set_term_inplace(wt, wp, make_sv(sv{}, val));
               return W;
            }
            auto nv = make_sv(sv{}, val);
            ++_size;  // ADD a term → grow → rebuild (W private)
            return rebuild(true, wt, wp, edit{edit::set_term, 0, nv.first, nv.second});
         }
         const uint8_t byte = uint8_t(key[depth]);
         bt            ct;
         uint32_t      cpp;
         if (!node_child(wt, wp, byte, ct, cpp))  // INS: byte absent
         {
            auto nv = make_sv(key.substr(depth + 1), val);
            ++_size;
            if (wt == bt::full)  // O(1) direct-index insert (W is private)
            {
               full* n = static_cast<full*>(resolve(_e, wt, wp));
               tag3_set(n->types(), byte, nv.first);
               n->branches()[byte] = nv.second;
               ++n->nbranch;
               return W;
            }
            // setlist: in-place insert if the slot has room; else grow (rebuild).
            if (static_cast<setlist*>(resolve(_e, wt, wp))->total() <
                static_cast<setlist*>(resolve(_e, wt, wp))->cap)
            {
               setlist_insert_inplace(wt, wp, byte, nv.first, nv.second);
               return W;
            }
            return rebuild(true, wt, wp, edit{edit::ins, byte, nv.first, nv.second});
         }
         // REP: recurse the byte-child in the latched mode, overwrite the slot.
         auto nc = upsert_rec<CM>(ct, cpp, key, depth + 1, val);
         set_branch_inplace(wt, wp, byte, nc);
         return W;
      }

      // ── DIRECTLY-MUTABLE nested_setlist edit (kills promote-on-write) ───────────
      // Re-pack the nested line so the result is BYTE-IDENTICAL to what
      // pack_nested_leaves would have produced for the same (prefix, entries, term):
      // both funnel through lay_nested_into_. UNIQUE re-packs the SAME slot; SHARED
      // allocs a fresh cl1 slot and re-packs into it (retaining the carried-over
      // external children + inline-setlist grandchildren, releasing the old node
      // once). Only on a genuine 128B overflow does it fall back to promote_nested +
      // the flat path (the ONLY remaining promote). `depth` is the depth at node
      // ENTRY (the nested prefix not yet consumed).
      template <cow_mode CM>
      std::pair<bt, uint32_t> nested_edit(bt t, uint32_t p, sv key, uint32_t depth, sv val)
      {
         const uint32_t entry_depth = depth;  // for the overflow promote re-run
         nested_setlist* nn = static_cast<nested_setlist*>(resolve(_e, t, p));
         const sv       pfx  = nn->prefix();
         const sv       ksuf = key.substr(depth);
         const uint32_t cp   = lcp(pfx, ksuf);
         // PREFIX SPLIT: the node's prefix diverges from the key → it needs a new
         // parent. This is a genuine shape change; promote the nested to a flat node
         // and re-run the edit at the SAME entry depth (the flat node keeps the
         // prefix, upsert_rec re-consumes it). (The hot case is cp==pfx.size().)
         if (cp < pfx.size())
         {
            auto fl = promote_nested(t, p);
            return upsert_rec<cow_mode::unique>(fl.first, fl.second, key, entry_depth, val);
         }
         depth += uint32_t(pfx.size());

         // SINGLE clone+edit that MIRRORS the flat edit_node<shared> discipline (NOT a
         // clone-then-re-dispatch-as-unique shortcut — that re-ran the child recursion in
         // <unique>, re-checking each child's refcount and mutating a committed child in
         // place, diverging from the proven flat path and over-releasing a root to ref 0
         // → snap::acquire livelock). The edit body runs ONCE with CM threaded through:
         //   • UNIQUE: the node is private — re-pack the SAME slot, release a
         //     replaced/dropped child directly, recurse children <unique>.
         //   • SHARED: the node is snapshot-pinned — read its entries from a PRIVATE
         //     scratch copy (never touch the live node), recurse the matched child
         //     <shared> (so a COMMITTED child gets CLONED, never mutated in place);
         //     do NOT release carry-over children inline (the old node's eventual drain
         //     cascade does that). The carry-over retains + the single old-node
         //     release_ref happen at EMIT (emit_nested_<shared>), mirroring rebuild<shared>.
         // Snapshot safety: before a <shared> child recursion we RETAIN that child (so the
         // recursion's internal release consumes OUR retain, leaving the snapshot's ref —
         // held via the old node's still-live body — intact); that retain is net-zero
         // (recursion consumes it), so the fresh child arrives owned and is NOT re-retained
         // at emit. This is exactly reprefix(0)'s "retain all children up front" split into
         // {the recursed child} + {the carry-overs}.
         constexpr bool is_unique = (CM == cow_mode::unique);

         // Decode the current entries into a path-agnostic set (read from a private
         // scratch copy so a same-slot rewrite can read while it overwrites). A
         // nested_setlist is 128 bytes so it holds far fewer than 64 entries; sizing the
         // scratch tightly (vs 257) keeps the per-frame footprint small — this descent
         // recurses once per nested level, and a fat frame would overflow the stack on a
         // deep key.
         char     scratch128[128];
         nentry   es[nested_setlist::es_cap];
         uint32_t ne;
         bool     term;
         read_nested_entries_(nn, scratch128, es, ne, term);
         const uint32_t nbr = ne - (term ? 1u : 0u);

         // staging for any NEW inline-sv body this edit mints (build_sv_body_, pure
         // bytes — never make_sv'd for a carry-over). Kept alive across the re-emit.
         char     newbody[127];
         bool     inc_size   = false;
         // a FRESHLY-OWNED branch this edit minted (a make_sv external leaf, a deeper-
         // split child, or a recursed clone) — owned by us, so the shared re-emit must
         // NOT retain it (only genuine carry-overs from the old node are co-owned with
         // the snapshot). have_fresh tracks the ONE such entry an edit produces.
         bool     have_fresh = false;
         uint32_t fresh_idx  = 0;
         // mint an EXTERNAL leaf into entry idx (value too big to inline) + mark it fresh.
         auto mint_external = [&](uint32_t idx, sv suffix)
         {
            auto mv          = make_sv(suffix, val);
            es[idx].kind     = 0;
            es[idx].otype    = mv.first;
            es[idx].ext      = mv.second;
            have_fresh       = true;
            fresh_idx        = idx;
         };

         if (depth == key.size())  // TERM
         {
            // find the term entry (last, index ne-1) if present
            if (term)
            {
               const uint32_t ti = ne - 1;
               nentry&        te = es[ti];
               if (te.kind == 0)  // external term → release old leaf (unique only), REP with a fresh inline-sv (or external if too big)
               {
                  if (is_unique)
                     release_ref(te.otype, te.ext);  // shared: old node cascades it later
                  if (sv_body_fits_(sv{}, val))
                  {
                     te.kind  = 1;
                     te.otype = bt::suffix_value;
                     te.body  = newbody;
                     te.blen  = build_sv_body_(newbody, sv{}, val);
                  }
                  else
                     mint_external(ti, sv{});
               }
               else  // inline-sv term (data) → REP its body bytes
               {
                  if (sv_body_fits_(sv{}, val))
                  {
                     te.kind = 1;
                     te.body = newbody;
                     te.blen = build_sv_body_(newbody, sv{}, val);
                  }
                  else  // grew past inline budget → externalize
                     mint_external(ti, sv{});
               }
            }
            else  // absent term → append a new inline-sv (0-suffix) term
            {
               const uint32_t ti = ne;  // term goes LAST
               nentry&        te = es[ne++];
               te.okey           = 0;
               if (sv_body_fits_(sv{}, val))
               {
                  te.kind  = 1;
                  te.otype = bt::suffix_value;
                  te.body  = newbody;
                  te.blen  = build_sv_body_(newbody, sv{}, val);
               }
               else
                  mint_external(ti, sv{});
               term     = true;
               inc_size = true;
            }
         }
         else  // BYTE entry
         {
            const uint8_t byte = uint8_t(key[depth]);
            // locate among the byte-branches (sorted okeys, indices [0,nbr))
            int pos = -1;
            for (uint32_t i = 0; i < nbr; ++i)
               if (es[i].okey == byte)
               {
                  pos = int(i);
                  break;
               }
            if (pos < 0)  // ABSENT → INS a new inline-sv (or external big), sorted
            {
               // find the sorted insert position (term, if any, stays LAST)
               uint32_t ip = 0;
               while (ip < nbr && es[ip].okey < byte)
                  ++ip;
               for (uint32_t i = ne; i > ip; --i)
                  es[i] = es[i - 1];
               ++ne;
               nentry& nE = es[ip];
               nE.okey    = byte;
               if (sv_body_fits_(key.substr(depth + 1), val))
               {
                  nE.kind  = 1;
                  nE.otype = bt::suffix_value;
                  nE.body  = newbody;
                  nE.blen  = build_sv_body_(newbody, key.substr(depth + 1), val);
               }
               else  // value too big to inline → external leaf (owned, mark fresh)
                  mint_external(ip, key.substr(depth + 1));
               // if have_fresh was already set for an earlier index it cannot happen
               // here (one op per edit); fresh_idx (if mint) is the new index ip.
               inc_size = true;
            }
            else if (es[uint32_t(pos)].kind == 0)  // PRESENT external → recurse, REP its ptr
            {
               nentry& ce = es[uint32_t(pos)];
               // SHARED: retain the child BEFORE recursing <shared> so the recursion's own
               // COW release consumes OUR retain (net-zero), leaving the snapshot's
               // reference (held via the still-live old node body) intact. The fresh ptr
               // arrives owned, so emit must NOT retain it (it is the fresh branch).
               if constexpr (CM == cow_mode::shared)
                  retain(_e, *_lses, ce.otype, ce.ext);
               auto nc    = upsert_rec<CM>(ce.otype, ce.ext, key, depth + 1, val);
               ce.otype   = nc.first;
               ce.ext     = nc.second;
               ce.kind    = 0;
               have_fresh = true;
               fresh_idx  = uint32_t(pos);
            }
            else if (es[uint32_t(pos)].kind == 1)  // PRESENT inline-sv
            {
               nentry& ce = es[uint32_t(pos)];
               // decode its suffix from the scratch body
               uint16_t v16, s16;
               std::memcpy(&v16, ce.body, 2);
               std::memcpy(&s16, ce.body + 2, 2);
               const sv esuf{ce.body + sv_leaf::hdr + v16, s16};
               if (esuf == key.substr(depth + 1))  // SAME key → REP its value bytes
               {
                  if (sv_body_fits_(esuf, val))
                  {
                     // esuf views ce.body (scratch128); build into newbody, THEN repoint
                     // ce.body — read esuf before it is shadowed by the write.
                     uint16_t bl = build_sv_body_(newbody, esuf, val);
                     ce.body     = newbody;
                     ce.blen     = bl;
                  }
                  else  // grew past inline → externalize (owned, mark fresh)
                     mint_external(uint32_t(pos), esuf);
               }
               else  // DEEPER SPLIT: the two leaves share key[depth] but diverge below
               {
                  // build a small child node over {existing inline leaf, new leaf};
                  // store it inline (kind2 setlist) if it fits, else external. The
                  // child is built fresh (its leaves are owned), so on a shared clone
                  // it is NOT retained — mark it the fresh branch.
                  std::string esuf_s(esuf);
                  std::string eval_s(sv{ce.body + sv_leaf::hdr, v16});
                  const sv    nsuf = key.substr(depth + 1);
                  const uint32_t cp2 = lcp(esuf_s, nsuf);
                  br          brs[2];
                  uint32_t    cnb  = 0;
                  bool        cterm = false;
                  bt          ctt  = bt::null;
                  uint32_t    ctp  = 0;
                  auto place = [&](const std::string& s, sv v)
                  {
                     if (cp2 == s.size())
                     {
                        cterm   = true;
                        auto t2 = make_sv(sv{}, v);
                        ctt     = t2.first;
                        ctp     = t2.second;
                     }
                     else
                     {
                        auto t2     = make_sv(sv(s).substr(cp2 + 1), v);
                        brs[cnb++]  = br{uint8_t(s[cp2]), t2.first, t2.second};
                     }
                  };
                  place(esuf_s, eval_s);
                  place(std::string(nsuf), val);
                  if (cnb == 2 && brs[0].byte > brs[1].byte)
                     std::swap(brs[0], brs[1]);
                  // The OLD inline-sv body is pure data (no refcount) — it is simply
                  // replaced; nothing to release. build the child node (flat, no inline
                  // — its leaves must stay external so the child is a real setlist).
                  auto child = build_node(esuf_s.substr(0, cp2), brs, cnb, cterm, ctt, ctp,
                                          /*may_inline=*/false);
                  // try to inline the child as kind2 if it is a setlist that fits
                  if (child.first == bt::setlist && _e.cl1.unique(child.second))
                  {
                     const setlist* c = static_cast<const setlist*>(_e.cl1.get(child.second));
                     const uint32_t b = setlist::bytes_for_cap(c->total(), c->prefixlen());
                     if (b <= 127)
                     {
                        char* dst = newbody;  // reuse the stack body buffer
                        copy_setlist_tight(dst, c);
                        if (_e.cl1.release_one(child.second))
                           _e.cl1.free(child.second);  // grandchildren transfer into the line
                        ce.kind  = 2;
                        ce.otype = bt::setlist;
                        ce.body  = dst;
                        ce.blen  = uint16_t(b);
                        // The kind2 grandchildren are FRESH (make_sv, owned) — mark
                        // this entry the fresh branch so the shared re-emit does NOT
                        // retain them (they arrive owned, not co-owned with a snapshot).
                        have_fresh = true;
                        fresh_idx  = uint32_t(pos);
                     }
                     else
                     {
                        ce.kind  = 0;
                        ce.otype = child.first;
                        ce.ext   = child.second;
                        have_fresh = true;
                        fresh_idx  = uint32_t(pos);
                     }
                  }
                  else
                  {
                     ce.kind  = 0;
                     ce.otype = child.first;
                     ce.ext   = child.second;
                     have_fresh = true;
                     fresh_idx  = uint32_t(pos);
                  }
                  inc_size = true;
               }
            }
            else  // PRESENT inline-setlist (kind2) → externalize, recurse, keep external
            {
               nentry& ce = es[uint32_t(pos)];
               // Alloc a standalone cl1 setlist from the inline bytes (pure copy — its u32
               // grandchildren are aliased, not retained, by externalize itself). SHARED:
               // RETAIN those grandchildren first so the externalized private slot is fully
               // co-owned with the snapshot's inline body — then the <unique> recursion
               // clones any grandchild it touches and its release consumes our retain
               // (net-zero), leaving the snapshot's grandchild refs intact. The carried-over
               // grandchildren end up OWNED by the re-inlined/external result, so emit must
               // NOT retain them again (this entry is marked fresh). UNIQUE: the inline
               // setlist was private, so its grandchildren are already private — no retain.
               const setlist* insl = reinterpret_cast<const setlist*>(ce.body);
               if constexpr (CM == cow_mode::shared)
               {
                  for (uint32_t j = 0; j < insl->nbranch; ++j)
                     retain(_e, *_lses, insl->btype(j), insl->branch(j));
                  if (insl->has_term())
                     retain(_e, *_lses, insl->term_type(), insl->term_branch());
               }
               uint32_t s = externalize_inline_setlist_(ce.body, ce.blen);
               // route the next byte THROUGH this now-external (fully owned) setlist.
               auto nc = upsert_rec<cow_mode::unique>(bt::setlist, s, key, depth + 1, val);
               // try to RE-INLINE if it is still a setlist that fits
               bool reinlined = false;
               if (nc.first == bt::setlist && _e.cl1.unique(nc.second))
               {
                  const setlist* c = static_cast<const setlist*>(_e.cl1.get(nc.second));
                  const uint32_t b = setlist::bytes_for_cap(c->total(), c->prefixlen());
                  if (b <= 127)
                  {
                     char* dst = newbody;
                     copy_setlist_tight(dst, c);
                     if (_e.cl1.release_one(nc.second))
                        _e.cl1.free(nc.second);
                     ce.kind   = 2;
                     ce.otype  = bt::setlist;
                     ce.body   = dst;
                     ce.blen   = uint16_t(b);
                     reinlined = true;
                     // grandchildren came from `s` (which the shared externalize already
                     // retained, or which were private in unique mode) — they are now
                     // OWNED by this inline body, so the shared re-emit must NOT retain
                     // them again. Mark the entry fresh.
                     have_fresh = true;
                     fresh_idx  = uint32_t(pos);
                  }
               }
               if (!reinlined)
               {
                  ce.kind    = 0;
                  ce.otype   = nc.first;
                  ce.ext     = nc.second;
                  have_fresh = true;
                  fresh_idx  = uint32_t(pos);
               }
            }
         }

         return emit_nested_<CM>(t, p, pfx, es, ne, term, have_fresh, fresh_idx, inc_size);
      }

      // Build a FLAT routing node (setlist / double / full) from an ALREADY-EDITED
      // nentry set — the overflow / de-nest path. The recursion + make_sv side-effects
      // are DONE (the nentry set incorporates them), so we must NOT re-run the edit;
      // we convert this set into standalone children exactly as promote_nested would,
      // but reading from the mutated set, not the node. Inline-sv → fresh standalone
      // leaf (make_sv: the inline body is data with no CB). Inline-setlist → standalone
      // cl1 (copy body; grandchildren transfer). External → keep the ptr. Then drop the
      // OLD nested node: UNIQUE husk-frees it (its inline data dies with the slot, its
      // external carry-overs transfer); SHARED retains the carry-over externals +
      // inline-setlist grandchildren first, then release_ref once.
      std::pair<bt, uint32_t> nentry_to_flat_(bt t, uint32_t p, bool is_unique, sv pfx,
                                              const nentry* es, uint32_t ne, bool term,
                                              bool have_fresh, uint32_t fresh_idx)
      {
         const uint32_t nbr = ne - (term ? 1u : 0u);
         br             scratch[257];
         uint32_t       nb = 0;
         bt             tt = bt::null;
         uint32_t       tp = 0;
         auto resolve_entry = [&](const nentry& e, uint8_t& okeyo, bt& tago, uint32_t& ptro)
         {
            okeyo = e.okey;
            if (e.kind == 0)  // external — keep the ptr
            {
               tago = e.otype;
               ptro = e.ext;
            }
            else if (e.kind == 2)  // inline setlist → standalone cl1 (copy body)
            {
               uint32_t s = _s1.alloc();
               std::memcpy(_e.cl1.get(s), e.body, e.blen);
               tago = bt::setlist;
               ptro = s;
            }
            else  // inline-sv → fresh standalone SAL leaf
            {
               uint16_t v16, s16;
               std::memcpy(&v16, e.body, 2);
               std::memcpy(&s16, e.body + 2, 2);
               const sv val{e.body + sv_leaf::hdr, v16};
               const sv suf{e.body + sv_leaf::hdr + v16, s16};
               auto     mv = make_sv(suf, val);
               tago        = mv.first;
               ptro        = mv.second;
            }
         };
         for (uint32_t i = 0; i < nbr; ++i)
         {
            br b;
            resolve_entry(es[i], b.byte, b.tag, b.ptr);
            scratch[nb++] = b;
         }
         if (term)
         {
            uint8_t dummy;
            resolve_entry(es[ne - 1], dummy, tt, tp);
         }
         auto out = build_node(pfx, scratch, nb, term, tt, tp, /*may_inline=*/false);
         if (is_unique)
         {
            // external carry-overs + inline-setlist grandchildren TRANSFERRED into the
            // standalone children / new node; inline-sv data died with the old slot's
            // copy but was already re-minted via make_sv above (so the old standalone
            // leaf, if any external arm, transferred). Husk-free the old nested slot.
            release_husk(bt::nested_setlist, p);
         }
         else
         {
            // SHARED: the new flat node's standalone children co-reference the carry-over
            // externals + inline-setlist grandchildren (kind2 allocs a fresh standalone
            // setlist whose u32 grandchildren ALIAS the old ones; kind0 keeps the ptr).
            // Retain each such carry-over (EXCEPT the freshly-owned branch — a recursed/
            // re-inlined/make_sv child arrives ref 1) so it is co-owned by old + new, then
            // drop ONE ref on the old node (snapshot keeps its copy; its eventual drain
            // cascade balances these retains). inline-sv → re-minted via make_sv (owned),
            // never retained. EXACTLY rebuild<shared>'s discipline on the de-nested path.
            retain_shared_carryovers_(es, ne, have_fresh, fresh_idx);
            release_ref(t, p);
         }
         return out;
      }

      // Re-emit a nested line from a mutated nentry set. UNIQUE re-packs the SAME
      // slot p; SHARED allocs a fresh cl1 slot. The carried-over external children +
      // inline-setlist grandchildren are RETAINED on a shared clone (the fresh branch
      // is NOT — it arrives owned); inline-sv bodies are pure data, never retained. On
      // a genuine 128B overflow the ALREADY-EDITED set is built flat (nentry_to_flat_)
      // — NOT re-run from the original (the recursion side-effects are already done).
      template <cow_mode CM>
      std::pair<bt, uint32_t> emit_nested_(bt t, uint32_t p, sv pfx, const nentry* es, uint32_t ne,
                                           bool term, bool have_fresh, uint32_t fresh_idx,
                                           bool inc_size)
      {
         if constexpr (CM == cow_mode::unique)
         {
            // Same-slot re-pack. lay_nested_into_ size-checks BEFORE writing, so on
            // overflow the original slot is still intact in p (we read the set from a
            // private scratch128, not p) → flat-build from the edited set safely.
            nested_setlist* n = static_cast<nested_setlist*>(resolve(_e, t, p));
            if (lay_nested_into_(n, pfx, es, ne, term))
            {
               if (inc_size)
                  ++_size;
               return {t, p};
            }
            auto out = nentry_to_flat_(t, p, /*is_unique=*/true, pfx, es, ne, term, have_fresh,
                                       fresh_idx);
            if (inc_size)
               ++_size;
            return out;
         }
         else
         {
            // SHARED: alloc a fresh slot, re-pack into it, then RETAIN the fresh node's
            // carry-over children (every refcountable entry EXCEPT the freshly-owned one)
            // and drop ONE ref on the old node — EXACTLY rebuild<shared>'s "retain copied
            // children except is_new, release old once" (the carry-over retains are NOT
            // pre-done by the body; the body only retained the ONE child it recurse-clones,
            // which the recursion consumed). The snapshot keeps the old node's copy; its
            // eventual drain cascade balances these retains.
            const uint32_t  s  = _s1.alloc();
            nested_setlist* ns = static_cast<nested_setlist*>(_e.cl1.get(s));
            if (lay_nested_into_(ns, pfx, es, ne, term))
            {
               retain_shared_carryovers_(es, ne, have_fresh, fresh_idx);
               release_ref(t, p);  // drop one ref on the old node (snapshot keeps its copy)
               if (inc_size)
                  ++_size;
               return {bt::nested_setlist, s};
            }
            // OVERFLOW on the fresh slot: free it, flat-build from the edited set —
            // nentry_to_flat_<shared> does its OWN carry-over retains + the single old-node
            // release_ref (the recursion side-effects are already incorporated in es[]).
            _s1.free(s);
            auto out = nentry_to_flat_(t, p, /*is_unique=*/false, pfx, es, ne, term, have_fresh,
                                       fresh_idx);
            if (inc_size)
               ++_size;
            return out;
         }
      }

      void set_term_inplace(bt t, uint32_t p, std::pair<bt, uint32_t> nv)
      {
         if (t == bt::full)
         {
            auto* n     = static_cast<full*>(resolve(_e, t, p));
            n->term_tag = uint8_t(nv.first);
            n->term_ptr = nv.second;
         }
         else
         {
            auto* n = static_cast<setlist*>(resolve(_e, t, p));
            tag3_set(n->types(), n->nbranch, nv.first);
            n->branches()[n->nbranch] = nv.second;
         }
      }
      void set_branch_inplace(bt t, uint32_t p, uint8_t byte, std::pair<bt, uint32_t> nc)
      {
         if (t == bt::full)
         {
            auto* n = static_cast<full*>(resolve(_e, t, p));
            tag3_set(n->types(), byte, nc.first);
            n->branches()[byte] = nc.second;
         }
         else
         {
            auto*     n   = static_cast<setlist*>(resolve(_e, t, p));
            const int idx = n->find_index(byte);
            tag3_set(n->types(), uint32_t(idx), nc.first);
            n->branches()[uint32_t(idx)] = nc.second;
         }
      }

      // Reprefix a routing node to prefix[drop:] (build a copy, retain children,
      // release old). Returns the new (tag, ptr). drop==0 is the COW clone hot
      // path: same slot class → verbatim memcpy (no scratch/re-pack) + retain.
      std::pair<bt, uint32_t> reprefix(bt t, uint32_t p, uint32_t drop)
      {
         if (drop == 0)
         {
            std::pair<bt, uint32_t> out;
            if (t == bt::full)
            {
               uint32_t s = _s9.alloc();
               std::memcpy(_e.cl9.get(s), resolve(_e, t, p), full::slot_bytes);
               out = {bt::full, s};
            }
            else if (t == bt::double_setlist || t == bt::big_nested)
            {
               uint32_t s = _s2.alloc();
               std::memcpy(_e.cl2.get(s), resolve(_e, t, p), 256);
               out = {t, s};
            }
            else
            {
               uint32_t s = _s1.alloc();
               std::memcpy(_e.cl1.get(s), resolve(_e, t, p), 128);
               out = {t, s};
            }
            bt       tt;
            uint32_t tp;
            read_branches(out.first, out.second, [&](const br& b) { retain(_e, *_lses, b.tag, b.ptr); });
            if (node_term(out.first, out.second, tt, tp))
               retain(_e, *_lses, tt, tp);
            release_ref(t, p);
            return out;
         }
         br         scratch[257];
         uint32_t   nb   = 0;
         bt         tt   = bt::null;
         uint32_t   tp   = 0;
         const bool term = node_term(t, p, tt, tp);
         read_branches(t, p, [&](const br& b) { scratch[nb++] = b; });
         std::string np(node_prefix(t, p).substr(drop));
         auto        out = build_node(np, scratch, nb, term, tt, tp);
         read_branches(out.first, out.second, [&](const br& b) { retain(_e, *_lses, b.tag, b.ptr); });
         if (term)
            retain(_e, *_lses, tt, tp);
         release_ref(t, p);
         return out;
      }

      // Un-nest a nested_setlist into a flat structure: each inner setlist becomes
      // a STANDALONE cl1 setlist, and a flat OUTER setlist points to them. Returns
      // the flat (tag,ptr). The write path promotes-on-write so every edit path
      // only ever mutates flat nodes (nested is effectively read-only after create;
      // re-nesting happens later in build_node). PRIVATE: inner u32 children are
      // TRANSFERRED to the new setlists, the nested slot is just freed. SHARED: the
      // new setlists co-reference the children, so retain each then release the old
      // nested (its ref>1 so it survives for the snapshot).
      std::pair<bt, uint32_t> promote_nested(bt t, uint32_t p)
      {
         const bool      priv  = unique(_e, t, p);
         nested_setlist* n     = static_cast<nested_setlist*>(resolve(_e, t, p));
         const uint32_t  onb   = n->nbranch;
         const bool      oterm = n->has_term();
         br              scratch[257];
         uint32_t        nb = 0;
         for (uint32_t i = 0; i < onb; ++i)
         {
            if (!n->is_inline(i))  // external → keep the ptr
            {
               scratch[nb++] = br{n->okeys()[i], n->otype(i), n->ext_ptr(i)};
               continue;
            }
            if (n->otype(i) == bt::setlist)  // inline setlist → standalone cl1 (copy body)
            {
               uint32_t s = _s1.alloc();
               std::memcpy(_e.cl1.get(s), n->entry(i), n->entry_size(i));
               scratch[nb++] = br{n->okeys()[i], bt::setlist, s};
            }
            else  // inline suffix_value → fresh standalone SAL leaf
            {
               // A SAL leaf needs an alloc_header + CB that a raw memcpy of the
               // inline body would NOT produce, so reconstruct via make_sv (the inline
               // body header is still [u16 vlen][u16 slen], so in_suffix/in_value read
               // it unchanged). make_sv routes through _lses->alloc<sv_leaf>.
               auto lv = make_sv(n->in_suffix(i), n->in_value(i));
               scratch[nb++] = br{n->okeys()[i], lv.first, lv.second};
            }
         }
         bt       tt = bt::null;
         uint32_t tp = 0;
         if (oterm)
         {
            const uint32_t ti = onb;
            if (!n->is_inline(ti))
            {
               tt = n->otype(ti);
               tp = n->ext_ptr(ti);
            }
            else
            {
               auto lv = make_sv(sv{}, n->in_value(ti));
               tt      = lv.first;
               tp      = lv.second;
            }  // inline term → 0-suffix SAL leaf
         }
         auto out = build_node(n->prefix(), scratch, nb, oterm, tt, tp);
         if (priv)
         {
            // external ptrs + inline-setlist grandchildren TRANSFERRED; inline data
            // dropped with the slot. release the husk (ref 1→0→free, no cascade).
            release_husk(bt::nested_setlist, p);
         }
         else  // shared: the new flat co-references the external/grandchild ptrs → retain, then drop one nested ref
         {
            for_each_nested_branch_(n, [&](bt ct, uint32_t cp) { retain(_e, *_lses, ct, cp); });
            if (oterm && !n->is_inline(onb))
               retain(_e, *_lses, n->otype(onb), n->ext_ptr(onb));
            release_ref(t, p);
         }
         return out;
      }

      // ── remove path (mirrors upsert_rec/edit_node; presence is guaranteed by
      //    the caller's find_in pre-check, so there is NO not-found branch) ──────
      // Descend to the leaf (consuming the matched prefix), removing it; on the
      // way back up REP the changed child, or — when a child's subtree vanished —
      // rebuild this node without that branch. A node that drops to zero entries
      // returns {null,0} (the parent then drops ITS branch, cascading up). The
      // rebuild re-runs build_node(may_inline) so a node that shrank back to a
      // line of small children RE-NESTS (Dan: merge into a nested setlist on
      // remove, don't leave it expanded) and de-widens full→double→setlist.
      template <cow_mode M>
      std::pair<bt, uint32_t> remove_rec(bt t, uint32_t p, sv key, uint32_t depth)
      {
         if (is_leaf_tag(t))  // the leaf (find_in guaranteed the match)
         {
            release_ref(t, p);
            return {bt::null, 0};
         }
         if (t == bt::nested_setlist)  // DIRECTLY MUTABLE — re-pack in place (no promote)
         {
            // Latch (mirrors the flat remove dispatch @below): private → same-slot
            // <unique>; snapshot-pinned → fresh-slot clone <shared>.
            if constexpr (M == cow_mode::unique)
               if (unique(_e, t, p))
                  return nested_remove<cow_mode::unique>(t, p, key, depth);
            return nested_remove<cow_mode::shared>(t, p, key, depth);
         }
         // full: prefetch the offset CL for the speculative routing byte before
         // node_prefix touches the header (cold-miss overlap window, §7).
         if (t == bt::full && depth < key.size())
            prefetch(static_cast<const char*>(resolve(_e, t, p)) + full::branches_off +
                     (uint32_t(uint8_t(key[depth])) << 2));
         depth += uint32_t(node_prefix(t, p).size());  // prefix matched (pre-check); consume it
         // Overlap the cold refcount load (.rc array) with the node-body load:
         // resolve() is pure address arithmetic (no deref), so prefetch the body
         // BEFORE unique() — the body is in flight while the refcount stalls, and
         // edit_remove dereferences it resident (one memory latency, not two).
         // Pure hint; correctness-neutral.
         if constexpr (M == cow_mode::unique)
         {
            prefetch(resolve(_e, t, p));
            if (unique(_e, t, p))
               return edit_remove<cow_mode::unique>(t, p, key, depth);
         }
         return edit_remove<cow_mode::shared>(t, p, key, depth);
      }

      // Mutate a routing node to drop the key. Like edit_node: <unique> mutates
      // (t,p) (W is private); <shared> clones to a private W (reprefix retains the
      // children + drops one old ref). Either way W is private, so the rebuild is
      // the unique-path one.
      template <cow_mode CM>
      std::pair<bt, uint32_t> edit_remove(bt t, uint32_t p, sv key, uint32_t depth)
      {
         std::pair<bt, uint32_t> W;
         if constexpr (CM == cow_mode::unique)
            W = {t, p};
         else
            W = reprefix(t, p, 0);  // clone the shared node → private W
         const bt       wt = W.first;
         const uint32_t wp = W.second;

         if (depth == key.size())  // remove the TERM at this node
         {
            bt       tt;
            uint32_t tp;
            node_term(wt, wp, tt, tp);  // guaranteed present
            release_ref(tt, tp);
            return rebuild_remove(wt, wp, /*del_term=*/true, /*del_byte=*/false, 0);
         }
         const uint8_t byte = uint8_t(key[depth]);
         bt            ct;
         uint32_t      cpp;
         node_child(wt, wp, byte, ct, cpp);  // guaranteed present
         auto nc = remove_rec<CM>(ct, cpp, key, depth + 1);
         if (nc.first == bt::null)  // child subtree vanished → drop this branch
            return rebuild_remove(wt, wp, /*del_term=*/false, /*del_byte=*/true, byte);
         set_branch_inplace(wt, wp, byte, nc);  // child collapsed/changed → REP the slot
         return W;
      }

      // Rebuild a PRIVATE node minus one entry (a byte-branch or the term). The
      // dropped entry's child was ALREADY released by the caller (recursion /
      // term release). count==0 → free the husk, return {null,0}. Otherwise
      // build_node TRANSFERS the surviving children (no retain — they move from
      // the freed husk to the new node) and may re-nest / de-widen.
      std::pair<bt, uint32_t> rebuild_remove(bt       t,
                                             uint32_t p,
                                             bool     del_term,
                                             bool     del_byte,
                                             uint8_t  byte)
      {
         br       scratch[257];
         uint32_t nb   = 0;
         bt       tt   = bt::null;
         uint32_t tp   = 0;
         bool     term = node_term(t, p, tt, tp);
         read_branches(t, p,
                       [&](const br& b)
                       {
                          if (!(del_byte && b.byte == byte))
                             scratch[nb++] = b;
                       });
         if (del_term)
            term = false;
         if (nb + (term ? 1u : 0u) == 0)
         {
            release_husk(t, p);
            return {bt::null, 0};
         }
         const std::string pfx(node_prefix(t, p));  // stable (husk freed below)
         auto              out = build_node(pfx, scratch, nb, term, tt, tp, /*may_inline=*/true);
         release_husk(t, p);  // children transferred to `out` (ref 1→0→free, no cascade)
         return out;
      }

      // ── DIRECTLY-MUTABLE nested_setlist remove (mirrors nested_edit) ────────────
      // Presence is guaranteed by the caller's find_in pre-check. Decode the entries,
      // drop / REP the matched one, re-emit through emit_nested_ (a remove only ever
      // SHRINKS the line, so a same-slot UNIQUE re-pack never overflows). On shrink to
      // zero the husk is freed and {null,0} bubbles up. `depth` is at node ENTRY.
      template <cow_mode CM>
      std::pair<bt, uint32_t> nested_remove(bt t, uint32_t p, sv key, uint32_t depth)
      {
         // SINGLE clone+remove MIRRORING the flat edit_remove<shared> discipline (NOT a
         // clone-then-re-dispatch-as-unique shortcut). One body, CM threaded through:
         //   • UNIQUE: private — drop/REP the matched entry in place, re-pack the same slot.
         //   • SHARED: snapshot-pinned — read entries from a PRIVATE scratch (never touch the
         //     live node), recurse the matched child <shared> (a COMMITTED child gets CLONED,
         //     never mutated/freed in place); do NOT release dropped entries inline (the old
         //     node's eventual drain cascade does). The carry-over retains + the single
         //     old-node release_ref happen at EMIT (mirroring rebuild<shared>).
         nested_setlist* nn  = static_cast<nested_setlist*>(resolve(_e, t, p));
         const sv        pfx = nn->prefix();
         depth += uint32_t(pfx.size());  // prefix matched (find_in pre-check); consume it

         char     scratch128[128];
         nentry   es[nested_setlist::es_cap];
         uint32_t ne;
         bool     term;
         read_nested_entries_(nn, scratch128, es, ne, term);
         const uint32_t nbr = ne - (term ? 1u : 0u);

         char     reinline_body[127];  // staging for a re-inlined setlist (separate from scratch128)
         bool     have_fresh = false;
         uint32_t fresh_idx  = 0;
         // UNIQUE: a dropped child is released directly, the recursion mutates in place.
         // SHARED: dropped entries are NOT released here (the old node's drain cascade does);
         // a recursed child is retained BEFORE the <shared> recursion so the recursion's COW
         // release consumes our retain, keeping the snapshot's reference intact.
         constexpr bool is_unique = (CM == cow_mode::unique);

         if (depth == key.size())  // remove the TERM
         {
            // term is the last entry (index ne-1). external → release; inline → data.
            nentry& te = es[ne - 1];
            if (te.kind == 0 && is_unique)
               release_ref(te.otype, te.ext);  // shared: old node cascades it later
            // drop the term entry
            --ne;
            term = false;
         }
         else  // remove a byte entry
         {
            const uint8_t byte = uint8_t(key[depth]);
            int           pos  = -1;
            for (uint32_t i = 0; i < nbr; ++i)
               if (es[i].okey == byte)
               {
                  pos = int(i);
                  break;
               }
            // pos must be >= 0 (find_in guaranteed presence)
            const uint32_t pi = uint32_t(pos);
            if (es[pi].kind == 0)  // external child → recurse
            {
               // SHARED: retain the child BEFORE recursing <shared> so the recursion's own
               // COW release (clone-and-replace, or full drop on vanish) consumes OUR retain
               // (net-zero), leaving the snapshot's reference (via the still-live old node)
               // intact. A surviving fresh ptr arrives owned (marked fresh, not re-retained);
               // a vanished child's entry is simply dropped (not a carry-over).
               if constexpr (CM == cow_mode::shared)
                  retain(_e, *_lses, es[pi].otype, es[pi].ext);
               auto nc = remove_rec<CM>(es[pi].otype, es[pi].ext, key, depth + 1);
               if (nc.first == bt::null)  // child subtree vanished → drop this entry
               {
                  for (uint32_t i = pi; i + 1 < ne; ++i)
                     es[i] = es[i + 1];
                  --ne;
               }
               else  // child changed → REP its ptr (the recursion owns the new ptr)
               {
                  es[pi].otype = nc.first;
                  es[pi].ext   = nc.second;
                  have_fresh   = true;
                  fresh_idx    = pi;
               }
            }
            else if (es[pi].kind == 1)  // inline-sv leaf (the matched key) → DROP (pure data)
            {
               for (uint32_t i = pi; i + 1 < ne; ++i)
                  es[i] = es[i + 1];
               --ne;
            }
            else  // inline-setlist → externalize, recurse, then DROP-or-REP
            {
               // SHARED: RETAIN the grandchildren (they alias the snapshot's inline body)
               // before externalizing so the externalized private slot is fully co-owned;
               // the <unique> recursion then clones any grandchild it touches and its
               // release consumes our retain (net-zero), leaving the snapshot's refs intact.
               // The surviving result OWNS the carried-over grandchildren (entry marked
               // fresh → not re-retained at emit); a vanished setlist drops the entry.
               // UNIQUE: the inline setlist was private, its grandchildren private — no retain.
               const setlist* insl = reinterpret_cast<const setlist*>(es[pi].body);
               if constexpr (CM == cow_mode::shared)
               {
                  for (uint32_t j = 0; j < insl->nbranch; ++j)
                     retain(_e, *_lses, insl->btype(j), insl->branch(j));
                  if (insl->has_term())
                     retain(_e, *_lses, insl->term_type(), insl->term_branch());
               }
               uint32_t s = externalize_inline_setlist_(es[pi].body, es[pi].blen);
               auto nc = remove_rec<cow_mode::unique>(bt::setlist, s, key, depth + 1);
               if (nc.first == bt::null)  // shrank to empty → drop
               {
                  for (uint32_t i = pi; i + 1 < ne; ++i)
                     es[i] = es[i + 1];
                  --ne;
               }
               else  // try to RE-INLINE the surviving setlist, else keep external
               {
                  bool reinlined = false;
                  if (nc.first == bt::setlist && _e.cl1.unique(nc.second))
                  {
                     const setlist* c = static_cast<const setlist*>(_e.cl1.get(nc.second));
                     const uint32_t b = setlist::bytes_for_cap(c->total(), c->prefixlen());
                     if (b <= 127)
                     {
                        // re-inline into a SEPARATE staging buffer (scratch128 still
                        // backs the OTHER carry-over inline entries — must not stomp it).
                        char* dst = reinline_body;
                        copy_setlist_tight(dst, c);
                        if (_e.cl1.release_one(nc.second))
                           _e.cl1.free(nc.second);
                        es[pi].kind  = 2;
                        es[pi].otype = bt::setlist;
                        es[pi].body  = dst;
                        es[pi].blen  = uint16_t(b);
                        reinlined    = true;
                        // grandchildren are OWNED by this inline body (externalize
                        // retained them in shared mode) — don't re-retain on re-emit.
                        have_fresh   = true;
                        fresh_idx    = pi;
                     }
                  }
                  if (!reinlined)
                  {
                     es[pi].kind  = 0;
                     es[pi].otype = nc.first;
                     es[pi].ext   = nc.second;
                     have_fresh   = true;
                     fresh_idx    = pi;
                  }
               }
            }
         }

         if (ne == 0)  // line emptied → free the husk, cascade {null,0} up
         {
            if constexpr (CM == cow_mode::unique)
               release_husk(bt::nested_setlist, p);
            else
               release_ref(t, p);  // shared: drop one ref (snapshot keeps its copy)
            return {bt::null, 0};
         }
         // remove never grows the line → the same-slot re-pack always fits.
         return emit_nested_<CM>(t, p, pfx, es, ne, term, have_fresh, fresh_idx,
                                 /*inc_size=*/false);
      }

      engine&                              _e;
      sal_hybrid::fixed_allocator<128>::session  _s1;
      sal_hybrid::fixed_allocator<256>::session  _s2;
      sal_hybrid::fixed_allocator<1152>::session _s9;
      sal_hybrid::allocator_session_ptr          _lses;  // SAL leaf session (this work tree's own)
      bt                                   _rt   = bt::null;
      uint32_t                             _rp   = 0;
      uint64_t                             _size = 0;
      std::string _scan;  // reused buffer for remove()'s presence pre-check (no per-call alloc)
   };
}  // namespace rcart

// gcart — the spec node taxonomy over the three regions (version-liveness COW:
// the in-place-vs-clone latch is BIRTH==V; refcount survives only for reclamation).
//
// Routing lives in FIX (cl1 setlist, cl2 double, cl9 full); terminal (suffix,
// value) leaves live in SAL (32B); large values in BIG. A branch is a 4-byte
// region-typed pointer: its 3-bit `branch_type` tag (in the parent's packed
// types[] array) selects the region AND lets dispatch PREFETCH the child's
// cacheline before any load that depends on it. There is no leaf node — a leaf
// is a suffix_value with a 0-length suffix (the term).
//
// Liveness/reclaim = per-slot refcount in each region; the COW in-place-vs-clone
// decision is the node header's BIRTH version (birth==V ⟹ private this txn).
// Mutation
// is copy-construct-rebuild from the stable source into a right-sized slot (the
// new slot bubbles up; COW-to-root); pointer children are retained per region,
// inline-none (suffix_value is its own SAL object, refcounted there).
#pragma once
#include <gcart/branch_tag.hpp>  // branch_type, tag3_get/set, tag3_bytes
#include <ucc/lower_bound.hpp>   // ucc::lower_bound_padded (vectorized in-node scan)
#include <ucc/round.hpp>         // ucc::round_up_multiple (cacheline-rounded leaf alloc)
#include <sal2/fixed_allocator.hpp>  // FIX routing (cl1/cl2/cl9) — UNCHANGED, direct, no CB
// ── Phase 2: the LEAF region rides the copied SAL 1.0 allocator ──────────────
// (libraries/sal2/include/sal2/alloc/* — namespace sal2 — carries the real
//  machinery: background compactor, MFU read-cache (get_ref SAMPLES the active
//  bit), segment threads, control_block refcount). Routing stays on the
//  from-scratch fixed_allocator above; only suffix_value / big_suffix_value
//  leaves move here, so the MFU cache + compactor become functional on the leaf.
//  NOTE: movable_region.hpp / big_region.hpp are NO LONGER included — they pull
//  in the from-scratch sal2/control_block_alloc.hpp which redefines sal2::
//  ptr_address (uint32_t) and would collide with the typed ptr_address the SAL
//  1.0 subsystem declares. gcart no longer references those regions at all.
#include <sal2/alloc/allocator.hpp>
#include <sal2/alloc/allocator_session_impl.hpp>
#include <sal2/alloc/read_lock.hpp>
#include <sal2/alloc/smart_ptr.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace gcart
{
   using sv = std::string_view;
   using bt = branch_type;
   // The LEAF region's SAL-1.0 pointer is a typed_int<uint32_t> — the SAME width
   // as the engine's u32 routing-branch slot, so a leaf branch (suffix_value /
   // big_suffix_value) keeps storing a bare u32 (the ptr_address value) and only
   // reconstructs leaf_addr() at the SAL-1.0 call boundary.
   using leaf_addr_t = sal2::ptr_address;
   inline leaf_addr_t leaf_addr(uint32_t p) noexcept { return leaf_addr_t(p); }

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
   // node once committed to <shared>. <unique> reads each routing node's BIRTH
   // version (birth==V ⟹ born this txn ⟹ private ⟹ mutate in place); the FIRST node
   // with birth<V (born in an earlier, possibly snapshot-pinned txn) CLONES (the
   // clone is stamped birth=V, so its subtree is in-txn), and latches the rest of
   // the descent to <shared>, which clones without re-checking. This is the spec's
   // version-liveness latch LITERALLY — the COW decision is birth==V, NOT refcount
   // (refcount is now used ONLY for reclamation; readers no longer bump it for the
   // COW decision, the prerequisite for hazard-pointer readers).
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
   // sv_leaf / big_leaf are SAL-1.0 alloc_header nodes (declared just below the
   // engine); the leaf body cap is what fits a single SAL-1.0 cacheline-rounded
   // allocation comfortably below the 2 KB inline ceiling psitri kept. The SAL 1.0
   // large-object path takes everything bigger automatically (one allocator,
   // small + large), so there is no longer a separate "big region": a leaf whose
   // body exceeds sal_max_body is a big_suffix_value (a bigger SAL 1.0 alloc) but
   // still lives in the SAME allocator (its CB ref is the liveness).
   struct engine
   {
      sal2::fixed_allocator<128>  cl1;  // setlist / nested_setlist  (FIX, direct, no CB)
      sal2::fixed_allocator<256>  cl2;  // double_setlist / big_nested (FIX)
      sal2::fixed_allocator<1152> cl9;  // full / node256             (FIX)

      // The copied SAL 1.0 allocator drives ALL leaf data (suffix_value AND
      // big_suffix_value): the CB region holds small leaves, the large-object
      // path holds big ones — exactly how sal_art_ref drives one sal::allocator
      // for both leaf_node and large values. Opened in the ctor, threads started
      // (compactor + MFU promoter). A per-thread session (thread_local) drives
      // get_ref / retain / release; the writer's session is bound per tree.
      std::unique_ptr<sal2::allocator> leaf;

      // Leaf body cap below which a (suffix,value) is a suffix_value (inline-able
      // into a nested_setlist); larger bodies are big_suffix_value. Both are SAL
      // 1.0 sv_leaf allocations (the tag only changes whether the body can inline).
      static constexpr uint32_t sal_max_body = 2048;

      // Per-region segment cap = max addressable size before the u32 slot index
      // wraps (each region's branch pointer is a 4-byte intra-region index). A
      // segment is PROT_NONE-reserved up front and committed on demand, so a large
      // cap costs virtual address space, not RSS — sized here for big workloads
      // (Silkworm-scale sync). FIX caps sit at the u32 ceiling (16384·32MB=512GB);
      // sal's grain index ceiling is 16384·8MB=128GB. Override per region via env
      // (GCART_CL1_SEGS / _CL2_ / _CL9_ / _SAL_ / _BIG_SEGS) for tuning.
      static uint32_t seg_cap(const char* env, uint32_t def) noexcept
      {
         if (const char* v = std::getenv(env))
            if (const unsigned long n = std::strtoul(v, nullptr, 10))
               return uint32_t(n);
         return def;
      }
      explicit engine(std::filesystem::path dir)
          : cl1({.dir = dir, .name = "cl1", .max_segments = seg_cap("GCART_CL1_SEGS", 4096)}),
            cl2({.dir = dir, .name = "cl2", .max_segments = seg_cap("GCART_CL2_SEGS", 2048)}),
            cl9({.dir = dir, .name = "cl9", .max_segments = seg_cap("GCART_CL9_SEGS", 2048)}),
            _dir(std::move(dir))
      {
         // The leaf allocator (the copied SAL 1.0) opens its own files under a
         // "leaf" subdir of the engine dir: register the leaf node type ops BEFORE
         // start_background_threads (the SAL contract — the compactor/release loops
         // read the non-atomic type-ops array on every object they touch), then turn
         // the background compactor + MFU promoter ON. Defined out-of-line below
         // (init_leaf_) because sv_leaf/big_leaf are declared after `engine`.
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
         // reference it). Member destruction order is reverse-declaration, which
         // already puts these after `leaf`, but be explicit (and ordered vs the bg
         // join above — _bg_ses is the reclaimer thread's session). Assigning a null
         // ptr drops the held session (end_session) BEFORE `leaf` is destroyed.
         _admin_ses = sal2::allocator_session_ptr{nullptr};
         _bg_ses    = sal2::allocator_session_ptr{nullptr};
         // The leaf allocator destructor stops its own background threads.
      }

      // ── leaf sessions: one per session-HOLDER, never a global map ──────────────
      // SAL 1.0 sessions are single-thread-owned (each thread atomically claims a
      // slot) and every get_ref copies a NON-atomic session-ptr refcount, so each
      // thread MUST drive get_ref/retain/release through its OWN session, never
      // another thread's. There is NO global session map and NO mutex: each holder
      // (a read_session's cursors, a work tree's writer) OWNS its session, created
      // via leaf_create_session() and used only on that one thread. The protection
      // for a read is the snapshot pin (the held root's refcount), not a lock.
      sal2::allocator_session_ptr leaf_create_session() { return leaf->create_session(); }
      // Cold single-thread admin path (leaf_live_bytes_, snapshot::get, get_root):
      // an engine-owned session used only on the main/admin thread.
      sal2::allocator_session& admin_ses() noexcept { return *_admin_ses; }

      void init_leaf_();  // out-of-line (needs sv_leaf/big_leaf complete)

      // raw live bytes; drain() first for an accurate (settled) figure.
      uint64_t live_bytes_raw() noexcept
      {
         return cl1.live_bytes() + cl2.live_bytes() + cl9.live_bytes() + leaf_live_bytes_();
      }
      uint64_t live_bytes() noexcept
      {
         drain();
         return live_bytes_raw();
      }
      uint64_t leaf_live_bytes_() noexcept;  // out-of-line (iterates the SAL 1.0 live set)

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
      // Engine-owned sessions (created in init_leaf_ AFTER `leaf` is up, dropped in
      // ~engine BEFORE `leaf`): the background reclaimer's session (used only by the
      // bg thread) and the cold admin/main-thread session. Every other holder owns
      // its own via leaf_create_session().
      sal2::allocator_session_ptr    _bg_ses{nullptr};
      sal2::allocator_session_ptr    _admin_ses{nullptr};
      release_ring<release_ring_cap> _rq;
      std::thread                    _bg;
      std::atomic<bool>              _bg_stop{false};
      std::atomic<uint64_t>          _bg_inflight{0};
   };

   // ── suffix_value (SAL 1.0 leaf): alloc_header + [u32 birth][u16 vlen][u8 slen][value][suffix]
   // slen==0 => the term (a "leaf" is suffix_value with a 0-length suffix).
   //
   // sv_leaf is a SAL-1.0 alloc_header-derived NODE (mirrors sal_art_ref::leaf_node):
   // alloc_size(suffix,value) sizes a cacheline-rounded allocation; the ctor writes
   // the body; the leaf session's alloc<sv_leaf>(suffix,value) hands back a
   // ptr_address; get_ref<sv_leaf>(pa) reads it back (and SAMPLES the MFU active
   // bit — the whole point of moving the leaf onto SAL 1.0). It owns no children
   // (a byte leaf is pure data), so destroy/visit_children are no-ops (the
   // alloc_header defaults). The CB refcount IS the leaf's liveness (Phase 2 v1).
   struct sv_leaf : sal2::alloc_header
   {
      static constexpr sal2::header_type type_id = sal2::header_type::start_user_type;  // 3

      // body (immediately after the 14-B packed alloc_header):
      //   [u32 birth][u16 vlen][u8 slen][value][suffix].
      // birth = the committing-txn version that created this leaf (version liveness;
      // separate from alloc_header's allocation seq). The INLINE-into-nested form
      // (nested::in_value) is birth-FREE — it shares the nested node's header birth —
      // so bytes_for() is the LOGICAL (birth-free) body size used for inline packing.
      static constexpr uint32_t birth_bytes = 4;
      static uint32_t bytes_for(uint32_t vlen, uint32_t slen) noexcept { return 3 + vlen + slen; }

      char*       body() noexcept { return reinterpret_cast<char*>(this) + sizeof(sal2::alloc_header); }
      const char* body() const noexcept
      {
         return reinterpret_cast<const char*>(this) + sizeof(sal2::alloc_header);
      }

      // ── SAL 1.0 construction contract ──────────────────────────────────────
      static uint32_t alloc_size(sv suffix, sv value) noexcept
      {
         const uint32_t raw = uint32_t(sizeof(sal2::alloc_header) + birth_bytes +
                                       bytes_for(uint32_t(value.size()), uint32_t(suffix.size())));
         return ucc::round_up_multiple<sal2::cacheline_size>(raw);
      }
      sv_leaf(uint32_t asize, sal2::ptr_address_seq seq, sv suffix, sv value) noexcept
          : sal2::alloc_header(asize, type_id, seq)
      {
         const uint16_t vl = uint16_t(value.size());
         char*          d  = body();
         *reinterpret_cast<uint32_t*>(d) = 0;  // birth: stamped by make_sv after alloc (GC-2c)
         std::memcpy(d + birth_bytes, &vl, 2);
         d[birth_bytes + 2] = char(uint8_t(suffix.size()));
         std::memcpy(d + birth_bytes + 3, value.data(), value.size());
         std::memcpy(d + birth_bytes + 3 + value.size(), suffix.data(), suffix.size());
      }

      uint32_t birth() const noexcept { return *reinterpret_cast<const uint32_t*>(body()); }
      void     set_birth(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(body()) = v; }
      sv       value() const noexcept
      {
         const char* p = body() + birth_bytes;
         uint16_t    v16;
         std::memcpy(&v16, p, 2);
         return {p + 3, v16};
      }
      sv suffix() const noexcept
      {
         const char* p = body() + birth_bytes;
         uint16_t    v16;
         std::memcpy(&v16, p, 2);
         return {p + 3 + v16, uint8_t(p[2])};
      }
   };

   // ── big_leaf (SAL 1.0 large-object leaf): alloc_header + [u32 birth][u32 vlen][u8 slen][value][suffix]
   // The same (suffix,value) leaf as sv_leaf but with a u32 vlen (big values can
   // exceed 64K) and never inlined into a nested node. Reached as a
   // bt::big_suffix_value branch. It rides the SAME SAL 1.0 leaf allocator as
   // sv_leaf — SAL 1.0's large-object path handles the bigger size (one allocator,
   // small + large, exactly how sal_art_ref stores both leaf_node and large values).
   struct big_leaf : sal2::alloc_header
   {
      static constexpr sal2::header_type type_id =
          sal2::header_type(uint8_t(sal2::header_type::start_user_type) + 1);  // 4

      static constexpr uint32_t birth_bytes = 4;
      // birth-free body size: [u32 vlen][u8 slen][value][suffix] = 5 + vlen + slen.
      static uint32_t bytes_for(uint32_t vlen, uint32_t slen) noexcept { return 5 + vlen + slen; }

      char*       body() noexcept { return reinterpret_cast<char*>(this) + sizeof(sal2::alloc_header); }
      const char* body() const noexcept
      {
         return reinterpret_cast<const char*>(this) + sizeof(sal2::alloc_header);
      }

      static uint32_t alloc_size(sv suffix, sv value) noexcept
      {
         const uint32_t raw = uint32_t(sizeof(sal2::alloc_header) + birth_bytes +
                                       bytes_for(uint32_t(value.size()), uint32_t(suffix.size())));
         return ucc::round_up_multiple<sal2::cacheline_size>(raw);
      }
      big_leaf(uint32_t asize, sal2::ptr_address_seq seq, sv suffix, sv value) noexcept
          : sal2::alloc_header(asize, type_id, seq)
      {
         const uint32_t vl = uint32_t(value.size());
         char*          d  = body();
         *reinterpret_cast<uint32_t*>(d) = 0;  // birth stamped by make_sv after alloc (GC-2c)
         std::memcpy(d + birth_bytes, &vl, 4);
         d[birth_bytes + 4] = char(uint8_t(suffix.size()));
         std::memcpy(d + birth_bytes + 5, value.data(), vl);
         std::memcpy(d + birth_bytes + 5 + vl, suffix.data(), suffix.size());
      }

      uint32_t birth() const noexcept { return *reinterpret_cast<const uint32_t*>(body()); }
      void     set_birth(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(body()) = v; }
      sv       value() const noexcept
      {
         const char* p = body() + birth_bytes;
         uint32_t    vl;
         std::memcpy(&vl, p, 4);
         return {p + 5, vl};
      }
      sv suffix() const noexcept
      {
         const char* p = body() + birth_bytes;
         uint32_t    vl;
         std::memcpy(&vl, p, 4);
         return {p + 5 + vl, uint8_t(p[4])};
      }
   };

   // ── engine::init_leaf_ / leaf_live_bytes_ (out-of-line: sv_leaf/big_leaf complete) ──
   inline void engine::init_leaf_()
   {
      std::filesystem::create_directories(_dir / "leaf");
      sal2::runtime_config cfg;  // MFU read-cache defaults ON (the lever this phase earns)
      leaf = std::make_unique<sal2::allocator>(_dir / "leaf", cfg, /*start_threads_now=*/false);
      leaf->register_type_ops<sv_leaf>();   // ops BEFORE start_background_threads (SAL contract)
      leaf->register_type_ops<big_leaf>();
      leaf->start_background_threads();      // compactor + MFU promoter + release thread ON
      // Engine-owned sessions, created AFTER the leaf allocator is up + its threads
      // started: the bg reclaimer's session (used only by the bg thread / spill
      // cascade) and the cold admin/main-thread session.
      _bg_ses    = leaf->create_session();
      _admin_ses = leaf->create_session();
   }
   inline uint64_t engine::leaf_live_bytes_() noexcept
   {
      uint64_t b = 0;
      _admin_ses->for_each_live_object(
          [&](auto, auto, const sal2::alloc_header* h) { b += h->size(); });
      return b;
   }

   // ── leaf dispatch: a terminal is suffix_value OR big_suffix_value — both SAL
   //    1.0 alloc_header leaves; read its (suffix,value) without the caller caring
   //    which. The leaf is resolved via the LEAF SESSION's get_ref<T> (which SAMPLES
   //    the MFU active bit — the whole point of Phase 2). The returned sv VIEWS leaf
   //    memory: valid while a leaf read_lock covering the read is held AND the leaf's
   //    refcount keeps it alive (the snapshot/work root pins it). ──────────────────
   // Each caller passes its OWN leaf session (a cursor's, a work tree's): no global
   // map, no lock — the read's protection is the snapshot pin, not a session probe.
   inline sv leaf_suffix(sal2::allocator_session& ses, bt t, uint32_t p) noexcept
   {
      return t == bt::big_suffix_value
                 ? ses.get_ref<big_leaf>(leaf_addr(p)).obj()->suffix()
                 : ses.get_ref<sv_leaf>(leaf_addr(p)).obj()->suffix();
   }
   inline sv leaf_value(sal2::allocator_session& ses, bt t, uint32_t p) noexcept
   {
      return t == bt::big_suffix_value
                 ? ses.get_ref<big_leaf>(leaf_addr(p)).obj()->value()
                 : ses.get_ref<sv_leaf>(leaf_addr(p)).obj()->value();
   }
   inline bool is_leaf_tag(bt t) noexcept
   {
      return t == bt::suffix_value || t == bt::big_suffix_value;
   }
   // The birth version stamped on a leaf — read via the caller's owned session
   // (used by make_sv / pack_nested_leaves to decide private-this-txn inlining).
   inline uint32_t sv_birth(sal2::allocator_session& ses, uint32_t p) noexcept
   {
      return ses.get_ref<sv_leaf>(leaf_addr(p)).obj()->birth();
   }
   inline uint32_t big_birth(sal2::allocator_session& ses, uint32_t p) noexcept
   {
      return ses.get_ref<big_leaf>(leaf_addr(p)).obj()->birth();
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
      // [3] pad; [4..7] birth version (GC: the committing txn version, design §4.2);
      // [8..11] death version (Phase B: the displacing-commit version; 0 = alive).

      static constexpr uint32_t slot_bytes  = 128;
      static constexpr uint32_t birth_off   = 4;
      static constexpr uint32_t death_off   = 8;   // Phase B: u32 death (0 = alive)
      static constexpr uint32_t hdr         = 12;  // was 8; +u32 death (version liveness reaper)
      uint32_t birth() const noexcept  // offset 4 is 4-aligned in the cacheline slot
      {
         return *reinterpret_cast<const uint32_t*>(base() + birth_off);
      }
      void set_birth(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + birth_off) = v; }
      // Phase B death stamp: the version of the commit that DISPLACED this node (a
      // reader pinned at version < death still sees it). 0 = alive. Set at COW
      // displacement; the floor-gated reaper frees the node only once floor >= death.
      uint32_t death() const noexcept { return *reinterpret_cast<const uint32_t*>(base() + death_off); }
      void set_death(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + death_off) = v; }
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
      static bool fits(uint32_t nbranch, uint32_t prefixlen, bool term, uint32_t slot) noexcept
      {
         return bytes_for_cap(nbranch + (term ? 1u : 0u), prefixlen) <= slot;
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
   //   [u16 vlen][u8 slen][value][suffix] (sv_leaf body). offsets are from-END so a
   //   verbatim memcpy keeps them valid.
   struct nested_setlist
   {
      uint8_t meta;     // outer term:1 (bit7), prefixlen:7 (bits0-6)
      uint8_t nbranch;  // outer byte-branch count

      static constexpr uint32_t slot_bytes  = 128;
      static constexpr uint32_t birth_off   = 4;
      static constexpr uint32_t death_off   = 8;   // Phase B: u32 death (0 = alive)
      static constexpr uint32_t hdr         = 12;  // was 8; +u32 death (version liveness reaper)
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
      uint32_t birth() const noexcept { return *reinterpret_cast<const uint32_t*>(base() + birth_off); }
      void set_birth(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + birth_off) = v; }
      uint32_t death() const noexcept { return *reinterpret_cast<const uint32_t*>(base() + death_off); }
      void set_death(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + death_off) = v; }
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
      sv in_value(uint32_t i) const noexcept
      {
         const char* q = entry(i);
         uint16_t    v;
         std::memcpy(&v, q, 2);
         return {q + 3, v};
      }
      sv in_suffix(uint32_t i) const noexcept
      {
         const char* q = entry(i);
         uint16_t    v;
         std::memcpy(&v, q, 2);
         const uint8_t s = uint8_t(q[2]);
         return {q + 3 + v, s};
      }

      int find_index(uint8_t byte) const noexcept
      {
         int pos = int(ucc::lower_bound_padded(okeys(), nbranch, byte));
         return (pos < int(nbranch) && okeys()[pos] == byte) ? pos : -1;
      }
   };
   static_assert(sizeof(nested_setlist) == 2);

   // ── nentry: the decoded, PATH-AGNOSTIC entry the canonical packer (lay_nested_into_)
   //    and the direct-mutable nested editor (nested_edit/nested_remove) both speak.
   //    A nested_setlist's bytes are a PURE FUNCTION of (prefix, sorted nentry set, term);
   //    funnelling initial-pack, unique in-place edit, and shared clone+edit ALL through
   //    lay_nested_into_(nentry*) is what makes the body bytes byte-identical across the
   //    three paths (killing promote-on-write). gcart's INLINE-SV body is the 3-byte
   //    form [u16 vlen][u8 slen][value][suffix] (sv_leaf::bytes_for = 3+v+s; NO birth
   //    prefix — it shares the nested node header's birth), NOT rcart's 4-byte form.
   //    kind 0 = EXTERNAL  → (otype, ext) is the region-typed 4-byte child ptr.
   //    kind 1 = inline suffix_value → (otype=suffix_value); body/blen point at the
   //             gcart sv_leaf inline body bytes ([u16 vlen][u8 slen][value][suffix]).
   //    kind 2 = inline setlist → (otype=setlist); body/blen point at a TIGHT setlist
   //             body to memcpy (its u32 grandchildren ride along unchanged).
   //    For kinds 1/2 `body` points at a stack-scratch copy (re-pack of an existing
   //    nested) OR a re-resolved/tight-staged body (initial pack) — lay_nested_into_
   //    ONLY ever memcpy's raw bytes, never refcounts (Phase C: no refcount at all).
   struct nentry
   {
      uint8_t     okey;   // outer byte (sorted; unused for the term entry)
      uint8_t     kind;   // 0=ext u32 ptr, 1=inline-sv body, 2=inline-setlist body
      bt          otype;  // resolved branch_type tag of this entry
      uint32_t    ext;    // kind 0: the external 4-byte region ptr
      const char* body;   // kind 1/2: bytes to memcpy into the tail-heap
      uint16_t    blen;   // kind 1/2: byte count of body
   };

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
      static constexpr uint32_t birth_off    = 8;                        // u32 birth (version liveness)
      static constexpr uint32_t death_off    = 12;                       // Phase B: u32 death (0 = alive)
      static constexpr uint32_t types_off    = 16;                       // was 12; +4 death
      static constexpr uint32_t branches_off = 112;                      // 16 + 96
      static constexpr uint32_t prefix_off   = 1136;                     // 112 + 1024
      static constexpr uint32_t max_prefix   = slot_bytes - prefix_off;  // 16

      bool           has_term() const noexcept { return meta & 0x80; }
      uint32_t       prefixlen() const noexcept { return meta & 0x7f; }
      char*          base() noexcept { return reinterpret_cast<char*>(this); }
      const char*    base() const noexcept { return reinterpret_cast<const char*>(this); }
      uint32_t birth() const noexcept { return *reinterpret_cast<const uint32_t*>(base() + birth_off); }
      void set_birth(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + birth_off) = v; }
      uint32_t death() const noexcept { return *reinterpret_cast<const uint32_t*>(base() + death_off); }
      void set_death(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + death_off) = v; }
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
   // Which region a tag lives in. ROUTING tags → FIX (cl1/cl2/cl9), DIRECT (no CB,
   // address-stable). LEAF tags → the SAL 1.0 leaf allocator via get_ref (MFU-sampled);
   // resolve returns the leaf's BODY pointer (past the alloc_header) so the existing
   // body readers (leaf_value_body / find_in's standalone-leaf path) work verbatim —
   // a standalone body still begins with the [birth] prefix. The returned pointer is
   // valid while a leaf read_lock covering the use is held (the compactor may relocate
   // a leaf otherwise); the kv read entry points scope a short read_lock per call.
   // Routing tags resolve via the direct FIX allocators (e.cl*); LEAF tags via the
   // caller-supplied leaf SESSION (get_ref, MFU-sampled). The caller passes its OWN
   // session (cursor's, work tree's) — there is no global session map and no lock.
   inline void* resolve(engine& e, sal2::allocator_session& ses, bt t, uint32_t p) noexcept
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
            return const_cast<char*>(ses.get_ref<big_leaf>(leaf_addr(p)).obj()->body());
         default:
            return nullptr;  // null
      }
   }
   // ROUTING-ONLY resolve (no session): cl1/cl2/cl9 are direct, address-stable FIX
   // allocators that need no leaf session. The write path only calls this form on
   // routing nodes; a leaf tag needs a session (use the session overload above).
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
         default:
            assert(false && "leaf resolve needs a session (use resolve(e, ses, t, p))");
            return nullptr;
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
   // NOTE: there is intentionally NO refcount-based `unique(engine&,bt,uint32_t)`
   // here anymore. The COW in-place-vs-clone decision is BIRTH==V (node_birth below),
   // not refcount — refcount survives only for RECLAMATION (release_one / release_
   // subtree). Removing the refcount latch is the prerequisite for hazard-pointer
   // readers (Part B): once readers stop bumping refcounts, a refcount of 1 no longer
   // means "no snapshot pins this", so it MUST NOT gate in-place mutation.
   //
   // version-liveness latch (design §5.2): a routing node's BIRTH version, read from
   // its header. birth==V (the current txn) ⟹ private (mutate in place); birth<V ⟹
   // born in an earlier committed txn ⟹ potentially snapshot-shared (clone). This is
   // the spec's latch — independent of any refcount. setlist & nested_setlist share
   // birth_off=4 (cl1/cl2); full has it at +8.
   inline uint32_t node_birth(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full:
            return static_cast<const full*>(e.cl9.get(p))->birth();
         case bt::double_setlist:
         case bt::big_nested:
            return static_cast<const setlist*>(e.cl2.get(p))->birth();
         case bt::setlist:
         case bt::nested_setlist:
            return static_cast<const setlist*>(e.cl1.get(p))->birth();
         default:
            return 0;
      }
   }
   // stamp a routing node's header birth (e.g. after a verbatim clone copies the
   // source's birth, or after promoting an inline child to a standalone node).
   inline void set_node_birth(engine& e, bt t, uint32_t p, uint32_t v) noexcept
   {
      switch (t)
      {
         case bt::full:
            static_cast<full*>(e.cl9.get(p))->set_birth(v);
            return;
         case bt::double_setlist:
         case bt::big_nested:
            static_cast<setlist*>(e.cl2.get(p))->set_birth(v);
            return;
         case bt::setlist:
         case bt::nested_setlist:
            static_cast<setlist*>(e.cl1.get(p))->set_birth(v);
            return;
         default:
            return;
      }
   }
   // Phase B: stamp a routing node's DEATH version (the displacing-commit version).
   // Called at COW displacement (the writer supersedes a birth<V node). Same region
   // dispatch as set_node_birth — setlist & nested_setlist share death_off (cl1/cl2),
   // full has it past birth. 0 = alive; a fresh clone resets it (set_node_death(...,0)).
   inline void set_node_death(engine& e, bt t, uint32_t p, uint32_t v) noexcept
   {
      switch (t)
      {
         case bt::full:
            static_cast<full*>(e.cl9.get(p))->set_death(v);
            return;
         case bt::double_setlist:
         case bt::big_nested:
            static_cast<setlist*>(e.cl2.get(p))->set_death(v);
            return;
         case bt::setlist:
         case bt::nested_setlist:
            static_cast<setlist*>(e.cl1.get(p))->set_death(v);
            return;
         default:
            return;
      }
   }
   inline uint32_t node_death(engine& e, bt t, uint32_t p) noexcept
   {
      switch (t)
      {
         case bt::full:
            return static_cast<const full*>(e.cl9.get(p))->death();
         case bt::double_setlist:
         case bt::big_nested:
            return static_cast<const setlist*>(e.cl2.get(p))->death();
         case bt::setlist:
         case bt::nested_setlist:
            return static_cast<const setlist*>(e.cl1.get(p))->death();
         default:
            return 0;
      }
   }
   // Phase C: there is NO retain()/try_retain()/release_one() free function anymore.
   // Per-node refcount is GONE from the entire gcart engine — the COW in-place-vs-clone
   // latch is birth==V, sharing a node down a clone costs ZERO (no retain), and
   // reclamation is the database's floor-gated drain (free_node_shallow_ / free_subtree_
   // deep_). The OWNED snapshot is a version pin (owned_pin), not a refcount. The FIX
   // allocators' retain/release_one still EXIST (rcart, the refcount fork, uses them) but
   // gcart never calls them; the leaf CB ref stays 1 for the object's whole life (set at
   // alloc, dropped by the single leaf_free at death) — kept only for relocation/MFU.
   // Phase C — NO per-node refcount. Liveness is the version stamp (birth + death)
   // gated by the floor; reclamation is the floor-gated drain (start_ver). Sharing a
   // node down a COW clone costs ZERO (no retain, no release, no per-child refcount
   // traffic — start_ver tree_alloc.hpp:19). A node frees EXACTLY ONCE: a routing
   // husk via free_node_ (slot reclaim), a leaf via leaf_free (a single SAL release
   // from its lifetime ref of 1). retain()/try_retain()/release_one() above are kept
   // ONLY for the OWNED-snapshot path (snap::acquire / write_session::get_root) — the
   // cold, non-reader pin — never the hot read/write/commit path.
   //
   // LEAF reclaim (Phase C): a leaf carries CB ref==1 for its whole life (never
   // retained on a COW share — start_ver §19); leaf_free is the SINGLE decrement that
   // takes it 1→0, destroying+freeing the SAL object (a byte leaf owns no children,
   // so destroy is a no-op). The CB survives ONLY for relocation/MFU (the compactor
   // moves leaves; get_ref MFU-samples) — it is NOT the liveness signal. The drain
   // calls leaf_free when the leaf's death <= floor (born-this-txn leaves free inline).
   inline void leaf_free(sal2::allocator_session& ses, uint32_t p) noexcept
   {
      ses.release(leaf_addr(p));  // ref 1→0 → destroy+free (the single, only release)
   }
   // a leaf's birth version (the committing txn that created it). 0 = the inline /
   // pre-version form. Read for the COW privacy decision (born this txn ⟹ free inline).
   inline uint32_t leaf_birth(sal2::allocator_session& ses, bt t, uint32_t p) noexcept
   {
      return t == bt::big_suffix_value ? ses.get_ref<big_leaf>(leaf_addr(p)).obj()->birth()
                                       : ses.get_ref<sv_leaf>(leaf_addr(p)).obj()->birth();
   }
}  // namespace gcart

namespace gcart
{
   // a decoded branch (scratch, for copy-construct-rebuild of a routing node)
   struct br
   {
      uint8_t  byte;
      bt       tag;
      uint32_t ptr;
   };

   inline bool pfx_ok_(sv pfx, sv key, uint32_t depth) noexcept
   {
      return key.size() - depth >= pfx.size() &&
             (pfx.empty() || std::memcmp(key.data() + depth, pfx.data(), pfx.size()) == 0);
   }

   // ── body-based leaf readers (used by find_in's leaf hit + the ordered cursor) ──
   // Read (suffix,value) from a RESOLVED leaf BODY pointer — resolve()/get_ref
   // returns body() (past the alloc_header), which begins with the [u32 birth]
   // prefix these skip. A single get_ref per leaf feeds both readers (one MFU
   // sample per logical read). Inline-in-nested suffix_value bodies are birth-FREE
   // and read via nested::in_value/in_suffix instead.
   inline sv leaf_suffix_body(bt t, const void* b) noexcept
   {
      const char* p = static_cast<const char*>(b) + sv_leaf::birth_bytes;  // skip birth (==big's)
      if (t == bt::big_suffix_value)
      {
         uint32_t v;
         std::memcpy(&v, p, 4);
         return {p + 5 + v, uint8_t(p[4])};
      }
      uint16_t v;
      std::memcpy(&v, p, 2);
      return {p + 3 + v, uint8_t(p[2])};
   }
   inline sv leaf_value_body(bt t, const void* b) noexcept
   {
      const char* p = static_cast<const char*>(b) + sv_leaf::birth_bytes;  // skip birth (==big's)
      if (t == bt::big_suffix_value)
      {
         uint32_t v;
         std::memcpy(&v, p, 4);
         return {p + 5, v};
      }
      uint16_t v;
      std::memcpy(&v, p, 2);
      return {p + 3, v};
   }

   // Read-path find from a (tag,ptr) root — usable by readers/cursors with NO
   // write sessions. Dispatch on the branch tag, PREFETCH the child before the
   // dependent load (full prefetches the offset cacheline before the type hdr).
   // out is a VIEW into the value bytes (no copy). Valid while the snapshot/work
   // root is alive (the caller pins it); the caller constructs/copies if it needs
   // ownership. Mirrors start_ref's view-based pg_find — the read hot path never
   // allocates or copies the value (the bank reader discards it entirely).
   inline bool find_in(engine& e, sal2::allocator_session& ses, bt t, uint32_t p, sv key, sv& out)
   {
      uint32_t depth = 0;
      for (;;)
      {
         if (t == bt::null)
            return false;
         void* node = resolve(e, ses, t, p);  // leaf: get_ref (MFU-sampled) → body pointer
         if (is_leaf_tag(t))  // suffix_value / big_suffix_value (both SAL 1.0 leaves)
         {
            // Read suffix + value from the ONE resolved body (the get_ref already
            // sampled MFU); leaf_*_body skip the [birth] prefix. No second get_ref.
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
            // (key[depth] — correct when this full has no prefix, the common case)
            // BEFORE touching the header/type cacheline, so the header's COLD MISS
            // (pfx_ok_ / btype below) overlaps this branch-ptr prefetch (Dan §7).
            // The header is NOT prefetched — it IS the overlap window. A prefix
            // makes this mispredict (a wasted prefetch, never wrong). Pays only on
            // COLD runs at scale; invisible in L2/L3-resident benches.
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
            p = n->branch(byte);  // prefetched (no-prefix) → hit; mispredicted → miss
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
                  uint16_t    v;
                  std::memcpy(&v, q, 2);
                  const uint8_t s = uint8_t(q[2]);
                  if (sv{q + 3 + v, s} != key.substr(depth + 1))
                     return false;
                  out = sv{q + 3, v};  // view into the line
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

   // ── version-liveness reclamation (Phase C — NO refcount) ────────────────────
   // start_ver's two free shapes (tree_alloc.hpp:13-21):
   //   SHALLOW (free_node_shallow_): the CLONE supersession — a routing node N was
   //     cloned (COW); N's children were CARRIED by the successor clone, so freeing N
   //     frees ONLY N's own slot line (its inline data dies with the line; its
   //     external children + an inline-setlist's u32 grandchildren are SHARED with the
   //     clone and MUST NOT be touched). This is the entire normal write-path free:
   //     every COW-displaced interior node along the spine is shallow.
   //   DEEP (free_subtree_deep_): a genuinely-DROPPED subtree (no successor carries
   //     it) — cascade-free every node + term + leaf reachable, UNCONDITIONALLY (no
   //     refcount: the floor gate already proved no reader can reach it). A leaf is
   //     childless so its deep == its single leaf_free.
   // BOTH are reached only through the floor-gated drain (a node is freed only once
   // its death <= floor): a reader pinned at version < death can still reach it.
   inline void free_node_shallow_(engine& e, sal2::allocator_session& ses, bt t, uint32_t p) noexcept
   {
      if (is_leaf_tag(t))  // a leaf carries no shared children → shallow == its free
      {
         leaf_free(ses, p);
         return;
      }
      if (is_routing(t))
         free_node_(e, t, p);  // free ONLY this line; children carried by the clone
   }
   inline void free_subtree_deep_(engine& e, sal2::allocator_session& ses, bt t, uint32_t p) noexcept
   {
      if (t == bt::null)
         return;
      if (is_leaf_tag(t))
      {
         leaf_free(ses, p);  // childless
         return;
      }
      if (!is_routing(t))
         return;
      br       kids[257];
      uint32_t nb;
      bt       tt;
      uint32_t tp;
      bool     term;
      gather_children_(e, t, p, kids, nb, tt, tp, term);  // read children BEFORE freeing
      for (uint32_t i = 0; i < nb; ++i)
         free_subtree_deep_(e, ses, kids[i].tag, kids[i].ptr);
      if (term)
         free_subtree_deep_(e, ses, tt, tp);
      free_node_(e, t, p);
   }
   // The engine bg release ring (spill_fn / release_async / bg_loop_) is no longer on
   // any gcart path (Phase C: reclaim is the database's floor-gated drain, never an
   // engine-side refcount cascade). The bg thread is never started; these definitions
   // remain only to satisfy the engine's member declarations. release_async deep-frees
   // unconditionally (no refcount) should any stray caller reach it.
   inline void engine::spill_fn::operator()(bt t, uint32_t p) const noexcept
   {
      free_subtree_deep_(*e, *e->_bg_ses, t, p);
   }
   inline void engine::release_async(bt t, uint32_t p) noexcept
   {
      if (t == bt::null)
         return;
      free_subtree_deep_(*this, *_bg_ses, t, p);
   }
   inline void engine::bg_loop_() noexcept {}  // unused (no bg ring in Phase C)

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
         // The work tree OWNS its leaf session (_lses): the writer runs on one
         // thread and drives every leaf get_ref/retain/release/alloc through it.
         // No engine session map, no lock.
      }

      uint64_t size() const noexcept { return _size; }

      // ── read path: dispatch on the branch tag, prefetch the child BEFORE the
      //    dependent load (Dan's hard rule; full node prefetches the offset
      //    cacheline before reading the 3-bit type header). ────────────────────
      bool find(sv key, std::string& out) const
      {
         auto lk = _lses->lock();  // pin leaf pages for the descent + value copy
         sv   v;
         if (!find_in(const_cast<engine&>(_e), *_lses, _rt, _rp, key, v))
            return false;
         out.assign(v.data(), v.size());  // string-out API: one copy
         return true;
      }
      // view-returning find for the hot read path (no copy; caller pins the root).
      bool find(sv key, sv& out) const
      {
         return find_in(const_cast<engine&>(_e), *_lses, _rt, _rp, key, out);
      }

      bt       root_tag() const noexcept { return _rt; }
      uint32_t root_ptr() const noexcept { return _rp; }
      void     set_root(bt t, uint32_t p) noexcept
      {
         _rt = t;
         _rp = p;
      }
      // version-liveness latch (design §5.2): the birth stamped on nodes created by
      // this tree. spec_kv sets it to a fresh per-txn version at start_transaction so
      // committed nodes (birth<V) clone and in-txn nodes (birth==V) mutate in place.
      uint32_t write_version() const noexcept { return _V; }
      void     set_write_version(uint32_t v) noexcept { _V = v; }
      // Mark the FIX-allocator transaction boundary (start_ref::session model): the
      // routing-node sessions (cl1/cl2/cl9) latch _floor = _cursor = the current
      // frontier so this txn's at/after-floor frees recycle hot and the freeset scan
      // first-fits forward from here. Called at write-transaction start. The leaf
      // allocator (copied SAL 1.0) manages its own txn/segment lifecycle (active
      // segment + background compactor), so it needs no per-txn begin marker here.
      void begin_txn() noexcept
      {
         _s1.begin_txn();
         _s2.begin_txn();
         _s9.begin_txn();
         _clone_log.clear();  // fresh per-txn supersession log (Phase C drain)
      }
      engine& eng() noexcept { return _e; }
      // The work tree's owned leaf session (the writer thread's). transaction::abort
      // and start_transaction borrow it for the snap that pins/releases the root.
      sal2::allocator_session* leaf_session() const noexcept { return _lses.get(); }
      // Public deep-drop a (tag,ptr) subtree (abort: the diverged work tree). Phase C:
      // version-liveness, NO refcount — born-this-txn nodes free inline, committed
      // nodes log for the floor-gated drain (commit then publishes the log).
      void release(bt t, uint32_t p) { log_deep(t, p); }
      // Phase C ABORT walk (start_ver vabort_release): free the diverged work spine.
      // Recurse ONLY into nodes born THIS txn (birth==_V); a committed node (birth<_V)
      // is SHARED with the restored root — stop there, never free it. Frees the
      // born-this-txn clones + fresh leaves inline (no refcount). Born-this-txn leaves
      // are freed; committed leaves (shared with the restored root) are skipped.
      void vabort_release(bt t, uint32_t p)
      {
         if (t == bt::null)
            return;
         if (is_leaf_tag(t))
         {
            if (leaf_birth(*_lses, t, p) == _V)  // born this txn ⟹ diverged garbage
               leaf_free(*_lses, p);
            return;  // committed leaf: shared with the restored root, leave it
         }
         if (!is_routing(t))
            return;
         if (node_birth(_e, t, p) != _V)
            return;  // committed node: the restored root still owns it (and its subtree)
         read_branches(t, p, [&](const br& b) { vabort_release(b.tag, b.ptr); });
         bt       tt;
         uint32_t tp;
         if (node_term(t, p, tt, tp))
            vabort_release(tt, tp);
         free_node(t, p);  // the diverged node itself (born this txn)
      }

      // ── Phase C: per-txn SUPERSESSION LOG (start_ver clone-log) ───────────────
      // Every node DISPLACED this txn that was born in an EARLIER (possibly
      // snapshot-pinned) committed txn is logged here, NOT freed: a reader pinned at
      // version < this txn's commit version still reaches it through the old root, so
      // it can be freed only once the floor passes that version. The commit hands the
      // log to the database's floor-gated drain. A node born THIS txn (birth==V) is
      // never reader-visible, so it is freed INLINE (free_node_/leaf_free) — never
      // logged (start_ver: window-born husks free now; only pre-window nodes log).
      struct dead_entry
      {
         bt       tag;
         uint32_t ptr;
         bool     deep;  // true = DROPPED subtree (cascade); false = CLONE husk (node-only)
      };
      std::vector<dead_entry>& clone_log() noexcept { return _clone_log; }
      void                     clear_clone_log() noexcept { _clone_log.clear(); }

     private:
      // SHALLOW supersession (the clone path): a routing node cloned by COW, or a
      // genuinely-dropped LEAF (childless ⟹ shallow == its single free). Born this
      // txn ⟹ free inline (private, never reader-visible); else log for the drain.
      void log_shallow(bt t, uint32_t p) noexcept
      {
         if (t == bt::null)
            return;
         const bool born_this_txn = is_leaf_tag(t) ? (leaf_birth(*_lses, t, p) == _V)
                                                   : (node_birth(_e, t, p) == _V);
         if (born_this_txn)
            free_node_shallow_(_e, *_lses, t, p);  // private husk: free now
         else
            _clone_log.push_back({t, p, /*deep=*/false});  // committed: floor-gated
      }
      // DEEP supersession (a dropped subtree with live children). Born this txn ⟹
      // cascade-free inline; else log deep for the floor-gated drain.
      void log_deep(bt t, uint32_t p) noexcept
      {
         if (t == bt::null)
            return;
         const bool born_this_txn = is_leaf_tag(t) ? (leaf_birth(*_lses, t, p) == _V)
                                                   : (node_birth(_e, t, p) == _V);
         if (born_this_txn)
            free_subtree_deep_(_e, *_lses, t, p);  // private subtree: free now
         else
            _clone_log.push_back({t, p, /*deep=*/true});  // committed: floor-gated
      }

     public:

      // ── write path ─────────────────────────────────────────────────────────
      // Public entry descends in <unique> and latches to <shared> on the first
      // snapshot-pinned node (doc §5.2). One refcount read at the root decides
      // in-place vs clone; the cost is cached down the descent.
      void upsert(sv key, sv val)
      {
         // The COW descent reads leaf bytes (suffix/value on a split, inline-promote)
         // via get_ref against the leaf allocator; hold a leaf residency read_lock so a
         // background compaction cannot recycle a leaf segment under the writer's
         // resolved pointers for the duration of the mutation.
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
         auto lk = _lses->lock();  // pin leaf pages for the descent + reclaim
         sv   _scanv;  // presence pre-check only — discards the value view
         if (!find_in(_e, *_lses, _rt, _rp, key, _scanv))
            return 0;  // absent — no-op, no clone
         auto [t, p] = remove_rec<cow_mode::unique>(_rt, _rp, key, 0);
         _rt         = t;
         _rp         = p;
         --_size;
         return 1;
      }

     private:
      // Build a terminal leaf via the SAL 1.0 leaf allocator: a small body is a
      // suffix_value (inline-able into a nested node), a large body is a
      // big_suffix_value (a bigger SAL 1.0 alloc, never inlined). BOTH are SAL 1.0
      // alloc<…>(suffix,value) → ptr_address (CB ref = 1, MFU-trackable), mirroring
      // sal_art_ref's make_leaf. Returns the tag so callers stay region-agnostic.
      // The leaf's birth is stamped here (version liveness; born this txn).
      std::pair<bt, uint32_t> make_sv(sv suffix, sv val)
      {
         auto& ses = *_lses;
         if (sv_leaf::bytes_for(uint32_t(val.size()), uint32_t(suffix.size())) <=
             engine::sal_max_body)
         {
            const leaf_addr_t a = ses.alloc<sv_leaf>(suffix, val);
            ses.get_ref<sv_leaf>(a).modify().get()->set_birth(_V);
            return {bt::suffix_value, uint32_t(*a)};
         }
         const leaf_addr_t a = ses.alloc<big_leaf>(suffix, val);
         ses.get_ref<big_leaf>(a).modify().get()->set_birth(_V);
         return {bt::big_suffix_value, uint32_t(*a)};
      }
      // Phase C: WRITER deep-drop (a whole subtree leaves the tree — abort's diverged
      // work, or a dropped subtree on remove). No refcount: born-this-txn nodes free
      // inline; committed nodes log deep for the floor-gated drain. Used by abort
      // (where the diverged work is entirely born-this-txn ⟹ all free inline).
      void drop_subtree(bt t, uint32_t p) { log_deep(t, p); }

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
         // nested_setlist is now DIRECTLY MUTABLE (nested_edit/nested_remove handle it
         // before any flat node_child call), and on the prefix-split/de-nest fallback
         // promote_nested converts to a FLAT node first — so node_child never sees a
         // nested node. The old "only after promote_nested" assert no longer holds (the
         // contract changed: nested is editable in place), so it is removed.
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
         // page already dirty) recycles HOT via the per-session fcache; cold frees
         // fall through to the freeset authority inside session::free (start_ref §7).
         if (t == bt::full)
            _s9.free(p);
         else if (t == bt::double_setlist || t == bt::big_nested)
            _s2.free(p);
         else
            _s1.free(p);
      }
      // Phase C: release a husk whose children were already TRANSFERRED into a freshly
      // built node (rebuild / rebuild_remove / promote, private path). This is a
      // SHALLOW supersession — free ONLY the husk line; its children now live in the
      // new node (NOT touched). In every caller the husk is the working node W, which
      // is private (born this txn — the unique latch, or a fresh reprefix clone), so
      // log_shallow frees it inline; the committed-node fallback (log for the drain)
      // is the safety belt. NO refcount.
      void release_husk(bt t, uint32_t p) { log_shallow(t, p); }

      // Build a routing node from a branch list (sorted by byte) + optional term
      // + prefix; pick cl1/cl2/full by size. Branch ptrs are TAKEN (not retained).
      // may_inline: try a nested_setlist first (inline private small leaf children
      // into the line). Pass true ONLY where the caller does NOT release the child
      // set afterward (rebuild-unique free_node / fresh splits) — inlined leaves are
      // consumed (release_ref'd), which would double-free under a later release_ref(t).
      static bool nested_enabled() noexcept
      {
         static const bool on = (std::getenv("GCART_NO_NESTED") == nullptr);
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
         // A full node's prefix slot is only full::max_prefix bytes, but a long
         // shared prefix can precede a high-fanout point (Silkworm storage: dbi +
         // 32B account hash, then a 256-way fan-out). Split: a wrapper holds the
         // HEAD and routes ONE byte to a full that holds the TAIL (≤max_prefix) +
         // the fan-out. Full nodes USUALLY have no prefix (a dense fan-out's
         // children diverge), so the descent prefetches the child full's offset
         // cacheline on key[depth] speculatively — correct in the common no-prefix
         // case, a harmless mispredict (wasted prefetch, never wrong) when prefixed.
         if (pl > full::max_prefix)
         {
            const uint32_t h   = pl - full::max_prefix - 1;  // head len; 1 byte links to the full
            const uint8_t  sep = uint8_t(prefix[h]);
            const uint32_t fs =
                pack_full(prefix.substr(h + 1), brs, nb, term, tt, tp);  // tail ≤ max_prefix
            br link{sep, bt::full, fs};
            return build_node(prefix.substr(0, h), &link, 1, false, bt::null, 0,
                              false);  // wrapper (recurses if head>127)
         }
         return {bt::full, pack_full(prefix, brs, nb, term, tt, tp)};
      }

      // ── gcart inline-suffix_value body builder (the 3-BYTE form) ───────────────
      // gcart's inline-sv body is [u16 vlen][u8 slen][value][suffix] (sv_leaf::hdr=3,
      // birth-free — it shares the nested node header's birth). This is DELIBERATELY
      // NOT rcart's 4-byte [u16 vlen][u16 slen] form: a 4-byte header would shift every
      // value/suffix offset and corrupt the line (in_value/in_suffix read q+3). The
      // u8 slen caps an INLINE suffix at 255 bytes — a longer suffix MUST externalize
      // (make_sv), so sv_body_fits_ enforces BOTH the 127-byte tail cap AND slen<=255.
      static bool sv_body_fits_(sv suffix, sv value) noexcept
      {
         return suffix.size() <= 255 &&
                size_t(sv_leaf::bytes_for(uint32_t(value.size()), uint32_t(suffix.size()))) <= 127;
      }
      // Write the 3-byte inline body into `dst` (no SAL alloc — pure carry-over bytes,
      // the only make_sv-free way to mint an inline-sv for a direct nested edit). dst
      // must have room for 3 + value.size() + suffix.size() bytes (<=127; caller buffer
      // is 127). Returns the total body byte count (== sv_leaf::bytes_for(vlen, slen)).
      static uint16_t build_sv_body_(char* dst, sv suffix, sv value) noexcept
      {
         assert(sv_body_fits_(suffix, value) && "build_sv_body_ overflow — check sv_body_fits_ first");
         const uint16_t v16 = uint16_t(value.size());
         std::memcpy(dst, &v16, 2);
         dst[2] = char(uint8_t(suffix.size()));  // u8 slen (gcart 3-byte form)
         std::memcpy(dst + 3, value.data(), value.size());
         std::memcpy(dst + 3 + value.size(), suffix.data(), suffix.size());
         return uint16_t(3 + value.size() + suffix.size());
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
      // into <unique> + re-inlined. PURE COPY — no refcount. The grandchild ptrs are
      // aliases of the inline-setlist's grandchildren (Phase C: sharing a child down a
      // clone costs ZERO — the carried ptr is verbatim, the old line's death-stamped
      // drain balances liveness). The fresh slot is a born-THIS-txn working node about
      // to be recursed <unique>, so set_birth(_V)/set_death(0) (the recycled slot may
      // carry a stale death; copy_setlist body never touched birth/death — set them).
      uint32_t externalize_inline_setlist_(const char* body, uint16_t blen)
      {
         const uint32_t s = _s1.alloc();
         std::memcpy(_e.cl1.get(s), body, blen);
         setlist* d = static_cast<setlist*>(_e.cl1.get(s));
         d->set_birth(_V);  // born this txn (working node)
         d->set_death(0);   // alive (the recycled slot may hold a stale death)
         return s;
      }

      // ── THE CANONICAL nested_setlist byte packer ──────────────────────────────
      // Writes header(meta/nbranch/prefix) + okeys + otypes + ooff + the from-END
      // tail-heap of a nested_setlist into slot `n` from a decoded nentry set. PURE
      // BYTES from hdr onward: it NEVER touches birth/death (the caller stamps those)
      // and has NO refcount/alloc/free side-effects (the consume/transfer bookkeeping
      // is the CALLER's job). `es[0..ne)` are the entries IN INDEX ORDER (byte-branches
      // sorted ascending, then — if `term` — the term entry LAST at index ne-1; the
      // term's okey is unused). Returns false WITHOUT writing a single byte if the line
      // would overflow (prefix >= inline_flag, tail>127, or head+tail>128), so a
      // same-slot unique re-pack can fail safely and the caller de-nests from the
      // still-intact original. This is the one helper that initial-pack, unique in-place
      // edit, and shared clone+edit ALL funnel through, making a nested line's body
      // bytes a pure function of (prefix, nentry set, term) — the byte-identity that
      // keeps find_in/cursor offset math correct across all three paths.
      bool lay_nested_into_(nested_setlist* n, sv prefix, const nentry* es, uint32_t ne, bool term)
      {
         const uint32_t pl    = uint32_t(prefix.size());
         const uint32_t nbr   = ne - (term ? 1u : 0u);  // byte-branch count
         const uint32_t nslot = ne;
         // SIZE CHECK BEFORE WRITING (a half-written slot can't be safely de-nested).
         // head = hdr + prefix + okeys[nbr] + otypes(3bit×nslot) + ooff[nslot].
         uint32_t tail = 0;
         for (uint32_t i = 0; i < ne; ++i)
            tail += (es[i].kind ? es[i].blen : 4u);
         const uint32_t head = nested_setlist::hdr + pl + nbr + tag3_bytes(nslot) + nslot;
         if (pl >= nested_setlist::inline_flag || tail > 127 || head + tail > 128)
            return false;  // overflow — caller de-nests (the original slot is untouched)

         n->meta    = uint8_t((term ? 0x80 : 0) | (pl & 0x7f));
         n->nbranch = uint8_t(nbr);
         std::memcpy(n->base() + nested_setlist::hdr, prefix.data(), pl);
         for (uint32_t i = 0; i < nbr; ++i)
            n->okeys()[i] = es[i].okey;
         // tag3_set is a read-modify-write of a u16, so the otypes array's trailing
         // UNUSED bits (3*nslot may not be byte-aligned) would otherwise carry whatever
         // stale garbage the recycled slot held — making the laid bytes depend on slot
         // history. find_in masks otype to 3 bits so reads are unaffected, but the
         // byte-IDENTITY invariant (create == unique re-pack == shared clone) requires
         // deterministic bytes: zero the tag3 region first so a fresh slot and a recycled
         // one produce the SAME line for the same (prefix, entry set, term).
         std::memset(n->otypes(), 0, tag3_bytes(nslot));
         uint32_t eoff = 0;
         for (uint32_t idx = 0; idx < ne; ++idx)
         {
            const nentry&  e   = es[idx];
            const uint32_t esz = (e.kind ? e.blen : 4u);
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

      // Decode a nested_setlist into a PATH-AGNOSTIC nentry set. memcpy's the WHOLE
      // 128 B node to `scratch128` FIRST (a same-slot unique re-pack rewrites the live
      // slot, so we must read everything from a private snapshot — the from-END
      // tail-heap means a new entry's start can overlap an old entry's bytes). Inline
      // bodies are kept as POINTERS INTO scratch128 (pure carry-over bytes, never
      // make_sv'd, so birth-free 3-byte gcart bodies); externals keep their u32 ptr.
      // No refcount, no birth read (the node header birth is read separately).
      void read_nested_entries_(const nested_setlist* src,
                                char                  scratch128[128],
                                nentry                out[],
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
            if (tag == bt::suffix_value && sv_birth(*_lses, ptr) == _V)
            {
               // born this txn ⟹ private (in the writable active segment, not yet
               // sealed/compactable): safe to consume the SAL leaf + inline its body.
               // Point past the alloc_header + standalone birth prefix → the birth-FREE
               // body, copied verbatim into the (birth-free) inline entry. `src` stays
               // valid through build_node: no leaf alloc happens between here and the
               // memcpy (build_node only allocs FIX nodes), so this-txn leaves don't move.
               src = _lses->get_ref<sv_leaf>(leaf_addr(ptr)).obj()->body() + sv_leaf::birth_bytes;
               uint16_t v16;
               std::memcpy(&v16, src, 2);
               const uint32_t b = sv_leaf::bytes_for(v16, uint8_t(src[2]));
               if (b <= 127)
               {
                  k  = 1;
                  sz = b;
               }
               else
                  src = nullptr;
            }
            else if (tag == bt::setlist &&
                     static_cast<const setlist*>(_e.cl1.get(ptr))->birth() == _V)
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

         // Build the PATH-AGNOSTIC nentry set and STAGE inline bodies into a scratch
         // (lay_nested_into_ only memcpy's raw bytes — no re-resolve, no tight-pack).
         // kind1: the inline-sv body is the birth-FREE 3-byte gcart form (skip the
         //   standalone leaf's [u32 birth] prefix) — copied verbatim. kind2:
         //   copy_setlist_tight into the scratch so the cap==total tight body is what
         //   gets laid (byte-identity with the unique/shared edit paths). The CONSUME
         //   side-effects (free the standalone leaf / free the source setlist / take the
         //   external) run AFTER the bytes are laid — lay_nested_into_ is pure bytes.
         // inline bodies sum to <=127 (the tail budget, rejected above), so ONE 128-byte
         // staging buffer holds every inline body contiguously.
         char     stage[128];
         uint32_t soff = 0;
         nentry   es[257];
         auto     make_entry = [&](uint32_t idx, uint8_t okey, bt tag, uint32_t ptr)
         {
            nentry& o = es[idx];
            o.okey    = okey;
            if (kind[idx] == 1)
            {
               char* dst = stage + soff;
               soff += esize[idx];
               std::memcpy(dst, esrc[idx], esize[idx]);  // esrc points past birth (3-byte body)
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
         n->set_birth(_V);  // version liveness: born this txn
         n->set_death(0);   // Phase B: born alive (the recycled slot may hold a stale death)
         lay_nested_into_(n, prefix, es, nslot, term);  // measured to fit (reject @above)

         // CONSUME side-effects, in pack order (NOT inside lay_nested_into_).
         auto consume = [&](uint32_t idx, uint32_t ptr)
         {
            if (kind[idx] == 1)
               // born-this-txn (gated above) ⟹ private ⟹ free the standalone leaf now
               // (its body now lives inline in this nested line). NO refcount.
               leaf_free(*_lses, ptr);
            else if (kind[idx] == 2)
               // born-this-txn setlist (gated above): SHALLOW free the source line — its
               // u32 grandchildren transferred (copied tight) into the inline body.
               _e.cl1.free(ptr);
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
         n->set_birth(_V);  // version liveness: born this txn
         n->set_death(0);   // Phase B: born alive (the recycled slot may hold a stale death)
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
         n->set_birth(_V);  // version liveness: born this txn
         n->set_death(0);   // Phase B: born alive (the recycled slot may hold a stale death)
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
         // Phase C — NO refcount. Whether private (unique) or snapshot-shared, the new
         // node `out` CARRIES the old node's children (the copied branch list); freeing
         // the old node is a SHALLOW supersession (its line only — the children live in
         // `out`). is_unique drives may_inline above, NOT the free shape: a born-this-txn
         // (private) node frees inline, a committed (snapshot-shared) node logs for the
         // floor-gated drain — log_shallow decides by birth. Sharing the children down to
         // `out` costs ZERO (no retain — start_ver tree_alloc.hpp:19).
         log_shallow(t, p);
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
            const sv esuf = leaf_suffix(*_lses, t, p);
            const sv nsuf = key.substr(depth);
            if (esuf == nsuf)
            {
               auto nv = make_sv(nsuf, val);
               log_deep(t, p);  // old leaf DROPPED (childless): inline if born-this-txn else drain
               return nv;
            }
            // split two leaves into a setlist
            std::string es(esuf), ns(nsuf);  // stable copies (esuf views the old obj we free)
            std::string eval_s(leaf_value(*_lses, t, p));  // existing value
            log_deep(t, p);  // old leaf DROPPED (childless)
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
            // The latch (same as the flat path @below): in <unique> read this node's
            // BIRTH once — born this txn (birth==_V) ⟹ private, re-pack the SAME slot
            // <unique>; an earlier-committed (birth<_V), possibly snapshot-pinned one
            // clones a fresh slot <shared> and recurses children <shared>. In <shared>
            // there is NO birth read — it is the clone path only. nested_edit consumes
            // the node prefix INTERNALLY (it needs the entry depth for the prefix-split
            // fallback), so the latch keys off node_birth, NOT a refcount.
            if constexpr (M == cow_mode::unique)
               if (node_birth(_e, t, p) == _V)
                  return nested_edit<cow_mode::unique>(t, p, key, depth, val);
            return nested_edit<cow_mode::shared>(t, p, key, depth, val);
         }
         // full node: prefetch the offset cacheline for the SPECULATIVE routing byte
         // (key[depth] — correct when prefixless, the common case) BEFORE node_prefix
         // touches the header cacheline, so the header's cold miss overlaps it (§7).
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
         // The latch, compile-time (design §5.2): in <unique> read this node's BIRTH
         // once — birth==V (born this txn) ⟹ private, mutate in place (children stay
         // <unique>); birth<V (an earlier committed txn) ⟹ potentially snapshot-shared,
         // clone and recurse children <shared>. In <shared> there is NO birth read at
         // all — the instantiation is the clone path only.
         if constexpr (M == cow_mode::unique)
            if (node_birth(_e, t, p) == _V)
               return edit_node<cow_mode::unique>(t, p, key, depth, val);
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
               log_deep(tt, tp);  // old term leaf DROPPED (childless)
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

      // Reprefix a routing node to prefix[drop:] (build a copy that CARRIES the
      // children, then SHALLOW-supersede the old). Returns the new (tag, ptr).
      // drop==0 is the COW clone hot path: same slot class → verbatim memcpy (the copy
      // carries every child ptr) → SHALLOW free of the old line. Phase C: NO refcount —
      // sharing the children down to the clone costs ZERO (start_ver tree_alloc.hpp:19).
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
            // the verbatim copy carried the source's (older) birth; this clone is
            // born THIS txn → stamp _V so the latch treats it as private. It also
            // carried the source's death — a fresh clone is ALIVE, so reset death=0.
            set_node_birth(_e, out.first, out.second, _V);
            set_node_death(_e, out.first, out.second, 0);
            log_shallow(t, p);  // old line dies; its children are carried by `out`
            return out;
         }
         br         scratch[257];
         uint32_t   nb   = 0;
         bt         tt   = bt::null;
         uint32_t   tp   = 0;
         const bool term = node_term(t, p, tt, tp);
         read_branches(t, p, [&](const br& b) { scratch[nb++] = b; });
         std::string np(node_prefix(t, p).substr(drop));
         auto        out = build_node(np, scratch, nb, term, tt, tp);  // carries children
         log_shallow(t, p);  // old line dies; children live in `out`
         return out;
      }

      // Un-nest a nested_setlist into a flat structure: each inner setlist becomes
      // a STANDALONE cl1 setlist, and a flat OUTER setlist points to them. Returns
      // the flat (tag,ptr). FALLBACK-ONLY now: nested is DIRECTLY MUTABLE (nested_edit/
      // nested_remove re-pack the line in place), so this no longer runs on every edit —
      // only for (a) nested_edit's PREFIX-SPLIT arm (the key diverges from the node
      // prefix, a genuine shape change), and (b) nentry_to_flat_ reuses this de-nest
      // logic on a true 128B emit overflow. Phase C — NO refcount: the new flat `out`
      // CARRIES the nested's external ptrs + inline-setlist u32 grandchildren and
      // reconstructs FRESH leaves for the inline suffix_values, so the husk is a SHALLOW
      // supersession (log_shallow decides inline-vs-drain by birth).
      std::pair<bt, uint32_t> promote_nested(bt t, uint32_t p)
      {
         const bool      priv  = node_birth(_e, t, p) == _V;  // born this txn ⟹ private
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
               static_cast<setlist*>(_e.cl1.get(s))->set_birth(_V);  // born this txn
               static_cast<setlist*>(_e.cl1.get(s))->set_death(0);   // alive (Phase B)
               scratch[nb++] = br{n->okeys()[i], bt::setlist, s};
            }
            else  // inline suffix_value → fresh standalone SAL 1.0 leaf (born this txn)
            {
               // Reconstruct the (suffix,value) from the inline (birth-free) body and
               // allocate a fresh sv_leaf via the leaf session — make_sv stamps birth.
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
         // Phase C — NO refcount. The new flat `out` CARRIES the nested's external ptrs
         // and its inline-setlist u32 grandchildren (copied into the standalone setlist),
         // and reconstructs FRESH leaves for the inline suffix_values; the nested's own
         // inline data dies with its line. So the husk is a SHALLOW supersession in BOTH
         // cases — born-this-txn (priv) frees inline, committed (shared) logs for the
         // floor-gated drain. log_shallow decides by birth (priv is just the inline case).
         (void)priv;
         log_shallow(bt::nested_setlist, p);
         return out;
      }

      // ── DIRECTLY-MUTABLE nested_setlist edit (kills promote-on-write) ───────────
      // Re-pack the nested line so the result is BYTE-IDENTICAL (body bytes) to what
      // pack_nested_leaves would have produced for the same (prefix, entries, term):
      // both funnel through lay_nested_into_. UNIQUE re-packs the SAME slot (birth
      // already _V — never re-stamp); SHARED allocs a FRESH cl1 slot (set_birth(_V)/
      // set_death(0)) and re-packs into it, then log_shallow's the OLD (committed) line
      // ONCE at emit (the snapshot keeps it; the floor-gated drain cascades its
      // superseded children later). Only on a genuine 128B overflow does it fall back
      // to nentry_to_flat_ (the de-nest); a key-prefix DIVERGENCE falls back to
      // promote_nested + the flat path (the only remaining promote). `depth` is the
      // depth at node ENTRY (the nested prefix not yet consumed). Phase C — NO refcount:
      // every rcart retain()/release_ref()/retain_shared_carryovers_ is DELETED. The
      // <shared> rule (the single most error-prone translation): a replaced/dropped
      // child is NOT log_deep'd here (the committed old line still points at it for the
      // snapshot) — only the OLD LINE is log_shallow'd once at emit. <unique> is the
      // opposite (log_deep the dropped child immediately — the slot is truly private).
      template <cow_mode CM>
      std::pair<bt, uint32_t> nested_edit(bt t, uint32_t p, sv key, uint32_t depth, sv val)
      {
         const uint32_t  entry_depth = depth;  // for the overflow / prefix-split promote re-run
         nested_setlist* nn          = static_cast<nested_setlist*>(resolve(_e, t, p));
         const sv        pfx         = nn->prefix();
         const sv        ksuf        = key.substr(depth);
         const uint32_t  cp          = lcp(pfx, ksuf);
         // PREFIX SPLIT: the node's prefix diverges from the key → it needs a new
         // parent. A genuine shape change; promote the nested to a flat node and re-run
         // the edit at the SAME entry depth (the flat node keeps the prefix, upsert_rec
         // re-consumes it). (The hot case is cp==pfx.size().)
         if (cp < pfx.size())
         {
            auto fl = promote_nested(t, p);
            return upsert_rec<cow_mode::unique>(fl.first, fl.second, key, entry_depth, val);
         }
         depth += uint32_t(pfx.size());

         // Decode the current entries into a path-agnostic set, reading from a PRIVATE
         // scratch copy so a same-slot rewrite can read while it overwrites (the from-END
         // tail-heap means a new entry's start can overlap an old entry). es[] is sized
         // es_cap (64), NOT 257 — this descent recurses once per nested level and a fat
         // frame would overflow the stack on a deep key.
         char     scratch128[128];
         nentry   es[nested_setlist::es_cap];
         uint32_t ne;
         bool     term;
         read_nested_entries_(nn, scratch128, es, ne, term);
         const uint32_t nbr = ne - (term ? 1u : 0u);

         // staging for any NEW inline-sv body this edit mints (build_sv_body_, the gcart
         // 3-byte form — pure bytes, never make_sv'd for a carry-over). Alive across emit.
         char newbody[127];
         bool inc_size = false;
         // mint an EXTERNAL leaf into entry idx (value too big to inline). The leaf is
         // freshly owned (make_sv); on a shared clone it is NOT a carry-over (it never
         // existed on the old line) so nothing extra is needed — there are no carry-over
         // retains in gcart at all.
         auto mint_external = [&](uint32_t idx, sv suffix)
         {
            auto mv       = make_sv(suffix, val);
            es[idx].kind  = 0;
            es[idx].otype = mv.first;
            es[idx].ext   = mv.second;
         };

         if (depth == key.size())  // TERM
         {
            if (term)
            {
               const uint32_t ti = ne - 1;
               nentry&        te = es[ti];
               if (te.kind == 0)  // external term → drop old leaf, REP with inline-sv/ext
               {
                  // The old external term is NOT recursed (a term leaf has no child
                  // recursion), and gcart's OLD nested line is SHALLOW-superseded at emit
                  // (frees only the line, never cascades) — so the replaced term leaf MUST
                  // be logged HERE in BOTH modes (log_deep decides inline-vs-drain by
                  // birth: a committed leaf logs for the floor-gated drain, keeping it
                  // alive for any snapshot; a born-this-txn one frees inline). gcart's
                  // shallow-supersession analog of the flat edit_node's unconditional
                  // log_deep(tt,tp) on a term REPLACE (NOT rcart's "shared defers").
                  log_deep(te.otype, te.ext);
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
            int           pos  = -1;
            for (uint32_t i = 0; i < nbr; ++i)
               if (es[i].okey == byte)
               {
                  pos = int(i);
                  break;
               }
            if (pos < 0)  // ABSENT → INS a new inline-sv (or external big), sorted
            {
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
               else  // value too big to inline → external leaf (freshly owned)
                  mint_external(ip, key.substr(depth + 1));
               inc_size = true;
            }
            else if (es[uint32_t(pos)].kind == 0)  // PRESENT external → recurse, REP its ptr
            {
               nentry& ce = es[uint32_t(pos)];
               // Recurse <CM> so a COMMITTED child is CLONED (never mutated in place).
               // The recursion returns the new (tag,ptr); the OLD ptr's supersession is
               // handled INSIDE the recursion (it log_shallow/log_deep's its own husk by
               // birth). No retain (gcart sharing is free).
               auto nc  = upsert_rec<CM>(ce.otype, ce.ext, key, depth + 1, val);
               ce.otype = nc.first;
               ce.ext   = nc.second;
               ce.kind  = 0;
            }
            else if (es[uint32_t(pos)].kind == 1)  // PRESENT inline-sv
            {
               nentry& ce = es[uint32_t(pos)];
               // decode its (value,suffix) from the 3-byte scratch body
               uint16_t v16;
               std::memcpy(&v16, ce.body, 2);
               const uint8_t s8   = uint8_t(ce.body[2]);
               const sv      esuf{ce.body + 3 + v16, s8};
               if (esuf == key.substr(depth + 1))  // SAME key → REP its value bytes
               {
                  if (sv_body_fits_(esuf, val))
                  {
                     // esuf views ce.body (scratch128); build into newbody, THEN repoint
                     // ce.body — read esuf before it is shadowed by the write.
                     std::string esuf_s(esuf);
                     uint16_t    bl = build_sv_body_(newbody, esuf_s, val);
                     ce.body        = newbody;
                     ce.blen        = bl;
                  }
                  else  // grew past inline → externalize
                     mint_external(uint32_t(pos), std::string(esuf));
               }
               else  // DEEPER SPLIT: the two leaves share key[depth] but diverge below
               {
                  std::string    esuf_s(esuf);
                  std::string    eval_s(sv{ce.body + 3, v16});
                  const sv       nsuf = key.substr(depth + 1);
                  const uint32_t cp2  = lcp(esuf_s, nsuf);
                  br             brs[2];
                  uint32_t       cnb   = 0;
                  bool           cterm = false;
                  bt             ctt   = bt::null;
                  uint32_t       ctp   = 0;
                  auto           place = [&](const std::string& s, sv v)
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
                        auto t2    = make_sv(sv(s).substr(cp2 + 1), v);
                        brs[cnb++] = br{uint8_t(s[cp2]), t2.first, t2.second};
                     }
                  };
                  place(esuf_s, eval_s);
                  place(std::string(nsuf), val);
                  if (cnb == 2 && brs[0].byte > brs[1].byte)
                     std::swap(brs[0], brs[1]);
                  // The OLD inline-sv body is pure data (no SAL object) — simply replaced;
                  // nothing to drop. Build the child node FLAT (no inline — its leaves stay
                  // external so the child is a real setlist), then try to inline it kind2.
                  auto child = build_node(esuf_s.substr(0, cp2), brs, cnb, cterm, ctt, ctp,
                                          /*may_inline=*/false);
                  if (child.first == bt::setlist &&
                      static_cast<const setlist*>(_e.cl1.get(child.second))->birth() == _V)
                  {
                     const setlist* c = static_cast<const setlist*>(_e.cl1.get(child.second));
                     const uint32_t b = setlist::bytes_for_cap(c->total(), c->prefixlen());
                     if (b <= 127)
                     {
                        char* dst = newbody;  // reuse the stack body buffer
                        copy_setlist_tight(dst, c);
                        _e.cl1.free(child.second);  // grandchildren transfer into the line
                        ce.kind  = 2;
                        ce.otype = bt::setlist;
                        ce.body  = dst;
                        ce.blen  = uint16_t(b);
                     }
                     else
                     {
                        ce.kind  = 0;
                        ce.otype = child.first;
                        ce.ext   = child.second;
                     }
                  }
                  else
                  {
                     ce.kind  = 0;
                     ce.otype = child.first;
                     ce.ext   = child.second;
                  }
                  inc_size = true;
               }
            }
            else  // PRESENT inline-setlist (kind2) → externalize, recurse, keep/re-inline
            {
               nentry& ce = es[uint32_t(pos)];
               // Alloc a standalone cl1 setlist from the inline bytes (pure copy — its u32
               // grandchildren are aliased/carried, no refcount). Then route the next byte
               // THROUGH it <unique> (it is a born-this-txn working node). The recursion
               // clones any committed grandchild it touches (recursing <unique> over a
               // born-_V node still clones a birth<_V grandchild). gcart sharing is free —
               // no grandchild retains.
               uint32_t s  = externalize_inline_setlist_(ce.body, ce.blen);
               auto     nc = upsert_rec<cow_mode::unique>(bt::setlist, s, key, depth + 1, val);
               bool     reinlined = false;
               if (nc.first == bt::setlist &&
                   static_cast<const setlist*>(_e.cl1.get(nc.second))->birth() == _V)
               {
                  const setlist* c = static_cast<const setlist*>(_e.cl1.get(nc.second));
                  const uint32_t b = setlist::bytes_for_cap(c->total(), c->prefixlen());
                  if (b <= 127)
                  {
                     char* dst = newbody;
                     copy_setlist_tight(dst, c);
                     _e.cl1.free(nc.second);
                     ce.kind   = 2;
                     ce.otype  = bt::setlist;
                     ce.body   = dst;
                     ce.blen   = uint16_t(b);
                     reinlined = true;
                  }
               }
               if (!reinlined)
               {
                  ce.kind  = 0;
                  ce.otype = nc.first;
                  ce.ext   = nc.second;
               }
            }
         }

         return emit_nested_<CM>(t, p, pfx, es, ne, term, inc_size);
      }

      // Build a FLAT routing node (setlist / double / full) from an ALREADY-EDITED
      // nentry set — the overflow / de-nest path. The recursion + make_sv side-effects
      // are DONE (the set incorporates them), so we must NOT re-run the edit; we convert
      // this set into standalone children exactly as promote_nested would, but reading
      // from the mutated set. kind0 external → keep the ptr (TAKEN). kind2 inline-setlist
      // → alloc a standalone cl1, memcpy the body, set_birth(_V)/set_death(0) (its u32
      // grandchildren are carried). kind1 inline-sv → reconstruct (suffix,value) from the
      // 3-byte body and make_sv (re-adds the [u32 birth] prefix, birth-stamped _V). Then
      // log_shallow the old husk (a SHALLOW supersession for BOTH paths: its external
      // ptrs + inline-setlist grandchildren are carried into the standalone children, its
      // inline-sv data dies with the line). Phase C — NO refcount in either branch.
      std::pair<bt, uint32_t> nentry_to_flat_(bt       t,
                                              uint32_t p,
                                              bool     is_unique,
                                              sv       pfx,
                                              const nentry* es,
                                              uint32_t ne,
                                              bool     term)
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
            else if (e.kind == 2)  // inline setlist → standalone cl1 (copy body, stamp birth)
            {
               uint32_t s = _s1.alloc();
               std::memcpy(_e.cl1.get(s), e.body, e.blen);
               setlist* d = static_cast<setlist*>(_e.cl1.get(s));
               d->set_birth(_V);  // born this txn (recycled slot may hold stale death)
               d->set_death(0);
               tago = bt::setlist;
               ptro = s;
            }
            else  // inline-sv (3-byte body) → fresh standalone SAL-1.0 leaf
            {
               uint16_t v16;
               std::memcpy(&v16, e.body, 2);
               const uint8_t s8 = uint8_t(e.body[2]);
               const sv      val{e.body + 3, v16};
               const sv      suf{e.body + 3 + v16, s8};
               auto          mv = make_sv(suf, val);
               tago             = mv.first;
               ptro             = mv.second;
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
         (void)is_unique;  // both paths SHALLOW-supersede the husk (children carried)
         log_shallow(bt::nested_setlist, p);
         return out;
      }

      // Re-emit a nested line from a mutated nentry set. UNIQUE re-packs the SAME slot p
      // (birth already _V — do NOT re-stamp); SHARED allocs a FRESH cl1 (set_birth(_V)/
      // set_death(0)) and lay's into it, then log_shallow's the OLD line ONCE (the
      // snapshot keeps it; the floor-gated drain cascades the superseded children later).
      // On a genuine 128B overflow the ALREADY-EDITED set is built flat (nentry_to_flat_)
      // — NOT re-run from the original (the recursion side-effects are already done). The
      // original slot p is intact on overflow (read from scratch128, not p), so the
      // de-nest is safe. NO retain/carry-over plumbing (gcart sharing is free).
      template <cow_mode CM>
      std::pair<bt, uint32_t> emit_nested_(bt       t,
                                           uint32_t p,
                                           sv       pfx,
                                           const nentry* es,
                                           uint32_t ne,
                                           bool     term,
                                           bool     inc_size)
      {
         if constexpr (CM == cow_mode::unique)
         {
            nested_setlist* n = static_cast<nested_setlist*>(resolve(_e, t, p));
            if (lay_nested_into_(n, pfx, es, ne, term))  // same-slot re-pack (birth stays _V)
            {
               if (inc_size)
                  ++_size;
               return {t, p};
            }
            auto out = nentry_to_flat_(t, p, /*is_unique=*/true, pfx, es, ne, term);
            if (inc_size)
               ++_size;
            return out;
         }
         else
         {
            // SHARED: alloc a fresh slot, stamp birth=_V/death=0, re-pack into it, then
            // log_shallow the OLD (committed) line ONCE — the snapshot keeps the old line's
            // copy; the floor-gated drain cascades its superseded children later. (Do NOT
            // log the individual replaced/dropped children — the old line still points at
            // them for the snapshot.)
            const uint32_t  s  = _s1.alloc();
            nested_setlist* ns = static_cast<nested_setlist*>(_e.cl1.get(s));
            ns->set_birth(_V);  // born this txn (recycled slot may hold a stale death)
            ns->set_death(0);
            if (lay_nested_into_(ns, pfx, es, ne, term))
            {
               log_shallow(t, p);  // old line dies (committed → drain; born-_V → inline)
               if (inc_size)
                  ++_size;
               return {bt::nested_setlist, s};
            }
            // OVERFLOW on the fresh slot: free it, flat-build from the edited set —
            // nentry_to_flat_ log_shallow's the old line once.
            _s1.free(s);
            auto out = nentry_to_flat_(t, p, /*is_unique=*/false, pfx, es, ne, term);
            if (inc_size)
               ++_size;
            return out;
         }
      }

      // ── DIRECTLY-MUTABLE nested_setlist remove (mirrors nested_edit) ────────────
      // Presence is guaranteed by the caller's find_in pre-check. Decode the entries,
      // drop / REP the matched one, re-emit through emit_nested_ (a remove only ever
      // SHRINKS the line, so a same-slot UNIQUE re-pack never overflows — the overflow
      // arm there is dead but harmless). On shrink to zero the husk is log_shallow'd and
      // {null,0} cascades up. `depth` is at node ENTRY. Phase C — NO refcount: <unique>
      // log_deep's a dropped external child immediately; <shared> does NOT (the committed
      // old line still points at it for the snapshot — only emit log_shallow's the line).
      template <cow_mode CM>
      std::pair<bt, uint32_t> nested_remove(bt t, uint32_t p, sv key, uint32_t depth)
      {
         nested_setlist* nn  = static_cast<nested_setlist*>(resolve(_e, t, p));
         const sv        pfx = nn->prefix();
         depth += uint32_t(pfx.size());  // prefix matched (find_in pre-check); consume it

         char     scratch128[128];
         nentry   es[nested_setlist::es_cap];
         uint32_t ne;
         bool     term;
         read_nested_entries_(nn, scratch128, es, ne, term);
         const uint32_t nbr = ne - (term ? 1u : 0u);

         char reinline_body[127];  // staging for a re-inlined setlist (≠ scratch128)

         if (depth == key.size())  // remove the TERM
         {
            nentry& te = es[ne - 1];  // term is the last entry
            if (te.kind == 0)
            {
               // External term dropped — NOT recursed, and the OLD nested line is
               // SHALLOW-superseded at emit (never cascades). So log the dropped term
               // leaf HERE in BOTH modes (log_deep: committed → drain; born-_V → inline).
               log_deep(te.otype, te.ext);
            }
            // inline-sv term → pure data, just dropped
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
            const uint32_t pi = uint32_t(pos);  // find_in guaranteed presence (pos>=0)
            if (es[pi].kind == 0)  // external child → recurse
            {
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
               }
            }
            else if (es[pi].kind == 1)  // inline-sv leaf (the matched key) → DROP (pure data)
            {
               for (uint32_t i = pi; i + 1 < ne; ++i)
                  es[i] = es[i + 1];
               --ne;
            }
            else  // inline-setlist → externalize, recurse <unique>, then DROP-or-REP
            {
               uint32_t s  = externalize_inline_setlist_(es[pi].body, es[pi].blen);
               auto     nc = remove_rec<cow_mode::unique>(bt::setlist, s, key, depth + 1);
               if (nc.first == bt::null)  // shrank to empty → drop
               {
                  for (uint32_t i = pi; i + 1 < ne; ++i)
                     es[i] = es[i + 1];
                  --ne;
               }
               else  // try to RE-INLINE the surviving setlist, else keep external
               {
                  bool reinlined = false;
                  if (nc.first == bt::setlist &&
                      static_cast<const setlist*>(_e.cl1.get(nc.second))->birth() == _V)
                  {
                     const setlist* c = static_cast<const setlist*>(_e.cl1.get(nc.second));
                     const uint32_t b = setlist::bytes_for_cap(c->total(), c->prefixlen());
                     if (b <= 127)
                     {
                        // re-inline into a SEPARATE staging buffer (scratch128 still backs
                        // the OTHER carry-over inline entries — must not stomp it).
                        char* dst = reinline_body;
                        copy_setlist_tight(dst, c);
                        _e.cl1.free(nc.second);
                        es[pi].kind  = 2;
                        es[pi].otype = bt::setlist;
                        es[pi].body  = dst;
                        es[pi].blen  = uint16_t(b);
                        reinlined    = true;
                     }
                  }
                  if (!reinlined)
                  {
                     es[pi].kind  = 0;
                     es[pi].otype = nc.first;
                     es[pi].ext   = nc.second;
                  }
               }
            }
         }

         if (ne == 0)  // line emptied → free the husk, cascade {null,0} up
         {
            log_shallow(bt::nested_setlist, p);  // committed → drain; born-_V → inline
            return {bt::null, 0};
         }
         // remove never grows the line → the same-slot re-pack always fits.
         return emit_nested_<CM>(t, p, pfx, es, ne, term, /*inc_size=*/false);
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
            log_deep(t, p);  // removed leaf DROPPED (childless)
            return {bt::null, 0};
         }
         if (t == bt::nested_setlist)  // DIRECTLY MUTABLE — re-pack in place (no promote)
         {
            // Same birth==_V latch as the flat remove (@below): born this txn ⟹ private,
            // re-pack the SAME slot <unique>; an earlier-committed (possibly snapshot-
            // pinned) one clones a fresh slot <shared> and recurses children <shared>.
            // nested_remove consumes the node prefix internally.
            if constexpr (M == cow_mode::unique)
               if (node_birth(_e, t, p) == _V)
                  return nested_remove<cow_mode::unique>(t, p, key, depth);
            return nested_remove<cow_mode::shared>(t, p, key, depth);
         }
         // full: prefetch the offset CL for the speculative routing byte before
         // node_prefix touches the header (the cold-miss overlap window, §7).
         if (t == bt::full && depth < key.size())
            prefetch(static_cast<const char*>(resolve(_e, t, p)) + full::branches_off +
                     (uint32_t(uint8_t(key[depth])) << 2));
         depth += uint32_t(node_prefix(t, p).size());  // prefix matched (pre-check); consume it
         if constexpr (M == cow_mode::unique)
            if (node_birth(_e, t, p) == _V)  // born this txn ⟹ private, edit in place
               return edit_remove<cow_mode::unique>(t, p, key, depth);
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
            log_deep(tt, tp);  // removed term leaf DROPPED (childless)
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

      engine&                              _e;
      sal2::fixed_allocator<128>::session  _s1;
      sal2::fixed_allocator<256>::session  _s2;
      sal2::fixed_allocator<1152>::session _s9;
      // The writer's OWNED leaf session (single-thread): every leaf op on the write
      // path drives *_lses — no engine session map, no lock. Created in the ctor.
      sal2::allocator_session_ptr          _lses;
      bt                                   _rt   = bt::null;
      uint32_t                             _rp   = 0;
      uint64_t                             _size = 0;
      // Write version: the birth stamped on every node this tree creates. Bumped
      // per transaction (by spec_kv at txn start), so nodes committed by an earlier
      // txn read back birth<_V → the latch clones them (COW); nodes born this txn
      // read birth==_V → mutate in place. Default 1 = the always-in-place case used
      // by direct (snapshot-free) tree drivers like the spec oracle.
      uint32_t                             _V    = 1;
      std::string _scan;  // reused buffer for remove()'s presence pre-check (no per-call alloc)
      // Phase C per-txn SUPERSESSION LOG (start_ver clone-log): committed nodes
      // displaced this txn, awaiting the floor-gated drain. Cleared at begin_txn;
      // the commit moves it to the database drain. Single-thread (the writer).
      std::vector<dead_entry>              _clone_log;
   };
}  // namespace gcart

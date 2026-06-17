// gcart — the spec node taxonomy over the three regions (refcount era).
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
#include <gcart/branch_tag.hpp>  // branch_type, tag3_get/set, tag3_bytes
#include <ucc/lower_bound.hpp>   // ucc::lower_bound_padded (vectorized in-node scan)
#include <sal2/big_allocator.hpp>
#include <sal2/fixed_allocator.hpp>
#include <sal2/var_allocator.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace gcart
{
   using sv = std::string_view;
   using bt = branch_type;

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
      sal2::fixed_allocator<128>  cl1;  // setlist / nested_setlist
      sal2::fixed_allocator<256>  cl2;  // double_setlist / big_nested
      sal2::fixed_allocator<1152> cl9;  // full / node256
      sal2::var_allocator<32>     sal;  // suffix_value leaves (≤2KB)
      sal2::big_allocator<64>     big;  // big_suffix_value leaves (>2KB, BIG region)

      // values whose sv_leaf body exceeds the SAL var-allocator's object cap go BIG.
      static constexpr uint32_t sal_max_body = sal2::var_allocator<32>::max_object_bytes;

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
            sal({.dir = dir, .name = "sal", .max_segments = seg_cap("GCART_SAL_SEGS", 16384)}),
            big({.dir = dir, .name = "big", .max_segments = seg_cap("GCART_BIG_SEGS", 8192)})
      { /* A/B: bg release thread disabled (sync release) */
      }

      ~engine()
      {
         if (_bg.joinable())
         {
            _bg_stop.store(true, std::memory_order_release);
            _bg.join();
         }
      }

      // raw live bytes; drain() first for an accurate (settled) figure.
      uint64_t live_bytes_raw() noexcept
      {
         return cl1.live_bytes() + cl2.live_bytes() + cl9.live_bytes() + sal.live_bytes() +
                big.live_bytes();
      }
      uint64_t live_bytes() noexcept
      {
         drain();
         return live_bytes_raw();
      }

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

      release_ring<release_ring_cap> _rq;
      std::thread                    _bg;
      std::atomic<bool>              _bg_stop{false};
      std::atomic<uint64_t>          _bg_inflight{0};
   };

   // ── suffix_value (SAL leaf): [u16 vlen][u8 slen][value][suffix] ───────────
   // slen==0 => the term (a "leaf" is suffix_value with a 0-length suffix).
   struct sv_leaf
   {
      static uint32_t bytes_for(uint32_t vlen, uint32_t slen) noexcept { return 3 + vlen + slen; }
      static uint32_t make(engine& e, sal2::var_allocator<32>::session& ses, sv suffix, sv value)
      {
         const uint32_t vl = uint32_t(value.size()), sl = uint32_t(suffix.size());
         const uint32_t slot = ses.alloc(bytes_for(vl, sl));
         char*          p    = static_cast<char*>(e.sal.get(slot));
         const uint16_t v16  = uint16_t(vl);
         std::memcpy(p, &v16, 2);
         p[2] = char(uint8_t(sl));
         std::memcpy(p + 3, value.data(), vl);
         std::memcpy(p + 3 + vl, suffix.data(), sl);
         return slot;
      }
      static sv value(const engine& e, uint32_t slot) noexcept
      {
         const char* p = static_cast<const char*>(e.sal.get(slot));
         uint16_t    v16;
         std::memcpy(&v16, p, 2);
         return {p + 3, v16};
      }
      static sv suffix(const engine& e, uint32_t slot) noexcept
      {
         const char* p = static_cast<const char*>(e.sal.get(slot));
         uint16_t    v16;
         std::memcpy(&v16, p, 2);
         const uint8_t sl = uint8_t(p[2]);
         return {p + 3 + v16, sl};
      }
      static uint32_t total_bytes(const engine& e, uint32_t slot) noexcept
      {
         const char* p = static_cast<const char*>(e.sal.get(slot));
         uint16_t    v16;
         std::memcpy(&v16, p, 2);
         return bytes_for(v16, uint8_t(p[2]));
      }
   };

   // ── big_leaf (BIG region): [u32 vlen][u8 slen][value][suffix] ──────────────
   // The same (suffix,value) leaf as sv_leaf, but the body is too large for the
   // SAL var-allocator (>2KB) so it lives as ONE contiguous object in the BIG
   // region (address-stable, refcounted). vlen is u32 (big values can exceed 64K);
   // slen stays u8 (the trie suffix is short — keys are small). Reached as a
   // bt::big_suffix_value branch; no SAL slot, no chunk chain.
   struct big_leaf
   {
      static uint32_t body_bytes(uint32_t vlen, uint32_t slen) noexcept { return 5 + vlen + slen; }
      static uint32_t make(engine& e, sv suffix, sv value)
      {
         const uint32_t vl = uint32_t(value.size()), sl = uint32_t(suffix.size());
         const uint32_t h = e.big.alloc(body_bytes(vl, sl));
         char*          p = static_cast<char*>(e.big.get(h));
         std::memcpy(p, &vl, 4);
         p[4] = char(uint8_t(sl));
         std::memcpy(p + 5, value.data(), vl);
         std::memcpy(p + 5 + vl, suffix.data(), sl);
         return h;
      }
      static sv value(const engine& e, uint32_t h) noexcept
      {
         const char* p = static_cast<const char*>(e.big.get(h));
         uint32_t    vl;
         std::memcpy(&vl, p, 4);
         return {p + 5, vl};
      }
      static sv suffix(const engine& e, uint32_t h) noexcept
      {
         const char* p = static_cast<const char*>(e.big.get(h));
         uint32_t    vl;
         std::memcpy(&vl, p, 4);
         const uint8_t sl = uint8_t(p[4]);
         return {p + 5 + vl, sl};
      }
   };

   // ── leaf dispatch: a terminal is suffix_value (SAL) OR big_suffix_value (BIG);
   //    read its (suffix,value) without the caller caring which region. ──────────
   inline sv leaf_suffix(const engine& e, bt t, uint32_t p) noexcept
   {
      return t == bt::big_suffix_value ? big_leaf::suffix(e, p) : sv_leaf::suffix(e, p);
   }
   inline sv leaf_value(const engine& e, bt t, uint32_t p) noexcept
   {
      return t == bt::big_suffix_value ? big_leaf::value(e, p) : sv_leaf::value(e, p);
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
      // [3] pad; [4..7] birth version (GC: the committing txn version, design §4.2).

      static constexpr uint32_t slot_bytes  = 128;
      static constexpr uint32_t birth_off   = 4;
      static constexpr uint32_t hdr         = 8;  // was 3; +pad+u32 birth (version liveness)
      uint32_t birth() const noexcept  // offset 4 is 4-aligned in the cacheline slot
      {
         return *reinterpret_cast<const uint32_t*>(base() + birth_off);
      }
      void set_birth(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + birth_off) = v; }
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
      static constexpr uint32_t hdr         = 8;  // was 2; +pad+u32 birth (version liveness)
      static constexpr uint8_t  inline_flag = 0x80;  // ooff bit7

      bool     has_term() const noexcept { return meta & 0x80; }
      uint32_t prefixlen() const noexcept { return meta & 0x7f; }
      uint32_t nslot() const noexcept { return nbranch + (has_term() ? 1u : 0u); }

      char*       base() noexcept { return reinterpret_cast<char*>(this); }
      const char* base() const noexcept { return reinterpret_cast<const char*>(this); }
      uint32_t birth() const noexcept { return *reinterpret_cast<const uint32_t*>(base() + birth_off); }
      void set_birth(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + birth_off) = v; }
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
      static constexpr uint32_t types_off    = 12;                       // was 8; +4 birth
      static constexpr uint32_t branches_off = 108;                      // 12 + 96
      static constexpr uint32_t prefix_off   = 1132;                     // 108 + 1024
      static constexpr uint32_t max_prefix   = slot_bytes - prefix_off;  // 20

      bool           has_term() const noexcept { return meta & 0x80; }
      uint32_t       prefixlen() const noexcept { return meta & 0x7f; }
      char*          base() noexcept { return reinterpret_cast<char*>(this); }
      const char*    base() const noexcept { return reinterpret_cast<const char*>(this); }
      uint32_t birth() const noexcept { return *reinterpret_cast<const uint32_t*>(base() + birth_off); }
      void set_birth(uint32_t v) noexcept { *reinterpret_cast<uint32_t*>(base() + birth_off) = v; }
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
   // Which FIX class a routing tag lives in (cl1/cl2/cl9). suffix_value → SAL.
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
         case bt::suffix_value:
            return e.sal.get(p);
         case bt::big_suffix_value:
            return e.big.get(p);
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
   [[nodiscard]] inline bool try_retain(engine& e, bt t, uint32_t p) noexcept
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
            return e.sal.try_retain(p);
         case bt::big_suffix_value:
            return e.big.try_retain(p);
         default:
            return false;
      }
   }
   inline void retain(engine& e, bt t, uint32_t p)
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
            e.sal.retain(p);
            break;
         case bt::big_suffix_value:
            e.big.retain(p);
            break;
         default:
            break;
      }
   }
   // returns true iff the slot hit zero (caller recurses into children + frees).
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
         case bt::suffix_value:
            return e.sal.release_one(p);
         case bt::big_suffix_value:
            return e.big.release_one(p);
         default:
            return false;
      }
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

   // Read-path find from a (tag,ptr) root — usable by readers/cursors with NO
   // write sessions. Dispatch on the branch tag, PREFETCH the child before the
   // dependent load (full prefetches the offset cacheline before the type hdr).
   inline bool find_in(engine& e, bt t, uint32_t p, sv key, std::string& out)
   {
      uint32_t depth = 0;
      for (;;)
      {
         if (t == bt::null)
            return false;
         void* node = resolve(e, t, p);
         if (is_leaf_tag(t))  // suffix_value (SAL) or big_suffix_value (BIG)
         {
            if (leaf_suffix(e, t, p) == key.substr(depth))
            {
               sv v = leaf_value(e, t, p);
               out.assign(v.data(), v.size());
               return true;
            }
            return false;
         }
         if (t == bt::full)
         {
            const full* n = static_cast<const full*>(node);
            if (!pfx_ok_(n->prefix(), key, depth))
               return false;
            depth += n->prefixlen();
            if (depth == key.size())
            {
               if (!n->has_term())
                  return false;
               t = bt(n->term_tag);
               p = n->term_ptr;
               prefetch(resolve(e, t, p));
               continue;
            }
            const uint8_t byte = uint8_t(key[depth]);
            prefetch(n->branches() + byte);  // offset cacheline before the type header
            const bt ct = n->btype(byte);
            if (ct == bt::null)
               return false;
            t = ct;
            p = n->branch(byte);
            ++depth;
            prefetch(resolve(e, t, p));
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
                  sv v = nn->in_value(ti);
                  out.assign(v.data(), v.size());
                  return true;
               }
               t = nn->otype(ti);
               p = nn->ext_ptr(ti);
               prefetch(resolve(e, t, p));
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
                  out.assign(q + 3, v);
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
                  prefetch(resolve(e, t, p));
                  continue;
               }
               const int ii = in->find_index(uint8_t(key[depth]));
               if (ii < 0)
                  return false;
               t = in->btype(uint32_t(ii));
               p = in->branch(uint32_t(ii));
               ++depth;
               prefetch(resolve(e, t, p));
               continue;
            }
            t = nn->otype(ui);
            p = nn->ext_ptr(ui);
            ++depth;  // external child
            prefetch(resolve(e, t, p));
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
            prefetch(resolve(e, t, p));
            continue;
         }
         const int idx = n->find_index(uint8_t(key[depth]));
         if (idx < 0)
            return false;
         t = n->btype(uint32_t(idx));
         p = n->branch(uint32_t(idx));
         ++depth;
         prefetch(resolve(e, t, p));
         continue;
      }
   }

   // ── ordered cursor descents (read-only; usable by snapshot cursors) ─────────
   // Body-based leaf readers: read (suffix,value) from a RESOLVED leaf body — works
   // for a standalone leaf (resolve(e,t,p)) and an inline-in-nested suffix_value body.
   inline sv leaf_suffix_body(bt t, const void* b) noexcept
   {
      const char* p = static_cast<const char*>(b);
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
      const char* p = static_cast<const char*>(b);
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

   // Smallest key in the subtree rooted at the RESOLVED node (t, node), with `acc`
   // the key bytes accumulated ABOVE this node. cur_leftmost APPENDS this node's
   // prefix itself. The term (key ends at the prefix) sorts before all byte-branches.
   inline void cur_leftmost(engine&      e,
                            bt           t,
                            const void*  node,
                            std::string  acc,
                            std::string& ok,
                            std::string& ov)
   {
      if (is_leaf_tag(t))
      {
         acc.append(leaf_suffix_body(t, node));
         ok   = std::move(acc);
         sv v = leaf_value_body(t, node);
         ov.assign(v.data(), v.size());
         return;
      }
      if (t == bt::full)
      {
         const full* n = static_cast<const full*>(node);
         acc.append(n->prefix());
         if (n->has_term())
         {
            bt tt = bt(n->term_tag);
            cur_leftmost(e, tt, resolve(e, tt, n->term_ptr), std::move(acc), ok, ov);
            return;
         }
         for (uint32_t b = 0; b < 256; ++b)
         {
            bt ct = n->btype(b);
            if (ct != bt::null)
            {
               acc.push_back(char(b));
               cur_leftmost(e, ct, resolve(e, ct, n->branch(b)), std::move(acc), ok, ov);
               return;
            }
         }
         return;  // unreachable: a full has >=1 child or term
      }
      if (t == bt::nested_setlist)
      {
         const nested_setlist* n = static_cast<const nested_setlist*>(node);
         acc.append(n->prefix());
         if (n->has_term())
         {
            const uint32_t ti = n->nbranch;
            if (n->is_inline(ti))
            {
               ok   = std::move(acc);
               sv v = n->in_value(ti);
               ov.assign(v.data(), v.size());
               return;
            }
            bt tt = n->otype(ti);
            cur_leftmost(e, tt, resolve(e, tt, n->ext_ptr(ti)), std::move(acc), ok, ov);
            return;
         }
         const uint8_t ob = n->okeys()[0];
         acc.push_back(char(ob));
         if (n->is_inline(0))
         {
            if (n->otype(0) == bt::suffix_value)
            {
               acc.append(n->in_suffix(0));
               ok   = std::move(acc);
               sv v = n->in_value(0);
               ov.assign(v.data(), v.size());
               return;
            }
            cur_leftmost(e, bt::setlist, n->in_setlist(0), std::move(acc), ok, ov);
            return;
         }
         bt ct = n->otype(0);
         cur_leftmost(e, ct, resolve(e, ct, n->ext_ptr(0)), std::move(acc), ok, ov);
         return;
      }
      const setlist* n = static_cast<const setlist*>(node);
      acc.append(n->prefix());
      if (n->has_term())
      {
         bt tt = n->term_type();
         cur_leftmost(e, tt, resolve(e, tt, n->term_branch()), std::move(acc), ok, ov);
         return;
      }
      const uint8_t b0 = n->keys()[0];
      acc.push_back(char(b0));
      bt ct = n->btype(0);
      cur_leftmost(e, ct, resolve(e, ct, n->branch(0)), std::move(acc), ok, ov);
   }

   // Smallest FULL key >= `target` in the subtree (t, node), matching target from
   // `depth`; `acc` = key bytes above this node. Fills ok/ov + returns true on a hit.
   inline bool cur_ge(engine&      e,
                      bt           t,
                      const void*  node,
                      sv           target,
                      uint32_t     depth,
                      std::string  acc,
                      std::string& ok,
                      std::string& ov)
   {
      if (is_leaf_tag(t))
      {
         std::string fk = acc;
         fk.append(leaf_suffix_body(t, node));
         if (fk < target)
            return false;
         ok   = std::move(fk);
         sv v = leaf_value_body(t, node);
         ov.assign(v.data(), v.size());
         return true;
      }
      const sv P = (t == bt::full)             ? static_cast<const full*>(node)->prefix()
                   : (t == bt::nested_setlist) ? static_cast<const nested_setlist*>(node)->prefix()
                                               : static_cast<const setlist*>(node)->prefix();
      const uint32_t rem = uint32_t(target.size()) - depth;
      const uint32_t pl  = uint32_t(P.size());
      const uint32_t cl  = pl < rem ? pl : rem;
      uint32_t       i   = 0;
      while (i < cl && P[i] == target[depth + i])
         ++i;
      if (i < cl)  // prefix diverged within the compared region
      {
         if (uint8_t(P[i]) > uint8_t(target[depth + i]))
         {
            cur_leftmost(e, t, node, std::move(acc), ok, ov);
            return true;
         }
         return false;
      }
      if (pl > rem)
      {
         cur_leftmost(e, t, node, std::move(acc), ok, ov);
         return true;
      }  // target is a strict prefix of P → whole subtree > target
      // P fully matches target[depth:depth+pl]. If target ends exactly here, the
      // smallest key >= target is this node's leftmost (term==target, or first
      // child > target) — cur_leftmost appends P, so do NOT pre-append it.
      if (depth + pl == target.size())
      {
         cur_leftmost(e, t, node, std::move(acc), ok, ov);
         return true;
      }
      acc.append(P);
      depth += pl;  // consume P, descend into children at target[depth]
      const uint8_t tb = uint8_t(target[depth]);
      if (t == bt::full)
      {
         const full* n = static_cast<const full*>(node);
         {
            bt ct = n->btype(tb);
            if (ct != bt::null)
            {
               std::string a = acc;
               a.push_back(char(tb));
               if (cur_ge(e, ct, resolve(e, ct, n->branch(tb)), target, depth + 1, std::move(a), ok,
                          ov))
                  return true;
            }
         }
         for (uint32_t b = tb + 1; b < 256; ++b)
         {
            bt c2 = n->btype(b);
            if (c2 != bt::null)
            {
               std::string a = acc;
               a.push_back(char(b));
               cur_leftmost(e, c2, resolve(e, c2, n->branch(b)), std::move(a), ok, ov);
               return true;
            }
         }
         return false;
      }
      if (t == bt::nested_setlist)
      {
         const nested_setlist* n     = static_cast<const nested_setlist*>(node);
         int                   idx   = int(ucc::lower_bound_padded(n->okeys(), n->nbranch, tb));
         uint32_t              start = uint32_t(idx);
         if (idx < int(n->nbranch) && n->okeys()[idx] == tb)
         {
            const uint32_t ui = uint32_t(idx);
            if (n->is_inline(ui))
            {
               if (n->otype(ui) == bt::suffix_value)
               {
                  std::string fk = acc;
                  fk.push_back(char(tb));
                  fk.append(n->in_suffix(ui));
                  if (fk >= target)
                  {
                     ok   = std::move(fk);
                     sv v = n->in_value(ui);
                     ov.assign(v.data(), v.size());
                     return true;
                  }
               }
               else
               {
                  std::string a = acc;
                  a.push_back(char(tb));
                  if (cur_ge(e, bt::setlist, n->in_setlist(ui), target, depth + 1, std::move(a), ok,
                             ov))
                     return true;
               }
            }
            else
            {
               bt          ct = n->otype(ui);
               std::string a  = acc;
               a.push_back(char(tb));
               if (cur_ge(e, ct, resolve(e, ct, n->ext_ptr(ui)), target, depth + 1, std::move(a),
                          ok, ov))
                  return true;
            }
            start = ui + 1;
         }
         if (start < n->nbranch)
         {
            const uint8_t ob = n->okeys()[start];
            std::string   a  = acc;
            a.push_back(char(ob));
            if (n->is_inline(start))
            {
               if (n->otype(start) == bt::suffix_value)
               {
                  a.append(n->in_suffix(start));
                  ok   = std::move(a);
                  sv v = n->in_value(start);
                  ov.assign(v.data(), v.size());
                  return true;
               }
               cur_leftmost(e, bt::setlist, n->in_setlist(start), std::move(a), ok, ov);
               return true;
            }
            bt ct = n->otype(start);
            cur_leftmost(e, ct, resolve(e, ct, n->ext_ptr(start)), std::move(a), ok, ov);
            return true;
         }
         return false;
      }
      const setlist* n     = static_cast<const setlist*>(node);
      int            idx   = int(ucc::lower_bound_padded(n->keys(), n->nbranch, tb));
      uint32_t       start = uint32_t(idx);
      if (idx < int(n->nbranch) && n->keys()[idx] == tb)
      {
         bt          ct = n->btype(uint32_t(idx));
         std::string a  = acc;
         a.push_back(char(tb));
         if (cur_ge(e, ct, resolve(e, ct, n->branch(uint32_t(idx))), target, depth + 1,
                    std::move(a), ok, ov))
            return true;
         start = uint32_t(idx) + 1;
      }
      if (start < n->nbranch)
      {
         const uint8_t ob = n->keys()[start];
         bt            ct = n->btype(start);
         std::string   a  = acc;
         a.push_back(char(ob));
         cur_leftmost(e, ct, resolve(e, ct, n->branch(start)), std::move(a), ok, ov);
         return true;
      }
      return false;
   }

   // Largest key in the subtree (t, node); cur_rightmost appends this node's prefix.
   // The term (key ends at prefix) is the SMALLEST, so it is the rightmost only when
   // there are no byte-branches.
   inline void cur_rightmost(engine&      e,
                             bt           t,
                             const void*  node,
                             std::string  acc,
                             std::string& ok,
                             std::string& ov)
   {
      if (is_leaf_tag(t))
      {
         acc.append(leaf_suffix_body(t, node));
         ok   = std::move(acc);
         sv v = leaf_value_body(t, node);
         ov.assign(v.data(), v.size());
         return;
      }
      if (t == bt::full)
      {
         const full* n = static_cast<const full*>(node);
         acc.append(n->prefix());
         for (int b = 255; b >= 0; --b)
         {
            bt ct = n->btype(uint32_t(b));
            if (ct != bt::null)
            {
               acc.push_back(char(b));
               cur_rightmost(e, ct, resolve(e, ct, n->branch(uint32_t(b))), std::move(acc), ok, ov);
               return;
            }
         }
         bt tt = bt(n->term_tag);
         cur_rightmost(e, tt, resolve(e, tt, n->term_ptr), std::move(acc), ok, ov);
         return;
      }
      if (t == bt::nested_setlist)
      {
         const nested_setlist* n = static_cast<const nested_setlist*>(node);
         acc.append(n->prefix());
         if (n->nbranch > 0)
         {
            const uint32_t i  = n->nbranch - 1u;
            const uint8_t  ob = n->okeys()[i];
            acc.push_back(char(ob));
            if (n->is_inline(i))
            {
               if (n->otype(i) == bt::suffix_value)
               {
                  acc.append(n->in_suffix(i));
                  ok   = std::move(acc);
                  sv v = n->in_value(i);
                  ov.assign(v.data(), v.size());
                  return;
               }
               cur_rightmost(e, bt::setlist, n->in_setlist(i), std::move(acc), ok, ov);
               return;
            }
            bt ct = n->otype(i);
            cur_rightmost(e, ct, resolve(e, ct, n->ext_ptr(i)), std::move(acc), ok, ov);
            return;
         }
         const uint32_t ti = n->nbranch;  // term only
         if (n->is_inline(ti))
         {
            ok   = std::move(acc);
            sv v = n->in_value(ti);
            ov.assign(v.data(), v.size());
            return;
         }
         bt tt = n->otype(ti);
         cur_rightmost(e, tt, resolve(e, tt, n->ext_ptr(ti)), std::move(acc), ok, ov);
         return;
      }
      const setlist* n = static_cast<const setlist*>(node);
      acc.append(n->prefix());
      if (n->nbranch > 0)
      {
         const uint32_t i = n->nbranch - 1u;
         acc.push_back(char(n->keys()[i]));
         bt ct = n->btype(i);
         cur_rightmost(e, ct, resolve(e, ct, n->branch(i)), std::move(acc), ok, ov);
         return;
      }
      bt tt = n->term_type();
      cur_rightmost(e, tt, resolve(e, tt, n->term_branch()), std::move(acc), ok, ov);
   }

   // Largest FULL key STRICTLY < `target` in (t, node), matching from `depth`.
   // Mirror of cur_ge: prefer the ==tb child (largest < target within it), then the
   // largest byte-branch < tb (its rightmost), then this node's term (= acc < target).
   inline bool cur_lt(engine&      e,
                      bt           t,
                      const void*  node,
                      sv           target,
                      uint32_t     depth,
                      std::string  acc,
                      std::string& ok,
                      std::string& ov)
   {
      if (is_leaf_tag(t))
      {
         std::string fk = acc;
         fk.append(leaf_suffix_body(t, node));
         if (!(fk < target))
            return false;
         ok   = std::move(fk);
         sv v = leaf_value_body(t, node);
         ov.assign(v.data(), v.size());
         return true;
      }
      const sv P = (t == bt::full)             ? static_cast<const full*>(node)->prefix()
                   : (t == bt::nested_setlist) ? static_cast<const nested_setlist*>(node)->prefix()
                                               : static_cast<const setlist*>(node)->prefix();
      const uint32_t rem = uint32_t(target.size()) - depth;
      const uint32_t pl  = uint32_t(P.size());
      const uint32_t cl  = pl < rem ? pl : rem;
      uint32_t       i   = 0;
      while (i < cl && P[i] == target[depth + i])
         ++i;
      if (i < cl)  // diverged
      {
         if (uint8_t(P[i]) < uint8_t(target[depth + i]))
         {
            cur_rightmost(e, t, node, std::move(acc), ok, ov);
            return true;
         }
         return false;
      }
      if (pl > rem)
         return false;  // target is a strict prefix of P → all keys here > target
      // P fully matched. If target ends here, every key here is >= target → none < target.
      if (depth + pl == target.size())
         return false;
      acc.append(P);
      depth += pl;
      const uint8_t tb = uint8_t(target[depth]);
      auto term_here = [&](bt& tt, uint32_t& tp) -> int {  // 0 none, 1 external (tt,tp), 2 inline-nested
         if (t == bt::full) { const full* n = static_cast<const full*>(node); if (!n->has_term()) return 0; tt = bt(n->term_tag); tp = n->term_ptr; return 1; }
         if (t == bt::nested_setlist) { const nested_setlist* n = static_cast<const nested_setlist*>(node); if (!n->has_term()) return 0; if (n->is_inline(n->nbranch)) return 2; tt = n->otype(n->nbranch); tp = n->ext_ptr(n->nbranch); return 1; }
         const setlist* n = static_cast<const setlist*>(node); if (!n->has_term()) return 0; tt = n->term_type(); tp = n->term_branch(); return 1;
      };
      if (t == bt::full)
      {
         const full* n = static_cast<const full*>(node);
         {
            bt ct = n->btype(tb);
            if (ct != bt::null)
            {
               std::string a = acc;
               a.push_back(char(tb));
               if (cur_lt(e, ct, resolve(e, ct, n->branch(tb)), target, depth + 1, std::move(a), ok,
                          ov))
                  return true;
            }
         }
         for (int b = int(tb) - 1; b >= 0; --b)
         {
            bt c2 = n->btype(uint32_t(b));
            if (c2 != bt::null)
            {
               std::string a = acc;
               a.push_back(char(b));
               cur_rightmost(e, c2, resolve(e, c2, n->branch(uint32_t(b))), std::move(a), ok, ov);
               return true;
            }
         }
      }
      else if (t == bt::nested_setlist)
      {
         const nested_setlist* n   = static_cast<const nested_setlist*>(node);
         int                   idx = int(ucc::lower_bound_padded(n->okeys(), n->nbranch, tb));
         if (idx < int(n->nbranch) && n->okeys()[idx] == tb)
         {
            const uint32_t ui = uint32_t(idx);
            if (n->is_inline(ui))
            {
               if (n->otype(ui) == bt::suffix_value)
               {
                  std::string fk = acc;
                  fk.push_back(char(tb));
                  fk.append(n->in_suffix(ui));
                  if (fk < target)
                  {
                     ok   = std::move(fk);
                     sv v = n->in_value(ui);
                     ov.assign(v.data(), v.size());
                     return true;
                  }
               }
               else
               {
                  std::string a = acc;
                  a.push_back(char(tb));
                  if (cur_lt(e, bt::setlist, n->in_setlist(ui), target, depth + 1, std::move(a), ok,
                             ov))
                     return true;
               }
            }
            else
            {
               bt          ct = n->otype(ui);
               std::string a  = acc;
               a.push_back(char(tb));
               if (cur_lt(e, ct, resolve(e, ct, n->ext_ptr(ui)), target, depth + 1, std::move(a),
                          ok, ov))
                  return true;
            }
         }
         for (int j = idx - 1; j >= 0; --j)  // okeys[0..idx-1] are all < tb (idx = first >= tb)
         {
            const uint32_t uj = uint32_t(j);
            const uint8_t  ob = n->okeys()[uj];
            std::string    a  = acc;
            a.push_back(char(ob));
            if (n->is_inline(uj))
            {
               if (n->otype(uj) == bt::suffix_value)
               {
                  a.append(n->in_suffix(uj));
                  ok   = std::move(a);
                  sv v = n->in_value(uj);
                  ov.assign(v.data(), v.size());
                  return true;
               }
               cur_rightmost(e, bt::setlist, n->in_setlist(uj), std::move(a), ok, ov);
               return true;
            }
            bt ct = n->otype(uj);
            cur_rightmost(e, ct, resolve(e, ct, n->ext_ptr(uj)), std::move(a), ok, ov);
            return true;
         }
      }
      else
      {
         const setlist* n   = static_cast<const setlist*>(node);
         int            idx = int(ucc::lower_bound_padded(n->keys(), n->nbranch, tb));
         if (idx < int(n->nbranch) && n->keys()[idx] == tb)
         {
            bt          ct = n->btype(uint32_t(idx));
            std::string a  = acc;
            a.push_back(char(tb));
            if (cur_lt(e, ct, resolve(e, ct, n->branch(uint32_t(idx))), target, depth + 1,
                       std::move(a), ok, ov))
               return true;
         }
         for (int j = int(idx) - 1; j >= 0; --j)
         {
            const uint32_t uj = uint32_t(j);
            const uint8_t  ob = n->keys()[uj];
            bt             ct = n->btype(uj);
            std::string    a  = acc;
            a.push_back(char(ob));
            cur_rightmost(e, ct, resolve(e, ct, n->branch(uj)), std::move(a), ok, ov);
            return true;
         }
      }
      // no byte-branch < tb produced a key — fall back to this node's term (= acc < target)
      bt        tt;
      uint32_t  tp;
      const int tk = term_here(tt, tp);
      if (tk == 2)
      {
         const nested_setlist* n = static_cast<const nested_setlist*>(node);
         ok                      = std::move(acc);
         sv v                    = n->in_value(n->nbranch);
         ov.assign(v.data(), v.size());
         return true;
      }
      if (tk == 1)
      {
         ok   = std::move(acc);
         sv v = leaf_value_body(tt, resolve(e, tt, tp));
         ov.assign(v.data(), v.size());
         return true;
      }
      return false;
   }

   // ── batched in-order scan (amortizes cursor next(): one descent collects up to
   //    `limit` keys instead of re-descending per step) ──────────────────────────
   using kv_pair  = std::pair<std::string, std::string>;
   using kv_batch = std::vector<kv_pair>;

   // Emit EVERY key in the subtree (t, node) in order (term first, then byte-branches
   // ascending), appending this node's prefix, until out.size() == limit.
   inline void cur_scan_all(engine&     e,
                            bt          t,
                            const void* node,
                            std::string acc,
                            kv_batch&   out,
                            size_t      limit)
   {
      if (out.size() >= limit)
         return;
      if (is_leaf_tag(t))
      {
         acc.append(leaf_suffix_body(t, node));
         sv v = leaf_value_body(t, node);
         out.emplace_back(std::move(acc), std::string(v.data(), v.size()));
         return;
      }
      if (t == bt::full)
      {
         const full* n = static_cast<const full*>(node);
         acc.append(n->prefix());
         if (n->has_term())
         {
            bt tt = bt(n->term_tag);
            cur_scan_all(e, tt, resolve(e, tt, n->term_ptr), acc, out, limit);
            if (out.size() >= limit)
               return;
         }
         for (uint32_t b = 0; b < 256; ++b)
         {
            bt ct = n->btype(b);
            if (ct != bt::null)
            {
               std::string a = acc;
               a.push_back(char(b));
               cur_scan_all(e, ct, resolve(e, ct, n->branch(b)), std::move(a), out, limit);
               if (out.size() >= limit)
                  return;
            }
         }
         return;
      }
      if (t == bt::nested_setlist)
      {
         const nested_setlist* n = static_cast<const nested_setlist*>(node);
         acc.append(n->prefix());
         if (n->has_term())
         {
            const uint32_t ti = n->nbranch;
            if (n->is_inline(ti))
            {
               sv v = n->in_value(ti);
               out.emplace_back(acc, std::string(v.data(), v.size()));
            }
            else
            {
               bt tt = n->otype(ti);
               cur_scan_all(e, tt, resolve(e, tt, n->ext_ptr(ti)), acc, out, limit);
            }
            if (out.size() >= limit)
               return;
         }
         for (uint32_t i = 0; i < n->nbranch; ++i)
         {
            const uint8_t ob = n->okeys()[i];
            std::string   a  = acc;
            a.push_back(char(ob));
            if (n->is_inline(i))
            {
               if (n->otype(i) == bt::suffix_value)
               {
                  a.append(n->in_suffix(i));
                  sv v = n->in_value(i);
                  out.emplace_back(std::move(a), std::string(v.data(), v.size()));
               }
               else
                  cur_scan_all(e, bt::setlist, n->in_setlist(i), std::move(a), out, limit);
            }
            else
            {
               bt ct = n->otype(i);
               cur_scan_all(e, ct, resolve(e, ct, n->ext_ptr(i)), std::move(a), out, limit);
            }
            if (out.size() >= limit)
               return;
         }
         return;
      }
      const setlist* n = static_cast<const setlist*>(node);
      acc.append(n->prefix());
      if (n->has_term())
      {
         bt tt = n->term_type();
         cur_scan_all(e, tt, resolve(e, tt, n->term_branch()), acc, out, limit);
         if (out.size() >= limit)
            return;
      }
      for (uint32_t i = 0; i < n->nbranch; ++i)
      {
         std::string a = acc;
         a.push_back(char(n->keys()[i]));
         bt ct = n->btype(i);
         cur_scan_all(e, ct, resolve(e, ct, n->branch(i)), std::move(a), out, limit);
         if (out.size() >= limit)
            return;
      }
   }

   // Emit keys >= `target` in (t, node) in order, until out.size() == limit
   // (cur_ge generalized to collect the run, not just the first hit).
   inline void cur_scan(engine&     e,
                        bt          t,
                        const void* node,
                        sv          target,
                        uint32_t    depth,
                        std::string acc,
                        kv_batch&   out,
                        size_t      limit)
   {
      if (out.size() >= limit)
         return;
      if (is_leaf_tag(t))
      {
         std::string fk = acc;
         fk.append(leaf_suffix_body(t, node));
         if (fk < target)
            return;
         sv v = leaf_value_body(t, node);
         out.emplace_back(std::move(fk), std::string(v.data(), v.size()));
         return;
      }
      const sv P = (t == bt::full)             ? static_cast<const full*>(node)->prefix()
                   : (t == bt::nested_setlist) ? static_cast<const nested_setlist*>(node)->prefix()
                                               : static_cast<const setlist*>(node)->prefix();
      const uint32_t rem = uint32_t(target.size()) - depth;
      const uint32_t pl  = uint32_t(P.size());
      const uint32_t cl  = pl < rem ? pl : rem;
      uint32_t       i   = 0;
      while (i < cl && P[i] == target[depth + i])
         ++i;
      if (i < cl)
      {
         if (uint8_t(P[i]) > uint8_t(target[depth + i]))
            cur_scan_all(e, t, node, std::move(acc), out, limit);
         return;
      }
      if (pl > rem)
      {
         cur_scan_all(e, t, node, std::move(acc), out, limit);
         return;
      }
      if (depth + pl == target.size())
      {
         cur_scan_all(e, t, node, std::move(acc), out, limit);
         return;
      }
      acc.append(P);
      depth += pl;
      const uint8_t tb = uint8_t(target[depth]);
      if (t == bt::full)
      {
         const full* n = static_cast<const full*>(node);
         {
            bt ct = n->btype(tb);
            if (ct != bt::null)
            {
               std::string a = acc;
               a.push_back(char(tb));
               cur_scan(e, ct, resolve(e, ct, n->branch(tb)), target, depth + 1, std::move(a), out,
                        limit);
               if (out.size() >= limit)
                  return;
            }
         }
         for (uint32_t b = tb + 1; b < 256; ++b)
         {
            bt c2 = n->btype(b);
            if (c2 != bt::null)
            {
               std::string a = acc;
               a.push_back(char(b));
               cur_scan_all(e, c2, resolve(e, c2, n->branch(b)), std::move(a), out, limit);
               if (out.size() >= limit)
                  return;
            }
         }
         return;
      }
      if (t == bt::nested_setlist)
      {
         const nested_setlist* n     = static_cast<const nested_setlist*>(node);
         int                   idx   = int(ucc::lower_bound_padded(n->okeys(), n->nbranch, tb));
         uint32_t              start = uint32_t(idx);
         if (idx < int(n->nbranch) && n->okeys()[idx] == tb)
         {
            const uint32_t ui = uint32_t(idx);
            if (n->is_inline(ui))
            {
               if (n->otype(ui) == bt::suffix_value)
               {
                  std::string fk = acc;
                  fk.push_back(char(tb));
                  fk.append(n->in_suffix(ui));
                  if (fk >= target)
                  {
                     sv v = n->in_value(ui);
                     out.emplace_back(std::move(fk), std::string(v.data(), v.size()));
                  }
               }
               else
               {
                  std::string a = acc;
                  a.push_back(char(tb));
                  cur_scan(e, bt::setlist, n->in_setlist(ui), target, depth + 1, std::move(a), out,
                           limit);
               }
            }
            else
            {
               bt          ct = n->otype(ui);
               std::string a  = acc;
               a.push_back(char(tb));
               cur_scan(e, ct, resolve(e, ct, n->ext_ptr(ui)), target, depth + 1, std::move(a), out,
                        limit);
            }
            start = ui + 1;
            if (out.size() >= limit)
               return;
         }
         for (uint32_t j = start; j < n->nbranch; ++j)
         {
            const uint8_t ob = n->okeys()[j];
            std::string   a  = acc;
            a.push_back(char(ob));
            if (n->is_inline(j))
            {
               if (n->otype(j) == bt::suffix_value)
               {
                  a.append(n->in_suffix(j));
                  sv v = n->in_value(j);
                  out.emplace_back(std::move(a), std::string(v.data(), v.size()));
               }
               else
                  cur_scan_all(e, bt::setlist, n->in_setlist(j), std::move(a), out, limit);
            }
            else
            {
               bt ct = n->otype(j);
               cur_scan_all(e, ct, resolve(e, ct, n->ext_ptr(j)), std::move(a), out, limit);
            }
            if (out.size() >= limit)
               return;
         }
         return;
      }
      const setlist* n     = static_cast<const setlist*>(node);
      int            idx   = int(ucc::lower_bound_padded(n->keys(), n->nbranch, tb));
      uint32_t       start = uint32_t(idx);
      if (idx < int(n->nbranch) && n->keys()[idx] == tb)
      {
         bt          ct = n->btype(uint32_t(idx));
         std::string a  = acc;
         a.push_back(char(tb));
         cur_scan(e, ct, resolve(e, ct, n->branch(uint32_t(idx))), target, depth + 1, std::move(a),
                  out, limit);
         start = uint32_t(idx) + 1;
         if (out.size() >= limit)
            return;
      }
      for (uint32_t j = start; j < n->nbranch; ++j)
      {
         std::string a = acc;
         a.push_back(char(n->keys()[j]));
         bt ct = n->btype(j);
         cur_scan_all(e, ct, resolve(e, ct, n->branch(j)), std::move(a), out, limit);
         if (out.size() >= limit)
            return;
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

   // ── free reclamation (snapshots / readers; no write sessions needed) ───────
   // release_one() drops the last ref to the DYING sentinel (not 0), so the slot's
   // content stays readable AFTER the decrement — a concurrent writer's write-around
   // reuses only ref==0 (free()'d) slots, never a dying one. Hence we read children
   // AFTER release_one and ONLY when we actually own the kill: a still-shared node
   // (the common COW case) returns immediately without touching its content.
   inline void release_subtree(engine& e, bt t, uint32_t p)
   {
      if (t == bt::null)
         return;
      if (t == bt::suffix_value)
      {
         if (e.sal.release_one(p))
            e.sal.free(p, sv_leaf::total_bytes(e, p));  // dying → content still readable
         return;
      }
      if (t == bt::big_suffix_value)
      {
         if (e.big.release_one(p))
            e.big.free(p);
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
         release_subtree(e, kids[i].tag, kids[i].ptr);
      if (term)
         release_subtree(e, tt, tp);
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
   inline void release_async_(engine& e, bt t, uint32_t p, int& budget, Spill& spill)
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
         if (e.sal.release_one(p))
            e.sal.free(p, sv_leaf::total_bytes(e, p));  // dying → content still readable
         return;
      }
      if (t == bt::big_suffix_value)
      {
         --budget;
         if (e.big.release_one(p))
            e.big.free(p);
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
         release_async_(e, kids[i].tag, kids[i].ptr, budget, spill);
      if (term)
         release_async_(e, tt, tp, budget, spill);
      free_node_(e, t, p);
   }

   // ── engine reclaimer (out-of-line: depends on release_async_) ───────────────
   inline void engine::spill_fn::operator()(bt t, uint32_t p) const noexcept
   {
      if (e->_rq.try_push(kv_pack_(t, p)))
         return;
      int big = 0x7fffffff;  // ring full → finish inline (backpressure)
      release_async_(*e, t, p, big, e->_spill);
   }
   inline void engine::release_async(bt t, uint32_t p) noexcept
   {
      if (t == bt::null)
         return;
      int budget = inline_destroy_budget;
      release_async_(*this, t, p, budget, _spill);
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
            release_async_(*this, bt(uint8_t(v >> 32)), uint32_t(v), big, _spill);
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
            _ssal(e.sal.start_session())
      {
      }

      uint64_t size() const noexcept { return _size; }

      // ── read path: dispatch on the branch tag, prefetch the child BEFORE the
      //    dependent load (Dan's hard rule; full node prefetches the offset
      //    cacheline before reading the 3-bit type header). ────────────────────
      bool find(sv key, std::string& out) const
      {
         return find_in(const_cast<engine&>(_e), _rt, _rp, key, out);
      }

      bt       root_tag() const noexcept { return _rt; }
      uint32_t root_ptr() const noexcept { return _rp; }
      void     set_root(bt t, uint32_t p) noexcept
      {
         _rt = t;
         _rp = p;
      }
      engine& eng() noexcept { return _e; }
      // release a (tag,ptr) subtree through refcount (snapshot drop / abort).
      void release(bt t, uint32_t p) { release_ref(t, p); }

      // ── write path ─────────────────────────────────────────────────────────
      // Public entry descends in <unique> and latches to <shared> on the first
      // snapshot-pinned node (doc §5.2). One refcount read at the root decides
      // in-place vs clone; the cost is cached down the descent.
      void upsert(sv key, sv val)
      {
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
         if (!find_in(_e, _rt, _rp, key, _scan))
            return 0;  // absent — no-op, no clone
         auto [t, p] = remove_rec<cow_mode::unique>(_rt, _rp, key, 0);
         _rt         = t;
         _rp         = p;
         --_size;
         return 1;
      }

     private:
      // Build a terminal leaf: SAL suffix_value if the body fits the var-allocator
      // object cap, else a BIG big_suffix_value (single contiguous object in the
      // address-stable BIG region). Returns the tag so callers stay region-agnostic.
      std::pair<bt, uint32_t> make_sv(sv suffix, sv val)
      {
         if (sv_leaf::bytes_for(uint32_t(val.size()), uint32_t(suffix.size())) <=
             engine::sal_max_body)
            return {bt::suffix_value, sv_leaf::make(_e, _ssal, suffix, val)};
         return {bt::big_suffix_value, big_leaf::make(_e, suffix, val)};
      }
      // WRITER-ONLY COW reclaim (work tree husks). Single-threaded with the writer's
      // alloc, and a release cascade never allocates, so no concurrent slot reuse can
      // race with reading a node's children — unlike the reader path (release_subtree)
      // this needs NO gather-before-release, keeping the COW write path fast. Readers
      // never call this; they release via snap → release_subtree.
      void release_ref(bt t, uint32_t p)
      {
         if (t == bt::null)
            return;
         if (t == bt::suffix_value)
         {
            if (_e.sal.release_one(p))
               _e.sal.free(p, sv_leaf::total_bytes(_e, p));
            return;
         }
         if (t == bt::big_suffix_value)
         {
            if (_e.big.release_one(p))
               _e.big.free(p);
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
         if (t == bt::full)
            _e.cl9.free(p);
         else if (t == bt::double_setlist || t == bt::big_nested)
            _e.cl2.free(p);
         else
            _e.cl1.free(p);
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
            if (tag == bt::suffix_value && _e.sal.unique(ptr))
            {
               src = static_cast<const char*>(_e.sal.get(ptr));  // resolve the SAL slot ONCE
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

         const uint32_t  slot = _s1.alloc();
         nested_setlist* n    = static_cast<nested_setlist*>(_e.cl1.get(slot));
         n->meta              = uint8_t((term ? 0x80 : 0) | (pl & 0x7f));
         n->nbranch           = uint8_t(nb);
         std::memcpy(n->base() + nested_setlist::hdr, prefix.data(), pl);
         for (uint32_t i = 0; i < nb; ++i)
            n->okeys()[i] = brs[i].byte;
         uint32_t eoff = 0;
         auto     lay  = [&](uint32_t idx, bt tag, uint32_t ptr)
         {
            eoff += esize[idx];
            n->ooff()[idx] = uint8_t((kind[idx] ? nested_setlist::inline_flag : 0) | eoff);
            char* dst      = n->base() + 128 - eoff;
            if (kind[idx] == 1)  // inline leaf: copy body, consume the SAL leaf
            {
               std::memcpy(dst, esrc[idx], esize[idx]);
               tag3_set(n->otypes(), idx, bt::suffix_value);
               release_ref(bt::suffix_value, ptr);
            }
            else if (
                kind[idx] ==
                2)  // inline setlist: tight copy, free the source slot (grandchildren transfer)
            {
               copy_setlist_tight(dst, reinterpret_cast<const setlist*>(esrc[idx]));
               tag3_set(n->otypes(), idx, bt::setlist);
               if (_e.cl1.release_one(ptr))
                  _e.cl1.free(ptr);
            }
            else  // external: 4-byte region ptr, TAKEN
            {
               std::memcpy(dst, &ptr, 4);
               tag3_set(n->otypes(), idx, tag);
            }
         };
         for (uint32_t i = 0; i < nb; ++i)
            lay(i, brs[i].tag, brs[i].ptr);
         if (term)
            lay(nb, tt, tp);
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
                             retain(_e, b.tag, b.ptr);
                       });
         if (new_term)
         {
            bt         ot;
            uint32_t   op2;
            const bool had         = node_term(t, p, ot, op2);
            const bool term_is_new = (ed.k == edit::set_term) || !had;
            if (!term_is_new)
               retain(_e, new_tt, new_tp);  // carried over unchanged
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
            const sv esuf = leaf_suffix(_e, t, p);
            const sv nsuf = key.substr(depth);
            if (esuf == nsuf)
            {
               auto nv = make_sv(nsuf, val);
               release_ref(t, p);
               return nv;
            }
            // split two leaves into a setlist
            std::string es(esuf), ns(nsuf);  // stable copies (esuf views the old obj we free)
            std::string eval_s(leaf_value(_e, t, p));  // existing value
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
         if (t == bt::nested_setlist)  // promote-on-write: every edit path mutates flat nodes only.
         {
            auto fl = promote_nested(t, p);
            return upsert_rec<cow_mode::unique>(fl.first, fl.second, key, depth, val);
         }
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
         if constexpr (M == cow_mode::unique)
            if (unique(_e, t, p))
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
            read_branches(out.first, out.second, [&](const br& b) { retain(_e, b.tag, b.ptr); });
            if (node_term(out.first, out.second, tt, tp))
               retain(_e, tt, tp);
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
         read_branches(out.first, out.second, [&](const br& b) { retain(_e, b.tag, b.ptr); });
         if (term)
            retain(_e, tt, tp);
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
            else  // inline suffix_value → fresh SAL leaf (verbatim body copy)
            {
               const uint32_t sz = n->entry_size(i);
               const uint32_t s  = _ssal.alloc(sz);
               std::memcpy(_e.sal.get(s), n->entry(i), sz);
               scratch[nb++] = br{n->okeys()[i], bt::suffix_value, s};
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
            for_each_nested_branch_(n, [&](bt ct, uint32_t cp) { retain(_e, ct, cp); });
            if (oterm && !n->is_inline(onb))
               retain(_e, n->otype(onb), n->ext_ptr(onb));
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
         if (t == bt::nested_setlist)  // promote-on-write, then remove on the flat node
         {
            auto fl = promote_nested(t, p);
            return remove_rec<cow_mode::unique>(fl.first, fl.second, key, depth);
         }
         depth += uint32_t(node_prefix(t, p).size());  // prefix matched (pre-check); consume it
         if constexpr (M == cow_mode::unique)
            if (unique(_e, t, p))
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

      engine&                              _e;
      sal2::fixed_allocator<128>::session  _s1;
      sal2::fixed_allocator<256>::session  _s2;
      sal2::fixed_allocator<1152>::session _s9;
      sal2::var_allocator<32>::session     _ssal;
      bt                                   _rt   = bt::null;
      uint32_t                             _rp   = 0;
      uint64_t                             _size = 0;
      std::string _scan;  // reused buffer for remove()'s presence pre-check (no per-call alloc)
   };
}  // namespace gcart

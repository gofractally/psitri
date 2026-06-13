// kv_tree<T> — the headerless adaptive radix map of design4.md.
//
// Tiers implemented in this v1:
//   leaf<T>      value_ptr  — suffix (rest of key) + value, one alloc (lazy expansion)
//   prefix_node  prefix_node— long shared prefix + one next pointer (path compression)
//   setlist      setlist_u8 — sparse byte router, 1 cacheline, up to 16 branches (R1)
//   node_full    node_full  — dense direct-index router, full_lines_v lines, skip-CL0 (SFULL)
//
// Routing is off the pointer tag: setlist/prefix read the node (R1); node_full
// computes the segment line from the key byte, PREFETCHES it, then reads the slot
// (skip-CL0). Growth: a full setlist widens to node_full. Mutation is in place
// (no COW). The narrow bitset tiers (r32/64/128, c2/c4/c8) and wide setlists are
// the documented next step; the dispatch and growth scaffolding already fits them.
#pragma once
#include <bit>
#include <map>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <start_ver/hkv/cnode.hpp>
#include <start_ver/hkv/key_codec.hpp>
#include <start_ver/hkv/tagged_ptr.hpp>
#include <start_ver/hkv/value_ops.hpp>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace start_ver::hkv
{
   // Compile-time structural policy — a flag set, combinable with operator|:
   //   kv_tree<T>                              plain adaptive-radix
   //   kv_tree<T, mode::buckets>               last-mile buckets from the root
   //   kv_tree<T, mode::adaptive | mode::dense_tiers>
   enum class mode : unsigned
   {
      none        = 0,
      buckets     = 1u << 0,  // terminals are last-mile buckets (collapse sparse/deep tails)
      adaptive    = 1u << 1,  // radix that collapses deep-narrow leaf-splits to buckets
      dense_tiers = 1u << 2,  // allow the segmented-bitset cnode routers (setlist→cN→full)
      wide_stems  = 1u << 3,  // sparse routers consume 2 bytes/hop (setlist_u16); re-stride if dense
   };
   constexpr mode operator|(mode a, mode b) noexcept { return mode(unsigned(a) | unsigned(b)); }
   constexpr mode operator&(mode a, mode b) noexcept { return mode(unsigned(a) & unsigned(b)); }
   constexpr bool has_mode(mode set, mode flag) noexcept
   {
      return (unsigned(set) & unsigned(flag)) != 0;
   }

namespace detail  // implementation types & node helpers — not part of the public API
{
   // ── sparse router: 1 cacheline, byte[i] -> branch[i] (sorted) ───────────────
   // Carries a short shared prefix INLINE (rh.hdr.prefix_len bytes, in the trailing
   // `prefix` field) so a path-compressed run does NOT cost a separate prefix_node
   // hop — measured: dict shared prefixes avg <2 bytes and 99.9% are <=8, so this
   // removes essentially all prefix_node pointer-chases.
   //
   // INLINE CHILDREN (`inl` byte): a setlist can host small leaf children INSIDE its
   // own 128-byte line instead of as separate terminal allocations. `inl` bit 7 =
   // "inline mode active"; bits 0..6 flag, per branch 0..6, whether branch[i] is an
   // INLINE leaf — its handle then holds a tail-offset (resolve at line+128-offset),
   // not an arena address. The leaf bytes live in the line's free region, which is one
   // contiguous block [&branch[nbranch], line+128): members are ordered so the unused
   // branch slots abut the line tail. Inline payloads pack against the tail (offset
   // 128), growing down, in ascending branch-index order — a CANONICAL layout
   // re-established after every structural change (sl_repack_). A key terminating
   // under an inline branch costs NO extra cacheline — the leaf shares the line the
   // descent already loaded. Only setlist_u8 does this (other tiers leave inl == 0).
   template <class Ptr>
   struct setlist_t
   {
      static constexpr int PREFIX_CAP = 7;   // was 8; the `inl` byte reclaims one, keeping
                                             // CAP at 16 (no widen-point shift). 7 still covers
                                             // ~all runs; an 8-byte one takes a prefix_node.
      static constexpr int LANES      = 16;  // stem lanes the SIMD scans read; [CAP..LANES) stay 0xFF
      // CAP derived so the struct fits one line WITH the `inl` byte. branch[] is LAST
      // so the unused slots [nbranch..CAP) plus the line tail [sizeof..128) form one
      // contiguous inline region growing down from the tail.
      static constexpr int CAP = std::min(
          LANES, (int(cacheline_bytes) - int(sizeof(router_hdr_t<Ptr>)) - 1 - PREFIX_CAP - LANES) /
                     int(sizeof(Ptr)));  // 16 at 6B
      router_hdr_t<Ptr>    rh;
      uint8_t              inl;                 // bit7 inline-active; bits0-6 per-branch inline flag
      uint8_t              bytes[LANES];
      uint8_t              prefix[PREFIX_CAP];   // rh.hdr.prefix_len bytes used (moved before branch)
      Ptr                  branch[CAP];          // LAST: tail past branch[nbranch] is the inline region
   };
   using setlist = setlist_t<packed_ptr>;
   static_assert(sizeof(setlist) <= 128, "setlist fits one cacheline (tail is the inline region)");

   // Stems are kept SORTED ascending; unused lanes are padded with 0xFF (the max
   // byte). That makes this container ordered (lower_bound / range scans work) AND
   // lets the SIMD lower_bound be a clean "count of stems strictly < byte" over all
   // 16 lanes with NO lane-masking — a 0xFF pad is never < byte, so padding never
   // counts. This is heart's `ucc::lower_bound`: count-less-than (not find-first-bit),
   // a tiny scalar add-chain for small N, a NEON horizontal-add otherwise.
   template <class Ptr>
   inline int setlist_lb(const setlist_t<Ptr>* s, uint8_t byte) noexcept
   {
#if defined(__ARM_NEON)
      const uint8x16_t v  = vld1q_u8(s->bytes);
      const uint8x16_t lt = vandq_u8(vcltq_u8(v, vdupq_n_u8(byte)), vdupq_n_u8(1));
      return int(vaddlvq_u8(lt));  // # lanes with stem < byte (padding 0xFF never counts)
#elif defined(__SSE2__)
      const __m128i v  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s->bytes));
      const __m128i lt = _mm_cmplt_epi8(_mm_xor_si128(v, _mm_set1_epi8(char(0x80))),
                                        _mm_set1_epi8(char(byte ^ 0x80)));  // unsigned via sign-flip
      return std::popcount(unsigned(uint16_t(_mm_movemask_epi8(lt))));
#else
      int c = 0;
      for (int i = 0; i < setlist_t<Ptr>::CAP; ++i)
         c += s->bytes[i] < byte;  // padded 0xFF lanes never satisfy <byte
      return c;
#endif
   }
   // Point lookup: SWAR byte search (heart's find_byte), NOT a NEON movemask. For the
   // tiny setlists path-compressed data produces (avg ~2 children), the SWAR zero-byte
   // trick over two u64 words has far lower latency than a 16-lane vector + two
   // horizontal-adds. Stems are unique and padding is 0xFF, so a match for byte<0xFF
   // is always a real stem; the `i < n` guard rejects a 0xFF lookup hitting padding.
   template <class Ptr>
   inline int setlist_index(const setlist_t<Ptr>* s, uint8_t byte) noexcept
   {
      constexpr uint64_t ONES = 0x0101010101010101ULL, HIGH = 0x8080808080808080ULL;
      const int          n      = s->rh.hdr.nbranch;
      const uint64_t     target = uint64_t(byte) * ONES;
      uint64_t           d, x, m;
      std::memcpy(&d, s->bytes, 8);
      x = d ^ target;
      m = (x - ONES) & ~x & HIGH;
      if (m) { const int i = std::countr_zero(m) >> 3; if (i < n) return i; }
      std::memcpy(&d, s->bytes + 8, 8);
      x = d ^ target;
      m = (x - ONES) & ~x & HIGH;
      if (m) { const int i = 8 + (std::countr_zero(m) >> 3); if (i < n) return i; }
      return -1;
   }
   template <class Ptr>
   inline Ptr setlist_find(const setlist_t<Ptr>* s, uint8_t byte) noexcept
   {
      const int i = setlist_index(s, byte);
      return i >= 0 ? s->branch[i] : Ptr::null();
   }
   // Sorted update-or-insert: replace in place if present, else open a gap at the
   // lower-bound position and insert (keeps stems sorted). False only when full.
   template <class Ptr>
   inline bool setlist_set(setlist_t<Ptr>* s, uint8_t byte, Ptr p) noexcept
   {
      const int n   = s->rh.hdr.nbranch;
      const int idx = setlist_lb(s, byte);
      if (idx < n && s->bytes[idx] == byte)
      {
         s->branch[idx] = p;  // replace
         return true;
      }
      if (n >= setlist_t<Ptr>::CAP)
         return false;
      std::memmove(&s->bytes[idx + 1], &s->bytes[idx], size_t(n - idx));
      std::memmove(&s->branch[idx + 1], &s->branch[idx], size_t(n - idx) * sizeof(Ptr));
      s->bytes[idx]     = byte;
      s->branch[idx]    = p;
      s->rh.hdr.nbranch = uint16_t(n + 1);
      return true;
   }
   // Remove `byte`: close the gap, restore the 0xFF pad in the freed tail lane.
   template <class Ptr>
   inline void setlist_remove(setlist_t<Ptr>* s, uint8_t byte) noexcept
   {
      const int i = setlist_index(s, byte);
      if (i < 0)
         return;
      const int n = s->rh.hdr.nbranch;
      std::memmove(&s->bytes[i], &s->bytes[i + 1], size_t(n - 1 - i));
      std::memmove(&s->branch[i], &s->branch[i + 1], size_t(n - 1 - i) * sizeof(Ptr));
      s->bytes[n - 1]   = 0xFF;  // keep unused lanes padded for the SIMD lower_bound
      s->rh.hdr.nbranch = uint16_t(n - 1);
   }

   // ── wide router: a setlist that consumes TWO key bytes per hop ───────────────
   // Same shape as `setlist` but the stem is a 16-bit big-endian (key[d]<<8|key[d+1]),
   // so a numeric stem compare == lexicographic order on the two bytes. One cacheline
   // holds the header + 14 stems + 14 branches: it consumes 2 key bytes for 1 cacheline
   // touch (0.5 CL/byte — strictly better than two u8 setlists), collapsing two sparse
   // radix levels into one. It is built ONLINE (heart's model) only where fanout stays
   // small: a node that would exceed 14 stems re-strides back to u8 (self-tuning — dense
   // regions stay byte-at-a-time, sparse chains go wide). Term-less by construction at
   // build; carries an optional term (a key ending exactly here). No inline prefix.
   template <class Ptr>
   struct setlist16_t
   {
      static constexpr int LANES = 16;  // stems padded to 16 so the SIMD lower_bound's
                                        // two 8-lane loads never read past the array
      static constexpr int CAP =
          std::min(LANES, (int(cacheline_bytes) - int(sizeof(router_hdr_t<Ptr>)) - LANES * 2) /
                              int(sizeof(Ptr)));  // 14 at 6B
      router_hdr_t<Ptr>    rh;
      uint16_t             stems[LANES];  // sorted ascending; lanes [CAP..16) always 0xFFFF
      Ptr                  branch[CAP];
      uint8_t              pad[128 - int(sizeof(router_hdr_t<Ptr>)) - LANES * 2 - CAP * int(sizeof(Ptr))];
   };
   using setlist16 = setlist16_t<packed_ptr>;
   static_assert(sizeof(setlist16) == 128, "setlist16 is one cacheline");

   // lower_bound: count of stems strictly < q (0xFFFF padding never counts). NEON over
   // two 8-lane u16 vectors; scalar add-chain otherwise. Same count-less-than trick as
   // the u8 setlist, just 16-bit lanes.
   template <class Ptr>
   inline int sl16_lb(const setlist16_t<Ptr>* s, uint16_t q) noexcept
   {
#if defined(__ARM_NEON)
      const uint16x8_t qd = vdupq_n_u16(q);
      const uint16x8_t lo = vandq_u16(vcltq_u16(vld1q_u16(s->stems), qd), vdupq_n_u16(1));
      const uint16x8_t hi = vandq_u16(vcltq_u16(vld1q_u16(s->stems + 8), qd), vdupq_n_u16(1));
      // lanes 14,15 of the high vector are padding (0xFFFF) → never < q, so no masking.
      return int(vaddvq_u16(lo)) + int(vaddvq_u16(hi));
#else
      int c = 0;
      for (int i = 0; i < setlist16_t<Ptr>::CAP; ++i) c += s->stems[i] < q;
      return c;
#endif
   }
   // The 16-bit big-endian stem for the two bytes at s[i], s[i+1] (numeric order == lex order).
   inline uint16_t two_stem(std::string_view s, size_t i) noexcept
   {
      return uint16_t(uint16_t(uint8_t(s[i])) << 8 | uint8_t(s[i + 1]));
   }
   template <class Ptr>
   inline Ptr sl16_find(const setlist16_t<Ptr>* s, uint16_t q) noexcept
   {
      const int i = sl16_lb(s, q);
      return (i < s->rh.hdr.nbranch && s->stems[i] == q) ? s->branch[i] : Ptr::null();
   }
   template <class Ptr>
   inline Ptr* sl16_find_slot(setlist16_t<Ptr>* s, uint16_t q) noexcept
   {
      const int i = sl16_lb(s, q);
      return (i < s->rh.hdr.nbranch && s->stems[i] == q) ? &s->branch[i] : nullptr;
   }
   // Sorted update-or-insert of a 2-byte stem. False only when full (the re-stride signal).
   template <class Ptr>
   inline bool sl16_set(setlist16_t<Ptr>* s, uint16_t q, Ptr p) noexcept
   {
      const int n   = s->rh.hdr.nbranch;
      const int idx = sl16_lb(s, q);
      if (idx < n && s->stems[idx] == q) { s->branch[idx] = p; return true; }
      if (n >= setlist16_t<Ptr>::CAP) return false;
      std::memmove(&s->stems[idx + 1], &s->stems[idx], size_t(n - idx) * 2);
      std::memmove(&s->branch[idx + 1], &s->branch[idx], size_t(n - idx) * sizeof(Ptr));
      s->stems[idx]     = q;
      s->branch[idx]    = p;
      s->rh.hdr.nbranch = uint16_t(n + 1);
      return true;
   }
   template <class Ptr>
   inline void sl16_remove(setlist16_t<Ptr>* s, uint16_t q) noexcept
   {
      const int n   = s->rh.hdr.nbranch;
      const int idx = sl16_lb(s, q);
      if (idx >= n || s->stems[idx] != q) return;
      std::memmove(&s->stems[idx], &s->stems[idx + 1], size_t(n - 1 - idx) * 2);
      std::memmove(&s->branch[idx], &s->branch[idx + 1], size_t(n - 1 - idx) * sizeof(Ptr));
      s->stems[n - 1]   = 0xFFFF;
      s->rh.hdr.nbranch = uint16_t(n - 1);
   }
   template <class Ptr, class F>
   inline void sl16_for_each(const setlist16_t<Ptr>* s, F&& f)  // (stem, branch) ascending
   {
      for (int i = 0; i < s->rh.hdr.nbranch; ++i) f(s->stems[i], s->branch[i]);
   }

   // ── dense router: CL0 header + 13 direct-index lines, 21 pointers/line ──────
   // Direct-index dense node, derived from sizeof(Ptr): as many branches as fit a cacheline
   // beside the 2-byte (nset,rsvd) population counter, and enough lines to cover all 256 bytes.
   template <class Ptr>
   inline constexpr unsigned full_per_line_v = (cacheline_bytes - 2) / unsigned(sizeof(Ptr));  // 21 at 6B
   template <class Ptr>
   inline constexpr unsigned full_lines_v = (256 + full_per_line_v<Ptr> - 1) / full_per_line_v<Ptr>;  // 13 at 6B
   template <class Ptr>
   struct cline_direct_t
   {
      Ptr     ptr[full_per_line_v<Ptr>];
      uint8_t nset;  // population of this line; optional, iteration only
      uint8_t rsvd;
      [[no_unique_address]] pad_t<cacheline_bytes - full_per_line_v<Ptr> * unsigned(sizeof(Ptr)) - 2>
          pad;  // tile to one CL (node_full_t static_asserts it per instantiation)
   };
   using cline_direct = cline_direct_t<packed_ptr>;
   static_assert(sizeof(cline_direct) == 128, "direct line is one cacheline");

   template <class Ptr>
   struct node_full_t
   {
      static_assert(sizeof(cline_direct_t<Ptr>) == cacheline_bytes, "direct line tiles one cacheline");
      router_hdr_t<Ptr>   rh;
      uint8_t             pad[cacheline_bytes - sizeof(router_hdr_t<Ptr>)];  // CL0 = header (no inline prefix)
      cline_direct_t<Ptr> ranges[full_lines_v<Ptr>];  // bytes 0..255, branchfree reciprocal index
   };
   using node_full = node_full_t<packed_ptr>;
   static_assert(sizeof(node_full) == 128 + full_lines_v<packed_ptr> * 128, "node_full = header + lines");

   // byte -> (line, slot) with no divide: line = byte/21, slot = byte%21.
   // 3121 = ceil(2^16/21); exact for all bytes 0..255 (the smaller s=12 magic errs at 251).
   template <class Ptr>
   inline void full_index(uint8_t byte, unsigned& line, unsigned& slot) noexcept
   {
      constexpr unsigned PL = full_per_line_v<Ptr>;
      if constexpr ((PL & (PL - 1)) == 0)  // power-of-2 lines (8B=16, 4B=32) → shift/mask, no magic
      {
         constexpr unsigned SH = unsigned(std::countr_zero(PL));
         line = unsigned(byte) >> SH;
         slot = unsigned(byte) & (PL - 1);
      }
      else  // non-power-of-2: reciprocal-multiply divide (3121 at 21/line), exactness proven 0..255
      {
         constexpr unsigned RM = (1u << 16) / PL + 1;  // ceil(2^16/PL)
         static_assert([] {
            for (unsigned b = 0; b < 256; ++b)
               if ((b * RM) >> 16 != b / PL) return false;
            return true;
         }(), "full_index reciprocal must be exact for every byte");
         line = (unsigned(byte) * RM) >> 16;
         slot = unsigned(byte) - line * PL;
      }
   }
   // Precomputed byte -> byte-offset of ranges[byte/N].ptr[byte%N] from the node base.
   // Collapses the reciprocal + slot multiply on the descent hot path to one L1 load.
   template <class Ptr>
   inline constexpr std::array<uint16_t, 256> make_full_offtab() noexcept
   {
      std::array<uint16_t, 256> t{};
      for (int b = 0; b < 256; ++b)
      {
         const int li = b / int(full_per_line_v<Ptr>), slot = b - li * int(full_per_line_v<Ptr>);
         t[size_t(b)] = uint16_t(128 + li * 128 + slot * int(sizeof(Ptr)));
      }
      return t;
   }
   template <class Ptr>
   inline constexpr std::array<uint16_t, 256> full_offtab_v = make_full_offtab<Ptr>();
   template <class Ptr>
   inline Ptr full_find(const node_full_t<Ptr>* f, uint8_t byte) noexcept
   {
      unsigned li, slot;
      full_index<Ptr>(byte, li, slot);
      return f->ranges[li].ptr[slot];
   }
   template <class Ptr>
   inline void full_set(node_full_t<Ptr>* f, uint8_t byte, Ptr p) noexcept
   {
      unsigned li, slot;
      full_index<Ptr>(byte, li, slot);
      const bool was_null = f->ranges[li].ptr[slot].is_null();
      f->ranges[li].ptr[slot] = p;
      if (was_null && !p.is_null())
      {
         ++f->ranges[li].nset;
         ++f->rh.hdr.nbranch;
      }
      else if (!was_null && p.is_null())
      {
         --f->ranges[li].nset;
         --f->rh.hdr.nbranch;
      }
   }
   // part COW demotion target: the EXPLICIT full router (design5 implicit->
   // explicit). One cacheline: header + per-line pointers to cline_direct lines
   // that are INDIVIDUALLY refcounted and shareable. Created only by COW from a
   // shared node_full; reads pay one extra dependent load (SPEC principle 7).
   template <class Ptr>
   struct node_fullx_t
   {
      router_hdr_t<Ptr> rh;
      Ptr               lines[full_lines_v<Ptr>];  // tag bits unused (always node_full)
   };

   // part COW demotion target for the segmented routers: CL0 carries the
   // per-segment line pointers (2/4/8 of them); each segment line keeps the
   // cnode seg layout (presence bitmap + rank-dense branches) and is
   // individually refcounted/shareable.
   template <class Ptr>
   struct node_cx_t
   {
      router_hdr_t<Ptr> rh;
      uint8_t           lg;  // log2(NSEG): 1, 2, 3
      uint8_t           pad0;
      Ptr               seglines[8];  // first (1<<lg) used; tag bits unused
   };
   // runtime-geometry seg ops (the cnode<N> helpers are compile-time N)
   inline unsigned cx_segw(unsigned lg) noexcept { return 256u >> lg; }
   inline unsigned cx_maskb(unsigned lg) noexcept { return (256u >> lg) / 8; }
   template <class Ptr>
   inline unsigned cx_cap(unsigned lg) noexcept
   {
      return (cacheline_bytes - cx_maskb(lg)) / unsigned(sizeof(Ptr));
   }
   template <class Ptr>
   inline Ptr* cx_branches(uint8_t* seg, unsigned lg) noexcept
   {
      return reinterpret_cast<Ptr*>(seg + cx_maskb(lg));
   }

   template <class Ptr>
   inline void full_init(node_full_t<Ptr>* f) noexcept
   {
      f->rh.hdr  = node_hdr{0, 0, 0};
      f->rh.term = Ptr::null();
      for (auto& line : f->ranges)
      {
         std::memset(line.ptr, 0xFF, sizeof(line.ptr));  // every slot -> null
         line.nset = 0;
         line.rsvd = 0;
      }
   }

   // ── bucket: the "last-mile" terminal — a flat, lexicographically-sorted set of
   // (suffix,value) entries collapsing a sparse/deep radix tail into ONE node.
   //   [ nent | tail_used | div[nent](u16) ‖ slot[nent](u16) | free | …entries… ]
   // div[i] = big-endian first 2 bytes of entry i's suffix (the SIMD lower_bound key;
   // just an accelerator — the full suffix in the entry is the source of truth, so the
   // bucket is correct for arbitrary binary keys). Entries grow from the tail and never
   // move; insert/remove shift only the small u16 index arrays. Overflow → split to radix.
   struct bucket
   {
      static constexpr int PAGE = 512;  // 4 cachelines; fills then splits to radix
      uint16_t             nent;
      uint16_t             tail_used;
      uint8_t              data[PAGE - 4];  // div[]/slot[] from front; entries from back
   };
   static_assert(sizeof(bucket) == 512, "bucket page");

   // ── prefix_node: [u16 plen][Ptr next][u8 prefix[plen]] — layout derives from sizeof(Ptr) ──
   inline uint16_t pn_plen(const char* p) noexcept { uint16_t v; std::memcpy(&v, p, 2); return v; }
   template <class Ptr>
   inline Ptr pn_next(const char* p) noexcept { Ptr r; std::memcpy(r.b, p + 2, sizeof(Ptr)); return r; }
   template <class Ptr>
   inline void pn_set_next(char* p, Ptr n) noexcept { std::memcpy(p + 2, n.b, sizeof(Ptr)); }
   template <class Ptr>
   inline const uint8_t* pn_pfx(const char* p) noexcept
   {
      return reinterpret_cast<const uint8_t*>(p + 2 + sizeof(Ptr) + 4);  // +4: vstamp
   }
   template <class Ptr>
   inline uint8_t* pn_pfx(char* p) noexcept { return reinterpret_cast<uint8_t*>(p + 2 + sizeof(Ptr) + 4); }
   template <class Ptr>
   inline size_t pn_size(size_t plen) noexcept { return 2 + sizeof(Ptr) + 4 + plen; }

   // ── Stage B: the branch-handle type comes from the Allocator ─────────────────
   // An allocator opts in with a member type `using hkv_handle = packed_ptr_t<N>;`
   // (8 = full pointer, drops the 48-bit VA assumption; 4 = pool index). Default: the
   // 6-byte packed_ptr, so std::allocator trees keep today's exact layout and codegen.
   template <class A, class = void>
   struct handle_of { using type = packed_ptr; };
   template <class A>
   struct handle_of<A, std::void_t<typename A::hkv_handle>> { using type = typename A::hkv_handle; };

   // Structural tuning is compile-time policy (zero runtime branch cost; dead modes are
   // discarded). Defaults = plain adaptive-radix. Buckets: terminals are last-mile buckets
   // from the root. Adaptive: radix that collapses deep-narrow leaf-splits to buckets.
   // DenseTiers: allow the segmented-bitset cnode routers between setlist and node_full.
   // The ELEGANT value mode: T = bytes_value makes the LEAF the value — raw
   // bytes sized to the data, stored after a u16 vlen; find/iterate hand back
   // VIEWS into the leaf (zero tree-side copies, one copy at the API edge).
   struct bytes_value
   {
      std::string_view v;
      static bytes_value from_bytes(std::string_view b) noexcept { return {b}; }
      std::string_view  to_bytes() const noexcept { return v; }
   };

   template <class Key, class T, mode Mode = mode::none, class Allocator = std::allocator<T>>
   class kv_tree
   {
     public:
      using key_type        = Key;
      using mapped_type     = T;
      using value_type      = std::pair<const Key, T>;
      using size_type       = std::size_t;
      using difference_type = std::ptrdiff_t;
      using allocator_type  = Allocator;
      static constexpr mode policy = Mode;

     private:
      // The branch handle: the Allocator's choice of width (default packed_ptr_t<6>). These
      // local spellings shadow the namespace defaults, so every node type inside the tree
      // re-tiles from this one alias; the free helpers deduce Ptr from their node argument.
      using Ptr        = typename handle_of<Allocator>::type;
      using packed_ptr = Ptr;
      using router_hdr = router_hdr_t<Ptr>;
      using setlist    = setlist_t<Ptr>;
      using setlist16  = setlist16_t<Ptr>;
      using node_full  = node_full_t<Ptr>;
      using node_fullx = node_fullx_t<Ptr>;
      using node_cx    = node_cx_t<Ptr>;
      template <int N>
      using cnode = cnode_t<N, Ptr>;
      static_assert(cnode_layout_ok_v<2, Ptr> && cnode_layout_ok_v<4, Ptr> &&
                    cnode_layout_ok_v<8, Ptr>,
                    "cnode CL0 must tile one line at this handle width");

      using codec = key_codec<Key>;
      // Pass cheap keys (string_view, integers) BY VALUE — a const-ref forces a stack
      // spill+reload on the hot lookup path; owning keys (std::string, vector) stay by-ref.
      using key_param =
          std::conditional_t<std::is_trivially_copyable_v<Key> && sizeof(Key) <= 16, Key, const Key&>;
      // Zero-copy codecs (string-like keys) view the key bytes in place — no scratch object
      // on the hot path. Others serialize into a std::string scratch. Detected from an
      // optional `zero_copy` member so existing single-method codecs still work.
      template <class C, class = void>
      struct codec_zc_ : std::false_type {};
      template <class C>
      struct codec_zc_<C, std::void_t<decltype(C::zero_copy)>> : std::bool_constant<C::zero_copy> {};
      static constexpr bool codec_zero_copy = codec_zc_<codec>::value;
      struct no_scratch_ {};
      // Serializing codecs may bring their own scratch type (fixed-width codecs use a plain
      // char buffer — no std::string machinery per encode); default is std::string.
      template <class C, class = void>
      struct codec_scratch_ { using type = std::string; };
      template <class C>
      struct codec_scratch_<C, std::void_t<typename C::scratch>> { using type = typename C::scratch; };
      using scratch_t =
          std::conditional_t<codec_zero_copy, no_scratch_, typename codec_scratch_<codec>::type>;
      static std::string_view enc_(key_param k, scratch_t& s)
      {
         if constexpr (codec_zero_copy) { (void)s; return codec::view(k); }
         else                            return codec::encode(k, s);
      }

     private:
      // The flag set unpacked into the if-constexpr gates used throughout.
      static constexpr bool Buckets    = has_mode(Mode, mode::buckets);
      static constexpr bool Adaptive   = has_mode(Mode, mode::adaptive);
      static constexpr bool DenseTiers = has_mode(Mode, mode::dense_tiers);
      static constexpr bool WideStems  = has_mode(Mode, mode::wide_stems);

     public:

      // A value small enough (and trivially copyable) to live in the pointer's bytes
      // instead of a leaf allocation: the handle width minus the 1-byte tag (5 at the
      // default 6-byte handle). Declared up front: it gates the by-reference element-access
      // API and the iterator's value storage (constraints / conditional types parsed before
      // the rest of the class).
      static constexpr size_t inline_cap   = sizeof(Ptr) - 1;
      static constexpr bool   raw_vals     = std::is_same_v<T, bytes_value>;
      static constexpr bool   inlineable   = !raw_vals &&
                                           std::is_trivially_copyable_v<T> && sizeof(T) <= inline_cap;
      // Clamped so pack/unpack are instantiation-safe even for non-inlineable T (dead
      // code under the runtime guard); == sizeof(T) on the path that actually inlines.
      static constexpr size_t inline_bytes = sizeof(T) <= inline_cap ? sizeof(T) : inline_cap;

     private:
      // Per-node allocation. Every node/value blob is a run of 128-byte-aligned blocks taken
      // straight from the tree's Allocator (rebound to a 128-byte unit), so the low 7 bits stay
      // free for the pointer tag and sizes round up to whole blocks. There is no pool: each node
      // is freed individually (free_all_ walks the tree at teardown / on a throwing path), so the
      // allocator — not a hidden arena — owns and accounts for every byte.
      static constexpr size_t node_block_bytes = 128;
      struct alignas(node_block_bytes) node_block { std::byte raw[node_block_bytes]; };
      using node_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<node_block>;
      using node_at      = std::allocator_traits<node_alloc_t>;
      static constexpr size_t nblocks_(size_t n) noexcept { return (n + node_block_bytes - 1) / node_block_bytes; }
      void* node_alloc_(size_t n)  // throws bad_alloc — the strong-guarantee fault point
      {
         node_alloc_t na(alloc_);
         return node_at::allocate(na, nblocks_(n));
      }
      // Compile-time-size variant (Dan: n IS constant at typed call sites —
      // make it constant; the runtime chain verifiably does not fold). Falls
      // back to the runtime path for allocators without the hook (heap mode).
      template <size_t Bytes>
      void* node_alloc_c_()
      {
         node_alloc_t na(alloc_);
         if constexpr (requires { na.template allocate_c<Bytes>(); })
            return na.template allocate_c<Bytes>();
         else
            return node_at::allocate(na, nblocks_(Bytes));
      }
      void node_free_(void* p, size_t n) noexcept
      {
         node_alloc_t na(alloc_);
         node_at::deallocate(na, static_cast<node_block*>(p), nblocks_(n));
      }

      // ── typed-value lifecycle hooks (raw_vals, allocator-carried) ─────────
      // When the allocator exposes vhooks (vpart's typed-value registry), the
      // tree reports every PHYSICAL death of value bytes (on_release) and
      // every engine-internal byte duplication (on_copy). A COW leaf clone
      // and its husk's later drain fire a balanced copy/release pair; a
      // same-scope move (split rebuild: bytes leave the old leaf into exactly
      // one new node, then the old leaf is freed) fires neither. The only NET
      // calls are therefore user-visible births and deaths, so a registry's
      // refcounts stay exact across overwrite, remove, abort, supersession
      // and drain — including values that die inside the drain thread.
      static constexpr bool vh_on =
          raw_vals && requires(Allocator& a) { a.vhooks->on_release; };
      void                  vh_release_(std::string_view b) noexcept
      {
         if constexpr (vh_on)
            if (auto* h = alloc_.vhooks; h && h->on_release)
               h->on_release(b.data(), b.size(), h->user);
      }
      void vh_copy_(std::string_view b) noexcept
      {
         if constexpr (vh_on)
            if (auto* h = alloc_.vhooks; h && h->on_copy)
               h->on_copy(b.data(), b.size(), h->user);
      }
      void vh_leaf_dies_(const char* L) noexcept
      {
         if constexpr (vh_on)
            vh_release_(leaf_bytes(L));
      }
      void vh_node_dies_(Ptr p) noexcept  // funnel for tag-dispatched frees
      {
         if constexpr (vh_on)
            if (p.tag() == K::value_ptr)
               vh_leaf_dies_(static_cast<const char*>(deref_(p)));
      }

      // Address-resolution chokepoint. Direct handles compile to exactly the old static code
      // (base_ is an empty member and `this` folds away under inlining — asm-gated). Indexed
      // handles resolve base_ + (28-bit line index << 7); base_ is cached from the allocator
      // by rebase_() whenever alloc_ changes, so the hot path is one shift-add, no indirection.
      static constexpr bool indexed = Ptr::indexed;
      struct no_base_t
      {
      };
      void* deref_(packed_ptr p) const noexcept
      {
         if constexpr (indexed)
            return const_cast<std::byte*>(base_) + ((p.raw() & ~uint64_t(0xF)) << 3);
         else
            return p.ptr();
      }
      // Same resolution for a raw handle value already in a register (the descend hot path).
      const char* deref_raw_(uint64_t raw) const noexcept
      {
         if constexpr (indexed)
            return reinterpret_cast<const char*>(base_) + ((raw & ~uint64_t(0xF)) << 3);
         else
            return reinterpret_cast<const char*>(raw & ~uint64_t(0x7f));
      }
      // ── version stamps (vpart liveness, cow_enabled+indexed only) ──────────
      // Every stamped node carries a u32 transaction id in kind-specific slack
      // (routers: pad bytes — read-path offsets untouched; leaf/prefix_node:
      // a real +4 B field, see their layout blocks). Offsets assume the 4-byte
      // line_ptr instantiation (router_hdr_t = 6 B); guarded by k_has_vstamp_.
      static constexpr int vstamp_off_(K k) noexcept
      {
         switch (k)
         {
            case K::value_ptr:   return 0;                    // leaf prepend
            case K::prefix_node: return 2 + int(sizeof(Ptr));  // after [plen][next]
            case K::setlist_u8:  return 96;   // header-ward (just past sizeof==94): the clone
                                              // copies only [0,94), never the empty [96,128) tail
            case K::setlist_u16: return 104;  // tail pad (102 used)
            case K::c2:
            case K::c4:
            case K::c8:          return 8;    // CL0 pad after router_hdr
            case K::node_full:   return 8;    // CL0 pad after router_hdr
            case K::hdr:         return 8;    // same CL0 slot (prefix at 12)
            default:             return -1;   // bucket/cbucket/fx/cx: no stamp
         }
      }
      static constexpr bool k_has_vstamp_(K k) noexcept { return vstamp_off_(k) >= 0; }
      uint32_t vstamp_get_(Ptr p) const noexcept
      {
         const int off = vstamp_off_(p.tag());
         if (off < 0)
            return 0;  // stampless kinds read as "ancient" (always shared)
         uint32_t v;
         std::memcpy(static_cast<void*>(&v), static_cast<const char*>(deref_(p)) + off, 4);
         return v;
      }
      packed_ptr pack_(const void* node, K k) const noexcept
      {
         if constexpr (indexed)
         {
            if constexpr (cow_stamped_)
            {
               // stamp at creation (also safe on in-place re-tags: those only
               // run on privatized nodes, whose stamp already == the window)
               static_assert(sizeof(Ptr) == 4, "vstamp offsets assume the 4 B line_ptr");
               if (const int off = vstamp_off_(k); off >= 0)
               {
                  const uint32_t v = alloc_.hkv_vtxn();
                  std::memcpy(const_cast<char*>(static_cast<const char*>(node)) + off,
                              static_cast<const void*>(&v), 4);
               }
            }
            return packed_ptr::from_off(size_t(static_cast<const std::byte*>(node) - base_), k);
         }
         else
            return packed_ptr::from(node, k);
      }
      // Cache the allocator's mapping base (indexed handles only; a no-op for direct).
      void rebase_() noexcept
      {
         if constexpr (indexed) base_ = alloc_.hkv_base();
      }

     public:
      kv_tree() { rebase_(); }
      explicit kv_tree(const Allocator& a) : alloc_(a) { rebase_(); }
      kv_tree(const kv_tree&)            = delete;
      kv_tree& operator=(const kv_tree&) = delete;

      // Move construction always steals (no allocation) — noexcept iff the allocator's move
      // is. The source is left valid-empty.
      kv_tree(kv_tree&& o) noexcept(std::is_nothrow_move_constructible_v<Allocator>)
          : alloc_(std::move(o.alloc_)), root_(o.root_), count_(o.count_)
      {
         rebase_();
         o.root_  = packed_ptr::null();
         o.count_ = 0;
      }
      // Move assignment per [container.requirements]: steal when the allocator propagates or
      // the allocators are equal; otherwise (unequal, non-propagating — e.g. PMR) the foreign
      // allocator can't free our nodes, so move the elements into our own allocator. noexcept
      // exactly when allocators are always equal (the steal path can't throw).
      kv_tree& operator=(kv_tree&& o) noexcept(std::allocator_traits<Allocator>::is_always_equal::value)
      {
         if (this == &o) return *this;
         using AT = std::allocator_traits<Allocator>;
         if constexpr (AT::propagate_on_container_move_assignment::value)
         {
            clear();
            alloc_ = std::move(o.alloc_);  // take the allocator that owns o's nodes, then the nodes
         }
         else if (alloc_ == o.alloc_)
         {
            clear();  // equal allocators are interchangeable → our alloc_ can free o's nodes
         }
         else
         {
            clear();
            move_elements_from_(o);  // foreign allocator can't free our nodes → reallocate
            o.clear();
            return *this;
         }
         rebase_();
         root_    = o.root_;
         count_   = o.count_;
         o.root_  = packed_ptr::null();
         o.count_ = 0;
         return *this;
      }
      ~kv_tree() { free_all_(root_); }  // walk the tree, destroy every T and free every node

      allocator_type get_allocator() const noexcept { return alloc_; }

      // Insert or update. Returns true if the key was newly inserted.
      bool insert(key_param k, const T& v)
      {
         T tmp(v);
         return insert(k, std::move(tmp));
      }
      bool insert(key_param k, T&& v) { return insert_<true>(k, std::move(v)); }

      // Construct the value from args and insert ONLY if the key is absent (std::map
      // emplace/try_emplace semantics: a present key is left untouched). Returns true iff
      // inserted. The value is constructed from args exactly once, AT its final address
      // (leaf / bucket entry) — and not at all when the key turns out to exist (stronger
      // than std::map, which may construct and discard).
      template <class... Args>
      bool emplace(key_param k, Args&&... args)
      {
         return insert_<false>(k, em_t<Args...>{std::forward_as_tuple(std::forward<Args>(args)...)});
      }
      template <class... Args>
      bool try_emplace(key_param k, Args&&... args)
      {
         return emplace(k, std::forward<Args>(args)...);
      }

     private:
      // The one insert descent. Assign decides the exact-match action: overwrite (insert)
      // or leave untouched (emplace); everything else — splits, widens, growth — is shared.
      template <bool Assign, class VF>
      bool insert_(key_param k, VF&& v)
      {
         scratch_t ks_;
         std::string_view key = enc_(k, ks_);
         if constexpr (Buckets)  // bucket-mode: terminals are buckets, routers from splits
         {
            bool ins = false;
            root_ = bkt_put<Assign>(root_, key, 0, std::forward<VF>(v), ins);
            count_ += ins;
            return ins;
         }
         // template<Shared> (Dan / psitri's proven pattern): the descent starts
         // in the UNIQUE instantiation — every ownership branch compiled out —
         // and tail-switches into SHARED at the first old node; below that the
         // children-of-shared-are-shared invariant means SHARED needs no checks
         // either: own_suffix_(transition index) clones with NO scan.
         ins_path path_;
         path_[0] = &root_;
         return insert_impl_<Assign, false>(key, std::forward<VF>(v), path_, 1, 0, 0, 0);
      }

      // Insert descend stack: inline for every practical depth, heap-spilled
      // for pathological/long keys (each frame is one descend hop; long keys
      // can hop once per byte in the worst pre-compression shapes).
      struct ins_path
      {
         static constexpr uint32_t INLINE = 72;
         packed_ptr*  sbo[INLINE];
         packed_ptr** d   = sbo;
         uint32_t     cap = INLINE;
         ~ins_path()
         {
            if (d != sbo)
               ::operator delete(d);
         }
         packed_ptr*& operator[](uint32_t i) noexcept { return d[i]; }
         void         ensure(uint32_t n)
         {
            if (n < cap) [[likely]]
               return;
            const uint32_t ncap = cap * 2;
            auto** nd = static_cast<packed_ptr**>(::operator new(size_t(ncap) * sizeof(packed_ptr*)));
            std::memcpy(nd, d, size_t(cap) * sizeof(packed_ptr*));
            if (d != sbo)
               ::operator delete(d);
            d   = nd;
            cap = ncap;
         }
      };

      // The one insert descent body, instantiated per (Assign, Shared).
      // slot == path[pdep-1]; fshared = the Shared-mode transition index; the
      // transition tail-call resumes at the SAME position (depth/lowfan carry).
      template <bool Assign, bool Shared, class VF>
      bool insert_impl_(std::string_view key, VF&& v, ins_path& path_, uint32_t pdep_,
                        uint32_t fshared_, size_t depth, unsigned lowfan_in)
      {
         packed_ptr* slot   = path_[pdep_ - 1];
         [[maybe_unused]] packed_ptr* pslot = nullptr;  // slot holding the parent of *slot (for fusion)
         [[maybe_unused]] unsigned lowfan = lowfan_in;  // routers with <3 branches (deep-narrow signature)
         (void)fshared_;
         for (;;)
         {
            packed_ptr cur = *slot;
            if (cur.is_null())
            {
               if constexpr (Shared)  // parents of this slot must be private
                  slot = own_suffix_(path_, pdep_, fshared_);
               *slot = make_value_at(key, depth, std::forward<VF>(v));
               ++count_;
               return true;
            }
            // UNIQUE → SHARED transition: one well-predicted branch per hop;
            // the SHARED instantiation never tests again (invariant).
            if constexpr (cow_stamped_ && !Shared)
               if (alloc_.hkv_vtxn() && cur.tag() != K::value_inline && cow_shared_(cur))
                  return insert_impl_<Assign, true>(key, std::forward<VF>(v), path_, pdep_,
                                                    pdep_ - 1, depth, lowfan);
            const K k = cur.tag();
            // Adaptive: track the deep-narrow signature. Uniform data widens fast
            // (high fanout → lowfan resets), so it never trips; clustered/deep data
            // accumulates low-fanout levels and collapses its leaf-splits to buckets.
            if constexpr (Adaptive)
               if (is_router(k))
               {
                  const unsigned nb = static_cast<router_hdr*>(deref_(cur))->hdr.nbranch;
                  lowfan            = (nb < 3) ? lowfan + 1 : 0;
               }
            // Terminal bucket: only the bucket/adaptive policies ever create one, so the
            // other modes don't pay this tag test on every descend hop.
            if constexpr (Buckets || Adaptive)
               if (k == K::bucket)
               {
                  bucket*   b = static_cast<bucket*>(deref_(cur));
                  const int r = bkt_insert_entry<Assign>(b, key.substr(depth), std::forward<VF>(v));
                  if (r >= 0) { count_ += (r == 1); return r == 1; }
                  *slot = bkt_split(cur);  // overflow → radix; re-descend
                  continue;
               }
            // Hoist the dominant descend: node_full child slot via the offset table
            // (no reciprocal, no router_find_slot re-dispatch).
            if (k == K::node_full)
            {
               char* f = static_cast<char*>(deref_(cur));
               if (depth == key.size())
               {
                  if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                  f              = static_cast<char*>(deref_(cur));
                  const bool ins = set_or_update_term<Assign>(reinterpret_cast<router_hdr*>(f), std::forward<VF>(v));
                  count_ += ins;
                  return ins;
               }
               const uint8_t byte = uint8_t(key[depth]);
               packed_ptr*   cs   = reinterpret_cast<packed_ptr*>(f + full_offtab_v<Ptr>[byte]);
               if (!cs->is_null())  // existing child -> descend
               {
                  if constexpr (WideStems) pslot = slot;
                  slot               = cs;
                  assert(pdep_ < 72);
                  path_.ensure(pdep_ + 1);
                  path_[pdep_++] = slot;
                  ++depth;
                  continue;
               }
               if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
               f   = static_cast<char*>(deref_(cur));
               full_set(reinterpret_cast<node_full*>(f), byte,
                        make_value_at(key, depth + 1, std::forward<VF>(v)));  // node_full never overflows
               ++count_;
               return true;
            }
            if constexpr (cow_enabled)
               if (k == K::full_x)
               {
                  // Explicit full router: same shape as node_full but the slot
                  // lives in a (privatized) shared direct line.
                  node_fullx* x = static_cast<node_fullx*>(deref_(cur));
                  if (depth == key.size())
                  {
                     const bool ins = set_or_update_term<Assign>(&x->rh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  unsigned      li, sl;
                  full_index<Ptr>(byte, li, sl);
                  if (x->lines[li].is_null())
                     x->lines[li] = fx_new_line_();  // empty range materializes on demand
                  cline_direct_t<Ptr>* L  = fx_line_(x, li);
                  packed_ptr*          cs = &L->ptr[sl];
                  if (!cs->is_null())
                  {
                     slot           = cs;
                     path_.ensure(pdep_ + 1);
                  path_[pdep_++] = slot;
                     ++depth;
                     continue;
                  }
                  *cs = make_value_at(key, depth + 1, std::forward<VF>(v));
                  ++L->nset;
                  ++x->rh.hdr.nbranch;
                  ++count_;
                  return true;
               }
            if constexpr (cow_enabled)
               if (k == K::cx)
               {
                  node_cx* x = static_cast<node_cx*>(deref_(cur));
                  if (depth == key.size())
                  {
                     const bool ins = set_or_update_term<Assign>(&x->rh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  if (packed_ptr* cs = router_find_slot(x, byte))
                  {
                     slot           = cs;
                     path_.ensure(pdep_ + 1);
                  path_[pdep_++] = slot;
                     ++depth;
                     continue;
                  }
                  const unsigned sg = byte >> (8 - x->lg);
                  if (x->seglines[sg].is_null())  // empty range materializes on demand
                     x->seglines[sg] = cx_new_seg_(x->lg);
                  packed_ptr lf = make_value_at(key, depth + 1, std::forward<VF>(v));
                  if (router_try_set(x, byte, lf))
                  {
                     ++count_;
                     return true;
                  }
                  // Segment line full: widen (cx lg+1 ... full_x), re-anchored at *slot
                  build_guard glf{this, &lf};
                  grow_router_(slot, cur, byte, lf);
                  glf.release();
                  ++count_;
                  return true;
               }
            if constexpr (cb_enabled)
               if (k == K::cbucket)
               {
                  if constexpr (std::is_same_v<std::decay_t<VF>, T>)
                  {
                     cb_hdr*        h  = static_cast<cb_hdr*>(deref_(cur));
                     const unsigned pl = h->rh.hdr.prefix_len;
                     if (pl)
                     {
                        const size_t c =
                            lcp(std::string_view(reinterpret_cast<char*>(h->prefix), pl),
                                key.substr(depth));
                        if (c < pl)
                        {  // prefix diverges: burst; the setlist split path takes over
                           *slot = cb_split_(cur);
                           continue;
                        }
                        depth += pl;
                     }
                     if (depth == key.size())
                     {
                        const bool ins = set_or_update_term<Assign>(&h->rh, std::forward<VF>(v));
                        count_ += ins;
                        return ins;
                     }
                     const auto bytes = v.to_bytes();
                     const int  r = cb_insert_<Assign>(h, key.substr(depth),
                                                       std::string_view(bytes.data(), bytes.size()));
                     if (r >= 0)
                     {
                        count_ += (r == 1);
                        return r == 1;
                     }
                     // Full: burst and retry. The burst re-encodes the bucket's
                     // prefix in a prefix_node wrapper, so the re-descent must
                     // start BEFORE the prefix we just consumed — without the
                     // rewind the key lands pl bytes off (find-unreachable).
                     depth -= pl;
                     *slot = cb_split_(cur);
                     continue;
                  }
                  else
                  {
                     assert(false && "cbucket paths take a materialized T");
                     __builtin_unreachable();
                  }
               }
            switch (k)
            {
               case K::value_ptr:
               case K::value_inline:
               {
                  if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }  // leaf update / split: the leaf and its parents
                  if constexpr (cb_create)
                     if (k == K::value_ptr && std::is_same_v<std::decay_t<VF>, T>)
                     {
                        // collision under COW: collect both entries into a fresh
                        // composed bucket when both fit sub-page cells
                        char*            L  = static_cast<char*>(deref_(cur));
                        std::string_view ls(reinterpret_cast<const char*>(leaf_suf(L)),
                                            leaf_slen(L));
                        std::string_view ks = key.substr(depth);
                        if (ls != ks)
                        {
                           const auto       nb = v.to_bytes();
                           const T*         ov = leaf_val(L);
                           const auto       ob = ov->to_bytes();
                           const size_t     c  = lcp(ls, ks);
                           if (c <= 36 && 3 + (ls.size() - c) + ob.size() <= CB_CELLMAX &&
                               3 + (ks.size() - c) + nb.size() <= CB_CELLMAX)
                           {
                              std::string_view pfx = ks.substr(0, c);
                              Ptr nbk = cb_make2_(pfx, ls.substr(c),
                                                  std::string_view(ob.data(), ob.size()),
                                                  ks.substr(c),
                                                  std::string_view(nb.data(), nb.size()));
                              *slot = nbk;
                              node_free_(L, leaf_alloc_size(L));  // exclusive (privatized)
                              ++count_;
                              return true;
                           }
                        }
                     }
                  bool inserted;
                  if constexpr (Adaptive)
                     if (lowfan >= 2)  // deep-narrow leaf collision → bucket
                     {
                        *slot = collapse_to_bucket<Assign>(cur, key, depth, std::forward<VF>(v), inserted);
                        count_ += inserted;
                        return inserted;
                     }
                  *slot = split_leaf<Assign>(cur, key, depth, std::forward<VF>(v), inserted);
                  // Wide-stems: a split just created a router under the parent — if the parent
                  // is now a sparse 2-level chain, fuse it into one wide u16 node.
                  if constexpr (WideStems)
                     if (inserted && pslot) try_fuse_parent(pslot);
                  count_ += inserted;
                  return inserted;
               }
               case K::prefix_node:
               {
                  char*            P = static_cast<char*>(deref_(cur));
                  std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
                  const size_t     c = lcp(PP, key.substr(depth));
                  if (c == PP.size())  // whole prefix matched -> descend into next
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot           = reinterpret_cast<packed_ptr*>(P + 2);  // the next-pointer slot
                     assert(pdep_ < 72);
                     path_.ensure(pdep_ + 1);
                  path_[pdep_++] = slot;
                     depth += PP.size();
                     continue;
                  }
                  if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }  // split mutates this node in place
                  *slot = split_prefix(cur, key, depth, c, std::forward<VF>(v), [&](auto&& x) {
                     return make_value_at(key, depth + c + 1, std::forward<decltype(x)>(x));
                  });
                  ++count_;  // a diverging prefix always inserts
                  return true;
               }
               case K::setlist_u8:  // the only tier with an INLINE prefix
               {
                  setlist*       s  = static_cast<setlist*>(deref_(cur));
                  const unsigned pl = s->rh.hdr.prefix_len;
                  if (pl)
                  {
                     const size_t c = lcp(std::string_view(reinterpret_cast<char*>(s->prefix), pl),
                                          key.substr(depth));
                     if (c < pl)  // inline prefix diverges -> split it
                     {
                        if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }  // split re-prefixes this node in place
                        if constexpr (inline_children_ok)
                        {
                           setlist* sp_ = static_cast<setlist*>(deref_(cur));
                           if (sl_inl_active(sp_)) sl_externalize_all_(sp_);
                        }
                        *slot = split_setlist_prefix(cur, key, depth, c, std::forward<VF>(v), [&](auto&& x) {
                           return make_value_at(key, depth + c + 1, std::forward<decltype(x)>(x));
                        });
                        ++count_;
                        return true;
                     }
                     depth += pl;  // whole inline prefix consumed
                  }
                  if (depth == key.size())
                  {
                     if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                     s              = static_cast<setlist*>(deref_(cur));
                     const bool ins = set_or_update_term<Assign>(&s->rh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  if (int i = setlist_index(s, byte); i >= 0)  // existing child -> descend
                  {
                     if constexpr (inline_children_ok)
                        if (sl_is_inline(s, i))
                        {  // the leaf to mutate rides this line — own it, then handle in place
                           if constexpr (Shared)
                           {
                              slot = own_suffix_(path_, pdep_, fshared_);
                              cur  = *slot;
                              s    = static_cast<setlist*>(deref_(cur));
                           }
                           bool ins = upsert_inline_<Assign>(s, i, key, depth, std::forward<VF>(v));
                           count_ += ins;
                           return ins;
                        }
                     if constexpr (WideStems) pslot = slot;
                     slot           = &s->branch[i];
                     assert(pdep_ < 72);
                     path_.ensure(pdep_ + 1);
                  path_[pdep_++] = slot;
                     ++depth;
                     continue;
                  }
                  if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                  s   = static_cast<setlist*>(deref_(cur));
                  // split-only inline: dissolve inline payloads so setlist_set can write the new
                  // slot (re-inlined at the next split). grow-keeps-inline (sl_try_inline_add_)
                  // is implemented but DEFERRED — it still double-frees under the heavy COW
                  // workload (in-place inline mutation vs supersession/repack); needs focused
                  // debugging before enabling.
                  if constexpr (inline_children_ok)
                     if (sl_inl_active(s)) sl_externalize_all_(s);
                  packed_ptr lf = make_value_at(key, depth + 1, std::forward<VF>(v));
                  if (setlist_set(s, byte, lf))  // room: in place (prefix untouched)
                  {
                     ++count_;
                     return true;
                  }
                  // full -> grow through the widen ladder, re-anchoring *slot each step and
                  // preserving the inline prefix (both handled by grow_router_).
                  build_guard glf{this, &lf};  // widen() can throw; don't orphan the new value
                  grow_router_(slot, cur, byte, lf);
                  glf.release();  // lf is now linked into the widened node
                  ++count_;
                  return true;
               }
               case K::setlist_u16:  // wide router: 2-byte hop, re-stride to u8 when it can't fit
               {
                  setlist16* s = static_cast<setlist16*>(deref_(cur));
                  if (key.size() - depth < 2)
                  {
                     if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                     s   = static_cast<setlist16*>(deref_(cur));
                     if (depth == key.size())  // key ends exactly here → term
                     {
                        const bool ins = set_or_update_term<Assign>(&s->rh, std::forward<VF>(v));
                        count_ += ins;
                        return ins;
                     }
                     *slot = restride_u16_to_u8(cur);  // 1 byte left: can't form a stem → re-stride, retry
                     continue;
                  }
                  const uint16_t q = two_stem(key, depth);
                  if (packed_ptr* cs = sl16_find_slot(s, q))  // existing stem → descend 2 bytes
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot           = cs;
                     assert(pdep_ < 72);
                     path_.ensure(pdep_ + 1);
                  path_[pdep_++] = slot;
                     depth += 2;
                     continue;
                  }
                  if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                  s   = static_cast<setlist16*>(deref_(cur));
                  if (s->rh.hdr.nbranch >= setlist16::CAP)  // full → re-stride, retry (v untouched)
                  {
                     *slot = restride_u16_to_u8(cur);
                     continue;
                  }
                  sl16_set(s, q, make_value_at(key, depth + 2, std::forward<VF>(v)));  // room: in place
                  ++count_;
                  return true;
               }
               case K::hdr:  // prefixed wide router (see-header, CL0 prefix)
               {
                  char*          hp  = static_cast<char*>(deref_(cur));
                  router_hdr*    hrh = reinterpret_cast<router_hdr*>(hp);
                  const unsigned hpl = hrh->hdr.prefix_len;
                  const size_t   hc =
                      lcp(std::string_view(hp + WPFX_OFF, hpl), key.substr(depth));
                  if (hc < hpl)  // diverges inside the wide prefix -> split
                  {
                     if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }  // split shortens this node's prefix in place
                     *slot = split_hdr_prefix_(cur, key, depth, hc, std::forward<VF>(v));
                     ++count_;
                     return true;
                  }
                  depth += hpl;
                  if (depth == key.size())
                  {
                     if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                     hrh            = reinterpret_cast<router_hdr*>(deref_(cur));
                     const bool ins = set_or_update_term<Assign>(hrh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t hbyte = uint8_t(key[depth]);
                  if (packed_ptr* hcs = router_find_slot(cur, hbyte))
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot           = hcs;
                     assert(pdep_ < 72);
                     path_.ensure(pdep_ + 1);
                  path_[pdep_++] = slot;
                     ++depth;
                     continue;
                  }
                  if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                  packed_ptr hlf = make_value_at(key, depth + 1, std::forward<VF>(v));
                  build_guard hg{this, &hlf};
                  grow_router_(slot, cur, hbyte, hlf);  // try_set strips hdr; widen carries the prefix
                  hg.release();
                  ++count_;
                  return true;
               }
               case K::c2:  // dense tiers: CL0 prefix rides the hdr tag when present
               case K::c4:
               case K::c8:
               {
                  router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
                  if (depth == key.size())
                  {
                     if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                     rh             = static_cast<router_hdr*>(deref_(cur));
                     const bool ins = set_or_update_term<Assign>(rh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  if (packed_ptr* cs = router_find_slot(cur, byte))  // existing child -> descend
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot           = cs;
                     assert(pdep_ < 72);
                     path_.ensure(pdep_ + 1);
                  path_[pdep_++] = slot;
                     ++depth;
                     continue;
                  }
                  if constexpr (Shared)
                  {
                     slot = own_suffix_(path_, pdep_, fshared_);
                     cur  = *slot;
                  }
                  packed_ptr lf = make_value_at(key, depth + 1, std::forward<VF>(v));
                  build_guard glf{this, &lf};  // grow can throw; don't orphan the new value
                  grow_router_(slot, cur, byte, lf);  // re-anchors *slot across the widen ladder
                  glf.release();
                  ++count_;
                  return true;
               }
               // exhaustive (no default): node_full/bucket descend before the switch and
               // null inserts at the top of the loop, so these can't reach it — but a NEW
               // kind added to K must fail compilation here and get an explicit decision.
               case K::node_full: case K::full_x: case K::cx: case K::bucket:
               case K::cbucket: case K::null:
                  assert(false && "handled before the switch");
                  __builtin_unreachable();
            }
         }
      }

     public:
      // ── descend_read<Op>: one shared read-descent skeleton ──────────────────────
      // The hoisted switch (node_full / value_ptr-leaf / setlist hoisted ahead of the jump
      // table; tuned by measurement). The router hops are identical for every read; the Op
      // supplies only the terminal/miss actions and the return type, so find/locate_/… are
      // thin Ops over ONE loop — compile-time (template) dispatch, fully inlined, no vtable.
      // Op interface: leaf(p,rem,rlen) inl(raw,at_end) bkt(b,rem,rlen) term(rh) miss().
#define HKV_PTR(R)  deref_raw_(R)
#define HKV_NEXT(ND) { depth = (ND); continue; }
#define HKV_BODY_VALUE_PTR    return op.leaf(HKV_PTR(raw), kp + depth, klen - depth);
#define HKV_BODY_VALUE_INLINE return op.inl(raw, depth == klen);
#define HKV_BODY_BUCKET       return op.bkt(reinterpret_cast<const bucket*>(HKV_PTR(raw)), kp + depth, klen - depth);
#define HKV_BODY_PREFIX { \
   const char* p = HKV_PTR(raw); const uint16_t pl = pn_plen(p); \
   if (klen - depth < pl || std::memcmp(pn_pfx<Ptr>(p), kp + depth, pl) != 0) return op.miss(); \
   std::memcpy(&raw, p + 2, sizeof(Ptr)); HKV_NEXT(depth + pl); }
#define HKV_BODY_SETLIST { \
   const setlist* s = reinterpret_cast<const setlist*>(HKV_PTR(raw)); const unsigned pl = s->rh.hdr.prefix_len; \
   if (pl) { if (klen - depth < pl || std::memcmp(s->prefix, kp + depth, pl) != 0) return op.miss(); depth += pl; } \
   if (depth == klen) return op.term(&s->rh); \
   const int sli_ = setlist_index(s, uint8_t(kp[depth])); \
   if (sli_ < 0) return op.miss(); \
   if constexpr (inline_children_ok) if (sl_is_inline(s, sli_)) /* leaf rides this line: resolve here */ \
      return op.leaf(sl_inline_leaf(s, sli_), kp + depth + 1, klen - depth - 1); \
   raw = s->branch[sli_].raw(); HKV_NEXT(depth + 1); }
#define HKV_BODY_SETLIST16 { \
   const setlist16* s = reinterpret_cast<const setlist16*>(HKV_PTR(raw)); \
   if (klen - depth < 2) return depth == klen ? op.term(&s->rh) : op.miss(); \
   const uint16_t q = uint16_t(uint16_t(uint8_t(kp[depth])) << 8 | uint8_t(kp[depth + 1])); \
   raw = sl16_find(s, q).raw(); HKV_NEXT(depth + 2); }
#define HKV_BODY_FULL { \
   const char* p = HKV_PTR(raw); \
   if (depth == klen) return op.term(reinterpret_cast<const router_hdr*>(p)); \
   std::memcpy(&raw, p + full_offtab_v<Ptr>[uint8_t(kp[depth])], sizeof(Ptr)); HKV_NEXT(depth + 1); }
#define HKV_BODY_FULLX { \
   const char* p = HKV_PTR(raw); \
   if (depth == klen) return op.term(reinterpret_cast<const router_hdr*>(p)); \
   unsigned li_, sl_; full_index<Ptr>(uint8_t(kp[depth]), li_, sl_); \
   uint64_t lraw = 0; std::memcpy(&lraw, p + sizeof(router_hdr) + li_ * sizeof(Ptr), sizeof(Ptr)); \
   { Ptr lp_; std::memcpy(&lp_, &lraw, sizeof(Ptr)); if (lp_.is_null()) return op.miss(); } \
   std::memcpy(&raw, HKV_PTR(lraw) + sl_ * sizeof(Ptr), sizeof(Ptr)); HKV_NEXT(depth + 1); }
#define HKV_BODY_CBUCKET { \
   if constexpr (cb_enabled) { \
   const cb_hdr* h = reinterpret_cast<const cb_hdr*>(HKV_PTR(raw)); \
   const unsigned pl = h->rh.hdr.prefix_len; \
   if (pl) { if (klen - depth < pl || std::memcmp(h->prefix, kp + depth, pl) != 0) return op.miss(); depth += pl; } \
   if (depth == klen) return op.term(&h->rh); \
   const uint8_t* c = cb_find_(h, std::string_view(kp + depth, klen - depth)); \
   return c ? op.cbkt(c) : op.miss(); \
   } else return op.miss(); }
#define HKV_BODY_CX { \
   const char* p = HKV_PTR(raw); \
   if (depth == klen) return op.term(reinterpret_cast<const router_hdr*>(p)); \
   const unsigned lg_ = uint8_t(p[sizeof(router_hdr)]); \
   const uint8_t  by_ = uint8_t(kp[depth]); \
   const unsigned sg_ = by_ >> (8 - lg_); \
   const unsigned d_  = by_ & ((256u >> lg_) - 1); \
   uint64_t lraw = 0; std::memcpy(&lraw, p + sizeof(router_hdr) + 2 + sg_ * sizeof(Ptr), sizeof(Ptr)); \
   { Ptr lp_; std::memcpy(&lp_, &lraw, sizeof(Ptr)); if (lp_.is_null()) return op.miss(); } \
   const uint8_t* m_ = reinterpret_cast<const uint8_t*>(HKV_PTR(lraw)); \
   if (!bm_test(m_, int(d_))) return op.miss(); \
   std::memcpy(&raw, m_ + ((256u >> lg_) / 8) + unsigned(bm_rank(m_, int(d_))) * sizeof(Ptr), sizeof(Ptr)); \
   HKV_NEXT(depth + 1); }
#define HKV_BODY_HDR { \
   const char* p = HKV_PTR(raw); \
   const unsigned pl = reinterpret_cast<const router_hdr*>(p)->hdr.prefix_len; \
   if (klen - depth < pl || std::memcmp(p + WPFX_OFF, kp + depth, pl) != 0) return op.miss(); \
   raw = (raw & ~uint64_t(0xF)) | uint64_t(uint8_t(p[6]) & 0xF); \
   HKV_NEXT(depth + pl); }
#define HKV_BODY_CN(N) { \
   const char* p = HKV_PTR(raw); \
   if (depth == klen) return op.term(reinterpret_cast<const router_hdr*>(p)); \
   raw = cnode_step(reinterpret_cast<const cnode<N>*>(p), uint8_t(kp[depth])).raw(); HKV_NEXT(depth + 1); }

      template <class Op>
      [[gnu::always_inline]] auto descend_read(std::string_view key, Op op) const -> decltype(op.miss())
      {
         uint64_t raw = 0;
         std::memcpy(&raw, root_.b, sizeof(Ptr));
         const char* const kp   = key.data();
         const size_t      klen = key.size();
         size_t            depth = 0;
         for (;;)
         {
            const unsigned kk = unsigned(raw & 0xF);
            if (kk == unsigned(K::node_full)) HKV_BODY_FULL
            if constexpr (!inlineable) { if (kk == unsigned(K::value_ptr)) HKV_BODY_VALUE_PTR }
            if (kk == unsigned(K::setlist_u8)) HKV_BODY_SETLIST
            if (kk > K_max) return op.miss();
            switch (static_cast<K>(kk))  // exhaustive, no default → -Wswitch guards a new K
            {
               case K::value_ptr:    HKV_BODY_VALUE_PTR
               case K::value_inline: HKV_BODY_VALUE_INLINE
               case K::prefix_node:  HKV_BODY_PREFIX
               case K::setlist_u8:   HKV_BODY_SETLIST
               case K::node_full:    HKV_BODY_FULL
               case K::hdr:          HKV_BODY_HDR
               case K::c2:           HKV_BODY_CN(2)
               case K::c4:           HKV_BODY_CN(4)
               case K::c8:           HKV_BODY_CN(8)
               case K::bucket:       HKV_BODY_BUCKET
               case K::setlist_u16:  HKV_BODY_SETLIST16
               case K::full_x:       HKV_BODY_FULLX
               case K::cx:           HKV_BODY_CX
               case K::cbucket:      HKV_BODY_CBUCKET
               case K::null:         return op.miss();
            }
            __builtin_unreachable();
         }
      }
#undef HKV_NEXT
#undef HKV_PTR
#undef HKV_BODY_VALUE_PTR
#undef HKV_BODY_VALUE_INLINE
#undef HKV_BODY_PREFIX
#undef HKV_BODY_SETLIST
#undef HKV_BODY_SETLIST16
#undef HKV_BODY_FULL
#undef HKV_BODY_CBUCKET
#undef HKV_BODY_CX
#undef HKV_BODY_FULLX
#undef HKV_BODY_CN
#undef HKV_BODY_BUCKET

      // Copy-out lookup: writes the value into `out` and returns true if present.
      // (Copy-out, not pointer, because a small value can live INLINE in the pointer bits.)
      bool find(key_param k, T& out) const
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         struct op_t
         {
            const kv_tree* self;
            T*             out;
            bool leaf(const char* p, const char* rem, size_t rl) const
            {
               if (std::string_view(reinterpret_cast<const char*>(leaf_suf(p)), leaf_slen(p)) !=
                   std::string_view(rem, rl))
                  return false;
               if constexpr (raw_vals)
                  *out = T{leaf_bytes(p)};  // VIEW into the leaf: zero copy
               else
                  self->copy_value(*out, leaf_val(p));
               return true;
            }
            bool inl(uint64_t raw, bool at_end) const
            {
               if (!at_end) return false;
               const uint64_t shifted = raw >> 8;
               std::memcpy(out, &shifted, inline_bytes);
               return true;
            }
            bool bkt(const bucket* b, const char* rem, size_t rl) const
            {
               return self->bkt_find(b, std::string_view(rem, rl), *out);
            }
            bool term(const router_hdr* rh) const { return self->read_term(rh, *out); }
            bool miss() const { return false; }
            bool cbkt(const uint8_t* c) const
            {
               if constexpr (cb_enabled)
               {
                  *out = T::from_bytes(cb_cell{const_cast<uint8_t*>(c)}.value());
                  return true;
               }
               else
                  return false;
            }
         };
         return descend_read(key, op_t{this, &out});
      }
      // Convenience: copy-out as an optional (hkv extension; STL find() below returns an
      // iterator). Named distinctly to avoid clashing with the iterator-returning find.
      std::optional<T> find_opt(key_param k) const
      {
         T v;
         return find(k, v) ? std::optional<T>(std::move(v)) : std::nullopt;
      }
      // Write-stamp of the key's LEAF (the version that last wrote it) — the
      // retire-if-unchanged primitive (L5: retire only keys with stamp <= F).
      // UINT32_MAX = unknown-or-missing (inline/bucket terminals carry no own
      // stamp): callers must treat it as "do not retire" — conservative-safe.
      static constexpr uint32_t no_stamp = UINT32_MAX;
      uint32_t write_stamp(key_param k) const
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         struct op_t
         {
            const kv_tree* self;
            uint32_t leaf(const char* p, const char* rem, size_t rl) const
            {
               if (std::string_view(reinterpret_cast<const char*>(leaf_suf(p)), leaf_slen(p)) !=
                   std::string_view(rem, rl))
                  return no_stamp;
               if constexpr (k_has_vstamp_(K::value_ptr))
               {
                  uint32_t v;
                  std::memcpy(&v, p + vstamp_off_(K::value_ptr), 4);
                  return v;
               }
               return no_stamp;
            }
            uint32_t inl(uint64_t, bool) const { return no_stamp; }
            uint32_t bkt(const bucket*, const char*, size_t) const { return no_stamp; }
            uint32_t term(const router_hdr* rh) const
            {
               if (!rh->term.is_null() && rh->term.tag() == K::value_ptr)
                  return self->vstamp_get_(rh->term);
               return no_stamp;
            }
            uint32_t miss() const { return no_stamp; }
            uint32_t cbkt(const uint8_t*) const { return no_stamp; }
         };
         return descend_read(key, op_t{this});
      }

      // Presence test WITHOUT reading the value — find() copies it out (a real cost for
      // allocating T like std::string), a membership test never needs to.
      bool contains(key_param k) const
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         struct op_t
         {
            bool leaf(const char* p, const char* rem, size_t rl) const
            {
               return std::string_view(reinterpret_cast<const char*>(leaf_suf(p)), leaf_slen(p)) ==
                      std::string_view(rem, rl);
            }
            bool inl(uint64_t, bool at_end) const { return at_end; }
            bool bkt(const bucket* b, const char* rem, size_t rl) const
            {
               return bkt_locate(b, std::string_view(rem, rl)) != nullptr;
            }
            bool term(const router_hdr* rh) const { return !rh->term.is_null(); }
            bool miss() const { return false; }
            bool cbkt(const uint8_t*) const { return true; }
         };
         return descend_read(key, op_t{});
      }

      // ── upsert / update (STL-ish value-write semantics) ─────────────────────────
      // upsert = insert-or-assign: insert if absent, overwrite if present. Returns true
      // iff NEWLY inserted. (hkv's insert already has this behavior; these are the
      // explicit/STL names.)
      bool upsert(key_param k, const T& v) { return insert(k, v); }
      bool upsert(key_param k, T&& v) { return insert(k, std::move(v)); }
      bool insert_or_assign(key_param k, const T& v) { return insert(k, v); }
      bool insert_or_assign(key_param k, T&& v) { return insert(k, std::move(v)); }

      // update = assign ONLY if the key already exists; never inserts. Returns true iff
      // the key was present (and its value overwritten). A single read-style descent —
      // no node creation, no split, no growth.
      bool update(key_param k, const T& v)
      {
         T t(v);
         return update(k, std::move(t));
      }
      // ── descend_mut<Op>: shared MUTATING descent (slot + parent tracking) ────────
      // Like descend_read but threads the writable slot* and the parent/edge so the Op can
      // overwrite or unlink at the terminal. update/remove are thin Ops over this one loop.
      // (insert is NOT here: it restructures mid-descent at every node — split/widen/fuse —
      // so it isn't a terminal visitor.) Op: leaf/inl/term/bkt/miss; exhaustive switch(K).
      template <class Op>
      [[gnu::always_inline]] auto descend_mut(std::string_view key, Op op) -> decltype(op.miss())
      {
         packed_ptr* slot   = &root_;
         packed_ptr  parent = packed_ptr::null();
         uint8_t     pbyte  = 0;
         uint16_t    pstem  = 0;
         size_t      depth  = 0;
         for (;;)
         {
            packed_ptr cur = *slot;
            switch (cur.tag())
            {
               case K::null:         return op.miss();
               case K::value_ptr:
                  return op.leaf(slot, static_cast<char*>(deref_(cur)), key.substr(depth), parent, pbyte, pstem);
               case K::value_inline: return op.inl(slot, depth == key.size(), parent, pbyte, pstem);
               case K::bucket:       return op.bkt(slot, cur, key.substr(depth));
               case K::cbucket:      return op.miss();  // at()/[] unsupported over cbuckets
               case K::prefix_node:
               {
                  char*          P  = static_cast<char*>(deref_(cur));
                  const uint16_t pl = pn_plen(P);
                  if (key.size() - depth < pl || std::memcmp(pn_pfx<Ptr>(P), key.data() + depth, pl) != 0)
                     return op.miss();
                  parent = cur; slot = reinterpret_cast<packed_ptr*>(P + 2); depth += pl; continue;
               }
               case K::setlist_u16:
               {
                  setlist16* s = static_cast<setlist16*>(deref_(cur));
                  if (key.size() - depth < 2) return depth == key.size() ? op.term(&s->rh) : op.miss();
                  const uint16_t q  = two_stem(key, depth);
                  packed_ptr*    cs = sl16_find_slot(s, q);
                  if (!cs) return op.miss();
                  parent = cur; pstem = q; slot = cs; depth += 2; continue;
               }
               case K::setlist_u8: case K::c2: case K::c4: case K::c8: case K::node_full:
               case K::hdr:
               {
                  router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
                  if (const K rk = cur.tag(); rk == K::setlist_u8 || rk == K::hdr)
                  {
                     const unsigned pl = rh->hdr.prefix_len;
                     if (pl)
                     {
                        const char* pb =
                            rk == K::setlist_u8
                                ? reinterpret_cast<const char*>(
                                      static_cast<const setlist*>(deref_(cur))->prefix)
                                : static_cast<const char*>(deref_(cur)) + WPFX_OFF;
                        if (key.size() - depth < pl || std::memcmp(pb, key.data() + depth, pl) != 0)
                           return op.miss();
                        depth += pl;
                     }
                  }
                  if (depth == key.size()) return op.term(rh);
                  const uint8_t byte = uint8_t(key[depth]);
                  packed_ptr*   cs   = router_find_slot(cur, byte);
                  if (!cs) return op.miss();
                  parent = cur; pbyte = byte; slot = cs; ++depth; continue;
               }
            }
         }
      }

      bool update(key_param k, T&& v)
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         struct op_t
         {
            kv_tree* self;
            T*       v;
            bool assign(T* p) const  // overwrite an existing value object in place
            {
               v_destroy(self->alloc_, p);
               v_construct(self->alloc_, p, std::move(*v));
               return true;
            }
            bool leaf(packed_ptr*, char* L, std::string_view rem, packed_ptr, uint8_t, uint16_t) const
            {
               if (std::string_view(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L)) != rem)
                  return false;
               return assign(leaf_val(L));
            }
            bool inl(packed_ptr* slot, bool at_end, packed_ptr, uint8_t, uint16_t) const
            {
               if (!at_end) return false;
               if constexpr (inlineable) *slot = pack_inline(*v);
               return true;
            }
            bool term(router_hdr* rh) const
            {
               if (rh->term.is_null()) return false;
               if constexpr (inlineable)
                  if (rh->term.tag() == K::value_inline) { rh->term = pack_inline(*v); return true; }
               return assign(leaf_val(static_cast<char*>(self->deref_(rh->term))));
            }
            bool bkt(packed_ptr*, packed_ptr cur, std::string_view rem) const
            {
               if constexpr (bucketable)
                  if (const char* e = bkt_locate(static_cast<bucket*>(self->deref_(cur)), rem))
                     return assign(bkt_val(const_cast<char*>(e)));
               return false;
            }
            bool miss() const { return false; }
         };
         return descend_mut(key, op_t{this, &v});
      }

      // ── element access by reference (non-inline T only) ─────────────────────────
      // For non-inlineable T every value is a real, aligned, constructed object in a
      // leaf or bucket entry, so these hand out genuine T& into the tree. The reference
      // is valid until the next insert/upsert/update/remove that restructures the
      // affected path (leaf split, bucket compact/rebuild/split) — a std::vector-like
      // invalidation rule. (Inlineable small-POD T has no addressable object; use the
      // copy-out find() for those — these overloads are constrained off.)
      T* find_ptr(key_param k)
         requires(!inlineable)
      {
         scratch_t ks_;
         return locate_(enc_(k, ks_));
      }
      const T* find_ptr(key_param k) const
         requires(!inlineable)
      {
         scratch_t ks_;
         return locate_(enc_(k, ks_));
      }
      T& at(key_param k)
         requires(!inlineable)
      {
         scratch_t ks_;
         if (T* p = locate_(enc_(k, ks_))) return *p;
         throw std::out_of_range("start_ver::hkv::kv_tree::at: key not found");
      }
      const T& at(key_param k) const
         requires(!inlineable)
      {
         scratch_t ks_;
         if (const T* p = locate_(enc_(k, ks_))) return *p;
         throw std::out_of_range("start_ver::hkv::kv_tree::at: key not found");
      }
      // operator[]: returns the value, inserting a default-constructed one if absent.
      T& operator[](key_param k)
         requires(!inlineable && std::is_default_constructible_v<T>)
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         if (T* p = locate_(key)) return *p;
         insert(k, T{});                 // default-insert (may restructure)
         return *locate_(key);    // re-descend to the now-stable slot (ks_ still valid)
      }

     private:
      // A grown-once byte buffer for assembling keys during a scan/iteration — psitri's
      // _key_buf idea: write a byte / set the length back, no std::string (no NUL writes,
      // no SSO branch) and no per-step allocation. Reallocates only when a key runs deeper
      // than anything seen so far (geometric growth), so steady-state writes are bare
      // stores. view() hands out a string_view over the live bytes.
      struct key_buf
      {
         std::vector<char> b;  // sized on first use — an unread key buffer costs nothing
         size_t            len = 0;
         void ensure(size_t need)
         {
            if (need > b.size())
            {
               size_t cap = b.size() ? b.size() : 256;
               while (cap < need) cap <<= 1;
               b.resize(cap);
            }
         }
         void push(uint8_t c)
         {
            ensure(len + 1);
            b[len++] = char(c);
         }
         void append(const void* p, size_t n)
         {
            ensure(len + n);
            char*       d = b.data() + len;
            const char* s = static_cast<const char*>(p);
            len += n;
            // Suffixes/prefixes are short; inlined byte stores beat a runtime-size memcpy
            // call. Fall back to memcpy only for the rare long copy (long shared prefix).
            if (n > 16) { std::memcpy(d, s, n); return; }
            for (size_t i = 0; i < n; ++i) d[i] = s[i];
         }
         std::string_view view() const noexcept { return std::string_view(b.data(), len); }
      };

      // ── ordered forward iterator ────────────────────────────────────────────────
      // Yields (key, value) in ascending lexicographic key order. Keys aren't stored
      // whole — they're rebuilt along the descent path into key_ (a key_buf: push
      // prefix/byte on the way down, set the length back on backtrack — no std::string
      // churn). key() / *it return a string_view over that buffer. Bucket entries are
      // sorted by full suffix on entry (the bucket's div index orders only by 2 bytes).
     public:
      class const_iterator
      {
         struct Frame  // node + scan cursor; plen/cbase used only once key maintenance is on
         {
            packed_ptr node;
            int        cur;    // router: -1 term-pending then next-byte 0..256; prefix: 0/1; bucket: entry#
            uint32_t   plen;   // key_ length on entry (backtrack point)
            uint32_t   cbase;  // key_ length past this router's inline prefix (term/child base)
         };
         // Descent stack with inline storage: begin()/bound seeks allocate NOTHING for paths
         // up to INLINE frames (each frame consumes >= 1 key byte, so that is every practical
         // tree); deeper long-key paths spill to the heap transparently. Frames are trivially
         // copyable, so copy/move/grow are memcpy.
         struct frame_stack
         {
            static constexpr uint32_t INLINE = 16;
            static_assert(std::is_trivially_copyable_v<Frame>);
            Frame    sbo_[INLINE];
            Frame*   d_   = sbo_;
            uint32_t n_   = 0;
            uint32_t cap_ = INLINE;

            frame_stack() = default;
            frame_stack(const frame_stack& o) { copy_(o); }
            frame_stack(frame_stack&& o) noexcept { steal_(o); }
            frame_stack& operator=(const frame_stack& o)
            {
               if (this != &o) { release_(); copy_(o); }
               return *this;
            }
            frame_stack& operator=(frame_stack&& o) noexcept
            {
               if (this != &o) { release_(); steal_(o); }
               return *this;
            }
            ~frame_stack() { release_(); }

            bool         empty() const noexcept { return n_ == 0; }
            uint32_t     size() const noexcept { return n_; }
            Frame&       operator[](size_t i) noexcept { return d_[i]; }
            const Frame& operator[](size_t i) const noexcept { return d_[i]; }
            Frame&       back() noexcept { return d_[n_ - 1]; }
            Frame*       begin() noexcept { return d_; }
            Frame*       end() noexcept { return d_ + n_; }
            void         pop_back() noexcept { --n_; }
            void         push_back(const Frame& f)
            {
               if (n_ == cap_) grow_();
               d_[n_++] = f;
            }

           private:
            void grow_()
            {
               const uint32_t ncap = cap_ * 2;
               Frame*         nd   = static_cast<Frame*>(::operator new(ncap * sizeof(Frame)));
               std::memcpy(nd, d_, size_t(n_) * sizeof(Frame));
               release_();
               d_   = nd;
               cap_ = ncap;
            }
            void release_() noexcept
            {
               if (d_ != sbo_) ::operator delete(d_);
            }
            void copy_(const frame_stack& o)
            {
               if (o.n_ > INLINE)
               {
                  d_   = static_cast<Frame*>(::operator new(size_t(o.cap_) * sizeof(Frame)));
                  cap_ = o.cap_;
               }
               else
               {
                  d_   = sbo_;
                  cap_ = INLINE;
               }
               n_ = o.n_;
               std::memcpy(d_, o.d_, size_t(n_) * sizeof(Frame));
            }
            void steal_(frame_stack& o) noexcept
            {
               if (o.d_ != o.sbo_)
               {
                  d_   = o.d_;
                  cap_ = o.cap_;
               }
               else
               {
                  d_   = sbo_;
                  cap_ = INLINE;
                  std::memcpy(sbo_, o.sbo_, size_t(o.n_) * sizeof(Frame));
               }
               n_     = o.n_;
               o.d_   = o.sbo_;
               o.cap_ = INLINE;
               o.n_   = 0;
            }
         };

         const kv_tree*       t_ = nullptr;  // owning tree: deref base for indexed handles
         // composed-bucket entry values are raw bytes, materialized here per
         // yield (raw storage: T's name isn't needed at declaration parse time)
         alignas(alignof(T)) mutable unsigned char cvalbuf_[sizeof(T)];
         mutable frame_stack  stk_;  // mutable: first key() primes plen/cbase
         std::vector<int>           bord_;    // sorted entry order for the CURRENT bucket (top frame)
         // Shadows the tree's deref_ so every handle resolution below routes through t_.
         void* deref_(packed_ptr p) const noexcept { return t_->deref_(p); }
         mutable key_buf            key_;     // current key (maintained once read) / scratch
         mutable bool               maintain_ = false;  // false → ++ skips key work; flips on first key()
         const uint8_t*             sfx_    = nullptr;   // current terminal suffix; null/0 = none
         uint32_t                   sfxlen_ = 0;
         const char*                full_str_ = nullptr;  // set at a full-key leaf → key() is zero-copy
         uint32_t                   full_len_ = 0;
         // Non-inline T: point at the real stored object (no copy → value() is a true
         // reference into the tree). Inline T: keep a local copy (no addressable object).
         std::conditional_t<inlineable, T, const T*> val_{};
         // val_ points into the TREE (stable). Bucket-cell values are materialized
         // into cvalbuf_ instead; a self-pointer would dangle under the defaulted
         // copy/move, so a flag marks "value lives in cvalbuf_" and value() rebases
         // off `this`.
         bool                                        val_local_ = false;
         bool                                        end_       = true;

         void set_ptr_(const T* p)
         {
            val_local_ = false;
            if constexpr (inlineable)
               copy_value(val_, p);
            else
               val_ = p;
         }
         void emit_(packed_ptr p)  // value from a terminal pointer (value_ptr "" / value_inline)
         {
            if constexpr (inlineable)
               if (p.tag() == K::value_inline) { unpack_inline(p, val_); return; }
            if constexpr (kv_tree::raw_vals)
            {
               *reinterpret_cast<T*>(cvalbuf_) =
                   T{leaf_bytes(static_cast<const char*>(deref_(p)))};
               val_local_ = true;
            }
            else
               set_ptr_(leaf_val(static_cast<const char*>(deref_(p))));
         }
         // Position the cursor on a resolved leaf pointer L (suffix view + value into the
         // local buffer / proxy). Shared by descend_/descend_max_ (external leaf via deref_)
         // and the inline paths (leaf rides the parent setlist's line). base = key length at
         // the leaf's start.
         void emit_leaf_(const char* L, size_t base)
         {
            sfx_    = leaf_suf(L);
            sfxlen_ = leaf_slen(L);
            if (leaf_has_full(L)) { full_str_ = leaf_str(L); full_len_ = leaf_strlen(L); }
            else                    full_str_ = nullptr;
            if constexpr (kv_tree::raw_vals)
            {
               *reinterpret_cast<T*>(cvalbuf_) = T{leaf_bytes(L)};
               val_local_ = true;
            }
            else
               set_ptr_(leaf_val(L));
            if (maintain_) { key_.len = base; if (!full_str_) key_.append(sfx_, sfxlen_); }
         }
         // True iff `child` of setlist parent `parent` (reached via `byte`) is an INLINE leaf;
         // when so, *L is set to the in-line leaf pointer (resolved against the parent line).
         bool kid_inline_(packed_ptr parent, uint8_t byte, const char** L) const noexcept
         {
            if constexpr (inline_children_ok)
               if (parent.tag() == K::setlist_u8)
               {
                  const setlist* s = static_cast<const setlist*>(t_->deref_(parent));
                  const int      i = setlist_index(s, byte);
                  if (i >= 0 && kv_tree::sl_is_inline(s, i)) { *L = t_->sl_inline_leaf(s, i); return true; }
               }
            (void)parent; (void)byte; (void)L;
            return false;
         }
         // Forward: position at `child` of `parent` (reached via `byte`). Inline leaf -> emit
         // here (true); otherwise delegate to descend_.
         bool descend_kid_(packed_ptr parent, uint8_t byte, packed_ptr child, size_t base)
         {
            const char* L;
            if (kid_inline_(parent, byte, &L)) { emit_leaf_(L, base); return true; }
            return descend_(child, base);
         }
         // Backward mirror: descend to the rightmost terminal of `child`, or emit its inline leaf.
         void descend_kid_max_(packed_ptr parent, uint8_t byte, packed_ptr child, size_t base)
         {
            const char* L;
            if (kid_inline_(parent, byte, &L)) { emit_leaf_(L, base); return; }
            descend_max_(child, base);
         }
         // Byte-routers only — u16 (2-byte) and non-routers must be handled by the caller
         // BEFORE this (with_router asserts on them; a silent default here is exactly what
         // once hid the lower_bound/u16 miss).
         packed_ptr router_child_(packed_ptr node, uint8_t byte) const
         {
            return t_->with_router(node,
                                   [&](auto* n) noexcept { return t_->router_find_d(n, byte); });
         }
         bool next_child_(packed_ptr node, int from, uint8_t& ob, packed_ptr& oc) const
         {
            if (from > 255)
               return false;
            return t_->with_router(node,
                                   [&](auto* n) noexcept { return t_->router_next_d(n, from, ob, oc); });
         }
         bool prev_child_(packed_ptr node, int from, uint8_t& ob, packed_ptr& oc) const
         {
            // tree's router_prev contract: largest byte STRICTLY BELOW `from`
            // (from=256 -> max); donor's <=from maps to strictly-below(from+1).
            if (from < 0)
               return false;
            return t_->router_prev_d(node, from + 1, ob, oc);
         }
         // Descend to the RIGHTMOST terminal of `child` (mirror of descend_+seek_).
         void descend_max_(packed_ptr child, size_t base)
         {
            for (;;)
            {
               switch (child.tag())
               {
                  case K::value_ptr:
                     emit_leaf_(static_cast<const char*>(deref_(child)), base);
                     return;
                  case K::value_inline:
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     val_local_ = false;
                     if constexpr (inlineable)
                        unpack_inline(child, val_);
                     if (maintain_) key_.len = base;
                     return;
                  case K::prefix_node:
                  {
                     const char* P  = static_cast<const char*>(deref_(child));
                     uint32_t    cb = uint32_t(base);
                     if (maintain_) { key_.len = base; key_.append(pn_pfx<Ptr>(P), pn_plen(P)); cb = uint32_t(key_.len); }
                     stk_.push_back(Frame{child, 1, uint32_t(base), cb});  // single child: consumed
                     child = pn_next<Ptr>(const_cast<char*>(P));
                     base  = cb;
                     continue;
                  }
                  case K::bucket:
                  {
                     bucket* b = static_cast<bucket*>(deref_(child));
                     bord_.clear();
                     bkt_append_order(b, bord_);
                     const int n = int(bord_.size());
                     stk_.push_back(Frame{child, n, uint32_t(base), uint32_t(base)});
                     const char* e = bkt_entry(b, bord_[size_t(n - 1)]);
                     sfx_      = bkt_suf(e);
                     sfxlen_   = bkt_slen(e);
                     full_str_ = nullptr;
                     set_ptr_(bkt_val(e));
                     if (maintain_) { key_.len = base; key_.append(sfx_, sfxlen_); }
                     return;
                  }
                  case K::setlist_u16:
                  {
                     const setlist16* s  = static_cast<const setlist16*>(deref_(child));
                     const int        nb = s->rh.hdr.nbranch;
                     uint32_t         cb = uint32_t(base);
                     if (maintain_) { key_.len = base; cb = uint32_t(key_.len); }
                     if (nb > 0)
                     {
                        stk_.push_back(Frame{child, nb, uint32_t(base), cb});
                        const uint16_t st = s->stems[nb - 1];
                        if (maintain_)
                        { key_.push(uint8_t(st >> 8)); key_.push(uint8_t(st & 0xFF)); }
                        child = s->branch[nb - 1];
                        base  = maintain_ ? size_t(key_.len) : size_t(cb) + 2;
                        continue;
                     }
                     stk_.push_back(Frame{child, 0, uint32_t(base), cb});  // term-only router
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(s->rh.term);
                     if (maintain_) key_.len = cb;
                     return;
                  }
                  default:  // byte routers (setlist / c2 / c4 / c8 / node_full / fullx / cx / hdr)
                  {
                     const router_hdr* rh = static_cast<const router_hdr*>(deref_(child));
                     uint32_t          cb = uint32_t(base);
                     if (maintain_)
                     {
                        key_.len = base;
                        if (child.tag() == K::setlist_u8 && rh->hdr.prefix_len)
                           key_.append(static_cast<const setlist*>(deref_(child))->prefix,
                                       rh->hdr.prefix_len);
                        else if (child.tag() == K::hdr && rh->hdr.prefix_len)
                           key_.append(reinterpret_cast<const uint8_t*>(deref_(child)) +
                                           kv_tree::WPFX_OFF,
                                       rh->hdr.prefix_len);
                        cb = uint32_t(key_.len);
                     }
                     uint8_t    bb;
                     packed_ptr nc;
                     if (prev_child_(child, 255, bb, nc))
                     {
                        stk_.push_back(Frame{child, int(bb) + 1, uint32_t(base), cb});
                        if (maintain_) key_.push(bb);
                        const size_t nbase = maintain_ ? size_t(key_.len) : size_t(cb) + 1;
                        const char*  IL;
                        if (kid_inline_(child, bb, &IL)) { emit_leaf_(IL, nbase); return; }
                        child = nc;
                        base  = nbase;
                        continue;
                     }
                     stk_.push_back(Frame{child, 0, uint32_t(base), cb});  // childless: term only
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(rh->term);
                     if (maintain_) key_.len = cb;
                     return;
                  }
               }
            }
         }
         void retreat_()  // step to the PREVIOUS terminal; the mirror of seek_
         {
            while (!stk_.empty())
            {
               Frame&  f = stk_.back();
               const K k = f.node.tag();
               if (k == K::prefix_node)
               {  // single child, no term: backward-exhausted on re-entry
                  if (maintain_) key_.len = f.plen;
                  stk_.pop_back();
                  continue;
               }
               if constexpr (Buckets || Adaptive)
                  if (k == K::bucket)
                  {
                     if (f.cur >= 2)  // current = bord_[cur-1]; previous = bord_[cur-2]
                     {
                        --f.cur;
                        bucket*     b = static_cast<bucket*>(deref_(f.node));
                        const char* e = bkt_entry(b, bord_[size_t(f.cur - 1)]);
                        sfx_ = bkt_suf(e); sfxlen_ = bkt_slen(e); full_str_ = nullptr;
                        set_ptr_(bkt_val(e));
                        if (maintain_) { key_.len = f.cbase; key_.append(sfx_, sfxlen_); }
                        return;
                     }
                     if (maintain_) key_.len = f.plen;
                     stk_.pop_back();
                     continue;
                  }
               if (k == K::setlist_u16)
               {
                  const setlist16* s = static_cast<const setlist16*>(deref_(f.node));
                  if (f.cur >= 2)  // current = branch[cur-1]; previous = branch[cur-2]
                  {
                     --f.cur;
                     const int      i  = f.cur - 1;
                     const uint16_t st = s->stems[i];
                     if (maintain_)
                     { key_.len = f.cbase; key_.push(uint8_t(st >> 8)); key_.push(uint8_t(st & 0xFF)); }
                     descend_max_(s->branch[i], maintain_ ? size_t(key_.len) : 0);
                     return;
                  }
                  if (f.cur == 1 && !s->rh.term.is_null())  // before branch[0] comes the term
                  {
                     f.cur = 0;
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(s->rh.term);
                     if (maintain_) key_.len = f.cbase;
                     return;
                  }
                  if (maintain_) key_.len = f.plen;
                  stk_.pop_back();
                  continue;
               }
               // byte routers
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(f.node));
               if (f.cur >= 1)  // current = child at byte cur-1
               {
                  uint8_t    bb;
                  packed_ptr child;
                  if (prev_child_(f.node, f.cur - 2, bb, child))
                  {
                     f.cur = int(bb) + 1;
                     if (maintain_) { key_.len = f.cbase; key_.push(bb); }
                     descend_kid_max_(f.node, bb, child, maintain_ ? size_t(key_.len) : 0);
                     return;
                  }
                  if (!rh->term.is_null())  // the term precedes the first child
                  {
                     f.cur = 0;
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(rh->term);
                     if (maintain_) key_.len = f.cbase;
                     return;
                  }
               }
               if (maintain_) key_.len = f.plen;  // cur <= 0: nothing precedes here
               stk_.pop_back();
               continue;
            }
            end_ = true;  // retreated past the first key (--begin(), as std::map: rend)
         }
         // true → positioned at a terminal (sets val_ + suffix); false → pushed a frame.
         // `base` is the key length at this child's start; key_ is grown only when
         // maintain_ is on (the eager fast path), otherwise the path in stk_ suffices.
         bool descend_(packed_ptr child, size_t base)
         {
            switch (child.tag())
            {
               case K::value_ptr:
                  // external leaf; the inline-leaf path emits the same way via descend_kid_
                  emit_leaf_(static_cast<const char*>(deref_(child)), base);
                  return true;
               case K::value_inline:
                  sfx_       = nullptr;
                  sfxlen_    = 0;
                  full_str_  = nullptr;
                  val_local_ = false;
                  if constexpr (inlineable)
                     unpack_inline(child, val_);
                  if (maintain_) key_.len = base;
                  return true;
               case K::prefix_node:
               {
                  const char* P  = static_cast<const char*>(deref_(child));
                  uint32_t    cb = uint32_t(base);
                  if (maintain_) { key_.len = base; key_.append(pn_pfx<Ptr>(P), pn_plen(P)); cb = uint32_t(key_.len); }
                  stk_.push_back(Frame{child, 0, uint32_t(base), cb});
                  return false;
               }
               case K::bucket:
               {
                  bord_.clear();  // sort the current bucket's entries by full suffix
                  bkt_append_order(static_cast<bucket*>(deref_(child)), bord_);
                  stk_.push_back(Frame{child, 0, uint32_t(base), uint32_t(base)});
                  return false;
               }
               case K::cbucket:
               {
                  if constexpr (kv_tree::cb_enabled)
                  {
                     // pages are SELF-CONTAINED (entry key = page prefix + suffix):
                     // the radix path above is the page's DIVIDER, never part of
                     // the key — the key buffer restarts at zero inside a page
                     // and pops back to the divider path (f.plen) on exit.
                     const auto* h =
                         reinterpret_cast<const typename kv_tree::cb_hdr*>(t_->deref_(child));
                     uint32_t cb = 0;
                     if (maintain_)
                     {
                        key_.len = 0;
                        key_.append(h->prefix, h->rh.hdr.prefix_len);
                        cb = uint32_t(key_.len);
                     }
                     stk_.push_back(Frame{child, -1, uint32_t(base), cb});
                     return false;
                  }
                  else
                     return false;
               }
               default:  // routers (setlist / c2 / c4 / c8 / node_full / hdr)
               {
                  const router_hdr* rh = static_cast<const router_hdr*>(deref_(child));
                  uint32_t          cb = uint32_t(base);
                  if (maintain_)
                  {
                     key_.len = base;
                     if (child.tag() == K::setlist_u8 && rh->hdr.prefix_len)
                        key_.append(static_cast<const setlist*>(deref_(child))->prefix, rh->hdr.prefix_len);
                     else if (child.tag() == K::hdr && rh->hdr.prefix_len)
                        key_.append(reinterpret_cast<const uint8_t*>(deref_(child)) +
                                        kv_tree::WPFX_OFF,
                                    rh->hdr.prefix_len);
                     cb = uint32_t(key_.len);
                  }
                  stk_.push_back(Frame{child, -1, uint32_t(base), cb});
                  return false;
               }
            }
         }
         void seek_()  // advance to next terminal; maintains key_ incrementally iff maintain_
         {
            while (!stk_.empty())
            {
               Frame&  f = stk_.back();
               const K k = f.node.tag();
               if (k == K::prefix_node)
               {
                  if (f.cur == 0)
                  {
                     f.cur = 1;
                     if (descend_(pn_next<Ptr>(static_cast<char*>(deref_(f.node))), f.cbase)) return;
                  }
                  else { if (maintain_) key_.len = f.plen; stk_.pop_back(); }
                  continue;
               }
               if constexpr (kv_tree::cb_enabled)
                  if (k == K::cbucket)
                  {
                     const auto* h =
                         reinterpret_cast<const typename kv_tree::cb_hdr*>(t_->deref_(f.node));
                     if (f.cur == -1)
                     {
                        f.cur = 0;
                        if (!h->rh.term.is_null() && descend_(h->rh.term, f.cbase))
                           return;
                     }
                     if (f.cur < int32_t(h->rh.hdr.nbranch))
                     {
                        unsigned r  = unsigned(f.cur++);
                        unsigned li = 0;
                        while (li < 3 && r >= h->cnt[li])  // ragged rank → (line, slot)
                           r -= h->cnt[li++];
                        const auto* sp = t_->cb_spl_(h, li);
                        auto        c  = t_->cb_cell_at_(h, sp->lref[r]);
                        sfx_      = reinterpret_cast<const uint8_t*>(c.p + 3);
                        sfxlen_   = c.p[0];
                        full_str_ = nullptr;
                        T* cv      = reinterpret_cast<T*>(cvalbuf_);
                        *cv        = T::from_bytes(c.value());  // trivially copyable by cb contract
                        val_local_ = true;
                        if (maintain_)
                        {
                           key_.len = f.cbase;
                           key_.append(sfx_, sfxlen_);
                        }
                        return;
                     }
                     if (maintain_)
                        key_.len = f.plen;
                     stk_.pop_back();
                     continue;
                  }
               if constexpr (Buckets || Adaptive)  // bucket frames exist only under these modes
                  if (k == K::bucket)
                  {
                     bucket* b = static_cast<bucket*>(deref_(f.node));
                     if (f.cur < b->nent)
                     {
                        const char* e = bkt_entry(b, bord_[size_t(f.cur++)]);
                        sfx_          = bkt_suf(e);
                        sfxlen_       = bkt_slen(e);
                        full_str_     = nullptr;
                        set_ptr_(bkt_val(e));
                        if (maintain_) { key_.len = f.cbase; key_.append(sfx_, sfxlen_); }
                        return;
                     }
                     if (maintain_) key_.len = f.plen;
                     stk_.pop_back();
                     continue;
                  }
               if (k == K::setlist_u16)  // wide router: f.cur = -1 (term pending) then stem index
               {
                  const setlist16* s = static_cast<const setlist16*>(deref_(f.node));
                  if (f.cur == -1)
                  {
                     f.cur = 0;
                     if (!s->rh.term.is_null())
                     {
                        if constexpr (kv_tree::cb_create)
                        {  // page mode: a router term IS a page — descend into it
                           if (descend_(s->rh.term, maintain_ ? key_.len : 0))
                              return;
                           continue;
                        }
                        sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                        emit_(s->rh.term);
                        if (maintain_) key_.len = f.cbase;
                        return;
                     }
                  }
                  if (f.cur < s->rh.hdr.nbranch)  // stems sorted ascending == key order
                  {
                     const uint16_t stem  = s->stems[f.cur];
                     packed_ptr     child = s->branch[f.cur];
                     ++f.cur;
                     if (maintain_)
                     { key_.len = f.cbase; key_.push(uint8_t(stem >> 8)); key_.push(uint8_t(stem & 0xFF)); }
                     if (descend_(child, key_.len)) return;
                  }
                  else { if (maintain_) key_.len = f.plen; stk_.pop_back(); }
                  continue;
               }
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(f.node));
               if (f.cur == -1)
               {
                  f.cur = 0;
                  if (!rh->term.is_null())
                  {
                     if constexpr (kv_tree::cb_create)
                     {  // page mode: a router term IS a page — descend into it
                        if (descend_(rh->term, maintain_ ? key_.len : 0))
                           return;
                        continue;
                     }
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(rh->term);
                     if (maintain_) key_.len = f.cbase;
                     return;
                  }
               }
               uint8_t    bb;
               packed_ptr child;
               if (next_child_(f.node, f.cur, bb, child))
               {
                  f.cur = bb + 1;
                  if (maintain_) { key_.len = f.cbase; key_.push(bb); }
                  if (descend_kid_(f.node, bb, child, key_.len)) return;
               }
               else { if (maintain_) key_.len = f.plen; stk_.pop_back(); }
            }
            end_ = true;
         }

         // Directed descent for lower_bound/upper_bound. Follows `key`; returns true if it
         // positioned directly at a qualifying terminal, false if it left the frame stack
         // such that a subsequent seek_() yields the bound (next-greater leftmost, or the
         // next sibling after an all-less subtree). `upper`: strict (first key > key).
         bool descend_lb_(packed_ptr cur, std::string_view key, size_t depth, bool upper)
         {
            const K k = cur.tag();
            if (k == K::value_ptr)
            {
               const char*      L = static_cast<const char*>(deref_(cur));
               std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
               const int        cmp = S.compare(key.substr(depth));
               if (upper ? cmp > 0 : cmp >= 0) return descend_(cur, 0);  // qualifies → position
               return false;                                            // < target → seek_ from parent
            }
            if (k == K::value_inline)
            {
               // stored key == path (empty suffix). Qualifies only for lower_bound when the
               // target ends here exactly (path == target).
               if (!upper && depth == key.size()) return descend_(cur, 0);
               return false;
            }
            if constexpr (Buckets || Adaptive)  // buckets exist only under these modes
               if (k == K::bucket)
               {
                  bucket* b = static_cast<bucket*>(deref_(cur));
                  bord_.clear();
                  bkt_append_order(b, bord_);
                  const std::string_view R = key.substr(depth);
                  int                    i = 0;
                  for (; i < b->nent; ++i)
                  {
                     const char*      e = bkt_entry(b, bord_[size_t(i)]);
                     std::string_view S(reinterpret_cast<const char*>(bkt_suf(e)), bkt_slen(e));
                     const int        cmp = S.compare(R);
                     if (upper ? cmp > 0 : cmp >= 0) break;
                  }
                  stk_.push_back(Frame{cur, i, 0, 0});  // seek_ yields bord_[i] (or pops if i==nent)
                  return false;
               }
            if constexpr (kv_tree::cb_enabled)
               if (k == K::cbucket)
               {
                  const auto* h =
                      reinterpret_cast<const typename kv_tree::cb_hdr*>(t_->deref_(cur));
                  std::string_view P(reinterpret_cast<const char*>(h->prefix),
                                     h->rh.hdr.prefix_len);
                  std::string_view R = key.substr(depth);
                  const size_t     c = lcp(P, R);
                  if (c < P.size())
                  {
                     if (c == R.size() || uint8_t(P[c]) > uint8_t(R[c]))
                     {
                        descend_(cur, 0);  // whole bucket > target → leftmost
                        return false;
                     }
                     // whole bucket < target → exhausted frame; seek_ pops it
                     stk_.push_back(Frame{cur, int32_t(h->rh.hdr.nbranch), 0, 0});
                     return false;
                  }
                  R = R.substr(c);
                  if (R.empty() && !upper)
                  {
                     stk_.push_back(Frame{cur, -1, 0, 0});  // term (== target) first
                     return false;
                  }
                  // rank of the first entry >= R: storage is totally ordered, so
                  // walk the ragged lines (suffix compare only within stem ties)
                  const uint16_t stem = kv_tree::cb_stem_(R);
                  int32_t        rank = 0;
                  for (unsigned li = 0; li < 4; ++li)
                  {
                     if (!h->cnt[li] || h->sp_line[li].is_null())
                        continue;
                     const auto*    sp  = t_->cb_spl_(h, li);
                     const unsigned lb2 = kv_tree::cb_lb_(sp, h->cnt[li], stem);
                     rank += int32_t(lb2);
                     for (unsigned i = lb2; i < h->cnt[li]; ++i)
                     {
                        if (sp->stems[i] != stem)
                           goto cb_lb_done;
                        const int cmp = t_->cb_cell_at_(h, sp->lref[i]).suffix().compare(R);
                        if (upper ? cmp > 0 : cmp >= 0)
                           goto cb_lb_done;
                        ++rank;
                     }
                  }
               cb_lb_done:
                  stk_.push_back(Frame{cur, rank, 0, 0});  // seek_ yields rank (or pops)
                  return false;
               }
            if (k == K::prefix_node)
            {
               const char*      P = static_cast<const char*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
               std::string_view R = key.substr(depth);
               const size_t     c = lcp(PP, R);
               if (c == PP.size())  // prefix matched → descend the single child
               {
                  stk_.push_back(Frame{cur, 1, 0, 0});
                  return descend_lb_(pn_next<Ptr>(const_cast<char*>(P)), key, depth + c, upper);
               }
               if (c == R.size() || uint8_t(PP[c]) > uint8_t(R[c]))
               {
                  descend_(cur, 0);  // subtree > target → leftmost
                  return false;
               }
               stk_.push_back(Frame{cur, 1, 0, 0});  // subtree < target → pop to parent's next
               return false;
            }
            if (k == K::setlist_u16)  // wide router: position on the 2-byte stem lower-bound
            {
               const setlist16* s = static_cast<const setlist16*>(deref_(cur));
               if (depth == key.size())  // key ends here: lower yields term then stems; upper stems only
               {
                  stk_.push_back(Frame{cur, upper ? 0 : -1, 0, 0});
                  return false;
               }
               // <2 bytes left: pad the missing low byte with 0 → first stem with hi >= key[depth]
               const uint16_t q = (key.size() - depth < 2)
                                      ? uint16_t(uint16_t(uint8_t(key[depth])) << 8)
                                      : uint16_t(uint16_t(uint8_t(key[depth])) << 8 | uint8_t(key[depth + 1]));
               const int lb = sl16_lb(s, q);
               if (key.size() - depth >= 2 && lb < s->rh.hdr.nbranch && s->stems[lb] == q)
               {
                  stk_.push_back(Frame{cur, lb + 1, 0, 0});  // exact stem → recurse; resume after it
                  return descend_lb_(s->branch[lb], key, depth + 2, upper);
               }
               stk_.push_back(Frame{cur, lb, 0, 0});  // seek_ yields stems from the lower-bound index
               return false;
            }
            // routers (setlist / c2 / c4 / c8 / node_full / hdr)
            const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
            if ((k == K::setlist_u8 || k == K::hdr) && rh->hdr.prefix_len)
            {
               const char* pb =
                   k == K::setlist_u8
                       ? reinterpret_cast<const char*>(
                             static_cast<const setlist*>(deref_(cur))->prefix)
                       : static_cast<const char*>(deref_(cur)) + kv_tree::WPFX_OFF;
               std::string_view PP(pb, rh->hdr.prefix_len);
               std::string_view R = key.substr(depth);
               const size_t     c = lcp(PP, R);
               if (c < PP.size())
               {
                  if (c == R.size() || uint8_t(PP[c]) > uint8_t(R[c])) { descend_(cur, 0); return false; }
                  stk_.push_back(Frame{cur, 256, 0, 0});  // subtree < target → pop router
                  return false;
               }
               depth += PP.size();
            }
            if (depth == key.size())  // key ends at this router
            {
               // lower: term (== key) then children (>). upper: skip term, children only.
               stk_.push_back(Frame{cur, upper ? 0 : -1, 0, 0});
               return false;
            }
            const uint8_t    byte  = uint8_t(key[depth]);
            const packed_ptr child = router_child_(cur, byte);
            if (!child.is_null())  // exact child → follow the key deeper
            {
               const char* IL;
               if (kid_inline_(cur, byte, &IL))
               {  // inline leaf: lower-bound compare in place (mirror of the value_ptr case)
                  stk_.push_back(Frame{cur, byte + 1, 0, 0});
                  std::string_view S(reinterpret_cast<const char*>(leaf_suf(IL)), leaf_slen(IL));
                  const int        cmp = S.compare(key.substr(depth + 1));
                  if (upper ? cmp > 0 : cmp >= 0) { emit_leaf_(IL, 0); return true; }
                  return false;
               }
               stk_.push_back(Frame{cur, byte + 1, 0, 0});
               return descend_lb_(child, key, depth + 1, upper);
            }
            stk_.push_back(Frame{cur, byte, 0, 0});  // no exact child → seek_ takes next-greater leftmost
            return false;
         }

        public:
         // LegacyForwardIterator surface. NOTE: reference is a proxy (key is rebuilt, value
         // may be inline), so this is NOT a C++20 std::forward_iterator — same status as
         // std::vector<bool>::iterator. Works with range-for and the classic algorithms.
         using iterator_category = std::forward_iterator_tag;
         using key_type          = Key;
         using mapped_type       = T;
         using value_type        = std::pair<const Key, T>;
         using reference         = std::pair<Key, const T&>;  // key decoded by value (see key_codec)
         using difference_type   = std::ptrdiff_t;
         struct arrow_proxy  // operator-> for a proxy reference (holds the pair, hands back &it)
         {
            reference        kv;
            const reference* operator->() const noexcept { return &kv; }
         };
         using pointer = arrow_proxy;

         const_iterator() = default;
         struct end_tag
         {
         };
         // end() WITH tree identity: compares equal to any end iterator, and --it
         // re-seats at the rightmost key (std::map's --end() / rbegin()).
         const_iterator(const kv_tree* t, end_tag) : t_(t) {}
         explicit const_iterator(const kv_tree* t, packed_ptr root) : t_(t)
         {
            if (!root.is_null())
            {
               end_ = false;
               if (!descend_(root, 0)) seek_();
            }
         }
         // lower_bound (upper=false) / upper_bound (upper=true) seek.
         const_iterator(const kv_tree* t, packed_ptr root, std::string_view key, bool upper) : t_(t)
         {
            if (!root.is_null())
            {
               end_ = false;
               if constexpr (kv_tree::cb_create)
                  pg_lb_(key, upper);  // page mode: floor route + in-page rank
               else if (!descend_lb_(root, key, 0, upper))
                  seek_();
            }
         }
         // page-mode lower_bound: floor-route, walk the divider path pushing
         // frames (cur = byte+1 so ++ resumes after the taken branch), enter the
         // page, rank-position within it. A fully-smaller page leaves cur at
         // nbranch and seek_ pops into the NEXT page.
         void pg_lb_(std::string_view key, bool upper)
         {
            if constexpr (kv_tree::cb_enabled)
            {
               typename kv_tree::pg_div div;
               t_->pg_route_(key, &div);
               packed_ptr cur   = t_->root_;
               size_t     depth = 0;
               while (cur.tag() != K::cbucket)
               {
                  const K k = cur.tag();
                  if (k == K::prefix_node)
                  {
                     const char* P = static_cast<const char*>(t_->deref_(cur));
                     stk_.push_back(Frame{cur, 1, 0, 0});
                     depth += pn_plen(P);
                     cur = pn_next<Ptr>(const_cast<char*>(P));
                     continue;
                  }
                  const auto* rh = static_cast<const router_hdr*>(t_->deref_(cur));
                  if (k == K::setlist_u8 && rh->hdr.prefix_len)
                     depth += rh->hdr.prefix_len;
                  if (depth == div.len)
                  {  // the page hangs at this node's term; branches resume at 0
                     stk_.push_back(Frame{cur, 0, 0, 0});
                     cur = rh->term;
                     continue;
                  }
                  const uint8_t b = div.b[depth];
                  stk_.push_back(Frame{cur, int32_t(b) + 1, 0, 0});
                  cur = router_child_(cur, b);
                  ++depth;
               }
               const auto* h = reinterpret_cast<const typename kv_tree::cb_hdr*>(t_->deref_(cur));
               const std::string_view pfx(reinterpret_cast<const char*>(h->prefix),
                                          h->rh.hdr.prefix_len);
               // key vs (pfx + sfx), lexicographic
               auto cmp_full = [&](std::string_view sfx) -> int
               {
                  const size_t m1 = std::min(pfx.size(), key.size());
                  int          c  = m1 ? std::memcmp(pfx.data(), key.data(), m1) : 0;
                  if (c != 0)
                     return c;
                  if (pfx.size() >= key.size())
                     return (pfx.size() > key.size() || !sfx.empty()) ? 1 : 0;
                  const std::string_view kr = key.substr(pfx.size());
                  const size_t           m2 = std::min(sfx.size(), kr.size());
                  c = m2 ? std::memcmp(sfx.data(), kr.data(), m2) : 0;
                  if (c != 0)
                     return c;
                  return sfx.size() == kr.size() ? 0 : sfx.size() < kr.size() ? -1 : 1;
               };
               int32_t pos    = 0;
               bool    term_q = false;
               if (!h->rh.term.is_null())
               {
                  const int c = cmp_full(std::string_view{});
                  term_q      = upper ? c > 0 : c >= 0;
               }
               if (!term_q)
                  for (unsigned li = 0; li < 4; ++li)
                  {
                     if (!h->cnt[li] || h->sp_line[li].is_null())
                        continue;
                     const auto* sp = t_->cb_spl_(h, li);
                     for (unsigned i = 0; i < h->cnt[li]; ++i)
                     {
                        auto      cell = t_->cb_cell_at_(h, sp->lref[i]);
                        const int c    = cmp_full(cell.suffix());
                        if (upper ? c > 0 : c >= 0)
                           goto pg_lb_done;
                        ++pos;
                     }
                  }
            pg_lb_done:
               stk_.push_back(Frame{cur, term_q ? -1 : pos, 0, 0});
               seek_();
            }
         }
         // Pure position comparison (no key materialization): two iterators are equal iff
         // they sit at the same terminal, which the frame stack (node + cursor per level)
         // identifies uniquely regardless of how each was reached.
         bool operator==(const const_iterator& o) const noexcept
         {
            if (end_ != o.end_) return false;
            if (end_) return true;
            if (stk_.size() != o.stk_.size()) return false;
            for (size_t i = 0; i < stk_.size(); ++i)
               if (stk_[i].node.raw() != o.stk_[i].node.raw() || stk_[i].cur != o.stk_[i].cur)
                  return false;
            return true;
         }
         bool            operator!=(const const_iterator& o) const noexcept { return !(*this == o); }
         const_iterator& operator++() { seek_(); return *this; }
         const_iterator  operator++(int)
         {
            const_iterator tmp = *this;
            seek_();
            return tmp;
         }
         const_iterator& operator--()
         {
            if (end_)
            {  // --end(): seat at the rightmost key (needs a tree-carrying end iterator)
               if (t_ && !t_->root_.is_null())
               {
                  end_ = false;
                  stk_ = frame_stack();
                  if (maintain_) key_.len = 0;
                  descend_max_(t_->root_, 0);
               }
               return *this;
            }
            retreat_();
            return *this;
         }
         const_iterator operator--(int)
         {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
         }
         const T&        value() const noexcept
         {
            if constexpr (inlineable)
               return val_;
            else
               return val_local_ ? *reinterpret_cast<const T*>(cvalbuf_) : *val_;
         }
         // Adaptive key materialization. Until key() is first read, ++ does NO key work
         // (value-only iteration pays nothing). The first read replays the whole path,
         // priming each frame's plen/cbase, and flips maintain_ on — from then ++ keeps
         // key_ current incrementally (only the diverged tail is rewritten), so repeated
         // key reads are as cheap as the eager scan. Assumes "read once → read again".
         // Raw key bytes (the radix's native form): zero-copy where possible. key() decodes
         // these back to the typed Key (by value — a non-string key has no stored object).
         Key              key() const { return key_codec<Key>::decode(key_bytes()); }
         std::string_view key_bytes() const
         {
            if (full_str_)
               return std::string_view(full_str_, full_len_);  // zero-copy: full key in the leaf
            if (maintain_)
               return key_.view();  // already current
            key_.len = 0;
            for (Frame& fr : stk_)
            {
               fr.plen   = uint32_t(key_.len);
               const K k = fr.node.tag();
               if (k == K::prefix_node)
               {
                  const char* P = static_cast<const char*>(deref_(fr.node));
                  key_.append(pn_pfx<Ptr>(P), pn_plen(P));
                  fr.cbase = uint32_t(key_.len);
               }
               else if (k == K::setlist_u8)
               {
                  const setlist* s = static_cast<const setlist*>(deref_(fr.node));
                  if (s->rh.hdr.prefix_len) key_.append(s->prefix, s->rh.hdr.prefix_len);
                  fr.cbase = uint32_t(key_.len);
                  if (fr.cur >= 1) key_.push(uint8_t(fr.cur - 1));
               }
               else if (k == K::setlist_u16)  // wide router: replay the 2 bytes of the descended stem
               {
                  const setlist16* s = static_cast<const setlist16*>(deref_(fr.node));
                  fr.cbase = uint32_t(key_.len);
                  if (fr.cur >= 1)
                  { const uint16_t st = s->stems[fr.cur - 1]; key_.push(uint8_t(st >> 8)); key_.push(uint8_t(st & 0xFF)); }
               }
               else if (k == K::cbucket)  // self-contained page: key restarts here
               {
                  if constexpr (kv_tree::cb_enabled)
                  {
                     const auto* h =
                         reinterpret_cast<const typename kv_tree::cb_hdr*>(t_->deref_(fr.node));
                     key_.len = 0;
                     if (h->rh.hdr.prefix_len)
                        key_.append(h->prefix, h->rh.hdr.prefix_len);
                  }
                  fr.cbase = uint32_t(key_.len);
               }
               else if (k == K::hdr)  // see-header wide: CL0 prefix, then edge
               {
                  const char* hp = static_cast<const char*>(deref_(fr.node));
                  if (const unsigned pl =
                          reinterpret_cast<const router_hdr*>(hp)->hdr.prefix_len)
                     key_.append(reinterpret_cast<const uint8_t*>(hp) + kv_tree::WPFX_OFF,
                                 pl);
                  fr.cbase = uint32_t(key_.len);
                  if (fr.cur >= 1) key_.push(uint8_t(fr.cur - 1));
               }
               else  // bucket (no prefix/edge) or c2/c4/c8/node_full (edge only)
               {
                  fr.cbase = uint32_t(key_.len);
                  if (k != K::bucket && fr.cur >= 1) key_.push(uint8_t(fr.cur - 1));
               }
            }
            if (sfxlen_) key_.append(sfx_, sfxlen_);
            maintain_ = true;
            return key_.view();
         }
         reference  operator*() const { return reference{key(), value()}; }
         arrow_proxy operator->() const { return arrow_proxy{operator*()}; }
      };

      using iterator = const_iterator;  // read-only container (no element-mutating iterator)

      const_iterator begin() const { return const_iterator(this, root_); }
      const_iterator end() const { return const_iterator(); }
      // Iterator seated at the RIGHTMOST key (end() when empty) — the cursor's
      // seek_last; equivalent to --end() on a tree-carrying end iterator.
      const_iterator last() const
      {
         const_iterator it(this, typename const_iterator::end_tag{});
         --it;
         return it;
      }
      const_iterator cbegin() const { return begin(); }
      const_iterator cend() const { return end(); }

      // Ordered positioning (lexicographic, unsigned-byte). lower_bound: first key >= key;
      // upper_bound: first key > key; equal_range: [lower, upper) (≤1 element since keys
      // are unique). All return end() when no such key exists.
      const_iterator lower_bound(key_param k) const
      {
         scratch_t ks_;
         return const_iterator(this, root_, enc_(k, ks_), false);
      }
      const_iterator upper_bound(key_param k) const
      {
         scratch_t ks_;
         return const_iterator(this, root_, enc_(k, ks_), true);
      }
      std::pair<const_iterator, const_iterator> equal_range(key_param k) const
      {
         scratch_t        ks_;
         std::string_view kb = enc_(k, ks_);
         return {const_iterator(this, root_, kb, false), const_iterator(this, root_, kb, true)};
      }

      // Map-style lookup/erase returning the STL shapes (iterator / count).
      const_iterator find(key_param k) const
      {
         scratch_t        ks_;
         std::string_view kb = enc_(k, ks_);
         const_iterator   it(this, root_, kb, false);  // lower_bound on the bytes
         return (it != end() && it.key_bytes() == kb) ? it : end();
      }
      size_type count(key_param k) const { return contains(k) ? 1 : 0; }
      size_type erase(key_param k) { return remove(k) ? 1 : 0; }

      // Fast ordered scan (the libart art_iter model): a recursive DFS visitor that
      // enumerates children directly (no per-element resumption, no byte-probing) and
      // grows one shared key buffer. ~3x the iterator on cache-resident scans. Calls
      // f(std::string_view key, const T& value) in ascending key order.
      template <class F>
      void for_each(F&& f) const
      {
         if (root_.is_null()) return;
         key_buf          kb;
         std::vector<int> bscratch;
         walk_(root_, kb, bscratch, f);
      }

      // Ordered value-only scan: visits each value in ascending key order WITHOUT
      // building keys. Calls f(const T&). Doubles as a diagnostic: the gap vs for_each
      // is the key-reconstruction cost; what's left is the raw node walk.
      template <class F>
      void for_each_value(F&& f) const
      {
         if (!root_.is_null()) walk_v_(root_, f);
      }

      // ── bucket engine (gated; trivially-copyable T) ─────────────────────────────
      // A bucket is an inline (suffix,value) collector that collapses a sparse/deep
      // radix tail into one node. Enabled via enable_buckets(); terminals become
      // buckets, routers (setlist/cN/full) appear only when a bucket overflows.
     public:
      static constexpr bool bucketable = true;  // works for any T (entries never memmove'd)

     private:
      // Entry layout in the tail: [ T value | u16 suffix_len | suffix bytes ], with the
      // entry START aligned to alignof(T) (T placement-new'd in place — non-POD safe).
      // Sizes are alignof(T)-rounded so the tail stays aligned as it grows downward.
      static constexpr size_t TALIGN = alignof(T);
      static size_t bkt_round(size_t x) noexcept { return (x + TALIGN - 1) & ~(TALIGN - 1); }
      static uint16_t bkt_stem2(std::string_view s) noexcept
      {
         return uint16_t((s.size() > 0 ? unsigned(uint8_t(s[0])) << 8 : 0u) |
                         (s.size() > 1 ? unsigned(uint8_t(s[1])) : 0u));
      }
      static uint16_t*       bkt_div(bucket* b) noexcept { return reinterpret_cast<uint16_t*>(b->data); }
      static const uint16_t* bkt_div(const bucket* b) noexcept { return reinterpret_cast<const uint16_t*>(b->data); }
      static uint16_t*       bkt_slot(bucket* b) noexcept { return bkt_div(b) + b->nent; }
      static const uint16_t* bkt_slot(const bucket* b) noexcept { return bkt_div(b) + b->nent; }
      static char*           bkt_entry(bucket* b, int i) noexcept
      {
         return reinterpret_cast<char*>(b->data) + (bucket::PAGE - 4) - bkt_slot(b)[i];
      }
      static const char* bkt_entry(const bucket* b, int i) noexcept
      {
         return reinterpret_cast<const char*>(b->data) + (bucket::PAGE - 4) - bkt_slot(b)[i];
      }
      static T*             bkt_val(char* e) noexcept { return reinterpret_cast<T*>(e); }            // aligned
      static const T*       bkt_val(const char* e) noexcept { return reinterpret_cast<const T*>(e); }
      static uint16_t       bkt_slen(const char* e) noexcept { uint16_t v; std::memcpy(&v, e + sizeof(T), 2); return v; }
      static const uint8_t* bkt_suf(const char* e) noexcept { return reinterpret_cast<const uint8_t*>(e + sizeof(T) + 2); }
      static size_t         bkt_esize(size_t slen) noexcept { return bkt_round(sizeof(T) + 2 + slen); }
      // first index with div >= q (count of div < q). Deliberately scalar: PAGE caps n at
      // a few dozen, and a lane-masked NEON count measured flat (buckets bench, 2M keys —
      // the page touch dominates), so the branchless 2-op loop stays.
      static int bkt_lower(const uint16_t* d, int n, uint16_t q) noexcept
      {
         int c = 0;
         for (int i = 0; i < n; ++i) c += (d[i] < q);
         return c;
      }
      // entry i's suffix vs (s): -1/0/1
      static int bkt_cmp(const bucket* b, int i, std::string_view s) noexcept
      {
         const char* e  = bkt_entry(b, i);
         const int   el = bkt_slen(e), sn = int(s.size());
         const int   m  = el < sn ? el : sn;
         const int   c  = m ? std::memcmp(bkt_suf(e), s.data(), size_t(m)) : 0;
         if (c) return c < 0 ? -1 : 1;
         return el == sn ? 0 : (el < sn ? -1 : 1);
      }
      // The entry whose suffix == rem, else nullptr — the one bucket point-lookup, shared by
      // find/update/locate_. div[] narrows to the 2-byte stem group; the full suffix decides.
      static const char* bkt_locate(const bucket* b, std::string_view rem) noexcept
      {
         const uint16_t  q2 = bkt_stem2(rem);
         const uint16_t* d  = bkt_div(b);
         for (int i = bkt_lower(d, b->nent, q2); i < b->nent && d[i] == q2; ++i)
         {
            const char* e = bkt_entry(b, i);
            if (bkt_slen(e) == rem.size() &&
                (rem.empty() || std::memcmp(bkt_suf(e), rem.data(), rem.size()) == 0))
               return e;
         }
         return nullptr;
      }
      // Entry indices of b appended to `ord`, sorted by FULL suffix — div[] orders only by
      // the first 2 bytes; ordered iteration needs total lexicographic order.
      static void bkt_append_order(const bucket* b, std::vector<int>& ord)
      {
         const size_t first = ord.size();
         for (int i = 0; i < b->nent; ++i) ord.push_back(i);
         std::sort(ord.begin() + std::ptrdiff_t(first), ord.end(), [b](int a, int c) {
            const char* ea = bkt_entry(b, a);
            const char* ec = bkt_entry(b, c);
            return std::string_view(reinterpret_cast<const char*>(bkt_suf(ea)), bkt_slen(ea)) <
                   std::string_view(reinterpret_cast<const char*>(bkt_suf(ec)), bkt_slen(ec));
         });
      }

      // Node creators placement-new into the allocator's bytes: it starts the object's
      // lifetime ISO-cleanly and costs nothing (trivial default-init emits no code).
      bucket* new_bucket()
      {
         bucket* b   = ::new (node_alloc_(sizeof(bucket))) bucket;
         b->nent     = 0;
         b->tail_used = 0;
         return b;
      }
      // A fresh single-entry bucket holding (key[at:], val) — bucket mode's new-value branch.
      template <class VF>
      packed_ptr make_bucket_branch(std::string_view key, size_t at, VF&& val)
      {
         bucket* b = new_bucket();
         bkt_insert_entry(b, key.substr(at), std::forward<VF>(val));
         return pack_(b, K::bucket);
      }
      // Adaptive collapse: turn the leaf at a deep-narrow split point into a bucket
      // holding the old entry + the new key. Frees the old leaf.
      template <bool Assign, class VF>
      packed_ptr collapse_to_bucket(packed_ptr cur, std::string_view key, size_t depth, VF&& val,
                                    bool& inserted)
      {
         bucket* b = new_bucket();
         if (cur.tag() == K::value_inline)
         {
            T ov;
            unpack_inline(cur, ov);
            bkt_insert_entry(b, std::string_view{}, std::move(ov));
         }
         else
         {
            char*            L = static_cast<char*>(deref_(cur));
            std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
            T                ov(std::move(*leaf_val(L)));  // move out, then destruct the leaf's T
            v_destroy(alloc_, leaf_val(L));
            bkt_insert_entry(b, S, std::move(ov));
            node_free_(L, leaf_alloc_size(L));
         }
         inserted = (bkt_insert_entry<Assign>(b, key.substr(depth), std::forward<VF>(val)) == 1);
         return pack_(b, K::bucket);
      }
      bool bkt_find(const bucket* b, std::string_view rem, T& out) const
      {
         const char* e = bkt_locate(b, rem);
         if (!e)
            return false;
         copy_value(out, bkt_val(e));
         return true;
      }
      // 1 inserted, 0 updated, -1 overflow (caller splits). `val` is forwarded — moved
      // on a successful insert/update, left untouched on overflow (so the caller can
      // re-pass it after splitting).
      template <bool Assign = true, class V>
      int bkt_insert_entry(bucket* b, std::string_view suf, V&& val)
      {
         const uint16_t q2  = bkt_stem2(suf);
         uint16_t*      d   = bkt_div(b);
         int            pos = bkt_lower(d, b->nent, q2);
         for (; pos < b->nent && d[pos] == q2; ++pos)
         {
            const int c = bkt_cmp(b, pos, suf);
            if (c == 0)
            {
               if constexpr (Assign)
               {
                  char* e = bkt_entry(b, pos);  // update value in place (key/suffix unchanged)
                  v_destroy(alloc_, bkt_val(e));
                  vf_construct(std::forward<V>(val), bkt_val(e));
               }
               return 0;
            }
            if (c > 0) break;
         }
         const size_t esz   = bkt_esize(suf.size());
         const size_t freeb = size_t(bucket::PAGE - 4) - 4u * b->nent - b->tail_used;
         if (freeb < 4 + esz) return -1;  // page full
         b->tail_used += uint16_t(esz);
         char* e = reinterpret_cast<char*>(b->data) + (bucket::PAGE - 4) - b->tail_used;  // aligned
         vf_construct(std::forward<V>(val), bkt_val(e));  // construct at the entry (non-POD safe)
         const uint16_t sl = uint16_t(suf.size());
         std::memcpy(e + sizeof(T), &sl, 2);
         std::memcpy(e + sizeof(T) + 2, suf.data(), suf.size());
         const uint16_t off = b->tail_used;
         d                  = bkt_div(b);
         uint16_t* so       = d + b->nent;      // old slot base
         uint16_t* sn       = d + b->nent + 1;  // new slot base (div grew by one)
         std::memmove(sn + pos + 1, so + pos, size_t(b->nent - pos) * 2);
         std::memmove(sn, so, size_t(pos) * 2);
         std::memmove(d + pos + 1, d + pos, size_t(b->nent - pos) * 2);
         d[pos]  = q2;
         sn[pos] = off;
         ++b->nent;
         return 1;
      }
      // Remove (s) and RECLAIM its tail space (hkv has no background compactor). bp may
      // be updated (non-POD path rebuilds into a fresh bucket). Returns true if removed.
      bool bkt_remove_entry(packed_ptr& bp, std::string_view suf)
      {
         bucket*        b   = static_cast<bucket*>(deref_(bp));
         const uint16_t q2  = bkt_stem2(suf);
         uint16_t*      d   = bkt_div(b);
         int            pos = bkt_lower(d, b->nent, q2);
         for (; pos < b->nent && d[pos] == q2; ++pos)
         {
            const int c = bkt_cmp(b, pos, suf);
            if (c > 0)
               return false;
            if (c == 0)
               break;
         }
         if (pos >= b->nent || d[pos] != q2)
            return false;

         char* const base_end = reinterpret_cast<char*>(b->data) + (bucket::PAGE - 4);
         char*       dead     = base_end - bkt_slot(b)[pos];
         v_destroy(alloc_, bkt_val(dead));  // run T's destructor

         if constexpr (std::is_trivially_copyable_v<T>)
         {
            // compact the tail in place: shift the entries below the hole up by E.
            const uint16_t S          = bkt_slot(b)[pos];
            const size_t   E          = bkt_esize(bkt_slen(dead));
            char*          tail_start = base_end - b->tail_used;
            std::memmove(tail_start + E, tail_start, size_t(dead - tail_start));
            b->tail_used -= uint16_t(E);
            uint16_t* sl = bkt_slot(b);
            for (int i = 0; i < b->nent; ++i)
               if (sl[i] > S) sl[i] -= uint16_t(E);
            // drop the index pair at pos
            std::memmove(d + pos, d + pos + 1, size_t(b->nent - 1 - pos) * 2);
            std::memmove(d + b->nent - 1, d + b->nent, size_t(pos) * 2);
            std::memmove(d + b->nent - 1 + pos, d + b->nent + pos + 1, size_t(b->nent - 1 - pos) * 2);
            --b->nent;
            return true;
         }
         else
         {
            // non-POD: can't byte-shift T objects → rebuild a fresh tight bucket
            // (copy-construct live entries, destruct originals). Avoids any relocation.
            bucket* nb = new_bucket();
            for (int i = 0; i < b->nent; ++i)
            {
               if (i == pos) continue;
               char*            e = bkt_entry(b, i);
               std::string_view s(reinterpret_cast<const char*>(bkt_suf(e)), bkt_slen(e));
               bkt_insert_entry(nb, s, std::move(*bkt_val(e)));  // MOVE into nb
               v_destroy(alloc_, bkt_val(e));
            }
            node_free_(b, sizeof(bucket));
            bp = pack_(nb, K::bucket);
            return true;
         }
      }

      // Overflow: split a bucket at `depth` into a radix subtree. Extract the common
      // prefix of all (sorted) suffixes, then re-insert every entry through bkt_put.
      packed_ptr bkt_split(packed_ptr bp)
      {
         bucket*    b = static_cast<bucket*>(deref_(bp));
         const int  n = b->nent;
         // common prefix of all suffixes == lcp(first, last) since lex-sorted
         const char*      e0 = bkt_entry(b, 0);
         const char*      eN = bkt_entry(b, n - 1);
         std::string_view f0(reinterpret_cast<const char*>(bkt_suf(e0)), bkt_slen(e0));
         std::string_view fN(reinterpret_cast<const char*>(bkt_suf(eN)), bkt_slen(eN));
         const size_t     c = lcp(f0, fN);
         setlist*         r = new_setlist();
         packed_ptr       top;
         {
            // common bytes from a stable copy (entries get freed when we drop b)
            uint8_t cbuf[setlist::PREFIX_CAP > 8 ? setlist::PREFIX_CAP : 8];
            const bool fits = c <= setlist::PREFIX_CAP;
            if (fits && c) std::memcpy(cbuf, bkt_suf(e0), c);
            // A short common prefix goes inline; a longer one is NOT wrapped here — the
            // re-inserted full suffixes below re-form it naturally as prefix nodes.
            top = fits ? apply_prefix(r, std::string_view(reinterpret_cast<char*>(cbuf), c))
                       : pack_(r, K::setlist_u8);
         }
         // Destructive rebuild: values are MOVED out of b. Strong isn't achievable without
         // copying (T may be move-only), but basic-guarantee + leak-freedom is: guard the
         // partial `top` so a throw reclaims it, and DON'T destroy the moved-from originals
         // until the whole rebuild succeeds — on a throw, b keeps a full set of valid
         // (moved-from) values, so it stays consistent and is reclaimed normally at teardown.
         build_guard g{this, &top};
         for (int i = 0; i < n; ++i)
         {
            char*            e = bkt_entry(b, i);
            std::string_view suf(reinterpret_cast<const char*>(bkt_suf(e)), bkt_slen(e));
            bool             ins = false;
            top                  = bkt_put<true>(top, suf, 0, std::move(*bkt_val(e)), ins);  // MOVE into target
         }
         g.release();
         for (int i = 0; i < n; ++i)  // committed: destroy all moved-from originals, free b
            v_destroy(alloc_, bkt_val(bkt_entry(b, i)));
         node_free_(b, sizeof(bucket));
         return top;
      }

      // Recursive bucket-mode insert into the subtree at `node`. Terminals are buckets;
      // routers reused from the radix family. `val` is MOVED to its single resting place
      // (each path consumes it exactly once — on overflow bkt_insert_entry leaves it
      // intact, so the re-pass after the split still moves a live value).
      template <bool Assign, class VF>
      packed_ptr bkt_put(packed_ptr node, std::string_view key, size_t depth, VF&& val, bool& ins)
      {
         if (node.is_null())
         {
            ins = true;
            return make_bucket_branch(key, depth, std::forward<VF>(val));
         }
         switch (node.tag())
         {
            case K::bucket:
            {
               bucket*   b = static_cast<bucket*>(deref_(node));
               const int r = bkt_insert_entry<Assign>(b, key.substr(depth), std::forward<VF>(val));
               if (r >= 0) { ins = (r == 1); return node; }
               packed_ptr router = bkt_split(node);                          // overflow → radix
               return bkt_put<Assign>(router, key, depth, std::forward<VF>(val), ins);      // val untouched on overflow
            }
            case K::prefix_node:
            {
               char*            P = static_cast<char*>(deref_(node));
               std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
               const size_t     c = lcp(PP, key.substr(depth));
               if (c == PP.size())
               {
                  packed_ptr nn = bkt_put<Assign>(pn_next<Ptr>(P), key, depth + c, std::forward<VF>(val), ins);
                  pn_set_next(P, nn);
                  return node;
               }
               ins = true;
               return split_prefix(node, key, depth, c, std::forward<VF>(val), [&](auto&& x) {
                  return make_bucket_branch(key, depth + c + 1, std::forward<decltype(x)>(x));
               });
            }
            case K::setlist_u8:
            {
               setlist*       s  = static_cast<setlist*>(deref_(node));
               const unsigned pl = s->rh.hdr.prefix_len;
               if (pl)
               {
                  const size_t c = lcp(std::string_view(reinterpret_cast<char*>(s->prefix), pl),
                                       key.substr(depth));
                  if (c < pl)
                  {
                     ins = true;
                     return split_setlist_prefix(node, key, depth, c, std::forward<VF>(val), [&](auto&& x) {
                        return make_bucket_branch(key, depth + c + 1, std::forward<decltype(x)>(x));
                     });
                  }
                  depth += pl;
               }
               if (depth == key.size()) { ins = set_or_update_term<Assign>(&s->rh, std::forward<VF>(val)); return node; }
               const uint8_t byte = uint8_t(key[depth]);
               const int     i    = setlist_index(s, byte);
               if (i >= 0)
               {
                  s->branch[i] = bkt_put<Assign>(s->branch[i], key, depth + 1, std::forward<VF>(val), ins);
                  return node;
               }
               ins              = true;
               packed_ptr child = make_bucket_branch(key, depth + 1, std::forward<VF>(val));
               if (setlist_set(s, byte, child)) return node;
               build_guard gc{this, &child};  // widen() can throw; don't orphan the new bucket
               uint8_t pfx[setlist::PREFIX_CAP];
               std::memcpy(pfx, s->prefix, pl);
               s->rh.hdr.prefix_len = 0;
               packed_ptr w         = node;
               do w = widen(w);
               while (!router_try_set(w, byte, child));
               gc.release();
               return pl ? make_prefix(std::string_view(reinterpret_cast<char*>(pfx), pl), w) : w;
            }
            default:  // c2 / c4 / c8 / node_full — no inline prefix
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(node));
               if (depth == key.size()) { ins = set_or_update_term<Assign>(rh, std::forward<VF>(val)); return node; }
               const uint8_t byte = uint8_t(key[depth]);
               if (packed_ptr* cs = router_find_slot(node, byte))
               {
                  *cs = bkt_put<Assign>(*cs, key, depth + 1, std::forward<VF>(val), ins);
                  return node;
               }
               ins              = true;
               packed_ptr child = make_bucket_branch(key, depth + 1, std::forward<VF>(val));
               build_guard gc{this, &child};  // widen() can throw; don't orphan the new bucket
               packed_ptr w     = node;
               while (!router_try_set(w, byte, child)) w = widen(w);
               gc.release();
               return w;
            }
         }
      }

     private:
      struct rm_lvl  // one descent level: where node j lives + how its parent reaches it
      {
         packed_ptr* slot;
         uint16_t    edge;  // byte (u8 routers/full/cN), 2-byte stem (u16); unused for pn/root
      };
      static constexpr uint32_t RM_LVN = 64;  // inline frames; deeper paths heap-spill
      // Full-depth removal path: inline for the common case, heap past it.
      // (The old fixed ring CUT the upward collapse at the window edge and
      // orphaned every empty ancestor above — the compat drain gate caught
      // 7 leaked lines on a 600-deep chain.)
      struct rm_stack
      {
         rm_lvl  sbo[RM_LVN];
         rm_lvl* d   = sbo;
         uint32_t cap = RM_LVN;
         ~rm_stack()
         {
            if (d != sbo)
               ::operator delete(d);
         }
         rm_lvl& operator[](uint32_t i) noexcept { return d[i]; }
         void    ensure(uint32_t n)
         {
            if (n < cap) [[likely]]
               return;
            const uint32_t ncap = cap * 2;
            auto* nd = static_cast<rm_lvl*>(::operator new(size_t(ncap) * sizeof(rm_lvl)));
            std::memcpy(nd, d, size_t(cap) * sizeof(rm_lvl));
            if (d != sbo)
               ::operator delete(d);
            d   = nd;
            cap = ncap;
         }
      };
      static constexpr int      SHRINK_MAX = setlist::CAP - 4;  // de-widen full/cN below the
                                                                // setlist CAP with hysteresis
                                                                // against widen/shrink ping-pong

      // Free the node at *L.slot (already unlinked logically); kinds of router only.
      void rm_free_(packed_ptr nd) noexcept
      {
         node_free_(deref_(nd), nd.tag() == K::hdr ? node_size(hdr_base_(deref_(nd)))
                                                   : node_size(nd.tag()));
      }

      // Collapse a single-branch term-less router: absorb its inline prefix (setlist only)
      // + the edge to its one child INTO the child — free when the child is a setlist with
      // prefix room, else via a fresh prefix node (best-effort: kept as-is on bad_alloc).
      // pfx holds the bytes to prepend (router prefix ++ edge byte(s)).
      void rm_collapse_(packed_ptr* slot, packed_ptr nd, const uint8_t* pfx, unsigned pl,
                        packed_ptr child) noexcept
      {
         if (child.tag() == K::setlist_u8 &&
             !(cow_stamped_ && alloc_.hkv_vtxn() && cow_shared_(child)))
         {  // in-place absorb requires an OWNED child; shared falls to make_prefix
            setlist*       cs  = static_cast<setlist*>(deref_(child));
            const unsigned cpl = cs->rh.hdr.prefix_len;
            if (cpl + pl <= setlist::PREFIX_CAP)
            {
               std::memmove(cs->prefix + pl, cs->prefix, cpl);
               std::memcpy(cs->prefix, pfx, pl);
               cs->rh.hdr.prefix_len = uint16_t(cpl + pl);
               *slot                 = child;
               rm_free_(nd);
               return;
            }
         }
         try  // child can't absorb: hang it under a small prefix node (pn->pn chains are fine)
         {
            packed_ptr top = make_prefix(std::string_view(reinterpret_cast<const char*>(pfx), pl), child);
            *slot          = top;
            rm_free_(nd);
         }
         catch (...)
         {
         }  // OOM: stay unshrunk — the tree is still correct
      }

      // Evaluate the router at *L.slot after it lost a branch/term. Returns true iff the
      // node was freed outright (empty) — the caller must then unlink it from ITS parent.
      bool rm_shrink_(packed_ptr* slot) noexcept
      {
         for (;;)  // a de-widened node is re-evaluated once as a setlist
         {
            const packed_ptr nd = *slot;
            const K          k  = nd.tag();
            if constexpr (cow_enabled)
               if (k == K::full_x || k == K::cx)
               {
                  router_hdr*    xrh = static_cast<router_hdr*>(deref_(nd));
                  const unsigned xnb = xrh->hdr.nbranch;
                  const bool     xtm = !xrh->term.is_null();
                  if (xnb == 0 && !xtm)
                  {
                     cow_release(nd);  // line-wise: dying lines release their content
                     return true;      // empty shell: caller unlinks and cascades
                  }
                  if (xnb == 0)
                  {  // only payload is the term: it replaces the node
                     const Ptr t = xrh->term;
                     if (!cow_retain(t))
                        return false;  // pathological refcount: keep the node
                     *slot = t;
                     cow_release(nd);
                     return false;
                  }
                  if (xnb <= unsigned(SHRINK_MAX))
                  {
                     try  // de-widen to one setlist line (shared-line safe:
                     {    // +1 per copied child pairs the -1 from dying lines)
                        setlist* s2         = new_setlist();
                        s2->rh.hdr.has_term = xrh->hdr.has_term;
                        s2->rh.term         = xrh->term;
                        cow_fix_child_(&s2->rh.term);
                        router_for_each(nd, [&](uint8_t bb, packed_ptr br) {
                           const bool ok = setlist_set(s2, bb, br);
                           assert(ok && "SHRINK_MAX <= setlist CAP");
                           (void)ok;
                        });
                        for (int i = 0; i < s2->rh.hdr.nbranch; ++i)
                           cow_fix_child_(&s2->branch[i]);
                        *slot = pack_(s2, K::setlist_u8);
                        cow_release(nd);
                        continue;  // re-evaluate as a setlist
                     }
                     catch (...)
                     {
                     }
                  }
                  return false;
               }
            if (k == K::prefix_node || !is_router(k))
               return false;
            router_hdr*    rh = static_cast<router_hdr*>(deref_(nd));
            const unsigned nb = rh->hdr.nbranch;
            const bool     tm = !rh->term.is_null();
            if (nb == 0)
            {
               if (!tm)
               {
                  rm_free_(nd);
                  return true;  // empty shell: cascade
               }
               // Only payload is a key ending HERE: the term handle (an empty-suffix leaf /
               // inline value) replaces the router. An inline prefix (setlist or
               // see-header CL0) is part of that key's path — it must survive as a
               // prefix node above the term.
               if (k == K::setlist_u8 || k == K::hdr)
               {
                  const char* pp = static_cast<const char*>(deref_(nd));
                  const char* pb = k == K::setlist_u8
                                       ? reinterpret_cast<const char*>(
                                             static_cast<const setlist*>(deref_(nd))->prefix)
                                       : pp + WPFX_OFF;
                  if (const unsigned pl =
                          reinterpret_cast<const router_hdr*>(pp)->hdr.prefix_len)
                  {
                     try
                     {
                        *slot = make_prefix(std::string_view(pb, pl), rh->term);
                        rm_free_(nd);
                     }
                     catch (...)
                     {
                     }  // OOM: keep the router
                     return false;
                  }
               }
               *slot = rh->term;
               rm_free_(nd);
               return false;
            }
            // Under COW, an implicit cN/full may share its tail lines with a
            // demoted explicit node: the raw single-child collapse would move
            // the child out without a retain. Route those through the
            // de-widen-to-setlist below (shared-safe); the setlist re-collapses
            // on the next loop iteration.
            const bool cow_dewiden_first =
                cow_enabled && k != K::setlist_u8 && k != K::setlist_u16;
            if (nb == 1 && !tm && !cow_dewiden_first)
            {
               uint8_t  pfx[setlist::PREFIX_CAP + 2];
               unsigned pl = 0;
               if (k == K::setlist_u8)
               {
                  setlist* s = static_cast<setlist*>(deref_(nd));
                  if constexpr (inline_children_ok)
                     if (sl_is_inline(s, 0))
                        return false;  // lone inline survivor: folding needs an alloc (noexcept) — leave it
                  pl         = s->rh.hdr.prefix_len;
                  std::memcpy(pfx, s->prefix, pl);
                  pfx[pl++] = s->bytes[0];
                  rm_collapse_(slot, nd, pfx, pl, s->branch[0]);
               }
               else if (k == K::setlist_u16)
               {
                  setlist16* s = static_cast<setlist16*>(deref_(nd));
                  pfx[pl++]    = uint8_t(s->stems[0] >> 8);
                  pfx[pl++]    = uint8_t(s->stems[0] & 0xFF);
                  rm_collapse_(slot, nd, pfx, pl, s->branch[0]);
               }
               else  // cN/full: fetch the lone child, absorb its byte
               {
                  packed_ptr child = packed_ptr::null();
                  router_for_each(nd, [&](uint8_t bb, packed_ptr br) {
                     pfx[0] = bb;
                     child  = br;
                  });
                  pl = 1;
                  rm_collapse_(slot, nd, pfx, pl, child);
               }
               return false;
            }
            if (nb <= unsigned(SHRINK_MAX) && k != K::setlist_u8 && k != K::setlist_u16)
            {
               try  // sparse dense node: de-widen back to one cacheline (best-effort)
               {
                  setlist* s2         = new_setlist();
                  s2->rh.term         = rh->term;
                  s2->rh.hdr.has_term = rh->hdr.has_term;
                  router_for_each(nd, [&](uint8_t bb, packed_ptr br) {
                     const bool ok = setlist_set(s2, bb, br);
                     assert(ok && "SHRINK_MAX <= setlist CAP");
                     (void)ok;
                  });
                  packed_ptr s2p = pack_(s2, K::setlist_u8);
                  if (k == K::hdr)  // carry the CL0 prefix (inline if <= 8, else
                  {                 // a legitimately-long prefix_node)
                     const char*    hp = static_cast<const char*>(deref_(nd));
                     const unsigned pl =
                         reinterpret_cast<const router_hdr*>(hp)->hdr.prefix_len;
                     if (pl)
                        s2p = apply_prefix(s2, std::string_view(hp + WPFX_OFF, pl));
                  }
                  *slot = s2p;
                  // vpart: nd is a privatized HUSK — its children moved into
                  // s2, so free its lines WITHOUT cascading (under refcounts
                  // this paired fix_child retains against a cascading release;
                  // with stamps there is nothing to pair).
                  rm_free_(nd);
                  continue;  // re-evaluate as a setlist (may now be 1-branch -> collapse)
               }
               catch (...)
               {
               }
            }
            return false;
         }
      }

      // Walk the recorded path upward: drop the freed child's edge from its parent, then
      // shrink each ancestor until one survives. `j` starts at the freed node's parent.
      void rm_unwind_(rm_stack& path, uint32_t n, uint32_t j) noexcept
      {
         (void)n;
         bool unlink = true;
         for (;; --j)
         {
            rm_lvl&          L  = path[j];
            const packed_ptr nd = *L.slot;
            const K          k  = nd.tag();
            if (unlink)
            {
               if (k == K::prefix_node)
               {  // pn lost its only child: it dies too — keep cascading
                  char* P = static_cast<char*>(deref_(nd));
                  node_free_(P, pn_size<Ptr>(pn_plen(P)));
                  if (j == 0) break;
                  continue;
               }
               const uint16_t edge = path[j + 1].edge;
               if (k == K::setlist_u16)
                  sl16_remove(static_cast<setlist16*>(deref_(nd)), edge);
               else
                  router_remove(nd, uint8_t(edge));
               unlink = false;
            }
            if (!rm_shrink_(L.slot))
               return;  // node survived (possibly replaced in *L.slot)
            if (j == 0) break;  // root died
            unlink = true;
         }
         root_ = packed_ptr::null();
      }

     public:
      // Remove `key` if present; returns true iff it was found and removed. The structure
      // SHRINKS on the way out: empty routers/buckets are freed (cascading up), a router
      // whose only payload is its term collapses to the term handle, a single-branch
      // term-less router collapses into its child (edge byte + inline prefix absorbed, a
      // prefix node allocated only when the child can't take them for free), and a sparse
      // dense node (full/cN, <= 12 branches) de-widens back to a setlist. Allocating
      // collapses are best-effort: on bad_alloc the tree simply stays unshrunk. Ancestors
      // beyond the deepest 64 levels are left unshrunk (the window covers any practical key).
      // NOTE: not noexcept — removing a non-POD value from a bucket rebuilds it (allocates),
      // and a custom key_codec may allocate. Nothrow in practice for POD values / radix mode.
      // ── L5 retirement: stamp-bounded subtree detach (Dan's algorithm) ────────
      // COW-to-root invariant: a node's stamp == the max stamp in its subtree
      // (any deeper write would have cloned it on the way up). So retirement
      // never modifies a <=F parent — it DETACHES it whole; only the >F spine
      // (nodes above data written since the floor) is touched, owned once per
      // retire txn, then mutated in place. Cost: O(post-floor spine), not
      // O(keys retired). Three phases so the writer lock is held only for A:
      //   A retire_detach(F)  writer-locked txn: unlink <=F subtrees -> list
      //   B retire_count()    lock-free: count keys in the detached (private) list
      //   C retire_release()  writer-locked txn: deep-log roots; the DRAIN
      //                       thread cascades the frees in the background.
      // Deep-log an externally-held root (the frozen WARM tree after its
      // flush): the drain cascades the whole tree once no version pin
      // precedes the logging txn. Must run inside a transaction.
      void release_root(packed_ptr root)
      {
         if (!root.is_null())
            cow_release(root);
      }
      size_t retire_detach(uint32_t F)
      {
         retire_list_.clear();
         retire_spine_(&root_, F);
         return retire_list_.size();
      }
      uint64_t retire_count() const
      {
         uint64_t n = 0;
         for (packed_ptr p : retire_list_)
            n += count_keys_(p);
         return n;
      }
      void retire_release(uint64_t counted)
      {
         for (packed_ptr p : retire_list_)
            cow_release(p);  // stamp <= floor: deep-logged, drain cascades
         retire_list_.clear();
         count_ -= size_t(counted);
      }

     private:
      std::vector<packed_ptr> retire_list_;  // merger-path scratch (not hot)

      void retire_spine_(packed_ptr* slot, uint32_t F)
      {
         packed_ptr cur = *slot;
         if (cur.is_null() || cur.tag() == K::value_inline)
            return;  // inline terminals carry no stamp: they age out at the next floor
         if (vstamp_get_(cur) <= F)
         {  // the WHOLE subtree precedes the floor: detach, never modify
            retire_list_.push_back(cur);
            *slot = packed_ptr::null();
            return;
         }
         const K k = cur.tag();
         if (k == K::value_ptr)
            return;  // post-floor leaf stays
         // >F spine node: own it once per retire txn, then mutate in place
         if constexpr (cow_stamped_)
            if (alloc_.hkv_vtxn() && cow_shared_(cur))
            {
               cur = cow_clone_into_(slot, cur);
            }
         if (k == K::prefix_node)
         {
            char*       P  = static_cast<char*>(deref_(cur));
            packed_ptr* ns = reinterpret_cast<packed_ptr*>(P + 2);
            retire_spine_(ns, F);
            if (ns->is_null())
            {  // child detached: the (own-era, private) shell goes too
               *slot = packed_ptr::null();
               cow_release(cur);  // shallow husk; nothing below it
            }
            return;
         }
         // routers (hdr resolves to its base shape for dispatch; rh is at CL0)
         router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
         if (!rh->term.is_null() && rh->term.tag() == K::value_ptr &&
             vstamp_get_(rh->term) <= F)
         {
            retire_list_.push_back(rh->term);
            rh->term         = packed_ptr::null();
            rh->hdr.has_term = 0;
         }
         if (k == K::setlist_u8)
         {  // inline children carry no separate stamp — they ride the line and age with it;
            // skip them (never recurse/deref). sl_remove_branch_ keeps the inl bits in sync.
            setlist* s = static_cast<setlist*>(deref_(cur));
            for (int i = 0; i < s->rh.hdr.nbranch;)
            {
               if constexpr (inline_children_ok)
                  if (sl_is_inline(s, i)) { ++i; continue; }
               retire_spine_(&s->branch[i], F);
               if (s->branch[i].is_null())
                  sl_remove_branch_(s, i);  // shifts down; re-check this index
               else
                  ++i;
            }
         }
         else if (k == K::setlist_u16)
         {
            setlist16* sw = static_cast<setlist16*>(deref_(cur));
            uint16_t   dead[setlist16::CAP];
            int        nd = 0;
            for (int i = 0; i < sw->rh.hdr.nbranch; ++i)
            {
               retire_spine_(&sw->branch[i], F);
               if (sw->branch[i].is_null())
                  dead[nd++] = sw->stems[i];
            }
            for (int i = 0; i < nd; ++i)
               sl16_remove(sw, dead[i]);
         }
         else
         {
            uint8_t dead[256];
            int     nd = 0;
            router_for_each(cur, [&](uint8_t b, packed_ptr) { dead[nd++] = b; });
            const int nkids = nd;
            nd              = 0;
            for (int i = 0; i < nkids; ++i)
            {
               const uint8_t b  = dead[i];
               packed_ptr*   cs = router_find_slot(cur, b);
               retire_spine_(cs, F);
               if (cs->is_null())
                  dead[nd++] = b;  // compact the dead bytes in place
            }
            if (k == K::node_full)
            {  // slots were nulled directly; fix the population counters
               node_full* f = static_cast<node_full*>(deref_(cur));
               for (int i = 0; i < nd; ++i)
               {
                  unsigned li, sl;
                  full_index<Ptr>(dead[i], li, sl);
                  --f->ranges[li].nset;
                  --f->rh.hdr.nbranch;
               }
            }
            else
               for (int i = 0; i < nd; ++i)
                  router_remove(cur, dead[i]);
         }
         if (rh->hdr.nbranch == 0 && rh->term.is_null())
         {  // emptied shell (own-era, private): unlink and shallow-release
            *slot = packed_ptr::null();
            cow_release(cur);
         }
      }

      uint64_t count_keys_(packed_ptr cur) const
      {
         if (cur.is_null())
            return 0;
         switch (cur.tag())
         {
            case K::value_ptr:
            case K::value_inline:
               return 1;
            case K::prefix_node:
               return count_keys_(pn_next<Ptr>(static_cast<char*>(deref_(cur))));
            default:
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
               uint64_t          n  = rh->term.is_null() ? 0 : 1;
               router_for_each(cur, [&](uint8_t, packed_ptr br) { n += count_keys_(br); });
               return n;
            }
         }
      }

     public:
      bool remove(key_param k)
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         rm_stack         path;
         uint32_t         n    = 0;  // levels recorded (path[0] = root)
         packed_ptr*      slot = &root_;
         size_t           depth = 0;
         path.ensure(n + 1);
         path[n++] = {&root_, 0};
         auto free_leaf         = [&](char* L) {
            if constexpr (!raw_vals)
               v_destroy(alloc_, leaf_val(L));
            vh_leaf_dies_(L);
            node_free_(L, leaf_alloc_size(L));
         };
         // BOTTOM-UP COW (Dan): descend READ-ONLY; own_rm_ clones the shared
         // suffix of the recorded path only when the key is actually FOUND —
         // misses clone nothing. Slot fix-up by offset (clone = memcpy).
         auto own_rm_ = [&]()
         {
            if constexpr (cow_stamped_)
               if (alloc_.hkv_vtxn())
               {
                  uint32_t i = 0;
                  while (i < n && !((*path[i].slot).tag() != K::value_inline &&
                                    cow_shared_(*path[i].slot)))
                     ++i;
                  for (; i < n; ++i)
                  {
                     const packed_ptr old = *path[i].slot;
                     if (old.is_null() || old.tag() == K::value_inline)
                        break;
                     char*            ob = static_cast<char*>(deref_(old));
                     const packed_ptr nn = cow_clone_into_(path[i].slot, old);
                     if (i + 1 < n)
                        path[i + 1].slot = reinterpret_cast<packed_ptr*>(
                            static_cast<char*>(deref_(nn)) +
                            (reinterpret_cast<char*>(path[i + 1].slot) - ob));
                  }
                  slot = path[n - 1].slot;
               }
         };
         for (;;)
         {
            packed_ptr cur = *slot;
            switch (cur.tag())
            {
               case K::null:
                  return false;
               case K::value_ptr:
               {
                  char* L = static_cast<char*>(deref_(cur));
                  if (std::string_view(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L)) !=
                      key.substr(depth))
                     return false;
                  own_rm_();  // found: own the whole path, then unlink/shrink
                  L = static_cast<char*>(deref_(*slot));
                  free_leaf(L);
                  --count_;
                  if (n >= 2) rm_unwind_(path, n, n - 2);  // n-2: the leaf's parent
                  else root_ = packed_ptr::null();         // the leaf WAS the root
                  return true;
               }
               case K::value_inline:
               {
                  if (depth != key.size())
                     return false;  // inline terminal matches only at key end; nothing to free
                  own_rm_();  // the unwind mutates the recorded ancestors
                  --count_;
                  if (n >= 2) rm_unwind_(path, n, n - 2);
                  else root_ = packed_ptr::null();
                  return true;
               }
               case K::bucket:
               {
                  if constexpr (Buckets || Adaptive)
                  {
                     packed_ptr nb = cur;
                     if (!bkt_remove_entry(nb, key.substr(depth)))  // may rebuild (non-POD)
                        return false;
                     --count_;
                     bucket* b = static_cast<bucket*>(deref_(nb));
                     if (b->nent == 0)
                     {
                        node_free_(b, sizeof(bucket));
                        if (n >= 2) rm_unwind_(path, n, n - 2);
                        else root_ = packed_ptr::null();
                     }
                     else if (nb.raw() != cur.raw())
                        *slot = nb;  // rebuilt → update the slot
                     return true;
                  }
                  return false;
               }
               case K::prefix_node:
               {
                  char*          P  = static_cast<char*>(deref_(cur));
                  const uint16_t pl = pn_plen(P);
                  if (key.size() - depth < pl || std::memcmp(pn_pfx<Ptr>(P), key.data() + depth, pl) != 0)
                     return false;
                  slot = reinterpret_cast<packed_ptr*>(P + 2);
                  path.ensure(n + 1);
                  path[n++] = {slot, 0};
                  depth += pl;
                  continue;
               }
               case K::setlist_u16:
               {
                  setlist16* s = static_cast<setlist16*>(deref_(cur));
                  if (key.size() - depth < 2)
                  {
                     if (depth != key.size())
                        return false;
                     goto rm_term;
                  }
                  {
                     const uint16_t q  = two_stem(key, depth);
                     packed_ptr*    cs = sl16_find_slot(s, q);
                     if (!cs)
                        return false;
                     slot               = cs;
                     path.ensure(n + 1);
                     path[n++] = {slot, q};
                     depth += 2;
                  }
                  continue;
               }
               case K::cbucket:
               {
                  if constexpr (cb_enabled)
                  {
                     cb_hdr*        h  = static_cast<cb_hdr*>(deref_(*slot));
                     const unsigned pl = h->rh.hdr.prefix_len;
                     if (pl)
                     {
                        if (key.size() - depth < pl ||
                            std::memcmp(h->prefix, key.data() + depth, pl) != 0)
                           return false;
                        depth += pl;
                     }
                     if (depth == key.size())
                        goto rm_term;
                     if (!cb_remove_(h, key.substr(depth)))
                        return false;
                     --count_;
                     if (h->rh.hdr.nbranch == 0 && h->rh.term.is_null())
                     {  // emptied: release line-wise and cascade up
                        cow_release(*slot);
                        if (n >= 2)
                           rm_unwind_(path, n, n - 2);
                        else
                           root_ = packed_ptr::null();
                     }
                     return true;
                  }
                  else
                     return false;
               }
               case K::setlist_u8:
               case K::c2:
               case K::c4:
               case K::c8:
               case K::node_full:
               case K::hdr:     // prefixed wide router (see-header)
               case K::full_x:  // explicit kinds: same descent via the slot
               case K::cx:      // dispatchers; rm_shrink_ leaves them unshrunk
               {
                  if (const K rk = cur.tag(); rk == K::setlist_u8 || rk == K::hdr)
                  {
                     const char*    pp = static_cast<const char*>(deref_(cur));
                     const unsigned pl =
                         reinterpret_cast<const router_hdr*>(pp)->hdr.prefix_len;
                     if (pl)
                     {
                        const char* pb =
                            rk == K::setlist_u8
                                ? reinterpret_cast<const char*>(
                                      static_cast<const setlist*>(deref_(cur))->prefix)
                                : pp + WPFX_OFF;
                        if (key.size() - depth < pl ||
                            std::memcmp(pb, key.data() + depth, pl) != 0)
                           return false;
                        depth += pl;
                     }
                  }
                  if (depth == key.size())
                     goto rm_term;
                  {
                     const uint8_t byte = uint8_t(key[depth]);
                     if constexpr (inline_children_ok)
                        if (cur.tag() == K::setlist_u8)
                        {
                           setlist*  s = static_cast<setlist*>(deref_(cur));
                           const int i = setlist_index(s, byte);
                           if (i >= 0 && sl_is_inline(s, i))
                           {  // target leaf rides this line: drop the branch, no node to free
                              const char* L = sl_inline_leaf(s, i);
                              if (std::string_view(reinterpret_cast<const char*>(leaf_suf(L)),
                                                   leaf_slen(L)) != key.substr(depth + 1))
                                 return false;  // key not present
                              own_rm_();  // own the path (incl. this setlist); leaf rides the clone
                              setlist*  os = static_cast<setlist*>(deref_(*slot));
                              const int oi = setlist_index(os, byte);
                              vh_leaf_dies_(sl_inline_leaf(os, oi));
                              sl_remove_branch_(os, oi);
                              --count_;
                              if (rm_shrink_(slot))  // emptied/collapsed: unlink + cascade up
                              {
                                 if (n >= 2) rm_unwind_(path, n, n - 2);
                                 else        root_ = packed_ptr::null();
                              }
                              return true;
                           }
                        }
                     packed_ptr* cs = router_find_slot(cur, byte);
                     if (!cs)
                        return false;
                     slot               = cs;
                     path.ensure(n + 1);
                     path[n++] = {slot, byte};
                     ++depth;
                  }
                  continue;
               }
            }
            __builtin_unreachable();
         rm_term:  // the key ends AT this router: drop its term, then shrink from this level
         {
            own_rm_();  // found: own the path before the term drop + shrink
            router_hdr* rh = static_cast<router_hdr*>(deref_(*slot));
            if (rh->term.is_null())
               return false;
            if (rh->term.tag() == K::value_ptr)  // leaf("",v); an inline term has nothing to free
               free_leaf(static_cast<char*>(deref_(rh->term)));
            rh->term         = packed_ptr::null();
            rh->hdr.has_term = 0;
            --count_;
            if (rm_shrink_(slot))  // emptied outright: unlink from the parent and cascade
            {
               if (n >= 2) rm_unwind_(path, n, n - 2);
               else root_ = packed_ptr::null();  // the root router emptied
            }
            return true;
         }
         }
      }

      size_t size() const noexcept { return count_; }
      bool   empty() const noexcept { return count_ == 0; }
      size_t max_size() const noexcept { return ~size_t(0); }

      // Remove all elements: destroy every T and free every node.
      void clear() noexcept
      {
         free_all_(root_);
         root_  = packed_ptr::null();
         count_ = 0;
      }
      void swap(kv_tree& o) noexcept(std::allocator_traits<Allocator>::is_always_equal::value)
      {
         using std::swap;
         // Allocators exchange iff they propagate on swap; otherwise they must be equal
         // (UB if not, per the standard), so leaving them is correct for the valid cases.
         if constexpr (std::allocator_traits<Allocator>::propagate_on_container_swap::value)
            swap(alloc_, o.alloc_);
         rebase_();
         o.rebase_();
         swap(root_, o.root_);
         swap(count_, o.count_);
      }
      // Non-member ADL swap (hidden friend) — found by generic code / std::ranges::swap.
      friend void swap(kv_tree& a, kv_tree& b) noexcept(noexcept(a.swap(b))) { a.swap(b); }

      // Debug: node-type census + avg terminal depth (hops). Reveals how many
      // prefix_node hops the descent pays vs router/leaf nodes.
      struct dbg_counts
      {
         uint64_t leaf = 0, inl = 0, prefix = 0, setlist = 0, setlist16 = 0, c2 = 0, c4 = 0, c8 = 0, full = 0;
         uint64_t terminals = 0, sum_depth = 0;
         uint64_t pfx_le4 = 0, pfx_le8 = 0, pfx_le16 = 0, pfx_sum = 0;
         // Wide-stem opportunity. router_hops = sum over terminals of routers on its path
         // (≈ cacheline misses/lookup). collapsible = router nodes whose parent is also a
         // term-less router → a u16 node could fuse the two levels, saving 1 hop for every
         // key beneath. collapse_saved weights that by the terminals under each fused edge:
         // the predicted total router-hops removed if every eligible pair were fused.
         uint64_t router_hops = 0, collapsible = 0, collapse_saved = 0;
      };
      // Returns the number of terminals in cur's subtree (used to weight collapse_saved).
      // routers_above = routers already on the path; parent_noterm_router = the parent is a
      // term-less router (so a u16 node could fuse this router into it).
      uint64_t dbg_walk(packed_ptr cur, dbg_counts& d, int depth, int routers_above,
                        bool parent_noterm_router) const
      {
         if (cur.is_null())
            return 0;
         switch (cur.tag())
         {
            case K::value_ptr:    ++d.leaf; ++d.terminals; d.sum_depth += depth;
                                  d.router_hops += routers_above; return 1;
            case K::value_inline: ++d.inl;  ++d.terminals; d.sum_depth += depth;
                                  d.router_hops += routers_above; return 1;
            case K::prefix_node:
            {
               ++d.prefix;
               const uint16_t pl = pn_plen(static_cast<char*>(deref_(cur)));
               d.pfx_sum += pl;
               d.pfx_le4 += (pl <= 4);
               d.pfx_le8 += (pl <= 8);
               d.pfx_le16 += (pl <= 16);
               // A prefix_node breaks a router→router chain (the child can't fuse into the
               // parent across a multi-byte prefix), so parent_noterm_router resets to false.
               return dbg_walk(pn_next<Ptr>(static_cast<char*>(deref_(cur))), d, depth + 1,
                               routers_above, false);
            }
            case K::setlist_u8:   ++d.setlist; break;
            case K::setlist_u16:  ++d.setlist16; break;
            case K::c2:           ++d.c2; break;
            case K::c4:           ++d.c4; break;
            case K::c8:           ++d.c8; break;
            case K::node_full:    ++d.full; break;
            case K::bucket:       return 0;  // census doesn't descend buckets
            case K::null:         return 0;  // exhaustive (no default)
         }
         const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
         const bool        noterm = rh->term.is_null();
         if (parent_noterm_router) ++d.collapsible;
         uint64_t sub = dbg_walk(rh->term, d, depth + 1, routers_above + 1, false);
         router_for_each(cur, [&](uint8_t, packed_ptr br) {
            sub += dbg_walk(br, d, depth + 1, routers_above + 1, noterm);
         });
         // If this router fuses up into its (term-less) parent, every terminal beneath it
         // loses one router hop.
         if (parent_noterm_router) d.collapse_saved += sub;
         return sub;
      }
      dbg_counts debug_stats() const { dbg_counts d; dbg_walk(root_, d, 0, 0, false); return d; }

     private:
      // ── leaf<T> layout: [u16 suffix_len][u16 pfx_field][pad to alignof(T)][T value][key bytes] ─
      // pfx_field = (has_full << 15) | prefix_len. The trailing key bytes are either the
      // suffix only (has_full=0, the compressed form) OR the WHOLE key = prefix ++ suffix
      // laid out contiguously (has_full=1) — stored opportunistically when the prefix fits
      // in the 128-byte slot we'd allocate anyway, so the iterator can hand back the full
      // key as a zero-copy string_view (same cacheline as the value). leaf_suf() points at
      // the suffix in both forms, so point lookup is identical.
      static constexpr uint16_t LEAF_FULL = 0x8000;
      static constexpr size_t   leaf_value_off() noexcept
      {
         if constexpr (raw_vals)
            return 4 + 3 * sizeof(uint16_t);  // [vstamp][slen][pfx_field][vlen]
         else
         {
            constexpr size_t a = alignof(T);
            return (4 + 2 * sizeof(uint16_t) + a - 1) / a * a;  // [vstamp][slen][pfx]
         }
      }
      static size_t leaf_size(size_t strlen, size_t vlen = sizeof(T)) noexcept
      {
         if constexpr (raw_vals)
            return leaf_value_off() + vlen + strlen;
         else
            return leaf_value_off() + sizeof(T) + strlen;
      }
      static uint16_t leaf_vlen(const char* p) noexcept
      {
         if constexpr (raw_vals)
         {
            uint16_t v;
            std::memcpy(&v, p + 8, 2);
            return v;
         }
         else
            return uint16_t(sizeof(T));
      }
      static std::string_view leaf_bytes(const char* p) noexcept
         requires raw_vals
      {
         return std::string_view(p + leaf_value_off(), leaf_vlen(p));
      }
      static uint16_t leaf_slen(const char* p) noexcept { uint16_t v; std::memcpy(&v, p + 4, 2); return v; }
      static uint16_t leaf_pfxfield(const char* p) noexcept { uint16_t v; std::memcpy(&v, p + 6, 2); return v; }
      static bool     leaf_has_full(const char* p) noexcept { return leaf_pfxfield(p) & LEAF_FULL; }
      static uint16_t leaf_plen(const char* p) noexcept { return leaf_pfxfield(p) & 0x7FFF; }
      static uint16_t leaf_strlen(const char* p) noexcept
      {
         const uint16_t f = leaf_pfxfield(p);
         return uint16_t(leaf_slen(p) + ((f & LEAF_FULL) ? (f & 0x7FFF) : 0));
      }
      static size_t leaf_alloc_size(const char* p) noexcept
      {
         return leaf_size(leaf_strlen(p), leaf_vlen(p));
      }
      static T* leaf_val(char* p) noexcept
         requires(!raw_vals)
      {
         return reinterpret_cast<T*>(p + leaf_value_off());
      }
      static const T* leaf_val(const char* p) noexcept
         requires(!raw_vals)
      {
         return reinterpret_cast<const T*>(p + leaf_value_off());
      }
      static char* leaf_str(char* p) noexcept
      {
         return p + leaf_value_off() + leaf_vlen(p);
      }
      static const char* leaf_str(const char* p) noexcept
      {
         return p + leaf_value_off() + leaf_vlen(p);
      }
      // Full key as a contiguous view — valid ONLY when leaf_has_full(p).
      static std::string_view leaf_fullview(const char* p) noexcept
      {
         return std::string_view(leaf_str(p), leaf_strlen(p));
      }

      // ── setlist inline children (libart++ `inl` byte, vpart-adapted) ─────────────
      // A setlist_u8 can host small leaf children INSIDE its own 128 B line instead of
      // as separate value_ptr terminals. inl bit7 = inline-active, bits0-6 flag branch
      // 0..6. An inline branch's handle keeps tag K::value_ptr but its address bits hold
      // a LINE OFFSET (the leaf starts at line_base + (raw>>4)) — NEVER an arena address,
      // so the audit rule is: an inline branch is resolved ONLY via sl_inline_leaf()
      // (parent in hand); deref_/vstamp_get_/cow_release/node_free_ must never touch it.
      //
      // vpart specifics libart++ lacks: (1) the setlist carries a version stamp at line
      // offset vstamp_off_(setlist_u8); inline payloads pack DOWN from SL_INLINE_TOP (the
      // stamp slot), not 128, so they never collide with it. (2) The minimal COW unit IS
      // the cacheline: updating an inline value clones the line (carrying the bytes) and
      // overwrites in place — no externalize. Externalize only when a leaf no longer fits
      // (value grows past the region, or a structural split/widen needs the slot).
      // (3) Typed-value hooks: a line CLONE duplicates inline bytes -> vh_copy_ each; a
      // line DEATH drops them -> vh_release_ each; a same-scope MOVE (compact/externalize)
      // fires neither. Only raw_vals leaves carry hooks; trivially-copyable T is POD.
      // Master switch for the inline-children feature. WIP: with it ON the read/creation/
      // write paths are correct AND ~20-40% faster (start_ver-tests fully green, kv-compat
      // fastest in every phase), but there is ONE open bug — under kv-compat's heavy
      // update-all-then-remove-half battery, removing an externalized key can drop its
      // externalized sibling, exposed only by flush_releases() (a stamp/drain supersession
      // interaction that inline creation sets up; repro: removing "str/31510" drops
      // "str/31514"). Default OFF so the suite stays green; flip to true to develop the
      // feature / reproduce the bug. Type gate unchanged (raw_vals || trivially-copyable T).
      static constexpr bool     kInlineChildren    = false;
      static constexpr bool     inline_children_ok =
          kInlineChildren && (raw_vals || std::is_trivially_copyable_v<T>);
      static constexpr unsigned SL_LINE          = cacheline_bytes;  // 128
      static constexpr int      SL_INLINE_BRANCH = 7;  // inl bits 0..6 cover branches 0..6
      static constexpr unsigned sl_branch_off =
          unsigned(sizeof(router_hdr_t<packed_ptr>)) + 1u + setlist::LANES + setlist::PREFIX_CAP;
      // Inline payloads must stop below the setlist's version-stamp slot (the 4 B line_ptr
      // COW instantiation stamps at offset 96); a stampless instantiation uses the full tail.
      // A function (not a data member): cow_stamped_ is declared later in the class, and a
      // function body is parsed after the class is complete — a class-scope initializer is not.
      // Inline payloads must stay within (a) the bytes cow_clone_node_ memcpys, which is
      // node_size(setlist_u8) == sizeof(setlist) (the 4 B line_ptr struct is only 94 B though
      // it lives in a 128 B block), and (b) below the version-stamp slot. sizeof is the tighter
      // bound on the COW path; take the min so a payload is never truncated on clone.
      static constexpr unsigned sl_inline_top() noexcept
      {
         // The clone copies sizeof(setlist) bytes; inline payloads must fit within that AND
         // below the version stamp (min handles both — sizeof < stamp on the 4 B COW handle).
         const unsigned copied = unsigned(sizeof(setlist));
         const unsigned stamp  = cow_stamped_ ? unsigned(vstamp_off_(K::setlist_u8)) : SL_LINE;
         return copied < stamp ? copied : stamp;
      }
      // Inline leaf start aligns so leaf_value_off() lands on alignof(T). For raw_vals the
      // value is RAW BYTES (leaf_bytes, offset 10, no object alignment) — align 1, else the
      // alignof(bytes_value)=8 padding wastes the small inline region and blocks compaction
      // (2 typical leaves no longer fit), making the whole feature dormant.
      static constexpr unsigned SL_VAL_ALIGN =
          raw_vals ? 1u : (alignof(T) < 1 ? 1u : unsigned(alignof(T)));
      // Inline leaves need no OWN version stamp — the setlist's stamp governs their liveness
      // (they are never a separate COW unit). For raw_vals strip the leading 4 B vstamp:
      // store only [slen..suffix] and hand accessors a LOGICAL pointer 4 B before the stored
      // bytes. The phantom [logical, logical+4) is never read for an inline leaf; externalize's
      // memcpy(full sz)+pack_ re-materializes a real stamp there. +4 B of region per leaf.
      static constexpr unsigned SL_VSTAMP = raw_vals ? 4u : 0u;

      static bool sl_inl_active(const setlist* s) noexcept { return s->inl & 0x80; }
      static bool sl_is_inline(const setlist* s, int i) noexcept
      {
         return (s->inl & 0x80) && i < SL_INLINE_BRANCH && (s->inl & (1u << i));
      }
      static packed_ptr sl_inline_handle(unsigned line_off) noexcept
      {
         packed_ptr p;
         std::memset(p.b, 0, sizeof(packed_ptr));
         const uint64_t raw = (uint64_t(line_off) << 4) | uint8_t(K::value_ptr);
         std::memcpy(p.b, &raw, sizeof(packed_ptr));
         return p;
      }
      static unsigned sl_inline_off(packed_ptr p) noexcept { return unsigned(p.raw() >> 4); }
      // Resolve inline branch i's leaf bytes. `s` IS the 128 B line base (deref_ of the
      // setlist), so the leaf lives at s + line_off. Caller guarantees sl_is_inline(s, i).
      char* sl_inline_leaf(setlist* s, int i) const noexcept
      {
         return reinterpret_cast<char*>(s) + sl_inline_off(s->branch[i]);
      }
      const char* sl_inline_leaf(const setlist* s, int i) const noexcept
      {
         return reinterpret_cast<const char*>(s) + sl_inline_off(s->branch[i]);
      }
      static unsigned sl_used_branch_end(const setlist* s) noexcept
      {
         return sl_branch_off + unsigned(s->rh.hdr.nbranch) * unsigned(sizeof(packed_ptr));
      }
      // Typed-value hooks for line clone (each inline leaf's bytes are duplicated) and line
      // death (each dies). No-ops unless vh_on (raw_vals + registered allocator).
      void sl_inline_vh_copy_(const setlist* s) noexcept
      {
         if constexpr (vh_on)
            if (sl_inl_active(s))
            {
               const int n = std::min<int>(s->rh.hdr.nbranch, SL_INLINE_BRANCH);
               for (int i = 0; i < n; ++i)
                  if (sl_is_inline(s, i))
                     vh_copy_(leaf_bytes(sl_inline_leaf(s, i)));
            }
      }
      void sl_inline_vh_dies_(const setlist* s) noexcept
      {
         if constexpr (vh_on)
            if (sl_inl_active(s))
            {
               const int n = std::min<int>(s->rh.hdr.nbranch, SL_INLINE_BRANCH);
               for (int i = 0; i < n; ++i)
                  if (sl_is_inline(s, i))
                     vh_release_(leaf_bytes(sl_inline_leaf(s, i)));
            }
      }
      // Re-pack inline payloads contiguously against SL_INLINE_TOP in ascending branch
      // order (canonical). Allocation-free; trivially-/raw-copyable payloads move by memcpy
      // and the new positions never overlap the (smaller-or-equal) used branch array.
      void sl_repack_(setlist* s) const noexcept
      {
         if (!sl_inl_active(s)) return;
         const int n = std::min<int>(s->rh.hdr.nbranch, SL_INLINE_BRANCH);
         char      tmp[SL_LINE];
         unsigned  pos = sl_inline_top(), newoff[SL_INLINE_BRANCH];
         for (int i = 0; i < n; ++i)
            if (sl_is_inline(s, i))
            {
               const char*    lf   = sl_inline_leaf(s, i);          // logical (4 B before physical)
               const unsigned sz   = unsigned(leaf_alloc_size(lf)); // full size (incl vstamp)
               const unsigned phys = sz - SL_VSTAMP;                // stored bytes
               pos = (pos - phys) & ~(SL_VAL_ALIGN - 1);            // physical start
               std::memcpy(tmp + pos, lf + SL_VSTAMP, phys);        // lf+SL_VSTAMP = physical start
               newoff[i] = pos - SL_VSTAMP;                         // logical offset
            }
         std::memcpy(reinterpret_cast<char*>(s) + pos, tmp + pos, sl_inline_top() - pos);
         for (int i = 0; i < n; ++i)
            if (sl_is_inline(s, i)) s->branch[i] = sl_inline_handle(newoff[i]);
      }
      // Pull external leaf children INTO the line, all-or-nothing: allocation-free (frees
      // the externals — a same-scope MOVE, so no vh hooks), canonical by construction.
      // The creation workhorse — called once after building/growing a small setlist (split).
      void sl_inline_compact_(setlist* s) noexcept
      {
         if constexpr (!inline_children_ok) { (void)s; return; }
         else
         {
            const int n = std::min<int>(s->rh.hdr.nbranch, SL_INLINE_BRANCH);
            char      tmp[SL_LINE];
            unsigned  pos = sl_inline_top(), newoff[SL_INLINE_BRANCH];
            uint8_t   newinl = 0x80;
            bool      any = false, was_ext[SL_INLINE_BRANCH] = {};
            for (int i = 0; i < n; ++i)
            {
               const char* src = nullptr;
               if (sl_is_inline(s, i)) src = sl_inline_leaf(s, i);
               else if (s->branch[i].tag() == K::value_ptr)
               {
                  // Only WINDOW-BORN external leaves move inline (a MOVE we own). A SHARED leaf
                  // (committed, a snapshot/old generation holds it) stays external — it's carried
                  // by this clone for COW supersession; dropping it has no refcount under stamp
                  // MVCC and would double-free. (Non-COW: everything is window-born.)
                  bool window_born = true;
                  if constexpr (cow_enabled)
                     window_born = !alloc_.hkv_vtxn() || !cow_shared_(s->branch[i]);
                  if (window_born)
                  {
                     src        = static_cast<const char*>(deref_(s->branch[i]));
                     was_ext[i] = true;
                  }
               }
               if (!src) continue;
               const unsigned sz   = unsigned(leaf_alloc_size(src));  // full leaf size (incl vstamp)
               const unsigned phys = sz - SL_VSTAMP;                  // stored bytes (vstamp stripped)
               if (phys > pos) return;  // too big for the region — all-or-nothing bail
               const unsigned np = (pos - phys) & ~(SL_VAL_ALIGN - 1);  // physical start
               if (np < sl_used_branch_end(s)) return;  // would overlap the branch array
               pos = np;
               std::memcpy(tmp + np, src + SL_VSTAMP, phys);  // skip the source's vstamp
               newoff[i] = np - SL_VSTAMP;                    // LOGICAL offset (4 B before physical)
               newinl |= uint8_t(1u << i);
               any = true;
            }
            if (!any) return;
            for (int i = 0; i < n; ++i)
               if (was_ext[i])  // only window-born husks reach here — MOVE: free, no vh hook
                  node_free_(deref_(s->branch[i]), leaf_alloc_size(static_cast<const char*>(deref_(s->branch[i]))));
            std::memcpy(reinterpret_cast<char*>(s) + pos, tmp + pos, sl_inline_top() - pos);
            for (int i = 0; i < n; ++i)
               if (newinl & (1u << i)) s->branch[i] = sl_inline_handle(newoff[i]);
            s->inl = newinl;
         }
      }
      // CHURN-FREE grow: add a new child (byte -> the window-born external leaf `lf`) to the
      // OWNED setlist `s`, keeping it inline, with NO externalize and NO O(n) re-alloc. The
      // existing inline payloads DON'T move — only the branch/byte/inl arrays shift and the new
      // leaf drops into the gap (the unused branch-pointer slots between used_branch_end and the
      // lowest existing payload). Returns true and frees lf if it fit inline; false otherwise
      // (caller falls back to externalize + external setlist_set / widen). raw_vals only (the
      // stripped-vstamp logical-pointer scheme); typed-T routes through compact instead.
      [[maybe_unused]] bool sl_try_inline_add_(setlist* s, uint8_t byte, packed_ptr lf)
      {
         if constexpr (!inline_children_ok) { (void)s; (void)byte; (void)lf; return false; }
         else
         {
            const int K = s->rh.hdr.nbranch;
            if (K >= setlist::CAP) return false;  // branch array full → caller widens
            const unsigned new_used = sl_branch_off + unsigned(K + 1) * unsigned(sizeof(packed_ptr));
            unsigned       lowest   = sl_inline_top();  // lowest existing inline payload (physical)
            for (int i = 0; i < K; ++i)
               if (sl_is_inline(s, i))
               {
                  const unsigned off = sl_inline_off(s->branch[i]) + SL_VSTAMP;  // physical start
                  if (off < lowest) lowest = off;
               }
            // The new branch slot consumes the BOTTOM 4 B of the gap [used,lowest); if the inline
            // payloads have grown down to block even that, bail (caller externalizes + widens —
            // happens at most once per setlist, near full, so no oscillating churn).
            if (lowest < new_used) return false;
            // Commit IN PLACE: insert byte at its sorted slot; existing payloads never move.
            const int idx = setlist_lb(s, byte);
            std::memmove(&s->bytes[idx + 1], &s->bytes[idx], size_t(K - idx));
            std::memmove(&s->branch[idx + 1], &s->branch[idx], size_t(K - idx) * sizeof(packed_ptr));
            const uint8_t flags = uint8_t(s->inl & 0x7F);  // shift inl bits up across the insert
            const uint8_t lo    = uint8_t(flags & ((1u << idx) - 1));
            const uint8_t hi    = uint8_t((flags << 1) & ~((1u << (idx + 1)) - 1) & 0x7F);
            const char*    lsrc = static_cast<const char*>(deref_(lf));
            const unsigned phys = unsigned(leaf_alloc_size(lsrc)) - SL_VSTAMP;
            const unsigned np   = (phys <= lowest) ? ((lowest - phys) & ~(SL_VAL_ALIGN - 1)) : 0;
            s->bytes[idx] = byte;
            if (idx < SL_INLINE_BRANCH && phys <= lowest && np >= new_used)
            {  // the new leaf fits the gap (above the new slot) — inline it, free the external husk
               std::memcpy(reinterpret_cast<char*>(s) + np, lsrc + SL_VSTAMP, phys);
               s->branch[idx] = sl_inline_handle(np - SL_VSTAMP);
               s->inl         = uint8_t(0x80 | lo | hi | (1u << idx));
               node_free_(const_cast<char*>(lsrc), leaf_alloc_size(lsrc));
            }
            else
            {  // payload won't fit (or idx>=7): keep it EXTERNAL in place — do NOT dissolve the
               // other inline children (that's what oscillated into an alloc livelock).
               s->branch[idx]      = lf;
               const uint8_t bits  = uint8_t(lo | hi);
               s->inl              = bits ? uint8_t(0x80 | bits) : 0;
            }
            s->rh.hdr.nbranch = uint16_t(K + 1);
            return true;
         }
      }
      // Move one inline child OUT to a real value_ptr leaf (throws on alloc). A same-scope
      // MOVE — no vh hooks. pack_ refreshes the externalized leaf's own version stamp.
      void sl_externalize_(setlist* s, int i)
      {
         char*          lf  = sl_inline_leaf(s, i);
         const unsigned sz  = unsigned(leaf_alloc_size(lf));
         char*          ext = static_cast<char*>(node_alloc_(sz));  // throws
         std::memcpy(ext, lf, sz);
         s->branch[i] = pack_(ext, K::value_ptr);
         s->inl &= uint8_t(~(1u << i));
         if (!(s->inl & 0x7F)) s->inl = 0;
         else sl_repack_(s);
      }
      void sl_externalize_all_(setlist* s)  // basic guarantee: a partial result is valid
      {
         if (!sl_inl_active(s)) return;

         const int n = std::min<int>(s->rh.hdr.nbranch, SL_INLINE_BRANCH);
         for (int i = 0; i < n; ++i)
            if (sl_is_inline(s, i))
            {
               char*          lf  = sl_inline_leaf(s, i);
               const unsigned sz  = unsigned(leaf_alloc_size(lf));
               char*          ext = static_cast<char*>(node_alloc_(sz));  // throws
               std::memcpy(ext, lf, sz);
               s->branch[i] = pack_(ext, K::value_ptr);
               s->inl &= uint8_t(~(1u << i));
            }
         s->inl = 0;
      }
      // Drop branch i, shifting stems/handles AND the inl bits down; reclaims an inline
      // payload if branch i had one. Allocation-free — the inline-aware setlist_remove.
      void sl_remove_branch_(setlist* s, int i) const noexcept
      {
         const int n = s->rh.hdr.nbranch;
         std::memmove(&s->bytes[i], &s->bytes[i + 1], size_t(n - 1 - i));
         std::memmove(&s->branch[i], &s->branch[i + 1], size_t(n - 1 - i) * sizeof(packed_ptr));
         s->bytes[n - 1]   = 0xFF;
         s->rh.hdr.nbranch = uint16_t(n - 1);
         if (s->inl & 0x80)
         {
            const uint8_t flags = uint8_t(s->inl & 0x7F);  // drop active bit before the shift
            const uint8_t lo    = uint8_t(flags & ((1u << i) - 1));
            const uint8_t hi    = uint8_t((flags >> 1) & ~((1u << i) - 1) & 0x7F);
            const uint8_t bits  = uint8_t(lo | hi);
            s->inl = bits ? uint8_t(0x80 | bits) : 0;
            sl_repack_(s);
         }
      }

      // Invoke f(key, value) for a terminal pointer (value_ptr leaf or value_inline).
      template <class F>
      void call_val_(std::string_view key, packed_ptr p, F& f) const
      {
         if constexpr (inlineable)
            if (p.tag() == K::value_inline)
            {
               T tmp;
               unpack_inline(p, tmp);
               f(key, static_cast<const T&>(tmp));
               return;
            }
         f(key, static_cast<const T&>(*leaf_val(static_cast<char*>(deref_(p)))));
      }

      // Value of a terminal pointer (value_ptr leaf or value_inline), into tmp if inline.
      const T* val_of_(packed_ptr p, T& tmp) const
      {
         if constexpr (inlineable)
            if (p.tag() == K::value_inline) { unpack_inline(p, tmp); return &tmp; }
         return leaf_val(static_cast<char*>(deref_(p)));
      }
      // Value-only DFS (no key assembly) backing for_each_value.
      template <class F>
      void walk_v_(packed_ptr cur, F& f) const
      {
         const K k = cur.tag();
         switch (k)
         {
            case K::value_ptr:
               f(static_cast<const T&>(*leaf_val(static_cast<char*>(deref_(cur)))));
               return;
            case K::value_inline:
               if constexpr (inlineable) { T tmp; unpack_inline(cur, tmp); f(static_cast<const T&>(tmp)); }
               return;
            case K::prefix_node:
               walk_v_(pn_next<Ptr>(static_cast<char*>(deref_(cur))), f);
               return;
            case K::bucket:
               if constexpr (bucketable)
               {
                  bucket* bk = static_cast<bucket*>(deref_(cur));
                  for (int i = 0; i < bk->nent; ++i)  // sum/visit order within bucket unspecified here
                     f(static_cast<const T&>(*bkt_val(bkt_entry(bk, i))));
               }
               return;
            default:
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
               if (!rh->term.is_null()) { T tmp; f(*val_of_(rh->term, tmp)); }
               auto recurse = [&](uint8_t, packed_ptr ch) { walk_v_(ch, f); };
               switch (k)
               {
                  case K::setlist_u8:
                  {
                     const setlist* s = static_cast<const setlist*>(deref_(cur));
                     const int      n = s->rh.hdr.nbranch;
                     for (int i = 0; i < n; ++i)
                     {
                        if constexpr (inline_children_ok)
                           if (sl_is_inline(s, i))  // leaf rides the line
                           {
                              const char* L = sl_inline_leaf(s, i);
                              if constexpr (raw_vals) { T tmp{leaf_bytes(L)}; f(static_cast<const T&>(tmp)); }
                              else                      f(static_cast<const T&>(*leaf_val(L)));
                              continue;
                           }
                        walk_v_(s->branch[i], f);
                     }
                     break;
                  }
                  case K::c2: cnode_for_each(static_cast<const cnode<2>*>(deref_(cur)), recurse); break;
                  case K::c4: cnode_for_each(static_cast<const cnode<4>*>(deref_(cur)), recurse); break;
                  case K::c8: cnode_for_each(static_cast<const cnode<8>*>(deref_(cur)), recurse); break;
                  default:
                  {
                     const node_full* fn = static_cast<const node_full*>(deref_(cur));
                     for (int byte = 0; byte < 256; ++byte)
                     {
                        packed_ptr ch = full_find(fn, uint8_t(byte));
                        if (!ch.is_null()) walk_v_(ch, f);
                     }
                     break;
                  }
               }
               return;
            }
         }
      }

      // Recursive in-order DFS backing for_each. Enumerates children directly: setlist
      // by its sorted bytes[], cnode via cnode_for_each (ctz bitmap walk), node_full by
      // a single 0..255 pass. `key` grows on descent and the length is reset on return;
      // bscratch holds the current bucket's suffix-sort order (reused — only one bucket
      // is active at a time in a DFS).
      template <class F>
      void walk_(packed_ptr cur, key_buf& key, std::vector<int>& bscratch, F& f) const
      {
         const K k = cur.tag();
         switch (k)
         {
            case K::value_ptr:
            {
               char* L = static_cast<char*>(deref_(cur));
               if (leaf_has_full(L))  // whole key contiguous in the leaf → zero-copy, no append
               {
                  f(leaf_fullview(L), static_cast<const T&>(*leaf_val(L)));
                  return;
               }
               const size_t b = key.len;
               key.append(leaf_suf(L), leaf_slen(L));
               f(key.view(), static_cast<const T&>(*leaf_val(L)));
               key.len = b;
               return;
            }
            case K::value_inline:
               call_val_(key.view(), cur, f);  // empty suffix → key already complete
               return;
            case K::prefix_node:
            {
               char*        P = static_cast<char*>(deref_(cur));
               const size_t b = key.len;
               key.append(pn_pfx<Ptr>(P), pn_plen(P));
               walk_(pn_next<Ptr>(P), key, bscratch, f);
               key.len = b;
               return;
            }
            case K::bucket:
               if constexpr (bucketable)
               {
                  bucket*      bk   = static_cast<bucket*>(deref_(cur));
                  const size_t base = bscratch.size();
                  bkt_append_order(bk, bscratch);
                  const size_t kb = key.len;
                  for (size_t i = base; i < bscratch.size(); ++i)
                  {
                     const char* e = bkt_entry(bk, bscratch[i]);
                     key.append(bkt_suf(e), bkt_slen(e));
                     f(key.view(), static_cast<const T&>(*bkt_val(e)));
                     key.len = kb;
                  }
                  bscratch.resize(base);
               }
               return;
            default:  // routers
            {
               const router_hdr* rh   = static_cast<const router_hdr*>(deref_(cur));
               const size_t      base = key.len;
               if (k == K::setlist_u8 && rh->hdr.prefix_len)
                  key.append(static_cast<const setlist*>(deref_(cur))->prefix, rh->hdr.prefix_len);
               const size_t cb = key.len;
               if (!rh->term.is_null())
                  call_val_(key.view(), rh->term, f);  // key ends here (path + prefix)
               auto recurse = [&](uint8_t byte, packed_ptr ch) {
                  key.push(byte);
                  walk_(ch, key, bscratch, f);
                  key.len = cb;
               };
               switch (k)
               {
                  case K::setlist_u8:
                  {
                     const setlist* s = static_cast<const setlist*>(deref_(cur));
                     const int      n = s->rh.hdr.nbranch;
                     for (int i = 0; i < n; ++i)  // bytes[] sorted
                     {
                        if constexpr (inline_children_ok)
                           if (sl_is_inline(s, i))  // leaf rides the line: emit key+byte+suffix here
                           {
                              const char* L = sl_inline_leaf(s, i);
                              key.push(s->bytes[i]);
                              if (leaf_has_full(L))
                                 f(leaf_fullview(L), static_cast<const T&>(*leaf_val(L)));
                              else
                              {
                                 const size_t b2 = key.len;
                                 key.append(leaf_suf(L), leaf_slen(L));
                                 f(key.view(), static_cast<const T&>(*leaf_val(L)));
                                 key.len = b2;
                              }
                              key.len = cb;
                              continue;
                           }
                        recurse(s->bytes[i], s->branch[i]);
                     }
                     break;
                  }
                  case K::c2: cnode_for_each(static_cast<const cnode<2>*>(deref_(cur)), recurse); break;
                  case K::c4: cnode_for_each(static_cast<const cnode<4>*>(deref_(cur)), recurse); break;
                  case K::c8: cnode_for_each(static_cast<const cnode<8>*>(deref_(cur)), recurse); break;
                  default:
                  {
                     const node_full* fn = static_cast<const node_full*>(deref_(cur));
                     for (int byte = 0; byte < 256; ++byte)
                     {
                        packed_ptr ch = full_find(fn, uint8_t(byte));
                        if (!ch.is_null()) recurse(uint8_t(byte), ch);
                     }
                     break;
                  }
               }
               key.len = base;
               return;
            }
         }
      }

      // Read-style descent returning a pointer to the STORED T for `key`, or nullptr.
      // Only meaningful when values are stable (non-inlineable T): backs find_ptr/at/[].
      // Pointer to the stored value for `key`, or nullptr — over the shared read skeleton.
      // Returns mutable T* (the value storage is mutable; descend_read is const only in that
      // it doesn't restructure). value_inline has no addressable object → nullptr.
      T* locate_(std::string_view key) const
      {
         struct op_t
         {
            const kv_tree* self;
            T* of_term(packed_ptr t) const  // a router/u16 term: T* iff it's a value_ptr leaf
            {
               return (t.is_null() || t.tag() == K::value_inline)
                          ? nullptr
                          : leaf_val(static_cast<char*>(self->deref_(t)));
            }
            T* leaf(const char* p, const char* rem, size_t rl) const
            {
               if (std::string_view(reinterpret_cast<const char*>(leaf_suf(p)), leaf_slen(p)) !=
                   std::string_view(rem, rl))
                  return nullptr;
               return const_cast<T*>(leaf_val(p));
            }
            T* inl(uint64_t, bool) const { return nullptr; }  // no addressable object
            T* bkt(const bucket* b, const char* rem, size_t rl) const
            {
               if constexpr (bucketable)
                  if (const char* e = bkt_locate(b, std::string_view(rem, rl)))
                     return bkt_val(const_cast<char*>(e));
               return nullptr;
            }
            T* term(const router_hdr* rh) const { return of_term(rh->term); }
            T* miss() const { return nullptr; }
         };
         return descend_read(key, op_t{this});
      }
      // Suffix bytes (the lazy-expansion remainder) — after the stored prefix when full.
      // Branchless: a suffix-only leaf has pfx_field == 0, so the masked prefix_len is 0 and
      // this is just leaf_str; a full leaf adds prefix_len. Keeps the find hot path cheap.
      static uint8_t* leaf_suf(char* p) noexcept
      {
         return reinterpret_cast<uint8_t*>(leaf_str(p) + (leaf_pfxfield(p) & 0x7FFF));
      }
      static const uint8_t* leaf_suf(const char* p) noexcept
      {
         return reinterpret_cast<const uint8_t*>(leaf_str(p) + (leaf_pfxfield(p) & 0x7FFF));
      }

      static size_t lcp(std::string_view a, std::string_view b) noexcept
      {
         const size_t n = a.size() < b.size() ? a.size() : b.size();
         return size_t(std::mismatch(a.begin(), a.begin() + n, b.begin()).first - a.begin());
      }

      // emplace's deferred constructor: carries the caller's argument refs down the descent
      // and constructs T exactly ONCE, at the value's final address (leaf / bucket entry).
      // A present key never constructs at all. Plain T&& remains the currency for insert and
      // for relocating existing values; vf_construct/vf_make dispatch between the two.
      template <class... Args>
      struct em_t
      {
         std::tuple<Args&&...> args;
         void construct(Allocator& a, T* dst)
         {
            std::apply([&](Args&&... as)
                       { std::allocator_traits<Allocator>::construct(
                             a, dst, std::forward<Args>(as)...); },
                       std::move(args));
         }
         T make() { return std::make_from_tuple<T>(std::move(args)); }
      };
      template <class VF>
      void vf_construct(VF&& vf, T* dst)  // consume the value source at its final address
      {
         if constexpr (std::is_same_v<std::remove_cvref_t<VF>, T>)
            v_construct(alloc_, dst, std::forward<VF>(vf));
         else
            std::forward<VF>(vf).construct(alloc_, dst);
      }
      template <class VF>
      static std::string_view vf_peek_(const VF& vf) noexcept
         requires raw_vals
      {
         if constexpr (std::is_same_v<std::remove_cvref_t<VF>, T>)
            return vf.v;
         else
            return vf.make().v;
      }
      template <class VF>
      static T vf_make(VF&& vf)  // materialize (the inline-packing path; small trivial T)
      {
         if constexpr (std::is_same_v<std::remove_cvref_t<VF>, T>)
            return std::forward<VF>(vf);
         else
            return std::forward<VF>(vf).make();
      }

      static packed_ptr pack_inline(const T& v) noexcept
      {
         packed_ptr p;
         std::memset(p.b, 0, sizeof(Ptr));
         p.b[0] = uint8_t(K::value_inline);
         std::memcpy(p.b + 1, &v, inline_bytes);  // value in bytes [1..1+sizeof)
         return p;
      }
      static void unpack_inline(packed_ptr p, T& out) noexcept { std::memcpy(&out, p.b + 1, inline_bytes); }
      static void copy_value(T& out, const T* src)
      {
         if constexpr (std::is_trivially_copyable_v<T>)
            std::memcpy(&out, src, sizeof(T));
         else
            out = *src;
      }
      // Read a terminal value slot (inline or leaf) into out. Used at key-end (the
      // suffix is empty for terminals reached this way).
      void read_value(packed_ptr v, T& out) const
      {
         if (v.tag() == K::value_inline)
            unpack_inline(v, out);
         else if constexpr (raw_vals)
            out = T{leaf_bytes(static_cast<const char*>(deref_(v)))};
         else
            copy_value(out, leaf_val(static_cast<const char*>(deref_(v))));
      }
      bool read_term(const router_hdr* rh, T& out) const
      {
         if (rh->term.is_null())
            return false;
         read_value(rh->term, out);
         return true;
      }

      static void leaf_set_hdr(char* p, uint16_t slen, uint16_t pfxfield) noexcept
      {
         std::memcpy(p + 4, &slen, 2);  // offset 0..3 = vstamp (written by pack_)
         std::memcpy(p + 6, &pfxfield, 2);
      }
      template <class VF>
      packed_ptr make_leaf(std::string_view suf, VF&& val)  // suffix-only (compressed) leaf
      {
         if constexpr (raw_vals)
         {
            const std::string_view b = vf_make(std::forward<VF>(val)).v;
            char* p = static_cast<char*>(node_alloc_(leaf_size(suf.size(), b.size())));
            leaf_set_hdr(p, uint16_t(suf.size()), 0);
            const uint16_t vl = uint16_t(b.size());
            std::memcpy(p + 8, &vl, 2);
            std::memcpy(p + leaf_value_off(), b.data(), b.size());
            std::memcpy(leaf_str(p), suf.data(), suf.size());
            return pack_(p, K::value_ptr);
         }
         else
         {
            char* p = static_cast<char*>(node_alloc_(leaf_size(suf.size())));
            leaf_set_hdr(p, uint16_t(suf.size()), 0);
            std::memcpy(leaf_str(p), suf.data(), suf.size());
            vf_construct(std::forward<VF>(val), leaf_val(p));
            return pack_(p, K::value_ptr);
         }
      }
      template <class VF>
      packed_ptr make_leaf_full(std::string_view pfx, std::string_view suf, VF&& val)  // whole key
      {
         if constexpr (raw_vals)
         {
            const std::string_view b = vf_make(std::forward<VF>(val)).v;
            char* p = static_cast<char*>(
                node_alloc_(leaf_size(pfx.size() + suf.size(), b.size())));
            leaf_set_hdr(p, uint16_t(suf.size()), uint16_t(LEAF_FULL | pfx.size()));
            const uint16_t vl = uint16_t(b.size());
            std::memcpy(p + 8, &vl, 2);
            std::memcpy(p + leaf_value_off(), b.data(), b.size());
            std::memcpy(leaf_str(p), pfx.data(), pfx.size());
            std::memcpy(leaf_str(p) + pfx.size(), suf.data(), suf.size());
            return pack_(p, K::value_ptr);
         }
         else
         {
            char* p = static_cast<char*>(node_alloc_(leaf_size(pfx.size() + suf.size())));
            leaf_set_hdr(p, uint16_t(suf.size()), uint16_t(LEAF_FULL | pfx.size()));
            std::memcpy(leaf_str(p), pfx.data(), pfx.size());
            std::memcpy(leaf_str(p) + pfx.size(), suf.data(), suf.size());
            vf_construct(std::forward<VF>(val), leaf_val(p));
            return pack_(p, K::value_ptr);
         }
      }
      // A terminal value with key suffix `suf`: inline in the pointer when the suffix
      // is empty and T fits (no allocation), else a suffix-only leaf.
      template <class VF>
      packed_ptr make_value(std::string_view suf, VF&& val)
      {
         if constexpr (inlineable)
            if (suf.empty())
               return pack_inline(vf_make(std::forward<VF>(val)));
         return make_leaf(suf, std::forward<VF>(val));
      }
      // Full-key-aware terminal: `key` is the whole key, the leaf sits at `depth` (suffix
      // = key[depth:]). Stores the WHOLE key contiguously iff the prefix fits in the node
      // blocks we'd allocate anyway (free) → enables zero-copy key() during iteration;
      // else falls back to suffix-only. depth must fit the leaf's 15-bit prefix_len field.
      template <class VF>
      packed_ptr make_value_at(std::string_view key, size_t depth, VF&& val)
      {
         const std::string_view suf = key.substr(depth);
         if constexpr (inlineable)
            if (suf.empty())
               return pack_inline(vf_make(std::forward<VF>(val)));
         const std::string_view pfx = key.substr(0, depth);
         size_t                 vsz = sizeof(T);
         if constexpr (raw_vals)
            vsz = vf_peek_(val).size();
         if (depth < 0x8000 && nblocks_(leaf_size(pfx.size() + suf.size(), vsz)) ==
                                   nblocks_(leaf_size(suf.size(), vsz)))
            return make_leaf_full(pfx, suf, std::forward<VF>(val));
         return make_leaf(suf, std::forward<VF>(val));
      }
      packed_ptr make_prefix(std::string_view pfx, packed_ptr next)
      {
         if (pfx.empty())
            return next;
         char*    p  = static_cast<char*>(node_alloc_(pn_size<Ptr>(pfx.size())));
         uint16_t pl = uint16_t(pfx.size());
         std::memcpy(p, &pl, 2);
         pn_set_next(p, next);
         std::memcpy(pn_pfx<Ptr>(p), pfx.data(), pfx.size());
         return pack_(p, K::prefix_node);
      }
      setlist* new_setlist()
      {
         setlist* s = ::new (node_alloc_c_<sizeof(setlist)>()) setlist;
         s->rh.hdr  = node_hdr{0, 0, 0};
         s->rh.term = packed_ptr::null();
         s->inl     = 0;  // no inline children (the aggregate has no ctor — must zero explicitly)
         std::memset(s->bytes, 0xFF, sizeof(s->bytes));  // pad unused lanes with max for sorted SIMD lower_bound
         return s;
      }
      setlist16* new_setlist16()
      {
         setlist16* s = ::new (node_alloc_(sizeof(setlist16))) setlist16;
         s->rh.hdr    = node_hdr{0, 0, 0};
         s->rh.term   = packed_ptr::null();
         std::memset(s->stems, 0xFF, sizeof(s->stems));  // all lanes 0xFFFF for the SIMD lower_bound
         return s;
      }
      // Online fusion (the WideStems creation policy): when a leaf split has just turned a
      // child of setlist_u8 `*pslot` into a router, collapse `*pslot` + ALL its children into
      // one wide u16 node — iff that's a genuinely sparse 2-level chain (so the win is real
      // and it won't immediately re-stride). Predicate: the parent is prefix-less, and every
      // child is a prefix-less, term-less ONE-byte router (a leaf/term/bucket/u16 child can't
      // be represented by a 2-byte hop), and the fan-in fits one cacheline with headroom.
      // Fires only at splits (off the find hot path); dense nodes fail the predicate untouched.
      static constexpr int FUSE_THRESH = setlist16::CAP - 2;  // headroom so a few more inserts don't re-stride
      // noexcept: fusion is a best-effort optimization that runs AFTER the insert has already
      // succeeded, so it must never fail (or corrupt) it. The one allocation (the wide node) is
      // wrapped — on OOM we simply leave the 2-level chain in place.
      void try_fuse_parent(packed_ptr* pslot) noexcept
      {
         const packed_ptr Pp = *pslot;
         if (Pp.tag() != K::setlist_u8)
            return;
         setlist*  P  = static_cast<setlist*>(deref_(Pp));
         if (P->rh.hdr.prefix_len != 0)
            return;
         const int nb = P->rh.hdr.nbranch;
         if (nb == 0)
            return;
         int total = 0;
         for (int i = 0; i < nb; ++i)
         {
            const packed_ptr c  = P->branch[i];
            const K          ck = c.tag();
            if (ck == K::setlist_u16 || !is_router(ck))  // 2-byte child or leaf/term/bucket/prefix
               return;
            const router_hdr* crh = static_cast<const router_hdr*>(deref_(c));
            if (!crh->term.is_null())                    // a key ends 1 byte in → not representable
               return;
            if (ck == K::setlist_u8 && crh->hdr.prefix_len != 0)
               return;
            total += crh->hdr.nbranch;
            if (total > FUSE_THRESH)
               return;
         }
         // commit: build the wide node (b0,b1)→grandchild, carrying the parent's term. The wide
         // node is the only allocation; on OOM skip fusion (the 2-level chain is already correct).
         setlist16* U;
         try { U = new_setlist16(); }
         catch (...) { return; }
         U->rh.term       = P->rh.term;
         U->rh.hdr.has_term = P->rh.hdr.has_term;
         for (int i = 0; i < nb; ++i)
         {
            const uint8_t b0 = P->bytes[i];
            router_for_each(P->branch[i],
                            [&](uint8_t b1, packed_ptr gc) { sl16_set(U, uint16_t(uint16_t(b0) << 8 | b1), gc); });
         }
         for (int i = 0; i < nb; ++i)  // the child routers are absorbed; grandchildren are reused
            node_free_(deref_(P->branch[i]), node_size(P->branch[i].tag()));
         node_free_(P, sizeof(setlist));
         *pslot = pack_(U, K::setlist_u16);
      }

      // Re-stride a full (or 1-byte-stuck) wide router back to the byte-at-a-time family:
      // a u16 stem (hi,lo)→child becomes hi→(u8 setlist)→lo→child, so each child's stored
      // suffix (already past depth+2) stays correct as the two u8 levels consume hi then lo.
      // The node's term (a key ending exactly here, at depth) carries over unchanged. This is
      // the self-tuning fallback: dense regions that overflow 14 stems revert to byte routing.
      packed_ptr restride_u16_to_u8(packed_ptr cur)
      {
         static_assert(setlist16::CAP <= setlist::CAP, "every u16 hi-group / hi-set fits one u8 setlist");
         setlist16* N = static_cast<setlist16*>(deref_(cur));
         const int  n = N->rh.hdr.nbranch;
         // A u16 holds <= setlist16::CAP (14) stems, so each hi-group has <= 14 children and
         // there are <= 14 distinct hi bytes — both within setlist u8 CAP (16). No child or
         // parent setlist ever widens, so the new nodes are ALL plain setlists; a throw mid-build
         // frees them shell-only (grandchildren stay owned by the still-intact N). N itself is
         // only read until the final node_free_, so on a throw *slot keeps pointing at it.
         setlist* P         = new_setlist();
         P->rh.term         = N->rh.term;  // term is at this depth → unchanged
         P->rh.hdr.has_term = N->rh.hdr.has_term;
         restride_guard g{this, P};
         int            i = 0;
         while (i < n)
         {
            const uint8_t hi = uint8_t(N->stems[i] >> 8);
            setlist*      C  = new_setlist();  // the only throwing op in the loop
            while (i < n && uint8_t(N->stems[i] >> 8) == hi)
            {
               const bool ok = setlist_set(C, uint8_t(N->stems[i] & 0xFF), N->branch[i]);
               assert(ok && "restride: a u16 hi-group always fits one u8 setlist");
               (void)ok;
               ++i;
            }
            const bool ok = setlist_set(P, hi, pack_(C, K::setlist_u8));
            assert(ok && "restride: distinct-hi count always fits one u8 setlist");
            (void)ok;
         }
         g.release();
         node_free_(N, sizeof(setlist16));
         return pack_(P, K::setlist_u8);
      }
      // Attach a shared prefix to a freshly-built setlist: store it INLINE if it fits
      // (no extra hop), else fall back to a separate prefix_node. Returns the top ptr.
      packed_ptr apply_prefix(setlist* r, std::string_view pfx)
      {
         packed_ptr top = pack_(r, K::setlist_u8);
         if (pfx.empty())
            return top;
         if (pfx.size() <= setlist::PREFIX_CAP)
         {
            r->rh.hdr.prefix_len = uint16_t(pfx.size());
            std::memcpy(r->prefix, pfx.data(), pfx.size());
            return top;
         }
         return make_prefix(pfx, top);
      }
      template <class VF>
      void set_term(router_hdr* rh, VF&& val)
      {
         rh->term         = make_value(std::string_view{}, std::forward<VF>(val));  // "" -> inline
         rh->hdr.has_term = 1;
      }

      // ── router dispatch over the tier family {setlist, c2, c4, c8, node_full} ───
      static size_t node_size(K k) noexcept
      {
         switch (k)
         {
            case K::setlist_u8:  return sizeof(setlist);
            case K::setlist_u16: return sizeof(setlist16);
            case K::c2:          return sizeof(cnode<2>);
            case K::c4:          return sizeof(cnode<4>);
            case K::c8:          return sizeof(cnode<8>);
            case K::node_full:   return sizeof(node_full);
            case K::full_x:      return sizeof(node_fullx) <= 128 ? 128 : sizeof(node_fullx);
            case K::cx:          return 128;
            case K::cbucket:     return 128;  // CL0 only; pieces are separate
            // exhaustive (no default): a new node kind must declare its size here.
            case K::value_ptr: case K::value_inline: case K::prefix_node:
            case K::bucket: case K::hdr: case K::null:
               assert(false); __builtin_unreachable();
         }
      }
      // ── see-header (K::hdr) wide routers: CL0 layout
      //    [router_hdr 6][base_kind u8][pad][vstamp u32][prefix bytes ..]
      // prefix_len lives in node_hdr.prefix_len (6 bits, cap 63). A wide
      // router with prefix_len == 0 must carry its FAST tag (the headerless
      // certificate); only prefixed ones pay the hdr tag + CL0 visit.
      static constexpr unsigned WPFX_OFF = 12;
      static constexpr unsigned WPFX_CAP = 63;  // node_hdr.prefix_len is 6 bits
      static K hdr_base_(const void* cl0) noexcept
      {
         return K(static_cast<const uint8_t*>(cl0)[6] & 0xF);
      }
      static void hdr_set_base_(void* cl0, K base) noexcept
      {
         static_cast<uint8_t*>(cl0)[6] = uint8_t(base);
      }
      static const uint8_t* wpfx_(const void* cl0) noexcept
      {
         return static_cast<const uint8_t*>(cl0) + WPFX_OFF;
      }
      Ptr retag_(Ptr p, K k) const noexcept
      {
         Ptr r;
         const uint32_t raw = (uint32_t(p.raw()) & ~0xFu) | uint8_t(k);
         std::memcpy(r.b, &raw, 4);
         return r;
      }
      static bool is_wide_implicit_(K k) noexcept
      {
         return k == K::c2 || k == K::c4 || k == K::c8 || k == K::node_full;
      }

      // ONE typed visitor for the byte-router family: switch on the tag once, hand the
      // TYPED node to f. Every byte-router dispatcher below is a packed_ptr -> typed-overload
      // one-liner over this, so tag exhaustiveness lives in exactly one switch (u16 is not a
      // byte router; the callers that can see one handle it first).
      // always_inline + noexcept propagation: the typed switch must dissolve into each
      // dispatcher exactly as the hand-written switches did — an outlined call (or a
      // may-throw edge inside the noexcept dispatchers) would bloat every caller with
      // EH/terminate scaffolding. Verified per-function asm-identical against those.
      template <class F>
      [[gnu::always_inline]] decltype(auto) with_router(packed_ptr cur, F&& f) const noexcept(
          std::is_nothrow_invocable_v<F&, setlist*>)
      {
         switch (cur.tag())
         {
            case K::setlist_u8: return f(static_cast<setlist*>(deref_(cur)));
            case K::c2:         return f(static_cast<cnode<2>*>(deref_(cur)));
            case K::c4:         return f(static_cast<cnode<4>*>(deref_(cur)));
            case K::c8:         return f(static_cast<cnode<8>*>(deref_(cur)));
            case K::node_full:  return f(static_cast<node_full*>(deref_(cur)));
            case K::full_x:     return f(static_cast<node_fullx*>(deref_(cur)));
            case K::cx:         return f(static_cast<node_cx*>(deref_(cur)));
            case K::hdr:  // see-header: CL0 byte 6 names the real router type
            {
               void* n = deref_(cur);
               switch (K(static_cast<const uint8_t*>(n)[6] & 0xF))
               {
                  case K::c2:        return f(static_cast<cnode<2>*>(n));
                  case K::c4:        return f(static_cast<cnode<4>*>(n));
                  case K::c8:        return f(static_cast<cnode<8>*>(n));
                  case K::node_full: return f(static_cast<node_full*>(n));
                  default: assert(false && "hdr wraps wide implicit routers only");
                           __builtin_unreachable();
               }
            }
            // exhaustive (no default): a new K must be decided here — and only here.
            case K::value_ptr: case K::value_inline: case K::prefix_node:
            case K::setlist_u16: case K::bucket: case K::cbucket: case K::null:
               assert(false && "byte-routers only"); __builtin_unreachable();
         }
      }

      // byte -> child (read step), per tier and dispatched.
      static packed_ptr router_find(setlist* s, uint8_t byte) noexcept { return setlist_find(s, byte); }
      template <int N>
      static packed_ptr router_find(cnode<N>* c, uint8_t byte) noexcept { return cnode_step(c, byte); }
      static packed_ptr router_find(node_full* f, uint8_t byte) noexcept { return full_find(f, byte); }

      // First child with byte >= from (iterator stepping). Each tier resumes from its own
      // structure — sorted stems, presence-bitmap ctz, nset line skip — instead of probing
      // every byte through a find.
      static bool router_next(setlist* s, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         const int n = s->rh.hdr.nbranch;
         for (int i = 0; i < n; ++i)  // bytes[] sorted ascending
            if (s->bytes[i] >= from) { ob = s->bytes[i]; oc = s->branch[i]; return true; }
         return false;
      }
      template <int N>
      static bool router_next(cnode<N>* c, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         constexpr int SEGW = cnode<N>::SEGW, SHIFT = cnode<N>::SHIFT, MASKB = cnode<N>::MASKB;
         for (int sg = from >> SHIFT; sg < N; ++sg)
         {
            const typename cnode<N>::seg* s  = &c->segs[sg];
            const int                     lo = (sg == from >> SHIFT) ? (from & (SEGW - 1)) : 0;
            for (int w = lo >> 6; w < (MASKB + 7) / 8; ++w)
            {
               uint64_t bits = bm_word(s->present, w);
               if constexpr (MASKB % 8 != 0)
                  bits &= (1ull << (MASKB * 8)) - 1;  // mask the documented over-read (c8: 4-byte map)
               if (w == lo >> 6 && (lo & 63))
                  bits &= ~0ull << (lo & 63);
               if (bits)
               {
                  const int d = w * 64 + std::countr_zero(bits);
                  ob = uint8_t(sg * SEGW + d);
                  oc = s->branch[bm_rank(s->present, d)];
                  return true;
               }
            }
         }
         return false;
      }
      static bool router_next(node_full* f, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(uint8_t(from), li, sl);
         for (; li < full_lines_v<Ptr>; ++li, sl = 0)
         {
            const cline_direct_t<Ptr>& ln = f->ranges[li];
            if (ln.nset == 0)
               continue;  // empty line: population is tracked
            for (; sl < full_per_line_v<Ptr>; ++sl)
            {
               const unsigned byte = li * full_per_line_v<Ptr> + sl;
               if (byte > 255)
                  return false;
               if (!ln.ptr[sl].is_null())
               {
                  ob = uint8_t(byte);
                  oc = ln.ptr[sl];
                  return true;
               }
            }
         }
         return false;
      }

      // ── node_fullx (explicit full) helpers: member, they deref line ptrs ─────
      cline_direct_t<Ptr>* fx_line_(const node_fullx* x, unsigned li) const noexcept
      {
         return static_cast<cline_direct_t<Ptr>*>(deref_(x->lines[li]));
      }
      packed_ptr router_find(node_fullx* x, uint8_t byte) const noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(byte, li, sl);
         if (x->lines[li].is_null())
            return packed_ptr::null();
         return fx_line_(x, li)->ptr[sl];
      }
      bool router_next(node_fullx* x, int from, uint8_t& ob, packed_ptr& oc) const noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(uint8_t(from), li, sl);
         for (; li < full_lines_v<Ptr>; ++li, sl = 0)
         {
            if (x->lines[li].is_null())
               continue;
            const cline_direct_t<Ptr>* ln = fx_line_(x, li);
            if (ln->nset == 0)
               continue;
            for (; sl < full_per_line_v<Ptr>; ++sl)
            {
               const unsigned byte = li * full_per_line_v<Ptr> + sl;
               if (byte > 255)
                  return false;
               if (!ln->ptr[sl].is_null())
               {
                  ob = uint8_t(byte);
                  oc = ln->ptr[sl];
                  return true;
               }
            }
         }
         return false;
      }
      packed_ptr* router_find_slot(node_fullx* x, uint8_t byte) const noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(byte, li, sl);
         if (x->lines[li].is_null())
            return nullptr;
         packed_ptr* p = &fx_line_(x, li)->ptr[sl];
         return p->is_null() ? nullptr : p;
      }
      template <class F>
      void router_for_each(node_fullx* x, F&& f) const
      {
         for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
         {
            if (x->lines[li].is_null())
               continue;
            cline_direct_t<Ptr>* ln = fx_line_(x, li);
            if (ln->nset == 0)
               continue;
            for (unsigned sl = 0; sl < full_per_line_v<Ptr>; ++sl)
            {
               const unsigned byte = li * full_per_line_v<Ptr> + sl;
               if (byte > 255)
                  break;
               if (!ln->ptr[sl].is_null())
                  f(uint8_t(byte), ln->ptr[sl]);
            }
         }
      }

      template <class N>
      packed_ptr router_find_d(N* n, uint8_t byte) const noexcept
      {
         return router_find(n, byte);
      }
      packed_ptr router_find_d(node_fullx* n, uint8_t byte) const noexcept
      {
         return router_find(n, byte);
      }
      template <class N>
      bool router_next_d(N* n, int from, uint8_t& ob, packed_ptr& oc) const noexcept
      {
         return router_next(n, from, ob, oc);
      }
      bool router_next_d(node_fullx* n, int from, uint8_t& ob, packed_ptr& oc) const noexcept
      {
         return router_next(n, from, ob, oc);
      }

      // ── node_cx (explicit segmented) helpers: member, runtime geometry ───────
      uint8_t* cx_seg_(const node_cx* x, unsigned sg) const noexcept
      {
         return static_cast<uint8_t*>(deref_(x->seglines[sg]));
      }
      packed_ptr router_find(node_cx* x, uint8_t byte) const noexcept
      {
         const unsigned sg = byte >> (8 - x->lg);
         if (x->seglines[sg].is_null())  // empty range: no line allocated
            return packed_ptr::null();
         const unsigned d = byte & (cx_segw(x->lg) - 1);
         uint8_t*       m = cx_seg_(x, sg);
         return bm_test(m, int(d)) ? cx_branches<Ptr>(m, x->lg)[bm_rank(m, int(d))]
                                   : packed_ptr::null();
      }
      packed_ptr* router_find_slot(node_cx* x, uint8_t byte) const noexcept
      {
         const unsigned sg = byte >> (8 - x->lg);
         if (x->seglines[sg].is_null())
            return nullptr;
         const unsigned d = byte & (cx_segw(x->lg) - 1);
         uint8_t*       m = cx_seg_(x, sg);
         return bm_test(m, int(d)) ? &cx_branches<Ptr>(m, x->lg)[bm_rank(m, int(d))] : nullptr;
      }
      // Update-or-insert; false when THIS segment line is full (promote signal).
      bool router_try_set(node_cx* x, uint8_t byte, packed_ptr p) const noexcept
      {
         const unsigned sg = byte >> (8 - x->lg);
         if (x->seglines[sg].is_null())
            return false;  // no line for this range: caller allocates (cx_ensure_seg_)
         const unsigned d   = byte & (cx_segw(x->lg) - 1);
         uint8_t*       m   = cx_seg_(x, sg);
         Ptr*           br  = cx_branches<Ptr>(m, x->lg);
         const int      pos = bm_rank(m, int(d));
         if (bm_test(m, int(d)))
         {
            br[pos] = p;
            return true;
         }
         const int cnt = bm_count(m, int(cx_maskb(x->lg)));
         if (unsigned(cnt) >= cx_cap<Ptr>(x->lg))
            return false;
         std::memmove(&br[pos + 1], &br[pos], size_t(cnt - pos) * sizeof(Ptr));
         br[pos] = p;
         bm_set(m, int(d));
         ++x->rh.hdr.nbranch;
         return true;
      }
      void router_remove(node_cx* x, uint8_t byte) noexcept
      {
         const unsigned sg = byte >> (8 - x->lg);
         if (x->seglines[sg].is_null())
            return;
         uint8_t* m = cx_seg_(x, sg);
         const unsigned d = byte & (cx_segw(x->lg) - 1);
         if (!bm_test(m, int(d)))
            return;
         Ptr*      br  = cx_branches<Ptr>(m, x->lg);
         const int cnt = bm_count(m, int(cx_maskb(x->lg)));
         const int pos = bm_rank(m, int(d));
         std::memmove(&br[pos], &br[pos + 1], size_t(cnt - 1 - pos) * sizeof(Ptr));
         m[d >> 3] &= uint8_t(~(1u << (d & 7)));
         --x->rh.hdr.nbranch;
         if (cnt == 1)
         {  // range emptied: drop the line (private here — the remove path
            // privatized it), so an emptied range costs nothing until reused
            cx_release_line_(x->seglines[sg], x->lg);
            x->seglines[sg] = packed_ptr::null();
         }
      }
      bool router_next(node_cx* x, int from, uint8_t& ob, packed_ptr& oc) const noexcept
      {
         const unsigned segw = cx_segw(x->lg), nseg = 1u << x->lg, maskb = cx_maskb(x->lg);
         for (unsigned sg = unsigned(from) / segw; sg < nseg; ++sg)
         {
            if (x->seglines[sg].is_null())
               continue;
            uint8_t*       m  = cx_seg_(x, sg);
            const unsigned lo = (sg == unsigned(from) / segw) ? unsigned(from) % segw : 0;
            for (unsigned w = lo / 64; w < (maskb + 7) / 8; ++w)
            {
               uint64_t bits = bm_word(m, int(w));
               if (maskb % 8 != 0)
                  bits &= (1ull << (maskb * 8)) - 1;
               if (w == lo / 64 && (lo & 63))
                  bits &= ~0ull << (lo & 63);
               if (bits)
               {
                  const int d = int(w) * 64 + std::countr_zero(bits);
                  ob          = uint8_t(sg * segw + unsigned(d));
                  oc          = cx_branches<Ptr>(m, x->lg)[bm_rank(m, d)];
                  return true;
               }
            }
         }
         return false;
      }
      template <class F>
      void router_for_each(node_cx* x, F&& f) const
      {
         const unsigned segw = cx_segw(x->lg), nseg = 1u << x->lg, maskb = cx_maskb(x->lg);
         for (unsigned sg = 0; sg < nseg; ++sg)
         {
            if (x->seglines[sg].is_null())
               continue;
            uint8_t* m   = cx_seg_(x, sg);
            int      idx = 0;
            for (unsigned wo = 0; wo < maskb; wo += 8)
            {
               uint64_t       w  = 0;
               const unsigned wb = maskb - wo < 8 ? maskb - wo : 8;
               std::memcpy(&w, m + wo, size_t(wb));
               while (w)
               {
                  const int b = std::countr_zero(w);
                  w &= w - 1;
                  f(uint8_t(sg * segw + wo * 8 + unsigned(b)),
                    cx_branches<Ptr>(m, x->lg)[idx++]);
               }
            }
         }
      }
      packed_ptr router_find_d(node_cx* n, uint8_t byte) const noexcept
      {
         return router_find(n, byte);
      }
      // ── router_prev: the LARGEST branch byte STRICTLY BELOW `from` (from=256 → max).
      // The floor-radix walk's mirror of router_next; structural path only (page routing,
      // splits), so plain backward scans are fine.
      static bool router_prev(setlist* sl, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         int best = -1;
         for (int i = 0; i < sl->rh.hdr.nbranch; ++i)
            if (int(sl->bytes[i]) < from)
               best = i;
            else
               break;  // bytes ascend
         if (best < 0)
            return false;
         ob = sl->bytes[best];
         oc = sl->branch[best];
         return true;
      }
      template <int N>
      static bool router_prev(cnode<N>* c, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         for (int b = (from > 256 ? 256 : from) - 1; b >= 0; --b)
         {
            const auto* sg = &c->segs[b >> cnode<N>::SHIFT];
            const int   d  = b & (cnode<N>::SEGW - 1);
            if (bm_test(sg->present, d))
            {
               ob = uint8_t(b);
               oc = sg->branch[bm_rank(sg->present, d)];
               return true;
            }
         }
         return false;
      }
      static bool router_prev(node_full* f, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         for (int b = (from > 256 ? 256 : from) - 1; b >= 0; --b)
         {
            unsigned li, sl;
            full_index<Ptr>(uint8_t(b), li, sl);
            if (!f->ranges[li].ptr[sl].is_null())
            {
               ob = uint8_t(b);
               oc = f->ranges[li].ptr[sl];
               return true;
            }
         }
         return false;
      }
      bool router_prev(node_fullx* x, int from, uint8_t& ob, packed_ptr& oc) const noexcept
      {
         for (int b = (from > 256 ? 256 : from) - 1; b >= 0; --b)
         {
            unsigned li, sl;
            full_index<Ptr>(uint8_t(b), li, sl);
            if (x->lines[li].is_null())
            {  // skip the whole empty range
               b = int(li * full_per_line_v<Ptr>);
               continue;
            }
            cline_direct_t<Ptr>* L = fx_line_(x, li);
            if (!L->ptr[sl].is_null())
            {
               ob = uint8_t(b);
               oc = L->ptr[sl];
               return true;
            }
         }
         return false;
      }
      bool router_prev(node_cx* x, int from, uint8_t& ob, packed_ptr& oc) const noexcept
      {
         for (int b = (from > 256 ? 256 : from) - 1; b >= 0; --b)
         {
            const unsigned sg = unsigned(b) >> (8 - x->lg);
            if (x->seglines[sg].is_null())
            {
               b = int(sg * cx_segw(x->lg));
               continue;
            }
            uint8_t*       m = cx_seg_(x, sg);
            const unsigned d = unsigned(b) & (cx_segw(x->lg) - 1);
            if (bm_test(m, int(d)))
            {
               ob = uint8_t(b);
               oc = cx_branches<Ptr>(m, x->lg)[bm_rank(m, int(d))];
               return true;
            }
         }
         return false;
      }
      static bool router_prev(setlist16*, int, uint8_t&, packed_ptr&) noexcept
      {
         assert(false && "page radix never contains wide-stem routers");
         return false;
      }
      bool router_prev_d(packed_ptr cur, int from, uint8_t& ob, packed_ptr& oc) const noexcept
      {
         return with_router(cur, [&](auto* n) noexcept { return router_prev(n, from, ob, oc); });
      }
      bool router_next_d(node_cx* n, int from, uint8_t& ob, packed_ptr& oc) const noexcept
      {
         return router_next(n, from, ob, oc);
      }

      // The slot holding byte's child if present, else nullptr — ONE dispatch for the
      // insert descend (find-or-add), instead of a separate find + child_slot pair.
      static packed_ptr* router_find_slot(setlist* s, uint8_t byte) noexcept
      {
         const int i = setlist_index(s, byte);
         return i >= 0 ? &s->branch[i] : nullptr;
      }
      template <int N>
      static packed_ptr* router_find_slot(cnode<N>* c, uint8_t byte) noexcept
      {
         return cnode_find_slot(c, byte);
      }
      static packed_ptr* router_find_slot(node_full* f, uint8_t byte) noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(byte, li, sl);
         packed_ptr* p = &f->ranges[li].ptr[sl];
         return p->is_null() ? nullptr : p;
      }
      packed_ptr* router_find_slot(packed_ptr cur, uint8_t byte) const noexcept
      {
         return with_router(cur, [&](auto* n) noexcept { return router_find_slot(n, byte); });
      }

      // Update-or-insert byte -> p; false when the tier is full (the widen signal).
      static bool router_try_set(setlist* s, uint8_t byte, packed_ptr p) noexcept { return setlist_set(s, byte, p); }
      template <int N>
      static bool router_try_set(cnode<N>* c, uint8_t byte, packed_ptr p) noexcept { return cnode_set(c, byte, p); }
      static bool router_try_set(node_full* f, uint8_t byte, packed_ptr p) noexcept
      {
         full_set(f, byte, p);  // node_full never overflows
         return true;
      }
      bool router_try_set(node_fullx* x, uint8_t byte, packed_ptr p) const noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(byte, li, sl);
         if (x->lines[li].is_null())
            return p.is_null();  // clearing an absent slot is a no-op; setting needs
                                 // the line allocated first (fx insert arm / ensure)
         cline_direct_t<Ptr>* L        = fx_line_(x, li);
         const bool           was_null = L->ptr[sl].is_null();
         L->ptr[sl]                    = p;
         if (was_null && !p.is_null())
         {
            ++L->nset;
            ++x->rh.hdr.nbranch;
         }
         else if (!was_null && p.is_null())
         {
            --L->nset;
            --x->rh.hdr.nbranch;
         }
         return true;
      }
      bool router_try_set(packed_ptr cur, uint8_t byte, packed_ptr p) const noexcept
      {
         return with_router(cur, [&](auto* n) noexcept { return router_try_set(n, byte, p); });
      }

      // Remove byte's branch (keeps each tier's invariants: setlist gap, cnode rank/present,
      // full direct slot + nset).
      static void router_remove(setlist* s, uint8_t byte) noexcept { setlist_remove(s, byte); }
      template <int N>
      static void router_remove(cnode<N>* c, uint8_t byte) noexcept { cnode_remove(c, byte); }
      static void router_remove(node_full* f, uint8_t byte) noexcept { full_set(f, byte, packed_ptr::null()); }
      void router_remove(node_fullx* x, uint8_t byte) noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(byte, li, sl);
         if (x->lines[li].is_null())
            return;
         cline_direct_t<Ptr>* L = fx_line_(x, li);
         if (L->ptr[sl].is_null())
            return;
         L->ptr[sl] = packed_ptr::null();
         --L->nset;
         --x->rh.hdr.nbranch;
         if (L->nset == 0)
         {  // range emptied: drop the line (private here via the remove path)
            fx_release_line_(x->lines[li]);
            x->lines[li] = packed_ptr::null();
         }
      }
      void router_remove(packed_ptr cur, uint8_t byte) noexcept
      {
         with_router(cur, [&](auto* n) noexcept { router_remove(n, byte); });
      }

      // Visit (byte, branch) ascending, per tier; the dispatcher also accepts setlist_u16
      // for STRUCTURAL callers (destroy/census/fusion) — they get the stem's high byte and
      // recurse the branch, never route on it.
      template <class F>
      static void router_for_each(setlist* s, F&& f)
      {
         for (int i = 0; i < s->rh.hdr.nbranch; ++i) f(s->bytes[i], s->branch[i]);
      }
      template <int N, class F>
      static void router_for_each(cnode<N>* c, F&& f)
      {
         cnode_for_each(c, f);
      }
      template <class F>
      static void router_for_each(node_full* fn, F&& f)
      {
         for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
         {
            if (fn->ranges[li].nset == 0)  // skip empty lines (population is tracked)
               continue;
            for (unsigned sl = 0; sl < full_per_line_v<Ptr>; ++sl)
            {
               const unsigned byte = li * full_per_line_v<Ptr> + sl;
               if (byte < 256 && !fn->ranges[li].ptr[sl].is_null())
                  f(uint8_t(byte), fn->ranges[li].ptr[sl]);
            }
         }
      }
      template <class F>
      void router_for_each(packed_ptr cur, F&& f) const
      {
         if (cur.tag() == K::setlist_u16)
         {
            const setlist16* s = static_cast<const setlist16*>(deref_(cur));
            for (int i = 0; i < s->rh.hdr.nbranch; ++i) f(uint8_t(s->stems[i] >> 8), s->branch[i]);
            return;
         }
         with_router(cur, [&](auto* n) { router_for_each(n, f); });
      }

      // Widen a full router. Default: setlist→node_full directly (the fast path —
      // O(1) direct-index inserts, one rebuild). With dense tiers enabled, climb the
      // memory-saving ladder setlist→c2→c4→c8→node_full instead: ~half the bytes on
      // moderate-fanout nodes, but rank-dense inserts are O(n) and each rung is a
      // rebuild, so it trades insert throughput for footprint. Pick per workload.
      packed_ptr widen(packed_ptr cur)
      {
         // Prefix carry (vpart L1b): a prefixed source — setlist with an inline
         // prefix, or a see-header wide router — keeps its prefix THROUGH the
         // ladder: the wider node is wrapped hdr-style, never lifted into a
         // prefix_node (a prefix_node above a node with buffer room is a BUG).
         uint8_t cpfx[WPFX_CAP];
         unsigned cpl = 0;
         if (cur.tag() == K::hdr)
         {
            char* hp = static_cast<char*>(deref_(cur));
            cpl      = reinterpret_cast<router_hdr*>(hp)->hdr.prefix_len;
            std::memcpy(cpfx, hp + WPFX_OFF, cpl);
            cur = retag_(cur, hdr_base_(hp));  // ladder works on the base form
         }
         else if (cur.tag() == K::setlist_u8)
         {
            setlist* sp = static_cast<setlist*>(deref_(cur));
            if ((cpl = sp->rh.hdr.prefix_len) != 0)
               std::memcpy(cpfx, sp->prefix, cpl);
         }
         const K      k  = cur.tag();
         const size_t os = node_size(k);
         packed_ptr   out;
         if constexpr (cow_stamped_ && !DenseTiers)
         {
            // vpart default: NO dense-tier ladder. The c2/c4/c8 tiers exist
            // for part's refcount economics (cheap explicit-tier line sharing
            // on COW); with stamps that reason is gone and the ladder costs
            // ~18 ns per dict find (measured: hkv mode::none 109.3 vs
            // dense_tiers 127.9) — the segmented bitmap loses to node_full's
            // direct index. Memory trade: sparse-wide routers carry empty
            // lines (rand 1M: +45% live). mode::dense_tiers RESTORES the
            // ladder — density matters for cache footprint under real
            // working-set pressure, which micro-benches cannot see (Dan);
            // both forms stay measurable per instantiation.
            out = build_wider<node_full>(cur, K::node_full);
         }
         else if constexpr (cow_enabled)
         {
            // part ladder: setlist -> {c2|cx1} -> {c4|cx2} -> {c8|cx3} ->
            // {node_full|full_x}, the form chosen by TARGET occupancy (Dan's
            // rule): implicit (monolithic, skip-CL0 1-load descent) only when
            // at most ONE of its range lines would be empty; otherwise the
            // explicit per-line form where empty ranges are NULL lines. Dense
            // tops keep the direct-node read path; sparse ones stop paying
            // dead clines. COW demotion later turns implicit into explicit
            // with the same null-empty rule, so write-hot subtrees converge
            // to the shareable form and read-only ones keep the fast one.
            const unsigned next_lg = k == K::setlist_u8 ? 1
                                   : k == K::cx ? unsigned(static_cast<node_cx*>(deref_(cur))->lg) + 1
                                   : k == K::c2 ? 2
                                   : k == K::c4 ? 3
                                   : k == K::c8 ? 4
                                   : (assert(false), 0u);  // 4 = the full tier
            unsigned used = 0;
            const unsigned nrange = next_lg == 4 ? full_lines_v<Ptr> : (1u << next_lg);
            {
               uint64_t mask = 0;
               router_for_each(cur, [&](uint8_t b, packed_ptr) {
                  mask |= 1ull << (next_lg == 4 ? b / full_per_line_v<Ptr> : b >> (8 - next_lg));
               });
               used = unsigned(std::popcount(mask));
            }
            // vpart L1a: ALWAYS implicit — explicit tiers are disabled (sharing
            // is free, so their reason to exist — dodging per-child refcount
            // traffic on COW — is gone; sparse wide routers pay empty lines,
            // a measured trade until per-line stamps land).
            const bool implicit_ok = true;
            (void)nrange;
            (void)used;
            const bool src_cx      = k == K::cx;
            if (next_lg == 4)
               out = implicit_ok ? (src_cx ? implicit_from_cx_<node_full>(cur, K::node_full)
                                           : build_wider<node_full>(cur, K::node_full))
                                 : (src_cx ? fx_from_cx_(cur) : fx_from_implicit_(cur));
            else if (implicit_ok)
            {
               const K nt = next_lg == 1 ? K::c2 : next_lg == 2 ? K::c4 : K::c8;
               if (src_cx)
                  out = next_lg == 1   ? implicit_from_cx_<cnode<2>>(cur, nt)
                        : next_lg == 2 ? implicit_from_cx_<cnode<4>>(cur, nt)
                                       : implicit_from_cx_<cnode<8>>(cur, nt);
               else
                  out = next_lg == 1   ? build_wider<cnode<2>>(cur, nt)
                        : next_lg == 2 ? build_wider<cnode<4>>(cur, nt)
                                       : build_wider<cnode<8>>(cur, nt);
            }
            else
               out = cx_from_router_(cur, next_lg);  // handles cx AND implicit sources
         }
         else if constexpr (!DenseTiers)
            out = build_wider<node_full>(cur, K::node_full);  // fast path: straight to full
         else if (k == K::setlist_u8) out = build_wider<cnode<2>>(cur, K::c2);
         else if (k == K::c2)         out = build_wider<cnode<4>>(cur, K::c4);
         else if (k == K::c4)         out = build_wider<cnode<8>>(cur, K::c8);
         else if (k == K::c8)         out = build_wider<node_full>(cur, K::node_full);
         else { assert(false); __builtin_unreachable(); }
         node_free_(deref_(cur), os);
         if (cpl)  // re-wrap: the widened node carries the prefix in ITS CL0
            out = wrap_hdr_(out, cpfx, cpl);
         return out;
      }
      // Wrap a wide IMPLICIT router with a CL0 prefix: see-header form.
      packed_ptr wrap_hdr_(packed_ptr node, const uint8_t* pfx, unsigned pl) noexcept
      {
         assert(is_wide_implicit_(node.tag()) && pl && pl <= WPFX_CAP);
         char* p = static_cast<char*>(deref_(node));
         hdr_set_base_(p, node.tag());
         reinterpret_cast<router_hdr*>(p)->hdr.prefix_len = uint16_t(pl);
         std::memcpy(p + WPFX_OFF, pfx, pl);
         return retag_(node, K::hdr);
      }
      template <class N>
      packed_ptr build_wider(packed_ptr src, K newtag)
      {
         N* n = ::new (node_alloc_c_<sizeof(N)>()) N;
         if constexpr (std::is_same_v<N, node_full>)
            full_init(n);
         else
            cnode_init(n);
         router_hdr* srh      = static_cast<router_hdr*>(deref_(src));
         n->rh.hdr.has_term   = srh->hdr.has_term;
         n->rh.term           = srh->term;
         packed_ptr np        = pack_(n, newtag);
         router_for_each(src, [&](uint8_t b, packed_ptr br) {
            const bool ok = router_try_set(np, b, br);
            assert(ok && "next tier always has room for a rebuild");
            (void)ok;
         });
         return np;
      }
      // Grow the router at *slot until byte->child fits, then leave the result in *slot. The
      // widen ladder (setlist→c2→c4→c8→full under dense tiers) can take MULTIPLE steps, and the
      // first widen frees the original node — so we re-anchor *slot after EVERY widen: a later
      // widen throwing then leaves *slot pointing at a valid (wider) node, never a freed one,
      // and the key set is unchanged (the new child simply wasn't added). A setlist's inline
      // prefix is first lifted into a prefix_node so re-anchoring through the ladder can't drop
      // it. `child` must be guarded by the caller (a throw here orphans it).
      void grow_router_(packed_ptr* slot, packed_ptr cur, uint8_t byte, packed_ptr child)
      {
         // vpart L1b: NO prefix lift. widen() carries a setlist's inline prefix
         // (and a see-header node's CL0 prefix) into the wider node's own CL0 —
         // each ladder step publishes a complete prefixed node, so re-anchoring
         // *slot stays throw-safe without a prefix_node above it.
         char*      pn   = nullptr;  // retained for the re-anchor sites below
         packed_ptr node = cur;
         while (!router_try_set(node, byte, child))
         {
            if constexpr (cow_enabled)
            {
               if (node.tag() == K::cx)
               {
                  node_cx*       x  = static_cast<node_cx*>(deref_(node));
                  const unsigned sg = byte >> (8 - x->lg);
                  if (x->seglines[sg].is_null())
                  {  // try_set failed only because the range had no line yet
                     x->seglines[sg] = cx_new_seg_(x->lg);
                     continue;
                  }
               }
               if (node.tag() == K::full_x)
               {  // terminal tier: try_set can only fail on an absent range line
                  node_fullx* x = static_cast<node_fullx*>(deref_(node));
                  unsigned    li, sl;
                  full_index<Ptr>(byte, li, sl);
                  assert(x->lines[li].is_null());
                  x->lines[li] = fx_new_line_();
                  continue;
               }
            }
            node = widen(node);  // frees the old node; re-anchor before the next (throwing) widen
            if (pn) pn_set_next(pn, node);
            else    *slot = node;
         }
      }

      template <bool Assign, class VF>
      bool set_or_update_term(router_hdr* rh, VF&& val)
      {
         if (!rh->term.is_null())  // present: overwrite (insert) or leave (emplace)
         {
            if constexpr (Assign)
            {
               if constexpr (cow_stamped_)  // fused privatize: own the term leaf
                  if (alloc_.hkv_vtxn() && rh->term.tag() == K::value_ptr &&
                      cow_shared_(rh->term))
                     cow_clone_into_(&rh->term, rh->term);
               if (rh->term.tag() == K::value_inline)
                  rh->term = pack_inline(vf_make(std::forward<VF>(val)));
               else
               {
                  char* tl = static_cast<char*>(deref_(rh->term));
                  if constexpr (raw_vals)
                  {  // same-size: bytes in place; else rebuild the term leaf
                     const std::string_view nb = vf_peek_(val);
                     if (nb.size() == leaf_vlen(tl))
                     {
                        vh_leaf_dies_(tl);  // old bytes clobbered in place
                        std::memcpy(tl + leaf_value_off(), nb.data(), nb.size());
                     }
                     else
                     {
                        const Ptr old = rh->term;
                        rh->term      = make_leaf(
                            std::string_view(reinterpret_cast<const char*>(leaf_suf(tl)),
                                             leaf_slen(tl)),
                            std::forward<VF>(val));
                        cow_release(old);
                     }
                  }
                  else
                  {
                     v_destroy(alloc_, leaf_val(tl));
                     vf_construct(std::forward<VF>(val), leaf_val(tl));
                  }
               }
            }
            return false;
         }
         set_term(rh, std::forward<VF>(val));
         return true;
      }

      // Upsert into an INLINE child: the leaf rides s's (already-owned) line. `depth` is the
      // byte index that selected child i, so the leaf's key tail is key.substr(depth+1). Per
      // Dan's COW insight: the line is already cloned, so a same-size value update overwrites
      // in place — NO externalize. Externalize only when the leaf no longer fits (raw value
      // grows/shrinks) or the key diverges (a split needs the slot). Returns true iff a new
      // key was inserted (an exact-key update returns false).
      template <bool Assign, class VF>
      bool upsert_inline_(setlist* s, int i, std::string_view key, size_t depth, VF&& val)
      {
         char*            L = sl_inline_leaf(s, i);
         std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
         std::string_view R = key.substr(depth + 1);
         if (S == R)  // exact key → value update
         {
            if constexpr (Assign)
            {
               if constexpr (raw_vals)
               {
                  const std::string_view nb = vf_peek_(val);
                  if (nb.size() == leaf_vlen(L))
                  {  // same size: overwrite in the owned line — no externalize (Dan)
                     vh_leaf_dies_(L);
                     std::memcpy(L + leaf_value_off(), nb.data(), nb.size());
                  }
                  else
                  {  // size changed: no longer fits this slot — externalize, then rebuild
                     sl_externalize_(s, i);
                     bool ins;
                     s->branch[i] =
                         split_leaf<Assign>(s->branch[i], key, depth + 1, std::forward<VF>(val), ins);
                  }
               }
               else
               {  // trivially-copyable fixed-size T: overwrite in the owned line
                  v_destroy(alloc_, leaf_val(L));
                  vf_construct(std::forward<VF>(val), leaf_val(L));
               }
            }
            return false;  // key existed
         }
         // diverging key → externalize the leaf, then split it like a normal terminal
         sl_externalize_(s, i);
         bool inserted;
         s->branch[i] = split_leaf<Assign>(s->branch[i], key, depth + 1, std::forward<VF>(val), inserted);
         return inserted;
      }

      // Split a leaf whose suffix diverges from the key: build a setlist holding the
      // old value and the new one, optionally under a shared-prefix node. Returns the
      // new subtree (or the same leaf, value updated, when the key matches exactly).
      template <bool Assign, class VF>
      packed_ptr split_leaf(packed_ptr cur, std::string_view key, size_t depth, VF&& val, bool& inserted)
      {
         if (cur.tag() == K::value_inline)
         {
            // Inline terminal: stored key ended here (empty suffix). No allocation to free.
            std::string_view R = key.substr(depth);
            if (R.empty())  // exact key
            {
               inserted = false;
               if constexpr (Assign) return pack_inline(vf_make(std::forward<VF>(val)));
               else return cur;
            }
            T oldv;
            unpack_inline(cur, oldv);  // inlineable T is trivially copyable
            inserted = true;
            setlist* r = new_setlist();
            set_term(&r->rh, std::move(oldv));                                    // old key terminates
            setlist_set(r, uint8_t(R[0]), make_value_at(key, depth + 1, std::forward<VF>(val)));  // new continues
            return pack_(r, K::setlist_u8);  // c == 0, no prefix node needed
         }
         char*            L = static_cast<char*>(deref_(cur));
         std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
         std::string_view R = key.substr(depth);
         if (S == R)  // exact key
         {
            if constexpr (Assign)
            {
               if constexpr (raw_vals)
               {
                  const std::string_view nb = vf_peek_(val);
                  if (nb.size() == leaf_vlen(L))
                  {
                     vh_leaf_dies_(L);  // old bytes clobbered in place
                     std::memcpy(L + leaf_value_off(), nb.data(), nb.size());
                  }
                  else
                  {  // size changed: rebuild the leaf verbatim around new bytes
                     const size_t Lsz = leaf_alloc_size(L);
                     const Ptr    np  = leaf_has_full(L)
                                            ? make_leaf_full(
                                                  std::string_view(leaf_str(L), leaf_plen(L)),
                                                  std::string_view(
                                                      reinterpret_cast<const char*>(leaf_suf(L)),
                                                      leaf_slen(L)),
                                                  std::forward<VF>(val))
                                            : make_leaf(
                                                  std::string_view(
                                                      reinterpret_cast<const char*>(leaf_suf(L)),
                                                      leaf_slen(L)),
                                                  std::forward<VF>(val));
                     vh_leaf_dies_(L);  // old bytes die with the old leaf
                     node_free_(L, Lsz);
                     inserted = false;
                     return np;
                  }
               }
               else
               {
                  v_destroy(alloc_, leaf_val(L));
                  vf_construct(std::forward<VF>(val), leaf_val(L));
               }
            }
            inserted = false;
            return cur;
         }
         const size_t           c       = lcp(S, R);
         const size_t           Lsize   = leaf_alloc_size(L);  // capture before node_free_ clobbers L
         const bool             oldfull = leaf_has_full(L);
         const std::string_view ofv     = oldfull ? leaf_fullview(L) : std::string_view{};  // into L (read pre-free)

         // Strong exception safety: do every throwing allocation BEFORE touching L. The new
         // value + prefix are built first; the OLD value is moved out LAST — and make_value
         // allocates then moves, so even that final alloc failing leaves L's value in place.
         // Any bad_alloc therefore leaves L (and the whole container) unchanged.
         setlist*   r       = new_setlist();                         // (1)
         packed_ptr partial = pack_(r, K::setlist_u8);
         build_guard g{this, &partial};  // a later alloc throwing must not orphan the new value
         if (c == R.size())                                           // (2) new value (consumes val)
            set_term(&r->rh, std::forward<VF>(val));
         else
            setlist_set(r, uint8_t(R[c]), make_value_at(key, depth + c + 1, std::forward<VF>(val)));
         partial = apply_prefix(r, key.substr(depth, c));             // (3) shared prefix (== S[0..c])
         if constexpr (raw_vals)
         {  // bytes MOVE out of the OLD leaf into exactly one new node, then the
            // old leaf is freed — one live copy throughout, so no vh hooks fire
            const T ov{leaf_bytes(L)};
            if (c == S.size())  // (4) old value, LAST
               set_term(&r->rh, ov);
            else if (oldfull)
               setlist_set(r, uint8_t(S[c]), make_value_at(ofv, depth + c + 1, ov));
            else
               setlist_set(r, uint8_t(S[c]), make_value(S.substr(c + 1), ov));
            node_free_(L, Lsize);
         }
         else
         {
            if (c == S.size())                                           // (4) old value, LAST
               set_term(&r->rh, std::move(*leaf_val(L)));
            else if (oldfull)  // reuse the old leaf's contiguous full key so the child stays full
               setlist_set(r, uint8_t(S[c]), make_value_at(ofv, depth + c + 1, std::move(*leaf_val(L))));
            else
               setlist_set(r, uint8_t(S[c]), make_value(S.substr(c + 1), std::move(*leaf_val(L))));
            v_destroy(alloc_, leaf_val(L));  // old value moved out → destroy moved-from + free
            node_free_(L, Lsize);
         }
         // establish-once: pull r's small leaf children into its line (alloc-free; a same-scope
         // MOVE, no vh hooks). Done at the split shape only — re-compacting per insert is churn.
         if constexpr (inline_children_ok)
            sl_inline_compact_(r);
         g.release();  // committed: the caller adopts `partial`
         inserted = true;
         return partial;
      }

      // Split a prefix_node whose prefix diverges from the key at position c (< plen): a new
      // parent setlist routes the new value's branch and the old continuation. The new value
      // branch + outer prefix are built while `r` references ONLY new nodes, so a bad_alloc
      // lets the guard reclaim it without touching the live child `next`; the graft of `next`
      // (and freeing the old node) is a non-throwing tail. make_branch(v) builds the terminal
      // for the diverging remainder key[depth+c+1:] — a leaf in radix mode, a bucket in
      // bucket mode (the two callers differ in nothing else).
      template <class MakeBranch, class VF>
      packed_ptr split_prefix(packed_ptr cur, std::string_view key, size_t depth, size_t c,
                              VF&& val, MakeBranch&& make_branch)
      {
         char*            P = static_cast<char*>(deref_(cur));
         std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
         packed_ptr       next  = pn_next<Ptr>(P);
         const size_t     Psize = pn_size<Ptr>(PP.size());  // capture before node_free_ clobbers P
         std::string_view R     = key.substr(depth);
         setlist*         r       = new_setlist();
         packed_ptr       partial = pack_(r, K::setlist_u8);
         build_guard      g{this, &partial};
         if (c == R.size())
            set_term(&r->rh, std::forward<VF>(val));  // the new key ends inside the prefix
         else
            setlist_set(r, uint8_t(R[c]), make_branch(std::forward<VF>(val)));
         partial = apply_prefix(r, key.substr(depth, c));  // == PP[0..c]; inline if short
         packed_ptr cont = (c + 1 == PP.size()) ? next : make_prefix(PP.substr(c + 1), next);
         setlist_set(r, uint8_t(PP[c]), cont);  // non-throwing graft of the live child
         g.release();
         node_free_(P, Psize);
         return partial;
      }

      // Split a setlist's INLINE prefix that diverges from the key at position c (< prefix_len):
      // a new parent setlist takes the shared part [0..c), routes the new value's branch, and
      // routes the old continuation byte back to this setlist (re-prefixed in place to [c+1..)).
      // Everything that can throw happens BEFORE the live setlist is grafted or mutated, so a
      // bad_alloc leaves the container unchanged (strong) and the guard reclaims only new nodes
      // — never `cur`. make_branch as in split_prefix.
      template <class MakeBranch, class VF>
      packed_ptr split_setlist_prefix(packed_ptr cur, std::string_view key, size_t depth,
                                      size_t c, VF&& val, MakeBranch&& make_branch)
      {
         setlist*       s   = static_cast<setlist*>(deref_(cur));
         const unsigned pl  = s->rh.hdr.prefix_len;
         uint8_t        common[setlist::PREFIX_CAP];
         std::memcpy(common, s->prefix, c);              // shared part [0..c)  (read, no mutation)
         const uint8_t  oldbyte = s->prefix[c];          // byte leading to the old setlist
         const unsigned newpl   = pl - unsigned(c) - 1;  // old setlist keeps [c+1..pl)
         setlist*       r       = new_setlist();
         packed_ptr     partial = pack_(r, K::setlist_u8);
         build_guard    g{this, &partial};
         std::string_view R = key.substr(depth);
         if (c == R.size())
            set_term(&r->rh, std::forward<VF>(val));  // the new key ends inside the prefix
         else
            setlist_set(r, uint8_t(R[c]), make_branch(std::forward<VF>(val)));
         partial = apply_prefix(r, std::string_view(reinterpret_cast<char*>(common), c));
         // commit (non-throwing): graft the live setlist, then re-prefix s in place
         setlist_set(r, oldbyte, cur);  // route old byte -> s (s still has its full prefix)
         std::memmove(s->prefix, s->prefix + c + 1, newpl);
         s->rh.hdr.prefix_len = newpl;
         g.release();
         return partial;
      }

      // Split a see-header wide router whose CL0 prefix diverges from the key at
      // offset c: a fresh setlist takes the shared part [0..c) (inline if <= 8,
      // else a legitimately-long prefix_node via apply_prefix), routes the old
      // byte back to this node (prefix shortened in place — re-certified to its
      // fast tag when the prefix empties) and the new key's branch beside it.
      template <class VF>
      packed_ptr split_hdr_prefix_(packed_ptr cur, std::string_view key, size_t depth,
                                   size_t c, VF&& val)
      {
         char*          p  = static_cast<char*>(deref_(cur));
         router_hdr*    rh = reinterpret_cast<router_hdr*>(p);
         const unsigned pl = rh->hdr.prefix_len;
         uint8_t        common[WPFX_CAP];
         std::memcpy(common, p + WPFX_OFF, c);
         const uint8_t  oldbyte = uint8_t(p[WPFX_OFF + c]);
         const unsigned newpl   = pl - unsigned(c) - 1;
         setlist*       r       = new_setlist();
         packed_ptr     partial = pack_(r, K::setlist_u8);
         build_guard    g{this, &partial};
         std::string_view R = key.substr(depth);
         if (c == R.size())
            set_term(&r->rh, std::forward<VF>(val));  // the new key ends inside the prefix
         else
            setlist_set(r, uint8_t(R[c]),
                        make_value_at(key, depth + c + 1, std::forward<VF>(val)));
         partial = apply_prefix(r, std::string_view(reinterpret_cast<char*>(common), c));
         // commit (non-throwing): graft cur, then shorten its prefix in place
         std::memmove(p + WPFX_OFF, p + WPFX_OFF + c + 1, newpl);
         rh->hdr.prefix_len = uint16_t(newpl);
         packed_ptr child = cur;
         if (newpl == 0)  // re-certify: header-free reads are valid again
            child = retag_(cur, hdr_base_(p));
         setlist_set(r, oldbyte, child);
         g.release();
         return partial;
      }

      // Move-assignment fallback for unequal, non-propagating allocators: re-insert every
      // element of `o` using OUR allocator (the foreign arena can't be stolen). Inline
      // values are trivially copyable; others are moved out of o's stored slots.
      void move_elements_from_(kv_tree& o)
      {
         for (auto it = o.begin(); it != o.end(); ++it)
         {
            if constexpr (inlineable)
               insert(it.key(), it.value());
            else
               insert(it.key(), std::move(*o.locate_(it.key_bytes())));
         }
      }

      // Tear down a (possibly partial) subtree: destroy each T AND free each node. Drives
      // teardown (~kv_tree / clear), and the exception-safety guard's cleanup of a half-built
      // subtree when a mutate op throws before committing it.
      void free_all_(packed_ptr cur) noexcept
      {
         if (cur.is_null())
            return;
         switch (cur.tag())
         {
            case K::value_ptr:
            {
               char* L = static_cast<char*>(deref_(cur));
               if constexpr (!std::is_trivially_destructible_v<T>) v_destroy(alloc_, leaf_val(L));
               vh_leaf_dies_(L);
               node_free_(L, leaf_alloc_size(L));
               return;
            }
            case K::value_inline:
               return;
            case K::prefix_node:
            {
               char* P = static_cast<char*>(deref_(cur));
               free_all_(pn_next<Ptr>(P));
               node_free_(P, pn_size<Ptr>(pn_plen(P)));
               return;
            }
            case K::setlist_u8:
            {  // inline children ride this line — release their bytes, never deref/free them
               setlist*  s  = static_cast<setlist*>(deref_(cur));
               free_all_(s->rh.term);
               const int nb = s->rh.hdr.nbranch;
               for (int i = 0; i < nb; ++i)
               {
                  if constexpr (inline_children_ok)
                     if (sl_is_inline(s, i)) { vh_leaf_dies_(sl_inline_leaf(s, i)); continue; }
                  free_all_(s->branch[i]);
               }
               node_free_(s, node_size(K::setlist_u8));
               return;
            }
            case K::setlist_u16:
            case K::c2:
            case K::c4:
            case K::c8:
            case K::node_full:
            case K::hdr:
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
               free_all_(rh->term);
               router_for_each(cur, [&](uint8_t, packed_ptr br) { free_all_(br); });
               node_free_(deref_(cur), cur.tag() == K::hdr
                                           ? node_size(hdr_base_(deref_(cur)))
                                           : node_size(cur.tag()));
               return;
            }
            case K::bucket:
            {
               bucket* b = static_cast<bucket*>(deref_(cur));
               if constexpr (!std::is_trivially_destructible_v<T>)
                  for (int i = 0; i < b->nent; ++i) v_destroy(alloc_, bkt_val(bkt_entry(b, i)));
               node_free_(b, sizeof(bucket));
               return;
            }
            case K::full_x:
            {
               if constexpr (cow_enabled)
               {
                  node_fullx* x = static_cast<node_fullx*>(deref_(cur));
                  free_all_(x->rh.term);
                  for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                     fx_release_line_(x->lines[li]);
                  node_free_(x, cacheline_bytes);
               }
               else
                  assert(false && "full_x exists only under COW allocators");
               return;
            }
            case K::cx:
            {
               if constexpr (cow_enabled)
               {
                  node_cx* x = static_cast<node_cx*>(deref_(cur));
                  free_all_(x->rh.term);
                  for (unsigned sg = 0; sg < (1u << x->lg); ++sg)
                     cx_release_line_(x->seglines[sg], x->lg);
                  node_free_(x, cacheline_bytes);
               }
               else
                  assert(false && "cx exists only under COW allocators");
               return;
            }
            case K::cbucket:
            {
               if constexpr (cb_enabled)
               {
                  cow_destroy_dead_(cur);  // releases term + every piece
                  node_free_(deref_(cur), cacheline_bytes);
               }
               else
                  assert(false && "cbucket needs the bytes bridge");
               return;
            }
            case K::null:
               return;
         }
      }
      // Nodes are allocated per-node (no bulk-reclaiming pool), so a node orphaned on a throwing
      // mid-build path is a real memory leak for ANY T — the guard must always reclaim. (Kept as
      // a flag so a future bulk-reclaiming allocator could set it true and elide the guard for
      // trivially-destructible T, as the old pool did.)
      static constexpr bool pool_bulk_reclaims = false;
      static constexpr bool guard_needed = !std::is_trivially_destructible_v<T> || !pool_bulk_reclaims;

      // RAII: frees the tracked partial subtree on unwind unless committed. Point `*slot` at the
      // outermost partial node as a mutate op builds it; call release() at the noexcept commit.
      struct build_guard
      {
         kv_tree*    self;
         packed_ptr* slot;
         bool        armed = true;
         ~build_guard()
         {
            if constexpr (guard_needed)
               if (armed && slot && !slot->is_null()) self->free_all_(*slot);
         }
         void release() noexcept { armed = false; }
      };

      // RAII for restride_u16_to_u8: on unwind, free the freshly-built u8 setlists SHELL-ONLY
      // (the new parent + its child setlists). The grandchildren they route to are still owned
      // by the live u16 node, so they must NOT be recursed into (that would double-free them).
      struct restride_guard
      {
         kv_tree* self;
         setlist* parent;
         bool     armed = true;
         ~restride_guard()
         {
            if (!armed) return;
            const int nb = parent->rh.hdr.nbranch;
            for (int i = 0; i < nb; ++i)
               self->node_free_(self->deref_(parent->branch[i]), sizeof(setlist));  // child setlists, shell-only
            self->node_free_(parent, sizeof(setlist));
         }
         void release() noexcept { armed = false; }
      };

      // ════ part COW extension (part's copy only — not upstream hkv) ═══════════
      // Strategy: privatize-then-mutate. cow_insert/cow_remove first clone every
      // node on the key's path whose head-line refcount is >= 2 (top-down;
      // cloning retains the clone's children, so sharedness propagates downward
      // automatically), then run the UNCHANGED in-place mutation on what is now
      // an exclusively-owned path. Old versions drain through cow_release
      // cascades when snapshots/root slots drop their references. Enabled when
      // the Allocator exposes per-line refcount hooks (part_alloc):
      //   hkv_ref / hkv_retain / hkv_release(line, destroy) / hkv_free_run.
     private:
      template <class A, class = void>
      struct cow_cap_ : std::false_type
      {
      };
      template <class A>
      struct cow_cap_<A, std::void_t<decltype(A::hkv_cow)>> : std::true_type
      {
      };

     public:
      static constexpr bool cow_enabled = cow_cap_<Allocator>::value && indexed;

     private:
      // vpart L1a: stamp-based liveness is active exactly when the COW
      // extension is (the part_alloc protocol carries the version hooks).
      static constexpr bool cow_stamped_ = cow_cap_<Allocator>::value && indexed;

     public:

      Ptr    root_handle() const noexcept { return root_; }
      size_t size_hint() const noexcept { return count_; }
      // vpart: the kv layer wires per-transaction state (vtxn, vlog) into the
      // tree's stored allocator between transactions.
      Allocator& allocator_mut() noexcept { return alloc_; }
      // Non-owning view management (snapshot cursors): adopt a pinned root and
      // null it back out before destruction so the dtor frees nothing.
      void adopt_root(Ptr p, size_t n = 0) noexcept
      {
         root_  = p;
         count_ = n;
      }
      uint32_t line_of(Ptr p) const noexcept { return uint32_t(p.raw() >> 4); }

      static constexpr unsigned PG_DIV_MAX = 36 + 256;
      struct pg_div  // a divider / radix-path buffer
      {
         uint8_t  b[PG_DIV_MAX];
         uint16_t len = 0;
         void     push(uint8_t c) noexcept
         {
            assert(len < PG_DIV_MAX);
            b[len++] = c;
         }
         void append(const void* p, size_t n) noexcept
         {
            assert(len + n <= PG_DIV_MAX);
            std::memcpy(b + len, p, n);
            len += uint16_t(n);
         }
         std::string_view view() const noexcept
         {
            return std::string_view(reinterpret_cast<const char*>(b), len);
         }
      };
      struct cb_eref  // one entry by reference: suffix + value views into the page
      {
         const uint8_t* sfx;
         const uint8_t* val;
         uint8_t        slen, vlen;
      };
      // ── v2 PAGE MODE (BUCKET_SPEC v2): diver rule + floor-routed divider radix ──
      // Keys terminate in self-contained cbucket pages; the radix maps divider ->
      // page with FLOOR semantics (page of the largest divider <= key); pages hang
      // ONLY at router terms (or root_). All entry points are pg_*.

      // floor route; when `div` is non-null it receives the routed page's divider
      // (the exact radix path — what pg_path_mut_ re-descends by).
      Ptr pg_route_(std::string_view key, pg_div* div = nullptr) const
      {
         Ptr    cur   = root_;
         size_t depth = 0;
         Ptr    fb_node = Ptr::null();  // deepest node passed that has a floor option
         int    fb_from = -1;
         size_t fb_dlen = 0;
         for (;;)
         {
            const K k = cur.tag();
            if (k == K::cbucket)
               return cur;
            if (k == K::prefix_node)
            {
               const char*      P = static_cast<const char*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
               std::string_view R = key.substr(depth);
               const size_t     c = lcp(PP, R);
               if (c < PP.size())
               {
                  if (c == R.size() || uint8_t(R[c]) < uint8_t(PP[c]))
                     return pg_fallback_(key, fb_node, fb_from, fb_dlen, div);
                  if (div)
                     div->append(PP.data(), PP.size());
                  return pg_max_(pn_next<Ptr>(const_cast<char*>(P)), div);
               }
               if (div)
                  div->append(PP.data(), PP.size());
               depth += PP.size();
               cur = pn_next<Ptr>(const_cast<char*>(P));
               continue;
            }
            // routers; a setlist may carry an inline prefix
            const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
            if (k == K::setlist_u8 && rh->hdr.prefix_len)
            {
               const setlist*   sl = static_cast<const setlist*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(sl->prefix), rh->hdr.prefix_len);
               std::string_view R = key.substr(depth);
               const size_t     c = lcp(PP, R);
               if (c < PP.size())
               {
                  if (c == R.size() || uint8_t(R[c]) < uint8_t(PP[c]))
                     return pg_fallback_(key, fb_node, fb_from, fb_dlen, div);
                  if (div)
                     div->append(PP.data(), PP.size());
                  return pg_max_(cur, div, /*skip_prefix=*/true);
               }
               if (div)
                  div->append(PP.data(), PP.size());
               depth += PP.size();
            }
            if (depth == key.size())
            {
               if (!rh->term.is_null())
                  return rh->term;
               return pg_fallback_(key, fb_node, fb_from, fb_dlen, div);
            }
            const uint8_t byte  = uint8_t(key[depth]);
            Ptr           child =
                with_router(cur, [&](auto* n) noexcept { return router_find_d(n, byte); });
            if (!child.is_null())
            {
               uint8_t ob;
               Ptr     oc;
               if (!rh->term.is_null() || router_prev_d(cur, byte, ob, oc))
               {  // a floor option exists at this node: remember it
                  fb_node = cur;
                  fb_from = byte;
                  fb_dlen = div ? div->len : 0;
               }
               if (div)
                  div->push(byte);
               cur = child;
               ++depth;
               continue;
            }
            uint8_t ob;
            Ptr     oc;
            if (router_prev_d(cur, byte, ob, oc))
            {
               if (div)
                  div->push(ob);
               return pg_max_(oc, div);
            }
            if (!rh->term.is_null())
               return rh->term;
            return pg_fallback_(key, fb_node, fb_from, fb_dlen, div);
         }
      }
      Ptr pg_fallback_(std::string_view, Ptr fb_node, int fb_from, size_t fb_dlen,
                       pg_div* div) const
      {
         assert(!fb_node.is_null() && "the radix root always holds the \"\" divider");
         if (div)
            div->len = uint16_t(fb_dlen);
         uint8_t ob;
         Ptr     oc;
         if (router_prev_d(fb_node, fb_from, ob, oc))
         {
            if (div)
               div->push(ob);
            return pg_max_(oc, div);
         }
         const router_hdr* rh = static_cast<const router_hdr*>(deref_(fb_node));
         assert(!rh->term.is_null());
         return rh->term;
      }
      // rightmost page of a subtree (largest divider under it)
      Ptr pg_max_(Ptr cur, pg_div* div, bool skip_prefix = false) const
      {
         for (;;)
         {
            const K k = cur.tag();
            if (k == K::cbucket)
               return cur;
            if (k == K::prefix_node)
            {
               const char* P = static_cast<const char*>(deref_(cur));
               if (div)
                  div->append(pn_pfx<Ptr>(P), pn_plen(P));
               cur = pn_next<Ptr>(const_cast<char*>(P));
               continue;
            }
            const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
            if (k == K::setlist_u8 && rh->hdr.prefix_len && !skip_prefix && div)
               div->append(static_cast<const setlist*>(deref_(cur))->prefix,
                           rh->hdr.prefix_len);
            skip_prefix = false;
            uint8_t ob;
            Ptr     oc;
            if (router_prev_d(cur, 256, ob, oc))
            {
               if (div)
                  div->push(ob);
               cur = oc;
               continue;
            }
            assert(!rh->term.is_null() && "a routers-only subtree must hold a page");
            return rh->term;
         }
      }
      // ── fused privatizing floor walk: ONE descent for writes (route and
      // privatize together; the old route-then-path_mut double walk is gone).
      // Nodes are privatized on entry and branch LINES before descending
      // through them; rare fallback turns may clone a node the final route
      // abandons — correct, marginal waste. Collects the divider.
      void pg_own_line_(Ptr cur, uint8_t byte)
      {
         if constexpr (cow_enabled)
         {
            const K k = cur.tag();
            if (k == K::full_x)
            {
               node_fullx* x = static_cast<node_fullx*>(deref_(cur));
               unsigned    li, sl2;
               full_index<Ptr>(byte, li, sl2);
               if (!x->lines[li].is_null() && alloc_.hkv_ref(line_of(x->lines[li])) >= 2)
               {
                  const Ptr old = x->lines[li];
                  x->lines[li]  = fx_clone_line_(old);
                  fx_release_line_(old);
               }
            }
            else if (k == K::cx)
            {
               node_cx*       x  = static_cast<node_cx*>(deref_(cur));
               const unsigned sg = byte >> (8 - x->lg);
               if (!x->seglines[sg].is_null() &&
                   alloc_.hkv_ref(line_of(x->seglines[sg])) >= 2)
               {
                  const Ptr old   = x->seglines[sg];
                  x->seglines[sg] = cx_clone_line_(old, x->lg);
                  cx_release_line_(old, x->lg);
               }
            }
         }
      }
      Ptr* pg_max_mut_(Ptr* slot, pg_div* div, bool skip_prefix = false)
      {
         for (;;)
         {
            if (slot->tag() == K::cbucket)
               return slot;  // pages are NOT privatized here: ops decide
            if (cow_shared_(*slot))
               cow_clone_into_(slot, *slot);
            Ptr     cur = *slot;
            const K k   = cur.tag();
            if (k == K::prefix_node)
            {
               char* P = static_cast<char*>(deref_(cur));
               if (div)
                  div->append(pn_pfx<Ptr>(P), pn_plen(P));
               slot = reinterpret_cast<Ptr*>(P + 2);
               continue;
            }
            router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
            if (k == K::setlist_u8 && rh->hdr.prefix_len && !skip_prefix && div)
               div->append(static_cast<const setlist*>(deref_(cur))->prefix,
                           rh->hdr.prefix_len);
            skip_prefix = false;
            uint8_t ob;
            Ptr     oc;
            if (router_prev_d(cur, 256, ob, oc))
            {
               pg_own_line_(cur, ob);
               Ptr* cs = with_router(cur, [&](auto* n) noexcept { return router_find_slot(n, ob); });
               if (div)
                  div->push(ob);
               slot = cs;
               continue;
            }
            assert(!rh->term.is_null() && "a routers-only subtree must hold a page");
            return &rh->term;
         }
      }
      Ptr* pg_fb_mut_(Ptr* fb_slot, int fb_from, size_t fb_dlen, pg_div* div)
      {
         assert(fb_slot && "the radix root always holds the \"\" divider");
         if (div)
            div->len = uint16_t(fb_dlen);
         Ptr     cur = *fb_slot;  // already privatized on the way down
         uint8_t ob;
         Ptr     oc;
         if (router_prev_d(cur, fb_from, ob, oc))
         {
            pg_own_line_(cur, ob);
            Ptr* cs = with_router(cur, [&](auto* n) noexcept { return router_find_slot(n, ob); });
            if (div)
               div->push(ob);
            return pg_max_mut_(cs, div);
         }
         router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
         assert(!rh->term.is_null());
         return &rh->term;
      }
      Ptr* pg_route_mut_(std::string_view key, pg_div* div)
      {
         Ptr*   slot    = &root_;
         size_t depth   = 0;
         Ptr*   fb_slot = nullptr;
         int    fb_from = -1;
         size_t fb_dlen = 0;
         for (;;)
         {
            if (slot->tag() == K::cbucket)
               return slot;  // pages are NOT privatized here: ops decide
            if (cow_shared_(*slot))
               cow_clone_into_(slot, *slot);
            Ptr     cur = *slot;
            const K k   = cur.tag();
            if (k == K::prefix_node)
            {
               char*            P = static_cast<char*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
               std::string_view R = key.substr(depth);
               const size_t     c = lcp(PP, R);
               if (c < PP.size())
               {
                  if (c == R.size() || uint8_t(R[c]) < uint8_t(PP[c]))
                     return pg_fb_mut_(fb_slot, fb_from, fb_dlen, div);
                  if (div)
                     div->append(PP.data(), PP.size());
                  return pg_max_mut_(reinterpret_cast<Ptr*>(P + 2), div);
               }
               if (div)
                  div->append(PP.data(), PP.size());
               depth += PP.size();
               slot = reinterpret_cast<Ptr*>(P + 2);
               continue;
            }
            router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
            if (k == K::setlist_u8 && rh->hdr.prefix_len)
            {
               const setlist*   sl = static_cast<const setlist*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(sl->prefix), rh->hdr.prefix_len);
               std::string_view R = key.substr(depth);
               const size_t     c = lcp(PP, R);
               if (c < PP.size())
               {
                  if (c == R.size() || uint8_t(R[c]) < uint8_t(PP[c]))
                     return pg_fb_mut_(fb_slot, fb_from, fb_dlen, div);
                  if (div)
                     div->append(PP.data(), PP.size());
                  return pg_max_mut_(slot, div, /*skip_prefix=*/true);
               }
               if (div)
                  div->append(PP.data(), PP.size());
               depth += PP.size();
            }
            if (depth == key.size())
            {
               if (!rh->term.is_null())
                  return &rh->term;
               return pg_fb_mut_(fb_slot, fb_from, fb_dlen, div);
            }
            const uint8_t byte = uint8_t(key[depth]);
            Ptr           child =
                with_router(cur, [&](auto* n) noexcept { return router_find_d(n, byte); });
            if (!child.is_null())
            {
               uint8_t ob;
               Ptr     oc;
               if (!rh->term.is_null() || router_prev_d(cur, byte, ob, oc))
               {
                  fb_slot = slot;
                  fb_from = byte;
                  fb_dlen = div ? div->len : 0;
               }
               pg_own_line_(cur, byte);
               Ptr* cs = with_router(cur, [&](auto* n) noexcept { return router_find_slot(n, byte); });
               if (div)
                  div->push(byte);
               slot = cs;
               ++depth;
               continue;
            }
            uint8_t ob;
            Ptr     oc;
            if (router_prev_d(cur, byte, ob, oc))
            {
               pg_own_line_(cur, ob);
               Ptr* cs = with_router(cur, [&](auto* n) noexcept { return router_find_slot(n, ob); });
               if (div)
                  div->push(ob);
               return pg_max_mut_(cs, div);
            }
            if (!rh->term.is_null())
               return &rh->term;
            return pg_fb_mut_(fb_slot, fb_from, fb_dlen, div);
         }
      }

      // equality re-descent by a collected divider, PRIVATIZING the path; returns
      // the slot holding the routed page (a router's term slot, or root_ itself).
      Ptr* pg_path_mut_(std::string_view D)
      {
         Ptr*   slot  = &root_;
         size_t depth = 0;
         for (;;)
         {
            if (slot->tag() == K::cbucket)
            {
               assert(depth == D.size());
               return slot;  // the page itself is NOT privatized here
            }
            if (cow_shared_(*slot))
               cow_clone_into_(slot, *slot);
            Ptr     cur = *slot;
            const K k   = cur.tag();
            if (k == K::prefix_node)
            {
               char* P = static_cast<char*>(deref_(cur));
               assert(D.compare(depth, pn_plen(P),
                                reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P)) == 0);
               depth += pn_plen(P);
               slot = reinterpret_cast<Ptr*>(P + 2);
               continue;
            }
            router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
            if (k == K::setlist_u8 && rh->hdr.prefix_len)
               depth += rh->hdr.prefix_len;  // verified by construction (route built D)
            if (depth == D.size())
            {
               assert(!rh->term.is_null());
               return &rh->term;
            }
            const uint8_t byte = uint8_t(D[depth]);
            // privatize the branch line for the explicit kinds (per-line COW)
            if constexpr (cow_enabled)
            {
               if (k == K::full_x)
               {
                  node_fullx* x = static_cast<node_fullx*>(deref_(cur));
                  unsigned    li, sl2;
                  full_index<Ptr>(byte, li, sl2);
                  if (!x->lines[li].is_null() &&
                      alloc_.hkv_ref(line_of(x->lines[li])) >= 2)
                  {
                     const Ptr old = x->lines[li];
                     x->lines[li]  = fx_clone_line_(old);
                     fx_release_line_(old);
                  }
               }
               else if (k == K::cx)
               {
                  node_cx*       x  = static_cast<node_cx*>(deref_(cur));
                  const unsigned sg = byte >> (8 - x->lg);
                  if (!x->seglines[sg].is_null() &&
                      alloc_.hkv_ref(line_of(x->seglines[sg])) >= 2)
                  {
                     const Ptr old   = x->seglines[sg];
                     x->seglines[sg] = cx_clone_line_(old, x->lg);
                     cx_release_line_(old, x->lg);
                  }
               }
            }
            Ptr* cs = with_router(cur, [&](auto* n) noexcept { return router_find_slot(n, byte); });
            assert(cs && !cs->is_null());
            slot = cs;
            ++depth;
            continue;
         }
      }
      // hang page R at divider D (which is NOT yet present). Path nodes are
      // privatized; missing structure is built; prefix edges split as needed.
      void pg_link_(std::string_view D, Ptr R)
      {
         Ptr*   slot  = &root_;
         size_t depth = 0;
         for (;;)
         {
            if (!slot->is_null() && cow_shared_(*slot))
               cow_clone_into_(slot, *slot);
            Ptr cur = *slot;
            if (cur.is_null())
            {
               *slot = pg_tail_(D, depth, R);
               return;
            }
            const K k = cur.tag();
            assert(k != K::cbucket && "pages hang at terms; link never lands on one");
            if (k == K::prefix_node)
            {
               char*            P = static_cast<char*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
               std::string_view R2 = D.substr(depth);
               const size_t     c  = lcp(PP, R2);
               if (c == PP.size())
               {
                  depth += c;
                  slot = reinterpret_cast<Ptr*>(P + 2);
                  continue;
               }
               // split the edge at c
               setlist* s2 = new_setlist();
               Ptr      sp = pack_(s2, K::setlist_u8);
               Ptr old_tail = pn_next<Ptr>(P);
               if (PP.size() - c > 1)
                  old_tail = make_prefix(std::string_view(PP.substr(c + 1)), old_tail);
               const bool ok = setlist_set(s2, uint8_t(PP[c]), old_tail);
               assert(ok);
               (void)ok;
               if (depth + c == D.size())
               {
                  s2->rh.term         = R;
                  s2->rh.hdr.has_term = 1;
               }
               else
               {
                  const bool ok2 =
                      setlist_set(s2, uint8_t(D[depth + c]), pg_tail_(D, depth + c + 1, R));
                  assert(ok2);
                  (void)ok2;
               }
               Ptr out = c ? make_prefix(std::string_view(PP.substr(0, c)), sp) : sp;
               *slot   = out;
               node_free_(P, pn_size<Ptr>(uint16_t(PP.size())));  // children moved
               return;
            }
            router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
            if (k == K::setlist_u8 && rh->hdr.prefix_len)
            {
               setlist*         sl = static_cast<setlist*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(sl->prefix), rh->hdr.prefix_len);
               if (lcp(PP, D.substr(depth)) < PP.size())
               {  // divider diverges inside the inline prefix: lift it, re-loop
                  uint8_t pfx[setlist::PREFIX_CAP];
                  std::memcpy(pfx, sl->prefix, PP.size());
                  Ptr pnp = make_prefix(std::string_view(reinterpret_cast<char*>(pfx), PP.size()), cur);
                  sl->rh.hdr.prefix_len = 0;
                  *slot                 = pnp;
                  continue;  // handled by the prefix_node arm
               }
               depth += PP.size();
            }
            if (depth == D.size())
            {
               assert(rh->term.is_null() && "divider already present");
               rh->term         = R;
               rh->hdr.has_term = 1;
               return;
            }
            const uint8_t byte = uint8_t(D[depth]);
            if constexpr (cow_enabled)
            {
               if (k == K::full_x)
               {
                  node_fullx* x = static_cast<node_fullx*>(deref_(cur));
                  unsigned    li, sl2;
                  full_index<Ptr>(byte, li, sl2);
                  if (!x->lines[li].is_null() &&
                      alloc_.hkv_ref(line_of(x->lines[li])) >= 2)
                  {
                     const Ptr old = x->lines[li];
                     x->lines[li]  = fx_clone_line_(old);
                     fx_release_line_(old);
                  }
               }
               else if (k == K::cx)
               {
                  node_cx*       x  = static_cast<node_cx*>(deref_(cur));
                  const unsigned sg = byte >> (8 - x->lg);
                  if (!x->seglines[sg].is_null() &&
                      alloc_.hkv_ref(line_of(x->seglines[sg])) >= 2)
                  {
                     const Ptr old   = x->seglines[sg];
                     x->seglines[sg] = cx_clone_line_(old, x->lg);
                     cx_release_line_(old, x->lg);
                  }
               }
            }
            Ptr* cs = with_router(cur, [&](auto* n) noexcept { return router_find_slot(n, byte); });
            if (cs)
            {
               slot = cs;
               ++depth;
               continue;
            }
            Ptr child = pg_tail_(D, depth + 1, R);
            build_guard gc{this, &child};
            grow_router_(slot, cur, byte, child);
            gc.release();
            return;
         }
      }
      Ptr pg_tail_(std::string_view D, size_t depth, Ptr R)
      {
         setlist* s2         = new_setlist();
         s2->rh.term         = R;
         s2->rh.hdr.has_term = 1;
         Ptr sp              = pack_(s2, K::setlist_u8);
         return depth == D.size() ? sp : make_prefix(D.substr(depth), sp);
      }
      // remove divider D: clears the term slot (caller releases the page) and
      // cascades now-empty nodes; then collapses a term-only root router.
      void pg_unlink_(std::string_view D)
      {
         if (pg_unlink_rec_(&root_, D, 0))
            root_ = Ptr::null();  // children leave slots dangling for the PARENT
         if (root_.is_null() || root_.tag() == K::cbucket || root_.tag() == K::prefix_node)
            return;
         router_hdr* rh = static_cast<router_hdr*>(deref_(root_));
         if (rh->hdr.nbranch == 0 && !rh->term.is_null())
         {  // term-only root router: the "" page becomes the root
            const Ptr t      = rh->term;
            rh->term         = Ptr::null();
            rh->hdr.has_term = 0;
            cow_release(root_);
            root_ = t;
         }
      }
      bool pg_unlink_rec_(Ptr* slot, std::string_view D, size_t depth)
      {
         if (cow_shared_(*slot))
            cow_clone_into_(slot, *slot);
         Ptr     cur = *slot;
         const K k   = cur.tag();
         if (k == K::prefix_node)
         {
            char* P = static_cast<char*>(deref_(cur));
            if (pg_unlink_rec_(reinterpret_cast<Ptr*>(P + 2), D, depth + pn_plen(P)))
            {
               node_free_(P, pn_size<Ptr>(pn_plen(P)));
               return true;  // slot left dangling: the PARENT's router_remove
            }                // must see non-null to fix its counts/lines
            return false;
         }
         router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
         if (k == K::setlist_u8 && rh->hdr.prefix_len)
            depth += rh->hdr.prefix_len;
         if (depth == D.size())
         {
            assert(!rh->term.is_null());
            rh->term         = Ptr::null();  // page released by the caller
            rh->hdr.has_term = 0;
         }
         else
         {
            const uint8_t byte = uint8_t(D[depth]);
            // privatize the branch LINE before writing through it (per-line COW —
            // pg_path_mut_/pg_link_ do the same; without this the descent slot
            // and router_remove mutate a line shared with older generations)
            if constexpr (cow_enabled)
            {
               if (k == K::full_x)
               {
                  node_fullx* x = static_cast<node_fullx*>(deref_(cur));
                  unsigned    li, sl2;
                  full_index<Ptr>(byte, li, sl2);
                  if (!x->lines[li].is_null() &&
                      alloc_.hkv_ref(line_of(x->lines[li])) >= 2)
                  {
                     const Ptr old = x->lines[li];
                     x->lines[li]  = fx_clone_line_(old);
                     fx_release_line_(old);
                  }
               }
               else if (k == K::cx)
               {
                  node_cx*       x  = static_cast<node_cx*>(deref_(cur));
                  const unsigned sg = byte >> (8 - x->lg);
                  if (!x->seglines[sg].is_null() &&
                      alloc_.hkv_ref(line_of(x->seglines[sg])) >= 2)
                  {
                     const Ptr old   = x->seglines[sg];
                     x->seglines[sg] = cx_clone_line_(old, x->lg);
                     cx_release_line_(old, x->lg);
                  }
               }
            }
            Ptr* cs = with_router(cur, [&](auto* n) noexcept { return router_find_slot(n, byte); });
            assert(cs && !cs->is_null());
            if (!pg_unlink_rec_(cs, D, depth + 1))
               return false;
            router_remove(cur, byte);  // nulls the (freed) slot + fixes counts
         }
         if (rh->hdr.nbranch == 0 && rh->term.is_null())
         {
            cow_release(cur);  // empty shell (no children to cascade into);
            return true;       // slot dangles — the parent cleans it up
         }
         return false;
      }

      // debug: assert a page's storage is totally ordered (stem, then suffix)
      void pg_validate_(Ptr pg, const char* tag) const
      {
         if (!getenv("PG_VAL"))
            return;
         const auto* h = static_cast<const cb_hdr*>(deref_(pg));
         std::string prev;
         bool        first = true;
         cb_for_each_(h, [&](std::string_view s2, std::string_view) {
            if (!first && !(prev < std::string(s2)))
            {
               fprintf(stderr, "PGVAL %s ORDER plen=%u nbr=%u cnt=[%u %u %u %u]:\n", tag,
                       unsigned(h->rh.hdr.prefix_len), unsigned(h->rh.hdr.nbranch), h->cnt[0],
                       h->cnt[1], h->cnt[2], h->cnt[3]);
               for (unsigned li = 0; li < 4; ++li)
               {
                  if (!h->cnt[li] || h->sp_line[li].is_null())
                     continue;
                  const cb_sp* sp = cb_spl_(h, li);
                  for (unsigned i2 = 0; i2 < h->cnt[li]; ++i2)
                  {
                     cb_cell c2 = cb_cell_at_(h, sp->lref[i2]);
                     fprintf(stderr, "  [%u:%u] stem=%04x lref=%04x slen=%u s=%.6s\n", li, i2,
                             sp->stems[i2], sp->lref[i2], c2.p[0],
                             reinterpret_cast<const char*>(c2.p + 3));
                  }
               }
               abort();
            }
            prev  = std::string(s2);
            first = false;
         });
      }

      // ── page-mode entry points ──────────────────────────────────────────────
      bool pg_find(std::string_view key, std::string_view& out) const
      {
         if constexpr (cb_enabled)
         {
            if (root_.is_null())
               return false;
            const Ptr     pg = pg_route_(key);
            const cb_hdr* h  = static_cast<const cb_hdr*>(deref_(pg));
            const std::string_view pfx(reinterpret_cast<const char*>(h->prefix),
                                       h->rh.hdr.prefix_len);
            if (key.size() < pfx.size() || key.compare(0, pfx.size(), pfx) != 0)
               return false;
            const std::string_view sfx = key.substr(pfx.size());
            if (sfx.empty())
            {
               if (h->rh.term.is_null())
                  return false;
               const auto b = leaf_val(static_cast<const char*>(deref_(h->rh.term)))->to_bytes();
               out          = std::string_view(b.data(), b.size());
               return true;
            }
            const uint8_t* c = cb_find_(h, sfx);
            if (!c)
            {
               if (getenv("PG_DBG2"))
               {
                  fprintf(stderr, "PGFIND miss: page nbr=%u plen=%u sfx.len=%zu sfx0=%02x cnt=[%u %u %u %u]\n",
                          unsigned(h->rh.hdr.nbranch), unsigned(h->rh.hdr.prefix_len),
                          sfx.size(), uint8_t(sfx[0]), h->cnt[0], h->cnt[1], h->cnt[2], h->cnt[3]);
                  cb_for_each_(h, [&](std::string_view s2, std::string_view) {
                     fprintf(stderr, "  entry slen=%zu s0=%02x%02x\n", s2.size(),
                             s2.size() > 0 ? uint8_t(s2[0]) : 0, s2.size() > 1 ? uint8_t(s2[1]) : 0);
                  });
               }
               return false;
            }
            out = cb_cell{const_cast<uint8_t*>(c)}.value();  // view into the page
            return true;
         }
         else
            return false;
      }
      template <bool Assign = true>
      bool pg_upsert(std::string_view key, std::string_view vbytes)
      {
         if constexpr (!cb_enabled)
            return false;
         else
         {
            if (root_.is_null())
               root_ = cb_make0_(key.substr(0, std::min<size_t>(key.size(), 36)));
            pg_div div;
            for (int pass = 0;; ++pass)
            {
               assert(pass < 64 && "page split did not converge");
               div.len = 0;
               Ptr* slot = pg_route_mut_(key, &div);  // ONE privatizing descent
               assert(key >= div.view() && "floor route: divider must not exceed the key");
               if (getenv("PG_OPS"))
                  fprintf(stderr, "OP pass=%d page=%u div.len=%u dtail=%02x%02x\n", pass,
                          unsigned(slot->raw() >> 4), div.len,
                          div.len > 1 ? div.b[div.len - 2] : 0,
                          div.len ? div.b[div.len - 1] : 0);
               assert(slot->tag() == K::cbucket && "divider path must end at a page");
               bool    shared = cow_shared_(*slot);
               cb_hdr* h      = static_cast<cb_hdr*>(deref_(*slot));
               std::string_view pfx(reinterpret_cast<const char*>(h->prefix),
                                    h->rh.hdr.prefix_len);
               pg_validate_(*slot, "pre-op");
               bool must_split = false;
               if (key.size() < pfx.size() || key.compare(0, pfx.size(), pfx) != 0)
               {  // loosen the page LCP, then re-run against the rebuilt page;
                  // when the loosened rebuild can't fit, split the ORIGINAL
                  // (its entries still share the tight prefix) and retry
                  const Ptr np = cb_reprefix_(*slot, lcp(key, pfx));
                  if (np.is_null())
                     must_split = true;
                  else
                  {
                     *slot = np;
                     h     = static_cast<cb_hdr*>(deref_(*slot));
                     pfx   = std::string_view(reinterpret_cast<const char*>(h->prefix),
                                              h->rh.hdr.prefix_len);
                     pg_validate_(*slot, "post-reprefix");
                  }
               }
               if (!must_split)
               {
               const std::string_view sfx = key.substr(pfx.size());
               if (shared && !sfx.empty() && !cb_find_(h, sfx) &&
                   h->rh.hdr.nbranch < CB_MAX)
               {  // SHARED page, NEW key: clone-with-the-gap in ONE pass — no
                  // verbatim clone followed by an index memmove (Dan)
                  const Ptr np = cb_clone_insert_(h, sfx, vbytes);
                  if (!np.is_null())
                  {
                     const Ptr old = *slot;
                     *slot         = np;
                     cow_release(old);  // one ref; other generations keep theirs
                     ++count_;
                     return true;
                  }
                  // no room in this shape: canonicalize-with-key / split below
                  // build from the SHARED source directly (no pre-clone)
               }
               else if (shared)
               {  // shared + update/term path: verbatim clone, mutate in place
                  cow_clone_into_(slot, *slot);
                  h      = static_cast<cb_hdr*>(deref_(*slot));
                  shared = false;  // private now: the normal insert path applies
               }
               if (sfx.empty())
               {  // key == page prefix: the page's own term
                  if (h->rh.term.is_null())
                  {
                     h->rh.term         = cb_leaf_(std::string_view{}, T::from_bytes(vbytes));
                     h->rh.hdr.has_term = 1;
                     ++count_;
                     return true;
                  }
                  if constexpr (Assign)
                  {
                     const Ptr old = h->rh.term;
                     h->rh.term    = cb_leaf_(std::string_view{}, T::from_bytes(vbytes));
                     cow_release(old);
                  }
                  return false;
               }
               int r = shared ? -3 : cb_insert_<Assign>(h, sfx, vbytes);
               if ((r == -2 || r == -3) &&
                   cb_reformat_fits_(h, sfx.size(), vbytes.size()))
               {  // a dense rebuild WITH the new key folded in — built straight
                  // from the (possibly shared) source, one pass, no pre-clone
                  cb_eref  refs[CB_MAX + 2];
                  unsigned n = cb_collect_refs_(h, refs);
                  // splice the new entry at its rank (refs are sorted)
                  unsigned at = 0;
                  while (at < n)
                  {
                     const std::string_view es(reinterpret_cast<const char*>(refs[at].sfx),
                                               refs[at].slen);
                     if (es >= sfx)
                        break;
                     ++at;
                  }
                  const bool dup = at < n && refs[at].slen == sfx.size() &&
                                   std::memcmp(refs[at].sfx, sfx.data(), sfx.size()) == 0;
                  if (dup)
                  {  // update-grow landed here: REPLACE, never duplicate
                     refs[at].val  = reinterpret_cast<const uint8_t*>(vbytes.data());
                     refs[at].vlen = uint8_t(vbytes.size());
                  }
                  else
                  {
                     std::memmove(refs + at + 1, refs + at, (n - at) * sizeof(cb_eref));
                     refs[at] = cb_eref{reinterpret_cast<const uint8_t*>(sfx.data()),
                                        reinterpret_cast<const uint8_t*>(vbytes.data()),
                                        uint8_t(sfx.size()), uint8_t(vbytes.size())};
                     ++n;
                  }
                  if (n <= CB_MAX + (refs[0].slen == 0 ? 1 : 0))
                  {
                     const std::string_view pfx2(reinterpret_cast<const char*>(h->prefix),
                                                 h->rh.hdr.prefix_len);
                     const Ptr np = cb_build_page_bulk_(pfx2, {}, refs, 0, n, 0);
                     if (!np.is_null())
                     {
                        const Ptr old = *slot;
                        *slot         = np;
                        cow_release(old);
                        count_ += unsigned(!dup);
                        return !dup;
                     }
                  }
                  r = -1;  // fall to split
               }
               if (getenv("PG_OPS"))
                  fprintf(stderr, "OP   r=%d nbr=%u\n", r, unsigned(h->rh.hdr.nbranch));
               if (r >= 0)
               {
                  pg_validate_(*slot, "post-insert");
                  count_ += unsigned(r == 1);
                  return r == 1;
               }
               }  // !must_split
               // full: split by the shortest round separator near the median, retry
               cb_eref        refs[CB_MAX + 1];
               const unsigned n  = cb_collect_refs_(h, refs);
               const auto     sp = cb_get_split_sep_(refs, n);
               pg_div         nd;  // divider = page prefix + separator
               nd.append(pfx.data(), pfx.size());
               nd.append(sp.b, sp.len);
               if (getenv("PG_DBG"))
               {
                  auto hx = [](const void* p, size_t n2) {
                     std::string o;
                     for (size_t j = 0; j < n2; ++j)
                     { char b2[4]; snprintf(b2, 4, "%02x", ((const uint8_t*)p)[j]); o += b2; }
                     return o;
                  };
                  fprintf(stderr, "SPLIT div=%s pfx.len=%zu n=%u at=%u sep=%s s0=%s sN=%s\n",
                          hx(div.b, div.len).c_str(), pfx.size(), n, sp.index,
                          hx(sp.b, sp.len).c_str(), hx(refs[0].sfx, refs[0].slen).c_str(),
                          hx(refs[n - 1].sfx, refs[n - 1].slen).c_str());
               }
               // tighten each half's prefix by its own suffix-LCP (cap 36 total)
               auto half_pfx = [&](unsigned l2, unsigned h2, unsigned& scl) {
                  scl = es_scl_(refs, l2, h2);
                  scl = std::min<size_t>(scl, pfx.size() < 36 ? 36 - pfx.size() : 0);
                  return scl;
               };
               uint8_t  pb[36];
               unsigned sclL, sclR;
               half_pfx(0, sp.index, sclL);
               std::memcpy(pb, pfx.data(), pfx.size());
               std::memcpy(pb + pfx.size(), refs[0].sfx, sclL);
               const Ptr L = cb_build_page_bulk_(
                   std::string_view(reinterpret_cast<const char*>(pb), pfx.size() + sclL),
                   {}, refs, 0, sp.index, sclL);
               half_pfx(sp.index, n, sclR);
               std::memcpy(pb + pfx.size(), refs[sp.index].sfx, sclR);
               const Ptr R = cb_build_page_bulk_(
                   std::string_view(reinterpret_cast<const char*>(pb), pfx.size() + sclR),
                   {}, refs, sp.index, n, sclR);
               assert(!L.is_null() && !R.is_null() && "split halves fit by construction");
               pg_validate_(L, "split-L");
               pg_validate_(R, "split-R");
               // L/R built from views into the OLD page: release it only now
               if (slot == &root_)
               {  // the root page grows a radix above it (the "" divider = term)
                  const Ptr old       = root_;
                  setlist*  s2        = new_setlist();
                  s2->rh.term         = L;
                  s2->rh.hdr.has_term = 1;
                  root_               = pack_(s2, K::setlist_u8);
                  pg_link_(nd.view(), R);
                  cow_release(old);
               }
               else
               {
                  const Ptr old = *slot;  // re-resolve below: pg_link_ can restructure
                  pg_link_(nd.view(), R);
                  Ptr* s3 = pg_path_mut_(div.view());
                  assert(s3->raw() == old.raw());
                  *s3 = L;
                  cow_release(old);
               }
            }
         }
      }
      bool pg_remove(std::string_view key)
      {
         if constexpr (!cb_enabled)
            return false;
         else
         {
            if (root_.is_null())
               return false;
            {  // const miss-check first: no privatization on misses
               std::string_view tmp;
               if (!pg_find(key, tmp))
                  return false;
            }
            pg_div div;
            Ptr*   slot = pg_route_mut_(key, &div);  // ONE privatizing descent
            assert(slot->tag() == K::cbucket && "divider path must end at a page");
            if (cow_shared_(*slot))
               cow_clone_into_(slot, *slot);
            cb_hdr*                h = static_cast<cb_hdr*>(deref_(*slot));
            const std::string_view pfx(reinterpret_cast<const char*>(h->prefix),
                                       h->rh.hdr.prefix_len);
            const std::string_view sfx = key.substr(pfx.size());
            if (sfx.empty())
            {
               cow_release(h->rh.term);
               h->rh.term         = Ptr::null();
               h->rh.hdr.has_term = 0;
            }
            else
            {
               const bool ok = cb_remove_(h, sfx);
               assert(ok);
               (void)ok;
            }
            --count_;
            if (h->rh.hdr.nbranch == 0 && !h->rh.hdr.has_term)
            {
               if (slot == &root_)
               {
                  cow_release(root_);
                  root_ = Ptr::null();
               }
               else if (div.len)
               {  // drop the divider; the "" page persists while dividers remain
                  const Ptr dead = *slot;
                  pg_unlink_(div.view());
                  cow_release(dead);
                  // the unlink may have collapsed the radix onto the "" page:
                  // if that page is ALSO empty, the tree is empty
                  if (!root_.is_null() && root_.tag() == K::cbucket)
                  {
                     const cb_hdr* rhh = static_cast<const cb_hdr*>(deref_(root_));
                     if (rhh->rh.hdr.nbranch == 0 && !rhh->rh.hdr.has_term)
                     {
                        cow_release(root_);
                        root_ = Ptr::null();
                     }
                  }
               }
            }
            return true;
         }
      }


      // +1 on a subtree root (root-slot / snapshot bookkeeping). False would
      // mean 190 state refs on one node — unreachable under sane root usage.
      bool cow_retain(Ptr p)
      {
         (void)p;  // vpart: sharing is FREE — no per-node state to bump
         return true;
      }

      // ── vpart liveness: supersession instead of reference counting ─────────
      //
      // cow_release(p) records that the CURRENT transaction dropped the
      // subtree at p from the tree (a DEEP supersession):
      //   • window-born (stamp == txn): never visible to anyone — free now,
      //     cascading inline; old children met during the cascade are deep-
      //     logged (their death is this window).
      //   • reclaim context (the drain thread, floor already passed): free
      //     now unconditionally, cascading through everything reachable.
      //   • otherwise: one 8 B deep log entry. The drain expands it later.
      //
      // cow_release_shallow_(p) is the CLONE path's supersession: children
      // were carried by the successor, so the drain frees p's own lines only.
      void cow_release(Ptr p) noexcept
      {
         if constexpr (cow_enabled)
         {
            if (p.is_null() || p.tag() == K::value_inline)
               return;
            if (alloc_.vreclaim || !alloc_.hkv_vold(vstamp_get_(p)))
            {
               cow_destroy_dead_(p);  // cascade children (window/drain rules inside)
               vh_node_dies_(p);
               node_free_(deref_(p), cow_node_bytes_(p));
               return;
            }
            alloc_.hkv_vdeep(uint32_t(p.raw()), vstamp_get_(p));
         }
      }
      void cow_release_shallow_(Ptr p) noexcept
      {
         if constexpr (cow_enabled)
         {
            if (p.is_null() || p.tag() == K::value_inline)
               return;
            // privatize only clones SHARED (old) nodes; a window-born node is
            // already private and never re-cloned.
            assert(alloc_.vreclaim || alloc_.hkv_vold(vstamp_get_(p)));
            alloc_.hkv_vshallow(uint32_t(p.raw()), vstamp_get_(p));
         }
      }
      uint32_t rel_budget_ = 64;  // kept for source compat; unused by vpart
      void     release_budget(uint32_t n) noexcept { rel_budget_ = n; }
      // drain entry: one committed clone-log entry (raw, birth|DEEP)
      void vdrain_entry(uint32_t raw, uint32_t birth_flags) noexcept
      {
         if constexpr (cow_enabled)
         {
            Ptr p;
            std::memcpy(p.b, &raw, 4);
            if (p.is_null() || p.tag() == K::value_inline)
               return;
            if (birth_flags & 0x80000000u)  // DEEP: dropped subtree — cascade
            {
               assert(alloc_.vreclaim);
               cow_release(p);
            }
            else  // SHALLOW: children carried by the successor — free p only
            {
               vh_node_dies_(p);  // a superseded leaf's bytes die with the husk
               // inline children are NOT carried (the clone has its own vh_copy_'d copies):
               // their bytes die with this husk line, so balance the clone's copy hook here.
               if constexpr (vh_on)
                  if (p.tag() == K::setlist_u8)
                     sl_inline_vh_dies_(static_cast<setlist*>(deref_(p)));
               node_free_(deref_(p), cow_node_bytes_(p));
            }
         }
      }
      void cow_release_now(Ptr p) noexcept { cow_release(p); }
      // ABORT walk: free the diverged window. Recurse only into nodes the
      // CURRENT transaction created; anything older is shared with the
      // restored root and must survive untouched (and unlogged).
      void vabort_release(Ptr p) noexcept
      {
         if constexpr (cow_enabled)
         {
            if (p.is_null() || p.tag() == K::value_inline)
               return;
            if (alloc_.hkv_vold(vstamp_get_(p)))
               return;  // pre-txn node: the restored root still owns it
            switch (p.tag())
            {
               case K::prefix_node:
                  vabort_release(pn_next<Ptr>(static_cast<char*>(deref_(p))));
                  break;
               case K::setlist_u8:
               {  // inline children ride this line: release their bytes, never recurse/free them
                  setlist*  s  = static_cast<setlist*>(deref_(p));
                  vabort_release(s->rh.term);
                  const int nb = s->rh.hdr.nbranch;
                  for (int i = 0; i < nb; ++i)
                  {
                     if constexpr (inline_children_ok)
                        if (sl_is_inline(s, i)) { vh_leaf_dies_(sl_inline_leaf(s, i)); continue; }
                     vabort_release(s->branch[i]);
                  }
                  break;
               }
               case K::setlist_u16:
               case K::c2:
               case K::c4:
               case K::c8:
               case K::node_full:
               case K::hdr:
               {
                  router_hdr* rh = static_cast<router_hdr*>(deref_(p));
                  vabort_release(rh->term);
                  router_for_each(p, [&](unsigned, Ptr c) { vabort_release(c); });
                  break;
               }
               default:
                  break;  // leaves/buckets: no children
            }
            vh_node_dies_(p);
            node_free_(deref_(p), cow_node_bytes_(p));
         }
      }

      template <class V>
      bool cow_insert(key_param k, V&& v)
      {
         rel_budget_ = 64;  // per-op inline release budget
         static_assert(cow_enabled, "allocator has no refcount hooks");
         static_assert(std::is_trivially_copyable_v<T>, "COW memcpy-clones leaf values");
         // privatize is FUSED into insert_'s descent (clone-if-shared per hop)
         T tmp(std::forward<V>(v));
         return insert_<true>(k, std::move(tmp));
      }
      // NOTE: remove()'s fusion paths free child nodes unconditionally; callers
      // must not hold snapshots across cow_remove. Fusion-under-sharing is
      // future work; the bank workload never removes.
      bool cow_remove(key_param k)
      {
         rel_budget_ = 64;  // per-op inline release budget
         static_assert(cow_enabled, "allocator has no refcount hooks");
         // privatize is FUSED into remove's descent (clone-if-shared per hop)
         return remove(k);
      }

     public:
      // ════ composed bucket (BUCKET_SPEC.md) ═══════════════════════════════════
      // CL0 + FP line + 4 ragged stem/lref lines + <=12 sub-pages (2 lines,
      // bump-only, free_space dead ledger) + dedicated lines for big entries.
      // Cells store LIVE value bytes (self-describing, vtype.has_cap capacity
      // record in the shrink gap) — variable-length values, sized to user data.
      struct cb_hdr
      {
         router_hdr rh;  // offset 0: term machinery reused verbatim
         uint8_t    open_spi;
         uint8_t    open_off8;
         uint8_t    cnt[4];
         uint16_t   stem_lo[4];
         uint8_t    nbig;
         uint8_t    dead8;   // saturating dead-byte ledger (8B units): >0 means
                             // the next COW clone COMPACTS instead of memcpy
         uint8_t    rsv[2];  // (fp line removed: stems verify by suffix)
         Ptr        sp_line[4];
         Ptr        page_line[12];
         uint8_t    prefix[36];
      };
      struct cb_sp
      {
         uint16_t stems[16];
         uint16_t lref[16];
         uint8_t  rsv[64];
      };
      static constexpr unsigned CB_MAX     = 64;   // entries
      static constexpr unsigned CB_PAGES   = 12;   // sub-page table slots
      static constexpr unsigned CB_PGCAP8  = 31;   // payload 8B units per page (248B)
      static constexpr unsigned CB_CELLMAX = 64;   // bytes; bigger -> dedicated lines
      static constexpr uint8_t  VT_HASCAP  = 0x80;

      static constexpr uint16_t cb_lref(unsigned big, unsigned spi, unsigned off8,
                                        unsigned len8) noexcept
      {
         return uint16_t((big << 15) | (spi << 11) | (off8 << 6) | len8);
      }
      static constexpr unsigned cb_big(uint16_t r) noexcept { return r >> 15; }
      static constexpr unsigned cb_spi(uint16_t r) noexcept { return (r >> 11) & 15; }
      static constexpr unsigned cb_off8(uint16_t r) noexcept { return (r >> 6) & 31; }
      static constexpr unsigned cb_len8(uint16_t r) noexcept { return r & 63; }

      // cell: [klen u8][vlen u8][vtype u8][suffix][value][cap u8 if has_cap]
      static unsigned cb_cell_cap_(const uint8_t* c) noexcept
      {
         return (c[2] & VT_HASCAP) ? c[3 + c[0] + c[1]] : c[1];
      }
      static unsigned cb_cell_alloc8_(const uint8_t* c) noexcept
      {
         return (3u + c[0] + cb_cell_cap_(c) + 7) / 8;
      }

      uint8_t* cb_page_(const cb_hdr* h, unsigned spi) const noexcept
      {
         return static_cast<uint8_t*>(deref_(h->page_line[spi]));
      }
      cb_sp* cb_spl_(const cb_hdr* h, unsigned li) const noexcept
      {
         return static_cast<cb_sp*>(deref_(h->sp_line[li]));
      }
      static uint16_t cb_stem_(std::string_view sfx) noexcept
      {
         const uint8_t a = sfx.size() > 0 ? uint8_t(sfx[0]) : 0;
         const uint8_t b = sfx.size() > 1 ? uint8_t(sfx[1]) : 0;
         return uint16_t((a << 8) | b);
      }

      struct cb_cell
      {
         uint8_t*         p;
         std::string_view suffix() const
         {
            return std::string_view(reinterpret_cast<const char*>(p + 3), p[0]);
         }
         std::string_view value() const
         {
            return std::string_view(reinterpret_cast<const char*>(p + 3 + p[0]), p[1]);
         }
      };
      cb_cell cb_cell_at_(const cb_hdr* h, uint16_t lr) const noexcept
      {
         uint8_t* base = cb_big(lr)
                             ? static_cast<uint8_t*>(deref_(h->page_line[cb_spi(lr)]))
                             : cb_page_(h, cb_spi(lr)) + cb_off8(lr) * 8;
         return cb_cell{base};
      }

      // Total-order placement: the line and in-line slot of the first entry
      // >= (stem, sfx). Storage is kept FULLY sorted — stems ascend across the
      // concatenation of nonempty lines and equal-stem runs are suffix-sorted —
      // so iteration and bursts never sort. Holes (lines emptied by removes)
      // are skipped and stay dead until a rebalance repacks them. Cell derefs
      // happen only on equal-stem ties.
      std::pair<unsigned, unsigned> cb_place_(const cb_hdr* h, uint16_t stem,
                                              std::string_view sfx) const noexcept
      {
         unsigned last = 0;
         bool     any  = false;
         for (unsigned li = 0; li < 4; ++li)
         {
            if (!h->cnt[li] || h->sp_line[li].is_null())
               continue;
            const cb_sp* sp  = cb_spl_(h, li);
            unsigned     pos = cb_lb_(sp, h->cnt[li], stem);
            while (pos < h->cnt[li] && sp->stems[pos] == stem &&
                   cb_cell_at_(h, sp->lref[pos]).suffix() < sfx)
               ++pos;
            if (pos < h->cnt[li])
               return {li, pos};  // first entry >= (stem, sfx) is here
            last = li;            // whole line below the new entry: keep walking
            any  = true;
         }
         return {last, any ? unsigned(h->cnt[last]) : 0u};  // append after everything
      }
      static unsigned cb_lb_(const cb_sp* sp, unsigned n, uint16_t stem) noexcept
      {
         unsigned i = 0;
         while (i < n && sp->stems[i] < stem)
            ++i;
         return i;
      }

      const uint8_t* cb_find_(const cb_hdr* h, std::string_view sfx) const noexcept
      {
         const uint16_t stem = cb_stem_(sfx);
         for (unsigned li = 0; li < 4; ++li)
         {
            if (!h->cnt[li] || h->sp_line[li].is_null())
               continue;
            if (h->stem_lo[li] > stem)
               break;  // lines ascend by stem_lo: the run cannot start later
            const cb_sp* sp = cb_spl_(h, li);
            if (sp->stems[h->cnt[li] - 1] < stem)
               continue;  // whole line below the stem
            for (unsigned i = cb_lb_(sp, h->cnt[li], stem);
                 i < h->cnt[li] && sp->stems[i] == stem; ++i)
            {
               cb_cell c = cb_cell_at_(h, sp->lref[i]);
               if (c.suffix() == sfx)
                  return c.p;
            }
         }
         return nullptr;
      }

      // bump-only payload allocation per BUCKET_SPEC §4/§5.
      // Returns lref, or 0xFFFF when the page table is exhausted (split signal).
      uint16_t cb_alloc_cell_(cb_hdr* h, std::string_view sfx, std::string_view val)
      {
         const unsigned bytes = 3 + unsigned(sfx.size()) + unsigned(val.size());
         const unsigned u8s   = (bytes + 7) / 8;
         if (bytes > CB_CELLMAX)
         {
            unsigned spi = CB_PAGES;
            for (unsigned i = 0; i < CB_PAGES; ++i)
               if (h->page_line[i].is_null())
               {
                  spi = i;
                  break;
               }
            if (spi == CB_PAGES)
               return 0xFFFF;
            const unsigned nl = (bytes + 127) / 128;
            void*          d  = node_alloc_(nl * 128ull);
            // BUCKET_SPEC: the spare tag bits distinguish {subpage | bigval} —
            // the small-cell bump path must never treat a dedicated big line
            // as a sub-page (open_spi can name a slot a big alloc claimed)
            h->page_line[spi] = pack_(d, K::value_ptr);  // bigval slot
            ++h->nbig;
            uint8_t* c = static_cast<uint8_t*>(d);
            c[0]       = uint8_t(sfx.size());
            c[1]       = uint8_t(val.size());
            c[2]       = 0;
            std::memcpy(c + 3, sfx.data(), sfx.size());
            std::memcpy(c + 3 + sfx.size(), val.data(), val.size());
            return cb_lref(1, spi, 0, nl & 63);
         }
         for (int attempt = 0; attempt < 2; ++attempt)
         {
            const bool open_is_sub = !h->page_line[h->open_spi].is_null() &&
                                     h->page_line[h->open_spi].tag() == K::node_full;
            if (open_is_sub && h->open_off8 + u8s <= 1 + CB_PGCAP8)
            {
               uint8_t*       pg = cb_page_(h, h->open_spi);
               uint8_t*       c  = pg + h->open_off8 * 8;
               const uint16_t lr = cb_lref(0, h->open_spi, h->open_off8, u8s);
               h->open_off8      = uint8_t(h->open_off8 + u8s);
               c[0]              = uint8_t(sfx.size());
               c[1]              = uint8_t(val.size());
               c[2]              = 0;
               std::memcpy(c + 3, sfx.data(), sfx.size());
               std::memcpy(c + 3 + sfx.size(), val.data(), val.size());
               return lr;
            }
            if (open_is_sub)
            {  // close the open page: fold its untouched tail into free_space
               uint8_t* pg = cb_page_(h, h->open_spi);
               pg[0]       = uint8_t(pg[0] + (1 + CB_PGCAP8 - h->open_off8));
            }
            unsigned spi = CB_PAGES;
            for (unsigned i = 0; i < CB_PAGES; ++i)
               if (h->page_line[i].is_null())
               {
                  spi = i;
                  break;
               }
            if (spi == CB_PAGES)
               return 0xFFFF;
            void* d = node_alloc_(256);
            std::memset(d, 0, 8);
            h->page_line[spi] = pack_(d, K::node_full);
            h->open_spi       = uint8_t(spi);
            h->open_off8      = 1;
         }
         return 0xFFFF;
      }

      void cb_free_cell_(cb_hdr* h, uint16_t lr)
      {
         if (cb_big(lr))
         {
            const unsigned spi = cb_spi(lr);
            cb_release_lines_(h->page_line[spi], cb_len8(lr) ? cb_len8(lr) : 1);
            h->page_line[spi] = Ptr::null();
            --h->nbig;
            h->dead8 = uint8_t(std::min(255u, unsigned(h->dead8) + 1));  // slot churn
            return;
         }
         const unsigned spi = cb_spi(lr);
         uint8_t* pg = cb_page_(h, spi);  // page private with its CL0 (COW unit)
         cb_cell  c  = cb_cell_at_(h, lr);
         const unsigned a8 = cb_cell_alloc8_(c.p);
         pg[0]             = uint8_t(pg[0] + a8);
         h->dead8          = uint8_t(std::min(255u, unsigned(h->dead8) + a8));
         if (spi != h->open_spi && pg[0] >= CB_PGCAP8)
         {  // fully dead page: release; the table slot becomes reusable
            cb_release_lines_(h->page_line[spi], 2);
            h->page_line[spi] = Ptr::null();
         }
      }

      // ── self-privatization: bucket ops own (clone-if-shared) exactly the
      // lines they are about to write; CL0 is privatized by the descent.
      Ptr cb_copy_lines_(Ptr lp, unsigned nl)
      {
         void* d = node_alloc_(nl * 128ull);
         std::memcpy(d, deref_(lp), nl * 128ull);
         return pack_(d, lp.tag());
      }
      Ptr cb_clone_lines_(Ptr lp, unsigned nl)
      {
         void* d = node_alloc_(nl * 128ull);
         std::memcpy(d, deref_(lp), nl * 128ull);
         return pack_(d, lp.tag());  // preserve the {subpage|bigval} slot tag
      }
      void cb_release_lines_(Ptr lp, unsigned nl) noexcept
      {
         alloc_.hkv_release(line_of(lp),
                            [&](uint32_t head)
                            {
                               if (nl > 1)
                                  alloc_.hkv_free_run(head + 1, nl - 1);
                            });
      }

      // T <-> raw cell bytes bridge: cbucket activates only when T opts in.
      template <class X, class = void>
      struct cb_cap_ : std::false_type
      {
      };
      template <class X>
      struct cb_cap_<X, std::void_t<decltype(X::from_bytes(std::string_view{})),
                                    decltype(std::declval<const X&>().to_bytes())>>
          : std::true_type
      {
      };
#ifndef PART_CBUCKET
#define PART_CBUCKET 0  // OFF (Dan, 2026-06-11): pages amplified copies/complexity vs the pure radix and did not pay on disk pages. The page engine stays behind this gate; the PURE RADIX (occupancy ladder, no buckets) is the configuration of record.
                        // WIP: 3 oracle-found bugs fixed (run-spanning lookup,
                        // recursive split, burst capacity); a duplicate-update
                        // edge remains — needs the BUCKET_SPEC TDD pass before
                        // default-on.
#endif
      static constexpr bool cb_enabled = cow_enabled && cb_cap_<T>::value;
      static constexpr bool cb_create  = cb_enabled && (PART_CBUCKET != 0);

      // Leaf for a bucket-relative SUFFIX: make_value_at would record the
      // suffix as the leaf's zero-copy FULL key (the path prefix is not in
      // scope here), so clear LEAF_FULL — layout is identical at plen==0 and
      // the iterator reconstructs the key from the frame stack instead.
      Ptr cb_leaf_(std::string_view sfx, T&& v)
      {
         Ptr p = make_value_at(sfx, 0, std::move(v));
         if (p.tag() == K::value_ptr)
         {
            char*    L = static_cast<char*>(deref_(p));
            uint16_t f;
            std::memcpy(&f, L + 2, 2);
            f &= uint16_t(~LEAF_FULL);
            std::memcpy(L + 2, &f, 2);
         }
         return p;
      }

      // create a bucket from two colliding entries (the leaf-split replacement):
      // CL0 + FP + SP0 + one page = 5 lines
      struct cb_ent_  // (sfx holds a SUFFIX in the v1 burst, a FULL key in v2 pages)
      {
         std::string sfx, val;
      };
      // a fresh EMPTY page: CL0 + FP + first SP line, no entries (the diver
      // rule starts every tree as one of these and splits never re-shard)
      Ptr cb_make0_(std::string_view pfx)
      {
         cb_hdr* h = static_cast<cb_hdr*>(node_alloc_(128));
         std::memset(h, 0, 128);
         h->rh.term = Ptr::null();
         for (auto& pl : h->page_line)
            pl = Ptr::null();
         for (auto& sl : h->sp_line)
            sl = Ptr::null();
         h->rh.hdr.prefix_len = uint16_t(std::min<size_t>(pfx.size(), 36));
         std::memcpy(h->prefix, pfx.data(), std::min<size_t>(pfx.size(), 36));
         h->sp_line[0] = pack_(node_alloc_(128), K::node_full);  // no memset:
         // stems/lrefs beyond cnt are never read
         h->open_spi  = 0;
         h->open_off8 = 0;  // page_line[0] null: first alloc opens it
         return pack_(h, K::cbucket);
      }
      Ptr cb_make2_(std::string_view pfx, std::string_view s1, std::string_view v1,
                    std::string_view s2, std::string_view v2)
      {
         const Ptr hp = cb_make0_(pfx);
         cb_hdr*   h  = static_cast<cb_hdr*>(deref_(hp));
         // a key that IS the common prefix has an empty suffix: it lives in
         // the term (lookups for it end at depth == key.size())
         auto put = [&](std::string_view sfx, std::string_view val)
         {
            if (sfx.empty())
            {
               h->rh.term         = cb_leaf_(std::string_view{}, T::from_bytes(val));
               h->rh.hdr.has_term = 1;
            }
            else
               cb_insert_(h, sfx, val);
         };
         put(s1, v1);
         put(s2, v2);
         return hp;
      }

      // ── v2 page machinery — NO dynamic temporaries (Dan): splits/rebuilds
      // work on stack REF arrays pointing into the live page (cells stay valid
      // until the old page is released, after the builds), and dividers live in
      // fixed buffers bounded by prefix(36) + suffix(255) + 1.
      // refs in key order (term — the prefix-exact key, slen 0 — first); count.
      unsigned cb_collect_refs_(const cb_hdr* h, cb_eref* out) const
      {
         unsigned n = 0;
         if (!h->rh.term.is_null())
         {
            const auto b = leaf_val(static_cast<const char*>(deref_(h->rh.term)))->to_bytes();
            out[n++] = cb_eref{nullptr, reinterpret_cast<const uint8_t*>(b.data()), 0,
                               uint8_t(b.size())};
         }
         for (unsigned li = 0; li < 4; ++li)
         {
            if (!h->cnt[li] || h->sp_line[li].is_null())
               continue;
            const cb_sp* sp = cb_spl_(h, li);
            for (unsigned i = 0; i < h->cnt[li]; ++i)
            {
               cb_cell c = cb_cell_at_(h, sp->lref[i]);
               out[n++]  = cb_eref{c.p + 3, c.p + 3 + c.p[0], c.p[0], c.p[1]};
            }
         }
         return n;
      }
      // ONE page from ref entries [lo,hi), built by SEQUENTIAL COPY — the
      // entries are already sorted, so cells bump straight into fresh
      // sub-pages/dedicated lines and the SP/FP index is written once, in
      // order, spread evenly across the four lines (no per-entry search, no
      // memmove, no cb_insert_). `prepend` rides in front of every suffix
      // (reprefix loosening); splits pass "". New prefix = old_pfx + prepend?
      // — callers state the page prefix explicitly. Returns null when the
      // entry mix exhausts the 12-slot table (caller splits first).
      Ptr cb_build_page_bulk_(std::string_view pfx, std::string_view prepend,
                              const cb_eref* es, unsigned lo, unsigned hi, unsigned strip)
      {
         assert(hi > lo && hi - lo <= CB_MAX + 1);
         const Ptr hp = cb_make0_(pfx);
         cb_hdr*   h  = static_cast<cb_hdr*>(deref_(hp));
         unsigned  b0 = lo;
         {  // an entry that IS the page prefix becomes the term
            const unsigned el = unsigned(prepend.size()) + es[lo].slen - strip;
            if (el == 0)
            {
               const std::string_view val(reinterpret_cast<const char*>(es[lo].val),
                                          es[lo].vlen);
               h->rh.term         = cb_leaf_(std::string_view{}, T::from_bytes(val));
               h->rh.hdr.has_term = 1;
               ++b0;
            }
         }
         const unsigned n = hi - b0;
         if (n == 0)
            return hp;
         if (n > CB_MAX)
         {  // a reprefix can grow the term into a 65th CELL: page can't hold
            // it — report overflow so the caller splits the original first
            if (!h->rh.term.is_null())
               cow_release(h->rh.term);
            cb_release_lines_(h->sp_line[0], 1);
            node_free_(h, 128);
            return Ptr::null();
         }
         // single sequential pass: cells bump-copied, index written in order
         uint16_t stems[CB_MAX];
         uint16_t lrefs[CB_MAX];
         uint8_t  slot_nl[CB_PAGES];  // true line count per slot (failure cleanup)
         unsigned open = ~0u, next_slot = 0;
         uint8_t  off8 = 1;
         // a partial page has NO lrefs in its index yet: the generic destroy
         // cannot size its big slots, so failure releases pieces by hand
         auto fail = [&]() noexcept {
            for (unsigned t = 0; t < next_slot; ++t)
               if (!h->page_line[t].is_null())
                  cb_release_lines_(h->page_line[t], slot_nl[t]);
            if (!h->rh.term.is_null())
               cow_release(h->rh.term);
            for (unsigned li = 0; li < 4; ++li)
               if (!h->sp_line[li].is_null())
                  cb_release_lines_(h->sp_line[li], 1);
            node_free_(h, 128);
            return Ptr::null();
         };
         for (unsigned i = 0; i < n; ++i)
         {
            const cb_eref& e    = es[b0 + i];
            const unsigned slen = unsigned(prepend.size()) + e.slen - strip;
            const unsigned bytes = 3 + slen + e.vlen;
            const unsigned u8s   = (bytes + 7) / 8;
            uint8_t*       c;
            if (bytes > CB_CELLMAX)
            {  // dedicated line(s)
               if (next_slot >= CB_PAGES)
                  return fail();
               const unsigned nl = (bytes + 127) / 128;
               void*          d  = node_alloc_(nl * 128ull);
               slot_nl[next_slot]      = uint8_t(nl);
               h->page_line[next_slot] = pack_(d, K::value_ptr);  // bigval slot
               ++h->nbig;
               c        = static_cast<uint8_t*>(d);
               lrefs[i] = cb_lref(1, next_slot++, 0, nl & 63);
            }
            else
            {
               if (open == ~0u || off8 + u8s > 1 + CB_PGCAP8)
               {
                  if (open != ~0u)  // close: fold the untouched tail
                     cb_page_(h, open)[0] =
                         uint8_t(cb_page_(h, open)[0] + (1 + CB_PGCAP8 - off8));
                  if (next_slot >= CB_PAGES)
                     return fail();
                  void* d = node_alloc_(256);
                  std::memset(d, 0, 8);
                  slot_nl[next_slot]      = 2;
                  h->page_line[next_slot] = pack_(d, K::node_full);  // subpage
                  open                    = next_slot++;
                  off8                    = 1;
               }
               c        = cb_page_(h, open) + off8 * 8;
               lrefs[i] = cb_lref(0, open, off8, u8s);
               off8     = uint8_t(off8 + u8s);
            }
            c[0] = uint8_t(slen);
            c[1] = e.vlen;
            c[2] = 0;
            std::memcpy(c + 3, prepend.data(), prepend.size());
            std::memcpy(c + 3 + prepend.size(), e.sfx + strip, e.slen - strip);
            std::memcpy(c + 3 + slen, e.val, e.vlen);
            const std::string_view sv(reinterpret_cast<const char*>(c + 3), slen);
            stems[i] = cb_stem_(sv);
         }
         h->open_spi  = uint8_t(open == ~0u ? 0 : open);
         h->open_off8 = open == ~0u ? 0 : off8;
         // SP/FP written once, spread evenly (the rebalance layout)
         const unsigned per = (n + 3) / 4;
         unsigned       k2  = 0;
         for (unsigned li = 0; li < 4 && k2 < n; ++li)
         {
            const unsigned take = std::min(per, n - k2);
            if (h->sp_line[li].is_null())
               h->sp_line[li] = pack_(node_alloc_(128), K::node_full);
            cb_sp* sp = cb_spl_(h, li);
            std::memcpy(sp->stems, stems + k2, take * sizeof(uint16_t));
            std::memcpy(sp->lref, lrefs + k2, take * sizeof(uint16_t));
            h->cnt[li]     = uint8_t(take);
            h->stem_lo[li] = sp->stems[0];
            k2 += take;
         }
         h->rh.hdr.nbranch = uint16_t(n);
         return hp;
      }

      static unsigned es_scl_(const cb_eref* es, unsigned lo, unsigned hi) noexcept
      {
         unsigned scl = es[lo].slen;
         for (unsigned i = lo + 1; i < hi && scl; ++i)
         {
            const unsigned m = std::min<unsigned>(scl, es[i].slen);
            unsigned       c = 0;
            while (c < m && es[lo].sfx[c] == es[i].sfx[c])
               ++c;
            scl = c;
         }
         return scl;
      }
      // heart's split rule in SUFFIX space: candidates in the middle window
      // [3n/8, 5n/8]; the separator between neighbours a,b is b[0..p]+v with
      // v in (a[p], b[p]] snapped toward TRAILING ZEROS; pick (shortest,
      // roundest, most median). The divider = page prefix + separator.
      struct cb_sep_
      {
         uint16_t index;
         uint8_t  len;
         uint8_t  b[260];
      };
      static cb_sep_ cb_get_split_sep_(const cb_eref* es, unsigned n)
      {
         assert(n >= 2);
         // Dan's rule — no candidate scan: probe the two keys at median ±10%
         // (60 keys → 24 and 36). Their common prefix is the divider prefix;
         // the next byte is the MOST-ZEROS value strictly between them — one
         // xor + clz (clear b's bits below the highest bit differing from a).
         // The divider lands between the probes, so both halves keep >=~40%.
         unsigned i = (n * 4) / 10, j = (n * 6) / 10;
         if (j <= i)
            j = i + 1;
         if (j >= n)
         {
            j = n - 1;
            i = j - 1;
         }
         const cb_eref& a = es[i];
         const cb_eref& b = es[j];
         const unsigned m = a.slen < b.slen ? a.slen : b.slen;
         unsigned       p = 0;
         while (p < m && a.sfx[p] == b.sfx[p])
            ++p;
         assert(p < b.slen && "strict order: b cannot be a prefix of a");
         uint8_t v;
         if (p >= a.slen)
            v = 0;  // a IS the shared prefix: any byte qualifies; 0 = max zeros
         else
         {
            const uint8_t d = uint8_t(a.sfx[p] ^ b.sfx[p]);  // a<b ⇒ nonzero
            const int     k = 31 - __builtin_clz(unsigned(d));
            v               = uint8_t(b.sfx[p] & ~((1u << k) - 1u));  // >a[p], <=b[p]
         }
         cb_sep_ r;
         r.len = uint8_t(p + 1);
         std::memcpy(r.b, b.sfx, p);
         r.b[p] = v;
         // partition = first entry >= separator; it lies in (i, j]
         unsigned idx = i + 1;
         for (; idx <= j; ++idx)
         {
            const cb_eref& e  = es[idx];
            const unsigned mm = e.slen < unsigned(p + 1) ? e.slen : unsigned(p + 1);
            const int      c  = mm ? std::memcmp(e.sfx, r.b, mm) : 0;
            if (c > 0 || (c == 0 && e.slen >= unsigned(p + 1)))
               break;  // e >= separator
         }
         r.index = uint16_t(idx);
         return r;
      }
      // fused shared-page INSERT: build the private successor with the new
      // entry already in place — one pass, partial copies only (used SP
      // prefixes with the gap pre-made, used open-subpage prefix, full closed
      // pieces; no rsv halves, no memset of unallocated space, no memmove).
      // Returns null when the layout needs more (full SP line / table): the
      // caller canonicalizes-with-key or splits — both from the SHARED source.
      Ptr cb_clone_insert_(const cb_hdr* h, std::string_view sfx, std::string_view val)
      {
         const unsigned bytes = 3 + unsigned(sfx.size()) + unsigned(val.size());
         const unsigned u8s   = (bytes + 7) / 8;
         const bool     big   = bytes > CB_CELLMAX;
         auto [li, pos]       = cb_place_(h, cb_stem_(sfx), sfx);
         if (h->cnt[li] >= 16)
            return Ptr::null();  // needs a rebalance: rebuild path handles it
         // where does the new cell go?
         unsigned tgt = CB_PAGES;  // table slot for {big | new subpage}
         bool     into_open = false;
         const bool open_is_sub = !h->page_line[h->open_spi].is_null() &&
                                  h->page_line[h->open_spi].tag() == K::node_full;
         if (!big && open_is_sub && h->open_off8 + u8s <= 1 + CB_PGCAP8)
            into_open = true;
         else
         {
            for (unsigned i = 0; i < CB_PAGES; ++i)
               if (h->page_line[i].is_null())
               {
                  tgt = i;
                  break;
               }
            if (tgt == CB_PAGES)
               return Ptr::null();  // table exhausted: reformat/split decides
         }
         cb_hdr* d = static_cast<cb_hdr*>(node_alloc_(128));
         std::memcpy(d, h, 128);
         cow_fix_child_(&d->rh.term);
         // SP lines: used halves only; the insertion line gets the gap pre-made
         for (unsigned l2 = 0; l2 < 4; ++l2)
         {
            if (d->sp_line[l2].is_null())
               continue;
            const cb_sp* src = cb_spl_(h, l2);
            const unsigned n2 = h->cnt[l2];
            if (n2 == 0 && l2 != li)
            {
               d->sp_line[l2] = Ptr::null();  // hole: nothing to carry
               continue;
            }
            cb_sp* ns      = static_cast<cb_sp*>(node_alloc_(128));
            d->sp_line[l2] = pack_(ns, K::node_full);
            if (l2 == li)
            {
               std::memcpy(ns->stems, src->stems, pos * 2);
               std::memcpy(ns->stems + pos + 1, src->stems + pos, (n2 - pos) * 2);
               std::memcpy(ns->lref, src->lref, pos * 2);
               std::memcpy(ns->lref + pos + 1, src->lref + pos, (n2 - pos) * 2);
               ns->stems[pos] = cb_stem_(sfx);
               // lref written below once the cell lands
            }
            else
            {
               std::memcpy(ns->stems, src->stems, n2 * 2);
               std::memcpy(ns->lref, src->lref, n2 * 2);
            }
         }
         if (d->sp_line[li].is_null())
         {  // first entry of a fresh line
            cb_sp* ns      = static_cast<cb_sp*>(node_alloc_(128));
            d->sp_line[li] = pack_(ns, K::node_full);
            ns->stems[0]   = cb_stem_(sfx);
         }
         // payload pieces: closed verbatim; the open subpage copies its USED
         // prefix only (the dead tail past alloc_pos is never carried)
         uint8_t nl_of[CB_PAGES];
         for (auto& x2 : nl_of)
            x2 = 2;
         for (unsigned l2 = 0; l2 < 4; ++l2)
         {
            if (!h->cnt[l2] || h->sp_line[l2].is_null())
               continue;
            const cb_sp* sp = cb_spl_(h, l2);
            for (unsigned i = 0; i < h->cnt[l2]; ++i)
               if (cb_big(sp->lref[i]))
                  nl_of[cb_spi(sp->lref[i])] =
                      uint8_t(cb_len8(sp->lref[i]) ? cb_len8(sp->lref[i]) : 1);
         }
         for (unsigned spi = 0; spi < CB_PAGES; ++spi)
         {
            if (h->page_line[spi].is_null())
               continue;
            const bool sub = h->page_line[spi].tag() == K::node_full;
            if (sub && spi == h->open_spi && open_is_sub)
            {  // open page: copy [0, alloc_pos) only
               void* nd = node_alloc_(256);
               std::memcpy(nd, deref_(h->page_line[spi]), size_t(h->open_off8) * 8);
               d->page_line[spi] = pack_(nd, K::node_full);
            }
            else
               d->page_line[spi] = cb_copy_lines_(h->page_line[spi], sub ? 2 : nl_of[spi]);
         }
         // land the new cell
         uint16_t lr;
         if (into_open)
         {
            uint8_t* pg = static_cast<uint8_t*>(deref_(d->page_line[d->open_spi])) +
                          size_t(d->open_off8) * 8;
            lr = cb_lref(0, d->open_spi, d->open_off8, u8s);
            d->open_off8 = uint8_t(d->open_off8 + u8s);
            pg[0]        = uint8_t(sfx.size());
            pg[1]        = uint8_t(val.size());
            pg[2]        = 0;
            std::memcpy(pg + 3, sfx.data(), sfx.size());
            std::memcpy(pg + 3 + sfx.size(), val.data(), val.size());
         }
         else if (big)
         {
            const unsigned nl = (bytes + 127) / 128;
            uint8_t*       c2 = static_cast<uint8_t*>(node_alloc_(nl * 128ull));
            d->page_line[tgt] = pack_(c2, K::value_ptr);
            ++d->nbig;
            c2[0] = uint8_t(sfx.size());
            c2[1] = uint8_t(val.size());
            c2[2] = 0;
            std::memcpy(c2 + 3, sfx.data(), sfx.size());
            std::memcpy(c2 + 3 + sfx.size(), val.data(), val.size());
            lr = cb_lref(1, tgt, 0, nl & 63);
         }
         else
         {  // fresh subpage (close the old open one's tail into free_space)
            if (open_is_sub)
            {
               uint8_t* opg = static_cast<uint8_t*>(deref_(d->page_line[d->open_spi]));
               opg[0]       = uint8_t(opg[0] + (1 + CB_PGCAP8 - d->open_off8));
            }
            uint8_t* nd = static_cast<uint8_t*>(node_alloc_(256));
            std::memset(nd, 0, 8);
            d->page_line[tgt] = pack_(nd, K::node_full);
            d->open_spi       = uint8_t(tgt);
            d->open_off8      = uint8_t(1 + u8s);
            nd[8]             = uint8_t(sfx.size());
            nd[9]             = uint8_t(val.size());
            nd[10]            = 0;
            std::memcpy(nd + 11, sfx.data(), sfx.size());
            std::memcpy(nd + 11 + sfx.size(), val.data(), val.size());
            lr = cb_lref(0, tgt, 1, u8s);
         }
         cb_spl_(d, li)->lref[pos] = lr;
         ++d->cnt[li];
         d->stem_lo[li] = cb_spl_(d, li)->stems[0];
         ++d->rh.hdr.nbranch;
         return pack_(d, K::cbucket);
      }
      // would a compact reformat make room for one more (slen, vlen) cell?
      // Exact scan of live extents — runs only at the would-split decision.
      bool cb_reformat_fits_(const cb_hdr* h, size_t slen, size_t vlen) const
      {
         const unsigned bytes = 3 + unsigned(slen) + unsigned(vlen);
         unsigned       live8 = 0, bigs = 0;
         for (unsigned li = 0; li < 4; ++li)
         {
            if (!h->cnt[li] || h->sp_line[li].is_null())
               continue;
            const cb_sp* sp = cb_spl_(h, li);
            for (unsigned i = 0; i < h->cnt[li]; ++i)
            {
               if (cb_big(sp->lref[i]))
                  ++bigs;
               else
                  live8 += cb_cell_alloc8_(cb_cell_at_(h, sp->lref[i]).p);
            }
         }
         unsigned need_small8 = live8, need_bigs = bigs;
         if (bytes > CB_CELLMAX)
            ++need_bigs;
         else
            need_small8 += (bytes + 7) / 8;
         const unsigned sub = (need_small8 + CB_PGCAP8 - 1) / CB_PGCAP8;
         return sub + need_bigs <= CB_PAGES;  // optimistic: a failed retry splits
      }
      // demand-time compaction: rebuild THIS (private) page densely
      Ptr cb_canonicalize_(Ptr cur)
      {
         cb_hdr*        h = static_cast<cb_hdr*>(deref_(cur));
         cb_eref        refs[CB_MAX + 1];
         const unsigned n = cb_collect_refs_(h, refs);
         const std::string_view pfx(reinterpret_cast<const char*>(h->prefix),
                                    h->rh.hdr.prefix_len);
         const Ptr np = n ? cb_build_page_bulk_(pfx, {}, refs, 0, n, 0) : cb_make0_(pfx);
         if (np.is_null())
            return np;
         cow_release(cur);
         return np;
      }
      // a key diverging from the page LCP loosens it: rebuild with the shorter
      // prefix (heart: "factoring only ever loosens"); returns the new page.
      // The stripped prefix bytes are re-joined per entry in one stack buffer.
      Ptr cb_reprefix_(Ptr cur, size_t new_plen)
      {
         cb_hdr* h = static_cast<cb_hdr*>(deref_(cur));
         assert(new_plen < h->rh.hdr.prefix_len);
         cb_eref        refs[CB_MAX + 1];
         const unsigned n = cb_collect_refs_(h, refs);
         const std::string_view old_pfx(reinterpret_cast<const char*>(h->prefix),
                                        h->rh.hdr.prefix_len);
         // stripped prefix bytes ride in front of every suffix — ONE sequential
         // copy into the new page, no per-entry machinery
         const Ptr np = cb_build_page_bulk_(old_pfx.substr(0, new_plen),
                                            old_pfx.substr(new_plen), refs, 0, n, 0);
         if (np.is_null())
            return np;  // caller splits the original first
         cow_release(cur);
         return np;
      }

      // insert-or-update. Returns: 0 updated, 1 inserted, -1 needs split.
      template <bool Assign = true>
      int cb_insert_(cb_hdr* h, std::string_view sfx, std::string_view val)
      {
         const uint16_t stem = cb_stem_(sfx);
         // update path: equal-stem runs may span SP lines
         for (unsigned uli = 0; uli < 4; ++uli)
         {
            if (!h->cnt[uli] || h->sp_line[uli].is_null())
               continue;
            if (h->stem_lo[uli] > stem)
               break;
            cb_sp* usp = cb_spl_(h, uli);
            if (usp->stems[h->cnt[uli] - 1] < stem)
               continue;
            for (unsigned i = cb_lb_(usp, h->cnt[uli], stem);
                 i < h->cnt[uli] && usp->stems[i] == stem; ++i)
            {
               cb_cell c = cb_cell_at_(h, usp->lref[i]);
               if (c.suffix() != sfx)
                  continue;
               usp           = cb_spl_(h, uli);
               const unsigned li = uli;
               cb_sp*         sp = usp;
               (void)li;
               (void)sp;
            if constexpr (!Assign)
               return 0;  // present: emplace semantics leave it untouched
            const unsigned cap = cb_cell_cap_(c.p);
            if (val.size() <= cap)
            {  // in place — own the cell's page first
               c = cb_cell_at_(h, sp->lref[i]);
               c.p[1] = uint8_t(val.size());
               std::memcpy(c.p + 3 + c.p[0], val.data(), val.size());
               if (val.size() < cap)
               {
                  c.p[2] |= VT_HASCAP;
                  c.p[3 + c.p[0] + c.p[1]] = uint8_t(cap);
               }
               else
                  c.p[2] &= uint8_t(~VT_HASCAP);
               return 0;
            }
            const uint16_t nlr = cb_alloc_cell_(h, sfx, val);
            if (nlr == 0xFFFF)
               return -2;  // no space: a reformat may make room
            cb_free_cell_(h, cb_spl_(h, uli)->lref[i]);
            cb_spl_(h, uli)->lref[i] = nlr;
            return 0;
            }
         }
         // insert path
         if (h->rh.hdr.nbranch >= CB_MAX)
            return -1;
         auto [li, pos] = cb_place_(h, stem, sfx);
         cb_sp* sp = h->sp_line[li].is_null() ? nullptr : cb_spl_(h, li);
         if (h->cnt[li] == 16)
         {  // rebalance: redistribute evenly (page already private)
            cb_rebalance_(h);
            std::tie(li, pos) = cb_place_(h, stem, sfx);  // entries moved
            sp                = cb_spl_(h, li);
            if (h->cnt[li] == 16)
               return -1;  // pathological all-equal stems: split
         }
         const uint16_t lr = cb_alloc_cell_(h, sfx, val);
         if (lr == 0xFFFF)
            return -2;  // no space: a reformat may make room
         if (h->sp_line[li].is_null())
         {
            h->sp_line[li] = pack_(node_alloc_(128), K::node_full);
            sp             = cb_spl_(h, li);
         }
         std::memmove(&sp->stems[pos + 1], &sp->stems[pos],
                      (h->cnt[li] - pos) * sizeof(uint16_t));
         std::memmove(&sp->lref[pos + 1], &sp->lref[pos],
                      (h->cnt[li] - pos) * sizeof(uint16_t));
         sp->stems[pos] = stem;
         sp->lref[pos]  = lr;
         ++h->cnt[li];
         h->stem_lo[li] = sp->stems[0];
         ++h->rh.hdr.nbranch;
         return 1;
      }

      // spread entries evenly across the 4 SP lines (keeps global stem order;
      // all index lines are private when this runs)
      void cb_rebalance_(cb_hdr* h)
      {
         uint16_t st[CB_MAX], lr[CB_MAX];
         unsigned n   = 0;
         for (unsigned li = 0; li < 4; ++li)
         {
            if (h->cnt[li] == 0)
               continue;
            cb_sp* sp = cb_spl_(h, li);
            for (unsigned i = 0; i < h->cnt[li]; ++i, ++n)
            {
               st[n] = sp->stems[i];
               lr[n] = sp->lref[i];
            }
         }
         const unsigned per = (n + 3) / 4;
         unsigned       k   = 0;
         for (unsigned li = 0; li < 4; ++li)
         {
            const unsigned take = std::min(per, n - k);
            if (take && h->sp_line[li].is_null())
            {
               h->sp_line[li] = pack_(node_alloc_(128), K::node_full);
               std::memset(deref_(h->sp_line[li]), 0, 128);
            }
            h->cnt[li] = uint8_t(take);
            if (!take)
            {
               h->stem_lo[li] = 0xFFFF;
               continue;
            }
            cb_sp* sp = cb_spl_(h, li);
            std::memcpy(sp->stems, st + k, take * sizeof(uint16_t));
            std::memcpy(sp->lref, lr + k, take * sizeof(uint16_t));
            h->stem_lo[li] = sp->stems[0];
            k += take;
         }
      }

      bool cb_remove_(cb_hdr* h, std::string_view sfx)
      {
         const uint16_t stem = cb_stem_(sfx);
         for (unsigned li = 0; li < 4; ++li)
         {
         if (!h->cnt[li] || h->sp_line[li].is_null())
            continue;
         if (h->stem_lo[li] > stem)
            break;
         {
            cb_sp* probe = cb_spl_(h, li);
            if (probe->stems[h->cnt[li] - 1] < stem)
               continue;
         }
         cb_sp* sp = cb_spl_(h, li);
         for (unsigned i = cb_lb_(sp, h->cnt[li], stem);
              i < h->cnt[li] && sp->stems[i] == stem; ++i)
         {
            cb_cell c = cb_cell_at_(h, sp->lref[i]);
            if (c.suffix() != sfx)
               continue;
            cb_free_cell_(h, sp->lref[i]);
            std::memmove(&sp->stems[i], &sp->stems[i + 1],
                         (h->cnt[li] - 1 - i) * sizeof(uint16_t));
            std::memmove(&sp->lref[i], &sp->lref[i + 1],
                         (h->cnt[li] - 1 - i) * sizeof(uint16_t));
            --h->cnt[li];
            h->stem_lo[li] = h->cnt[li] ? sp->stems[0] : 0xFFFF;
            --h->rh.hdr.nbranch;
            return true;
         }
         }
         return false;
      }

      // visit (suffix, value bytes) for every entry, stem-ordered (equal-stem
      // runs suffix-sorted by the caller when full order matters)
      template <class F>
      void cb_for_each_(const cb_hdr* h, F&& f) const
      {
         for (unsigned li = 0; li < 4; ++li)
         {
            if (!h->cnt[li])
               continue;
            const cb_sp* sp = cb_spl_(h, li);
            for (unsigned i = 0; i < h->cnt[li]; ++i)
            {
               cb_cell c = cb_cell_at_(h, sp->lref[i]);
               f(c.suffix(), c.value());
            }
         }
      }

      // overflow: burst into children keyed by the next byte; groups bigger
      // than a comfortable bucket recurse (strip shared bytes into prefix
      // wrappers), so no entry is ever dropped regardless of skew.
      Ptr cb_build_(std::vector<cb_ent_>& es, size_t lo, size_t hi)
      {
         if (hi - lo == 1)
         {
            T tmp = T::from_bytes(es[lo].val);
            return cb_leaf_(std::string_view(es[lo].sfx), std::move(tmp));
         }
         // strip the run of shared leading bytes into a wrapper prefix
         std::string shared;
         for (;;)
         {
            if (es[lo].sfx.empty())
               break;
            const char b   = es[lo].sfx[0];
            bool       all = true;
            for (size_t i = lo + 1; i < hi && all; ++i)
               all = !es[i].sfx.empty() && es[i].sfx[0] == b;
            if (!all)
               break;
            shared.push_back(b);
            for (size_t i = lo; i < hi; ++i)
               es[i].sfx.erase(0, 1);
         }
         Ptr out;
         if (hi - lo <= 48)
         {
            // fits one bucket comfortably (equal stems spread across 4 lines)
            size_t      first = lo;
            std::string tval;
            bool        has_term = false;
            if (es[lo].sfx.empty())
            {
               tval     = es[lo].val;
               has_term = true;
               ++first;
            }
            assert(hi - first >= 1);
            Ptr bp;
            if (hi - first == 1)
            {
               if (has_term)
               {  // a 1-entry group with a term: ONE tiny bucket, no throwaway leaf
                  bp       = cb_make2_(std::string_view{}, std::string_view{}, tval,
                                       es[first].sfx, es[first].val);
                  has_term = false;
               }
               else
               {
                  T tmp = T::from_bytes(es[first].val);
                  bp    = cb_leaf_(std::string_view(es[first].sfx), std::move(tmp));
               }
            }
            else
            {
               bp = cb_make2_(std::string_view{}, es[first].sfx, es[first].val,
                              es[first + 1].sfx, es[first + 1].val);
               cb_hdr* ch = static_cast<cb_hdr*>(deref_(bp));
               for (size_t k2 = first + 2; k2 < hi; ++k2)
               {
                  const int r = cb_insert_(ch, es[k2].sfx, es[k2].val);
                  assert(r >= 0 && "<=48 entries must fit a fresh bucket");
                  (void)r;
               }
               if (has_term)
               {
                  T tmp = T::from_bytes(tval);
                  ch->rh.term = cb_leaf_(std::string_view{}, std::move(tmp));
                  ch->rh.hdr.has_term = 1;
               }
            }
            out = bp;
         }
         else
         {
            // count distinct lead bytes: a setlist only when it fits, else a
            // node_full (a dropped branch was the silent-loss bug)
            size_t   groups = 0;
            for (size_t i = lo + (es[lo].sfx.empty() ? 1 : 0); i < hi;)
            {
               const uint8_t b = uint8_t(es[i].sfx[0]);
               while (i < hi && !es[i].sfx.empty() && uint8_t(es[i].sfx[0]) == b)
                  ++i;
               ++groups;
            }
            router_hdr* rh2;
            Ptr         rp;
            const bool  use_full = groups > size_t(setlist::CAP);
            if (use_full)
            {
               node_full* f = static_cast<node_full*>(node_alloc_(sizeof(node_full)));
               full_init(f);
               rh2 = &f->rh;
               rp  = pack_(f, K::node_full);
            }
            else
            {
               setlist* s2 = new_setlist();
               rh2         = &s2->rh;
               rp          = pack_(s2, K::setlist_u8);
            }
            size_t i = lo;
            if (es[i].sfx.empty())
            {
               T tmp = T::from_bytes(es[i].val);
               rh2->term         = cb_leaf_(std::string_view{}, std::move(tmp));
               rh2->hdr.has_term = 1;
               ++i;
            }
            while (i < hi)
            {
               const uint8_t b = uint8_t(es[i].sfx[0]);
               size_t        j = i;
               while (j < hi && !es[j].sfx.empty() && uint8_t(es[j].sfx[0]) == b)
                  ++j;
               for (size_t k2 = i; k2 < j; ++k2)
                  es[k2].sfx.erase(0, 1);
               const Ptr child = cb_build_(es, i, j);
               if (use_full)
                  full_set(static_cast<node_full*>(deref_(rp)), b, child);
               else
               {
                  const bool ok = setlist_set(static_cast<setlist*>(deref_(rp)), b, child);
                  assert(ok && "group count pre-checked against setlist CAP");
                  (void)ok;
               }
               i = j;
            }
            out = rp;
         }
         if (!shared.empty())
            out = make_prefix(std::string_view(shared), out);
         return out;
      }
      Ptr cb_split_(Ptr cur)
      {
         cb_hdr* h = static_cast<cb_hdr*>(deref_(cur));
         std::vector<cb_ent_> es;
         es.reserve(CB_MAX + 2);
         // The term rides the rebuild as the empty-suffix entry (FIRST — it
         // sorts before every nonempty suffix): cb_build_ places it on
         // whatever node ends up on top, and an empty suffix stops the
         // shared-prefix strip, so a term bucket can never burst into a bare
         // prefix/leaf top. The old term leaf is released with the bucket.
         if (!h->rh.term.is_null())
         {
            cb_ent_ te{std::string(), std::string()};
            if constexpr (inlineable)
            {
               if (h->rh.term.tag() == K::value_inline)
               {
                  T tv;
                  unpack_inline(h->rh.term, tv);
                  const auto b = tv.to_bytes();
                  te.val.assign(b.data(), b.size());
               }
            }
            if (h->rh.term.tag() == K::value_ptr)
            {
               const char* tl = static_cast<const char*>(deref_(h->rh.term));
               std::string_view b;
               if constexpr (raw_vals)
                  b = leaf_bytes(tl);
               else
                  b = leaf_val(tl)->to_bytes();
               te.val.assign(b.data(), b.size());
            }
            es.push_back(std::move(te));
         }
         // storage is totally ordered (cb_place_), so for_each IS key order
         cb_for_each_(h, [&](std::string_view s2, std::string_view v2)
                      { es.push_back({std::string(s2), std::string(v2)}); });
         assert(std::is_sorted(es.begin(), es.end(), [](const cb_ent_& x, const cb_ent_& y)
                               { return x.sfx < y.sfx; }) &&
                "cb storage must be totally ordered");
         const std::string pfx(reinterpret_cast<const char*>(h->prefix),
                               h->rh.hdr.prefix_len);
         Ptr np  = cb_build_(es, 0, es.size());
         Ptr out = np;
         if (!pfx.empty())
            out = make_prefix(std::string_view(pfx), out);
         cow_release(cur);  // line-wise: shared pieces survive
         return out;
      }

     private:
     public:
      // ── debug refcheck: visit every line the live tree REFERENCES with the
      // count of references the tree itself holds (root extra-refs passed in).
      // The caller compares against the arena's refcount bytes.
      template <class F>
      void refcheck(F&& f, unsigned root_extra = 0) const
      {
         if (root_.is_null())
            return;
         std::map<uint32_t, uint32_t> exp;
         refwalk_(root_, exp);
         exp[line_of(root_)] += root_extra;
         for (auto& [ln, n] : exp)
            f(ln, n);
      }
      void refadd_(std::map<uint32_t, uint32_t>& exp, Ptr p, unsigned nlines = 1) const
      {
         for (unsigned k2 = 0; k2 < nlines; ++k2)
            ++exp[line_of(p) + k2];
      }
      void refwalk_(Ptr cur, std::map<uint32_t, uint32_t>& exp) const
      {
         if (cur.is_null() || cur.tag() == K::value_inline)
            return;
         const K k = cur.tag();
         refadd_(exp, cur, uint32_t((cow_node_bytes_(cur) + 127) / 128));
         switch (k)
         {
            case K::value_ptr:
            case K::bucket:
            case K::value_inline:
            case K::null:
               return;
            case K::prefix_node:
               refwalk_(pn_next<Ptr>(static_cast<char*>(deref_(cur))), exp);
               return;
            case K::setlist_u8:
            {  // inline children occupy NO separate line — they ride this one (already counted)
               const setlist* s = static_cast<const setlist*>(deref_(cur));
               refwalk_(s->rh.term, exp);
               const int nb = s->rh.hdr.nbranch;
               for (int i = 0; i < nb; ++i)
               {
                  if constexpr (inline_children_ok)
                     if (sl_is_inline(s, i)) continue;
                  refwalk_(s->branch[i], exp);
               }
               return;
            }
            case K::setlist_u16:
            case K::c2:
            case K::c4:
            case K::c8:
            case K::node_full:
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
               refwalk_(rh->term, exp);
               router_for_each(cur, [&](uint8_t, Ptr br) { refwalk_(br, exp); });
               return;
            }
            case K::full_x:
            {
               if constexpr (cow_enabled)
               {
                  const node_fullx* x = static_cast<const node_fullx*>(deref_(cur));
                  refwalk_(x->rh.term, exp);
                  for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                  {
                     if (x->lines[li].is_null())
                        continue;
                     refadd_(exp, x->lines[li]);  // the line itself
                     const cline_direct_t<Ptr>* L = fx_line_(x, li);
                     for (unsigned sl = 0; sl < full_per_line_v<Ptr>; ++sl)
                        if (!L->ptr[sl].is_null())
                           refwalk_(L->ptr[sl], exp);
                  }
               }
               return;
            }
            case K::cx:
            {
               if constexpr (cow_enabled)
               {
                  const node_cx* x = static_cast<const node_cx*>(deref_(cur));
                  refwalk_(x->rh.term, exp);
                  for (unsigned sg = 0; sg < (1u << x->lg); ++sg)
                  {
                     if (x->seglines[sg].is_null())
                        continue;
                     refadd_(exp, x->seglines[sg]);
                     cx_for_each_seg_(x, sg, [&](uint8_t, Ptr br) { refwalk_(br, exp); });
                  }
               }
               return;
            }
            case K::cbucket:
            {
               if constexpr (cb_enabled)
               {
                  const cb_hdr* h = static_cast<const cb_hdr*>(deref_(cur));
                  refwalk_(h->rh.term, exp);
                  for (unsigned li = 0; li < 4; ++li)
                     if (!h->sp_line[li].is_null())
                        refadd_(exp, h->sp_line[li]);
                  bool    big[CB_PAGES]  = {};
                  uint8_t blen[CB_PAGES] = {};
                  for (unsigned li = 0; li < 4; ++li)
                  {
                     if (!h->cnt[li] || h->sp_line[li].is_null())
                        continue;
                     const cb_sp* sp = cb_spl_(h, li);
                     for (unsigned i = 0; i < h->cnt[li]; ++i)
                        if (cb_big(sp->lref[i]))
                        {
                           big[cb_spi(sp->lref[i])]  = true;
                           blen[cb_spi(sp->lref[i])] = uint8_t(std::max(1u, cb_len8(sp->lref[i])));
                        }
                  }
                  for (unsigned spi = 0; spi < CB_PAGES; ++spi)
                     if (!h->page_line[spi].is_null())
                        refadd_(exp, h->page_line[spi], big[spi] ? blen[spi] : 2);
               }
               return;
            }
         }
      }

      // ── debug census: walk the live tree, report every node (read-only) ──
      // f(kind, inclusive_bytes, nbranch, lines_used, lines_alloc); inclusive
      // means demoted kinds / cbucket count their riding lines, so Σbytes is
      // the tree's true allocated footprint (shared lines counted once per
      // reachable path — exact for a single-root snapshot).
      struct census_row
      {
         K        kind;
         uint32_t bytes;
         uint16_t nbranch;
         uint8_t  lines_used, lines_alloc;
      };
      template <class F>
      void census(F&& f) const
      {
         census_(root_, f);
      }
      template <class F>
      void census_(packed_ptr cur, F& f) const
      {
         if (cur.is_null())
            return;
         const K k = cur.tag();
         switch (k)
         {
            case K::value_ptr:
            {
               const char*    L  = static_cast<const char*>(deref_(cur));
               const uint32_t sz = uint32_t(leaf_alloc_size(const_cast<char*>(L)));
               f(census_row{k, sz, 0, uint8_t((sz + 127) / 128), uint8_t((sz + 127) / 128)});
               return;
            }
            case K::value_inline:
               f(census_row{k, 0, 0, 0, 0});
               return;
            case K::prefix_node:
            {
               const char* P = static_cast<const char*>(deref_(cur));
               f(census_row{k, uint32_t(pn_size<Ptr>(pn_plen(P))), 1, 1, 1});
               census_(pn_next<Ptr>(const_cast<char*>(P)), f);
               return;
            }
            case K::setlist_u8:
            {  // inline children ride this line — skip (counted in this node's lines)
               const setlist* s  = static_cast<const setlist*>(deref_(cur));
               const uint8_t  la = uint8_t(node_size(K::setlist_u8) / 128);
               f(census_row{k, uint32_t(node_size(K::setlist_u8)), s->rh.hdr.nbranch, la, la});
               census_(s->rh.term, f);
               const int nb = s->rh.hdr.nbranch;
               for (int i = 0; i < nb; ++i)
               {
                  if constexpr (inline_children_ok)
                     if (sl_is_inline(s, i)) continue;
                  census_(s->branch[i], f);
               }
               return;
            }
            case K::setlist_u16:
            case K::c2:
            case K::c4:
            case K::c8:
            case K::node_full:
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
               uint8_t           lu = 1, la = uint8_t(node_size(k) / 128);
               if (k == K::node_full)
               {
                  const node_full* nf = static_cast<const node_full*>(deref_(cur));
                  for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                     lu += nf->ranges[li].nset > 0;
               }
               else
                  lu = la;
               f(census_row{k, uint32_t(node_size(k)), rh->hdr.nbranch, lu, la});
               census_(rh->term, f);
               router_for_each(cur, [&](uint8_t, packed_ptr br) { census_(br, f); });
               return;
            }
            case K::hdr:  // prefixed wide router — its own census row
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
               const K           bk = hdr_base_(deref_(cur));
               const uint8_t     la = uint8_t(node_size(bk) / 128);
               f(census_row{k, uint32_t(node_size(bk)), rh->hdr.nbranch, la, la});
               census_(rh->term, f);
               router_for_each(cur, [&](uint8_t, packed_ptr br) { census_(br, f); });
               return;
            }
            case K::full_x:
            {
               if constexpr (cow_enabled)
               {
                  const node_fullx* x  = static_cast<const node_fullx*>(deref_(cur));
                  uint8_t           lu = 1, la = 1;
                  for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                  {
                     la += !x->lines[li].is_null();
                     lu += !x->lines[li].is_null() &&
                           static_cast<const cline_direct_t<Ptr>*>(deref_(x->lines[li]))->nset > 0;
                  }
                  f(census_row{k, uint32_t(la) * 128, x->rh.hdr.nbranch, lu, la});
                  census_(x->rh.term, f);
                  router_for_each(cur, [&](uint8_t, packed_ptr br) { census_(br, f); });
               }
               return;
            }
            case K::cx:
            {
               if constexpr (cow_enabled)
               {
                  const node_cx* x  = static_cast<const node_cx*>(deref_(cur));
                  uint8_t        la = 1;
                  for (unsigned sg = 0; sg < (1u << x->lg); ++sg)
                     la += !x->seglines[sg].is_null();
                  f(census_row{k, uint32_t(la) * 128, x->rh.hdr.nbranch, la, la});
                  census_(x->rh.term, f);
                  router_for_each(cur, [&](uint8_t, packed_ptr br) { census_(br, f); });
               }
               return;
            }
            case K::bucket:
            {
               const bucket* b = static_cast<const bucket*>(deref_(cur));
               f(census_row{k, uint32_t(sizeof(bucket)), b->nent, uint8_t(sizeof(bucket) / 128),
                            uint8_t(sizeof(bucket) / 128)});
               return;
            }
            case K::cbucket:
            {
               if constexpr (cb_enabled)
               {
                  const cb_hdr* h  = static_cast<const cb_hdr*>(deref_(cur));
                  uint32_t      ln = 1;  // CL0 (fp line removed)
                  for (unsigned li = 0; li < 4; ++li)
                     ln += !h->sp_line[li].is_null();
                  bool big[CB_PAGES] = {};
                  uint8_t blen[CB_PAGES] = {};
                  for (unsigned li = 0; li < 4; ++li)
                  {
                     if (!h->cnt[li] || h->sp_line[li].is_null())
                        continue;
                     const cb_sp* sp = cb_spl_(h, li);
                     for (unsigned i = 0; i < h->cnt[li]; ++i)
                        if (cb_big(sp->lref[i]))
                        {
                           big[cb_spi(sp->lref[i])]  = true;
                           blen[cb_spi(sp->lref[i])] = uint8_t(std::max(1u, cb_len8(sp->lref[i])));
                        }
                  }
                  for (unsigned spi = 0; spi < CB_PAGES; ++spi)
                     if (!h->page_line[spi].is_null())
                        ln += big[spi] ? blen[spi] : 2;
                  f(census_row{k, ln * 128, h->rh.hdr.nbranch, uint8_t(ln), uint8_t(ln)});
                  census_(h->rh.term, f);
               }
               return;
            }
            case K::null:
               return;
         }
      }

     private:
      size_t cow_node_bytes_(Ptr p) const noexcept
      {
         char* n = static_cast<char*>(deref_(p));
         switch (p.tag())
         {
            case K::value_ptr:
               return leaf_alloc_size(n);
            case K::prefix_node:
               return pn_size<Ptr>(pn_plen(n));
            case K::setlist_u8:
            case K::setlist_u16:
            case K::c2:
            case K::c4:
            case K::c8:
            case K::node_full:
            case K::full_x:
            case K::cx:
            case K::cbucket:
               return node_size(p.tag());
            case K::hdr:
               return node_size(hdr_base_(n));
            case K::bucket:
               return sizeof(bucket);
            case K::value_inline:
            case K::null:
               return 0;
         }
         return 0;
      }

      // Cascade the CHILDREN of a dead node (vpart node-grain: a node and all
      // its lines live and die as one allocation — the caller frees the run).
      void cow_destroy_dead_(Ptr p) noexcept
      {
         const K k = p.tag();
         switch (k)
         {
            case K::value_ptr:
            case K::bucket:  // POD payloads (asserted in cow_insert): no dtors
               break;
            case K::prefix_node:
               cow_release(pn_next<Ptr>(static_cast<char*>(deref_(p))));
               break;
            case K::setlist_u8:
            {  // inline children ride this line: release their bytes, never cow_release them
               setlist*  s  = static_cast<setlist*>(deref_(p));
               cow_release(s->rh.term);
               const int nb = s->rh.hdr.nbranch;
               for (int i = 0; i < nb; ++i)
               {
                  if constexpr (inline_children_ok)
                     if (sl_is_inline(s, i)) { vh_leaf_dies_(sl_inline_leaf(s, i)); continue; }
                  cow_release(s->branch[i]);
               }
               break;
            }
            case K::setlist_u16:
            case K::c2:
            case K::c4:
            case K::c8:
            case K::node_full:
            case K::hdr:
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(p));
               cow_release(rh->term);
               router_for_each(p, [&](unsigned, Ptr c) { cow_release(c); });
               break;
            }
            case K::full_x:
            case K::cx:
               assert(false && "vpart L1a: explicit tiers are never created");
               break;
            case K::cbucket:
            {
               if constexpr (cb_enabled)
               {
                  cb_hdr* h = static_cast<cb_hdr*>(deref_(p));
                  cow_release(h->rh.term);
                  uint16_t big_seen = 0;
                  for (unsigned li = 0; li < 4; ++li)
                  {
                     if (!h->cnt[li])
                        continue;
                     const cb_sp* sp = cb_spl_(h, li);
                     for (unsigned i = 0; i < h->cnt[li]; ++i)
                        if (cb_big(sp->lref[i]))
                        {
                           const unsigned spi = cb_spi(sp->lref[i]);
                           if (!(big_seen & (1u << spi)))
                           {
                              big_seen |= uint16_t(1u << spi);
                              cb_release_lines_(h->page_line[spi],
                                                cb_len8(sp->lref[i])
                                                    ? cb_len8(sp->lref[i])
                                                    : 1);
                           }
                        }
                  }
                  for (unsigned spi = 0; spi < CB_PAGES; ++spi)
                     if (!h->page_line[spi].is_null() && !(big_seen & (1u << spi)))
                        cb_release_lines_(h->page_line[spi], 2);
                  for (unsigned li = 0; li < 4; ++li)
                     if (!h->sp_line[li].is_null())
                        cb_release_lines_(h->sp_line[li], 1);
               }
               break;
            }
            case K::value_inline:
            case K::null:
               return;
         }
         // vpart node-grain: the CALLER frees the node's whole line run
         // (head + tails) in one release; nothing left to do here.
      }

      // Release one reference to a shareable direct line of a full_x; when the
      // line dies, the <=31 children packed inside it are released first.
      void fx_release_line_(Ptr lp) noexcept
      {
         if (lp.is_null())
            return;
         alloc_.hkv_release(line_of(lp),
                            [&](uint32_t)
                            {
                               auto* L = static_cast<cline_direct_t<Ptr>*>(deref_(lp));
                               for (unsigned sl = 0; sl < full_per_line_v<Ptr>; ++sl)
                                  if (!L->ptr[sl].is_null())
                                     cow_release(L->ptr[sl]);
                            });
      }
      // Same for a cx segment line: a dying line releases the rank-dense
      // branches recorded by its presence bitmap.
      void cx_release_line_(Ptr lp, unsigned lg) noexcept
      {
         if (lp.is_null())
            return;
         alloc_.hkv_release(line_of(lp),
                            [&](uint32_t)
                            {
                               uint8_t*  m   = static_cast<uint8_t*>(deref_(lp));
                               Ptr*      br  = cx_branches<Ptr>(m, lg);
                               const int cnt = bm_count(m, int(cx_maskb(lg)));
                               for (int i = 0; i < cnt; ++i)
                                  cow_release(br[i]);
                            });
      }
      void cx_own_line_(Ptr* slot, unsigned lg)
      {
         if (slot->is_null())
            return;
         if (alloc_.hkv_retain(line_of(*slot)))
            return;
         *slot = cx_clone_line_(*slot, lg);
      }
      // fresh, empty pieces for the explicit tiers (bitmap/slots cleared; the
      // rank-dense branch tail is never read past the population)
      Ptr cx_new_seg_(unsigned lg)
      {
         assert(false && "vpart L1a: explicit tiers are never created");
         void* L = node_alloc_(cacheline_bytes);
         std::memset(L, 0, cx_maskb(lg));
         return pack_(L, K::node_full /*tag unused on seg lines*/);
      }
      Ptr fx_new_line_()
      {
         assert(false && "vpart L1a: explicit tiers are never created");
         auto* L = static_cast<cline_direct_t<Ptr>*>(node_alloc_(cacheline_bytes));
         for (unsigned sl = 0; sl < full_per_line_v<Ptr>; ++sl)
            L->ptr[sl] = packed_ptr::null();
         L->nset = 0;
         L->rsvd = 0;
         return pack_(L, K::node_full /*tag unused on direct lines*/);
      }
      Ptr cx_clone_line_(Ptr lp, unsigned lg)
      {
         void* dst = node_alloc_(cacheline_bytes);
         std::memcpy(dst, deref_(lp), cacheline_bytes);
         uint8_t*  m   = static_cast<uint8_t*>(dst);
         Ptr*      br  = cx_branches<Ptr>(m, lg);
         const int cnt = bm_count(m, int(cx_maskb(lg)));
         for (int i = 0; i < cnt; ++i)
            cow_fix_child_(&br[i]);
         return pack_(dst, K::node_full /*tag unused on seg lines*/);
      }
      // ── widen builders for the explicit ladder ──────────────────────────────
      // Both MOVE branches out of `src` (widen() raw-frees its CL0 after) with
      // line-granular care: an EXCLUSIVE source seg line transfers its branch
      // references as-is and is raw-released; a SHARED one is copied with
      // cow_fix_child_ per branch (retain / clone-forward) and released — the
      // sharer's eventual line destroy pairs with our retains. Throw safety:
      // target lines are allocated up front, shared segs are copied BEFORE any
      // exclusive seg is consumed, and the unwind releases exactly the fixed
      // prefix — the source is untouched until the nothrow move phase.
      Ptr cx_from_router_(Ptr src, unsigned lg)
      {
         node_cx*    nx  = static_cast<node_cx*>(node_alloc_(cacheline_bytes));
         router_hdr* srh = static_cast<router_hdr*>(deref_(src));
         nx->rh          = *srh;  // term MOVES with the rebuild
         nx->lg          = uint8_t(lg);
         nx->pad0        = 0;
         for (auto& sl2 : nx->seglines)
            sl2 = packed_ptr::null();
         const unsigned segw = cx_segw(lg);
         // nothrow placement into a preallocated target seg (ascending feed)
         auto place = [&](uint8_t b, packed_ptr br) noexcept -> Ptr*
         {
            uint8_t*  m = static_cast<uint8_t*>(deref_(nx->seglines[b >> (8 - lg)]));
            const int d = int(b & (segw - 1));
            Ptr*      p = &cx_branches<Ptr>(m, lg)[bm_rank(m, d)];
            *p = br;
            bm_set(m, d);
            return p;
         };
         unsigned fixed = 0;  // shared-copy progress, for unwind
         try
         {
            if (src.tag() != K::cx)
            {  // implicit source (setlist / cnode): whole-node private — pure move
               router_for_each(src, [&](uint8_t b, packed_ptr) {  // pass 1: targets
                  Ptr& t = nx->seglines[b >> (8 - lg)];
                  if (t.is_null())
                     t = cx_new_seg_(lg);
               });
               router_for_each(src, [&](uint8_t b, packed_ptr br) { place(b, br); });
               return pack_(nx, K::cx);
            }
            node_cx*       sx    = static_cast<node_cx*>(deref_(src));
            const unsigned snseg = 1u << sx->lg;
            bool           shared[8] = {};
            for (unsigned sg = 0; sg < snseg; ++sg)  // pass 1: classify + targets
            {
               if (sx->seglines[sg].is_null())
                  continue;
               shared[sg] = alloc_.hkv_ref(line_of(sx->seglines[sg])) >= 2;
               cx_for_each_seg_(sx, sg, [&](uint8_t b, packed_ptr) {
                  Ptr& t = nx->seglines[b >> (8 - lg)];
                  if (t.is_null())
                     t = cx_new_seg_(lg);
               });
            }
            for (unsigned sg = 0; sg < snseg; ++sg)  // pass 2: copy SHARED (may throw)
            {
               if (sx->seglines[sg].is_null() || !shared[sg])
                  continue;
               cx_for_each_seg_(sx, sg, [&](uint8_t b, packed_ptr br) {
                  Ptr* p = place(b, br);
                  cow_fix_child_(p);  // retain / clone-forward
                  ++fixed;
               });
            }
            for (unsigned sg = 0; sg < snseg; ++sg)  // pass 3: nothrow
            {
               if (sx->seglines[sg].is_null())
                  continue;
               if (shared[sg])
                  cx_release_line_(sx->seglines[sg], sx->lg);  // sharer keeps the children
               else
               {
                  cx_for_each_seg_(sx, sg, [&](uint8_t b, packed_ptr br) { place(b, br); });
                  alloc_.hkv_release(line_of(sx->seglines[sg]), [](uint32_t) {});  // refs moved
               }
               sx->seglines[sg] = packed_ptr::null();
            }
            return pack_(nx, K::cx);
         }
         catch (...)
         {
            cx_unwind_built_(nx, lg, fixed);
            throw;
         }
      }
      Ptr fx_from_cx_(Ptr src)
      {
         node_cx*    sx = static_cast<node_cx*>(deref_(src));
         node_fullx* nx = static_cast<node_fullx*>(node_alloc_(cacheline_bytes));
         nx->rh         = sx->rh;  // term moves
         for (auto& l2 : nx->lines)
            l2 = packed_ptr::null();
         auto place = [&](uint8_t b, packed_ptr br) noexcept -> Ptr*
         {
            unsigned li, sl;
            full_index<Ptr>(b, li, sl);
            cline_direct_t<Ptr>* L = fx_line_(nx, li);
            L->ptr[sl]             = br;
            ++L->nset;
            return &L->ptr[sl];
         };
         const unsigned snseg = 1u << sx->lg;
         bool           shared[8] = {};
         unsigned       fixed     = 0;
         try
         {
            for (unsigned sg = 0; sg < snseg; ++sg)  // pass 1
            {
               if (sx->seglines[sg].is_null())
                  continue;
               shared[sg] = alloc_.hkv_ref(line_of(sx->seglines[sg])) >= 2;
               cx_for_each_seg_(sx, sg, [&](uint8_t b, packed_ptr) {
                  unsigned li, sl;
                  full_index<Ptr>(b, li, sl);
                  if (nx->lines[li].is_null())
                     nx->lines[li] = fx_new_line_();
               });
            }
            for (unsigned sg = 0; sg < snseg; ++sg)  // pass 2: shared copies
            {
               if (sx->seglines[sg].is_null() || !shared[sg])
                  continue;
               cx_for_each_seg_(sx, sg, [&](uint8_t b, packed_ptr br) {
                  Ptr* p = place(b, br);
                  cow_fix_child_(p);
                  ++fixed;
               });
            }
            for (unsigned sg = 0; sg < snseg; ++sg)  // pass 3: nothrow moves
            {
               if (sx->seglines[sg].is_null())
                  continue;
               if (shared[sg])
                  cx_release_line_(sx->seglines[sg], sx->lg);
               else
               {
                  cx_for_each_seg_(sx, sg, [&](uint8_t b, packed_ptr br) { place(b, br); });
                  alloc_.hkv_release(line_of(sx->seglines[sg]), [](uint32_t) {});
               }
               sx->seglines[sg] = packed_ptr::null();
            }
            return pack_(nx, K::full_x);
         }
         catch (...)
         {
            fx_unwind_built_(nx, fixed);
            throw;
         }
      }
      // cx -> MONOLITHIC target (cnode<X> or node_full): the dense-occupancy
      // widen. Same line-granular move/copy discipline as the explicit builders.
      template <class N>
      Ptr implicit_from_cx_(Ptr src, K newtag)
      {
         node_cx* sx = static_cast<node_cx*>(deref_(src));
         N*       n  = ::new (node_alloc_(sizeof(N))) N;
         if constexpr (std::is_same_v<N, node_full>)
            full_init(n);
         else
            cnode_init(n);
         n->rh.hdr.has_term = sx->rh.hdr.has_term;
         n->rh.term         = sx->rh.term;  // moves
         auto place = [&](uint8_t b, packed_ptr br) noexcept -> Ptr*
         {
            if constexpr (std::is_same_v<N, node_full>)
            {
               full_set(n, b, br);
               return reinterpret_cast<packed_ptr*>(reinterpret_cast<char*>(n) +
                                                    full_offtab_v<Ptr>[b]);
            }
            else
            {
               const bool ok = cnode_set(n, b, br);
               assert(ok && "occupancy-checked target always has room");
               (void)ok;
               return cnode_child_slot(n, b);
            }
         };
         const unsigned snseg     = 1u << sx->lg;
         bool           shared[8] = {};
         unsigned       fixed     = 0;
         try
         {
            for (unsigned sg = 0; sg < snseg; ++sg)
               if (!sx->seglines[sg].is_null())
                  shared[sg] = alloc_.hkv_ref(line_of(sx->seglines[sg])) >= 2;
            for (unsigned sg = 0; sg < snseg; ++sg)  // shared copies first (may throw)
            {
               if (sx->seglines[sg].is_null() || !shared[sg])
                  continue;
               cx_for_each_seg_(sx, sg, [&](uint8_t b, packed_ptr br) {
                  Ptr* p = place(b, br);
                  cow_fix_child_(p);
                  ++fixed;
               });
            }
            for (unsigned sg = 0; sg < snseg; ++sg)  // nothrow moves + releases
            {
               if (sx->seglines[sg].is_null())
                  continue;
               if (shared[sg])
                  cx_release_line_(sx->seglines[sg], sx->lg);
               else
               {
                  cx_for_each_seg_(sx, sg, [&](uint8_t b, packed_ptr br) { place(b, br); });
                  alloc_.hkv_release(line_of(sx->seglines[sg]), [](uint32_t) {});
               }
               sx->seglines[sg] = packed_ptr::null();
            }
            return pack_(n, newtag);
         }
         catch (...)
         {
            // release the fixed prefix (ascending bytes = placement order)
            router_for_each(pack_(n, newtag), [&](uint8_t, packed_ptr br) {
               if (fixed) { cow_release(br); --fixed; }
            });
            node_free_(n, sizeof(N));
            throw;
         }
      }
      // implicit source (c8: whole-node private) -> sparse full_x with null lines
      Ptr fx_from_implicit_(Ptr src)
      {
         router_hdr* srh = static_cast<router_hdr*>(deref_(src));
         node_fullx* nx  = static_cast<node_fullx*>(node_alloc_(cacheline_bytes));
         nx->rh          = *srh;
         for (auto& l2 : nx->lines)
            l2 = packed_ptr::null();
         router_for_each(src, [&](uint8_t b, packed_ptr) {  // pass 1: lines
            unsigned li, sl;
            full_index<Ptr>(b, li, sl);
            if (nx->lines[li].is_null())
               nx->lines[li] = fx_new_line_();
         });
         router_for_each(src, [&](uint8_t b, packed_ptr br) {  // pure move
            unsigned li, sl;
            full_index<Ptr>(b, li, sl);
            cline_direct_t<Ptr>* L = fx_line_(nx, li);
            L->ptr[sl]             = br;
            ++L->nset;
         });
         return pack_(nx, K::full_x);
      }
      // iterate one seg line of a cx: (absolute byte, branch) ascending
      template <class F>
      void cx_for_each_seg_(const node_cx* x, unsigned sg, F&& f) const
      {
         const unsigned segw = cx_segw(x->lg), maskb = cx_maskb(x->lg);
         uint8_t*       m    = static_cast<uint8_t*>(deref_(x->seglines[sg]));
         int            idx  = 0;
         for (unsigned wo = 0; wo < maskb; wo += 8)
         {
            uint64_t       w  = 0;
            const unsigned wb = maskb - wo < 8 ? maskb - wo : 8;
            std::memcpy(&w, m + wo, size_t(wb));
            while (w)
            {
               const int b = std::countr_zero(w);
               w &= w - 1;
               f(uint8_t(sg * segw + wo * 8 + unsigned(b)), cx_branches<Ptr>(m, x->lg)[idx++]);
            }
         }
      }
      // unwind a partially-built widen target: release the first `fixed`
      // shared-copied branches (in feed order), then raw-free its lines + CL0.
      void cx_unwind_built_(node_cx* nx, unsigned lg, unsigned fixed) noexcept
      {
         for (unsigned sg = 0; sg < (1u << lg) && fixed; ++sg)
         {
            if (nx->seglines[sg].is_null())
               continue;
            cx_for_each_seg_(nx, sg, [&](uint8_t, packed_ptr br) {
               if (fixed) { cow_release(br); --fixed; }
            });
         }
         for (unsigned sg = 0; sg < (1u << lg); ++sg)
            if (!nx->seglines[sg].is_null())
               alloc_.hkv_release(line_of(nx->seglines[sg]), [](uint32_t) {});
         node_free_(nx, cacheline_bytes);
      }
      void fx_unwind_built_(node_fullx* nx, unsigned fixed) noexcept
      {
         for (unsigned li = 0; li < full_lines_v<Ptr> && fixed; ++li)
         {
            if (nx->lines[li].is_null())
               continue;
            cline_direct_t<Ptr>* L = fx_line_(nx, li);
            for (unsigned sl = 0; sl < full_per_line_v<Ptr> && fixed; ++sl)
               if (!L->ptr[sl].is_null()) { cow_release(L->ptr[sl]); --fixed; }
         }
         for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
            if (!nx->lines[li].is_null())
               alloc_.hkv_release(line_of(nx->lines[li]), [](uint32_t) {});
         node_free_(nx, cacheline_bytes);
      }

      // Retain a direct line for a new owner; at the writer band, clone the
      // 128B line forward instead (its <=31 children get retained).
      void fx_own_line_(Ptr* slot)
      {
         if (slot->is_null())
            return;
         if (alloc_.hkv_retain(line_of(*slot)))
            return;
         *slot = fx_clone_line_(*slot);
      }
      Ptr fx_clone_line_(Ptr lp)
      {
         void* dst = node_alloc_(sizeof(cline_direct_t<Ptr>));
         std::memcpy(dst, deref_(lp), sizeof(cline_direct_t<Ptr>));
         auto* L = static_cast<cline_direct_t<Ptr>*>(dst);
         for (unsigned sl = 0; sl < full_per_line_v<Ptr>; ++sl)
            if (!L->ptr[sl].is_null())
               cow_fix_child_(&L->ptr[sl]);
         return pack_(dst, K::node_full /*tag unused on direct lines*/);
      }

      // vpart: child slots of a memcpy'd clone need NO re-owning — sharing is
      // free (no counts), and birth stamps ride inside the children untouched.
      void cow_fix_child_(Ptr*) noexcept {}

      Ptr cow_clone_node_(Ptr cur)
      {
         const K k = cur.tag();
         // vpart L1a: NO implicit->explicit demotion. Sharing is free (no
         // refcount traffic to dodge), so a clone of any multi-line implicit
         // router is a straight memcpy of the whole run — O(1) bookkeeping,
         // children untouched. The explicit tiers (full_x/cx) are never
         // created; per-line sub-node COW returns as a measured follow-up.
         if constexpr (cb_enabled)
            if (k == K::cbucket)
            {
               // page = one COW unit AND the clone is the compactor: rebuild
               // the page through the bulk builder — dead sub-page bytes
               // (removed cells, outgrown values) vanish at every generation
               // instead of riding the verbatim copy forever.
               // verbatim piece copy ALWAYS (Dan): reclamation happens on
               // DEMAND — when an insert lacks space and the dead ledger says
               // a reformat would fit it (cb_canonicalize_ in pg_upsert_)
               cb_hdr* h  = static_cast<cb_hdr*>(deref_(cur));
               cb_hdr* d2 = static_cast<cb_hdr*>(node_alloc_(128));
               std::memcpy(d2, h, 128);
               cow_fix_child_(&d2->rh.term);
               for (unsigned li = 0; li < 4; ++li)
                  if (!d2->sp_line[li].is_null())
                     d2->sp_line[li] = cb_copy_lines_(d2->sp_line[li], 1);
               uint8_t nl_of[CB_PAGES];
               for (auto& x2 : nl_of)
                  x2 = 2;
               for (unsigned li = 0; li < 4; ++li)
               {
                  if (!d2->cnt[li])
                     continue;
                  const cb_sp* sp = cb_spl_(d2, li);
                  for (unsigned i = 0; i < d2->cnt[li]; ++i)
                     if (cb_big(sp->lref[i]))
                        nl_of[cb_spi(sp->lref[i])] =
                            uint8_t(cb_len8(sp->lref[i]) ? cb_len8(sp->lref[i]) : 1);
               }
               for (unsigned spi = 0; spi < CB_PAGES; ++spi)
                  if (!d2->page_line[spi].is_null())
                     d2->page_line[spi] = cb_copy_lines_(d2->page_line[spi], nl_of[spi]);
               return pack_(d2, K::cbucket);
            }
         const size_t bytes = cow_node_bytes_(cur);
         void*        dst   = node_alloc_(bytes);
         std::memcpy(dst, deref_(cur), bytes);
         const Ptr np = pack_(dst, k);
         switch (k)
         {
            case K::value_ptr:
               // duplicated value bytes; the shallow husk's drain balances this
               if constexpr (vh_on)
                  vh_copy_(leaf_bytes(static_cast<const char*>(dst)));
               break;
            case K::bucket:
               break;
            case K::prefix_node:
               cow_fix_child_(reinterpret_cast<Ptr*>(static_cast<char*>(dst) + 2));
               break;
            case K::setlist_u8:
            {
               setlist* s = static_cast<setlist*>(dst);
               cow_fix_child_(&s->rh.term);
               for (int i = 0; i < s->rh.hdr.nbranch; ++i)
                  cow_fix_child_(&s->branch[i]);
               // inline children rode the memcpy (offsets are line-relative, no re-point);
               // their bytes are now duplicated — fire the balanced copy hook.
               sl_inline_vh_copy_(s);
               break;
            }
            case K::setlist_u16:
            {
               setlist16* s = static_cast<setlist16*>(dst);
               cow_fix_child_(&s->rh.term);
               for (int i = 0; i < s->rh.hdr.nbranch; ++i)
                  cow_fix_child_(&s->branch[i]);
               break;
            }
            case K::c2:
            {
               auto* cn = static_cast<cnode<2>*>(dst);
               cow_fix_child_(&cn->rh.term);
               for (int sg = 0; sg < 2; ++sg)
                  for (int i = 0, n = bm_count(cn->segs[sg].present, cnode<2>::MASKB); i < n; ++i)
                     cow_fix_child_(&cn->segs[sg].branch[i]);
               break;
            }
            case K::c4:
            {
               auto* cn = static_cast<cnode<4>*>(dst);
               cow_fix_child_(&cn->rh.term);
               for (int sg = 0; sg < 4; ++sg)
                  for (int i = 0, n = bm_count(cn->segs[sg].present, cnode<4>::MASKB); i < n; ++i)
                     cow_fix_child_(&cn->segs[sg].branch[i]);
               break;
            }
            case K::c8:
            {
               auto* cn = static_cast<cnode<8>*>(dst);
               cow_fix_child_(&cn->rh.term);
               for (int sg = 0; sg < 8; ++sg)
                  for (int i = 0, n = bm_count(cn->segs[sg].present, cnode<8>::MASKB); i < n; ++i)
                     cow_fix_child_(&cn->segs[sg].branch[i]);
               break;
            }
            case K::node_full:
            case K::hdr:  // CL0 (base, prefix, stamp slot) rides the memcpy
               break;  // vpart: whole-run memcpy clone; children carried as-is
            case K::full_x:
            {
               node_fullx* x = static_cast<node_fullx*>(dst);
               cow_fix_child_(&x->rh.term);
               for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                  fx_own_line_(&x->lines[li]);
               break;
            }
            case K::cx:
            {
               node_cx* x = static_cast<node_cx*>(dst);
               cow_fix_child_(&x->rh.term);
               for (unsigned sg = 0; sg < (1u << x->lg); ++sg)
                  cx_own_line_(&x->seglines[sg], x->lg);
               break;
            }
            case K::cbucket:
               assert(false && "pages clone via the compacting bulk builder above");
               break;
                        case K::value_inline:
            case K::null:
               break;
         }
         return np;
      }

      // Clone the recorded path's shared suffix starting at `from` (the
      // template<Shared> transition index — no scan: children of shared
      // nodes are shared). Idempotent: post-mutation re-entry (restride
      // continue paths) finds private entries and skips them. Each next
      // slot is re-based by OFFSET into the fresh clone (memcpy ⇒ stable).
      packed_ptr* own_suffix_(ins_path& path, uint32_t n, uint32_t from)
      {
         for (uint32_t i = from; i < n; ++i)
         {
            const packed_ptr old = *path[i];
            if (old.is_null() || old.tag() == K::value_inline)
               break;
            if (!cow_shared_(old))
               continue;  // already owned (idempotent re-entry)
            char*            ob = static_cast<char*>(deref_(old));
            const packed_ptr nn = cow_clone_into_(path[i], old);
            if (i + 1 < n)
               path[i + 1] = reinterpret_cast<packed_ptr*>(
                   static_cast<char*>(deref_(nn)) +
                   (reinterpret_cast<char*>(path[i + 1]) - ob));
         }
         return path[n - 1];
      }

      Ptr cow_clone_into_(Ptr* slot, Ptr cur)
      {
         const Ptr np = cow_clone_node_(cur);
         *slot        = np;
         cow_release_shallow_(cur);  // children carried by np; log the husk only
         return np;
      }

      bool cow_shared_(Ptr p) const noexcept
      {
         // vpart privacy test: anything not minted by the CURRENT write window
         // is shared (visible to a committed boundary) and must be COW'd.
         // Stampless kinds (bucket) read birth 0 -> always shared (safe).
         return alloc_.hkv_vold(vstamp_get_(p));
      }

      // Privatize the terminal leaf hanging off a router header (in-place value
      // overwrite mutates it).
      void cow_privatize_term_(router_hdr* rh)
      {
         const Ptr t = rh->term;
         if (!t.is_null() && t.tag() == K::value_ptr && cow_shared_(t))
            cow_clone_into_(&rh->term, t);
      }

      // Walk the key's path, cloning every shared node top-down. Stops where the
      // descent stops: everything an in-place insert/remove can touch — path
      // routers, the diverging prefix node, the terminal leaf/bucket, the router
      // whose term gets overwritten — is exclusively owned afterwards.
      void cow_privatize_(std::string_view key)
      {
         if (root_.is_null())
            return;
         Ptr*   slot  = &root_;
         size_t depth = 0;
         for (;;)
         {
            Ptr cur = *slot;
            if (cur.is_null() || cur.tag() == K::value_inline)
               return;
            if (cow_shared_(cur))
               cur = cow_clone_into_(slot, cur);
            const K k = cur.tag();
            switch (k)
            {
               case K::value_ptr:
               case K::bucket:
                  return;  // terminal reached and private
               case K::cbucket:
                  if constexpr (cb_enabled)
                  {  // CL0 is private; ops own their lines; term may need a clone
                     cb_hdr*        h  = static_cast<cb_hdr*>(deref_(cur));
                     const unsigned pl = h->rh.hdr.prefix_len;
                     size_t         d2 = depth;
                     if (pl)
                     {
                        const size_t c =
                            lcp(std::string_view(reinterpret_cast<char*>(h->prefix), pl),
                                key.substr(d2));
                        if (c < pl)
                           return;
                        d2 += pl;
                     }
                     if (d2 == key.size())
                        cow_privatize_term_(&h->rh);
                  }
                  return;
               case K::prefix_node:
               {
                  char*            P = static_cast<char*>(deref_(cur));
                  std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
                  const size_t     c = lcp(PP, key.substr(depth));
                  if (c < PP.size())
                     return;  // insert splits THIS node (already private)
                  slot = reinterpret_cast<Ptr*>(P + 2);
                  depth += PP.size();
                  break;
               }
               case K::hdr:  // prefixed wide router: consume CL0 prefix, route
               {
                  char*          hp  = static_cast<char*>(deref_(cur));
                  router_hdr*    hrh = reinterpret_cast<router_hdr*>(hp);
                  const unsigned hpl = hrh->hdr.prefix_len;
                  const size_t   hc =
                      lcp(std::string_view(hp + WPFX_OFF, hpl), key.substr(depth));
                  if (hc < hpl)
                     return;  // prefix split happens in this (private) node
                  depth += hpl;
                  if (depth == key.size())
                  {
                     cow_privatize_term_(hrh);
                     return;
                  }
                  Ptr* hcs = router_find_slot(cur, uint8_t(key[depth]));
                  if (!hcs)
                     return;  // insert adds a branch here
                  slot = hcs;
                  ++depth;
                  break;
               }
               case K::setlist_u8:
               {
                  setlist*       s  = static_cast<setlist*>(deref_(cur));
                  const unsigned pl = s->rh.hdr.prefix_len;
                  if (pl)
                  {
                     const size_t c =
                         lcp(std::string_view(reinterpret_cast<char*>(s->prefix), pl),
                             key.substr(depth));
                     if (c < pl)
                        return;  // prefix split happens in this (private) node
                     depth += pl;
                  }
                  if (depth == key.size())
                  {
                     cow_privatize_term_(&s->rh);
                     return;
                  }
                  const int i = setlist_index(s, uint8_t(key[depth]));
                  if (i < 0)
                     return;  // insert adds a branch here
                  slot = &s->branch[i];
                  ++depth;
                  break;
               }
               case K::setlist_u16:
               {
                  setlist16* s = static_cast<setlist16*>(deref_(cur));
                  if (key.size() - depth < 2)
                  {
                     if (depth == key.size())
                        cow_privatize_term_(&s->rh);
                     return;  // term update or restride — both on this private node
                  }
                  Ptr* cs = sl16_find_slot(s, two_stem(key, depth));
                  if (!cs)
                     return;
                  slot = cs;
                  depth += 2;
                  break;
               }
               case K::node_full:
               {
                  char* f = static_cast<char*>(deref_(cur));
                  if (depth == key.size())
                  {
                     cow_privatize_term_(reinterpret_cast<router_hdr*>(f));
                     return;
                  }
                  Ptr* cs =
                      reinterpret_cast<Ptr*>(f + full_offtab_v<Ptr>[uint8_t(key[depth])]);
                  if (cs->is_null())
                     return;
                  slot = cs;
                  ++depth;
                  break;
               }
               case K::c2:
               case K::c4:
               case K::c8:
               {
                  if (depth == key.size())
                  {
                     cow_privatize_term_(static_cast<router_hdr*>(deref_(cur)));
                     return;
                  }
                  Ptr* cs = router_find_slot(cur, uint8_t(key[depth]));
                  if (!cs)
                     return;
                  slot = cs;
                  ++depth;
                  break;
               }
               case K::full_x:
               {
                  node_fullx* x = static_cast<node_fullx*>(deref_(cur));
                  if (depth == key.size())
                  {
                     cow_privatize_term_(&x->rh);
                     return;
                  }
                  unsigned li, sl;
                  full_index<Ptr>(uint8_t(key[depth]), li, sl);
                  if (x->lines[li].is_null())
                     return;  // absent range: insert materializes it in the private node
                  // The descent (or its writeback) mutates this direct line:
                  // it must be exclusively owned too (per-line COW).
                  if (alloc_.hkv_ref(line_of(x->lines[li])) >= 2)
                  {
                     const Ptr old = x->lines[li];
                     x->lines[li]  = fx_clone_line_(old);
                     fx_release_line_(old);  // the CL0 slot's former reference
                  }
                  packed_ptr* cs = &fx_line_(x, li)->ptr[sl];
                  if (cs->is_null())
                     return;  // insert adds in this (now private) line
                  slot = cs;
                  ++depth;
                  break;
               }
               case K::cx:
               {
                  node_cx* x = static_cast<node_cx*>(deref_(cur));
                  if (depth == key.size())
                  {
                     cow_privatize_term_(&x->rh);
                     return;
                  }
                  const uint8_t  byte = uint8_t(key[depth]);
                  const unsigned sg   = byte >> (8 - x->lg);
                  if (x->seglines[sg].is_null())
                     return;  // absent range: insert materializes it in the private node
                  if (alloc_.hkv_ref(line_of(x->seglines[sg])) >= 2)
                  {
                     const Ptr old   = x->seglines[sg];
                     x->seglines[sg] = cx_clone_line_(old, x->lg);
                     cx_release_line_(old, x->lg);
                  }
                  packed_ptr* cs = router_find_slot(x, byte);
                  if (!cs)
                     return;  // insert adds in this (now private) line
                  slot = cs;
                  ++depth;
                  break;
               }
               case K::value_inline:
               case K::null:
                  return;
            }
         }
      }

     private:
      [[no_unique_address]] Allocator alloc_{};  // element (T) allocator; rebound per-node for bytes
      [[no_unique_address]] std::conditional_t<indexed, const std::byte*, no_base_t> base_{};
      packed_ptr                      root_  = packed_ptr::null();
      size_t                          count_ = 0;
   };
}  // namespace detail

using detail::bytes_value;
using detail::kv_tree;  // re-export only the public class; impl stays in start_ver::hkv::detail
}  // namespace start_ver::hkv

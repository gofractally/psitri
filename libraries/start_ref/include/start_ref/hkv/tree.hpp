// kv_tree<T> — the headerless adaptive radix map of design4.md.
//
// Tiers:
//   leaf<T>      value_ptr / value_large — suffix (rest of key) + value, one alloc
//   prefix_node  prefix_node — long shared prefix + one next pointer (path compression)
//   setlist      setlist_u8  — sparse byte router, 1 cacheline, up to 16 branches
//   cnode<N>     c2 / c4 / c8 — segmented routers between the setlist and the full node
//   node_full    node_full   — dense direct-index router, full_lines_v lines, skip-CL0
//   node_fullx / node_cx     — the explicit forms a shared full / segmented router is
//                              cloned into, which share the original's lines
//
// Routing is off the pointer tag: setlist/prefix read the node; node_full
// computes the segment line from the key byte, PREFETCHES it, then reads the slot
// (skip-CL0). Growth climbs setlist -> c2 -> c4 -> c8 -> node_full. With an
// allocator that has reference-count hooks the tree is copy-on-write: a node
// shared with another version is cloned before it is changed (see "part COW
// extension" below); with a plain allocator mutation is in place.
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

#include <start_ref/config.hpp>  // large_grain_bytes / line_bytes / max_alloc_lines
#include <start_ref/hkv/cnode.hpp>
#include <start_ref/hkv/key_codec.hpp>
#include <start_ref/hkv/tagged_ptr.hpp>
#include <start_ref/hkv/value_ops.hpp>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace start_ref::hkv
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
   // removes essentially all prefix_node pointer-chases. bytes[]/branch[] keep their
   // offsets (the SIMD scan is unchanged); the prefix lives in the old pad bytes.
   template <class Ptr>
   struct setlist_t
   {
      static constexpr int PREFIX_CAP = 8;
      static constexpr int LANES      = 16;  // stem lanes the SIMD scans read; [CAP..LANES) stay 0xFF
      // one cacheline: header + LANES stems + CAP branches + PREFIX_CAP. Derived from sizeof(Ptr).
      static constexpr int CAP =
          std::min(LANES, (int(cacheline_bytes) - int(sizeof(router_hdr_t<Ptr>)) - PREFIX_CAP - LANES) /
                              int(sizeof(Ptr)));  // 16 at 6B
      router_hdr_t<Ptr>    rh;
      uint8_t              bytes[LANES];
      Ptr                  branch[CAP];
      uint8_t              prefix[PREFIX_CAP];  // rh.hdr.prefix_len bytes used
   };
   using setlist = setlist_t<packed_ptr>;
   static_assert(sizeof(setlist) == 128, "setlist is one cacheline");

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

   // ── prefix_node: [u16 plen][Ptr next][u8 prefix[plen]] — layout derives from sizeof(Ptr) ──
   inline uint16_t pn_plen(const char* p) noexcept { uint16_t v; std::memcpy(&v, p, 2); return v; }
   template <class Ptr>
   inline Ptr pn_next(const char* p) noexcept { Ptr r; std::memcpy(r.b, p + 2, sizeof(Ptr)); return r; }
   template <class Ptr>
   inline void pn_set_next(char* p, Ptr n) noexcept { std::memcpy(p + 2, n.b, sizeof(Ptr)); }
   template <class Ptr>
   inline const uint8_t* pn_pfx(const char* p) noexcept
   {
      return reinterpret_cast<const uint8_t*>(p + 2 + sizeof(Ptr));
   }
   template <class Ptr>
   inline uint8_t* pn_pfx(char* p) noexcept { return reinterpret_cast<uint8_t*>(p + 2 + sizeof(Ptr)); }
   template <class Ptr>
   inline size_t pn_size(size_t plen) noexcept { return 2 + sizeof(Ptr) + plen; }

   // ── Stage B: the branch-handle type comes from the Allocator ─────────────────
   // An allocator opts in with a member type `using hkv_handle = packed_ptr_t<N>;`
   // (8 = full pointer, drops the 48-bit VA assumption; 4 = pool index). Default: the
   // 6-byte packed_ptr, so std::allocator trees keep today's exact layout and codegen.
   template <class A, class = void>
   struct handle_of { using type = packed_ptr; };
   template <class A>
   struct handle_of<A, std::void_t<typename A::hkv_handle>> { using type = typename A::hkv_handle; };

   // Structural tuning is compile-time policy (zero runtime branch cost; dead modes are
   // discarded). Defaults = plain adaptive-radix.
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
      // value_large gate, defined up front so the leaf helpers below can use it.
      // The allocator exposes a second 32 KiB-grain arena via has_large; heap/PMR
      // allocators lack it and large_ok_ is false, making every leaf value_ptr
      // (the large path is dead code under `if constexpr`).
      template <class A, class = void>
      struct large_cap0_ : std::false_type
      {
      };
      template <class A>
      struct large_cap0_<A, std::void_t<decltype(A::has_large)>> : std::true_type
      {
      };
      static constexpr bool large_ok_ = large_cap0_<Allocator>::value && Ptr::indexed;

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

      // ── large-value leaves: a second 32 KiB-grain arena (value_large) ────────
      // A leaf whose byte size exceeds the line arena's single-CAS cap lives in
      // the allocator's large arena instead. Allocation/free route there; the
      // leaf bytes and accessors are value_ptr's verbatim, so once a value_large
      // handle is deref'd, EVERY leaf_*() reads it identically. Liveness is the
      // SAME per-line refcount the line arena uses (no version drain): a
      // value_large leaf is freed when its refcount reaches 0 via the COW
      // cascade through node_free_p_.
      // Does a leaf of this many bytes need the large arena? (true => value_large)
      bool leaf_needs_large_(size_t bytes) const noexcept
      {
         if constexpr (large_ok_)
            return bytes > Allocator::large_threshold_bytes;
         else
            return false;
      }
      // Allocate a leaf of `bytes`; sets out_tag to value_ptr or value_large.
      char* leaf_alloc_(size_t bytes, K& out_tag)
      {
         if constexpr (large_ok_)
            if (bytes > Allocator::large_threshold_bytes)
            {
               out_tag           = K::value_large;
               const size_t goff = alloc_.allocate_large(bytes);  // throws past 2 MiB
               return static_cast<char*>(
                   deref_large_(uint64_t(goff / start_ref::large_grain_bytes) << 4));
            }
         out_tag = K::value_ptr;
         return static_cast<char*>(node_alloc_(bytes));
      }
      // Free a leaf, routing on its tag (large vs line arena). p is the deref'd
      // leaf base; bytes is its alloc size.
      void leaf_free_(void* p, size_t bytes, K tag) noexcept
      {
         if constexpr (large_ok_)
            if (tag == K::value_large)
            {
               const size_t goff =
                   size_t(static_cast<const std::byte*>(p) - large_base_);
               alloc_.deallocate_large(goff, bytes);
               return;
            }
         node_free_(p, bytes);
      }
      // Free a node BY HANDLE: routes a value_large leaf to the large arena,
      // everything else to the line arena. The generic node-death sites (COW
      // release cascade, free_all_) funnel through this so they never need to
      // special-case the tag themselves. THIS is the refcount-cascade free site.
      void node_free_p_(Ptr p, size_t bytes) noexcept
      {
         if constexpr (large_ok_)
            if (p.tag() == K::value_large)
            {
               leaf_free_(deref_(p), bytes, K::value_large);
               return;
            }
         node_free_(deref_(p), bytes);
      }

      // ── typed-value lifecycle hooks (raw_vals, allocator-carried) ─────────
      // When the allocator exposes vhooks (the kv database's typed-value
      // registry), the tree reports every PHYSICAL death of value bytes
      // (on_release) and every engine-internal byte duplication (on_copy).
      // A privatize clone and the old generation's later refcount-zero death
      // fire a balanced copy/release pair; a same-scope move (split rebuild)
      // fires neither. The only NET calls are user-visible births and deaths,
      // so a registry's refcounts stay exact across overwrite, remove, abort,
      // retirement and background-releaser cascades.
      static constexpr bool vh_on =
          raw_vals && requires(Allocator& a) { a.vhooks->on_release; };
      void vh_release_(std::string_view b) noexcept
      {
         if constexpr (vh_on)
            if (auto* h = alloc_.vhooks; h && h->on_release)
               h->on_release(b.data(), b.size(), h->user);
      }
      void vh_copy_(std::string_view b)
      {
         if constexpr (vh_on)
            if (auto* h = alloc_.vhooks; h && h->on_copy)
               h->on_copy(const_cast<char*>(b.data()), b.size(), h->user, alloc_.lses);
      }
      void vh_leaf_dies_(const char* L) noexcept
      {
         if constexpr (vh_on)
            vh_release_(leaf_bytes(L));
      }
      void vh_node_dies_(Ptr p) noexcept  // funnel for tag-dispatched frees
      {
         if constexpr (vh_on)
            if (is_leaf(p.tag()))  // value_ptr or value_large carry value bytes
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
      // value_large grain (32 KiB) vs line grain (128 B): the index field is
      // identical; only the multiplier (and base) differ. (raw & ~0xF) is the
      // index << 4, so the deref shift is log2(grain) - 4 (128 B -> 3, the line
      // path; 32 KiB -> 11). One predictable, almost-always-false tag compare.
      static constexpr unsigned large_deref_shift_ =
          large_ok_ ? unsigned(std::countr_zero(start_ref::large_grain_bytes)) - 4u : 0u;
      static_assert(!large_ok_ || (start_ref::large_grain_bytes &
                                   (start_ref::large_grain_bytes - 1)) == 0,
                    "large grain must be a power of two");
      void* deref_large_(uint64_t raw) const noexcept
      {
         if constexpr (large_ok_)
            return const_cast<std::byte*>(large_base_) +
                   ((raw & ~uint64_t(0xF)) << large_deref_shift_);
         else
            return nullptr;
      }
      void* deref_(packed_ptr p) const noexcept
      {
         if constexpr (indexed)
         {
            if constexpr (large_ok_)
               if (p.tag() == K::value_large)
                  return deref_large_(p.raw());
            return const_cast<std::byte*>(base_) + ((p.raw() & ~uint64_t(0xF)) << 3);
         }
         else
            return p.ptr();
      }
      // Same resolution for a raw handle value already in a register (the descend hot path).
      // PURE line-arena resolution — value_large gets its own descend body so the
      // common router hops never pay a tag compare here.
      const char* deref_raw_(uint64_t raw) const noexcept
      {
         if constexpr (indexed)
            return reinterpret_cast<const char*>(base_) + ((raw & ~uint64_t(0xF)) << 3);
         else
            return reinterpret_cast<const char*>(raw & ~uint64_t(0x7f));
      }
      packed_ptr pack_(const void* node, K k) const noexcept
      {
         if constexpr (indexed)
         {
            // value_large: the node lives in the large arena — index it off the
            // large base in 32 KiB grains, not the line base in 128 B lines.
            if constexpr (large_ok_)
               if (k == K::value_large)
                  return packed_ptr::from_grain_off(
                      size_t(static_cast<const std::byte*>(node) - large_base_), k);
            return packed_ptr::from_off(size_t(static_cast<const std::byte*>(node) - base_), k);
         }
         else
            return packed_ptr::from(node, k);
      }
      // Cache the allocator's mapping base (indexed handles only; a no-op for direct).
      void rebase_() noexcept
      {
         if constexpr (indexed) base_ = alloc_.hkv_base();
         if constexpr (large_ok_) large_base_ = alloc_.hkv_large_base();
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
      Allocator&     allocator_mut() noexcept { return alloc_; }  // kv wiring (vhooks)

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
         return insert_impl_<Assign, false>(key, std::forward<VF>(v), &root_, 0, 0);
      }

      // template<Shared> (Dan / psitri's proven pattern, ported for apples-to-
      // apples vs vpart): UNIQUE carries zero ownership work and tail-switches
      // at the first ref>=2 node; SHARED clones EVERY hop unconditionally and
      // check-free — correct because cloning a shared parent RETAINS its
      // children (ref>=2 by construction), i.e. refcounts encode the
      // unique->shared latch dynamically. This is the old privatize pass
      // fused into the descent: same clones, same order, one walk.
      template <bool Assign, bool Shared, class VF>
      bool insert_impl_(std::string_view key, VF&& v, packed_ptr* slot, size_t depth,
                        unsigned lowfan_in)
      {
         [[maybe_unused]] packed_ptr* pslot = nullptr;  // slot holding the parent of *slot (for fusion)
         [[maybe_unused]] unsigned lowfan = lowfan_in;  // routers with <3 branches (deep-narrow signature)
         for (;;)
         {
            packed_ptr cur = *slot;
            if (cur.is_null())
            {
               *slot = make_value_at(key, depth, std::forward<VF>(v));
               ++count_;
               return true;
            }
            if constexpr (cow_enabled && !Shared)
               if (cur.tag() != K::value_inline && cow_shared_(cur))
                  return insert_impl_<Assign, true>(key, std::forward<VF>(v), slot, depth,
                                                    lowfan);
            if constexpr (cow_enabled && Shared)
               // eager: every hop is shared. A raw-bytes leaf is NOT cloned: the
               // leaf code below never edits a shared leaf, it builds the leaf's
               // successor straight from the shared bytes and drops one reference
               // (leaf_unshared_). Cloning it first copied the whole value, up
               // to 2 MiB, only to free the copy in the same call.
               if (cur.tag() != K::value_inline && !(leaf_no_clone_ && is_leaf(cur.tag())))
                  cur = cow_clone_into_(slot, cur);
            const K k = cur.tag();
            // Hoist the dominant descend: node_full child slot via the offset table
            // (no reciprocal, no router_find_slot re-dispatch).
            if (k == K::node_full)
            {
               char* f = static_cast<char*>(deref_(cur));
               if (depth == key.size())
               {
                  const bool ins = set_or_update_term<Assign>(reinterpret_cast<router_hdr*>(f), std::forward<VF>(v));
                  count_ += ins;
                  return ins;
               }
               const uint8_t byte = uint8_t(key[depth]);
               packed_ptr*   cs   = reinterpret_cast<packed_ptr*>(f + full_offtab_v<Ptr>[byte]);
               if (!cs->is_null())  // existing child -> descend
               {
                  if constexpr (WideStems) pslot = slot;
                  slot  = cs;
                  ++depth;
                  continue;
               }
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
                  else if (alloc_.hkv_ref(line_of(x->lines[li])) >= 2)
                  {  // per-line COW (was privatize's job): the descent or its
                     // writeback mutates this direct line — own it first
                     const Ptr old = x->lines[li];
                     x->lines[li]  = fx_clone_line_(old);
                     fx_release_line_(old);
                  }
                  cline_direct_t<Ptr>* L  = fx_line_(x, li);
                  packed_ptr*          cs = &L->ptr[sl];
                  if (!cs->is_null())
                  {
                     slot = cs;
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
                  const unsigned sg = byte >> (8 - x->lg);
                  if (!x->seglines[sg].is_null() &&
                      alloc_.hkv_ref(line_of(x->seglines[sg])) >= 2)
                  {  // per-line COW (was privatize's job)
                     const Ptr old   = x->seglines[sg];
                     x->seglines[sg] = cx_clone_line_(old, x->lg);
                     cx_release_line_(old, x->lg);
                  }
                  if (packed_ptr* cs = router_find_slot(x, byte))
                  {
                     slot = cs;
                     ++depth;
                     continue;
                  }
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
            switch (k)
            {
               case K::value_ptr:
               case K::value_large:  // a leaf in the large arena: same update/split logic
               case K::value_inline:
               {
                  bool inserted;
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
                     slot  = reinterpret_cast<packed_ptr*>(P + 2);  // the next-pointer slot
                     depth += PP.size();
                     continue;
                  }
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
                     const bool ins = set_or_update_term<Assign>(&s->rh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  if (int i = setlist_index(s, byte); i >= 0)  // existing child -> descend
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot  = &s->branch[i];
                     ++depth;
                     continue;
                  }
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
                     slot  = cs;
                     depth += 2;
                     continue;
                  }
                  if (s->rh.hdr.nbranch >= setlist16::CAP)  // full → re-stride, retry (v untouched)
                  {
                     *slot = restride_u16_to_u8(cur);
                     continue;
                  }
                  sl16_set(s, q, make_value_at(key, depth + 2, std::forward<VF>(v)));  // room: in place
                  ++count_;
                  return true;
               }
               case K::c2:  // dense tiers: no inline prefix (carried by a prefix_node)
               case K::c4:
               case K::c8:
               {
                  router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
                  if (depth == key.size())
                  {
                     const bool ins = set_or_update_term<Assign>(rh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  if (packed_ptr* cs = router_find_slot(cur, byte))  // existing child -> descend
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot  = cs;
                     ++depth;
                     continue;
                  }
                  packed_ptr lf = make_value_at(key, depth + 1, std::forward<VF>(v));
                  build_guard glf{this, &lf};  // grow can throw; don't orphan the new value
                  grow_router_(slot, cur, byte, lf);  // re-anchors *slot across the widen ladder
                  glf.release();
                  ++count_;
                  return true;
               }
               // exhaustive (no default): node_full descends before the switch and null
               // inserts at the top of the loop, so these can't reach it — but a NEW
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
// value_large: same op.leaf, but the handle indexes the LARGE arena. A separate
// body keeps deref_raw_ (the router hot path) free of any tag check.
#define HKV_BODY_VALUE_LARGE  return op.leaf(static_cast<const char*>(deref_large_(raw)), kp + depth, klen - depth);
#define HKV_BODY_VALUE_INLINE return op.inl(raw, depth == klen);
#define HKV_BODY_PREFIX { \
   const char* p = HKV_PTR(raw); const uint16_t pl = pn_plen(p); \
   if (klen - depth < pl || std::memcmp(pn_pfx<Ptr>(p), kp + depth, pl) != 0) return op.miss(); \
   std::memcpy(&raw, p + 2, sizeof(Ptr)); HKV_NEXT(depth + pl); }
#define HKV_BODY_SETLIST { \
   const setlist* s = reinterpret_cast<const setlist*>(HKV_PTR(raw)); const unsigned pl = s->rh.hdr.prefix_len; \
   if (pl) { if (klen - depth < pl || std::memcmp(s->prefix, kp + depth, pl) != 0) return op.miss(); depth += pl; } \
   if (depth == klen) return op.term(&s->rh); \
   raw = setlist_find(s, uint8_t(kp[depth])).raw(); HKV_NEXT(depth + 1); }
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
               case K::value_large:  HKV_BODY_VALUE_LARGE
               case K::value_inline: HKV_BODY_VALUE_INLINE
               case K::prefix_node:  HKV_BODY_PREFIX
               case K::setlist_u8:   HKV_BODY_SETLIST
               case K::node_full:    HKV_BODY_FULL
               case K::c2:           HKV_BODY_CN(2)
               case K::c4:           HKV_BODY_CN(4)
               case K::c8:           HKV_BODY_CN(8)
               case K::bucket:       return op.miss();  // reserved tag: never produced
               case K::setlist_u16:  HKV_BODY_SETLIST16
               case K::full_x:       HKV_BODY_FULLX
               case K::cx:           HKV_BODY_CX
               case K::cbucket:      return op.miss();  // reserved tag: never produced
               case K::null:         return op.miss();
            }
            __builtin_unreachable();
         }
      }
#undef HKV_NEXT
#undef HKV_PTR
#undef HKV_BODY_VALUE_PTR
#undef HKV_BODY_VALUE_LARGE
#undef HKV_BODY_VALUE_INLINE
#undef HKV_BODY_PREFIX
#undef HKV_BODY_SETLIST
#undef HKV_BODY_SETLIST16
#undef HKV_BODY_FULL
#undef HKV_BODY_CX
#undef HKV_BODY_FULLX
#undef HKV_BODY_CN

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
            bool term(const router_hdr* rh) const { return self->read_term(rh, *out); }
            bool miss() const { return false; }
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
            bool term(const router_hdr* rh) const { return !rh->term.is_null(); }
            bool miss() const { return false; }
         };
         return descend_read(key, op_t{});
      }

      // ── upsert (STL-ish value-write semantics) ──────────────────────────────────
      // upsert = insert-or-assign: insert if absent, overwrite if present. Returns true
      // iff NEWLY inserted. (hkv's insert already has this behavior; these are the
      // explicit/STL names.)
      bool upsert(key_param k, const T& v) { return insert(k, v); }
      bool upsert(key_param k, T&& v) { return insert(k, std::move(v)); }
      bool insert_or_assign(key_param k, const T& v) { return insert(k, v); }
      bool insert_or_assign(key_param k, T&& v) { return insert(k, std::move(v)); }

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
         throw std::out_of_range("hkv::kv_tree::at: key not found");
      }
      const T& at(key_param k) const
         requires(!inlineable)
      {
         scratch_t ks_;
         if (const T* p = locate_(enc_(k, ks_))) return *p;
         throw std::out_of_range("hkv::kv_tree::at: key not found");
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
         // a raw-bytes value is a view of the leaf, materialized here per yield
         // (raw storage: T's name isn't needed at declaration parse time)
         alignas(alignof(T)) mutable unsigned char cvalbuf_[sizeof(T)];
         mutable frame_stack  stk_;  // mutable: first key() primes plen/cbase
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
         // true → positioned at a terminal (sets val_ + suffix); false → pushed a frame.
         // `base` is the key length at this child's start; key_ is grown only when
         // maintain_ is on (the eager fast path), otherwise the path in stk_ suffices.
         bool descend_(packed_ptr child, size_t base)
         {
            switch (child.tag())
            {
               case K::value_ptr:
               case K::value_large:  // a leaf in either arena: deref_ routes by tag
               {
                  const char* L = static_cast<const char*>(deref_(child));
                  sfx_          = leaf_suf(L);
                  sfxlen_       = leaf_slen(L);
                  if (leaf_has_full(L)) { full_str_ = leaf_str(L); full_len_ = leaf_strlen(L); }
                  else                    full_str_ = nullptr;
                  if constexpr (kv_tree::raw_vals)
                  {
                     *reinterpret_cast<T*>(cvalbuf_) = T{leaf_bytes(L)};
                     val_local_ = true;
                  }
                  else
                     set_ptr_(leaf_val(L));
                  // maintain the path in key_ for the NEXT non-full terminal; skip the suffix
                  // append here when this leaf is full (key() will return its zero-copy view).
                  if (maintain_) { key_.len = base; if (!full_str_) key_.append(sfx_, sfxlen_); }
                  return true;
               }
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
               case K::cbucket:  // reserved tags: never produced
                  return false;
               default:  // routers (setlist / c2 / c4 / c8 / node_full)
               {
                  const router_hdr* rh = static_cast<const router_hdr*>(deref_(child));
                  uint32_t          cb = uint32_t(base);
                  if (maintain_)
                  {
                     key_.len = base;
                     if (child.tag() == K::setlist_u8 && rh->hdr.prefix_len)
                        key_.append(static_cast<const setlist*>(deref_(child))->prefix, rh->hdr.prefix_len);
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
               if (k == K::setlist_u16)  // wide router: f.cur = -1 (term pending) then stem index
               {
                  const setlist16* s = static_cast<const setlist16*>(deref_(f.node));
                  if (f.cur == -1)
                  {
                     f.cur = 0;
                     if (!s->rh.term.is_null())
                     {
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
                  if (descend_(child, key_.len)) return;
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
            if (is_leaf(k))  // value_ptr or value_large: same suffix-compare positioning
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
            // routers (setlist / c2 / c4 / c8 / node_full)
            const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
            if (k == K::setlist_u8 && rh->hdr.prefix_len)
            {
               const setlist*   s = static_cast<const setlist*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(s->prefix), rh->hdr.prefix_len);
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
               if (!descend_lb_(root, key, 0, upper))
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
               else  // c2/c4/c8/node_full (edge only)
               {
                  fr.cbase = uint32_t(key_.len);
                  if (fr.cur >= 1) key_.push(uint8_t(fr.cur - 1));
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
      void rm_free_(packed_ptr nd) noexcept { node_free_(deref_(nd), node_size(nd.tag())); }

      // Collapse a single-branch term-less router: absorb its inline prefix (setlist only)
      // + the edge to its one child INTO the child — free when the child is a setlist with
      // prefix room, else via a fresh prefix node (best-effort: kept as-is on bad_alloc).
      // pfx holds the bytes to prepend (router prefix ++ edge byte(s)).
      void rm_collapse_(packed_ptr* slot, packed_ptr nd, const uint8_t* pfx, unsigned pl,
                        packed_ptr child) noexcept
      {
         // Absorb the parent's edge prefix into the child setlist IN PLACE — only when
         // the child is exclusively owned. Under COW the child may be shared with a
         // snapshot (refcount >= 2 once the parent was cloned on the way down); mutating
         // its prefix bytes would corrupt that snapshot's view of the subtree. When
         // shared, fall through to make_prefix (creates a NEW prefix node above the
         // child — no mutation of the shared node).
         if (child.tag() == K::setlist_u8 && !(cow_enabled && cow_shared_(child)))
         {
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
               // inline value) replaces the router. A setlist's inline prefix is part of that
               // key's path — it must survive as a prefix node above the term.
               if (k == K::setlist_u8)
               {
                  setlist* s = static_cast<setlist*>(deref_(nd));
                  if (const unsigned pl = s->rh.hdr.prefix_len)
                  {
                     try
                     {
                        *slot = make_prefix(
                            std::string_view(reinterpret_cast<char*>(s->prefix), pl), rh->term);
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
                  *slot = pack_(s2, K::setlist_u8);
                  if constexpr (cow_enabled)
                  {  // tails may be shared (post-demotion): retain the copies,
                     // release the node line-wise — pairs exactly either way
                     cow_fix_child_(&s2->rh.term);
                     for (int i = 0; i < s2->rh.hdr.nbranch; ++i)
                        cow_fix_child_(&s2->branch[i]);
                     cow_release(nd);
                  }
                  else
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
      // collapses are best-effort: on bad_alloc the tree simply stays unshrunk. The whole
      // path is recorded (64 levels inline, the rest on the heap), so every ancestor is
      // considered however deep the key.
      // NOTE: not noexcept — removing a non-POD value from a bucket rebuilds it (allocates),
      // and a custom key_codec may allocate. Nothrow in practice for POD values / radix mode.
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
         // A leaf shared with another version (cow_remove no longer clones it
         // first) only loses this tree's reference.
         auto free_leaf         = [&](packed_ptr lp) {
            char* L = static_cast<char*>(deref_(lp));
            if constexpr (!raw_vals)
               v_destroy(alloc_, leaf_val(L));
            leaf_drop_(lp, L, leaf_alloc_size(L), lp.tag());  // routes value_large to the large arena
         };
         for (;;)
         {
            const packed_ptr cur = *slot;
            switch (cur.tag())
            {
               case K::null:
                  return false;
               case K::value_ptr:
               case K::value_large:  // a leaf in the large arena: same remove logic
               {
                  char* L = static_cast<char*>(deref_(cur));
                  if (std::string_view(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L)) !=
                      key.substr(depth))
                     return false;
                  free_leaf(cur);
                  --count_;
                  if (n >= 2) rm_unwind_(path, n, n - 2);  // n-2: the leaf's parent
                  else root_ = packed_ptr::null();         // the leaf WAS the root
                  return true;
               }
               case K::value_inline:
               {
                  if (depth != key.size())
                     return false;  // inline terminal matches only at key end; nothing to free
                  --count_;
                  if (n >= 2) rm_unwind_(path, n, n - 2);
                  else root_ = packed_ptr::null();
                  return true;
               }
               case K::bucket:
               case K::cbucket:  // reserved tags: never produced
                  return false;
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
               case K::setlist_u8:
               case K::c2:
               case K::c4:
               case K::c8:
               case K::node_full:
               case K::full_x:  // explicit kinds: same descent via the slot
               case K::cx:      // dispatchers; rm_shrink_ leaves them unshrunk
               {
                  if (cur.tag() == K::setlist_u8)
                  {
                     const setlist* s  = static_cast<const setlist*>(deref_(cur));
                     const unsigned pl = s->rh.hdr.prefix_len;
                     if (pl)
                     {
                        if (key.size() - depth < pl ||
                            std::memcmp(s->prefix, key.data() + depth, pl) != 0)
                           return false;
                        depth += pl;
                     }
                  }
                  if (depth == key.size())
                     goto rm_term;
                  {
                     const uint8_t byte = uint8_t(key[depth]);
                     packed_ptr*   cs   = router_find_slot(cur, byte);
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
            router_hdr* rh = static_cast<router_hdr*>(deref_(*slot));
            if (rh->term.is_null())
               return false;
            if (is_leaf(rh->term.tag()))  // leaf("",v); an inline term has nothing to free
               free_leaf(rh->term);
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
            case K::value_ptr:
            case K::value_large:  ++d.leaf; ++d.terminals; d.sum_depth += depth;
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
      // ── leaf<T> layout: [u16 suffix_len][u16 pfx_field][u32 vlen / pad to alignof(T)][value][key bytes] ─
      // pfx_field = (has_full << 15) | prefix_len. The trailing key bytes are either the
      // suffix only (has_full=0, the compressed form) OR the WHOLE key = prefix ++ suffix
      // laid out contiguously (has_full=1) — stored opportunistically when the prefix fits
      // in the 128-byte slot we'd allocate anyway, so the iterator can hand back the full
      // key as a zero-copy string_view (same cacheline as the value). leaf_suf() points at
      // the suffix in both forms, so point lookup is identical.
      static constexpr uint16_t LEAF_FULL = 0x8000;
      // raw_vals value length is u32 (offset 4): values up to the large arena's
      // 2 MiB cap (value_large) need >16 bits; a u16 capped at 65535. The +2 B
      // over the old u16 layout is uniform for value_ptr and value_large, so the
      // two are byte-identical (the design's "same leaf layout"). suffix/prefix
      // lengths stay u16 (keys never exceed 64 KB).
      static constexpr uint16_t LEAF_VLEN_OFF = 4;
      static constexpr size_t   leaf_value_off() noexcept
      {
         if constexpr (raw_vals)
            return 2 * sizeof(uint16_t) + sizeof(uint32_t);  // [slen][pfx_field][vlen u32]
         else
         {
            constexpr size_t a = alignof(T);
            return (2 * sizeof(uint16_t) + a - 1) / a * a;
         }
      }
      static size_t leaf_size(size_t strlen, size_t vlen = sizeof(T)) noexcept
      {
         if constexpr (raw_vals)
            return leaf_value_off() + vlen + strlen;
         else
            return leaf_value_off() + sizeof(T) + strlen;
      }
      static uint32_t leaf_vlen(const char* p) noexcept
      {
         if constexpr (raw_vals)
         {
            uint32_t v;
            std::memcpy(&v, p + LEAF_VLEN_OFF, 4);
            return v;
         }
         else
            return uint32_t(sizeof(T));
      }
      static void leaf_set_vlen(char* p, uint32_t vlen) noexcept
         requires raw_vals
      {
         std::memcpy(p + LEAF_VLEN_OFF, &vlen, 4);
      }
      static std::string_view leaf_bytes(const char* p) noexcept
         requires raw_vals
      {
         return std::string_view(p + leaf_value_off(), leaf_vlen(p));
      }
      static uint16_t leaf_slen(const char* p) noexcept { uint16_t v; std::memcpy(&v, p, 2); return v; }
      static uint16_t leaf_pfxfield(const char* p) noexcept { uint16_t v; std::memcpy(&v, p + 2, 2); return v; }
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
            case K::value_large:  // a leaf in either arena: deref_ routes by tag
               f(static_cast<const T&>(*leaf_val(static_cast<char*>(deref_(cur)))));
               return;
            case K::value_inline:
               if constexpr (inlineable) { T tmp; unpack_inline(cur, tmp); f(static_cast<const T&>(tmp)); }
               return;
            case K::prefix_node:
               walk_v_(pn_next<Ptr>(static_cast<char*>(deref_(cur))), f);
               return;
            case K::bucket:
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
                     for (int i = 0; i < n; ++i) walk_v_(s->branch[i], f);
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
            case K::value_large:  // a leaf in either arena: deref_ routes by tag
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
                     for (int i = 0; i < n; ++i) recurse(s->bytes[i], s->branch[i]);  // bytes[] sorted
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
      // GATHER protocol (raw_vals): the value source emits its bytes DIRECTLY
      // into the leaf's value region — no intermediate contiguous view, no
      // pre-built envelope, one copy at the single materialization point. A
      // plain T source (bytes_value) is a view; any other source exposes
      // size()/write(dst). Sources are view-like, so reading size then bytes
      // (two reads) is fine — callers pass `val` by const-ref here.
      template <class VF>
      static size_t vf_emit_size_(const VF& vf) noexcept
         requires raw_vals
      {
         if constexpr (std::is_same_v<std::remove_cvref_t<VF>, T>)
            return vf.v.size();
         else
            return vf.size();
      }
      template <class VF>
      static void vf_emit_(const VF& vf, char* dst) noexcept
         requires raw_vals
      {
         if constexpr (std::is_same_v<std::remove_cvref_t<VF>, T>)
            std::memcpy(dst, vf.v.data(), vf.v.size());
         else
            vf.write(dst);
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
         std::memcpy(p, &slen, 2);
         std::memcpy(p + 2, &pfxfield, 2);
      }
      template <class VF>
      packed_ptr make_leaf(std::string_view suf, VF&& val)  // suffix-only (compressed) leaf
      {
         if constexpr (raw_vals)
         {
            const size_t sz = vf_emit_size_(val);  // gather: source emits in place
            K            kt = K::value_ptr;         // leaf_alloc_ may pick value_large
            char* p = leaf_alloc_(leaf_size(suf.size(), sz), kt);
            leaf_set_hdr(p, uint16_t(suf.size()), 0);
            leaf_set_vlen(p, uint32_t(sz));
            vf_emit_(val, p + leaf_value_off());
            std::memcpy(leaf_str(p), suf.data(), suf.size());
            return pack_(p, kt);
         }
         else
         {
            K     kt = K::value_ptr;
            char* p  = leaf_alloc_(leaf_size(suf.size()), kt);
            leaf_set_hdr(p, uint16_t(suf.size()), 0);
            std::memcpy(leaf_str(p), suf.data(), suf.size());
            vf_construct(std::forward<VF>(val), leaf_val(p));
            return pack_(p, kt);
         }
      }
      template <class VF>
      packed_ptr make_leaf_full(std::string_view pfx, std::string_view suf, VF&& val)  // whole key
      {
         if constexpr (raw_vals)
         {
            const size_t sz = vf_emit_size_(val);  // gather: source emits in place
            K            kt = K::value_ptr;
            char* p = leaf_alloc_(leaf_size(pfx.size() + suf.size(), sz), kt);
            leaf_set_hdr(p, uint16_t(suf.size()), uint16_t(LEAF_FULL | pfx.size()));
            leaf_set_vlen(p, uint32_t(sz));
            vf_emit_(val, p + leaf_value_off());
            std::memcpy(leaf_str(p), pfx.data(), pfx.size());
            std::memcpy(leaf_str(p) + pfx.size(), suf.data(), suf.size());
            return pack_(p, kt);
         }
         else
         {
            K     kt = K::value_ptr;
            char* p  = leaf_alloc_(leaf_size(pfx.size() + suf.size()), kt);
            leaf_set_hdr(p, uint16_t(suf.size()), uint16_t(LEAF_FULL | pfx.size()));
            std::memcpy(leaf_str(p), pfx.data(), pfx.size());
            std::memcpy(leaf_str(p) + pfx.size(), suf.data(), suf.size());
            vf_construct(std::forward<VF>(val), leaf_val(p));
            return pack_(p, kt);
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
            vsz = vf_emit_size_(val);
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
            // exhaustive (no default): a new node kind must declare its size here.
            case K::value_ptr: case K::value_large: case K::value_inline:
            case K::prefix_node: case K::bucket: case K::cbucket: case K::null:
               assert(false); __builtin_unreachable();  // leaves size via leaf_alloc_size
         }
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
            // exhaustive (no default): a new K must be decided here — and only here.
            case K::value_ptr: case K::value_large: case K::value_inline:
            case K::prefix_node: case K::setlist_u16: case K::bucket:
            case K::cbucket: case K::null:
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
         const K      k  = cur.tag();
         const size_t os = node_size(k);
         packed_ptr   out;
         if constexpr (cow_enabled)
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
            const bool implicit_ok = used + 1 >= nrange;  // <=1 empty range line
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
         return out;
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
         char* pn = nullptr;  // when set, re-anchor target is this prefix_node's next field
         if (cur.tag() == K::setlist_u8)
         {
            setlist*       s  = static_cast<setlist*>(deref_(cur));
            const unsigned pl = s->rh.hdr.prefix_len;
            if (pl)
            {
               uint8_t pfx[setlist::PREFIX_CAP];
               std::memcpy(pfx, s->prefix, pl);
               packed_ptr pnp = make_prefix(std::string_view(reinterpret_cast<char*>(pfx), pl), cur);  // throws → unchanged
               s->rh.hdr.prefix_len = 0;  // prefix now carried by pnp above the (to-be-widened) router
               *slot = pnp;
               pn    = static_cast<char*>(deref_(pnp));
            }
         }
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
               if constexpr (cow_enabled && !leaf_no_clone_)  // fused privatize: own the term leaf
                  if (is_leaf(rh->term.tag()) && cow_shared_(rh->term))
                     cow_clone_into_(&rh->term, rh->term);
               if (rh->term.tag() == K::value_inline)
                  rh->term = pack_inline(vf_make(std::forward<VF>(val)));
               else
               {
                  char* tl = static_cast<char*>(deref_(rh->term));
                  if constexpr (raw_vals)
                  {  // same-size and ours alone: bytes in place; else the term
                     // leaf is rebuilt and the old one loses a reference (a
                     // shared leaf is never written, and never copied first)
                     const size_t sz = vf_emit_size_(val);
                     if (sz == leaf_vlen(tl) && leaf_unshared_(rh->term))
                     {
                        vh_leaf_dies_(tl);  // old bytes clobbered in place
                        vf_emit_(val, tl + leaf_value_off());
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
         const K          lt = cur.tag();  // value_ptr or value_large — frees route by it
         char*            L = static_cast<char*>(deref_(cur));
         std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
         std::string_view R = key.substr(depth);
         if (S == R)  // exact key
         {
            if constexpr (Assign)
            {
               if constexpr (raw_vals)
               {
                  const size_t sz = vf_emit_size_(val);
                  if (sz == leaf_vlen(L) && leaf_unshared_(cur))
                  {
                     vh_leaf_dies_(L);  // old bytes clobbered in place
                     vf_emit_(val, L + leaf_value_off());
                  }
                  else
                  {  // size changed, or the leaf is shared with another version:
                     // rebuild the leaf verbatim around new bytes
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
                     leaf_drop_(cur, L, Lsz, lt);  // ours alone: bytes die; shared: one reference less
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
         {  // The leaf is ours alone: its bytes MOVE into exactly one new node and
            // the old leaf is freed, one live copy throughout, so no vh hooks
            // fire. The leaf is shared with another version: the new node holds
            // a second copy (the copy hook fires, as it does for a clone) and
            // the old leaf only loses this tree's reference.
            const bool mine = leaf_unshared_(cur);
            const T    ov{leaf_bytes(L)};
            packed_ptr moved;
            if (c == S.size())  // (4) old value, LAST
               moved = make_value(std::string_view{}, ov);
            else if (oldfull)
               moved = make_value_at(ofv, depth + c + 1, ov);
            else
               moved = make_value(S.substr(c + 1), ov);
            if constexpr (vh_on)
               if (!mine)
               {
                  try { vh_copy_(leaf_bytes(static_cast<const char*>(deref_(moved)))); }
                  catch (...)
                  {  // the copy was refused: the new leaf's bytes never lived
                     char* M = static_cast<char*>(deref_(moved));
                     leaf_free_(M, leaf_alloc_size(M), moved.tag());
                     throw;
                  }
               }
            if (c == S.size())
               set_term_ptr_(&r->rh, moved);
            else
               setlist_set(r, uint8_t(S[c]), moved);
            if (mine)
               leaf_free_(L, Lsize, lt);  // routes value_large to the large arena
            else
               cow_release(cur);
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
            leaf_free_(L, Lsize, lt);  // routes value_large to the large arena
         }
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
            case K::value_large:  // a leaf in the large arena: free there
            {
               char* L = static_cast<char*>(deref_(cur));
               if constexpr (!std::is_trivially_destructible_v<T>) v_destroy(alloc_, leaf_val(L));
               vh_leaf_dies_(L);
               leaf_free_(L, leaf_alloc_size(L), cur.tag());  // routes value_large to the large arena
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
            case K::setlist_u16:
            case K::c2:
            case K::c4:
            case K::c8:
            case K::node_full:
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
               free_all_(rh->term);
               router_for_each(cur, [&](uint8_t, packed_ptr br) { free_all_(br); });
               node_free_(deref_(cur), node_size(cur.tag()));
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
            case K::bucket:
            case K::cbucket:  // reserved tags: never produced
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

      Ptr    root_handle() const noexcept { return root_; }
      size_t size_hint() const noexcept { return count_; }

      // TEMP shape audit: single-child routers + node histogram + avg leaf depth.
      struct shape_stats
      {
         size_t total = 0, routers = 0, setlist = 0, setlist1 = 0, router1 = 0, prefix = 0,
                leaves = 0, buckets = 0, leafdepth = 0, nleaf = 0;
      };
      void audit_walk_(Ptr p, unsigned depth, shape_stats& s) const
      {
         if (p.is_null() || p.tag() == K::value_inline) return;
         const K k = p.tag();
         s.total++;
         if (k == K::value_ptr || k == K::value_large) { s.leaves++; s.leafdepth += depth; s.nleaf++; return; }
         if (k == K::prefix_node)
         {
            s.prefix++;
            audit_walk_(pn_next<Ptr>(static_cast<const char*>(deref_(p))), depth + 1, s);
            return;
         }
         // everything else is a router (setlist_u8/u16, c2/c4/c8, cx, full_x, node_full)
         s.routers++;
         auto*          rh = static_cast<router_hdr*>(deref_(p));
         const unsigned nb = rh->hdr.nbranch;
         if (nb == 1) s.router1++;
         if (k == K::setlist_u8) { s.setlist++; if (nb == 1) s.setlist1++; }
         if (!rh->term.is_null()) audit_walk_(rh->term, depth + 1, s);
         router_for_each(p, [&](unsigned, Ptr c) { audit_walk_(c, depth + 1, s); });
      }
      shape_stats audit_shape() const { shape_stats s; audit_walk_(root_, 0, s); return s; }
      shape_stats audit_shape_from(Ptr root) const { shape_stats s; audit_walk_(root, 0, s); return s; }
      // Non-owning view management (snapshot cursors): adopt a pinned root and
      // null it back out before destruction so the dtor frees nothing.
      void adopt_root(Ptr p, size_t n = 0) noexcept
      {
         root_  = p;
         count_ = n;
      }
      uint32_t line_of(Ptr p) const noexcept { return uint32_t(p.raw() >> 4); }

      // +1 on a subtree root (root-slot / snapshot bookkeeping). False would
      // mean 190 state refs on one node — unreachable under sane root usage.
      bool cow_retain(Ptr p)
      {
         if constexpr (cow_enabled)
         {
            if (p.is_null() || p.tag() == K::value_inline)
               return true;
            if constexpr (large_ok_)
               if (p.tag() == K::value_large)  // share count lives in the large arena
                  return alloc_.hkv_large_retain(line_of(p));
            return alloc_.hkv_retain(line_of(p));
         }
         else
            return true;
      }

      // Drop one reference to the subtree at p. A release that would hit ZERO
      // on a CASCADING node can be a huge subtree even mid-write (remove,
      // privatize, retired generations) — hand the whole thing to the
      // background releaser when one is registered. In shared mode the
      // decrement cannot reach zero (another generation owns it indirectly),
      // so deferral triggers only on unique ownership; the deferred node's
      // LAST reference rides the ring (no decrement here).
      void cow_release(Ptr p) noexcept
      {
         if constexpr (cow_enabled)
         {
            if (p.is_null() || p.tag() == K::value_inline)
               return;
            if constexpr (large_ok_)
               if (p.tag() == K::value_large)  // a single large-arena leaf: free there, no cascade
               {
                  cow_release_now(p);
                  return;
               }
            if (!is_leaf(p.tag()) && alloc_.hkv_ref(line_of(p)) == 1)
            {
               // bounded foreground work per op: small cascades run inline
               // (and feed the writer's hot-hole fcache); once the budget is
               // spent, remaining SUBTREES offload to the background releaser
               if (rel_budget_ == 0)
               {
                  if (alloc_.hkv_defer(p.raw()))
                     return;  // ownership moved to the releaser
               }
               else
                  --rel_budget_;
            }
            cow_release_now(p);
         }
      }
      uint32_t rel_budget_ = 64;  // refilled at op entry; counts to-zero cascades
      void     release_budget(uint32_t n) noexcept { rel_budget_ = n; }
      // the inline path (also the releaser's worker path: its allocator does
      // not defer, so cascades run here to completion)
      void cow_release_now(Ptr p) noexcept
      {
         if constexpr (cow_enabled)
         {
            if (p.is_null() || p.tag() == K::value_inline)
               return;
            if constexpr (large_ok_)
               if (p.tag() == K::value_large)
               {
                  // value_large leaf: refcount + free route to the LARGE arena by
                  // grain. The object spans bytes/grain units; the head grain is the
                  // share count, the run is freed when it dies (vh hook fires once).
                  char*          L     = static_cast<char*>(deref_(p));
                  const uint32_t units = uint32_t(
                      (leaf_alloc_size(L) + start_ref::large_grain_bytes - 1) /
                      start_ref::large_grain_bytes);
                  alloc_.hkv_large_release(line_of(p), units, [&] { vh_node_dies_(p); });
                  return;
               }
            // A node that dies releases its children, which may die and
            // release theirs: one call level per level of the tree, and a
            // tree is as deep as its longest key. The caller may be any
            // thread with any stack (a handle or cursor being dropped), so
            // the descent is cut off at rel_max_depth_: a subtree this call
            // alone owns is parked and destroyed from the outermost frame.
            if (rel_depth_ >= rel_max_depth_ && !is_leaf(p.tag()) &&
                alloc_.hkv_ref(line_of(p)) == 1)
            {
               try
               {
                  rel_parked_.push_back(p);
                  return;
               }
               catch (...)
               {  // no memory for the list: descend after all
               }
            }
            ++rel_depth_;
            rel_one_(p);
            if (--rel_depth_ == 0)
               while (!rel_parked_.empty())
               {
                  const Ptr q = rel_parked_.back();
                  rel_parked_.pop_back();
                  ++rel_depth_;
                  rel_one_(q);
                  --rel_depth_;
               }
         }
      }
      void rel_one_(Ptr p) noexcept
      {
         alloc_.hkv_release(line_of(p),
                            [&](uint32_t)
                            {
                               vh_node_dies_(p);  // refcount-zero: bytes die
                               cow_destroy_dead_(p);
                            });
      }
      // Call levels a release may descend before parking the rest. Each level
      // is a few hundred bytes of stack in an optimized build, about a
      // kilobyte in an unoptimized one.
      static constexpr uint32_t rel_max_depth_ = 32;
      uint32_t                  rel_depth_     = 0;
      std::vector<Ptr>          rel_parked_;

      template <class V>
      bool cow_insert(key_param k, V&& v)
      {
         rel_budget_ = 64;  // per-op inline release budget
         static_assert(cow_enabled, "allocator has no refcount hooks");
         static_assert(std::is_trivially_copyable_v<T>, "COW memcpy-clones leaf values");
         // privatize is FUSED into insert_'s template<Shared> descent.
         // The value SOURCE (a plain T, or a VF that gathers into the leaf) is
         // forwarded straight through — it materializes exactly once at the leaf,
         // never as an intermediate T here.
         return insert_<true>(k, std::forward<V>(v));
      }
      // NOTE: remove()'s fusion paths free child nodes unconditionally; callers
      // must not hold snapshots across cow_remove. Fusion-under-sharing is
      // future work; the bank workload never removes.
      bool cow_remove(key_param k)
      {
         rel_budget_ = 64;  // per-op inline release budget
         static_assert(cow_enabled, "allocator has no refcount hooks");
         {
            scratch_t ks;
            cow_privatize_(enc_(k, ks));
         }
         return remove(k);
      }

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
         // value_large refs live in the SEPARATE large arena (grain-indexed), not
         // the line-arena refcount bytes this map validates — skip it (no children).
         if constexpr (large_ok_)
            if (cur.tag() == K::value_large)
               return;
         const K k = cur.tag();
         refadd_(exp, cur, uint32_t((cow_node_bytes_(cur) + 127) / 128));
         switch (k)
         {
            case K::value_ptr:
            case K::value_large:  // unreachable (skipped above); keeps the switch exhaustive
            case K::bucket:
            case K::value_inline:
            case K::null:
               return;
            case K::prefix_node:
               refwalk_(pn_next<Ptr>(static_cast<char*>(deref_(cur))), exp);
               return;
            case K::setlist_u8:
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
            case K::cbucket:  // reserved tag: never produced
               return;
         }
      }

      // ── recovery: re-derive liveness from a root ────────────────────────────
      // Visits everything reachable from `p` and tells the marker `m` which
      // storage it occupies, so the arena's free map and reference counts can
      // be rebuilt from the tree alone. It is the exact mirror of the release
      // cascade (cow_release / cow_destroy_dead_): what a release would
      // decrement, this counts; what a death would free, this claims.
      //
      //   m.check_line(i) / m.check_large(g)  throw unless the index is inside the arena
      //   m.line(i)        one more reference to line i; true the first time
      //   m.tail(i, n)     n lines owned by a node head (always one reference)
      //   m.large(g, u)    one more reference to a u-grain large object; true the first time
      //   m.leaf_value(b)  the value bytes of a leaf seen for the first time
      //   m.unsupported()  a node kind this walk does not know
      //
      // A node reached a second time only gains a reference: its contents were
      // counted when it was first reached. Depth follows key length.
      template <class M>
      void recover_mark(Ptr p, M& m) const
      {
         if (p.is_null() || p.tag() == K::value_inline)
            return;
         const K k = p.tag();
         if constexpr (large_ok_)
            if (k == K::value_large)
            {
               m.check_large(line_of(p));
               const uint32_t units = uint32_t(
                   (cow_node_bytes_(p) + start_ref::large_grain_bytes - 1) /
                   start_ref::large_grain_bytes);
               if (m.large(line_of(p), units))
                  recover_leaf_(p, m);
               return;
            }
         m.check_line(line_of(p));
         if (!m.line(line_of(p)))
            return;
         const size_t bytes = cow_node_bytes_(p);
         switch (k)
         {
            case K::value_ptr:
               recover_leaf_(p, m);
               break;
            case K::value_large:
            case K::value_inline:
            case K::null:
               return;
            case K::bucket:
            case K::cbucket:
               m.unsupported();
               return;
            case K::prefix_node:
               recover_mark(pn_next<Ptr>(static_cast<char*>(deref_(p))), m);
               break;
            case K::setlist_u8:
            case K::setlist_u16:
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(p));
               recover_mark(rh->term, m);
               router_for_each(p, [&](unsigned, Ptr c) { recover_mark(c, m); });
               break;
            }
            // Implicit multi-line routers: each line after the head carries its
            // own count and owns the children inside it.
            case K::c2:
            case K::c4:
            case K::c8:
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(p));
               recover_mark(rh->term, m);
               const unsigned lg        = k == K::c2 ? 1 : k == K::c4 ? 2 : 3;
               const uint32_t base_line = line_of(p) + 1;
               for (unsigned sg = 0; sg < (1u << lg); ++sg)
                  recover_cx_line_(Ptr::from_off(size_t(base_line + sg) << 7, K::node_full), lg,
                                   m);
               return;
            }
            case K::node_full:
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(p));
               recover_mark(rh->term, m);
               const uint32_t base_line = line_of(p) + 1;
               for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                  recover_fx_line_(Ptr::from_off(size_t(base_line + li) << 7, K::node_full), m);
               return;
            }
            case K::full_x:
            {
               if constexpr (cow_enabled)
               {
                  const node_fullx* x = static_cast<const node_fullx*>(deref_(p));
                  recover_mark(x->rh.term, m);
                  for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                     recover_fx_line_(x->lines[li], m);
               }
               break;
            }
            case K::cx:
            {
               if constexpr (cow_enabled)
               {
                  const node_cx* x = static_cast<const node_cx*>(deref_(p));
                  recover_mark(x->rh.term, m);
                  for (unsigned sg = 0; sg < (1u << x->lg); ++sg)
                     recover_cx_line_(x->seglines[sg], x->lg, m);
               }
               break;
            }
         }
         const uint32_t nl = uint32_t((bytes + 127) / 128);
         if (nl > 1)
            m.tail(line_of(p) + 1, nl - 1);
      }

     private:
      template <class M>
      void recover_leaf_(Ptr p, M& m) const
      {
         if constexpr (raw_vals)
            m.leaf_value(leaf_bytes(static_cast<const char*>(deref_(p))));
      }
      template <class M>
      void recover_fx_line_(Ptr lp, M& m) const
      {
         if (lp.is_null())
            return;
         m.check_line(line_of(lp));
         if (!m.line(line_of(lp)))
            return;
         const auto* L = static_cast<const cline_direct_t<Ptr>*>(deref_(lp));
         for (unsigned sl = 0; sl < full_per_line_v<Ptr>; ++sl)
            if (!L->ptr[sl].is_null())
               recover_mark(L->ptr[sl], m);
      }
      template <class M>
      void recover_cx_line_(Ptr lp, unsigned lg, M& m) const
      {
         if (lp.is_null())
            return;
         m.check_line(line_of(lp));
         if (!m.line(line_of(lp)))
            return;
         uint8_t*  seg = static_cast<uint8_t*>(deref_(lp));
         Ptr*      br  = cx_branches<Ptr>(seg, lg);
         const int cnt = bm_count(seg, int(cx_maskb(lg)));
         for (int i = 0; i < cnt; ++i)
            recover_mark(br[i], m);
      }

     public:
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
            case K::value_large:  // a leaf (line or large arena): report its bytes
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
            case K::cbucket:  // reserved tags: never produced
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
            case K::value_large:  // same leaf layout — leaf_alloc_size is arena-agnostic
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
               return node_size(p.tag());
            case K::bucket:
            case K::cbucket:  // reserved tags: never produced
            case K::value_inline:
            case K::null:
               return 0;
         }
         return 0;
      }

      void cow_destroy_dead_(Ptr p) noexcept
      {
         const K      k     = p.tag();
         const size_t bytes = cow_node_bytes_(p);
         switch (k)
         {
            case K::value_ptr:
            case K::value_large:  // a leaf has no children to cascade
            case K::bucket:
            case K::cbucket:  // reserved tags: never produced
               break;
            case K::prefix_node:
               cow_release(pn_next<Ptr>(static_cast<char*>(deref_(p))));
               break;
            case K::setlist_u8:
            case K::setlist_u16:
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(p));
               cow_release(rh->term);
               router_for_each(p, [&](unsigned, Ptr c) { cow_release(c); });
               break;
            }
            // Implicit multi-line routers release their LINES, not the children
            // inside them (design5: ownership of branches rides the line, which
            // a demoted explicit node may share). A line that dies releases its
            // own children; a shared line survives with its children intact.
            case K::c2:
            case K::c4:
            case K::c8:
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(p));
               cow_release(rh->term);
               const unsigned lg = k == K::c2 ? 1 : k == K::c4 ? 2 : 3;
               const uint32_t base_line = line_of(p) + 1;
               for (unsigned sg = 0; sg < (1u << lg); ++sg)
                  cx_release_line_(Ptr::from_off(size_t(base_line + sg) << 7, K::node_full),
                                   lg);
               // tail lines freed via their own refcounts — only CL0 remains
               return;
            }
            case K::node_full:
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(p));
               cow_release(rh->term);
               const uint32_t base_line = line_of(p) + 1;
               for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                  fx_release_line_(
                      Ptr::from_off(size_t(base_line + li) << 7, K::node_full));
               return;
            }
            case K::full_x:
            {
               node_fullx* x = static_cast<node_fullx*>(deref_(p));
               cow_release(x->rh.term);
               for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
                  fx_release_line_(x->lines[li]);
               break;
            }
            case K::cx:
            {
               node_cx* x = static_cast<node_cx*>(deref_(p));
               cow_release(x->rh.term);
               for (unsigned sg = 0; sg < (1u << x->lg); ++sg)
                  cx_release_line_(x->seglines[sg], x->lg);
               break;
            }
            case K::value_inline:
            case K::null:
               return;
         }
         const uint32_t nl = uint32_t((bytes + 127) / 128);
         if (nl > 1)  // the head line is freed by the release that ran this hook
            alloc_.hkv_free_run(line_of(p) + 1, nl - 1);
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
         void* L = node_alloc_(cacheline_bytes);
         std::memset(L, 0, cx_maskb(lg));
         return pack_(L, K::node_full /*tag unused on seg lines*/);
      }
      Ptr fx_new_line_()
      {
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

      // Re-own every child slot of a freshly memcpy'd clone: retain, or — at
      // the writer-clone band (SPEC §9) — clone the child forward instead.
      void cow_fix_child_(Ptr* cs)
      {
         const Ptr c = *cs;
         if (c.is_null() || c.tag() == K::value_inline)
            return;
         // value_large children's share count lives in the LARGE arena — route the
         // retain there (cow_retain dispatches by tag). hkv_retain on the raw line
         // index would bump the WRONG arena and leave the large head grain at 1,
         // so the next cow_release would free a still-shared leaf (double-free).
         if (!cow_retain(c))
            *cs = cow_clone_node_(c);
      }

      Ptr cow_clone_node_(Ptr cur)
      {
         const K k = cur.tag();
         if (k == K::c2 || k == K::c4 || k == K::c8)
         {
            // design5 demotion for the segmented routers: 128B CL0 + NSEG line
            // retains; the per-segment branches stay where they are.
            auto* x = static_cast<node_cx*>(node_alloc_(cacheline_bytes));
            auto* rh = static_cast<router_hdr*>(deref_(cur));
            x->rh    = *rh;
            x->lg    = k == K::c2 ? 1 : k == K::c4 ? 2 : 3;
            x->pad0  = 0;
            cow_fix_child_(&x->rh.term);
            const uint32_t base_line = line_of(cur) + 1;  // segs follow CL0
            for (unsigned sg = 0; sg < (1u << x->lg); ++sg)
            {
               const uint8_t* m = static_cast<const uint8_t*>(deref_(cur)) + 128 + sg * 128;
               if (bm_count(m, int(cx_maskb(x->lg))) == 0)
               {  // empty range: no line, no retain — reclaimable immediately
                  x->seglines[sg] = packed_ptr::null();
                  continue;
               }
               x->seglines[sg] =
                   Ptr::from_off(size_t(base_line + sg) << 7, K::node_full);
               cx_own_line_(&x->seglines[sg], x->lg);
            }
            return pack_(x, K::cx);
         }
         if (k == K::node_full)
         {
            // design5 demotion (SPEC §2.7/§9): implicit -> explicit. One 128B
            // CL0 + retains on the 9 direct LINES; the ~256 children are
            // untouched — their references ride the shared lines.
            static_assert(sizeof(node_fullx) <= cacheline_bytes,
                          "full_x CL0 must tile one line at this handle width");
            auto* x = static_cast<node_fullx*>(node_alloc_(cacheline_bytes));
            auto* f = static_cast<node_full*>(deref_(cur));
            x->rh   = f->rh;
            cow_fix_child_(&x->rh.term);
            const uint32_t base_line = line_of(cur) + 1;  // direct lines follow CL0
            for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
            {
               if (f->ranges[li].nset == 0)
               {  // empty range: null line instead of a retained dead cline
                  x->lines[li] = packed_ptr::null();
                  continue;
               }
               x->lines[li] =
                   Ptr::from_off(size_t(base_line + li) << 7, K::node_full);
               fx_own_line_(&x->lines[li]);
            }
            return pack_(x, K::full_x);
         }
         const size_t bytes = cow_node_bytes_(cur);
         // value_large: the clone is another large-arena leaf. Allocate there,
         // memcpy the verbatim leaf, pack with the large (grain) encoding.
         if constexpr (large_ok_)
            if (k == K::value_large)
            {
               K     kt  = K::value_ptr;
               char* dst = leaf_alloc_(bytes, kt);
               assert(kt == K::value_large && "a value_large clone must stay large");
               std::memcpy(dst, deref_(cur), bytes);
               const Ptr np = pack_(dst, K::value_large);
               if constexpr (vh_on)
               {
                  try { vh_copy_(leaf_bytes(static_cast<const char*>(dst))); }
                  catch (...) { leaf_free_(dst, bytes, kt); throw; }
               }
               return np;
            }
         void*        dst   = node_alloc_(bytes);
         std::memcpy(dst, deref_(cur), bytes);
         const Ptr np = pack_(dst, k);
         switch (k)
         {
            case K::value_large:  // handled above (large-arena clone); never here
               __builtin_unreachable();
            case K::value_ptr:
               // duplicated value bytes; the old generation's refcount-zero
               // death balances this
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
               assert(false && "node_full demotes in cow_clone_node_, not here");
               break;
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
            case K::cbucket:  // reserved tag: never produced
            case K::value_inline:
            case K::null:
               break;
         }
         return np;
      }

      Ptr cow_clone_into_(Ptr* slot, Ptr cur)
      {
         const Ptr np = cow_clone_node_(cur);
         *slot        = np;
         cow_release(cur);  // the slot's former reference (handles a racing death)
         return np;
      }

      bool cow_shared_(Ptr p) const noexcept
      {
         if constexpr (large_ok_)
            if (p.tag() == K::value_large)  // share count lives in the large arena
               return alloc_.hkv_large_ref(line_of(p)) >= 2;
         return alloc_.hkv_ref(line_of(p)) >= 2;
      }

      // ── leaves under copy-on-write ────────────────────────────────────────
      // A raw-bytes leaf is never cloned on the way to being replaced or
      // removed (gofractally/psitri#33): cloning copied the whole value only
      // for the caller to free the copy, and on a full store the copy could
      // not be allocated at all, so a remove was refused. The writer reads a
      // shared leaf where it is, builds what replaces it, and drops this
      // tree's one reference.
      static constexpr bool leaf_no_clone_ = cow_enabled && raw_vals;
      // True when this tree holds the only reference to the leaf at p, so its
      // bytes may be overwritten or freed in place.
      bool leaf_unshared_(Ptr p) const noexcept
      {
         if constexpr (cow_enabled)
            return !cow_shared_(p);
         else
            return true;
      }
      // The leaf at p (deref'd L, alloc size bytes) leaves this tree: its bytes
      // die when the leaf was ours alone; a shared leaf loses one reference.
      void leaf_drop_(Ptr p, char* L, size_t bytes, K tag) noexcept
      {
         if constexpr (cow_enabled)
            if (cow_shared_(p))
            {
               cow_release(p);
               return;
            }
         vh_leaf_dies_(L);
         leaf_free_(L, bytes, tag);
      }
      void set_term_ptr_(router_hdr* rh, Ptr leaf) noexcept
      {
         rh->term         = leaf;
         rh->hdr.has_term = 1;
      }

      // Privatize the terminal leaf hanging off a router header (in-place value
      // overwrite mutates it).
      void cow_privatize_term_(router_hdr* rh)
      {
         if constexpr (leaf_no_clone_)
            return;  // the term leaf is dropped by remove(), not edited
         const Ptr t = rh->term;
         if (!t.is_null() && is_leaf(t.tag()) && cow_shared_(t))
            cow_clone_into_(&rh->term, t);
      }

      // Walk the key's path for cow_remove, cloning every shared node top-down.
      // Stops where the descent stops: everything an in-place remove can touch
      // (path routers, the diverging prefix node, the router whose term is
      // dropped) is exclusively owned afterwards. A raw-bytes leaf, at the
      // end of the path or as a router's term, is left as it is.
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
            if constexpr (leaf_no_clone_)
               if (is_leaf(cur.tag()))
                  return;  // terminal reached: remove() drops a reference, never edits it
            if (cow_shared_(cur))
               cur = cow_clone_into_(slot, cur);
            const K k = cur.tag();
            switch (k)
            {
               case K::value_ptr:
               case K::value_large:  // a leaf in either arena: terminal reached and private
               case K::bucket:
               case K::cbucket:  // reserved tags: never produced
                  return;  // terminal reached and private
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
      // Large-arena base (value_large deref): cached like base_, only when the
      // allocator has a second arena. An empty member otherwise — zero cost.
      [[no_unique_address]] std::conditional_t<large_ok_, const std::byte*, no_base_t> large_base_{};
      packed_ptr                      root_  = packed_ptr::null();
      size_t                          count_ = 0;
   };
}  // namespace detail

using detail::bytes_value;
using detail::kv_tree;  // re-export only the public class; impl stays in hkv::detail
}  // namespace start_ref::hkv

// gcart::node — FIX-region radix nodes (design §7), built in the required style:
//   * alloc_size(args)  ↔  constructor(asize_lines, args)   (one contract)
//   * copy-construct-insert: a ctor takes a STABLE source + an edit ticket and
//     builds the result in ONE pass (no temporaries, source never mutated).
// Pointers are sal2 line indices (nptr); a node's type is its first byte, read
// on descent. Three kinds: leaf (full key + value bytes), setlist (sparse,
// SWAR-scanned), node256 (dense, direct-indexed). Path compression via a per-node
// prefix; a key that ends at an inner node is held in `term` (a leaf nptr).
#pragma once
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

#include <sal2/arena.hpp>
#include <ucc/lower_bound.hpp>

namespace gcart::legacy
{
   using nptr                     = sal2::line_idx;
   inline constexpr nptr null_ptr = sal2::null_idx;
   inline constexpr uint32_t LINE = sal2::line_bytes;

   enum class ntype : uint8_t { leaf = 0, setlist = 1, node256 = 2, value_chunk = 3 };

   inline uint32_t lines_for(uint32_t bytes) noexcept { return (bytes + LINE - 1) / LINE; }
   inline uint8_t  type_of(const void* p) noexcept { return *static_cast<const uint8_t*>(p); }
   inline bool     is_leaf(const void* p) noexcept { return type_of(p) == uint8_t(ntype::leaf); }

   // ── value_chunk: a segment of a LARGE value's chain ──────────────────────
   // [hdr 12B][bytes len]. Used when a value won't fit in an inline leaf node
   // (> the FIX-allocator's max_alloc_lines). The chain is owned 1:1 by its
   // leaf — it is reached only through that leaf, so it needs no refcount of its
   // own; when the leaf's ref hits zero the whole chain is freed.
   struct value_chunk
   {
      uint8_t  type;    // ntype::value_chunk
      uint8_t  nlines;  // allocation size in lines
      uint16_t _rsv;
      uint32_t len;     // value bytes carried by this chunk
      nptr     next;    // next chunk, or null_ptr at the tail
      // data follows: bytes[len]

      static constexpr uint32_t hdr         = 12;
      static constexpr uint32_t max_payload = sal2::max_alloc_lines * LINE - hdr;  // ~8180 B
      static uint32_t alloc_size(uint32_t chunk_len) noexcept { return lines_for(hdr + chunk_len); }
      value_chunk(uint32_t nl, std::string_view bytes, nptr nxt) noexcept
      {
         type   = uint8_t(ntype::value_chunk);
         nlines = uint8_t(nl);
         _rsv   = 0;
         len    = uint32_t(bytes.size());
         next   = nxt;
         std::memcpy(data(), bytes.data(), bytes.size());
      }
      char*            data() noexcept { return reinterpret_cast<char*>(this) + hdr; }
      const char*      data() const noexcept { return reinterpret_cast<const char*>(this) + hdr; }
      std::string_view bytes() const noexcept { return {data(), len}; }
      uint32_t         line_count() const noexcept { return nlines; }
   };
   static_assert(sizeof(value_chunk) == 12);

   // ── leaf ─────────────────────────────────────────────────────────────────
   // inline (default): [hdr 12B][key klen][value vlen]
   // large (_rsv & FLAG_LARGE): [hdr 12B][key klen][nptr chunk_head] — vlen is
   //   the FULL value length, the bytes live in a value_chunk chain.
   struct leaf
   {
      uint8_t  type;    // ntype::leaf
      uint8_t  nlines;  // allocation size in lines (for free)
      uint16_t _rsv;    // FLAG_LARGE bit
      uint32_t klen;
      uint32_t vlen;    // value length (inline byte count, or total length if large)

      static constexpr uint32_t hdr        = 12;
      static constexpr uint16_t FLAG_LARGE = 1;

      // inline leaf (value fits in the node)
      static uint32_t alloc_size(std::string_view key, std::string_view val) noexcept
      {
         return lines_for(hdr + uint32_t(key.size()) + uint32_t(val.size()));
      }
      leaf(uint32_t nl, std::string_view key, std::string_view val) noexcept
      {
         type   = uint8_t(ntype::leaf);
         nlines = uint8_t(nl);
         _rsv   = 0;
         klen   = uint32_t(key.size());
         vlen   = uint32_t(val.size());
         std::memcpy(data(), key.data(), key.size());
         std::memcpy(data() + klen, val.data(), val.size());
      }

      // large leaf (value spilled to a value_chunk chain)
      struct large_op { std::string_view key; uint32_t total_vlen; nptr chunk_head; };
      static uint32_t alloc_size(const large_op& o) noexcept
      {
         return lines_for(hdr + uint32_t(o.key.size()) + uint32_t(sizeof(nptr)));
      }
      leaf(uint32_t nl, const large_op& o) noexcept
      {
         type   = uint8_t(ntype::leaf);
         nlines = uint8_t(nl);
         _rsv   = FLAG_LARGE;
         klen   = uint32_t(o.key.size());
         vlen   = o.total_vlen;
         std::memcpy(data(), o.key.data(), o.key.size());
         std::memcpy(data() + klen, &o.chunk_head, sizeof(nptr));
      }

      char*            data() noexcept { return reinterpret_cast<char*>(this) + hdr; }
      const char*      data() const noexcept { return reinterpret_cast<const char*>(this) + hdr; }
      std::string_view key() const noexcept { return {data(), klen}; }
      bool             is_large() const noexcept { return _rsv & FLAG_LARGE; }
      uint32_t         total_vlen() const noexcept { return vlen; }
      // inline value (valid only when !is_large)
      std::string_view value() const noexcept { return {data() + klen, vlen}; }
      nptr             chunk_head() const noexcept
      {
         nptr h;
         std::memcpy(&h, data() + klen, sizeof(nptr));
         return h;
      }
      uint32_t line_count() const noexcept { return nlines; }
   };
   static_assert(sizeof(leaf) == 12);

   // ── inner-node shared header (12 B) ──────────────────────────────────────
   struct inner_hdr
   {
      uint8_t  type;    // setlist / node256
      uint8_t  nlines;  // allocation size in lines
      uint8_t  cap;     // setlist capacity (children); 0 for node256
      uint8_t  n;       // setlist child count; node256 uses n16
      uint16_t plen;    // prefix length
      uint16_t n16;     // node256 child count
      nptr     term;    // leaf for a key ending at this node, or null_ptr
   };
   static_assert(sizeof(inner_hdr) == 12);

   // ── edit tickets (copy-construct-insert) ─────────────────────────────────
   struct ins_branch    { uint8_t byte; nptr child; };   // add a (byte -> child)
   struct rep_branch    { uint8_t byte; nptr child; };   // replace existing byte's child
   struct set_term_op   { nptr term; };                  // set/replace term

   // ════════════════════════ setlist ════════════════════════════════════════
   // layout: [inner_hdr 12B] keys[cap](u8) [pad to 4] children[cap](nptr) prefix[plen]
   struct setlist
   {
      inner_hdr h;

      static uint32_t keys_off() noexcept { return sizeof(inner_hdr); }
      static uint32_t children_off(uint32_t cap) noexcept
      {
         return (sizeof(inner_hdr) + cap + 3u) & ~3u;  // 4-byte align children
      }
      static uint32_t bytes_for(uint32_t cap, uint32_t plen) noexcept
      {
         return children_off(cap) + cap * uint32_t(sizeof(nptr)) + plen;
      }
      // capacity that fits one more child than `n` while staying cacheline-tight.
      static uint32_t pick_cap(uint32_t n, uint32_t plen) noexcept
      {
         uint32_t cap = n;
         while (lines_for(bytes_for(cap, plen)) == lines_for(bytes_for(cap + 1, plen)) &&
                cap < setlist_max)
            ++cap;  // soak up the slack lines into extra capacity (free growth room)
         return cap < setlist_max ? cap : setlist_max;
      }
      static constexpr uint32_t setlist_max = 48;  // upgrade to node256 beyond this

      uint8_t*       keys() noexcept { return reinterpret_cast<uint8_t*>(this) + keys_off(); }
      const uint8_t* keys() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(this) + keys_off();
      }
      nptr*       children() noexcept
      {
         return reinterpret_cast<nptr*>(reinterpret_cast<char*>(this) + children_off(h.cap));
      }
      const nptr* children() const noexcept
      {
         return reinterpret_cast<const nptr*>(reinterpret_cast<const char*>(this) +
                                              children_off(h.cap));
      }
      char* prefix_ptr() noexcept
      {
         return reinterpret_cast<char*>(this) + children_off(h.cap) + h.cap * sizeof(nptr);
      }
      const char* prefix_ptr() const noexcept
      {
         return reinterpret_cast<const char*>(this) + children_off(h.cap) + h.cap * sizeof(nptr);
      }
      std::string_view prefix() const noexcept { return {prefix_ptr(), h.plen}; }
      uint32_t         line_count() const noexcept { return h.nlines; }

      // sorted lower_bound on the (zero-padded) key array — branchless SWAR.
      int find_index(uint8_t byte) const noexcept
      {
         int pos = int(ucc::lower_bound_padded(keys(), h.n, byte));
         return (pos < int(h.n) && keys()[pos] == byte) ? pos : -1;
      }

      // ── ctors (alloc_size ↔ ctor) ─────────────────────────────────────────
      // 1) fresh node with explicit prefix + term, no children (caller fills).
      struct fresh_op { uint32_t cap; std::string_view prefix; nptr term; };
      static uint32_t alloc_size(const fresh_op& o) noexcept
      {
         return lines_for(bytes_for(o.cap, uint32_t(o.prefix.size())));
      }
      setlist(uint32_t nl, const fresh_op& o) noexcept
      {
         init_(nl, o.cap, 0, o.prefix, o.term);
      }

      // 2) copy `src` and SPLICE one new (byte->child) at its sorted position.
      struct clone_insert_op { const setlist* src; ins_branch ins; };
      static uint32_t alloc_size(const clone_insert_op& o) noexcept
      {
         const uint32_t n   = o.src->h.n + 1u;
         const uint32_t cap = pick_cap(n, o.src->h.plen);
         return lines_for(bytes_for(cap, o.src->h.plen));
      }
      setlist(uint32_t nl, const clone_insert_op& o) noexcept
      {
         const setlist* s   = o.src;
         const uint32_t n   = s->h.n + 1u;
         const uint32_t cap = pick_cap(n, s->h.plen);
         init_(nl, cap, n, s->prefix(), s->h.term);
         // ONE pass: emit src keys/children, splice the insert at lower_bound pos.
         uint8_t*    k  = keys();
         nptr*       c  = children();
         const uint8_t* sk = s->keys();
         const nptr*    sc = s->children();
         uint32_t pos = uint32_t(ucc::lower_bound_padded(sk, s->h.n, o.ins.byte));
         uint32_t w = 0;
         for (uint32_t i = 0; i < pos; ++i) { k[w] = sk[i]; c[w] = sc[i]; ++w; }
         k[w] = o.ins.byte; c[w] = o.ins.child; ++w;
         for (uint32_t i = pos; i < s->h.n; ++i) { k[w] = sk[i]; c[w] = sc[i]; ++w; }
      }

      // 3) copy `src`, REPLACING one existing byte's child pointer.
      struct clone_replace_op { const setlist* src; rep_branch rep; };
      static uint32_t alloc_size(const clone_replace_op& o) noexcept
      {
         return lines_for(bytes_for(o.src->h.cap, o.src->h.plen));
      }
      setlist(uint32_t nl, const clone_replace_op& o) noexcept
      {
         const setlist* s = o.src;
         init_(nl, s->h.cap, s->h.n, s->prefix(), s->h.term);
         std::memcpy(keys(), s->keys(), s->h.n);
         std::memcpy(children(), s->children(), s->h.n * sizeof(nptr));
         const int idx = find_index(o.rep.byte);
         children()[idx] = o.rep.child;  // idx >= 0 by contract
      }

      // 4) copy `src` with a new term (key now ends at this node).
      struct clone_term_op { const setlist* src; nptr term; };
      static uint32_t alloc_size(const clone_term_op& o) noexcept
      {
         return lines_for(bytes_for(o.src->h.cap, o.src->h.plen));
      }
      setlist(uint32_t nl, const clone_term_op& o) noexcept
      {
         const setlist* s = o.src;
         init_(nl, s->h.cap, s->h.n, s->prefix(), o.term);
         std::memcpy(keys(), s->keys(), s->h.n);
         std::memcpy(children(), s->children(), s->h.n * sizeof(nptr));
      }

      // 5) copy `src` but DROP the first `drop` prefix bytes (prefix split keeps
      //    the SUFFIX of the prefix); same children/term.
      struct clone_reprefix_op { const setlist* src; uint16_t drop; };
      static uint32_t alloc_size(const clone_reprefix_op& o) noexcept
      {
         return lines_for(bytes_for(o.src->h.cap, uint32_t(o.src->h.plen - o.drop)));
      }
      setlist(uint32_t nl, const clone_reprefix_op& o) noexcept
      {
         const setlist* s = o.src;
         init_(nl, s->h.cap, s->h.n,
               {s->prefix_ptr() + o.drop, uint32_t(s->h.plen - o.drop)}, s->h.term);
         std::memcpy(keys(), s->keys(), s->h.n);
         std::memcpy(children(), s->children(), s->h.n * sizeof(nptr));
      }

      // ── in-place edits (unique mode; caller guarantees ref==1 + capacity) ──
      bool can_insert() const noexcept { return h.n < h.cap; }
      void insert_inplace(uint8_t byte, nptr child) noexcept
      {
         uint8_t* k = keys();
         nptr*    c = children();
         uint32_t pos = uint32_t(ucc::lower_bound_padded(k, h.n, byte));
         for (uint32_t i = h.n; i > pos; --i) { k[i] = k[i - 1]; c[i] = c[i - 1]; }
         k[pos] = byte; c[pos] = child;
         ++h.n;
      }
      void replace_inplace(int idx, nptr child) noexcept { children()[idx] = child; }

     private:
      void init_(uint32_t nl, uint32_t cap, uint32_t n, std::string_view pfx, nptr term) noexcept
      {
         h.type   = uint8_t(ntype::setlist);
         h.nlines = uint8_t(nl);
         h.cap    = uint8_t(cap);
         h.n      = uint8_t(n);
         h.plen   = uint16_t(pfx.size());
         h.n16    = 0;
         h.term   = term;
         // zero key tail so the padded SWAR scan reads deterministic bytes.
         std::memset(keys(), 0, cap);
         if (!pfx.empty())
            std::memcpy(prefix_ptr(), pfx.data(), pfx.size());
      }
   };

   // ════════════════════════ node256 ════════════════════════════════════════
   // layout: [inner_hdr 12B] children[256](nptr) prefix[plen]
   struct node256
   {
      inner_hdr h;

      static constexpr uint32_t children_off = sizeof(inner_hdr);
      static constexpr uint32_t fixed        = sizeof(inner_hdr) + 256 * sizeof(nptr);
      static uint32_t bytes_for(uint32_t plen) noexcept { return fixed + plen; }

      nptr* children() noexcept
      {
         return reinterpret_cast<nptr*>(reinterpret_cast<char*>(this) + children_off);
      }
      const nptr* children() const noexcept
      {
         return reinterpret_cast<const nptr*>(reinterpret_cast<const char*>(this) + children_off);
      }
      char*       prefix_ptr() noexcept { return reinterpret_cast<char*>(this) + fixed; }
      const char* prefix_ptr() const noexcept { return reinterpret_cast<const char*>(this) + fixed; }
      std::string_view prefix() const noexcept { return {prefix_ptr(), h.plen}; }
      uint32_t         line_count() const noexcept { return h.nlines; }
      nptr             child(uint8_t byte) const noexcept { return children()[byte]; }

      // promote a full setlist to node256, splicing one new (byte->child).
      struct from_setlist_op { const setlist* src; ins_branch ins; };
      static uint32_t alloc_size(const from_setlist_op& o) noexcept
      {
         return lines_for(bytes_for(o.src->h.plen));
      }
      node256(uint32_t nl, const from_setlist_op& o) noexcept
      {
         const setlist* s = o.src;
         init_(nl, s->prefix(), s->h.term);
         const uint8_t* sk = s->keys();
         const nptr*    sc = s->children();
         for (uint32_t i = 0; i < s->h.n; ++i) { children()[sk[i]] = sc[i]; h.n16++; }
         children()[o.ins.byte] = o.ins.child; h.n16++;
      }

      // copy + replace one child pointer.
      struct clone_replace_op { const node256* src; rep_branch rep; };
      static uint32_t alloc_size(const clone_replace_op& o) noexcept
      {
         return lines_for(bytes_for(o.src->h.plen));
      }
      node256(uint32_t nl, const clone_replace_op& o) noexcept
      {
         const node256* s = o.src;
         init_(nl, s->prefix(), s->h.term);
         std::memcpy(children(), s->children(), 256 * sizeof(nptr));
         h.n16 = s->h.n16;
         children()[o.rep.byte] = o.rep.child;
      }

      // copy + insert one new child (byte was absent).
      struct clone_insert_op { const node256* src; ins_branch ins; };
      static uint32_t alloc_size(const clone_insert_op& o) noexcept
      {
         return lines_for(bytes_for(o.src->h.plen));
      }
      node256(uint32_t nl, const clone_insert_op& o) noexcept
      {
         const node256* s = o.src;
         init_(nl, s->prefix(), s->h.term);
         std::memcpy(children(), s->children(), 256 * sizeof(nptr));
         h.n16 = uint16_t(s->h.n16 + 1);
         children()[o.ins.byte] = o.ins.child;
      }

      // copy + new term.
      struct clone_term_op { const node256* src; nptr term; };
      static uint32_t alloc_size(const clone_term_op& o) noexcept
      {
         return lines_for(bytes_for(o.src->h.plen));
      }
      node256(uint32_t nl, const clone_term_op& o) noexcept
      {
         const node256* s = o.src;
         init_(nl, s->prefix(), o.term);
         std::memcpy(children(), s->children(), 256 * sizeof(nptr));
         h.n16 = s->h.n16;
      }

      // DROP the first `drop` prefix bytes (prefix split keeps the SUFFIX).
      struct clone_reprefix_op { const node256* src; uint16_t drop; };
      static uint32_t alloc_size(const clone_reprefix_op& o) noexcept
      {
         return lines_for(bytes_for(uint32_t(o.src->h.plen - o.drop)));
      }
      node256(uint32_t nl, const clone_reprefix_op& o) noexcept
      {
         const node256* s = o.src;
         init_(nl, {s->prefix_ptr() + o.drop, uint32_t(s->h.plen - o.drop)}, s->h.term);
         std::memcpy(children(), s->children(), 256 * sizeof(nptr));
         h.n16 = s->h.n16;
      }

      void insert_inplace(uint8_t byte, nptr child) noexcept
      {
         children()[byte] = child;
         ++h.n16;
      }
      void replace_inplace(uint8_t byte, nptr child) noexcept { children()[byte] = child; }

     private:
      void init_(uint32_t nl, std::string_view pfx, nptr term) noexcept
      {
         h.type   = uint8_t(ntype::node256);
         h.nlines = uint8_t(nl);
         h.cap    = 0;
         h.n      = 0;
         h.plen   = uint16_t(pfx.size());
         h.n16    = 0;
         h.term   = term;
         std::memset(children(), 0xFF, 256 * sizeof(nptr));  // null_ptr fill
         if (!pfx.empty())
            std::memcpy(prefix_ptr(), pfx.data(), pfx.size());
      }
   };
}  // namespace gcart::legacy

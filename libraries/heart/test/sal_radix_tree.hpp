#pragma once
// SAL-backed unique-mode radix tree — fifth prototype: RADIX ROUTERS + FLAT LEAVES.
//
// Pure radix forces a node at every byte where keys fork, so a prefix-heavy long
// tail (a dictionary) builds deep, cache-unfriendly chains. psitri avoids this
// with a B-tree-style leaf catch-all. heart does the same here:
//
//   * leaf_node — a FLAT catch-all holding up to CAP (suffix,value) pairs, each
//     discriminated by a u32 `stem` (first 4 bytes, a fast pre-filter) plus a
//     full-suffix compare. Distinct stems => path compression for free; stem
//     collisions just fall through to the suffix compare. NO child pointers.
//   * routers (setlist/bitset/full, the ART N16/N48/N256 tiers) — branch on one
//     byte to child nodes (leaves or routers), plus a term value. Built only when
//     a leaf overflows.
//
// A leaf that exceeds CAP SPLITS via build(): find the suffixes' common prefix,
// branch on the first differing byte, and recurse — producing a router over leaf
// children. Because leaves hold full suffixes (no pointers), the split rebuilds
// purely from stored bytes. Routers grow their fanout via the setlist->bitset->
// full upgrade ladder. Sibling-locality alloc hints (peer branch column) as before.
//
// Single writer, unique mode (ref==1), single version. Suffix+value must fit the
// 64-byte inline cap (big-value promotion: TODO).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>

#include <sal/allocator.hpp>
#include <sal/allocator_session_impl.hpp>
#include <ucc/round.hpp>

#include <ucc/lower_bound.hpp>  // ucc::find_u32x16 (SIMD)

#include <heart/bitset.hpp>
#include <heart/inline_value.hpp>
#include <heart/radix_full.hpp>
#include <heart/smart_address.hpp>

namespace heart::sal_proto
{
   using sal::ptr_address;
   using sal::ptr_address_seq;
   using bytes = std::span<const uint8_t>;

   enum class rtype : uint8_t
   {
      leaf    = uint8_t(sal::header_type::start_user_type),      // 3  flat catch-all
      setlist = uint8_t(sal::header_type::start_user_type) + 1,  // 4  router <=16
      bitset  = uint8_t(sal::header_type::start_user_type) + 2,  // 5  router 17..48
      full    = uint8_t(sal::header_type::start_user_type) + 3,  // 6  router 256
   };

   // ── shared paged-arena base: inline-value heap grows down from tail() ─────
   struct paged_node : sal::alloc_header
   {
      uint16_t      heap_used;
      uint16_t      nprefix;
      smart_address term_value;  // null | tail inline_value (key ends at this node's prefix)

      paged_node(uint32_t asize, sal::header_type t, ptr_address_seq seq) noexcept
          : sal::alloc_header(asize, t, seq)
      {
      }

      char*       tail_top() noexcept { return tail() - heap_used; }
      const char* tail_top() const noexcept { return tail() - heap_used; }
      uint32_t    free_after(const char* head_end) const noexcept
      {
         return uint32_t(tail_top() - head_end);
      }
      char* alloc_tail(uint16_t n) noexcept
      {
         heap_used += n;
         return tail_top();
      }
      inline_value* resolve_inline(smart_address sa) noexcept
      {
         return reinterpret_cast<inline_value*>(tail() - sa.tail_offset());
      }
      const inline_value* resolve_inline(smart_address sa) const noexcept
      {
         return reinterpret_cast<const inline_value*>(tail() - sa.tail_offset());
      }
      void blit_head_tail(alloc_header* dst, uint32_t hsize) const noexcept
      {
         alloc_header saved = *dst;
         std::memcpy(dst, this, hsize);
         std::memcpy(dst, &saved, sizeof(alloc_header));
         auto* out = static_cast<paged_node*>(dst);
         std::memcpy(out->tail() - heap_used, tail() - heap_used, heap_used);
         out->clear_checksum();
      }
   };
   static_assert(sizeof(paged_node) == 24);

   // ── flat catch-all leaf (u32 stem + full-suffix inline) ───────────────────
   struct leaf_node : paged_node
   {
      static constexpr sal::header_type type_id  = sal::header_type(rtype::leaf);
      static constexpr int              CAP      = 32;  // SIMD-scanned (2x find_u32x16)
      static constexpr uint32_t         floor    = 256;
      static constexpr uint32_t         headroom = 64;
      struct copy_tag
      {
      };

      uint16_t      count;       // @24
      uint32_t      stems[CAP];  // @28  first-4-byte fast pre-filter per entry
      smart_address vals[CAP];   // @92  → inline_value{suffix, value}
      // prefix bytes at +sizeof(leaf_node).

      uint8_t*       prefix() noexcept { return reinterpret_cast<uint8_t*>(this) + sizeof(leaf_node); }
      const uint8_t* prefix() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(this) + sizeof(leaf_node);
      }
      uint32_t head_size() const noexcept { return uint32_t(sizeof(leaf_node) + nprefix); }
      uint32_t free_bytes() const noexcept
      {
         return free_after(reinterpret_cast<const char*>(this) + head_size());
      }

      // First 4 bytes packed big-endian (short keys left-justified, low bytes 0).
      // Big-endian packing makes u32 compare == lexicographic on the first 4 bytes,
      // so a suffix-sorted entry list has ASCENDING stems and equal-stem runs are
      // contiguous — which is what makes both the SIMD find and lower_bound work.
      static uint32_t stem4(bytes s) noexcept
      {
         uint32_t x = 0;
         int      n = int(std::min<size_t>(4, s.size()));
         for (int i = 0; i < n; ++i)
            x = (x << 8) | s[size_t(i)];
         x <<= 8 * (4 - n);
         return x;
      }
      // Lexicographic suffix order (the leaf's sort key).
      static int cmp(bytes a, bytes b) noexcept
      {
         size_t m = std::min(a.size(), b.size());
         int    c = m ? std::memcmp(a.data(), b.data(), m) : 0;
         if (c)
            return c;
         return a.size() < b.size() ? -1 : a.size() > b.size() ? 1 : 0;
      }
      // SIMD point find (ucc::find_u32x16, chunked for CAP>16): locate the stem,
      // then walk the contiguous equal-stem run confirming the full suffix.
      int find_idx(bytes rem) const noexcept
      {
         uint32_t q     = stem4(rem);
         int      first = count;
         for (int base = 0; base < count; base += 16)
         {
            int chunk = std::min(16, count - base);
            int r     = ucc::find_u32x16(stems + base, size_t(chunk), q);
            if (r < chunk)
            {
               first = base + r;
               break;
            }
         }
         for (int i = first; i < count && stems[i] == q; ++i)
         {
            const inline_value* iv = resolve_inline(vals[i]);
            if (iv->suffix_len == rem.size() &&
                std::memcmp(iv->data, rem.data(), iv->suffix_len) == 0)
               return i;
         }
         return -1;
      }
      // First entry index whose suffix is >= rem (strict: > rem). Sorted scan.
      int lower_bound_idx(bytes rem, bool strict) const noexcept
      {
         int i = 0;
         while (i < count)
         {
            const inline_value* iv = resolve_inline(vals[i]);
            int                 c  = cmp({iv->data, iv->suffix_len}, rem);
            if (c > 0 || (c == 0 && !strict))
               break;
            ++i;
         }
         return i;
      }

      void visit_children(const std::function<void(ptr_address)>&) const {}  // no node children
      sal::alloc_hint branch_hint() const noexcept { return {}; }            // not a hint source

      static uint32_t page_size_for(uint32_t need) noexcept
      {
         return std::max(floor, ucc::round_up_multiple<sal::cacheline_size>(need));
      }
      static uint32_t alloc_size(bytes pfx) noexcept
      {
         return page_size_for(uint32_t(sizeof(leaf_node) + pfx.size()));
      }
      leaf_node(uint32_t asize, ptr_address_seq seq, bytes pfx) noexcept
          : paged_node(asize, type_id, seq)
      {
         heap_used  = 0;
         nprefix    = uint16_t(pfx.size());
         term_value = smart_address::null();
         count      = 0;
         for (int i = 0; i < CAP; ++i)
            stems[i] = 0xFFFFFFFFu;  // sentinel: SIMD find reads the full lane width
         std::memcpy(prefix(), pfx.data(), pfx.size());
      }
      leaf_node(uint32_t asize, ptr_address_seq seq, copy_tag, const leaf_node* src) noexcept
          : paged_node(asize, type_id, seq)
      {
         src->blit_head_tail(this, src->head_size());
      }
      uint32_t cow_size() const noexcept { return page_size_for(head_size() + heap_used + headroom); }
      uint32_t compact_size() const noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(head_size() + heap_used);
      }
      void copy_to(alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
      void compact_to(alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
   };
   static_assert(sizeof(leaf_node) == 24 + 4 + leaf_node::CAP * 4 + leaf_node::CAP * 4,
                 "24 + count(+pad) + stems[CAP] + vals[CAP]");

   // ── router tier 1: small fanout (ART Node16) ──────────────────────────────
   struct setlist_node : paged_node
   {
      static constexpr sal::header_type type_id  = sal::header_type(rtype::setlist);
      static constexpr int              CAP      = 16;
      static constexpr uint32_t         floor    = 256;
      static constexpr uint32_t         headroom = 64;
      struct copy_tag
      {
      };
      struct upgrade_tag
      {
      };

      uint8_t       count;
      uint8_t       keys[CAP];
      smart_address vals[CAP];

      uint8_t*       prefix() noexcept { return reinterpret_cast<uint8_t*>(this) + sizeof(setlist_node); }
      const uint8_t* prefix() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(this) + sizeof(setlist_node);
      }
      uint32_t head_size() const noexcept { return uint32_t(sizeof(setlist_node) + nprefix); }
      uint32_t free_bytes() const noexcept
      {
         return free_after(reinterpret_cast<const char*>(this) + head_size());
      }
      smart_address find(uint8_t b) const noexcept
      {
         for (int i = 0; i < count; ++i)
            if (keys[i] == b)
               return vals[i];
         return smart_address::null();
      }
      bool has_branch(uint8_t b) const noexcept
      {
         for (int i = 0; i < count; ++i)
            if (keys[i] == b)
               return true;
         return false;
      }
      bool branch_has_room(uint8_t b) const noexcept { return count < CAP || has_branch(b); }
      void set_branch(uint8_t b, smart_address sa) noexcept
      {
         for (int i = 0; i < count; ++i)
            if (keys[i] == b)
            {
               vals[i] = sa;
               return;
            }
         assert(count < CAP);
         keys[count] = b;
         vals[count] = sa;
         ++count;
      }
      void visit_children(const std::function<void(ptr_address)>& visit) const
      {
         if (term_value.is_pointer())
            visit(term_value.as_pointer());
         for (int i = 0; i < count; ++i)
            if (vals[i].is_pointer())
               visit(vals[i].as_pointer());
      }
      sal::alloc_hint branch_hint() const noexcept
      {
         return {reinterpret_cast<const ptr_address*>(vals), size_t(count)};
      }
      static uint32_t page_size_for(uint32_t need) noexcept
      {
         return std::max(floor, ucc::round_up_multiple<sal::cacheline_size>(need));
      }
      static uint32_t alloc_size(bytes pfx) noexcept
      {
         return page_size_for(uint32_t(sizeof(setlist_node) + pfx.size()));
      }
      setlist_node(uint32_t asize, ptr_address_seq seq, bytes pfx) noexcept
          : paged_node(asize, type_id, seq)
      {
         heap_used  = 0;
         nprefix    = uint16_t(pfx.size());
         term_value = smart_address::null();
         count      = 0;
         std::memcpy(prefix(), pfx.data(), pfx.size());
      }
      setlist_node(uint32_t asize, ptr_address_seq seq, copy_tag, const setlist_node* src) noexcept
          : paged_node(asize, type_id, seq)
      {
         src->blit_head_tail(this, src->head_size());
      }
      uint32_t cow_size() const noexcept { return page_size_for(head_size() + heap_used + headroom); }
      uint32_t compact_size() const noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(head_size() + heap_used);
      }
      void copy_to(alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
      void compact_to(alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
   };
   static_assert(sizeof(setlist_node) == 24 + 1 + 16 + 3 + 64);

   // ── router tier 2: bitset (ART N48, 17..48) ───────────────────────────────
   struct bitset_node : paged_node
   {
      static constexpr sal::header_type type_id  = sal::header_type(rtype::bitset);
      static constexpr int              CAP      = 48;
      static constexpr uint32_t         floor    = 512;
      static constexpr uint32_t         headroom = 128;
      struct copy_tag
      {
      };
      struct upgrade_tag
      {
      };

      heart::bitset<256> present;
      uint16_t           count;
      smart_address      vals[CAP];

      uint8_t*       prefix() noexcept { return reinterpret_cast<uint8_t*>(this) + sizeof(bitset_node); }
      const uint8_t* prefix() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(this) + sizeof(bitset_node);
      }
      uint32_t head_size() const noexcept { return uint32_t(sizeof(bitset_node) + nprefix); }
      uint32_t free_bytes() const noexcept
      {
         return free_after(reinterpret_cast<const char*>(this) + head_size());
      }
      smart_address find(uint8_t b) const noexcept
      {
         if (!present.test(b))
            return smart_address::null();
         return vals[present.rank(b)];
      }
      bool branch_has_room(uint8_t b) const noexcept { return present.test(b) || count < CAP; }
      void set_branch(uint8_t b, smart_address sa) noexcept
      {
         int r = present.rank(b);
         if (present.test(b))
         {
            vals[r] = sa;
            return;
         }
         assert(count < CAP);
         std::memmove(vals + r + 1, vals + r, size_t(count - r) * sizeof(smart_address));
         vals[r] = sa;
         present.set(b);
         ++count;
      }
      void visit_children(const std::function<void(ptr_address)>& visit) const
      {
         if (term_value.is_pointer())
            visit(term_value.as_pointer());
         for (int i = 0; i < count; ++i)
            if (vals[i].is_pointer())
               visit(vals[i].as_pointer());
      }
      sal::alloc_hint branch_hint() const noexcept
      {
         return {reinterpret_cast<const ptr_address*>(vals), size_t(count)};
      }
      static uint32_t page_size_for(uint32_t need) noexcept
      {
         return std::max(floor, ucc::round_up_multiple<sal::cacheline_size>(need));
      }
      static uint32_t alloc_size(bytes pfx) noexcept
      {
         return page_size_for(uint32_t(sizeof(bitset_node) + pfx.size()));
      }
      bitset_node(uint32_t asize, ptr_address_seq seq, bytes pfx) noexcept
          : paged_node(asize, type_id, seq)
      {
         heap_used  = 0;
         nprefix    = uint16_t(pfx.size());
         term_value = smart_address::null();
         present    = heart::bitset<256>{};
         count      = 0;
         std::memcpy(prefix(), pfx.data(), pfx.size());
      }
      bitset_node(uint32_t asize, ptr_address_seq seq, copy_tag, const bitset_node* src) noexcept
          : paged_node(asize, type_id, seq)
      {
         src->blit_head_tail(this, src->head_size());
      }
      bitset_node(uint32_t asize, ptr_address_seq seq, upgrade_tag, const setlist_node* src) noexcept
          : paged_node(asize, type_id, seq)
      {
         heap_used  = src->heap_used;
         nprefix    = src->nprefix;
         term_value = src->term_value;
         present    = heart::bitset<256>{};
         count      = 0;
         for (int i = 0; i < src->count; ++i)
            set_branch(src->keys[i], src->vals[i]);
         std::memcpy(prefix(), src->prefix(), nprefix);
         std::memcpy(tail() - heap_used, src->tail() - heap_used, heap_used);
         clear_checksum();
      }
      uint32_t cow_size() const noexcept { return page_size_for(head_size() + heap_used + headroom); }
      uint32_t compact_size() const noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(head_size() + heap_used);
      }
      void copy_to(alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
      void compact_to(alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
   };
   static_assert(sizeof(bitset_node) == 256);

   // ── router tier 3: dense full radix (ART Node256) ─────────────────────────
   struct full_node : paged_node
   {
      static constexpr sal::header_type type_id  = sal::header_type(rtype::full);
      static constexpr uint32_t         headroom = 1024;
      struct copy_tag
      {
      };
      struct upgrade_tag
      {
      };

      radix_full<256> children;

      uint8_t*       prefix() noexcept { return reinterpret_cast<uint8_t*>(this) + sizeof(full_node); }
      const uint8_t* prefix() const noexcept
      {
         return reinterpret_cast<const uint8_t*>(this) + sizeof(full_node);
      }
      uint32_t head_size() const noexcept { return uint32_t(sizeof(full_node) + nprefix); }
      uint32_t free_bytes() const noexcept
      {
         return free_after(reinterpret_cast<const char*>(this) + head_size());
      }
      smart_address find(uint8_t b) const noexcept { return children.find(b); }
      bool          branch_has_room(uint8_t) const noexcept { return true; }
      void          set_branch(uint8_t b, smart_address sa) noexcept { children.set_branch(b, sa); }
      void visit_children(const std::function<void(ptr_address)>& visit) const
      {
         if (term_value.is_pointer())
            visit(term_value.as_pointer());
         for (int b = 0; b < 256; ++b)
            if (children.branches[b].is_pointer())
               visit(children.branches[b].as_pointer());
      }
      sal::alloc_hint branch_hint() const noexcept
      {
         return {reinterpret_cast<const ptr_address*>(&children.branches[0]), 256};
      }
      static uint32_t page_size_for(uint32_t need) noexcept
      {
         return ucc::round_up_multiple<uint32_t(1024)>(need);
      }
      static uint32_t alloc_size(bytes pfx) noexcept
      {
         return page_size_for(uint32_t(sizeof(full_node) + pfx.size()));
      }
      full_node(uint32_t asize, ptr_address_seq seq, bytes pfx) noexcept
          : paged_node(asize, type_id, seq)
      {
         heap_used  = 0;
         nprefix    = uint16_t(pfx.size());
         term_value = smart_address::null();
         children.init();
         std::memcpy(prefix(), pfx.data(), pfx.size());
      }
      full_node(uint32_t asize, ptr_address_seq seq, copy_tag, const full_node* src) noexcept
          : paged_node(asize, type_id, seq)
      {
         src->blit_head_tail(this, src->head_size());
      }
      full_node(uint32_t asize, ptr_address_seq seq, upgrade_tag, const bitset_node* src) noexcept
          : paged_node(asize, type_id, seq)
      {
         heap_used  = src->heap_used;
         nprefix    = src->nprefix;
         term_value = src->term_value;
         children.init();
         int r = 0;
         for (int b = src->present.next_set(0); b >= 0; b = src->present.next_set(b + 1))
            children.set_branch(b, src->vals[r++]);
         std::memcpy(prefix(), src->prefix(), nprefix);
         std::memcpy(tail() - heap_used, src->tail() - heap_used, heap_used);
         clear_checksum();
      }
      uint32_t cow_size() const noexcept { return page_size_for(head_size() + heap_used + headroom); }
      uint32_t compact_size() const noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(head_size() + heap_used);
      }
      void copy_to(alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
      void compact_to(alloc_header* dst) const noexcept { blit_head_tail(dst, head_size()); }
   };
   static_assert(sizeof(full_node) == 24 + 1024);

   class sal_radix_tree
   {
     public:
      explicit sal_radix_tree(const std::filesystem::path& dir)
          : _alloc(make_alloc(dir)), _ses(_alloc->get_session())
      {
         _alloc->register_type_ops<leaf_node>();
         _alloc->register_type_ops<setlist_node>();
         _alloc->register_type_ops<bitset_node>();
         _alloc->register_type_ops<full_node>();
         _root = _ses->alloc<leaf_node>(bytes{});  // tree starts as one empty leaf
      }

      uint64_t object_count() const noexcept { return _alloc->total_allocated_objects(); }

      struct tier_counts
      {
         uint64_t leaf = 0, setlist = 0, bitset = 0, full = 0;
      };
      tier_counts count_tiers()
      {
         auto        lock = _ses->lock();
         tier_counts tc;
         walk_tiers(_root, tc);
         return tc;
      }
      size_t bytes_used()
      {
         auto lock = _ses->lock();
         return walk_bytes(_root, false);
      }
      size_t bytes_compacted()
      {
         auto lock = _ses->lock();
         return walk_bytes(_root, true);
      }

      void insert(bytes key, uint64_t value)
      {
         auto  lock = _ses->lock();
         bytes vspan(reinterpret_cast<const uint8_t*>(&value), sizeof(value));
         _root = insert_node(_root, key, 0, vspan, {});
      }

      bool find(bytes key, uint64_t& out)
      {
         auto        lock  = _ses->lock();
         ptr_address addr  = _root;
         uint32_t    depth = 0;
         for (;;)
         {
            auto ref = _ses->get_ref<sal::alloc_header>(addr);
            auto t   = ref->type();
            if (t == leaf_node::type_id)
               return leaf_find(ref.as<leaf_node>().obj(), key, depth, out);
            bool        done, found;
            ptr_address next = ptr_address(0);
            if (t == setlist_node::type_id)
               found = route_step(ref.as<setlist_node>().obj(), key, depth, out, done, next);
            else if (t == bitset_node::type_id)
               found = route_step(ref.as<bitset_node>().obj(), key, depth, out, done, next);
            else
               found = route_step(ref.as<full_node>().obj(), key, depth, out, done, next);
            if (done)
               return found;
            addr = next;
         }
      }

      // Smallest stored key >= key (lower_bound) / > key (upper_bound). Returns
      // that key's value, or false if none. The whole tree is ordered: leaf
      // entries are sorted, routers iterate branches in byte order, and a node's
      // term (key ending at its prefix) precedes its branches.
      bool lower_bound(bytes key, uint64_t& out)
      {
         auto lk = _ses->lock();
         return lb(_root, key, 0, false, out);
      }
      bool upper_bound(bytes key, uint64_t& out)
      {
         auto lk = _ses->lock();
         return lb(_root, key, 0, true, out);
      }

     private:
      static std::unique_ptr<sal::allocator> make_alloc(const std::filesystem::path& dir)
      {
         std::filesystem::remove_all(dir);
         std::filesystem::create_directories(dir);
         sal::runtime_config cfg;
         cfg.sync_mode               = sal::sync_type::none;
         cfg.write_protect_on_commit = false;
         cfg.enable_read_cache       = false;
         return std::make_unique<sal::allocator>(dir, cfg, true);
      }

      // ── ordered-query dispatch helpers (term/nprefix/heap are common base
      //    fields; prefix()/find/branch-iteration differ per tier) ────────────
      sal::header_type node_type(ptr_address a) { return _ses->get_ref<sal::alloc_header>(a)->type(); }
      smart_address    node_term(ptr_address a)
      {
         return reinterpret_cast<const paged_node*>(_ses->get_ref<sal::alloc_header>(a).obj())
             ->term_value;
      }
      const inline_value* node_resolve(ptr_address a, smart_address sa)
      {
         return reinterpret_cast<const paged_node*>(_ses->get_ref<sal::alloc_header>(a).obj())
             ->resolve_inline(sa);
      }
      bytes node_prefix(ptr_address a)
      {
         auto     r  = _ses->get_ref<sal::alloc_header>(a);
         auto     t  = r->type();
         uint16_t np = reinterpret_cast<const paged_node*>(r.obj())->nprefix;
         const uint8_t* p = (t == leaf_node::type_id)      ? r.as<leaf_node>().obj()->prefix()
                            : (t == setlist_node::type_id) ? r.as<setlist_node>().obj()->prefix()
                            : (t == bitset_node::type_id)  ? r.as<bitset_node>().obj()->prefix()
                                                           : r.as<full_node>().obj()->prefix();
         return {p, np};
      }
      smart_address router_find(ptr_address a, uint8_t b)  // routers only
      {
         auto r = _ses->get_ref<sal::alloc_header>(a);
         auto t = r->type();
         if (t == setlist_node::type_id)
            return r.as<setlist_node>().obj()->find(b);
         if (t == bitset_node::type_id)
            return r.as<bitset_node>().obj()->find(b);
         return r.as<full_node>().obj()->find(b);
      }
      int router_min_byte_ge(ptr_address a, int b)  // smallest branch byte >= b, or -1
      {
         if (b < 0)
            b = 0;
         if (b > 255)
            return -1;
         auto r = _ses->get_ref<sal::alloc_header>(a);
         auto t = r->type();
         if (t == setlist_node::type_id)
         {
            auto* n    = r.as<setlist_node>().obj();
            int   best = 256;
            for (int i = 0; i < n->count; ++i)
               if (n->keys[i] >= b && n->keys[i] < best)
                  best = n->keys[i];
            return best < 256 ? best : -1;
         }
         if (t == bitset_node::type_id)
            return r.as<bitset_node>().obj()->present.next_set(b);
         auto* n = r.as<full_node>().obj();
         for (int i = b; i < 256; ++i)
            if (!n->children.branches[i].is_null())
               return i;
         return -1;
      }

      // Smallest key in the subtree at `addr`.
      bool node_smallest(ptr_address a, uint64_t& out)
      {
         smart_address term = node_term(a);
         if (!term.is_null())  // a key ending at this prefix is the smallest
         {
            read_value(node_resolve(a, term), out);
            return true;
         }
         if (node_type(a) == leaf_node::type_id)
         {
            auto* L = _ses->get_ref<leaf_node>(a).obj();
            if (L->count == 0)
               return false;
            read_value(L->resolve_inline(L->vals[0]), out);  // sorted → vals[0] smallest
            return true;
         }
         int bb = router_min_byte_ge(a, 0);
         if (bb < 0)
            return false;
         smart_address sa = router_find(a, uint8_t(bb));
         if (sa.is_tail())
         {
            read_value(node_resolve(a, sa), out);
            return true;
         }
         return node_smallest(sa.as_pointer(), out);
      }

      // First key >= query (strict: > query) in the subtree at `addr`.
      bool lb(ptr_address a, bytes query, uint32_t depth, bool strict, uint64_t& out)
      {
         bytes  pfx  = node_prefix(a);
         bytes  qrem = query.subspan(depth);
         size_t m    = std::min(pfx.size(), qrem.size());
         int    c    = m ? std::memcmp(pfx.data(), qrem.data(), m) : 0;
         if (c > 0)
            return node_smallest(a, out);  // whole subtree > query
         if (c < 0)
            return false;  // whole subtree < query
         if (pfx.size() > qrem.size())
            return node_smallest(a, out);  // query is a proper prefix → all keys > query
         depth += uint32_t(pfx.size());
         qrem = query.subspan(depth);
         auto t = node_type(a);

         if (qrem.empty())  // query ends exactly at this node
         {
            smart_address term = node_term(a);
            if (!strict && !term.is_null())
            {
               read_value(node_resolve(a, term), out);
               return true;
            }
            if (t == leaf_node::type_id)
            {
               auto* L = _ses->get_ref<leaf_node>(a).obj();
               if (L->count == 0)
                  return false;
               read_value(L->resolve_inline(L->vals[0]), out);
               return true;
            }
            int bb = router_min_byte_ge(a, 0);
            if (bb < 0)
               return false;
            smart_address sa = router_find(a, uint8_t(bb));
            if (sa.is_tail())
            {
               read_value(node_resolve(a, sa), out);
               return true;
            }
            return node_smallest(sa.as_pointer(), out);
         }

         uint8_t b = qrem[0];
         if (t == leaf_node::type_id)
         {
            auto* L   = _ses->get_ref<leaf_node>(a).obj();
            int   idx = L->lower_bound_idx(qrem, strict);
            if (idx < L->count)
            {
               read_value(L->resolve_inline(L->vals[idx]), out);
               return true;
            }
            return false;
         }
         // router: try branch b, then fall to the smallest branch byte > b.
         smart_address sa = router_find(a, b);
         if (!sa.is_null())
         {
            if (sa.is_tail())
            {
               const inline_value* iv = node_resolve(a, sa);
               int c2 = leaf_node::cmp({iv->data, iv->suffix_len}, qrem.subspan(1));  // b==qrem[0]
               if (c2 > 0 || (c2 == 0 && !strict))
               {
                  read_value(iv, out);
                  return true;
               }
            }
            else if (lb(sa.as_pointer(), query, depth + 1, strict, out))
               return true;
         }
         int bb = router_min_byte_ge(a, b + 1);
         if (bb < 0)
            return false;
         smart_address sa2 = router_find(a, uint8_t(bb));
         if (sa2.is_tail())
         {
            read_value(node_resolve(a, sa2), out);
            return true;
         }
         return node_smallest(sa2.as_pointer(), out);
      }

      static void read_value(const inline_value* iv, uint64_t& out) noexcept
      {
         out = 0;
         std::memcpy(&out, iv->data + iv->suffix_len, iv->value_len);
      }

      template <class N>
      N* mut(ptr_address a)
      {
         return _ses->get_ref<N>(a).modify().get();
      }

      // ── find ──────────────────────────────────────────────────────────────
      bool leaf_find(const leaf_node* n, bytes key, uint32_t depth, uint64_t& out) const
      {
         if (n->nprefix > key.size() - depth)
            return false;
         if (n->nprefix && std::memcmp(n->prefix(), key.data() + depth, n->nprefix) != 0)
            return false;
         depth += n->nprefix;
         bytes rem = key.subspan(depth);
         if (rem.empty())
         {
            if (n->term_value.is_null())
               return false;
            read_value(n->resolve_inline(n->term_value), out);
            return true;
         }
         int idx = n->find_idx(rem);
         if (idx < 0)
            return false;
         read_value(n->resolve_inline(n->vals[idx]), out);
         return true;
      }

      template <class N>
      bool route_step(const N* n, bytes key, uint32_t& depth, uint64_t& out, bool& done,
                      ptr_address& next) const
      {
         if (n->nprefix > key.size() - depth)
         {
            done = true;
            return false;
         }
         if (n->nprefix && std::memcmp(n->prefix(), key.data() + depth, n->nprefix) != 0)
         {
            done = true;
            return false;
         }
         depth += n->nprefix;
         if (depth == key.size())
         {
            done = true;
            if (n->term_value.is_null())
               return false;
            read_value(n->resolve_inline(n->term_value), out);
            return true;
         }
         smart_address sa = n->find(key[depth]);
         if (sa.is_null())
         {
            done = true;
            return false;
         }
         if (sa.is_tail())  // inline terminal: a unique key continuing past this byte
         {
            done                   = true;
            const inline_value* iv = n->resolve_inline(sa);
            bytes               suf = key.subspan(depth + 1);
            if (iv->suffix_len == suf.size() &&
                std::memcmp(iv->data, suf.data(), iv->suffix_len) == 0)
            {
               read_value(iv, out);
               return true;
            }
            return false;
         }
         next = sa.as_pointer();  // pointer → child leaf or router
         depth += 1;
         done = false;
         return false;
      }

      // ── leaf storage ────────────────────────────────────────────────────────
      ptr_address make_leaf(bytes pfx, sal::alloc_hint hint) { return _ses->alloc<leaf_node>(hint, pfx); }

      // Append (suffix,value) to a leaf, growing the page (realloc) if needed.
      void leaf_add(ptr_address& addr, leaf_node*& n, bytes suffix, bytes val)
      {
         uint32_t need = uint32_t(inline_value::byte_size(int(suffix.size()), int(val.size())));
         if (n->free_bytes() < need)
         {
            auto     ref   = _ses->get_ref<leaf_node>(addr);
            uint32_t newsz = leaf_node::page_size_for(n->head_size() + n->heap_used + need +
                                                      leaf_node::headroom);
            auto     nr    = _ses->realloc<leaf_node>(ref, newsz, leaf_node::copy_tag{}, n);
            addr           = nr.address();
            n              = nr.modify().get();
         }
         auto* iv       = reinterpret_cast<inline_value*>(n->alloc_tail(uint16_t(need)));
         iv->suffix_len = uint8_t(suffix.size());
         iv->value_len  = uint8_t(val.size());
         iv->value_type = 0;
         std::memcpy(iv->data, suffix.data(), suffix.size());
         std::memcpy(iv->data + suffix.size(), val.data(), val.size());
         // insert at the sorted (lexicographic-by-suffix) position so stems stay
         // ascending — keeps SIMD find's equal-run contiguous and lower_bound valid.
         int p = n->lower_bound_idx(suffix, /*strict=*/false);
         std::memmove(n->stems + p + 1, n->stems + p, size_t(n->count - p) * sizeof(uint32_t));
         std::memmove(n->vals + p + 1, n->vals + p, size_t(n->count - p) * sizeof(smart_address));
         n->stems[p] = leaf_node::stem4(suffix);
         n->vals[p]  = smart_address::from_tail(n->heap_used);
         ++n->count;
      }

      // Write an inline value into a node's heap and point a branch at it.
      // branch < 0 → term_value (key ends at this node's prefix); else → branch
      // byte (a unique key continuing one byte past). Grows the page if needed.
      template <class N>
      void put_inline(ptr_address& addr, N*& n, int branch, bytes suffix, bytes val)
      {
         if (branch < 0 && !n->term_value.is_null())
         {
            inline_value* iv = n->resolve_inline(n->term_value);
            assert(iv->value_len == val.size());
            std::memcpy(iv->data + iv->suffix_len, val.data(), val.size());
            return;
         }
         uint32_t need = uint32_t(inline_value::byte_size(int(suffix.size()), int(val.size())));
         if (n->free_bytes() < need)
         {
            auto     ref   = _ses->get_ref<N>(addr);
            uint32_t newsz = N::page_size_for(n->head_size() + n->heap_used + need + N::headroom);
            auto     nr    = _ses->realloc<N>(ref, newsz, typename N::copy_tag{}, n);
            addr           = nr.address();
            n              = nr.modify().get();
         }
         auto* iv       = reinterpret_cast<inline_value*>(n->alloc_tail(uint16_t(need)));
         iv->suffix_len = uint8_t(suffix.size());
         iv->value_len  = uint8_t(val.size());
         iv->value_type = 0;
         std::memcpy(iv->data, suffix.data(), suffix.size());
         std::memcpy(iv->data + suffix.size(), val.data(), val.size());
         smart_address sa = smart_address::from_tail(n->heap_used);
         if (branch < 0)
            n->term_value = sa;
         else if constexpr (requires(N x, uint8_t b, smart_address s) { x.set_branch(b, s); })
            n->set_branch(uint8_t(branch), sa);
         else
            assert(false && "inline branch only valid on routers");
      }
      template <class N>
      void set_term(ptr_address& addr, N*& n, bytes val)
      {
         put_inline(addr, n, -1, bytes{}, val);
      }

      // ── radix build (leaf overflow → router subtree over leaf children) ──────
      struct ent
      {
         bytes suf;
         bytes val;
      };
      static int common_prefix(const ent* e, int n) noexcept
      {
         size_t m = e[0].suf.size();
         for (int i = 1; i < n; ++i)
            m = std::min(m, e[i].suf.size());
         int p = 0;
         while (p < int(m))
         {
            uint8_t c = e[0].suf[size_t(p)];
            for (int i = 1; i < n; ++i)
               if (e[i].suf[size_t(p)] != c)
                  return p;
            ++p;
         }
         return int(m);
      }

      // Build a subtree for `ents` (n>=1) whose suffixes follow `lead`. Returns the
      // node address. `ents` is mutated (sorted / suffixes advanced) in place.
      ptr_address build(bytes lead, ent* ents, int n, sal::alloc_hint hint)
      {
         int     cp = common_prefix(ents, n);
         uint8_t pbuf[300];
         assert(lead.size() + size_t(cp) <= sizeof(pbuf));
         std::memcpy(pbuf, lead.data(), lead.size());
         std::memcpy(pbuf + lead.size(), ents[0].suf.data(), size_t(cp));
         bytes pfx(pbuf, lead.size() + size_t(cp));
         for (int i = 0; i < n; ++i)
            ents[i].suf = ents[i].suf.subspan(size_t(cp));  // now relative to after the node prefix

         if (n <= leaf_node::CAP)
         {
            ptr_address a = make_leaf(pfx, hint);
            leaf_node*  ln = mut<leaf_node>(a);
            for (int i = 0; i < n; ++i)
            {
               if (ents[i].suf.empty())
                  set_term(a, ln, ents[i].val);
               else
                  leaf_add(a, ln, ents[i].suf, ents[i].val);
            }
            return a;
         }

         // router: sort by first byte (empty suffix = the term, sorts first)
         std::sort(ents, ents + n,
                   [](const ent& x, const ent& y)
                   {
                      int bx = x.suf.empty() ? -1 : int(x.suf[0]);
                      int by = y.suf.empty() ? -1 : int(y.suf[0]);
                      return bx < by;
                   });
         int i = 0;
         int distinct = 0;
         for (int k = (ents[0].suf.empty() ? 1 : 0); k < n;)
         {
            uint8_t b = ents[k].suf[0];
            ++distinct;
            while (k < n && !ents[k].suf.empty() && ents[k].suf[0] == b)
               ++k;
         }
         ptr_address router = (distinct <= setlist_node::CAP)  ? make_router<setlist_node>(pfx, hint)
                              : (distinct <= bitset_node::CAP) ? make_router<bitset_node>(pfx, hint)
                                                              : make_router<full_node>(pfx, hint);
         // Fanout width drives the size-1 decision (NOT a fixed count):
         //   wide  (≈uniform, most groups singletons) → dense radix + inline terminals,
         //         else uniform data explodes into single-entry leaves that re-split.
         //   narrow (clustered, e.g. a dictionary)    → keep stem-leaves (even of one
         //         entry) so the tree stays shallow — recovers the query win.
         bool wide = distinct * 2 >= n;
         if (ents[0].suf.empty())  // term
         {
            set_router_term(router, ents[0].val);
            i = 1;
         }
         while (i < n)
         {
            uint8_t b = ents[i].suf[0];
            int     j = i;
            while (j < n && ents[j].suf[0] == b)
               ++j;
            if (j - i == 1 && wide)
            {
               // unique continuation in a wide split → inline terminal (no child).
               set_router_inline(router, b, ents[i].suf.subspan(1), ents[i].val);
            }
            else
            {
               for (int k = i; k < j; ++k)
                  ents[k].suf = ents[k].suf.subspan(1);  // consume the branch byte
               ptr_address child = build(bytes{}, ents + i, j - i, router_hint(router));
               set_router_branch(router, b, child);
            }
            i = j;
         }
         return router;
      }

      template <class N>
      ptr_address make_router(bytes pfx, sal::alloc_hint hint)
      {
         return _ses->alloc<N>(hint, pfx);
      }
      // Dispatch helpers so build() can drive any router tier by address.
      sal::alloc_hint router_hint(ptr_address a)
      {
         auto ref = _ses->get_ref<sal::alloc_header>(a);
         auto t   = ref->type();
         if (t == setlist_node::type_id)
            return ref.as<setlist_node>().obj()->branch_hint();
         if (t == bitset_node::type_id)
            return ref.as<bitset_node>().obj()->branch_hint();
         return ref.as<full_node>().obj()->branch_hint();
      }
      void set_router_branch(ptr_address a, uint8_t b, ptr_address child)
      {
         auto ref = _ses->get_ref<sal::alloc_header>(a);
         auto t   = ref->type();
         if (t == setlist_node::type_id)
            ref.as<setlist_node>().modify().get()->set_branch(b, smart_address::from_pointer(child));
         else if (t == bitset_node::type_id)
            ref.as<bitset_node>().modify().get()->set_branch(b, smart_address::from_pointer(child));
         else
            ref.as<full_node>().modify().get()->set_branch(b, smart_address::from_pointer(child));
      }
      void set_router_term(ptr_address a, bytes val)
      {
         auto ref = _ses->get_ref<sal::alloc_header>(a);
         auto t   = ref->type();
         if (t == setlist_node::type_id)
         {
            auto* n = ref.as<setlist_node>().modify().get();
            set_term(a, n, val);
         }
         else if (t == bitset_node::type_id)
         {
            auto* n = ref.as<bitset_node>().modify().get();
            set_term(a, n, val);
         }
         else
         {
            auto* n = ref.as<full_node>().modify().get();
            set_term(a, n, val);
         }
      }
      // Inline terminal branch (unique key continuing past byte b) on any router.
      void set_router_inline(ptr_address a, uint8_t b, bytes suffix, bytes val)
      {
         auto ref = _ses->get_ref<sal::alloc_header>(a);
         auto t   = ref->type();
         if (t == setlist_node::type_id)
         {
            auto* n = ref.as<setlist_node>().modify().get();
            put_inline(a, n, b, suffix, val);
         }
         else if (t == bitset_node::type_id)
         {
            auto* n = ref.as<bitset_node>().modify().get();
            put_inline(a, n, b, suffix, val);
         }
         else
         {
            auto* n = ref.as<full_node>().modify().get();
            put_inline(a, n, b, suffix, val);
         }
      }

      // ── insert ──────────────────────────────────────────────────────────────
      ptr_address insert_node(ptr_address addr, bytes key, uint32_t depth, bytes val,
                              sal::alloc_hint peer_hint)
      {
         auto t = _ses->get_ref<sal::alloc_header>(addr)->type();
         if (t == leaf_node::type_id)
            return leaf_insert(addr, key, depth, val, peer_hint);
         if (t == setlist_node::type_id)
            return router_insert<setlist_node>(addr, key, depth, val, peer_hint);
         if (t == bitset_node::type_id)
            return router_insert<bitset_node>(addr, key, depth, val, peer_hint);
         return router_insert<full_node>(addr, key, depth, val, peer_hint);
      }

      // Split a node's prefix at p: a new setlist parent takes key[depth..depth+p];
      // the old node (demoted: prefix loses p+1 bytes) hangs under branch old_byte;
      // the new key gets a fresh leaf (or the parent's term). Returns the parent.
      template <class N>
      ptr_address prefix_split(ptr_address addr, N*& n, bytes key, uint32_t depth, int p, bytes val,
                               sal::alloc_hint peer_hint)
      {
         uint8_t     ob  = n->prefix()[p];
         ptr_address np  = make_router<setlist_node>(key.subspan(depth, size_t(p)), peer_hint);
         auto*       pnp = mut<setlist_node>(np);
         n               = mut<N>(addr);
         int newpl       = n->nprefix - (p + 1);
         std::memmove(n->prefix(), n->prefix() + p + 1, size_t(newpl));
         n->nprefix = uint16_t(newpl);
         pnp->set_branch(ob, smart_address::from_pointer(addr));
         uint32_t d2 = depth + uint32_t(p);
         if (d2 == key.size())
            set_term(np, pnp, val);
         else
            put_inline(np, pnp, key[d2], key.subspan(d2 + 1), val);  // inline terminal on parent
         return np;
      }

      ptr_address leaf_insert(ptr_address addr, bytes key, uint32_t depth, bytes val,
                              sal::alloc_hint peer_hint)
      {
         leaf_node* n     = mut<leaf_node>(addr);
         uint32_t   avail = uint32_t(key.size() - depth);
         int        maxp  = int(std::min<uint32_t>(n->nprefix, avail));
         int        p     = 0;
         while (p < maxp && n->prefix()[p] == key[depth + p])
            ++p;
         if (p < n->nprefix)
            return prefix_split<leaf_node>(addr, n, key, depth, p, val, peer_hint);

         depth += n->nprefix;
         bytes rem = key.subspan(depth);
         if (rem.empty())
         {
            set_term(addr, n, val);
            return addr;
         }
         int idx = n->find_idx(rem);
         if (idx >= 0)
         {
            inline_value* iv = n->resolve_inline(n->vals[idx]);
            assert(iv->value_len == val.size());
            std::memcpy(iv->data + iv->suffix_len, val.data(), val.size());
            return addr;
         }
         if (n->count < leaf_node::CAP)
         {
            leaf_add(addr, n, rem, val);
            return addr;
         }
         return leaf_overflow(addr, n, rem, val, peer_hint);
      }

      // Leaf is full: gather its entries + the new one, build a router subtree,
      // free the old leaf. The leaf prefix becomes the build `lead`.
      ptr_address leaf_overflow(ptr_address addr, leaf_node* n, bytes rem, bytes val,
                                sal::alloc_hint peer_hint)
      {
         ent ents[leaf_node::CAP + 2];
         int m = 0;
         if (!n->term_value.is_null())
         {
            const inline_value* iv = n->resolve_inline(n->term_value);
            ents[m++]              = {bytes{}, iv->value()};
         }
         for (int i = 0; i < n->count; ++i)
         {
            const inline_value* iv = n->resolve_inline(n->vals[i]);
            ents[m++]              = {iv->suffix(), iv->value()};
         }
         ents[m++] = {rem, val};

         uint8_t leadbuf[300];
         std::memcpy(leadbuf, n->prefix(), n->nprefix);
         bytes lead(leadbuf, n->nprefix);

         ptr_address sub = build(lead, ents, m, peer_hint);
         _ses->release(addr);  // old leaf replaced
         return sub;
      }

      template <class N>
      ptr_address router_insert(ptr_address addr, bytes key, uint32_t depth, bytes val,
                                sal::alloc_hint peer_hint)
      {
         const uint32_t depth0 = depth;
         N*             n      = mut<N>(addr);
         uint32_t       avail  = uint32_t(key.size() - depth);
         int            maxp   = int(std::min<uint32_t>(n->nprefix, avail));
         int            p      = 0;
         while (p < maxp && n->prefix()[p] == key[depth + p])
            ++p;
         if (p < n->nprefix)
            return prefix_split<N>(addr, n, key, depth, p, val, peer_hint);

         depth += n->nprefix;
         if (depth == key.size())
         {
            set_term(addr, n, val);
            return addr;
         }
         uint8_t       b  = key[depth];
         smart_address sa = n->find(b);
         if (sa.is_null())
         {
            if (!n->branch_has_room(b))
            {
               if constexpr (std::is_same_v<N, setlist_node>)
                  upgrade<bitset_node, setlist_node>(addr);
               else if constexpr (std::is_same_v<N, bitset_node>)
                  upgrade<full_node, bitset_node>(addr);
               return insert_node(addr, key, depth0, val, peer_hint);
            }
            put_inline(addr, n, b, key.subspan(depth + 1), val);  // unique key → inline terminal
            return addr;
         }
         if (sa.is_tail())
         {
            // collision: an existing inline terminal shares this byte → move both
            // into a flat LEAF (the catch-all) that absorbs further collisions.
            inline_value* iv  = n->resolve_inline(sa);
            bytes         suf = key.subspan(depth + 1);
            if (iv->suffix_len == suf.size() &&
                std::memcmp(iv->data, suf.data(), iv->suffix_len) == 0)
            {
               std::memcpy(iv->data + iv->suffix_len, val.data(), val.size());  // update
               return addr;
            }
            uint8_t buf[inline_value::max_inline_size];
            std::memcpy(buf, iv, iv->byte_size());
            auto* eiv = reinterpret_cast<inline_value*>(buf);
            ent   two[2] = {{{eiv->data, eiv->suffix_len}, {eiv->data + eiv->suffix_len, eiv->value_len}},
                            {suf, val}};
            ptr_address lf = build(bytes{}, two, 2, n->branch_hint());  // 2 ≤ CAP → a leaf
            n              = mut<N>(addr);
            n->set_branch(b, smart_address::from_pointer(lf));
            return addr;
         }
         ptr_address child = sa.as_pointer();
         ptr_address nc    = insert_node(child, key, depth + 1, val, n->branch_hint());
         if (nc != child)
         {
            n = mut<N>(addr);
            n->set_branch(b, smart_address::from_pointer(nc));
         }
         return addr;
      }

      template <class To, class From>
      void upgrade(ptr_address addr)
      {
         auto     ref   = _ses->get_ref<From>(addr);
         uint32_t newsz = To::page_size_for(uint32_t(sizeof(To)) + ref->nprefix + ref->heap_used +
                                            To::headroom);
         (void)_ses->realloc<To>(ref, newsz, typename To::upgrade_tag{}, ref.obj());
      }

      // ── walks ────────────────────────────────────────────────────────────────
      void walk_tiers(ptr_address a, tier_counts& tc)
      {
         auto ref = _ses->get_ref<sal::alloc_header>(a);
         auto t   = ref->type();
         auto rec = [&](ptr_address c) { walk_tiers(c, tc); };
         if (t == leaf_node::type_id)
            ++tc.leaf;  // leaves have no node children
         else if (t == setlist_node::type_id)
         {
            ++tc.setlist;
            ref.as<setlist_node>().obj()->visit_children(rec);
         }
         else if (t == bitset_node::type_id)
         {
            ++tc.bitset;
            ref.as<bitset_node>().obj()->visit_children(rec);
         }
         else
         {
            ++tc.full;
            ref.as<full_node>().obj()->visit_children(rec);
         }
      }
      size_t walk_bytes(ptr_address a, bool compacted)
      {
         auto   ref = _ses->get_ref<sal::alloc_header>(a);
         auto   t   = ref->type();
         size_t sum = 0;
         auto   rec = [&](ptr_address c) { sum += walk_bytes(c, compacted); };
         if (t == leaf_node::type_id)
         {
            auto* n = ref.as<leaf_node>().obj();
            sum += compacted ? n->compact_size() : n->size();
         }
         else if (t == setlist_node::type_id)
         {
            auto* n = ref.as<setlist_node>().obj();
            sum += compacted ? n->compact_size() : n->size();
            n->visit_children(rec);
         }
         else if (t == bitset_node::type_id)
         {
            auto* n = ref.as<bitset_node>().obj();
            sum += compacted ? n->compact_size() : n->size();
            n->visit_children(rec);
         }
         else
         {
            auto* n = ref.as<full_node>().obj();
            sum += compacted ? n->compact_size() : n->size();
            n->visit_children(rec);
         }
         return sum;
      }

      std::unique_ptr<sal::allocator> _alloc;
      sal::allocator_session_ptr      _ses;
      ptr_address                     _root;
   };
}  // namespace heart::sal_proto

// vpart_alloc — plugs the vpart arena into vhkv::kv_tree via hkv's allocator
// protocol (hkv_handle + hkv_base + allocate/deallocate). The handle is the
// 4-byte indexed line_ptr: [28-bit line index][4-bit kind tag], little-endian,
// so the tree's deref is base + (idx << 7) and descent dispatch never loads a
// header. The node layout and descent path are hkv's, verbatim: that is the
// read-performance contract.
//
// LIVENESS (vpart L1a, docs/unified_tier_design.md): per-line refcounts are
// RETIRED from the tree protocol. Every node carries a 4-byte version stamp
// (the transaction id that created it, written at pack_); the write path's
// privacy test is one compare (stamp == current txn -> mutate in place; else
// COW). Superseded nodes append to per-transaction chain-line logs:
//   clone-log  — (raw, birth|DEEP) 8 B entries. SHALLOW = children carried by
//                the successor clone (drain frees the node's lines only);
//                DEEP = subtree dropped (drain cascade-frees recursively).
//   create-log — (line, nl) 8 B entries appended by allocate(); abort frees
//                every still-allocated entry (kv-compat 4.7b rollback).
// Reclaim rule (floor): a committed chain with death D drains once every
// pinned version is >= D (a pin at V sees exactly the state of commit V).
// Sharing costs ZERO: no retain, no release, no per-child refcount traffic.
// (refs.part still initializes to 1 per allocation — it now serves only as
// the allocated/free bit for release_run; retire it entirely in a follow-up.)
#pragma once
#include <vpart/arena.hpp>
#include <vpart/hkv/tagged_ptr.hpp>

namespace vpart
{
   namespace detail
   {
      // 4-byte indexed branch handle (hkv line_ptr): same byte-array shape as
      // packed_ptr_t (memcpy load/store, unaligned-safe in packed node arrays).
      struct line_ptr
      {
         static constexpr bool indexed = true;
         uint8_t               b[4];

         static line_ptr null() noexcept
         {
            line_ptr p;
            std::memset(p.b, 0xFF, 4);  // tag nibble becomes 0xF == K::null
            return p;
         }
         static line_ptr from_off(size_t byte_off, vhkv::detail::K k) noexcept
         {
            assert((byte_off & 0x7f) == 0 && "vpart: node offsets are line-aligned");
            assert(byte_off >> 35 == 0 && "vpart: offset exceeds the 28-bit line index");
            line_ptr       p;
            const uint32_t raw = uint32_t(byte_off >> 3) | uint8_t(k);  // (off>>7)<<4 == off>>3
            std::memcpy(p.b, &raw, 4);
            return p;
         }
         uint64_t raw() const noexcept
         {
            uint32_t v;
            std::memcpy(&v, b, 4);
            return v;
         }
         vhkv::detail::K tag() const noexcept { return vhkv::detail::K(b[0] & 0xF); }
         bool            is_null() const noexcept { return tag() == vhkv::detail::K::null; }

         line_idx index() const noexcept { return line_idx(raw() >> 4); }
      };
      static_assert(sizeof(line_ptr) == 4, "line_ptr is exactly 4 bytes");
   }  // namespace detail

   // Per-transaction log state, shared by every allocator copy of one writer
   // (kv::write_session owns it; view/reader allocators carry nullptr).
   // Logs are linked lists of 128 B arena lines, push-to-front, allocated from
   // the writer's active segments (dirty-anyway pages: zero extra sync).
   // Line format (both logs): [u32 prev][u32 n][15 x (u32 a, u32 b)]
   //   clone-log entry:  a = tagged raw handle, b = birth stamp | DEEP flag
   //   create-log entry: a = line_idx,          b = run length (lines)
   struct vlog_state
   {
      static constexpr uint32_t npos = 0xFFFFFFFFu;
      static constexpr uint32_t cap  = 15;
      static constexpr uint32_t DEEP = 0x80000000u;  // flag on the birth word

      uint32_t clone_head  = npos;  // newest clone-log line (line_idx)
      uint32_t clone_n     = cap;   // entries used in the head line
      uint32_t create_head = npos;
      uint32_t create_n    = cap;

      void reset() noexcept
      {
         clone_head = create_head = npos;
         clone_n = create_n = cap;
      }
      bool empty() const noexcept
      {
         return clone_head == npos && create_head == npos;
      }
   };

   template <class T>
   struct part_alloc
   {
      using value_type = T;
      using hkv_handle = detail::line_ptr;

      arena*      a        = nullptr;  // deref base (any thread)
      session*    ses      = nullptr;  // allocation context (writer thread only);
                                       // null for read-only snapshot views
      vlog_state* vlog     = nullptr;  // writer's per-txn logs (null on views)
      uint32_t    vtxn     = 0;        // current write window id (0 = no window)
      bool        vreclaim = false;    // drain context: free-now instead of log

      part_alloc() = default;
      explicit part_alloc(session* s) noexcept : a(&s->store()), ses(s) {}
      explicit part_alloc(session& s) noexcept : a(&s.store()), ses(&s) {}
      explicit part_alloc(arena* ar) noexcept : a(ar) {}  // read-only views
      part_alloc(arena* ar, bool reclaim) noexcept : a(ar), vreclaim(reclaim) {}
      template <class U>
      part_alloc(const part_alloc<U>& o) noexcept
          : a(o.a), ses(o.ses), vlog(o.vlog), vtxn(o.vtxn), vreclaim(o.vreclaim)
      {
      }

      const std::byte* hkv_base() const noexcept { return a ? a->base() : nullptr; }

      static constexpr size_t lines_(size_t bytes) noexcept
      {
         return (bytes + line_bytes - 1) / line_bytes;
      }
      T* allocate(size_t n)
      {
         static_assert(alignof(T) <= line_bytes, "vpart lines are 128-byte aligned");
         assert(ses && "read-only view allocators cannot allocate");
         const uint32_t nl = uint32_t(lines_(n * sizeof(T)));
         const line_idx i  = ses->alloc_lines(nl);
         if (vlog && vtxn)  // create-log: abort frees every still-allocated entry
            vappend_(vlog->create_head, vlog->create_n, i, nl);
         return static_cast<T*>(a->get(i));
      }
      void deallocate(T* p, size_t n) noexcept
      {
         const line_idx i =
             line_idx((reinterpret_cast<const std::byte*>(p) - a->base()) / line_bytes);
         if (ses)
            ses->release_run(i, uint32_t(lines_(n * sizeof(T))));
         else
            a->release_run(i, uint32_t(lines_(n * sizeof(T))));
      }

      template <class U>
      bool operator==(const part_alloc<U>& o) const noexcept
      {
         return a == o.a && ses == o.ses;
      }

      // ── version-liveness hooks (consumed by the tree's vpart-COW extension) ──
      static constexpr bool hkv_cow = true;

      // privacy test: was this stamp minted by the CURRENT write window?
      bool hkv_vold(uint32_t stamp) const noexcept { return stamp != vtxn || vtxn == 0; }
      uint32_t hkv_vtxn() const noexcept { return vtxn; }

      // clone-log appends. raw = full tagged handle; birth = the dead node's stamp.
      void hkv_vshallow(uint32_t raw, uint32_t birth) noexcept
      {
         assert(vlog && vtxn && "shallow supersession outside a write window");
         vappend_(vlog->clone_head, vlog->clone_n, raw, birth & ~vlog_state::DEEP);
      }
      void hkv_vdeep(uint32_t raw, uint32_t birth) noexcept
      {
         assert(vlog && vtxn && "deep supersession outside a write window");
         vappend_(vlog->clone_head, vlog->clone_n, raw, birth | vlog_state::DEEP);
      }

      // legacy refcount hooks: only reachable from L1a-disabled paths
      // (explicit tiers, cbucket). They must never run.
      uint8_t hkv_ref(uint32_t) const noexcept
      {
         assert(false && "vpart L1a: refcount hook on a disabled path");
         return 2;
      }
      bool hkv_defer(uint32_t) noexcept { return false; }
      bool hkv_retain(uint32_t) noexcept { return true; }
      template <class D>
      bool hkv_release(uint32_t, D&&) noexcept
      {
         assert(false && "vpart L1a: refcount release on a disabled path");
         return false;
      }
      void hkv_free_run(uint32_t line, uint32_t n) noexcept
      {
         if (ses)
            ses->release_run(line, n);
         else
            a->release_run(line, n);
      }

     private:
      uint32_t* line_u32_(uint32_t line) const noexcept
      {
         return static_cast<uint32_t*>(a->get(line));
      }
      void vappend_(uint32_t& head, uint32_t& n, uint32_t a32, uint32_t b32) noexcept
      {
         if (n == vlog_state::cap)
         {
            const line_idx i = ses->alloc_lines(1);  // log lines are not create-logged
            uint32_t*      w = line_u32_(i);
            w[0]             = head;
            w[1]             = 0;
            head             = i;
            n                = 0;
         }
         uint32_t* w  = line_u32_(head);
         w[2 + n * 2] = a32;
         w[3 + n * 2] = b32;
         w[1]         = ++n;
      }
   };
}  // namespace vpart

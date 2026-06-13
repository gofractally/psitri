// vpart_alloc — plugs the vpart arena into start_ver::hkv::kv_tree via hkv's allocator
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
// COW). Superseded nodes append to a per-transaction clone-log:
//   (raw, birth|DEEP) 8 B entries. SHALLOW = children carried by the
//   successor clone (drain frees the node's lines only); DEEP = subtree
//   dropped (drain cascade-frees recursively). Abort needs no log at all:
//   it walks the window (kv-compat 4.7b rollback).
// Reclaim rule (floor): a committed chain with death D drains once every
// pinned version is >= D (a pin at V sees exactly the state of commit V).
// Sharing costs ZERO: no retain, no release, no per-child refcount traffic.
// (refs.part still initializes to 1 per allocation — it now serves only as
// the allocated/free bit for release_run; retire it entirely in a follow-up.)
#pragma once
#include <bit>
#include <start_ver/arena.hpp>
#include <start_ver/hkv/tagged_ptr.hpp>

namespace start_ver
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
         static line_ptr from_off(size_t byte_off, start_ver::hkv::detail::K k) noexcept
         {
            assert((byte_off & 0x7f) == 0 && "vpart: node offsets are line-aligned");
            assert(byte_off >> 35 == 0 && "vpart: offset exceeds the 28-bit line index");
            line_ptr       p;
            const uint32_t raw = uint32_t(byte_off >> 3) | uint8_t(k);  // (off>>7)<<4 == off>>3
            std::memcpy(p.b, &raw, 4);
            return p;
         }
         // LARGE arena: the index counts 8 KB GRAINS, not 128 B lines. The handle
         // is otherwise identical (28-bit index in the high nibbles, 4-bit tag);
         // deref multiplies the index by large_grain_bytes instead of line_bytes.
         static line_ptr from_grain_off(size_t byte_off, start_ver::hkv::detail::K k) noexcept
         {
            assert((byte_off % large_grain_bytes) == 0 && "vpart: large objects are grain-aligned");
            const size_t grain = byte_off / large_grain_bytes;
            assert(grain >> 28 == 0 && "vpart: large index exceeds the 28-bit field");
            line_ptr       p;
            const uint32_t raw = uint32_t(grain << 4) | uint8_t(k);
            std::memcpy(p.b, &raw, 4);
            return p;
         }
         uint64_t raw() const noexcept
         {
            uint32_t v;
            std::memcpy(&v, b, 4);
            return v;
         }
         start_ver::hkv::detail::K tag() const noexcept { return start_ver::hkv::detail::K(b[0] & 0xF); }
         bool            is_null() const noexcept { return tag() == start_ver::hkv::detail::K::null; }

         line_idx index() const noexcept { return line_idx(raw() >> 4); }
      };
      static_assert(sizeof(line_ptr) == 4, "line_ptr is exactly 4 bytes");
   }  // namespace detail

   // Drain -> writer recycle ring (SPSC: the drain thread pushes freed runs,
   // the writer pops them into its session fcache at transaction start).
   // Entries keep their freeset bits CLAIMED — recycling skips the freeset
   // round-trip entirely and hands hot, exact-size runs back to the writer
   // (Dan: over-allocating steals cache from others; this removes both the
   // size-class rounding waste AND the odd-size scan). NOTE: recycled lines
   // re-dirty old pages — a write-amp trade at sync modes (fine at none).
   struct recycle_ring
   {
      static constexpr uint32_t cap = 1u << 14;
      std::atomic<uint32_t>     head{0}, tail{0};
      uint64_t                  slot[cap];

      bool push(uint32_t line, uint32_t nl) noexcept  // producer: drain only
      {
         const uint32_t h = head.load(std::memory_order_relaxed);
         if (h - tail.load(std::memory_order_acquire) >= cap)
            return false;
         slot[h % cap] = (uint64_t(line) << 6) | nl;
         head.store(h + 1, std::memory_order_release);
         return true;
      }
      bool pop(uint32_t& line, uint32_t& nl) noexcept  // consumer: writer only
      {
         const uint32_t t = tail.load(std::memory_order_relaxed);
         if (t == head.load(std::memory_order_acquire))
            return false;
         const uint64_t v = slot[t % cap];
         tail.store(t + 1, std::memory_order_release);
         line = uint32_t(v >> 6);
         nl   = uint32_t(v & 63);
         return true;
      }
   };

   // Per-transaction clone-log state, shared by every allocator copy of one
   // writer (kv::write_session owns it; view/reader allocators carry nullptr).
   // The log is a linked list of 128 B arena lines, push-to-front, allocated
   // from the writer's active segments (dirty-anyway pages: zero extra sync).
   // Line format: [u32 prev][u32 n][15 x (u32 raw, u32 birth|DEEP)]
   // There is NO create-log: abort recovers the diverged generation by
   // WALKING the window (recurse into current-stamped nodes only — old nodes
   // are shared with the restored root and stop the walk), so the allocation
   // hot path carries zero log traffic.
   struct vlog_state
   {
      static constexpr uint32_t npos = 0xFFFFFFFFu;
      static constexpr uint32_t cap  = 15;
      static constexpr uint32_t DEEP = 0x80000000u;  // flag on the birth word

      uint32_t clone_head = npos;  // newest clone-log line (line_idx)
      uint32_t clone_n    = cap;   // entries used in the head line

      void reset() noexcept
      {
         clone_head = npos;
         clone_n    = cap;
      }
      bool empty() const noexcept { return clone_head == npos; }
   };

   // Typed-value lifecycle hooks (LOCKED API §9): fired when stored value
   // bytes are copied between leaves (retain) or physically die with
   // their leaf (release) — including drain/reaper husks. Installed by
   // the kv database on EVERY engine it constructs; hooks parse the
   // (typenum, payload) envelope and dispatch the type registry.
   // Namespace-scope (not nested) so allocator REBINDS share the pointer.
   struct value_hooks
   {
      void (*on_copy)(const char* bytes, size_t n, void* user);
      void (*on_release)(const char* bytes, size_t n, void* user);
      void* user;
   };

   template <class T>
   struct part_alloc
   {
      using value_type = T;
      using hkv_handle = detail::line_ptr;

      arena*      a        = nullptr;  // deref base (any thread)
      session*    ses      = nullptr;  // allocation context (writer thread only);
                                       // null for read-only snapshot views
      vlog_state*   vlog     = nullptr;  // writer's per-txn logs (null on views)
      const value_hooks* vhooks = nullptr;

      recycle_ring* vring    = nullptr;  // drain context: freed runs recycle here
      uint32_t      vtxn     = 0;        // current write window id (0 = no window)
      bool          vreclaim = false;    // drain context: free-now instead of log

      // ── large-value second arena (8 KB grain; value_large leaves) ─────────
      // lses (writer-only) is the allocation context; the deref/free base comes
      // from a->companion() so EVERY allocator copy — views, drain, reaper —
      // resolves and frees a large leaf without carrying lses. The same COW /
      // refcount / recovery machinery the line arena has (large_arena IS the
      // grain-parameterized arena), with no per-line-arena overhead.
      large_session* lses = nullptr;

      part_alloc() = default;
      explicit part_alloc(session* s) noexcept : a(&s->store()), ses(s) {}
      explicit part_alloc(session& s) noexcept : a(&s.store()), ses(&s) {}
      explicit part_alloc(arena* ar) noexcept : a(ar) {}  // read-only views
      part_alloc(arena* ar, bool reclaim) noexcept : a(ar), vreclaim(reclaim) {}
      template <class U>
      part_alloc(const part_alloc<U>& o) noexcept
          : a(o.a), ses(o.ses), vlog(o.vlog), vhooks(o.vhooks), vring(o.vring), vtxn(o.vtxn),
            vreclaim(o.vreclaim), lses(o.lses)
      {
      }

      const std::byte* hkv_base() const noexcept { return a ? a->base() : nullptr; }

      // ── large arena addressing/alloc/free (value_large leaves) ────────────
      // The companion large_arena is held by the LINE arena (set once at db
      // construction), so any allocator copy reaches it through `a`.
      large_arena* large_() const noexcept
      {
         return a ? static_cast<large_arena*>(a->companion()) : nullptr;
      }
      const std::byte* hkv_large_base() const noexcept
      {
         large_arena* la = large_();
         return la ? la->base() : nullptr;
      }
      static constexpr bool has_large = true;
      // Largest value the line arena can hold (64 lines minus leaf/suffix
      // overhead): the tree decides line-vs-large against this.
      static constexpr size_t large_threshold_bytes = size_t(max_alloc_lines) * line_bytes;
      static constexpr size_t large_max_bytes       = large_max_object_bytes;

      // Allocate a large leaf of `bytes`; returns its grain offset (from the
      // large arena base) — the tree packs it with from_grain_off + value_large.
      size_t allocate_large(size_t bytes)
      {
         large_arena*   la = large_();
         assert(la && lses && "large-value store has no large arena/session");
         if (bytes > large_max_bytes)
            throw std::runtime_error("vpart: value exceeds the large-arena 0.5 MB object cap");
         const uint32_t units = uint32_t((bytes + large_grain_bytes - 1) / large_grain_bytes);
         const line_idx g     = lses->alloc_lines(units);
         return size_t(g) * large_grain_bytes;
      }
      void deallocate_large(size_t grain_off, size_t bytes) noexcept
      {
         large_arena*   la    = large_();
         const line_idx g     = line_idx(grain_off / large_grain_bytes);
         const uint32_t units = uint32_t((bytes + large_grain_bytes - 1) / large_grain_bytes);
         if (lses)
            lses->release_run(g, units);
         else if (la)
            la->release_run(g, units);
      }

      static constexpr size_t lines_(size_t bytes) noexcept
      {
         return (bytes + line_bytes - 1) / line_bytes;
      }
      // EXACT line counts (Dan: over-allocating steals cache from others —
      // the earlier {1,2,4,8,16} rounding wasted 6 lines per node_full). The
      // odd-size scan the rounding dodged is now dodged better: steady-state
      // odd runs come back through the drain->fcache recycle ring O(1).
      T* allocate(size_t n)
      {
         static_assert(alignof(T) <= line_bytes, "vpart lines are 128-byte aligned");
         assert(ses && "read-only view allocators cannot allocate");
         const line_idx i = ses->alloc_lines(uint32_t(lines_(n * sizeof(T))));
         return static_cast<T*>(a->get(i));
      }
      // compile-time-N variant: the class is the INSTANTIATION (Dan: since
      // n is constant at the typed call sites, make it constant).
      static constexpr bool has_alloc_c = true;
      template <size_t Bytes>
      T* allocate_c()
      {
         assert(ses && "read-only view allocators cannot allocate");
         constexpr uint32_t NL = uint32_t((Bytes + line_bytes - 1) / line_bytes);
         const line_idx     i  = ses->template alloc_lines_c<NL>();
         return static_cast<T*>(a->get(i));
      }
      void deallocate(T* p, size_t n) noexcept
      {
         const line_idx i =
             line_idx((reinterpret_cast<const std::byte*>(p) - a->base()) / line_bytes);
         const uint32_t nl = uint32_t(lines_(n * sizeof(T)));
         if (vreclaim && vring && nl <= 16 && vring->push(i, nl))
            return;  // recycled: bits stay claimed; the writer fcaches the run
         if (ses)
            ses->release_run(i, nl);
         else
            a->release_run(i, nl);
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
}  // namespace start_ver

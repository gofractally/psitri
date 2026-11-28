// part_alloc — plugs the part arena into hkv::kv_tree via hkv's allocator
// protocol (hkv_handle + hkv_base + allocate/deallocate). The handle is the
// 4-byte indexed line_ptr from hkv's Stage C pool work: [28-bit line index]
// [4-bit kind tag], little-endian — bit-identical to part's ptr design, so the
// tree's deref is base + (idx << 7) and descent dispatch never loads a header
// (SPEC §2.7). The node layout and descent path are hkv's, verbatim: that is
// the read-performance contract.
//
// Allocation routes through a start_ref::session (first-fit-forward from the txn
// floor); deallocation releases each line's refcount (1 -> 0 frees). Under
// exclusive ownership that is exactly hkv's free; under COW the same call
// safely just drops a reference when a line is shared.
#pragma once
#include <start_ref/arena.hpp>
#include <start_ref/hkv/tagged_ptr.hpp>

namespace start_ref
{
   namespace detail
   {
      // 5-byte indexed branch handle (hkv line_ptr): same byte-array shape as
      // packed_ptr_t (memcpy load/store, unaligned-safe in packed node arrays).
      // The index is 36 bits (40-bit handle - 4-bit tag) => 2^36 lines * 128 B =
      // 8 TB, comfortably above the arena's 32-bit line_idx ceiling (sentinel_base
      // 0xFFFF0000 lines). A 4-byte handle held only 28 bits => 4096 segments /
      // 34 GB, which the COW-bloated arena silently overflowed past 2.4M.
      static constexpr unsigned HANDLE_BYTES = 5;
      struct line_ptr
      {
         static constexpr bool indexed = true;
         uint8_t               b[HANDLE_BYTES];

         static line_ptr null() noexcept
         {
            line_ptr p;
            std::memset(p.b, 0xFF, HANDLE_BYTES);  // tag nibble becomes 0xF == K::null
            return p;
         }
         static line_ptr from_off(size_t byte_off, hkv::detail::K k) noexcept
         {
            assert((byte_off & 0x7f) == 0 && "part: node offsets are line-aligned");
            assert(byte_off >> 43 == 0 && "part: offset exceeds the 36-bit line index");
            line_ptr       p;
            const uint64_t raw = uint64_t(byte_off >> 3) | uint8_t(k);  // (off>>7)<<4 == off>>3
            std::memcpy(p.b, &raw, HANDLE_BYTES);
            return p;
         }
         // LARGE arena: the index counts 32 KiB GRAINS, not 128 B lines. The handle
         // is otherwise identical (36-bit index in the high nibbles, 4-bit tag);
         // deref multiplies the index by large_grain_bytes instead of line_bytes.
         static line_ptr from_grain_off(size_t byte_off, hkv::detail::K k) noexcept
         {
            assert((byte_off % large_grain_bytes) == 0 && "part: large objects are grain-aligned");
            const size_t grain = byte_off / large_grain_bytes;
            assert(grain >> 36 == 0 && "part: large index exceeds the 36-bit field");
            line_ptr       p;
            const uint64_t raw = (uint64_t(grain) << 4) | uint8_t(k);
            std::memcpy(p.b, &raw, HANDLE_BYTES);
            return p;
         }
         uint64_t raw() const noexcept
         {
            uint64_t v = 0;
            std::memcpy(&v, b, HANDLE_BYTES);
            return v;
         }
         hkv::detail::K tag() const noexcept { return hkv::detail::K(b[0] & 0xF); }
         bool           is_null() const noexcept { return tag() == hkv::detail::K::null; }

         line_idx index() const noexcept { return line_idx(raw() >> 4); }
      };
      static_assert(sizeof(line_ptr) == HANDLE_BYTES, "line_ptr is exactly 5 bytes");
   }  // namespace detail

   // Typed-value lifecycle hooks (LOCKED API, adb): fired when stored value
   // bytes are copied between leaves (retain) or physically die with their
   // leaf (release) — including refcount-zero cascades on the background
   // releaser. Installed by the kv database on EVERY tree it constructs;
   // hooks parse the (typenum, payload) envelope and dispatch the registry.
   // Namespace-scope (not nested) so allocator REBINDS share the pointer.
   struct value_hooks
   {
      void (*on_copy)(char* bytes, size_t n, void* user, large_session* session);
      void (*on_release)(const char* bytes, size_t n, void* user);
      void* user;
   };

   template <class T>
   struct part_alloc
   {
      using value_type = T;
      using hkv_handle = detail::line_ptr;

      arena*   a   = nullptr;  // deref base + refcounts (any thread)
      session* ses = nullptr;  // allocation context (writer thread only);
                               // null for read-only snapshot views
      bool defer = false;      // writer-side trees may hand release-to-zero
                               // cascades to the background releaser
      const value_hooks* vhooks = nullptr;

      // ── large-value second arena (32 KiB grain; value_large leaves) ─────────
      // lses (writer-only) is the allocation context; the deref/free base comes
      // from a->companion() so EVERY allocator copy — views, drain, releaser —
      // resolves and frees a large leaf without carrying lses. The same COW /
      // refcount / recovery machinery the line arena has (large_arena IS the
      // grain-parameterized arena). value_large leaves free via the refcount
      // cascade (node_free_p_ -> deallocate_large), no version drain.
      large_session* lses = nullptr;

      part_alloc() = default;
      explicit part_alloc(session* s) noexcept : a(&s->store()), ses(s), defer(true) {}
      explicit part_alloc(session& s) noexcept : a(&s.store()), ses(&s), defer(true) {}
      explicit part_alloc(arena* ar) noexcept : a(ar) {}  // read-only views
      part_alloc(arena* ar, bool d) noexcept : a(ar), defer(d) {}  // writer temp views
      template <class U>
      part_alloc(const part_alloc<U>& o) noexcept
          : a(o.a), ses(o.ses), defer(o.defer), vhooks(o.vhooks), lses(o.lses)
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
      // Largest value the line arena can hold (64 lines): the tree decides
      // line-vs-large against this.
      static constexpr size_t large_threshold_bytes = size_t(max_alloc_lines) * line_bytes;
      static constexpr size_t large_max_bytes       = large_max_object_bytes;

      // Allocate a large leaf of `bytes`; returns its grain offset (from the
      // large arena base) — the tree packs it with from_grain_off + value_large.
      size_t allocate_large(size_t bytes)
      {
         large_arena*   la = large_();
         assert(la && lses && "large-value store has no large arena/session");
         (void)la;
         if (bytes > large_max_bytes)
            throw std::runtime_error("part: physical allocation exceeds the large-arena 2 MiB cap");
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
      T* allocate(size_t n)
      {
         static_assert(alignof(T) <= line_bytes, "part lines are 128-byte aligned");
         assert(ses && "read-only view allocators cannot allocate");
         const line_idx i = ses->alloc_lines(uint32_t(lines_(n * sizeof(T))));
         return static_cast<T*>(a->get(i));
      }
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

      // ── COW refcount hooks (consumed by the tree's part-COW extension) ─────
      // All arena-level: safe from any thread (cascades often run on a reader
      // dropping the last snapshot reference).
      static constexpr bool hkv_cow = true;
      uint8_t hkv_ref(uint32_t line) const noexcept { return a->ref(line); }
      bool hkv_defer(uint64_t raw) noexcept { return defer && a->try_defer(raw); }
      bool    hkv_retain(uint32_t line) noexcept { return a->retain(line); }
      template <class D>
      bool hkv_release(uint32_t line, D&& d) noexcept(noexcept(d(line)))
      {
         return a->release(line, static_cast<D&&>(d));
      }
      void hkv_free_run(uint32_t line, uint32_t n) noexcept { a->release_run(line, n); }

      // ── large-arena refcount hooks (value_large leaves) ────────────────────
      // The same per-line refcount liveness as the line arena, but indexed into
      // the LARGE arena by GRAIN. As with the line arena, the HEAD grain's count
      // is the object's share count (snapshot pins / COW clones retain the head);
      // the tail grains stay at their allocation count of 1 and are freed in one
      // run when the head dies. No version drain — a value_large leaf is freed
      // exactly when its head grain reaches 0.
      uint8_t hkv_large_ref(uint32_t grain) const noexcept
      {
         large_arena* la = large_();
         return la ? la->ref(grain) : 0;
      }
      bool hkv_large_retain(uint32_t grain) noexcept
      {
         large_arena* la = large_();
         return la ? la->retain(grain) : true;
      }
      // Release ONE reference to a `units`-grain large object at `grain`. When the
      // head grain reaches 0 the object dies: `d()` runs (the leaf-dies hook),
      // then all `units` grains are freed in one run. Mirrors the line arena's
      // head-line release + tail free_run.
      template <class D>
      void hkv_large_release(uint32_t grain, uint32_t units, D&& d) noexcept(noexcept(d()))
      {
         large_arena* la = large_();
         if (!la)
            return;
         // release() runs the destroy while the head grain is still live, then
         // frees it; fire d() there (it reads the leaf bytes) and free the tail.
         la->release(grain,
                     [&](line_idx)
                     {
                        d();  // object died: fire the leaf-dies hook once
                        // The tail grains carry the count of one they were
                        // allocated with (only the head is ever shared): they
                        // are released, not just freed, so that no free grain
                        // is left with a count.
                        if (units > 1)
                           la->release_run(grain + 1, units - 1);
                     });
      }
   };
}  // namespace start_ref

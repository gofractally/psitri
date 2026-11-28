#pragma once
#include <sal/alloc_header.hpp>
#include <sal/allocator.hpp>
#include <sal/allocator_impl.hpp>
#include <sal/allocator_session.hpp>
#include <sal/transaction.hpp>

namespace sal
{


   [[nodiscard]] inline read_lock allocator_session::lock() noexcept
   {
      return read_lock(*this);
   }

   template <typename T>
   T* allocator_session::get(location loc) noexcept
   {
      if constexpr (debug_memory)
      {
         /// TODO: add sanity checks here
         // verify that the location isn't beyond the end of allocated memory
         // verify that the location is not in an unallocated segment or
         // unallocated space within a segment
         // verify that the type can be converted to T
         // verify that if the location is not read only, then the session owns the segment
      }
      return reinterpret_cast<T*>(_block_base_ptr + *loc.offset());
   }
   /**
    * Allocates a node in the active segment and returns a pointer to the node, the
    * modify lock must be released after all writes are done on the allocated node.
    * 
    * The caller does not know what segment this will end up written to, so it cannot
    * write lock in advance. 

    * constructs T(size, std::forward<decltype(args)>(args)...)
    * 
    * @param size, must be a multiple of 64
    * @param vage, the virtual age of the object, if 0, the current time will be used
    * @param args, arguments to forward to the constructor of the object after size.
    */

   template <typename T>
   std::pair<location, T*> allocator_session::alloc_data(uint32_t size, auto&&... args)
   {
      size = ucc::round_up_multiple<cacheline_size>(size);   // every object is grain-aligned
      prepare_alloc(size, get_current_time_msec());
      auto head = _alloc_seg_ptr->alloc<T>(size, std::forward<decltype(args)>(args)...);
      auto loc  = location::from_absolute_address((char*)head - _block_base_ptr);
      return {loc, head};
   }  // namespace std::pair
   template <typename T>
   std::pair<location, T*> allocator_session::alloc_data_vage(uint32_t       size,
                                                              msec_timestamp vage,
                                                              auto&&... args)
   {
      size = ucc::round_up_multiple<cacheline_size>(size);   // every object is grain-aligned
      prepare_alloc(size, vage);
      auto head = _alloc_seg_ptr->alloc<T>(size, std::forward<decltype(args)>(args)...);
      auto loc  = location::from_absolute_address((char*)head - _block_base_ptr);
      return {loc, head};
   }
   inline void allocator_session::prepare_alloc(uint32_t size, msec_timestamp vage)
   {
      assert(size <= fresh_allocation_capacity);
      assert(size == ucc::round_up_multiple<cacheline_size>(size));

      if (not _alloc_seg_ptr) [[unlikely]]
         init_active_segment();

      if (not _alloc_seg_ptr->can_alloc(size)) [[unlikely]]
      {
         finalize_active_segment();  // final bookkeeping before getting a new segment
         init_active_segment();      // get a new segment
      }
      // Update the vage_accumulator with the current allocation
      /// TODO: make the caller provide the vage value rather than force the condition on every call
      _alloc_seg_ptr->age_accumulator.add(size, *vage);
   }

   /**
    * Generate a random number for cache decisions
    * 
    * @return A random 64-bit number
    */
   inline uint64_t allocator_session::get_random() noexcept
   {
      return _session_rng.next();
   }

   template <typename T>
   [[nodiscard]] ptr_address allocator_session::alloc(uint32_t   size,
                                                      alloc_hint hint,
                                                      auto&&... args) noexcept
   {
      assert(size >= sizeof(T));
      assert(size % 64 == 0);
      static_assert(std::is_base_of_v<alloc_header, T>, "T must derive from alloc_header");

      allocation palloc = _ptr_alloc.alloc(hint);
      SAL_CRASH_POINT("alloc:cb_claimed");

      auto [loc, node_ptr] =
          alloc_data<T>(size, palloc.addr_seq, std::forward<decltype(args)>(args)...);
      SAL_CRASH_POINT("alloc:data_written");

      palloc.ptr->store(control_block_data().set_loc(loc).set_ref(1),
                        std::memory_order_release);
      SAL_CRASH_POINT("alloc:cb_stored");

      return palloc.addr_seq.address;
   }

   template <typename T>
   [[nodiscard]] smart_ptr<T> allocator_session::smart_alloc(uint32_t   size,
                                                             alloc_hint hint,
                                                             auto&&... args) noexcept
   {
      return smart_ptr<T>(allocator_session_ptr(this, true),
                          alloc<T>(size, hint, std::forward<decltype(args)>(args)...));
   }

   /**
    *  Allocates new memory but reuses the control block from an existing object whose
    * reference count is 1, the space from the existing object will be freed because the
    * control block will be pointing to a new location.
    */
   template <typename To, typename From>
   [[nodiscard]] smart_ref<To> allocator_session::realloc(const smart_ref<From>& from,
                                                          uint32_t               size,
                                                          auto&&... args)
   {
      assert(size >= sizeof(To));
      assert(size % 64 == 0);
      static_assert(std::is_base_of_v<alloc_header, To>, "To must derive from alloc_header");
      static_assert(std::is_base_of_v<alloc_header, From>, "From must derive from alloc_header");

      assert(from.ref() == 1);

      auto [loc, node_ptr] =
          alloc_data<To>(size, from->address_seq(), std::forward<decltype(args)>(args)...);
      assert(loc != from.loc());

      control_block_data old_control = from.control().move(loc);
      // Some tree algorithms take one transient lifetime retain while cascading a
      // unique remove, then still relocate the same logical object. That extra
      // retain is not a second published owner; it is released before the mutation
      // unwinds back to the parent.
      assert(old_control.ref == 1 || old_control.ref == 2);

      assert(old_control.loc() != loc);
      //      SAL_WARN("realloc from: {:#x} to: {:#x} ", get<From>(old_control.loc()), get<To>(loc));

      // we have to refetch the pointer just incase the compactor moved the object
      // between when from read it and when we exchanged it.
      record_freed_space(get<From>(old_control.loc()), "session_realloc");

      return smart_ref<To>(get_session_ptr(), node_ptr, from.control(),
                           from.control().load(std::memory_order_relaxed));
   }
   /**
    * MVCC realloc: like realloc but allows ref > 1.
    * Atomically redirects a shared control block to new data constructed
    * with the given arguments.  Old data is freed (epoch-reclaimed).
    */
   template <typename To, typename From>
   [[nodiscard]] smart_ref<To> allocator_session::mvcc_realloc(const smart_ref<From>& from,
                                                                uint32_t               size,
                                                                auto&&... args)
   {
      assert(size >= sizeof(To));
      assert(size % 64 == 0);
      static_assert(std::is_base_of_v<alloc_header, To>, "To must derive from alloc_header");
      static_assert(std::is_base_of_v<alloc_header, From>, "From must derive from alloc_header");

      auto [loc, node_ptr] =
          alloc_data<To>(size, from->address_seq(), std::forward<decltype(args)>(args)...);
      assert(loc != from.loc());

      control_block_data old_control = from.control().move(loc);
      // No ref == 1 assertion — MVCC: other snapshots may reference this CB

      record_freed_space(get<From>(old_control.loc()), "session_mvcc_realloc");

      return smart_ref<To>(get_session_ptr(), node_ptr, from.control(),
                           from.control().load(std::memory_order_relaxed));
   }

   template <typename T>
   [[nodiscard]] T* allocator_session::copy_on_write(smart_ref<T>& ptr)
   {
      if (can_modify(ptr.loc()))
         return const_cast<T*>(ptr.obj());

      uint32_t asize;

      if constexpr (std::is_same_v<T, alloc_header>)
         asize = _sega.type_ops(ptr.obj()).cow_size(ptr.obj());
      else
         asize = ptr->cow_size();

      auto [loc, node_ptr] =
          alloc_data<alloc_header>(asize, (sal::header_type)ptr->type(), ptr->address_seq());

      _sega.type_ops(ptr.obj()).active_copy_to(ptr.obj(), node_ptr, get_session_ptr());

      SAL_CRASH_POINT("cow:copied");
      auto old_cached = ptr._control.move(loc);
      SAL_CRASH_POINT("cow:redirected");
      record_freed_space(get<alloc_header>(old_cached.loc()), "session_move");
      ptr._cached = old_cached;
      ptr._obj    = node_ptr;
      return static_cast<T*>(node_ptr);
   }

   template <typename T>
   [[nodiscard]] smart_ref<T> allocator_session::get_ref(ptr_address adr) noexcept
   {
      assert(!is_reserved_ptr_address(adr));
      auto& cb    = _ptr_alloc.get(adr);
      auto  cread = cb.load(std::memory_order_acquire);
      auto  ptr   = reinterpret_cast<T*>(_block_base_ptr + *cread.loc().offset());
      assert(int(T::type_id) == int(alloc_header::type_id) or int(T::type_id) == int(ptr->type()));
      if (_sega._mapped_state->_config.enable_read_cache && cread.ref > 0 &&
          is_read_only(cread.loc()) && should_cache(adr, ptr->size()) &&
          cb.try_mark_read_cache_pending())
      {
         (void)_rcache_queue.try_push(adr);
      }
      // assert(cread.ref > 0);
      return smart_ref<T>(get_session_ptr(), ptr, cb, cread);
   }

   /// Issue a hardware prefetch for the node at the given address.
   /// Resolves the control block → offset → mmap pointer, then prefetches
   /// the first cache line without constructing a smart_ref or touching
   /// the ref count.  Use this to pipeline page faults during sorted merges.
   inline void allocator_session::prefetch(ptr_address adr) const noexcept
   {
      if (is_reserved_ptr_address(adr)) [[unlikely]]   // null / tombstone / sentinel
         return;
      auto& cb    = _ptr_alloc.get(adr);
      auto  cread = cb.load(std::memory_order_relaxed);
      auto  ptr   = _block_base_ptr + *cread.loc().offset();
      __builtin_prefetch(ptr, 0, 3);
   }

   inline bool allocator_session::is_read_only(ptr_address adr) const noexcept
   {
      assert(!is_reserved_ptr_address(adr));
      auto& cb    = _ptr_alloc.get(adr);
      auto  cread = cb.load(std::memory_order_acquire);
      return is_read_only(cread.loc());
   }
   /**
    * Free an object in a segment
    * 
    * @param segment The segment number containing the object
    * @param object_size The size of the object to free
    */
   inline void allocator_session::record_freed_space(const alloc_header* obj,
                                                      const char* tag) noexcept
   {
      _sega.record_freed_space(_session_num, obj, tag);
   }

   /**
    * Check if a node location is read-only
    * 
    * @param loc The node location to check
    * @return true if the location is read-only, false otherwise
    */
   inline bool allocator_session::is_read_only(location loc) const noexcept
   {
      return _sega.is_read_only(loc);
   }

   /**
    * 
    */
   inline bool allocator_session::can_modify(location loc) const noexcept
   {
      return _sega.can_modify(_session_num, loc);
   }

   /**
    * Get the cache difficulty value which is used for determining read bit updates
    * 
    * @return The current cache difficulty value
    */
   inline uint64_t allocator_session::get_cache_difficulty() const noexcept
   {
      return _sega.get_cache_difficulty();
   }

   /**
    * Check if an object should be cached based on its size and difficulty threshold
    * 
    * @param adr The control block address being dereferenced
    * @param size The size of the object in bytes
    * @return true if the object should be cached, false otherwise
    */
   inline bool allocator_session::should_cache(ptr_address adr, uint32_t size) noexcept
   {
      return _sega._mapped_state->_cache_difficulty_state.should_cache(
          get_random() ^ uint64_t(*adr), size);
   }
   inline ptr_address allocator_session::alloc_custom_cb(uint64_t user_value) noexcept
   {
      allocation palloc = _ptr_alloc.alloc(alloc_hint{});

      // Store user value in the location field with {active=0, pending_cache=1} marker
      auto loc = location::from_cacheline(user_value);
      palloc.ptr->store(
          control_block_data().set_loc(loc).set_ref(1).set_pending_cache(true),
          std::memory_order_release);

      return palloc.addr_seq.address;
   }

   inline uint64_t allocator_session::read_custom_cb(ptr_address adr) const noexcept
   {
      auto& cb   = const_cast<control_block_alloc&>(_ptr_alloc).get(adr);
      auto  data = cb.load(std::memory_order_acquire);
      assert(is_custom_cb(data));
      return uint64_t(data.cacheline_offset);
   }

   inline std::optional<uint64_t>
   allocator_session::try_read_custom_cb(ptr_address adr) const noexcept
   {
      auto* cb = const_cast<control_block_alloc&>(_ptr_alloc).try_get(adr);
      if (!cb)
         return std::nullopt;

      auto data = cb->load(std::memory_order_acquire);
      if (!is_custom_cb(data))
         return std::nullopt;

      return uint64_t(data.cacheline_offset);
   }

   inline bool allocator_session::is_custom_cb(control_block_data cbd) noexcept
   {
      return !cbd.active && cbd.pending_cache;
   }

   inline uint32_t allocator_session::custom_cb_share(ptr_address adr) const noexcept
   {
      return const_cast<control_block_alloc&>(_ptr_alloc).get(adr).ref();
   }

   inline bool allocator_session::custom_cb_cas_value(ptr_address adr,
                                                      uint64_t    expect,
                                                      uint64_t    desired) noexcept
   {
      return get(adr).cas_user_value(expect, desired);
   }

   inline void allocator_session::retain(ptr_address adr)
   {
      get(adr).retain();
   }

   // ── per-range branch-set refcounts: forward to the allocator's implicit_ref_pool ──
   inline ptr_address allocator_session::rr_alloc_fresh() { return _sega.iref().alloc_fresh(); }
   inline std::optional<ptr_address> allocator_session::rr_alloc_in_region(const implicit_ref_hint& h)
   {
      return _sega.iref().alloc_in_region(h);
   }
   inline bool    allocator_session::rr_try_retain(ptr_address a) { return _sega.iref().try_retain(a); }
   inline bool    allocator_session::rr_release(ptr_address a) { return _sega.iref().release(a); }
   inline uint8_t allocator_session::rr_count(ptr_address a) { return _sega.iref().count(a); }
   // Inline-destroy node budget per top-level release before the cascade spills to the background
   // queue (caps the worst-case writer latency of a single release — a big subtree/snapshot drop
   // can't stall the writer unboundedly). A COW-to-root path is a handful of nodes, so the common
   // drop is always fully inline; only a rare large drop spills the remainder.
   inline constexpr uint32_t inline_destroy_budget = 256;

#ifdef SAL_RELEASE_DEBUG_COUNTERS
   // Opt-in flow accounting for the bounded-inline release / background-release pipeline
   // (leak hunts). Compiled out unless the TU defines SAL_RELEASE_DEBUG_COUNTERS.
   inline std::atomic<uint64_t> g_sal_spill_push{0};           // budget-0 spills queued
   inline std::atomic<uint64_t> g_sal_spill_fallthrough{0};    // queue full → destroyed inline
   inline std::atomic<uint64_t> g_sal_inline_destroy{0};       // release() destroys
   inline std::atomic<uint64_t> g_sal_final_release{0};        // bg pops processed
   inline std::atomic<uint64_t> g_sal_final_release_early{0};  // bg pop: ref>1 → no destroy
   inline std::atomic<uint64_t> g_sal_final_destroy{0};        // bg destroys
   inline std::atomic<uint64_t> g_sal_final_null_loc{0};       // bg pop: NULL LOC → destroy skipped
   inline std::atomic<uint64_t> g_sal_producer_queue{0};       // &_release_queue the spiller wrote
   inline std::atomic<uint64_t> g_sal_producer_session{0};     // spiller's session number
   inline std::atomic<uint64_t> g_sal_consumer_queue0{0};      // &queue(0) the bg loop reads
   inline std::atomic<uint64_t> g_sal_consumer_queue1{0};      // &queue(1) the bg loop reads
#endif

   namespace detail
   {
      /// For tests: called on the last-reference path of allocator_session::release at the
      /// point where the object is committed to being destroyed but must still be counted:
      /// just before its reference is handed to the release queue (deferred == true), or
      /// just before the count is dropped under the read lock (deferred == false). Null in
      /// production; one relaxed load per last-reference release.
      using release_probe_fn = void (*)(const control_block& cb, bool deferred) noexcept;
      inline std::atomic<release_probe_fn> release_last_reference_probe{nullptr};
   }  // namespace detail

   inline void allocator_session::release(ptr_address adr) noexcept
   {
      auto& cb = get(adr);
      // Others still own it: one decrement and done.
      if (cb.release_if_shared())
      {
         SAL_CRASH_POINT("release:shared_decremented");
         return;
      }
      // We hold the ONLY reference, so no other thread can hold or acquire one and the count
      // stays at 1 until this thread changes it. It must not reach zero before the object's
      // bytes are safe from recycling: the compactor treats a zero count as already freed and
      // leaves the object behind when it empties the segment, after which the segment can be
      // reused while the destroy below, or the release thread's, still reads the object from
      // it. Reading recycled bytes as a node releases references that belong to live objects
      // of another tree. So: either the reference is handed on as it is, or the read lock is
      // taken first and only then is the count dropped.
      auto pre = cb.load(std::memory_order_relaxed);  // custom marker, before the decrement
      if (is_custom_cb(pre))  // custom CB: no object to destroy, just notify + free
      {
         (void)cb.release();
         if (_sega._on_custom_cb_released) [[unlikely]]
            _sega._on_custom_cb_released(pre.cacheline_offset);
         _ptr_alloc.free(adr);
         return;
      }
      // Mid-cascade and out of budget → spill: hand our one reference to the background
      // release thread so a huge drop can't stall this thread unboundedly. A session marked
      // defer_destroy (read sessions: dropping a pinned snapshot must never run the dead
      // lineage's destroy cascade on the reader thread) spills EVERY last-ref release. The
      // count is not touched: the queue entry now owns the reference, and the release thread
      // drops it under its own read lock.
      if (_defer_destroy || _destroy_budget == 0)
      {
         if (auto probe = detail::release_last_reference_probe.load(std::memory_order_relaxed))
             [[unlikely]]
            probe(cb, true);
         if (_release_queue.try_push(adr)) [[likely]]
         {
            SAL_CRASH_POINT("release:queued");
#ifdef SAL_RELEASE_DEBUG_COUNTERS
            ++g_sal_spill_push;
            g_sal_producer_queue   = (uint64_t)&_release_queue;
            g_sal_producer_session = *_session_num;
#endif
            return;
         }
         // queue full (rare): destroy inline. We still hold the one reference.
#ifdef SAL_RELEASE_DEBUG_COUNTERS
         ++g_sal_spill_fallthrough;
#endif
#if PSITRI_DEEP_INVARIANTS
         fprintf(stderr, "SPILL-FALLTHROUGH: adr=%u queue full — destroying inline\n", *adr);
#endif
      }
      const bool top = (_destroy_budget == UINT32_MAX);
      if (top)
      {
         _destroy_budget = inline_destroy_budget;
         retain_read_lock();  // block the compactor from recycling/relocating segments we read
      }
      if (_destroy_budget != UINT32_MAX && _destroy_budget != 0)
         --_destroy_budget;

      // The read lock is held (taken above when this is the top of a cascade, by the top
      // otherwise), so the bytes cannot be recycled from now on. Drop the last reference: the
      // word returned carries the object's CURRENT location, wherever the compactor moved it
      // while it was still referenced; at zero it is not moved again.
      if (auto probe = detail::release_last_reference_probe.load(std::memory_order_relaxed))
          [[unlikely]]
         probe(cb, false);
      SAL_CRASH_POINT("release:before_last_decrement");
      auto prev = cb.release();
      SAL_CRASH_POINT("release:at_zero");

      // Destroy `adr` via the fast templated per-tier destroy (NOT the std::function vtable
      // visit_children) — it releases each child through release() above, so the cascade unwinds
      // depth-first and bounds itself on _destroy_budget.
      this->retain_session();
      allocator_session_ptr ses_ptr(this);
      const alloc_header*   node = get<alloc_header>(prev.loc());
      _sega.type_ops(node).destroy(node, ses_ptr);
      SAL_CRASH_POINT("release:children_released");
      record_freed_space(node, "release_destroy");
      SAL_CRASH_POINT("release:space_freed");
#ifdef SAL_RELEASE_DEBUG_COUNTERS
      ++g_sal_inline_destroy;
#endif
      _ptr_alloc.free(adr);
      SAL_CRASH_POINT("release:cb_freed");

      if (top)
      {
         _destroy_budget = UINT32_MAX;
         release_read_lock();
      }
   }
   inline void allocator_session::final_release(ptr_address adr) noexcept
   {
      auto& cb  = get(adr);
      // Read custom marker BEFORE release() clears the bits
      auto  pre = cb.load(std::memory_order_relaxed);
      bool  custom = is_custom_cb(pre);

      auto  prev = cb.release();
      SAL_CRASH_POINT("final_release:decremented");
#ifdef SAL_RELEASE_DEBUG_COUNTERS
      ++g_sal_final_release;
      if (prev.ref > 1)
         ++g_sal_final_release_early;
#endif
      if (prev.ref > 1)
         return;

      if (!custom) [[likely]]
      {
         location loc = prev.loc();
         if (loc != location::null()) [[likely]]
         {
            this->retain_session();
            allocator_session_ptr ptr(this);

            const alloc_header* nptr = get<alloc_header>(loc);
            _sega.type_ops(nptr).destroy(nptr, ptr);
            SAL_CRASH_POINT("final_release:children_released");
            record_freed_space(nptr, "release_destroy");
            SAL_CRASH_POINT("final_release:space_freed");
#ifdef SAL_RELEASE_DEBUG_COUNTERS
            ++g_sal_final_destroy;
#endif
         }
#ifdef SAL_RELEASE_DEBUG_COUNTERS
         else
            ++g_sal_final_null_loc;  // destroy SKIPPED — children of this node strand
#endif
      }
      else if (_sega._on_custom_cb_released) [[unlikely]]
      {
         // Custom CB released — notify callback with stored version number
         _sega._on_custom_cb_released(pre.cacheline_offset);
      }
      _ptr_alloc.free(adr);
      SAL_CRASH_POINT("final_release:cb_freed");
   }

   /// called by the allocator_session_ptr destructor to release the session,
   /// notifies the allocator that the session is no longer in use when counter reaches 0
   inline void allocator_session::end_session()
   {
      //      SAL_INFO("allocator_session: end_session: {} {} ref: {}", this, _session_num, _ref_count);
      if (--_ref_count == 0)
      {
         SAL_INFO("allocator_session: end_session: {} {} ref: {}", this, _session_num, _ref_count);
         delete this;  // dtor releases the session number to the pool
      }
   }

   template <typename T>
   smart_ptr<T> allocator_session::get_root(root_object_number ro) noexcept
   {
      auto tid = _sega.get(ro);
      SAL_CRASH_POINT("root:get_retained");
      return smart_ptr<T>(allocator_session_ptr(this, true), tid.root, false);
   }

   inline uint64_t allocator_session::next_root_commit(root_object_number ro) const noexcept
   {
      return _sega.next_root_commit(ro);
   }

   inline sync_root_info make_sync_root_info(const allocator_session& session,
                                             root_object_number       ro,
                                             tree_id                  tid) noexcept
   {
      // root_version numbers the commits of this root: one more than the number in the
      // root's record. The same number goes into the record with this commit and into every
      // sync header the commit writes, so power-loss recovery can tell which of the two saw
      // the newer commit of a root. (A header is gone once its segment is compacted and
      // recycled, so the newest header that survives may be older than the roots file.)
      // It is read from the record the commit is about to write: no clock, no counter
      // shared between roots, nothing a reader's cache line holds that was not already
      // written by every commit.
      return sync_root_info{*ro, *tid.root, *null_ptr_address, session.next_root_commit(ro)};
   }

   template <typename T>
   inline smart_ptr<T> allocator_session::set_root(root_object_number ro,
                                                   smart_ptr<T>       ptr,
                                                   sync_type          st) noexcept
   {
      tree_id new_tid{ptr.take()};
      auto root_info = make_sync_root_info(*this, ro, new_tid);
      SAL_CRASH_POINT("root:publish_begin");
      sync(st, _sega._mapped_state->_config, root_info);
      SAL_CRASH_POINT("root:publish_synced");
      auto old_tid = _sega.set(ro, new_tid, st, root_info.root_version);
      SAL_CRASH_POINT("root:publish_stored");
      return smart_ptr<T>(allocator_session_ptr(this, true), old_tid.root, false);
   }

   template <typename T, typename U>
   inline smart_ptr<T> allocator_session::cas_root(root_object_number ro,
                                                   smart_ptr<T>       expect,
                                                   smart_ptr<U>       desired,
                                                   sync_type          st) noexcept
   {
      tree_id expect_tid{expect.address()};
      tree_id desired_tid{desired.address()};
      auto    root_info = make_sync_root_info(*this, ro, desired_tid);
      if (_sega.cas_root(ro, expect_tid, desired_tid, st,
                         root_info.root_version,
                         [this, st, &root_info]() noexcept
                         {
                            sync(st, _sega._mapped_state->_config, root_info);
                         }))
      {
         desired.take();  // _sega took it, so don't release it
         // let the caller determine how and when to release prior value
         return smart_ptr<T>(allocator_session_ptr(this, true), expect_tid.root, false);
      }
      return smart_ptr<T>();
   }

   inline transaction_ptr allocator_session::start_transaction(root_object_number ro) noexcept
   {
      return std::make_unique<transaction>(allocator_session_ptr(this, true), ro);
   }

   inline allocator_session_ptr allocator_session::get_session_ptr() noexcept
   {
      return allocator_session_ptr(this, true);
   }

   inline smart_ptr<alloc_header> allocator_session::transaction_commit(
       root_object_number      ro,
       smart_ptr<alloc_header> desired,
                                                   sync_type               st) noexcept
   {
      tree_id new_tid{desired.take()};
      auto root_info = make_sync_root_info(*this, ro, new_tid);
      sync(st, _sega._mapped_state->_config, root_info);
      auto old_tid =
          _sega.transaction_commit(ro, new_tid, st, root_info.root_version);
      return smart_ptr<alloc_header>(get_session_ptr(), old_tid.root, false);
   }
   inline void allocator_session::transaction_abort(root_object_number ro) noexcept
   {
      _sega.transaction_abort(ro);
   }

   inline void allocator_session::retain_read_lock() noexcept
   {
      if (_nested_read_lock++)
         return;
      _session_rlock.lock();
      SAL_CRASH_POINT("rlock:locked");
   }
   inline void allocator_session::release_read_lock() noexcept
   {
      assert(_nested_read_lock > 0);
      if (--_nested_read_lock)
         return;
      _session_rlock.unlock();
      SAL_CRASH_POINT("rlock:unlocked");
   }

   inline uint64_t allocator_session::get_total_allocated_objects() const noexcept
   {
      return _ptr_alloc.used();
   }

   inline uint64_t allocator_session::get_pending_release_count() const noexcept
   {
      return _release_queue.usage();
   }

   inline allocator_session::operation_scope
   allocator_session::record_operation(mapped_memory::session_operation op,
                                       uint64_t                         count) noexcept
   {
      const bool is_outer_operation = detail::session_operation_depth == 0;
      ++detail::session_operation_depth;
      if (is_outer_operation)
         _sega.record_session_operation(_session_num, op, count);
      return operation_scope(true);
   }

   template <typename Visitor>
   void allocator_session::for_each_live_object(Visitor&& visitor) const noexcept
   {
      _ptr_alloc.for_each_allocated(
          [&](ptr_address adr, uint32_t ref)
          {
             auto& cb   = const_cast<control_block_alloc&>(_ptr_alloc).get(adr);
             auto  data = cb.load(std::memory_order_relaxed);
             // Skip custom control blocks (no segment data to inspect)
             if (is_custom_cb(data))
                return;
             auto* obj  = reinterpret_cast<alloc_header*>(_block_base_ptr + *data.loc().offset());
             visitor(adr, ref, obj);
          });
   }

}  // namespace sal

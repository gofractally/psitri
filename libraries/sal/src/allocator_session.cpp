#include <sal/allocator.hpp>
#include <sal/allocator_impl.hpp>
#include <sal/allocator_session.hpp>
#include <sal/mapped_memory/read_lock_queue.hpp>
#include <sal/mapped_memory/segment_impl.hpp>
#include <sal/time.hpp>
#include <ucc/round.hpp>
#include <chrono>

namespace sal
{

   allocator_session::allocator_session(allocator& a, allocator_session_number ses_num)
       : _block_base_ptr(a._block_alloc.get(offset_ptr(0))),
         _control_block_base_ptr(a._ptr_alloc.try_get(ptr_address(0))),
         _nested_read_lock(0),
         _session_num(ses_num),
         _rcache_queue(a.get_rcache_queue(ses_num)),
         _release_queue(a.get_release_queue(ses_num)),
         _sega(a),
         _ptr_alloc(a._ptr_alloc),
         _alloc_seg_ptr(nullptr),
         _session_rng(uint64_t(reinterpret_cast<uintptr_t>(this)) ^
                      (uint64_t(reinterpret_cast<uintptr_t>(&a)) << 17) ^
                      (uint64_t(*ses_num) << 32) ^ 0xABBA7777ull),
         _dirty_segments(a._mapped_state->_session_data.dirty_segments(ses_num)),
         _alloc_seg_num(-1),
         _session_rlock(a.get_session_rlock(ses_num))
   {
      SAL_INFO("allocator_session: constructor: {} {} ref: {}", this, ses_num, _ref_count);
      _block_base_ptr = a._block_alloc.get(offset_ptr(0));
   }

   allocator_session::~allocator_session()
   {
      SAL_INFO("allocator_session: destructor: {} {} ref: {}", this, _session_num, _ref_count);
      if (_session_num == allocator_session_number(-1))
         return;
      finalize_active_segment();
      // The queue belongs to the session NUMBER, in mapped state. Left
      // undrained, the segment just finalized stays flagged active for good:
      // the next open resets the queue, and a later session that is given
      // this number only finds it if it happens to sync. So it is drained
      // here, at the allocator's configured sync level: everything committed
      // in the segment was already synced at the level its commit asked for,
      // and what follows the last commit is not part of any commit.
      try
      {
         drain_dirty_segments(_sega._mapped_state->_config.sync_mode, _sega._mapped_state->_config,
                              {});
      }
      catch (...)
      {
         // A destructor cannot report it; the segments stay flagged active
         // (never compacted), which wastes space and loses nothing.
      }
      _sega.release_session_num(_session_num);
      _session_num = allocator_session_number(-1);
   }

   void allocator_session::finalize_active_segment()
   {
      // sync() owns terminal-marker and trailing-space accounting when this
      // dirty segment is drained. Rollover callers (including chunk placement)
      // must not also credit the skipped tail as freed space here.
      if (_alloc_seg_ptr)
      {
         if (not _alloc_seg_ptr->is_finalized())
            _alloc_seg_ptr->finalize();
         _dirty_segments.push(_alloc_seg_num);
      }
      _alloc_seg_ptr = nullptr;
      _alloc_seg_num = segment_number(-1);
   }

   void allocator_session::init_active_segment()
   {
      auto t0 = std::chrono::steady_clock::now();

      auto [num, ptr] = _sega.get_new_segment(_alloc_to_pinned);
      _alloc_seg_num  = num;
      _alloc_seg_ptr  = ptr;

      // Initialize segment header with session information
      _alloc_seg_ptr->_session_id = _session_num;
      _alloc_seg_ptr->_seg_sequence =
          _sega._mapped_state->_session_data.next_session_segment_seq(_session_num);
      _alloc_seg_ptr->_open_time_usec  = sal::get_current_time_msec();
      _alloc_seg_ptr->_close_time_usec = msec_timestamp(0);  // Will be set when segment is closed

      ++_seg_alloc_count;
      _seg_alloc_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - t0).count();
   }

   /**
    * Reclaims the most recently allocated size bytes
    * 
    * @param size The size of the object to reclaim
    * @return true if the object was reclaimed, false otherwise
    */
   bool allocator_session::unalloc(uint32_t size)
   {
      if (not _alloc_seg_ptr) [[unlikely]]
         return false;

      size = ucc::round_up_multiple<cacheline_size>(size);   // inverse of alloc_data's grain rounding

      assert(reinterpret_cast<alloc_header*>(_alloc_seg_ptr->data +
                                             _alloc_seg_ptr->get_alloc_pos() - size)
                 ->size() == size);

      _alloc_seg_ptr->unalloc(size);
      return true;
   }
   void allocator_session::sync(sync_type st)
   {
      sync(st, _sega._mapped_state->_config, {});
   }

   void allocator_session::sync(sync_type st, const runtime_config& cfg, std::span<char> user_data)
   {
      // Sync the active segment to advance _first_writable_page past all
      // committed data. Without this, is_read_only() returns false for
      // objects in the current segment even after transaction commit.
      if (_alloc_seg_ptr and not _alloc_seg_ptr->is_finalized() and
          (_alloc_seg_ptr->get_alloc_pos() > _alloc_seg_ptr->get_first_write_pos() ||
           !user_data.empty()))
      {
         auto pos_before = _alloc_seg_ptr->get_alloc_pos();
         _sega.record_session_write(_session_num,
                                    _alloc_seg_ptr->sync(st, cfg, user_data));
         // Count sync header padding as reclaimable space so the compactor
         // knows segments filled with many small commits are mostly free.
         auto sync_hdr_size = _alloc_seg_ptr->get_alloc_pos() - pos_before;
         _sega._mapped_state->_segment_data.add_freed_space(
             _alloc_seg_num,
             _alloc_seg_ptr,
             _alloc_seg_ptr->data + pos_before,
             sync_hdr_size,
             "active_sync_padding");
      }

      drain_dirty_segments(st, cfg, user_data);
      _sega.sync(st);
   }

   // Every segment this session finalized since its last sync: write its
   // terminal sync header, make it read-only, sync it at `st`, and hand it to
   // the compactor. This runs whether or not the session has an active
   // segment: after next_allocation_segment() (a big value's chunk placement)
   // it has none, and the commit that follows must still make what it wrote
   // read-only and durable.
   void allocator_session::drain_dirty_segments(sync_type             st,
                                                const runtime_config& cfg,
                                                std::span<char>       user_data)
   {
      segment_number seg_num = _dirty_segments.pop();
      while (seg_num != segment_number(-1))
      {
         auto seg = _sega.get_segment(seg_num);

         // Snapshot alloc_pos before sync(). If sync()'s fall-through path
         // wrote a small terminal sync_header (the partial-segment finalize
         // case), the unused tail [pos_before_terminal, segment_size - footer)
         // is reclaimable by the compactor and must be recorded as
         // freed_space exactly once. The caller — not segment::sync() — owns
         // this accounting because the segment-side sync_header's size()
         // intentionally only covers the marker itself.
         const uint32_t pos_before = seg->get_alloc_pos();
         const bool was_partial_finalized =
             seg->is_finalized() and not seg->is_read_only();

         const auto bytes_synced = seg->sync(st, cfg, user_data);
         _sega.record_session_write(_session_num, bytes_synced);

         // Only attribute last_aheader to freed_space when sync() actually
         // wrote a new sync_header in *this* drain pass. The natural-fill
         // path already accounted for the terminal sync_header in the
         // active-segment block of allocator_session::sync; calling
         // record_freed_space on the same last_aheader during the drain
         // would double-count its bytes against this segment's freed_space
         // counter and trip the overflow assertion when live data is small
         // (heavy-COW workloads like Bitcoin Core's chainstate flush).
         if (bytes_synced > 0)
         {
            auto ahead = seg->get_last_aheader();
            _sega.record_freed_space(_session_num, ahead, "drain_terminal_sync_header");
         }

         // If this drain pass transitioned a partial-finalized segment to
         // fully read-only via the fall-through path, the bytes past the
         // terminal sync_header are unallocated tail — reclaimable by the
         // compactor, and must be recorded once. (Must match segment_impl's grain.)
         constexpr uint32_t terminal_size = sal::cacheline_size;
         const bool wrote_fall_through_terminal =
             was_partial_finalized and seg->is_read_only() and bytes_synced > 0 and
             seg->get_alloc_pos() == pos_before + terminal_size and
             seg->get_last_aheader()->size() == terminal_size;
         if (wrote_fall_through_terminal)
         {
            constexpr uint32_t footer        = mapped_memory::segment_footer_size;
            const uint32_t     tail_start    = pos_before + terminal_size;
            if (tail_start + footer < segment_size)
            {
               const uint32_t tail = segment_size - footer - tail_start;
               _sega._mapped_state->_segment_data.add_freed_space(
                   seg_num,
                   seg,
                   seg->data + tail_start,
                   tail,
                   "drain_fall_through_tail");
            }
         }

         assert(seg->is_finalized() ? seg->is_read_only() : true);
         if (seg->is_read_only())
            _sega._mapped_state->_segment_data.prepare_for_compaction(
                seg_num, seg->age_accumulator.average());
         seg_num = _dirty_segments.pop();
      }
   }
}  // namespace sal

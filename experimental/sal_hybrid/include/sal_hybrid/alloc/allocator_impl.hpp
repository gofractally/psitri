#pragma once
#include <sal_hybrid/alloc/allocator_session_impl.hpp>

namespace sal_hybrid
{
   inline void allocator::release(ptr_address adr) noexcept
   {
      // Sessionless SYNCHRONOUS release: a mutex-guarded shared utility
      // session — legal under the session contract (two threads must not be
      // ACTIVE simultaneously; serialized use is fine). Callers with their
      // own session should use it directly; truly-async callers can
      // post_release() to the release thread instead.
      std::lock_guard g(_utility_mx);
      if (!_utility_init)
      {
         _utility_ses  = create_session();
         _utility_init = true;
      }
      _utility_ses->release(adr);
   }
   /**
        * When an object is moved its space is freed and we need to record the freed space
        * so the compactor has the metadata it needs to efficiently identify segments that
        * can be compacted.
        * 
        * @param obj The object on the segment being freed
        * @param seg The segment number containing the object
        */
   template <typename T>
   inline void allocator::record_freed_space(allocator_session_number /*ses_num*/, T* obj,
                                              const char* tag)
   {
      auto seg_num = get_segment_for_object(obj);
      auto* seg    = get_segment(seg_num);
      // alloc_pos is read INSIDE add_freed_space under SAL_TRACK_LOCK to
      // avoid a race with concurrent main-thread allocation.
      _mapped_state->_segment_data.add_freed_space(seg_num, seg, obj, tag);
   }
}  // namespace sal_hybrid
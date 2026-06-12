#pragma once
#include <psitri/database.hpp>
#include <psitri/database_impl.hpp>
#include <psitri/read_session.hpp>
#include <sal/allocator_session_impl.hpp>

namespace psitri
{

   template <class LockPolicy>
   inline basic_read_session<LockPolicy>::basic_read_session(database_type& db)
       : _db(db.shared_from_this()),
         _allocator_session(db._allocator.create_session())
   {
   }

   template <class LockPolicy>
   inline tree basic_read_session<LockPolicy>::get_root(uint32_t root_index)
   {
      return tree(_allocator_session->template get_root<>(
          sal::root_object_number(root_index)));
   }

   template <class LockPolicy>
   inline cursor basic_read_session<LockPolicy>::create_cursor(uint32_t root_index)
   {
      // MVCC removed: no per-snapshot version; the cursor sees the single (version-0) state.
      return cursor(get_root(root_index));
   }

   template <class LockPolicy>
   inline cursor basic_read_session<LockPolicy>::snapshot_cursor(uint32_t root_index)
   {
      return create_cursor(root_index);
   }

}  // namespace psitri

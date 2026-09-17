#pragma once
#include <psitri/database_impl.hpp>
#include <psitri/dwal/dwal_database.hpp>
#include <psitri/dwal/merge_cursor.hpp>
#include <psitri/read_session_impl.hpp>
#include <mutex>

namespace psitri::dwal
{
   /// A committed byte-value snapshot across RW, frozen RO, and the base tree.
   /// Capturing advances COW once under the root transaction lock; reads retain
   /// their arena owners and fixed roots without holding a writer lock.
   class read_snapshot
   {
     public:
      read_snapshot(dwal_database& db, uint32_t index)
          : session_(db.underlying_db()->start_read_session())
      {
         auto& state = db.root(index);
         std::unique_lock lock(state.tx_mutex);
         rw_ = state.rw_layer;
         if (rw_)
         {
            rw_view_.emplace(*rw_);
            rw_->map.bump_cow_seq();
         }
         {
            std::lock_guard buffered_lock(state.buffered_mutex);
            ro_ = state.buffered_ptr;
         }
         if (ro_) ro_view_.emplace(*ro_);
         root_ = session_->get_root(index).copy_root();
      }

      bool get(std::string_view key, std::string* value) const
      {
         for (auto* layer : {&rw_view_, &ro_view_})
         {
            if (!*layer) continue;
            if (auto* entry = (*layer)->map.get(key))
            {
               if (entry->is_tombstone()) return false;
               if (value) value->assign(entry->data);
               return true;
            }
            if ((*layer)->tombstones.is_deleted(key)) return false;
         }
         psitri::cursor cursor(root_);
         return cursor.get(key, value) >= 0;
      }

      owned_merge_cursor create_cursor() const
      {
         return {rw_, ro_, rw_view_, ro_view_, psitri::cursor(root_)};
      }

     private:
      std::shared_ptr<psitri::read_session> session_;
      sal::smart_ptr<sal::alloc_header> root_;
      std::shared_ptr<btree_layer> rw_, ro_;
      std::optional<btree_layer_view> rw_view_, ro_view_;
   };
}

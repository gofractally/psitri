#pragma once
// Internal byte-key/value adapter for DWAL consumers such as the MDBX shim.
// This is not the public psitri::engine concept: persistent subtree snapshots
// and synchronous commit policies are not exposed by this adapter.
#include <psitri/dwal/dwal_database_impl.hpp>
#include <psitri/dwal/merge_cursor_impl.hpp>
#include <unordered_map>

namespace psitri::dwal
{
   template <class Base>
   struct byte_engine
   {
      using dwal_type = basic_dwal_database<std_lock_policy, Base>;
      struct config
      {
         typename Base::config base;
         dwal_config buffering;
      };
      class database;

      struct snapshot
      {
         std::shared_ptr<database> owner;
         std::shared_ptr<typename Base::read_session> session;
         std::shared_ptr<btree_layer> rw, ro;
         std::optional<btree_layer_view> rw_view, ro_view;
         std::optional<typename Base::cursor> base;

         snapshot(std::shared_ptr<database> db, uint32_t index, bool writer_owned = false)
             : owner(std::move(db)), session(owner->dwal.underlying_db()->start_read_session())
         {
            auto& state = owner->dwal.root(index);
            std::unique_lock lock(state.tx_mutex, std::defer_lock);
            if (!writer_owned) lock.lock();
            rw = state.rw_layer;
            if (rw)
            {
               rw_view.emplace(*rw);
               rw->map.bump_cow_seq();
            }
            std::lock_guard buffered_lock(state.buffered_mutex);
            ro = state.buffered_ptr;
            if (ro) ro_view.emplace(*ro);
            base.emplace(session->create_cursor(index));
         }

         template <class F>
         bool get(std::string_view key, F&& fn)
         {
            for (auto* layer : {&rw_view, &ro_view})
            {
               if (!*layer) continue;
               if (auto* value = (*layer)->map.get(key))
               {
                  if (!value->is_data()) return false;
                  fn(value->data);
                  return true;
               }
               if ((*layer)->tombstones.is_deleted(key)) return false;
            }
            return base->get(key, std::forward<F>(fn));
         }
      };

      class cursor
      {
         std::shared_ptr<snapshot> view_;
         std::optional<basic_merge_cursor<Base>> merge_;
         auto& ordered()
         {
            if (!merge_)
               merge_.emplace(view_->rw_view, view_->ro_view, *view_->base);
            return *merge_;
         }
       public:
         explicit cursor(std::shared_ptr<snapshot> view) : view_(std::move(view)) {}
         bool seek_begin() { return ordered().seek_begin(); }
         bool seek_last() { return ordered().seek_last(); }
         bool lower_bound(std::string_view k) { return ordered().lower_bound(k); }
         // The shim's base protocol defines upper_bound as the floor.
         bool upper_bound(std::string_view k)
         {
            if (lower_bound(k)) return key() == k || prev();
            return seek_last();
         }
         bool next() { return ordered().next(); }
         bool prev() { return ordered().prev(); }
         bool is_end() const { return !merge_ || merge_->is_end(); }
         bool is_rend() const { return !merge_ || merge_->is_rend(); }
         std::string_view key() const { return merge_->key(); }
         template <class F> bool get(std::string_view k, F&& fn)
         { return view_->get(k, std::forward<F>(fn)); }
         template <class T> std::optional<T> get(std::string_view k)
         {
            std::optional<T> result;
            get(k, [&](std::string_view v) { result.emplace(v); });
            return result;
         }
         template <class T> std::optional<T> value()
         {
            if (is_end() || is_rend() || merge_->is_subtree()) return {};
            if (merge_->current_source() == basic_merge_cursor<Base>::source::tri)
               return merge_->tri_cursor()->template value<T>();
            return T(merge_->current_value().data);
         }
      };

      class read_session
      {
         std::shared_ptr<database> db_;
         std::unordered_map<uint32_t, std::shared_ptr<snapshot>> snapshots_;
       public:
         explicit read_session(std::shared_ptr<database> db) : db_(std::move(db)) {}
         cursor create_cursor(uint32_t index)
         {
            auto& view = snapshots_[index];
            if (!view) view = std::make_shared<snapshot>(db_, index);
            return cursor(view);
         }
      };

      class transaction
      {
         std::shared_ptr<database> db_;
         uint32_t index_;
         typename dwal_type::dwal_transaction_type tx_;
       public:
         transaction(std::shared_ptr<database> db, uint32_t index)
             : db_(std::move(db)), index_(index), tx_(db_->dwal.start_write_transaction(index)) {}
         transaction(transaction&&) = default;
         void upsert(std::string_view k, std::string_view v) { tx_.upsert(k, v); }
         bool remove(std::string_view k) { return tx_.remove(k); }
         template <class F> bool get(std::string_view k, F&& fn) const
         {
            auto result = tx_.get(k);
            if (!result.found || !result.value.is_data()) return false;
            fn(result.value.data);
            return true;
         }
         template <class T> std::optional<T> get(std::string_view k) const
         {
            std::optional<T> result;
            get(k, [&](std::string_view v) { result.emplace(v); });
            return result;
         }
         cursor create_cursor()
         { return cursor(std::make_shared<snapshot>(db_, index_, true)); }
         void commit()
         {
            // Submit the record while DWAL still owns the root's writer lock.
            tx_.commit(sal::sync_type::none);
         }
         void abort() { tx_.abort(); }
      };

      class write_session
      {
         std::shared_ptr<database> db_;
       public:
         explicit write_session(std::shared_ptr<database> db) : db_(std::move(db)) {}
         transaction start_transaction(uint32_t index) { return transaction(db_, index); }
      };

      class database : public std::enable_shared_from_this<database>
      {
       public:
         dwal_type dwal;
         database(std::shared_ptr<typename Base::database> base,
                  const std::filesystem::path& path, const dwal_config& cfg)
             : dwal(std::move(base), path / "wal", cfg) {}
         static std::shared_ptr<database> open(const std::filesystem::path& path,
                                               sal::open_mode mode, const config& cfg)
         {
            auto base = Base::database::open(path, mode, cfg.base);
            return std::make_shared<database>(std::move(base), path, cfg.buffering);
         }
         std::shared_ptr<write_session> start_write_session()
         { return std::make_shared<write_session>(this->shared_from_this()); }
         std::shared_ptr<read_session> start_read_session()
         { return std::make_shared<read_session>(this->shared_from_this()); }
      };
   };
}

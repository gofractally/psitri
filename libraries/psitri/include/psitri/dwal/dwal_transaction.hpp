#pragma once
#include <psitri/dwal/btree_layer.hpp>
#include <psitri/dwal/btree_value.hpp>
#include <psitri/dwal/dwal_root.hpp>
#include <psitri/dwal/merge_cursor.hpp>
#include <psitri/dwal/undo_log.hpp>
#include <psitri/dwal/wal_writer.hpp>
#include <psitri/lock_policy.hpp>

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace psitri::dwal
{
   template <class LockPolicy>
   class basic_dwal_database;

   template <class LockPolicy>
   class basic_dwal_transaction;

   /**
    * @brief Lazy result from dwal_transaction::remove().
    *
    * If the key was found in the RW layer, the answer is known immediately.
    * If not, converting to bool triggers a lookup in lower layers (RO + Tri).
    * Callers that discard the result pay no lookup cost.
    */
   template <class LockPolicy = std_lock_policy>
   class basic_remove_result
   {
     public:
      using transaction_type = basic_dwal_transaction<LockPolicy>;

      /// Implicit conversion — resolves the lazy check if needed.
      operator bool()
      {
         if (!_resolved)
            resolve();
         return _existed;
      }

     private:
      friend class basic_dwal_transaction<LockPolicy>;

      explicit basic_remove_result(bool existed) : _existed(existed), _resolved(true) {}

      basic_remove_result(const transaction_type* tx, std::string key)
          : _tx(tx), _key(std::move(key)), _resolved(false)
      {
      }

      void resolve();

      const transaction_type* _tx = nullptr;
      std::string             _key;
      bool                    _existed  = false;
      bool                    _resolved = true;
   };

   using remove_result = basic_remove_result<std_lock_policy>;

   /// Mode for a root within a transaction.
   enum class root_mode : uint8_t
   {
      read_write,  ///< Full access: RW btree mutations, undo log, WAL recording
      read_only,   ///< Reads only: 3-layer lookup, no undo/WAL, no mutations
   };

   /**
    * @brief A buffered write transaction on a single DWAL root.
    *
    * Provides ACID semantics over PsiTri's DWAL write path.  Mutations
    * accumulate in the RW ART map with WAL durability; reads see uncommitted
    * writes via layered lookup (RW → RO → Tri).
    *
    * ## Threading
    *
    * Single writer per root.  The transaction must be created, used, and
    * destroyed on the same thread.  No concurrent access — the RW layer
    * is writer-private.
    *
    * ## Lifecycle
    *
    *   auto tx = dwal_db.start_write_transaction(root_index);
    *   tx.upsert("key", "value");       // buffered in RW ART map
    *   auto val = tx.get("key");         // sees uncommitted write
    *   auto cur = tx.create_cursor();    // iterate over uncommitted + committed
    *   tx.commit();                      // WAL entry written, undo discarded
    *   // — or —
    *   tx.abort();                       // undo log replayed, RW map restored
    *
    * If neither commit() nor abort() is called, the destructor aborts.
    *
    * ## Nested Transactions
    *
    * sub_transaction() creates a child that commits back into the parent's
    * RW map.  Only the outermost commit writes a WAL entry.  Inner abort
    * undoes only the child's mutations.
    *
    * ## Direct Mode
    *
    * For large batch operations that exceed the ART buffer capacity, use
    * transaction_mode::direct when starting the transaction.  This flushes
    * the DWAL buffer and writes directly to the PsiTri COW trie.
    */
   template <class LockPolicy = std_lock_policy>
   class basic_dwal_transaction
   {
     public:
      using database_type      = basic_dwal_database<LockPolicy>;
      using dwal_root_type     = basic_dwal_root<LockPolicy>;
      using remove_result_type = basic_remove_result<LockPolicy>;

      basic_dwal_transaction(dwal_root_type& root,
                             wal_writer*     wal,
                             uint32_t        root_index,
                             database_type*  db     = nullptr,
                             bool            nested = false,
                             root_mode       mode   = root_mode::read_write);

      ~basic_dwal_transaction();

      basic_dwal_transaction(const basic_dwal_transaction&)            = delete;
      basic_dwal_transaction& operator=(const basic_dwal_transaction&) = delete;
      basic_dwal_transaction(basic_dwal_transaction&& other) noexcept;
      basic_dwal_transaction& operator=(basic_dwal_transaction&&) = delete;

      // ── Mutations ──────────────────────────────────────────────────

      void upsert(std::string_view key, std::string_view value);

      /**
       * @brief Insert or update a subtree reference.
       *
       * Stores a PsiTri subtree root address as the value.  The caller
       * must have obtained the address via smart_ptr::take() — the
       * transaction assumes ownership of one reference count.
       *
       * @pre  !is_committed() && !is_aborted() && !is_read_only()
       * @param key   Key bytes
       * @param tid   PsiTri subtree tree_id (one ref consumed)
       */
      void upsert_subtree(std::string_view key, sal::tree_id tid);

      /**
       * @brief Remove a single key.
       *
       * Inserts a tombstone into the RW ART map, shadowing the key in
       * lower layers.  Returns a lazy remove_result:
       *   - If the key was in the RW map, the result resolves immediately.
       *   - If not, converting to bool checks RO + Tri layers.
       *   - Discarding the result skips the lower-layer lookup entirely.
       *
       * @pre  !is_committed() && !is_aborted() && !is_read_only()
       */
      remove_result_type remove(std::string_view key);

      /**
       * @brief Remove all keys in [low, high).
       *
       * Adds a range tombstone to the RW layer.  Keys in the range that
       * exist in lower layers are shadowed without per-key cost.
       *
       * @pre  !is_committed() && !is_aborted() && !is_read_only()
       */
      void remove_range(std::string_view low, std::string_view high);

      // ── Reads ──────────────────────────────────────────────────────

      /**
       * @brief Result of a point lookup or cursor value read.
       */
	      struct lookup_result
	      {
	         bool        found = false;
	         btree_value value;
	         std::string owned_data;
	         std::shared_ptr<const btree_layer> layer_guard;

	         lookup_result() = default;
	         lookup_result(bool f, btree_value v) : found(f), value(v) {}
	         lookup_result(bool f, btree_value v, std::shared_ptr<const btree_layer> guard)
	             : found(f), value(v), layer_guard(std::move(guard))
	         {
	         }

	         lookup_result(const lookup_result& other)
	             : found(other.found),
	               value(other.value),
	               owned_data(other.owned_data),
	               layer_guard(other.layer_guard)
	         {
	            rebind_owned_data();
	         }

	         lookup_result(lookup_result&& other) noexcept
	             : found(other.found),
	               value(other.value),
	               owned_data(std::move(other.owned_data)),
	               layer_guard(std::move(other.layer_guard))
	         {
	            rebind_owned_data();
	         }

	         lookup_result& operator=(const lookup_result& other)
	         {
	            if (this == &other)
	               return *this;
	            found       = other.found;
	            value       = other.value;
	            owned_data  = other.owned_data;
	            layer_guard = other.layer_guard;
	            rebind_owned_data();
	            return *this;
	         }

	         lookup_result& operator=(lookup_result&& other) noexcept
	         {
	            if (this == &other)
	               return *this;
	            found       = other.found;
	            value       = other.value;
	            owned_data  = std::move(other.owned_data);
	            layer_guard = std::move(other.layer_guard);
	            rebind_owned_data();
	            return *this;
	         }

	         static lookup_result make_owned(std::string data)
	         {
	            lookup_result r;
	            r.found      = true;
	            r.owned_data = std::move(data);
	            r.value      = btree_value::make_data(r.owned_data);
	            return r;
	         }

	         static lookup_result make_borrowed(btree_value value,
	                                            std::shared_ptr<const btree_layer> guard)
	         {
	            return lookup_result(true, value, std::move(guard));
	         }

	      private:
	         void rebind_owned_data() noexcept
	         {
	            if (found && value.is_data() && !owned_data.empty())
	               value = btree_value::make_data(owned_data);
	         }
	      };

      lookup_result      get(std::string_view key) const;
      owned_merge_cursor create_cursor() const;

      // ── Transaction Control ────────────────────────────────────────

      void commit();
      void commit_multi(uint64_t tx_id, uint16_t participants, bool is_commit);
      void abort();
      basic_dwal_transaction sub_transaction();

      // ── Accessors ──────────────────────────────────────────────────

      bool     is_committed() const noexcept { return _committed; }
      bool     is_aborted() const noexcept { return _aborted; }
      bool     is_read_only() const noexcept { return _mode == root_mode::read_only; }
      uint32_t root_index() const noexcept { return _root_index; }

     private:
      friend class basic_remove_result<LockPolicy>;

      lookup_result ro_get(std::string_view key) const;
      bool          exists_in_lower_layers(std::string_view key) const;
      void          record_undo_for_upsert(std::string_view key);
      void          record_undo_for_remove(std::string_view key);

      dwal_root_type* _root       = nullptr;
      wal_writer*     _wal        = nullptr;
      database_type*  _db         = nullptr;
      uint32_t        _root_index = 0;
      bool            _committed  = false;
      bool            _aborted    = false;
      bool            _nested     = false;
      root_mode       _mode       = root_mode::read_write;

      undo_log _undo;
   };

   using dwal_transaction = basic_dwal_transaction<std_lock_policy>;

}  // namespace psitri::dwal

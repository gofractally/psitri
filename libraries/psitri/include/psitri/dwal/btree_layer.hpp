#pragma once
#include <art/art_map.hpp>
#include <psitri/dwal/btree_value.hpp>
#include <psitri/dwal/range_tombstone_list.hpp>
#include <sal/allocator.hpp>

#include <cstring>
#include <string_view>

namespace psitri::dwal
{
   /// A btree layer: ART map + range tombstones.
   ///
   /// Keys are stored in the ART arena. Value data (string_view payloads)
   /// is also stored in the ART arena — no separate pool needed.
   /// The arena is freed as a unit when the layer is discarded.
   ///
   /// Ref-count contract for subtree values:
   /// While a subtree `sal::ptr_address` lives in this layer's map, the
   /// layer holds one reference on it. The reference is taken on insert
   /// (store_subtree), dropped when the entry is overwritten by data /
   /// tombstone / another subtree, dropped when the entry is erased, and
   /// dropped for any remaining subtree entries on destruction. This
   /// mirrors the retain-on-drain semantics in merge_pool_impl.hpp so
   /// callers can pass a bare `ptr_address` without having to manage a
   /// smart_ptr across the handoff.
   ///
   /// Ref-count operations go through the `sal::allocator` directly
   /// rather than through a thread-affine `allocator_session_ptr`:
   /// `retain()` is an atomic increment, and `release()` forwards to the
   /// calling thread's thread-local session, so either can be invoked
   /// safely from any thread that touches this layer (writer, merge
   /// thread, or the database-destructor flush thread).
   struct btree_layer
   {
      using map_type = art::art_map<btree_value>;
      using iterator = map_type::iterator;

      map_type             map;
      range_tombstone_list tombstones;
      uint32_t             generation = 0;

      /// Allocator used to retain/release subtree addresses. Null when
      /// the layer is constructed without one (e.g. standalone tests
      /// that stash fake addresses and never actually reference real
      /// SAL objects). In that mode store_subtree is a no-op for
      /// ref-counting.
      sal::allocator* alloc = nullptr;

      btree_layer() : map(1u << 20)
      {
         tombstones.set_copy_fn(
             [](void* ctx, std::string_view src) -> std::string_view
             {
                return static_cast<btree_layer*>(ctx)->store_string(src);
             },
             this);
      }

      // Non-copyable, non-movable (arena addresses must stay stable).
      btree_layer(const btree_layer&)            = delete;
      btree_layer& operator=(const btree_layer&) = delete;
      btree_layer(btree_layer&&)                 = delete;
      btree_layer& operator=(btree_layer&&)      = delete;

      ~btree_layer()
      {
         release_all_subtrees();
      }

      /// Bind or rebind the allocator used for subtree ref-counting.
      /// Typically called once, before the first store_subtree. Safe to
      /// call repeatedly; the pointer is not owning.
      void set_allocator(sal::allocator* a) noexcept { alloc = a; }

      /// Copy a string into the ART arena, returning a stable string_view.
      /// Used for range tombstone bounds (not for value data — that goes inline in leaves).
      std::string_view store_string(std::string_view src)
      {
         if (src.empty())
            return {};
         auto& arena = map.get_arena();
         auto  off   = arena.allocate(src.size());
         auto* buf   = arena.as<char>(off);
         std::memcpy(buf, src.data(), src.size());
         return {buf, src.size()};
      }

      /// Store a key/data-value pair into the map.
      /// Value data is stored inline in the ART leaf allocation — no separate alloc.
      void store_data(std::string_view key, std::string_view value)
      {
         // If we're overwriting a subtree entry, drop its refcount.
         release_prior_subtree_if_any(key);

         // upsert_inline stores value bytes right after the btree_value struct
         // in the leaf. We then fix up the data string_view to point at the
         // inline copy.
         auto* bv = map.upsert_inline(key, btree_value::make_data({}), value);
         if (bv && !value.empty())
         {
            // The inline data starts right after the btree_value struct
            auto* inline_ptr = reinterpret_cast<const char*>(bv) + sizeof(btree_value);
            bv->data = {inline_ptr, value.size()};
         }
      }

      /// Store a key/subtree pair. Retains one refcount on each of the
      /// tree_id's root + ver addresses — the caller does NOT need to
      /// hold a smart_ptr across this call.
      void store_subtree(std::string_view key, sal::tree_id tid)
      {
         // If we're replacing another subtree entry, drop the old refs first.
         release_prior_subtree_if_any(key);

         // Retain root so our map slot owns one refcount (MVCC removed: no ver).
         if (alloc && tid.root != sal::null_ptr_address)
            alloc->retain(tid.root);

         map.upsert(key, btree_value::make_subtree(tid));
      }

      /// Store a tombstone for a key.
      void store_tombstone(std::string_view key)
      {
         // Tombstone also overwrites a subtree if one was at this key.
         release_prior_subtree_if_any(key);
         map.upsert(key, btree_value::make_tombstone());
      }

      /// Erase a single key. If the prior entry was a subtree, release its ref.
      void erase(std::string_view key)
      {
         release_prior_subtree_if_any(key);
         map.erase(key);
      }

      /// Number of entries (including tombstones).
      size_t size() const noexcept { return map.size(); }

      /// True if both the map and tombstone list are empty.
      bool empty() const noexcept { return map.empty() && tombstones.empty(); }

      /// Release refs for any subtree entries currently in the map.
      /// Safe to call multiple times; after the first call the subtree
      /// fields are zeroed so subsequent calls are no-ops.
      void release_all_subtrees() noexcept
      {
         if (!alloc)
            return;
         for (auto it = map.begin(); it != map.end(); ++it)
         {
            auto& v = it.value();
            if (v.is_subtree())
            {
               if (v.subtree.root != sal::null_ptr_address)
                  alloc->release(v.subtree.root);
               v.subtree = sal::null_tree_id;
            }
         }
      }

     private:
      /// If the entry at `key` is a subtree, release its refs and zero the
      /// addresses in place so a subsequent release cannot double-decrement.
      void release_prior_subtree_if_any(std::string_view key) noexcept
      {
         if (!alloc)
            return;
         auto* v = map.get(key);
         if (v && v->is_subtree())
         {
            if (v->subtree.root != sal::null_ptr_address)
               alloc->release(v->subtree.root);
            v->subtree = sal::null_tree_id;
         }
      }
   };

   /// Immutable root and range-deletion view. The caller retains the arena owner
   /// and advances its COW generation while holding the root transaction lock.
   struct btree_layer_view
   {
      struct map_view
      {
         art::arena* arena;
         art::offset_t root;
         const btree_layer::map_type* live_map = nullptr;
         art::offset_t current_root() const { return live_map ? live_map->snapshot_root() : root; }
         const btree_value* get(std::string_view key) const
         { return art::get<btree_value>(*arena, current_root(), key); }
         auto begin() const { return art::make_begin<btree_value>(*arena, current_root()); }
         auto end() const { return art::make_end<btree_value>(*arena, current_root()); }
         auto lower_bound(std::string_view key) const
         { return art::make_lower_bound<btree_value>(*arena, current_root(), key); }
         auto upper_bound(std::string_view key) const
         {
            auto it = lower_bound(key);
            if (it != end() && it.key() == key) ++it;
            return it;
         }
         bool empty() const { return current_root() == art::null_offset; }
      };
      struct tombstone_view
      {
         std::vector<range_tombstone_list::range> ranges;
         const range_tombstone_list* live_tombstones = nullptr;
         bool is_deleted(std::string_view key) const
         {
            if (live_tombstones) return live_tombstones->is_deleted(key);
            auto it = std::upper_bound(ranges.begin(), ranges.end(), key,
                [](auto k, const auto& range) { return k < range.low; });
            return it != ranges.begin() && std::prev(it)->contains(key);
         }
      };
      map_view map;
      tombstone_view tombstones;
      explicit btree_layer_view(const btree_layer& layer)
          : map{const_cast<art::arena*>(&layer.map.get_arena()), layer.map.snapshot_root()},
            tombstones{layer.tombstones.ranges()} {}
      static btree_layer_view live(const btree_layer& layer)
      {
         return {layer, false};
      }
     private:
      btree_layer_view(const btree_layer& layer, bool)
          : map{const_cast<art::arena*>(&layer.map.get_arena()), art::null_offset, &layer.map},
            tombstones{{}, &layer.tombstones} {}

   };

}  // namespace psitri::dwal

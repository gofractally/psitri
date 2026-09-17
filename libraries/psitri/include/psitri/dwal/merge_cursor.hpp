#pragma once
#include <psitri/cursor.hpp>
#include <psitri/dwal/btree_layer.hpp>
#include <psitri/dwal/btree_value.hpp>
#include <psitri/dwal/kv_engine.hpp>

#include <optional>
#include <string>
#include <string_view>

namespace psitri::dwal
{
   /// A merge cursor that presents a unified sorted view across the three
   /// DWAL layers: RW btree, RO btree, and PsiTri COW tree.
   ///
   /// At each position, the cursor holds the smallest live (non-tombstoned)
   /// key across all sources. Higher layers (RW > RO > Tri) shadow lower
   /// layers for the same key.
   ///
   /// Layer pointers may be null to exclude layers (e.g., read_mode::buffered
   /// omits RW, read_mode::trie omits RW+RO).
   template <class Engine>
   class basic_merge_cursor
   {
     public:
      using btree_iter = btree_layer::iterator;
      using cursor_t   = typename Engine::cursor;

      /// Construct a merge cursor over up to three layers.
      /// Any layer pointer may be null to exclude that layer.
      /// The tri cursor is optional — pass a default-constructed optional to skip.
      basic_merge_cursor(const btree_layer*       rw,
                         const btree_layer*       ro,
                         std::optional<cursor_t>  tri);

      basic_merge_cursor(std::optional<btree_layer_view> rw,
                         std::optional<btree_layer_view> ro,
                         std::optional<cursor_t> tri);

      // ── Positioning ────────────────────────────────────────────────

      bool seek_begin();
      bool seek_last();
      bool lower_bound(std::string_view key);
      bool upper_bound(std::string_view key);
      bool seek(std::string_view key);
      bool find(std::string_view key);

      // ── Navigation ─────────────────────────────────────────────────

      bool next();
      bool prev();

      // ── State ──────────────────────────────────────────────────────

      bool is_end() const noexcept { return _at_end; }
      bool is_rend() const noexcept { return _at_rend; }

      /// Current key (valid only when !is_end() && !is_rend()).
      std::string_view key() const noexcept { return _current_key; }

      /// Current value. Only valid for data values from btree layers.
      /// For PsiTri values, the caller should use the tri cursor directly.
      const btree_value& current_value() const noexcept { return _current_value; }

      /// Which layer produced the current key.
      enum class source
      {
         rw,
         ro,
         tri,
         none
      };
      source current_source() const noexcept { return _source; }

      /// Whether the current value is a subtree.
      bool is_subtree() const noexcept;

      /// Access the underlying tri cursor (e.g., for value reads, subtree access).
      cursor_t*       tri_cursor() noexcept;
      const cursor_t* tri_cursor() const noexcept;

      // ── Counting ───────────────────────────────────────────────────

      /// Count live keys in [lower, upper). Uses iteration (exact).
      uint64_t count_keys(std::string_view lower = {}, std::string_view upper = {});

     private:
      /// Advance the merge state to find the next live key >= current positions.
      /// Returns false if all sources are exhausted (at end).
      bool advance_forward();

      /// Retreat the merge state to find the previous live key.
      bool advance_backward();

      /// Check if a key is shadowed by tombstones in higher layers.
      bool is_tombstoned_by_rw(std::string_view k) const;
      bool is_tombstoned_by_ro(std::string_view k) const;

      /// Skip the RO and/or Tri iterators past a key that was taken by a higher layer.
      void skip_ro_past(std::string_view k);
      void skip_tri_past(std::string_view k);

      // Layer pointers (null = excluded).
      std::optional<btree_layer_view> _rw;
      std::optional<btree_layer_view> _ro;

      // Btree iterators.
      btree_iter _rw_it, _rw_end;
      btree_iter _ro_it, _ro_end;

      // PsiTri cursor (optional).
      std::optional<cursor_t> _tri;

      // Current merged position.
      std::string  _current_key;
      btree_value  _current_value;
      source       _source  = source::none;
      bool         _at_end  = true;
      bool         _at_rend = true;
      bool         _backward = false;
   };

   /// A merge cursor that owns its layer snapshots.
   ///
   /// Returned by dwal_database::create_cursor(). Holds shared_ptr copies
   /// of the RW/RO btree layers so the caller does not need to manage
   /// layer lifetimes or hold any DWAL locks during iteration.
   template <class Engine>
   class basic_owned_merge_cursor
   {
     public:
      using merge_cursor_t = basic_merge_cursor<Engine>;
      using cursor_t       = typename Engine::cursor;

      basic_owned_merge_cursor(std::shared_ptr<btree_layer> rw,
                               std::shared_ptr<btree_layer> ro,
                               std::optional<cursor_t>      tri)
          : _rw(std::move(rw)),
            _ro(std::move(ro)),
            _cursor(_rw.get(), _ro.get(), std::move(tri))
      {
      }

      basic_owned_merge_cursor(std::shared_ptr<btree_layer> rw,
                               std::shared_ptr<btree_layer> ro,
                               std::optional<btree_layer_view> rw_view,
                               std::optional<btree_layer_view> ro_view,
                               std::optional<cursor_t> tri)
          : _rw(std::move(rw)), _ro(std::move(ro)),
            _cursor(std::move(rw_view), std::move(ro_view), std::move(tri)) {}

      merge_cursor_t&       cursor() noexcept { return _cursor; }
      const merge_cursor_t& cursor() const noexcept { return _cursor; }

      merge_cursor_t*       operator->() noexcept { return &_cursor; }
      const merge_cursor_t* operator->() const noexcept { return &_cursor; }

      merge_cursor_t&       operator*() noexcept { return _cursor; }
      const merge_cursor_t& operator*() const noexcept { return _cursor; }

     private:
      std::shared_ptr<btree_layer> _rw;
      std::shared_ptr<btree_layer> _ro;
      merge_cursor_t               _cursor;
   };

   /// psitri-engine aliases (the historical concrete types — external ABI).
   using merge_cursor       = basic_merge_cursor<psitri_engine<std_lock_policy>>;
   using owned_merge_cursor = basic_owned_merge_cursor<psitri_engine<std_lock_policy>>;

}  // namespace psitri::dwal

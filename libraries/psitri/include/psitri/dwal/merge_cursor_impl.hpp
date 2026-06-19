#pragma once
// Template bodies for basic_merge_cursor (engine-generic 3-tier merge).
// The psitri-engine instantiation is compiled into the psitri library
// (src/dwal/merge_cursor.cpp); other engines instantiate from this header.
#include <psitri/dwal/merge_cursor.hpp>

#include <cassert>

namespace psitri::dwal
{

   template <class Engine>
   basic_merge_cursor<Engine>::basic_merge_cursor(const btree_layer*            rw,
                              const btree_layer*            ro,
                              std::optional<cursor_t>  tri)
       : _rw(rw), _ro(ro), _tri(std::move(tri))
   {
      if (_rw)
      {
         _rw_it  = _rw->map.begin();
         _rw_end = _rw->map.end();
      }
      if (_ro)
      {
         _ro_it  = _ro->map.begin();
         _ro_end = _ro->map.end();
      }
   }

   // ── Positioning ──────────────────────────────────────────────────

   template <class Engine>
   bool basic_merge_cursor<Engine>::seek_begin()
   {
      if (_rw)
      {
         _rw_it  = _rw->map.begin();
         _rw_end = _rw->map.end();
      }
      if (_ro)
      {
         _ro_it  = _ro->map.begin();
         _ro_end = _ro->map.end();
      }
      if (_tri)
         _tri->seek_begin();

      _at_end  = false;
      _at_rend = false;
      return advance_forward();
   }

   template <class Engine>
   bool basic_merge_cursor<Engine>::seek_last()
   {
      // Position all sources at their last element, then merge backward.
      if (_rw)
      {
         _rw_end = _rw->map.end();
         _rw_it  = _rw_end;
         if (!_rw->map.empty())
            --_rw_it;  // point at last element
      }
      if (_ro)
      {
         _ro_end = _ro->map.end();
         _ro_it  = _ro_end;
         if (!_ro->map.empty())
            --_ro_it;
      }
      if (_tri)
         _tri->seek_last();

      _at_end  = false;
      _at_rend = false;
      return advance_backward();
   }

   template <class Engine>
   bool basic_merge_cursor<Engine>::lower_bound(std::string_view key)
   {
      if (_rw)
      {
         _rw_it  = _rw->map.lower_bound(key);
         _rw_end = _rw->map.end();
      }
      if (_ro)
      {
         _ro_it  = _ro->map.lower_bound(key);
         _ro_end = _ro->map.end();
      }
      if (_tri)
         _tri->lower_bound(key);

      _at_end  = false;
      _at_rend = false;
      return advance_forward();
   }

   // Mathematical upper bound: the LARGEST live key v <= key (the floor; the dual
   // of lower_bound). Built from this cursor's own merge-aware lower_bound/prev/
   // seek_last so the 3-way dedup + tombstone logic is reused, not re-derived.
   template <class Engine>
   bool basic_merge_cursor<Engine>::upper_bound(std::string_view key)
   {
      if (lower_bound(key))          // smallest live v >= key
      {
         if (_current_key == key)    // key present (live) → it is its own floor
            return true;
         return prev();              // landed strictly > key → predecessor = largest v < key
      }
      return seek_last();            // no live key >= key → the last key is the floor
   }

	   template <class Engine>
	   bool basic_merge_cursor<Engine>::seek(std::string_view key)
	   {
	      if (!lower_bound(key))
	         return false;
      if (_current_key != key)
      {
         _at_end = true;
         _source = source::none;
         return false;
      }
	      return true;
	   }

	   template <class Engine>
	   bool basic_merge_cursor<Engine>::find(std::string_view key)
	   {
	      return seek(key);
	   }

	   // ── Navigation ───────────────────────────────────────────────────

   template <class Engine>
   bool basic_merge_cursor<Engine>::next()
   {
      if (_at_end)
         return false;

      // Advance the source(s) that produced the current key.
      std::string_view cur = _current_key;

      // Advance RW past current key.
      if (_rw && _rw_it != _rw_end && _rw_it.key() == cur)
         ++_rw_it;

      // Advance RO past current key.
      if (_ro && _ro_it != _ro_end && _ro_it.key() == cur)
         ++_ro_it;

      // Advance Tri past current key.
      if (_tri && !_tri->is_end() && _tri->key() == cur)
         _tri->next();

      return advance_forward();
   }

   template <class Engine>
   bool basic_merge_cursor<Engine>::prev()
   {
      if (_at_rend)
         return false;

      // When at end, position all sources at their last element.
      if (_at_end)
      {
         if (_rw)
         {
            _rw_end = _rw->map.end();
            _rw_it  = _rw_end;
            if (_rw_it != _rw->map.begin())
               --_rw_it;
         }
         if (_ro)
         {
            _ro_end = _ro->map.end();
            _ro_it  = _ro_end;
            if (_ro_it != _ro->map.begin())
               --_ro_it;
         }
         if (_tri)
            _tri->seek_last();

         _at_end = false;
         return advance_backward();
      }

      // Retreat the source(s) past the current key.
      std::string_view cur = _current_key;

      // Retreat RW.
      if (_rw)
      {
         if (_rw_it != _rw_end)
         {
            if (_rw_it.key() >= cur)
            {
               --_rw_it;
               if (_rw_it != _rw_end && _rw_it.key() == cur)
                  --_rw_it;
            }
         }
      }

      // Retreat RO (same logic).
      if (_ro)
      {
         if (_ro_it != _ro_end)
         {
            if (_ro_it.key() >= cur)
            {
               --_ro_it;
               if (_ro_it != _ro_end && _ro_it.key() == cur)
                  --_ro_it;
            }
         }
      }

      // Retreat Tri.
      if (_tri && !_tri->is_rend())
      {
         if (!_tri->is_end() && _tri->key() >= cur)
            _tri->prev();
         if (!_tri->is_rend() && _tri->key() == cur)
            _tri->prev();
      }

      return advance_backward();
   }

   // ── Helpers ──────────────────────────────────────────────────────

   template <class Engine>
   bool basic_merge_cursor<Engine>::is_tombstoned_by_rw(std::string_view k) const
   {
      if (!_rw)
         return false;
      // Check point tombstone.
      auto* v = _rw->map.get(k);
      if (v && v->is_tombstone())
         return true;
      // Check range tombstone.
      return _rw->tombstones.is_deleted(k);
   }

   template <class Engine>
   bool basic_merge_cursor<Engine>::is_tombstoned_by_ro(std::string_view k) const
   {
      if (!_ro)
         return false;
      auto* v = _ro->map.get(k);
      if (v && v->is_tombstone())
         return true;
      return _ro->tombstones.is_deleted(k);
   }

   template <class Engine>
   void basic_merge_cursor<Engine>::skip_ro_past(std::string_view k)
   {
      if (_ro && _ro_it != _ro_end && _ro_it.key() == k)
         ++_ro_it;
   }

   template <class Engine>
   void basic_merge_cursor<Engine>::skip_tri_past(std::string_view k)
   {
      if (_tri && !_tri->is_end() && _tri->key() == k)
         _tri->next();
   }

   template <class Engine>
   bool basic_merge_cursor<Engine>::advance_forward()
   {
      // Find the smallest live key across all sources.
      for (;;)
      {
         // Gather candidates.
         bool             have_rw = _rw && _rw_it != _rw_end;
         bool             have_ro = _ro && _ro_it != _ro_end;
         bool             have_tri = _tri && !_tri->is_end();

         if (!have_rw && !have_ro && !have_tri)
         {
            _at_end = true;
            _source = source::none;
            return false;
         }

         // Find minimum key.
         std::string_view min_key;
         source           min_src = source::none;

         if (have_rw)
         {
            min_key = _rw_it.key();
            min_src = source::rw;
         }
         if (have_ro)
         {
            if (min_src == source::none || _ro_it.key() < min_key)
            {
               min_key = _ro_it.key();
               min_src = source::ro;
            }
         }
         if (have_tri)
         {
            if (min_src == source::none || _tri->key() < min_key)
            {
               min_key = _tri->key();
               min_src = source::tri;
            }
         }

         // Determine the winning source (highest priority for ties).
         // RW > RO > Tri priority.
         source winner = source::none;
         btree_value winner_val;

         // Check RW at this key.
         if (have_rw && _rw_it.key() == min_key)
         {
            if (_rw_it.value().is_tombstone())
            {
               // RW tombstone shadows everything — skip this key in all sources.
               ++_rw_it;
               skip_ro_past(min_key);
               skip_tri_past(min_key);
               continue;
            }
            winner     = source::rw;
            winner_val = _rw_it.value();
            // Skip duplicates in lower layers.
            skip_ro_past(min_key);
            skip_tri_past(min_key);
         }
         else if (have_ro && _ro_it.key() == min_key)
         {
            // Check if RW range-tombstones this key.
            if (_rw && _rw->tombstones.is_deleted(min_key))
            {
               ++_ro_it;
               skip_tri_past(min_key);
               continue;
            }
            if (_ro_it.value().is_tombstone())
            {
               // RO tombstone shadows Tri.
               ++_ro_it;
               skip_tri_past(min_key);
               continue;
            }
            winner     = source::ro;
            winner_val = _ro_it.value();
            skip_tri_past(min_key);
         }
         else
         {
            // Tri is the source.
            // Check if shadowed by RW or RO tombstones/range tombstones.
            if (is_tombstoned_by_rw(min_key) || is_tombstoned_by_ro(min_key))
            {
               _tri->next();
               continue;
            }
            winner     = source::tri;
            winner_val = {};  // Tri values accessed through tri cursor.
         }

         _current_key   = std::string(min_key);
         _current_value = winner_val;
         _source        = winner;
         _at_end        = false;
         _at_rend       = false;
         return true;
      }
   }

   template <class Engine>
   bool basic_merge_cursor<Engine>::advance_backward()
   {
      // Find the largest live key across all sources (backward merge).
      for (;;)
      {
         bool have_rw  = _rw && _rw_it != _rw_end;
         bool have_ro  = _ro && _ro_it != _ro_end;
         bool have_tri = _tri && !_tri->is_rend() && !_tri->is_end();

         if (!have_rw && !have_ro && !have_tri)
         {
            _at_rend = true;
            _source  = source::none;
            return false;
         }

         // Find maximum key.
         std::string_view max_key;
         source           max_src = source::none;

         if (have_rw)
         {
            max_key = _rw_it.key();
            max_src = source::rw;
         }
         if (have_ro)
         {
            if (max_src == source::none || _ro_it.key() > max_key)
            {
               max_key = _ro_it.key();
               max_src = source::ro;
            }
         }
         if (have_tri)
         {
            if (max_src == source::none || _tri->key() > max_key)
            {
               max_key = _tri->key();
               max_src = source::tri;
            }
         }

         // Determine winner (highest priority for ties).
         source winner = source::none;
         btree_value winner_val;

         // Check if RW has this key.
         if (have_rw && _rw_it.key() == max_key)
         {
            if (_rw_it.value().is_tombstone())
            {
               // Skip — move all sources past this key backward.
               --_rw_it;
               // Move RO backward past this key.
               if (have_ro && _ro_it.key() == max_key)
                  --_ro_it;
               // Move Tri backward past this key.
               if (have_tri && _tri->key() == max_key)
                  _tri->prev();
               continue;
            }
            winner     = source::rw;
            winner_val = _rw_it.value();
            // Skip duplicates in lower layers backward.
            if (have_ro && _ro_it.key() == max_key)
               --_ro_it;
            if (have_tri && _tri->key() == max_key)
               _tri->prev();
         }
         else if (have_ro && _ro_it.key() == max_key)
         {
            if (_rw && _rw->tombstones.is_deleted(max_key))
            {
               --_ro_it;
               if (have_tri && _tri->key() == max_key)
                  _tri->prev();
               continue;
            }
            if (_ro_it.value().is_tombstone())
            {
               --_ro_it;
               if (have_tri && _tri->key() == max_key)
                  _tri->prev();
               continue;
            }
            winner     = source::ro;
            winner_val = _ro_it.value();
            if (have_tri && _tri->key() == max_key)
               _tri->prev();
         }
         else
         {
            if (is_tombstoned_by_rw(max_key) || is_tombstoned_by_ro(max_key))
            {
               _tri->prev();
               continue;
            }
            winner     = source::tri;
            winner_val = {};
         }

         _current_key   = std::string(max_key);
         _current_value = winner_val;
         _source        = winner;
         _at_end        = false;
         _at_rend       = false;
         return true;
      }
   }

   // ── Access ───────────────────────────────────────────────────────

   template <class Engine>
   bool basic_merge_cursor<Engine>::is_subtree() const noexcept
   {
      if (_source == source::rw || _source == source::ro)
         return _current_value.is_subtree();
      if (_source == source::tri && _tri)
         return _tri->is_subtree();
      return false;
   }

   template <class Engine>
   typename basic_merge_cursor<Engine>::cursor_t* basic_merge_cursor<Engine>::tri_cursor() noexcept
   {
      return _tri ? &*_tri : nullptr;
   }

   template <class Engine>
   const typename basic_merge_cursor<Engine>::cursor_t* basic_merge_cursor<Engine>::tri_cursor() const noexcept
   {
      return _tri ? &*_tri : nullptr;
   }

   template <class Engine>
   uint64_t basic_merge_cursor<Engine>::count_keys(std::string_view lower, std::string_view upper)
   {
      // Simple iteration-based count (exact).
      if (lower.empty())
         seek_begin();
      else
         lower_bound(lower);

      uint64_t count = 0;
      while (!is_end())
      {
         if (!upper.empty() && key() >= upper)
            break;
         ++count;
         next();
      }
      return count;
   }

}  // namespace psitri::dwal

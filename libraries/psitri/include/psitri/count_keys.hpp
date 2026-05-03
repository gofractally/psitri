#pragma once
#include <psitri/node/inner.hpp>
#include <psitri/node/leaf.hpp>
#include <psitri/node/value_node.hpp>
#include <psitri/util.hpp>
#include <psitri/version_compare.hpp>
#include <sal/allocator_session_impl.hpp>

namespace psitri
{
   struct key_range
   {
      key_view lower_bound;  ///< inclusive lower bound, empty = unbounded
      key_view upper_bound;  ///< exclusive upper bound, empty = unbounded

      bool is_unbounded() const { return lower_bound.empty() && upper_bound.empty(); }

      bool is_empty_range() const
      {
         return !upper_bound.empty() && !lower_bound.empty() &&
                upper_bound <= lower_bound;
      }
   };

   // Forward declaration
   uint64_t count_keys(sal::allocator_session& session,
                       ptr_address             addr,
                       key_range               range,
                       uint64_t                version = UINT64_MAX);

   /**
    * Returns true when the leaf branch carries a value that is visible at the
    * given snapshot version.  A branch is hidden when it was created strictly
    * after the snapshot, or when the branch's value_node resolves to a
    * tombstone or null at the snapshot.  Used by both count_keys and the
    * cursor's MVCC visibility check so the two stay in sync.
    */
   inline bool leaf_entry_visible_at(sal::allocator_session& session,
                                     const leaf_node&        node,
                                     branch_number           branch,
                                     uint64_t                version)
   {
      const uint64_t created_at = node.get_version(branch);
      if (version != UINT64_MAX && created_at != 0 &&
          version_newer_than(created_at, version, value_version_bits))
         return false;

      if (node.get_value_type(branch) != leaf_node::value_type_flag::value_node)
         return true;

      auto ref           = session.get_ref<value_node>(node.get_value_address(branch));
      auto [offset, idx] = ref->find_version(version);
      return offset != value_node::offset_tombstone && offset != value_node::offset_null;
   }

   inline uint64_t count_keys_leaf(sal::allocator_session& session,
                                   const leaf_node&        node,
                                   key_range               range,
                                   uint64_t                version);

   /// Count all keys in the subtree rooted at addr.
   /// Recursively sums children for inner nodes (O(N) — epoch replaced descendents).
   inline uint64_t count_child_keys(sal::allocator_session& session,
                                    ptr_address             addr,
                                    uint64_t                version = UINT64_MAX)
   {
      auto ref = session.get_ref(addr);
      switch (node_type(ref->type()))
      {
         case node_type::inner:
         {
            auto     in    = ref.as<inner_node>();
            uint64_t total = 0;
            for (uint16_t i = 0; i < in->num_branches(); ++i)
               total += count_child_keys(session, in->get_branch(branch_number(i)), version);
            return total;
         }
         case node_type::inner_prefix:
         {
            auto     ipn   = ref.as<inner_prefix_node>();
            uint64_t total = 0;
            for (uint16_t i = 0; i < ipn->num_branches(); ++i)
               total += count_child_keys(session, ipn->get_branch(branch_number(i)), version);
            return total;
         }
         case node_type::leaf:
            return count_keys_leaf(session, *ref.as<leaf_node>(), {}, version);
         case node_type::value:
            return 1;
         default:
            std::unreachable();
      }
   }

   /// Count keys in a leaf node within [lower, upper).  When version is not
   /// UINT64_MAX, branches hidden at the snapshot (newer than the snapshot or
   /// resolved as tombstone/null) are excluded; older snapshots therefore see
   /// the count they would have seen at commit time, even after later inserts
   /// or removes.
   inline uint64_t count_keys_leaf(sal::allocator_session& session,
                                   const leaf_node&        node,
                                   key_range               range,
                                   uint64_t                version)
   {
      branch_number lo = range.lower_bound.empty() ? branch_number(0)
                                                   : node.lower_bound(range.lower_bound);
      branch_number hi = range.upper_bound.empty() ? branch_number(node.num_branches())
                                                   : node.lower_bound(range.upper_bound);
      if (*hi < *lo)
         return 0;

      uint64_t total = 0;
      for (uint16_t i = *lo; i < *hi; ++i)
         if (leaf_entry_visible_at(session, node, branch_number(i), version))
            ++total;
      return total;
   }

   /// Core inner node range counting with adaptive inclusion/exclusion.
   ///
   /// Key design: psitri's inner_node does NOT consume key bytes during routing.
   /// Only inner_prefix_node consumes prefix bytes. Therefore, the full range
   /// (after prefix stripping) is passed to child branches — no byte stripping
   /// at inner node boundaries.
   template <any_inner_node_type NodeT>
   uint64_t count_keys_inner(sal::allocator_session& session,
                             const NodeT&            node,
                             key_range               range,
                             uint64_t                version)
   {
      // Handle prefix for inner_prefix_node — narrows the range
      if constexpr (is_inner_prefix_node<NodeT>)
      {
         key_view prefix    = node.prefix();
         key_view new_lower = range.lower_bound;
         key_view new_upper = range.upper_bound;

         // Narrow lower bound by prefix
         if (!range.lower_bound.empty())
         {
            auto cpre = common_prefix(prefix, range.lower_bound);
            if (cpre.size() == prefix.size())
            {
               new_lower = range.lower_bound.substr(prefix.size());
            }
            else if (cpre.size() < range.lower_bound.size())
            {
               if (static_cast<uint8_t>(prefix[cpre.size()]) >
                   static_cast<uint8_t>(range.lower_bound[cpre.size()]))
                  new_lower = key_view();
               else
                  return 0;
            }
            else
            {
               // lower_bound is a prefix of node prefix → all keys > lower
               new_lower = key_view();
            }
         }

         // Narrow upper bound by prefix
         if (!range.upper_bound.empty())
         {
            auto cpre = common_prefix(prefix, range.upper_bound);
            if (cpre.size() == prefix.size())
            {
               if (range.upper_bound.size() == prefix.size())
                  return 0;  // prefix == upper exactly → exclusive, no keys
               new_upper = range.upper_bound.substr(prefix.size());
            }
            else if (cpre.size() < range.upper_bound.size())
            {
               if (static_cast<uint8_t>(prefix[cpre.size()]) >=
                   static_cast<uint8_t>(range.upper_bound[cpre.size()]))
                  return 0;
               else
                  new_upper = key_view();
            }
            else
            {
               // upper_bound is a prefix of node prefix → all keys >= upper
               return 0;
            }
         }

         range = {new_lower, new_upper};
      }

      if (range.is_unbounded())
      {
         uint64_t total = 0;
         for (uint16_t i = 0; i < node.num_branches(); ++i)
            total += count_child_keys(session, node.get_branch(branch_number(i)), version);
         return total;
      }

      if (range.is_empty_range())
         return 0;

      // Route: find which branches the lower and upper bounds map to
      branch_number start = range.lower_bound.empty()
                                ? branch_number(0)
                                : node.lower_bound(range.lower_bound);
      if (*start >= node.num_branches())
         return 0;

      branch_number end      = branch_number(node.num_branches());
      branch_number boundary = end;
      bool          has_boundary = false;

      if (!range.upper_bound.empty())
      {
         boundary = node.lower_bound(range.upper_bound);
         end      = boundary;
         if (*boundary < node.num_branches())
            has_boundary = true;
      }

      // If start == boundary, both bounds are in the same branch — recurse with full range
      if (*start == *boundary && has_boundary)
         return count_keys(session, node.get_branch(start), range, version);

      // Adaptive decision: count by inclusion or exclusion
      uint16_t in_range     = *end - *start;
      uint16_t total_br     = node.num_branches();
      uint16_t out_of_range = total_br - in_range - (has_boundary ? 1 : 0);

      uint64_t count = 0;

      if (in_range > out_of_range)
      {
         // Exclusion: total - before - after
         for (uint16_t i = 0; i < node.num_branches(); ++i)
            count += count_child_keys(session, node.get_branch(branch_number(i)), version);

         // Subtract branches before start
         for (uint16_t i = 0; i < *start; ++i)
            count -= count_child_keys(session, node.get_branch(branch_number(i)), version);

         // Subtract keys below lower_bound in start branch
         if (!range.lower_bound.empty())
         {
            uint64_t total_in_start =
                count_child_keys(session, node.get_branch(start), version);
            uint64_t in_range_start =
                count_keys(session, node.get_branch(start), range, version);
            count -= (total_in_start - in_range_start);
         }

         // Subtract branches after boundary (or after end)
         uint16_t after_start = has_boundary ? *boundary + 1 : *end;
         for (uint16_t i = after_start; i < total_br; ++i)
            count -= count_child_keys(session, node.get_branch(branch_number(i)), version);

         // Subtract keys >= upper_bound in boundary branch
         if (has_boundary)
         {
            uint64_t total_in_boundary =
                count_child_keys(session, node.get_branch(boundary), version);
            uint64_t in_range_boundary =
                count_keys(session, node.get_branch(boundary), range, version);
            count -= (total_in_boundary - in_range_boundary);
         }
      }
      else
      {
         // Inclusion: count start + middle + boundary

         // Start branch: pass full range (child handles routing)
         count += count_keys(session, node.get_branch(start), range, version);

         // Middle branches: fully contained
         for (uint16_t i = *start + 1; i < *end; ++i)
            count += count_child_keys(session, node.get_branch(branch_number(i)), version);

         // Boundary branch: pass full range
         if (has_boundary)
            count += count_keys(session, node.get_branch(boundary), range, version);
      }

      return count;
   }

   /// Dispatch count_keys to typed node implementations
   inline uint64_t count_keys(sal::allocator_session& session,
                              ptr_address             addr,
                              key_range               range,
                              uint64_t                version)
   {
      if (addr == sal::null_ptr_address)
         return 0;

      auto ref = session.get_ref(addr);
      switch (node_type(ref->type()))
      {
         case node_type::inner:
            return count_keys_inner(session, *ref.as<inner_node>(), range, version);
         case node_type::inner_prefix:
            return count_keys_inner(session, *ref.as<inner_prefix_node>(), range, version);
         case node_type::leaf:
            return count_keys_leaf(session, *ref.as<leaf_node>(), range, version);
         case node_type::value:
         {
            bool in_range = (range.lower_bound.empty() || key_view() >= range.lower_bound) &&
                            (range.upper_bound.empty() || key_view() < range.upper_bound);
            return in_range ? 1 : 0;
         }
         default:
            std::unreachable();
      }
   }

}  // namespace psitri

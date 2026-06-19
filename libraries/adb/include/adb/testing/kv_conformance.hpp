// adb — shared behavioral conformance for the adb cursor contract. Authored ONCE
// here, instantiated against EVERY engine's cursor (the spec's test-authoring rule:
// the same check runs on all engines, never copy-pasted per engine). A std::map is
// the differential oracle.
//
// The two ordered bounds are mathematical duals, validated together:
//   lower_bound(k) = the smallest key >= k   (the ceiling / infimum of {v : v >= k})
//   upper_bound(k) = the LARGEST  key <= k   (the floor  / supremum of {v : v <= k})
//
// upper_bound is NOT std::-style "first > k": it is the mathematical upper bound,
// the dual of lower_bound. The std::map oracle for the floor is therefore
// `m.upper_bound(k)` (first > k) stepped back one (the element just below it).
#pragma once
#include <iterator>
#include <map>
#include <string>
#include <string_view>

namespace adb::testing
{
   // Probe `cur` against `oracle` for lower_bound (ceiling) and upper_bound (floor)
   // at every key, a proper prefix of each key (< key), just past each key (> key),
   // the empty key, and a key past the maximum. `ok(bool)` records one check; the
   // caller supplies the accounting (Catch2 CHECK, a checks/fails pair, etc.).
   //
   // `cur` is taken by value and re-positioned for each probe — any adb cursor
   // (kv_cursor + the upper_bound/prev of mergeable_cursor) works, including the
   // tiered/merge cursors, so this is the single definition of "bounds are correct".
   template <class Cursor, class Ok>
   void check_bounds(Cursor cur, const std::map<std::string, std::string>& oracle, Ok&& ok)
   {
      auto probe = [&](std::string_view q)
      {
         const std::string qs(q);

         // ── ceiling: smallest key >= q (== std::map::lower_bound) ──
         {
            const bool got = cur.lower_bound(q);
            auto       oi  = oracle.lower_bound(qs);
            ok(got == (oi != oracle.end()));
            if (got && oi != oracle.end())
               ok(std::string(cur.key()) == oi->first);
         }
         // ── floor: largest key <= q (the mathematical upper bound) ──
         {
            const bool got = cur.upper_bound(q);
            auto       gt  = oracle.upper_bound(qs);    // first > q
            const bool ofl = (gt != oracle.begin());     // a key <= q exists
            auto       fi  = ofl ? std::prev(gt) : oracle.end();  // largest <= q
            ok(got == ofl);
            if (got && ofl)
               ok(std::string(cur.key()) == fi->first);
         }
      };

      for (const auto& [k, v] : oracle)
      {
         probe(k);                                   // a present key
         if (!k.empty())
            probe(k.substr(0, k.size() - 1));        // a proper prefix (strictly < k)
         {
            std::string after(k);
            after.push_back('\x01');                 // just past k (strictly > k)
            probe(after);
         }
      }
      probe("");                                     // at/below the minimum
      probe(std::string(8, char(0xff)));             // past the maximum
   }
}  // namespace adb::testing

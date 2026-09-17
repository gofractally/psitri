#pragma once
#include <psitri/concepts.hpp>
#include <sal_art_ref/engine.hpp>

namespace psitri
{
   /// Refcounted movable storage for databases that scale beyond RAM.
   /// "Big" describes total database scale, not individual value size.
   using big_engine = sal_art_ref::radix::engine;
   using default_engine = big_engine;
   static_assert(engine<big_engine>);

   using compact_engine = big_engine;  // compatibility with the initial API spelling
}

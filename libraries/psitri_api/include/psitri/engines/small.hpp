#pragma once
#include <psitri/concepts.hpp>
#include <start_ref/kv.hpp>

namespace psitri
{
   /// Refcounted fixed-address storage for databases that fit in RAM.
   /// "Small" describes the intended workload, not the handle width.
   using small_engine = start_ref::kv::engine;
   static_assert(engine<small_engine>);

   using fixed_engine = small_engine;  // compatibility with the initial API spelling
}

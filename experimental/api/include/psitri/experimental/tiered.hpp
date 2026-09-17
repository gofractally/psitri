#pragma once
#include <psitri/api.hpp>
#include <delta_art/tiered_kv.hpp>

namespace psitri::experimental
{
   // No portable owned composite snapshot yet; deliberately not psitri::engine.
   template <psitri::engine Hot = small_engine, psitri::engine Base = big_engine>
   using tiered_engine = delta_art::engine<Hot, Base>;
   using write_engine = tiered_engine<>;
}

#pragma once
// heart as the DWAL base tier ("Tri" layer): the kv_engine adapter.
//
// Heart's public API was built as a psitri conformance target (kv-compat
// drives both engines with one test body), so the adapter is pure typedefs —
// the concept check below is the contract. Lives with heart (not in psitri's
// dwal headers) so psitri never depends on heart.
//
// Consumers instantiating dwal over this engine must include the dwal
// template bodies (*_impl.hpp) in their TU — the psitri library only ships
// the psitri-engine instantiation.
#include <psitri/dwal/kv_engine.hpp>
#include <radix_engine.hpp>

namespace heart::radix
{
   struct dwal_engine
   {
      using database      = heart::radix::database;
      using write_session = heart::radix::write_session;
      using read_session  = heart::radix::read_session;
      using transaction   = heart::radix::transaction;
      using cursor        = heart::radix::cursor;
   };
   static_assert(psitri::dwal::kv_engine<dwal_engine>,
                 "heart's public API must satisfy the DWAL base-tier concept");
}  // namespace heart::radix

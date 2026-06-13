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

namespace sal_art_ref::radix
{
   struct dwal_engine
   {
      using database      = sal_art_ref::radix::database;
      using write_session = sal_art_ref::radix::write_session;
      using read_session  = sal_art_ref::radix::read_session;
      using transaction   = sal_art_ref::radix::transaction;
      using cursor        = sal_art_ref::radix::cursor;
      using config        = sal::runtime_config;
   };
   static_assert(psitri::dwal::kv_engine<dwal_engine>,
                 "heart's public API must satisfy the DWAL base-tier concept");
}  // namespace sal_art_ref::radix

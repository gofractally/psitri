#pragma once
#include <adb/concepts.hpp>
#include <sal_art_ref/radix_engine.hpp>

namespace sal_art_ref::radix
{
   struct engine
   {
      static constexpr uint64_t max_value_size = large_value_detail::max_value_size;
      using database      = sal_art_ref::radix::database;
      using write_session = sal_art_ref::radix::write_session;
      using read_session  = sal_art_ref::radix::read_session;
      using transaction   = sal_art_ref::radix::transaction;
      using cursor        = sal_art_ref::radix::cursor;
      using config        = sal_art_ref::radix::runtime_config;
   };
   using dwal_engine = engine;  // historical spelling
   static_assert(adb::openable_engine<engine, sal::open_mode>);
}

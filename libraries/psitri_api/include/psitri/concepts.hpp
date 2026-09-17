#pragma once
#include <adb/concepts.hpp>
#include <sal/config.hpp>

namespace psitri
{
   // Source-level API version. This does not version an engine's disk format.
   inline constexpr unsigned api_version = 1;
   using open_mode = sal::open_mode;
   using sync_mode = sal::sync_type;
   using adb::read_all;

   /// The complete portable byte-database contract. See docs/api/reference.md.
   template <class E>
   concept engine = adb::openable_engine<E, open_mode> && requires(typename E::config& cfg) {
      cfg.sync_mode = sync_mode::none;
   };

   /// Optional range reads; the version-1 engine concept remains unchanged.
   template <class E>
   concept range_engine = engine<E> && adb::range_engine<E>;

   template <engine E>
   using database_for = typename E::database;
   template <engine E>
   using config_for = typename E::config;
   template <engine E>
   using snapshot_for = decltype(std::declval<typename E::write_session&>().get_root(0u));
}

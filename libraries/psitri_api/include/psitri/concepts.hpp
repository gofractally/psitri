#pragma once
#include <adb/concepts.hpp>
#include <sal/config.hpp>

#include <array>
#include <string_view>

namespace psitri
{
   // Source-level API version. This does not version an engine's disk format.
   inline constexpr unsigned api_version = 2;
   using open_mode = sal::open_mode;
   using sync_mode = sal::sync_type;
   using adb::read_all;

   namespace detail
   {
      // One byte longer than the longest key any engine takes (1,024 bytes), all 0xFF.
      inline constexpr std::array<char, 1025> above_every_key_data = []()
      {
         std::array<char, 1025> bytes{};
         for (auto& c : bytes)
            c = '\xFF';
         return bytes;
      }();
   }  // namespace detail
   /// An upper bound above every key. Ranges are half-open, [low, high), with no special
   /// meaning for an empty bound; pass this as `high` to mean "to the end".
   inline constexpr std::string_view above_every_key{detail::above_every_key_data.data(),
                                                     detail::above_every_key_data.size()};

   /// The complete portable byte-database contract. See docs/api/reference.md.
   ///
   /// Beyond what the signatures say, an engine promises:
   ///  - Roots are independent. Write sessions working on different roots run
   ///    at the same time without affecting each other.
   ///  - A transaction holds its root's writer from start_transaction until it
   ///    commits or aborts. A second writer on that root waits and starts from
   ///    what the first published; no commit is lost.
   ///  - Readers see only committed states and never wait for a writer's work.
   ///    A cursor is a snapshot: it does not change after it is created.
   ///  - A transaction's own cursor shows the root as that transaction has
   ///    changed it so far, in both directions, and is likewise a snapshot.
   /// tests/api/writers.cpp and tests/api/transactions.cpp check these on every
   /// engine listed there.
   template <class E>
   concept engine = adb::openable_engine<E, open_mode> &&
                    requires(typename E::config& cfg, typename E::transaction& tx,
                             std::string_view key) {
                       cfg.sync_mode = sync_mode::none;
                       { tx.snapshot_cursor() } -> adb::mergeable_cursor;
                       { tx.remove_range_counted(key, key) } -> std::convertible_to<std::uint64_t>;
                    };

   /// Optional range reads.
   template <class E>
   concept range_engine = engine<E> && adb::range_engine<E>;

   template <engine E>
   using database_for = typename E::database;
   template <engine E>
   using config_for = typename E::config;
   template <engine E>
   using snapshot_for = decltype(std::declval<typename E::write_session&>().get_root(0u));
}

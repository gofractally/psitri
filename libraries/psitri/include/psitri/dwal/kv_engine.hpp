#pragma once
// The base-tier ("Tri") engine concept for the DWAL layer.
//
// dwal historically hardwired psitri as its COW base. Everything dwal needs
// from the base is the psitri-shaped public API that heart implemented as a
// conformance target (kv-compat proves one test body drives both engines), so
// the base becomes a template parameter constrained by `kv_engine`.
//
// An Engine is a TRAITS BUNDLE (not the database itself): five member types
// whose instances interact per the requires-clauses below. `psitri_engine`
// adapts psitri; heart's adapter lives with heart (heart_dwal_engine.hpp) so
// the psitri library never depends on heart.
//
// Optional capabilities are feature-detected at the call sites with
// `if constexpr (requires ...)` rather than required here:
//   tx.upsert_sorted(k, v)        — sorted-insert sibling-prefetch hint
//   tx.remove_range_any(lo, hi)   — engine-native range removal
//   cur.refresh(idx)              — re-pin an existing cursor at the live root
//   ws.allocator_session()        — SAL stats for merge telemetry
//   root_handle.address()         — observability root id (wal_status)
#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/fwd.hpp>
#include <psitri/lock_policy.hpp>

#include <concepts>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace psitri::dwal
{
   template <class E>
   concept kv_engine =
       requires {
          typename E::database;
          typename E::write_session;
          typename E::read_session;
          typename E::transaction;
          typename E::cursor;
       } &&
       requires(typename E::database& db) {
          { db.start_write_session() } -> std::convertible_to<std::shared_ptr<typename E::write_session>>;
          { db.start_read_session() } -> std::convertible_to<std::shared_ptr<typename E::read_session>>;
       } &&
       requires(typename E::write_session& ws, uint32_t idx) {
          { ws.start_transaction(idx) } -> std::same_as<typename E::transaction>;
       } &&
       requires(typename E::transaction& tx, std::string_view k, std::string_view v) {
          tx.upsert(k, v);
          tx.remove(k);
          tx.commit();
          tx.abort();
       } &&
       requires(typename E::read_session& rs, uint32_t idx) {
          { rs.create_cursor(idx) } -> std::same_as<typename E::cursor>;
       } &&
       requires(typename E::cursor& cur, std::string_view k) {
          { cur.seek_begin() } -> std::convertible_to<bool>;
          { cur.seek_last() } -> std::convertible_to<bool>;
          { cur.lower_bound(k) } -> std::convertible_to<bool>;
          { cur.upper_bound(k) } -> std::convertible_to<bool>;
          { cur.next() } -> std::convertible_to<bool>;
          { cur.prev() } -> std::convertible_to<bool>;
          { cur.is_end() } -> std::convertible_to<bool>;
          { cur.is_rend() } -> std::convertible_to<bool>;
          { cur.is_subtree() } -> std::convertible_to<bool>;
          { cur.key() } -> std::convertible_to<std::string_view>;
          { cur.template get<std::string>(k) } -> std::convertible_to<std::optional<std::string>>;
          { cur.template value<std::string>() } -> std::convertible_to<std::optional<std::string>>;
       };

   /// psitri as the DWAL base tier (the historical hard-wiring, now an adapter).
   template <class LockPolicy = std_lock_policy>
   struct psitri_engine
   {
      using database      = psitri::basic_database<LockPolicy>;
      using write_session = psitri::basic_write_session<LockPolicy>;
      using read_session  = psitri::basic_read_session<LockPolicy>;
      using transaction   = psitri::transaction;
      using cursor        = psitri::cursor;
   };
}  // namespace psitri::dwal

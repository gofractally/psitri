// part::shape — C++ concepts that type-check the psitri/heart/part database
// shape (SPEC §12.3). The shape is enforced by the compiler, not convention:
// psitri::database and heart::kv::database must PASS these (validating the
// concepts against ground truth — see test/shape_check.cpp), and part's
// classes static_assert them so a drift is a compile error naming the class,
// not a harness surprise. Zero dependencies beyond the standard library.
#pragma once
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace part::shape
{
   template <class C>
   concept kv_cursor = requires(C c, std::string_view k) {
      c.seek_begin();
      c.lower_bound(k);
      c.next();
      { c.is_end() } -> std::convertible_to<bool>;
      { c.key() } -> std::convertible_to<std::string_view>;
      c.template value<std::string>();  // optional-like result
   };

   template <class T>
   concept kv_tree = requires(const T t) {
      { t.snapshot_cursor() } -> kv_cursor;
   };

   template <class TX>
   concept kv_transaction = requires(TX tx, std::string_view k, std::string_view v) {
      tx.upsert(k, v);
      { tx.remove(k) } -> std::convertible_to<int>;
      tx.commit();
      { tx.template get<std::string>(k) };                          // optional-like
      { tx.get(k, [](std::string_view) {}) } -> std::convertible_to<bool>;  // lambda form
   };

   template <class WS>
   concept kv_write_session = requires(WS ws, uint32_t root) {
      { ws.start_transaction(root) } -> kv_transaction;
      { ws.get_root(root) } -> kv_tree;
   };

   template <class DB>
   concept kv_database = requires(DB& db, std::filesystem::path dir) {
      { db.start_write_session() };
      requires kv_write_session<
          typename std::pointer_traits<decltype(db.start_write_session())>::element_type>;
   };

   // ═══ Concept extensions (locked 2026-06-12, Dan; conformance enforced by
   // kv-compat running these feature batteries on EVERY engine) ═══════════

   // ── typed values with lifecycle hooks ─────────────────────────────────
   // A value slot is (T, blob); T = 0 = POD (today's behavior, no hooks).
   // Registered types get retain/release callbacks invoked at EXACTLY the
   // points subtrees are retained/released (leaf copy/share; leaf destroy:
   // overwrite, remove, abort, background reclaim). Hooks run on GC threads
   // too: they must be thread-safe and non-blocking (sal type-ops contract).
   // Native subtrees are the engine's built-in registration of this same
   // mechanism; user types (e.g. tiered's ttree ids) bring their own hooks.
   struct value_type_ops
   {
      // payload = the stored blob; both may run on background reclaim threads
      void (*retain)(std::string_view payload, void* user);
      void (*release)(std::string_view payload, void* user);
      void* user = nullptr;
   };

   template <class DB>
   concept typed_values =
       requires(DB& db, uint16_t type_id, value_type_ops ops) {
          db.register_value_type(type_id, ops);
       } &&
       requires(typename std::pointer_traits<decltype(std::declval<DB&>().start_write_session())>::
                    element_type& ws,
                std::string_view k, std::string_view v, uint16_t t) {
          { ws.start_transaction(0) };
          // typed write: the engine stores (t, v) and fires hooks on t != 0
          std::declval<decltype(ws.start_transaction(0))&>().upsert(k, v, t);
       };

   // ── multi-root transactions (psitri's model) ──────────────────────────
   // open_root in STRICTLY ASCENDING index order (deadlock-free by
   // construction); one tree_handle abstraction targets any held root;
   // abort rolls back all roots; cross-root publish is ORDERED (descending
   // index), not atomic — engines with a WAL may strengthen.
   // set_root(tree) -> tree is THE exchange: assign, return the displaced
   // tree as the engine's owned-tree value.
   template <class H>
   concept root_handle = requires(H h, std::string_view k, std::string_view v) {
      h.upsert(k, v);
      { h.remove(k) } -> std::convertible_to<int>;
      { h.template get<std::string>(k) };
      { h.set_root({}) };               // assign empty, RETURN the displaced tree
      { h.set_root(h.set_root({})) };   // the rotation composes from the exchange
   };

   template <class TX>
   concept multi_root_transaction = requires(TX tx, uint32_t idx) {
      { tx.primary() } -> root_handle;
      { tx.open_root(idx) } -> root_handle;
   };

   // ── snapshots are stored trees (composable subtrees, §8 identity) ─────
   // snapshot() = O(1) shared owned-tree of the live root; storing one is
   // just a (typed) value write at a HIGHER layer (engine-opaque), or the
   // engine's native subtree support where it exists. The concept requires
   // only the snapshot primitive; identity management lives above.
   template <class WS>
   concept tree_snapshots = requires(WS& ws) {
      { ws.snapshot(0u) };  // owned-tree value for root 0's committed state
   };
}  // namespace part::shape

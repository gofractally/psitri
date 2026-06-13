// adb — abstract database concepts: the engine shape every psitri library
// implements (database → sessions → transactions → cursors + the locked
// concept extensions). Enforced by the compiler, not convention: every
// engine static_asserts these and the behavioral battery instantiates
// against them, so drift is a compile error naming the class, not a
// harness surprise. Zero dependencies beyond the standard library.
#pragma once
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace adb
{
   // Base forward cursor. Positioning returns bool (landed-on-a-key) — a void
   // return passes a naive `c.seek_begin();` check but can't drive a merge
   // (`bool more = c.seek_begin()`), so it is REQUIRED here: an engine whose
   // cursor returns void is not composable, and that must be a compile error
   // at conformance, not a surprise at the first delta_art instantiation.
   template <class C>
   concept kv_cursor = requires(C c, std::string_view k) {
      { c.seek_begin() } -> std::convertible_to<bool>;
      { c.lower_bound(k) } -> std::convertible_to<bool>;
      { c.next() } -> std::convertible_to<bool>;
      { c.is_end() } -> std::convertible_to<bool>;
      { c.key() } -> std::convertible_to<std::string_view>;
      c.template value<std::string>();  // optional-like result
   };

   // Mergeable cursor: the full bidirectional surface a tiered/merge engine
   // (delta_art) drives — reverse iteration, point reads, subtree probe. An
   // engine usable as a delta_art tier MUST satisfy this; the base kv_cursor
   // alone does not (delta_art static_asserts this on its W and B cursors).
   template <class C>
   concept mergeable_cursor = kv_cursor<C> && requires(C c, std::string_view k) {
      { c.seek_last() } -> std::convertible_to<bool>;
      { c.upper_bound(k) } -> std::convertible_to<bool>;
      { c.prev() } -> std::convertible_to<bool>;
      { c.is_rend() } -> std::convertible_to<bool>;
      { c.is_subtree() } -> std::convertible_to<bool>;
      { c.get_value([](std::string_view) {}) } -> std::convertible_to<bool>;
      { c.get(k, [](std::string_view) {}) } -> std::convertible_to<bool>;
      { c.template get<std::string>(k) };  // optional-like
   };

   template <class T>
   concept kv_tree = requires(const T t) {
      { t.snapshot_cursor() } -> kv_cursor;
   };

   // An owned snapshot usable as a shared warm/merge leg: COPYABLE (multiple
   // holders — merger flush cursor, reader/writer warm legs — each keep it
   // alive) and readable through a mergeable cursor. delta_art's warm handle
   // (get_root's return) must satisfy this; a move-only one-ref tree does not.
   template <class T>
   concept shared_snapshot = std::copy_constructible<T> && requires(const T t) {
      { t.snapshot_cursor() } -> mergeable_cursor;
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

   // ── typed values: POD tokens + traits (Dan's take/make model, §9/§10) ──
   // A value slot is (typenum, pod bytes); typenum 0 = plain bytes (today's
   // zero-cost path). A storable type T provides:
   //   T::type_id                                    (collisions assert)
   //   T::t_pod  — trivially copyable token
   //   t_pod T::take_pod() &&                        (CONSUMES the ref)
   //   static T T::make_from_pod(const t_pod&, bool retain)   (MINTS a ref)
   //   static void t_pod::retain (void* session, const t_pod&)
   //   static void t_pod::release(void* session, const t_pod&)
   // The engine fires retain at exactly the subtree-retain points (leaf
   // copy/share) and release at the subtree-release points (overwrite,
   // remove, abort, background reclaim — hooks run on GC threads: they must
   // be thread-safe and non-blocking). Reads: THREE OUTCOMES everywhere —
   // found / not-found / INVALID-TYPE (wrong-pod reads are a type error,
   // never a fake absence, never a reinterpret).
   template <class T>
   concept storable_type = requires(T t, const typename T::t_pod& p, void* ses) {
      { T::type_id } -> std::convertible_to<uint16_t>;
      requires std::is_trivially_copyable_v<typename T::t_pod>;
      { std::move(t).take_pod() } -> std::same_as<typename T::t_pod>;
      { T::make_from_pod(p, true) } -> std::same_as<T>;
      T::t_pod::retain(ses, p);
      T::t_pod::release(ses, p);
   };

   // any_value — the dynamic, LOCK-BUNDLING read form (§10): returned by
   // get/find without a type argument; holds {typenum, pod, read lock}.
   // as<T>() retains (the only civilian exit); bytes() on typenum != 0 is
   // invalid_type (a typed pod exposed raw would be a forgeable handle);
   // raw_pod() is the expert/merger identity path, balanced by the typed
   // STORE form firing the retain hook.
   template <class V, class T>
   concept any_value_for = requires(V v) {
      { v.type() } -> std::convertible_to<uint16_t>;
      { v.template is<T>() } -> std::convertible_to<bool>;
      { v.template as<T>() } -> std::same_as<T>;
      { v.bytes() } -> std::convertible_to<std::string_view>;
      { v.raw_pod() } -> std::convertible_to<std::string_view>;
      { bool(v) };  // empty = not-found
   };

   template <class DB, class T>
   concept typed_values =
       storable_type<T> &&
       requires(DB& db) { db.template register_value_types<T>(); } &&
       requires(typename std::pointer_traits<decltype(std::declval<DB&>().start_write_session())>::
                    element_type& ws,
                std::string_view k, T t) {
          { ws.start_transaction(0) };
          std::declval<decltype(ws.start_transaction(0))&>().upsert(k, std::move(t));
          {
             std::declval<decltype(ws.start_transaction(0))&>().template get<T>(k)
          };  // retained T or throws invalid_type
          requires any_value_for<
              std::remove_cvref_t<decltype(std::declval<decltype(ws.start_transaction(0))&>().get_any(k))>,
              T>;
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

   // ── a tree IS a snapshot ──────────────────────────────────────────────
   // There is no separate snapshot() primitive: kv_write_session::get_root(i)
   // (base concept above) returns the owned-tree value, which is a FROZEN
   // snapshot of root i's committed state — under COW the owned reference
   // keeps that version alive across later commits, so the holder keeps
   // reading the contents as of the get_root() call (call get_root again to
   // observe later state). That same owned tree is what set_root() exchanges
   // and what a higher layer stores as a value (composable subtrees, §8).
   // No live-vs-frozen duality, no owned_tree alias — one type per engine.
   template <class WS>
   concept tree_snapshots = requires(WS& ws) {
      { ws.get_root(0u) };  // the owned, frozen snapshot of root 0
   };
}  // namespace adb

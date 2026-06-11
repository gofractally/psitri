// vpart::shape — C++ concepts that type-check the psitri/heart/part database
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

namespace vpart::shape
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
}  // namespace vpart::shape

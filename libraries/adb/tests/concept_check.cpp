// adb concept self-check — a minimal mock engine that SATISFIES every adb
// concept, plus negative checks. This guards the concepts themselves (they
// compile, they bind, they reject junk); engine ground-truth conformance is
// the behavioral battery's job (tests/kv_compat).
#include <adb/concepts.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace mock
{
   struct cursor
   {
      // positioning returns bool (landed-on-a-key) — required by kv_cursor
      bool seek_begin() { return false; }
      bool lower_bound(std::string_view) { return false; }
      bool next() { return false; }
      // bidirectional surface — required by mergeable_cursor
      bool seek_last() { return false; }
      bool upper_bound(std::string_view) { return false; }
      bool prev() { return false; }
      bool             is_rend() const { return false; }
      bool             is_subtree() const { return false; }
      template <class F>
      bool get_value(F&&) const
      {
         return false;
      }
      template <class F>
      bool get(std::string_view, F&&) const
      {
         return false;
      }
      template <class S = std::string>
      std::optional<S> get(std::string_view) const
      {
         return std::nullopt;
      }
      bool             is_end() const { return true; }
      std::string_view key() const { return {}; }
      template <class S>
      std::optional<S> value() const
      {
         return std::nullopt;
      }
   };
   struct tree
   {
      cursor snapshot_cursor() const { return {}; }
   };
   struct owned_tree
   {
   };
   struct root_handle
   {
      void upsert(std::string_view, std::string_view) {}
      int  remove(std::string_view) { return 0; }
      template <class S>
      std::optional<S> get(std::string_view) const
      {
         return std::nullopt;
      }
      owned_tree set_root(owned_tree) { return {}; }
   };
   struct transaction
   {
      void upsert(std::string_view, std::string_view) {}
      int  remove(std::string_view) { return 0; }
      void commit() {}
      template <class S>
      std::optional<S> get(std::string_view) const
      {
         return std::nullopt;
      }
      template <class F>
      bool get(std::string_view, F&&) const
      {
         return false;
      }
      root_handle primary() { return {}; }
      root_handle open_root(uint32_t) { return {}; }
   };
   struct write_session
   {
      transaction start_transaction(uint32_t) { return {}; }
      tree        get_root(uint32_t) { return {}; }
      owned_tree  snapshot(uint32_t) { return {}; }
   };
   struct database
   {
      std::shared_ptr<write_session> start_write_session() { return {}; }
   };

   struct ref
   {
      static constexpr uint16_t type_id = 1;
      struct t_pod
      {
         uint64_t    id;
         static void retain(void*, const t_pod&) {}
         static void release(void*, const t_pod&) {}
      };
      t_pod      take_pod() && { return {0}; }
      static ref make_from_pod(const t_pod&, bool) { return {}; }
   };
   struct any_value
   {
      uint16_t type() const { return 0; }
      template <class T>
      bool is() const
      {
         return false;
      }
      template <class T>
      T as() const
      {
         return {};
      }
      std::string_view bytes() const { return {}; }
      std::string_view raw_pod() const { return {}; }
      explicit operator bool() const { return false; }
   };
}  // namespace mock

static_assert(adb::kv_cursor<mock::cursor>);
static_assert(adb::mergeable_cursor<mock::cursor>);   // bidirectional merge surface
static_assert(adb::kv_tree<mock::tree>);
static_assert(adb::shared_snapshot<mock::tree>);      // copyable + mergeable cursor
static_assert(adb::kv_transaction<mock::transaction>);
static_assert(adb::kv_write_session<mock::write_session>);
static_assert(adb::kv_database<mock::database>);
static_assert(adb::storable_type<mock::ref>);
static_assert(adb::any_value_for<mock::any_value, mock::ref>);
static_assert(adb::root_handle<mock::root_handle>);
static_assert(adb::multi_root_transaction<mock::transaction>);
static_assert(adb::tree_snapshots<mock::write_session>);

static_assert(!adb::kv_database<int>);
static_assert(!adb::kv_cursor<std::string>);
static_assert(!adb::storable_type<std::string>);

int main()
{
   return 0;  // the assertions above ARE the test
}

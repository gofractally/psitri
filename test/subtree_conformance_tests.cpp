// Differential subtree conformance: the SAME test bodies (written in psitri's public API verbatim)
// run against BOTH engines. The engine is selected by a PURE type-alias policy — `using database
// = <ns>::database;` — and everything else is deduced with `auto`/`auto&&`. There is NO adapter and
// NO facade: psitri is the identity (its own API), and heart provides the same surface natively
// (heart::radix::database -> write_session -> transaction / tree / tree_handle / cursor). A
// behavioral divergence shows up as one engine FAIL / the other PASS on the identical assertion.
//
// `auto&&` for handles bridges the one shape difference: psitri's tree_handle is a copyable
// targeting scope (returned by value), heart's is a move-only owner (returned by reference).
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <optional>
#include <string>

#include <radix_engine.hpp>  // heart engine + its native psitri-shaped public API

#include <psitri/cursor.hpp>
#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_session_impl.hpp>

namespace
{
   std::filesystem::path conf_dir(const char* eng)
   {
      auto d = std::filesystem::temp_directory_path() / (std::string("subtree_conf_") + eng);
      std::filesystem::remove_all(d);
      std::filesystem::create_directories(d);
      return d;
   }
   using sopt = std::optional<std::string>;
   sopt S(std::string_view s) { return sopt(std::string(s)); }

   // The ONLY engine-specific code: which namespace's `database` to use. Pure aliases — no logic.
   struct psitri_policy
   {
      static constexpr const char* nm = "psitri";
      using database                  = psitri::database;
   };
   struct heart_policy
   {
      static constexpr const char* nm = "heart";
      using database                  = heart::radix::database;
   };
}  // namespace

// Build a detached subtree, attach it, read it back (count + values) + a sibling byte value.
TEMPLATE_TEST_CASE("conformance: subtree store + retrieve", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   {
      auto tx  = ses->start_transaction(0);
      auto sub = ses->start_write_transaction(ses->create_temporary_tree());
      sub.upsert("a", "1");
      sub.upsert("b", "2");
      tx.upsert_subtree("child", sub.get_tree());
      tx.upsert("data", "root");
      tx.commit();
   }
   auto root = ses->get_root(0);
   CHECK(root.template get<std::string>("data") == S("root"));
   auto child = root.get_subtree("child");
   REQUIRE(bool(child));
   CHECK(child.snapshot_cursor().count_keys() == 2);
   CHECK(child.template get<std::string>("a") == S("1"));
   CHECK(child.template get<std::string>("b") == S("2"));
}

// discrimination: a subtree key is a subtree; a byte key is not.
TEMPLATE_TEST_CASE("conformance: subtree discrimination", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   {
      auto tx  = ses->start_transaction(0);
      auto sub = ses->start_write_transaction(ses->create_temporary_tree());
      sub.upsert("k", "1");
      tx.upsert_subtree("tree", sub.get_tree());
      tx.upsert("plain", "v");
      tx.commit();
   }
   {
      auto tx = ses->start_transaction(0);
      CHECK(tx.is_subtree("tree"));
      CHECK_FALSE(tx.is_subtree("plain"));
      CHECK(tx.template get<std::string>("plain") == S("v"));
      tx.abort();
   }
   CHECK(bool(ses->get_root(0).get_subtree("tree")));
   CHECK_FALSE(bool(ses->get_root(0).get_subtree("plain")));
}

// in-place: create a subtree via the primary handle, populate, persist.
TEMPLATE_TEST_CASE("conformance: create_subtree via primary handle", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   {
      auto   tx = ses->start_transaction(0);
      auto&& h  = tx.primary();
      auto&& s  = h.create_subtree("new_tree");
      s.upsert("inner_a", "va");
      s.upsert("inner_b", "vb");
      tx.commit();
   }
   auto sub = ses->get_root(0).get_subtree("new_tree");
   REQUIRE(bool(sub));
   CHECK(sub.snapshot_cursor().count_keys() == 2);
   CHECK(sub.template get<std::string>("inner_a") == S("va"));
}

// nested subtrees (tree of trees) created via handles, written back bottom-up.
TEMPLATE_TEST_CASE("conformance: nested subtrees", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   {
      auto   tx    = ses->start_transaction(0);
      auto&& h     = tx.primary();
      auto&& outer = h.create_subtree("outer");
      outer.upsert("od", "hi");
      auto&& inner = outer.create_subtree("inner");
      inner.upsert("deep", "v");
      tx.commit();
   }
   auto outer = ses->get_root(0).get_subtree("outer");
   REQUIRE(bool(outer));
   CHECK(outer.template get<std::string>("od") == S("hi"));
   auto inner = outer.get_subtree("inner");
   REQUIRE(bool(inner));
   CHECK(inner.template get<std::string>("deep") == S("v"));
}

// multiple sibling subtrees in one transaction all persist.
TEMPLATE_TEST_CASE("conformance: multiple subtrees", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   {
      auto tx = ses->start_transaction(0);
      for (int i = 0; i < 6; ++i)
      {
         auto sub = ses->start_write_transaction(ses->create_temporary_tree());
         sub.upsert("v", std::to_string(i));
         tx.upsert_subtree("t" + std::to_string(i), sub.get_tree());
      }
      tx.commit();
   }
   auto root = ses->get_root(0);
   for (int i = 0; i < 6; ++i)
   {
      auto s = root.get_subtree("t" + std::to_string(i));
      REQUIRE(bool(s));
      CHECK(s.template get<std::string>("v") == S(std::to_string(i)));
   }
}

// abort discards a stored subtree (its key never appears).
TEMPLATE_TEST_CASE("conformance: abort discards subtree", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   {
      auto tx  = ses->start_transaction(0);
      auto sub = ses->start_write_transaction(ses->create_temporary_tree());
      sub.upsert("k", "v");
      tx.upsert_subtree("gone", sub.get_tree());
      tx.abort();
   }
   CHECK_FALSE(bool(ses->get_root(0).get_subtree("gone")));
}

// Differential subtree conformance: the SAME templated tests run against BOTH engines (heart's
// radix_tree and psitri), via thin adapters that expose a uniform surface. A divergence shows up
// as one engine FAIL / the other PASS on the same assertion — porting psitri's battle-tested
// subtree semantics onto heart as a hard gate.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include <radix_engine.hpp>  // heart engine (header)

#include <psitri/database.hpp>
#include <psitri/transaction.hpp>
#include <psitri/cursor.hpp>
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

   // ── heart adapter ──────────────────────────────────────────────────────────────────────
   struct heart_eng
   {
      static const char*               name() { return "heart"; }
      heart::radix::radix_tree         t;
      explicit heart_eng() : t(conf_dir("heart")) {}
      using HT = heart::radix::radix_tree;

      struct Sub
      {
         HT::tree_handle* h;
         void             upsert(std::string_view k, std::string_view v) { h->upsert(k, v); }
         sopt             get(std::string_view k) { return h->get<std::string>(k); }
         uint64_t         count() { return h->count_keys(); }
         Sub              create(std::string_view k) { return {&h->create_subtree(k)}; }
         Sub              open(std::string_view k) { return {&h->open_subtree(k)}; }
      };
      struct Writer
      {
         HT::transaction tx;
         void            upsert(std::string_view k, std::string_view v) { tx.upsert(k, v); }
         int             remove(std::string_view k) { return tx.remove(k); }
         bool            is_subtree(std::string_view k) { return tx.is_subtree(k); }
         sopt            get(std::string_view k) { return tx.get<std::string>(k); }
         Sub             create(std::string_view k) { return {&tx.create_subtree(k)}; }
         Sub             open(std::string_view k) { return {&tx.open_subtree(k)}; }
         void            commit() { tx.commit(); }
         void            abort() { tx.abort(); }
      };
      Writer writer(uint32_t idx = 0) { return Writer{t.start_transaction(idx)}; }

      struct Reader
      {
         HT::read_handle r;
         explicit        operator bool() const { return bool(r); }
         sopt            get(std::string_view k) { return r.get<std::string>(k); }
         bool            is_subtree(std::string_view k) { return r.is_subtree(k); }
         Reader          sub(std::string_view k) { return {r.get_subtree(k)}; }
         uint64_t        count() { return r.count_keys(); }
      };
      Reader reader(uint32_t idx = 0) { return Reader{t.get_root(idx)}; }
   };

   // ── psitri adapter ─────────────────────────────────────────────────────────────────────
   struct psitri_eng
   {
      static const char*                     name() { return "psitri"; }
      std::shared_ptr<psitri::database>      db;
      std::shared_ptr<psitri::write_session> ses;
      explicit psitri_eng()
      {
         db  = psitri::database::open(conf_dir("psitri").string());
         ses = db->start_write_session();
      }
      struct Sub
      {
         psitri::tree_handle h;
         void                upsert(std::string_view k, std::string_view v) { h.upsert(k, v); }
         sopt                get(std::string_view k) { return h.get<std::string>(k); }
         uint64_t            count() { return h.snapshot_cursor().count_keys(); }
         Sub                 create(std::string_view k) { return {h.create_subtree(k)}; }
         Sub                 open(std::string_view k) { return {h.open_subtree(k)}; }
      };
      struct Writer
      {
         psitri::transaction tx;
         void                upsert(std::string_view k, std::string_view v) { tx.upsert(k, v); }
         int                 remove(std::string_view k) { return tx.remove(k); }
         bool                is_subtree(std::string_view k) { return tx.primary().is_subtree(k); }
         sopt                get(std::string_view k) { return tx.get<std::string>(k); }
         Sub                 create(std::string_view k) { return {tx.primary().create_subtree(k)}; }
         Sub                 open(std::string_view k) { return {tx.primary().open_subtree(k)}; }
         void                commit() { tx.commit(); }
         void                abort() { tx.abort(); }
      };
      Writer writer(uint32_t idx = 0) { return Writer{ses->start_transaction(idx)}; }

      struct Reader
      {
         psitri::tree r;
         explicit     operator bool() const { return bool(r); }
         sopt         get(std::string_view k) { return r.get<std::string>(k); }
         bool         is_subtree(std::string_view k) { return bool(r.get_subtree(k)); }
         Reader       sub(std::string_view k) { return {r.get_subtree(k)}; }
         uint64_t count() { return r.cursor().count_keys(); }
      };
      Reader reader(uint32_t idx = 0) { return Reader{ses->get_root(idx)}; }
   };
}  // namespace

// store a subtree, read it back (count + values) + a sibling byte value.
TEMPLATE_TEST_CASE("conformance: subtree store + retrieve", "[conformance]", heart_eng, psitri_eng)
{
   TestType eng;
   {
      auto w = eng.writer();
      auto s = w.create("child");
      s.upsert("a", "1");
      s.upsert("b", "2");
      w.upsert("data", "root");
      w.commit();
   }
   auto r = eng.reader();
   CHECK(r.get("data") == S("root"));
   auto sub = r.sub("child");
   REQUIRE(bool(sub));
   CHECK(sub.count() == 2);
   CHECK(sub.get("a") == S("1"));
   CHECK(sub.get("b") == S("2"));
}

// discrimination: a subtree key is a subtree; a byte key is not.
TEMPLATE_TEST_CASE("conformance: subtree discrimination", "[conformance]", heart_eng, psitri_eng)
{
   TestType eng;
   {
      auto w = eng.writer();
      w.upsert("plain", "v");
      auto s = w.create("tree");
      s.upsert("k", "1");
      w.commit();
   }
   {
      auto w = eng.writer();
      CHECK(w.is_subtree("tree"));
      CHECK_FALSE(w.is_subtree("plain"));
      CHECK(w.get("plain") == S("v"));
      w.abort();
   }
   CHECK(eng.reader().is_subtree("tree"));
   CHECK_FALSE(eng.reader().is_subtree("plain"));
}

// open an existing subtree, modify, persist.
TEMPLATE_TEST_CASE("conformance: open_subtree modify persists", "[conformance]", heart_eng, psitri_eng)
{
   TestType eng;
   {
      auto w = eng.writer();
      auto s = w.create("child");
      s.upsert("a", "1");
      s.upsert("b", "2");
      w.commit();
   }
   {
      auto w = eng.writer();
      auto s = w.open("child");
      CHECK(s.get("a") == S("1"));
      s.upsert("c", "3");
      w.commit();
   }
   auto sub = eng.reader().sub("child");
   REQUIRE(bool(sub));
   CHECK(sub.get("c") == S("3"));
   CHECK(sub.count() == 3);
}

// nested subtrees (tree of trees) written back bottom-up.
TEMPLATE_TEST_CASE("conformance: nested subtrees", "[conformance]", heart_eng, psitri_eng)
{
   TestType eng;
   {
      auto w     = eng.writer();
      auto outer = w.create("outer");
      outer.upsert("od", "hi");
      auto inner = outer.create("inner");
      inner.upsert("deep", "v");
      w.commit();
   }
   auto outer = eng.reader().sub("outer");
   REQUIRE(bool(outer));
   CHECK(outer.get("od") == S("hi"));
   auto inner = outer.sub("inner");
   REQUIRE(bool(inner));
   CHECK(inner.get("deep") == S("v"));
}

// multiple sibling subtrees in one transaction all persist.
TEMPLATE_TEST_CASE("conformance: multiple subtrees", "[conformance]", heart_eng, psitri_eng)
{
   TestType eng;
   {
      auto w = eng.writer();
      for (int i = 0; i < 6; ++i)
      {
         auto s = w.create("t" + std::to_string(i));
         s.upsert("v", std::to_string(i));
      }
      w.commit();
   }
   auto r = eng.reader();
   for (int i = 0; i < 6; ++i)
   {
      auto s = r.sub("t" + std::to_string(i));
      REQUIRE(bool(s));
      CHECK(s.get("v") == S(std::to_string(i)));
   }
}

// abort discards a stored subtree (and its contents never appear).
TEMPLATE_TEST_CASE("conformance: abort discards subtree", "[conformance]", heart_eng, psitri_eng)
{
   TestType eng;
   {
      auto w = eng.writer();
      auto s = w.create("gone");
      s.upsert("k", "v");
      w.abort();
   }
   CHECK_FALSE(bool(eng.reader().sub("gone")));
}

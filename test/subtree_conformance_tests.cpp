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
#include <vector>

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

// pinned_value(): the symmetric lock-bundling proxy returns a value view kept valid by a held
// read_lock (same type/shape on both engines).
TEMPLATE_TEST_CASE("conformance: pinned_value borrowed view", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   {
      auto tx  = ses->start_transaction(0);
      auto sub = ses->start_write_transaction(ses->create_temporary_tree());
      sub.upsert("k", "hello-value");
      tx.upsert_subtree("w", sub.get_tree());
      tx.commit();
   }
   auto wsub = ses->get_root(0).get_subtree("w");
   REQUIRE(bool(wsub));
   auto c = wsub.snapshot_cursor();
   REQUIRE(c.find("k"));  // position at "k"; find() returns true when positioned on the key
   CHECK(c.key() == "k");
   auto v = c.pinned_value();  // value_ref on both engines; holds the read_lock while alive
   CHECK(std::string_view(v) == "hello-value");  // value_ref → value_view (std::string_view)
   CHECK(v.str() == "hello-value");
}

// cursor positioning convention parity: first/last/seek/lower_bound/upper_bound all follow the
// "true = positioned on a valid entry" convention on BOTH engines (psitri is the reference).
TEMPLATE_TEST_CASE("conformance: cursor positioning convention", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   {
      auto tx  = ses->start_transaction(0);
      auto sub = ses->start_write_transaction(ses->create_temporary_tree());
      for (const char* k : {"b", "d", "f"}) sub.upsert(k, "v");
      tx.upsert_subtree("w", sub.get_tree());
      tx.commit();
   }
   auto t = ses->get_root(0).get_subtree("w");
   REQUIRE(bool(t));

   auto c = t.snapshot_cursor();
   CHECK(c.first());  // true = positioned
   CHECK(c.key() == "b");
   CHECK(c.last());
   CHECK(c.key() == "f");

   // lower_bound: >= ; upper_bound: strictly >
   CHECK(c.lower_bound("d"));
   CHECK(c.key() == "d");
   CHECK(c.lower_bound("c"));
   CHECK(c.key() == "d");
   CHECK(c.upper_bound("d"));
   CHECK(c.key() == "f");
   CHECK_FALSE(c.lower_bound("z"));  // nothing >= z → at end, false
   CHECK_FALSE(c.upper_bound("f"));  // nothing > f → at end, false

   // seek: deprecated exact alias — true only on exact match
   CHECK(c.seek("d"));         // exact present
   CHECK_FALSE(c.seek("c"));   // absent (but positioned at lower_bound 'd')
   CHECK(c.key() == "d");

   // find: exact lookup, same convention
   CHECK(c.find("f"));
   CHECK_FALSE(c.find("e"));
}

// replace a subtree value with a different subtree — new content visible, no stale data.
TEMPLATE_TEST_CASE("conformance: replace subtree value", "[conformance]", psitri_policy, heart_policy)
{
   auto db  = TestType::database::open(conf_dir(TestType::nm));
   auto ses = db->start_write_session();
   auto attach = [&](const char* k, int n, const char* prefix) {
      auto tx  = ses->start_transaction(0);
      auto sub = ses->start_write_transaction(ses->create_temporary_tree());
      for (int i = 0; i < n; ++i) sub.upsert(std::string(prefix) + std::to_string(i), "v");
      tx.upsert_subtree(k, sub.get_tree());
      tx.commit();
   };
   attach("tree", 5, "first");
   CHECK(ses->get_root(0).get_subtree("tree").snapshot_cursor().count_keys() == 5);
   attach("tree", 9, "second");  // replace
   auto sub = ses->get_root(0).get_subtree("tree");
   REQUIRE(bool(sub));
   CHECK(sub.snapshot_cursor().count_keys() == 9);            // new content
   CHECK(sub.template get<std::string>("first0").has_value() == false);  // old gone
   CHECK(sub.template get<std::string>("second0") == S("v"));
}

// removing a subtree key frees the whole subtree — allocation returns to baseline (no leak).
TEMPLATE_TEST_CASE("conformance: subtree remove no leak", "[conformance][leak]", psitri_policy, heart_policy)
{
   auto db   = TestType::database::open(conf_dir(TestType::nm));
   auto ses  = db->start_write_session();
   const auto base = ses->get_total_allocated_objects();
   {
      auto tx  = ses->start_transaction(0);
      auto sub = ses->start_write_transaction(ses->create_temporary_tree());
      for (int i = 0; i < 50; ++i) sub.upsert("s" + std::to_string(i), "value");
      tx.upsert_subtree("tree", sub.get_tree());
      tx.commit();
   }
   CHECK(ses->get_total_allocated_objects() > base);  // it grew
   {
      auto tx = ses->start_transaction(0);
      tx.remove("tree");
      tx.commit();
   }
   REQUIRE(db->wait_for_compactor());
   CHECK(ses->get_total_allocated_objects() == base);  // each engine back to its own baseline
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

// count_keys on a subtree: cursor count_keys() == manual is_end()/next() iteration == N, on a
// multi-level subtree read through an UNCOMMITTED write_transaction (working-tree reads).
TEMPLATE_TEST_CASE("conformance: count_keys on subtree", "[conformance]", psitri_policy, heart_policy)
{
   auto db   = TestType::database::open(conf_dir(TestType::nm));
   auto ses  = db->start_write_session();
   auto pad6 = [](int i) {  // zero-padded keys → good branch distribution (multi-level tree)
      std::string s = std::to_string(i);
      s.insert(0, 6 - s.size(), '0');
      return s;
   };
   const int N   = 500;
   auto      sub = ses->start_write_transaction(ses->create_temporary_tree());
   for (int i = 0; i < N; ++i) sub.upsert("sub" + pad6(i), "value_for_sub" + pad6(i));
   auto tx = ses->start_write_transaction(ses->create_temporary_tree());
   tx.upsert_subtree("tree", sub.get_tree());

   auto sub_root = tx.get_subtree("tree");
   REQUIRE(bool(sub_root));
   auto c = sub_root.snapshot_cursor();
   CHECK(c.count_keys() == uint64_t(N));
   uint64_t manual = 0;  // ground truth: manual iteration (psitri's count_via_cursor shape)
   c.seek_begin();
   while (!c.is_end())
   {
      ++manual;
      c.next();
   }
   CHECK(manual == uint64_t(N));
}

// large-value subtrees (values > inline threshold: psitri value_nodes / heart external leaves):
// attach, verify content, replace, remove — allocation returns to baseline (no leak).
TEMPLATE_TEST_CASE("conformance: large-value subtrees no leak", "[conformance][leak]", psitri_policy, heart_policy)
{
   auto db   = TestType::database::open(conf_dir(TestType::nm));
   auto ses  = db->start_write_session();
   const auto base = ses->get_total_allocated_objects();
   auto make_big = [&](int n, char tag) {  // n values of 200 B (> 64 B forces out-of-node storage)
      auto        sub = ses->start_write_transaction(ses->create_temporary_tree());
      std::string big(200, 'X');
      for (int i = 0; i < n; ++i)
      {
         big[0] = tag;
         big[1] = char('a' + i % 26);
         sub.upsert("big" + std::to_string(i), big);
      }
      return sub.get_tree();
   };
   {
      auto tx = ses->start_transaction(0);
      tx.upsert_subtree("big_tree", make_big(100, 'A'));
      tx.commit();
   }
   {
      auto sub = ses->get_root(0).get_subtree("big_tree");
      REQUIRE(bool(sub));
      CHECK(sub.snapshot_cursor().count_keys() == 100);
      auto v = sub.template get<std::string>("big7");
      REQUIRE(v.has_value());
      std::string expect(200, 'X');
      expect[0] = 'A';
      expect[1] = char('a' + 7);
      CHECK(*v == expect);
   }
   {
      auto tx = ses->start_transaction(0);
      tx.upsert_subtree("big_tree", make_big(60, 'B'));  // replace → the old subtree is released
      tx.commit();
   }
   CHECK(ses->get_root(0).get_subtree("big_tree").snapshot_cursor().count_keys() == 60);
   {
      auto tx = ses->start_transaction(0);
      tx.remove("big_tree");
      tx.commit();
   }
   REQUIRE(db->wait_for_compactor());
   CHECK(ses->get_total_allocated_objects() == base);
}

// update churn: transactional UPDATEs of existing keys (the COW shared-mode path — every commit
// re-clones the path) must not leak — allocation returns to baseline after dropping the keys.
// GATE for the heart two-cow orphan bug (one leaked clone per transactional update): the bank
// bench leaked ~2 objects/commit through exactly this shape until the update sites cow'd ONCE.
TEMPLATE_TEST_CASE("conformance: update churn no leak", "[conformance][leak]", psitri_policy, heart_policy)
{
   auto db   = TestType::database::open(conf_dir(TestType::nm));
   auto ses  = db->start_write_session();
   const auto base = ses->get_total_allocated_objects();
   {
      auto tx = ses->start_transaction(0);
      for (int i = 0; i < 200; ++i) tx.upsert("acct" + std::to_string(i), "00000000");
      tx.commit();
   }
   for (int round = 0; round < 100; ++round)  // bench transfer shape: batch=1 update commits
   {
      auto tx = ses->start_transaction(0);
      tx.upsert("acct" + std::to_string((7 * round) % 200), "11111111");
      tx.upsert("acct" + std::to_string((13 * round + 5) % 200), "22222222");
      tx.commit();
   }
   {
      auto c = ses->get_root(0).snapshot_cursor();
      CHECK(c.count_keys() == 200);  // updates added no keys
   }
   {
      auto tx = ses->start_transaction(0);
      tx.remove_range_counted("acct", "acct\xFF");
      tx.commit();
   }
   REQUIRE(db->wait_for_compactor());
   CHECK(ses->get_total_allocated_objects() == base);
}

// range_remove over a mix of byte keys and subtree keys: [key_02, key_08) removed (subtree values
// cascade-freed, each counting as ONE key of the parent), survivors exact; then a second range
// sweep empties the tree — allocation returns to baseline (no leak).
TEMPLATE_TEST_CASE("conformance: range_remove over subtree keys no leak", "[conformance][leak]", psitri_policy, heart_policy)
{
   auto db   = TestType::database::open(conf_dir(TestType::nm));
   auto ses  = db->start_write_session();
   const auto base = ses->get_total_allocated_objects();
   {
      auto tx = ses->start_transaction(0);
      for (int i = 0; i < 10; ++i)
      {
         std::string key = "key_0" + std::to_string(i);
         if (i % 3 == 0)  // key_00, key_03, key_06, key_09 hold subtrees
         {
            auto sub = ses->start_write_transaction(ses->create_temporary_tree());
            for (int j = 0; j < 40; ++j)
               sub.upsert("s" + std::to_string(i) + "_" + std::to_string(j), "v");
            tx.upsert_subtree(key, sub.get_tree());
         }
         else
            tx.upsert(key, "regular");
      }
      CHECK(tx.remove_range_counted("key_02", "key_08") == 6);  // 02..07; subtrees at 03 + 06
      auto rc = tx.snapshot_cursor();  // survivors: exact set, in order
      std::vector<std::string> remaining;
      rc.seek_begin();
      while (!rc.is_end())
      {
         remaining.push_back(std::string(rc.key()));
         rc.next();
      }
      REQUIRE(remaining.size() == 4);
      CHECK(remaining[0] == "key_00");
      CHECK(remaining[1] == "key_01");
      CHECK(remaining[2] == "key_08");
      CHECK(remaining[3] == "key_09");
      tx.commit();
   }
   {
      auto tx = ses->start_transaction(0);
      CHECK(tx.remove_range_counted("key_00", "key_99") == 4);  // sweep the survivors
      tx.commit();
   }
   REQUIRE(db->wait_for_compactor());
   CHECK(ses->get_total_allocated_objects() == base);
}

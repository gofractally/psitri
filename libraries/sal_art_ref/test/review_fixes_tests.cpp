// Defects found by the review of the Large Engine (gofractally/psitri#19).
// Each case is the reproduction of one issue and fails on the code before
// its fix.
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include <sal_art_ref/engine.hpp>

namespace
{
   using engine = sal_art_ref::radix::engine;
   using frame  = sal_art_ref::radix::radix_tree::frame;

   struct temp_dir
   {
      std::filesystem::path path;
      explicit temp_dir(const char* name)
          : path(std::filesystem::temp_directory_path() /
                 (std::string("psitri-review-") + name + "-" + std::to_string(::getpid())))
      {
         std::filesystem::remove_all(path);
      }
      ~temp_dir() { std::filesystem::remove_all(path); }
   };

   using model_t = std::map<std::string, std::string>;

   // Every key of the model reads back by key, and a forward and a backward walk
   // of the committed root return exactly the model's keys and values in order.
   void check_against(engine::write_session& ws, const model_t& model)
   {
      auto tree = ws.get_root(0);
      for (const auto& [k, v] : model)
      {
         auto got = tree.get<std::string>(k);
         INFO("key of " << k.size() << " bytes");
         REQUIRE(got.has_value());
         REQUIRE(*got == v);
      }
      auto   c  = tree.snapshot_cursor();
      auto   it = model.begin();
      size_t n  = 0;
      for (bool ok = c.seek_begin(); ok; ok = c.next(), ++it, ++n)
      {
         REQUIRE(it != model.end());
         REQUIRE(std::string(c.key()) == it->first);
         REQUIRE(*c.value<std::string>() == it->second);
      }
      REQUIRE(n == model.size());
      auto rit = model.rbegin();
      n        = 0;
      for (bool ok = c.seek_last(); ok; ok = c.prev(), ++rit, ++n)
      {
         REQUIRE(rit != model.rend());
         REQUIRE(std::string(c.key()) == rit->first);
      }
      REQUIRE(n == model.size());
   }
   void put(engine::write_session& ws, model_t& model, const std::string& k, const std::string& v)
   {
      auto tx = ws.start_transaction(0);
      tx.upsert(k, v);
      tx.commit();
      model[k] = v;
   }
   void del(engine::write_session& ws, model_t& model, const std::string& k)
   {
      auto tx = ws.start_transaction(0);
      CHECK(tx.remove(k) == (model.count(k) ? 0 : -1));
      tx.commit();
      model.erase(k);
   }
}  // namespace

// gofractally/psitri#26: a moved-from savepoint stayed active, and its destructor
// put its empty saved root in place of the transaction's working root.
TEST_CASE("review: a savepoint that was moved does not wipe the working root", "[review]")
{
   temp_dir dir("frame-move");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 100; ++i)
         tx.upsert("base" + std::to_string(i), "v");
      tx.commit();
   }
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("in-tx", "v");
      {
         std::optional<frame> sp;
         sp.emplace(tx.sub_transaction());  // move-constructed from the temporary
         sp->upsert("in-frame", "v");
         sp->commit();
      }
      CHECK(tx.snapshot_cursor().count_keys() == 102);
      {
         // A moved savepoint that is left to abort restores what it saved.
         std::optional<frame> sp;
         sp.emplace(tx.sub_transaction());
         sp->upsert("dropped", "v");
      }
      CHECK(tx.snapshot_cursor().count_keys() == 102);
      CHECK_FALSE(tx.get<std::string>("dropped"));
      tx.commit();
   }
   CHECK(ws->get_root(0).count_keys() == 102);
}

// The same for a transaction: the moved-from object is closed and the
// destination commits what was written.
TEST_CASE("review: a transaction that was moved commits once", "[review]")
{
   temp_dir dir("tx-move");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      std::optional<sal_art_ref::radix::radix_tree::transaction> held;
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "1");
         held.emplace(std::move(tx));
      }  // the moved-from transaction dies here
      held->upsert("b", "2");
      held->commit();
   }
   CHECK(ws->get_root(0).count_keys() == 2);
   auto again = ws->start_transaction(0);  // the root's writer lock was released
   again.upsert("c", "3");
   again.commit();
   CHECK(ws->get_root(0).count_keys() == 3);
}

// gofractally/psitri#28: a key stored with its full bytes in an external record
// (its suffix was over 255 bytes when it arrived) kept only a 123-byte fragment
// in the page. Once splits had tightened the page's common prefix, an update
// replaced the external record by one without the key.
TEST_CASE("review: updating a long key keeps the key", "[review]")
{
   temp_dir dir("long-key-update");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   model_t  model;

   const std::string prefix(200, 'p');
   const std::string first = prefix + 'm' + std::string(199, 'q');  // 400 bytes
   put(*ws, model, first, "first-value");
   for (int i = 0; i < 300; ++i)
   {
      std::string k = prefix;
      k += char('a' + i / 26 % 26);
      k += char('a' + i % 26);
      k += char('0' + i / 676);
      put(*ws, model, k, "0123456789");
   }
   check_against(*ws, model);
   put(*ws, model, first, "second-value");
   check_against(*ws, model);
   // Every storage kind, updated to a small value, a value too large to stay in
   // the page, and back.
   const std::string big(600, 'B');
   for (const auto& [k, v] : model_t(model))
   {
      put(*ws, model, k, big);
      put(*ws, model, k, "s");
   }
   check_against(*ws, model);
   for (int len : {130, 254, 255, 256, 257, 400, 1024})
   {
      std::string k = len < int(prefix.size())
                          ? std::string(size_t(len), 'k')
                          : prefix + std::string(size_t(len) - prefix.size(), char('A' + len % 26));
      put(*ws, model, k, "v1");
      put(*ws, model, k, big);
      put(*ws, model, k, "v3");
   }
   check_against(*ws, model);
}

// gofractally/psitri#27: a page keeps its keys' common prefix and its divider in
// its own 2 KB, and nothing checked that they and the records fit. Keys sharing
// about 700 bytes overflowed the page (the prefix once, the divider again); a
// key outside a prefix of more than 255 bytes could never be added to the page
// it routed to ("split invariant violated", forever).
TEST_CASE("review: keys that share a long prefix", "[review]")
{
   // Shared prefix lengths around every boundary: the one-byte suffix length
   // (255/256), where prefix plus divider passed a page (700, 800), and the key
   // limit (keys of exactly 1,024 bytes).
   for (int shared : {100, 255, 256, 300, 700, 800, 1020})
   {
      INFO("shared prefix of " << shared << " bytes");
      temp_dir dir("long-prefix");
      auto     db = engine::database::open(dir.path);
      auto     ws = db->start_write_session();
      model_t  model;
      std::mt19937_64 rng{uint64_t(shared)};

      const std::string prefix(size_t(shared), 'z');
      auto key_of = [&](int i) {
         char suf[8];
         std::snprintf(suf, sizeof suf, "%04d", i);
         return prefix + suf;
      };
      const std::string values[] = {"v", std::string(100, 'm'), std::string(200, 'L'),
                                    std::string(5000, 'X')};
      // inserts in an order that splits pages in the middle and at both ends
      std::vector<int> order(400);
      for (int i = 0; i < 400; ++i)
         order[size_t(i)] = i;
      std::shuffle(order.begin() + 100, order.end(), rng);
      for (int i : order)
         put(*ws, model, key_of(i), values[size_t(i) % 4]);
      check_against(*ws, model);

      // keys outside the prefix: below it, inside it but shorter, and above it
      for (const std::string& k :
           {std::string("a"), std::string("zzzzb"), std::string("~"), prefix.substr(0, 50),
            prefix.substr(0, size_t(shared) - 1) + "y", prefix.substr(0, size_t(shared) - 1) + "{",
            prefix, std::string(1024, 'z')})
         put(*ws, model, k, "outside");
      check_against(*ws, model);

      // updates across every size class, removes, and re-inserts
      for (int round = 0; round < 3; ++round)
      {
         for (int i = 0; i < 400; ++i)
         {
            const auto k = key_of(int(rng() % 400));
            switch (rng() % 3)
            {
               case 0: put(*ws, model, k, values[rng() % 4]); break;
               case 1: del(*ws, model, k); break;
               default: put(*ws, model, k, values[(rng() % 4)] + "u"); break;
            }
         }
         check_against(*ws, model);
      }
      // a second family under a different long prefix, interleaved with the first
      const std::string other(size_t(shared), 'y');
      for (int i = 0; i < 200; ++i)
      {
         put(*ws, model, other + std::to_string(1000 + i), values[size_t(i) % 4]);
         if (i % 3 == 0)
            del(*ws, model, key_of(i));
      }
      check_against(*ws, model);
      // drain it: every page empties and is dropped or reused
      for (const auto& [k, v] : model_t(model))
         del(*ws, model, k);
      check_against(*ws, model);
      put(*ws, model, "after", "empty");
      put(*ws, model, key_of(7), "again");
      check_against(*ws, model);
   }
}

// The case that could not make progress: every key shares 300 bytes, so the
// pages' common prefix is over 255 bytes, and a key outside it is added.
TEST_CASE("review: a key outside a page's long common prefix gets in", "[review]")
{
   temp_dir dir("outside-prefix");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   model_t  model;
   const std::string prefix(300, 'z');
   for (int i = 0; i < 400; ++i)
   {
      char suf[8];
      std::snprintf(suf, sizeof suf, "%04d", i);
      put(*ws, model, prefix + suf, "v");
   }
   for (const char* k : {"a", "zzzzb", "~"})
      put(*ws, model, k, "small");
   check_against(*ws, model);
}

// gofractally/psitri#29: a cursor kept its descent path in 256 frames and wrote
// past them. The routing index is as deep as its dividers nest, and keys that
// are each a prefix of the next (a, aa, aaa, ...) nest once per byte, up to the
// key limit.
TEST_CASE("review: a cursor walks a tree nested as deep as the longest key", "[review]")
{
   temp_dir dir("deep-cursor");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   model_t  model;

   // Groups a^n b ?? for every n the key limit allows: 60 keys per group so each
   // group's page splits and leaves a divider at its depth.
   constexpr int deepest = 1024 - 3;
   for (int n = 1; n <= deepest; ++n)
   {
      auto tx = ws->start_transaction(0);
      for (int j = 0; j < 60; ++j)
      {
         std::string k(size_t(n), 'a');
         k += 'b';
         k += char('A' + j / 10);
         k += char('0' + j % 10);
         tx.upsert(k, "v");
         model[k] = "v";
      }
      tx.commit();
   }
   REQUIRE(model.rbegin()->first.size() <= 1024);
   check_against(*ws, model);

   auto tree = ws->get_root(0);
   auto c    = tree.snapshot_cursor();
   for (int n : {1, 200, 255, 256, 257, 280, 400, 1000, deepest})
   {
      std::string k(size_t(n), 'a');
      k += "bA0";
      INFO("depth " << n);
      REQUIRE(c.lower_bound(k));
      REQUIRE(std::string(c.key()) == k);
      auto it = model.find(k);
      REQUIRE(c.next());
      REQUIRE(std::string(c.key()) == std::next(it)->first);
      REQUIRE(c.prev());
      REQUIRE(c.prev() == (it != model.begin()));
      if (it != model.begin())
         REQUIRE(std::string(c.key()) == std::prev(it)->first);
   }
}

// gofractally/psitri#30 (1): commit published the primary root, then flushed and
// published the other roots one by one, so a later root's flush that threw left
// the earlier ones published by a commit that failed.
TEST_CASE("review: a commit that throws publishes no root", "[review]")
{
   temp_dir dir("commit-throw");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("kept", "old");
      tx.commit();
   }
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("kept", "new");
      tx.upsert("added", "x");
      // An opened child that cannot be written back: its key is over the limit, which
      // is found when root 1 is flushed at commit.
      tx.open_root(1).create_subtree(std::string(2000, 'k')).upsert("a", "b");
      CHECK_THROWS(tx.commit());
   }
   CHECK(ws->get_root(0).get<std::string>("kept") == std::optional<std::string>("old"));
   CHECK_FALSE(ws->get_root(0).get<std::string>("added"));
   CHECK(ws->get_root(1).count_keys() == 0);
   // both roots are free again and a later transaction commits
   auto tx = ws->start_transaction(0);
   tx.upsert("added", "y");
   tx.open_root(1).upsert("r1", "z");
   tx.commit();
   CHECK(ws->get_root(0).get<std::string>("added") == std::optional<std::string>("y"));
   CHECK(ws->get_root(1).count_keys() == 1);
}

// gofractally/psitri#30 (2): a savepoint saved the transaction's working root and
// how many subtrees were open, but not the working roots of those open subtrees,
// so aborting it kept what was written through a subtree opened before it.
TEST_CASE("review: aborting a savepoint undoes writes through subtrees opened before it",
          "[review]")
{
   temp_dir dir("frame-subtree");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      tx.create_subtree("sub").upsert("before", "1");
      tx.commit();
   }
   auto  tx  = ws->start_transaction(0);
   auto& sub = tx.open_subtree("sub");
   sub.upsert("in-tx", "2");
   auto& nested = sub.create_subtree("nested");
   nested.upsert("n1", "3");
   {
      auto sp = tx.sub_transaction();
      sub.upsert("in-frame", "4");                        // through a handle opened before
      nested.upsert("n2", "5");                           // and one opened below it
      sub.create_subtree("made-in-frame").upsert("x", "6");
      sp.upsert("top-in-frame", "7");
      sp.abort();
   }
   CHECK(sub.get<std::string>("in-tx") == std::optional<std::string>("2"));
   CHECK_FALSE(sub.get<std::string>("in-frame"));
   CHECK(nested.get<std::string>("n1") == std::optional<std::string>("3"));
   CHECK_FALSE(nested.get<std::string>("n2"));
   CHECK_FALSE(tx.get<std::string>("top-in-frame"));
   {
      auto sp = tx.sub_transaction();
      sub.upsert("kept-by-commit", "8");
      sp.commit();
   }
   tx.commit();
   auto committed = ws->get_root(0).get_subtree("sub");
   CHECK(committed.count_keys() == 4);  // before, in-tx, nested, kept-by-commit
   CHECK(committed.get<std::string>("kept-by-commit") == std::optional<std::string>("8"));
   CHECK_FALSE(committed.get<std::string>("in-frame"));
   CHECK(committed.get_subtree("nested").count_keys() == 1);
}

// gofractally/psitri#30 (3): cursor::subtree_shared() bound the shared tree to the
// writer context of the session the cursor came from. A shared tree is meant to
// outlive sessions; this one read the session's freed context when it was copied
// or dropped. (Shown by AddressSanitizer; without it the freed memory usually
// still holds the pointer.)
TEST_CASE("review: a shared subtree outlives the write session it was taken through", "[review]")
{
   temp_dir                        dir("shared-subtree");
   auto                            db = engine::database::open(dir.path);
   sal_art_ref::radix::shared_tree held;
   {
      auto ws = db->start_write_session();
      {
         auto tx = ws->start_transaction(0);
         tx.create_subtree("sub").upsert("inside", "v");
         tx.commit();
      }
      auto root = ws->get_root(0);
      auto c    = root.snapshot_cursor();
      REQUIRE(c.lower_bound("sub"));
      REQUIRE(c.is_subtree());
      held = c.subtree_shared();
   }  // the write session and its writer context are gone
   sal_art_ref::radix::shared_tree copy = held;
   auto                            rs   = db->start_read_session();
   auto                            c    = rs->create_cursor(copy);
   REQUIRE(c.seek_begin());
   CHECK(std::string(c.key()) == "inside");
   held.reset();
   copy.reset();
}

// gofractally/psitri#30 (4): the subtree handed to upsert_subtree was taken out of
// its owning pointer before the insert that could throw, so a refused insert (here:
// a key over the limit) left the whole subtree allocated with nothing referring to it.
TEST_CASE("review: a subtree whose attach is refused is not leaked", "[review]")
{
   temp_dir dir("attach-leak");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("base", "v");
      tx.commit();
   }
   auto settle = [&] { REQUIRE(db->wait_for_compactor(std::chrono::milliseconds(30000))); };
   settle();
   const uint64_t baseline = ws->get_total_allocated_objects();

   const std::string too_long(2000, 'k');
   {
      // a detached tree attached under a key that is refused
      auto edit = ws->start_write_transaction(ws->create_temporary_tree());
      for (int i = 0; i < 500; ++i)
         edit.upsert("key" + std::to_string(i), std::string(300, 'v'));
      auto tx = ws->start_transaction(0);
      CHECK_THROWS(tx.upsert_subtree(too_long, edit.get_tree()));
      tx.abort();
   }
   settle();
   CHECK(ws->get_total_allocated_objects() == baseline);
   {
      // an opened child that cannot be written back at commit
      auto  tx    = ws->start_transaction(0);
      auto& child = tx.create_subtree(too_long);
      for (int i = 0; i < 500; ++i)
         child.upsert("key" + std::to_string(i), std::string(300, 'v'));
      CHECK_THROWS(tx.commit());
   }
   settle();
   CHECK(ws->get_total_allocated_objects() == baseline);
   // and a plain write afterwards is a plain write
   auto tx = ws->start_transaction(0);
   tx.upsert("after", "bytes");
   tx.commit();
   CHECK(ws->get_root(0).get<std::string>("after") == std::optional<std::string>("bytes"));
}

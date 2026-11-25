// Typed values in the Large Engine (gofractally/psitri#18).
//
// A value is either the caller's bytes or a typed value: a type number and a
// small token whose type registered retain/release hooks. The kind is the
// record's own metadata. These tests hold the engine to two things:
//
//   1. Plain bytes are never inspected: whatever a value's first bytes are,
//      storing, copying and removing it runs no hook.
//   2. A typed value's hooks are balanced: storing hands the engine one
//      reference, every physical copy of the stored bytes takes one more, and
//      every copy that dies gives one back — through overwrite, remove, abort,
//      copy-on-write of a shared page, page splits, external records, the
//      background release thread, and across close and recovery.
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

#include <sal_art_ref/engine.hpp>

namespace
{
   using engine = sal_art_ref::radix::engine;
   using sal_art_ref::invalid_type;
   using sal_art_ref::kRawType;

   constexpr uint16_t counted_type = 7;
   constexpr uint16_t other_type   = 0x4142;  // "BA": also two printable bytes
   constexpr size_t   id_count     = 1 << 16;

   // The token: an id. `owners[id]` is how many references the engine holds on it:
   // +1 when the test stores it (custody moves in), +1 per retain, -1 per release.
   struct token
   {
      uint64_t id;
   };
   std::array<std::atomic<int32_t>, id_count> owners{};
   std::atomic<uint64_t>                      retains{0}, releases{0}, malformed{0};

   void on_retain(void*, const char* p, size_t n)
   {
      token t;
      if (n != sizeof(t))
      {
         ++malformed;
         return;
      }
      std::memcpy(&t, p, sizeof(t));
      if (t.id >= id_count)
      {
         ++malformed;
         return;
      }
      ++owners[t.id];
      ++retains;
   }
   void on_release(void*, const char* p, size_t n)
   {
      token t;
      if (n != sizeof(t))
      {
         ++malformed;
         return;
      }
      std::memcpy(&t, p, sizeof(t));
      if (t.id >= id_count)
      {
         ++malformed;
         return;
      }
      --owners[t.id];
      ++releases;
   }
   void reset_counts()
   {
      sal_art_ref::g_value_types[counted_type] = {&on_retain, &on_release};
      sal_art_ref::g_value_types[other_type]   = {&on_retain, &on_release};
      for (auto& o : owners)
         o = 0;
      retains = releases = malformed = 0;
   }
   std::string_view bytes_of(const token& t)
   {
      return {reinterpret_cast<const char*>(&t), sizeof(t)};
   }
   // Store a typed value: the engine now holds one reference on `id`.
   template <class Writer>
   void store(Writer& w, std::string_view key, uint64_t id, uint16_t type = counted_type)
   {
      const token t{id};
      w.upsert_typed(key, type, bytes_of(t));
      ++owners[id];
   }
   int32_t owned(uint64_t id) { return owners[id].load(); }
   int64_t total_owned()
   {
      int64_t n = 0;
      for (auto& o : owners)
         n += o.load();
      return n;
   }

   struct temp_dir
   {
      std::filesystem::path path;
      explicit temp_dir(const char* name)
          : path(std::filesystem::temp_directory_path() /
                 (std::string("psitri-typed-") + name + "-" + std::to_string(::getpid())))
      {
         std::filesystem::remove_all(path);
      }
      ~temp_dir() { std::filesystem::remove_all(path); }
   };

   // What a value of any kind reads as.
   struct any
   {
      uint16_t    type = kRawType;
      std::string bytes;
   };
   template <class Reader>
   std::optional<any> read_any(Reader&& r, std::string_view key)
   {
      std::optional<any> out;
      r.get_any(key, [&](uint16_t t, std::string_view v) { out = any{t, std::string(v)}; });
      return out;
   }
   uint64_t id_of(const any& a)
   {
      token t{~0ull};
      if (a.bytes.size() == sizeof(t))
         std::memcpy(&t, a.bytes.data(), sizeof(t));
      return t.id;
   }
   void drain(engine::database& db)
   {
      REQUIRE(db.wait_for_compactor(std::chrono::milliseconds(30000)));
      // The count reaches zero when the queue is taken, a moment before the last
      // destroy has run its hooks.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      REQUIRE(db.wait_for_compactor(std::chrono::milliseconds(30000)));
   }
   std::string numbered(const char* prefix, unsigned i)
   {
      char buf[48];
      std::snprintf(buf, sizeof(buf), "%s%06u", prefix, i);
      return buf;
   }
}  // namespace

// The defect: the first two bytes of a caller's value were read as a type number
// whenever its page was copied or the value died, and the type's hooks ran on the
// rest of the caller's bytes.
TEST_CASE("typed values: plain bytes are never read as a type number", "[typed]")
{
   reset_counts();
   temp_dir dir("raw");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();

   // Values that begin with a registered type number, one of them exactly the
   // size of a typed value's stored form (2 + sizeof(token)).
   const token       fake{3};
   const std::string same_size = std::string("\x07\x00", 2) + std::string(bytes_of(fake));
   const std::string longer    = std::string("\x07\x00", 2) + std::string(40, 'x');
   const std::string other     = "BA" + std::string(bytes_of(fake));
   {
      auto tx = ws->start_transaction(0);
      for (unsigned i = 0; i < 40; ++i)
         tx.upsert(numbered("n/", i), "neighbour");
      tx.upsert("n/000010a", same_size);
      tx.upsert("n/000020a", longer);
      tx.upsert("n/000030a", other);
      tx.commit();
   }
   {
      auto snapshot = ws->get_root(0);  // every page is now shared
      {
         auto tx = ws->start_transaction(0);
         tx.upsert(numbered("n/", 11), "changed");  // copy-on-write of the values' page
         tx.upsert("n/000010a", longer);            // replace one
         tx.remove("n/000020a");                    // remove one
         tx.commit();
      }
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("n/000030a", "x");  // replace in a page this version owns
         tx.remove("n/000010a");
         tx.remove("n/000030a");
         tx.commit();
      }
      // The snapshot still reads the caller's bytes, unchanged.
      REQUIRE(snapshot.get<std::string>("n/000010a") == same_size);
      REQUIRE(snapshot.get<std::string>("n/000020a") == longer);
      REQUIRE(snapshot.get<std::string>("n/000030a") == other);
      auto raw = read_any(snapshot, "n/000010a");
      REQUIRE(raw);
      REQUIRE(raw->type == kRawType);
      REQUIRE(raw->bytes == same_size);
   }
   ws->set_root(0, ws->create_temporary_tree());  // the rest dies on the release thread
   drain(*db);

   CHECK(retains.load() == 0);
   CHECK(releases.load() == 0);
   CHECK(malformed.load() == 0);
   CHECK(total_owned() == 0);
}

TEST_CASE("typed values: stored and read with their type, never as bytes", "[typed]")
{
   reset_counts();
   temp_dir dir("read");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      store(tx, "t/a", 100);
      store(tx, "t/b", 101, other_type);
      tx.upsert("t/raw", "plain");
      tx.upsert("t/empty", "");

      // Reads inside the transaction see the working tree.
      auto a = read_any(tx, "t/a");
      REQUIRE(a);
      CHECK(a->type == counted_type);
      CHECK(id_of(*a) == 100);
      CHECK_THROWS_AS(tx.get<std::string>("t/a"), invalid_type);
      CHECK_THROWS_AS(tx.get("t/a", [](std::string_view) {}), invalid_type);
      tx.commit();
   }
   {
      auto t = ws->get_root(0);
      auto a = read_any(t, "t/a");
      auto b = read_any(t, "t/b");
      auto r = read_any(t, "t/raw");
      auto e = read_any(t, "t/empty");
      REQUIRE((a && b && r && e));
      CHECK((a->type == counted_type && id_of(*a) == 100));
      CHECK((b->type == other_type && id_of(*b) == 101));
      CHECK((r->type == kRawType && r->bytes == "plain"));
      CHECK((e->type == kRawType && e->bytes.empty()));
      CHECK_FALSE(read_any(t, "t/absent"));
      CHECK(t.get<std::string>("t/raw") == "plain");
      CHECK_THROWS_AS(t.get<std::string>("t/b"), invalid_type);

      auto c = t.snapshot_cursor();
      REQUIRE(c.find("t/a"));
      CHECK(c.value_typenum() == counted_type);
      CHECK(c.value_size() == int64_t(sizeof(token)));
      CHECK_THROWS_AS(c.value<std::string>(), invalid_type);
      CHECK_THROWS_AS(c.get_value([](std::string_view) {}), invalid_type);
      any seen;
      CHECK(c.value_any([&](uint16_t ty, std::string_view v) { seen = any{ty, std::string(v)}; }));
      CHECK((seen.type == counted_type && id_of(seen) == 100));
      REQUIRE(c.find("t/raw"));
      CHECK(c.value_typenum() == kRawType);
      CHECK(c.value<std::string>() == "plain");

      // A reader thread's session reads the same.
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      auto ra  = read_any(cur, "t/a");
      REQUIRE(ra);
      CHECK((ra->type == counted_type && id_of(*ra) == 100));
      CHECK_THROWS_AS(cur.get<std::string>("t/a"), invalid_type);
   }
   // Reading takes no reference: each value has its one owner, the tree.
   drain(*db);
   CHECK(owned(100) == 1);
   CHECK(owned(101) == 1);
   CHECK(retains.load() == releases.load());

   {  // what a typed value may be
      auto        tx = ws->start_transaction(0);
      const token t{1};
      CHECK_THROWS_AS(tx.upsert_typed("t/zero", kRawType, bytes_of(t)), std::invalid_argument);
      CHECK_THROWS_AS(tx.upsert_typed("t/last", 0xFFFF, bytes_of(t)), std::invalid_argument);
      const std::string big(sal_art_ref::max_typed_token_size + 1, 'x');
      CHECK_THROWS_AS(tx.upsert_typed("t/big", counted_type, big), std::invalid_argument);
      tx.abort();
   }
   ws->set_root(0, ws->create_temporary_tree());
   drain(*db);
   CHECK(total_owned() == 0);
   CHECK(malformed.load() == 0);
}

TEST_CASE("typed values: overwrite, remove and abort give back what they displace", "[typed]")
{
   reset_counts();
   temp_dir dir("life");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      store(tx, "k/typed", 1);
      tx.upsert("k/bytes", "bytes");
      store(tx, "k/gone", 2);
      tx.commit();
   }
   drain(*db);
   CHECK((owned(1) == 1 && owned(2) == 1));
   {
      auto tx = ws->start_transaction(0);
      store(tx, "k/typed", 3);       // typed by typed
      store(tx, "k/bytes", 4);       // bytes by typed
      tx.remove("k/gone");           // remove
      tx.commit();
   }
   drain(*db);
   CHECK(owned(1) == 0);
   CHECK(owned(2) == 0);
   CHECK(owned(3) == 1);
   CHECK(owned(4) == 1);
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("k/typed", "now bytes");  // typed by bytes
      tx.commit();
   }
   drain(*db);
   CHECK(owned(3) == 0);
   CHECK(ws->get_root(0).get<std::string>("k/typed") == "now bytes");
   {
      // An aborted transaction's stores die with its working tree.
      auto tx = ws->start_transaction(0);
      store(tx, "k/aborted", 5);
      store(tx, "k/bytes", 6);  // replaces 4 in the working tree only
      tx.remove("k/typed");
      tx.abort();
   }
   drain(*db);
   CHECK(owned(5) == 0);
   CHECK(owned(6) == 0);
   CHECK(owned(4) == 1);
   {
      auto t = ws->get_root(0);
      auto v = read_any(t, "k/bytes");
      REQUIRE(v);
      CHECK((v->type == counted_type && id_of(*v) == 4));
      CHECK_FALSE(read_any(t, "k/aborted"));
   }
   ws->set_root(0, ws->create_temporary_tree());
   drain(*db);
   CHECK(total_owned() == 0);
   CHECK(malformed.load() == 0);
}

TEST_CASE("typed values: a shared page's copy owns its typed values too", "[typed]")
{
   reset_counts();
   temp_dir dir("cow");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      for (unsigned i = 0; i < 20; ++i)
         store(tx, numbered("c/", i), 1000 + i);
      tx.upsert("c/plain", "plain");
      tx.commit();
   }
   drain(*db);
   {
      auto snapshot = ws->get_root(0);
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("c/plain", "changed");      // the page is copied: every typed value twice
         store(tx, numbered("c/", 3), 2003);   // replaced in the copy
         tx.remove(numbered("c/", 4));         // removed from the copy
         tx.commit();
      }
      drain(*db);
      for (unsigned i = 0; i < 20; ++i)
      {
         INFO("value " << i);
         // 3 and 4 remain only in the snapshot's page; the rest are in both.
         CHECK(owned(1000 + i) == ((i == 3 || i == 4) ? 1 : 2));
      }
      CHECK(owned(2003) == 1);
      auto old = read_any(snapshot, numbered("c/", 3));
      REQUIRE(old);
      CHECK(id_of(*old) == 1003);
      CHECK(read_any(snapshot, numbered("c/", 4)));
      {
         // Another change: the published page is copied again (a transaction continues
         // from a root the root table also holds), and the copy it replaces dies.
         auto tx = ws->start_transaction(0);
         tx.upsert("c/plain", "again");
         tx.commit();
      }
      drain(*db);
      for (unsigned i = 0; i < 20; ++i)
      {
         INFO("value " << i);
         CHECK(owned(1000 + i) == ((i == 3 || i == 4) ? 1 : 2));
      }
   }
   drain(*db);  // the snapshot's page is destroyed on the release thread
   for (unsigned i = 0; i < 20; ++i)
   {
      INFO("value " << i);
      CHECK(owned(1000 + i) == ((i == 3 || i == 4) ? 0 : 1));
   }
   CHECK(owned(2003) == 1);
   ws->set_root(0, ws->create_temporary_tree());
   drain(*db);
   CHECK(total_owned() == 0);
   CHECK(malformed.load() == 0);
}

TEST_CASE("typed values: page splits, external records and removal stay balanced", "[typed]")
{
   reset_counts();
   temp_dir dir("split");
   auto     db = engine::database::open(dir.path);
   auto     ws = db->start_write_session();

   constexpr unsigned n = 6000;
   // A third of the keys are too long for a record in the page: their typed value
   // lives in an external record, alone (500 bytes) or with its whole key (1,000).
   auto key_for = [](unsigned i)
   {
      std::string k = numbered("s/", i);
      if (i % 3 == 1)
         k += std::string(500, char('a' + i % 26));
      else if (i % 9 == 2)
         k += std::string(1000 - k.size(), char('A' + i % 26));
      return k;
   };
   std::map<std::string, uint64_t> model;
   auto check_model = [&]
   {
      drain(*db);
      std::array<int32_t, 2 * n + 16> expected{};
      for (auto& [k, id] : model)
         ++expected[id];
      for (uint64_t id = 0; id < expected.size(); ++id)
         if (owned(id) != expected[id])
         {
            FAIL_CHECK("id " << id << " has " << owned(id) << " owners, expected " << expected[id]);
            return;
         }
      auto     t     = ws->get_root(0);
      auto     c     = t.snapshot_cursor();
      unsigned count = 0;
      for (bool ok = c.lower_bound(std::string_view{}); ok; ok = c.next(), ++count)
      {
         auto found = model.find(std::string(c.key()));
         REQUIRE(found != model.end());
         REQUIRE(c.value_typenum() == counted_type);
         any seen;
         REQUIRE(c.value_any([&](uint16_t ty, std::string_view v) { seen = any{ty, std::string(v)}; }));
         REQUIRE(id_of(seen) == found->second);
      }
      CHECK(count == model.size());
   };

   // Many transactions, a snapshot held across some of them: splits of shared
   // and of unshared pages.
   std::optional<sal_art_ref::radix::radix_tree::tree> held;
   for (unsigned i = 0; i < n;)
   {
      auto tx = ws->start_transaction(0);
      for (unsigned j = 0; j < 97 && i < n; ++j, ++i)
      {
         // not in key order, so pages fill and split everywhere
         const unsigned at = unsigned((uint64_t(i) * 2654435761u) % n);
         const auto     k  = key_for(at);
         if (!model.count(k))
         {
            store(tx, k, at);
            model[k] = at;
         }
      }
      tx.commit();
      if (i % 970 == 0)
         held = ws->get_root(0);
      else if (i % 485 == 0)
         held.reset();
   }
   held.reset();
   check_model();
   CHECK(model.size() > n / 2);

   {  // overwrite every other value, in place where the page is unshared
      auto tx = ws->start_transaction(0);
      unsigned i = 0;
      for (auto& [k, id] : model)
         if (i++ % 2 == 0)
         {
            const uint64_t fresh = n + (id % n);
            store(tx, k, fresh);
            id = fresh;
         }
      tx.commit();
   }
   check_model();

   {  // remove most of it under a snapshot, then the rest: pages empty and are dropped
      auto     snapshot = ws->get_root(0);
      auto     tx       = ws->start_transaction(0);
      unsigned i        = 0;
      for (auto it = model.begin(); it != model.end();)
         if (i++ % 5 != 0)
         {
            REQUIRE(tx.remove(it->first) == 0);
            it = model.erase(it);
         }
         else
            ++it;
      tx.commit();
   }
   check_model();
   {
      auto tx = ws->start_transaction(0);
      for (auto& [k, id] : model)
         REQUIRE(tx.remove(k) == 0);
      model.clear();
      tx.commit();
   }
   check_model();
   CHECK(total_owned() == 0);
   CHECK(malformed.load() == 0);
}

TEST_CASE("typed values: kept across a clean close and across recovery", "[typed]")
{
   for (const bool clean : {true, false})
   {
      INFO((clean ? "clean close" : "killed without closing"));
      reset_counts();
      temp_dir dir(clean ? "reopen" : "recover");
      auto     write = [&]
      {
         auto db = engine::database::open(dir.path);
         auto ws = db->start_write_session();
         auto tx = ws->start_transaction(0);
         for (unsigned i = 0; i < 300; ++i)
            store(tx, numbered("r/", i) + (i % 4 == 0 ? std::string(400, 'k') : std::string()),
                  i);
         tx.upsert("r/raw", std::string("\x07\x00", 2) + "raw bytes");
         tx.commit();
      };
      if (clean)
      {
         write();
      }
      else
      {
         const pid_t child = ::fork();
         REQUIRE(child >= 0);
         if (child == 0)
         {
            write();        // the write session and database are never closed
            ::_exit(0);
         }
         int status = 0;
         REQUIRE(::waitpid(child, &status, 0) == child);
         REQUIRE((WIFEXITED(status) && WEXITSTATUS(status) == 0));
         for (unsigned i = 0; i < 300; ++i)  // the child's stores, counted here
            ++owners[i];
      }
      const auto retained_before = retains.load();
      const auto released_before = releases.load();
      {
         auto db = engine::database::open(dir.path, sal::open_mode::open_existing);
         auto ws = db->start_write_session();
         // Opening (and recovering) copies and frees no value.
         CHECK(retains.load() == retained_before);
         CHECK(releases.load() == released_before);
         auto t = ws->get_root(0);
         for (unsigned i = 0; i < 300; ++i)
         {
            auto v = read_any(
                t, numbered("r/", i) + (i % 4 == 0 ? std::string(400, 'k') : std::string()));
            REQUIRE(v);
            REQUIRE(v->type == counted_type);
            REQUIRE(id_of(*v) == i);
         }
         CHECK(t.get<std::string>("r/raw") == std::string("\x07\x00", 2) + "raw bytes");
         t = {};
         {
            auto tx = ws->start_transaction(0);
            for (unsigned i = 0; i < 300; i += 2)
               REQUIRE(tx.remove(numbered("r/", i) +
                                 (i % 4 == 0 ? std::string(400, 'k') : std::string())) == 0);
            tx.commit();
         }
         drain(*db);
         for (unsigned i = 0; i < 300; ++i)
         {
            INFO("value " << i);
            CHECK(owned(i) == (i % 2 == 0 ? 0 : 1));
         }
         ws->set_root(0, ws->create_temporary_tree());
         drain(*db);
      }
      CHECK(total_owned() == 0);
      CHECK(malformed.load() == 0);
   }
}

#include <psitri/api.hpp>
#include <psitri/experimental/tiered.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

static_assert(psitri::engine<psitri::small_engine>);
static_assert(psitri::engine<psitri::big_engine>);
static_assert(psitri::range_engine<psitri::small_engine>);
static_assert(psitri::range_engine<psitri::big_engine>);
static_assert(!psitri::range_engine<int>);
static_assert(!psitri::engine<psitri::experimental::write_engine>);
static_assert(!psitri::engine<int>);
static_assert(adb::multi_root_transaction<psitri::small_engine::transaction>);
static_assert(adb::multi_root_transaction<psitri::big_engine::transaction>);

// These must fail even though the other engine roles still conform.
struct mismatched_roles : psitri::small_engine
{
   using cursor = psitri::big_engine::cursor;
};
struct missing_open : psitri::small_engine
{
   struct database
   {
      std::shared_ptr<write_session> start_write_session();
      std::shared_ptr<read_session> start_read_session();
   };
};
struct no_abort : psitri::small_engine::transaction
{
   void abort() = delete;
};
struct engine_without_abort : psitri::small_engine
{
   using transaction = no_abort;
   struct write_session : psitri::small_engine::write_session
   {
      transaction start_transaction(uint32_t);
   };
   struct database : psitri::small_engine::database
   {
      static std::shared_ptr<database> open(const std::filesystem::path&, psitri::open_mode,
                                           const config&);
      std::shared_ptr<write_session> start_write_session();
   };
};
struct config_without_sync : psitri::small_engine
{
   struct config
   {
      operator psitri::small_engine::config() const { return {}; }
   };
};
static_assert(!psitri::engine<mismatched_roles>);
static_assert(!psitri::engine<missing_open>);
static_assert(adb::kv_transaction<no_abort>);  // original experimental contract
static_assert(adb::kv_engine<engine_without_abort>);
static_assert(!psitri::engine<engine_without_abort>);  // public profile requires explicit abort
static_assert(!psitri::engine<config_without_sync>);

namespace
{
   struct temporary_directory
   {
      std::filesystem::path path;
      temporary_directory()
      {
         std::string pattern = (std::filesystem::temp_directory_path() / "psitri-api-XXXXXX").string();
         auto* result = ::mkdtemp(pattern.data());
         if (!result) throw std::runtime_error("mkdtemp failed");
         path = result;
      }
      ~temporary_directory()
      {
         std::error_code ignored;
         std::filesystem::remove_all(path, ignored);
      }
   };

   template <psitri::engine E>
   typename E::config config()
   {
      typename E::config cfg;
      cfg.sync_mode = psitri::sync_mode::fsync;
      if constexpr (std::same_as<E, psitri::small_engine>) cfg.max_segments = 16;
      else cfg.max_pinned_cache_size_mb = 128;
      return cfg;
   }
}

TEMPLATE_TEST_CASE("Public API: bytes, transactions, snapshots, bounds, reopen", "[api]",
                   psitri::small_engine, psitri::big_engine)
{
   using E = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto path = temp.path / "db";
   auto cfg = config<E>();
   REQUIRE_THROWS(DB::open(path, psitri::open_mode::open_existing, cfg));
   const std::string binary_key("b\0\xff", 3);
   const std::string binary_value("\0\x80\xff", 3);
   const std::vector<std::string> keys{"", "a", binary_key, "c", "z"};
   {
      auto db = DB::open(path, psitri::open_mode::create_only, cfg);
      REQUIRE_THROWS(DB::open(path, psitri::open_mode::create_only, cfg));
      auto writer = db->start_write_session();
      auto empty = writer->get_root(0);
      {
         auto tx = writer->start_transaction(0);
         tx.upsert("", "");
         tx.upsert("a", "first");
         tx.upsert(binary_key, binary_value);
         tx.upsert("c", "third");
         tx.upsert("z", std::string(65535, 'x'));
         REQUIRE(tx.template get<std::string>("z") == std::string(65535, 'x'));
         REQUIRE(tx.template get<std::string>(binary_key) == binary_value);
         REQUIRE(tx.remove("absent") == -1);
         int calls = 0;
         REQUIRE(tx.get("", [&](std::string_view value) { ++calls; REQUIRE(value.empty()); }));
         REQUIRE(calls == 1);
         REQUIRE_FALSE(tx.get("absent", [&](std::string_view) { ++calls; }));
         REQUIRE(calls == 1);
         tx.commit();
      }
      REQUIRE_FALSE(empty.snapshot_cursor().seek_begin());
      auto snapshot = writer->get_root(0);
      auto copy = snapshot;
      {
         auto tx = writer->start_transaction(0);
         tx.upsert("a", "aborted");
         REQUIRE(tx.remove("c") == 0);
         // implicit rollback
      }
      {
         auto tx = writer->start_transaction(0);
         tx.upsert("a", "also aborted");
         tx.abort();
      }
      auto reader = db->start_read_session();
      auto cur = reader->create_cursor(0);
      REQUIRE(cur.template get<std::string>("a") == "first");
      REQUIRE(cur.template get<std::string>("c") == "third");
      std::vector<std::string> seen;
      for (bool more = cur.seek_begin(); more; more = cur.next()) seen.emplace_back(cur.key());
      REQUIRE(seen == keys);
      REQUIRE_FALSE(cur.template value<std::string>());
      int calls = 0;
      REQUIRE_FALSE(cur.get_value([&](std::string_view) { ++calls; }));
      REQUIRE(calls == 0);
      seen.clear();
      for (bool more = cur.seek_last(); more; more = cur.prev()) seen.emplace_back(cur.key());
      REQUIRE(seen == std::vector<std::string>(keys.rbegin(), keys.rend()));
      REQUIRE(cur.is_rend());
      REQUIRE_FALSE(cur.template value<std::string>());
      REQUIRE_FALSE(cur.get_value([&](std::string_view) { ++calls; }));
      REQUIRE(calls == 0);
      REQUIRE(cur.lower_bound("b"));
      REQUIRE(cur.key() == binary_key);
      REQUIRE(cur.upper_bound("b"));
      REQUIRE(cur.key() == "a");
      REQUIRE(cur.upper_bound(binary_key));
      REQUIRE(cur.key() == binary_key);
      REQUIRE(cur.get_value([&](std::string_view v) { ++calls; REQUIRE(v == binary_value); }));
      REQUIRE(calls == 1);
      {
         auto tx = writer->start_transaction(0);
         tx.upsert("a", "second");
         tx.commit();
      }
      REQUIRE(cur.template get<std::string>("a") == "first");
      REQUIRE(copy.snapshot_cursor().template get<std::string>("a") == "first");
      REQUIRE(writer->get_root(0).snapshot_cursor().template get<std::string>("a") == "second");

      // A session and its cursor are created, used, and destroyed on the reader
      // thread. No session or snapshot handle crosses threads.
      std::string read_result;
      std::thread reader_thread([&] {
         auto rs = db->start_read_session();
         auto c = rs->create_cursor(0);
         read_result = c.template get<std::string>("a").value_or("missing");
      });
      reader_thread.join();
      REQUIRE(read_result == "second");
   }
   {
      auto db = DB::open(path, psitri::open_mode::open_existing, cfg);
      auto rs = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE(cur.template get<std::string>("a") == "second");
      REQUIRE(cur.template get<std::string>(binary_key) == binary_value);
      REQUIRE(cur.template get<std::string>("z") == std::string(65535, 'x'));
   }
   if constexpr (std::same_as<E, psitri::big_engine>)
   {
      auto db = DB::open(path, psitri::open_mode::read_only, cfg);
      auto rs = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE(cur.template get<std::string>("a") == "second");
      REQUIRE(db->underlying_allocator().get_runtime_config().read_only);
   }
   else
   {
      REQUIRE_THROWS(DB::open(path, psitri::open_mode::read_only, cfg));
   }
}

TEMPLATE_TEST_CASE("Public API extension: root exchange and multi-root rollback", "[api]",
                   psitri::small_engine, psitri::big_engine)
{
   using E = TestType;
   temporary_directory temp;
   auto db = E::database::open(temp.path, psitri::open_mode::create_only, config<E>());
   auto ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("key", "zero");
      tx.open_root(1).upsert("key", "one");
      tx.commit();
   }
   auto before = ws->get_root(0);
   {
      auto tx = ws->start_transaction(0);
      auto displaced = tx.primary().set_root({});
      tx.open_root(1).set_root(displaced);
      // Abort both roots on scope exit.
   }
   REQUIRE(ws->get_root(0).snapshot_cursor().template get<std::string>("key") == "zero");
   REQUIRE(ws->get_root(1).snapshot_cursor().template get<std::string>("key") == "one");
   {
      auto tx = ws->start_transaction(0);
      auto displaced = tx.primary().set_root({});
      tx.open_root(1).set_root(displaced);
      tx.commit();
   }
   REQUIRE_FALSE(ws->get_root(0).snapshot_cursor().seek_begin());
   REQUIRE(ws->get_root(1).snapshot_cursor().template get<std::string>("key") == "zero");
   REQUIRE(before.snapshot_cursor().template get<std::string>("key") == "zero");
}

TEMPLATE_TEST_CASE("Public API: value ranges preserve full reads and snapshots", "[api][ranges]",
                   psitri::small_engine, psitri::big_engine)
{
   using E = TestType;
   temporary_directory temp;
   auto db = E::database::open(temp.path, psitri::open_mode::create_only, config<E>());
   auto ws = db->start_write_session();
   std::string payload(65535, '\0');
   for (size_t i = 0; i < payload.size(); ++i) payload[i] = char(i % 256);
   struct range { uint64_t offset, length; };
   const range cases[] = {{0, psitri::read_all}, {0, 0}, {1, 255}, {127, 2},
                          {1024, 4096}, {65534, 9}, {65535, 1}, {65536, 1},
                          {psitri::read_all, psitri::read_all}, {7, psitri::read_all}};
   auto check_reader = [&](auto& reader) {
      REQUIRE(reader.template get<std::string>("bytes") == payload);
      // Literal zero must select the owning offset overload, not a callback
      // or Big's legacy get(key, std::string*) overload.
      REQUIRE(reader.get("bytes", 0) == payload);
      REQUIRE(reader.template get<std::string>("bytes", 7) == payload.substr(7));
      for (auto [offset, length] : cases)
      {
         const auto start = std::min<uint64_t>(offset, payload.size());
         const auto expected = payload.substr(start, std::min<uint64_t>(length, payload.size() - start));
         REQUIRE(reader.template get<std::string>("bytes", offset, length) == expected);
         unsigned calls = 0;
         REQUIRE(reader.get("bytes", [&](std::string_view value) {
            ++calls;
            REQUIRE(value == expected);
         }, offset, length));
         REQUIRE(calls == 1);
      }
      unsigned calls = 0;
      REQUIRE(reader.get("bytes", [&](std::string_view value) {
         ++calls;
         REQUIRE(value == std::string_view(payload).substr(7));
      }, 7));
      REQUIRE(calls == 1);
      REQUIRE(reader.template get<std::string>("empty", 0, psitri::read_all) == "");
      REQUIRE_FALSE(reader.template get<std::string>("missing", 0, 0));
      REQUIRE_FALSE(reader.get("missing", [&](std::string_view) { ++calls; }, 0, 0));
      REQUIRE(calls == 1);
      REQUIRE(reader.get("empty", [&](std::string_view value) {
         ++calls;
         REQUIRE(value.empty());
      }, psitri::read_all, 0));
      REQUIRE(calls == 2);
   };
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("bytes", payload);
      tx.upsert("empty", "");
      check_reader(tx);
      auto&& primary = tx.primary();
      check_reader(primary);
      tx.commit();
   }
   auto snapshot = ws->get_root(0);
   check_reader(snapshot);
   const auto& const_snapshot = snapshot;
   check_reader(const_snapshot);
   auto cur = snapshot.snapshot_cursor();
   check_reader(cur);
   REQUIRE(cur.lower_bound("bytes"));
   REQUIRE(cur.template value<std::string>() == payload);
   REQUIRE(cur.template value<std::string>(7) == payload.substr(7));
   for (auto [offset, length] : cases)
   {
      const auto start = std::min<uint64_t>(offset, payload.size());
      const auto expected = payload.substr(start, std::min<uint64_t>(length, payload.size() - start));
      REQUIRE(cur.template value<std::string>(offset, length) == expected);
      unsigned calls = 0;
      REQUIRE(cur.get_value([&](std::string_view value) {
         ++calls;
         REQUIRE(value == expected);
      }, offset, length));
      REQUIRE(calls == 1);
   }
   // Same protected address proves range callbacks borrow without a payload copy.
   REQUIRE(cur.get_value([&](std::string_view whole) {
      REQUIRE(cur.get_value([&](std::string_view part) {
         REQUIRE(part.data() == whole.data() + 17);
         REQUIRE(part.size() == 31);
      }, 17, 31));
      if constexpr (std::same_as<E, psitri::big_engine>)
      {
         auto held = cur.pinned_value(17, 31);
         REQUIRE(held.data() == whole.data() + 17);
         REQUIRE(held.view() == whole.substr(17, 31));
      }
   }));
   REQUIRE_THROWS_AS(cur.get_value([](std::string_view) { throw std::runtime_error("callback"); },
                                   1, 2), std::runtime_error);
   REQUIRE(cur.template value<std::string>(1, 2) == payload.substr(1, 2));
   if constexpr (std::same_as<E, psitri::big_engine>)
   {
      std::string legacy_output;
      REQUIRE(cur.get("bytes", &legacy_output) == int32_t(payload.size()));
      REQUIRE(legacy_output == payload);
   }
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("bytes", "new");
      tx.commit();
   }
   REQUIRE(cur.get("bytes", 0, 3) == payload.substr(0, 3));
   auto fresh = ws->get_root(0);
   REQUIRE(fresh.get("bytes", 0, 3) == "new");
   REQUIRE(cur.seek_last());
   REQUIRE_FALSE(cur.next());
   REQUIRE_FALSE(cur.template value<std::string>(0));
   REQUIRE_FALSE(cur.get_value([](std::string_view) { FAIL("callback at end"); }, 0));
   REQUIRE(cur.seek_begin());
   REQUIRE_FALSE(cur.prev());
   REQUIRE_FALSE(cur.template value<std::string>(0));
   REQUIRE_FALSE(cur.get_value([](std::string_view) { FAIL("callback at rend"); }, 0));
}

TEST_CASE("Big API forwards allocator configuration", "[api]")
{
   temporary_directory temp;
   auto cfg = config<psitri::big_engine>();
   cfg.enable_read_cache = true;
   cfg.read_cache_window_sec = 1234;
   auto db = psitri::big_engine::database::open(temp.path, psitri::open_mode::create_only, cfg);
   const auto& actual = db->underlying_allocator().get_runtime_config();
   // The allocator may lower this budget to the OS mlock limit.
   REQUIRE(actual.max_pinned_cache_size_mb <= cfg.max_pinned_cache_size_mb);
   REQUIRE(actual.enable_read_cache == cfg.enable_read_cache);
   REQUIRE(actual.read_cache_window_sec == cfg.read_cache_window_sec);
   REQUIRE(actual.sync_mode == cfg.sync_mode);
}

TEST_CASE("Big API widens value storage and retains key limits", "[api]")
{
   temporary_directory temp;
   auto db = psitri::big_engine::database::open(temp.path, psitri::open_mode::create_only,
                                                  config<psitri::big_engine>());
   auto ws = db->start_write_session();
   auto tx = ws->start_transaction(0);
   tx.upsert("key", "original");
   tx.upsert("key", std::string(65536, 'x'));
   REQUIRE(tx.get<std::string>("key") == std::string(65536, 'x'));
   REQUIRE_THROWS_AS(tx.upsert(std::string(1025, 'k'), "value"), std::invalid_argument);
   tx.upsert(std::string(1024, 'k'), "limit");
   tx.commit();
   auto cur = ws->get_root(0).snapshot_cursor();
   REQUIRE(cur.get<std::string>("key") == std::string(65536, 'x'));
   REQUIRE(cur.get<std::string>(std::string(1024, 'k')) == "limit");
}

#include <catch2/catch_test_macros.hpp>
#include <psitri/dwal/read_snapshot.hpp>
#include <psitri/dwal/dwal_transaction.hpp>
#include <chrono>
#include <filesystem>
#include <thread>
#include <future>
#include <latch>
#include <unistd.h>

TEST_CASE("DWAL admission serves a queued snapshot before the next writer", "[dwal][admission-regression]")
{
   psitri::std_lock_policy::shared_mutex_type mutex;
   std::atomic<bool> snapshot_acquired{false};
   mutex.lock_shared();
   auto snapshot = std::async(std::launch::async, [&] {
      std::unique_lock lock(mutex);
      snapshot_acquired.store(true);
   });
   // A pending exclusive capture must close admission to later shared owners.
   // Probe on a separate thread: recursive shared locking is not required.
   auto queued = std::async(std::launch::async, [&] {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < deadline)
      {
         if (!mutex.try_lock_shared()) return true;
         mutex.unlock_shared();
         std::this_thread::yield();
      }
      return false;
   }).get();
   mutex.unlock_shared();
   mutex.lock(); // A new writer may not overtake the already queued capture.
   const bool served_first = snapshot_acquired.load();
   mutex.unlock();
   snapshot.get(); // Release all owners before reporting a test failure.
   CHECK(queued);
   CHECK(served_first);
}

TEST_CASE("DWAL admission allows shared owners and excludes writers", "[dwal][admission-regression]")
{
   psitri::std_lock_policy::shared_mutex_type mutex;
   mutex.lock_shared();
   auto shared = std::async(std::launch::async, [&] {
      const bool read = mutex.try_lock_shared();
      if (read) mutex.unlock_shared();
      const bool write = mutex.try_lock();
      if (write) mutex.unlock();
      return std::pair{read, write};
   }).get();
   mutex.unlock_shared();
   CHECK(shared.first);
   CHECK_FALSE(shared.second);
   mutex.lock();
   auto exclusive = std::async(std::launch::async, [&] {
      const bool read = mutex.try_lock_shared();
      if (read) mutex.unlock_shared();
      const bool write = mutex.try_lock();
      if (write) mutex.unlock();
      return std::pair{read, write};
   }).get();
   mutex.unlock();
   CHECK_FALSE(exclusive.first);
   CHECK_FALSE(exclusive.second);
}

TEST_CASE("DWAL snapshots retain values and range deletes across arena rotation", "[dwal][snapshot-regression]")
{
   auto path = std::filesystem::temp_directory_path() /
               ("dwal_snapshot_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   auto db = psitri::database::create(path / "db");
   {
      psitri::dwal::dwal_config config;
      config.max_rw_entries = 8;
      psitri::dwal::dwal_database dwal(db, path / "wal", config);
      {
         auto tx = dwal.start_write_transaction(0);
         tx.upsert("a", "original");
         tx.upsert("b", "keep");
         tx.upsert("c", "old");
         tx.commit();
      }
      psitri::dwal::read_snapshot before(dwal, 0);
      {
         auto tx = dwal.start_write_transaction(0);
         tx.upsert("a", "aborted");
         tx.abort();
      }
      {
         auto tx = dwal.start_write_transaction(0);
         tx.upsert("a", "changed");
         tx.remove_range("b", "d");
         tx.commit();
      }
      psitri::dwal::read_snapshot deleted(dwal, 0);
      // Force rotation/merge repeatedly while both earlier snapshots stay live.
      for (int i = 0; i < 80; ++i)
      {
         auto tx = dwal.start_write_transaction(0);
         tx.upsert("extra-" + std::to_string(i), "new");
         tx.commit();
      }
      {
         auto tx = dwal.start_write_transaction(0);
         tx.upsert("b", "resurrected");
         tx.commit();
      }
      REQUIRE(dwal.root(0).generation.load() > 0);
      std::string value;
      REQUIRE(before.get("a", &value)); CHECK(value == "original");
      REQUIRE(before.get("b", &value)); CHECK(value == "keep");
      REQUIRE(before.get("c", &value)); CHECK(value == "old");
      REQUIRE(deleted.get("a", &value)); CHECK(value == "changed");
      CHECK_FALSE(deleted.get("b", &value));
      CHECK_FALSE(deleted.get("c", &value));
      CHECK_FALSE(before.get("extra-0", &value));
      auto cursor = before.create_cursor();
      REQUIRE(cursor->seek_begin()); CHECK(cursor->key() == "a");
      REQUIRE(cursor->next()); CHECK(cursor->key() == "b");
      REQUIRE(cursor->next()); CHECK(cursor->key() == "c");
      CHECK_FALSE(cursor->next());
      psitri::dwal::read_snapshot current(dwal, 0);
      REQUIRE(current.get("b", &value)); CHECK(value == "resurrected");
   }
   db.reset();
   std::filesystem::remove_all(path);
}

TEST_CASE("DWAL snapshot capture waits for an in-progress writer", "[dwal][snapshot-regression]")
{
   auto path = std::filesystem::temp_directory_path() /
               ("dwal_snapshot_pending_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   auto db = psitri::database::create(path / "db");
   {
      psitri::dwal::dwal_database dwal(db, path / "wal");
      {
         auto tx = dwal.start_write_transaction(0);
         tx.upsert("a", "old"); tx.upsert("b", "old"); tx.commit();
      }
      auto writer = dwal.start_write_transaction(0);
      writer.upsert("a", "new");
      std::latch started(1);
      auto reader = std::async(std::launch::async, [&] {
         started.count_down();
         psitri::dwal::read_snapshot snapshot(dwal, 0);
         std::pair<std::string, std::string> values;
         snapshot.get("a", &values.first); snapshot.get("b", &values.second);
         return values;
      });
      started.wait();
      bool waited = reader.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
      writer.upsert("b", "new");
      writer.commit();
      auto values = reader.get();
      CHECK(waited);
      CHECK(values.first == "new");
      CHECK(values.second == "new");
   }
   db.reset();
   std::filesystem::remove_all(path);
}

TEST_CASE("DWAL writer cursor reseeks the current root after mutation", "[dwal][snapshot-regression]")
{
   psitri::dwal::btree_layer layer;
   layer.store_data("b", "old");
   psitri::dwal::merge_cursor cursor(&layer, nullptr, std::nullopt);
   REQUIRE(cursor.seek_begin()); CHECK(cursor.key() == "b");
   layer.map.bump_cow_seq();
   layer.store_data("a", "new");
   REQUIRE(cursor.seek_begin()); CHECK(cursor.key() == "a");
   layer.tombstones.add("a", "b");
   layer.erase("a");
   REQUIRE(cursor.seek_begin()); CHECK(cursor.key() == "b");
}

TEST_CASE("DWAL merge cursor changes direction across exhausted layers", "[dwal][snapshot-regression]")
{
   psitri::dwal::btree_layer rw, ro;
   rw.store_data("a", "rw"); rw.store_data("c", "rw");
   ro.store_data("b", "ro");
   psitri::dwal::merge_cursor cursor(&rw, &ro, std::nullopt);
   REQUIRE(cursor.lower_bound("bb")); CHECK(cursor.key() == "c");
   REQUIRE(cursor.prev()); CHECK(cursor.key() == "b");
   REQUIRE(cursor.prev()); CHECK(cursor.key() == "a");
   REQUIRE(cursor.next()); CHECK(cursor.key() == "b");
   REQUIRE(cursor.next()); CHECK(cursor.key() == "c");
   CHECK_FALSE(cursor.next());
   REQUIRE(cursor.prev()); CHECK(cursor.key() == "c");
   REQUIRE(cursor.prev()); CHECK(cursor.key() == "b");
   REQUIRE(cursor.prev()); CHECK(cursor.key() == "a");
   CHECK_FALSE(cursor.prev());
   REQUIRE(cursor.next()); CHECK(cursor.key() == "a");
}

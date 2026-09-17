#include <catch2/catch_test_macros.hpp>
#include <sqlite3.h>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>

namespace
{
   struct connection
   {
      sqlite3* db = nullptr;
      explicit connection(const std::filesystem::path& path)
      { REQUIRE(sqlite3_open(path.c_str(), &db) == SQLITE_OK); }
      ~connection() { sqlite3_close(db); }
      void exec(const char* sql)
      { INFO(sql); REQUIRE(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK); }
      std::string scalar(const char* sql)
      {
         INFO(sql);
         sqlite3_stmt* raw = nullptr;
         REQUIRE(sqlite3_prepare_v2(db, sql, -1, &raw, nullptr) == SQLITE_OK);
         std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(raw, sqlite3_finalize);
         REQUIRE(sqlite3_step(raw) == SQLITE_ROW);
         auto text = sqlite3_column_text(raw, 0);
         REQUIRE(text != nullptr);
         std::string result(reinterpret_cast<const char*>(text));
         REQUIRE(sqlite3_step(raw) == SQLITE_DONE);
         return result;
      }
   };
}

TEST_CASE("SQLite keeps primary keys stable and counts only the selected table", "[sqlite][bank-regression]")
{
   auto path = std::filesystem::temp_directory_path() /
               ("sqlite_bank_regression_" + std::to_string(getpid()));
   std::filesystem::remove_all(path);
   {
      connection writer(path);
      CHECK(writer.scalar("PRAGMA journal_mode=WAL") == "wal");
      writer.exec("PRAGMA synchronous=OFF");
      writer.exec("CREATE TABLE accounts(k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID");
      writer.exec("CREATE TABLE other(k INTEGER PRIMARY KEY, v TEXT)");
      writer.exec("INSERT INTO other VALUES(1,'other'),(2,'other')");
      writer.exec("INSERT INTO accounts VALUES(x'00',x'0100'),(x'01',x'0200')");
      connection reader(path);
      CHECK(reader.scalar("PRAGMA journal_mode") == "wal");
      reader.exec("BEGIN");
      CHECK(reader.scalar("SELECT hex(v) FROM accounts WHERE k=x'00'") == "0100");
      for (int i = 0; i < 4; ++i)
      {
         writer.exec("BEGIN IMMEDIATE");
         writer.exec("INSERT INTO accounts VALUES(x'00',x'0000') ON CONFLICT(k) DO UPDATE SET v=excluded.v");
         CHECK(writer.scalar("SELECT hex(v) FROM accounts WHERE k=x'00'") == "0000");
         writer.exec("INSERT INTO accounts VALUES(x'00',x'0300') ON CONFLICT(k) DO UPDATE SET v=excluded.v");
         CHECK(writer.scalar("SELECT hex(v) FROM accounts WHERE k=x'00'") == "0300");
         writer.exec(i % 2 ? "COMMIT" : "ROLLBACK");
         CHECK(writer.scalar("SELECT count(*) FROM accounts") == "2");
         CHECK(writer.scalar("SELECT count(*) FROM other") == "2");
      }
      CHECK(reader.scalar("SELECT hex(v) FROM accounts WHERE k=x'00'") == "0100");
      reader.exec("COMMIT");
      CHECK(reader.scalar("SELECT hex(v) FROM accounts WHERE k=x'00'") == "0300");
   }
   {
      connection reopened(path);
      CHECK(reopened.scalar("PRAGMA journal_mode") == "wal");
      CHECK(reopened.scalar("SELECT count(*) FROM accounts") == "2");
      CHECK(reopened.scalar("SELECT hex(v) FROM accounts WHERE k=x'00'") == "0300");
   }
   std::filesystem::remove_all(path);
}

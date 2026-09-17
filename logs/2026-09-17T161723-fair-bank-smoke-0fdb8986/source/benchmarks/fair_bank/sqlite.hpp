#pragma once
#include "common.hpp"
#include <sqlite3.h>
namespace fair_bank
{
inline constexpr auto engine_name = "sqlite";
struct Store
{
   struct Connection
   {
      sqlite3* db = nullptr;
      sqlite3_stmt* get_stmt = nullptr;
      sqlite3_stmt* put_stmt = nullptr;
      void check(int rc) { require(rc == SQLITE_OK, db ? sqlite3_errmsg(db) : "SQLite open failed"); }
      void exec(const char* sql) { check(sqlite3_exec(db, sql, nullptr, nullptr, nullptr)); }
      Connection(const Config& c, bool create)
      {
         try
         {
            check(sqlite3_open_v2((c.path / "bank.sqlite").c_str(), &db,
                                 SQLITE_OPEN_READWRITE | (create ? SQLITE_OPEN_CREATE : 0) | SQLITE_OPEN_NOMUTEX, nullptr));
            check(sqlite3_extended_result_codes(db, 1));
            check(sqlite3_busy_timeout(db, 30000));
            if (create) exec("PRAGMA journal_mode=WAL");
            exec(c.sync ? "PRAGMA synchronous=FULL" : "PRAGMA synchronous=OFF");
            exec(("PRAGMA cache_size=-" + std::to_string(c.cache_mib * 1024)).c_str());
            exec("PRAGMA mmap_size=0");
            if (create) exec("CREATE TABLE accounts(k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID");
            check(sqlite3_prepare_v2(db, "SELECT v FROM accounts WHERE k=?", -1, &get_stmt, nullptr));
            check(sqlite3_prepare_v2(db, "INSERT INTO accounts(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", -1, &put_stmt, nullptr));
         }
         catch (...) { close(); throw; }
      }
      ~Connection() { close(); }
      void close() { sqlite3_finalize(get_stmt); sqlite3_finalize(put_stmt); if (db) sqlite3_close_v2(db); }
      std::string get(const std::string& key)
      {
         check(sqlite3_bind_blob(get_stmt, 1, key.data(), int(key.size()), SQLITE_TRANSIENT));
         int rc = sqlite3_step(get_stmt);
         require(rc == SQLITE_ROW, std::string("SQLite get: ") + sqlite3_errmsg(db));
         const auto* bytes = static_cast<const char*>(sqlite3_column_blob(get_stmt, 0));
         std::string value(bytes, size_t(sqlite3_column_bytes(get_stmt, 0)));
         require(sqlite3_step(get_stmt) == SQLITE_DONE, "SQLite get returned multiple rows or error");
         check(sqlite3_reset(get_stmt)); check(sqlite3_clear_bindings(get_stmt));
         return value;
      }
      void put(const std::string& key, const std::string& value)
      {
         check(sqlite3_bind_blob(put_stmt, 1, key.data(), int(key.size()), SQLITE_TRANSIENT));
         check(sqlite3_bind_blob(put_stmt, 2, value.data(), int(value.size()), SQLITE_TRANSIENT));
         require(sqlite3_step(put_stmt) == SQLITE_DONE, std::string("SQLite put: ") + sqlite3_errmsg(db));
         require(sqlite3_changes(db) == 1, "SQLite put did not change exactly one row");
         check(sqlite3_reset(put_stmt)); check(sqlite3_clear_bindings(put_stmt));
      }
      template <class F> void transaction(F&& fn, bool write, bool commit)
      {
         exec(write ? "BEGIN IMMEDIATE" : "BEGIN");
         try { fn(*this); exec(commit ? "COMMIT" : "ROLLBACK"); }
         catch (...) { sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr); throw; }
      }
   };
   Config cfg;
   Connection writer;
   static std::string version() { return std::string(sqlite3_libversion()) + " " + sqlite3_sourceid(); }
   static std::string settings(const Config& c) { return std::string("journal_mode=WAL; synchronous=") + (c.sync ? "FULL" : "OFF") + "; WITHOUT ROWID; mmap_size=0; wal_autocheckpoint=1000; per_connection_cache_mib=" + std::to_string(c.cache_mib); }
   Store(const Config& c, bool reopen) : cfg(c), writer(c, !reopen) {}
   template <class F> void write(F&& fn, bool commit = true) { writer.transaction(std::forward<F>(fn), true, commit); }
   struct Reader
   {
      Connection connection;
      explicit Reader(Store& s) : connection(s.cfg, false) {}
      template <class F> void read(F&& fn) { connection.transaction(std::forward<F>(fn), false, true); }
   };
};
}

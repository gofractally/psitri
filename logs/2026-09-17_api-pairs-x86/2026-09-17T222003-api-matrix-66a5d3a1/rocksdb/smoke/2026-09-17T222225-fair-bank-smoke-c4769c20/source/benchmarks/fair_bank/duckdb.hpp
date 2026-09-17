#pragma once
#include "common.hpp"
#include <duckdb.h>
namespace fair_bank
{
inline constexpr auto engine_name = "duckdb";
struct Store
{
   duckdb_database db = nullptr;
   struct Connection
   {
      duckdb_connection conn = nullptr;
      duckdb_prepared_statement get_stmt = nullptr, put_stmt = nullptr;
      struct Result
      {
         duckdb_result value{};
         ~Result() { duckdb_destroy_result(&value); }
         void check(duckdb_state state)
         {
            if (state != DuckDBSuccess)
            {
               auto error = duckdb_result_error(&value);
               throw std::runtime_error(error ? error : "DuckDB operation failed");
            }
         }
      };
      void exec(const char *sql)
      {
         Result r;
         r.check(duckdb_query(conn, sql, &r.value));
      }
      void prepare(const char *sql, duckdb_prepared_statement *stmt)
      {
         if (duckdb_prepare(conn, sql, stmt) != DuckDBSuccess)
         {
            auto error = duckdb_prepare_error(*stmt);
            throw std::runtime_error(error ? error : "DuckDB prepare failed");
         }
      }
      Connection(duckdb_database db, bool create)
      {
         require(duckdb_connect(db, &conn) == DuckDBSuccess, "DuckDB connect failed");
         try
         {
            if (create)
               exec("CREATE TABLE accounts(k BLOB PRIMARY KEY, v BLOB NOT NULL)");
            prepare("SELECT v FROM accounts WHERE k=?", &get_stmt);
            prepare(
                "INSERT INTO accounts(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                &put_stmt);
         }
         catch (...)
         {
            close();
            throw;
         }
      }
      ~Connection() { close(); }
      void close()
      {
         duckdb_destroy_prepare(&get_stmt);
         duckdb_destroy_prepare(&put_stmt);
         duckdb_disconnect(&conn);
      }
      static void bind(duckdb_prepared_statement stmt, idx_t index, const std::string &value)
      {
         require(duckdb_bind_blob(stmt, index, value.data(), value.size()) == DuckDBSuccess,
                 "DuckDB bind failed");
      }
      std::string get(const std::string &key)
      {
         bind(get_stmt, 1, key);
         Result r;
         r.check(duckdb_execute_prepared(get_stmt, &r.value));
         require(duckdb_row_count(&r.value) == 1 && !duckdb_value_is_null(&r.value, 0, 0),
                 "DuckDB missing account");
         auto blob = duckdb_value_blob(&r.value, 0, 0);
         require(blob.data, "DuckDB blob allocation failed");
         std::string value;
         try
         {
            value.assign(static_cast<char *>(blob.data), blob.size);
         }
         catch (...)
         {
            duckdb_free(blob.data);
            throw;
         }
         duckdb_free(blob.data);
         require(duckdb_clear_bindings(get_stmt) == DuckDBSuccess, "DuckDB clear bindings failed");
         return value;
      }
      void put(const std::string &key, const std::string &value)
      {
         bind(put_stmt, 1, key);
         bind(put_stmt, 2, value);
         Result r;
         r.check(duckdb_execute_prepared(put_stmt, &r.value));
         require(duckdb_rows_changed(&r.value) == 1, "DuckDB put did not change exactly one row");
         require(duckdb_clear_bindings(put_stmt) == DuckDBSuccess, "DuckDB clear bindings failed");
      }
      template <class F> void transaction(F &&fn, bool commit)
      {
         exec("BEGIN TRANSACTION");
         try
         {
            fn(*this);
            exec(commit ? "COMMIT" : "ROLLBACK");
         }
         catch (...)
         {
            try
            {
               exec("ROLLBACK");
            }
            catch (...)
            {
            }
            throw;
         }
      }
   };
   std::unique_ptr<Connection> writer;
   static bool supports(std::string_view level) { return level == FAIR_DUCKDB_SYNC_LEVEL; }
   static std::string version() { return duckdb_library_version(); }
   static std::string settings(const Config &c)
   {
      return "persistent database; synchronous WAL commits; platform_sync=" FAIR_DUCKDB_SYNC_LEVEL
             "; threads=1; memory_limit_mib=" +
             std::to_string(c.cache_mib);
   }
   Store(const Config &c, bool reopen)
   {
      require(supports(c.durability), "DuckDB does not match the requested durability tier");
      duckdb_config config = nullptr;
      require(duckdb_create_config(&config) == DuckDBSuccess, "DuckDB config failed");
      auto mem = std::to_string(c.cache_mib) + "MiB";
      auto a = duckdb_set_config(config, "threads", "1");
      auto b = duckdb_set_config(config, "memory_limit", mem.c_str());
      if (a != DuckDBSuccess || b != DuckDBSuccess)
      {
         duckdb_destroy_config(&config);
         throw std::runtime_error("DuckDB config rejected");
      }
      char *error = nullptr;
      auto state = duckdb_open_ext((c.path / "bank.duckdb").c_str(), &db, config, &error);
      duckdb_destroy_config(&config);
      if (state != DuckDBSuccess)
      {
         std::string message = error ? error : "DuckDB open failed";
         duckdb_free(error);
         throw std::runtime_error(message);
      }
      try
      {
         writer = std::make_unique<Connection>(db, !reopen);
      }
      catch (...)
      {
         duckdb_close(&db);
         throw;
      }
   }
   ~Store()
   {
      writer.reset();
      duckdb_close(&db);
   }
   template <class F> void write(F &&fn, bool commit = true)
   {
      writer->transaction(std::forward<F>(fn), commit);
   }
   struct Reader
   {
      Connection connection;
      explicit Reader(Store &s) : connection(s.db, false) {}
      template <class F> void read(F &&fn) { connection.transaction(std::forward<F>(fn), true); }
   };
   uint64_t count()
   {
      Connection::Result r;
      r.check(duckdb_query(writer->conn, "SELECT COUNT(*) FROM accounts", &r.value));
      require(duckdb_row_count(&r.value) == 1, "DuckDB count failed");
      return duckdb_value_uint64(&r.value, 0, 0);
   }
};
} // namespace fair_bank

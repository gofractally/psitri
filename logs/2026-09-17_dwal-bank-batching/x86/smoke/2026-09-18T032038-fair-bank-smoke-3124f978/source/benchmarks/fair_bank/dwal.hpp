#pragma once
#include "common.hpp"
#include <psitri/dwal/byte_engine.hpp>
#include <sal_art_ref/engine.hpp>
namespace fair_bank
{
using Engine = psitri::dwal::byte_engine<sal_art_ref::radix::engine>;
inline constexpr auto engine_name = "dwal";
struct Store
{
   static bool supports(std::string_view level) { return level == "none"; }
   static std::string version() { return "DWAL over Large Engine; WAL format 2"; }
   static std::string settings(const Config& c)
   {
      return "base=Large Engine; base_sync=none; WAL=pwrite per commit, no per-commit fsync; "
             "rotation/close=full barrier; max_rw_entries=100000; max_wal_bytes=67108864; "
             "max_rw_arena_bytes=1073741824; merge_threads=2; enable_read_cache=false; "
             "max_pinned_cache_size_mb=" + std::to_string(c.cache_mib);
   }
   std::shared_ptr<Engine::database> db;
   std::shared_ptr<Engine::write_session> ws;
   Store(const Config& c, bool reopen)
   {
      Engine::config cfg;
      cfg.base.sync_mode = sal::sync_type::none;
      cfg.base.enable_read_cache = false;
      cfg.base.max_pinned_cache_size_mb = c.cache_mib;
      cfg.base.max_database_size = c.map_gib * (uint64_t(1) << 30);
      db = Engine::database::open(c.path, reopen ? sal::open_mode::open_existing : sal::open_mode::create_only, cfg);
      ws = db->start_write_session();
   }
   template <class T> struct View
   {
      T& tx;
      std::string get(const std::string& key)
      {
         auto value = tx.template get<std::string>(key);
         require(bool(value), "Missing account");
         return std::move(*value);
      }
      void put(const std::string& key, const std::string& value) { tx.upsert(key, value); }
   };
   template <class F> void write(F&& fn, bool commit = true)
   {
      auto tx = ws->start_transaction(0);
      View view{tx};
      fn(view);
      if (commit) tx.commit(); else tx.abort();
   }
   struct Reader
   {
      Store& store;
      explicit Reader(Store& s) : store(s) {}
      template <class F> void read(F&& fn)
      {
         auto rs = store.db->start_read_session();
         auto cur = rs->create_cursor(0);
         View view{cur};
         fn(view);
      }
   };
   uint64_t count()
   {
      auto rs = db->start_read_session();
      auto cur = rs->create_cursor(0);
      uint64_t n = 0;
      for (bool valid = cur.seek_begin(); valid; valid = cur.next()) ++n;
      return n;
   }
};
}

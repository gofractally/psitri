#pragma once
#include "common.hpp"
#if FAIR_small
#include <psitri/engines/small.hpp>
#else
#include <psitri/engines/big.hpp>
#endif
namespace fair_bank
{
#if FAIR_small
using Engine = psitri::small_engine;
inline constexpr auto engine_name = "small";
#else
using Engine = psitri::big_engine;
inline constexpr auto engine_name = "large";
#endif
struct Store
{
   static std::string version() { return "psitri API " + std::to_string(psitri::api_version); }
   static std::string settings(const Config& c)
   {
      return std::string("sync_mode=") + (c.sync ? "fsync" : "none") +
#if FAIR_small
         "; fixed arena; default max_segments=4096; background release enabled";
#else
         "; enable_read_cache=false; max_pinned_cache_size_mb=" + std::to_string(c.cache_mib);
#endif
   }
   std::shared_ptr<Engine::database> db;
   std::shared_ptr<Engine::write_session> ws;
   Store(const Config& c, bool reopen)
   {
      Engine::config config;
      config.sync_mode = c.sync ? psitri::sync_mode::fsync : psitri::sync_mode::none;
#if FAIR_large
      config.max_pinned_cache_size_mb = c.cache_mib;
      config.max_database_size = c.map_gib * (uint64_t(1) << 30);
#endif
      db = Engine::database::open(c.path, reopen ? psitri::open_mode::open_existing : psitri::open_mode::create_only, config);
      ws = db->start_write_session();
   }
   template <class T> struct View
   {
      T& tx;
      std::string get(const std::string& key)
      {
         auto value = tx.template get<std::string>(key);
         require(bool(value), "Missing account");
         return *value;
      }
      void put(const std::string& key, const std::string& value) { tx.upsert(key, value); }
   };
   template <class F> void write(F&& fn, bool commit = true)
   {
      auto tx = ws->start_transaction(0);
      View view{tx}; fn(view);
      if (commit) tx.commit(); else tx.abort();
   }
   struct Reader
   {
      std::shared_ptr<Engine::read_session> rs;
      explicit Reader(Store& store) : rs(store.db->start_read_session()) {}
      template <class F> void read(F&& fn)
      {
         auto cur = rs->create_cursor(0);
         View view{cur}; fn(view);
      }
   };
};
}

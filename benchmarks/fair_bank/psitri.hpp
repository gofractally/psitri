#pragma once
#include "common.hpp"
#if FAIR_small
#include <psitri/engines/small.hpp>
#elif FAIR_tiered
#include <psitri/experimental/tiered.hpp>
#else
#include <psitri/engines/big.hpp>
#endif
namespace fair_bank
{
#if FAIR_small
using Engine = psitri::small_engine;
inline constexpr auto engine_name = "small";
#elif FAIR_tiered
using Engine = psitri::experimental::tiered_engine<>;
inline constexpr auto engine_name = "tiered";
#else
using Engine = psitri::big_engine;
inline constexpr auto engine_name = "large";
#endif
struct Store
{
   static bool supports(std::string_view) { return true; }
   static psitri::sync_mode sync_policy(const Config &c)
   {
      if (c.durability == "full")
         return psitri::sync_mode::full;
      if (c.durability == "fsync")
         return psitri::sync_mode::fsync;
      if (c.durability == "msync")
         return psitri::sync_mode::msync_sync;
      if (c.durability == "async")
         return psitri::sync_mode::msync_async;
      return psitri::sync_mode::none;
   }
   static std::string version() { return "psitri API " + std::to_string(psitri::api_version); }
   static std::string settings(const Config &c)
   {
      return std::string("sync_mode=") + c.durability +
#if FAIR_tiered
             "; tiered Small Engine -> Large Engine; latest view; no WAL; background flush enabled; "
             "hi_water_bytes=16777216; lo_water_bytes=4194304; flush_chunk=10000000; "
             "Small Engine default max_segments=4096; background release enabled; "
             "Large Engine enable_read_cache=false; max_pinned_cache_size_mb=" + std::to_string(c.cache_mib);
#elif FAIR_small
             "; fixed arena; default max_segments=4096; background release enabled";
#else
             "; enable_read_cache=false; max_pinned_cache_size_mb=" + std::to_string(c.cache_mib);
#endif
   }
   std::shared_ptr<Engine::database> db;
   std::shared_ptr<Engine::write_session> ws;
   Store(const Config &c, bool reopen)
   {
      Engine::config config;
      // full requests F_FULLFSYNC on macOS and fsync on other platforms.
#if FAIR_tiered
      config.commit_sync = sync_policy(c);
      config.bcfg.enable_read_cache = false;
      config.bcfg.max_pinned_cache_size_mb = c.cache_mib;
      config.bcfg.max_database_size = c.map_gib * (uint64_t(1) << 30);
      // Exercise migration within this resident dataset; defaults would keep
      // the entire bank run in the hot tier. Record the threshold explicitly.
      config.knobs.hi_water_bytes = 16ull << 20;
      config.knobs.lo_water_bytes = 4ull << 20;
#else
      config.sync_mode = sync_policy(c);
#endif
#if FAIR_large
      config.max_pinned_cache_size_mb = c.cache_mib;
      config.max_database_size = c.map_gib * (uint64_t(1) << 30);
#endif
      db = Engine::database::open(
          c.path, reopen ? psitri::open_mode::open_existing : psitri::open_mode::create_only,
          config);
      ws = db->start_write_session();
   }
   template <class T> struct View
   {
      T &tx;
      std::string get(const std::string &key)
      {
         auto value = tx.template get<std::string>(key);
         require(bool(value), "Missing account");
         return std::move(*value);
      }
      void put(const std::string &key, const std::string &value) { tx.upsert(key, value); }
   };
   template <class F> void write(F &&fn, bool commit = true)
   {
      auto tx = ws->start_transaction(0);
      View view{tx};
      fn(view);
      if (commit)
         tx.commit();
      else
         tx.abort();
   }
   struct Reader
   {
      std::shared_ptr<Engine::read_session> rs;
      explicit Reader(Store &store) : rs(store.db->start_read_session()) {}
      template <class F> void read(F &&fn)
      {
         auto cur = rs->create_cursor(0);
         View view{cur};
         fn(view);
      }
   };
   uint64_t count()
   {
      auto rs = db->start_read_session();
      auto cursor = rs->create_cursor(0);
      uint64_t n = 0;
      for (bool valid = cursor.seek_begin(); valid; valid = cursor.next())
         ++n;
#if FAIR_tiered
      std::cerr << "tiered base commits: " << db->base_commit_count() << "\n";
#endif
      return n;
   }
};
} // namespace fair_bank

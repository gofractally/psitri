#pragma once
#include "common.hpp"
#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/table.h>
#ifdef FAIR_API_PAIR
#include <rocksdb/write_batch.h>
#include <map>
#else
#include <rocksdb/utilities/write_batch_with_index.h>
#endif
#ifndef FAIR_SHIM
#include <rocksdb/version.h>
#endif
namespace fair_bank
{
#ifdef FAIR_SHIM
inline constexpr auto engine_name = "psitrirocks";
#else
inline constexpr auto engine_name = "rocksdb";
#endif
inline void check(const rocksdb::Status &status)
{
   if (!status.ok())
      throw std::runtime_error(status.ToString());
}
struct Store
{
   std::unique_ptr<rocksdb::DB> db;
   rocksdb::WriteOptions wo;
   static bool supports(std::string_view level)
   {
      return level == "none" || level == FAIR_ROCKSDB_SYNC_LEVEL;
   }
   static std::string version()
   {
#ifdef FAIR_SHIM
      return "psitrirocks " FAIR_SHIM_REVISION;
#else
      return std::to_string(ROCKSDB_MAJOR) + "." + std::to_string(ROCKSDB_MINOR) + "." +
             std::to_string(ROCKSDB_PATCH);
#endif
   }
   static std::string settings(const Config &c)
   {
      return std::string("WAL=true; sync=") + (c.synchronous() ? "true" : "false") +
             "; platform_sync=" FAIR_ROCKSDB_SYNC_LEVEL
             "; compression=none; write_buffer_size=64MiB; max_write_buffer_number=2; "
             "block_cache_mib=" +
             std::to_string(c.cache_mib);
   }
   Store(const Config &c, bool reopen)
   {
      rocksdb::Options options;
      options.create_if_missing = !reopen;
      options.error_if_exists = !reopen;
      options.compression = rocksdb::kNoCompression;
      options.write_buffer_size = 64 * 1024 * 1024;
      options.max_write_buffer_number = 2;
      rocksdb::BlockBasedTableOptions table;
      table.block_cache = rocksdb::NewLRUCache(c.cache_mib * 1024 * 1024);
      options.table_factory = decltype(options.table_factory)(rocksdb::NewBlockBasedTableFactory(table));
      rocksdb::DB *ptr = nullptr;
      check(rocksdb::DB::Open(options, c.path.string(), &ptr));
      db.reset(ptr);
      wo.sync = c.synchronous();
      wo.disableWAL = false;
   }
   struct Writer
   {
      Store &store;
#ifdef FAIR_API_PAIR
      // Both implementations receive the identical RocksDB API call sequence.
      // The shim supports WriteBatch; the application supplies read-your-writes.
      rocksdb::WriteBatch batch{};
      std::map<std::string, std::string> pending;
      std::string get(const std::string &key)
      {
         if (auto it = pending.find(key); it != pending.end())
            return it->second;
         std::string value;
         check(store.db->Get({}, key, &value));
         return value;
      }
      void put(const std::string &key, const std::string &value)
      {
         check(batch.Put(key, value));
         pending[key] = value;
      }
#else
      rocksdb::WriteBatchWithIndex batch{rocksdb::BytewiseComparator(), 0, true};
      std::string get(const std::string &key)
      {
         std::string value;
         check(batch.GetFromBatchAndDB(store.db.get(), {}, key, &value));
         return value;
      }
      void put(const std::string &key, const std::string &value) { check(batch.Put(key, value)); }
#endif
   };
   template <class F> void write(F &&fn, bool commit = true)
   {
      Writer writer{*this};
      fn(writer);
      if (commit)
#ifdef FAIR_API_PAIR
         check(db->Write(wo, &writer.batch));
#else
         check(db->Write(wo, writer.batch.GetWriteBatch()));
#endif
   }
   struct Reader
   {
      Store &store;
      rocksdb::ReadOptions options;
      explicit Reader(Store &s) : store(s) {}
      std::string get(const std::string &key)
      {
         std::string value;
         check(store.db->Get(options, key, &value));
         return value;
      }
      template <class F> void read(F &&fn)
      {
         options.snapshot = store.db->GetSnapshot();
         require(options.snapshot, "RocksDB snapshot unavailable");
         try
         {
            fn(*this);
         }
         catch (...)
         {
            store.db->ReleaseSnapshot(options.snapshot);
            options.snapshot = nullptr;
            throw;
         }
         store.db->ReleaseSnapshot(options.snapshot);
         options.snapshot = nullptr;
      }
   };
   uint64_t count()
   {
      std::unique_ptr<rocksdb::Iterator> iterator(db->NewIterator({}));
      uint64_t n = 0;
      for (iterator->SeekToFirst(); iterator->Valid(); iterator->Next())
         ++n;
      check(iterator->status());
      return n;
   }
};
} // namespace fair_bank

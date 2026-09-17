#pragma once
#include "common.hpp"
#include <mdbx.h>
namespace fair_bank
{
inline constexpr auto engine_name = "mdbx";
inline void check(int rc)
{
   if (rc != MDBX_SUCCESS)
      throw std::runtime_error(mdbx_strerror(rc));
}
struct Store
{
   MDBX_env *env = nullptr;
   MDBX_dbi dbi;
   static bool supports(std::string_view level)
   {
      if (level == "none" || level == "async")
         return true;
#ifdef __APPLE__
      std::string_view options = mdbx_build.options ? mdbx_build.options : "";
      return (level == "full" &&
              options.find("MDBX_APPLE_SPEED_INSTEADOF_DURABILITY=0") != options.npos) ||
             (level == "fsync" &&
              options.find("MDBX_APPLE_SPEED_INSTEADOF_DURABILITY=1") != options.npos);
#else
      return level == "full" || level == "fsync";
#endif
   }
   static const char *policy(const Config &c)
   {
      return c.synchronous()           ? "SYNC_DURABLE"
             : c.durability == "async" ? "SAFE_NOSYNC"
                                       : "UTTERLY_NOSYNC";
   }
   static std::string version() { return mdbx_version.git.describe; }
   static std::string settings(const Config &c)
   {
      return std::string("WRITEMAP; ") + policy(c) + "; map_gib=" + std::to_string(c.map_gib) +
             "; build_options=" + (mdbx_build.options ? mdbx_build.options : "unknown") +
             "; compiler=" + (mdbx_build.compiler ? mdbx_build.compiler : "unknown") +
             "; build_flags=" + (mdbx_build.flags ? mdbx_build.flags : "unknown");
   }
   Store(const Config &c, bool reopen)
   {

      fs::create_directories(c.path);
      check(mdbx_env_create(&env));
      try
      {
         check(mdbx_env_set_maxreaders(env, uint32_t(c.readers + 16)));
         check(mdbx_env_set_geometry(env, -1, -1, int64_t(c.map_gib) << 30, -1, -1, -1));
         check(mdbx_env_open(
             env, c.path.c_str(),
             MDBX_env_flags_t(MDBX_WRITEMAP | (c.synchronous()           ? MDBX_SYNC_DURABLE
                                               : c.durability == "async" ? MDBX_SAFE_NOSYNC
                                                                         : MDBX_UTTERLY_NOSYNC)),
             0600));
         MDBX_txn *t = nullptr;
         check(mdbx_txn_begin(env, nullptr, MDBX_TXN_READWRITE, &t));
         int rc = mdbx_dbi_open(t, nullptr, MDBX_DB_DEFAULTS, &dbi);
         if (rc)
         {
            mdbx_txn_abort(t);
            check(rc);
         }
         check(mdbx_txn_commit(t));
      }
      catch (...)
      {
         mdbx_env_close(env);
         throw;
      }
   }
   ~Store() { mdbx_env_close(env); }
   struct Transaction
   {
      Store &store;
      MDBX_txn *tx = nullptr;
      Transaction(Store &s, bool write) : store(s)
      {
         check(mdbx_txn_begin(s.env, nullptr, write ? MDBX_TXN_READWRITE : MDBX_TXN_RDONLY, &tx));
      }
      ~Transaction()
      {
         if (tx)
            mdbx_txn_abort(tx);
      }
      std::string get(const std::string &key)
      {
         MDBX_val k{const_cast<char *>(key.data()), key.size()}, v;
         check(mdbx_get(tx, store.dbi, &k, &v));
         return std::string(static_cast<char *>(v.iov_base), v.iov_len);
      }
      void put(const std::string &key, const std::string &value)
      {
         MDBX_val k{const_cast<char *>(key.data()), key.size()},
             v{const_cast<char *>(value.data()), value.size()};
         check(mdbx_put(tx, store.dbi, &k, &v, MDBX_UPSERT));
      }
      void commit()
      {
         auto t = tx;
         tx = nullptr;
         check(mdbx_txn_commit(t));
      }
   };
   template <class F> void write(F &&fn, bool commit = true)
   {
      Transaction tx(*this, true);
      fn(tx);
      if (commit)
         tx.commit();
   }
   struct Reader
   {
      Store &store;
      explicit Reader(Store &s) : store(s) {}
      template <class F> void read(F &&fn)
      {
         Transaction tx(store, false);
         fn(tx);
      }
   };
   uint64_t count()
   {
      Transaction tx(*this, false);
      MDBX_stat stat;
      check(mdbx_dbi_stat(tx.tx, dbi, &stat, sizeof(stat)));
      return stat.ms_entries;
   }
};
} // namespace fair_bank

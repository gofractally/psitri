#pragma once
// heart's NATIVE psitri-shaped public API: the same database → write_session → transaction
// → cursor structure psitri exposes, so heart is a drop-in for psitri's transaction/session
// API. heart::radix::database already carries the shape natively (bank_bench drives it
// directly); this header only papers over the two spelling gaps the templated harness
// needs — DB::create(dir, runtime_config) over database::open(dir), and
// transaction::get(key, lambda) over get<std::string> — and re-exports the rest.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <sal/config.hpp>  // sal::sync_type

#include "radix_engine.hpp"

namespace heart::kv
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   // Mirrors psitri::runtime_config's relevant fields (same sal types) so a templated caller
   // can fill `cfg.sync_mode` etc. uniformly for either engine.
   struct runtime_config
   {
      sal::sync_type sync_mode               = sal::sync_type::none;
      bool           write_protect_on_commit = false;
      bool           checksum_on_commit      = false;
   };

   using engine = heart::radix::database;
   using cursor = heart::radix::radix_tree::cursor;
   using tree   = heart::radix::radix_tree::tree;  // has snapshot_cursor()

   // Thin spelling adapter over the native transaction (adds get(key, lambda)).
   class transaction
   {
     public:
      explicit transaction(heart::radix::radix_tree::transaction&& t) : _tx(std::move(t)) {}
      void insert(key_view k, value_view v) { _tx.upsert(k, v); }
      void upsert(key_view k, value_view v) { _tx.upsert(k, v); }
      int  remove(key_view k) { return _tx.remove(k); }
      void commit() { _tx.commit(); }
      void abort() noexcept { _tx.abort(); }

      template <class T = std::string>
      std::optional<T> get(key_view k)
      {
         return _tx.template get<T>(k);
      }
      // psitri::transaction::get(key, lambda) analogue: fn(value_view) on hit, return found.
      template <class F>
      bool get(key_view k, F&& fn)
      {
         auto v = _tx.template get<std::string>(k);
         if (!v)
            return false;
         fn(value_view(*v));
         return true;
      }

     private:
      heart::radix::radix_tree::transaction _tx;
   };

   class write_session
   {
     public:
      explicit write_session(std::shared_ptr<heart::radix::write_session> ws)
          : _ws(std::move(ws))
      {
      }
      transaction start_transaction(uint32_t root_index)
      {
         return transaction(_ws->start_transaction(root_index));
      }
      tree get_root(uint32_t root_index) { return _ws->get_root(root_index); }

     private:
      std::shared_ptr<heart::radix::write_session> _ws;
   };

   class database
   {
     public:
      static std::shared_ptr<database> create(std::filesystem::path dir, runtime_config = {})
      {
         std::filesystem::create_directories(dir);
         return std::shared_ptr<database>(new database(heart::radix::database::open(dir)));
      }
      std::shared_ptr<write_session> start_write_session()
      {
         return std::make_shared<write_session>(_db->start_write_session());
      }

     private:
      explicit database(std::shared_ptr<heart::radix::database> db) : _db(std::move(db)) {}
      std::shared_ptr<heart::radix::database> _db;
   };
}  // namespace heart::kv

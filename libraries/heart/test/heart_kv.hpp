#pragma once
// heart's NATIVE psitri-shaped public API: the same database → write_session → transaction
// → cursor structure psitri exposes, so heart is a drop-in for psitri's transaction/session
// API. This is heart's own API surface (not a per-engine test wrapper) — a templated
// test/bench can drive psitri::* and heart::kv::* through identical calls, differing only in
// which type names it names. Backed by the heart radix engine; commit() is the version/
// snapshot boundary (after it, writes COW), matching a psitri transaction commit.
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

   using engine = heart::radix::radix_tree;

   // A read view over the latest committed tree (psitri::cursor analogue). Constructed from a
   // "root" handle (here, the engine pointer that get_root() returns).
   class cursor
   {
     public:
      explicit cursor(engine* t) noexcept : _t(t) {}
      // psitri::cursor::get contract: >=0 (value size) on hit, <0 on miss; fills *buffer.
      int32_t get(key_view k, std::string* buffer)
      {
         std::string_view v;
         if (!_t->find(k, v))
            return -1;
         buffer->assign(v.data(), v.size());
         return int32_t(v.size());
      }

      // ── stateful ordered iteration (psitri::cursor analogue) ─────────────────────
      // Positioning returns is_end() like psitri. Forward only for now (next); reverse
      // (prev/last) needs a reverse ordered-seek in the engine — not yet wired.
      bool seek_begin() { return seek_lb(key_view{}); }               // position at first key
      bool first(key_view prefix = {}) { return seek_lb(prefix); }     // first key >= prefix
      bool lower_bound(key_view k) { return seek_lb(k); }              // first key >= k
      bool upper_bound(key_view k) { return seek_ub(k); }             // first key > k
      bool find(key_view k)                                            // exact; end on miss
      {
         std::string_view v;
         _valid = _t->find(k, v);
         if (_valid)
         {
            _key.assign(k.data(), k.size());
            _val.assign(v.data(), v.size());
         }
         return !_valid;
      }
      bool next()  // advance to the smallest key strictly greater than the current
      {
         if (!_valid)
            return true;
         std::string      nk;
         std::string_view v;
         _valid = _t->upper_bound(_key, nk, v);
         if (_valid)
         {
            _key.swap(nk);
            _val.assign(v.data(), v.size());
         }
         return !_valid;
      }
      bool     is_end() const noexcept { return !_valid; }
      key_view key() const noexcept { return _key; }
      template <class F>
      void get_value(F&& fn) const
      {
         fn(value_view(_val));
      }
      // psitri::cursor::value<T>() analogue: the current value as an optional buffer.
      template <class T = std::string>
      std::optional<T> value() const
      {
         if (!_valid)
            return std::nullopt;
         return T(_val.begin(), _val.end());
      }

     private:
      bool seek_lb(key_view k)
      {
         std::string_view v;
         _valid = _t->lower_bound(k, _key, v);
         if (_valid)
            _val.assign(v.data(), v.size());
         return !_valid;
      }
      bool seek_ub(key_view k)
      {
         std::string_view v;
         _valid = _t->upper_bound(k, _key, v);
         if (_valid)
            _val.assign(v.data(), v.size());
         return !_valid;
      }
      engine*     _t;
      std::string _key, _val;
      bool        _valid = false;
   };

   // A root handle (psitri::tree analogue) returned by get_root(); hands out read cursors.
   class tree
   {
     public:
      explicit tree(engine* t) noexcept : _t(t) {}
      cursor   snapshot_cursor() const { return cursor(_t); }  // call seek_begin()/lower_bound() to position
      explicit operator bool() const noexcept { return _t != nullptr; }

     private:
      engine* _t;
   };

   // A write transaction targeting one root. Mutations apply to the engine's working tree
   // (COW-to-root); commit() publishes the version (snapshot boundary).
   class transaction
   {
     public:
      explicit transaction(engine* t) noexcept : _t(t) {}
      void insert(key_view k, value_view v) { _t->upsert(k, v); }
      void upsert(key_view k, value_view v) { _t->upsert(k, v); }
      int  remove(key_view k) { return _t->remove(k) ? 0 : -1; }
      void commit() { _t->commit(); }
      void abort() noexcept {}
      // psitri::transaction::get(key, lambda) analogue: invoke fn(value_view) on hit, return found.
      template <class F>
      bool get(key_view k, F&& fn) const
      {
         value_view v;
         if (!_t->find(k, v))
            return false;
         fn(v);
         return true;
      }

     private:
      engine* _t;
   };

   class write_session
   {
     public:
      explicit write_session(engine* t) noexcept : _t(t) {}
      // Single-root for now (root_index must be 0) — enough for the drop-in comparison.
      transaction start_transaction(uint32_t root_index)
      {
         (void)root_index;
         return transaction(_t);
      }
      tree get_root(uint32_t root_index)
      {
         (void)root_index;
         return tree(_t);
      }

     private:
      engine* _t;
   };

   class database
   {
     public:
      static std::shared_ptr<database> create(std::filesystem::path dir, runtime_config = {})
      {
         return std::shared_ptr<database>(new database(std::move(dir)));
      }
      std::shared_ptr<write_session> start_write_session()
      {
         return std::make_shared<write_session>(&_engine);
      }

     private:
      explicit database(std::filesystem::path dir) : _engine(std::move(dir)) {}
      engine _engine;  // heart radix engine (cfg: sync none, matching the comparison setup)
   };
}  // namespace heart::kv

// gcart::st::kv — adb-style adapter over the spec engine (gcart::st), with
// lock-free MVCC. Mirrors the v1 gcart::kv discipline (smart_ptr IS the pin,
// try_retain+verify acquire, COW-to-root, give() hand-off) but the root is a
// (branch_type tag, u32 ptr) pair packed into an atomic u64 published slot.
#pragma once
#include <gcart/spec_tree.hpp>

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace gcart::st::kv
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   inline uint64_t pack(bt t, uint32_t p) noexcept { return (uint64_t(uint8_t(t)) << 32) | p; }
   inline bt       rtag(uint64_t r) noexcept { return bt(uint8_t(r >> 32)); }
   inline uint32_t rptr(uint64_t r) noexcept { return uint32_t(r); }
   inline constexpr uint64_t null_root = (uint64_t(uint8_t(bt::null)) << 32);

   // ── smart_ptr: a held (tag,ptr) root IS the snapshot pin ───────────────────
   class snap
   {
     public:
      snap() = default;
      static snap acquire(engine& e, std::atomic<uint64_t>& slot)
      {
         for (;;)
         {
            const uint64_t r = slot.load(std::memory_order_acquire);
            const bt t = rtag(r); const uint32_t p = rptr(r);
            if (t == bt::null) return snap(&e, bt::null, 0);
            if (!try_retain(e, t, p)) continue;                 // lost race with a freeing writer
            if (r == slot.load(std::memory_order_acquire)) return snap(&e, t, p);
            release_subtree(e, t, p);                           // root moved — undo, retry
         }
      }
      snap(const snap& o) : _e(o._e), _t(o._t), _p(o._p) { if (_e && _t != bt::null) retain(*_e, _t, _p); }
      snap& operator=(const snap& o)
      { if (this != &o) { reset(); _e = o._e; _t = o._t; _p = o._p; if (_e && _t != bt::null) retain(*_e, _t, _p); } return *this; }
      snap(snap&& o) noexcept : _e(o._e), _t(o._t), _p(o._p) { o._t = bt::null; }
      snap& operator=(snap&& o) noexcept { if (this != &o) { reset(); _e = o._e; _t = o._t; _p = o._p; o._t = bt::null; } return *this; }
      ~snap() { reset(); }

      bt       tag() const noexcept { return _t; }
      uint32_t ptr() const noexcept { return _p; }
      engine*  eng() const noexcept { return _e; }
      void give() noexcept { _t = bt::null; }   // relinquish without releasing
      // the pin OWNS its release. (A/B: SYNC release to isolate the bump allocator
      // from the async bg cascade.)
      void reset() noexcept { if (_e && _t != bt::null) release_subtree(*_e, _t, _p); _t = bt::null; }

     private:
      snap(engine* e, bt t, uint32_t p) : _e(e), _t(t), _p(p) {}
      engine*  _e = nullptr;
      bt       _t = bt::null;
      uint32_t _p = 0;
   };

   class database;

   // ── cursor: a pinned snapshot + point get ──────────────────────────────────
   class cursor
   {
     public:
      cursor() = default;
      explicit cursor(snap s) : _s(std::move(s)) {}
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if (!_s.eng() || _s.tag() == bt::null) return std::nullopt;
         std::string out;
         if (!find_in(*_s.eng(), _s.tag(), _s.ptr(), k, out)) return std::nullopt;
         return S(out.data(), out.size());
      }
     private:
      snap _s;
   };

   // ── transaction: COW snapshot over one root ────────────────────────────────
   class transaction
   {
     public:
      transaction(transaction&& o) noexcept
          : _db(o._db), _t(o._t), _slot(o._slot), _snap(std::move(o._snap)), _done(o._done) { o._db = nullptr; }
      transaction(const transaction&)            = delete;
      transaction& operator=(const transaction&) = delete;
      ~transaction();

      void insert(key_view k, value_view v) { _t->upsert(k, v); }
      void upsert(key_view k, value_view v) { _t->upsert(k, v); }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         std::string out;
         if (!_t->find(k, out)) return std::nullopt;
         return S(out.data(), out.size());
      }
      void commit();
      void abort() noexcept;

     private:
      friend class write_session;
      transaction(database* db, tree* t, uint32_t slot);
      database* _db = nullptr;
      tree*     _t  = nullptr;
      uint32_t  _slot = 0;
      snap      _snap;
      bool      _done = false;
   };

   class write_session
   {
     public:
      explicit write_session(database* db) : _db(db) {}
      transaction start_transaction(uint32_t idx = 0);
      tree        get_root_unused() = delete;
     private:
      database* _db;
   };

   class read_session
   {
     public:
      explicit read_session(database* db) : _db(db) {}
      cursor create_cursor(uint32_t idx);
     private:
      database* _db;
   };

   // ── database ───────────────────────────────────────────────────────────────
   class database
   {
     public:
      static std::shared_ptr<database> create(std::filesystem::path dir)
      { return std::shared_ptr<database>(new database(std::move(dir))); }
      static std::shared_ptr<database> open(std::filesystem::path dir) { return create(std::move(dir)); }

      std::shared_ptr<write_session> start_write_session() { return std::make_shared<write_session>(this); }
      std::shared_ptr<read_session>  start_read_session() { return std::make_shared<read_session>(this); }
      void sync(uint32_t = 0) noexcept {}

      engine&               eng() noexcept { return _e; }
      std::atomic<uint64_t>& root_slot(uint32_t i) noexcept { return _roots[i]; }
      tree&                 work_tree(uint32_t i)
      {
         if (!_work[i]) _work[i] = std::make_unique<tree>(_e);
         return *_work[i];
      }
      // settled once the engine's background cascade has caught up.
      uint64_t live_bytes() noexcept { return _e.live_bytes(); }
      void     drain() noexcept { _e.drain(); }   // measurement barrier

     private:
      explicit database(std::filesystem::path dir) : _e(std::move(dir))
      { for (auto& r : _roots) r.store(null_root, std::memory_order_relaxed); }

      engine                                  _e;
      std::array<std::atomic<uint64_t>, 64>   _roots;
      std::array<std::unique_ptr<tree>, 64>   _work{};
   };

   // ── bodies ─────────────────────────────────────────────────────────────────
   inline transaction::transaction(database* db, tree* t, uint32_t slot)
       : _db(db), _t(t), _slot(slot), _snap(snap::acquire(db->eng(), db->root_slot(slot)))
   {
      _t->set_root(_snap.tag(), _snap.ptr());  // working tree starts from the committed root
   }
   inline transaction::~transaction() { if (_db && !_done) abort(); }

   inline void transaction::commit()
   {
      if (_done) return;
      // publish the new working root atomically, then drop the snapshot pin on
      // the old committed root. _snap.reset() routes through the engine's
      // background reclaimer (the cascade runs off the writer's critical path);
      // the smart_ptr owns its own release.
      _db->root_slot(_slot).store(pack(_t->root_tag(), _t->root_ptr()), std::memory_order_release);
      _snap.reset();
      _done = true;
   }
   inline void transaction::abort() noexcept
   {
      if (_done) return;
      const bt wt = _t->root_tag(); const uint32_t wp = _t->root_ptr();
      _t->set_root(_snap.tag(), _snap.ptr());   // restore committed
      _snap.give();                              // the snap pin becomes the live tree's ownership ref
      release_subtree(_db->eng(), wt, wp);       // free the discarded work (balances give; no-op-safe if ==committed)
      _done = true;
   }

   inline transaction write_session::start_transaction(uint32_t idx)
   { return transaction(_db, &_db->work_tree(idx), idx); }

   inline cursor read_session::create_cursor(uint32_t idx)
   { return cursor(snap::acquire(_db->eng(), _db->root_slot(idx))); }
}  // namespace gcart::st::kv

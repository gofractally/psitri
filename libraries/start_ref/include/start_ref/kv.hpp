// start_ref::kv — part's NATIVE psitri-shaped public API (the same database →
// write_session → transaction → cursor structure psitri and heart::kv expose),
// so one templated test/bench body drives all three engines through identical
// calls. See heart/test/heart_kv.hpp for the canonical shape this mirrors and
// programs/kv_compat_test.cpp for the conformance driver.
//
// Engine: the verbatim-hkv tree on the part arena (part/hkv/tree.hpp +
// part_alloc), with the part COW extension supplying snapshot semantics:
// commit retains the new root for the database's live-root slot and releases
// the old one; snapshot cursors pin the live root (the 191..254 refcount band)
// and everything reachable under the pin is immutable by the COW discipline
// (SPEC §7), so cursor traversal needs no further refcounting. Values are POD
// inline byte strings (<= 126 B): COW clones are memcpys and every value byte
// lives in the arena.
#pragma once
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <chrono>
#include <condition_variable>
#include <thread>

#include <start_ref/hkv/tree.hpp>
#include <start_ref/tree_alloc.hpp>

namespace start_ref::kv
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   // Mirrors psitri::runtime_config's relevant fields (same sal types).
   struct runtime_config
   {
      sal::sync_type sync_mode               = sal::sync_type::none;
      bool           write_protect_on_commit = false;
      bool           checksum_on_commit      = false;
      uint32_t       max_segments            = 4096;  // x 8MB arena ceiling (32 GB default)
      // SPEC §19: when false, no releaser thread is spawned and the defer hook
      // is never registered — every release runs synchronously on its caller
      // (identical semantics, for deployments that cannot tolerate a thread).
      bool           background_release      = true;
   };

   // Inline POD value: len-prefixed bytes, trivially copyable (COW memcpy).
   // 80 B total keeps a typical leaf (4 B header + value + short suffix) inside
   // ONE 128 B line — at 128 B every 8-byte bank balance burned a 2-line leaf,
   // which alone doubled resident memory. Variable-length leaf values (design5
   // val_len) are the real fix and live in the tree layer's future.
   // (pval[80] retired: values are raw bytes in the leaf — hkv::bytes_value)

   // ELEGANT value mode: the leaf IS the value (raw bytes sized to the data);
   // reads hand back views — pval[80] retired.
   using bval     = hkv::bytes_value;
   using engine_t = hkv::kv_tree<std::string_view, bval, hkv::mode::none, part_alloc<bval>>;
   using root_ptr = start_ref::detail::line_ptr;

   // Multi-root: arena slots [0, kMaxRoots) hold per-root committed raws (the
   // arena table is sized max_roots=16). start_ref and sal_art_ref share the
   // SAME root/snapshot model — a tree is an index + a held refcount;
   // start_ref's control block is refcount-only (address-stable arena, no
   // relocation indirection). docs/start_ref_multiroot_design.md.
   inline constexpr uint32_t kMaxRoots = 8;

   class database;
   class write_session;
   class transaction;
   class tree_handle;

   // ── typed values (the LOCKED adb API) ─────────────────────────────────
   // EVERY stored slot is [u16 LE typenum][payload]. Typenum 0 = raw bytes
   // (the null-hook builtin); registered ids >= 1 dispatch retain/release
   // through the database registry from the tree's value-lifecycle hooks —
   // overwrite, remove, abort, privatize clone, generation retirement and
   // background-releaser cascades included. Reads are THREE-OUTCOME:
   // found / not-found / invalid_type.
   inline constexpr uint16_t kRawType = 0;

   struct invalid_type : std::runtime_error
   {
      invalid_type(uint16_t stored, uint16_t asked)
          : std::runtime_error("start_ref: value is type " + std::to_string(stored) +
                               ", read as type " + std::to_string(asked))
      {
      }
   };

   template <class T>
   concept storable = requires(T t, const typename T::t_pod& p, void* ses) {
      { T::type_id } -> std::convertible_to<uint16_t>;
      requires std::is_trivially_copyable_v<typename T::t_pod>;
      { std::move(t).take_pod() } -> std::same_as<typename T::t_pod>;
      { T::make_from_pod(p, true) } -> std::same_as<T>;
      T::t_pod::retain(ses, p);
      T::t_pod::release(ses, p);
   };

   inline uint16_t slot_type(value_view s) noexcept
   {
      uint16_t t;
      std::memcpy(&t, s.data(), 2);
      return t;
   }
   inline value_view slot_payload(value_view s) noexcept
   {
      return s.substr(2);
   }
   inline value_view env_raw_(value_view v)  // [0][bytes], thread_local scratch
   {
      thread_local std::string buf;
      buf.resize(2 + v.size());
      buf[0] = buf[1] = 0;
      std::memcpy(buf.data() + 2, v.data(), v.size());
      return {buf.data(), buf.size()};
   }
   template <class S>
   S slot_decode(value_view slot)
   {
      const uint16_t t = slot_type(slot);
      if constexpr (storable<S>)
      {
         if (t != S::type_id)
            throw invalid_type{t, S::type_id};
         typename S::t_pod pod;
         if (slot.size() - 2 != sizeof(pod))
            throw invalid_type{t, S::type_id};
         std::memcpy(&pod, slot.data() + 2, sizeof(pod));
         return S::make_from_pod(pod, true);  // a RETAINED handle out
      }
      else
      {
         if (t != kRawType)
            throw invalid_type{t, kRawType};
         const value_view p = slot_payload(slot);
         return S(p.data(), p.size());
      }
   }

   // any_value — pointer to the value SLOT in the tree, no copy; inside a
   // write transaction the transaction itself is the pin (invalidated by a
   // later write to the same key). as<T>() retains; bytes() on a typed slot
   // is invalid_type; raw_pod() is the expert/merger identity path.
   class any_value
   {
     public:
      any_value() = default;
      uint16_t type() const noexcept { return _s ? slot_type({_s, _n}) : kRawType; }
      template <class T>
      bool is() const noexcept
      {
         return _s && slot_type({_s, _n}) == T::type_id;
      }
      template <class T>
      T as() const
      {
         if (!is<T>())
            throw invalid_type{type(), T::type_id};
         typename T::t_pod pod;
         std::memcpy(&pod, _s + 2, sizeof(pod));
         return T::make_from_pod(pod, true);
      }
      std::string_view bytes() const
      {
         if (type() != kRawType)
            throw invalid_type{type(), kRawType};
         return {_s + 2, _n - 2};
      }
      std::string_view raw_pod() const noexcept
      {
         return _s ? std::string_view{_s + 2, _n - 2} : std::string_view{};
      }
      explicit operator bool() const noexcept { return _s != nullptr; }

     private:
      friend class transaction;
      any_value(const char* s, size_t n) : _s(s), _n(n) {}
      const char* _s = nullptr;
      size_t      _n = 0;  // the FULL slot (envelope included), in the tree
   };

   // Snapshot cursor: a pinned root + a non-owning view tree + an hkv iterator.
   // The dtor drops the pin; if that was the generation's last reference the
   // old version drains back to the arena.
   class cursor
   {
     public:
      cursor(cursor&& o) noexcept
          : _a(o._a), _hp(o._hp), _root(o._root), _view(std::move(o._view)), _it(_view.end())
      {
         o._a  = nullptr;  // moved-from cursor owns no pin; position resets
         o._hp = nullptr;
      }
      cursor(const cursor&) = delete;
      ~cursor()
      {
         if (!_a)
            return;
         const root_ptr r = _root;
         _view.adopt_root(root_ptr::null());  // the view owns nothing
         if (_hp)
            _hp->store(0, std::memory_order_seq_cst);  // readers only clear their slot:
         else if (!r.is_null())                        // reclamation is the writer's job
            _view.cow_release(r);  // write-side pinned cursor (scan_all path)
      }

      void seek_begin() { _it = _view.begin(); }
      void lower_bound(key_view k) { _it = _view.lower_bound(k); }
      void upper_bound(key_view k)
      {
         _it = _view.lower_bound(k);
         if (_it != _view.end() && (*_it).first == k)
            ++_it;
      }
      void find(key_view k)
      {
         if constexpr (engine_t::cb_create)
         {  // pages: position at lower_bound and demand an exact key match
            _it = _view.lower_bound(k);
            if (_it != _view.end() && (*_it).first != k)
               _it = _view.end();
         }
         else
            _it = _view.find(k);
      }
      void next() { ++_it; }
      bool is_end() const { return _it == _view.end(); }

      key_view key() const { return (*_it).first; }

      template <class S = std::string>
      std::optional<S> value() const
      {
         if (is_end())
            return std::nullopt;
         return slot_decode<S>((*_it).second.v);
      }
      template <class F>
      bool get_value(F&& f) const
      {
         if (is_end())
            return false;
         const value_view s = (*_it).second.v;
         if (slot_type(s) != kRawType)
            throw invalid_type{slot_type(s), kRawType};
         f(slot_payload(s));
         return true;
      }
      uint16_t value_type() const { return is_end() ? kRawType : slot_type((*_it).second.v); }
      template <class T>
      bool is_type() const
      {
         return !is_end() && slot_type((*_it).second.v) == T::type_id;
      }

      // Point read against the snapshot (the bank reader hot path).
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         if constexpr (engine_t::cb_create)
         {
            std::string_view out;
            if (!_view.pg_find(k, out))
               return std::nullopt;
            return slot_decode<S>(out);
         }
         bval out;
         if (!_view.find(k, out))
            return std::nullopt;
         return slot_decode<S>(out.v);
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         if constexpr (engine_t::cb_create)
         {
            std::string_view out;
            if (!_view.pg_find(k, out))
               return false;
            if (slot_type(out) != kRawType)
               throw invalid_type{slot_type(out), kRawType};
            f(slot_payload(out));
            return true;
         }
         bval out;
         if (!_view.find(k, out))
            return false;
         if (slot_type(out.v) != kRawType)
            throw invalid_type{slot_type(out.v), kRawType};
         f(slot_payload(out.v));
         return true;
      }

     private:
      friend class database;
      friend class read_session;
      friend class tree;
      cursor(start_ref::arena*  a,
             root_ptr            root,
             std::atomic<uint64_t>* hp = nullptr,
             const value_hooks*  vh = nullptr)
          : _a(a), _hp(hp), _root(root), _view(part_alloc<bval>(a)), _it(_view.end())
      {
         _view.allocator_mut().vhooks = vh;  // write-side cursors can release
         _view.adopt_root(_root);
      }

      start_ref::arena*             _a;
      std::atomic<uint64_t>*   _hp;  // sal paradigm: the reader's published slot
      root_ptr                 _root;
      mutable engine_t         _view;
      engine_t::const_iterator _it;
   };

   // A reader thread's handle: one start_ref::read_session slot for the lifetime of
   // the thread; cursors pin the live root per batch.
   class read_session
   {
     public:
      cursor create_cursor(uint32_t root_index);

     private:
      friend class database;
      explicit read_session(database* db);
      database*          _db;
      start_ref::read_session _rs;
   };

   class write_session;

   class transaction
   {
     public:
      // primary-root mutations/reads (forward to the primary handle's tree)
      void insert(key_view k, value_view v) { upsert(k, v); }
      void upsert(key_view k, value_view v);
      template <class T>
         requires storable<std::remove_cvref_t<T>>
      void upsert(key_view k, T&& v);
      any_value get_any(key_view k) const;
      int  remove(key_view k);
      // ── multi-root (psitri's model): open additional roots in STRICTLY
      // ASCENDING index order; all roots commit/abort together. set_root is a
      // pure ref-handle exchange (the working handle and the owned tree just
      // swap the ref each already holds — no retain/release).
      tree_handle primary();
      tree_handle open_root(uint32_t root_index);
      void commit();
      void abort() noexcept;
      ~transaction() { abort(); }
      transaction(transaction&& o) noexcept
          : _ws(o._ws), _tx(std::move(o._tx)), _roots(std::move(o._roots)), _done(o._done)
      {
         o._done = true;  // moved-from: nothing to roll back
      }
      transaction(const transaction&)            = delete;
      transaction& operator=(const transaction&) = delete;

      template <class F>
      bool get(key_view k, F&& fn) const;
      template <class S>
      std::optional<S> get(key_view k) const;

     private:
      friend class write_session;
      friend class tree_handle;
      transaction(write_session* ws, uint32_t root_index, start_ref::transaction&& tx);
      struct root_state
      {
         uint32_t                     index;
         root_ptr                     start_root;   // committed root at txn start
         size_t                       start_count;
         std::unique_lock<std::mutex> lock;         // per-root writer lock (ordered)
      };
      engine_t&   tree_(uint32_t root_index);       // working tree for a held root
      write_session*          _ws;
      start_ref::transaction  _tx;
      std::vector<root_state> _roots;  // [0] = primary; ascending order
      bool                    _done = false;
   };

   // A tree IS a snapshot (adb): an owned, FROZEN handle holding ONE refcount
   // on its root (under node-COW that ref keeps the version readable across
   // later commits). The single owned-tree type: get_root returns it, set_root
   // exchanges it, snapshot_cursor reads it. Move-only; default = empty.
   class tree
   {
     public:
      tree() = default;
      tree(tree&& o) noexcept : _db(o._db), _root(o._root), _count(o._count)
      {
         o._db = nullptr, o._root = root_ptr::null();
      }
      tree& operator=(tree&& o) noexcept
      {
         if (this != &o)
         {
            release_();
            _db = o._db, _root = o._root, _count = o._count;
            o._db = nullptr, o._root = root_ptr::null();
         }
         return *this;
      }
      tree(const tree&)            = delete;
      tree& operator=(const tree&) = delete;
      ~tree() { release_(); }
      cursor snapshot_cursor() const;  // out-of-line (needs database)
      template <class S = std::string>
      std::optional<S> get(key_view k) const;
      explicit operator bool() const noexcept { return _db != nullptr; }

     private:
      friend class database;
      friend class write_session;
      friend class transaction;
      friend class tree_handle;
      tree(database* db, root_ptr held, size_t count) : _db(db), _root(held), _count(count) {}
      void      release_();  // cow_release the held ref (out-of-line)
      database* _db    = nullptr;
      root_ptr  _root  = root_ptr::null();
      size_t    _count = 0;
   };

   // Targets one root inside a multi-root transaction (adb root_handle).
   class tree_handle
   {
     public:
      void insert(key_view k, value_view v) { upsert(k, v); }
      void upsert(key_view k, value_view v);
      int  remove(key_view k);
      template <class S = std::string>
      std::optional<S> get(key_view k) const;
      tree set_root(tree t);  // THE exchange: assign, return the displaced tree

     private:
      friend class transaction;
      tree_handle(transaction* tx, uint32_t root_index) : _tx(tx), _ri(root_index) {}
      transaction* _tx;
      uint32_t     _ri;
   };

   class write_session
   {
     public:
      transaction start_transaction(uint32_t root_index);
      tree        get_root(uint32_t root_index);  // owned frozen snapshot

      // Drop every working tree's ref the COW way (nodes shared with a live
      // slot / reader pin survive); hkv's unconditional teardown must not see
      // them.
      ~write_session()
      {
         _t.cow_release(_t.root_handle());
         _t.adopt_root(root_ptr::null());
         for (auto& a : _aux)
            if (a)
            {
               a->cow_release(a->root_handle());
               a->adopt_root(root_ptr::null());
            }
      }

      engine_t& tree_dbg() noexcept { return _t; }  // debug/tools only

     private:
      friend class database;
      friend class transaction;
      friend class tree;
      friend class tree_handle;
      explicit write_session(database* db);
      // working tree for root i: root 0 is THE hot member; roots >= 1 are lazy
      // (first open adopts the committed root and takes the working's ref).
      engine_t& tree_for(uint32_t i);

      database*          _db;
      start_ref::session _ses;
      engine_t           _t;  // root 0's working tree (continues across txns)
      std::array<std::optional<engine_t>, kMaxRoots> _aux;  // roots >= 1, lazy
   };

   class database
   {
     public:
      static std::shared_ptr<database> create(std::filesystem::path dir, runtime_config cfg = {})
      {
         return std::shared_ptr<database>(new database(std::move(dir), cfg));
      }
      static std::shared_ptr<database> open(std::filesystem::path dir, runtime_config cfg = {})
      {
         return create(std::move(dir), cfg);
      }
      // Unified open semantics (kv_engine concept construction clause).
      static std::shared_ptr<database> open(std::filesystem::path dir,
                                            sal::open_mode        mode,
                                            runtime_config        cfg = {})
      {
         const bool exists = std::filesystem::exists(dir / "data.part");
         switch (mode)
         {
            case sal::open_mode::create_only:
               if (exists)
                  throw std::runtime_error("database already exists: " + dir.generic_string());
               break;
            case sal::open_mode::open_existing:
               if (!exists)
                  throw std::runtime_error("database does not exist: " + dir.generic_string());
               break;
            case sal::open_mode::read_only:
               throw std::runtime_error("part: read-only opens not implemented yet");
            case sal::open_mode::create_or_open:
               break;
         }
         return create(std::move(dir), cfg);
      }

      std::shared_ptr<write_session> start_write_session()
      {
         return std::shared_ptr<write_session>(new write_session(this));
      }
      std::shared_ptr<read_session> start_read_session()
      {
         return std::shared_ptr<read_session>(new read_session(this));
      }

      void sync(sal::sync_type = sal::sync_type::none) {}  // commits sync per cfg already

      ~database()
      {
         // clean shutdown (SPEC §19): ring drained, releaser stopped and
         // JOINED before any member teardown; the rest runs foreground/exact
         flush_releases();
         _rel_stop.store(true, std::memory_order_release);
         _rel_cv.notify_one();
         if (_rel_thread.joinable())
            _rel_thread.join();
         _arena.set_defer(nullptr, nullptr);  // remaining releases are inline
         // No readers remain by contract: force-drain retired roots, then the
         // live root's slot reference.
         for (uint64_t raw : _retired)
         {
            root_ptr       p;
            const uint32_t r32 = uint32_t(raw);
            std::memcpy(p.b, &r32, 4);
            engine_t v{part_alloc<bval>(&_arena)};
            v.allocator_mut().vhooks = &_vh;
            v.cow_release(p);
         }
         for (uint32_t i = 0; i < kMaxRoots; ++i)
            if (!_live_root[i].is_null())
            {
               engine_t v{part_alloc<bval>(&_arena)};
               v.allocator_mut().vhooks = &_vh;
               v.cow_release(_live_root[i]);
            }
      }

      start_ref::arena& store() noexcept { return _arena; }

      // ── typed-value registry (locked adb API) ───────────────────────────
      // Registered ids dispatch retain/release from the tree's value-
      // lifecycle hooks — the writer AND the background releaser fire through
      // here (hooks must be thread-safe and non-blocking). Register types
      // BEFORE storing values of T; id 0 is the raw-bytes builtin.
      template <class... Ts>
      void register_value_types()
      {
         (reg_type_<Ts>(), ...);
      }
      // SPEC §19: wait until the background releaser has emptied its ring
      // (drain gates and shutdown ordering call this)
      void flush_releases() noexcept
      {
         while (_rel_tail.load(std::memory_order_acquire) !=
                _rel_head.load(std::memory_order_acquire))
            std::this_thread::yield();
      }
      // debug/tools only: raw live root 0 (no pin — caller must hold off writers)
      root_ptr     live_root_dbg() const noexcept { return _live_root[0]; }

      // An owned, FROZEN snapshot of root i: retain the committed root under
      // the publish mutex (a root can't be released between load and retain).
      // The owned tree holds that ref; COW keeps it readable across commits.
      tree get_root_owned(uint32_t idx)
      {
         std::lock_guard g(_root_mx);
         const root_ptr  r = _live_root[idx];
         if (!r.is_null())
            _arena.add_ref_or_throw(uint32_t(r.raw() >> 4));  // the snapshot's ref
         return tree(this, r, 0);
      }
      // debug/back-compat alias (root 0 snapshot cursor over an owned tree)
      cursor pin_snapshot() { return get_root_owned(0).snapshot_cursor(); }

     private:
      friend class write_session;
      friend class read_session;
      friend class transaction;
      friend class tree;
      friend class tree_handle;

      database(std::filesystem::path dir, runtime_config cfg)
          : _arena(make_cfg_(std::move(dir), cfg))
      {
         for (auto& r : _live_root)
            r = root_ptr::null();
         for (auto& r : _live_raw)
            r.store(0, std::memory_order_relaxed);
         // background releaser (SPEC §19): the writer pushes release-to-zero
         // subtree roots; this thread runs the cascades; a full ring makes the
         // writer release inline (graceful degradation to the old behavior).
         // OPTIONAL: without the thread the hook is never registered and the
         // queue is permanently "blocked" — everything runs synchronously.
         if (cfg.background_release)
         {
            _arena.set_defer(&defer_cb_, this);
            _rel_thread = std::thread([this] { rel_loop_(); });
         }
      }
      static bool defer_cb_(void* ctx, uint32_t raw) noexcept
      {
         return static_cast<database*>(ctx)->rel_push_(raw);
      }
      bool rel_push_(uint32_t raw) noexcept
      {
         const uint32_t h = _rel_head.load(std::memory_order_relaxed);
         const uint32_t t = _rel_tail.load(std::memory_order_acquire);
         if (h - t >= kRelRing)
            return false;  // full: the caller releases in the foreground
         _rel[h % kRelRing].store(raw, std::memory_order_relaxed);
         _rel_head.store(h + 1, std::memory_order_release);
         if (h == t)
            _rel_cv.notify_one();  // was empty: wake the releaser
         return true;
      }
      void rel_loop_() noexcept
      {
         engine_t v{part_alloc<bval>(&_arena, /*defer=*/false)};  // inline worker
         v.allocator_mut().vhooks = &_vh;  // typed values die in releaser cascades
         for (;;)
         {
            const uint32_t t = _rel_tail.load(std::memory_order_relaxed);
            if (t == _rel_head.load(std::memory_order_acquire))
            {
               if (_rel_stop.load(std::memory_order_acquire))
                  return;
               std::unique_lock lk(_rel_mx);
               _rel_cv.wait_for(lk, std::chrono::milliseconds(2));
               continue;
            }
            const uint32_t raw = uint32_t(_rel[t % kRelRing].load(std::memory_order_relaxed));
            root_ptr       p;
            std::memcpy(p.b, &raw, 4);
            v.cow_release(p);  // non-deferring allocator: full cascade here
            _rel_tail.store(t + 1, std::memory_order_release);
         }
      }
      static start_ref::arena_config make_cfg_(std::filesystem::path dir, runtime_config cfg)
      {
         start_ref::arena_config c;
         c.dir          = std::move(dir);
         c.sync_mode    = cfg.sync_mode;
         c.max_segments = cfg.max_segments;
         // COW churn frees whole generations but long-lived survivors pepper
         // old segments; demanding 50%-in-8-contig before reuse lets the file
         // grow without bound. 1/8 keeps the can-this-segment-serve-allocs
         // guarantee while reclaiming aggressively.
         c.reuse_threshold = 0.125;
         return c;
      }

      // Retired-root release is DEFERRED to the writer: a root leaves the
      // queue only when no reader slot publishes it (sal's reader paradigm —
      // readers never touch refcounts or locks).
      void drain_retired_()  // caller holds _root_mx
      {
         const bool tm = getenv("PART_DRAIN_TIME") != nullptr;
         const auto t0 = tm ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
         for (size_t i = 0; i < _retired.size();)
         {
            if (_arena.root_published(_retired[i]))
            {
               ++i;
               continue;
            }
            root_ptr p;
            const uint32_t r32 = uint32_t(_retired[i]);
            std::memcpy(p.b, &r32, 4);
            engine_t v{part_alloc<bval>(&_arena, /*defer=*/true)};
            v.allocator_mut().vhooks = &_vh;
            v.cow_release(p);  // big generations route to the releaser
            _retired[i] = _retired.back();
            _retired.pop_back();
            ++_drain_roots;
         }
         if (tm)
            _drain_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
      }

     public:
      uint64_t drain_ns_dbg() const noexcept { return _drain_ns; }
      uint64_t drain_roots_dbg() const noexcept { return _drain_roots; }

     private:
      uint64_t _drain_ns    = 0;
      uint64_t _drain_roots = 0;

      private:
      static constexpr uint32_t kRelRing = 4096;  // big cascades shard into many subtree chunks
      std::atomic<uint32_t>     _rel[kRelRing] = {};
      std::atomic<uint32_t>     _rel_head{0}, _rel_tail{0};
      std::mutex                _rel_mx;
      std::condition_variable   _rel_cv;
      std::thread               _rel_thread;
      std::atomic<bool>         _rel_stop{false};

      start_ref::arena      _arena;
      std::mutex            _root_mx;                  // serializes commit publication
      std::mutex            _root_locks[kMaxRoots];    // per-root writer locks (ordered)
      root_ptr              _live_root[kMaxRoots];     // committed root per slot
      std::atomic<uint64_t> _live_raw[kMaxRoots];      // raw per slot (reader publish-validate)
      std::vector<uint64_t> _retired;

      // ── typed-value dispatch (slot envelope -> registered hooks) ────────
      struct type_entry
      {
         void (*retain)(void*, const char*, size_t)  = nullptr;
         void (*release)(void*, const char*, size_t) = nullptr;
      };
      std::vector<type_entry> _types = std::vector<type_entry>(65536);
      template <class T>
      void reg_type_()
      {
         static_assert(storable<T>);
         static_assert(T::type_id != kRawType, "type_id 0 is the raw-bytes builtin");
         _types[T::type_id] = {&vh_retain_t_<T>, &vh_release_t_<T>};
      }
      template <class T>
      static void vh_retain_t_(void* db, const char* p, size_t n)
      {
         typename T::t_pod pod;
         assert(n == sizeof(pod));
         std::memcpy(&pod, p, sizeof(pod));
         T::t_pod::retain(db, pod);
      }
      template <class T>
      static void vh_release_t_(void* db, const char* p, size_t n)
      {
         typename T::t_pod pod;
         assert(n == sizeof(pod));
         std::memcpy(&pod, p, sizeof(pod));
         T::t_pod::release(db, pod);
      }
      void vh_fire_(const char* b, size_t n, bool retain) noexcept
      {
         if (n < 2)
            return;
         uint16_t t;
         std::memcpy(&t, b, 2);
         if (t == kRawType)
            return;
         const auto& e = _types[t];
         if (auto f = retain ? e.retain : e.release)
            f(this, b + 2, n - 2);
      }
      static void vh_copy_s_(const char* b, size_t n, void* u)
      {
         static_cast<database*>(u)->vh_fire_(b, n, true);
      }
      static void vh_release_s_(const char* b, size_t n, void* u)
      {
         static_cast<database*>(u)->vh_fire_(b, n, false);
      }
      value_hooks _vh{&vh_copy_s_, &vh_release_s_, this};
   };

   // ── bodies ────────────────────────────────────────────────────────────────
   inline write_session::write_session(database* db)
       : _db(db), _ses(db->_arena.start_session()), _t(part_alloc<bval>(&_ses))
   {
      _t.allocator_mut().vhooks = &db->_vh;
   }

   inline read_session::read_session(database* db)
       : _db(db), _rs(db->_arena.start_read_session())
   {
   }

   inline cursor read_session::create_cursor(uint32_t root_index)
   {
      auto& slot = _db->_arena.reader_slot(_rs.slot());
      for (;;)
      {
         const uint64_t r = _db->_live_raw[root_index].load(std::memory_order_seq_cst);
         slot.store(r, std::memory_order_seq_cst);  // publish FIRST...
         if (_db->_live_raw[root_index].load(std::memory_order_seq_cst) == r)
         {  // ...then validate: r was live after our publish, so the writer's
            // drain (which runs after replacing the root) must see our slot.
            root_ptr p = root_ptr::null();
            if (r)
            {
               const uint32_t r32 = uint32_t(r);
               std::memcpy(p.b, &r32, 4);
            }
            return cursor(&_db->_arena, p, &slot);
         }
      }
   }

   inline cursor tree::snapshot_cursor() const
   {
      if (!_db)
         return cursor(nullptr, root_ptr::null(), nullptr);
      // borrow the held ref for the cursor: retain again so the cursor's
      // write-side dtor release balances; the tree keeps its own ref.
      if (!_root.is_null())
         _db->_arena.add_ref_or_throw(uint32_t(_root.raw() >> 4));
      return cursor(&_db->_arena, _root, nullptr, &_db->_vh);
   }
   inline void tree::release_()
   {
      if (!_db || _root.is_null())
         return;
      engine_t v{part_alloc<bval>(&_db->_arena)};
      v.allocator_mut().vhooks = &_db->_vh;
      v.cow_release(_root);  // drop THIS handle's ref (cascades only at zero)
      _root = root_ptr::null();
      _db   = nullptr;
   }
   template <class S>
   inline std::optional<S> tree::get(key_view k) const
   {
      if (!_db)
         return std::nullopt;
      return snapshot_cursor().template get<S>(k);
   }

   inline void transaction::upsert(key_view k, value_view v)
   {
      const value_view env = env_raw_(v);  // [0][bytes]
      if constexpr (engine_t::cb_create)
         _ws->tree_for(_roots[0].index).pg_upsert(k, env);  // bytes straight to the page
      else
         _ws->tree_for(_roots[0].index).cow_insert(k, bval{env});  // view in; the leaf copies once
   }
   template <class T>
      requires storable<std::remove_cvref_t<T>>
   inline void transaction::upsert(key_view k, T&& v)
   {
      using U                     = std::remove_cvref_t<T>;
      const typename U::t_pod pod = std::move(v).take_pod();  // custody -> db
      char                    buf[2 + sizeof(pod)];
      const uint16_t          id = U::type_id;
      std::memcpy(buf, &id, 2);
      std::memcpy(buf + 2, &pod, sizeof(pod));
      const value_view env(buf, sizeof(buf));
      if constexpr (engine_t::cb_create)
         _ws->tree_for(_roots[0].index).pg_upsert(k, env);
      else
         _ws->tree_for(_roots[0].index).cow_insert(k, bval{env});
   }
   inline any_value transaction::get_any(key_view k) const
   {
      if constexpr (engine_t::cb_create)
      {
         std::string_view out;
         if (!_ws->tree_for(_roots[0].index).pg_find(k, out))
            return {};
         return any_value{out.data(), out.size()};
      }
      bval out;
      if (!_ws->tree_for(_roots[0].index).find(k, out))
         return {};
      return any_value{out.v.data(), out.v.size()};  // slot view, txn-pinned
   }
   inline int transaction::remove(key_view k)
   {
      if constexpr (engine_t::cb_create)
         return _ws->tree_for(_roots[0].index).pg_remove(k) ? 0 : -1;
      else
         return _ws->tree_for(_roots[0].index).cow_remove(k) ? 0 : -1;
   }
   template <class F>
   inline bool transaction::get(key_view k, F&& fn) const
   {
      if constexpr (engine_t::cb_create)
      {
         std::string_view out;
         if (!_ws->tree_for(_roots[0].index).pg_find(k, out))
            return false;
         if (slot_type(out) != kRawType)
            throw invalid_type{slot_type(out), kRawType};
         fn(slot_payload(out));
         return true;
      }
      bval out;
      if (!_ws->tree_for(_roots[0].index).find(k, out))
         return false;
      if (slot_type(out.v) != kRawType)
         throw invalid_type{slot_type(out.v), kRawType};
      fn(slot_payload(out.v));
      return true;
   }
   template <class S>
   inline std::optional<S> transaction::get(key_view k) const
   {
      if constexpr (engine_t::cb_create)
      {
         std::string_view out;
         if (!_ws->tree_for(_roots[0].index).pg_find(k, out))
            return std::nullopt;
         return slot_decode<S>(out);
      }
      bval out;
      if (!_ws->tree_for(_roots[0].index).find(k, out))
         return std::nullopt;
      return slot_decode<S>(out.v);
   }

   inline transaction::transaction(write_session* ws, uint32_t root_index,
                                   start_ref::transaction&& tx)
       : _ws(ws), _tx(std::move(tx))
   {
      auto& t = ws->tree_for(root_index);
      _roots.push_back({root_index, t.root_handle(), t.size_hint(),
                        std::unique_lock<std::mutex>(ws->_db->_root_locks[root_index])});
   }
   inline engine_t& transaction::tree_(uint32_t root_index) { return _ws->tree_for(root_index); }
   inline tree_handle transaction::primary() { return tree_handle(this, _roots[0].index); }
   inline tree_handle transaction::open_root(uint32_t root_index)
   {
      assert(root_index < kMaxRoots && !_roots.empty());
      assert(root_index > _roots.back().index &&
             "roots must be opened in ascending index order (deadlock freedom)");
      auto& t = _ws->tree_for(root_index);
      _roots.push_back({root_index, t.root_handle(), t.size_hint(),
                        std::unique_lock<std::mutex>(_ws->_db->_root_locks[root_index])});
      return tree_handle(this, root_index);
   }
   inline void transaction::abort() noexcept
   {
      if (_done)
         return;
      _done = true;
      // roll EVERY held root back to its txn-start root; release each diverged
      // generation (the working's ref on it). set_root displaced trees are
      // independent owned handles — the caller still holds them.
      for (auto it = _roots.rbegin(); it != _roots.rend(); ++it)
      {
         engine_t&      t        = _ws->tree_for(it->index);
         const root_ptr diverged = t.root_handle();
         if (diverged.raw() != it->start_root.raw())
         {
            t.cow_retain(it->start_root);  // the working re-takes the start ref
            t.adopt_root(it->start_root, it->start_count);
            t.cow_release(diverged);  // drop the working's diverged ref
         }
         it->lock.unlock();  // reverse order
      }
      _roots.clear();
   }
   inline void transaction::commit()
   {
      _done = true;
      _tx.commit();  // data msync + floor advance (layer 1)
      database*       db = _ws->_db;
      std::lock_guard g(db->_root_mx);
      // publish every held root: cow_retain the working root (the _live_root
      // ref), swap the slot, retire the old. UNCONDITIONAL (line-recycle ABA).
      for (auto& r : _roots)
      {
         engine_t&      t   = _ws->tree_for(r.index);
         const root_ptr nr  = t.root_handle();
         const root_ptr old = db->_live_root[r.index];
         if (!nr.is_null() && !t.cow_retain(nr))
            throw std::runtime_error("start_ref::kv: root retain refused");
         db->_live_root[r.index] = nr;
         db->_live_raw[r.index].store(nr.is_null() ? 0 : nr.raw(), std::memory_order_seq_cst);
         db->_arena.set_root(r.index, nr.is_null() ? 0 : nr.raw());
         if (!old.is_null())
            db->_retired.push_back(old.raw());
      }
      db->drain_retired_();
      for (auto it = _roots.rbegin(); it != _roots.rend(); ++it)
         it->lock.unlock();
      _roots.clear();
   }

   inline transaction write_session::start_transaction(uint32_t root_index)
   {
      return transaction(this, root_index, _ses.start_transaction(0));
   }
   inline tree write_session::get_root(uint32_t root_index)
   {
      return _db->get_root_owned(root_index);
   }
   // Working tree for root i: root 0 is _t; roots >= 1 materialize lazily,
   // adopting the committed root and taking the working's ref on it.
   inline engine_t& write_session::tree_for(uint32_t i)
   {
      if (i == 0)
         return _t;
      if (!_aux[i])
      {
         _aux[i].emplace(part_alloc<bval>(&_ses));
         _aux[i]->allocator_mut().vhooks = &_db->_vh;
         const root_ptr r = _db->_live_root[i];
         if (!r.is_null())
         {
            _aux[i]->cow_retain(r);  // the working takes its ref
            _aux[i]->adopt_root(r, 0);
         }
      }
      return *_aux[i];
   }

   inline void tree_handle::upsert(key_view k, value_view v)
   {
      _tx->tree_(_ri).cow_insert(k, bval{env_raw_(v)});
   }
   inline int tree_handle::remove(key_view k)
   {
      return _tx->tree_(_ri).cow_remove(k) ? 0 : -1;
   }
   template <class S>
   inline std::optional<S> tree_handle::get(key_view k) const
   {
      bval out;
      if (!_tx->tree_(_ri).find(k, out))
         return std::nullopt;
      return slot_decode<S>(out.v);
   }
   // THE exchange (pure ref transfer): the working handle and the owned tree
   // swap the ref each already holds — no retain/release. The displaced root
   // becomes the returned owned tree's ref; the incoming root becomes the
   // working's ref.
   inline tree tree_handle::set_root(tree t)
   {
      engine_t&      w       = _tx->tree_(_ri);
      const root_ptr old     = w.root_handle();
      const size_t   old_cnt = w.size_hint();
      w.adopt_root(t._root, t._count);  // repoint working to the incoming root
      t._db   = nullptr;  // the working now owns the incoming's ref
      t._root = root_ptr::null();
      return tree(_tx->_ws->_db, old, old_cnt);  // owned tree takes the working's old ref
   }

   // The engine as a kv_engine aggregate (delta_art's W/B type param).
   struct engine
   {
      using database      = start_ref::kv::database;
      using write_session = start_ref::kv::write_session;
      using read_session  = start_ref::kv::read_session;
      using transaction   = start_ref::kv::transaction;
      using cursor        = start_ref::kv::cursor;
      using config        = start_ref::kv::runtime_config;
   };
}  // namespace start_ref::kv

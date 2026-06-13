// start_ver::kv — part's NATIVE psitri-shaped public API (the same database →
// write_session → transaction → cursor structure psitri and heart::kv expose),
// so one templated test/bench body drives all engines through identical calls.
//
// Engine: the verbatim-hkv tree on the vpart arena, with STAMP LIVENESS
// (docs/unified_tier_design.md) instead of refcounts:
//   • the version IS the transaction — one boundary published per commit;
//   • the in-place fast path is within-transaction only (stamp == txn), so
//     the last committed root is stable for readers with ZERO coordination;
//   • superseded nodes ride per-transaction clone-logs (shallow = clone
//     husks, deep = dropped subtrees); commit publishes the chain under the
//     new version; abort frees the create-log and drops the chain unpublished;
//   • readers pin a VERSION by publishing (version<<32|root) in their reader
//     slot — they never mint version roots and never wait on the writer;
//   • a background drain frees every chain whose death version is <= the
//     floor (the oldest pinned version; everything, when nothing is pinned).
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

#include <start_ver/hkv/tree.hpp>
#include <start_ver/tree_alloc.hpp>

namespace start_ver::kv
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
      // When false, no drain thread is spawned: every committed chain drains
      // synchronously on the writer after commit (identical semantics).
      bool           background_release      = true;
      // Segment-reuse bar: only segments at least this empty are reclaimed
      // for the write window; emptier = more bump-like allocation. Dan:
      // standalone expects 50%; the DELTA role (tiered) overrides to 80%
      // (flush+retire drains segments nearly clean there).
      double         reuse_threshold         = 0.50;
   };

   // ELEGANT value mode: the leaf IS the value (raw bytes sized to the data);
   // reads hand back views.
   using bval = start_ver::hkv::bytes_value;
   // DEFAULT = the c2/c4/c8 ladder, matching part's form exactly (Dan: the
   // only part/vpart diff is supposed to be refcount vs version header; the
   // census proved straight-widen was +19 MB of node_full on rand 1M, ~the
   // whole "extra space" — the stamps themselves cost zero extra lines).
   // VPART_STRAIGHT_WIDEN=1 keeps the form experiment buildable: widen goes
   // setlist -> node_full directly (fastest dispatch, fatter sparse wides);
   // the placement/mover design owns the per-node choice long-term.
#if defined(VPART_STRAIGHT_WIDEN) && VPART_STRAIGHT_WIDEN
   using engine_t = start_ver::hkv::kv_tree<std::string_view, bval, start_ver::hkv::mode::none, part_alloc<bval>>;
#else
   using engine_t =
       start_ver::hkv::kv_tree<std::string_view, bval, start_ver::hkv::mode::dense_tiers, part_alloc<bval>>;
#endif
   using root_ptr = start_ver::detail::line_ptr;

   // Multi-root layout: arena slots [0, kMaxRoots) hold root raws; slot
   // kMaxRoots holds the (global) committed version. One version counter
   // spans all roots: version == transaction, whichever roots it touched.
   inline constexpr uint32_t kMaxRoots = 8;

   class database;
   class cursor;  // tree_root::snapshot_cursor() returns it (defined below)

   // ── typed values (the LOCKED API, audit §9-§12) ───────────────────────
   // EVERY stored slot is [u16 LE typenum][payload]. Typenum 0 = raw bytes
   // (the null-hook builtin: strings/views); registered ids >= 1 dispatch
   // retain/release through the database registry from the tree's value-
   // lifecycle hooks — overwrite, remove, abort, COW clone, supersession
   // drain, reaper: GC threads included. Reads are THREE-OUTCOME everywhere:
   // found / not-found / invalid_type (a wrong-pod read is a type error,
   // never a fake absence, never a reinterpret).
   inline constexpr uint16_t kRawType = 0;

   struct invalid_type : std::runtime_error
   {
      invalid_type(uint16_t stored, uint16_t asked)
          : std::runtime_error("vpart: value is type " + std::to_string(stored) +
                               ", read as type " + std::to_string(asked))
      {
      }
   };

   // part::shape::storable_type, restated locally (vpart doesn't include the
   // concept header; the conformance suite asserts the shapes line up).
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
   // Raw-bytes envelope, built in thread_local arena scratch (grows once).
   inline value_view env_raw_(value_view v)
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
            throw invalid_type{t, S::type_id};  // truncated/foreign pod bytes
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

   // any_value — the dynamic, lock-bundling read form (§10/§12): a pointer to
   // the value SLOT in the tree, no copy. Inside a write transaction the
   // transaction itself is the pin; the view is invalidated by a later write
   // to the same key in that transaction. as<T>() retains (the only civilian
   // exit); bytes() on a typed slot is invalid_type (a typed pod exposed raw
   // would be a forgeable handle); raw_pod() is the expert/merger identity
   // path, balanced by the typed STORE side firing the retain hook.
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
      std::string_view raw_pod() const noexcept { return _s ? std::string_view{_s + 2, _n - 2} : std::string_view{}; }
      explicit operator bool() const noexcept { return _s != nullptr; }

     private:
      friend class transaction;
      any_value(const char* s, size_t n) : _s(s), _n(n) {}
      const char* _s = nullptr;
      size_t      _n = 0;  // the FULL slot (envelope included), in the tree
   };

   // Refcounted handle to a FROZEN version root (the smart_ptr/"tree_root"
   // equivalent): copy freely across threads; the tree stays alive while any
   // handle (merger, reader warm legs) exists; the LAST drop enqueues the
   // root for vpart's own REAPER thread — garbage collection is built into
   // vpart, never a caller responsibility.
   class tree_root
   {
     public:
      tree_root() = default;
      uint32_t version() const noexcept { return _c ? _c->vf : 0; }
      uint32_t root_raw() const noexcept { return _c ? _c->raw : 0; }
      uint64_t key_count() const noexcept { return _c ? _c->count : 0; }
      explicit operator bool() const noexcept { return bool(_c); }
      void     reset() noexcept { _c.reset(); }
      // A tree IS a snapshot: read it through a cursor pinned at its frozen
      // (version, root). Defined out-of-line (needs database + cursor).
      cursor snapshot_cursor() const;
      template <class S = std::string>
      std::optional<S> get(key_view k) const;

     private:
      friend class database;
      friend class write_session;
      friend class transaction;
      friend class tree_handle;
      struct ctrl
      {
         database* db;
         uint32_t  vf, raw;
         uint32_t  idx = UINT32_MAX;  // root index this snapshot is OF (per-root floor)
         uint64_t  count = 0;     // keys in the tree (size() continuity)
         bool      armed = true;  // false until the displacing txn COMMITS
         // snapshot handles: a reader-slot pin holds the drain floor at vf
         // for the handle's lifetime (the nodes stay readable); DISARMED —
         // dropping a snapshot of the live lineage must never reap it. The
         // published value must be ZEROED on drop (floor_ scans values, not
         // slot-ownership bits) — exactly the cursor teardown protocol.
         std::optional<start_ver::read_session> pin;
         std::atomic<uint64_t>*             pin_slot = nullptr;
         ~ctrl();  // last handle dropped (and armed) -> reaper queue
      };
      explicit tree_root(std::shared_ptr<ctrl> c) : _c(std::move(c)) {}
      std::shared_ptr<ctrl> _c;
   };

   // Snapshot cursor: a version pin (reader slot publishing version<<32|root)
   // + a non-owning view tree + an hkv iterator. The dtor clears the slot;
   // reclamation is the drain thread's job (floor rule) — readers never free.
   class cursor
   {
     public:
      cursor(cursor&& o) noexcept
          : _hp(o._hp), _rs(std::move(o._rs)), _root(o._root), _view(std::move(o._view)),
            _it(_view.end()), _rend(o._rend)
      {
         o._hp = nullptr;
      }
      cursor(const cursor&) = delete;
      ~cursor()
      {
         _view.adopt_root(root_ptr::null());  // the view owns nothing
         if (_hp)
            _hp->store(0, std::memory_order_seq_cst);  // drop the version pin
      }

      bool seek_begin()
      {
         _rend = false;
         _it   = _view.begin();
         return !is_end();
      }
      bool seek_last()
      {
         _rend = false;
         _it   = _view.last();
         return !is_end();
      }
      bool lower_bound(key_view k)
      {
         _rend = false;
         _it   = _view.lower_bound(k);
         return !is_end();
      }
      bool upper_bound(key_view k)
      {
         _rend = false;
         _it   = _view.lower_bound(k);
         if (_it != _view.end() && (*_it).first == k)
            ++_it;
         return !is_end();
      }
      bool find(key_view k)
      {
         _rend = false;
         _it   = _view.find(k);
         return !is_end();
      }
      bool next()
      {
         if (_rend)
            return seek_begin();  // psitri convention: next from rend = first key
         ++_it;
         return !is_end();
      }
      bool prev()
      {
         if (_rend)
            return false;
         if (_it == _view.end())
            return seek_last();  // prev from end = last key
         --_it;
         if (_it == _view.end())
         {  // retreated past the first key
            _rend = true;
            return false;
         }
         return true;
      }
      bool is_end() const { return !_rend && _it == _view.end(); }
      bool is_rend() const { return _rend; }
      bool is_subtree() const { return false; }  // vpart has no subtree values

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
      // typenum of the current slot (locked API: cursor type discovery)
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
         bval out;
         if (!_view.find(k, out))
            return std::nullopt;
         return slot_decode<S>(out.v);
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
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
      cursor(start_ver::arena* a, root_ptr root, std::atomic<uint64_t>* hp,
             std::optional<start_ver::read_session> rs = std::nullopt)
          : _hp(hp), _rs(std::move(rs)), _root(root), _view(part_alloc<bval>(a)), _it(_view.end())
      {
         _view.adopt_root(_root);
      }

      std::atomic<uint64_t>*             _hp;  // the reader's published pin slot
      std::optional<start_ver::read_session> _rs;  // owned slot (writer-side snapshots)
      root_ptr                           _root;
      mutable engine_t                   _view;
      engine_t::const_iterator           _it;
      bool                               _rend = false;  // retreated past the first key
   };

   // A reader thread's handle: one start_ver::read_session slot for the lifetime of
   // the thread; cursors pin (publish) the latest committed version per batch.
   class read_session
   {
     public:
      cursor create_cursor(uint32_t root_index);

     private:
      friend class database;
      explicit read_session(database* db);
      database*           _db;
      start_ver::read_session _rs;
   };

   class write_session;

   class tree_handle;

   class transaction
   {
     public:
      void insert(key_view k, value_view v) { upsert(k, v); }
      void upsert(key_view k, value_view v);
      // typed store: take_pod() CONSUMES v — custody (one reference) moves
      // into the database; the slot's later death fires the release hook.
      template <class T>
         requires storable<std::remove_cvref_t<T>>
      void upsert(key_view k, T&& v);
      any_value get_any(key_view k) const;
      int  remove(key_view k);
      // ── multi-root (psitri's model; concept-locked) ─────────────────────
      // primary() targets the root this transaction started on; open_root(i)
      // opens an additional root — STRICTLY ASCENDING indices (deadlock
      // freedom by construction, asserted). All roots commit/abort together
      // (ordered publish, DESCENDING index: rotation's warm-before-hot).
      tree_handle primary();
      tree_handle open_root(uint32_t root_index);
      // Retire primitive (L5): remove ONLY if the key's last write-version is
      // <= vmax (i.e., it was already flushed at the merge floor). Keys
      // rewritten after the floor — and keys whose stamp is unknowable
      // (inline terminals) — are left alone. 0 = removed, -1 = kept/absent.
      int remove_if_older(key_view k, uint32_t vmax);
      // L5 retirement, Dan's stamp-bounded subtree algorithm (3 phases; A and
      // C run in (separate) write transactions under the caller's writer
      // serialization; B is lock-free — the detached list is private).
      size_t   retire_detach(uint32_t floor);
      uint64_t retire_count() const;
      void     retire_release(uint64_t counted);
      void commit();
      // CONTRACT (kv-compat 4.7b): dropping or aborting an uncommitted
      // transaction ROLLS BACK — the session tree returns to the txn-start
      // root, every node the transaction created is freed (create-log), and
      // its supersession chain is dropped unpublished.
      void abort() noexcept;
      ~transaction() { abort(); }
      transaction(transaction&& o) noexcept
          : _ws(o._ws), _tx(std::move(o._tx)), _displaced(std::move(o._displaced)),
            _roots(std::move(o._roots)), _done(o._done)
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
      transaction(write_session* ws, uint32_t root_index, start_ver::transaction&& tx);
      struct root_state
      {
         uint32_t                     index;
         root_ptr                     start_root;
         size_t                       start_count;
         std::unique_lock<std::mutex> lock;  // per-root writer lock (ordered)
      };
      engine_t&   tree_(uint32_t slot_i);  // tree for held-root slot i
      root_state* find_root_(uint32_t root_index);
      write_session*     _ws;
      start_ver::transaction _tx;
      // displaced trees ARM (reap-on-drop) only when this txn commits;
      // an abort leaves them disarmed — the rollback restores the slot.
      std::vector<std::shared_ptr<tree_root::ctrl>> _displaced;
      std::vector<root_state> _roots;  // [0] = primary; ascending order
      bool                    _done = false;
   };

   // Lightweight targeting scope (psitri's tree_handle): directs ops at one
   // root inside a multi-root transaction. Non-owning; no commit/abort.
   class tree_handle
   {
     public:
      void insert(key_view k, value_view v) { upsert(k, v); }
      void upsert(key_view k, value_view v);
      int  remove(key_view k);
      template <class F>
      bool get(key_view k, F&& fn) const;
      template <class S = std::string>
      std::optional<S> get(key_view k) const;
      uint64_t key_count() const;
      // THE exchange (Dan): assign a tree to this root, RETURN the displaced
      // tree. tree_root{} assigns empty. The returned handle arms (reaps on
      // last drop) only when the transaction commits; abort rolls back.
      tree_root set_root(tree_root t);

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
      // get_root(i) IS the snapshot primitive (a tree is a snapshot): an
      // owned, FROZEN tree_root of root i's committed state — a reader-slot
      // pin holds the drain floor at its version (the nodes stay readable)
      // and it is DISARMED (a view of the live lineage, never reaped from
      // here). Call get_root again to observe later state.
      tree_root get_root(uint32_t root_index);

      // vpart: the working trees' nodes are the committed state (or were
      // logged when superseded) — teardown owns nothing to free.
      ~write_session()
      {
         _t.adopt_root(root_ptr::null());
         for (auto& t : _aux)
            if (t)
               t->adopt_root(root_ptr::null());
      }

      engine_t& tree_dbg() noexcept { return _t; }  // debug/tools only
      // Tree for root i: root 0 is THE hot member (zero added indirection on
      // the classic path); roots >= 1 materialize lazily, adopting the
      // committed root from the database boundary.
      engine_t& tree_for(uint32_t i);

     private:
      friend class database;
      friend class transaction;
      friend class tree_handle;
      explicit write_session(database* db);

      database*      _db;
      start_ver::session _ses;
      vlog_state     _vlog;  // per-transaction clone/create logs (ALL roots)
      engine_t       _t;     // root 0's working tree (the classic hot path)
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
               throw std::runtime_error("vpart: read-only opens not implemented yet");
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
         // clean shutdown: reaper first (eats any last dead roots), then the
         // drain, then the final inline sweeps.
         {
            std::lock_guard g(_dead_mx);
            _reaper_stop = true;
         }
         _dead_cv.notify_one();
         if (_reaper.joinable())
            _reaper.join();
         _drain_stop.store(true, std::memory_order_release);
         _drain_cv.notify_one();
         if (_drain_thread.joinable())
            _drain_thread.join();
         drain_();
         {  // flush ring-parked runs back to the freeset
            uint32_t l, n;
            while (_recycle.pop(l, n))
               _arena.release_run(l, n);
         }
         // The live tree is durable state — nothing to release at teardown.
      }

      start_ver::arena& store() noexcept { return _arena; }
      // debug/tuning: (used_lines, total_lines) across the arena — distinguishes
      // un-reclaimed garbage (used high) from reclaimed-but-not-reused file
      // high-water (free high). Exact only at quiescence.
      std::pair<uint64_t, uint64_t> occupancy_dbg() noexcept
      {
         uint64_t freeL = 0;
         const uint32_t ns = _arena.num_segments();
         for (uint32_t s = 0; s < ns; ++s)
            freeL += _arena.freeset_of(s).free_lines();
         const uint64_t total = uint64_t(ns) * start_ver::lines_per_segment;
         return {total - freeL, total};
      }
      // debug/tuning: (segments_grown, segments_reused) — grow high => claim_
      // can't find free segments when the writer needs one (reclaim lag or
      // claim_ logic); reuse high => recycling works.
      std::pair<uint64_t, uint64_t> seg_stats_dbg() const noexcept
      {
         return {_arena.stats().segments_grown.load(std::memory_order_relaxed),
                 _arena.stats().segments_reused.load(std::memory_order_relaxed)};
      }

      // ── typed-value registry (locked API §9) ────────────────────────────
      // Registered ids dispatch retain/release from the tree's value-
      // lifecycle hooks — the writer, the DRAIN and the REAPER threads all
      // fire through here (hooks must be thread-safe and non-blocking, the
      // sal type-ops contract). Register types BEFORE storing values of T;
      // id 0 is the raw-bytes builtin. The psio schema persistence (§14)
      // binds stored ids to these callbacks by canonical schema on load.
      template <class... Ts>
      void register_value_types()
      {
         (reg_type_<Ts>(), ...);
      }

      // wait until the drain ring is empty (tests / shutdown ordering)
      void flush_releases() noexcept
      {
         auto spill_empty = [&] {
            std::lock_guard<std::mutex> g(_reg_spill_mx);
            return _reg_spill.empty();
         };
         while (_reg_tail.load(std::memory_order_acquire) !=
                    _reg_head.load(std::memory_order_acquire) ||
                _drain_active.load(std::memory_order_acquire) ||  // a pass holds
                !spill_empty())                                   // gated in-flight
         {
            _drain_cv.notify_one();
            std::this_thread::yield();
         }
         // return ring-parked runs to the freeset (the gate counts free BITS;
         // ring entries are claimed-but-reusable). Caller is the consumer side.
         uint32_t l, n;
         while (_recycle.pop(l, n))
            _arena.release_run(l, n);
      }
      // debug/tools only: the raw live root (no pin — caller must hold off writers)
      root_ptr live_root_dbg() const noexcept
      {
         root_ptr       p    = root_ptr::null();
         const uint64_t live = _live_arr[0].load(std::memory_order_seq_cst);
         // raw 0 is a VALID root (line 0, tag value_ptr): "nothing published"
         // is version 0 — test the WHOLE packed word, never the root half.
         if (live != 0)
         {
            const uint32_t r32 = uint32_t(live);
            std::memcpy(p.b, &r32, 4);
         }
         return p;
      }

      // Pinned cursor for a tree_root snapshot at (version, root). version 0 &&
      // root 0 = the empty/unpublished tree (raw 0 is a valid root at v>0, so
      // the empty test is the pair == {0,0}); its cursor reads as empty.
      cursor snapshot_cursor_for(uint32_t version, uint32_t root_raw,
                                 uint32_t root_index = UINT32_MAX)
      {
         auto  rs   = _arena.start_read_session();
         auto& slot = _arena.reader_slot(rs.slot());
         // attribute the pin to its root BEFORE publishing the slot (per-root
         // floor: a warm-root pin must not gate hot-root garbage)
         _arena.reader_root_idx(rs.slot()).store(root_index, std::memory_order_relaxed);
         slot.store((uint64_t(version) << 32) | root_raw, std::memory_order_seq_cst);
         root_ptr p = root_ptr::null();
         if (!(version == 0 && root_raw == 0))
            std::memcpy(p.b, &root_raw, 4);
         return cursor(&_arena, p, &slot, std::move(rs));
      }
      // An owned cursor over the empty tree (default tree_root has no db).
      static cursor empty_cursor() { return cursor(nullptr, root_ptr::null(), nullptr); }

     private:
      friend class write_session;
      friend class read_session;
      friend class transaction;

      database(std::filesystem::path dir, runtime_config cfg)
          : _arena(make_cfg_(std::move(dir), cfg))
      {
         // resume the version counter past every stamp that could be durable,
         // and re-expose every durable root as its live boundary. Version is
         // global (slot kMaxRoots); roots live in slots [0, kMaxRoots).
         _committed = uint32_t(_arena.get_root(kMaxRoots));
         for (uint32_t i = 0; i < kMaxRoots; ++i)
         {
            const uint32_t r32 = uint32_t(_arena.get_root(i));
            _live_arr[i].store(
                _committed ? (uint64_t(_committed) << 32) | r32 : 0,
                std::memory_order_seq_cst);
         }
         _live.store(_live_arr[0].load(std::memory_order_seq_cst),
                     std::memory_order_seq_cst);
         if (cfg.background_release)
            _drain_thread = std::thread([this] { drain_loop_(); });
         _reaper = std::thread([this] { reaper_loop_(); });
      }

      uint32_t committed_() const noexcept { return _committed; }

     public:
      // Arena footprint (segments held x segment bytes): the observer signal
      // (Dan: the compactor watches FREE SPACE and flushes early — bytes
      // written is the wrong sensor; a big arena also poisons claim_ scans).
      uint64_t footprint_bytes() const noexcept
      {
         return uint64_t(_arena.num_segments()) * (uint64_t(start_ver::lines_per_segment) * 128u);
      }

     private:
      friend struct tree_root::ctrl;
      void reap_root_(uint32_t raw) noexcept
      {
         {
            std::lock_guard g(_dead_mx);
            _dead_roots.push_back(raw);
         }
         _reap_enq.fetch_add(1, std::memory_order_release);
         _dead_cv.notify_one();
      }
      void reaper_loop_() noexcept
      {
         engine_t v{part_alloc<bval>(&_arena, /*reclaim=*/true)};
         v.allocator_mut().vhooks = &_vh;  // typed values die in reap cascades too
         // NO ring: _recycle is SPSC with the DRAIN as its sole producer —
         // a second producer corrupts it (measured: allocation livelock).
         // Reaper frees go straight to the thread-safe freesets; the writer
         // reclaims them through claim_.
         std::unique_lock lk(_dead_mx);
         for (;;)
         {
            _dead_cv.wait(lk, [&] { return _reaper_stop || !_dead_roots.empty(); });
            if (_reaper_stop && _dead_roots.empty())
               return;
            std::vector<uint32_t> work;
            work.swap(_dead_roots);
            lk.unlock();
            for (uint32_t raw : work)
            {
               root_ptr wp = root_ptr::null();
               std::memcpy(wp.b, &raw, 4);
               if (!wp.is_null())
                  v.vdrain_entry(raw, 0x80000000u);  // DEEP: cascade the whole tree
               _reap_done.fetch_add(1, std::memory_order_release);
            }
            lk.lock();
         }
      }

     public:
      // tests/shutdown: wait until every enqueued dead root is fully reaped
      // (queue-empty alone races the cascade running outside the lock)
      void flush_reaper() noexcept
      {
         while (_reap_done.load(std::memory_order_acquire) !=
                _reap_enq.load(std::memory_order_acquire))
         {
            _dead_cv.notify_one();
            std::this_thread::yield();
         }
      }

     private:

      // floor = oldest pinned version (committed when nothing is pinned).
      // seq_cst slot publication (reader) vs seq_cst scan (drain) closes the
      // race exactly like part's root_published dance.
      // Per-root floor: the oldest version pinned by a reader of THIS root
      // (plus any "gates-all" pin, UINT32_MAX). A pin on a DIFFERENT specific
      // root does not gate this root — a warm-root (1) reader cannot see
      // hot-root (0) garbage, so it must not block its reclamation. Pass
      // root == UINT32_MAX for the conservative GLOBAL floor (multi-root
      // chains, shutdown).
      uint32_t floor_for_(uint32_t root) noexcept
      {
         uint32_t f = _committed;
         for (uint32_t i = 0; i < max_read_sessions; ++i)
         {
            const uint64_t s = _arena.reader_slot(i).load(std::memory_order_seq_cst);
            if (!s)
               continue;
            const uint32_t pidx = _arena.reader_root_idx(i).load(std::memory_order_relaxed);
            if (root != UINT32_MAX && pidx != root && pidx != UINT32_MAX)
               continue;  // pin on a different specific root: doesn't gate `root`
            const uint32_t v = uint32_t(s >> 32);  // pin at V sees state V:
            if (v < f)                             // deaths <= V stay eligible
               f = v;
         }
         return f;
      }
      uint32_t floor_() noexcept { return floor_for_(UINT32_MAX); }  // global

      // SPSC registry: writer pushes (version, chain head) at commit; the
      // drain owner (thread, or the writer inline when no thread) consumes.
      void reg_push_(uint32_t ver, uint32_t chain_head, uint32_t root)
      {
         if (chain_head == vlog_state::npos)
            return;
         for (;;)
         {
            const uint32_t h = _reg_head.load(std::memory_order_relaxed);
            const uint32_t t = _reg_tail.load(std::memory_order_acquire);
            if (h - t < kRegRing)
            {
               _reg[h % kRegRing] = {ver, chain_head, root};
               _reg_head.store(h + 1, std::memory_order_release);
               _drain_cv.notify_one();
               return;
            }
            // ring full: a pin is blocking the floor. NEVER spin here — the
            // writer may hold locks a pin-holder needs to UNPIN (measured
            // 3-thread deadlock: writer spins in commit, drain floor-gated,
            // merger waits on the writer lock). Spill unbounded instead.
            {
               std::lock_guard g(_reg_spill_mx);
               _reg_spill.push_back({ver, chain_head, root});
            }
            _drain_cv.notify_one();
            return;
         }
      }

      // Drain committed chains whose death <= the floor OF THEIR ROOT. Single
      // consumer. PER-ROOT: a chain tagged root R drains at floor_for_(R), so
      // a warm-root (1) pin held across a long flush no longer blocks the hot
      // root's (0) garbage. Because eligibility is no longer a version-ordered
      // prefix, the ring is consumed IN FULL each pass: eligible chains are
      // eaten, gated chains move to the (unordered) spill and are retried next
      // pass — no head-of-line blocking (a blocked root-1 clear chain can't
      // freeze the root-0 chains behind it).
      void drain_() noexcept
      {
         _drain_active.store(true, std::memory_order_release);
         struct done_guard
         {
            std::atomic<bool>& f;
            ~done_guard() { f.store(false, std::memory_order_release); }
         } dg{_drain_active};
         engine_t v{part_alloc<bval>(&_arena, /*reclaim=*/true)};
         v.allocator_mut().vring  = &_recycle;  // freed runs go back to the writer
         v.allocator_mut().vhooks = &_vh;       // superseded typed values die HERE
         auto eat_chain = [&](uint32_t head) noexcept {
            uint32_t line = head;
            while (line != vlog_state::npos)
            {
               const uint32_t* w = static_cast<const uint32_t*>(_arena.get(line));
               const uint32_t  prev = w[0], n = w[1];
               for (uint32_t i = 0; i < n; ++i)
                  v.vdrain_entry(w[2 + i * 2], w[3 + i * 2]);
               _arena.release_run(line, 1);  // the chain line itself
               line = prev;
            }
         };
         // per-root floor cache (computed once per pass); index kMaxRoots = global
         uint32_t fcache[kMaxRoots + 1];
         bool     fhave[kMaxRoots + 1] = {};
         auto     floor_of = [&](uint32_t root) noexcept -> uint32_t {
            const uint32_t slot = root < kMaxRoots ? root : kMaxRoots;
            if (!fhave[slot])
            {
               fcache[slot] = floor_for_(slot == kMaxRoots ? UINT32_MAX : slot);
               fhave[slot]  = true;
            }
            return fcache[slot];
         };
         auto eligible = [&](const reg_entry& e) noexcept {
            return e.ver <= floor_of(e.root);
         };
         // consume the whole ring; eat eligible, spill gated
         std::vector<reg_entry> gated;
         for (;;)
         {
            const uint32_t t = _reg_tail.load(std::memory_order_relaxed);
            if (t == _reg_head.load(std::memory_order_acquire))
               break;
            const reg_entry e = _reg[t % kRegRing];
            _reg_tail.store(t + 1, std::memory_order_release);
            if (eligible(e))
               eat_chain(e.head);
            else
               gated.push_back(e);
         }
         // retry the spill (ring-overflow + freshly-gated): eat eligible, keep rest
         std::vector<reg_entry> local;
         {
            std::lock_guard g(_reg_spill_mx);
            local.swap(_reg_spill);
         }
         std::vector<reg_entry> keep = std::move(gated);
         for (const reg_entry& e : local)
            if (eligible(e))
               eat_chain(e.head);
            else
               keep.push_back(e);
         if (!keep.empty())
         {
            std::lock_guard g(_reg_spill_mx);
            _reg_spill.insert(_reg_spill.end(), keep.begin(), keep.end());
         }
      }
      void drain_loop_() noexcept
      {
         for (;;)
         {
            if (_reg_tail.load(std::memory_order_acquire) ==
                _reg_head.load(std::memory_order_acquire))
            {
               if (_drain_stop.load(std::memory_order_acquire))
                  return;
               std::unique_lock lk(_drain_mx);
               _drain_cv.wait_for(lk, std::chrono::milliseconds(2));
               continue;
            }
            drain_();
            if (_drain_stop.load(std::memory_order_acquire) &&
                _reg_tail.load(std::memory_order_acquire) ==
                    _reg_head.load(std::memory_order_acquire))
               return;
            std::this_thread::yield();
         }
      }

      static start_ver::arena_config make_cfg_(std::filesystem::path dir, runtime_config cfg)
      {
         start_ver::arena_config c;
         c.dir          = std::move(dir);
         c.sync_mode    = cfg.sync_mode;
         c.max_segments = cfg.max_segments;
         // COW churn frees whole generations but long-lived survivors pepper
         // old segments; 1/8 keeps the can-this-segment-serve-allocs
         // guarantee while reclaiming aggressively.
         c.reuse_threshold = cfg.reuse_threshold;
         return c;
      }

      struct reg_entry
      {
         uint32_t ver;
         uint32_t head;
         uint32_t root;  // root index this chain's garbage belongs to;
                         // UINT32_MAX = multi-root chain -> global floor
      };
      static constexpr uint32_t kRegRing = 1u << 16;  // 512 KB; covers deep pin backlogs
      reg_entry                 _reg[kRegRing];
      std::atomic<uint32_t>     _reg_head{0}, _reg_tail{0};
      std::mutex                _dead_mx;   // dead version roots -> the reaper
      std::condition_variable   _dead_cv;
      std::vector<uint32_t>     _dead_roots;
      std::atomic<uint64_t>     _reap_enq{0}, _reap_done{0};
      std::thread               _reaper;
      bool                      _reaper_stop = false;
      std::mutex                _reg_spill_mx;  // ring-overflow side list (writer never blocks)
      std::vector<reg_entry>    _reg_spill;
      std::mutex                _drain_mx;
      std::condition_variable   _drain_cv;
      std::thread               _drain_thread;
      std::atomic<bool>         _drain_stop{false};
      std::atomic<bool>         _drain_active{false};  // a drain pass is mid-flight
                                                       // (gated chains in a local) —
                                                       // flush_releases must wait

      start_ver::recycle_ring   _recycle;  // drain -> writer exact-size run reuse
      start_ver::arena          _arena;
      std::mutex            _root_mx;             // serializes commit publication
      std::mutex            _root_locks[kMaxRoots];  // per-root writer locks (ordered)
      std::atomic<uint64_t> _live_arr[kMaxRoots];    // (version<<32|root_i) per root
      std::atomic<uint64_t> _live{0};             // (version << 32) | root raw
      uint32_t              _committed = 0;  // last published version (writer-owned)

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
      start_ver::value_hooks _vh{&vh_copy_s_, &vh_release_s_, this};
   };

   // ── bodies ────────────────────────────────────────────────────────────────
   inline write_session::write_session(database* db)
       : _db(db), _ses(db->_arena.start_session()), _t(part_alloc<bval>(&_ses))
   {
      _t.allocator_mut().vlog   = &_vlog;
      _t.allocator_mut().vhooks = &db->_vh;
      // re-adopt the durable root on reopen (parity plus: part started empty)
      const uint64_t live = db->_live_arr[0].load(std::memory_order_seq_cst);
      if (live != 0)  // raw 0 is a valid root; version 0 = unpublished
      {
         const uint32_t r32 = uint32_t(live);
         root_ptr       p;
         std::memcpy(p.b, &r32, 4);
         _t.adopt_root(p);
      }
   }

   inline engine_t& write_session::tree_for(uint32_t i)
   {
      assert(i < kMaxRoots);
      if (i == 0)
         return _t;
      if (!_aux[i])
      {
         _aux[i].emplace(part_alloc<bval>(&_ses));
         _aux[i]->allocator_mut().vlog   = &_vlog;  // shared per-txn chains
         _aux[i]->allocator_mut().vhooks = &_db->_vh;
         const uint64_t live = _db->_live_arr[i].load(std::memory_order_seq_cst);
         if (live != 0)
         {
            const uint32_t r32 = uint32_t(live);
            root_ptr       p;
            std::memcpy(p.b, &r32, 4);
            _aux[i]->adopt_root(p);
         }
      }
      return *_aux[i];
   }

   inline transaction write_session::start_transaction(uint32_t root_index)
   {
      assert(root_index < kMaxRoots);
      _vlog.reset();
      {  // adopt drain-recycled runs: exact-size O(1) reuse of COW churn
         uint32_t l, n, k = 0;
         while (k++ < 512 && _db->_recycle.pop(l, n))
            _ses.recycle_drained(l, n);
      }
      const uint32_t v = _db->_committed + 1;  // the version IS the txn
      _t.allocator_mut().vtxn = v;             // shared allocator state: all roots
      return transaction(this, root_index, _ses.start_transaction(0));
   }

   inline read_session::read_session(database* db)
       : _db(db), _rs(db->_arena.start_read_session())
   {
   }

   inline cursor read_session::create_cursor(uint32_t root_index)
   {
      assert(root_index < kMaxRoots);
      auto& slot = _db->_arena.reader_slot(_rs.slot());
      // attribute the pin to its root (per-root floor) before publishing
      _db->_arena.reader_root_idx(_rs.slot()).store(root_index, std::memory_order_relaxed);
      for (;;)
      {
         const uint64_t live = _db->_live_arr[root_index].load(std::memory_order_seq_cst);
         slot.store(live, std::memory_order_seq_cst);  // publish the pin FIRST...
         if (_db->_live_arr[root_index].load(std::memory_order_seq_cst) == live)
         {  // ...then validate: the boundary was live after our publish, so
            // any drain that could free it must see our pinned version.
            root_ptr p = root_ptr::null();
            if (live != 0)  // raw 0 is a valid root; version 0 = unpublished
            {
               const uint32_t r32 = uint32_t(live);
               std::memcpy(p.b, &r32, 4);
            }
            return cursor(&_db->_arena, p, &slot);
         }
      }
   }

   inline cursor tree_root::snapshot_cursor() const
   {
      if (!_c)
         return database::empty_cursor();
      return _c->db->snapshot_cursor_for(_c->vf, _c->raw, _c->idx);
   }
   template <class S>
   inline std::optional<S> tree_root::get(key_view k) const
   {
      if (!_c)
         return std::nullopt;
      return snapshot_cursor().template get<S>(k);
   }

   inline tree_root::ctrl::~ctrl()
   {
      if (armed)
         db->reap_root_(raw);
      if (pin_slot)
         pin_slot->store(0, std::memory_order_seq_cst);  // drop the version pin
   }

   inline tree_root write_session::get_root(uint32_t root_index)
   {
      auto& arena = _db->_arena;
      auto  rs    = arena.start_read_session();
      auto& slot  = arena.reader_slot(rs.slot());
      // attribute the pin to this root (per-root floor) before publishing
      arena.reader_root_idx(rs.slot()).store(root_index, std::memory_order_relaxed);
      uint64_t live;
      for (;;)  // publish-validate (the pin_snapshot protocol)
      {
         live = _db->_live_arr[root_index].load(std::memory_order_seq_cst);
         slot.store(live, std::memory_order_seq_cst);
         if (_db->_live_arr[root_index].load(std::memory_order_seq_cst) == live)
            break;
      }
      auto c   = std::make_shared<tree_root::ctrl>();
      c->db    = _db;
      c->vf    = uint32_t(live >> 32);
      c->raw   = uint32_t(live);
      c->idx   = root_index;
      c->count = tree_for(root_index).size_hint();
      c->armed = false;  // view of the LIVE lineage: never reaped from here
      c->pin.emplace(std::move(rs));
      c->pin_slot = &slot;
      return tree_root(std::move(c));
   }


   inline void transaction::upsert(key_view k, value_view v)
   {
      _ws->_t.cow_insert(k, bval{env_raw_(v)});  // [0][bytes]; the leaf copies once
   }
   template <class T>
      requires storable<std::remove_cvref_t<T>>
   inline void transaction::upsert(key_view k, T&& v)
   {
      using U                = std::remove_cvref_t<T>;
      const typename U::t_pod pod = std::move(v).take_pod();  // custody -> db
      char                    buf[2 + sizeof(pod)];
      const uint16_t          id = U::type_id;
      std::memcpy(buf, &id, 2);
      std::memcpy(buf + 2, &pod, sizeof(pod));
      _ws->_t.cow_insert(k, bval{value_view(buf, sizeof(buf))});
   }
   inline any_value transaction::get_any(key_view k) const
   {
      bval out;
      if (!_ws->_t.find(k, out))
         return {};
      return any_value{out.v.data(), out.v.size()};  // slot view, txn-pinned
   }
   inline int transaction::remove(key_view k)
   {
      return _ws->_t.cow_remove(k) ? 0 : -1;
   }
   inline int transaction::remove_if_older(key_view k, uint32_t vmax)
   {
      const uint32_t st = _ws->_t.write_stamp(k);
      if (st == engine_t::no_stamp || st > vmax)
         return -1;
      return _ws->_t.cow_remove(k) ? 0 : -1;
   }
   inline size_t transaction::retire_detach(uint32_t floor)
   {
      return _ws->_t.retire_detach(floor);
   }
   inline uint64_t transaction::retire_count() const
   {
      return _ws->_t.retire_count();
   }
   inline void transaction::retire_release(uint64_t counted)
   {
      _ws->_t.retire_release(counted);
   }
   template <class F>
   inline bool transaction::get(key_view k, F&& fn) const
   {
      bval out;
      if (!_ws->_t.find(k, out))
         return false;
      if (slot_type(out.v) != kRawType)
         throw invalid_type{slot_type(out.v), kRawType};  // raw view of a typed pod
      fn(slot_payload(out.v));
      return true;
   }
   template <class S>
   inline std::optional<S> transaction::get(key_view k) const
   {
      bval out;
      if (!_ws->_t.find(k, out))
         return std::nullopt;
      return slot_decode<S>(out.v);
   }

   inline transaction::transaction(write_session* ws, uint32_t root_index,
                                   start_ver::transaction&& tx)
       : _ws(ws), _tx(std::move(tx))
   {
      auto&      t = ws->tree_for(root_index);
      root_state rs{root_index, t.root_handle(), t.size_hint(),
                    std::unique_lock<std::mutex>(ws->_db->_root_locks[root_index])};
      _roots.push_back(std::move(rs));
   }

   inline transaction::root_state* transaction::find_root_(uint32_t root_index)
   {
      for (auto& r : _roots)
         if (r.index == root_index)
            return &r;
      return nullptr;
   }

   inline engine_t& transaction::tree_(uint32_t root_index)
   {
      return _ws->tree_for(root_index);
   }

   inline tree_handle transaction::primary()
   {
      return tree_handle(this, _roots[0].index);
   }

   inline tree_handle transaction::open_root(uint32_t root_index)
   {
      assert(root_index < kMaxRoots);
      assert(!_roots.empty());
      assert(root_index > _roots.back().index &&
             "roots must be opened in ascending index order (deadlock freedom)");
      auto&      t = _ws->tree_for(root_index);
      root_state rs{root_index, t.root_handle(), t.size_hint(),
                    std::unique_lock<std::mutex>(_ws->_db->_root_locks[root_index])};
      _roots.push_back(std::move(rs));
      return tree_handle(this, root_index);
   }

   inline void tree_handle::upsert(key_view k, value_view v)
   {
      _tx->tree_(_ri).cow_insert(k, bval{env_raw_(v)});
   }
   inline int tree_handle::remove(key_view k)
   {
      return _tx->tree_(_ri).cow_remove(k) ? 0 : -1;
   }
   template <class F>
   inline bool tree_handle::get(key_view k, F&& fn) const
   {
      bval out;
      if (!_tx->tree_(_ri).find(k, out))
         return false;
      if (slot_type(out.v) != kRawType)
         throw invalid_type{slot_type(out.v), kRawType};
      fn(slot_payload(out.v));
      return true;
   }
   template <class S>
   inline std::optional<S> tree_handle::get(key_view k) const
   {
      bval out;
      if (!_tx->tree_(_ri).find(k, out))
         return std::nullopt;
      return slot_decode<S>(out.v);
   }
   inline uint64_t tree_handle::key_count() const
   {
      return _tx->tree_(_ri).size_hint();
   }

   inline tree_root tree_handle::set_root(tree_root t)
   {
      auto&          tr      = _tx->tree_(_ri);
      const root_ptr old     = tr.root_handle();
      const uint64_t old_cnt = tr.size_hint();
      // adopt the incoming tree (empty handle = empty tree); DISARM it — its
      // tree is live in this root now, never reaped by the handle's death
      root_ptr nr = root_ptr::null();
      uint64_t nc = 0;
      if (t._c)
      {
         const uint32_t r32 = t._c->raw;
         std::memcpy(nr.b, &r32, 4);
         nc          = t._c->count;
         t._c->armed = false;  // its tree is live in this root now
         // displaced-then-readopted IN THE SAME TXN (the rotation): commit
         // must NOT re-arm it — pull it off the displaced list
         auto& d = _tx->_displaced;
         for (size_t i = 0; i < d.size(); ++i)
            if (d[i] == t._c)
            {
               d.erase(d.begin() + long(i));
               break;
            }
      }
      tr.adopt_root(nr, size_t(nc));
      // the displaced tree: ARMS at COMMIT (abort restores the slot instead)
      auto c   = std::make_shared<tree_root::ctrl>();
      c->db    = _tx->_ws->_db;
      c->vf    = _tx->_ws->_t.allocator_mut().vtxn;
      c->raw   = uint32_t(old.raw());
      c->count = old_cnt;
      c->armed = false;
      _tx->_displaced.push_back(c);
      return tree_root(std::move(c));
   }

   namespace detail2
   {
      inline void free_chain_lines_(start_ver::arena& a, uint32_t head) noexcept
      {
         uint32_t line = head;
         while (line != vlog_state::npos)
         {
            const uint32_t* w    = static_cast<const uint32_t*>(a.get(line));
            const uint32_t  prev = w[0];
            a.release_run(line, 1);
            line = prev;
         }
      }
   }  // namespace detail2

   inline void transaction::abort() noexcept
   {
      if (_done)
         return;
      _done    = true;
      auto& vl = _ws->_vlog;
      auto& a  = _ws->_db->_arena;
      // ALL roots roll back together: free each diverged window by WALKING
      // it (recursion stops at pre-txn nodes — shared with the restored
      // root), restore each boundary root, drop the chain unpublished.
      // Displaced tree_roots stay DISARMED: the rollback re-seats their
      // trees, so their handles must never reap.
      for (auto it = _roots.rbegin(); it != _roots.rend(); ++it)
      {
         auto&          t        = _ws->tree_for(it->index);
         const root_ptr diverged = t.root_handle();
         if (diverged.raw() != it->start_root.raw())
            t.vabort_release(diverged);
         t.adopt_root(it->start_root, it->start_count);
         it->lock.unlock();  // reverse order
      }
      _roots.clear();
      _displaced.clear();
      detail2::free_chain_lines_(a, vl.clone_head);
      vl.reset();
      _ws->_t.allocator_mut().vtxn = 0;
   }

   inline void transaction::commit()
   {
      _done = true;
      _tx.commit();  // data msync + floor advance (layer 1)
      database*      db = _ws->_db;
      auto&          vl = _ws->_vlog;
      const uint32_t v  = _ws->_t.allocator_mut().vtxn;
      {
         std::lock_guard g(db->_root_mx);
         // Publish every held root in DESCENDING index order (the rotation's
         // crash-idempotence: warm (1) becomes durable before hot (0) — a
         // crash between leaves the tree in BOTH roots, masked by read
         // precedence and re-flushed idempotently). Null roots publish their
         // REAL bit pattern (all-FF): raw 0 is a VALID root (line 0, tag
         // value_ptr). Version 0 alone means "nothing ever published".
         for (auto it = _roots.rbegin(); it != _roots.rend(); ++it)
         {
            const root_ptr nr  = _ws->tree_for(it->index).root_handle();
            const uint32_t r32 = uint32_t(nr.raw());
            db->_arena.set_root(it->index, r32);
            db->_live_arr[it->index].store((uint64_t(v) << 32) | r32,
                                           std::memory_order_seq_cst);
         }
         db->_arena.set_root(kMaxRoots, v);  // the global version slot
         db->_committed = v;
      }
      // Tag the chain with its root for per-root reclaim: a single-root txn's
      // garbage is that root's lineage (the common hot write = root 0, drains
      // at root-0's floor unblocked by warm-root pins); a multi-root txn (the
      // rotation/clear) mixes lineages -> global floor (UINT32_MAX).
      const uint32_t chain_root = _roots.size() == 1 ? _roots[0].index : UINT32_MAX;
      db->reg_push_(v, vl.clone_head, chain_root);  // publish the supersession chain
      vl.reset();
      _ws->_t.allocator_mut().vtxn = 0;
      for (auto& c : _displaced)
         c->armed = true;  // committed: displaced trees now reap on last drop
      _displaced.clear();
      for (auto it = _roots.rbegin(); it != _roots.rend(); ++it)
         it->lock.unlock();
      _roots.clear();
      if (!db->_drain_thread.joinable())
         db->drain_();  // no thread: drain inline on the writer
   }
   // kv_engine concept surface (psitri::dwal::kv_engine): one aggregate naming
   // the five types + config. tiered_kv<W, Base> and the conformance
   // static_assert bind to THIS, never to the classes directly.
   struct engine
   {
      using database      = start_ver::kv::database;
      using write_session = start_ver::kv::write_session;
      using read_session  = start_ver::kv::read_session;
      using transaction   = start_ver::kv::transaction;
      using cursor        = start_ver::kv::cursor;
      using config        = start_ver::kv::runtime_config;
   };
}  // namespace start_ver::kv

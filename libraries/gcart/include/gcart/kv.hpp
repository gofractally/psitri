// gcart::legacy::kv — adb-conforming engine over gcart::legacy::tree on sal2.
//
// Transactions are COW: a txn pins the committed root (retain), mutates a private
// copy (writable clones the pinned root on first write), and on commit publishes
// the new root + releases the old; abort releases the work and restores the
// committed root. Snapshots (get_root / read cursors) hold a retained root and
// iterate it frozen — concurrent writers COW around them. v1 core surface
// (CRUD + ordered cursor + snapshot); typed-values / multi-root are deferred
// (their adb feature batteries are if-constexpr-gated, so they're simply skipped).
#pragma once
#include <gcart/tree.hpp>

#include <sal/config.hpp>  // sal::open_mode (embedder open factory)

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace gcart::legacy::kv
{
   using key_view   = std::string_view;
   using value_view = std::string_view;

   struct runtime_config
   {
      uint32_t max_segments = 16384;
   };

   // ── cursor: in-order path-stack iterator over a frozen root ───────────────
   class cursor
   {
     public:
      // A cursor holds a smart_ptr — the pin IS that handle: the snapshot stays
      // readable for the cursor's whole lifetime while a concurrent writer COWs
      // and frees other versions around it. Rule of zero: copy shares the pin,
      // move transfers it, destruction releases it — all via the smart_ptr
      // member. The raw _a/_root are cached from the pin for the hot read path.
      cursor() = default;
      explicit cursor(smart_ptr pin) : _pin(std::move(pin)), _a(_pin.arena()), _root(_pin.root()) {}

      bool seek_begin()
      {
         _depth = 0;
         _cur   = null_ptr;
         _rend  = false;
         if (_root == null_ptr) { _end = true; return false; }
         descend_leftmost(_root);
         _end = (_cur == null_ptr);
         return !_end;
      }
      bool seek_last()
      {
         _depth = 0;
         _cur   = null_ptr;
         _end   = false;
         if (_root == null_ptr) { _rend = true; return false; }
         descend_rightmost(_root);
         _rend = (_cur == null_ptr);
         return !_rend;
      }
      bool next()
      {
         if (_end) return false;
         advance_(/*forward=*/true);
         return !_end;
      }
      bool prev()
      {
         if (_rend) return false;
         advance_(/*forward=*/false);
         return !_rend;
      }
      bool lower_bound(key_view k)
      {
         _depth = 0; _cur = null_ptr; _end = false; _rend = false;
         if (_root == null_ptr) { _end = true; return false; }
         seek_lb_(_root, k, 0);
         _end = (_cur == null_ptr);
         return !_end;
      }
      bool upper_bound(key_view k)
      {
         if (!lower_bound(k)) return false;
         if (key() == k) return next();
         return true;
      }
      void find(key_view k)
      {
         if (lower_bound(k) && key() != k) { _end = true; _cur = null_ptr; }
      }

      bool is_end() const { return _end; }
      bool is_rend() const { return _rend; }
      bool is_subtree() const { return false; }

      key_view key() const
      {
         return _cur == null_ptr ? key_view{} : leaf_at(_cur)->key();
      }
      template <class S = std::string>
      std::optional<S> value() const
      {
         if (_cur == null_ptr) return std::nullopt;
         value_view v = gcart::legacy::read_value_view(*_a, leaf_at(_cur));
         return S(v.data(), v.size());
      }
      template <class F>
      bool get_value(F&& f) const
      {
         if (_cur == null_ptr) return false;
         f(gcart::legacy::read_value_view(*_a, leaf_at(_cur)));
         return true;
      }
      uint16_t value_type() const { return 0; }
      template <class T>
      bool is_type() const { return false; }

      // point reads (re-seek each call; cursor is a stable snapshot view)
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         const leaf* lf = find_leaf(_a, _root, k);
         if (!lf) return std::nullopt;
         value_view v = gcart::legacy::read_value_view(*_a, lf);
         return S(v.data(), v.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         const leaf* lf = find_leaf(_a, _root, k);
         if (!lf) return false;
         f(gcart::legacy::read_value_view(*_a, lf));
         return true;
      }

     private:
      const leaf* leaf_at(nptr n) const { return static_cast<const leaf*>(_a->get(n)); }

      static const leaf* find_leaf(const sal2::arena* a, nptr root, key_view key)
      {
         nptr cur = root; uint32_t depth = 0;
         while (cur != null_ptr)
         {
            const void* p = a->get(cur);
            if (is_leaf(p))
            { auto* lf = static_cast<const leaf*>(p); return lf->key() == key ? lf : nullptr; }
            const auto* h = static_cast<const inner_hdr*>(p);
            sv pfx = (h->type == uint8_t(ntype::setlist))
                         ? static_cast<const setlist*>(p)->prefix()
                         : static_cast<const node256*>(p)->prefix();
            if (key.size() - depth < pfx.size()) return nullptr;
            if (!pfx.empty() && std::memcmp(key.data() + depth, pfx.data(), pfx.size()) != 0)
               return nullptr;
            depth += uint32_t(pfx.size());
            if (depth == key.size())
            {
               if (h->term == null_ptr) return nullptr;
               auto* t = static_cast<const leaf*>(a->get(h->term));
               return t->key() == key ? t : nullptr;
            }
            cur = child_for(a, cur, uint8_t(key[depth++]));
         }
         return nullptr;
      }
      static nptr child_for(const sal2::arena* a, nptr n, uint8_t byte)
      {
         const void* p = a->get(n);
         if (type_of(p) == uint8_t(ntype::setlist))
         {
            auto* s = static_cast<const setlist*>(p);
            int   i = s->find_index(byte);
            return i < 0 ? null_ptr : s->children()[i];
         }
         return static_cast<const node256*>(p)->child(byte);
      }
      // i-th present child of a node (ascending byte order)
      nptr nth_child(nptr n, int i, uint8_t* out_byte = nullptr) const
      {
         const void* p = _a->get(n);
         if (type_of(p) == uint8_t(ntype::setlist))
         {
            auto* s = static_cast<const setlist*>(p);
            if (out_byte) *out_byte = s->keys()[i];
            return s->children()[i];
         }
         auto* nd = static_cast<const node256*>(p);
         int   seen = 0;
         const nptr* c = nd->children();
         for (int b = 0; b < 256; ++b)
            if (c[b] != null_ptr && seen++ == i) { if (out_byte) *out_byte = uint8_t(b); return c[b]; }
         return null_ptr;
      }
      int nchildren(nptr n) const
      {
         const void* p = _a->get(n);
         return type_of(p) == uint8_t(ntype::setlist) ? static_cast<const setlist*>(p)->h.n
                                                      : static_cast<const node256*>(p)->h.n16;
      }
      nptr term_of(nptr n) const { return static_cast<const inner_hdr*>(_a->get(n))->term; }

      struct frame { nptr node; int pos; };  // pos: -1 = term, else child index
      // depth bound = the adb key-length contract (1024 B) + margin; a key of N
      // bytes descends at most N nodes (no prefix compression in the worst case).
      std::array<frame, 1088> _stk{};
      int    _depth = 0;
      smart_ptr _pin;              // the snapshot pin (lifetime == this cursor)
      sal2::arena* _a = nullptr;   // cached _pin.arena() for the read hot path
      nptr   _root = null_ptr;     // cached _pin.root()
      nptr   _cur  = null_ptr;  // current leaf
      bool   _end = true, _rend = true;

      void descend_leftmost(nptr n)
      {
         for (;;)
         {
            if (is_leaf(_a->get(n))) { _cur = n; return; }
            if (term_of(n) != null_ptr) { _stk[_depth++] = {n, -1}; _cur = term_of(n); return; }
            _stk[_depth++] = {n, 0};
            n              = nth_child(n, 0);
         }
      }
      void descend_rightmost(nptr n)
      {
         for (;;)
         {
            if (is_leaf(_a->get(n))) { _cur = n; return; }
            int nc = nchildren(n);
            if (nc > 0) { _stk[_depth++] = {n, nc - 1}; n = nth_child(n, nc - 1); continue; }
            // no children → term is the only entry
            _stk[_depth++] = {n, -1}; _cur = term_of(n); return;
         }
      }
      void advance_(bool fwd)
      {
         while (_depth > 0)
         {
            frame& f  = _stk[_depth - 1];
            int    nc = nchildren(f.node);
            if (fwd)
            {
               if (f.pos == -1) { if (nc > 0) { f.pos = 0; descend_leftmost(nth_child(f.node, 0)); return; } }
               else if (f.pos + 1 < nc) { ++f.pos; descend_leftmost(nth_child(f.node, f.pos)); return; }
               --_depth;
            }
            else
            {
               if (f.pos > 0) { --f.pos; descend_rightmost(nth_child(f.node, f.pos)); return; }
               else if (f.pos == 0) { if (term_of(f.node) != null_ptr) { f.pos = -1; _cur = term_of(f.node); return; } --_depth; }
               else /* pos==-1 */ --_depth;
            }
         }
         if (fwd) _end = true; else _rend = true;
         _cur = null_ptr;
      }
      // position at the first key >= k
      void seek_lb_(nptr n, key_view key, uint32_t depth)
      {
         for (;;)
         {
            const void* p = _a->get(n);
            if (is_leaf(p))
            {
               auto* lf = static_cast<const leaf*>(p);
               if (lf->key() >= key) { _cur = n; return; }
               advance_(true);
               return;
            }
            sv pfx = node_prefix_(n);
            // compare key suffix vs prefix
            uint32_t krem = uint32_t(key.size() - depth);
            uint32_t m    = std::min<uint32_t>(krem, uint32_t(pfx.size()));
            int      c    = std::memcmp(key.data() + depth, pfx.data(), m);
            if (c < 0 || (c == 0 && krem < pfx.size()))
            { descend_leftmost(n); return; }            // whole subtree > key
            if (c > 0) { advance_(true); return; }       // whole subtree < key
            depth += uint32_t(pfx.size());
            if (depth == key.size()) { descend_leftmost(n); return; }  // term or first child >= key
            uint8_t byte = uint8_t(key[depth]);
            // find child position with byte (or first child > byte)
            const auto* h = static_cast<const inner_hdr*>(p);
            int pos, nc = (h->type == uint8_t(ntype::setlist)) ? h->n : h->n16;
            uint8_t cbyte;
            int idx = child_index_ge(n, byte, pos, cbyte);
            if (idx < 0) { advance_(true); return; }      // all children < byte
            _stk[_depth++] = {n, idx};
            if (cbyte == byte) { n = nth_child(n, idx); ++depth; continue; }  // exact, recurse
            descend_leftmost(nth_child(n, idx));          // first child > byte
            return;
            (void)pos; (void)nc;
         }
      }
      sv node_prefix_(nptr n) const
      {
         const void* p = _a->get(n);
         return type_of(p) == uint8_t(ntype::setlist) ? static_cast<const setlist*>(p)->prefix()
                                                      : static_cast<const node256*>(p)->prefix();
      }
      // first child index whose byte >= `byte`; sets cbyte. -1 if none.
      int child_index_ge(nptr n, uint8_t byte, int& pos, uint8_t& cbyte) const
      {
         const void* p = _a->get(n);
         if (type_of(p) == uint8_t(ntype::setlist))
         {
            auto* s = static_cast<const setlist*>(p);
            int   i = int(ucc::lower_bound_padded(s->keys(), s->h.n, byte));
            if (i >= s->h.n) return -1;
            cbyte = s->keys()[i]; pos = i; return i;
         }
         auto* nd = static_cast<const node256*>(p);
         const nptr* c = nd->children();
         int seen = 0;
         for (int b = 0; b < 256; ++b)
         {
            if (c[b] == null_ptr) continue;
            if (uint8_t(b) >= byte) { cbyte = uint8_t(b); pos = seen; return seen; }
            ++seen;
         }
         return -1;
      }
   };

   class database;

   // ── snapshot tree (owned, frozen) ─────────────────────────────────────────
   // A tree IS a snapshot: it holds a smart_ptr, so the version it was acquired
   // at stays readable for its lifetime. Rule of zero — the smart_ptr member
   // does all the refcount work (copy shares, move transfers, dtor releases).
   class tree
   {
     public:
      tree() = default;
      explicit tree(smart_ptr pin) : _pin(std::move(pin)) {}

      cursor snapshot_cursor() const { return cursor(_pin); }  // shares the pin
      template <class S = std::string>
      std::optional<S> get(key_view k) const { return snapshot_cursor().template get<S>(k); }
      explicit operator bool() const noexcept { return bool(_pin); }
      nptr root_raw() const noexcept { return _pin.root(); }

     private:
      smart_ptr _pin;
   };

   class write_session;
   class read_session;

   // ── transaction: COW-snapshot over one root's gcart::legacy::tree ──────────────────
   class transaction
   {
     public:
      transaction(transaction&& o) noexcept
          : _t(o._t), _a(o._a), _arena_slot(o._arena_slot), _snap(std::move(o._snap)), _done(o._done)
      { o._t = nullptr; }
      transaction(const transaction&)            = delete;
      transaction& operator=(const transaction&) = delete;
      ~transaction() { if (_t && !_done) abort(); }

      void insert(key_view k, value_view v) { _t->upsert(k, v); }
      void upsert(key_view k, value_view v) { _t->upsert(k, v); }
      int  remove(key_view k) { return _t->erase(k) ? 1 : 0; }

      template <class F>
      bool get(key_view k, F&& fn) const
      {
         const leaf* lf = _t->find(k);
         if (!lf) return false;
         fn(gcart::legacy::read_value_view(_t->arena(), lf));
         return true;
      }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         const leaf* lf = _t->find(k);
         if (!lf) return std::nullopt;
         value_view v = gcart::legacy::read_value_view(_t->arena(), lf);
         return S(v.data(), v.size());
      }

      void commit()
      {
         if (_done) return;
         // Publish the new root ATOMICALLY first: a concurrent reader acquiring
         // now sees the new version; one that already pinned the old root keeps
         // it alive. Then drop our snapshot pin on the old committed root — it is
         // freed unless a reader still holds it (lock-free, no reader barrier).
         _a->set_root(_arena_slot, _t->root());
         _snap.reset();
         _done = true;
      }
      void abort() noexcept
      {
         if (_done) return;
         // Hand the snapshot pin back to the live tree as its ownership ref
         // (give() = relinquish without releasing — never re-adopt a raw addr),
         // then free the discarded work tree. The release of `work` balances the
         // un-released give() so the restored root lands at the right refcount.
         const nptr work = _t->root();
         _t->set_root(_snap.give());
         if (work != null_ptr) gcart::legacy::release_subtree(*_a, work);
         _done = true;
      }

     private:
      friend class write_session;
      transaction(gcart::legacy::tree* t, sal2::arena* a, uint32_t slot)
          : _t(t), _a(a), _arena_slot(slot), _snap(smart_ptr::acquire(*a, slot))
      {
         // _snap pins the committed root (the rollback target + COW base). At a
         // txn boundary the working tree's root == the published arena root, so
         // acquire() pins exactly what _t is about to COW from.
      }
      gcart::legacy::tree* _t = nullptr;
      sal2::arena* _a = nullptr;
      uint32_t     _arena_slot = 0;
      smart_ptr    _snap;
      bool         _done = false;
   };

   // ── database ──────────────────────────────────────────────────────────────
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
      // Unified open semantics (adb::openable_database — the embedder contract).
      static std::shared_ptr<database> open(std::filesystem::path dir,
                                            sal::open_mode        mode,
                                            runtime_config        cfg = {})
      {
         const bool exists = std::filesystem::exists(dir / "cl1.dat");
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
               throw std::runtime_error("gcart: read-only opens not implemented yet");
            case sal::open_mode::create_or_open:
               break;
         }
         return create(std::move(dir), cfg);
      }

      std::shared_ptr<write_session> start_write_session();
      std::shared_ptr<read_session>  start_read_session();
      void sync(sal2::line_idx = 0) noexcept {}

      sal2::arena& store() noexcept { return _arena; }
      gcart::legacy::tree& tree_for(uint32_t idx)
      {
         if (!_trees[idx])
         {
            _trees[idx] = std::make_unique<gcart::legacy::tree>(_arena);
            _trees[idx]->set_root(_arena.get_root(idx));  // adopt any persisted root
         }
         return *_trees[idx];
      }

     private:
      explicit database(std::filesystem::path dir, runtime_config cfg)
          : _arena(sal2::arena_config{.dir = std::move(dir), .max_segments = cfg.max_segments})
      {
      }
      sal2::arena                                   _arena;
      std::array<std::unique_ptr<gcart::legacy::tree>, 64>  _trees{};
   };

   // ── write_session ──────────────────────────────────────────────────────────
   class write_session
   {
     public:
      explicit write_session(database* db) : _db(db) {}
      transaction start_transaction(uint32_t root_index = 0)
      {
         gcart::legacy::tree& t = _db->tree_for(root_index);
         return transaction(&t, &_db->store(), root_index);
      }
      tree get_root(uint32_t root_index)
      {
         // a tree IS a snapshot: lock-free pin of the COMMITTED (atomic) root —
         // never the writer's in-flight work tree.
         return tree(smart_ptr::acquire(_db->store(), root_index));
      }

     private:
      database* _db;
   };

   // ── read_session ────────────────────────────────────────────────────────────
   class read_session
   {
     public:
      explicit read_session(database* db) : _db(db) {}
      cursor create_cursor(uint32_t root_index)
      {
         // lock-free pin of the COMMITTED (atomic) root (try_retain + verify);
         // concurrent readers never block each other or the writer.
         return cursor(smart_ptr::acquire(_db->store(), root_index));
      }

     private:
      database* _db;
   };

   inline std::shared_ptr<write_session> database::start_write_session()
   {
      return std::make_shared<write_session>(this);
   }
   inline std::shared_ptr<read_session> database::start_read_session()
   {
      return std::make_shared<read_session>(this);
   }

   // ── engine traits bundle (psitri::dwal::kv_engine form) ────────────────────
   struct engine
   {
      using database      = gcart::legacy::kv::database;
      using write_session = gcart::legacy::kv::write_session;
      using read_session  = gcart::legacy::kv::read_session;
      using transaction   = gcart::legacy::kv::transaction;
      using cursor        = gcart::legacy::kv::cursor;
      using config        = gcart::legacy::kv::runtime_config;
   };
}  // namespace gcart::legacy::kv

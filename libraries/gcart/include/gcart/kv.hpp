// gcart::kv — adb-conforming engine over gcart::tree on sal2.
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

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace gcart::kv
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
      cursor() = default;
      cursor(const sal2::arena* a, nptr root) : _a(a), _root(root) {}

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
         value_view v = leaf_at(_cur)->value();
         return S(v.data(), v.size());
      }
      template <class F>
      bool get_value(F&& f) const
      {
         if (_cur == null_ptr) return false;
         f(leaf_at(_cur)->value());
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
         value_view v = lf->value();
         return S(v.data(), v.size());
      }
      template <class F>
      bool get(key_view k, F&& f) const
      {
         const leaf* lf = find_leaf(_a, _root, k);
         if (!lf) return false;
         f(lf->value());
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
      const sal2::arena* _a = nullptr;
      nptr   _root = null_ptr;
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
   class tree
   {
     public:
      tree() = default;
      tree(sal2::arena* a, nptr root) : _a(a), _root(root)
      {
         if (_root != null_ptr) _a->retain(_root);
      }
      tree(const tree& o) : _a(o._a), _root(o._root)
      {
         if (_root != null_ptr) _a->retain(_root);
      }
      tree& operator=(const tree& o)
      {
         if (this != &o) { reset_(); _a = o._a; _root = o._root; if (_root != null_ptr) _a->retain(_root); }
         return *this;
      }
      tree(tree&& o) noexcept : _a(o._a), _root(o._root) { o._root = null_ptr; }
      tree& operator=(tree&& o) noexcept
      {
         if (this != &o) { reset_(); _a = o._a; _root = o._root; o._root = null_ptr; }
         return *this;
      }
      ~tree() { reset_(); }

      cursor snapshot_cursor() const { return cursor(_a, _root); }
      template <class S = std::string>
      std::optional<S> get(key_view k) const { return snapshot_cursor().template get<S>(k); }
      explicit operator bool() const noexcept { return _root != null_ptr; }
      nptr root_raw() const noexcept { return _root; }

     private:
      void reset_()
      {
         if (_a && _root != null_ptr) gcart::release_subtree(*_a, _root);
         _root = null_ptr;
      }
      sal2::arena* _a    = nullptr;
      nptr         _root = null_ptr;
   };

   class write_session;
   class read_session;

   // ── transaction: COW-snapshot over one root's gcart::tree ──────────────────
   class transaction
   {
     public:
      transaction(transaction&& o) noexcept
          : _t(o._t), _a(o._a), _arena_slot(o._arena_slot), _snap(o._snap), _done(o._done)
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
         fn(lf->value());
         return true;
      }
      template <class S = std::string>
      std::optional<S> get(key_view k) const
      {
         const leaf* lf = _t->find(k);
         if (!lf) return std::nullopt;
         value_view v = lf->value();
         return S(v.data(), v.size());
      }

      void commit()
      {
         if (_done) return;
         if (_snap != null_ptr) gcart::release_subtree(*_a, _snap);  // drop the old committed root
         _a->set_root(_arena_slot, _t->root());                      // publish
         _done = true;
      }
      void abort() noexcept
      {
         if (_done) return;
         const nptr work = _t->root();
         _t->set_root(_snap);                       // restore committed (pin → live)
         if (work != null_ptr) gcart::release_subtree(*_a, work);
         _done = true;
      }

     private:
      friend class write_session;
      transaction(gcart::tree* t, sal2::arena* a, uint32_t slot) : _t(t), _a(a), _arena_slot(slot)
      {
         _snap = _t->root();
         if (_snap != null_ptr) _a->retain(_snap);  // pin committed root for rollback/COW
      }
      gcart::tree* _t = nullptr;
      sal2::arena* _a = nullptr;
      uint32_t     _arena_slot = 0;
      nptr         _snap = null_ptr;
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

      std::shared_ptr<write_session> start_write_session();
      std::shared_ptr<read_session>  start_read_session();
      void sync(sal2::line_idx = 0) noexcept {}

      sal2::arena& store() noexcept { return _arena; }
      gcart::tree& tree_for(uint32_t idx)
      {
         if (!_trees[idx])
         {
            _trees[idx] = std::make_unique<gcart::tree>(_arena);
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
      std::array<std::unique_ptr<gcart::tree>, 64>  _trees{};
   };

   // ── write_session ──────────────────────────────────────────────────────────
   class write_session
   {
     public:
      explicit write_session(database* db) : _db(db) {}
      transaction start_transaction(uint32_t root_index = 0)
      {
         gcart::tree& t = _db->tree_for(root_index);
         return transaction(&t, &_db->store(), root_index);
      }
      tree get_root(uint32_t root_index)
      {
         gcart::tree& t = _db->tree_for(root_index);
         return tree(&_db->store(), t.root());
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
         gcart::tree& t = _db->tree_for(root_index);
         return cursor(&_db->store(), t.root());
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
      using database      = gcart::kv::database;
      using write_session = gcart::kv::write_session;
      using read_session  = gcart::kv::read_session;
      using transaction   = gcart::kv::transaction;
      using cursor        = gcart::kv::cursor;
      using config        = gcart::kv::runtime_config;
   };
}  // namespace gcart::kv

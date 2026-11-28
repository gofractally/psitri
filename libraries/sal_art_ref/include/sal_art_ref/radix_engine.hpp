// The real engine on the DESIGNED segmented node taxonomy (heart/radix_nodes.hpp).
// ONE generic algorithm over a uniform node interface — no per-type special cases.
//
// Every branch/term slot stores a full smart_address: a tail (an inline_value in
// THIS node's tail heap), a pointer (a child node, or a SAL leaf_node holding a
// large value), or null. A terminal {suffix,value} is allocated by ONE size-gated
// rule: <= max_inline → a tail leaf in the node's heap; else a SAL leaf. Nodes GROW
// (realloc, copying the tail heap) up to max_node (4 KB); only when an encoding is
// genuinely full does set_branch fail and the engine widen (setlist→bitset→full),
// re-storing branches (re-inlining tails) into the wider encoding.
//
// Routing is by the control-block dispatch_shape: r8/r4/r2 jump straight to the
// segment cacheline (skip CL0); r1 (a prefixed node) reads CL0. Mutation model:
// COW-to-root (ref-based unique|shared); `versioned`/version_history unused.
#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <adb/value_range.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>  // std::runtime_error (ordered cursor hitting a bucket — see dispatch())
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <sal/allocator.hpp>
#include <sal/allocator_session_impl.hpp>
#include <sal/read_lock.hpp>
#include <sal/shutdown_record.hpp>
#include <ucc/round.hpp>

#include <sal/diag.hpp>            // PSITRI_STATS / PSITRI_DEBUG gating + sal::stat_counter
#include <sal_art_ref/bucket_node.hpp>
#include <sal_art_ref/env_config.hpp>  // single-init env knobs (no scattered getenv)
#include <sal_art_ref/leaf_node.hpp>
#include <sal_art_ref/key_val_node.hpp>
#include <sal_art_ref/value_read.hpp>
#include <sal_art_ref/leaf_page.hpp>
#include <sal_art_ref/value_hooks.hpp>
#include <sal_art_ref/radix_nodes.hpp>

namespace sal_art_ref::radix
{
   using sal::ptr_address;
   using radix_type_ids::kind_of;
   using radix_type_ids::radix_kind;
   using bytes      = std::span<const uint8_t>;
   using key_view   = std::string_view;
   using value_view = std::string_view;

   // Profile defaults retain Heart's baseline; other SAL settings remain available.
   struct runtime_config : sal::runtime_config
   {
      runtime_config()
      {
         write_protect_on_commit = false;
         enable_read_cache = false;
      }
   };

   namespace detail
   {
      // One byte longer than the longest key, all 0xFF: no key is above it.
      inline constexpr std::array<char, key_val_node::max_key_len + 1> max_key_data = []()
      {
         std::array<char, key_val_node::max_key_len + 1> arr{};
         for (auto& c : arr)
            c = '\xFF';
         return arr;
      }();
   }  // namespace detail
   /// An upper bound above every key: pass it as the end of a range to mean "to the end".
   /// It really is above every key (it is longer than any key can be), so code that only
   /// compares keys with it is right too; the engine also recognises it and skips the
   /// comparison.
   static constexpr key_view max_key{detail::max_key_data.data(), detail::max_key_data.size()};

   using setlist_u8  = setlist_node<uint8_t>;
   using setlist_u16 = setlist_node<uint16_t>;
   using setlist_u32 = setlist_node<uint32_t>;

   inline bytes as_bytes(std::string_view s) noexcept
   {
      return {reinterpret_cast<const uint8_t*>(s.data()), s.size()};
   }
   // Compile-time COW mode (psitri's upsert_mode pattern): the mode is a TEMPLATE parameter of
   // the whole mutation descent, never a runtime member — the unique fast path compiles with
   // zero shared-handling overhead and a stale latch is unrepresentable. `unique` = a node may
   // mutate in place when its own refcount is 1; `shared` = EVERY node on the path clones (a
   // ref==1 child below a shared node is still snapshot-reachable). Discovery of sharedness
   // (own ref>1 at a router entry, or a shared child range at a recursion site) selects the
   // make_shared() INSTANTIATION for the rest of that subtree. Extensible enum (psitri also
   // encodes op kinds); keep it a literal class so it can be a C++20 NTTP.
   struct upsert_mode
   {
      enum kind : uint8_t
      {
         unique = 0,
         shared = 1
      };
      kind k = unique;
      constexpr bool        is_shared() const noexcept { return k == shared; }
      constexpr upsert_mode make_shared() const noexcept { return {shared}; }
      constexpr bool        operator==(const upsert_mode&) const noexcept = default;
   };

   inline int common_prefix_len(bytes a, bytes b) noexcept
   {
      int n = int(std::min(a.size(), b.size()));
      int i = 0;
      while (i < n && a[size_t(i)] == b[size_t(i)])
         ++i;
      return i;
   }

   class radix_tree;

   /// An owning reference to a tree that is safe to copy, move, and drop on ANY thread,
   /// with no session and no lock. A `tree` is bound to the writer's session and a cursor to
   /// its reader's; this is what crosses between them: the writer shares what it built
   /// (write_session::share), any thread keeps it alive, and each reader thread opens its own
   /// cursor on it (read_session::create_cursor). Copies add a reference directly on the
   /// control block; the last drop is handed to the background release thread, so no destroy
   /// cascade ever runs on the dropping thread. The tree it names is immutable: holding a
   /// reference forces the writer to copy-on-write.
   class shared_tree
   {
      radix_tree*      _t    = nullptr;
      sal::ptr_address _root = sal::null_ptr_address;
      friend class radix_tree;
      friend class read_session;
      friend class write_session;
      struct adopt_reference
      {
      };
      shared_tree(radix_tree* t, sal::ptr_address root, adopt_reference) noexcept
          : _t(t), _root(root)
      {
      }
      shared_tree(radix_tree* t, sal::ptr_address root);  // adds a reference
      void drop() noexcept;

     public:
      shared_tree() = default;
      shared_tree(const shared_tree& other) : shared_tree(other._t, other._root) {}
      shared_tree(shared_tree&& other) noexcept : _t(other._t), _root(other._root)
      {
         other._root = sal::null_ptr_address;
      }
      shared_tree& operator=(const shared_tree& other)
      {
         if (this != &other)
            *this = shared_tree(other);
         return *this;
      }
      shared_tree& operator=(shared_tree&& other) noexcept
      {
         if (this != &other)
         {
            drop();
            _t          = other._t;
            _root       = other._root;
            other._root = sal::null_ptr_address;
         }
         return *this;
      }
      ~shared_tree() { drop(); }
      /// False for the empty tree.
      explicit operator bool() const noexcept { return _root != sal::null_ptr_address; }
      void     reset() noexcept { drop(); }
   };

   class radix_tree
   {
     public:
      class read_handle;  // internal non-owning view over a root
      class tree;         // psitri-compatible OWNING tree handle (get_root/get_subtree/get_tree)
      class tree_handle;  // mutable subtree editor (open_subtree/create_subtree, bottom-up commit)
      class cursor;
      class transaction;
      class frame;
      class subtree;
      explicit radix_tree(const std::filesystem::path& dir,
                          bool                         wipe      = true,
                          bool                         read_only = false,
                          sal::sync_type               sync      = sal::sync_type::none)
          : radix_tree(dir, wipe, config_of(read_only, sync))
      {
      }
      /// `recovery` selects what to rebuild before the store is used. The default,
      /// recovery_mode::none, decides from the store's shutdown record and from the
      /// durability the configuration asks for:
      ///   - clean close: nothing.
      ///   - the owning process died: app_crash (reference counts, range counters,
      ///     and segment bookkeeping; cost follows live data, not store size).
      ///   - the system stopped, or there is no record: power_loss (rebuild from
      ///     flushed segments; cost follows store size) ONLY when cfg.sync_mode is
      ///     fsync or stronger. A store that never asked for power-loss durability
      ///     is not charged for it and gets app_crash.
      /// deferred_cleanup opens at once and recovers nothing: what the dead owner
      /// held stays leaked.
      radix_tree(const std::filesystem::path& dir,
                 bool                         wipe,
                 const sal::runtime_config&   cfg,
                 sal::recovery_mode           recovery = sal::recovery_mode::none)
          : _prior_shutdown(prior_shutdown(dir, wipe)),
            _owned_alloc(make_alloc(dir, wipe, cfg)),
            _alloc(_owned_alloc.get()),
            _root_writers(std::make_shared<root_writers>()),
            _ses(_alloc->create_session()),
            _sync(cfg.sync_mode)
      {
         _alloc->register_type_ops<leaf_node>();
         _alloc->register_type_ops<setlist_u8>();
         _alloc->register_type_ops<setlist_u16>();
         _alloc->register_type_ops<setlist_u32>();
         _alloc->register_type_ops<bitset_c1_32>();
         _alloc->register_type_ops<bitset_c1_64>();
         _alloc->register_type_ops<bitset_c1_128>();
         _alloc->register_type_ops<bitset_r2_node>();
         _alloc->register_type_ops<bitset_r4_node>();
         _alloc->register_type_ops<full_radix_node>();
         _alloc->register_type_ops<bucket_node>();
         _alloc->register_type_ops<leaf_page>();
         _alloc->register_type_ops<key_val_node>();
         _alloc->register_type_ops<value_chunk>();
         // NULL IS A VALID EMPTY TREE — everywhere. No eager root: the direct-API working root
         // (`_root`, an OWNING smart_ptr — psitri tree_context's discipline) starts empty and
         // upsert_raw lazily creates on first write; an emptied tree returns to null. (The
         // database layer's published roots live in SAL's root table, likewise null until the
         // first commit.)
         // Start SAL's background threads ONLY AFTER the register_type_ops calls above —
         // exactly as psitri's database does (it also finishes RECOVERY first). The threads
         // are NOT waiting on this thread's session: create_session() is an explicit owned session
         // (each thread claims its own slot atomically), so session order is independent —
         // verified empirically, both orders clean under a debug build's ownership checks.
         // What actually forces the ordering is _type_ops: registration is a plain pointer
         // store into a non-atomic array the compactor/release/decay loops read on every
         // object they touch. Threads running before registration = a data race, and on a
         // reopened database the compactor would MOVE live heart nodes using the default
         // alloc_header ops (wrong visit/checksum/move semantics) — silent corruption.
         //
         // RECOVERY runs in the same window: after registration (it walks nodes through
         // their type ops) and before any thread or caller can observe stale reference
         // counts, range counters, or segment state left by an owner that never closed.
         recover_before_use(dir, cfg, recovery);
         _alloc->start_background_threads();
         SAL_CRASH_POINT("open:threads_started");
      }

      /// A writer's own view of an open store: the same allocator, roots and settings as
      /// `owner`, with its own allocator session and its own per-operation state. Each
      /// write session has one, so write sessions never share mutable state and may run
      /// on different threads at the same time. The owner must outlive it.
      struct writer_context
      {
      };
      radix_tree(writer_context, radix_tree& owner)
          : _store(owner._store),
            _alloc(owner._alloc),
            _root_writers(owner._root_writers),
            _ses(_alloc->create_session()),
            _sync(owner._sync),
            _page_mode(owner._page_mode),
            _wide_stems(owner._wide_stems),
            _bucket_mode(owner._bucket_mode),
            _adaptive(owner._adaptive),
            _c1_promote_on(owner._c1_promote_on)
      {
      }
      /// The engine the database opened. References that may outlive a write session, such
      /// as shared trees, refer to it and not to the session's writer context.
      radix_tree* store() noexcept { return _store; }
      // The writer lock of one root: held by a transaction from its start until it commits
      // or aborts. A transaction may be finished by another thread than the one that
      // started it (a session is not thread-safe, but it may move between threads when its
      // use is serialized), and a std::mutex may only be unlocked by the thread that locked
      // it. This lock has no owner, so any thread may release it: one word that is free,
      // held, or held with someone waiting. Taking and releasing it uncontended is one
      // atomic operation each and never a system call (a std::binary_semaphore's release
      // wakes through the kernel every time, waiter or not). One per cache line, so
      // writers of neighbouring roots do not share one.
      struct alignas(64) writer_gate
      {
         std::atomic<uint32_t> state{0};  // 0 free, 1 held, 2 held and at least one waiter
         bool try_lock() noexcept
         {
            uint32_t expected = 0;
            return state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                                 std::memory_order_relaxed);
         }
         void lock() noexcept
         {
            if (try_lock())
               return;
            // Mark the gate contended and sleep while it stays so; whoever gets it this way
            // leaves it marked, since others may still be waiting.
            while (state.exchange(2, std::memory_order_acquire) != 0)
               state.wait(2, std::memory_order_relaxed);
         }
         void unlock() noexcept
         {
            if (state.exchange(0, std::memory_order_release) == 2)
               state.notify_one();
         }
      };

      // Roots held by this writer context's own open transactions. A second claim on
      // one of them by the same context would wait for a lock the context itself holds,
      // which never ends, so it is refused instead. The record belongs to the context (a
      // write session), not to a thread: a session may move between threads.
      void claim_root(uint32_t idx)
      {
         for (uint32_t held : _held_roots)
            if (held == idx)
               throw std::logic_error("root " + std::to_string(idx) +
                                      " is already held by an open transaction of this write session");
         _held_roots.push_back(idx);
      }
      void unclaim_root(uint32_t idx) noexcept
      {
         for (auto& held : _held_roots)
            if (held == idx)
            {
               held = _held_roots.back();
               _held_roots.pop_back();
               return;
            }
      }
      /// The writer lock of root `idx`.
      writer_gate& root_writer(uint32_t idx)
      {
         if (idx >= _root_writers->mutex.size())
            throw std::out_of_range("root index out of range");
         return _root_writers->mutex[idx];
      }

      // Radix + bucket hybrid: leaf -> bucket-of-leaves -> radix with bucket/leaf branches.
      // The tree "starts as a bucket"; a full bucket splits into a radix node (the existing
      // router tiers) with bucket leaves. Fewer hops than pure radix (a bucket absorbs a
      // whole sparse terminal region), so it reclaims dict rand/miss vs psitri. Call on an
      // empty tree. (in-place mutation; COW/MVCC is a later add.)
      void set_bucket_mode(bool on)
      {
         _page_mode = false;  // legacy-engine knob (see set_wide_stems)
         if (on == _bucket_mode)
            return;
         _bucket_mode = on;
         // Bucket mode needs its page root up front; radix mode starts null (= empty tree).
         // smart_ptr assignment releases the previous root.
         _root = sal::smart_ptr<sal::alloc_header>(
             _ses, on ? make_bucket() : sal::ptr_address(sal::null_ptr_address));
      }

      // ── public API (psitri key_view/value_view; values are arbitrary bytes) ───
      // psitri tree_context's root discipline: `_root` is an OWNING smart_ptr; mutation takes
      // the address out, runs the COW (which releases the pre-image on change), and re-adopts.
      int64_t upsert(key_view key, value_view val)
      {
         _ins_value_type = inline_value::vt_bytes;
         // Bucket mode is the WHOLE-TREE hybrid (bucket leaves at any depth; a full bucket
         // splits into a radix router with bucket children, and insert_node's bucket case
         // handles buckets met during any descent). This branch is just the main tree's entry
         // into the bucket engine (bkt_insert: root-level bucket handling/splits) — the same
         // scoping the old `&root_ref == &_root` identity check expressed before _root became
         // a smart_ptr.
         if (_bucket_mode)
         {
            check_key_contract(key.size());  // bkt_insert recurses — enforce at the entry
            auto        lock = _ses->lock();
            ptr_address w    = _root.take();
            _root = sal::smart_ptr<sal::alloc_header>(_ses, bkt_insert(w, as_bytes(key), 0,
                                                                       as_bytes(val)));
            return _old_value_size;
         }
         mutate_root(_root,
                     [&](ptr_address pre) { return upsert_step(pre, as_bytes(key), as_bytes(val)); });
         return _old_value_size;
      }

      // ── typed values ────────────────────────────────────────────────────────────────────
      // A typed value is a type number plus a small trivially copyable token (value_hooks.hpp).
      // The engine writes [u16 typenum][token] under its own value kind (vt_typed); the type
      // number is never part of a caller's bytes, and a plain-bytes value is never inspected
      // for one. Storing CONSUMES one reference of the token's owner (no retain here); the
      // type's hooks fire as the stored bytes are duplicated or die.
      struct typed_stash
      {
         uint8_t buf[2 + max_typed_token_size];
         bytes   stored;
      };
      void stage_typed(typed_stash& st, uint16_t typenum, value_view token)
      {
         if (typenum == kRawType || typenum == 0xFFFF)
            throw std::invalid_argument(
                "sal_art_ref: type numbers 0 (plain bytes) and 65535 are reserved");
         if (token.size() > max_typed_token_size)
            throw std::invalid_argument("sal_art_ref: typed value token exceeds " +
                                        std::to_string(max_typed_token_size) + " bytes");
         if (!_page_mode || _bucket_mode)
            throw std::logic_error("sal_art_ref: typed values need the page model");
         std::memcpy(st.buf, &typenum, 2);
         if (!token.empty())
            std::memcpy(st.buf + 2, token.data(), token.size());
         st.stored = bytes{st.buf, 2 + token.size()};
      }
      [[nodiscard]] ptr_address typed_upsert_step(ptr_address root, key_view k, uint16_t typenum,
                                                  value_view token)
      {
         typed_stash st;
         stage_typed(st, typenum, token);
         _ins_value_type = inline_value::vt_typed;
         struct restore
         {
            uint8_t& v;
            ~restore() { v = inline_value::vt_bytes; }
         } r{_ins_value_type};
         return upsert_step(root, as_bytes(k), st.stored);
      }
      int64_t upsert_typed(key_view key, uint16_t typenum, value_view token)
      {
         mutate_root(_root, [&](ptr_address pre)
                     { return typed_upsert_step(pre, key, typenum, token); });
         return _old_value_size;
      }

      // ── first-class subtree values ──────────────────────────────────────────────────────
      // A subtree value: the value stored at `key` IS a nested tree (root `subtree`), not bytes.
      // Encoded as a vt_subtree terminal (value bytes = the 4-byte root); the terminal owns one
      // ref on the root, released when the terminal dies (leaf_node::destroy → cascade).
      //
      // create_subtree() hands back an empty root you fill via a write handle, then attach with
      // upsert_subtree(key, handle.root()). All subtree roots live in THIS tree's allocator.
      // RAII builder for a detached subtree root. Owns one ref on its root; attaching with
      // upsert_subtree retains a second, so when the handle dies the parent terminal is the sole
      // owner. A handle never attached releases its root → the subtree is freed (no leak).
      class subtree
      {
         friend class radix_tree;
         friend class transaction;
         friend class frame;
         radix_tree*                       _t = nullptr;
         sal::smart_ptr<sal::alloc_header> _root;  // RAII: owns the detached root until attached
         subtree(radix_tree* t, ptr_address r) noexcept : _t(t), _root(t->session(), r) {}  // adopt
         ptr_address root() const noexcept { return _root.address(); }       // internal
         ptr_address consume() noexcept { return _root.take(); }             // give ref to a terminal

        public:
         subtree()                          = default;
         subtree(subtree&&)                 = default;  // smart_ptr move
         subtree(const subtree&)            = delete;
         subtree& operator=(const subtree&) = delete;
         // Insert/update a key in the subtree (COW; the root may change → re-adopt it).
         int64_t upsert(key_view k, value_view v)
         {
            _t->_ins_value_type = inline_value::vt_bytes;
            _t->mutate_root(_root, [&](ptr_address pre)
                            { return _t->upsert_step(pre, as_bytes(k), as_bytes(v)); });
            return _t->_old_value_size;
         }
      };
      subtree create_subtree()
      {
         return subtree(this, _page_mode ? make_page() : make_node<setlist_u8>(bytes{}));
      }

      // Attach a built subtree under `key` (consumes the handle's ref → the terminal owns it).
      int64_t upsert_subtree(key_view key, subtree&& s)
      {
         ptr_address sub = s.consume();
         mutate_root(_root, [&](ptr_address pre) { return attach_subtree_step(pre, key, sub); });
         return _old_value_size;
      }
      // Is the value at `key` a subtree?  get_subtree returns it as a handle (falsy if absent).
      bool        is_subtree(key_view key);   // defined out-of-line (needs cursor)
      read_handle get_subtree(key_view key);  // defined out-of-line (needs cursor)

     private:
      // ── THE ROOT-TRANSITION DISCIPLINE (psitri tree_context) ───────────────────────────────
      // Every mutating entry owns the pre-image as a smart_ptr moved OUT of the root, runs the
      // descent against the borrowed address, restores the root on exception, and adopts the
      // result via give(). The pre-image's release is STRUCTURAL — the displaced smart_ptr's
      // destructor — never a manual _ses->release() in the entry, so no path (including throws)
      // can orphan or double-release a root. `op` is ptr_address pre → ptr_address post; the
      // *_step descents below never release the pre-image themselves.
      template <class Op>
      void mutate_root(sal::smart_ptr<sal::alloc_header>& root, Op&& op)
      {
         auto        old = std::move(root);  // own the pre-image; `root` is empty during the op
         ptr_address nr;
         try
         {
            nr = op(old.address());
         }
         catch (...)
         {
            root = std::move(old);  // restore on exception (psitri's guard)
            throw;
         }
         SAL_CRASH_POINT("mutate:descended");
         if (nr != old.address())
         {
            // Adopt the new root bound to THE ENGINE SESSION (a default-constructed tree{}'s
            // smart_ptr carries no session — its first write must bind one, or the eventual
            // release dereferences null). `old` releases the pre-image on scope exit.
            root = sal::smart_ptr<sal::alloc_header>(_ses, nr);
         }
         else
            root = std::move(old);  // unchanged / mutated in place — put it back
      }

      // The 1024-byte key contract is load-bearing: route/divider/rebuild buffers throughout
      // the engine are sized to max_key_len + 1. Enforced in EVERY build at the mutating
      // descents — under NDEBUG an oversized key would overflow those buffers (silent stack
      // corruption), so it is rejected as bad input, never trusted as an invariant. The throw
      // is safe at every call site: mutate_root restores the root on exception.
      // An object reached as a router is not one of the router kinds: the store is damaged
      // or the engine has a bug. Following it as some other kind would read and write
      // through the wrong layout, so the mutating descents stop here in every build. (The
      // read-side dispatch, dispatch_hdr, still tells the compiler the case cannot happen:
      // it is on the path of every lookup, and what a checked default costs there is to be
      // measured before it is added.)
      [[noreturn, gnu::cold, gnu::noinline]] static void unknown_node_kind() noexcept
      {
         std::fputs("sal_art_ref: an object in the tree is not a node the engine can route "
                    "through; stopping\n",
                    stderr);
         std::abort();
      }

      static void check_key_contract(size_t klen)
      {
         if (klen > key_val_node::max_key_len) [[unlikely]]
            throw std::invalid_argument("heart: key exceeds max_key_len (1024 bytes)");
      }

      // Shared insert descent for upsert/upsert_subtree/subtree::upsert (value-type already
      // stashed). BORROWS `root` (the caller owns the pre-image; see mutate_root) and returns
      // the post-image root — never releases the pre-image. A null root is the valid empty
      // tree: the descent creates the first node lazily.
      [[nodiscard]] ptr_address upsert_step(ptr_address root, bytes kb, bytes val)
      {
         if (val.size() > (_page_mode ? large_value_detail::max_value_size : 65535)) [[unlikely]]
            throw std::invalid_argument("heart: value exceeds this layout's maximum value size");
         check_key_contract(kb.size());
         auto lock       = _ses->lock();
         _old_value_size = -1;
         if (_page_mode)  // radix-indexed leaf pages (the default data layout)
            return page_upsert_step(root, kb, val);
         // The descent starts UNIQUE; sharedness is discovered per node/range and selects the
         // shared INSTANTIATION (upsert_mode is a template parameter — no runtime latch).
         if (root == sal::null_ptr_address)  // null = empty tree → first write creates the root
         {
            // Own the first node structurally: if the insert outgrows it the smart_ptr's
            // destructor releases the intermediate; otherwise hand the ref to the caller.
            sal::smart_ptr<sal::alloc_header> base(_ses, make_node<setlist_u8>(bytes{}));
            ptr_address nr = insert_node<upsert_mode{}>(base.address(), kb, 0, val, {});
            return nr == base.address() ? base.take() : nr;
         }
         return insert_node<upsert_mode{}>(root, kb, 0, val, {});
      }

     public:

      bool find(key_view key, value_view& out)
      {
         auto  lock = _ses->lock();
         bytes kb   = as_bytes(key);
         if (_bucket_mode)
            return bkt_find(kb, out);
         if (!_root)  // null = empty tree
            return false;
         if (_page_mode)
            return page_find_in(*_ses, _root.address(), kb, out);
         return find_in(*_ses, _root.address(), kb, out);
      }
      // The NATIVE iterative point lookup (skip-header dispatch + prefetch) on an ARBITRARY
      // pinned root via an ARBITRARY session — STATIC: no engine state, so reader-session
      // cursors share this exact hot path with the writer-thread find(). Returns false on a
      // miss AND for subtree values (use is_subtree()/get_subtree()).
      static bool find_in(sal::allocator_session& ses, ptr_address addr, bytes kb, value_view& out)
      {
         uint32_t depth = 0;
         for (;;)
         {
            auto    ref   = ses.get_ref<sal::alloc_header>(addr);
            uint8_t shape = ref.cb_type();
            if (shape == uint8_t(dispatch_shape::r8) || shape == uint8_t(dispatch_shape::r2) ||
                shape == uint8_t(dispatch_shape::r4))  // SKIP-HEADER: route off the segment line
            {
               const char* base = reinterpret_cast<const char*>(ref.obj());
               if (depth < kb.size())
               {
                  // Address-generate BEFORE switching on the algo. The key bits consumed
                  // before indexing the segment line differ per tier (r2:7 r4:6 r8:5), but
                  // the *line* is the same closed form for all three — so we compute the
                  // exact cacheline the algo will dereference, then PREFETCH it (never a
                  // line we won't read), then switch only on the in-line algo (r8 direct
                  // index vs r2/r4 popcount-rank). The segment-line miss overlaps the jump.
                  static constexpr uint8_t skip_shift[4] = {0, 7, 6, 5};  // [r1] r2 r4 r8
                  uint8_t                  byte          = kb[depth];
                  const char*              cl = base + 128 + (size_t(byte >> skip_shift[shape]) << 7);
                  __builtin_prefetch(cl);
                  smart_address sa;
                  if (shape == uint8_t(dispatch_shape::r8))
                     std::memcpy(&sa, cl + 4u * (byte & 31), sizeof(sa));
                  else if (shape == uint8_t(dispatch_shape::r2))
                     sa = reinterpret_cast<const bitset_r2_node::segment*>(cl)->find(byte & 127);
                  else
                     sa = reinterpret_cast<const bitset_r4_node::segment*>(cl)->find(byte & 63);
                  if (sa.is_null())
                     return false;
                  if (sa.is_tail())
                     return inline_at(ref.obj(), sa, kb.subspan(depth + 1), out);
                  addr = sa.as_pointer();
                  ++depth;
                  continue;
               }
               smart_address tv = (shape == uint8_t(dispatch_shape::r8))
                                      ? ref.as<full_radix_node>().obj()->term_value()
                                      : (shape == uint8_t(dispatch_shape::r2)
                                             ? ref.as<bitset_r2_node>().obj()->term_value()
                                             : ref.as<bitset_r4_node>().obj()->term_value());
               return term_lookup(ses, ref.obj(), tv, kb, depth, out);
            }
            // R1: read CL0, dispatch by format. ONE get_ref resolves the node; the switch
            // (jump table) reinterprets that single resolved pointer per case (alloc_header is
            // at offset 0 of every node) — no second per-case get_ref on the same address.
            bool                     cont  = false;
            bool                     found = false;
            const sal::alloc_header* b     = ses.get_ref<sal::alloc_header>(addr).obj();
            switch (kind_of(b->type()))
            {
               case radix_kind::leaf:
                  return leaf_read(ses, addr, kb, depth, out);
               case radix_kind::bucket:  // a bucket LEAF: search its key suffix
               {
                  const inline_value* iv = reinterpret_cast<const bucket_node*>(b)->find(
                      kb.data() + depth, int(kb.size() - depth));
                  if (!iv)
                     return false;
                  out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                   iv->value_len);
                  return true;
               }
               case radix_kind::setlist_u8:
                  cont = route_r1(ses, reinterpret_cast<const setlist_u8*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::setlist_u16:
                  cont = route_r1(ses, reinterpret_cast<const setlist_u16*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::setlist_u32:
                  cont = route_r1(ses, reinterpret_cast<const setlist_u32*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::c1_32:
                  cont = route_r1(ses, reinterpret_cast<const bitset_c1_32*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::c1_64:
                  cont = route_r1(ses, reinterpret_cast<const bitset_c1_64*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::c1_128:
                  cont = route_r1(ses, reinterpret_cast<const bitset_c1_128*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::bitset_r2:
                  cont = route_r1(ses, reinterpret_cast<const bitset_r2_node*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::bitset_r4:
                  cont = route_r1(ses, reinterpret_cast<const bitset_r4_node*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::full:
                  cont = route_r1(ses, reinterpret_cast<const full_radix_node*>(b), kb, depth, out, addr, found);
                  break;
               case radix_kind::page:  // page-model trees route via page_find_in, never here
                  assert(false && "find_in on a page-model tree (use page_find_in)");
                  return false;
               case radix_kind::bitset1:
                  unknown_node_kind();  // not produced by the engine
            }
            if (!cont)
               return found;
         }
      }

      // ── remove ──────────────────────────────────────────────────────────────────
      // Delete `key` if present; returns true iff it was found+removed. COW-to-root just
      // like upsert (unique → mutate in place; shared/snapshot → clone the path, the old
      // tree stays valid for the snapshot). Releases every SAL object the deletion frees
      // — external leaf/subtree values and any node that drains empty — with no leak and
      // no double-free (mirrors insert's clone-retains-children / caller-releases-original
      // contract). An inline-tail value's bytes stay in the node's tail heap (intra-node
      // space, reclaimed on the next clone); that is not a refcount leak. A node that loses
      // its last entry is dropped from its parent and freed, propagating to the root; a
      // fully drained tree returns to null (the valid empty tree).
      bool remove(key_view key)
      {
         if (_bucket_mode)
         {
            check_key_contract(key.size());
            auto lock = _ses->lock();
            return bkt_remove(as_bytes(key));
         }
         bool removed = false;
         mutate_root(_root, [&](ptr_address pre) { return remove_step(pre, as_bytes(key), removed); });
         return removed;
      }
      // COW-to-root remove descent (radix mode). BORROWS `root` (the caller owns the pre-image;
      // see mutate_root) and returns the post-image root — null when the tree fully drained
      // (null IS the valid empty tree). Sets `removed` iff the key was found+deleted. Never
      // releases the pre-image; a cloned-then-drained root (a descent-internal intermediate)
      // is released here.
      [[nodiscard]] ptr_address remove_step(ptr_address root, bytes kb, bool& removed)
      {
         check_key_contract(kb.size());
         auto lock = _ses->lock();
         _removed  = false;
         _emptied  = false;
         removed   = false;
         if (_page_mode)
            return page_remove_step(root, kb, removed);
         if (root == sal::null_ptr_address)
            return root;
         ptr_address nr = remove_node<upsert_mode{}>(root, kb, 0);  // unique start; mode discovered
         removed         = _removed;
         if (!_removed)
            return root;
         if (_emptied)  // whole tree drained → the post-image is the empty (null) tree
         {
            if (nr != root)
               _ses->release(nr);  // cloned-then-drained intermediate (pre-image is the caller's)
            return sal::null_ptr_address;
         }
         return nr;
      }
      // Root-parameterized counted range-remove (radix mode), psitri semantics: delete every key in
      // [lower, upper) and return how many; `upper == max_key` means unbounded; an inverted/empty
      // range removes nothing. A removed subtree value cascades (and counts as ONE key of THIS
      // tree). v1 is a remove-the-front loop — re-seek lower_bound(lower), point-remove — i.e.
      // O(removed · depth), not psitri's recursive whole-subtree drop. The cursor is scoped to die
      // BEFORE each remove so the working root stays uniquely owned (in-place delete, no COW churn).
      uint64_t remove_range_counted_step(sal::smart_ptr<sal::alloc_header>& root, key_view lower,
                                         key_view upper)
      {
         if (!root)
            return 0;
         if (lower >= upper && upper != max_key)
            return 0;  // empty range
         if (upper == max_key)
            upper = {};  // internal unbounded representation
         uint64_t removed = 0;
         uint8_t  kb[1024];  // stack copy of the victim key (cursor keys are <= 1024 B; no heap)
         mutate_root(root, [&](ptr_address pre) {
            ptr_address cur = pre;  // borrowed; intermediate roots are owned (released) in-loop
            for (;;)
            {
               size_t klen = 0;
               {
                  cursor c(this, cur);  // pins cur — scoped: released before the remove
                  if (!(lower.empty() ? c.seek_begin() : c.lower_bound(lower)))
                     break;  // nothing >= lower
                  key_view k = c.key();
                  if (!upper.empty() && !(k < upper))
                     break;  // first candidate is already past the range
                  klen = k.size();
                  std::memcpy(kb, k.data(), klen);
               }
               bool        rem = false;
               ptr_address nxt = remove_step(cur, bytes{kb, klen}, rem);
               if (!rem)
                  break;  // defensive: the key just seen must be removable (single writer)
               if (nxt != cur && cur != pre)
                  _ses->release(cur);  // intermediate post-image displaced by this iteration
               cur = nxt;
               ++removed;
            }
            return cur;  // mutate_root releases `pre` iff cur != pre
         });
         return removed;
      }
      // Subtree-attach descent: store a vt_subtree terminal at `k` whose value is the subtree
      // root's 4 address bytes; the terminal takes ownership of the caller's DONATED ref (no
      // retain here). BORROWS `root` like every *_step (the caller owns the pre-image; null is
      // the valid empty tree). Shared stash discipline: _ins_value_type set and restored.
      // `sub` arrives as a reference its caller gave up (taken out of its owning pointer):
      // it is this function's until a record holds it. If the insert throws (the key is
      // refused, an allocation fails) the subtree is released here; nobody else can.
      [[nodiscard]] ptr_address attach_subtree_step(ptr_address root, key_view k, ptr_address sub)
      {
         sal::smart_ptr<sal::alloc_header> held(_ses, sub);
         struct subtree_kind  // the stashed value kind does not outlive this call, throw or not
         {
            uint8_t& kind;
            explicit subtree_kind(uint8_t& k) : kind(k) { kind = inline_value::vt_subtree; }
            ~subtree_kind() { kind = inline_value::vt_bytes; }
         } as_subtree{_ins_value_type};
         uint32_t    raw = *sub;
         ptr_address nr =
             upsert_step(root, as_bytes(k), bytes{reinterpret_cast<const uint8_t*>(&raw), sizeof(raw)});
         (void)held.take();  // stored: the record owns it now
         return nr;
      }

      // ── ordered queries (this is a SORTED container, not a hash map) ────────────
      // lower_bound: smallest key >= `key`; upper_bound (mathematical floor): the
      // LARGEST key v <= `key` (the dual of lower_bound). On a hit, out_key holds
      // the matched key (materialized) and out_val views its value in node memory
      // (valid like find()'s result). Returns false if none.
      bool lower_bound(key_view key, std::string& out_key, value_view& out_val)
      {
         return ordered_seek(key, false, out_key, out_val);
      }
      bool upper_bound(key_view key, std::string& out_key, value_view& out_val)
      {  // floor via the cursor (now floor); reference path — not perf-critical
         cursor c(this);
         if (!c.upper_bound(key))
            return false;
         out_key.assign(c.key().data(), c.key().size());
         out_val = legacy_cursor_value(c);
         return true;
      }
      // Visit every (key, value) in ascending key order: cb(key_view, value_view).
      template <class Cb>
      void for_each_inorder(Cb&& cb)
      {
         auto lock = _ses->lock();
         KB   kb;
         inorder_walk(_root.address(), kb, cb);
      }
      uint64_t object_count() const { return _alloc->total_allocated_objects(); }
      /// How the previous read-write owner left this store, as found at open.
      sal::shutdown_state prior_shutdown() const noexcept { return _prior_shutdown; }
      /// The recovery this open ran (recovery_mode::none when it ran none).
      sal::recovery_mode recovery_performed() const noexcept { return _recovered; }
      // Live count is exact only once the background release thread has drained — releases of
      // a sole-referenced object are deferred to it. Tests poll this to a quiescent point.
      uint64_t pending_releases() const { return _alloc->total_pending_releases(); }

      // ── gated wide-stem feature ────────────────────────────────────────────────
      // When off (default), the growth algorithm only builds setlist_u8 → the engine is
      // the libart-conformant byte radix. When on, the growth algorithm may pick
      // setlist_u16/u32 at the existing build points (fewer hops → fewer CB indirections).
      // Routing is already stride-aware; nothing else in the engine changes.
      // Selecting ANY legacy growth knob (wide stems / adaptive / c1-promote / bucket mode)
      // opts the tree out of the page model into the legacy byte-radix engine — those knobs
      // shape machinery that only routes data there.
      void set_wide_stems(bool on)
      {
         _page_mode  = false;
         _wide_stems = on;
      }
      bool wide_stems() const { return _wide_stems; }
      // Page model (the default): keys live in 2KB leaf_page terminals behind a lower-bound
      // divider radix. Off = the legacy byte-consuming radix.
      void set_page_mode(bool on) { _page_mode = on; }
      bool page_mode() const { return _page_mode; }

      // Adaptive radix↔bucket: start radix; when insert descent crosses 2 levels that each
      // have < 3 branches (children + term) — i.e. going deep without widening, the signature
      // of clustered/non-radix-friendly data caught while the region is still a provably small
      // number of keys — collapse that subtree into a bucket leaf. Wide (radix-friendly) data
      // never trips it. Off by default so the gated baseline stays libart-exact.
      void set_adaptive(bool on)
      {
         _page_mode = false;  // legacy-engine knob (see set_wide_stems)
         _adaptive  = on;
      }
      bool adaptive() const { return _adaptive; }
      void set_c1_promote(bool on)
      {
         _page_mode     = false;  // legacy-engine knob (see set_wide_stems)
         _c1_promote_on = on;
      }
      // ── direct-API hooks used by the engine's own tests (raw roots, no safety rails) ──
      // [testing] A SNAPSHOT IS A smart_ptr COPY — the psitri definition: copying retains,
      // destruction releases. Hold the returned smart_ptr to pin this version of the tree;
      // drop it (scope/assignment) to release. No engine-held pin state, no manual unpin.
      sal::smart_ptr<sal::alloc_header> root_snapshot() const { return _root; }
      // ── introspection (for the tree-structure dump / stats tool) ───────────────
      // COW/alloc accounting: how mutation actually played out over the run. `clones`
      // and `grows` are the COW-to-root cost (a clone forced by sharing or a full heap);
      // `in_place` is the cheap path (unique node, room to spare).
      // Hot-path COW/alloc accounting. Each field is a sal::stat_counter: a real
      // uint64_t under -DPSITRI_STATS, otherwise a zero-cost stand-in whose ++/+=
      // inline to nothing and whose reads yield 0 (so this struct, counters(), and
      // every ++_cc.* site below all compile unchanged — only the work disappears).
      struct cow_counters
      {
         sal::stat_counter node_allocs   = 0;   // routers built
         sal::stat_counter leaf_allocs   = 0;   // >60 SAL value-holders
         sal::stat_counter inline_values = 0;   // tail (local) terminal values
         sal::stat_counter clones        = 0;   // COW clones (shared OR heap-full)
         sal::stat_counter child_retains = 0;   // per-child retains done by clone() (the dense-clone cost)
         sal::stat_counter grows         = 0;   // subset of clones forced by a full tail heap
         sal::stat_counter in_place      = 0;   // unique + room: mutated in place (no copy)
         sal::stat_counter restrides     = 0;   // wide-node splits (re_stride to a narrower stride)
         sal::stat_counter upgrades      = 0;   // grow_and_set tier widenings (setlist→c1/r2/r4/full)
         sal::stat_counter page_splits   = 0;   // leaf_page splits (page model)
      };
      const cow_counters&        counters() const { return _cc; }
      sal::allocator_session_ptr session() const { return _ses; }
      // Commit durability for this tree, set at open from the adb commit config and used by
      // every publish()/set_root — without a real sync the commit record isn't written, so
      // segments never seal and the file grows by the full COW-garbage rate.
      sal::sync_type sync_mode() const noexcept { return _sync; }
      ptr_address                root_addr() const { return _root.address(); }

      struct tier_counts
      {
         uint64_t leaf = 0, setlist = 0, setlist_w16 = 0, setlist_w32 = 0;
         uint64_t bitset_c1 = 0, bitset_r2 = 0, bitset_r4 = 0, full = 0;
         // page model: 2KB leaf_page terminals + their entry total (avg keys/page =
         // page_entries / pages) + the dynamic prefix-column width census (1/2/4 bytes).
         uint64_t pages = 0, page_entries = 0;
         uint64_t pages_w1 = 0, pages_w2 = 0, pages_w4 = 0;
         // compact_addresses sharing: branches / live cacheline bases across all bitset
         // segments (avg sharing factor s = cbranches / cbases; 1.0 = no sharing).
         uint64_t cbranches = 0, cbases = 0, csegs = 0;
         // POINTER-only sharing (excludes inline-tail terminals, which can't co-locate):
         // s_ptr = cptr / cptr_cl. ctail = inline-tail branches (local, not external CBs).
         uint64_t cptr = 0, cptr_cl = 0, ctail = 0;
         // Same breakdown for SETLIST nodes (sparse extreme; flat branches): pointer children
         // co-location per node. s_setlist = slptr / slptr_cl.
         uint64_t slptr = 0, slptr_cl = 0, sltail = 0;
         // setlist headroom utilization: sum of tail_used (inline-value bytes actually used)
         // vs the 256-byte headroom each setlist reserves. Measures recoverable memory.
         uint64_t sl_tail_used = 0;
         // full_radix (R8) occupancy: sum of occupied 64-wide quarters per full node (1-4).
         // avg = full_quarters / full → if < 4, presence could eliminate the empty quarters.
         uint64_t full_quarters = 0;
         // adaptive/bucket-mode dense terminal leaves (sorted (suffix,value) pages) + entries.
         uint64_t bucket = 0, bucket_entries = 0;
      };
      tier_counts count_tiers()
      {
         auto        lock = _ses->lock();
         tier_counts tc;
         walk(_root.address(), tc);
         return tc;
      }

     private:
      static constexpr int      max_inline   = 60;    // suffix+value cap for a tail leaf
      static constexpr uint16_t tail_reserve = 16;    // tail offsets >=16 → compact-safe
      static constexpr uint32_t max_node     = 4096;  // grow cap; beyond → widen encoding

      template <class N>
      void walk_kids(ptr_address a, tier_counts& tc)
      {
         _ses->get_ref<N>(a).obj()->for_each_child([&](ptr_address c) { walk(c, tc); });
      }
      // Sum branches + live cacheline bases over a bitset node's non-empty segments —
      // measures the real compact_addresses sharing factor (cbranches/cbases).
      template <class N>
      void accum_compact(ptr_address a, tier_counts& tc)
      {
         const N* n = _ses->get_ref<N>(a).obj();
         for (int s = 0; s < N::nseg; ++s)
         {
            const auto& seg = n->seg(s);
            const int   b   = seg.branches.size();
            if (b == 0)
               continue;
            tc.cbranches += uint64_t(b);
            tc.cbases += uint64_t(seg.branches.live_base_count());
            ++tc.csegs;
            // Classify each branch: inline tail (local) vs pointer (external CB). For
            // pointers, count DISTINCT cachelines (raw>>4) to get true co-location.
            uint32_t cls[256];
            int      ncl = 0;
            for (int i = 0; i < b; ++i)
            {
               const smart_address sa = seg.branches.at(i);
               if (sa.is_tail())
               {
                  ++tc.ctail;
                  continue;
               }
               ++tc.cptr;
               const uint32_t cl = sa.raw >> 4;
               bool           seen = false;
               for (int j = 0; j < ncl; ++j)
                  if (cls[j] == cl) { seen = true; break; }
               if (!seen)
                  cls[ncl++] = cl;
            }
            tc.cptr_cl += uint64_t(ncl);
         }
      }
      // Pointer-child CB co-location for a setlist node (per node): distinct cachelines among
      // its pointer children vs pointer count. s_setlist = slptr / slptr_cl.
      template <class N>
      void accum_setlist(ptr_address a, tier_counts& tc)
      {
         tc.sl_tail_used += _ses->get_ref<N>(a).obj()->tail_used();
         uint32_t cls[64];
         int      ncl = 0;
         _ses->get_ref<N>(a).obj()->for_each_entry([&](auto, smart_address br) {
            if (br.is_tail())
            {
               ++tc.sltail;
               return;
            }
            ++tc.slptr;
            const uint32_t cl = br.raw >> 4;
            for (int j = 0; j < ncl; ++j)
               if (cls[j] == cl)
                  return;
            if (ncl < 64)
               cls[ncl++] = cl;
         });
         tc.slptr_cl += uint64_t(ncl);
      }
      void walk(ptr_address a, tier_counts& tc)
      {
         if (a == sal::null_ptr_address)  // null = empty tree
            return;
         switch (kind_of(_ses->get_ref<sal::alloc_header>(a)->type()))
         {
            case radix_kind::leaf:
               ++tc.leaf;
               break;
            case radix_kind::setlist_u8:
               ++tc.setlist;
               accum_setlist<setlist_u8>(a, tc);
               walk_kids<setlist_u8>(a, tc);
               break;
            case radix_kind::setlist_u16:
               ++tc.setlist_w16;
               walk_kids<setlist_u16>(a, tc);
               break;
            case radix_kind::setlist_u32:
               ++tc.setlist_w32;
               walk_kids<setlist_u32>(a, tc);
               break;
            case radix_kind::c1_32:
               ++tc.bitset_c1;
               walk_kids<bitset_c1_32>(a, tc);
               break;
            case radix_kind::c1_64:
               ++tc.bitset_c1;
               walk_kids<bitset_c1_64>(a, tc);
               break;
            case radix_kind::c1_128:
               ++tc.bitset_c1;
               walk_kids<bitset_c1_128>(a, tc);
               break;
            case radix_kind::bitset_r2:
               ++tc.bitset_r2;
               accum_compact<bitset_r2_node>(a, tc);
               walk_kids<bitset_r2_node>(a, tc);
               break;
            case radix_kind::bitset_r4:
               ++tc.bitset_r4;
               accum_compact<bitset_r4_node>(a, tc);
               walk_kids<bitset_r4_node>(a, tc);
               break;
            case radix_kind::full:
            {
               ++tc.full;
               uint8_t q = 0;  // which 64-wide quarters have any branch
               _ses->get_ref<full_radix_node>(a).obj()->for_each_entry(
                   [&](uint8_t byte, smart_address) { q |= uint8_t(1u << (byte >> 6)); });
               tc.full_quarters += uint64_t(std::popcount(unsigned(q)));
               walk_kids<full_radix_node>(a, tc);
               break;
            }
            case radix_kind::bucket:  // adaptive dense terminal leaf — a counted endpoint
               ++tc.bucket;
               tc.bucket_entries += _ses->get_ref<bucket_node>(a).obj()->nent;
               break;
            case radix_kind::page:  // 2KB leaf page — a counted endpoint (like leaf/bucket)
            {
               const leaf_page* pg = _ses->get_ref<leaf_page>(a).obj();
               ++tc.pages;
               tc.page_entries += pg->num_keys();
               switch (pg->prefix_width())
               {
                  case 1: ++tc.pages_w1; break;
                  case 2: ++tc.pages_w2; break;
                  default: ++tc.pages_w4; break;
               }
               break;
            }
            case radix_kind::bitset1:  // not produced
               break;
         }
      }

      static runtime_config config_of(bool read_only, sal::sync_type sync)
      {
         runtime_config cfg;
         cfg.read_only = read_only;
         cfg.sync_mode = sync;
         return cfg;
      }

      static std::unique_ptr<sal::allocator> make_alloc(const std::filesystem::path& dir,
                                                        bool wipe, sal::runtime_config cfg)
      {
         if (wipe)
         {
            std::filesystem::remove_all(dir);
            std::filesystem::create_directories(dir);
         }
         // Explicit configuration reaches SAL intact. Existing benchmark environment
         // overrides still take precedence when enabled.
         if (env_config::get().read_cache)
            cfg.enable_read_cache = true;
         if (const uint64_t w = env_config::get().read_cache_window_sec)
            cfg.read_cache_window_sec = w;
         if (const uint64_t pinned = env_config::get().pinned_mb)
            cfg.max_pinned_cache_size_mb = pinned;
         return std::make_unique<sal::allocator>(dir, cfg, false);
      }

      // ── node construction + COW ────────────────────────────────────────────────
      template <class N>
      N* mut(ptr_address a)
      {
         return _ses->get_ref<N>(a).modify().get();
      }
      template <class N>
      void stamp(ptr_address a)
      {
         auto ref = _ses->get_ref<N>(a);
         ref.control().set_type(uint8_t(ref->stamp_shape()));  // r8 if prefix-less else r1
      }
      // A bucket is a LEAF (no skip-header routing) → it wants dispatch shape r1. alloc already
      // defaults the control type to r1 (0), so no explicit stamp is needed: the infrastructure
      // makes the safe shape the default, and only the wide tiers (set by make_node/clone/reprefix)
      // ever stamp away from it.
      ptr_address make_bucket() { return _ses->alloc<bucket_node>(sal::alloc_hint{}); }
      template <class N>
      ptr_address make_node(bytes pfx, sal::alloc_hint hint = {})
      {
         ptr_address a = _ses->alloc<N>(hint, pfx);
         N*          n = _ses->get_ref<N>(a).modify().get();
         n->alloc_tail(tail_reserve);  // tail offsets >=16
         rr_birth<N>(n);               // big tier → its per-range counters (empty node, =1 each)
         stamp<N>(a);
         ++_cc.node_allocs;
         return a;
      }
      ptr_address make_leaf(bytes suffix, bytes val, sal::alloc_hint hint = {})
      {
         ++_cc.leaf_allocs;
         ptr_address a = _ses->alloc<leaf_node>(hint, suffix, val);
         if (_ins_value_type != inline_value::vt_bytes)  // stamp the pending value-type (subtree)
            _ses->get_ref<leaf_node>(a).modify().get()->value.value_type = _ins_value_type;
         return a;
      }

      // Copy src's payload into dst (same N): head [alloc_header, head_bytes()) and the
      // tail heap [tail()-tail_used, tail()) → dst's tail (which may be at a different
      // offset when dst is larger), so tail-relative smart_addresses stay valid.
      template <class N>
      static void blit(N* dst, const N* src) noexcept
      {
         // copy [alloc_header, head_size()) — the fixed header + the live index region
         // (head_size() == sizeof for the fixed tiers; variable only for a wide setlist).
         constexpr uint32_t hdr = uint32_t(sizeof(sal::alloc_header));
         std::memcpy(reinterpret_cast<char*>(dst) + hdr,
                     reinterpret_cast<const char*>(src) + hdr, src->head_size() - hdr);
         uint16_t hu = src->tail_used();
         std::memcpy(dst->tail() - hu, src->tail() - hu, hu);
      }

      // ── range-refcount engine layer (per-range branch-set counters; doc/micro_refcount.md) ──
      // A big node batches its child retain/release into RANGE_COUNT per-range counters in the SAL
      // implicit_ref_pool. clone/destroy branch on rc_has_ctrs(): countered nodes use the 8
      // counters, un-countered nodes (a birth site not yet wired) fall back to per-child — so
      // birth wiring is incrementally safe. is_big marks the tiers that have counters.
      template <class N>
      static constexpr bool is_big = requires { N::refcount_block_count; };

      // Birth: give a fresh big node its range_count per-range counters (each = 1, this version), co-located
      // in one region. The builder's per-child placement supplies each counter's base (cb.ref=1
      // per child). Re-homes if a region fills mid-birth (counters are unique=1 here, safe).
      template <class N>
      void rr_birth(N* n)
      {
         if constexpr (is_big<N>)
         {
            for (;;)
            {
               sal::ptr_address       a0     = _ses->rr_alloc_fresh();
               auto                   region = sal::region_of(a0);
               sal::implicit_ref_addr placed[range_count];
               placed[0] = sal::offset_of(a0);
               int  r  = 1;
               bool ok = true;
               for (; r < range_count; ++r)
               {
                  auto a = _ses->rr_alloc_in_region(
                      {region, std::span<const sal::implicit_ref_addr>(placed, size_t(r))});
                  if (not a) { ok = false; break; }
                  placed[r] = sal::offset_of(*a);
               }
               if (ok)
               {
                  n->rc_set_region(region);
                  for (int i = 0; i < range_count; ++i) n->rc_set_off(i, placed[i]);
                  return;
               }
               for (int j = 0; j < r; ++j)  // region filled mid-birth: undo the placed + retry
                  (void)_ses->rr_release(sal::combine(region, placed[j]));
            }
         }
      }

      // A big node's children are co-owned by another version iff any range counter > 1 (deferred
      // counts hide that from cb.ref).
      template <class N>
      bool children_shared(const N* cn)
      {
         if constexpr (is_big<N>)
         {
            if (not cn->rc_has_ctrs()) return false;
            for (int r = 0; r < range_count; ++r)
               if (_ses->rr_count(cn->rc_ctr(r)) > 1) return true;
         }
         return false;
      }

      // Is the child in range r co-owned? PER-RANGE — used when descending into a specific child so
      // we don't clone it just because some OTHER range is shared. After op1 diverges the path's
      // range (count→1), op2..opN walking the same path see this false → in-place. (children_shared,
      // the any-range form, would keep cloning until every range is private.)
      template <class N>
      bool range_shared(const N* cn, int r)
      {
         if constexpr (is_big<N>)
         {
            if (not cn->rc_has_ctrs()) return false;
            return _ses->rr_count(cn->rc_ctr(r)) > 1;
         }
         return false;
      }

      // Give up this version's share of range r's counter. The version that takes a counter
      // to zero is the last owner of that range's branch set and releases the branches, exactly
      // as large_radix::destroy does. A caller that has just counted more than one sharer can
      // still be the last by the time it leaves: the others may be versions the release thread
      // is destroying at that moment.
      template <class N>
      void rr_leave(const N* n, int r)
      {
         if constexpr (is_big<N>)
         {
            if (_ses->rr_release(n->rc_ctr(r)))
            {
               SAL_CRASH_POINT("range:last_share_released");
               n->for_each_entry([&](uint8_t byte, smart_address sa) {
                  if (sa.is_pointer() && range_of(byte) == r) _ses->release(sa.as_pointer());
               });
            }
         }
      }

      // Rare: a counter alloc found the node's region full. Fully privatize — retain ALL pointer
      // children (O(F)), drop this version's hold on all 8 old counters, birth fresh ones in a new
      // region. Every range becomes private (count 1).
      template <class N>
      void rr_rehome_full(N* n)
      {
         if constexpr (is_big<N>)
         {
            n->for_each_entry([&](uint8_t, smart_address sa)
                              { if (sa.is_pointer()) _ses->retain(sa.as_pointer()); });
            for (int r = 0; r < range_count; ++r) rr_leave<N>(n, r);
            rr_birth<N>(n);
         }
      }

      // Divergence: before mutating range r's branch set on a (cow'd) big node whose range-r
      // counter is SHARED, give this version its own range-r counter. Retain every current
      // pointer-child in range r (now co-owned by old set + this new one — the inherent ~32); the
      // per-site release(old_child) at drop/repoint cancels the changed child's retain, and an
      // add's new child is fresh. No-op for small tiers, un-countered nodes, already-private ranges.
      template <class N>
      void rr_diverge(N* n, int r)
      {
         if constexpr (is_big<N>)
         {
            if (not n->rc_has_ctrs() || _ses->rr_count(n->rc_ctr(r)) <= 1)
               return;
            auto a = _ses->rr_alloc_in_region(  // try a co-located new counter FIRST
                {n->rc_region(), std::span<const sal::implicit_ref_addr>(n->rc_offs(), range_count)});
            if (not a) { rr_rehome_full(n); return; }  // region full → privatize all (rare)
            n->for_each_entry([&](uint8_t byte, smart_address sa) {
               if (sa.is_pointer() && range_of(byte) == r) _ses->retain(sa.as_pointer());
            });
            SAL_CRASH_POINT("range:diverge_children_retained");
            rr_leave<N>(n, r);  // leave the old shared set (its branches are held above)
            SAL_CRASH_POINT("range:diverge_left_shared");
            n->rc_set_off(r, sal::offset_of(*a));
         }
      }

      // Clone from an already-resolved source node (no get_ref — the caller, e.g. cow(), holds
      // the ref pointing at exactly this node). `a` is unused for resolution but kept for symmetry.
      template <class N>
      ptr_address clone(ptr_address /*a*/, const N* src, uint32_t extra)
      {
         uint32_t need = ucc::round_up_multiple<sal::cacheline_size>(
             src->head_size() + src->tail_used() + extra + N::headroom);
         ptr_address na = _ses->alloc<N>(need, sal::alloc_hint{}, src->prefix());
         N*          nn = mut<N>(na);
         blit<N>(nn, src);
         SAL_CRASH_POINT("clone:copied");
         if constexpr (is_big<N>)
         {
            if (nn->rc_has_ctrs())  // share the per-range counters (blit copied region+offsets)
            {
               // A counter is one byte: it can stand for 255 versions of a range's branch
               // set. When this clone would be one more (many versions kept alive at once,
               // or dropped faster than the release thread finishes them), it does not
               // share: it owns its branches the way a newly built node does, with a
               // reference of its own on every child and fresh counters. The versions it
               // was cloned from are untouched.
               int r = 0;
               while (r < range_count && _ses->rr_try_retain(nn->rc_ctr(r)))
                  ++r;
               if (r < range_count) [[unlikely]]
               {
                  // Own every branch first, then give back the shares already taken on
                  // the earlier ranges. `src` holds each of those counters for as long
                  // as this runs, so a share given back is not the last; if one ever is,
                  // this thread has become that range set's last owner and releases its
                  // branches as a destroy would (rr_leave), which the references just
                  // taken keep alive.
                  nn->for_each_entry([&](uint8_t, smart_address sa)
                                     { if (sa.is_pointer()) _ses->retain(sa.as_pointer()); });
                  for (int j = 0; j < r; ++j)
                     rr_leave<N>(nn, j);
                  rr_birth<N>(nn);
               }
               smart_address tv = nn->term_value();
               if (tv.is_pointer()) _ses->retain(tv.as_pointer());  // term held directly
            }
            else  // un-countered big node (birth not wired) → per-child fallback (matches destroy)
               nn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         }
         else
            nn->for_each_child([&](ptr_address c) { _ses->retain(c); ++_cc.child_retains; });
         SAL_CRASH_POINT("clone:children_owned");
         stamp<N>(na);
         ++_cc.clones;
         return na;
      }
      // Address-only overload: resolve then delegate. For callers not holding the node's ref.
      template <class N>
      ptr_address clone(ptr_address a, uint32_t extra = 0)
      {
         return clone<N>(a, _ses->get_ref<N>(a).obj(), extra);
      }
      // Make a node writable for a mutation needing `extra` tail bytes, with the node already
      // resolved ONCE into `ref` (threaded by reference — no per-call get_ref, and a smart_ref
      // is borrowed so this avoids the by-value session-ptr refcount churn psitri warns about).
      //   shared (ref>1, a snapshot holds it) → clone to a NEW address (children retained);
      //   unique + room                       → mutate the held node IN PLACE via ref.modify()
      //                                         (reuses the resolved ref — no get_ref);
      //   unique + heap-full                  → grow (clone bigger, new address).
      // On a clone `addr` is updated to the new node and `ref` is left pointing at the OLD one
      // (the caller uses the returned writable pointer, not `ref`, after a COW, and releases the
      // old node per the COW-to-root contract — clone pre-sizes for `extra`, so we never release
      // here, which would double-free). modify()'s own SAL-COW is never reached: heart pre-clones
      // whenever ref>1, so a node that reaches modify() is unique and (write_protect off)
      // always can_modify → an in-place write.
      template <upsert_mode mode, class N>
      N* cow(sal::smart_ref<N>& ref, ptr_address& addr, uint32_t extra = 0)
      {
         // SHARED mode (snapshot reachable) OR a directly-shared node → must CLONE, never mutate
         // in place. The mode forces the clone even at ref==1, because a ref==1 child of a shared
         // parent is still reachable from the snapshot (the clone retains the siblings — the
         // COW-to-root cost). The runtime ref check remains in unique mode: it is the discovery
         // point for nodes that became snapshot-reachable since the entry dispatch.
         if (mode.is_shared() || ref.ref() > 1)
         {
            addr = clone<N>(addr, ref.obj(), extra);  // reuse the held ref — no second get_ref
            return mut<N>(addr);
         }
         if (ref->free_heap() >= extra)
         {
            ++_cc.in_place;
            return ref.modify().get();
         }
         ++_cc.grows;
         addr = clone<N>(addr, ref.obj(), extra);  // reuse the held ref — no second get_ref
         return mut<N>(addr);
      }
      // Address-only overload: resolves once and delegates. Behavior-identical to threading a
      // ref; used by call sites not (yet) holding a smart_ref.
      template <upsert_mode mode, class N>
      N* cow(ptr_address& addr, uint32_t extra = 0)
      {
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);
         return cow<mode, N>(ref, addr, extra);
      }
      // Writable access to a FRESH node — one THIS op just allocated (collision child, re_stride
      // target, prefix-split parent). A fresh node is never snapshot-reachable, so the shared
      // latch does NOT apply: grow in place when the heap has room, else realloc wider and FREE
      // the original (we hold its only ref). Using cow() here instead is a BUG — under the latch
      // it clones the fresh node and orphans the original (one leaked object per call).
      template <class N>
      N* grow_fresh(ptr_address& addr, uint32_t extra = 0)
      {
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);
         assert(ref.ref() == 1 && "grow_fresh targets a node this op just built");
         if (ref->free_heap() >= extra)
            return ref.modify().get();
         const ptr_address pre = addr;
         addr = clone<N>(addr, ref.obj(), extra);  // clone retains children…
         _ses->release(pre);  // …and destroying the fresh original releases its refs: balanced
         ++_cc.grows;
         return mut<N>(addr);
      }

      // ── terminal values: one size-gated rule ───────────────────────────────────
      static bool fits_inline(bytes suffix, bytes value) noexcept
      {
         return int(suffix.size() + value.size()) <= max_inline;
      }
      static const inline_value* inline_of(const sal::alloc_header* node, smart_address sa)
      {
         // A tail offset must point INSIDE the node's tail region (offset <= size); a larger offset
         // would mean `sa` is a misclassified/corrupt branch and this computes a wild pointer.
         assert(sa.tail_offset() <= node->size() && "inline tail offset exceeds node size");
         return reinterpret_cast<const inline_value*>(reinterpret_cast<const char*>(node) +
                                                      node->size() - sa.tail_offset());
      }
      // Tail bytes needed for an inline terminal {suffix,value}, or 0 if it spills to a SAL leaf.
      static uint32_t value_tail_need(bytes suffix, bytes value) noexcept
      {
         return fits_inline(suffix, value)
                    ? uint32_t(inline_value::byte_size(int(suffix.size()), int(value.size())))
                    : 0;
      }
      // Write a terminal {suffix,value} INTO an already-writable node `n` — NO cow (the caller
      // cow'd with value_tail_need() room). Small → a tail leaf in n's heap; else a SAL leaf.
      template <class N>
      smart_address make_value_into(N* n, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         // Subtree values (vt_subtree) are stored as a LEAF, never inline: the leaf's destroy owns
         // the root-release cascade, so the hot node clone/destroy paths need no inline value-scan.
         if (_ins_value_type == inline_value::vt_bytes && fits_inline(suffix, value))
         {
            uint32_t need = uint32_t(inline_value::byte_size(int(suffix.size()), int(value.size())));
            auto*    iv   = reinterpret_cast<inline_value*>(n->alloc_tail(uint16_t(need)));
            iv->suffix_len = uint8_t(suffix.size());
            iv->value_len  = uint8_t(value.size());
            iv->value_type = 0;
            std::memcpy(iv->data, suffix.data(), suffix.size());
            std::memcpy(iv->data + suffix.size(), value.data(), value.size());
            ++_cc.inline_values;
            return smart_address::from_tail(n->tail_used());
         }
         return smart_address::from_pointer(make_leaf(suffix, value, hint));
      }
      // Allocate a terminal attached to a FRESH node `addr` (every caller passes a node this op
      // just built — collision child, prefix-split parent, re_stride router): grow_fresh (room
      // for the value), then write it in. NOT for snapshot-reachable nodes — those go through
      // cow() exactly once at their mutation site (see the router same-key update).
      template <class N>
      smart_address make_value(ptr_address& addr, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         N* n = grow_fresh<N>(addr, value_tail_need(suffix, value));
         return make_value_into<N>(n, suffix, value, hint);
      }
      // Zero-copy view of a terminal's {suffix,value} as raw bytes — a `bytes` overload of
      // the value_view term_view() the read path uses. No copy is needed: the node arena is
      // a fixed PROT_NONE reservation committed with MAP_FIXED (block_allocator), so it
      // never relocates, and mutation callers consume these views before the node they live
      // in is COW-released. So views into node memory stay valid across later allocations.
      void term_view(const sal::alloc_header* node, smart_address sa, bytes& suf, bytes& val) const
      {
         term_view(*_ses, node, sa, suf, val);
      }
      static void term_view(sal::allocator_session& ses, const sal::alloc_header* node,
                            smart_address sa, bytes& suf, bytes& val)
      {
         const inline_value* iv = sa.is_tail()
                                      ? inline_of(node, sa)
                                      : &ses.get_ref<leaf_node>(sa.as_pointer()).obj()->value;
         suf = bytes{iv->data, size_t(iv->suffix_len)};
         val = bytes{iv->data + iv->suffix_len, size_t(iv->value_len)};
      }
      // The terminal's inline_value (tail or leaf) — exposes value_type so callers can tell a
      // byte value from a subtree value.
      const inline_value* term_iv(const sal::alloc_header* node, smart_address sa) const
      {
         return term_iv(*_ses, node, sa);
      }
      static const inline_value* term_iv(sal::allocator_session& ses, const sal::alloc_header* node,
                                         smart_address sa)
      {
         return sa.is_tail() ? inline_of(node, sa)
                             : &ses.get_ref<leaf_node>(sa.as_pointer()).obj()->value;
      }
      // Resolve an inline tail on `node`: its suffix must equal `rem`.
      static bool inline_at(const sal::alloc_header* node, smart_address sa, bytes rem,
                            value_view& out)
      {
         const inline_value* iv = inline_of(node, sa);
         if (iv->suffix_len != rem.size() ||
             (rem.size() && std::memcmp(iv->data, rem.data(), iv->suffix_len) != 0))
            return false;
         if (iv->is_subtree())  // a subtree value, not bytes — use is_subtree()/get_subtree()
            return false;
         out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
         return true;
      }
      static bool term_lookup(sal::allocator_session& ses, const sal::alloc_header* node,
                              smart_address tv, bytes key, uint32_t depth, value_view& out)
      {
         if (tv.is_null())
            return false;
         if (tv.is_tail())
            return inline_at(node, tv, key.subspan(depth), out);
         return leaf_read(ses, tv.as_pointer(), key, depth, out);
      }
      static bool leaf_read(sal::allocator_session& ses, ptr_address addr, bytes key,
                            uint32_t depth, value_view& out)
      {
         const leaf_node* n   = ses.get_ref<leaf_node>(addr).obj();
         bytes            suf = n->suffix();
         bytes            rem = key.subspan(depth);
         if (suf.size() != rem.size() ||
             (suf.size() && std::memcmp(suf.data(), rem.data(), suf.size()) != 0))
            return false;
         if (n->is_subtree())  // a subtree value, not bytes — use is_subtree()/get_subtree()
            return false;
         bytes v = n->value_bytes();
         out     = value_view(reinterpret_cast<const char*>(v.data()), v.size());
         return true;
      }
      template <upsert_mode mode>
      ptr_address update_leaf_value(ptr_address addr, bytes val)
      {
         auto             ref = _ses->get_ref<leaf_node>(addr);
         const leaf_node* ln  = ref.obj();
         // Never overwrite in place when a subtree is involved (old or new): route through
         // make_leaf so the OLD leaf is released (its destroy frees the old subtree root) and the
         // NEW leaf is stamped with the pending value_type. Also never in place under the SHARED
         // latch: a ref==1 leaf below a shared node is still reachable from the snapshot —
         // mutating it would corrupt the snapshot's view (same rule as cow()).
         if (_ins_value_type == inline_value::vt_bytes && !ln->is_subtree() && !mode.is_shared() &&
             ref.ref() == 1 && ln->value.value_len == val.size())
         {
            leaf_node* w = ref.modify().get();
            std::memcpy(w->value.data + w->value.suffix_len, val.data(), val.size());
            return addr;
         }
         return make_leaf(ln->suffix(), val);
      }

      // ── ordered-seek machinery (lower_bound / upper_bound / in-order walk) ──────
      // Stems are stored sorted (set_branch inserts at the lower-bound position) and the
      // bitset/full tiers iterate in byte order, so every node enumerates entries in key
      // order. A seek descends the matching path; when the query diverges it takes the
      // smallest sibling strictly greater than the query and descends to that subtree's
      // minimum (descend_min), backtracking to the parent when a subtree is all-less.

      // Reconstructed-key buffer (keys are materialized as we descend; values are viewed
      // directly in node memory). 1 KB covers any realistic key; asserts if exceeded.
      struct KB
      {
         uint8_t  buf[1024];
         uint32_t len = 0;
         void     append(const uint8_t* p, size_t n) noexcept
         {
            assert(len + n <= sizeof(buf));
            std::memcpy(buf + len, p, n);
            len += uint32_t(n);
         }
         void truncate(uint32_t l) noexcept { len = l; }
      };

      bool is_terminal_branch(smart_address sa) const { return is_terminal_branch(*_ses, sa); }
      static bool is_terminal_branch(sal::allocator_session& ses, smart_address sa)
      {
         if (sa.is_tail())
            return true;
         return ses.get_ref<sal::alloc_header>(sa.as_pointer())->type() == leaf_node::type_id;
      }
      // Suffix bytes + value of a terminal, both viewing node memory (no copy).
      void term_view(const sal::alloc_header* node, smart_address sa, bytes& suf,
                     value_view& val) const
      {
         term_view(*_ses, node, sa, suf, val);
      }
      static void term_view(sal::allocator_session& ses, const sal::alloc_header* node,
                            smart_address sa, bytes& suf, value_view& val)
      {
         const inline_value* iv =
             sa.is_tail() ? inline_of(node, sa)
                          : &ses.get_ref<leaf_node>(sa.as_pointer()).obj()->value;
         suf = bytes{iv->data, size_t(iv->suffix_len)};
         val = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
      }
      // Append a terminal's suffix to kb and return its value (both view node memory).
      value_view append_terminal(const sal::alloc_header* node, smart_address sa, KB& kb)
      {
         return append_terminal(*_ses, node, sa, kb);
      }
      static value_view append_terminal(sal::allocator_session& ses, const sal::alloc_header* node,
                                        smart_address sa, KB& kb)
      {
         bytes      suf;
         value_view v;
         term_view(ses, node, sa, suf, v);
         kb.append(suf.data(), suf.size());
         return v;
      }
      // The `cw` actual key bytes a stem cell represents (big-endian, top cw of stem_width).
      static int stem_bytes(uint32_t cell, int stem_width, int cw, uint8_t* o) noexcept
      {
         for (int k = 0; k < cw; ++k)
            o[k] = uint8_t(cell >> (8 * (stem_width - 1 - k)));
         return cw;
      }
      static int key_cmp(const KB& kb, bytes q) noexcept
      {
         const size_t m = kb.len < q.size() ? kb.len : q.size();
         const int    c = m ? std::memcmp(kb.buf, q.data(), m) : 0;
         if (c)
            return c < 0 ? -1 : 1;
         if (kb.len == q.size())
            return 0;
         return kb.len < q.size() ? -1 : 1;
      }

      // ---- per-tier adaptors: a uniform ordered cursor over a node's entries ----
      template <class N>
      struct SLAd  // setlist (u8 / u16 / u24 / u32): cursor = entry index
      {
         const N*                 p;
         const sal::alloc_header* node() const { return reinterpret_cast<const sal::alloc_header*>(p); }
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return p->consume_width(); }
         int           stem_width() const { return N::stem_width; }
         int           first() const { return 0; }
         bool          valid(int c) const { return c < p->nstems(); }
         int           next(int c) const { return c + 1; }
         int           last() const { return p->nstems() - 1; }  // reverse: max entry (-1 if empty)
         int           prev(int c) const { return c - 1; }       // reverse: previous (-1 = before first)
         bool          has_below(int c) const { return c > 0; }  // any entry before c (O(1) here)
         uint32_t      cell(int c) const { return uint32_t(p->stem_at(c)); }
         smart_address branch(int c) const { return p->branch_at(c); }
         int           lower(bytes r, bool& full, uint32_t& floorcell) const
         {
            const int m = int(r.size()) < cw() ? int(r.size()) : cw();
            full        = int(r.size()) >= cw();
            uint32_t v  = 0;
            for (int i = 0; i < m; ++i)
               v = (v << 8) | r[i];
            v <<= 8 * (N::stem_width - m);
            floorcell = v;
            return p->lower_bound_index(typename N::stem_type(v));
         }
      };
      template <class N, int SegBits>
      struct BSAd  // bitset_r2 / bitset_r4: cursor = present byte (or 256 = end)
      {
         const N*                 p;
         const sal::alloc_header* node() const { return reinterpret_cast<const sal::alloc_header*>(p); }
         int                      np(int b) const  // smallest present byte >= b, else 256
         {
            if (b < 0)
               b = 0;
            for (int s = b >> SegBits; s < (256 >> SegBits); ++s)
            {
               const int lo = (s == (b >> SegBits)) ? (b & ((1 << SegBits) - 1)) : 0;
               const int nx = p->seg(s).present.next_set(lo);
               if (nx >= 0)
                  return (s << SegBits) + nx;
            }
            return 256;
         }
         int pp(int b) const  // largest present byte <= b, else -1 (reverse of np)
         {
            if (b < 0) return -1;
            if (b > 255) b = 255;
            for (int s = b >> SegBits; s >= 0; --s)
            {
               const int hi = (s == (b >> SegBits)) ? (b & ((1 << SegBits) - 1)) : ((1 << SegBits) - 1);
               const int pv = p->seg(s).present.prev_set(hi);
               if (pv >= 0) return (s << SegBits) + pv;
            }
            return -1;
         }
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
         int           last() const { return pp(255); }      // reverse: highest present (-1 empty)
         int           prev(int c) const { return pp(c - 1); }
         bool          has_below(int c) const { return pp(c - 1) >= 0; }
         uint32_t      cell(int c) const { return uint32_t(c); }
         smart_address branch(int c) const { return p->find(uint8_t(c)); }
         int           lower(bytes r, bool& full, uint32_t& floorcell) const
         {
            full      = !r.empty();
            floorcell = r.empty() ? 0u : uint32_t(r[0]);
            return np(int(floorcell));
         }
      };
      struct FullAd  // full_radix: cursor = present byte (or 256 = end)
      {
         const full_radix_node*   p;
         const sal::alloc_header* node() const { return reinterpret_cast<const sal::alloc_header*>(p); }
         int                      np(int b) const
         {
            if (b >= 256)
               return 256;
            const int nx = p->branches.next_present(b);
            return nx < 0 ? 256 : nx;
         }
         int pp(int b) const  // largest present byte <= b, else -1
         {
            if (b < 0) return -1;
            return p->branches.prev_present(b > 255 ? 255 : b);
         }
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
         int           last() const { return pp(255); }
         int           prev(int c) const { return pp(c - 1); }
         bool          has_below(int c) const { return pp(c - 1) >= 0; }
         uint32_t      cell(int c) const { return uint32_t(c); }
         smart_address branch(int c) const { return p->find(uint8_t(c)); }
         int           lower(bytes r, bool& full, uint32_t& floorcell) const
         {
            full      = !r.empty();
            floorcell = r.empty() ? 0u : uint32_t(r[0]);
            return np(int(floorcell));
         }
      };
      template <class N>
      struct C1Ad  // bitset_c1_*: cursor = present absolute byte (or 256 = end)
      {
         const N*                 p;
         const sal::alloc_header* node() const { return reinterpret_cast<const sal::alloc_header*>(p); }
         int                      np(int b) const  // smallest present byte >= b, else 256
         {
            const int bse = p->window_base();
            const int nd  = p->next_present(b < bse ? 0 : b - bse);
            return nd < 0 ? 256 : bse + nd;
         }
         int pp(int b) const  // largest present byte <= b, else -1 (c1 has no prev_present → scan up)
         {
            int best = -1;
            for (int c = np(0); c >= 0 && c <= b; c = np(c + 1)) best = c;
            return best;
         }
         bytes         prefix() const { return p->prefix(); }
         smart_address term() const { return p->term_value(); }
         int           cw() const { return 1; }
         int           stem_width() const { return 1; }
         int           first() const { return np(0); }
         bool          valid(int c) const { return c < 256; }
         int           next(int c) const { return np(c + 1); }
         int           last() const { return pp(255); }
         int           prev(int c) const { return pp(c - 1); }
         bool          has_below(int c) const { return pp(c - 1) >= 0; }
         uint32_t      cell(int c) const { return uint32_t(c); }
         smart_address branch(int c) const { return p->find(uint8_t(c)); }
         int           lower(bytes r, bool& full, uint32_t& floorcell) const
         {
            full      = !r.empty();
            floorcell = r.empty() ? 0u : uint32_t(r[0]);
            return np(int(floorcell));
         }
      };
      // Dispatch `addr` to the right adaptor and invoke f(adaptor). f returns bool/void.
      // ONE get_ref resolves the node; the type-switch lowers to a jump table; each case
      // reinterprets that single resolved pointer (alloc_header is at offset 0 of every node)
      // — no second per-case get_ref. This is the per-step cost of an ordered scan.
      // ── READ-PATH SESSION CONTRACT ─────────────────────────────────────────────────────────
      // Every read-only descent helper below takes the SAL session as an explicit parameter and
      // is STATIC: the compiler proves the read path touches no engine state, so a cursor pinned
      // through a READER session (one session per thread; allocator_session_ptr's refcount is
      // non-atomic and must never be copied cross-thread — every get_ref copies it) can descend
      // concurrently with the single writer. The parameterless members forward *_ses (writer
      // thread). Write paths (insert/remove/flush) stay member-session by construction.
      template <class F>
      auto dispatch(ptr_address addr, F&& f)
      {
         return dispatch(*_ses, addr, std::forward<F>(f));
      }
      template <class F>
      static auto dispatch(sal::allocator_session& ses, ptr_address addr, F&& f)
      {
         return dispatch_hdr(ses.get_ref<sal::alloc_header>(addr).obj(), std::forward<F>(f));
      }
      // Same dispatch on an ALREADY-RESOLVED header — for loops that classify the node type
      // themselves first (the page route), so each hop costs exactly one get_ref.
      template <class F>
      static auto dispatch_hdr(const sal::alloc_header* b, F&& f)
      {
         switch (kind_of(b->type()))
         {
            case radix_kind::setlist_u8:
               return f(SLAd<setlist_u8>{reinterpret_cast<const setlist_u8*>(b)});
            case radix_kind::setlist_u16:
               return f(SLAd<setlist_u16>{reinterpret_cast<const setlist_u16*>(b)});
            case radix_kind::setlist_u32:
               return f(SLAd<setlist_u32>{reinterpret_cast<const setlist_u32*>(b)});
            case radix_kind::c1_32:
               return f(C1Ad<bitset_c1_32>{reinterpret_cast<const bitset_c1_32*>(b)});
            case radix_kind::c1_64:
               return f(C1Ad<bitset_c1_64>{reinterpret_cast<const bitset_c1_64*>(b)});
            case radix_kind::c1_128:
               return f(C1Ad<bitset_c1_128>{reinterpret_cast<const bitset_c1_128*>(b)});
            case radix_kind::bitset_r2:
               return f(BSAd<bitset_r2_node, 7>{reinterpret_cast<const bitset_r2_node*>(b)});
            case radix_kind::bitset_r4:
               return f(BSAd<bitset_r4_node, 6>{reinterpret_cast<const bitset_r4_node*>(b)});
            case radix_kind::full:
               return f(FullAd{reinterpret_cast<const full_radix_node*>(b)});
            case radix_kind::bucket:
               // The ordered cursor / in-order walk has NO bucket adaptor yet: a bucket is a
               // dense sorted terminal leaf, not a radix router. Descending into one is the
               // known gap (step 2 adds a bucket frame). Throw rather than UB so the break is
               // observable. find()/upsert have their own switch and handle buckets directly.
               throw std::runtime_error(
                   "heart ordered cursor: reached a bucket leaf — buckets are not traversed");
            case radix_kind::page:
               // Pages are terminals, not routers — every page-model walker resolves a page
               // BEFORE dispatching (page_at). Reaching here is a logic error; observable > UB.
               throw std::runtime_error(
                   "heart ordered cursor: dispatch() reached a leaf page — resolve pages first");
            case radix_kind::leaf:     // not ordered routers — dispatch() is never called on these
            case radix_kind::bitset1:
               break;
         }
         __builtin_unreachable();
      }

      // Resolve `a` to a bucket leaf, or nullptr if it is a radix node. Buckets are dense sorted
      // (suffix,value) pages, not radix routers, so the ordered descent/seek/inorder paths handle
      // them explicitly instead of through dispatch() (which has no bucket adaptor).
      const bucket_node* bucket_at(ptr_address a) { return bucket_at(*_ses, a); }
      static const bucket_node* bucket_at(sal::allocator_session& ses, ptr_address a)
      {
         const sal::alloc_header* b = ses.get_ref<sal::alloc_header>(a).obj();
         return b->type() == bucket_node::type_id ? reinterpret_cast<const bucket_node*>(b) : nullptr;
      }
      // Append a bucket entry's suffix to kb and view its value (entry order == key order).
      static value_view bucket_entry(const bucket_node* bn, int i, KB& kb)
      {
         const inline_value* iv = bn->entry(i);
         kb.append(iv->data, iv->suffix_len);
         return value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
      }

      // ════ RADIX-INDEXED LEAF PAGES (the page model — the DEFAULT data layout) ═══════════════
      // Keys live in 2KB leaf_page terminals; the radix above stores only DIVIDERS — short
      // separator keys emitted by page splits. A divider is FULLY consumed by the radix path
      // (node prefixes + branch bytes) and its slot points DIRECTLY at a leaf_page: no suffix
      // bytes and no inline/leaf_node terminals exist anywhere in a page-model radix. Pages are
      // ordinary pointer children, so clone/destroy/range-refcount machinery applies unchanged.
      //
      // ROUTING IS LOWER-BOUND, not exact: key K belongs to the page of the LARGEST divider
      // <= K. The empty divider "" (the radix root's term_value) always exists once a radix
      // does, so a fallback page always exists. At each node take the largest branch <= the
      // key byte: equal → descend (remembering this node's nearest lesser branch/term as the
      // fallback); less → that branch subtree's MAX page (max-path walk); none → this node's
      // term page, else the remembered ancestor fallback. A key diverging from a node's prefix
      // resolves the same way (above → this subtree's max; below → fallback). A miss must still
      // reach a page — that is the design's stated sacrifice (the page's fingerprint column is
      // the point-lookup advantage instead).
      static bool is_page(sal::allocator_session& ses, ptr_address a)
      {
         return ses.get_ref<sal::alloc_header>(a)->type() == leaf_page::type_id;
      }
      static const leaf_page* page_at(sal::allocator_session& ses, ptr_address a)
      {
         const sal::alloc_header* b = ses.get_ref<sal::alloc_header>(a).obj();
         return b->type() == leaf_page::type_id ? reinterpret_cast<const leaf_page*>(b) : nullptr;
      }
      const leaf_page* page_at(ptr_address a) { return page_at(*_ses, a); }

      ptr_address make_page() { return _ses->alloc<leaf_page>(sal::alloc_hint{}); }

      // Append helper for divider capture (route + max-path build the divider as they walk).
      static void div_append(uint8_t* div, uint32_t* dlen, const uint8_t* p, size_t n)
      {
         if (!div || !n)
            return;
         std::memcpy(div + *dlen, p, n);
         *dlen += uint32_t(n);
      }


      // The rightmost page under `addr` (the page of the subtree's MAX divider). Appends the
      // divider bytes it walks when `div` is non-null. ONE get_ref per hop.
      static const leaf_page* max_page(sal::allocator_session& ses, ptr_address addr, uint8_t* div,
                                       uint32_t* dlen)
      {
         for (;;)
         {
            const sal::alloc_header* h = ses.get_ref<sal::alloc_header>(addr).obj();
            if (h->type() == leaf_page::type_id)
               return reinterpret_cast<const leaf_page*>(h);
            ptr_address next{};
            dispatch_hdr(h, [&](auto ad) {
               bytes pfx = ad.prefix();
               div_append(div, dlen, pfx.data(), pfx.size());
               const int c = ad.last();
               if (c >= 0)
               {
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
                  div_append(div, dlen, sb, size_t(nb));
                  next = ad.branch(c).as_pointer();  // page-model branches are always pointers
               }
               else
               {
                  assert(!ad.term().is_null() && "page-model router with neither branches nor term");
                  next = ad.term().as_pointer();
               }
            });
            addr = next;
         }
      }

      // Lower-bound route: the page owning `kb` (the largest divider <= kb). When `div` is
      // non-null the owning divider's bytes are captured there — the currency for the mutating
      // descents and the cursor. A root that IS a page returns immediately (divider = "").
      // HOT-PATH SHAPE: one get_ref per hop (type classified from the same resolved header),
      // and the fallback memo is LAZY — the exact path only records (node, li); the lesser
      // sibling / term choice is computed once, IF a fallback is ever needed (a miss-below
      // event), never per hop. has_below() is O(1) for setlists and short-circuited by the
      // term check for the big tiers (the radix root always holds the "" term).
      static const leaf_page* route_to_page(sal::allocator_session& ses, ptr_address addr,
                                            bytes kb, uint8_t* div, uint32_t* dlen)
      {
         uint32_t depth = 0;
         // remembered fallback: the deepest exact-path node with a divider < kb under it
         ptr_address fb_node = sal::null_ptr_address;
         int         fb_li   = -1;  // the lower-bound cursor at that node (resolve lazily)
         uint32_t    fb_dlen = 0;
         bool        have_fb = false;
         // Resolve the lazy lower-bound fallback (the deepest exact-path node holding a divider <
         // kb): the lesser sibling at fb_li, or that node's term. Shared by a router miss (key
         // diverges below a node prefix) AND a page miss (key < a directly-attached page's divider).
         auto take_fallback = [&]() -> const leaf_page* {
            assert(have_fb && "no fallback: the radix root must hold the \"\" divider");
            (void)have_fb;
            if (dlen)
               *dlen = fb_dlen;
            ptr_address fnext{};
            bool        fpage = false;
            dispatch(ses, fb_node, [&](auto ad) {
               const int less = ad.prev(fb_li);  // resolved ONCE, only on fallback
               if (less < 0)
               {
                  fnext = ad.term().as_pointer();
                  fpage = true;
                  return;
               }
               uint8_t   sb[4];
               const int nb = stem_bytes(ad.cell(less), ad.stem_width(), ad.cw(), sb);
               div_append(div, dlen, sb, size_t(nb));
               fnext = ad.branch(less).as_pointer();
            });
            if (fpage)
               return reinterpret_cast<const leaf_page*>(
                   ses.get_ref<sal::alloc_header>(fnext).obj());
            return max_page(ses, fnext, div, dlen);
         };
         for (;;)
         {
            const sal::alloc_header* h = ses.get_ref<sal::alloc_header>(addr).obj();
            if (h->type() == leaf_page::type_id)
            {
               // Canonical collapse: a page attaches directly under a branch byte and carries its
               // own divider (_div). Discriminate the lower bound against _div[depth..] (the parent
               // hop already appended this page's branch byte; _div beyond it is page-internal, so
               // NOTHING is appended to div here). A root/leftmost page (no fallback) catches every
               // key below it — return it rather than failing the fallback.
               const leaf_page* pg    = reinterpret_cast<const leaf_page*>(h);
               bytes            dv    = pg->divider();
               bytes            dtail = dv.size() > depth ? dv.subspan(depth) : bytes{};
               bytes            rem   = kb.subspan(depth < kb.size() ? depth : kb.size());
               const size_t     m     = dtail.size() < rem.size() ? dtail.size() : rem.size();
               int              c     = m ? std::memcmp(rem.data(), dtail.data(), m) : 0;
               if (c == 0 && rem.size() < dtail.size())
                  c = -1;  // key exhausts inside the page divider → below this page
               if (c < 0 && have_fb)
                  return take_fallback();
               return pg;  // c >= 0 (this page owns kb), or no fallback (root/leftmost catch-all)
            }
            ptr_address next{};
            int action = 0;  // 0 descend, 1 done(next=page), 2 fallback, 3 max(addr), 4 max(next)
            dispatch_hdr(h, [&](auto ad) {
               bytes        pfx = ad.prefix();
               bytes        rem = kb.subspan(depth);
               const size_t m   = pfx.size() < rem.size() ? pfx.size() : rem.size();
               int          c   = m ? std::memcmp(rem.data(), pfx.data(), m) : 0;
               if (c == 0 && rem.size() < pfx.size())
                  c = -1;  // key exhausts inside the prefix → below every divider through here
               if (c < 0)
               {
                  action = 2;
                  return;
               }
               if (c > 0)  // above every divider in this subtree → its max page
               {
                  action = 3;
                  return;
               }
               div_append(div, dlen, pfx.data(), pfx.size());
               const uint32_t d2 = depth + uint32_t(pfx.size());
               if (d2 == kb.size())  // key == path exactly → its own divider (term) or fallback
               {
                  if (!ad.term().is_null())
                  {
                     next   = ad.term().as_pointer();
                     action = 1;
                  }
                  else
                     action = 2;
                  return;
               }
               bytes      r2 = kb.subspan(d2);
               bool       full;
               uint32_t   fc;
               const int  li    = ad.lower(r2, full, fc);
               const bool exact = full && ad.valid(li) && ad.cell(li) == fc;
               if (exact)
               {
                  if (!ad.term().is_null() || ad.has_below(li))
                  {
                     fb_node = addr;
                     fb_li   = li;
                     fb_dlen = dlen ? *dlen : 0;
                     have_fb = true;
                  }
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(li), ad.stem_width(), ad.cw(), sb);
                  div_append(div, dlen, sb, size_t(nb));
                  next   = ad.branch(li).as_pointer();
                  depth  = d2 + uint32_t(ad.cw());
                  action = 0;
                  return;
               }
               const int less = ad.prev(li);
               if (less >= 0)
               {
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(less), ad.stem_width(), ad.cw(), sb);
                  div_append(div, dlen, sb, size_t(nb));
                  next   = ad.branch(less).as_pointer();
                  action = 4;
                  return;
               }
               if (!ad.term().is_null())
               {
                  next   = ad.term().as_pointer();
                  action = 1;
                  return;
               }
               action = 2;
            });
            switch (action)
            {
               case 0:
                  addr = next;
                  break;
               case 1:
                  return reinterpret_cast<const leaf_page*>(
                      ses.get_ref<sal::alloc_header>(next).obj());
               case 3:
                  return max_page(ses, addr, div, dlen);
               case 4:
                  return max_page(ses, next, div, dlen);
               case 2:
                  return take_fallback();
            }
         }
      }

      // Retain the children `pg` shares with a clone successor — every child except the record
      // at `skip` (-1 = none). The successor's slot-swap release of the old page then balances:
      // each shared child nets one owner per owning page; the skipped child is freed when the
      // old page was unique, or stays owned by the snapshot's page when it was shared.
      // A page that was just built as a copy of (part of) another holds a second physical copy
      // of each inline typed value: one more owner for each, released when either page dies.
      void retain_page_typed(ptr_address page)
      {
         _ses->get_ref<leaf_page>(page).obj()->for_each_inline_typed(
             [](uint32_t, std::string_view stored) { fire_typed_retain(stored); });
      }
      void retain_page_children(const leaf_page* pg, int skip)
      {
         for (uint32_t i = 0; i < pg->num_keys(); ++i)
         {
            if (int(i) == skip)
               continue;
            const inline_value* iv = pg->record(i);
            if (inline_value::external_value(iv->value_type))
               _ses->retain(pg->external_addr(i));
            else if (iv->is_subtree() && iv->subtree_root() != sal::null_ptr_address)
               _ses->retain(iv->subtree_root());
            else if (iv->value_type == inline_value::vt_typed) [[unlikely]]
               // the clone duplicates this typed value's bytes; the original/clone
               // each die through leaf_page::destroy -> balanced release.
               // Plain bytes (vt_bytes) are the caller's and are never inspected.
               fire_typed_retain({reinterpret_cast<const char*>(iv->value().data()),
                                  iv->value().size()});
         }
      }

      // Apply an upsert to `page` under the page-COW discipline: shared (snapshot-reachable
      // parent OR own ref>1) → rebuild-clone; unique + room → mutate in place (the page IS the
      // COW unit — no node copies at all on this path). Returns the page's successor (== page
      // when in-place/untouched). When the entry can't fit even via a rebuild, sets _page_split
      // and leaves the page untouched — the caller splits and retries.
      // FULL-KEY fetch for extern fragments (the page compares call back through this when a
      // fragment ties) — session-parametric so reader-session cursors share it.
      static kv_resolver kvres(sal::allocator_session& ses) noexcept
      {
         return {&ses, [](const void* c, ptr_address a) -> std::span<const uint8_t> {
                    auto& s = *const_cast<sal::allocator_session*>(
                        static_cast<const sal::allocator_session*>(c));
                    return s.get_ref<key_val_node>(a)->key();
                 }};
      }

      ptr_address page_apply_upsert(ptr_address page, bool parent_shared, bytes kb, bytes val)
      {
         auto             ref    = _ses->get_ref<leaf_page>(page);
         const leaf_page* pg     = ref.obj();
         const bool       shared = parent_shared || ref.ref() > 1;
         assert(kb.size() <= key_val_node::max_key_len && "keys are <= 1024 bytes (psitri parity)");
         // Classify by the would-be suffix (post-cpfx). A rebuild can only shorten cpfx to
         // common(cpfx, key), which is exactly what we compute → stable. ANY item whose
         // suffix+payload exceeds the in-page cap (leaf_page::max_inline_entry) — big value OR big key, any value
         // type — gets an external record; the page keeps a fragment + code.
         auto     cp  = pg->common_prefix();
         uint32_t ncp = 0;
         {
            const size_t m = cp.size() < kb.size() ? cp.size() : kb.size();
            while (ncp < m && cp[ncp] == kb[ncp])
               ++ncp;
         }
         const uint32_t       slen = uint32_t(kb.size()) - ncp;
         leaf_page::insert_op ins{.key   = kb,
                                  .value = val,
                                  .vtype = _ins_value_type,
                                  .ext   = sal::null_ptr_address,
                                  .res   = kvres(*_ses)};
         const uint64_t pay0 = inline_value::inline_payload(_ins_value_type)
                                   ? uint64_t(val.size())
                                   : 4u /*subtree root*/;
         const int idx = pg->find(kb.data(), uint32_t(kb.size()), ins.res);
         // ONCE EXTERNAL, STAYS EXTERNAL: an extern record's in-page fragment may be a
         // truncated prefix of the real key — inlining it on a shrinking update would lose
         // key bytes. (Fresh classification: any item whose suffix+payload exceeds the
         // in-page cap gets an external record: the value alone, or key and value when the
         // suffix does not fit the page record's one-byte length.)
         const bool old_ext =
             idx >= 0 && inline_value::external_value(pg->record(uint32_t(idx))->value_type);
         // A record whose FULL KEY lives in its external node (vt_extern) keeps only a
         // fragment of the suffix in the page, however short the suffix has since become
         // (splits tighten the common prefix). Its replacement must carry the full key too:
         // a value-only node would leave the fragment as the whole key.
         const bool old_full_key =
             idx >= 0 && pg->record(uint32_t(idx))->value_type == inline_value::vt_extern;
         if (old_ext || slen + pay0 > leaf_page::max_inline_entry)
         {
            if (slen <= leaf_page::max_suffix_len && !old_full_key)
            {
               // VALUE-ONLY external (psitri's value_node model): the full post-cpfx suffix
               // stays in the page record; only the VALUE moves out, with an EMPTY key. No key duplication; exact-match stays in-page.
               ins.ext   = large_value_detail::allocate(*_ses, bytes{}, val, _ins_value_type);
               ins.vtype = inline_value::vt_extern_val;
            }
            else
            {
               // Rare long-suffix (> 255 B post-cpfx, can't fit the u8 in-page suffix): keep the
               // legacy FULL key + value in the external node, page holds only a fragment.
               ins.ext   = large_value_detail::allocate(*_ses, kb, val, _ins_value_type);
               ins.vtype = inline_value::vt_extern;
            }
            ins.value = {};
         }
         // Keep the completed external graph owned until a page adopts it.
         sal::smart_ptr<sal::alloc_header> pending;
         if (ins.ext != sal::null_ptr_address)
         {
            pending = sal::smart_ptr<sal::alloc_header>(_ses, ins.ext);
            SAL_CRASH_POINT("page:external_built");
         }
         // An external record built for a typed value that no page adopts (the page was
         // full: the caller splits and retries with a new record) must die without firing
         // the type's release hook: the value was never stored, and the reference it would
         // have carried is still the caller's until an upsert succeeds.
         // The same holds for a record built for a subtree (a subtree at a key whose suffix
         // does not fit a page record): destroying it as a subtree record would release the
         // subtree's root, which is still the caller's (attach_subtree_step holds it) and
         // which the retry stores again. The root would be one reference short, or already
         // destroyed, when the retried record is published.
         struct unadopted_owner
         {
            radix_tree*                        tree;
            sal::smart_ptr<sal::alloc_header>& record;
            bool                               owns;
            ~unadopted_owner()
            {
               if (owns && record)
                  tree->_ses->template get_ref<key_val_node>(record.address())
                      .modify()
                      .get()
                      ->value_type = inline_value::vt_bytes;
            }
         } unadopted{this, pending,
                     (_ins_value_type == inline_value::vt_typed ||
                      _ins_value_type == inline_value::vt_subtree) &&
                         ins.ext != sal::null_ptr_address};
         if (idx >= 0)  // UPDATE — capture the displaced record's identity first
         {
            const inline_value* old       = pg->record(uint32_t(idx));
            ptr_address         old_child = sal::null_ptr_address;
            if (inline_value::external_value(old->value_type))
            {
               old_child = pg->external_addr(uint32_t(idx));
               _old_value_size =
                   int64_t(_ses->get_ref<key_val_node>(old_child)->value_size());
            }
            else
            {
               _old_value_size = old->is_typed() ? int64_t(old->value_len) - 2 : old->value_len;
               if (old->is_subtree())
                  old_child = old->subtree_root();
            }
            leaf_page::update_op upd{
                .index = uint16_t(idx), .value = ins.value, .vtype = ins.vtype, .ext = ins.ext};
            if (pg->can_apply(upd) == leaf_page::can_apply_mode::modify)
            {
               if (!shared)
               {
                  {  // the replaced inline value's bytes die in place
                     const inline_value* oiv = pg->record(uint32_t(idx));
                     if (oiv->value_type == inline_value::vt_typed) [[unlikely]]
                        fire_typed_release(
                            {reinterpret_cast<const char*>(oiv->value().data()),
                             oiv->value().size()});
                  }
                  ref.modify().get()->apply(upd);
                  SAL_CRASH_POINT("page:update_in_place");
                  (void)pending.take();
                  if (old_child != sal::null_ptr_address)
                     _ses->release(old_child);  // the replaced external/subtree child
                  return page;
               }
               // shared fast path: verbatim 2KB blit + the one mutation in place — never the
               // per-record rebuild (the old child stays owned by the snapshot's page).
               ptr_address np =
                   _ses->alloc<leaf_page>(sal::alloc_hint{}, leaf_page::clone_op{pg});
               mut<leaf_page>(np)->apply(upd);
               SAL_CRASH_POINT("page:update_cloned");
               (void)pending.take();
               retain_page_children(pg, idx);
               return np;
            }
            if (pg->rebuilt_fits(upd))
            {
               // The slot swap's release of `page` balances the retains — and (unique) destroys
               // the old page, releasing every child including the replaced one.
               ptr_address np =
                   _ses->alloc<leaf_page>(sal::alloc_hint{}, leaf_page::clone_update_op{pg, upd, kvres(*_ses)});
               SAL_CRASH_POINT("page:update_rebuilt");
               (void)pending.take();
               retain_page_children(pg, idx);
               return np;
            }
            _page_split = page;
            _page_edge  = false;
            return page;
         }
         _old_value_size = -1;
         if (pg->can_apply(ins) == leaf_page::can_apply_mode::modify)
         {
            if (!shared)
            {
               ref.modify().get()->apply(ins);
               SAL_CRASH_POINT("page:insert_in_place");
               (void)pending.take();
               return page;
            }
            ptr_address np = _ses->alloc<leaf_page>(sal::alloc_hint{}, leaf_page::clone_op{pg});
            mut<leaf_page>(np)->apply(ins);
            SAL_CRASH_POINT("page:insert_cloned");
            (void)pending.take();
            retain_page_children(pg, -1);
            return np;
         }
         if (pg->rebuilt_fits(ins))
         {
            ptr_address np =
                _ses->alloc<leaf_page>(sal::alloc_hint{}, leaf_page::clone_insert_op{pg, ins});
            SAL_CRASH_POINT("page:insert_rebuilt");
            (void)pending.take();
            retain_page_children(pg, -1);
            return np;
         }
         // The page cannot take the key even rebuilt. A key outside the page's common prefix
         // sorts before or after everything the page holds, and halving the page does not
         // help it in: every half keeps the prefix. It gets a page of its own instead (see
         // page_edge_split). So does a key for an empty page whose old prefix and divider
         // leave it no room.
         _page_split = page;
         _page_edge  = pg->num_keys() == 0 || ncp < cp.size();
         return page;
      }

      // Remove `kb` from `page`. Miss → _removed=false, page untouched. The page's LAST entry
      // → _page_emptied; the page is left UNTOUCHED unless `force_empty` (the caller is about
      // to drop the divider + page, whose destroy cascades the children). The LEFTMOST page is
      // never dropped (its "" divider must survive), so its caller passes force_empty and the
      // removal applies for real, leaving a legitimately empty page. Otherwise in-place
      // (unique) or rebuild-clone.
      ptr_address page_apply_remove(ptr_address page, bool parent_shared, bytes kb,
                                    bool force_empty = false)
      {
         auto             ref = _ses->get_ref<leaf_page>(page);
         const leaf_page* pg  = ref.obj();
         const int        idx = pg->find(kb.data(), uint32_t(kb.size()), kvres(*_ses));
         if (idx < 0)
         {
            _removed = false;
            return page;
         }
         _removed                      = true;
         const bool          shared    = parent_shared || ref.ref() > 1;
         const inline_value* old       = pg->record(uint32_t(idx));
         ptr_address         old_child = sal::null_ptr_address;
         if (inline_value::external_value(old->value_type))
            old_child = pg->external_addr(uint32_t(idx));
         else if (old->is_subtree())
            old_child = old->subtree_root();
         _page_emptied = (pg->num_keys() == 1);
         if (_page_emptied && !force_empty)
            return page;
         if (!shared)
         {
            if (old->value_type == inline_value::vt_typed) [[unlikely]]  // a removed typed value dies
               fire_typed_release({reinterpret_cast<const char*>(old->value().data()),
                                   old->value().size()});
            ref.modify().get()->apply(leaf_page::remove_op{uint16_t(idx)});
            SAL_CRASH_POINT("page:remove_in_place");
            if (old_child != sal::null_ptr_address)
               _ses->release(old_child);
            return page;
         }
         // shared fast path: blit + in-place remove (always has room); the removed entry's
         // child stays owned by the snapshot's page (not retained, not referenced by the clone).
         ptr_address np = _ses->alloc<leaf_page>(sal::alloc_hint{}, leaf_page::clone_op{pg});
         mut<leaf_page>(np)->apply(leaf_page::remove_op{uint16_t(idx)});
         SAL_CRASH_POINT("page:remove_cloned");
         retain_page_children(pg, idx);
         return np;
      }

      // Mutating descent along an EXACT divider path: COW the radix path lazily (only when the
      // page op produced a successor) and swap the owning slot. `op(page, parent_shared)` runs
      // the page mutation; an in-place page op leaves the entire radix untouched — the page is
      // the COW unit.
      template <upsert_mode mode, class Op>
      ptr_address page_path_mut(ptr_address addr, bytes D, uint32_t depth, Op&& op)
      {
         switch (kind_of(_ses->get_ref<sal::alloc_header>(addr)->type()))
         {
            case radix_kind::setlist_u8:
               return page_router_mut<mode, setlist_u8>(addr, D, depth, std::forward<Op>(op));
            case radix_kind::c1_32:
               return page_router_mut<mode, bitset_c1_32>(addr, D, depth, std::forward<Op>(op));
            case radix_kind::c1_64:
               return page_router_mut<mode, bitset_c1_64>(addr, D, depth, std::forward<Op>(op));
            case radix_kind::c1_128:
               return page_router_mut<mode, bitset_c1_128>(addr, D, depth, std::forward<Op>(op));
            case radix_kind::bitset_r2:
               return page_router_mut<mode, bitset_r2_node>(addr, D, depth, std::forward<Op>(op));
            case radix_kind::bitset_r4:
               return page_router_mut<mode, bitset_r4_node>(addr, D, depth, std::forward<Op>(op));
            case radix_kind::full:
               return page_router_mut<mode, full_radix_node>(addr, D, depth, std::forward<Op>(op));
            case radix_kind::setlist_u16:  // the page-model radix never builds wide/bucket tiers
            case radix_kind::setlist_u32:
            case radix_kind::bucket:
            case radix_kind::leaf:
            case radix_kind::bitset1:
            case radix_kind::page:  // callers resolve pages before descending
               break;
         }
         unknown_node_kind();
      }
      template <upsert_mode mode, class N, class Op>
      ptr_address page_router_mut(ptr_address addr, bytes D, uint32_t depth, Op&& op)
      {
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);
         if constexpr (!mode.is_shared())
         {
            if (ref.ref() > 1)
               return page_router_mut<upsert_mode{upsert_mode::shared}, N>(addr, D, depth,
                                                                           std::forward<Op>(op));
         }
         const N* cn  = ref.obj();
         bytes    pfx = cn->prefix();
         assert(pfx.size() <= D.size() - depth &&
                (!pfx.size() || std::memcmp(pfx.data(), D.data() + depth, pfx.size()) == 0));
         depth += uint32_t(pfx.size());
         if (depth == D.size())  // the divider's page is this node's term
         {
            ptr_address page = cn->term_value().as_pointer();
            ptr_address np   = op(page, mode.is_shared());
            if (np != page)
            {
               N* n = cow<mode, N>(ref, addr);
               n->set_term_value(smart_address::from_pointer(np));
               SAL_CRASH_POINT("router:term_repointed");
               _ses->release(page);
            }
            return addr;
         }
         const uint8_t   b     = D[depth];
         sal::alloc_hint chint = cn->branch_hint(b);
         smart_address   sa    = cn->find(b);
         assert(sa.is_pointer());
         ptr_address child = sa.as_pointer();
         const bool  csh   = mode.is_shared() || range_shared<N>(cn, range_of(b));
         ptr_address nc;
         if (is_page(*_ses, child))
            nc = op(child, csh);
         else
            nc = csh ? page_path_mut<upsert_mode{upsert_mode::shared}>(child, D, depth + 1, op)
                     : page_path_mut<mode>(child, D, depth + 1, op);
         if (nc != child)
         {
            const ptr_address pre = addr;
            N*                n   = cow<mode, N>(ref, addr);
            SAL_CRASH_POINT("router:made_writable");
            rr_diverge<N>(n, range_of(b));
            if (!n->set_branch(b, smart_address::from_pointer(nc)))
            {
               if constexpr (!std::is_same_v<N, full_radix_node>)
               {
                  // grow_and_set re-homes skipping b and sets b→nc; releasing the COW'd addr
                  // destroys it, releasing the OLD child at b — so no manual child release.
                  ptr_address up =
                      grow_and_set<N>(addr, b, smart_address::from_pointer(nc), bytes{}, bytes{},
                                      chint);
                  SAL_CRASH_POINT("router:grown");
                  if (addr != pre)
                     _ses->release(addr);
                  return up;
               }
            }
            else
            {
               SAL_CRASH_POINT("router:branch_repointed");
               _ses->release(child);
            }
         }
         return addr;
      }

      // Insert divider D (fully path-consumed) → `page` into the radix. The caller's ref on
      // `page` is donated to the slot. Mirrors router_insert minus all terminal-suffix
      // machinery: every slot written is a direct page pointer.
      template <upsert_mode mode>
      ptr_address divider_insert(ptr_address addr, bytes D, uint32_t depth, ptr_address page)
      {
         switch (kind_of(_ses->get_ref<sal::alloc_header>(addr)->type()))
         {
            case radix_kind::setlist_u8:
               return divider_insert_in<mode, setlist_u8>(addr, D, depth, page);
            case radix_kind::c1_32:
               return divider_insert_in<mode, bitset_c1_32>(addr, D, depth, page);
            case radix_kind::c1_64:
               return divider_insert_in<mode, bitset_c1_64>(addr, D, depth, page);
            case radix_kind::c1_128:
               return divider_insert_in<mode, bitset_c1_128>(addr, D, depth, page);
            case radix_kind::bitset_r2:
               return divider_insert_in<mode, bitset_r2_node>(addr, D, depth, page);
            case radix_kind::bitset_r4:
               return divider_insert_in<mode, bitset_r4_node>(addr, D, depth, page);
            case radix_kind::full:
               return divider_insert_in<mode, full_radix_node>(addr, D, depth, page);
            case radix_kind::setlist_u16:
            case radix_kind::setlist_u32:
            case radix_kind::bucket:
            case radix_kind::leaf:
            case radix_kind::bitset1:
            case radix_kind::page:
               break;
         }
         unknown_node_kind();
      }
      // The divider's tail past `depth` compresses into a CHAIN of fresh children: each
      // node holds up to prefix_capacity tail bytes (long dividers — deep nested-prefix
      // keysets — need several), the deepest holds the page as its term. Returns the
      // branch target for byte D[depth]. Built bottom-up so every link is set on a node
      // this op just created.
      // The largest prefix a divider setlist_u8 may carry. A setlist_u8 holds up to
      // setlist_u8::prefix_capacity (97), but when it fills and grow_and_set upgrades it to a wider
      // tier, try_upgrade copies the prefix into that tier — whose prefix_capacity is SMALLER
      // (bitset_r2/r4 and full are all 65; c1 only takes empty-prefix nodes, see try_upgrade_c1).
      // A 66..97-byte prefix overflows the wider node's CL0 into its branch storage → corruption →
      // SIGBUS. INVARIANT: never create a divider node whose prefix exceeds divider_prefix_cap, so
      // it is always upgradeable. (This subsumes the old room_for_one cap — 65 < 92 < 97.)
      static constexpr int divider_prefix_cap = bitset_r2_node::prefix_capacity;
      static_assert(bitset_r4_node::prefix_capacity >= divider_prefix_cap &&
                        full_radix_node::prefix_capacity >= divider_prefix_cap,
                    "divider_prefix_cap must be the SMALLEST upgrade-tier prefix_capacity");

      smart_address make_divider_tail(bytes D, uint32_t depth, ptr_address page,
                                      sal::alloc_hint hint)
      {
         (void)D;
         (void)depth;
         (void)hint;
         return smart_address::from_pointer(page);
      }

      // Build a path-compression chain that consumes `seg` and ends by branching into `tail`
      // (seg.back() is the branch byte into tail). Each link is a fresh single-branch setlist_u8
      // whose prefix is capped at divider_prefix_cap so it always has room for its branch AND stays
      // upgradeable — the invariant make_divider_tail / divider_prefix_split rely on for long keys.
      ptr_address make_prefix_chain(bytes seg, ptr_address tail, sal::alloc_hint hint)
      {
         constexpr int branch_cap = divider_prefix_cap;
         ptr_address   node = tail;
         int           hi   = int(seg.size());
         while (hi > 0)
         {
            const int   bpos = hi - 1;  // branch byte into `node`
            const int   s    = bpos - branch_cap > 0 ? bpos - branch_cap : 0;
            ptr_address up = make_node<setlist_u8>(seg.subspan(size_t(s), size_t(bpos - s)), hint);
            [[maybe_unused]] bool ok =
                mut<setlist_u8>(up)->set_branch(seg[size_t(bpos)], smart_address::from_pointer(node));
            assert(ok && "prefix-chain node (prefix<=branch_cap) always fits one branch");
            node = up;
            hi   = s;
         }
         return node;
      }
      template <upsert_mode mode, class N>
      ptr_address divider_insert_in(ptr_address addr, bytes D, uint32_t depth, ptr_address page)
      {
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);
         if constexpr (!mode.is_shared())
         {
            if (ref.ref() > 1)
               return divider_insert_in<upsert_mode{upsert_mode::shared}, N>(addr, D, depth, page);
         }
         const N*  cn    = ref.obj();
         bytes     pfx   = cn->prefix();
         const int avail = int(D.size() - depth);
         const int maxp  = std::min(int(pfx.size()), avail);
         const int p = common_prefix_len(D.subspan(depth, size_t(maxp)), pfx.subspan(0, size_t(maxp)));
         if (p < int(pfx.size()))
            return divider_prefix_split<mode, N>(addr, D, depth, p, page);
         depth += uint32_t(pfx.size());
         if (depth == D.size())  // divider == path → this node's term
         {
            assert(cn->term_value().is_null() && "divider already present");
            N* n = cow<mode, N>(ref, addr);
            n->set_term_value(smart_address::from_pointer(page));
            return addr;
         }
         const uint8_t   b     = D[depth];
         sal::alloc_hint chint = cn->branch_hint(b);
         smart_address   sa    = cn->find(b);
         if (sa.is_null())  // new slot at b
         {
            smart_address     sab = make_divider_tail(D, depth, page, chint);
            const ptr_address pre = addr;
            N*                n   = cow<mode, N>(ref, addr);
            rr_diverge<N>(n, range_of(b));
            if (n->set_branch(b, sab))
               return addr;
            if constexpr (!std::is_same_v<N, full_radix_node>)
            {
               ptr_address up = grow_and_set<N>(addr, b, sab, bytes{}, bytes{}, chint);
               if (addr != pre)
                  _ses->release(addr);
               return up;
            }
            return addr;  // full_radix never fails set_branch
         }
         ptr_address child = sa.as_pointer();
         if (is_page(*_ses, child))
         {
            // The slot at b is a directly-attached page P (canonical collapse: P carries its own
            // divider in leaf_page::_div, no single-child setlist). D is the new page's divider; it
            // shares [path][b] with P and diverges in the tail → build np, the divergence router:
            // the SHARED continuation common(P._div, D) is np's prefix, with P under its divergence
            // byte (or np's term if P's divider ends at the shared prefix) AND D's page under its
            // own. Reading P._div (the true separator, NOT cpfx) keeps P's discriminator exact.
            // Mirrors divider_prefix_split: np capped at divider_prefix_cap (upgrade-safe), a longer
            // shared prefix spills to a leading single-branch make_prefix_chain.
            // depth+1 == D.size() is legitimate: the new divider ends AT the branch byte while an
            // existing page sits there with a LONGER divider — the new page becomes np's term (the
            // lesser divider) and the existing page hangs under its own tail byte. ntail is then
            // empty → cpl 0 → handled below. A true duplicate (D == P._div exactly) is caught by the
            // cpl!=ptail.size() assert in the d2==D.size() branch.
            bytes      pdv   = page_at(*_ses, child)->divider();
            bytes      ptail = pdv.size() > depth + 1 ? pdv.subspan(depth + 1) : bytes{};
            bytes      ntail = D.subspan(depth + 1);
            const int  mcl   = int(std::min(ptail.size(), ntail.size()));
            const int  cpl =
                common_prefix_len(ptail.subspan(0, size_t(mcl)), ntail.subspan(0, size_t(mcl)));
            constexpr int  npcap      = divider_prefix_cap;
            const int      np_pfx_len = cpl <= npcap ? cpl : npcap;
            const uint32_t np_from    = depth + 1 + uint32_t(cpl) - uint32_t(np_pfx_len);
            ptr_address    np = make_node<setlist_u8>(D.subspan(np_from, size_t(np_pfx_len)), chint);
            _ses->retain(child);  // np takes np's ref on the page (balances the old slot release)
            if (cpl == int(ptail.size()))  // P's divider ends at the shared prefix → P is np's term
               mut<setlist_u8>(np)->set_term_value(smart_address::from_pointer(child));
            else
            {
               [[maybe_unused]] bool okp = mut<setlist_u8>(np)->set_branch(
                   ptail[size_t(cpl)], smart_address::from_pointer(child));
               assert(okp && "fresh setlist always fits the first branch");
            }
            const uint32_t d2 = depth + 1 + uint32_t(cpl);
            if (d2 == D.size())  // the new divider ends at the shared prefix → its page is np's term
            {
               assert(cpl != int(ptail.size()) && "duplicate divider");
               mut<setlist_u8>(np)->set_term_value(smart_address::from_pointer(page));
            }
            else
            {
               smart_address         sab = make_divider_tail(D, d2, page, chint);
               [[maybe_unused]] bool ok  = mut<setlist_u8>(np)->set_branch(D[size_t(d2)], sab);
               assert(ok && "np (<=2 entries, capped prefix) always fits the second branch");
            }
            ptr_address nmid =
                cpl <= npcap
                    ? np
                    : make_prefix_chain(D.subspan(depth + 1, size_t(np_from - (depth + 1))), np,
                                        chint);
            smart_address msa = smart_address::from_pointer(nmid);
            const ptr_address pre = addr;
            N*                n   = cow<mode, N>(ref, addr);
            rr_diverge<N>(n, range_of(b));
            if (n->set_branch(b, msa))
            {
               _ses->release(child);  // the old branch's ref on the page (moved into mid's term)
               return addr;
            }
            if constexpr (!std::is_same_v<N, full_radix_node>)
            {
               // grow_and_set skips b; releasing the COW'd addr destroys it, releasing the old
               // page ref at b — so no manual release here.
               ptr_address up = grow_and_set<N>(addr, b, msa, bytes{}, bytes{}, chint);
               if (addr != pre)
                  _ses->release(addr);
               return up;
            }
            return addr;
         }
         // router child → recurse
         ptr_address nc;
         if constexpr (mode.is_shared())
            nc = divider_insert<mode>(child, D, depth + 1, page);
         else
            nc = range_shared<N>(cn, range_of(b))
                     ? divider_insert<upsert_mode{upsert_mode::shared}>(child, D, depth + 1, page)
                     : divider_insert<mode>(child, D, depth + 1, page);
         if (nc != child)
         {
            const ptr_address pre = addr;
            N*                n   = cow<mode, N>(ref, addr);
            rr_diverge<N>(n, range_of(b));
            if (!n->set_branch(b, smart_address::from_pointer(nc)))
            {
               if constexpr (!std::is_same_v<N, full_radix_node>)
               {
                  ptr_address up = grow_and_set<N>(addr, b, smart_address::from_pointer(nc),
                                                   bytes{}, bytes{}, chint);
                  if (addr != pre)
                     _ses->release(addr);
                  return up;
               }
            }
            else
               _ses->release(child);
         }
         return addr;
      }
      // Divider analogue of prefix_split: a new parent takes pfx[0..p); the demoted node hangs
      // under pfx[p]; the divider's continuation (term or compressed tail) attaches to the parent.
      template <upsert_mode mode, class N>
      ptr_address divider_prefix_split(ptr_address addr, bytes D, uint32_t depth, int p,
                                       ptr_address page)
      {
         uint8_t ob = _ses->get_ref<N>(addr).obj()->prefix()[size_t(p)];

         ptr_address dnode = addr;
         N*          d     = cow<mode, N>(dnode);
         int         newpl = int(d->prefix_length()) - (p + 1);
         uint8_t     pbuf[sal::cacheline_size];
         std::memcpy(pbuf, d->prefix().data() + p + 1, size_t(newpl));
         reprefix<N>(dnode, bytes{pbuf, size_t(newpl)});

         // np is the divergence node: it carries ob->dnode PLUS the divider's continuation (a 2nd
         // branch, or a term). That is up to TWO (stem,branch) pairs. divider_prefix_cap (65) leaves
         // ample room for both ((65 + 2*(stem_width+4)) = 75 <= 97) AND keeps np upgradeable. A
         // longer common prefix `p` cannot live in np (it would overflow a wider tier on upgrade ->
         // SIGBUS); represent its leading bytes as a single-branch chain above np instead.
         constexpr int     npcap       = divider_prefix_cap;
         const int         np_pfx_len  = p <= npcap ? p : npcap;
         const uint32_t    np_pfx_from = depth + uint32_t(p) - uint32_t(np_pfx_len);
         ptr_address       np = make_node<setlist_u8>(D.subspan(np_pfx_from, size_t(np_pfx_len)));
         mut<setlist_u8>(np)->set_branch(ob, smart_address::from_pointer(dnode));
         if (dnode == addr)
            _ses->retain(dnode);  // in-place demote: np is a 2nd parent; caller releases addr

         const uint32_t d2 = depth + uint32_t(p);
         if (d2 == D.size())
            mut<setlist_u8>(np)->set_term_value(smart_address::from_pointer(page));
         else
         {
            smart_address         sab = make_divider_tail(D, d2, page, sal::alloc_hint{});
            [[maybe_unused]] bool ok  = mut<setlist_u8>(np)->set_branch(D[d2], sab);
            assert(ok && "np prefix capped at npcap → two-entry setlist always fits");
         }
         if (p <= npcap)
            return np;
         // Leading chain for the common prefix bytes that did not fit in np: D[depth .. np_pfx_from)
         // (its last byte is the branch into np).
         return make_prefix_chain(D.subspan(depth, size_t(np_pfx_from - depth)), np,
                                  sal::alloc_hint{});
      }

      // Remove divider D from the radix. `out_page` receives the page the slot held; when
      // `keep` the caller takes ownership of one ref on it (promotion), else the working
      // tree's claim is released here (the legacy replace discipline). Sets _removed/_emptied.
      template <upsert_mode mode>
      ptr_address divider_remove(ptr_address addr, bytes D, uint32_t depth, ptr_address& out_page,
                                 bool keep)
      {
         switch (kind_of(_ses->get_ref<sal::alloc_header>(addr)->type()))
         {
            case radix_kind::setlist_u8:
               return divider_remove_in<mode, setlist_u8>(addr, D, depth, out_page, keep);
            case radix_kind::c1_32:
               return divider_remove_in<mode, bitset_c1_32>(addr, D, depth, out_page, keep);
            case radix_kind::c1_64:
               return divider_remove_in<mode, bitset_c1_64>(addr, D, depth, out_page, keep);
            case radix_kind::c1_128:
               return divider_remove_in<mode, bitset_c1_128>(addr, D, depth, out_page, keep);
            case radix_kind::bitset_r2:
               return divider_remove_in<mode, bitset_r2_node>(addr, D, depth, out_page, keep);
            case radix_kind::bitset_r4:
               return divider_remove_in<mode, bitset_r4_node>(addr, D, depth, out_page, keep);
            case radix_kind::full:
               return divider_remove_in<mode, full_radix_node>(addr, D, depth, out_page, keep);
            case radix_kind::setlist_u16:
            case radix_kind::setlist_u32:
            case radix_kind::bucket:
            case radix_kind::leaf:
            case radix_kind::bitset1:
            case radix_kind::page:
               break;
         }
         unknown_node_kind();
      }
      template <upsert_mode mode, class N>
      ptr_address divider_remove_in(ptr_address addr, bytes D, uint32_t depth,
                                    ptr_address& out_page, bool keep)
      {
         const ptr_address pre = addr;
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);
         if constexpr (!mode.is_shared())
         {
            if (ref.ref() > 1)
               return divider_remove_in<upsert_mode{upsert_mode::shared}, N>(addr, D, depth,
                                                                             out_page, keep);
         }
         const N* cn  = ref.obj();
         bytes    pfx = cn->prefix();
         assert(pfx.size() <= D.size() - depth &&
                (!pfx.size() || std::memcmp(pfx.data(), D.data() + depth, pfx.size()) == 0));
         depth += uint32_t(pfx.size());
         if (depth == D.size())  // the divider is this node's term
         {
            smart_address tv = cn->term_value();
            assert(tv.is_pointer());
            out_page = tv.as_pointer();
            N* n     = cow<mode, N>(ref, addr);
            if (!keep)
               _ses->release(out_page);
            n->set_term_value(smart_address::null());
            _removed = true;
            _emptied = (n->branch_count() == 0);
            return finalize_remove<N>(addr, pre);
         }
         const uint8_t b  = D[depth];
         smart_address sa = cn->find(b);
         assert(sa.is_pointer());
         ptr_address child = sa.as_pointer();
         if (is_page(*_ses, child))
         {
            assert(depth + 1 == D.size() && "divider path mismatch");
            out_page = child;
            N* n     = cow<mode, N>(ref, addr);
            rr_diverge<N>(n, range_of(b));
            if (!keep)
               _ses->release(out_page);
            n->remove_branch(typename N::stem_type(b));
            _removed = true;
            _emptied = (n->branch_count() == 0 && !n->has_term_value());
            return finalize_remove<N>(addr, pre);
         }
         ptr_address nc;
         if constexpr (mode.is_shared())
            nc = divider_remove<mode>(child, D, depth + 1, out_page, keep);
         else
            nc = range_shared<N>(cn, range_of(b))
                     ? divider_remove<upsert_mode{upsert_mode::shared}>(child, D, depth + 1,
                                                                        out_page, keep)
                     : divider_remove<mode>(child, D, depth + 1, out_page, keep);
         if (_emptied)  // the child router drained → drop its branch and free it
         {
            N* n = cow<mode, N>(ref, addr);
            rr_diverge<N>(n, range_of(b));
            n->remove_branch(typename N::stem_type(b));
            _ses->release(nc);
            if (nc != child)
               _ses->release(child);
            _emptied = (n->branch_count() == 0 && !n->has_term_value());
            return finalize_remove<N>(addr, pre);
         }
         if (nc != child)
         {
            N* n = cow<mode, N>(ref, addr);
            rr_diverge<N>(n, range_of(b));
            n->set_branch(b, smart_address::from_pointer(nc));
            _ses->release(child);
         }
         return addr;
      }

      // Split the full page (stashed in _page_split) into L/R halves, repoint its slot → L,
      // insert the new divider → R. `cur` is the current working root (== the page itself for
      // a single-page tree); `borrowed` is the caller's pre-image, never released here.
      ptr_address page_split_and_attach(ptr_address cur, ptr_address borrowed, bytes D)
      {
         const ptr_address P = _page_split;
         _page_split         = sal::null_ptr_address;
         auto             pr = _ses->get_ref<leaf_page>(P);
         const leaf_page* pg = pr.obj();
         if (pg->num_keys() < 2)  // a 1-entry page can never split — fail loudly in every build.
         {
            // Structurally unreachable while the key contract holds (max_inline_entry caps any
            // entry's in-page footprint, so a page always fits >= 2), i.e. reaching here means
            // a logic bug or corruption — but observable > UB and a server must not die on one
            // bad tree: throw instead of abort. Nothing is allocated yet at this point and the
            // page still hangs off its slot, so unwinding is clean (mutate_root restores the
            // root; page_upsert_step releases its working claim on the way out).
            throw std::runtime_error("heart: split invariant violated — page " +
                                     std::to_string(*P) + " holds " +
                                     std::to_string(unsigned(pg->num_keys())) +
                                     " key(s) (size " + std::to_string(pg->size()) + ")");
         }
         // Where to split, and each half's common prefix, chosen so that both halves are
         // known to fit a page before either is built (leaf_page::plan_split). Nothing is
         // allocated yet, so a plan that cannot be made unwinds like the case above.
         const auto plan = pg->plan_split(kvres(*_ses));
         if (!plan.ok)
            throw std::runtime_error("sal_art_ref: no split of page " + std::to_string(*P) +
                                     " (" + std::to_string(unsigned(pg->num_keys())) +
                                     " keys) fits two pages");
         const auto& sp = plan.pos;
         // The page divider rides on each half: L keeps pg's own divider (its radix slot is
         // unchanged); R's divider IS the split divider (sp.divider()), the same key inserted into
         // the radix below. This populates leaf_page::_div for the canonical-collapse read path.
         ptr_address      L  = _ses->alloc<leaf_page>(
             sal::alloc_hint{},
             leaf_page::split_op{pg, 0, sp.index, kvres(*_ses), pg->divider(), plan.left_cp,
                                 plan.left_w});
         _ses->get_ref<leaf_page>(L).obj()->for_each_child([&](ptr_address c) { _ses->retain(c); });
         retain_page_typed(L);
         SAL_CRASH_POINT("split:left_built");
         ptr_address R = _ses->alloc<leaf_page>(
             sal::alloc_hint{},
             leaf_page::split_op{pg, sp.index, uint16_t(pg->num_keys()), kvres(*_ses), sp.divider(),
                                 plan.right_cp, plan.right_w});
         _ses->get_ref<leaf_page>(R).obj()->for_each_child([&](ptr_address c) { _ses->retain(c); });
         retain_page_typed(R);
         SAL_CRASH_POINT("split:right_built");
         ++_cc.page_splits;
         if (cur == P)  // the root IS the page → grow a radix above it
         {
            ptr_address nr = make_node<setlist_u8>(bytes{});
            mut<setlist_u8>(nr)->set_term_value(smart_address::from_pointer(L));  // "" divider
            ptr_address nr2 = divider_insert<upsert_mode{}>(nr, sp.divider(), 0, R);
            if (nr2 != nr)
               _ses->release(nr);
            if (cur != borrowed)
               _ses->release(cur);  // an intermediate page root from an earlier pass
            return nr2;
         }
         // (a) swap P's slot → L (releases the working claim on P; halves' retains balance)
         ptr_address c2 = page_path_mut<upsert_mode{}>(
             cur, D, 0, [&](ptr_address pgp, bool) {
                assert(pgp == P);
                (void)pgp;
                return L;
             });
         // (b) the new divider → R. `cur` stays the caller's to release until both steps are
         // done: on a throw the caller releases it, and only the generation made here is
         // dropped here (it was released twice before: here after (a) and again by the caller).
         ptr_address c3;
         try
         {
            c3 = divider_insert<upsert_mode{}>(c2, sp.divider(), 0, R);
         }
         catch (...)
         {
            if (c2 != cur)
               _ses->release(c2);
            throw;
         }
         if (c2 != cur)
         {
            if (cur != borrowed)
               _ses->release(cur);
            if (c3 != c2)
               _ses->release(c2);
         }
         else if (c3 != cur && cur != borrowed)
            _ses->release(cur);
         return c3;
      }

      // Give the key a page of its own. Called when the page it routes to (stashed in
      // _page_split) cannot take it even rebuilt and halving that page would not help: the
      // key lies outside the page's common prefix, so it sorts below or above everything
      // the page holds, or the page is empty. The new page holds only the key:
      //   above: the page stays as it is; the new page is attached at the shortest
      //          separator between the page's last key and the key.
      //   below: the new page takes the page's slot and divider; the page is rebuilt under
      //          a new divider, a prefix of its first key, and attached there.
      //   empty: the new page replaces it under the same divider.
      // Returns the working root with the key stored; `cur`/`borrowed` as in
      // page_split_and_attach.
      ptr_address page_edge_split(ptr_address cur, ptr_address borrowed, bytes D, bytes kb,
                                  bytes val)
      {
         const ptr_address P = _page_split;
         _page_split         = sal::null_ptr_address;
         _page_edge          = false;
         auto             pr = _ses->get_ref<leaf_page>(P);
         const leaf_page* pg = pr.obj();
         enum class side { empty, above, below } where = side::empty;
         leaf_page::split_pos sp{};
         if (pg->num_keys() != 0)
         {
            auto     cp  = pg->common_prefix();
            uint32_t ncp = 0;
            while (ncp < cp.size() && ncp < kb.size() && cp[ncp] == kb[ncp])
               ++ncp;
            if (ncp >= cp.size())
               throw std::logic_error("sal_art_ref: a key inside its page's prefix was sent to a page of its own");
            where = (ncp < kb.size() && kb[ncp] > cp[ncp]) ? side::above : side::below;
            sp    = where == side::above
                        ? pg->divider_above(kb.data(), uint32_t(kb.size()), kvres(*_ses))
                        : pg->divider_below(kb.data(), uint32_t(kb.size()), kvres(*_ses));
            if (where == side::below && !pg->redivided_fits(sp.divider()))
               throw std::runtime_error("sal_art_ref: page " + std::to_string(*P) +
                                        " does not fit under its new divider");
         }
         // The key's page: empty, reached through its divider, then the ordinary insert.
         sal::smart_ptr<sal::alloc_header> fresh(
             _ses, _ses->alloc<leaf_page>(sal::alloc_hint{},
                                          leaf_page::empty_op{where == side::above
                                                                  ? sp.divider()
                                                                  : pg->divider()}));
         if (page_apply_upsert(fresh.address(), false, kb, val) != fresh.address() ||
             _page_split != sal::null_ptr_address)
         {
            _page_split = sal::null_ptr_address;
            _page_edge  = false;
            throw std::runtime_error("sal_art_ref: a key of " + std::to_string(kb.size()) +
                                     " bytes does not fit a page of its own");
         }
         // The page rebuilt under its new divider (below): it shares the old page's children.
         sal::smart_ptr<sal::alloc_header> moved;
         if (where == side::below)
         {
            moved = sal::smart_ptr<sal::alloc_header>(
                _ses, _ses->alloc<leaf_page>(sal::alloc_hint{},
                                             leaf_page::redivide_op{pg, sp.divider()}));
            _ses->get_ref<leaf_page>(moved.address())
                .obj()
                ->for_each_child([&](ptr_address c) { _ses->retain(c); });
            retain_page_typed(moved.address());
         }
         SAL_CRASH_POINT("split:edge_built");
         ++_cc.page_splits;
         // What stays in the old page's slot, and what is attached at the new divider.
         const ptr_address in_slot  = where == side::above ? P : fresh.address();
         const ptr_address attached = where == side::above   ? fresh.address()
                                      : where == side::below ? moved.address()
                                                             : sal::null_ptr_address;
         if (cur == P)  // the root IS the page
         {
            if (where == side::empty)
            {
               if (cur != borrowed)
                  _ses->release(cur);
               return fresh.take();
            }
            // grow a radix above: the "" divider leads to the slot's page
            sal::smart_ptr<sal::alloc_header> top(_ses, make_node<setlist_u8>(bytes{}));
            mut<setlist_u8>(top.address())->set_term_value(smart_address::from_pointer(in_slot));
            if (where == side::above)
               _ses->retain(P);     // the new root holds the page with a reference of its own
            else
               (void)fresh.take();  // the new root owns the key's page
            // A throw from here unwinds `top` with exactly what it owns.
            const ptr_address nr2 =
                divider_insert<upsert_mode{}>(top.address(), sp.divider(), 0, attached);
            (void)(where == side::above ? fresh.take() : moved.take());
            if (nr2 == top.address())
               (void)top.take();
            if (cur != borrowed)
               _ses->release(cur);  // the loop's claim on the old page
            return nr2;
         }
         ptr_address c2 = cur;
         if (where != side::above)  // the key's page takes the old page's slot
         {
            c2 = page_path_mut<upsert_mode{}>(cur, D, 0, [&](ptr_address pgp, bool) {
               assert(pgp == P);
               (void)pgp;
               return in_slot;
            });
            (void)fresh.take();
            if (where == side::empty)
            {
               if (c2 != cur && cur != borrowed)
                  _ses->release(cur);
               return c2;
            }
         }
         // `cur` stays the caller's to release until the whole change is in place: on a
         // throw the caller releases it, and only the generation made here is dropped here.
         ptr_address c3;
         try
         {
            c3 = divider_insert<upsert_mode{}>(c2, sp.divider(), 0, attached);
         }
         catch (...)
         {
            if (c2 != cur)
               _ses->release(c2);
            throw;
         }
         (void)(where == side::above ? fresh.take() : moved.take());
         if (c2 != cur)
         {
            if (cur != borrowed)
               _ses->release(cur);
            if (c3 != c2)
               _ses->release(c2);
         }
         else if (c3 != cur && cur != borrowed)
            _ses->release(cur);
         return c3;
      }

      // Page-model upsert entry: route → page op; in-place when unique, rebuild-clone when
      // shared/full, split-and-retry when the entry can't fit a rebuilt page.
      [[nodiscard]] ptr_address page_upsert_step(ptr_address root, bytes kb, bytes val)
      {
         _old_value_size = -1;
         _page_split     = sal::null_ptr_address;
         if (root == sal::null_ptr_address)  // first write → the tree is one page
         {
            sal::smart_ptr<sal::alloc_header> base(_ses, make_page());
            ptr_address nr = page_apply_upsert(base.address(), false, kb, val);
            assert(_page_split == sal::null_ptr_address && "a fresh page always fits one entry");
            return nr == base.address() ? base.take() : nr;
         }
         ptr_address cur = root;  // borrowed; intermediates are owned (released) in-loop
         for (int pass = 0;; ++pass)
         {
            // Hard guard in EVERY build — a silent livelock is the worst failure. Each pass
            // splits the page the key routes to and leaves fewer keys in it; a page of large
            // records can shed them one at a time, and holds a few hundred at most.
            if (pass >= 512)
            {
               // Every completed pass was a split (a content no-op), so the working generation
               // is a CONSISTENT tree holding all pre-existing keys; release it if we own it
               // and throw — mutate_root restores the pre-image root. (The old abort here also
               // printed *_page_split, which is always null at the loop head — split_and_attach
               // consumes the stash — so that diagnostic was itself dead.)
               if (cur != root)
                  _ses->release(cur);
               throw std::runtime_error("heart: page split did not converge after 512 passes (key " +
                                        std::to_string(kb.size()) + " bytes)");
            }
            _page_split = sal::null_ptr_address;
            _page_edge  = false;
            uint8_t     db[1056];  // divider <= max key + 1 (the 520 smashed at deep keys)
            uint32_t    dl = 0;
            ptr_address nxt;
            if (is_page(*_ses, cur))
               nxt = page_apply_upsert(cur, false, kb, val);
            else
            {
               route_to_page(*_ses, cur, kb, db, &dl);
               nxt = page_path_mut<upsert_mode{}>(
                   cur, bytes{db, dl}, 0,
                   [&](ptr_address pg, bool sh) { return page_apply_upsert(pg, sh, kb, val); });
            }
            if (_page_split == sal::null_ptr_address)
            {
               if (nxt != cur && cur != root)
                  _ses->release(cur);
               return nxt == cur ? cur : nxt;
            }
            const bool edge = _page_edge;
            try
            {
               if (edge)  // the key gets a page of its own: stored, no retry
                  return page_edge_split(cur, root, bytes{db, dl}, kb, val);
               cur = page_split_and_attach(cur, root, bytes{db, dl});
            }
            catch (...)
            {
               if (cur != root)  // the loop owns its intermediate generations — don't leak one
                  _ses->release(cur);
               throw;
            }
         }
      }

      // Page-model remove entry. An emptied page drops out of the radix together with its
      // divider; the leftmost ("" ) page stays when other pages remain (an empty page routes
      // every lookup to a miss — valid); a root that drains to term-only collapses to its page;
      // an empty single-page tree returns to null.
      [[nodiscard]] ptr_address page_remove_step(ptr_address root, bytes kb, bool& removed)
      {
         _removed      = false;
         _emptied      = false;
         _page_emptied = false;
         removed       = false;
         if (root == sal::null_ptr_address)
            return root;
         if (is_page(*_ses, root))  // single-page tree
         {
            ptr_address np = page_apply_remove(root, false, kb);
            removed        = _removed;
            if (!_removed)
               return root;
            if (_page_emptied)
               return sal::null_ptr_address;  // mutate_root releases the pre-image page
            return np;
         }
         uint8_t  db[1056];  // divider <= max key + 1
         uint32_t dl = 0;
         route_to_page(*_ses, root, kb, db, &dl);
         // The leftmost ("" divider) page is never dropped, so an emptying remove there must
         // APPLY the removal (leaving a legitimately empty page) instead of deferring to the
         // divider-drop path.
         const bool  leftmost = (dl == 0);
         ptr_address nr       = page_path_mut<upsert_mode{}>(
             root, bytes{db, dl}, 0,
             [&](ptr_address pg, bool sh) { return page_apply_remove(pg, sh, kb, leftmost); });
         removed = _removed;
         if (!_removed || !_page_emptied)
            return nr;
         if (leftmost)  // the (now empty) leftmost page stays while other dividers exist
            return page_collapse_root(nr, root);
         // A non-leftmost emptying remove leaves the page untouched (no slot swap): nr == root.
         assert(nr == root);
         _emptied         = false;
         ptr_address dead = sal::null_ptr_address;
         ptr_address n2 = divider_remove<upsert_mode{}>(nr, bytes{db, dl}, 0, dead, /*keep=*/false);
         assert(!_emptied && "the radix root always keeps its \"\" term");
         return page_collapse_root(n2, root);
      }

      // If the page-model radix root holds nothing but its "" term, the tree is one page —
      // promote it (and a drained empty page promotes to the null tree). `borrowed` is the
      // caller's pre-image (released by mutate_root, never here).
      ptr_address page_collapse_root(ptr_address cur, ptr_address borrowed)
      {
         if (cur == sal::null_ptr_address || is_page(*_ses, cur))
            return cur;
         bool        only_term = false;
         ptr_address pagea     = sal::null_ptr_address;
         dispatch(*_ses, cur, [&](auto ad) {
            if (ad.term().is_null())
               return;
            if (!ad.valid(ad.first()))
            {
               only_term = true;
               pagea     = ad.term().as_pointer();
            }
         });
         if (!only_term)
            return cur;
         _ses->retain(pagea);  // the root slot's claim; the dying router's term ref balances
         if (cur != borrowed)
            _ses->release(cur);  // intermediate root: its destroy releases the term ref
         if (_ses->get_ref<leaf_page>(pagea).obj()->num_keys() == 0)
         {
            _ses->release(pagea);  // a drained leftmost page → the empty (null) tree
            return sal::null_ptr_address;
         }
         return pagea;
      }

      // Page-model static point lookup: lower-bound route + in-page fingerprint find. Subtree
      // values read as not-found (use is_subtree()/get_subtree()), matching find_in.
      static bool page_find_value(sal::allocator_session& ses, ptr_address root, bytes kb,
                                  value_detail::stored_value& out)
      {
         const leaf_page* pg = route_to_page(ses, root, kb, nullptr, nullptr);
         const int idx = pg->find(kb.data(), uint32_t(kb.size()), kvres(ses));
         if (idx < 0) return false;
         return page_value(ses, pg, uint32_t(idx), out);
      }
      static bool page_value(sal::allocator_session& ses, const leaf_page* pg, uint32_t idx,
                             value_detail::stored_value& out)
      {
         const inline_value* iv = pg->record(idx);
         if (inline_value::external_value(iv->value_type))
         {
            const auto addr = pg->external_addr(idx);
            const auto& record = *ses.get_ref<key_val_node>(addr).obj();
            if (record.is_subtree()) return false;
            out = value_detail::from_external(record, addr);
            return true;
         }
         if (iv->is_subtree()) return false;
         const std::string_view stored{reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                       iv->value_len};
         if (iv->is_typed()) [[unlikely]]
            out = value_detail::from_typed(stored);
         else
            out = {stored, sal::null_ptr_address};
         return true;
      }
      bool page_find_in(sal::allocator_session& ses, ptr_address root, bytes kb, value_view& out)
      {
         value_detail::stored_value value;
         if (!page_find_value(ses, root, kb, value)) return false;
         out = direct_value_view(ses, value);
         return true;
      }
      // The direct API returns a naked view. A crossing value uses scratch valid until its
      // next direct read; the public cursor API returns value_ref.
      value_view direct_value_view(sal::allocator_session& ses, value_detail::stored_value v)
      {
         if (v.directory == sal::null_ptr_address) return v.bytes;
         std::optional<sal::read_lock> pin(ses.lock());
         auto value = value_detail::read(ses, v, 0, adb::read_all, pin);
         _read_buffer = std::move(value).copy_or_take<std::string>();
         return _read_buffer;
      }
      value_view external_bytes(ptr_address a)
      {
         return direct_value_view(*_ses, value_detail::from_external(*_ses, a));
      }

      // In-order emission of one page (keys are SELF-CONTAINED: cpfx + suffix — the radix path
      // above is divider bytes, never key bytes). vt_subtree values emit their raw 4-byte root
      // (legacy inline parity); vt_extern values dereference to the real bytes; a typed value
      // emits its token (the type number is not part of what a byte walk sees).
      template <class Cb>
      void emit_page(const leaf_page* pg, Cb& cb)
      {
         KB kb2;
         auto cp = pg->common_prefix();
         for (uint32_t i = 0; i < pg->num_keys(); ++i)
         {
            const auto* iv = pg->record(i);
            kb2.truncate(0);
            if (iv->value_type == inline_value::vt_extern)
            {
               const auto key = kvres(*_ses)(pg->external_addr(i));
               kb2.append(key.data(), key.size());
            }
            else
            {
               kb2.append(cp.data(), cp.size());
               const auto suffix = pg->suffix(i);
               kb2.append(suffix.data(), suffix.size());
            }
            if (inline_value::external_value(iv->value_type))
            {
               std::optional<sal::read_lock> pin(_ses->lock());
               auto value = value_detail::read(*_ses, value_detail::from_external(*_ses, pg->external_addr(i)),
                                               0, adb::read_all, pin);
               cb(key_view(reinterpret_cast<const char*>(kb2.buf), kb2.len), value.view());
            }
            else
            {
               const value_view stored(reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                       iv->value_len);
               // a typed value emits its token, as its external form does
               cb(key_view(reinterpret_cast<const char*>(kb2.buf), kb2.len),
                  iv->is_typed() ? typed_token(stored) : stored);
            }
         }
      }

      // Append an entry's stem bytes (+ inline suffix if terminal) and either yield its
      // value (terminal) or descend into its child to that subtree's minimum.
      template <class Ad>
      static void take_entry_min(sal::allocator_session& ses, Ad ad, int c, KB& kb, value_view& out)
      {
         uint8_t       sb[4];
         const int     nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
         kb.append(sb, size_t(nb));
         const smart_address br = ad.branch(c);
         if (is_terminal_branch(ses, br))
         {
            out = append_terminal(ses, ad.node(), br, kb);
            return;
         }
         descend_min(ses, br.as_pointer(), kb, out);
      }
      // Minimum key of this node's subtree, with the node's prefix ALREADY in kb.
      template <class Ad>
      static void take_min_here(sal::allocator_session& ses, Ad ad, KB& kb, value_view& out)
      {
         if (!ad.term().is_null())  // a key ends here → EOF sorts before any byte
         {
            out = append_terminal(ses, ad.node(), ad.term(), kb);
            return;
         }
         take_entry_min(ses, ad, ad.first(), kb, out);  // else the smallest stem
      }
      static void descend_min(sal::allocator_session& ses, ptr_address addr, KB& kb, value_view& out)
      {
         if (const bucket_node* bn = bucket_at(ses, addr))  // bucket subtree → its smallest entry
         {
            assert(bn->nent > 0);
            out = bucket_entry(bn, 0, kb);
            return;
         }
         dispatch(ses, addr, [&](auto ad) {
            bytes pfx = ad.prefix();
            kb.append(pfx.data(), pfx.size());
            take_min_here(ses, ad, kb, out);
         });
      }

      template <class Ad>
      static bool seek_in(sal::allocator_session& ses, Ad ad, bytes q, uint32_t depth, bool strict,
                          KB& kb, value_view& out)
      {
         const uint32_t save  = kb.len;
         bytes          pfx   = ad.prefix();
         int            avail = int(q.size()) - int(depth);
         if (avail < 0)
            avail = 0;
         const int m = int(pfx.size()) < avail ? int(pfx.size()) : avail;
         int       cmp = 0, j = 0;
         for (; j < m; ++j)
            if (q[depth + j] != pfx[j])
            {
               cmp = q[depth + j] < pfx[j] ? -1 : 1;
               break;
            }
         if (j < m && cmp < 0)  // q < this path → whole subtree > q → its minimum
         {
            kb.append(pfx.data(), pfx.size());
            take_min_here(ses, ad, kb, out);
            return true;
         }
         if (j < m && cmp > 0)  // q > this path → every key here < q
            return false;
         kb.append(pfx.data(), pfx.size());
         if (avail <= int(pfx.size()))
         {
            if (avail < int(pfx.size()))  // q is a strict prefix of the path → all > q
            {
               take_min_here(ses, ad, kb, out);
               return true;
            }
            // q == path exactly
            if (!strict && !ad.term().is_null())
            {
               out = append_terminal(ses, ad.node(), ad.term(), kb);
               return true;
            }
            if (!ad.valid(ad.first()))  // nothing greater (term-only & strict)
            {
               kb.truncate(save);
               return false;
            }
            take_entry_min(ses, ad, ad.first(), kb, out);
            return true;
         }
         // q continues past the prefix; the node term (== path) is < q, so skip it.
         const uint32_t d2 = depth + uint32_t(pfx.size());
         bytes          r  = q.subspan(d2);
         bool           full;
         uint32_t       fc;
         for (int c = ad.lower(r, full, fc); ad.valid(c); c = ad.next(c))
         {
            const uint32_t      cl    = ad.cell(c);
            const bool          exact = full && cl == fc;
            const smart_address br    = ad.branch(c);
            uint8_t             sb[4];
            const int           nb = stem_bytes(cl, ad.stem_width(), ad.cw(), sb);
            const uint32_t      s2 = kb.len;
            kb.append(sb, size_t(nb));
            if (exact)  // stem equals q's stem → q may continue into this branch
            {
               if (is_terminal_branch(ses, br))
               {
                  const value_view v  = append_terminal(ses, ad.node(), br, kb);
                  const int        kc = key_cmp(kb, q);
                  if (kc > 0 || (kc == 0 && !strict))
                  {
                     out = v;
                     return true;
                  }
                  kb.truncate(s2);  // this key <= q (or == q & strict) → try the next stem
               }
               else if (seek(ses, br.as_pointer(), q, d2 + uint32_t(ad.cw()), strict, kb, out))
                  return true;
               else
                  kb.truncate(s2);  // child held nothing >= q → next stem
            }
            else  // strictly greater stem → whole branch > q → its minimum
            {
               if (is_terminal_branch(ses, br))
               {
                  out = append_terminal(ses, ad.node(), br, kb);
                  return true;
               }
               descend_min(ses, br.as_pointer(), kb, out);
               return true;
            }
         }
         kb.truncate(save);
         return false;
      }
      // Seek inside a bucket: the smallest entry whose suffix >= q[depth:] (or > q if strict).
      static bool seek_bucket(const bucket_node* bn, bytes q, uint32_t depth, bool strict, KB& kb,
                              value_view& out)
      {
         const int      avail = int(q.size()) > int(depth) ? int(q.size()) - int(depth) : 0;
         const uint8_t* s     = q.data() + depth;  // not dereferenced when avail == 0
         int            i     = bn->lower_bound(s, avail);                  // first suffix >= q
         if (strict && i < int(bn->nent) && bn->cmp(i, s, avail) == 0) ++i;  // > q for upper_bound
         if (i >= int(bn->nent)) return false;                             // nothing >= q here
         out = bucket_entry(bn, i, kb);
         return true;
      }
      bool seek(ptr_address addr, bytes q, uint32_t depth, bool strict, KB& kb, value_view& out)
      {
         return seek(*_ses, addr, q, depth, strict, kb, out);
      }
      static bool seek(sal::allocator_session& ses, ptr_address addr, bytes q, uint32_t depth,
                       bool strict, KB& kb, value_view& out)
      {
         if (const bucket_node* bn = bucket_at(ses, addr))
            return seek_bucket(bn, q, depth, strict, kb, out);
         return dispatch(ses, addr,
                         [&](auto ad) { return seek_in(ses, ad, q, depth, strict, kb, out); });
      }
      bool ordered_seek(key_view key, bool strict, std::string& out_key, value_view& out_val)
      {
         auto       lock = _ses->lock();
         if (!_root)  // null = empty tree
            return false;
         if (_page_mode)
         {
            // LEAN PATH: one lower-bound route + one in-page SIMD lower_bound — no cursor
            // construction, no second descent. Falls back to the cursor only when the query
            // lands past the routed page's last entry (a page-boundary seek, the rare case).
            bytes             q  = as_bytes(key);
            const kv_resolver R  = kvres(*_ses);
            const leaf_page*  pg = route_to_page(*_ses, _root.address(), q, nullptr, nullptr);
            int               lb = pg->lower_bound(q.data(), uint32_t(q.size()), R);
            // does entry lb's FULL key equal q? (extern keys live whole in the key_val_node)
            auto entry_equals = [&](uint32_t i) {
               const inline_value* iv = pg->record(i);
               if (iv->value_type == inline_value::vt_extern)
               {
                  auto kk = R(pg->external_addr(i));
                  return kk.size() == q.size() &&
                         (q.empty() || std::memcmp(kk.data(), q.data(), q.size()) == 0);
               }
               auto cp  = pg->common_prefix();
               auto sfx = pg->suffix(i);
               return cp.size() + sfx.size() == q.size() &&
                      std::memcmp(cp.data(), q.data(), cp.size()) == 0 &&
                      std::memcmp(sfx.data(), q.data() + cp.size(), sfx.size()) == 0;
            };
            if (strict && lb < int(pg->num_keys()) && entry_equals(uint32_t(lb)))
               ++lb;  // upper_bound: skip the (single possible) exact match
            if (lb < int(pg->num_keys()))
            {
               const inline_value* iv = pg->record(uint32_t(lb));
               if (iv->value_type == inline_value::vt_extern)
               {
                  const auto key = kvres(*_ses)(pg->external_addr(uint32_t(lb)));
                  out_key.assign(reinterpret_cast<const char*>(key.data()), key.size());
                  out_val = external_bytes(pg->external_addr(uint32_t(lb)));
                  return true;
               }
               auto cp  = pg->common_prefix();
               auto sfx = pg->suffix(uint32_t(lb));
               out_key.assign(reinterpret_cast<const char*>(cp.data()), cp.size());
               out_key.append(reinterpret_cast<const char*>(sfx.data()), sfx.size());
               if (iv->value_type == inline_value::vt_extern_val)  // value-only external
               {
                  out_val = external_bytes(pg->external_addr(uint32_t(lb)));
               }
               else
                  out_val = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                       iv->value_len);
               return true;
            }
            // Spill: the answer is the NEXT page's first entry — the cursor frame walk finds it.
            cursor c(this);
            if (!(strict ? c.upper_bound(key) : c.lower_bound(key)))
               return false;
            out_key.assign(c.key().data(), c.key().size());
            out_val = legacy_cursor_value(c);
            return true;
         }
         bytes      q    = as_bytes(key);
         KB         kb;
         value_view v;
         if (!seek(_root.address(), q, 0, strict, kb, v))
            return false;
         out_key.assign(reinterpret_cast<const char*>(kb.buf), kb.len);
         out_val = v;
         return true;
      }

      template <class Ad, class Cb>
      void inorder_in(Ad ad, KB& kb, Cb& cb)
      {
         const uint32_t save = kb.len;
         bytes          pfx  = ad.prefix();
         kb.append(pfx.data(), pfx.size());
         if (!ad.term().is_null())
         {
            // Page model: a term slot holds a PAGE (divider == path); its keys are
            // self-contained (page cpfx + suffix), so recurse — never append_terminal.
            if (ad.term().is_pointer() && page_at(ad.term().as_pointer()))
               inorder_walk(ad.term().as_pointer(), kb, cb);
            else
            {
               const uint32_t   s = kb.len;
               const value_view v = append_terminal(ad.node(), ad.term(), kb);
               cb(key_view(reinterpret_cast<const char*>(kb.buf), kb.len), v);
               kb.truncate(s);
            }
         }
         for (int c = ad.first(); ad.valid(c); c = ad.next(c))
         {
            const uint32_t s = kb.len;
            uint8_t        sb[4];
            const int      nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
            kb.append(sb, size_t(nb));
            const smart_address br = ad.branch(c);
            if (is_terminal_branch(br))
            {
               const uint32_t   s3 = kb.len;
               const value_view v  = append_terminal(ad.node(), br, kb);
               cb(key_view(reinterpret_cast<const char*>(kb.buf), kb.len), v);
               kb.truncate(s3);
            }
            else
               inorder_walk(br.as_pointer(), kb, cb);
            kb.truncate(s);
         }
         kb.truncate(save);
      }
      template <class Cb>
      void inorder_walk(ptr_address addr, KB& kb, Cb& cb)
      {
         if (addr == sal::null_ptr_address)  // null = empty tree
            return;
         if (const leaf_page* pg = page_at(addr))  // page model: self-contained sorted entries
         {
            emit_page(pg, cb);
            return;
         }
         if (const bucket_node* bn = bucket_at(addr))  // bucket leaf: emit entries in key order
         {
            const uint32_t save = kb.len;
            for (int i = 0; i < int(bn->nent); ++i)
            {
               const value_view v = bucket_entry(bn, i, kb);
               cb(key_view(reinterpret_cast<const char*>(kb.buf), kb.len), v);
               kb.truncate(save);
            }
            return;
         }
         dispatch(addr, [&](auto ad) { inorder_in(ad, kb, cb); });
      }

     public:
      // A cursor's hold on its root. Normally it owns a reference (a pin). A BORROWED
      // root owns none: the caller guarantees the tree outlives the cursor (it holds a
      // shared_tree), so opening and closing the cursor never touches the root's shared
      // reference count -- which every reader of the same snapshot would otherwise
      // contend on. Copying a borrowed root yields an owning one.
      class cursor_root
      {
         sal::smart_ptr<sal::alloc_header> _ptr;
         bool                              _borrowed = false;
         void drop() noexcept
         {
            if (_borrowed)
               (void)_ptr.take();  // forget without releasing: nothing was retained
            _borrowed = false;
         }

        public:
         cursor_root() = default;
         cursor_root(sal::smart_ptr<sal::alloc_header> pinned, bool borrowed = false) noexcept
             : _ptr(std::move(pinned)), _borrowed(borrowed)
         {
         }
         cursor_root(const sal::allocator_session_ptr& ses, ptr_address root, bool retain) noexcept
             : _ptr(ses, root, retain)
         {
         }
         cursor_root(const cursor_root& other)
             : _ptr(other._ptr.session(), other._ptr.address(), /*retain=*/true)
         {
         }
         cursor_root(cursor_root&& other) noexcept
             : _ptr(std::move(other._ptr)), _borrowed(other._borrowed)
         {
            other._borrowed = false;
         }
         cursor_root& operator=(const cursor_root& other)
         {
            if (this != &other)
               *this = cursor_root(other);
            return *this;
         }
         cursor_root& operator=(cursor_root&& other) noexcept
         {
            if (this != &other)
            {
               drop();
               _ptr            = std::move(other._ptr);
               _borrowed       = other._borrowed;
               other._borrowed = false;
            }
            return *this;
         }
         ~cursor_root() { drop(); }
         const sal::allocator_session_ptr& session() const noexcept { return _ptr.session(); }
         ptr_address                       address() const noexcept { return _ptr.address(); }
      };

      // The native ordered cursor. Its owned root keeps the COW snapshot alive;
      // session read locks protect residency during navigation and value access.
      // Positioning builds a path which next()/prev() advance without re-seeking.
      // key() borrows _kbuf until the next navigation. Read values through the
      // owning-copy or scoped-callback API; do not retain a borrowed value view.
      class cursor
      {
       private:
         friend class radix_tree;
         friend class read_handle;
         friend class tree;
         // Cursor rooted at an arbitrary node in the same allocator (a subtree root). Pins it.
         // Private: callers reach it through a handle (get_root / subtree_cursor / snapshot_cursor),
         // never by passing a raw ptr_address — ptr_address stays internal (psitri-shaped API).
         cursor(radix_tree* t, ptr_address root) noexcept
             : _t(t), _root(t->session(), root, /*retain=*/true)
         {
         }
         // Pre-pinned ctor: adopt a root ALREADY retained via an arbitrary session — the reader-
         // session path (the retain happens under the publish mutex; see snapshot_cursor_at).
         cursor(radix_tree* t, sal::smart_ptr<sal::alloc_header> pinned) noexcept
             : _t(t), _root(std::move(pinned))
         {
         }
         // Borrowed ctor: read `root` without pinning it (see cursor_root).
         struct borrow_root
         {
         };
         cursor(radix_tree* t, const sal::allocator_session_ptr& ses, ptr_address root,
                borrow_root) noexcept
             : _t(t),
               _root(sal::smart_ptr<sal::alloc_header>(ses, root, /*retain=*/false), /*borrowed=*/true)
         {
         }
         // Every read_lock AND every node access (get_ref) this cursor makes goes through the
         // session that PINNED its root. SAL sessions are single-thread-owned (and every get_ref
         // copies the session ptr, whose refcount is NON-atomic): a reader-session cursor must
         // lock, descend, and release through its OWN session, never the writer's. For cursors
         // made on the writer thread _root.session() == the engine session — identical behavior.
         sal::read_lock          rlock() const noexcept { return _root.session()->lock(); }
         sal::allocator_session& ses() const noexcept { return *_root.session(); }

        public:
         explicit cursor(radix_tree* t) noexcept
             : _t(t), _root(t->session(), t->root_addr(), /*retain=*/true)
         {
         }

         // Positioning methods follow psitri's convention: the bool return is `true` when the cursor
         // is left ON a valid entry, `false` when it lands at end / the key is absent. (NOT is_end.)
         bool seek_begin()  // position at the first (smallest) key; true if there is one
         {
            if (_root.address() == sal::null_ptr_address)  // empty/uncommitted root
            {
               reset();
               return false;
            }
            auto lk = rlock();
            reset();
            descend_min(_root.address());
            if (!_valid && _depth > 0)  // the min path ended in an empty (leftmost) page
               advance_steps();
            return _valid;
         }
         bool lower_bound(key_view k) { position(as_bytes(k), false); return _valid; }  // first key >= k
         bool upper_bound(key_view k)  // mathematical floor: largest key v <= k (dual of lower_bound)
         {
            if (lower_bound(k))
            {
               if (key() == k) return true;
               return prev();
            }
            return seek_last();
         }
         bool find(key_view k)                                                          // exact lookup
         {
            position(as_bytes(k), false);
            if (_valid && key() == k)
               return true;  // exact hit
            reset();          // miss → leave the cursor at end
            return false;
         }
         bool next()  // advance to the next key in order; true if it landed on one
         {
            if (!_valid)
               return (_end < 0) ? seek_begin() : false;  // from rend → first key; from end → stay
            auto lk = rlock();
            if (_stack_deferred) materialize_stack_();  // [lazy] build the divider path now
            return advance_steps();
         }
         bool prev()  // retreat to the previous key in order; true if it landed on one
         {
            if (!_valid)
               return (_end >= 0) ? seek_last() : false;  // from end → last key; from rend → stay
            auto lk = rlock();
            if (_stack_deferred) materialize_stack_();  // [lazy] build the divider path now
            return retreat_steps();
         }
         bool seek_last()  // position at the last (largest) key; true if there is one
         {
            if (_root.address() == sal::null_ptr_address)  // empty/uncommitted root
            {
               reset();
               _end = -1;
               return false;
            }
            auto lk = rlock();
            reset();
            descend_max(_root.address());
            if (!_valid && _depth > 0)  // the max path ended in an empty (leftmost) page
               retreat_steps();
            if (!_valid) _end = -1;
            return _valid;
         }
         bool last(key_view prefix = {})  // last key; prefix-bounded last is seek_last for now
         {
            (void)prefix;
            return seek_last();
         }
         bool seek_end()  noexcept { reset(); return false; }            // after last key (invalid)
         bool seek_rend() noexcept { reset(); _end = -1; return true; }  // before first key (matches psitri)
         bool       is_end() const noexcept { return !_valid && _end >= 0; }
         bool       is_rend() const noexcept { return !_valid && _end < 0; }
         key_view   key() const noexcept { return key_view(reinterpret_cast<const char*>(_kbuf), _klen); }
         // ── first-class subtree values ── (handles only; ptr_address stays internal)
         bool        is_subtree() const noexcept { return _valid && _is_subtree; }  // value is a subtree
         read_handle subtree() const;                                    // the nested tree (handle)
         /// The nested tree as a thread-safe owning reference (alive: this cursor pins its parent).
         /// Bound to the store, like share(): it may outlive the session this cursor reads
         /// through, whose own context it must therefore not name.
         shared_tree subtree_shared() const { return shared_tree(_t->store(), _subtree); }
         /// The tree this cursor reads, as a thread-safe owning reference.
         shared_tree share() const { return shared_tree(_t->store(), _root.address()); }
         cursor      subtree_cursor() const  // iterate it (pinned via THIS cursor's session)
         {
            return cursor(_t, sal::smart_ptr<sal::alloc_header>(_root.session(), _subtree,
                                                                /*retain=*/true));
         }
         // Borrowed value VIEW, kept valid by a read_lock bundled into the return (the caller extends
         // the safe-read window by holding the value_ref). Empty if not positioned/subtree. Symmetric
         // with psitri::cursor::pinned_value() — the safe replacement for value(read_lock&).
         // WARNING: hold the returned value_ref only briefly — its read_lock stalls the compactor;
         // holding it during heavy writes causes temporary memory bloat. Copy via value<T>() if you
         // need the bytes longer.
         value_ref pinned_value(uint64_t offset = 0, uint64_t length = adb::read_all) const
         {
            std::optional<sal::read_lock> pin(rlock());
            if (!_valid || _is_subtree) return value_ref{*pin, value_view{"", 0}};
            require_bytes();
            return value_detail::read(ses(), current_value(), offset, length, pin);
         }
         template <class T = std::string>
         std::optional<T> value(uint64_t offset = 0, uint64_t length = adb::read_all) const
         {
            if (!_valid || _is_subtree) return std::nullopt;
            require_bytes();
            std::optional<sal::read_lock> pin(rlock());
            const auto stored = current_value();
            if (stored.directory == sal::null_ptr_address)
            {
               const auto part = adb::detail::value_slice(stored.bytes, offset, length);
               return T(part.data(), part.data() + part.size());
            }
            auto value = value_detail::read(ses(), stored, offset, length, pin);
            return std::move(value).template copy_or_take<T>();
         }
         template <class F>
            requires adb::detail::value_callback<F>
         bool get_value(F&& fn, uint64_t offset = 0, uint64_t length = adb::read_all) const
         {
            if (!_valid || _is_subtree) return false;
            require_bytes();
            std::optional<sal::read_lock> pin(rlock());
            const auto stored = current_value();
            if (stored.directory == sal::null_ptr_address)
            {
               fn(adb::detail::value_slice(stored.bytes, offset, length));
               return true;
            }
            auto value = value_detail::read(ses(), stored, offset, length, pin);
            fn(value.view());
            return true;
         }
         int64_t get(key_view k, std::string* buf)
         {
            if (!find(k)) return value_not_found;
            if (_is_subtree) return value_subtree;
            *buf = *value<std::string>();
            return int64_t(buf->size());
         }
         bool seek(key_view k) { return lower_bound(k) && key() == k; }
         bool first(key_view prefix = {}) { return prefix.empty() ? seek_begin() : lower_bound(prefix); }
         int64_t value_size() const noexcept
         {
            return !_valid ? value_not_found : (_is_subtree ? value_subtree : int64_t(_value_size));
         }
         // Point reads keep the owning root throughout; assembly releases the
         // directory lock and independently pins each intersecting chunk.
         template <class F>
            requires adb::detail::value_callback<F>
         bool get(key_view k, F&& fn, uint64_t offset = 0, uint64_t length = adb::read_all)
         {
            if (_root.address() == sal::null_ptr_address) return false;
            std::optional<sal::read_lock> pin(rlock());
            value_detail::stored_value stored;
            const bool ok = _t->_page_mode ? page_find_value(ses(), _root.address(), as_bytes(k), stored)
                                           : find_in(ses(), _root.address(), as_bytes(k), stored.bytes);
            if (!ok) return false;
            value_detail::require_bytes(stored);
            if (stored.directory == sal::null_ptr_address)
            {
               fn(adb::detail::value_slice(stored.bytes, offset, length));
               return true;
            }
            auto value = value_detail::read(ses(), stored, offset, length, pin);
            fn(value.view());
            return true;
         }
         // ── reads of a value of ANY kind ── fn(uint16_t typenum, value_view bytes): typenum 0
         // with the plain bytes, or a typed value's type number with its token. A subtree
         // value reads as not-found, as in get(). The view is valid only inside fn.
         template <class F>
            requires std::invocable<F, uint16_t, value_view>
         bool get_any(key_view k, F&& fn)
         {
            if (_root.address() == sal::null_ptr_address) return false;
            std::optional<sal::read_lock> pin(rlock());
            value_detail::stored_value stored;
            const bool ok = _t->_page_mode ? page_find_value(ses(), _root.address(), as_bytes(k), stored)
                                           : find_in(ses(), _root.address(), as_bytes(k), stored.bytes);
            if (!ok) return false;
            if (stored.directory == sal::null_ptr_address)
            {
               fn(stored.typenum, stored.bytes);
               return true;
            }
            auto value = value_detail::read(ses(), stored, 0, adb::read_all, pin);
            fn(uint16_t(kRawType), value.view());
            return true;
         }
         /// The current value, of any kind (see get_any).
         template <class F>
            requires std::invocable<F, uint16_t, value_view>
         bool value_any(F&& fn) const
         {
            if (!_valid || _is_subtree) return false;
            std::optional<sal::read_lock> pin(rlock());
            const auto stored = current_value();
            if (stored.directory == sal::null_ptr_address)
            {
               fn(stored.typenum, stored.bytes);
               return true;
            }
            auto value = value_detail::read(ses(), stored, 0, adb::read_all, pin);
            fn(uint16_t(kRawType), value.view());
            return true;
         }
         /// Type number of the current value: 0 for plain bytes (and when not on a value).
         uint16_t value_typenum() const noexcept { return _valid ? _typenum : kRawType; }
         template <class T = std::string, std::integral Offset = uint64_t>
         std::optional<T> get(key_view k, Offset offset = 0, uint64_t length = adb::read_all)
         {
            if (_root.address() == sal::null_ptr_address) return std::nullopt;
            std::optional<sal::read_lock> pin(rlock());
            value_detail::stored_value stored;
            const bool ok = _t->_page_mode ? page_find_value(ses(), _root.address(), as_bytes(k), stored)
                                           : find_in(ses(), _root.address(), as_bytes(k), stored.bytes);
            if (!ok) return std::nullopt;
            value_detail::require_bytes(stored);
            if (stored.directory == sal::null_ptr_address)
            {
               const auto part = adb::detail::value_slice(stored.bytes, uint64_t(offset), length);
               return T(part.data(), part.data() + part.size());
            }
            auto value = value_detail::read(ses(), stored, uint64_t(offset), length, pin);
            return std::move(value).template copy_or_take<T>();
         }
         // Legacy naked-view overload: caller's lock protects contiguous bytes;
         // crossing values use cursor scratch, valid until its next such read.
         value_view value(sal::read_lock& rl) const
         {
            (void)rl;
            if (!_valid || _is_subtree) return {};
            require_bytes();
            const auto stored = current_value();
            if (stored.directory == sal::null_ptr_address) return stored.bytes;
            _compat_buffer = *value<std::string>();
            return _compat_buffer;
         }
         // Count keys in [lo, hi) (empty bound = open). Walks a private copy so `this` is untouched.
         uint64_t count_keys(key_view lo = {}, key_view hi = {}) const
         {
            cursor   c(*this);  // copy re-pins the same snapshot root (RAII)
            uint64_t n = 0;
            for (bool ok = lo.empty() ? c.seek_begin() : c.lower_bound(lo); ok; ok = c.next())
            {
               if (!hi.empty() && !(c.key() < hi)) break;
               ++n;
            }
            return n;
         }
         static constexpr int32_t value_not_found = -1;  // get(key, buf) miss (psitri parity)
         static constexpr int32_t value_subtree   = -2;  // value is a subtree, not bytes

        private:
         // A frame's node address, stored raw. ptr_address zero-initializes itself, and a
         // cursor is constructed for every point read: an array of frames holding one would
         // be cleared, all of it, each time (1,026 frames is 8 KB). A frame is written
         // before it is read (next_frame, then ++_depth), so it needs no initial value.
         struct frame_addr
         {
            uint32_t raw;
            frame_addr() = default;
            frame_addr(ptr_address a) noexcept : raw(*a) {}
            operator ptr_address() const noexcept { return ptr_address(raw); }
         };
         struct frame  // 8 bytes (lean push/pop); psitri-style
         {
            frame_addr addr;
            int16_t    pos;    // -1 = at this node's term value; else entry cursor (0..256)
            uint16_t   klen1;  // _klen after this node's prefix (entry stems append from here)
         };
         static_assert(sizeof(frame) == 8, "lean 8-byte cursor frame");
         static_assert(std::is_trivially_default_constructible_v<frame>,
                       "constructing a cursor must not touch its path array");

         void reset() noexcept
         {
            _value_page     = sal::null_ptr_address;
            _value_size     = 0;
            _depth          = 0;
            _klen           = 0;
            _valid          = false;
            _end            = 0;
            _is_subtree     = false;
            _typenum        = kRawType;
            _stack_deferred = false;
         }
         void append(const uint8_t* p, size_t n)
         {
            if (_klen + n > sizeof(_kbuf)) [[unlikely]]
               throw std::logic_error("sal_art_ref cursor: a path is longer than the key limit");
            std::memcpy(_kbuf + _klen, p, n);
            _klen += uint32_t(n);
         }
         template <class Ad>
         void take_terminal(Ad ad, smart_address br)  // append suffix, capture value or subtree
         {
            const inline_value* iv = _t->term_iv(ses(), ad.node(), br);
            append(iv->data, iv->suffix_len);
            _value_page = sal::null_ptr_address;
            _value_size = iv->value_len;
            _typenum    = kRawType;  // typed values exist only in the page model
            if (iv->is_subtree())
            {
               _is_subtree = true;
               _subtree    = iv->subtree_root();
               _val        = {};
            }
            else
            {
               _is_subtree = false;
               _val = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                 iv->value_len);
            }
         }

         // ── leaf pages (the page model) ──────────────────────────────────────────────────
         // A leaf_page is a dense sorted terminal whose keys are SELF-CONTAINED (page cpfx +
         // suffix) — the radix path above is DIVIDER bytes, never key bytes, so a page frame
         // rebuilds the key from scratch per entry (klen1 still restores parents on pop).
         const leaf_page* as_page(ptr_address a) const { return radix_tree::page_at(ses(), a); }
         void take_page_entry(const leaf_page* pg, int i)
         {
            _klen = 0;
            _value_page = pg->address();
            _value_index = uint32_t(i);
            const auto* iv = pg->record(uint32_t(i));
            if (iv->value_type == inline_value::vt_extern)
            {
               const auto key = kvres(ses())(pg->external_addr(uint32_t(i)));
               append(key.data(), key.size());
            }
            else
            {
               const auto cp = pg->common_prefix();
               append(cp.data(), cp.size());
               const auto suffix = pg->suffix(uint32_t(i));
               append(suffix.data(), suffix.size());
            }
            if (inline_value::external_value(iv->value_type))
            {
               const auto& record = *ses().get_ref<key_val_node>(pg->external_addr(uint32_t(i))).obj();
               _is_subtree = record.is_subtree();
               _subtree = _is_subtree ? record.subtree_root() : sal::null_ptr_address;
               _typenum = record.is_typed() ? typed_typenum({reinterpret_cast<const char*>(
                                                                 record.value_bytes().data()),
                                                             record.value_bytes().size()})
                                            : kRawType;
               _value_size = record.value_size() - (_typenum != kRawType ? 2u : 0u);
               _val = {};  // resolve through the stable page/child IDs on every read
            }
            else
            {
               _is_subtree = iv->is_subtree();
               _subtree = _is_subtree ? iv->subtree_root() : sal::null_ptr_address;
               const value_view stored(reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                       iv->value_len);
               _typenum    = iv->is_typed() ? typed_typenum(stored) : kRawType;
               _value_size = iv->value_len - (_typenum != kRawType ? 2u : 0u);
               _val = _is_subtree ? value_view{}
                                  : (_typenum != kRawType ? typed_token(stored) : stored);
            }
         }
         void require_bytes() const
         {
            if (_typenum != kRawType) [[unlikely]]
               throw invalid_type{_typenum, kRawType};
         }
         value_detail::stored_value current_value() const
         {
            if (_value_page == sal::null_ptr_address) return {_val, sal::null_ptr_address, _typenum};
            value_detail::stored_value result;
            page_value(ses(), as_page(_value_page), _value_index, result);
            return result;
         }

         // ── bucket leaves ────────────────────────────────────────────────────────────────
         // A bucket is a dense, lexicographically-sorted (suffix, value) page — NOT a radix
         // router, so dispatch() has no adaptor for it. The cursor walks it as a flat entry
         // array (entry 0 = smallest), resolving the page ONCE per frame and advancing pos in
         // place. Entry i's full key = the path prefix [0, klen1) + the entry's stored suffix.
         const bucket_node* as_bucket(ptr_address a) const { return bucket_at(ses(), a); }
         void take_bucket_entry(const bucket_node* bn, int i, uint16_t klen1)
         {
            const inline_value* iv = bn->entry(i);
            _klen                  = klen1;
            append(iv->data, iv->suffix_len);
            _value_page = sal::null_ptr_address;
            _value_size = iv->value_len;
            _typenum    = kRawType;  // typed values exist only in the page model
            _is_subtree = iv->is_subtree();
            _subtree    = _is_subtree ? iv->subtree_root() : ptr_address{};
            _val        = _is_subtree ? value_view{}
                                      : value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len),
                                                   iv->value_len);
         }

         // The in-order advance loop shared by next() and the empty-page recovery in
         // seek_begin/position: walk the frame stack until a key lands or the top falls off.
         // An empty page subtree (descend_min lands invalid with the page frame popped) just
         // keeps advancing the parent — only the page model produces those.
         bool advance_steps()
         {
            while (_depth > 0)
            {
               const int r = advance_top();
               if (r == 1)
               {
                  _valid = true;
                  return true;  // landed on a terminal/entry
               }
               if (r == 2)
               {
                  descend_min(_nextaddr);
                  if (_valid)
                     return true;
                  if (_depth == 0)
                     break;  // defensive: nothing left to advance
               }
               // r == 0: this frame is exhausted, popped — keep advancing the parent
            }
            _valid = false;
            _end   = 0;  // fell off the top → at end
            return false;
         }
         bool retreat_steps()  // reverse of advance_steps (prev() and reverse recovery)
         {
            while (_depth > 0)
            {
               const int r = retreat_top();
               if (r == 1)
               {
                  _valid = true;
                  return true;
               }
               if (r == 2)
               {
                  descend_max(_nextaddr);
                  if (_valid)
                     return true;
                  if (_depth == 0)
                     break;
               }
            }
            _valid = false;
            _end   = -1;  // fell off the bottom → before begin (rend)
            return false;
         }

         // Push frames down to the minimum terminal of `addr`'s subtree; sets _valid/_val/_kbuf.
         void descend_min(ptr_address addr)
         {
            for (;;)
            {
               if (const leaf_page* pg = as_page(addr))  // leftmost path ends in a page
               {
                  frame& f = next_frame();
                  f.addr   = addr;
                  f.klen1  = uint16_t(_klen);
                  f.pos    = 0;
                  ++_depth;
                  if (pg->num_keys() == 0)  // an empty (leftmost) page — caller skips onward
                  {
                     --_depth;
                     _valid = false;
                     return;
                  }
                  take_page_entry(pg, 0);
                  _valid = true;
                  return;
               }
               if (const bucket_node* bn = as_bucket(addr))  // leftmost path ends in a bucket
               {
                  frame& f = next_frame();
                  f.addr   = addr;
                  f.klen1  = uint16_t(_klen);
                  f.pos    = 0;
                  ++_depth;
                  if (bn->nent == 0) { --_depth; _valid = false; return; }  // defensive
                  take_bucket_entry(bn, 0, f.klen1);  // smallest entry
                  _valid = true;
                  return;
               }
               const uint32_t k0 = _klen;  // key length before this node's prefix (local; not stored)
               const int      r  = _t->dispatch(ses(), addr, [&](auto ad) -> int {
                  frame& f = next_frame();
                  f.addr   = addr;
                  bytes pfx = ad.prefix();
                  append(pfx.data(), pfx.size());
                  f.klen1 = uint16_t(_klen);
                  ++_depth;
                  if (!ad.term().is_null())  // term (key ends here) sorts before any entry
                  {
                     f.pos = -1;
                     if (ad.term().is_pointer() && as_page(ad.term().as_pointer()))
                     {
                        _nextaddr = ad.term().as_pointer();  // page model: the divider's page
                        return 2;
                     }
                     take_terminal(ad, ad.term());
                     return 1;
                  }
                  const int c = ad.first();
                  if (!ad.valid(c))  // empty node (only the root can be empty)
                  {
                     --_depth;
                     _klen = k0;
                     return 0;
                  }
                  f.pos = int16_t(c);
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
                  append(sb, size_t(nb));
                  const smart_address br = ad.branch(c);
                  if (_t->is_terminal_branch(ses(), br))
                  {
                     take_terminal(ad, br);
                     return 1;
                  }
                  _nextaddr = br.as_pointer();
                  return 2;
               });
               if (r == 1) { _valid = true; return; }
               if (r == 0) { _valid = false; return; }
               addr = _nextaddr;
            }
         }

         // Advance within the top frame: 0 popped (exhausted), 1 found terminal, 2 descend _nextaddr.
         int advance_top()
         {
            frame& f = _fr[_depth - 1];
            if (const leaf_page* pg = as_page(f.addr))  // advance within a leaf page
            {
               if (++f.pos >= int(pg->num_keys()))
               {
                  --_depth;  // exhausted → pop; the parent advance restores its klen1
                  return 0;
               }
               take_page_entry(pg, f.pos);
               return 1;
            }
            if (const bucket_node* bn = as_bucket(f.addr))  // advance within a bucket page
            {
               _klen = f.klen1;
               if (++f.pos >= int(bn->nent)) { --_depth; return 0; }  // exhausted → pop
               take_bucket_entry(bn, f.pos, f.klen1);
               return 1;
            }
            return _t->dispatch(ses(), f.addr, [&](auto ad) -> int {
               const int nc = (f.pos < 0) ? ad.first() : ad.next(f.pos);
               _klen        = f.klen1;  // drop the previous entry's stem (+ any descent)
               if (!ad.valid(nc))
               {
                  --_depth;  // exhausted → pop; the parent's advance re-truncates to its klen1
                  return 0;
               }
               uint8_t   sb[4];
               const int nb = stem_bytes(ad.cell(nc), ad.stem_width(), ad.cw(), sb);
               append(sb, size_t(nb));
               f.pos                  = int16_t(nc);
               const smart_address br = ad.branch(nc);
               if (_t->is_terminal_branch(ses(), br))
               {
                  take_terminal(ad, br);
                  return 1;
               }
               _nextaddr = br.as_pointer();
               return 2;
            });
         }

         // Reverse of descend_min: push frames to the MAXIMUM terminal of `addr`'s subtree. In a
         // node, in-order is [term, entry0..entryN]; the max is entryN's subtree max, so we take
         // the LAST entry (the term, smallest, is reached only when entries are exhausted).
         void descend_max(ptr_address addr)
         {
            for (;;)
            {
               if (const leaf_page* pg = as_page(addr))  // rightmost path ends in a page
               {
                  frame& f = next_frame();
                  f.addr   = addr;
                  f.klen1  = uint16_t(_klen);
                  ++_depth;
                  if (pg->num_keys() == 0)  // an empty (leftmost) page — caller skips onward
                  {
                     --_depth;
                     _valid = false;
                     return;
                  }
                  f.pos = int16_t(pg->num_keys() - 1);
                  take_page_entry(pg, f.pos);
                  _valid = true;
                  _end   = 0;
                  return;
               }
               if (const bucket_node* bn = as_bucket(addr))  // rightmost path ends in a bucket
               {
                  frame& f = next_frame();
                  f.addr   = addr;
                  f.klen1  = uint16_t(_klen);
                  ++_depth;
                  if (bn->nent == 0) { --_depth; _valid = false; return; }  // defensive
                  f.pos = int16_t(bn->nent - 1);
                  take_bucket_entry(bn, f.pos, f.klen1);  // largest entry
                  _valid = true;
                  _end   = 0;
                  return;
               }
               const uint32_t k0 = _klen;
               const int      r  = _t->dispatch(ses(), addr, [&](auto ad) -> int {
                  frame& f = next_frame();
                  f.addr   = addr;
                  bytes pfx = ad.prefix();
                  append(pfx.data(), pfx.size());
                  f.klen1 = uint16_t(_klen);
                  ++_depth;
                  const int c = ad.last();
                  if (c >= 0)  // has entries → descend the largest
                  {
                     f.pos = int16_t(c);
                     uint8_t   sb[4];
                     const int nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
                     append(sb, size_t(nb));
                     const smart_address br = ad.branch(c);
                     if (_t->is_terminal_branch(ses(), br)) { take_terminal(ad, br); return 1; }
                     _nextaddr = br.as_pointer();
                     return 2;
                  }
                  if (!ad.term().is_null())  // no entries → the term is the only/largest key
                  {
                     f.pos = -1;
                     if (ad.term().is_pointer() && as_page(ad.term().as_pointer()))
                     {
                        _nextaddr = ad.term().as_pointer();  // page model: the divider's page
                        return 2;
                     }
                     take_terminal(ad, ad.term());
                     return 1;
                  }
                  --_depth;  // empty node (root only)
                  _klen = k0;
                  return 0;
               });
               if (r == 1) { _valid = true; _end = 0; return; }
               if (r == 0) { _valid = false; return; }
               addr = _nextaddr;
            }
         }

         // Reverse of advance_top: 0 popped, 1 found terminal, 2 descend_max _nextaddr.
         int retreat_top()
         {
            frame& f = _fr[_depth - 1];
            if (const leaf_page* pg = as_page(f.addr))  // retreat within a leaf page
            {
               if (--f.pos < 0)
               {
                  --_depth;  // before first → pop
                  return 0;
               }
               take_page_entry(pg, f.pos);
               return 1;
            }
            if (const bucket_node* bn = as_bucket(f.addr))  // retreat within a bucket page
            {
               _klen = f.klen1;
               if (--f.pos < 0) { --_depth; return 0; }  // before first → pop
               take_bucket_entry(bn, f.pos, f.klen1);
               return 1;
            }
            return _t->dispatch(ses(), f.addr, [&](auto ad) -> int {
               _klen = f.klen1;  // drop the previous entry's stem (+ any descent)
               if (f.pos < 0)    // we were at this node's term (last in reverse) → pop
               {
                  --_depth;
                  return 0;
               }
               const int pc = ad.prev(f.pos);
               if (pc >= 0)  // previous (smaller) entry → descend its MAX
               {
                  f.pos = int16_t(pc);
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(pc), ad.stem_width(), ad.cw(), sb);
                  append(sb, size_t(nb));
                  const smart_address br = ad.branch(pc);
                  if (_t->is_terminal_branch(ses(), br)) { take_terminal(ad, br); return 1; }
                  _nextaddr = br.as_pointer();
                  return 2;
               }
               if (!ad.term().is_null())  // entries exhausted → the term (smallest) is last in reverse
               {
                  f.pos = -1;
                  if (ad.term().is_pointer() && as_page(ad.term().as_pointer()))
                  {
                     _nextaddr = ad.term().as_pointer();  // page model: the divider's page
                     return 2;
                  }
                  take_terminal(ad, ad.term());
                  return 1;
               }
               --_depth;
               return 0;
            });
         }

         // Position at the first key >= q (strict → > q). Under the read-lock, on the PINNED
         // root: a raw inner seek finds the key, then build_to lays down the descent path.
         bool position(bytes q, bool strict)
         {
            if (_root.address() == sal::null_ptr_address)  // empty/uncommitted root
            {
               reset();
               return true;  // at end (matches a seek past every key)
            }
            auto lk = rlock();
            if (_t->_page_mode)
            {
               reset();
               ptr_address root = _root.address();
               if (const leaf_page* pg = as_page(root))  // single-page tree
               {
                  frame& f = _fr[0];
                  f.addr   = root;
                  f.klen1  = 0;
                  _depth   = 1;
                  enter_page_at(pg, q);
               }
               else
               {
                  // [lazy] Route to the owning page and position WITHIN it. Page entries are
                  // self-contained (key = page cpfx+suffix, value in-page), so a seek+read needs
                  // neither the divider-path frame stack nor a re-descent — take_page_entry sets
                  // key+value directly. Build the full stack lazily only when next()/prev() must
                  // cross a page boundary (materialize_stack_). A q PAST the page end must spill,
                  // and strict (upper_bound) may step past an exact match — both need the stack,
                  // so they fall back to the eager build_to_page.
                  uint8_t          db[1056];  // divider <= max key + 1
                  uint32_t         dl = 0;
                  const leaf_page* owner = route_to_page(ses(), root, q, db, &dl);
                  const int        lb = owner->lower_bound(q.data(), uint32_t(q.size()), kvres(ses()));
                  if (!strict && lb < int(owner->num_keys()))
                  {
                     take_page_entry(owner, lb);  // self-contained: sets _kbuf (key) + _val (value)
                     _valid          = true;
                     _stack_deferred = true;
                  }
                  else
                  {
                     build_to_page(bytes{db, dl}, q);  // spill / strict: build the full stack
                  }
               }
               if (_valid && strict &&
                   key() == key_view(reinterpret_cast<const char*>(q.data()), q.size()))
                  advance_steps();  // upper_bound: skip the exact match
               return !_valid;
            }
            KB         kb;
            value_view v;
            if (!_t->seek(ses(), _root.address(), q, 0, strict, kb, v))
            {
               reset();
               return true;
            }
            build_to(bytes{kb.buf, kb.len});
            return !_valid;
         }

         // Position the TOP page frame at the first entry >= q, spilling to the next page
         // (via the frame walk) when q is past every entry. The frame is already pushed.
         void enter_page_at(const leaf_page* pg, bytes q)
         {
            frame&    f  = _fr[_depth - 1];
            const int lb = pg->lower_bound(q.data(), uint32_t(q.size()), kvres(ses()));
            if (lb < int(pg->num_keys()))
            {
               f.pos = int16_t(lb);
               take_page_entry(pg, lb);
               _valid = true;
               return;
            }
            f.pos = int16_t(int(pg->num_keys()) - 1);  // ++ in advance_top pops (also empty: -1)
            advance_steps();
         }

         // Exact-build the frame path along divider D (the route's owning slot), ending in a
         // page frame positioned by q. Mirrors build_to, but every slot is a page pointer and
         // the divider is fully path-consumed (term or branch — never a suffix terminal).
         void build_to_page(bytes D, bytes q)
         {
            reset();
            ptr_address addr  = _root.address();
            uint32_t    depth = 0;
            for (;;)
            {
               if (const leaf_page* pg = as_page(addr))
               {
                  frame& f = next_frame();
                  f.addr   = addr;
                  f.klen1  = uint16_t(_klen);
                  ++_depth;
                  enter_page_at(pg, q);
                  return;
               }
               _t->dispatch(ses(), addr, [&](auto ad) {
                  frame& f = next_frame();
                  f.addr   = addr;
                  bytes pfx = ad.prefix();
                  append(pfx.data(), pfx.size());
                  f.klen1 = uint16_t(_klen);
                  ++_depth;
                  const uint32_t d2 = depth + uint32_t(pfx.size());
                  if (d2 == D.size())  // the divider is this node's term → its page
                  {
                     f.pos      = -1;
                     _nextaddr  = ad.term().as_pointer();
                     _nextdepth = d2;
                     return;
                  }
                  bytes     r2 = D.subspan(d2);
                  bool      full;
                  uint32_t  fc;
                  const int c = ad.lower(r2, full, fc);
                  f.pos       = int16_t(c);
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
                  append(sb, size_t(nb));
                  _nextaddr  = ad.branch(c).as_pointer();
                  _nextdepth = d2 + uint32_t(ad.cw());
               });
               addr  = _nextaddr;
               depth = _nextdepth;
            }
         }

         // [lazy] A deferred seek positioned the cursor inside its owning page WITHOUT building
         // the divider-path frame stack (key+value are page-self-contained). Build it now —
         // re-route from the current key and exact-build the path — so advance/retreat_steps can
         // cross page boundaries. Only the FIRST next()/prev() after a seek pays this; a pure
         // seek+read never does. Caller already holds the read lock.
         void materialize_stack_()
         {
            uint8_t        q[1024];
            const uint32_t ql = _klen;          // current key (set by the deferred take_page_entry)
            std::memcpy(q, _kbuf, ql);
            uint8_t  db[1056];
            uint32_t dl = 0;
            route_to_page(ses(), _root.address(), bytes{q, ql}, db, &dl);
            build_to_page(bytes{db, dl}, bytes{q, ql});  // rebuilds frames + repositions same entry
            _stack_deferred = false;
         }

         // Position at an EXACT existing key, building the descent path along the way.
         void build_to(bytes key)
         {
            reset();
            ptr_address addr  = _root.address();  // the pinned snapshot root
            uint32_t    depth = 0;
            for (;;)
            {
               if (const bucket_node* bn = as_bucket(addr))  // path ends in a bucket leaf
               {
                  frame& f = next_frame();
                  f.addr   = addr;
                  f.klen1  = uint16_t(_klen);
                  ++_depth;
                  const int avail = int(key.size()) > int(depth) ? int(key.size()) - int(depth) : 0;
                  const int i     = bn->lower_bound(key.data() + depth, avail);  // exact key exists
                  if (i >= int(bn->nent)) { --_depth; _valid = false; return; }  // defensive
                  f.pos = int16_t(i);
                  take_bucket_entry(bn, i, f.klen1);
                  _valid = true;
                  return;
               }
               const int r = _t->dispatch(ses(), addr, [&](auto ad) -> int {
                  frame& f = next_frame();
                  f.addr   = addr;
                  bytes pfx = ad.prefix();
                  append(pfx.data(), pfx.size());
                  f.klen1 = uint16_t(_klen);
                  ++_depth;
                  const uint32_t d2 = depth + uint32_t(pfx.size());
                  if (d2 == key.size())  // the key ends here → this node's term
                  {
                     f.pos = -1;
                     take_terminal(ad, ad.term());
                     return 1;
                  }
                  bytes     r2 = key.subspan(d2);
                  bool      full;
                  uint32_t  fc;
                  const int c = ad.lower(r2, full, fc);  // exact cell position (key exists)
                  f.pos       = int16_t(c);
                  uint8_t   sb[4];
                  const int nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sb);
                  append(sb, size_t(nb));
                  const smart_address br = ad.branch(c);
                  if (_t->is_terminal_branch(ses(), br))
                  {
                     take_terminal(ad, br);
                     return 1;
                  }
                  _nextaddr  = br.as_pointer();
                  _nextdepth = d2 + uint32_t(ad.cw());
                  return 2;
               });
               if (r == 1) { _valid = true; return; }
               addr  = _nextaddr;
               depth = _nextdepth;
            }
         }

         radix_tree*                     _t;
         cursor_root _root;  // pins the snapshot version (RAII refcount) unless borrowed
         // The descent path. Every router node on a path consumes at least one byte of a
         // divider, and a divider is at most a maximum-length key, so a path has at most
         // max_key_len router frames, one more for a node entered through its terminal, and
         // one for the page. next_frame() enforces the bound instead of trusting it.
         static constexpr int max_frames = int(key_val_node::max_key_len) + 2;
         frame                           _fr[max_frames];
         frame& next_frame()
         {
            if (_depth >= max_frames) [[unlikely]]
               throw std::logic_error("sal_art_ref cursor: the tree is deeper than the key limit allows");
            return _fr[_depth];
         }
         uint8_t                         _kbuf[1024];
         uint32_t    _klen      = 0;
         value_view  _val;
         ptr_address _value_page = sal::null_ptr_address;
         uint32_t _value_index = 0;
         uint64_t _value_size = 0;
         mutable std::string _compat_buffer;
         int         _depth     = 0;
         bool        _valid     = false;
         int8_t      _end       = 0;  // when !_valid: +1 = at/after end, -1 = at/before begin (rend)
         bool        _is_subtree = false;  // current terminal's value is a subtree (not bytes)
         uint16_t    _typenum    = kRawType;  // nonzero: the current value is a typed value
         ptr_address _subtree{};           // that subtree's root (valid when _is_subtree)
         ptr_address _nextaddr{};
         uint32_t    _nextdepth = 0;
         // [lazy] a seek positioned inside the owning page but did NOT build the divider-path
         // frame stack (key+value are page-self-contained). next()/prev() build it on demand.
         bool        _stack_deferred = false;
      };

     public:
      value_view legacy_cursor_value(const cursor& c)
      {
         auto pin = _ses->lock();
         return direct_value_view(*_ses, c.current_value());
      }

      // ── multi-root + transactions (psitri-shaped) ──────────────────────────────────────
      // Published roots live in SAL's root table — the SINGLE source of truth, exactly psitri's
      // model. SAL synchronizes it: set_root swaps under the per-root writer lock and writes the
      // commit sync-header (sealing filled segments for compaction/recycling); get_root retains
      // under the per-root SHARED lock, so readers scale and never wait behind a writer's tree
      // work. There is NO engine-side root index and no engine lock. A transaction acquires a
      // retained base from the table; the first write clones (COW-to-root), so a published root
      // is never mutated under a reader; commit hands the working root to set_root, abort drops
      // it. Savepoints (sub_transaction) retain the working root and restore it on abort. The
      // engine member `_root` is unrelated: it is the raw DIRECT-API working pointer (single-
      // threaded benches/tests, bucket mode) and never feeds the database layer.

      // A read-only TREE HANDLE over a root (or subtree root) — the psitri `tree`-equivalent.
      // Holds a ptr_address internally but never exposes it; reads go through cursors it makes.
      class read_handle
      {
       private:
         friend class radix_tree;
         friend class cursor;
         friend class transaction;
         friend class frame;
         radix_tree* _t    = nullptr;
         ptr_address _root = sal::null_ptr_address;
         read_handle(radix_tree* t, ptr_address r) noexcept : _t(t), _root(r) {}

        public:
         read_handle()                          = default;
         explicit operator bool() const noexcept { return _root != sal::null_ptr_address; }
         cursor   snapshot_cursor() const { return cursor(_t, _root); }
         uint64_t count_keys() const
         {
            if (_root == sal::null_ptr_address) return 0;
            cursor c(_t, _root);
            return c.count_keys();
         }
         bool is_subtree(key_view k) const
         {
            if (_root == sal::null_ptr_address) return false;
            cursor c(_t, _root);
            return c.find(k) && c.is_subtree();  // find() true = positioned on the key
         }
         read_handle get_subtree(key_view k) const
         {
            if (_root == sal::null_ptr_address) return {_t, sal::null_ptr_address};
            cursor c(_t, _root);
            if (!c.find(k) || !c.is_subtree()) return {_t, sal::null_ptr_address};
            return c.subtree();  // a handle, not an address
         }
         template <class T = std::string, std::integral Offset = uint64_t>
         std::optional<T> get(key_view k, Offset offset = 0, uint64_t length = adb::read_all) const
         {
            if (_root == sal::null_ptr_address) return std::nullopt;
            cursor c(_t, _root);
            return c.template get<T>(k, offset, length);
         }
         // Callback form (kv_engine concept): invoke f(value_view) iff present; true if found.
         template <class F>
            requires adb::detail::value_callback<F>
         bool get(key_view k, F&& f, uint64_t offset = 0, uint64_t length = adb::read_all) const
         {
            if (_root == sal::null_ptr_address) return false;
            cursor c(_t, _root);
            return c.get(k, std::forward<F>(f), offset, length);
         }
         /// A value of any kind: f(uint16_t typenum, value_view) (see cursor::get_any).
         template <class F>
            requires std::invocable<F, uint16_t, value_view>
         bool get_any(key_view k, F&& f) const
         {
            if (_root == sal::null_ptr_address) return false;
            cursor c(_t, _root);
            return c.get_any(k, std::forward<F>(f));
         }
      };
      // psitri-compatible OWNING tree handle: what get_root/get_subtree/get_tree return and what
      // upsert_subtree accepts. Owns a ref on its root (RAII smart_ptr); copyable (retains) like
      // psitri::tree. Exposes session()/address() (psitri does too) + the read surface.
      class tree
      {
       private:
         friend class radix_tree;
         friend class write_session;
         friend class write_transaction;
         friend class transaction;
         friend class tree_handle;
         friend class frame;
         radix_tree*                       _t = nullptr;
         sal::smart_ptr<sal::alloc_header> _root;  // owns a ref; empty = null/empty tree
         tree(radix_tree* t, ptr_address r, bool retain) : _t(t), _root(t->session(), r, retain) {}
         // Adopt an already-acquired root (SAL get_root's retained smart_ptr — the database path).
         tree(radix_tree* t, sal::smart_ptr<sal::alloc_header> r) : _t(t), _root(std::move(r)) {}

        public:
         tree()                       = default;
         tree(const tree&)            = default;
         tree(tree&&)                 = default;
         tree& operator=(const tree&) = default;
         tree& operator=(tree&&)      = default;
         explicit operator bool() const noexcept { return bool(_root); }
         const sal::allocator_session_ptr& session() const noexcept { return _root.session(); }
         ptr_address                       address() const noexcept { return _root.address(); }
         cursor snapshot_cursor() const { return cursor(_t, _root.address()); }
         uint64_t count_keys() const { return _root ? snapshot_cursor().count_keys() : 0; }
         bool     is_subtree(key_view k) const
         {
            if (!_root) return false;
            cursor c(_t, _root.address());
            return c.find(k) && c.is_subtree();  // find() true = positioned on the key
         }
         tree get_subtree(key_view k) const
         {
            if (!_root) return {};
            cursor c(_t, _root.address());
            if (!c.find(k) || !c.is_subtree()) return {};
            return tree(_t, c._subtree, /*retain=*/true);
         }
         template <class T = std::string, std::integral Offset = uint64_t>
         std::optional<T> get(key_view k, Offset offset = 0, uint64_t length = adb::read_all) const
         {
            if (!_root) return std::nullopt;
            cursor c(_t, _root.address());
            return c.template get<T>(k, offset, length);
         }
         template <class F>
            requires adb::detail::value_callback<F>
         bool get(key_view k, F&& fn, uint64_t offset = 0, uint64_t length = adb::read_all) const
         {
            if (!_root) return false;
            cursor c(_t, _root.address());
            return c.get(k, std::forward<F>(fn), offset, length);
         }
         /// A value of any kind: fn(uint16_t typenum, value_view) (see cursor::get_any).
         template <class F>
            requires std::invocable<F, uint16_t, value_view>
         bool get_any(key_view k, F&& fn) const
         {
            if (!_root) return false;
            cursor c(_t, _root.address());
            return c.get_any(k, std::forward<F>(fn));
         }
      };
      // The last COMMITTED root, acquired from SAL's table (retain under the per-root shared lock).
      tree get_root(uint32_t idx)
      {
         return tree(this, _ses->get_root<sal::alloc_header>(sal::root_object_number(idx)));
      }

      // A MUTABLE tree editor (psitri tree_handle): accumulates COW changes to a (sub)tree's working
      // root and lazily opens child subtrees, which are written back BOTTOM-UP when the owning
      // transaction commits. Reached via transaction::primary()/open_root() and
      // open_subtree()/create_subtree(); never constructed with a raw ptr_address by users.
      class tree_handle
      {
       private:
         friend class radix_tree;
         friend class transaction;
         friend class frame;
         using root_ptr = sal::smart_ptr<sal::alloc_header>;
         radix_tree* _t = nullptr;
         root_ptr    _working;  // this (sub)tree's COW working root (empty = null)
         struct open_child
         {
            std::string                  key;
            std::unique_ptr<tree_handle> h;
         };
         std::vector<open_child> _children;  // opened subtrees (txn-level bookkeeping, not a hot path)

         tree_handle(radix_tree* t, ptr_address base, bool retain)
             : _t(t), _working(t->session(), base, retain)
         {
         }
         // Adopt an already-acquired root (SAL get_root's retained smart_ptr — the database path).
         tree_handle(radix_tree* t, root_ptr base) : _t(t), _working(std::move(base)) {}
         // All mutations transition _working via mutate_root (the tree_context discipline):
         // the pre-image is owned across the descent and released structurally; null = the
         // valid empty tree (the descent lazily creates on first write).
         // Write every opened child back into _working (deepest first), then yield the final root.
         root_ptr flush()
         {
            for (auto& c : _children)
            {
               root_ptr    child = c.h->flush();  // recurse bottom-up
               ptr_address sub   = child.take();  // consume the child's ref → the terminal owns it
               _t->mutate_root(_working, [&](ptr_address pre)
                               { return _t->attach_subtree_step(pre, c.key, sub); });
               SAL_CRASH_POINT("subtree:written_back");
            }
            _children.clear();
            return std::move(_working);
         }

         // What a savepoint keeps of a handle: its working root and, for every subtree open
         // at that moment, the same again. Restoring puts each of them back and drops the
         // subtrees opened since. (Writes through a subtree handle go to that handle's own
         // working root, not to its parent's, until commit writes them back.)
         struct savepoint
         {
            root_ptr               working;
            size_t                 nchild = 0;
            std::vector<savepoint> children;
         };
         savepoint save() const
         {
            savepoint s{_working, _children.size(), {}};
            SAL_CRASH_POINT("savepoint:working_retained");
            s.children.reserve(_children.size());
            for (const auto& c : _children)
               s.children.push_back(c.h->save());
            return s;
         }
         void restore(savepoint&& s) noexcept
         {
            _working = std::move(s.working);
            SAL_CRASH_POINT("savepoint:working_restored");
            if (_children.size() > s.nchild)
               _children.resize(s.nchild);  // drop children opened since
            // (set_root() since the savepoint closed the handles open then: nothing of
            // theirs is left to restore.)
            for (size_t i = 0; i < _children.size() && i < s.children.size(); ++i)
               _children[i].h->restore(std::move(s.children[i]));
         }

        public:
         tree_handle()                              = default;
         tree_handle(tree_handle&&) noexcept        = default;
         tree_handle(const tree_handle&)            = delete;
         tree_handle& operator=(const tree_handle&) = delete;

         void upsert(key_view k, value_view v)
         {
            _t->_ins_value_type = inline_value::vt_bytes;
            _t->mutate_root(_working, [&](ptr_address pre)
                            { return _t->upsert_step(pre, as_bytes(k), as_bytes(v)); });
         }
         /// Store a typed value (see radix_tree::upsert_typed).
         void upsert_typed(key_view k, uint16_t typenum, value_view token)
         {
            _t->mutate_root(_working, [&](ptr_address pre)
                            { return _t->typed_upsert_step(pre, k, typenum, token); });
         }
         // THE exchange (adb multi_root): assign `t` as this handle's working root,
         // RETURN the displaced root as an owned tree. tree{} assigns empty; the
         // rotation composes — h1.set_root(h0.set_root(tree{})). The swap is
         // immediate (reads reflect it before commit); commit publishes whatever
         // each handle holds. Opened children belong to the OLD root → cleared.
         tree set_root(tree t)
         {
            tree old(_t, std::move(_working));  // wrap the current working (ref transfers)
            _working = std::move(t._root);      // adopt the incoming root (ref transfers)
            SAL_CRASH_POINT("tree:working_replaced");
            _children.clear();
            return old;
         }
         int remove(key_view k)
         {
            if (!_working) return -1;
            bool removed = false;
            _t->mutate_root(_working, [&](ptr_address pre)
                            { return _t->remove_step(pre, as_bytes(k), removed); });
            return removed ? 0 : -1;
         }
         uint64_t remove_range_counted(key_view lower, key_view upper)  // delete [lower,upper), count
         {
            return _t->remove_range_counted_step(_working, lower, upper);
         }
         void upsert_subtree(key_view k, subtree&& s)  // attach an already-built detached subtree
         {
            attach_subtree(k, s.consume());
         }
         void upsert_subtree(key_view k, tree t)  // psitri-compatible: attach an owning tree (consumes)
         {
            attach_subtree(k, t._root.take());
         }
       private:
         void attach_subtree(key_view k, ptr_address sub)  // store a vt_subtree terminal owning `sub`
         {
            _t->mutate_root(_working, [&](ptr_address pre)
                            { return _t->attach_subtree_step(pre, k, sub); });
         }
       public:
         // Open an existing subtree for in-place editing (deduped; written back on commit).
         tree_handle& open_subtree(key_view k)
         {
            for (auto& c : _children)
               if (c.key == k) return *c.h;
            read_handle rh{_t, _working.address()};
            read_handle sh   = rh.get_subtree(k);   // current child root (handle; empty if absent)
            ptr_address base = sh._root;             // friend access; null if not a subtree
            _children.push_back({std::string(k),
                                 std::unique_ptr<tree_handle>(new tree_handle(_t, base, /*retain=*/true))});
            return *_children.back().h;
         }
         tree_handle& create_subtree(key_view k)  // create a new empty subtree (deduped)
         {
            for (auto& c : _children)
               if (c.key == k) return *c.h;
            _children.push_back(
                {std::string(k),
                 std::unique_ptr<tree_handle>(new tree_handle(_t, sal::null_ptr_address, false))});
            return *_children.back().h;
         }
         // Reads reflect this handle's working root (an opened child's edits are seen via the child
         // handle, psitri-style — read its own working, not the parent's not-yet-flushed copy).
         read_handle handle() const { return {_t, _working.address()}; }
         cursor      snapshot_cursor() const { return handle().snapshot_cursor(); }
         uint64_t    count_keys() const { return handle().count_keys(); }
         bool        is_subtree(key_view k) const { return handle().is_subtree(k); }
         read_handle get_subtree(key_view k) const { return handle().get_subtree(k); }
         template <class T = std::string, std::integral Offset = uint64_t>
         std::optional<T> get(key_view k, Offset offset = 0, uint64_t length = adb::read_all) const { return handle().template get<T>(k, offset, length); }
         template <class F>
            requires adb::detail::value_callback<F>
         bool get(key_view k, F&& f, uint64_t offset = 0, uint64_t length = adb::read_all) const { return handle().get(k, std::forward<F>(f), offset, length); }
         template <class F>
            requires std::invocable<F, uint16_t, value_view>
         bool get_any(key_view k, F&& f) const { return handle().get_any(k, std::forward<F>(f)); }
      };

      // A transaction over one or more roots. Owns a primary tree_handle (the root at
      // `idx`) plus any additionally opened roots; on commit each handle is flushed bottom-up
      // (opened subtrees written back into their parents) and the roots published. Abort drops all
      // working lineages (RAII). Root slots publish separately, not atomically as a group.
      // The caller serializes writes because transactions share the engine's write session.
      class transaction
      {
       private:
         friend class radix_tree;
         friend class frame;
         using root_ptr = sal::smart_ptr<sal::alloc_header>;
         radix_tree* _t           = nullptr;
         uint32_t    _primary_idx = 0;
         bool        _open        = false;
         // Held from here until commit or abort, so a second writer on the root waits and
         // starts from what this one published. Declared before _primary: the root is read
         // only once the lock is held.
         // Roots this transaction claimed on its writer context (see radix_tree::claim_root),
         // declared before the locks so a refused claim takes no lock.
         struct claims
         {
            radix_tree*           t = nullptr;
            bool                  has_primary = false;
            uint32_t              primary     = 0;  // kept inline: no allocation per transaction
            std::vector<uint32_t> roots;            // roots opened beyond the primary
            claims(radix_tree* tree, uint32_t idx) : t(tree)
            {
               t->claim_root(idx);
               primary     = idx;
               has_primary = true;
            }
            claims(claims&& o) noexcept
                : t(o.t), has_primary(o.has_primary), primary(o.primary), roots(std::move(o.roots))
            {
               o.has_primary = false;
               o.roots.clear();
            }
            claims(const claims&)            = delete;
            claims& operator=(const claims&) = delete;
            ~claims() { drop(); }
            void add(uint32_t idx)
            {
               roots.reserve(roots.size() + 1);  // so recording a granted claim cannot throw
               t->claim_root(idx);
               roots.push_back(idx);
            }
            void undo_last() noexcept  // the claim just added got no lock
            {
               t->unclaim_root(roots.back());
               roots.pop_back();
            }
            void drop() noexcept
            {
               if (has_primary)
                  t->unclaim_root(primary);
               has_primary = false;
               for (uint32_t idx : roots)
                  t->unclaim_root(idx);
               roots.clear();
            }
         } _claims;
         std::unique_lock<writer_gate> _writer;
         tree_handle _primary;                                              // the root at _primary_idx
         std::vector<std::pair<uint32_t, std::unique_ptr<tree_handle>>> _extra;  // open_root(idx)
         std::vector<std::unique_lock<writer_gate>> _extra_writers;
         transaction(radix_tree* t, uint32_t idx)
             : _t(t),
               _primary_idx(idx),
               _open(true),
               _claims(t, idx),
               _writer(t->root_writer(idx)),
               _primary(t, t->_ses->get_root<sal::alloc_header>(sal::root_object_number(idx)))
         {
         }
         void unlock_roots() noexcept
         {
            _extra_writers.clear();
            if (_writer.owns_lock())
               _writer.unlock();
            _claims.drop();
         }
         // COMMIT a flushed working root: hand it to SAL's root table — set_root TAKES the
         // working ref, writes the commit sync-header (sealing filled segments so the compactor
         // can recycle them), and swaps under SAL's per-root writer lock; readers acquire under
         // the shared lock and never wait behind the writer's tree work. SAL's table IS the
         // published-root storage — no engine-side index, no engine lock (psitri's pattern).
         // The table's previous root releases on scope exit.
         void publish(uint32_t idx, root_ptr w)
         {
            auto old = _t->_ses->set_root<sal::alloc_header>(sal::root_object_number(idx),
                                                             std::move(w), _t->sync_mode());
            SAL_CRASH_POINT("commit:old_root_held");
         }

        public:
         // The moved-from transaction is closed: it holds no lock and no working root.
         transaction(transaction&& o) noexcept
             : _t(o._t),
               _primary_idx(o._primary_idx),
               _open(o._open),
               _claims(std::move(o._claims)),
               _writer(std::move(o._writer)),
               _primary(std::move(o._primary)),
               _extra(std::move(o._extra)),
               _extra_writers(std::move(o._extra_writers))
         {
            o._open = false;
         }
         transaction(const transaction&)                = delete;
         transaction& operator=(const transaction&)     = delete;
         ~transaction() { if (_open) abort(); }


         // ── primary-root mutations / reads (forward to the primary handle) ──
         void upsert(key_view k, value_view v) { _primary.upsert(k, v); }
         /// Store a typed value in the primary root (see radix_tree::upsert_typed).
         void upsert_typed(key_view k, uint16_t typenum, value_view token) { _primary.upsert_typed(k, typenum, token); }
         int  remove(key_view k) { return _primary.remove(k); }
         uint64_t remove_range_counted(key_view lower, key_view upper)
         {
            return _primary.remove_range_counted(lower, upper);
         }
         void upsert_subtree(key_view k, subtree&& s) { _primary.upsert_subtree(k, std::move(s)); }
         void upsert_subtree(key_view k, tree t) { _primary.upsert_subtree(k, std::move(t)); }  // psitri
         tree_handle& primary() { return _primary; }
         tree_handle& open_subtree(key_view k) { return _primary.open_subtree(k); }
         tree_handle& create_subtree(key_view k) { return _primary.create_subtree(k); }
         cursor       snapshot_cursor() { return _primary.snapshot_cursor(); }
         bool         is_subtree(key_view k) { return _primary.is_subtree(k); }
         tree get_subtree(key_view k)  // psitri: owning tree (over the working root's child)
         {
            return tree(_t, _primary._working.address(), /*retain=*/true).get_subtree(k);
         }
         template <class T = std::string, std::integral Offset = uint64_t>
         std::optional<T> get(key_view k, Offset offset = 0, uint64_t length = adb::read_all) { return _primary.template get<T>(k, offset, length); }
         template <class F>
            requires adb::detail::value_callback<F>
         bool get(key_view k, F&& f, uint64_t offset = 0, uint64_t length = adb::read_all) const { return _primary.get(k, std::forward<F>(f), offset, length); }
         template <class F>
            requires std::invocable<F, uint16_t, value_view>
         bool get_any(key_view k, F&& f) const { return _primary.get_any(k, std::forward<F>(f)); }

         // ── additional roots (multi-root transaction) ──
         tree_handle& open_root(uint32_t idx)
         {
            if (idx == _primary_idx) return _primary;
            for (auto& [i, h] : _extra)
               if (i == idx) return *h;
            // Roots opened in ascending order wait for their writer. Out of order, waiting
            // could deadlock against a transaction that holds them the other way round, so
            // the root is taken only if it is free.
            uint32_t highest = _primary_idx;
            for (auto& [i, h] : _extra)
               highest = std::max(highest, i);
            _claims.add(idx);  // refused if another transaction of this session holds it
            try
            {
               _extra_writers.reserve(_extra_writers.size() + 1);
               if (idx > highest)
                  _extra_writers.emplace_back(_t->root_writer(idx));
               else
               {
                  std::unique_lock<writer_gate> writer(_t->root_writer(idx), std::try_to_lock);
                  if (!writer.owns_lock())
                     throw std::runtime_error(
                         "root is being written; open roots in ascending order to wait for it");
                  _extra_writers.push_back(std::move(writer));
               }
            }
            catch (...)
            {
               _claims.undo_last();
               throw;
            }
            _extra.emplace_back(idx, std::unique_ptr<tree_handle>(new tree_handle(
                                         _t, _t->_ses->get_root<sal::alloc_header>(
                                                 sal::root_object_number(idx)))));
            return *_extra.back().second;
         }

         void commit()
         {
            if (!_open) return;
            // Flush every root bottom-up first: a flush can throw (a subtree that cannot be
            // written back, an allocation). Only when every root's final tree exists is any
            // of them published, so a commit that throws has published nothing; it leaves
            // the transaction aborted, because the flushed handles are spent.
            root_ptr                                   primary;
            std::vector<std::pair<uint32_t, root_ptr>> extras;
            try
            {
               extras.reserve(_extra.size());
               primary = _primary.flush();
               for (auto& [i, h] : _extra)
                  extras.emplace_back(i, h->flush());
            }
            catch (...)
            {
               abort();
               throw;
            }
            SAL_CRASH_POINT("commit:flushed");
            publish(_primary_idx, std::move(primary));
            for (auto& [i, w] : extras)
            {
               SAL_CRASH_POINT("commit:between_roots");
               publish(i, std::move(w));
            }
            SAL_CRASH_POINT("commit:published");
            _extra.clear();
            _open = false;
            unlock_roots();
            SAL_CRASH_POINT("commit:unlocked");
         }
         void abort() noexcept
         {
            if (!_open) return;
            SAL_CRASH_POINT("abort:begin");
            _primary._working.release();  // drop the working lineage + opened children (RAII)
            SAL_CRASH_POINT("abort:working_released");
            _primary._children.clear();
            _extra.clear();
            _open = false;
            unlock_roots();
            SAL_CRASH_POINT("abort:done");
         }
         frame sub_transaction();  // savepoint (defined after frame)
      };
      transaction start_transaction(uint32_t idx = 0) { return transaction(this, idx); }

      // A savepoint over a transaction's PRIMARY handle: snapshots its working root and those of
      // every subtree open under it; abort restores them and drops the subtrees opened during
      // the frame (LIFO). Other roots of the transaction (open_root) are not covered.
      class frame
      {
       private:
         friend class transaction;
         using root_ptr = sal::smart_ptr<sal::alloc_header>;
         transaction*           _tx = nullptr;
         tree_handle::savepoint _sp;  // the primary handle and its open subtrees, retained
         bool                   _active = false;
         frame(transaction* tx, tree_handle::savepoint sp)
             : _tx(tx), _sp(std::move(sp)), _active(true)
         {
         }

        public:
         // The moved-from frame is no longer the savepoint: left active, its destructor
         // would "abort" and put its now-empty saved root in place of the working one.
         frame(frame&& o) noexcept : _tx(o._tx), _sp(std::move(o._sp)), _active(o._active)
         {
            o._active = false;
         }
         frame(const frame&)            = delete;
         frame& operator=(const frame&) = delete;
         ~frame() { if (_active) abort(); }
         void upsert(key_view k, value_view v) { _tx->upsert(k, v); }
         void upsert_subtree(key_view k, subtree&& s) { _tx->upsert_subtree(k, std::move(s)); }
         int  remove(key_view k) { return _tx->remove(k); }
         uint64_t remove_range_counted(key_view lower, key_view upper)
         {
            return _tx->remove_range_counted(lower, upper);
         }
         tree_handle& open_subtree(key_view k) { return _tx->open_subtree(k); }
         tree_handle& create_subtree(key_view k) { return _tx->create_subtree(k); }
         bool         is_subtree(key_view k) { return _tx->is_subtree(k); }
         tree         get_subtree(key_view k) { return _tx->get_subtree(k); }
         template <class T = std::string, std::integral Offset = uint64_t>
         std::optional<T> get(key_view k, Offset offset = 0, uint64_t length = adb::read_all) { return _tx->template get<T>(k, offset, length); }
         template <class F>
            requires adb::detail::value_callback<F>
         bool get(key_view k, F&& fn, uint64_t offset = 0, uint64_t length = adb::read_all) const { return _tx->get(k, std::forward<F>(fn), offset, length); }
         void commit() noexcept { _sp = {}; _active = false; }  // keep working + children
         void abort() noexcept
         {
            _tx->_primary.restore(std::move(_sp));
            _active = false;
         }
      };

      // psitri-compatible editor for a DETACHED tree: start_write_transaction(create_temporary_tree()
      // or an existing tree) → upsert/remove/... → get_tree(). Owns a working tree; mutations COW it
      // in place; reads reflect it; get_tree() hands it off (e.g. to upsert_subtree).
      class write_transaction
      {
       private:
         radix_tree* _t = nullptr;
         tree        _wt;

        public:
         write_transaction(radix_tree* t, tree base) : _t(t), _wt(std::move(base))
         {
            _wt._t = t;  // bind: tolerate an engine-less tree{} (e.g. a get_subtree miss)
         }
         void upsert(key_view k, value_view v)
         {
            _t->_ins_value_type = inline_value::vt_bytes;
            _t->mutate_root(_wt._root, [&](ptr_address pre)
                            { return _t->upsert_step(pre, as_bytes(k), as_bytes(v)); });
         }
         /// Store a typed value (see radix_tree::upsert_typed).
         void upsert_typed(key_view k, uint16_t typenum, value_view token)
         {
            _t->mutate_root(_wt._root, [&](ptr_address pre)
                            { return _t->typed_upsert_step(pre, k, typenum, token); });
         }
         void upsert_subtree(key_view k, tree sub)  // psitri parity: attach an owning tree (consumes)
         {
            ptr_address donated = sub._root.take();  // the terminal owns the attached root's ref
            _t->mutate_root(_wt._root, [&](ptr_address pre)
                            { return _t->attach_subtree_step(pre, k, donated); });
         }
         int remove(key_view k)
         {
            if (!_wt._root) return -1;
            bool removed = false;
            _t->mutate_root(_wt._root, [&](ptr_address pre)
                            { return _t->remove_step(pre, as_bytes(k), removed); });
            return removed ? 0 : -1;
         }
         uint64_t remove_range_counted(key_view lower, key_view upper)  // delete [lower,upper), count
         {
            return _t->remove_range_counted_step(_wt._root, lower, upper);
         }
         // ── reads reflect the working tree (psitri write_transaction read surface) ──
         cursor snapshot_cursor() const { return _wt.snapshot_cursor(); }
         bool   is_subtree(key_view k) const { return _wt.is_subtree(k); }
         tree   get_subtree(key_view k) const { return _wt.get_subtree(k); }
         template <class T = std::string, std::integral Offset = uint64_t>
         std::optional<T> get(key_view k, Offset offset = 0, uint64_t length = adb::read_all) const
         {
            return _wt.template get<T>(k, offset, length);
         }
         template <class F>
            requires adb::detail::value_callback<F>
         bool get(key_view k, F&& fn, uint64_t offset = 0, uint64_t length = adb::read_all) const { return _wt.get(k, std::forward<F>(fn), offset, length); }
         template <class F>
            requires std::invocable<F, uint16_t, value_view>
         bool get_any(key_view k, F&& f) const { return _wt.get_any(k, std::forward<F>(f)); }
         tree get_tree() { return std::move(_wt); }  // hand off the built (owning) tree
      };

      // ── psitri-compatible engine entry points (database/write_session delegate here) ──
      // NOTE: the temporary tree must be ENGINE-BOUND (not tree{}): reads on the working tree
      // (write_transaction get/get_subtree/snapshot_cursor) construct cursors through tree::_t.
      tree create_temporary_tree() { return tree(this, sal::null_ptr_address, /*retain=*/false); }
      write_transaction start_write_transaction(tree base)
      {
         return write_transaction(this, std::move(base));
      }
      // COMMIT a tree into SAL's root table at idx (empty tree = clear). set_root takes the
      // tree's ref, writes the durable commit record (sealing filled segments — without commit
      // boundaries nothing ever compacts or recycles and the file grows by the full COW-garbage
      // rate), and swaps under SAL's per-root writer lock. The table's previous root releases on
      // scope exit. DIRECT mutations (upsert/remove on _root, bucket mode) are uncommitted by
      // definition and never touch the table.
      void set_root(uint32_t idx, tree t, sal::sync_type st = sal::sync_type::none)
      {
         claim_root(idx);  // refused while an open transaction of this session holds the root
         struct unclaim
         {
            radix_tree* t;
            uint32_t    idx;
            ~unclaim() { t->unclaim_root(idx); }
         } done{this, idx};
         std::lock_guard<writer_gate> writer(root_writer(idx));
         SAL_CRASH_POINT("set_root:locked");
         auto old =
             _ses->set_root<sal::alloc_header>(sal::root_object_number(idx), std::move(t._root), st);
         SAL_CRASH_POINT("set_root:old_root_held");
      }
      uint64_t get_total_allocated_objects() const { return object_count(); }
      uint64_t get_pending_release_count() const { return pending_releases(); }
      // Reader-session snapshot: acquire the LAST COMMITTED root from SAL's root table —
      // get_root retains under the per-root SHARED lock (readers scale; a committing writer
      // holds the exclusive lock only for the table exchange, never for its tree work). The
      // root is pinned via the CALLER's session, so the cursor's read locks and its release-
      // on-destruction run on that session (one SAL session per thread). Readers see commit
      // boundaries only — never the writer-private working state.
      cursor snapshot_cursor_at(uint32_t idx, const sal::allocator_session_ptr& ses)
      {
         return cursor(this,
                       ses->get_root<sal::alloc_header>(sal::root_object_number(idx)));
      }
      // A cursor on an arbitrary shared tree, pinned (and later released) through the CALLER's
      // session. The shared reference keeps the tree alive while the pin is taken.
      cursor shared_cursor(const shared_tree& shared, const sal::allocator_session_ptr& ses)
      {
         return cursor(this, sal::smart_ptr<sal::alloc_header>(ses, shared._root, /*retain=*/true));
      }
      // The same without pinning: the caller keeps `shared` alive for the cursor's lifetime.
      cursor borrowed_cursor(const shared_tree& shared, const sal::allocator_session_ptr& ses)
      {
         return cursor(this, ses, shared._root, cursor::borrow_root{});
      }
      // Writer side: a session-bound tree from a shared one, and back (consuming the tree's ref).
      tree        adopt(const shared_tree& shared) { return tree(this, shared._root, /*retain=*/true); }
      shared_tree share(tree t)
      {
         return shared_tree(_store, t._root.take(), shared_tree::adopt_reference{});
      }
      // One SAL session per thread: call on the thread that will use it (psitri read_session model).
      sal::allocator_session_ptr make_session() { return _alloc->create_session(); }
      sal::allocator&            underlying_allocator() noexcept { return *_alloc; }
      sal::seg_alloc_dump        dump() const { return _alloc->dump(); }
      void                       sync(sal::sync_type st) { _alloc->sync(st); }

     private:

      // ── descent helper for r1 nodes (prefixed routers) ─────────────────────────
      // STATIC + session-parametric (read-path contract): no engine state — `found` replaces the
      // old _route_found member side-channel, so reader-session cursors can run this descent
      // concurrently with the writer.
      template <class N>
      static bool route_r1(sal::allocator_session& ses, const N* n, bytes key, uint32_t& depth,
                           value_view& out, ptr_address& addr, bool& found)
      {
         bytes pfx = n->prefix();
         if (pfx.size() > key.size() - depth ||
             (pfx.size() && std::memcmp(pfx.data(), key.data() + depth, pfx.size()) != 0))
         {
            found = false;
            return false;
         }
         depth += uint32_t(pfx.size());
         if (depth == key.size())
         {
            found = term_lookup(ses, n, n->term_value(), key, depth, out);
            return false;
         }
         // Route by the node's consume-width key bytes. For setlist_u8 (stem_width==1)
         // the if-constexpr keeps the verbatim single-byte path → no change for u8.
         uint32_t      W;
         smart_address sa;
         if constexpr (N::stem_width == 1)
         {
            W  = 1;
            sa = n->find(key[depth]);
         }
         else
         {
            W = uint32_t(n->consume_width());  // 2 (u16) or 3/4 (u32 cell)
            if (key.size() - depth < W)        // key ends inside this node's stride -> miss
            {
               found = false;
               return false;
            }
            sa = n->find(read_stem<N>(key, depth, W));
         }
         if (sa.is_null())
         {
            found = false;
            return false;
         }
         if (sa.is_tail())
         {
            found = inline_at(n, sa, key.subspan(depth + W), out);
            return false;
         }
         addr = sa.as_pointer();
         depth += W;
         return true;
      }

      // Read `cw` key bytes at depth big-endian and left-justify into N's cell (stem_width
      // bytes) — so a u32-cell node consuming 3 bytes zero-pads the low byte, and an
      // integer compare on the cell == lexicographic compare on the consumed bytes.
      template <class N>
      static uint32_t read_stem(bytes key, uint32_t depth, uint32_t cw) noexcept
      {
         uint32_t v = 0;
         for (uint32_t i = 0; i < cw; ++i)
            v = (v << 8) | key[depth + i];
         return v << (8 * (uint32_t(N::stem_width) - cw));
      }

      // ── insert (tier dispatch is the ONLY place type matters) ───────────────────
      template <upsert_mode mode>
      ptr_address insert_node(ptr_address addr, bytes key, uint32_t depth, bytes val,
                              sal::alloc_hint peer_hint)
      {
         switch (kind_of(_ses->get_ref<sal::alloc_header>(addr)->type()))
         {
            case radix_kind::bucket:  // bucket LEAF: insert (split on overflow)
               return bucket_leaf_insert(addr, key, depth, val);
            case radix_kind::setlist_u8:
               return router_insert<mode, setlist_u8>(addr, key, depth, val, peer_hint);
            case radix_kind::setlist_u16:
               return router_insert_wide<mode, setlist_u16>(addr, key, depth, val, peer_hint);
            case radix_kind::setlist_u32:
               return router_insert_wide<mode, setlist_u32>(addr, key, depth, val, peer_hint);
            case radix_kind::c1_32:
               return router_insert<mode, bitset_c1_32>(addr, key, depth, val, peer_hint);
            case radix_kind::c1_64:
               return router_insert<mode, bitset_c1_64>(addr, key, depth, val, peer_hint);
            case radix_kind::c1_128:
               return router_insert<mode, bitset_c1_128>(addr, key, depth, val, peer_hint);
            case radix_kind::bitset_r2:
               return router_insert<mode, bitset_r2_node>(addr, key, depth, val, peer_hint);
            case radix_kind::bitset_r4:
               return router_insert<mode, bitset_r4_node>(addr, key, depth, val, peer_hint);
            case radix_kind::full:
               return router_insert<mode, full_radix_node>(addr, key, depth, val, peer_hint);
            case radix_kind::leaf:     // a terminal value — never inserted into directly
            case radix_kind::page:     // page-model writes go through page_upsert_step
            case radix_kind::bitset1:  // not produced by the engine
               break;
         }
         unknown_node_kind();
      }

      // ── remove descent (mirrors insert_node dispatch; routers only — a terminal value
      // is handled in-place by the parent via is_terminal_branch, never recursed into) ──
      template <upsert_mode mode>
      ptr_address remove_node(ptr_address addr, bytes key, uint32_t depth)
      {
         switch (kind_of(_ses->get_ref<sal::alloc_header>(addr)->type()))
         {
            case radix_kind::bucket:  // adaptive bucket LEAF: in-place remove
            {
               const uint8_t* s  = key.data() + depth;
               const int      sn = int(key.size() - depth);
               if (!_ses->get_ref<bucket_node>(addr).obj()->find(s, sn))
               {
                  _removed = false;
                  return addr;
               }
               bkt(addr)->remove(s, sn);  // buckets mutate in place (no-MVCC path)
               _removed = true;
               _emptied = (_ses->get_ref<bucket_node>(addr).obj()->nent == 0);
               return addr;
            }
            case radix_kind::setlist_u8:
               return remove_router<mode, setlist_u8>(addr, key, depth);
            case radix_kind::setlist_u16:
               return remove_router<mode, setlist_u16>(addr, key, depth);
            case radix_kind::setlist_u32:
               return remove_router<mode, setlist_u32>(addr, key, depth);
            case radix_kind::c1_32:
               return remove_router<mode, bitset_c1_32>(addr, key, depth);
            case radix_kind::c1_64:
               return remove_router<mode, bitset_c1_64>(addr, key, depth);
            case radix_kind::c1_128:
               return remove_router<mode, bitset_c1_128>(addr, key, depth);
            case radix_kind::bitset_r2:
               return remove_router<mode, bitset_r2_node>(addr, key, depth);
            case radix_kind::bitset_r4:
               return remove_router<mode, bitset_r4_node>(addr, key, depth);
            case radix_kind::full:
               return remove_router<mode, full_radix_node>(addr, key, depth);
            case radix_kind::leaf:     // a terminal value — never recursed into
            case radix_kind::page:     // page-model removes go through page_remove_step
            case radix_kind::bitset1:  // not produced by the engine
               break;
         }
         unknown_node_kind();
      }

      // Remove `key` from the subtree rooted at `addr`. Sets _removed (was it found) and,
      // when the node drains to nothing, _emptied (parent should drop+free it). Returns the
      // (possibly COW'd) node address. Releases external terminal values; for an emptied or
      // path-COW'd child it follows insert's contract — the PARENT releases the original.
      // After a reducing removal, downgrade `addr` to a smaller tier if it now fits one with
      // hysteresis; if the rebuild replaces a COW'd clone (shared mode, addr != pre), free the
      // clone here (in unique mode addr == pre and the PARENT frees the original). Skipped when
      // the node emptied (it's being dropped). Returns the node to hand back to the parent.
      template <class N>
      ptr_address finalize_remove(ptr_address addr, ptr_address pre)
      {
         if (_emptied)
            return addr;
         ptr_address dg = maybe_downgrade<N>(addr);
         if (dg != addr)
         {
            if (addr != pre)
               _ses->release(addr);
            addr = dg;
         }
         return addr;
      }

      template <upsert_mode mode, class N>
      ptr_address remove_router(ptr_address addr, bytes key, uint32_t depth)
      {
         const ptr_address pre = addr;  // original (pre-COW) node, for clone-release on downgrade
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);  // resolved once; threaded into cow()
         // SHAREDNESS IS THE INSTANTIATION (psitri upsert_mode): a unique-mode router whose own
         // node is snapshot-reachable re-enters as the SHARED instantiation — from here on `mode`
         // is the whole truth (no runtime latch). range_shared at the recursion site selects the
         // CHILD's instantiation, so a unique working node with co-owned children still mutates
         // IN-PLACE; only the shared child subtree clones (what lets a batch converge to an
         // in-place tree as nodes become unique).
         if constexpr (!mode.is_shared())
         {
            if (ref.ref() > 1)
               return remove_router<upsert_mode{upsert_mode::shared}, N>(addr, key, depth);
         }
         const N* cn = ref.obj();
         bytes             pfx = cn->prefix();
         if (pfx.size() > key.size() - depth ||
             (pfx.size() && std::memcmp(pfx.data(), key.data() + depth, pfx.size()) != 0))
         {
            _removed = false;  // key diverges from this node's compressed prefix → absent
            return addr;
         }
         depth += uint32_t(pfx.size());
         if (depth == key.size())  // key ends here → remove the term value
         {
            smart_address tv = cn->term_value();
            if (tv.is_null())
            {
               _removed = false;
               return addr;
            }
            N* n = cow<mode, N>(ref, addr);
            if (tv.is_pointer())
               _ses->release(tv.as_pointer());  // external leaf/subtree value
            n->set_term_value(smart_address::null());
            _removed = true;
            _emptied = (n->branch_count() == 0);
            return finalize_remove<N>(addr, pre);
         }
         // route by this node's stride (1 byte for every tier remove supports; wide setlist
         // consumes its consume_width key bytes per hop, same as route_r1).
         uint32_t            W;
         typename N::stem_type cell;
         if constexpr (N::stem_width == 1)
         {
            W    = 1;
            cell = typename N::stem_type(key[depth]);
         }
         else
         {
            W = uint32_t(cn->consume_width());
            if (key.size() - depth < W)
            {
               _removed = false;
               return addr;
            }
            cell = typename N::stem_type(read_stem<N>(key, depth, W));
         }
         smart_address sa = cn->find(cell);
         if (sa.is_null())
         {
            _removed = false;
            return addr;
         }
         if (is_terminal_branch(sa))  // a leaf/inline terminal here → this is (maybe) the key
         {
            bytes      esuf;
            value_view eval;
            term_view(cn, sa, esuf, eval);
            bytes suf = key.subspan(depth + W);
            if (esuf.size() != suf.size() ||
                (esuf.size() && std::memcmp(esuf.data(), suf.data(), esuf.size()) != 0))
            {
               _removed = false;  // same routing cell, different suffix → key absent
               return addr;
            }
            N* n = cow<mode, N>(ref, addr);
            rr_diverge<N>(n, range_of(cell));  // big tier: privatize range before dropping a branch
            if (sa.is_pointer())
               _ses->release(sa.as_pointer());  // external leaf value (inline tail just orphans)
            n->remove_branch(cell);
            _removed = true;
            _emptied = (n->branch_count() == 0 && !n->has_term_value());
            return finalize_remove<N>(addr, pre);
         }
         // pointer to an inner node → recurse. The child is shared (must clone) if THIS node is
         // shared OR its big-tier children are co-owned via shared counters (the child's MODE).
         ptr_address child = sa.as_pointer();
         ptr_address nc;
         if constexpr (mode.is_shared())
            nc = remove_node<mode>(child, key, depth + W);
         else  // PER-RANGE: only a co-owned range forces the child subtree into shared mode
            nc = range_shared<N>(cn, range_of(cell))
                     ? remove_node<upsert_mode{upsert_mode::shared}>(child, key, depth + W)
                     : remove_node<mode>(child, key, depth + W);
         if (!_removed)
            return addr;  // absent in subtree → nothing changed here
         if (_emptied)     // the child drained → drop its branch and free it
         {
            N* n = cow<mode, N>(ref, addr);  // ref valid: the recursion didn't COW this node
            rr_diverge<N>(n, range_of(cell));  // big tier: privatize range before dropping a branch
            n->remove_branch(cell);
            _ses->release(nc);  // the emptied child (in-place == child, or a clone)
            if (nc != child)
               _ses->release(child);  // shared: also drop the original child's ref
            _emptied = (n->branch_count() == 0 && !n->has_term_value());
            return finalize_remove<N>(addr, pre);
         }
         if (nc != child)  // child was path-COW'd (shared) → repoint, release the original
         {
            N* n = cow<mode, N>(ref, addr);  // ref valid: the recursion didn't COW this node
            rr_diverge<N>(n, range_of(cell));  // big tier: privatize range before re-pointing
            n->set_branch(cell, smart_address::from_pointer(nc));
            _ses->release(child);
         }
         _emptied = false;
         return addr;
      }

      template <upsert_mode mode, class N>
      ptr_address router_insert(ptr_address addr, bytes key, uint32_t depth, bytes val,
                                sal::alloc_hint peer_hint)
      {
         const ptr_address addr_orig = addr;
         const uint32_t    depth0     = depth;  // where addr's region begins (for bucket collapse)
         // Resolve the node ONCE and thread it by reference: cn covers every pre-COW read, and
         // cow(ref, …) reuses it for an in-place mutation (no second get_ref). After a COW the
         // ref is stale (clone → new address), so the post-COW writable `n` is used instead.
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);
         // SHAREDNESS IS THE INSTANTIATION (psitri upsert_mode): a unique-mode router whose own
         // node is snapshot-reachable re-enters as the SHARED instantiation — `mode` is the whole
         // truth from here (no runtime latch). range_shared at the recursion site selects the
         // CHILD's instantiation, so a unique working node with co-owned children still mutates
         // IN-PLACE; only the shared child subtree clones (what lets a batch converge to an
         // in-place tree as nodes become unique).
         if constexpr (!mode.is_shared())
         {
            if (ref.ref() > 1)
               return router_insert<upsert_mode{upsert_mode::shared}, N>(addr, key, depth, val,
                                                                         peer_hint);
         }
         const N* cn = ref.obj();
         bytes             pfx = cn->prefix();
         int               avail     = int(key.size() - depth);
         int               maxp      = std::min(int(pfx.size()), avail);
         int p = common_prefix_len(key.subspan(depth, size_t(maxp)), pfx.subspan(0, size_t(maxp)));
         if (p < int(pfx.size()))
            return prefix_split<mode, N>(addr, key, depth, p, val, peer_hint);

         depth += uint32_t(pfx.size());
         if (depth == key.size())  // key ends here → term
         {
            set_term<mode, N>(addr, val, peer_hint);
            return addr;
         }
         uint8_t         b     = key[depth];
         sal::alloc_hint chint = cn->branch_hint(b);
         smart_address   sa    = cn->find(b);
         bytes           suf   = key.subspan(depth + 1);

         if (sa.is_null())  // new terminal at b
         {
            // set_branch is the source of truth (a tail's compact base depends on its exact
            // offset, so a probe can't predict it). COW ONCE (room for the inline value, ref
            // threaded), write the value into that node, try to set; on failure widen (the value
            // is re-made on the wider node). The COW may clone addr → release that clone here
            // (the caller releases the original), else it orphans.
            const ptr_address pre = addr;
            N*                 n   = cow<mode, N>(ref, addr, value_tail_need(suf, val));
            rr_diverge<N>(n, range_of(b));  // big tier: privatize range before adding a branch
            smart_address      sab = make_value_into<N>(n, suf, val, chint);
            bool               to_c1 = false;
            if constexpr (std::is_same_v<N, setlist_u8>)
               to_c1 = want_c1_promote(n, b);  // opt-in; reuses the tested grow_and_set widen path
            if (!to_c1 && n->set_branch(b, sab))
               return addr;
            if constexpr (!std::is_same_v<N, full_radix_node>)
            {
               // Re-home wider, skipping b. If the value spilled to a LEAF, hand it through —
               // re-making it via (suf,val) would leak the leaf just made (same rule as the
               // update site). A tail is re-made on the wide node.
               ptr_address up =
                   sab.is_pointer()
                       ? grow_and_set<N>(addr, b, sab, bytes{}, bytes{}, chint)
                       : grow_and_set<N>(addr, b, smart_address::null(), suf, val, peer_hint);
               if (addr != pre)
                  _ses->release(addr);
               return up;
            }
            return addr;  // full_radix never fails set_branch
         }

         // Is this branch a TERMINAL (an inline tail, or a pointer to a leaf_node)?
         if (is_terminal_branch(sa))
         {
            const inline_value* eiv  = term_iv(cn, sa);  // existing terminal (suffix+value+type)
            bytes               esuf = {eiv->data, size_t(eiv->suffix_len)};
            bytes               eval = {eiv->data + eiv->suffix_len, size_t(eiv->value_len)};
            const uint8_t       evt  = eiv->value_type;
            if (esuf.size() == suf.size() &&
                std::memcmp(esuf.data(), suf.data(), esuf.size()) == 0)  // same key → update
            {
               // COW exactly ONCE per level (set_term's discipline). The old two-step —
               // make_value(addr) cowing, then cow(addr) again for set_branch — cloned TWICE in
               // shared mode (the latch clones even ref==1) and orphaned the first clone: one
               // leaked object per transactional update (the bank-bench bloat).
               _old_value_size = int(eval.size());
               const ptr_address pre = addr;
               if (sa.is_pointer())  // external (leaf_node) value
               {
                  smart_address nsa =
                      smart_address::from_pointer(update_leaf_value<mode>(sa.as_pointer(), val));
                  if (nsa == sa)
                     return addr;  // updated in place (unique + same size)
                  N* n = cow<mode, N>(ref, addr);
                  rr_diverge<N>(n, range_of(b));  // big tier: privatize range before replace
                  if (n->set_branch(b, nsa))
                  {
                     _ses->release(sa.as_pointer());  // replaced: return this node's ref on it
                     return addr;
                  }
                  if constexpr (!std::is_same_v<N, full_radix_node>)
                  {
                     // grow_and_set skips b and sets b→nsa; releasing the COW'd addr runs its
                     // destroy, releasing the OLD leaf at b — so don't release sa here.
                     ptr_address up = grow_and_set<N>(addr, b, nsa, bytes{}, bytes{}, chint);
                     if (addr != pre)
                        _ses->release(addr);
                     return up;
                  }
                  return addr;  // full_radix never fails set_branch
               }
               // inline tail value: ONE cow (room for the new tail entry), write it, repoint b.
               // The OLD tail entry's bytes leak intra-node only (reclaimed on the next clone).
               N*            n   = cow<mode, N>(ref, addr, value_tail_need(suf, val));
               rr_diverge<N>(n, range_of(b));  // big tier: privatize range before replace
               smart_address nsa = make_value_into<N>(n, suf, val, chint);
               if (n->set_branch(b, nsa))
                  return addr;
               if constexpr (!std::is_same_v<N, full_radix_node>)
               {
                  // Re-home wider, skipping b. If the value spilled to a LEAF, hand it through
                  // (re-making it would leak the one just made); a tail is re-made on the wide node.
                  ptr_address up =
                      nsa.is_pointer()
                          ? grow_and_set<N>(addr, b, nsa, bytes{}, bytes{}, chint)
                          : grow_and_set<N>(addr, b, smart_address::null(), suf, val, peer_hint);
                  if (addr != pre)
                     _ses->release(addr);
                  return up;
               }
               return addr;  // full_radix never fails set_branch
            }
            // diverge → child holding both continuations (a wide setlist_u32 when the gate
            // is on and both suffixes have ≥4 bytes past the common prefix; else u8)
            int           cp    = common_prefix_len(esuf, suf);
            ptr_address   child = make_collision_child(esuf, eval, evt, suf, val, cp, chint);
            smart_address csa   = smart_address::from_pointer(child);
            N*            n   = cow<mode, N>(ref, addr);  // ref valid: make_collision_child didn't COW addr
            rr_diverge<N>(n, range_of(b));  // big tier: privatize range before replace
            if (n->set_branch(b, csa))
            {
               if (sa.is_pointer())
                  _ses->release(sa.as_pointer());  // old SAL leaf replaced (a tail just leaks)
               return addr;
            }
            if constexpr (!std::is_same_v<N, full_radix_node>)  // re-home failed → widen
            {
               // grow_and_set skips b and sets b→child; releasing the COW'd `addr` runs
               // its destroy, releasing the OLD terminal at b (a leaf) — so don't release sa.
               ptr_address up = grow_and_set<N>(addr, b, csa, bytes{}, bytes{}, chint);
               if (addr != addr_orig)
                  _ses->release(addr);
               return up;
            }
            return addr;  // full_radix never fails set_branch
         }

         // pointer to an inner node → recurse
         ptr_address child = sa.as_pointer();
         // ADAPTIVE: this node + the child it routes into are BOTH narrow (< 3 branches incl
         // term) → 2 levels deep without widening = clustered, provably small (caught early).
         // Collapse addr's whole subtree into a bucket leaf instead of noding further.
         if constexpr (std::is_same_v<N, setlist_u8>)
         {
            if (_adaptive && !_in_bucket && cn->nstems() + (cn->has_term_value() ? 1 : 0) < 3 &&
                !is_bucket(child) && node_narrow(child))
            {
               ptr_address bp;
               if (convert_subtree_to_bucket(addr, bp))
                  return bucket_leaf_insert(bp, key, depth0, val);  // caller releases old addr
            }
         }
         // child is shared (must clone) if THIS node is shared OR its children are co-owned;
         // the child subtree's MODE (instantiation) is shared iff its range is co-owned.
         ptr_address nc;
         if constexpr (mode.is_shared())
            nc = insert_node<mode>(child, key, depth + 1, val, chint);
         else  // PER-RANGE: only a co-owned range forces the child subtree into shared mode
            nc = range_shared<N>(cn, range_of(b))
                     ? insert_node<upsert_mode{upsert_mode::shared}>(child, key, depth + 1, val,
                                                                     chint)
                     : insert_node<mode>(child, key, depth + 1, val, chint);
         if (nc != child)
         {
            smart_address nsab = smart_address::from_pointer(nc);
            N*            n    = cow<mode, N>(ref, addr);  // ref valid: the recursion didn't COW this node
            rr_diverge<N>(n, range_of(b));  // big tier: privatize range before re-pointing
            if (!n->set_branch(b, nsab))  // compact re-home failed → widen with the pointer
            {
               if constexpr (!std::is_same_v<N, full_radix_node>)
               {
                  // grow_and_set re-stores addr's entries (skipping b) and sets b→nc;
                  // releasing the COW'd `addr` runs its destroy, which releases all its
                  // children including the OLD child at b — so do NOT release `child`.
                  ptr_address up = grow_and_set<N>(addr, b, nsab, bytes{}, bytes{}, peer_hint);
                  if (addr != addr_orig)
                     _ses->release(addr);
                  return up;
               }
            }
            _ses->release(child);  // normal in-tier replace
         }
         return addr;
      }

      // ── wide-stem (gated) ───────────────────────────────────────────────────────
      // Build the child for a terminal collision: a wide setlist_u32 when the gate is on
      // AND both diverging suffixes have ≥4 bytes past the common prefix (so each forms a
      // full 4-byte stem); otherwise the byte-radix setlist_u8 (unchanged behavior).
      // Build a child holding the EXISTING terminal (esuf,eval, value_type=evt) + the NEW key
      // (suf,val, value_type = the caller's stashed _ins_value_type). The existing terminal is
      // being MOVED here while its old leaf is about to be released; if it's a subtree, its value
      // bytes are a shared root address — retain it once so the new copy keeps it alive (the old
      // leaf's release then nets to the right count). Each entry is stamped with its OWN value_type.
      ptr_address make_collision_child(bytes esuf, bytes eval, uint8_t evt, bytes suf, bytes val,
                                       int cp, sal::alloc_hint hint)
      {
         const uint8_t nvt = _ins_value_type;  // the NEW key's value-type (caller's stash)
         if (evt == inline_value::vt_subtree)  // moved subtree keeps its root alive across the move
         {
            uint32_t raw = 0;
            std::memcpy(&raw, eval.data(), sizeof(raw));
            _ses->retain(sal::ptr_address(raw));
         }
         ptr_address child;
         if (_wide_stems && int(esuf.size()) - cp >= 4 && int(suf.size()) - cp >= 4)
         {
            child = make_node<setlist_u32>(esuf.subspan(0, size_t(cp)), hint);
            _ins_value_type = evt;
            add_wide_entry(child, esuf.subspan(size_t(cp)), eval, hint);
            _ins_value_type = nvt;
            add_wide_entry(child, suf.subspan(size_t(cp)), val, hint);
         }
         else
         {
            child = make_node<setlist_u8>(esuf.subspan(0, size_t(cp)), hint);
            _ins_value_type = evt;
            add_continuation(child, esuf.subspan(size_t(cp)), eval, hint);
            _ins_value_type = nvt;
            add_continuation(child, suf.subspan(size_t(cp)), val, hint);
         }
         _ins_value_type = nvt;  // restore the caller's stash
         return child;
      }
      // Add one entry to a fresh wide (u32, consume-4) node: 4-byte stem + inline remainder.
      // Precondition: suffix.size() >= 4 (the create gate guarantees it).
      void add_wide_entry(ptr_address& child, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         uint32_t      q  = read_stem<setlist_u32>(suffix, 0, 4);
         smart_address sv = make_value<setlist_u32>(child, suffix.subspan(4), value, hint);
         setlist_u32*  w  = grow_fresh<setlist_u32>(child, uint32_t(setlist_u32::stem_width) + 4);
         [[maybe_unused]] bool ok = w->set_branch(q, sv);
         assert(ok && "fresh wide node always has room for an entry");
      }

      // Insert into a wide setlist (u16/u32). Stride = consume_width; new stems GROW the
      // node (clone bigger). Mid-stem / 4 KB split is step 4 (asserted unreachable here).
      template <upsert_mode mode, class N>
      ptr_address router_insert_wide(ptr_address addr, bytes key, uint32_t depth, bytes val,
                                     sal::alloc_hint peer_hint)
      {
         const uint32_t    depth0 = depth;
         sal::smart_ref<N> ref    = _ses->get_ref<N>(addr);  // resolved once (reads + safe cows)
         if constexpr (!mode.is_shared())  // own node snapshot-reachable → SHARED instantiation
         {
            if (ref.ref() > 1)
               return router_insert_wide<upsert_mode{upsert_mode::shared}, N>(addr, key, depth,
                                                                              val, peer_hint);
         }
         const N* cn = ref.obj();
         bytes             pfx    = cn->prefix();
         int            avail  = int(key.size() - depth);
         int            maxp   = std::min(int(pfx.size()), avail);
         int p = common_prefix_len(key.subspan(depth, size_t(maxp)), pfx.subspan(0, size_t(maxp)));
         if (p < int(pfx.size()))
            return prefix_split<mode, N>(addr, key, depth, p, val, peer_hint);

         depth += uint32_t(pfx.size());
         int rem = int(key.size() - depth);
         if (rem == 0)  // key ends here → term
         {
            set_term<mode, N>(addr, val, peer_hint);
            return addr;
         }
         int cw = cn->consume_width();
         if (rem < cw)  // key ends inside the stride → split to a stride that fits it
         {
            ptr_address na = re_stride_wide<N>(addr, rem, peer_hint);
            return insert_node<mode>(na, key, depth0, val, peer_hint);
         }
         uint32_t        q     = read_stem<N>(key, depth, uint32_t(cw));
         sal::alloc_hint chint = cn->branch_hint();
         smart_address   sa    = cn->find(q);
         bytes           suf   = key.subspan(depth + uint32_t(cw));

         if (sa.is_null())  // new stem
         {
            if (!cn->room_for_one() && cn->size() >= max_node)  // at 4 KB → split narrower
            {
               int         Wp = cw >= 3 ? 2 : 1;  // u32/u24 → u16; u16 → u8
               ptr_address na = re_stride_wide<N>(addr, Wp, peer_hint);
               return insert_node<mode>(na, key, depth0, val, peer_hint);
            }
            // ONE cow sized for the tail AND a stem slot (two cows would orphan the first
            // clone under the shared latch — one leaked object per insert).
            N* w = cow<mode, N>(ref, addr, value_tail_need(suf, val) + uint32_t(N::stem_width) + 4);
            smart_address         sv = make_value_into<N>(w, suf, val, chint);
            [[maybe_unused]] bool ok = w->set_branch(q, sv);
            assert(ok && "wide node grows → set_branch must succeed");
            return addr;
         }

         if (is_terminal_branch(sa))
         {
            const inline_value* eiv  = term_iv(cn, sa);  // existing terminal (suffix+value+type)
            bytes               esuf = {eiv->data, size_t(eiv->suffix_len)};
            bytes               eval = {eiv->data + eiv->suffix_len, size_t(eiv->value_len)};
            const uint8_t       evt  = eiv->value_type;
            if (esuf.size() == suf.size() &&
                std::memcmp(esuf.data(), suf.data(), esuf.size()) == 0)  // same key → update
            {
               // ONE cow per level (see the u8 router site): the old make_value + cow pair
               // cloned twice under the shared latch and orphaned the first clone.
               _old_value_size = int(eval.size());
               if (sa.is_pointer())  // external (leaf_node) value
               {
                  smart_address nsa =
                      smart_address::from_pointer(update_leaf_value<mode>(sa.as_pointer(), val));
                  if (nsa == sa)
                     return addr;  // updated in place (unique + same size)
                  N* w = cow<mode, N>(ref, addr, uint32_t(N::stem_width) + 4);
                  [[maybe_unused]] bool ok = w->set_branch(q, nsa);
                  assert(ok && "wide node grows → set_branch must succeed");
                  _ses->release(sa.as_pointer());  // replaced: return this node's ref on it
                  return addr;
               }
               // inline tail value: one cow with room for the new tail (+ stem slot), repoint q.
               N* w = cow<mode, N>(ref, addr, value_tail_need(suf, val) + uint32_t(N::stem_width) + 4);
               smart_address         nsa = make_value_into<N>(w, suf, val, chint);
               [[maybe_unused]] bool ok  = w->set_branch(q, nsa);
               assert(ok && "wide node grows → set_branch must succeed");
               return addr;
            }
            int           cp    = common_prefix_len(esuf, suf);
            ptr_address   child = make_collision_child(esuf, eval, evt, suf, val, cp, chint);
            N*            w     = cow<mode, N>(ref, addr);  // ref valid (same-key make_value not taken here)
            if (sa.is_pointer())
               _ses->release(sa.as_pointer());
            w->set_branch(q, smart_address::from_pointer(child));
            return addr;
         }

         // pointer to an inner node → recurse (wide setlist children are direct-refcounted:
         // co-owned children put the child subtree into the shared instantiation).
         ptr_address child = sa.as_pointer();
         ptr_address nc;
         if constexpr (mode.is_shared())
            nc = insert_node<mode>(child, key, depth + uint32_t(cw), val, chint);
         else  // wide setlist children are direct-refcounted: co-owned → shared instantiation
            nc = children_shared<N>(cn)
                     ? insert_node<upsert_mode{upsert_mode::shared}>(child, key,
                                                                     depth + uint32_t(cw), val,
                                                                     chint)
                     : insert_node<mode>(child, key, depth + uint32_t(cw), val, chint);
         if (nc != child)
         {
            N* w = cow<mode, N>(ref, addr);  // ref valid (insert_node recurses into child, not addr)
            w->set_branch(q, smart_address::from_pointer(nc));
            _ses->release(child);
         }
         return addr;
      }

      // ── re_stride (the wide-node split) ─────────────────────────────────────────
      // Allocate a router that consumes `cw` key bytes: cw=1 → byte router (setlist_u8,
      // widens by density); cw=2 → setlist_u16; cw=3 → setlist_u32 in consume-3 mode
      // (rtype setlist_u24); cw=4 → setlist_u32. `extra` sizes the growable variants.
      ptr_address make_wide_node(int cw, bytes pfx, sal::alloc_hint hint, uint32_t extra)
      {
         if (cw == 1)  // byte router: full_radix (direct 256 array, never widens during build)
            return make_sized<full_radix_node>(pfx, hint, extra);
         ptr_address a;
         if (cw == 2)
            a = make_sized<setlist_u16>(pfx, hint, extra);
         else
         {
            a = make_sized<setlist_u32>(pfx, hint, extra);
            if (cw == 3)
               mut<setlist_u32>(a)->set_rtype(radix_node_type::setlist_u24);
         }
         return a;
      }
      template <class N>
      ptr_address make_sized(bytes pfx, sal::alloc_hint hint, uint32_t extra)
      {
         uint32_t sz = ucc::round_up_multiple<sal::cacheline_size>(uint32_t(sizeof(N)) +
                                                                   uint32_t(pfx.size()) + extra +
                                                                   N::headroom);
         ptr_address a = _ses->alloc<N>(sz, hint, pfx);
         mut<N>(a)->alloc_tail(tail_reserve);
         stamp<N>(a);
         ++_cc.node_allocs;
         return a;
      }
      // `consume` key bytes (already isolated in b[0..consume)) → a left-justified cell
      // value for a node whose cell is `cell_width` bytes wide.
      static uint32_t encode_cell(const uint8_t* b, int consume, int cell_width) noexcept
      {
         uint32_t v = 0;
         for (int i = 0; i < consume; ++i)
            v = (v << 8) | b[i];
         return v << (8 * (cell_width - consume));
      }
      static int cell_width_for(int cw) noexcept { return cw == 1 ? 1 : cw == 2 ? 2 : 4; }

      // Move a branch slot of `src` into `dst` (a fresh node): inline/leaf → copy the
      // value into dst; node pointer → retain and reuse as-is (suffix unchanged).
      smart_address carry_value(ptr_address& dst, sal::header_type dtype, ptr_address src,
                                smart_address sa, sal::alloc_hint hint)
      {
         if (!is_terminal_branch(sa))
         {
            _ses->retain(sa.as_pointer());
            return sa;
         }
         bytes suf, val;  // zero-copy views into src (stable arena; src stays live here)
         term_view(_ses->get_ref<sal::alloc_header>(src).obj(), sa, suf, val);
         return mv_into(dst, dtype, suf, val, hint);
      }
      // re_stride routers are only ever full_radix (cw=1), setlist_u16 (cw=2), or
      // setlist_u32/u24 (cw=3/4) — never u8/bitset — so none of these widen by density.
      smart_address mv_into(ptr_address& node, sal::header_type t, bytes suf, bytes val,
                            sal::alloc_hint hint)
      {
         if (t == setlist_u16::type_id)
            return make_value<setlist_u16>(node, suf, val, hint);
         if (t == setlist_u32::type_id)
            return make_value<setlist_u32>(node, suf, val, hint);  // u32 / u24
         return make_value<full_radix_node>(node, suf, val, hint);
      }
      void add_ptr(ptr_address& node, sal::header_type t, uint32_t q, smart_address ptr,
                   sal::alloc_hint)
      {
         if (t == setlist_u16::type_id)
            grow_fresh<setlist_u16>(node, 8)->set_branch(uint16_t(q), ptr);
         else if (t == setlist_u32::type_id)
            grow_fresh<setlist_u32>(node, 8)->set_branch(q, ptr);
         else
         {
            auto* w = grow_fresh<full_radix_node>(node);  // direct, no grow/widen
            rr_diverge<full_radix_node>(w, range_of(uint8_t(q)));  // privatize range before set
            w->set_branch(uint8_t(q), ptr);
         }
      }
      void add_inline(ptr_address& node, sal::header_type t, uint32_t q, bytes suf, bytes val,
                      sal::alloc_hint hint)
      {
         smart_address sv = mv_into(node, t, suf, val, hint);
         add_existing(node, t, q, sv, hint);
      }

      // Split a wide node (always a u32 cell, consume W) into a router that consumes Wp<W,
      // pushing the leftover W-Wp tail bytes into each entry (inline → suffix; pointer →
      // a group child keyed by the tail). Returns the new subtree; caller releases `addr`.
      template <class Src>
      ptr_address re_stride_wide(ptr_address addr, int Wp, sal::alloc_hint hint)
      {
         ++_cc.restrides;
         // `old` (the source) is read-only here and addr is not mutated/freed until the
         // caller releases it, and the arena is address-stable, so we hold `old` and read
         // it in place across the build allocations. Stems are stored SORTED, so we walk
         // them in key order directly — no decode-all scratch array, no re-sort, no cap.
         const Src* old = _ses->get_ref<Src>(addr).obj();
         const int  W   = old->consume_width();
         const int  tw  = W - Wp;  // leftover stem bytes per entry (<=3: W<=4, Wp>=1)
         const int  n   = old->nstems();

         auto decode = [&](int i, uint8_t* b) {  // big-endian stem bytes of entry i (W bytes)
            const uint32_t c = old->stem_at(i);
            for (int k = 0; k < W; ++k)
               b[k] = uint8_t((c >> (8 * (Src::stem_width - 1 - k))) & 0xFF);
         };

         ptr_address      na    = make_wide_node(Wp, old->prefix(), hint,
                                                 old->head_size() + old->tail_used() + 64);
         sal::header_type nt    = _ses->get_ref<sal::alloc_header>(na)->type();
         const int        pcell = cell_width_for(Wp);

         if (smart_address term = old->term_value(); !term.is_null())
            set_term_of(na, nt, carry_value(na, nt, addr, term, hint));  // term is before the stride

         int     i = 0;
         uint8_t bi[4], bj[4];
         while (i < n)
         {
            decode(i, bi);
            int j = i + 1;
            for (; j < n; ++j)  // group entries sharing the high Wp bytes (sorted → contiguous)
            {
               decode(j, bj);
               if (std::memcmp(bi, bj, size_t(Wp)) != 0)
                  break;
            }
            const uint32_t pq = encode_cell(bi, Wp, pcell);
            if (j - i == 1 && is_terminal_branch(old->branch_at(i)))  // inline singleton
            {
               bytes suf, val;  // zero-copy views into `old` (stable arena, old stays live)
               term_view(old, old->branch_at(i), suf, val);
               uint8_t kb[4 + 256];  // tw (<=3) leftover stem bytes + inline suffix (len is u8)
               std::memcpy(kb, bi + Wp, size_t(tw));
               std::memcpy(kb + tw, suf.data(), suf.size());
               add_inline(na, nt, pq, bytes{kb, size_t(tw) + suf.size()}, val, hint);
            }
            else  // group child (consumes the tw tail bytes)
            {
               const int        ccell = cell_width_for(tw);
               ptr_address      child = make_wide_node(tw, bytes{}, hint, uint32_t(j - i) * 16 + 64);
               sal::header_type ct    = _ses->get_ref<sal::alloc_header>(child)->type();
               for (int k = i; k < j; ++k)
               {
                  uint8_t bk[4];
                  decode(k, bk);
                  const uint32_t      cq = encode_cell(bk + Wp, tw, ccell);
                  const smart_address cs = carry_value(child, ct, addr, old->branch_at(k), hint);
                  if (!is_terminal_branch(cs))  // a node pointer (carry_value retained it)
                     add_ptr(child, ct, cq, cs, hint);
                  else  // inline value already copied into child by carry_value → just set it
                     add_existing(child, ct, cq, cs, hint);
               }
               add_ptr(na, nt, pq, smart_address::from_pointer(child), hint);
            }
            i = j;
         }
         return na;
      }
      // Set a slot to an already-resident smart_address (no value creation, no widen).
      void add_existing(ptr_address& node, sal::header_type t, uint32_t q, smart_address sa,
                        sal::alloc_hint)
      {
         if (t == setlist_u16::type_id)
            grow_fresh<setlist_u16>(node, 8)->set_branch(uint16_t(q), sa);
         else if (t == setlist_u32::type_id)
            grow_fresh<setlist_u32>(node, 8)->set_branch(q, sa);
         else
         {
            auto* w = grow_fresh<full_radix_node>(node);
            rr_diverge<full_radix_node>(w, range_of(uint8_t(q)));  // privatize range before set
            w->set_branch(uint8_t(q), sa);
         }
      }
      void set_term_of(ptr_address node, sal::header_type t, smart_address sv)
      {
         if (t == setlist_u16::type_id)
            mut<setlist_u16>(node)->set_term_value(sv);
         else if (t == setlist_u32::type_id)
            mut<setlist_u32>(node)->set_term_value(sv);
         else
            mut<full_radix_node>(node)->set_term_value(sv);
      }

      // Add a (suffix→value) continuation under a fresh setlist `node` (inline if it
      // fits, else a SAL leaf). An empty suffix is the node's term.
      void add_continuation(ptr_address& node, bytes suffix, bytes value, sal::alloc_hint hint)
      {
         if (suffix.empty())
         {
            smart_address sv = make_value<setlist_u8>(node, bytes{}, value, hint);
            mut<setlist_u8>(node)->set_term_value(sv);
            return;
         }
         uint8_t       b  = suffix[0];
         smart_address sv = make_value<setlist_u8>(node, suffix.subspan(1), value, hint);
         [[maybe_unused]] bool ok = mut<setlist_u8>(node)->set_branch(b, sv);
         assert(ok && "two-entry setlist always fits");
      }

      template <upsert_mode mode, class N>
      void set_term(ptr_address& addr, bytes val, sal::alloc_hint peer_hint)
      {
         sal::smart_ref<N> ref = _ses->get_ref<N>(addr);  // resolved once; threaded into cow()
         smart_address     tv  = ref->term_value();
         if (tv.is_tail())  // update inline term
         {
            const inline_value* iv = inline_of(ref.obj(), tv);
            _old_value_size        = iv->value_len;
            if (iv->value_len == val.size())  // same size → overwrite in place
            {
               N*    n   = cow<mode, N>(ref, addr);
               auto* wiv = reinterpret_cast<inline_value*>(reinterpret_cast<char*>(n) + n->size() -
                                                           tv.tail_offset());
               std::memcpy(wiv->data + wiv->suffix_len, val.data(), val.size());
            }
            else  // grow → new tail (old leaks), cow once
            {
               N* n = cow<mode, N>(ref, addr, value_tail_need(bytes{}, val));
               n->set_term_value(make_value_into<N>(n, bytes{}, val, peer_hint));
            }
            return;
         }
         if (tv.is_pointer())  // update leaf term
         {
            ptr_address old = tv.as_pointer();
            _old_value_size = int(_ses->get_ref<leaf_node>(old).obj()->value_bytes().size());
            ptr_address tl  = update_leaf_value<mode>(old, val);
            if (tl != old)
            {
               N* n = cow<mode, N>(ref, addr);
               _ses->release(old);
               n->set_term_value(smart_address::from_pointer(tl));
            }
            return;
         }
         N* n = cow<mode, N>(ref, addr, value_tail_need(bytes{}, val));  // new term, cow once
         n->set_term_value(make_value_into<N>(n, bytes{}, val, peer_hint));
      }

      // Split `addr`'s prefix at p: a new setlist parent takes key[depth..depth+p];
      // the demoted node (prefix loses p+1 bytes) hangs under its old prefix byte.
      template <upsert_mode mode, class N>
      ptr_address prefix_split(ptr_address addr, bytes key, uint32_t depth, int p, bytes val,
                               sal::alloc_hint peer_hint)
      {
         uint8_t ob = _ses->get_ref<N>(addr).obj()->prefix()[size_t(p)];

         ptr_address dnode = addr;
         N*          d     = cow<mode, N>(dnode);
         int         newpl = int(d->prefix_length()) - (p + 1);
         uint8_t     pbuf[sal::cacheline_size];  // a prefix never exceeds prefix_capacity (<1 CL)
         std::memcpy(pbuf, d->prefix().data() + p + 1, size_t(newpl));
         reprefix<N>(dnode, bytes{pbuf, size_t(newpl)});  // changes prefix AND re-stamps the shape

         ptr_address np = make_node<setlist_u8>(key.subspan(depth, size_t(p)), peer_hint);
         mut<setlist_u8>(np)->set_branch(ob, smart_address::from_pointer(dnode));
         if (dnode == addr)
            _ses->retain(dnode);  // in-place demote: np is a 2nd parent; caller releases addr

         add_continuation(np, key.subspan(depth + uint32_t(p)), val, peer_hint);
         return np;
      }
      // The SINGLE prefix-change primitive — and it ALSO re-stamps the dispatch shape, because
      // the shape is a function of the prefix (a prefix forces r1; prefix-less keeps the wide
      // tier). Coupling the two here means no call site can leave a wide-stamped node holding a
      // prefix — the desync that made the skip-header find misroute (nondeterministically) in
      // adaptive mode. Prefix changes are the ONLY shape-affecting mutation; alloc already
      // defaults the control type to r1 (the safe slow path), so this is the only stamp a
      // mutation must do.
      template <class N>
      void reprefix(ptr_address a, bytes p)
      {
         N* n = mut<N>(a);
         // Every setlist width stores stems+branches at a prefix-relative offset, so a prefix
         // change must relocate that block — the node's own reprefix() (wide-aware for u16/u32,
         // end-anchored for u8). bitset/full keep branches in separate cachelines, so the cheap
         // base set_prefix() is correct for them.
         if constexpr (std::is_same_v<N, setlist_u8> || std::is_same_v<N, setlist_u16> ||
                       std::is_same_v<N, setlist_u32>)
            n->reprefix(p);
         else
            n->set_prefix(p);
         stamp<N>(a);
      }

      // ── widen: build a wider encoding from `from` + add (setb→{suf,val}) ─────────
      template <class To>
      smart_address reinline(To* nn, const inline_value* iv) noexcept
      {
         uint32_t n   = uint32_t(iv->byte_size());
         auto*    niv = reinterpret_cast<inline_value*>(nn->alloc_tail(uint16_t(n)));
         std::memcpy(niv, iv, n);
         return smart_address::from_tail(nn->tail_used());
      }
      // Build a wider encoding from `from`, then set setb. `setsa` non-null = set that
      // pointer as-is (a replaced child — fresh, not retained); null = make a new value
      // {suf,val} on the wider node. Re-inlines tails / copies+retains pointer children.
      template <class To, class From>
      ptr_address try_upgrade(ptr_address from, uint8_t setb, smart_address setsa, bytes suf,
                              bytes val, sal::alloc_hint hint, bool& ok)
      {
         const From* cn = _ses->get_ref<From>(from).obj();
         // INVARIANT: the source node's prefix must fit the wider tier, or alloc<To>(prefix) below
         // overflows To's CL0 into its branch storage → corruption → SIGBUS. Divider nodes are
         // capped at divider_prefix_cap precisely to guarantee this (see make_divider_tail).
         assert(cn->prefix().size() <= To::prefix_capacity &&
                "source prefix exceeds upgrade tier capacity — divider_prefix_cap violated");
         uint32_t    need =
             ucc::round_up_multiple<sal::cacheline_size>(To::head_bytes() + cn->tail_used() +
                                                         tail_reserve + 64 + To::headroom);
         ptr_address na = _ses->alloc<To>(need, hint, cn->prefix());
         To*         nn = mut<To>(na);
         nn->alloc_tail(tail_reserve);
         rr_birth<To>(nn);  // big tier → its counters; the term+branch retains below are the base
         stamp<To>(na);
         ok = true;
         smart_address tv = cn->term_value();
         if (tv.is_tail())
            nn->set_term_value(reinline<To>(nn, inline_of(cn, tv)));
         else if (!tv.is_null())
         {
            nn->set_term_value(tv);
            if (tv.is_pointer())
               _ses->retain(tv.as_pointer());
         }
         cn->for_each_entry([&](uint8_t byte, smart_address br) {
            if (!ok || byte == setb)
               return;
            if (br.is_tail())
            {
               if (!nn->set_branch(byte, reinline<To>(nn, inline_of(cn, br))))
                  ok = false;
            }
            else if (!nn->set_branch(byte, br))
               ok = false;
            else if (br.is_pointer())
               _ses->retain(br.as_pointer());
         });
         if (ok)  // set the new branch: a pointer as-is, or a freshly inlined value
         {
            if (!setsa.is_null())
               ok = nn->set_branch(setb, setsa);  // replaced child pointer (fresh, no retain)
            else
            {
               smart_address sv = make_value<To>(na, suf, val, hint);
               nn               = mut<To>(na);  // make_value may have grown na
               ok               = nn->set_branch(setb, sv);
            }
         }
         if (!ok)
         {
            _ses->release(na);
            return ptr_address(0);
         }
         return na;
      }
      // Proactive c1 promotion (opt-in, set_c1_promote): a setlist needn't OVERFLOW to be a c1.
      // Once a prefix-less setlist reaches c1_promote stems AND its whole byte range (incl the
      // new byte) fits a c1 window, force the (tested) grow_and_set path → smallest fitting c1.
      // Stems are sorted, so the range is first..last — O(1). Range grows monotonically, so this
      // fires at most once. Default OFF: on a memory-bound walk (dict) the dispatch win is hidden
      // under the per-hop miss; it pays when the working set is cache-resident / bounded-field.
      static constexpr int c1_promote = 8;
      bool                 want_c1_promote(const setlist_u8* n, uint8_t b) const noexcept
      {
         const int cnt = n->nstems();
         if (!_c1_promote_on || cnt < c1_promote || !n->prefix().empty())
            return false;
         int lo = n->stem_at(0), hi = n->stem_at(cnt - 1);
         lo = b < lo ? b : lo;
         hi = b > hi ? b : hi;
         return (hi - lo) < bitset_c1_128::width;
      }
      // Build a 1-CL narrow-window c1 node from `from`'s entries (+ setb) when they fit a
      // [base, base+To::width) window with <= To::cap branches. Returns 0 (ok=false) if the
      // key range is too wide or there are too many branches — the caller tries the next
      // width, then R2. c1 carries no prefix, so a prefixed source can't upgrade to c1.
      // Mirrors try_upgrade's tail-reinline / pointer-retain / setb-skip contract exactly.
      template <class To, class From>
      ptr_address try_upgrade_c1(ptr_address from, uint8_t setb, smart_address setsa, bytes suf,
                                 bytes val, sal::alloc_hint hint, bool& ok)
      {
         ok             = false;
         const From* cn = _ses->get_ref<From>(from).obj();
         if (!cn->prefix().empty())
            return ptr_address(0);  // c1 has no inline prefix
         // From is a (sorted) setlist, so its byte range is first..last — O(1). Fold in setb.
         const int en = cn->nstems();
         int       lo = setb, hi = setb;
         if (en)
         {
            const int slo = cn->stem_at(0), shi = cn->stem_at(en - 1);
            lo = slo < lo ? slo : lo;
            hi = shi > hi ? shi : hi;
         }
         const int n = en + 1;  // entries + the new branch being added
         if (hi - lo >= To::width || n > To::cap)
            return ptr_address(0);  // range too wide / too many branches → caller widens further
         uint32_t need = ucc::round_up_multiple<sal::cacheline_size>(
             To::head_bytes() + cn->tail_used() + tail_reserve + 64 + To::headroom);
         ptr_address na = _ses->alloc<To>(need, hint, bytes{});
         To*         nn = mut<To>(na);
         nn->alloc_tail(tail_reserve);
         rr_birth<To>(nn);  // big tier → its counters; the term+branch retains below are the base
         nn->set_window_base(uint8_t(lo));
         stamp<To>(na);
         ok               = true;
         smart_address tv = cn->term_value();
         if (tv.is_tail())
            nn->set_term_value(reinline<To>(nn, inline_of(cn, tv)));
         else if (!tv.is_null())
         {
            nn->set_term_value(tv);
            if (tv.is_pointer())
               _ses->retain(tv.as_pointer());
         }
         cn->for_each_entry([&](uint8_t byte, smart_address br) {
            if (!ok || byte == setb)
               return;
            if (br.is_tail())
            {
               if (!nn->set_branch(byte, reinline<To>(nn, inline_of(cn, br))))
                  ok = false;
            }
            else if (!nn->set_branch(byte, br))
               ok = false;
            else if (br.is_pointer())
               _ses->retain(br.as_pointer());
         });
         if (ok)
         {
            if (!setsa.is_null())
               ok = nn->set_branch(setb, setsa);
            else
            {
               smart_address sv = make_value<To>(na, suf, val, hint);
               nn               = mut<To>(na);  // make_value may have grown na
               ok               = nn->set_branch(setb, sv);
            }
         }
         if (!ok)
         {
            _ses->release(na);
            return ptr_address(0);
         }
         return na;
      }
      // Returns a wider node holding all of `from`'s entries plus setb→(setsa | {suf,val}).
      // The CALLER releases `from` (so a snapshot still referencing it stays valid).
      template <class From>
      ptr_address grow_and_set(ptr_address from, uint8_t b, smart_address setsa, bytes suf, bytes val,
                               sal::alloc_hint hint)
      {
         ++_cc.upgrades;
         bool        ok = false;
         ptr_address up;
         if constexpr (std::is_same_v<From, setlist_u8>)
         {
            // Dense-in-a-narrow-range first: a 1-CL c1 window holds MORE than a setlist
            // (23/22/20 vs ~19) and routes by popcount, not scan. Smallest window that
            // fits wins; only a too-wide / too-populous range falls through to R2.
            up = try_upgrade_c1<bitset_c1_32, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
            up = try_upgrade_c1<bitset_c1_64, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
            up = try_upgrade_c1<bitset_c1_128, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
            up = try_upgrade<bitset_r2_node, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
         }
         if constexpr (!std::is_same_v<From, bitset_r4_node>)
         {
            up = try_upgrade<bitset_r4_node, From>(from, b, setsa, suf, val, hint, ok);
            if (ok)
               return up;
         }
         return try_upgrade<full_radix_node, From>(from, b, setsa, suf, val, hint, ok);  // always fits
      }

      // ── downgrade (the inverse of widen) ─────────────────────────────────────────
      // After a remove shrinks a wide node, re-encode it into the smallest tier that holds
      // its entries with HYSTERESIS — the count must sit ~25% below where that tier would
      // re-widen, so churn around the boundary doesn't thrash grow↔shrink. A node stuck at a
      // big tier wastes cachelines (worse aggregate cache footprint on memory-bound queries)
      // and lengthens a setlist scan; shrinking keeps the tree tight. These caps are the
      // regrow points; the rebuild is the source of truth (it fails → we keep the tier).
      static constexpr int dn_setlist = 14;  // setlist re-widens ~19  → shrink to setlist at <=14
      static constexpr int dn_r2      = 40;  // r2 (2x~27) re-widens ~54 → shrink to r2 at <=40
      static constexpr int dn_r4      = 87;  // r4 (4x~29) re-widens ~116 → shrink to r4 at <=87

      // Rebuild ALL of `from`'s entries (+ term) into a fresh `To` (no added key); returns the
      // new node or 0 (ok=false) if they don't fit To. Same tail-reinline / pointer-retain
      // contract as try_upgrade — so the caller releases `from` and refcounts stay balanced.
      template <class To, class From>
      ptr_address rebuild_into(ptr_address from, bool& ok)
      {
         const From* cn   = _ses->get_ref<From>(from).obj();
         uint32_t    need = ucc::round_up_multiple<sal::cacheline_size>(
             To::head_bytes() + cn->tail_used() + tail_reserve + 64 + To::headroom);
         ptr_address na = _ses->alloc<To>(need, sal::alloc_hint{}, cn->prefix());
         To*         nn = mut<To>(na);
         nn->alloc_tail(tail_reserve);
         rr_birth<To>(nn);  // big tier → its counters; the term+branch retains below are the base
         stamp<To>(na);
         ok               = true;
         smart_address tv = cn->term_value();
         if (tv.is_tail())
            nn->set_term_value(reinline<To>(nn, inline_of(cn, tv)));
         else if (!tv.is_null())
         {
            nn->set_term_value(tv);
            if (tv.is_pointer())
               _ses->retain(tv.as_pointer());
         }
         cn->for_each_entry([&](uint8_t byte, smart_address br) {
            if (!ok)
               return;
            if (br.is_tail())
            {
               if (!nn->set_branch(byte, reinline<To>(nn, inline_of(cn, br))))
                  ok = false;
            }
            else if (!nn->set_branch(byte, br))
               ok = false;
            else if (br.is_pointer())
               _ses->retain(br.as_pointer());
         });
         if (!ok)
         {
            _ses->release(na);
            return ptr_address(0);
         }
         return na;
      }

      // If `addr` (a wide tier) has shrunk enough, return a smaller-tier rebuild (caller frees
      // the old per the COW contract); else return `addr` unchanged. setlist is the floor.
      template <class N>
      ptr_address maybe_downgrade(ptr_address addr)
      {
         if constexpr (std::is_same_v<N, setlist_u8> || std::is_same_v<N, setlist_u16> ||
                       std::is_same_v<N, setlist_u32>)
            return addr;  // already the floor / wide-stem variants don't byte-downgrade
         else
         {
            const int   cnt = _ses->get_ref<N>(addr).obj()->branch_count();
            bool        ok  = false;
            ptr_address dg;
            if (cnt <= dn_setlist)  // smallest first: a 1-CL setlist if it fits
            {
               dg = rebuild_into<setlist_u8, N>(addr, ok);
               if (ok)
                  return dg;
            }
            if constexpr (std::is_same_v<N, full_radix_node> || std::is_same_v<N, bitset_r4_node>)
               if (cnt <= dn_r2)
               {
                  dg = rebuild_into<bitset_r2_node, N>(addr, ok);
                  if (ok)
                     return dg;
               }
            if constexpr (std::is_same_v<N, full_radix_node>)
               if (cnt <= dn_r4)
               {
                  dg = rebuild_into<bitset_r4_node, N>(addr, ok);
                  if (ok)
                     return dg;
               }
            return addr;
         }
      }

      // ── bucket mode: radix routers (full_radix here) + bucket_node leaves ───────────
      // Reuses the engine's session/make_node/mut + bucket_node + full_radix_node. In-place
      // mutation (no MVCC snapshots). A full bucket splits into a router over child buckets;
      // a router whose prefix a new key diverges from is prefix-split (path compression).
      sal::header_type ntype(ptr_address a) { return _ses->get_ref<sal::alloc_header>(a)->type(); }
      bucket_node*     bkt(ptr_address a) { return _ses->get_ref<bucket_node>(a).modify().get(); }

      // ── adaptive radix→bucket (deep-narrow detection + subtree collapse) ─────────
      bool is_bucket(ptr_address a)
      {
         return _ses->get_ref<sal::alloc_header>(a)->type() == bucket_node::type_id;
      }
      // < 3 branches (children + term) → "narrow". Only setlist/single-child nodes can be
      // narrow; bitset/full are wide; leaf/bucket aren't radix routers.
      bool node_narrow(ptr_address a)
      {
         auto t = _ses->get_ref<sal::alloc_header>(a)->type();
         if (t == setlist_u8::type_id)
         {
            auto* n = _ses->get_ref<setlist_u8>(a).obj();
            return n->nstems() + (n->has_term_value() ? 1 : 0) < 3;
         }
         if (t == setlist_u16::type_id)
         {
            auto* n = _ses->get_ref<setlist_u16>(a).obj();
            return n->nstems() + (n->has_term_value() ? 1 : 0) < 3;
         }
         if (t == setlist_u32::type_id)
         {
            auto* n = _ses->get_ref<setlist_u32>(a).obj();
            return n->nstems() + (n->has_term_value() ? 1 : 0) < 3;
         }
         return false;
      }
      // Insert into a bucket LEAF; on overflow split it to a radix router (bkt_split) and route
      // the key into that router via the normal engine. Returns the (possibly new) address.
      ptr_address bucket_leaf_insert(ptr_address bp, bytes key, uint32_t depth, bytes val)
      {
         // Everything reached from here lives in a bucket subtree; suppress the radix→bucket
         // trigger so a split's rebuilt radix (collision setlists) doesn't re-collapse mid-build.
         const bool was = _in_bucket;
         _in_bucket     = true;
         ptr_address r  = bucket_leaf_insert_impl(bp, key, depth, val);
         _in_bucket     = was;
         return r;
      }
      ptr_address bucket_leaf_insert_impl(ptr_address bp, bytes key, uint32_t depth, bytes val)
      {
         if (!is_bucket(bp))  // already split to a router (e.g. mid-collapse) → route normally
            return insert_node<upsert_mode{}>(bp, key, depth, val, {});  // bucket mode: in-place engine
         int r = bkt(bp)->insert(key.data() + depth, int(key.size() - depth), val.data(),
                                 int(val.size()));
         if (r != 0)
         {
            _old_value_size = r < 0 ? 0 : -1;
            return bp;
         }
         ptr_address rp  = bkt_split(bp);                // overflow → router (bp freed by CALLER)
         ptr_address nrp = insert_node<upsert_mode{}>(rp, key, depth, val, {});  // bucket: in-place
         if (nrp != rp)
            _ses->release(rp);  // the freshly-split router got replaced (COW) → free it
         return nrp;
      }
      // Collect (suffix,value) of every key in addr's subtree, suffix relative to addr. Bounded:
      // returns false (abort, stay radix) if the subtree is bigger than a bucket should absorb —
      // which, with early detection, is rare.
      static constexpr int conv_sbuf = 8192, conv_vbuf = 4096, conv_cap = 300;
      struct CE { uint16_t so, vo; uint8_t sl, vl; };
      template <class Ad>
      bool collect_in(Ad ad, KB& acc, uint8_t* sb, uint16_t& sp, uint8_t* vb, uint16_t& vp,
                      CE* ents, int& n, int cap)
      {
         const uint32_t save = acc.len;
         bytes          pfx  = ad.prefix();
         acc.append(pfx.data(), pfx.size());
         auto record = [&](bytes suf, value_view v) -> bool {
            const int sl = int(acc.len) + int(suf.size());
            if (n >= cap || sl > 255 || int(v.size()) > 255 || int(sp) + sl > conv_sbuf ||
                int(vp) + int(v.size()) > conv_vbuf)
               return false;
            ents[n].so = sp;
            ents[n].sl = uint8_t(sl);
            std::memcpy(sb + sp, acc.buf, acc.len);
            std::memcpy(sb + sp + acc.len, suf.data(), suf.size());
            sp = uint16_t(sp + sl);
            ents[n].vo = vp;
            ents[n].vl = uint8_t(v.size());
            std::memcpy(vb + vp, v.data(), v.size());
            vp = uint16_t(vp + v.size());
            ++n;
            return true;
         };
         if (!ad.term().is_null())
         {
            bytes      s;
            value_view v;
            term_view(ad.node(), ad.term(), s, v);
            if (!record(s, v))
               return false;
         }
         for (int c = ad.first(); ad.valid(c); c = ad.next(c))
         {
            const uint32_t s0 = acc.len;
            uint8_t        sbb[4];
            const int      nb = stem_bytes(ad.cell(c), ad.stem_width(), ad.cw(), sbb);
            acc.append(sbb, size_t(nb));
            smart_address br = ad.branch(c);
            if (is_terminal_branch(br))
            {
               bytes      s;
               value_view v;
               term_view(ad.node(), br, s, v);
               if (!record(s, v))
                  return false;
            }
            else if (!collect_walk(br.as_pointer(), acc, sb, sp, vb, vp, ents, n, cap))
               return false;
            acc.truncate(s0);
         }
         acc.truncate(save);
         return true;
      }
      bool collect_walk(ptr_address a, KB& acc, uint8_t* sb, uint16_t& sp, uint8_t* vb, uint16_t& vp,
                        CE* ents, int& n, int cap)
      {
         if (is_bucket(a))  // a deeper region is already a bucket → don't try to collapse
            return false;
         return dispatch(a, [&](auto ad) { return collect_in(ad, acc, sb, sp, vb, vp, ents, n, cap); });
      }
      // Build a bucket holding the whole subtree at addr; false if it's too big to collapse.
      bool convert_subtree_to_bucket(ptr_address addr, ptr_address& out)
      {
         uint8_t  sb[conv_sbuf];
         uint8_t  vb[conv_vbuf];
         CE       ents[conv_cap];
         uint16_t sp = 0, vp = 0;
         int      n  = 0;
         KB       acc;
         acc.len = 0;
         if (!collect_walk(addr, acc, sb, sp, vb, vp, ents, n, conv_cap))
            return false;
         ptr_address bp = make_bucket();
         for (int i = 0; i < n; ++i)
         {
            ptr_address nb = bucket_leaf_insert(bp, bytes{sb + ents[i].so, ents[i].sl}, 0,
                                                bytes{vb + ents[i].vo, ents[i].vl});
            if (nb != bp)
               _ses->release(bp);  // bp split/COW'd into nb → free the consumed node
            bp = nb;
         }
         out = bp;
         return true;
      }
      ptr_address bkt_make(bytes suf, bytes val)
      {
         ptr_address bp = make_bucket();
         bkt(bp)->insert(suf.data(), int(suf.size()), val.data(), int(val.size()));
         return bp;
      }
      void bkt_set_term(ptr_address rp, bytes v)  // store value as an empty-suffix inline term
      {
         // If a term already exists and the new value fits its slot, overwrite IN PLACE.
         // Otherwise alloc_tail leaks the old slot on every update — and repeated updates of
         // the same terminal (e.g. duplicate-key upserts) grow the tail past the node's
         // headroom and into the branch array. (The leak on a genuine grow is rare + bounded.)
         smart_address tv = _ses->get_ref<full_radix_node>(rp).obj()->term_value();
         if (!tv.is_null())
         {
            full_radix_node* r  = mut<full_radix_node>(rp);
            auto*            iv = reinterpret_cast<inline_value*>(
                reinterpret_cast<char*>(r) + r->size() - tv.tail_offset());
            if (iv->value_len >= int(v.size()))  // empty-suffix term; reuse the slot
            {
               iv->value_len = uint8_t(v.size());
               std::memcpy(iv->data, v.data(), v.size());
               return;
            }
         }
         const int        need = inline_value::byte_size(0, int(v.size()));
         full_radix_node* r    = mut<full_radix_node>(rp);
         auto*            iv   = reinterpret_cast<inline_value*>(r->alloc_tail(uint16_t(need)));
         iv->suffix_len = 0;
         iv->value_len  = uint8_t(v.size());
         iv->value_type = 0;
         std::memcpy(iv->data, v.data(), v.size());
         mut<full_radix_node>(rp)->set_term_value(smart_address::from_tail(r->tail_used()));
      }
      // Store a single (suffix,value) terminal INLINE in router rp under byte b — a from_tail
      // branch in the router's own tail heap, NOT a separate object. This is what stops a dense
      // terminal byte (sequential keys) from exploding into one bucket per key. Returns false
      // if it doesn't fit the inline cap or the router's tail is full (caller uses a bucket).
      // Grow a compact router to its working page ONCE its inline-terminal tail overflows, so a
      // dense terminal byte fills one page instead of spilling to a per-key bucket. In-place
      // model (no MVCC sharing): alloc the bigger page, blit head+tail, retain children, release
      // the old router. Returns the new address (the old one is freed here).
      ptr_address bkt_grow_router(ptr_address rp)
      {
         // Copy the prefix to the stack FIRST: the alloc below can relocate/compact the arena,
         // so neither rp's pointer nor a span into it may be held across it.
         uint8_t pbuf[256];
         int     pl;
         {
            bytes p = _ses->get_ref<full_radix_node>(rp).obj()->prefix();
            pl      = int(p.size());
            std::memcpy(pbuf, p.data(), size_t(pl));
         }
         ptr_address np = _ses->alloc<full_radix_node>(full_radix_node::page_bytes, sal::alloc_hint{},
                                                       bytes{pbuf, size_t(pl)});
         full_radix_node* nn = mut<full_radix_node>(np);
         _ses->get_ref<full_radix_node>(rp).obj()->copy_to(nn);  // re-resolve rp AFTER the alloc
         rr_birth<full_radix_node>(nn);  // its counters; the for_each_child retains below are the base
         nn->for_each_child([&](ptr_address c) { _ses->retain(c); });
         stamp<full_radix_node>(np);  // dispatch shape (find reads cb_type) — copy_to didn't set it
         _ses->release(rp);  // drop old router (its destroy releases each child once → net no change)
         return np;
      }
      bool bkt_try_inline(ptr_address& rp, uint8_t b, bytes suf, bytes val)
      {
         if (!fits_inline(suf, val))
            return false;
         full_radix_node* r    = mut<full_radix_node>(rp);
         const uint32_t   need = uint32_t(inline_value::byte_size(int(suf.size()), int(val.size())));
         if (r->free_heap() < need)
         {
            if (r->size() >= full_radix_node::page_bytes)
               return false;  // already at the working page and still full → spill to a bucket
            rp = bkt_grow_router(rp);
            r  = mut<full_radix_node>(rp);
            if (r->free_heap() < need)
               return false;
         }
         auto* iv       = reinterpret_cast<inline_value*>(r->alloc_tail(uint16_t(need)));
         iv->suffix_len = uint8_t(suf.size());
         iv->value_len  = uint8_t(val.size());
         iv->value_type = 0;
         std::memcpy(iv->data, suf.data(), suf.size());
         std::memcpy(iv->data + suf.size(), val.data(), val.size());
         r->set_branch(b, smart_address::from_tail(r->tail_used()));
         return true;
      }
      // A new key collides with an existing inline terminal under `byte`. Either update it in
      // place (same key) or promote: build a bucket holding the existing + new entries and
      // repoint the branch at it (the old inline slot leaks — rare).
      ptr_address bkt_promote_inline(ptr_address rp, uint8_t byte, smart_address br, bytes key,
                                     uint32_t depth, bytes val)
      {
         full_radix_node*    r  = mut<full_radix_node>(rp);
         const inline_value* iv = inline_of(r, br);
         bytes               esuf{iv->data, size_t(iv->suffix_len)};
         bytes               nsuf = key.subspan(depth + 1);
         if (esuf.size() == nsuf.size() &&
             (nsuf.empty() || std::memcmp(esuf.data(), nsuf.data(), nsuf.size()) == 0))
         {  // same key -> overwrite the value in place if it fits the slot, else re-inline (leak)
            if (iv->value_len >= int(val.size()))
            {
               auto* w      = const_cast<inline_value*>(iv);
               w->value_len = uint8_t(val.size());
               std::memcpy(w->data + w->suffix_len, val.data(), val.size());
               _old_value_size = 0;
               return rp;
            }
            if (bkt_try_inline(rp, byte, esuf, val))
            {
               _old_value_size = 0;
               return rp;
            }
         }
         // different key (or a value grow with no room): snapshot the existing entry off the
         // router tail, then build a bucket holding it + the new key.
         uint8_t   tmp[inline_value::max_inline_size];
         const int sl = iv->suffix_len, vl = iv->value_len;
         std::memcpy(tmp, iv->data, size_t(sl) + size_t(vl));
         ptr_address nb = bkt_make(bytes{tmp, size_t(sl)}, bytes{tmp + sl, size_t(vl)});
         nb             = bkt_insert(nb, key, depth + 1, val);
         mut<full_radix_node>(rp)->set_branch(byte, smart_address::from_pointer(nb));
         return rp;
      }
      ptr_address bkt_split(ptr_address bp)
      {
         // SNAPSHOT all entries (suffix+value bytes) off the bucket BEFORE allocating any
         // child: a child-build allocation can trigger the SAL compactor to relocate `bp`,
         // dangling raw pointers into it. After the snapshot bp is released. (Same discipline
         // as term_get — guard against object relocation, not just mmap base moves.)
         struct E { uint16_t off; uint8_t sl; uint8_t vl; };
         E        ent[1200];
         uint8_t  buf[bucket_node::page_bytes];
         int      n   = 0;
         uint16_t pos = 0;
         _ses->get_ref<bucket_node>(bp).obj()->for_each([&](const inline_value& iv) {
            assert(n < 1200 && pos + iv.suffix_len + iv.value_len <= int(sizeof(buf)));
            ent[n] = {pos, iv.suffix_len, iv.value_len};
            std::memcpy(buf + pos, iv.data, size_t(iv.suffix_len) + iv.value_len);
            pos = uint16_t(pos + iv.suffix_len + iv.value_len);
            ++n;
         });
         assert(n >= 2);
         auto val = [&](int k) { return bytes{buf + ent[k].off + ent[k].sl, ent[k].vl}; };

         int cp = 0, m = ent[0].sl < ent[n - 1].sl ? ent[0].sl : ent[n - 1].sl;
         while (cp < m && buf[ent[0].off + cp] == buf[ent[n - 1].off + cp]) ++cp;
         ptr_address rp = make_node<full_radix_node>(bytes{buf + ent[0].off, size_t(cp)});
         int         i  = 0;
         if (ent[0].sl == cp)  // lone key ending exactly at the common prefix
         {
            bkt_set_term(rp, val(0));
            i = 1;
         }
         while (i < n)
         {
            const uint8_t bch = buf[ent[i].off + cp];
            int           j   = i;
            while (j < n && ent[j].sl > cp && buf[ent[j].off + cp] == bch) ++j;
            // a lone key under this byte → inline terminal in the router (no per-key object);
            // this is what keeps a dense terminal byte from exploding into one bucket per key.
            if (j - i == 1 &&
                bkt_try_inline(rp, bch, bytes{buf + ent[i].off + cp + 1, size_t(ent[i].sl - cp - 1)},
                               val(i)))
            {
               i = j;
               continue;
            }
            ptr_address cbp = make_bucket();
            for (int k = i; k < j; ++k)  // recursive: a bucket-sized group becomes a sub-radix
               cbp = bkt_insert(cbp, bytes{buf + ent[k].off + cp + 1, size_t(ent[k].sl - cp - 1)}, 0,
                                val(k));
            mut<full_radix_node>(rp)->set_branch(bch, smart_address::from_pointer(cbp));
            i = j;
         }
         // Engine COW contract: return the new node; the CALLER releases the old `bp` (exactly
         // once). Freeing it here too would double-free, since router_insert/upsert already do
         // `if (nc != child) release(child)`.
         return rp;
      }
      ptr_address bkt_prefix_split(ptr_address rp, bytes key, uint32_t depth, int p, bytes val)
      {
         bytes   P  = _ses->get_ref<full_radix_node>(rp).obj()->prefix();
         uint8_t ob = P[size_t(p)];
         uint8_t pbuf[128];
         int     newpl = int(P.size()) - (p + 1);
         std::memcpy(pbuf, P.data() + p + 1, size_t(newpl));
         ptr_address np = make_node<full_radix_node>(bytes{key.data() + depth, size_t(p)});
         reprefix<full_radix_node>(rp, bytes{pbuf, size_t(newpl)});  // changes prefix AND re-stamps
         mut<full_radix_node>(np)->set_branch(ob, smart_address::from_pointer(rp));
         const uint32_t d2 = depth + uint32_t(p);
         if (d2 == key.size())
            bkt_set_term(np, val);
         else
            mut<full_radix_node>(np)->set_branch(
                key[d2],
                smart_address::from_pointer(bkt_make({key.data() + d2 + 1, key.size() - d2 - 1}, val)));
         return np;
      }
      ptr_address bkt_insert(ptr_address addr, bytes key, uint32_t depth, bytes val)
      {
         if (ntype(addr) == bucket_node::type_id)
         {
            int r = bkt(addr)->insert(key.data() + depth, int(key.size() - depth), val.data(),
                                      int(val.size()));
            if (r != 0)
            {
               _old_value_size = r < 0 ? 0 : -1;
               return addr;
            }
            ptr_address split = bkt_split(addr);  // bkt_split no longer frees its input
            _ses->release(addr);                   // so THIS caller frees the consumed bucket
            return bkt_insert(split, key, depth, val);
         }
         bytes     P     = _ses->get_ref<full_radix_node>(addr).obj()->prefix();
         const int avail = int(key.size() - depth);
         const int mm    = int(P.size()) < avail ? int(P.size()) : avail;
         int       p     = 0;
         while (p < mm && key[depth + p] == P[p]) ++p;
         if (p < int(P.size()))
            return bkt_prefix_split(addr, key, depth, p, val);
         depth += uint32_t(P.size());
         if (depth == key.size())
         {
            bkt_set_term(addr, val);
            return addr;
         }
         const uint8_t byte = key[depth];
         smart_address br   = _ses->get_ref<full_radix_node>(addr).obj()->find(byte);
         if (br.is_null())
         {
            bytes suf{key.data() + depth + 1, key.size() - depth - 1};
            if (!bkt_try_inline(addr, byte, suf, val))  // inline terminal, else a child bucket
               mut<full_radix_node>(addr)->set_branch(byte,
                                                      smart_address::from_pointer(bkt_make(suf, val)));
            _old_value_size = -1;
            return addr;
         }
         if (br.is_tail())  // existing inline terminal under this byte: update or promote
            return bkt_promote_inline(addr, byte, br, key, depth, val);
         ptr_address nc = bkt_insert(br.as_pointer(), key, depth + 1, val);
         if (nc != br.as_pointer())
            mut<full_radix_node>(addr)->set_branch(byte, smart_address::from_pointer(nc));
         return addr;
      }
      bool bkt_find(bytes key, value_view& out)
      {
         ptr_address addr  = _root.address();
         uint32_t    depth = 0;
         for (;;)
         {
            if (ntype(addr) == bucket_node::type_id)
            {
               const inline_value* iv =
                   _ses->get_ref<bucket_node>(addr).obj()->find(key.data() + depth, int(key.size() - depth));
               if (!iv)
                  return false;
               out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
               return true;
            }
            const full_radix_node* r = _ses->get_ref<full_radix_node>(addr).obj();
            bytes                  P = r->prefix();
            if (key.size() - depth < P.size() ||
                (P.size() && std::memcmp(key.data() + depth, P.data(), P.size()) != 0))
               return false;
            depth += uint32_t(P.size());
            if (depth == key.size())
            {
               smart_address tv = r->term_value();
               if (tv.is_null())
                  return false;
               const inline_value* iv = inline_of(r, tv);
               out = value_view(reinterpret_cast<const char*>(iv->data + iv->suffix_len), iv->value_len);
               return true;
            }
            smart_address br = r->find(key[depth]);
            if (br.is_null())
               return false;
            if (br.is_tail())  // inline terminal in the router: suffix == key[depth+1:]
               return inline_at(r, br, key.subspan(depth + 1), out);
            addr = br.as_pointer();
            ++depth;
         }
      }

      // Bucket-mode remove (in-place, no MVCC — matches bkt_insert). Descends the full_radix
      // routers to the bucket leaf / inline terminal holding the key. An emptied bucket leaf is
      // released and its parent branch cleared; an inline terminal / router term is cleared in
      // place (its tail orphans). Empty routers are left (correct for find; not a leak).
      bool bkt_remove(bytes key)
      {
         ptr_address addr = _root.address(), parent{};
         uint8_t     pb = 0;
         bool        have_parent = false;
         uint32_t    depth       = 0;
         for (;;)
         {
            if (ntype(addr) == bucket_node::type_id)
            {
               const uint8_t* s  = key.data() + depth;
               const int      sn = int(key.size() - depth);
               if (!_ses->get_ref<bucket_node>(addr).obj()->find(s, sn))
                  return false;
               bkt(addr)->remove(s, sn);
               if (have_parent && _ses->get_ref<bucket_node>(addr).obj()->nent == 0)
               {
                  mut<full_radix_node>(parent)->remove_branch(pb);
                  _ses->release(addr);  // free the drained bucket
               }
               return true;
            }
            const full_radix_node* r = _ses->get_ref<full_radix_node>(addr).obj();
            bytes                  P = r->prefix();
            if (key.size() - depth < P.size() ||
                (P.size() && std::memcmp(key.data() + depth, P.data(), P.size()) != 0))
               return false;
            depth += uint32_t(P.size());
            if (depth == key.size())  // router term value
            {
               smart_address tv = r->term_value();
               if (tv.is_null())
                  return false;
               mut<full_radix_node>(addr)->set_term_value(smart_address::null());
               if (tv.is_pointer())
                  _ses->release(tv.as_pointer());
               return true;
            }
            uint8_t       b  = key[depth];
            smart_address br = r->find(b);
            if (br.is_null())
               return false;
            if (br.is_tail())  // inline terminal in the router
            {
               bytes      esuf;
               value_view ev;
               term_view(r, br, esuf, ev);
               bytes suf = key.subspan(depth + 1);
               if (esuf.size() != suf.size() ||
                   (esuf.size() && std::memcmp(esuf.data(), suf.data(), esuf.size()) != 0))
                  return false;
               mut<full_radix_node>(addr)->remove_branch(b);  // inline tail orphans
               return true;
            }
            parent      = addr;
            pb          = b;
            have_parent = true;
            addr        = br.as_pointer();
            ++depth;
         }
      }

      // ── unclean-shutdown detection and recovery ──────────────────────────────────
      static sal::shutdown_state prior_shutdown(const std::filesystem::path& dir, bool wipe)
      {
         // A store that does not exist yet (or is about to be wiped) has no history.
         if (wipe || !std::filesystem::exists(dir / "roots"))
            return sal::shutdown_state::clean;
         return sal::shutdown_record::read(dir);
      }

      // One dense node: its terminal value is held directly; each range's branch set is
      // owned once per range COUNTER, shared by every node version holding that counter.
      // Re-tally the counter per node, and follow the branches only on first sight.
      template <class N>
      bool recovery_walk_dense(const sal::alloc_header*                    obj,
                               const std::function<void(sal::ptr_address)>& retain)
      {
         if (obj->type() != N::type_id)
            return false;
         const N* n = reinterpret_cast<const N*>(obj);
         if (not n->rc_has_ctrs())
            return false;  // owns each child directly: the type's visit_children is exact
         const smart_address tv = n->term_value();
         if (tv.is_pointer())
            retain(tv.as_pointer());
         bool first[range_count];
         for (int r = 0; r < range_count; ++r)
            first[r] = _alloc->iref().retally(n->rc_ctr(r));
         n->for_each_entry([&](uint8_t byte, smart_address sa) {
            if (sa.is_pointer() && first[range_of(byte)])
               retain(sa.as_pointer());
         });
         return true;
      }
      bool recovery_walk(const sal::alloc_header*                    obj,
                         const std::function<void(sal::ptr_address)>& retain)
      {
         return recovery_walk_dense<bitset_r2_node>(obj, retain) ||
                recovery_walk_dense<bitset_r4_node>(obj, retain) ||
                recovery_walk_dense<full_radix_node>(obj, retain);
      }

      void recover_before_use(const std::filesystem::path& dir,
                              const sal::runtime_config&   cfg,
                              sal::recovery_mode           requested)
      {
         if (cfg.read_only)
         {
            // Recovery writes. Reading through whatever the dead owner left in the
            // unflushed control files could return garbage or fault.
            if (_prior_shutdown != sal::shutdown_state::clean)
               throw std::runtime_error(
                   "database was not shut down cleanly; recovery requires a read-write open "
                   "before it can be opened read-only: " + dir.generic_string());
            return;
         }
         const bool durable = cfg.sync_mode >= sal::sync_type::fsync;
         auto       mode    = requested;
         if (mode == sal::recovery_mode::none)
            mode = _prior_shutdown == sal::shutdown_state::clean
                       ? sal::recovery_mode::none
                       : _prior_shutdown == sal::shutdown_state::system_stopped && durable
                             ? sal::recovery_mode::power_loss
                             : sal::recovery_mode::app_crash;
         _alloc->set_recovery_hooks(
             {.walk = [this](const sal::alloc_header* obj,
                             const std::function<void(sal::ptr_address)>& retain)
              { return recovery_walk(obj, retain); },
              .rebuild_segment_metadata = true,
              .restart_threads          = false,
              .fail_on_unresolved       = true,
              .truncate_unsynced_tails  = durable});
         switch (mode)
         {
            case sal::recovery_mode::none:
               break;
            case sal::recovery_mode::deferred_cleanup:
               break;  // stale counts only ever over-count: a leak, never a dangling reference
            case sal::recovery_mode::app_crash:
               try
               {
                  _alloc->reset_reference_counts();
               }
               catch (const sal::recovery_error&)
               {
                  // The control blocks do not describe the roots, so the page cache
                  // was not intact after all. Rebuild from what was flushed instead.
                  if (requested != sal::recovery_mode::none)
                     throw;
                  mode = sal::recovery_mode::power_loss;
                  _alloc->recover_from_power_loss();
               }
               break;
            case sal::recovery_mode::power_loss:
               _alloc->recover_from_power_loss();
               break;
            case sal::recovery_mode::full_verify:
               _alloc->recover();
               break;
         }
         _recovered = mode;
         SAL_CRASH_POINT("open:recovered");
         // Durable before the first mutation: from here on, a stop without the matching
         // clean record below is detected by the next open.
         sal::shutdown_record::mark_open(dir, cfg.sync_mode);
         SAL_CRASH_POINT("open:record_written");
         _alloc->sync_control_state_on_close(cfg.sync_mode);
         _clean_close = {dir, cfg.sync_mode, true};
      }

      // Declared before _alloc so it is destroyed after it: the clean record is written
      // only once the allocator has flushed and closed every file.
      struct clean_close
      {
         std::filesystem::path dir;
         sal::sync_type        sync  = sal::sync_type::none;
         bool                  armed = false;
         clean_close()                              = default;
         clean_close(std::filesystem::path d, sal::sync_type s, bool a)
             : dir(std::move(d)), sync(s), armed(a)
         {
         }
         clean_close& operator=(clean_close&& other) noexcept
         {
            dir         = std::move(other.dir);
            sync        = other.sync;
            armed       = other.armed;
            other.armed = false;
            return *this;
         }
         ~clean_close()
         {
            if (armed)
            {
               SAL_CRASH_POINT("close:before_clean_record");
               sal::shutdown_record::mark_clean(dir, sync);
               SAL_CRASH_POINT("close:clean_record_written");
            }
         }
      };
      clean_close         _clean_close;
      sal::shutdown_state _prior_shutdown = sal::shutdown_state::clean;
      sal::recovery_mode  _recovered      = sal::recovery_mode::none;

      // The store. Owned by the engine a database opens; a writer context (see the
      // constructor taking writer_context) borrows it.
      radix_tree*                     _store = this;
      std::unique_ptr<sal::allocator> _owned_alloc;
      sal::allocator*                 _alloc = nullptr;
      // One writer at a time per root, held from the start of a transaction on that root
      // until it commits or aborts. Shared by every writer context of one store.
      struct root_writers
      {
         std::array<writer_gate, 1024> mutex;
      };
      std::shared_ptr<root_writers>   _root_writers;
      std::vector<uint32_t>           _held_roots;  // see claim_root
      sal::allocator_session_ptr      _ses;
      sal::sync_type                  _sync = sal::sync_type::none;  // commit durability (adb config)
      // The DIRECT-API working root — an OWNING smart_ptr (psitri tree_context's discipline);
      // empty = the valid empty tree. The database layer never reads it (SAL roots only).
      sal::smart_ptr<sal::alloc_header> _root;
      int64_t                         _old_value_size = -1;
      bool                            _removed        = false;  // remove(): key was found+deleted
      bool                            _emptied        = false;  // remove(): node drained → drop it
      // ── page model (the default) ──
      bool        _page_mode    = true;   // radix-indexed leaf pages; legacy knobs turn it off
      bool        _page_emptied = false;  // page op: the page drained → drop divider + page
      ptr_address _page_split   = sal::null_ptr_address;  // page op: full page → split + retry
      bool        _page_edge    = false;  // with _page_split: the key needs a page of its own
      static_assert(leaf_page::max_key_bytes == key_val_node::max_key_len);
      bool                            _wide_stems     = false;  // gate: off = byte radix
      bool                            _bucket_mode    = false;  // leaf=bucket, split to radix
      bool                            _adaptive       = false;  // gate: radix→bucket on deep-narrow
      bool                            _in_bucket      = false;  // re-entrancy: in a bucket subtree
      bool                            _c1_promote_on  = false;  // opt-in: narrow setlist→c1 below overflow
      // The value-type for the terminal the current upsert will create — STASHED on the tree (set
      // at upsert/upsert_subtree, read at make_leaf/make_value_into), so the insert descent never
      // threads it. vt_bytes for a normal upsert, vt_subtree for upsert_subtree.
      std::string                     _read_buffer;
      uint8_t                         _ins_value_type = inline_value::vt_bytes;
      cow_counters                    _cc;
   };

   // A read-only open frees nothing and runs no release thread, so its shared trees need no
   // counting at all: every object outlives the open.
   inline shared_tree::shared_tree(radix_tree* t, sal::ptr_address root) : _t(t), _root(root)
   {
      if (_root != sal::null_ptr_address && !_t->underlying_allocator().read_only())
         _t->underlying_allocator().retain(_root);
   }
   inline void shared_tree::drop() noexcept
   {
      if (_root != sal::null_ptr_address && !_t->underlying_allocator().read_only())
         _t->underlying_allocator().post_release(_root);
      _root = sal::null_ptr_address;
   }

   // Out-of-line: need the complete cursor type.
   inline bool radix_tree::is_subtree(key_view key)
   {
      cursor c(this);
      return c.find(key) && c.is_subtree();  // find() true = positioned on the key
   }
   inline radix_tree::read_handle radix_tree::get_subtree(key_view key)
   {
      cursor c(this);
      if (!c.find(key) || !c.is_subtree())
         return {};  // empty handle
      return c.subtree();
   }
   inline radix_tree::read_handle radix_tree::cursor::subtree() const
   {
      return read_handle(_t, _subtree);  // a handle, not a raw address (read_handle friends cursor)
   }
   inline radix_tree::frame radix_tree::transaction::sub_transaction()
   {
      // snapshot the primary handle and its open subtrees (copies of their roots → retained)
      return frame(this, _primary.save());
   }

   // ── psitri-compatible public entry: database → write_session → transaction/tree/... ──────────
   // These ARE heart's public API (not a facade): a database owns the engine (radix_tree); a
   // write_session is a handle onto it. psitri's tests compile against heart by selecting these
   // types (same names + signatures) — no adapter, no wrapper-translation layer.
   using cursor      = radix_tree::cursor;       // psitri-shaped namespace-level names
   using transaction = radix_tree::transaction;
   using tree        = radix_tree::tree;

   // psitri-shaped READ session: owns its own SAL session, handing out snapshot cursors pinned
   // through it. CREATE IT ON THE READER THREAD (SAL hands out one session per thread, and the
   // session is single-thread-owned). Reads run lock-free against the writer: the cursor
   // acquires the last COMMITTED root via SAL's root protocol (retain under the per-root
   // shared lock — a committing writer can't free it mid-acquire and never blocks readers
   // with its tree work), the writer's COW-to-root never mutates a published root, and every
   // cursor navigation takes THIS session's read_lock for page residency.
   class read_session
   {
      // Keeps the database open for as long as this session exists. First, so that it
      // is released last.
      std::shared_ptr<void>      _open;
      radix_tree*                _t;
      sal::allocator_session_ptr _ses;  // this reader thread's own SAL session

     public:
      void keep_open(std::shared_ptr<void> database) noexcept { _open = std::move(database); }
      explicit read_session(radix_tree* t) : _t(t), _ses(t->make_session())
      {
         // Dropping a pinned snapshot must never run the dead lineage's destroy cascade on the
         // reader thread — defer every last-ref release to the background release thread.
         _ses->set_defer_destroy(true);
      }
      radix_tree::cursor create_cursor(uint32_t idx = 0)
      {
         return _t->snapshot_cursor_at(idx, _ses);
      }
      /// Read any shared tree, published or not, through this reader's session.
      radix_tree::cursor create_cursor(const shared_tree& shared)
      {
         return _t->shared_cursor(shared, _ses);
      }
      /// Read a shared tree the caller keeps alive for as long as the cursor (and any
      /// cursor moved from it) exists. Takes no reference, so readers of one snapshot
      /// on many threads share no counter.
      radix_tree::cursor borrow_cursor(const shared_tree& shared)
      {
         return _t->borrowed_cursor(shared, _ses);
      }
      /// The published root as a reference that can leave this thread.
      shared_tree get_root(uint32_t idx = 0) { return create_cursor(idx).share(); }
   };

   class write_session
   {
      // Keeps the database open for as long as this session exists. First, so that it
      // is released last.
      std::shared_ptr<void>       _open;
      // This session's own writer context on the store. Trees, transactions and cursors
      // made through the session refer to it and must not outlive the session.
      std::unique_ptr<radix_tree> _own;
      radix_tree*                 _t = nullptr;

     public:
      void keep_open(std::shared_ptr<void> database) noexcept { _open = std::move(database); }
      explicit write_session(radix_tree* engine)
          : _own(std::make_unique<radix_tree>(radix_tree::writer_context{}, *engine)), _t(_own.get())
      {
      }
      radix_tree::transaction start_transaction(uint32_t idx = 0) { return _t->start_transaction(idx); }
      radix_tree::tree        create_temporary_tree() { return _t->create_temporary_tree(); }
      radix_tree::write_transaction start_write_transaction(radix_tree::tree base)
      {
         return _t->start_write_transaction(std::move(base));
      }
      radix_tree::tree get_root(uint32_t idx) { return _t->get_root(idx); }
      /// Hand a tree to other threads (consumes this writer's reference).
      shared_tree share(radix_tree::tree t) { return _t->share(std::move(t)); }
      /// Continue from a shared tree on the writer. Edits copy-on-write; the shared tree
      /// never changes.
      radix_tree::tree adopt(const shared_tree& shared) { return _t->adopt(shared); }
      void             set_root(uint32_t idx, radix_tree::tree t,
                                sal::sync_type sync = sal::sync_type::none)
      {
         _t->set_root(idx, std::move(t), sync);
      }
      uint64_t get_total_allocated_objects() const { return _t->get_total_allocated_objects(); }
      uint64_t get_pending_release_count() const { return _t->get_pending_release_count(); }
   };

   class database : public std::enable_shared_from_this<database>
   {
      radix_tree _engine;

     public:
      explicit database(const std::filesystem::path& dir,
                        bool                         wipe      = true,
                        bool                         read_only = false,
                        sal::sync_type               sync      = sal::sync_type::none)
          : _engine(dir, wipe, read_only, sync)
      {
      }
      database(const std::filesystem::path& dir,
               const sal::runtime_config&   cfg,
               sal::recovery_mode           recovery = sal::recovery_mode::none)
          : _engine(dir, /*wipe=*/false, cfg, recovery)
      {
      }
      // Unified open semantics (the kv_engine concept's construction clause —
      // identical shape and behavior across psitri/heart/part):
      //   open(path, open_mode, config) -> shared_ptr<database>
      // NOTE: open() no longer wipes; benches that want a fresh database
      // remove_all() themselves (they all already did).
      //
      // A read-write open first recovers a store whose previous owner did not close it
      // (see radix_tree's constructor); `recovery` forces a specific rebuild instead. A
      // read-only open of such a store throws: recovery has to write.
      static std::shared_ptr<database> open(const std::filesystem::path& dir,
                                            sal::open_mode mode = sal::open_mode::create_or_open,
                                            const sal::runtime_config& cfg = runtime_config{},
                                            sal::recovery_mode recovery = sal::recovery_mode::none)
      {
         const bool exists = std::filesystem::exists(dir / "roots");
         switch (mode)
         {
            case sal::open_mode::create_only:
               if (exists)
                  throw std::runtime_error("database already exists: " + dir.generic_string());
               break;
            case sal::open_mode::open_existing:
            case sal::open_mode::read_only:
               if (!exists)
                  throw std::runtime_error("database does not exist: " + dir.generic_string());
               break;
            case sal::open_mode::create_or_open:
               break;
         }
         if (!exists)
            std::filesystem::create_directories(dir);
         auto effective = cfg;
         effective.read_only = mode == sal::open_mode::read_only;
         return std::make_shared<database>(dir, effective, recovery);
      }
      // Open an EXISTING database (no wipe). read_only=true: no background
      // threads, data PROT_READ, control pages copy-on-write (file pristine),
      // flock LOCK_SH — refuses immediately if a writer holds the database.
      static std::shared_ptr<database> open_existing(const std::filesystem::path& dir,
                                                     bool read_only = false)
      {
         return open(dir, read_only ? sal::open_mode::read_only
                                    : sal::open_mode::open_existing);
      }
      std::shared_ptr<write_session> start_write_session()
      {
         auto session = std::make_shared<write_session>(&_engine);
         session->keep_open(weak_from_this().lock());
         return session;
      }
      // One per reader THREAD, created ON that thread (psitri model; see read_session).
      std::shared_ptr<read_session> start_read_session()
      {
         auto session = std::make_shared<read_session>(&_engine);
         session->keep_open(weak_from_this().lock());
         return session;
      }
      sal::seg_alloc_dump dump() { return _engine.dump(); }
      void sync(sal::sync_type st = sal::sync_type::msync_sync) { _engine.sync(st); }
      /// Whether the store at `dir` was closed by its last read-write owner, without
      /// opening it. False means a read-only open will refuse it and the next
      /// read-write open will recover it.
      static bool closed_cleanly(const std::filesystem::path& dir)
      {
         return sal::shutdown_record::read(dir) == sal::shutdown_state::clean;
      }
      /// How the previous read-write owner left this store, as found at open.
      sal::shutdown_state prior_shutdown() const noexcept { return _engine.prior_shutdown(); }
      /// The recovery this open ran (recovery_mode::none when it ran none).
      sal::recovery_mode recovery_performed() const noexcept { return _engine.recovery_performed(); }
      // [testing] the engine behind the public sessions: leak checks read object_count and
      // pending_releases of the same tree the sessions drive.
      radix_tree& engine_for_testing() { return _engine; }
      // The SAL allocator backing this database (dwal's btree_layer arena hook —
      // heart and psitri share sal::allocator, so the hook is engine-portable).
      sal::allocator& underlying_allocator() noexcept { return _engine.underlying_allocator(); }
      // Waits until the background RELEASE queue is empty, so that an object count taken
      // afterwards is exact. It does not wait for the compactor: the name is the one the
      // engines share.
      bool wait_for_compactor(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000))
      {
         auto end = std::chrono::steady_clock::now() + timeout;
         while (_engine.get_pending_release_count() > 0)
         {
            if (std::chrono::steady_clock::now() > end) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
         }
         return true;
      }
   };
}  // namespace sal_art_ref::radix

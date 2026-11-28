// start_ref::kv — part's NATIVE psitri-shaped public API (the same database →
// write_session → transaction → cursor structure psitri and heart::kv expose),
// so one templated test/bench body drives all three engines through identical
// calls. See heart/test/heart_kv.hpp for the canonical shape this mirrors and
// programs/kv_compat_test.cpp for the conformance driver.
//
// Engine: the hkv tree on the start_ref arena (hkv/tree.hpp + part_alloc),
// with the copy-on-write extension supplying snapshot semantics: commit
// retains the new root for the database's live-root slot and releases the old
// one. A reader's cursor protects the root it reads by publishing it in a
// reader slot (it takes no reference); an owned snapshot (get_root) holds a
// reference. Everything reachable from a protected root is immutable by the
// copy-on-write discipline (SPEC §7), so traversal needs no further counting.
// A value is a two-byte type tag written by the engine followed by the
// caller's bytes, stored in the leaf: in the line arena up to 64 lines, in
// the large-value arena up to 2 MiB, and in chunks beyond that.
#pragma once
#include <system_error>
#include <pthread.h>
#include <functional>
#include <adb/value_range.hpp>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <string_view>
#include <chrono>
#include <condition_variable>
#include <thread>

#include <start_ref/hkv/tree.hpp>
#include <start_ref/large_value.hpp>
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
      // Size of the reserve that lets a store at its ceiling remove keys (see
      // usage_info and docs/api/engines.md). 0: a sixty-fourth of the store,
      // between 128 KiB and 1 MiB. Raise it for a store whose transactions
      // remove many keys each at the ceiling; at most 4 MiB.
      uint64_t       reserve_bytes           = 0;
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

   // A root node at arena line 0 with a 0 kind-tag has raw()==0 (raw == (line<<4)
   // | tag). That aliases the "no root / unpublished" sentinel (0) shared by
   // _live_raw, arena.set_root, and the retired-token reclamation protocol — so
   // the very first/only key committed into a fresh tree (its single leaf sits at
   // line 0) would be published as 0 and read back as "empty". start_ver dodges
   // this incidentally by packing a non-zero version into the high bits; start_ref
   // is refcount-only with no version, so mark every published root with a high
   // bit. Readers recover the 4-byte raw via uint32_t() (which drops the marker)
   // and null-check via != 0, so only the write sites change. (See
   // [[sal_ptr_address_zero]]: address 0 is a VALID address, never a null sentinel.)
   // The handle is now 5 bytes (40-bit raw); the "published" marker must live
   // ABOVE that, so bit 40 (was bit 32, which a >=2^28-line index now occupies).
   inline constexpr uint64_t kHandleMask = (uint64_t(1) << (8 * detail::HANDLE_BYTES)) - 1;
   inline constexpr uint64_t kRootPubBit = uint64_t(1) << (8 * detail::HANDLE_BYTES);
   inline uint64_t           encode_root(root_ptr p) noexcept
   {
      return p.is_null() ? 0 : (p.raw() | kRootPubBit);
   }
   // Reconstruct the handle from a stored root word (durable slot or retired raw):
   // strip the publish bit / any high bits, keep the HANDLE_BYTES low bytes.
   inline root_ptr decode_root(uint64_t stored) noexcept
   {
      root_ptr       p;
      const uint64_t v = stored & kHandleMask;
      std::memcpy(p.b, &v, detail::HANDLE_BYTES);
      return p;
   }

   // Multi-root: arena slots [0, kMaxRoots) hold per-root committed raws (the
   // arena table is sized max_roots=16). start_ref and sal_art_ref share the
   // SAME root/snapshot model — a tree is an index + a held refcount;
   // start_ref's control block is refcount-only (address-stable arena, no
   // relocation indirection). docs/start_ref_multiroot_design.md.
   inline constexpr uint32_t kMaxRoots = 16;

   class database;
   class write_session;
   class transaction;
   class tree_handle;

   [[gnu::cold, noreturn]] inline void throw_root_references()
   {
      throw std::runtime_error(
          "start_ref::kv: too many references to one version of a root: write sessions and "
          "transactions can hold it up to 190 references, owned snapshots up to 254; release "
          "some snapshots");
   }

   // What database::usage() reports. All figures are bytes of arena storage,
   // which is file-backed mapped memory: `mapped` is what the store can have
   // resident and what its data files occupy.
   struct usage_info
   {
      uint64_t used_bytes   = 0;  // claimed 128-byte lines (live + retained + awaiting release)
      uint64_t mapped_bytes = 0;  // segments created so far; never shrinks (high-water mark)
      uint64_t limit_bytes  = 0;  // max_segments x 8 MiB: allocation beyond this throws
      uint64_t reserve_bytes = 0; // of the limit, kept for removes once the store cannot grow
      // Of the reserve (zero until the store is at its ceiling):
      // reserve_free_bytes: lines it holds now, whatever their arrangement.
      //   This is NOT what a remove can get: a remove needs runs of
      //   consecutive lines, and the next two say what runs there are.
      // reserve_longest_run_bytes: the longest single run it can hand out
      //   now. A remove that has to copy a longer node gets nothing from the
      //   reserve for it: it is refused unless the session's own segment, or
      //   failing that any segment of the store, has a free run that long.
      // reserve_long_runs: how many runs of nine lines (1,152 bytes) or more
      //   it can hand out now; nodes that long are routers of many branches,
      //   key prefixes over about a kilobyte, and leaves with long keys or
      //   values.
      // reserve_owed_bytes: lines it is waiting to get back. They return
      //   when the versions that removes replaced are released, which a
      //   reader still on such a version holds up for as long as it stays
      //   there, and only as runs at least as long as the ones lent; also
      //   lines not found free when the store was opened.
      uint64_t reserve_free_bytes        = 0;
      uint64_t reserve_owed_bytes        = 0;
      uint64_t reserve_longest_run_bytes = 0;
      uint32_t reserve_long_runs         = 0;
      // The companion arena that holds values too large for a 64-line run.
      uint64_t large_used_bytes   = 0;
      uint64_t large_mapped_bytes = 0;
      uint64_t large_limit_bytes  = 0;
   };

   // A thread with a stack of a stated size. Releasing a tree descends one
   // call level per level of the tree, and a tree is as deep as its longest
   // key allows (about one level per two key bytes), so the stack this needs
   // is bounded by the key length limit: a few hundred kilobytes in an
   // optimized build and well over a megabyte in an unoptimized one. A
   // platform's default for threads other than the first can be smaller than
   // that (512 KiB on macOS), so the engine does not rely on it.
   class sized_thread
   {
     public:
      sized_thread() = default;
      template <class F>
      sized_thread(std::size_t stack_bytes, F&& body)
      {
         auto fn = std::make_unique<std::function<void()>>(std::forward<F>(body));
         pthread_attr_t attr;
         pthread_attr_init(&attr);
         pthread_attr_setstacksize(&attr, stack_bytes);
         const int rc = pthread_create(&_thread, &attr, &run_, fn.get());
         pthread_attr_destroy(&attr);
         if (rc != 0)
            throw std::system_error(rc, std::generic_category(), "start_ref::kv: pthread_create");
         fn.release();
         _joinable = true;
      }
      sized_thread(const sized_thread&)            = delete;
      sized_thread& operator=(const sized_thread&) = delete;
      sized_thread& operator=(sized_thread&& o) noexcept
      {
         _thread     = o._thread;
         _joinable   = o._joinable;
         o._joinable = false;
         return *this;
      }
      bool joinable() const noexcept { return _joinable; }
      void join()
      {
         pthread_join(_thread, nullptr);
         _joinable = false;
      }

     private:
      static void* run_(void* arg)
      {
         std::unique_ptr<std::function<void()>> fn(static_cast<std::function<void()>*>(arg));
         (*fn)();
         return nullptr;
      }
      pthread_t _thread{};
      bool      _joinable = false;
   };

   // ── typed values (the LOCKED adb API) ─────────────────────────────────
   // EVERY stored slot is [u16 LE typenum][payload]. Typenum 0 = raw bytes
   // (the null-hook builtin); 65535 is internal chunked bytes. Registered
   // ids 1..65534 dispatch retain/release
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

   // Type numbers 0 and 65535 are the engine's own (plain bytes, and the
   // directory of a value stored in chunks). A type that claimed one of them
   // would be stored and then read back, released and recovered as if it were
   // that, so such a type is not storable at all: it cannot be registered,
   // written or read.
   template <class T>
   concept storable = requires(T t, const typename T::t_pod& p, void* ses) {
      { T::type_id } -> std::convertible_to<uint16_t>;
      requires T::type_id != 0 && T::type_id != 65535;
      requires std::is_trivially_copyable_v<typename T::t_pod>;
      { std::move(t).take_pod() } -> std::same_as<typename T::t_pod>;
      { T::make_from_pod(p, true) } -> std::same_as<T>;
      T::t_pod::retain(ses, p);
      T::t_pod::release(ses, p);
   };

   inline uint16_t storage_type(value_view s) noexcept
   {
      uint16_t t;
      std::memcpy(&t, s.data(), 2);
      return t;
   }
   inline uint16_t slot_type(value_view s) noexcept
   {
      const auto type = storage_type(s);
      return type == large_value_detail::chunked_type ? kRawType : type;
   }
   inline value_view slot_payload(value_view s) noexcept
   {
      return s.substr(2);
   }
   // TAGGED value-source: the engine's gather protocol (size()/write()) emits
   // [u16 tag][payload] DIRECTLY into the leaf's value region — upsert builds
   // nothing, one copy, heap-free at every size, no thread_local. This replaces
   // the old env_raw_ pre-built envelope.
   struct tagged_val
   {
      uint16_t   tag;
      value_view payload;
      size_t     size() const noexcept { return 2 + payload.size(); }
      void       write(char* d) const noexcept
      {
         std::memcpy(d, &tag, 2);
         std::memcpy(d + 2, payload.data(), payload.size());
      }
      // DEAD for raw_vals: a raw value never inline-packs and never lands in a
      // POD bucket, so these never RUN — they exist only to compile the VF
      // protocol's if-constexpr-discarded branches. NEVER return a real value.
      template <class A>
      void construct(A&, bval*) const
      {
         assert(false);
         __builtin_unreachable();
      }
      bval make() const
      {
         assert(false);
         __builtin_unreachable();
      }
   };
   // A tagged value that records whether its bytes were gathered into a leaf.
   struct tracked_val : tagged_val
   {
      bool* gathered;
      void  write(char* d) const noexcept
      {
         tagged_val::write(d);
         *gathered = true;
      }
   };
   struct chunked_val : tagged_val
   {
      large_value_detail::source chunks;
      chunked_val(large_session& session, uint32_t key_size, value_view value)
          : tagged_val{}, chunks(session, key_size, value) {}
      size_t size() const noexcept { return chunks.size(); }
      void write(char* dest) const noexcept { chunks.write(dest); }
   };

   // The key limit of the common API (docs/api/reference.md). The tree's own
   // length fields would take more; holding every engine to one limit keeps a
   // key that one accepts from being refused by another.
   inline constexpr size_t max_key_size = 1024;
   [[gnu::cold, noreturn]] inline void throw_key_too_long(size_t size)
   {
      throw std::invalid_argument("start_ref: key of " + std::to_string(size) +
                                  " bytes exceeds the " + std::to_string(max_key_size) +
                                  "-byte limit");
   }
   inline void check_key(key_view key)
   {
      if (key.size() > max_key_size) [[unlikely]]
         throw_key_too_long(key.size());
   }

   inline void upsert_bytes(engine_t& tree, key_view key, value_view value)
   {
      check_key(key);
      if (value.size() > large_value_detail::max_value_size)
         throw std::invalid_argument("start_ref: value exceeds UINT32_MAX");
      if (uint64_t(key.size()) + value.size() + large_value_detail::leaf_header_size + 2 >
          large_max_object_bytes)
      {
         chunked_val source(*tree.allocator_mut().lses, uint32_t(key.size()), value);
         tree.cow_insert(key, source);
      }
      else
         tree.cow_insert(key, tagged_val{kRawType, value});
   }

   template <class S>
   S slot_decode(value_view slot, const large_arena* store,
                 uint64_t offset = 0, uint64_t length = adb::read_all)
   {
      const uint16_t t = slot_type(slot);
      if constexpr (storable<S>)
      {
         if (offset || length != adb::read_all)
            throw std::invalid_argument("start_ref: byte ranges require a byte value");
         if (t != S::type_id || slot.size() - 2 != sizeof(typename S::t_pod))
            throw invalid_type{t, S::type_id};
         typename S::t_pod pod;
         std::memcpy(&pod, slot.data() + 2, sizeof(pod));
         return S::make_from_pod(pod, true);
      }
      else
      {
         if (t != kRawType) throw invalid_type{t, kRawType};
         if (storage_type(slot) == kRawType)
         {
            const auto p = adb::detail::value_slice(slot_payload(slot), offset, length);
            return S(p.data(), p.size());
         }
         return large_value_detail::read(store, slot, offset, length).template copy_or_take<S>();
      }
   }
   template <class F>
   void slot_visit(value_view slot, const large_arena* store, F&& fn,
                   uint64_t offset = 0, uint64_t length = adb::read_all)
   {
      if (slot_type(slot) != kRawType) throw invalid_type{slot_type(slot), kRawType};
      if (storage_type(slot) == kRawType)
         fn(adb::detail::value_slice(slot_payload(slot), offset, length));
      else
      {
         auto held = large_value_detail::read(store, slot, offset, length);
         fn(held.view());
      }
   }

   // any_value — pointer to the value SLOT in the tree, no copy; inside a
   // write transaction the transaction itself is the pin. It is invalidated
   // by the transaction's next write of ANY key: a write elsewhere can split,
   // rebuild or free the leaf this points into. as<T>() retains; bytes() on a
   // typed slot is invalid_type; raw_pod() is the expert/merger identity path.
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
      std::string_view bytes(uint64_t offset = 0, uint64_t length = adb::read_all) const
      {
         if (!_s) return {};
         if (type() != kRawType) throw invalid_type{type(), kRawType};
         _range = large_value_detail::read(_large, {_s, _n}, offset, length);
         return _range->view();
      }
      std::string_view raw_pod() const
      {
         if (_s && large_value_detail::chunked({_s, _n})) return bytes();
         return _s ? std::string_view{_s + 2, _n - 2} : std::string_view{};
      }
      explicit operator bool() const noexcept { return _s != nullptr; }

     private:
      friend class transaction;
      any_value(const char* s, size_t n, const large_arena* large) : _large(large), _s(s), _n(n) {}
      const large_arena* _large = nullptr;
      mutable std::optional<large_value_detail::range> _range;
      const char* _s = nullptr;
      size_t      _n = 0;  // the FULL slot (envelope included), in the tree
   };

   // Snapshot cursor: a pinned root + a non-owning view tree + an hkv iterator.
   // The dtor drops the pin; if that was the generation's last reference the
   // old version drains back to the arena.
   class cursor
   {
     public:
      // A moved cursor stays where it was. Its iterator points into its own
      // view of the tree, so the position travels as the key it stood on and
      // is found again in the new view.
      cursor(cursor&& o) noexcept : cursor(std::move(o), o.where_()) {}
      cursor(const cursor&) = delete;
      ~cursor()
      {
         if (!_a)
            return;
         const root_ptr r = _root;
         _view.adopt_root(root_ptr::null());  // the view owns nothing
         if (_hp)
            _hp->store(0, std::memory_order_seq_cst);  // readers only clear their slot:
         else if (!r.is_null() && !_borrowed)          // reclamation is the writer's job
            _view.cow_release(r);  // write-side pinned cursor (scan_all path)
         if (_session_slot_busy)
            *_session_slot_busy = false;
      }

      bool seek_begin() { _rend = false; _it = _view.begin(); return _it != _view.end(); }
      bool lower_bound(key_view k) { _rend = false; _it = _view.lower_bound(k); return _it != _view.end(); }
      // Mathematical upper bound: the LARGEST key v <= k (floor; the dual of
      // lower_bound = smallest v >= k). prev()/seek_last() are O(n) on start_ref.
      bool upper_bound(key_view k)
      {
         if (lower_bound(k))  // smallest v >= k
         {
            if (key() == k)   // k present → its own floor
               return true;
            return prev();    // landed strictly > k → predecessor = largest v < k
         }
         return seek_last();  // no key >= k → last key is the floor
      }
      void find(key_view k)
      {
         _rend = false;
         _it = _view.find(k);
      }
      bool next()
      {
         if (_rend)
            return seek_begin();  // psitri convention: next from rend = first
         ++_it;
         return _it != _view.end();
      }
      bool is_end() const { return !_rend && _it == _view.end(); }
      bool is_rend() const { return _rend; }
      bool is_subtree() const { return false; }  // start_ref has no subtree values
      // Backward positioning. The tree iterates forward only, so the key before
      // a position is found with lower_bound probes: a binary search over the
      // next byte at each level of the key, about nine probes per byte. The cost
      // follows key length, not the number of keys.
      bool seek_last()
      {
         _rend = false;
         _it   = _view.begin();
         if (_it == _view.end())
         {
            // Nothing to land on: the cursor is before the first key, as it
            // is after stepping back off the first key of a non-empty tree.
            _rend = true;
            return false;
         }
         std::string prefix;
         last_with_prefix_(prefix);
         return true;
      }
      bool prev()
      {
         if (_rend)
            return false;
         if (_it == _view.end())
            return seek_last();  // prev from end = last
         // The key before the current one shares some prefix P with it and
         // continues with a smaller byte, or is P itself. Try the longest P first.
         std::string prefix((*_it).first);
         while (!prefix.empty())
         {
            const int limit = static_cast<unsigned char>(prefix.back());
            prefix.pop_back();
            const int child = last_child_(prefix, limit);
            if (child >= 0)
            {
               prefix.push_back(static_cast<char>(child));
               last_with_prefix_(prefix);
               return true;
            }
            auto exact = _view.lower_bound(prefix);
            if (exact != _view.end() && std::string_view((*exact).first) == std::string_view(prefix))
            {
               _it = exact;
               return true;
            }
         }
         _rend = true;
         _it   = _view.end();
         return false;
      }

     private:
      // The largest byte d < limit for which some key starts with prefix+d, or -1.
      int last_child_(std::string& prefix, int limit)
      {
         // Is there a key starting with prefix whose next byte is in [lo, hi]?
         auto any = [&](int lo, int hi)
         {
            prefix.push_back(static_cast<char>(lo));
            auto it = _view.lower_bound(prefix);
            prefix.pop_back();
            if (it == _view.end())
               return false;
            const std::string_view k((*it).first);
            return k.size() > prefix.size() && k.substr(0, prefix.size()) == std::string_view(prefix) &&
                   static_cast<unsigned char>(k[prefix.size()]) <= hi;
         };
         if (limit <= 0 || !any(0, limit - 1))
            return -1;
         int lo = 0, hi = limit - 1;
         while (lo < hi)
         {
            const int mid = (lo + hi + 1) / 2;
            if (any(mid, hi))
               lo = mid;
            else
               hi = mid - 1;
         }
         return lo;
      }
      // Position on the largest key that starts with `prefix`; one exists.
      void last_with_prefix_(std::string& prefix)
      {
         for (;;)
         {
            const int child = last_child_(prefix, 256);
            if (child < 0)
               break;  // nothing longer: the prefix itself is the key
            prefix.push_back(static_cast<char>(child));
         }
         _it = _view.lower_bound(prefix);
      }

     public:
      key_view key() const { return (*_it).first; }

      template <class S = std::string>
      std::optional<S> value(uint64_t offset = 0, uint64_t length = adb::read_all) const
      {
         if (is_end() || is_rend()) return std::nullopt;
         return slot_decode<S>((*_it).second.v, large_(), offset, length);
      }
      template <class F> requires adb::detail::value_callback<F>
      bool get_value(F&& f, uint64_t offset = 0, uint64_t length = adb::read_all) const
      {
         if (is_end() || is_rend()) return false;
         slot_visit((*_it).second.v, large_(), std::forward<F>(f), offset, length);
         return true;
      }
      int64_t value_size() const
      {
         if (is_end() || is_rend()) return -1;
         const auto slot = (*_it).second.v;
         return large_value_detail::chunked(slot) ? large_value_detail::head_view(slot).size() : slot.size() - 2;
      }
      uint16_t value_type() const { return is_end() || is_rend() ? kRawType : slot_type((*_it).second.v); }
      template <class T> bool is_type() const
      {
         return !is_end() && !is_rend() && slot_type((*_it).second.v) == T::type_id;
      }
      template <class S = std::string>
      std::optional<S> get(key_view k, uint64_t offset = 0, uint64_t length = adb::read_all) const
      {
         value_view slot;
         if (!find_slot_(k, slot)) return std::nullopt;
         return slot_decode<S>(slot, large_(), offset, length);
      }
      template <class F> requires adb::detail::value_callback<F>
      bool get(key_view k, F&& f, uint64_t offset = 0, uint64_t length = adb::read_all) const
      {
         value_view slot;
         if (!find_slot_(k, slot)) return false;
         slot_visit(slot, large_(), std::forward<F>(f), offset, length);
         return true;
      }

     private:
      const large_arena* large_() const noexcept
      {
         return _a ? static_cast<const large_arena*>(_a->companion()) : nullptr;
      }
      bool find_slot_(key_view key, value_view& out) const
      {
         bval value;
         if (!_view.find(key, value)) return false;
         out = value.v;
         return true;
      }
      friend class database;
      friend class read_session;
      friend class tree;
      friend class transaction;
      friend class read_session;
      cursor(start_ref::arena*  a,
             root_ptr            root,
             std::atomic<uint64_t>* hp = nullptr,
             const value_hooks*  vh = nullptr)
          : _a(a), _hp(hp), _root(root), _view(part_alloc<bval>(a)), _it(_view.end())
      {
         _view.allocator_mut().vhooks = vh;  // write-side cursors can release
         _view.adopt_root(_root);
      }
      // Where a cursor stands, in a form that survives its view being moved.
      struct position
      {
         bool     on_key = false;
         bool     rend   = false;
         uint16_t size   = 0;
         char     key[max_key_size];
      };
      position where_() const
      {
         position p;
         p.rend = _rend;
         if (_a && !_rend && _it != _view.end())
         {
            const key_view k = (*_it).first;
            p.on_key         = true;
            p.size           = uint16_t(k.size());
            std::memcpy(p.key, k.data(), k.size());
         }
         return p;
      }
      cursor(cursor&& o, const position& at) noexcept
          : _own_slot(std::move(o._own_slot)),
            _session_slot_busy(o._session_slot_busy),
            _a(o._a),
            _hp(o._hp),
            _root(o._root),
            _view(std::move(o._view)),
            _it(_view.end()),
            _rend(at.rend),
            _borrowed(o._borrowed)
      {
         o._a  = nullptr;  // the moved-from cursor owns no pin
         o._hp = nullptr;
         o._session_slot_busy = nullptr;
         o._it   = o._view.end();
         o._rend = false;
         if (at.on_key)
            _it = _view.lower_bound(key_view(at.key, at.size));
      }

      // A reader's cursor protects its snapshot through one reader slot. The
      // first live cursor of a read session uses the session's slot; each
      // further one holds a slot of its own for as long as it lives.
      std::unique_ptr<start_ref::read_session> _own_slot;
      bool*                                    _session_slot_busy = nullptr;
      start_ref::arena*             _a;
      std::atomic<uint64_t>*   _hp;  // sal paradigm: the reader's published slot
      root_ptr                 _root;
      mutable engine_t         _view;
      engine_t::const_iterator _it;
      bool                     _rend = false;  // retreated past first
      // Reads through a reference somebody else holds for at least as long as
      // this cursor lives (a point read through an owned tree): nothing to
      // release, and no spare reference needed to read.
      bool                     _borrowed = false;
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
      // Keeps the database open for as long as this session exists. First, so
      // that it is released last.
      std::shared_ptr<database> _open;
      database*          _db;
      start_ref::read_session _rs;
      bool                    _slot_busy = false;  // a live cursor is using _rs's slot
   };

   class write_session;

   // A root's writer lock, held by a transaction from start to commit or
   // abort. It belongs to the transaction, not to a thread: a transaction may
   // be started on one thread and finished on another (with the hand-over
   // serialized by the caller), so the lock is a binary semaphore, which any
   // thread may release, and nothing about it depends on thread identity.
   class root_writer_lock
   {
     public:
      explicit root_writer_lock(std::binary_semaphore& gate) : _gate(&gate) { gate.acquire(); }
      root_writer_lock(root_writer_lock&& o) noexcept : _gate(o._gate) { o._gate = nullptr; }
      root_writer_lock(const root_writer_lock&)            = delete;
      root_writer_lock& operator=(const root_writer_lock&) = delete;
      ~root_writer_lock() { unlock(); }
      void unlock() noexcept
      {
         if (!_gate)
            return;
         _gate->release();
         _gate = nullptr;
      }

     private:
      std::binary_semaphore* _gate = nullptr;
   };

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
      // Optional range operation; includes uncommitted writes in [low, high).
      uint64_t remove_range_counted(key_view low, key_view high);
      // A cursor on the primary root as this transaction has changed it so far.
      // It is a snapshot: later writes of the transaction do not show in it.
      cursor snapshot_cursor();
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
          : _ws(o._ws), _tx(std::move(o._tx)), _large_tx(std::move(o._large_tx)), _roots(std::move(o._roots)), _held_roots(std::move(o._held_roots)), _done(o._done)
      {
         o._done = true;  // moved-from: nothing to roll back
      }
      transaction(const transaction&)            = delete;
      transaction& operator=(const transaction&) = delete;

      template <class F>
         requires adb::detail::value_callback<F>
      bool get(key_view k, F&& fn, uint64_t offset = 0, uint64_t length = adb::read_all) const;
      template <class S = std::string>
      std::optional<S> get(key_view k, uint64_t offset = 0, uint64_t length = adb::read_all) const;

     private:
      friend class write_session;
      friend class tree_handle;
      transaction(write_session* ws, uint32_t root_index, start_ref::transaction&& tx);
      struct root_state
      {
         uint32_t                     index;
         root_ptr                     start_root;   // committed root at txn start
         size_t                       start_count;
         root_writer_lock             lock;         // per-root writer lock (ordered)
      };
      engine_t&   tree_(uint32_t root_index);       // working tree for a held root
      write_session*          _ws;
      start_ref::transaction  _tx;
      start_ref::large_transaction _large_tx;
      std::vector<root_state> _roots;  // [0] = primary; ascending order
      // References of trees given to set_root, kept until the transaction
      // ends so that such a root is never uniquely held while written to.
      std::vector<root_ptr>   _held_roots;
      void                    release_held_roots_() noexcept;
      bool                    _done = false;
   };

   // A tree IS a snapshot (adb): an owned, FROZEN handle holding ONE refcount
   // on its root (under node-COW that ref keeps the version readable across
   // later commits). The single owned-tree type: get_root returns it, set_root
   // exchanges it, snapshot_cursor reads it. Copyable; default = empty.
   //
   // LIFETIME: a tree points at its database and gives its reference back to
   // it when destroyed, so every tree (and every cursor, read session and
   // write session) must be gone before the database is. Nothing keeps the
   // database alive for them. A build with assertions on counts the trees
   // that hold a root and aborts in ~database if one is left.
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
      tree(const tree& o);             // copy retains its own ref (out-of-line)
      tree& operator=(const tree& o)
      {
         if (this != &o)
         {
            release_();
            tree cp(o);  // retain
            _db = cp._db, _root = cp._root, _count = cp._count;
            cp._db = nullptr, cp._root = root_ptr::null();
         }
         return *this;
      }
      ~tree() { release_(); }
      cursor snapshot_cursor() const;  // out-of-line (needs database)
      cursor borrowed_cursor_() const; // for point reads: uses this tree's own reference
      template <class S = std::string>
      std::optional<S> get(key_view k, uint64_t offset = 0, uint64_t length = adb::read_all) const;
      explicit operator bool() const noexcept { return _db != nullptr; }
      template <class F>
         requires adb::detail::value_callback<F>
      bool get(key_view k, F&& fn, uint64_t offset = 0, uint64_t length = adb::read_all) const
      {
         return _db && borrowed_cursor_().get(k, std::forward<F>(fn), offset, length);
      }

     private:
      friend class database;
      friend class write_session;
      friend class transaction;
      friend class tree_handle;
      tree(database* db, root_ptr held, size_t count);  // out-of-line (counts the handle)
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
      std::optional<S> get(key_view k, uint64_t offset = 0, uint64_t length = adb::read_all) const;
      tree set_root(tree t);  // THE exchange: assign, return the displaced tree
      template <class F>
         requires adb::detail::value_callback<F>
      bool get(key_view k, F&& fn, uint64_t offset = 0, uint64_t length = adb::read_all) const;

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

      // A write session is not thread-safe. It may be used from any thread,
      // one at a time: the caller serializes. Builds with assertions enabled
      // stop the process when two threads are inside one session at once,
      // which otherwise corrupts the session's allocation state silently.
      // Re-entry from the same thread (a read callback that reads) is fine.
      struct use_scope
      {
#ifndef NDEBUG
         explicit use_scope(write_session* s) : _s(s)
         {
            const auto me = std::this_thread::get_id();
            if (s->_active.load(std::memory_order_relaxed) != me)
            {
               std::thread::id nobody{};
               if (!s->_active.compare_exchange_strong(nobody, me, std::memory_order_acquire))
               {
                  std::fputs("start_ref::kv: two threads are using one write session at the "
                             "same time; a session may move between threads only while "
                             "nothing else is using it\n",
                             stderr);
                  std::abort();
               }
            }
            ++s->_active_depth;
         }
         ~use_scope()
         {
            if (--_s->_active_depth == 0)
               _s->_active.store(std::thread::id{}, std::memory_order_release);
         }
         write_session* _s;
#else
         explicit use_scope(write_session*) noexcept {}
#endif
         use_scope(const use_scope&)            = delete;
         use_scope& operator=(const use_scope&) = delete;
      };
      // Present in every build so the layout does not depend on NDEBUG.
      std::atomic<std::thread::id> _active{};
      uint32_t                     _active_depth = 0;

      // working tree for root i: root 0 is THE hot member; roots >= 1 are lazy
      // (first open adopts the committed root and takes the working's ref).
      engine_t& tree_for(uint32_t i);
      // Caller holds this root's writer lock. Refresh a reused session.
      engine_t& refresh_tree(uint32_t i);

      // Keeps the database open for as long as this session exists. First, so
      // that it is released last.
      std::shared_ptr<database> _open;
      database*          _db;
      start_ref::session _ses;
      start_ref::large_session _lses;  // large-arena allocation context (writer)
      engine_t           _t;  // root 0's working tree (continues across txns)
      std::array<std::optional<engine_t>, kMaxRoots> _aux;  // roots >= 1, lazy
   };

   class database : public std::enable_shared_from_this<database>
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
               throw std::runtime_error(
                   "start_ref::kv: the Small Engine does not support read-only open "
                   "(open_mode::read_only) of " + dir.generic_string() +
                   "; open it read-write from the one process that owns it");
            case sal::open_mode::create_or_open:
               break;
         }
         return create(std::move(dir), cfg);
      }

      std::shared_ptr<write_session> start_write_session()
      {
         std::shared_ptr<write_session> session(new write_session(this));
         session->_open = weak_from_this().lock();
         return session;
      }
      std::shared_ptr<read_session> start_read_session()
      {
         std::shared_ptr<read_session> session(new read_session(this));
         session->_open = weak_from_this().lock();
         return session;
      }

      void sync(sal::sync_type = sal::sync_type::none) {}  // commits sync per cfg already

      /// Whether the store at `dir` was closed by its last owner, without
      /// opening it. False means that owner stopped without closing (killed,
      /// crashed, power lost) and the next open will rebuild the free map and
      /// reference counts from the published roots before returning. A
      /// directory with no store in it reports true: there is nothing to
      /// recover.
      /// A store that is open right now reports false: the mark is cleared at
      /// open and written back at close. Ask before opening, not during.
      static bool closed_cleanly(const std::filesystem::path& dir)
      {
         const auto lines  = start_ref::arena::read_clean_flag(dir);
         const auto grains = start_ref::large_arena::read_clean_flag(dir / "large");
         return lines.value_or(true) && grains.value_or(true);
      }
      /// Whether this open found the store not closed cleanly and rebuilt its
      /// free map and reference counts. Committed data is what the published
      /// roots held at the stop; see docs/api/operations.md for what each
      /// sync mode promises about that.
      bool recovered_on_open() const noexcept { return _recovered; }
      /// How many stored values of registered types (typed values) the
      /// recovery of this open found, counting a value shared by several
      /// roots once. Zero when the store was closed cleanly.
      ///
      /// Recovery rebuilds the engine's own counts only. It does not call a
      /// type's retain or release hooks: types are registered after open
      /// returns. So after an unclean stop the references that typed values
      /// hold on things outside the store are whatever the stopped process
      /// left: every stored value still has the reference it was stored
      /// with, and the values of the unfinished transaction and of versions
      /// that were waiting to be released keep references that nothing will
      /// ever release. An application whose types count such references
      /// has to reconcile them itself when this is not zero. Stores of plain
      /// bytes are not affected.
      uint64_t typed_values_found_by_recovery() const noexcept { return _recovered_typed; }

      ~database()
      {
         assert(_live_trees.load(std::memory_order_relaxed) == 0 &&
                "a tree (owned snapshot) outlived its database");
         // clean shutdown (SPEC §19): ring drained, releaser stopped and
         // JOINED before any member teardown; the rest runs foreground/exact
         flush_releases();
         _rel_stop.store(true, std::memory_order_release);
         _rel_cv.notify_one();
         if (_rel_thread.joinable())
            _rel_thread.join();
         START_REF_CRASH_POINT("close.after_releaser_stop");
         _arena.set_defer(nullptr, nullptr);  // remaining releases are inline
         _arena.set_reclaim(nullptr, nullptr);
         _large.set_reclaim(nullptr, nullptr);
         // No readers remain by contract: force-drain retired roots, then the
         // live root's slot reference.
         for (uint64_t raw : _retired)
         {
            const root_ptr p = decode_root(raw);
            engine_t v{part_alloc<bval>(&_arena)};
            v.allocator_mut().vhooks = &_vh;
            START_REF_CRASH_POINT("close.release_retired");
            v.cow_release(p);
         }
         START_REF_CRASH_POINT("close.before_arenas");
         // [persist/reopen] Do NOT release the published live roots on close.
         // They are the durable committed trees (arena.set_root persisted them);
         // a later reopen re-adopts them in the ctor. Releasing here would free
         // the on-disk tree — the old free-on-close (in-RAM-only) behavior that
         // made start_ref unable to survive an env close→reopen.
      }

      start_ref::arena&       store() noexcept { return _arena; }
      start_ref::large_arena& large_store() noexcept { return _large; }  // debug/tests
      // Used grains in the large arena (debug/tests: prove value_large fired and
      // that large leaves drain — a value_large leak shows up as a nonzero return
      // after every key is removed and the releaser has flushed).
      uint64_t large_used_grains() const noexcept
      {
         const uint32_t ns    = _large.num_segments();
         uint64_t       freeU = 0;
         for (uint32_t s = 0; s < ns; ++s)
            freeU += const_cast<start_ref::large_arena&>(_large).freeset_of(s).free_lines();
         return uint64_t(ns) * start_ref::lines_per_segment - freeU;
      }

      // Current storage use and the configured ceiling. Callable from any
      // thread; while writers run the figures are a moment-in-time estimate.
      // Call flush_releases() first for a figure that excludes nodes still
      // queued for the background releaser.
      usage_info usage() const noexcept
      {
         usage_info u;
         u.used_bytes         = _arena.used_bytes();
         u.mapped_bytes       = _arena.mapped_bytes();
         u.limit_bytes        = _arena.limit_bytes();
         u.reserve_bytes      = _arena.reserve_bytes();
         u.reserve_free_bytes = _arena.reserve_free_bytes();
         u.reserve_owed_bytes = _arena.reserve_owed_bytes();
         u.reserve_longest_run_bytes =
             uint64_t(_arena.reserve_longest_run()) * start_ref::line_bytes;
         u.reserve_long_runs = _arena.reserve_runs_at_least(9);
         u.large_used_bytes   = _large.used_bytes();
         u.large_mapped_bytes = _large.mapped_bytes();
         u.large_limit_bytes  = _large.limit_bytes();
         return u;
      }

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
      // It first releases every retired version that no reader holds any
      // more. That otherwise happens at the next commit, so after a writer's
      // last commit a version a reader let go of later would stay counted in
      // usage() for as long as nothing commits.
      void flush_releases() noexcept
      {
         {
            std::lock_guard g(_root_mx);
            drain_retired_();
         }
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
            retain_root_(r);  // the snapshot's ref
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

      void retain_root_(root_ptr root)
      {
         if (root.tag() == hkv::detail::K::value_large)
            _large.add_ref_or_throw(root.index());
         else
            _arena.add_ref_or_throw(root.index());
      }
      database(std::filesystem::path dir, runtime_config cfg)
          : _arena(checked_cfg_(dir, cfg)), _large(make_large_cfg_(dir, cfg))
      {
         // Link the large-value arena as _arena's companion BEFORE any tree is
         // built: every part_alloc copy (writer/view/releaser) resolves and frees
         // value_large leaves through a->companion().
         _arena.set_companion(&_large);
         for (auto& r : _live_root)
            r = root_ptr::null();
         for (auto& r : _live_raw)
            r.store(0, std::memory_order_relaxed);
         // [persist/reopen] Re-expose durable roots committed by a prior session
         // (mirrors start_ver's ctor): the arena reloaded the published root slots
         // from disk, so adopt them as the live boundary. Without this a reopen
         // (Silkworm opens the datadir twice at startup) sees an empty tree. A
         // slot of 0 means "never published" (zero-init mmap); a real publish
         // stores a non-zero raw or root_ptr::null()'s all-FF tag.
         for (uint32_t i = 0; i < kMaxRoots; ++i)
         {
            const uint64_t durable = _arena.get_root(i);
            if (durable == 0)
               continue;
            const root_ptr p = decode_root(durable);
            if (p.is_null())
               continue;
            _live_root[i] = p;
            _live_raw[i].store(durable, std::memory_order_seq_cst);
         }
         // The free map and reference counts are derived state that is only
         // settled by a clean close. If the last owner did not close, they
         // still describe its unfinished transaction, its retired versions
         // and its private reservations (and after a power loss, nothing in
         // particular), so rebuild them from the published roots.
         if (!_arena.was_clean() || !_large.was_clean())
         {
            recover_();
            _recovered = true;
         }
         // background releaser (SPEC §19): the writer pushes release-to-zero
         // subtree roots; this thread runs the cascades; a full ring makes the
         // writer release inline (graceful degradation to the old behavior).
         // OPTIONAL: without the thread the hook is never registered and the
         // queue is permanently "blocked" — everything runs synchronously.
         // A writer that is about to be told the store is full first gets
         // back what is already dead: versions no reader uses any more, and
         // whatever the releaser has queued.
         _arena.set_reclaim(&reclaim_cb_, this);
         _large.set_reclaim(&reclaim_cb_, this);
         if (cfg.background_release)
         {
            _arena.set_defer(&defer_cb_, this);
            _rel_thread = sized_thread(release_stack_bytes, [this] { rel_loop_(); });
         }
      }
      // Recovery marker: see kv_tree::recover_mark.
      struct recover_marker
      {
         start_ref::arena&       lines;
         start_ref::large_arena& grains;
         void check_line(uint32_t i) const { lines.check_index(i); }
         void check_large(uint32_t g) const { grains.check_index(g); }
         bool line(uint32_t i) { return lines.mark_run(i, 1); }
         void tail(uint32_t i, uint32_t n) { lines.claim_run(i, n); }
         bool large(uint32_t g, uint32_t units) { return grains.mark_run(g, units); }
         uint64_t typed = 0;  // values of a registered type the walk came across
         void leaf_value(std::string_view slot)
         {
            if (slot.size() >= 2 && slot_type(slot) != kRawType)
               ++typed;
            if (slot.size() < 2 || !large_value_detail::chunked(slot))
               return;
            // Each continuation is its own counted object in the large arena.
            const large_value_detail::head_view head(slot);
            for (size_t i = 0; i < head.entries().size(); ++i)
            {
               grains.check_index(head.entries()[i].grain);
               grains.mark_run(head.entries()[i].grain,
                               large_value_detail::units(head.length(i)));
            }
         }
         [[noreturn]] void unsupported()
         {
            throw std::runtime_error("recovery does not cover this node layout");
         }
      };
      // Rebuild both arenas' free maps and reference counts from the published
      // roots. Runs before any session exists. The walk recurses once per tree
      // level, so it gets the same stack the releaser does.
      void recover_()
      {
         std::string failure;
         sized_thread walker(release_stack_bytes, [&] {
            try
            {
               START_REF_CRASH_POINT("recover.begin");
               _arena.rebuild_begin();
               START_REF_CRASH_POINT("recover.between_resets");
               _large.rebuild_begin();
               recover_marker marker{_arena, _large};
               engine_t       view{part_alloc<bval>(&_arena)};
               for (uint32_t i = 0; i < kMaxRoots; ++i)
               {
                  view.recover_mark(_live_root[i], marker);  // the root slot's own reference
                  START_REF_CRASH_POINT("recover.after_root");
               }
               _arena.rebuild_end();
               START_REF_CRASH_POINT("recover.between_ends");
               _large.rebuild_end();
               _recovered_typed = marker.typed;
               START_REF_CRASH_POINT("recover.done");
            }
            catch (const std::exception& e)
            {
               failure = e.what();
            }
         });
         walker.join();
         if (!failure.empty())
            throw std::runtime_error(
                "start_ref::kv: the store at " + _arena.config().dir.generic_string() +
                " was not closed cleanly and its tree could not be walked (" + failure +
                "); its data and root slots are unchanged and it stays marked unclean, so "
                "the next open tries again (the free map and reference counts, which this "
                "rebuild resets, are derived from them)");
      }
      // Called by an allocation that found no room, on the writer's thread,
      // with no arena lock held. Releases retired versions that no reader
      // publishes any more (normally done at the next commit, which this
      // writer may never reach without the room), then waits for the
      // releaser's queue. True if either had anything to do.
      static bool reclaim_cb_(void* ctx, bool patient) noexcept
      {
         auto* db      = static_cast<database*>(ctx);
         bool  drained = false;
         {
            // A remove never waits here for a commit that is publishing. An
            // allocation about to be refused does: that commit may be about
            // to retire and release the very room it needs. (Nothing
            // allocates while holding this mutex, so the wait cannot be for
            // this thread.)
            std::unique_lock lock(db->_root_mx, std::defer_lock);
            if (patient)
               lock.lock();
            else
               (void)lock.try_lock();
            if (lock.owns_lock() && !db->_retired.empty())
            {
               const size_t before = db->_retired.size();
               db->drain_retired_();
               drained = db->_retired.size() != before;
            }
         }
         // Wait for the releaser to get through what was queued when we came
         // in, and no longer: other writers can keep adding to the queue for
         // ever, and this runs inside a writer's transaction.
         const uint32_t upto   = db->_rel_head.load(std::memory_order_acquire);
         const bool     queued = db->_rel_tail.load(std::memory_order_acquire) != upto;
         if (queued)
         {
            using clock   = std::chrono::steady_clock;
            uint32_t seen = db->_rel_tail.load(std::memory_order_acquire);
            auto     mark = clock::now();
            // Impatient: bounded in time (reclaim_max_wait). Patient: for as
            // long as the releaser keeps advancing, and reclaim_stall_limit
            // beyond its last step. The releaser may simply not be scheduled
            // (more runnable threads than processors), and what it holds is
            // room this writer is entitled to; only a releaser that has
            // stopped altogether ends the wait.
            const auto limit = patient ? clock::duration(reclaim_stall_limit)
                                       : clock::duration(reclaim_max_wait);
            for (uint32_t spins = 0;; ++spins)
            {
               const uint32_t tail = db->_rel_tail.load(std::memory_order_acquire);
               if (int32_t(tail - upto) >= 0)
                  break;
               const auto now = clock::now();
               if (patient && tail != seen)
                  seen = tail, mark = now;
               else if (now - mark >= limit)
                  break;
               if (patient && spins > 64)  // give the releaser a processor
                  std::this_thread::sleep_for(std::chrono::microseconds(50));
               else
                  std::this_thread::yield();
            }
         }
         return queued || drained;
      }
      // The longest one impatient call of the reclaim hook waits for the
      // releaser: a remove that the reserve cannot serve makes one.
      static constexpr std::chrono::microseconds reclaim_max_wait{500};
      // How long a patient call (an ordinary allocation about to be refused)
      // goes on waiting for a releaser that has not completed a single
      // queued release. Far beyond any scheduling delay: past it the
      // releaser is taken to be stuck (in a typed value's release hook, say)
      // and the allocation is refused after all.
      static constexpr std::chrono::seconds reclaim_stall_limit{30};
      static bool defer_cb_(void* ctx, uint64_t raw) noexcept
      {
         return static_cast<database*>(ctx)->rel_push_(raw);
      }
      // The release queue: many producers, one consumer. Any writer thread
      // can push (writers on different roots run at the same time, and each
      // hands over the subtrees its own operations and commits kill), so a
      // producer first claims a position by advancing the head with a
      // compare-exchange and only then fills its slot. The releaser treats a
      // claimed slot that is still zero as "not written yet" and waits for
      // it; it zeroes each slot before advancing the tail, so a position is
      // never claimed while its slot still holds an unconsumed entry.
      static constexpr uint64_t kRelFilled = uint64_t(1) << 63;  // a handle can itself be 0
      bool rel_push_(uint64_t raw) noexcept
      {
         uint32_t h = _rel_head.load(std::memory_order_relaxed);
         uint32_t t;
         do
         {
            t = _rel_tail.load(std::memory_order_acquire);
            if (h - t >= kRelRing)
               return false;  // full: the caller releases in the foreground
         } while (!_rel_head.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel,
                                                   std::memory_order_relaxed));
         _rel[h % kRelRing].store(raw | kRelFilled, std::memory_order_release);
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
            const uint64_t raw = _rel[t % kRelRing].load(std::memory_order_acquire);
            if (!(raw & kRelFilled))
            {
               std::this_thread::yield();  // claimed, about to be written
               continue;
            }
            const root_ptr p = decode_root(raw);
            START_REF_CRASH_POINT("releaser.before_release");
            v.cow_release(p);  // non-deferring allocator: full cascade here
            START_REF_CRASH_POINT("releaser.after_release");
            _rel[t % kRelRing].store(0, std::memory_order_release);
            _rel_tail.store(t + 1, std::memory_order_release);
         }
      }
      // Both arenas must fit the requested limit before either is opened: the
      // large-value arena is constructed second, and a refusal there would
      // come after the ordinary arena had been opened and marked in use.
      static start_ref::arena_config checked_cfg_(const std::filesystem::path& dir,
                                                  runtime_config               cfg)
      {
         // What this engine does not implement is refused, not ignored: a
         // caller that asked for page protection or checksums must not be
         // left believing it has them.
         if (cfg.sync_mode == sal::sync_type::mprotect)
            throw std::invalid_argument(
                "start_ref::kv: sync_mode::mprotect is not implemented by the Small Engine (it "
                "applies no page protection); open " + dir.generic_string() +
                " with none, msync_async, msync_sync, fsync or full");
         if (cfg.write_protect_on_commit)
            throw std::invalid_argument(
                "start_ref::kv: write_protect_on_commit is not implemented by the Small Engine; "
                "open " + dir.generic_string() + " without it");
         if (cfg.checksum_on_commit)
            throw std::invalid_argument(
                "start_ref::kv: checksum_on_commit is not implemented by the Small Engine (its "
                "nodes carry no checksums); open " + dir.generic_string() + " without it");
         const auto lines  = make_cfg_(dir, cfg);
         const auto grains = make_large_cfg_(dir, cfg);
         start_ref::arena::require_fits(lines.dir, lines.max_segments);
         start_ref::large_arena::require_fits(grains.dir, grains.max_segments);
         return lines;
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
         if (cfg.reserve_bytes)
            c.reserve_lines = uint32_t(std::min<uint64_t>(
                (cfg.reserve_bytes + start_ref::line_bytes - 1) / start_ref::line_bytes,
                start_ref::lines_per_segment / 2));
         return c;
      }
      // The large-value second arena lives in a `large/` subdirectory (its own
      // data/free/refs/meta.part), so it persists across reopen exactly like the
      // line arena. Direct 32-bit indices count 32 KiB grains, and each large
      // segment holds 2 GiB. Cap this separate reservation at 512 segments (1 TiB);
      // pages are touched as needed. A physical run is at most 2 MiB, while the
      // byte API spans runs through its embedded continuation directory.
      static start_ref::arena_config make_large_cfg_(std::filesystem::path dir, runtime_config cfg)
      {
         start_ref::arena_config c = make_cfg_(std::move(dir), cfg);
         c.dir /= "large";
         c.reserve_lines = 0;  // removes allocate nothing here
         c.max_segments = std::min<uint32_t>(c.max_segments, 512);
         return c;
      }

      // Retired-root release is DEFERRED to the writer: a root leaves the
      // queue only when no reader slot publishes it (sal's reader paradigm —
      // readers never touch refcounts or locks).
      void drain_retired_()  // caller holds _root_mx
      {
         // read once: this runs at every commit
         static const bool tm = getenv("PART_DRAIN_TIME") != nullptr;
         const auto t0 = tm ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
         for (size_t i = 0; i < _retired.size();)
         {
            if (_arena.root_published(_retired[i]))
            {
               ++i;
               continue;
            }
            const root_ptr p = decode_root(_retired[i]);
            engine_t v{part_alloc<bval>(&_arena, /*defer=*/true)};
            v.allocator_mut().vhooks = &_vh;
            START_REF_CRASH_POINT("retire.before_release");
            v.cow_release(p);  // big generations route to the releaser
            START_REF_CRASH_POINT("retire.after_release");
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
      std::atomic<uint64_t>     _rel[kRelRing] = {};  // full 40-bit handles
      std::atomic<uint32_t>     _rel_head{0}, _rel_tail{0};
      std::mutex                _rel_mx;
      std::condition_variable   _rel_cv;
      // Room for releasing a tree of the deepest keys in any build type.
      static constexpr std::size_t release_stack_bytes = 8u << 20;
      sized_thread              _rel_thread;
      std::atomic<bool>         _rel_stop{false};

      start_ref::arena      _arena;
      // Large-value second arena (32 KiB grain) in a `large/` subdirectory: same
      // COW/refcount/recovery machinery, max allocation 2 MiB. Linked to _arena as
      // its companion so every part_alloc (writer, view, releaser) reaches it.
      // value_large leaves live here; value_ptr leaves stay in _arena. Declared
      // AFTER _arena so it is constructed after (and destroyed before) it.
      start_ref::large_arena _large;
      std::mutex            _root_mx;                  // serializes commit publication
      // per-root writer locks (ordered); one permit each, released by
      // whichever thread finishes the transaction
      std::array<std::binary_semaphore, kMaxRoots> _root_locks = make_root_locks_();
      template <size_t... I>
      static std::array<std::binary_semaphore, kMaxRoots> make_root_locks_(std::index_sequence<I...>)
      {
         return {{(void(I), std::binary_semaphore{1})...}};
      }
      static std::array<std::binary_semaphore, kMaxRoots> make_root_locks_()
      {
         return make_root_locks_(std::make_index_sequence<kMaxRoots>{});
      }
      root_ptr              _live_root[kMaxRoots];     // committed root per slot
      std::atomic<uint64_t> _live_raw[kMaxRoots];      // raw per slot (reader publish-validate)
      std::vector<uint64_t> _retired;
      bool                  _recovered = false;  // this open rebuilt the derived state
      uint64_t              _recovered_typed = 0;  // typed values that rebuild walked over

      // ── typed-value dispatch (slot envelope -> registered hooks) ────────
      struct type_entry
      {
         void (*retain)(void*, const char*, size_t)  = nullptr;
         void (*release)(void*, const char*, size_t) = nullptr;
      };
      // One entry per type number, a megabyte in all: made when the first type
      // is registered, so a store that keeps only bytes never has it. The
      // releaser reads it from its own thread, hence the published pointer.
      // Trees that hold a root (see the lifetime note on class tree). Kept in
      // every build so the class has one layout; only assertions read it.
      std::atomic<int64_t> _live_trees{0};
      void tree_born_() noexcept { _live_trees.fetch_add(1, std::memory_order_relaxed); }
      void tree_gone_() noexcept { _live_trees.fetch_sub(1, std::memory_order_relaxed); }
      std::unique_ptr<type_entry[]>  _types_owner;
      std::atomic<type_entry*>       _types{nullptr};
      template <class T>
      void reg_type_()
      {
         static_assert(storable<T>);
         static_assert(T::type_id != kRawType && T::type_id != large_value_detail::chunked_type,
                       "type IDs 0 and 65535 are reserved for byte values");
         type_entry* types = _types.load(std::memory_order_acquire);
         if (!types)
         {
            _types_owner = std::make_unique<type_entry[]>(65536);
            types        = _types_owner.get();
         }
         types[T::type_id] = {&vh_retain_t_<T>, &vh_release_t_<T>};
         _types.store(types, std::memory_order_release);
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
         if (t == large_value_detail::chunked_type)
         {
            assert(!retain); // copies use the writer-aware mutable callback
            large_value_detail::release_head(_large, {b, n});
            return;
         }
         const type_entry* types = _types.load(std::memory_order_acquire);
         if (!types)
            return;  // no type was ever registered
         const auto& e = types[t];
         if (auto f = retain ? e.retain : e.release)
            f(this, b + 2, n - 2);
      }
      static void vh_copy_s_(char* b, size_t n, void* u, large_session* session)
      {
         if (large_value_detail::chunked({b, n}))
         {
            assert(session);
            large_value_detail::copy_head(*session, b, n);
         }
         else
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
       : _db(db), _ses(db->_arena.start_session()),
         _lses(db->_large.start_session()), _t(part_alloc<bval>(&_ses))
   {
      _t.allocator_mut().vhooks = &db->_vh;
      _t.allocator_mut().lses   = &_lses;  // large-value allocations on this writer
      // Session creation can overlap another writer's root publication.
      std::lock_guard lock(db->_root_mx);
      const root_ptr r = db->_live_root[0];
      if (!r.is_null())
      {
         // A refused reference must not be adopted: the session would later
         // give back a count it never took, and the version would be freed
         // while its other holders still read it.
         if (!_t.cow_retain(r))
            throw_root_references();
         _t.adopt_root(r, 0);
      }
   }

   inline read_session::read_session(database* db)
       : _db(db), _rs(db->_arena.start_read_session())
   {
   }

   inline cursor read_session::create_cursor(uint32_t root_index)
   {
      if (root_index >= kMaxRoots)
         throw std::out_of_range("root index out of range");
      // Every live cursor needs a slot to itself: sharing one would leave all
      // but the latest snapshot unprotected.
      std::unique_ptr<start_ref::read_session> own;
      if (_slot_busy)
         own = std::make_unique<start_ref::read_session>(_db->_arena.start_read_session());
      auto& slot = _db->_arena.reader_slot(own ? own->slot() : _rs.slot());
      for (;;)
      {
         const uint64_t r = _db->_live_raw[root_index].load(std::memory_order_seq_cst);
         slot.store(r, std::memory_order_seq_cst);  // publish FIRST...
         if (_db->_live_raw[root_index].load(std::memory_order_seq_cst) == r)
         {  // ...then validate: r was live after our publish, so the writer's
            // drain (which runs after replacing the root) must see our slot.
            root_ptr p = root_ptr::null();
            if (r)
               p = decode_root(r);
            cursor result(&_db->_arena, p, &slot);
            if (own)
               result._own_slot = std::move(own);
            else
            {
               _slot_busy                = true;
               result._session_slot_busy = &_slot_busy;
            }
            return result;
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
         _db->retain_root_(_root);
      return cursor(&_db->_arena, _root, nullptr, &_db->_vh);
   }
   inline cursor tree::borrowed_cursor_() const
   {
      cursor c(&_db->_arena, _root, nullptr, &_db->_vh);
      c._borrowed = true;
      return c;
   }
   inline void tree::release_()
   {
      if (!_db || _root.is_null())
         return;
      engine_t v{part_alloc<bval>(&_db->_arena)};
      v.allocator_mut().vhooks = &_db->_vh;
      v.cow_release(_root);  // drop THIS handle's ref (cascades only at zero)
      _db->tree_gone_();
      _root = root_ptr::null();
      _db   = nullptr;
   }
   // copy: each handle owns its OWN ref (warm legs share a snapshot)
   inline tree::tree(const tree& o) : _db(o._db), _root(o._root), _count(o._count)
   {
      if (_db && !_root.is_null())
      {
         _db->retain_root_(_root);
         _db->tree_born_();
      }
   }
   inline tree::tree(database* db, root_ptr held, size_t count)
       : _db(db), _root(held), _count(count)
   {
      if (_db && !_root.is_null())
         _db->tree_born_();
   }
   template <class S>
   inline std::optional<S> tree::get(key_view k, uint64_t offset, uint64_t length) const
   {
      if (!_db)
         return std::nullopt;
      return borrowed_cursor_().template get<S>(k, offset, length);
   }

   inline void transaction::upsert(key_view k, value_view v)
   {
      write_session::use_scope use(_ws);
      upsert_bytes(_ws->tree_for(_roots[0].index), k, v);
   }
   template <class T>
      requires storable<std::remove_cvref_t<T>>
   inline void transaction::upsert(key_view k, T&& v)
   {
      using U                     = std::remove_cvref_t<T>;
      write_session::use_scope use(_ws);
      check_key(k);  // before custody of the value moves
      const typename U::t_pod pod = std::move(v).take_pod();  // custody -> db
      {
         // [type_id][pod] gathered straight into the leaf. cow_insert copies the
         // pod synchronously (descends + materializes before returning) — the pod
         // local outlives the call, so a view over it is safe.
         //
         // The value's reference is the engine's from take_pod() on. A write
         // that fails after the leaf was built gives it back when that leaf
         // is torn down (the release hook); one that fails earlier, on the
         // way to the leaf, never built anything to tear down, so the
         // reference is given back here. Either way a refused upsert leaves
         // the caller owing nothing and owning nothing.
         bool gathered = false;
         try
         {
            _ws->tree_for(_roots[0].index)
                .cow_insert(k, tracked_val{{U::type_id,
                                            value_view(reinterpret_cast<const char*>(&pod),
                                                       sizeof(pod))},
                                           &gathered});
         }
         catch (...)
         {
            if (!gathered)
               U::t_pod::release(_ws->_db, pod);
            throw;
         }
      }
   }
   inline any_value transaction::get_any(key_view k) const
   {
      write_session::use_scope use(_ws);
      bval out;
      if (!_ws->tree_for(_roots[0].index).find(k, out))
         return {};
      return any_value{out.v.data(), out.v.size(), &_ws->_db->_large};  // slot view, txn-pinned
   }
   inline int transaction::remove(key_view k)
   {
      write_session::use_scope use(_ws);
      // A remove copies the path to its key and frees the old one at commit:
      // it may use the reserve, so that a full store can always be shrunk.
      const auto reserve = _ws->_ses.use_reserve();
      return _ws->tree_for(_roots[0].index).cow_remove(k) ? 0 : -1;
   }
   inline uint64_t transaction::remove_range_counted(key_view low, key_view high)
   {
      write_session::use_scope use(_ws);
      auto& t = tree_(_roots[0].index);
      uint64_t count = 0;
      auto it = t.lower_bound(low);
      while (it != t.end() && (*it).first < high)
      {
         // Mutation invalidates iterators and borrowed keys. Re-seek with an
         // owned key to bound temporary memory independently of range size.
         std::string key((*it).first);
         remove(key);
         ++count;
         it = t.lower_bound(key);
      }
      return count;
   }
   template <class F> requires adb::detail::value_callback<F>
   inline bool transaction::get(key_view k, F&& fn, uint64_t offset, uint64_t length) const
   {
      write_session::use_scope use(_ws);  // the callback's view borrows the session
      const auto found = get_any(k);
      if (!found) return false;
      slot_visit({found._s, found._n}, &_ws->_db->_large, std::forward<F>(fn), offset, length);
      return true;
   }
   template <class S>
   inline std::optional<S> transaction::get(key_view k, uint64_t offset, uint64_t length) const
   {
      const auto found = get_any(k);
      if (!found) return std::nullopt;
      return slot_decode<S>({found._s, found._n}, &_ws->_db->_large, offset, length);
   }

   inline void check_root_index(uint32_t root_index)
   {
      if (root_index >= kMaxRoots)
         throw std::out_of_range("root index out of range");
   }
   inline transaction::transaction(write_session* ws, uint32_t root_index,
                                   start_ref::transaction&& tx)
       : _ws(ws), _tx(std::move(tx)), _large_tx(ws->_lses.start_transaction())
   {
      root_writer_lock lock(ws->_db->_root_locks[root_index]);
      auto& t = ws->refresh_tree(root_index);
      _roots.push_back({root_index, t.root_handle(), t.size_hint(), std::move(lock)});
   }
   inline engine_t& transaction::tree_(uint32_t root_index) { return _ws->tree_for(root_index); }
   inline tree_handle transaction::primary() { return tree_handle(this, _roots[0].index); }
   inline tree_handle transaction::open_root(uint32_t root_index)
   {
      write_session::use_scope use(_ws);
      check_root_index(root_index);
      if (_roots.empty())
         throw std::logic_error("start_ref::kv: open_root on a finished transaction");
      // Ascending order is what keeps two multi-root transactions from each
      // waiting for a root the other holds.
      if (root_index <= _roots.back().index)
         throw std::logic_error(
             "start_ref::kv: open_root(" + std::to_string(root_index) +
             ") after root " + std::to_string(_roots.back().index) +
             ": a transaction opens each additional root once, in ascending index order");
      root_writer_lock lock(_ws->_db->_root_locks[root_index]);
      auto& t = _ws->refresh_tree(root_index);
      _roots.push_back({root_index, t.root_handle(), t.size_hint(), std::move(lock)});
      return tree_handle(this, root_index);
   }
   inline void transaction::abort() noexcept
   {
      if (_done)
         return;
      write_session::use_scope use(_ws);
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
            // The working re-takes the start ref. If the version is at its
            // reference ceiling it cannot; then it holds nothing, and the next
            // transaction adopts the committed root afresh (or reports the
            // ceiling where an error can be reported).
            if (t.cow_retain(it->start_root))
               t.adopt_root(it->start_root, it->start_count);
            else
               t.adopt_root(root_ptr::null(), 0);
            START_REF_CRASH_POINT("abort.before_release");
            t.cow_release(diverged);  // drop the working's diverged ref
            START_REF_CRASH_POINT("abort.after_release");
         }
         it->lock.unlock();  // reverse order
      }
      _roots.clear();
      release_held_roots_();
      _tx.close();  // the session may start its next transaction
      _large_tx.close();
   }
   inline void transaction::commit()
   {
      if (_done)
         throw std::logic_error("start_ref::kv: commit on a finished transaction");
      write_session::use_scope use(_ws);
      database* db = _ws->_db;
      // Take the root slots' references to the new roots BEFORE anything is
      // published, so that a refusal (a version at its reference ceiling)
      // leaves every root as it was and this transaction still open: the
      // caller aborts it, or frees references and commits again. A root this
      // transaction holds but did not change needs nothing: the slot already
      // references that very node (it cannot have been freed and its line
      // reused while the slot holds it, and nobody else can publish this root
      // while we hold its writer lock).
      const size_t count = _roots.size();
      size_t       taken = 0;
      for (; taken < count; ++taken)
      {
         const root_state& r  = _roots[taken];
         engine_t&         t  = _ws->tree_for(r.index);
         const root_ptr    nr = t.root_handle();
         if (nr.is_null() || nr.raw() == db->_live_root[r.index].raw())
            continue;
         if (!t.cow_retain(nr))
            break;
      }
      if (taken != count)
      {
         for (size_t i = 0; i < taken; ++i)
         {
            engine_t&      t  = _ws->tree_for(_roots[i].index);
            const root_ptr nr = t.root_handle();
            if (!nr.is_null() && nr.raw() != db->_live_root[_roots[i].index].raw())
               t.cow_release(nr);  // the working tree still holds its own
         }
         throw_root_references();
      }
      // Room for the versions this commit retires, so that nothing between
      // the first root's publication and the last can throw. Other commits
      // can use it up before the lock below is taken; then it is topped up
      // there, still before anything is published.
      const auto give_back = [&]() noexcept
      {
         for (size_t i = 0; i < count; ++i)
         {
            engine_t&      t  = _ws->tree_for(_roots[i].index);
            const root_ptr nr = t.root_handle();
            if (!nr.is_null() && nr.raw() != db->_live_root[_roots[i].index].raw())
               t.cow_release(nr);
         }
      };
      START_REF_CRASH_POINT("commit.before_flush");
      _large_tx.commit(); // continuation/head data must precede root publication
      _tx.commit();  // data msync + floor advance (layer 1)
      START_REF_CRASH_POINT("commit.after_flush");
      {
         std::lock_guard g(db->_root_mx);
         try
         {
            db->_retired.reserve(db->_retired.size() + count);
         }
         catch (...)
         {
            give_back();  // nothing published: still open, as for a refusal
            throw;
         }
         _done = true;
         for (auto& r : _roots)
         {
            const root_ptr nr  = _ws->tree_for(r.index).root_handle();
            const root_ptr old = db->_live_root[r.index];
            if (nr.raw() == old.raw())
               continue;  // unchanged: nothing to publish, nothing to retire
            db->_live_root[r.index] = nr;
            db->_live_raw[r.index].store(encode_root(nr), std::memory_order_seq_cst);
            db->_arena.set_root(r.index, encode_root(nr));
            if (!old.is_null())
               db->_retired.push_back(encode_root(old));
            START_REF_CRASH_POINT("commit.after_root");
         }
         START_REF_CRASH_POINT("commit.before_drain");
         db->drain_retired_();
         START_REF_CRASH_POINT("commit.after_drain");
      }
      START_REF_CRASH_POINT("commit.published");
      for (auto it = _roots.rbegin(); it != _roots.rend(); ++it)
         it->lock.unlock();
      _roots.clear();
      release_held_roots_();
      _tx.close();  // the session may start its next transaction
      _large_tx.close();
   }

   inline cursor transaction::snapshot_cursor()
   {
      write_session::use_scope use(_ws);
      database*      db   = _ws->_db;
      engine_t&      tree = _ws->tree_for(_roots[0].index);
      const root_ptr root = tree.root_handle();
      // The cursor holds its own reference, so the transaction's next write
      // copies instead of changing what the cursor reads.
      if (!root.is_null() && !tree.cow_retain(root))
         throw_root_references();
      return cursor(&db->_arena, root, nullptr, &db->_vh);
   }
   inline transaction write_session::start_transaction(uint32_t root_index)
   {
      use_scope use(this);
      check_root_index(root_index);
      if (_ses.in_transaction())
         throw std::logic_error(
             "start_ref::kv: this write session already has a transaction open; a session "
             "runs one transaction at a time (use open_root for more roots, or another "
             "session)");
      return transaction(this, root_index, _ses.start_transaction(0));
   }
   inline tree write_session::get_root(uint32_t root_index)
   {
      check_root_index(root_index);
      return _db->get_root_owned(root_index);
   }
   inline engine_t& write_session::refresh_tree(uint32_t i)
   {
      auto& t = tree_for(i);
      const root_ptr live = _db->_live_root[i];  // protected by root i's writer lock
      const root_ptr old = t.root_handle();
      if (live.raw() != old.raw())
      {
         if (!live.is_null() && !t.cow_retain(live))
            throw_root_references();
         t.adopt_root(live, 0);
         START_REF_CRASH_POINT("refresh.before_release");
         t.cow_release(old);
      }
      return t;
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
         _aux[i]->allocator_mut().lses   = &_lses;  // large-value allocations
         const root_ptr r = _db->_live_root[i];
         if (!r.is_null())
         {
            if (!_aux[i]->cow_retain(r))  // the working takes its ref
            {
               _aux[i].reset();
               throw_root_references();
            }
            _aux[i]->adopt_root(r, 0);
         }
      }
      return *_aux[i];
   }

   inline void tree_handle::upsert(key_view k, value_view v)
   {
      write_session::use_scope use(_tx->_ws);
      upsert_bytes(_tx->tree_(_ri), k, v);
   }
   inline int tree_handle::remove(key_view k)
   {
      write_session::use_scope use(_tx->_ws);
      const auto reserve = _tx->_ws->_ses.use_reserve();
      return _tx->tree_(_ri).cow_remove(k) ? 0 : -1;
   }
   template <class S>
   inline std::optional<S> tree_handle::get(key_view k, uint64_t offset, uint64_t length) const
   {
      bval out;
      if (!_tx->tree_(_ri).find(k, out))
         return std::nullopt;
      return slot_decode<S>(out.v, &_tx->_ws->_db->_large, offset, length);
   }
   template <class F>
      requires adb::detail::value_callback<F>
   inline bool tree_handle::get(key_view k, F&& fn, uint64_t offset, uint64_t length) const
   {
      bval out;
      if (!_tx->tree_(_ri).find(k, out))
         return false;
      slot_visit(out.v, &_tx->_ws->_db->_large, std::forward<F>(fn), offset, length);
      return true;
   }
   // Assign a tree to this root and return the one it displaces.
   //
   // The incoming tree keeps the reference it arrived with, held by the
   // transaction until it ends, and the working tree takes a second one. The
   // root is therefore never uniquely held while this transaction can write
   // to it, so the first write copies instead of writing in place. That is
   // required, not caution: an old version can share storage with newer ones
   // in a way its root's own count does not show (a wide router that a later
   // version copied has its tail lines retained by the copy), and a root
   // handed in with one reference used to be taken for private all the way
   // down. Trees the working tree builds itself are private for real.
   inline tree tree_handle::set_root(tree t)
   {
      write_session::use_scope use(_tx->_ws);
      engine_t&      w       = _tx->tree_(_ri);
      const root_ptr old     = w.root_handle();
      const size_t   old_cnt = w.size_hint();
      if (!t._root.is_null())
      {
         _tx->_held_roots.reserve(_tx->_held_roots.size() + 1);  // before any change
         if (!w.cow_retain(t._root))
            throw_root_references();
         _tx->_held_roots.push_back(t._root);  // the incoming tree's reference
      }
      START_REF_CRASH_POINT("set_root.before_adopt");
      w.adopt_root(t._root, t._count);  // repoint working to the incoming root
      START_REF_CRASH_POINT("set_root.after_adopt");
      if (t._db && !t._root.is_null())
         t._db->tree_gone_();  // the handle no longer holds a root: the transaction does
      t._db   = nullptr;
      t._root = root_ptr::null();
      return tree(_tx->_ws->_db, old, old_cnt);  // owned tree takes the working's old ref
   }
   inline void transaction::release_held_roots_() noexcept
   {
      if (_held_roots.empty())
         return;
      database* db = _ws->_db;
      engine_t  v{part_alloc<bval>(&db->_arena)};
      v.allocator_mut().vhooks = &db->_vh;
      for (const root_ptr r : _held_roots)
      {
         START_REF_CRASH_POINT("set_root.release_held");
         v.cow_release(r);
      }
      _held_roots.clear();
   }

   // The engine as a kv_engine aggregate (delta_art's W/B type param).
   struct engine
   {
      static constexpr uint64_t max_value_size = large_value_detail::max_value_size;
      static constexpr uint32_t root_count     = kMaxRoots;
      static constexpr size_t   max_key_size   = start_ref::kv::max_key_size;
      using database      = start_ref::kv::database;
      using write_session = start_ref::kv::write_session;
      using read_session  = start_ref::kv::read_session;
      using transaction   = start_ref::kv::transaction;
      using cursor        = start_ref::kv::cursor;
      using config        = start_ref::kv::runtime_config;
      /// Thrown by a write that needs more storage than config::max_segments
      /// allows. Abort the transaction; the database remains usable.
      using storage_full  = start_ref::arena_full;
      using usage         = start_ref::kv::usage_info;
   };
}  // namespace start_ref::kv

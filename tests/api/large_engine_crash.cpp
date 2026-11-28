// Large Engine: death of the process at chosen points (gofractally/psitri#44).
//
// The failure model is that the PROCESS can stop at any instruction, in any
// thread (killed, aborted), while the operating system keeps running and keeps
// every byte the process wrote to its shared mappings. A root publication (a
// transaction commit, or a session-level set_root) is acknowledged when the
// call returns. Power loss and operating-system failure are not defended
// against and are not tested here. The store runs in sync_mode::none.
//
// This file is compiled with SAL_CRASH_POINTS, which turns the
// SAL_CRASH_POINT("area:step") markers in the allocator and the engine into
// calls of a hook; the test binary compiles the allocator's sources itself so
// that every translation unit agrees. A forked child opens the store, checks
// it against a model and audits every reference count, then runs writers on
// several roots with readers holding snapshots, and kills itself with SIGKILL
// the n-th time a chosen marker is passed (or the parent kills it at a random
// moment). The next child does the same on the same store: it recovers,
// verifies, audits, and carries on writing.
//
// Environment (all optional):
//   LGE_CRASH_SEED       seed of the campaign (default 1)
//   LGE_CRASH_ROUNDS     deaths attempted per crash point (default 3)
//   LGE_CRASH_RANDOM     random-time kills (default 60)
//   LGE_CRASH_TXNS       transactions per writer per life (default 60; 120 for random kills)
//   LGE_CRASH_POINT      only this crash point (default: all)
//   LGE_CRASH_VERBOSE    1: one line per life
//   LGE_CRASH_DB_MB      max_database_size of the stores, in MiB (default 1024)
#include <psitri/engines/big.hpp>

#include <catch2/catch_test_macros.hpp>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef SAL_CRASH_POINTS
#error "this file must be compiled with SAL_CRASH_POINTS"
#endif

namespace
{
   using engine   = psitri::big_engine;
   using database = psitri::database_for<engine>;
   namespace fs   = std::filesystem;

   // Every SAL_CRASH_POINT name in the sources, collected by CMake.
   const char* const all_points[] = {
#include "large_engine_crash_points.inc"
   };

   struct temporary_directory
   {
      fs::path path;
      temporary_directory()
      {
         std::string pattern = (fs::temp_directory_path() / "psitri-large-crash-XXXXXX").string();
         auto*       result  = ::mkdtemp(pattern.data());
         if (!result)
            throw std::runtime_error("mkdtemp failed");
         path = result;
      }
      ~temporary_directory()
      {
         std::error_code ignored;
         fs::remove_all(path, ignored);
      }
   };

   uint64_t env_u64(const char* name, uint64_t fallback)
   {
      const char* v = std::getenv(name);
      return v && *v ? std::strtoull(v, nullptr, 10) : fallback;
   }
   std::string env_str(const char* name)
   {
      const char* v = std::getenv(name);
      return v ? v : "";
   }

   uint64_t mix(uint64_t a, uint64_t b, uint64_t c = 0)
   {
      uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0xBF58476D1CE4E5B9ull) * 0x94D049BB133111EBull ^
                   (c + 0x2545F4914F6CDD1Dull) * 0xD6E8FEB86659FD93ull;
      x ^= x >> 31;
      x *= 0x9E3779B97F4A7C15ull;
      return x ^ (x >> 29);
   }
   uint64_t hash_bytes(std::string_view s)
   {
      uint64_t h = 0x243F6A8885A308D3ull ^ s.size();
      size_t   i = 0;
      for (; i + 8 <= s.size(); i += 8)
      {
         uint64_t w;
         std::memcpy(&w, s.data() + i, 8);
         h = (h ^ w) * 0x9E3779B97F4A7C15ull;
         h ^= h >> 32;
      }
      uint64_t tail = 0;
      std::memcpy(&tail, s.data() + i, s.size() - i);
      h = (h ^ tail) * 0xD6E8FEB86659FD93ull;
      return h ^ (h >> 29);
   }

   // The node's configuration (psibase: read cache off, nothing pinned, never synced), on
   // a store capped small.
   engine::config store_config()
   {
      engine::config cfg;
      cfg.sync_mode                = psitri::sync_mode::none;
      cfg.enable_read_cache        = false;
      cfg.max_pinned_cache_size_mb = 0;
      cfg.max_database_size        = int64_t(env_u64("LGE_CRASH_DB_MB", 1024)) << 20;
      return cfg;
   }
   constexpr uint32_t segment_limit_for_writers = 24;  // of 32: writers wait for the compactor

   // ── messages from a child to the parent ─────────────────────────────────
   enum msg_type : uint32_t
   {
      m_opened = 1,    // a: recovery performed, b: prior shutdown state
      m_root,          // root, a: version found and verified
      m_stash,         // root (data root), a: version of its stash
      m_audit,         // a: objects, b: segments with data
      m_ready,         // the workload is about to start
      m_ack,           // root (data root or stash), a: version; sent AFTER the call returned
      m_dying,         // text: crash point, a: pass
      m_count,         // text: crash point, a: passes in this life
      m_fail,          // text: what
      m_done,          // the workload finished (a clean close follows unless armed)
      m_closed,        // the store was closed
      m_note,          // text: informational
   };
   struct message
   {
      uint32_t type = 0;
      uint32_t root = 0;
      uint64_t a    = 0;
      uint64_t b    = 0;
      char     text[104]{};
   };
   static_assert(sizeof(message) == 128);

   int g_out = -1;  // the child's end of the pipe
   void send(msg_type type, uint32_t root = 0, uint64_t a = 0, uint64_t b = 0,
             std::string_view text = {})
   {
      message m;
      m.type = type;
      m.root = root;
      m.a    = a;
      m.b    = b;
      std::memcpy(m.text, text.data(), std::min(text.size(), sizeof(m.text) - 1));
      if (::write(g_out, &m, sizeof(m)) != ssize_t(sizeof(m)))  // atomic: smaller than PIPE_BUF
         ::_exit(90);
   }
   [[noreturn]] void child_fail(const std::string& what)
   {
      std::fprintf(stderr, "large-engine crash child (pid %d): %s\n", int(::getpid()), what.c_str());
      send(m_fail, 0, 0, 0, what);
      ::_exit(2);
   }

   // ── the crash hook ──────────────────────────────────────────────────────
   namespace crash
   {
      char                  point[64] = "";
      std::atomic<uint64_t> countdown{0};
      std::atomic<bool>     armed{false};
      std::atomic<bool>     counting{false};
      // Called with the armed point's pass instead of dying (the tests that make the next
      // store of an operation the library offers no marker inside, then die).
      std::atomic<bool (*)()> special{nullptr};

      struct counter
      {
         std::atomic<const char*> name{nullptr};
         std::atomic<uint64_t>    passes{0};
      };
      std::array<counter, 512> counters;

      void count(const char* name)
      {
         for (size_t i = (reinterpret_cast<uintptr_t>(name) >> 3) % counters.size();;
              i        = (i + 1) % counters.size())
         {
            const char* cur = counters[i].name.load(std::memory_order_relaxed);
            if (cur == nullptr &&
                !counters[i].name.compare_exchange_strong(cur, name, std::memory_order_relaxed))
               cur = counters[i].name.load(std::memory_order_relaxed);
            if (cur == nullptr || cur == name)
            {
               counters[i].passes.fetch_add(1, std::memory_order_relaxed);
               return;
            }
         }
      }
      [[noreturn]] void die(const char* name, uint64_t pass)
      {
         send(m_dying, 0, pass, 0, name);
         // SIGKILL: no handlers, no destructors, no flushing; the other threads are
         // wherever they are.
         ::kill(::getpid(), SIGKILL);
         for (;;)
            ::pause();
      }
      void hook(const char* name)
      {
         if (counting.load(std::memory_order_relaxed))
            count(name);
         if (!armed.load(std::memory_order_relaxed) || std::strcmp(name, point) != 0)
            return;
         if (auto fn = special.load(std::memory_order_relaxed))
         {
            if (fn())
               die(name, 0);
            return;
         }
         if (countdown.fetch_sub(1, std::memory_order_relaxed) == 1)
            die(name, 0);
      }
      void install() { sal::testing::crash_hook.store(&hook, std::memory_order_release); }
      void arm(const std::string& name, uint64_t at)
      {
         std::snprintf(point, sizeof(point), "%s", name.c_str());
         countdown.store(at, std::memory_order_relaxed);
         armed.store(!name.empty(), std::memory_order_release);
      }
      std::map<std::string, uint64_t> counts()
      {
         std::map<std::string, uint64_t> out;
         for (auto& c : counters)
            if (const char* n = c.name.load())
               out[n] += c.passes.load();
         return out;
      }
   }  // namespace crash

   // ── the model ───────────────────────────────────────────────────────────
   // Each data root has its own sequence of publications. Publication k of root r is a pure
   // function of (seed, r, k) and of the root's contents before it, so a child that finds the
   // root at version k continues with k + 1, and the parent can say what any version holds.
   using value_ptr = std::shared_ptr<const std::string>;
   using submap    = std::map<std::string, std::string>;
   struct entry
   {
      value_ptr                     value;  // or
      std::shared_ptr<const submap> sub;    // a subtree
   };
   using model_map = std::map<std::string, entry>;

   constexpr uint32_t data_roots   = 8;
   constexpr uint32_t stash_base   = 8;  // root 8 + r: a snapshot of data root r
   constexpr uint32_t pair_a       = 4, pair_b = 5;
   constexpr uint32_t keep_root    = 6;  // session-level set_root of a tree the session keeps
   constexpr uint32_t replace_root = 7;  // a tree built outside, swapped in by a transaction
   constexpr uint64_t stash_every  = 16, stash_phase = 5;
   constexpr uint64_t revert_every = 37, revert_phase = 11;
   constexpr const char* seq_key   = "~m/seq";
   constexpr const char* sum_key   = "~m/sum";

   bool stash_scheduled(uint64_t k) { return k % stash_every == stash_phase; }
   bool reverts(uint32_t root, uint64_t k)
   {
      return root != pair_a && root != pair_b && k > stash_phase && k % revert_every == revert_phase;
   }
   bool has_subtrees(uint32_t root) { return root == 1 || root == replace_root || root == pair_b; }

   struct root_state
   {
      model_map model;
      uint64_t  version = 0;
      uint64_t  sum     = 0;  // xor of entry hashes of the data entries
      // The model at the latest version at which a stash is scheduled, and the one before:
      // a death between a commit and the stash that follows it leaves the older stash.
      model_map stash, prev_stash;
      uint64_t  stash_version = 0, prev_stash_version = 0;
      uint64_t  acked = 0, stash_acked = 0;  // parent: acknowledged publications
   };
   using campaign_state = std::array<root_state, data_roots>;

   uint64_t entry_hash(const std::string& key, const entry& e)
   {
      // A subtree counts as a marker only: readers check this sum without descending.
      return mix(hash_bytes(key), e.sub ? 0x5B7EEull : hash_bytes(*e.value), key.size());
   }

   std::string key_of(uint32_t i)
   {
      char num[16];
      std::snprintf(num, sizeof(num), "%05u", i);
      if (i % 9 == 0)
      {
         // long keys with long shared prefixes: fragments, external keys
         std::string k = "L/";
         k.append(100 + (i % 4) * 190, char('p' + i % 3));
         if (i % 27 == 0)
            k.append(260, 'z');
         return k + num;
      }
      const uint32_t h = uint32_t(i * 2654435761u) >> 16;  // two scrambled bytes: every fan-out
      std::string    k = "k/";
      k.push_back(char(h >> 8));
      k.push_back(char(h));
      return k + num;
   }
   value_ptr make_value(uint32_t root, uint64_t k, uint32_t op, size_t size)
   {
      auto           v        = std::make_shared<std::string>(size, char('a' + (k + op) % 26));
      const uint64_t stamp[2] = {k, (uint64_t(root) << 32) | op};
      std::memcpy(v->data(), stamp, std::min<size_t>(size, sizeof(stamp)));
      for (size_t at = 4096; at + 8 <= size; at += 65536)  // something all along a big value
         std::memcpy(v->data() + at, &at, 8);
      return v;
   }

   // LGE_CRASH_SHORT_SUBTREE_KEYS=1 keeps subtrees at short keys (to run a campaign past
   // the defect the long ones found; see the subtree test at the end of this file).
   bool long_subtree_keys()
   {
      static const bool on = env_u64("LGE_CRASH_SHORT_SUBTREE_KEYS", 0) == 0;
      return on;
   }

   struct profile
   {
      uint32_t keyspace = 900;
      uint32_t max_ops  = 20;
      uint32_t removes  = 55;  // percent of the remove operations drawn that are carried out
   };
   // Root 3 grows wide: thousands of keys, so its routers reach the dense kinds whose
   // versions share branch sets through counters.
   profile profile_of(uint32_t root)
   {
      return root == 3 ? profile{20000, 90, 25} : profile{};
   }

   // The operations of one phase of one publication, on the store through `h` (null: on the
   // model only) and on the model `m`. `fixed` holds keys this publication must not touch
   // again (a subtree opened for editing is written back at commit).
   template <class H, class WS>
   void run_ops(H* h, WS* ws, model_map& m, uint64_t& sum, std::mt19937_64& rng, uint32_t root,
                uint64_t k, uint32_t ops, uint32_t first_op, std::set<std::string>& fixed,
                bool may_open, const profile& p)
   {
      auto put = [&](const std::string& key, entry e) {
         auto it = m.find(key);
         if (it != m.end())
            sum ^= entry_hash(key, it->second);
         sum ^= entry_hash(key, e);
         m[key] = std::move(e);
      };
      auto drop = [&](model_map::iterator it) {
         sum ^= entry_hash(it->first, it->second);
         return m.erase(it);
      };
      auto is_book  = [](const std::string& key) { return key[0] == '~'; };
      auto data_key = [&](uint64_t r) -> model_map::iterator {
         auto it = m.lower_bound(key_of(uint32_t(r % p.keyspace)));
         if (it == m.end() || is_book(it->first))
            it = m.begin();
         if (it == m.end() || is_book(it->first))
            return m.end();
         return it;
      };
      auto build_subtree = [&](const submap& s) {
         auto wt = ws->start_write_transaction(ws->create_temporary_tree());
         for (auto& [sk, sv] : s)
            wt.upsert(sk, sv);
         return wt.get_tree();
      };
      for (uint32_t op = first_op; op < first_op + ops; ++op)
      {
         const uint32_t pick = uint32_t(rng() % 100);
         const uint64_t r    = rng();
         if (pick < 52)  // insert or replace
         {
            size_t size;
            if (pick < 30 || (root == 3 && pick < 50))
               size = r % 97;  // inline in the page
            else if (pick < 44)
               size = 120 + r % 5000;  // an external record
            else if (pick < 48 || root != 0)
               size = 9000 + r % 150000;
            else if (pick < 51)
               size = 300000 + r % 1700000;  // split in two when it meets the end of a segment
            else
               size = (2u << 20) + 200000 + r % (2u << 20);  // above the split threshold
            std::string key = size > (2u << 20) ? std::string("k/BIG")
                                                : key_of(uint32_t((r >> 20) % p.keyspace));
            if (fixed.count(key))
               continue;
            auto v = make_value(root, k, op, size);
            if (h)
               h->upsert(key, std::string_view(*v));
            put(key, entry{std::move(v), nullptr});
         }
         else if (pick < 60)  // replace an existing value: same size, then another size
         {
            auto it = data_key(r);
            if (it == m.end() || it->second.sub || fixed.count(it->first))
               continue;
            const std::string key  = it->first;
            const size_t      size = pick < 56 ? it->second.value->size()
                                               : (it->second.value->size() * 3 + 5) % 7000;
            auto v = make_value(root, k, op, size);
            if (h)
               h->upsert(key, std::string_view(*v));
            put(key, entry{std::move(v), nullptr});
         }
         else if (pick < 89 && (r >> 40) % 100 >= p.removes)
            continue;  // (what keeps a root at its size)
         else if (pick < 84)  // remove
         {
            auto it = data_key(r);
            if (it == m.end() || fixed.count(it->first))
               continue;
            if (h && h->remove(it->first) != 0)
               throw std::runtime_error("model: a key the model holds is not in the store");
            drop(it);
         }
         else if (pick < 89)  // a short range
         {
            auto it = data_key(r);
            if (it == m.end() || !fixed.empty())
               continue;
            const std::string low = it->first;
            uint32_t          n   = 0;
            auto              end = it;
            while (end != m.end() && n < 6 && !is_book(end->first))
               ++end, ++n;
            const std::string high = end == m.end() || is_book(end->first) ? "~" : end->first;
            if (h && h->remove_range_counted(low, high) != n)
               throw std::runtime_error("model: range remove count differs");
            while (it != end)
               it = drop(it);
         }
         else if (!has_subtrees(root))
            continue;
         else if (pick < 94)  // a new subtree, sometimes at a long key
         {
            char name[24];
            std::snprintf(name, sizeof(name), "s/%03u", uint32_t(r % 40));
            std::string key = name;
            if (r % 5 == 0 && long_subtree_keys())
               key.append(150 + r % 200, 'q');  // more than a page record's suffix can hold
            if (fixed.count(key))
               continue;
            auto           s = std::make_shared<submap>();
            const uint32_t n = 1 + uint32_t((r >> 8) % 12);
            for (uint32_t j = 0; j < n; ++j)
               (*s)["sub/" + std::to_string((r >> 16) % 50 + j)] =
                   std::string(size_t((r >> 24) % 300), char('A' + j));
            if (h)
               h->upsert_subtree(key, build_subtree(*s));
            put(key, entry{nullptr, std::move(s)});
         }
         else  // edit a subtree
         {
            auto it = m.lower_bound("s/");
            if (it == m.end() || !it->second.sub)
               continue;
            for (uint32_t skip = uint32_t(r % 7); skip && std::next(it) != m.end() &&
                                                  std::next(it)->second.sub;
                 --skip)
               ++it;
            const std::string key     = it->first;
            const bool        open_it = may_open && r % 2 == 0;  // edit it in place
            if (!open_it && fixed.count(key))
               continue;
            auto              s  = std::make_shared<submap>(*it->second.sub);
            const std::string sk = "sub/" + std::to_string((r >> 16) % 60);
            const std::string sv(size_t((r >> 24) % 900), 'e');
            (*s)[sk] = sv;
            if (h)
            {
               if (!open_it)
                  h->upsert_subtree(key, build_subtree(*s));
               else if constexpr (requires { h->open_subtree(key); })
                  h->open_subtree(key).upsert(sk, sv);  // written back when the handle is flushed
               else
                  throw std::logic_error("this handle cannot open a subtree");
            }
            if (open_it)
               fixed.insert(key);
            put(key, entry{nullptr, std::move(s)});
         }
      }
   }

   // Whether handles of root r can open a subtree in place (a transaction or a root handle
   // of one): decides the same thing in the parent, which has no handle.
   bool opens_subtrees(uint32_t root) { return root == 1 || root == pair_b; }

   // Publication k of `root`: the operations on the store through `h` and on the model.
   template <class H, class WS>
   void apply_publication(H* h, WS* ws, root_state& rs, uint64_t seed, uint32_t root, uint64_t k,
                          const profile&)
   {
      const profile p = profile_of(root);
      if (reverts(root, k))  // the caller has put the stash in place of the root
      {
         rs.model = rs.stash;
         rs.sum   = 0;
         for (auto& [key, e] : rs.model)
            if (key[0] != '~')
               rs.sum ^= entry_hash(key, e);
      }
      std::mt19937_64       rng(mix(seed, root, k));
      std::set<std::string> fixed;
      const uint32_t        ops     = 1 + uint32_t(rng() % p.max_ops);
      const uint32_t        first   = ops / 2;
      const bool            savept  = rng() % 3 == 0;
      const bool            discard = rng() % 2 == 0;
      const bool            opens   = opens_subtrees(root);
      // (in a model-only replay `h` is null and run_ops takes the same decisions)
      run_ops(h, ws, rs.model, rs.sum, rng, root, k, first, 0, fixed, opens, p);
      if (savept)
      {
         // A savepoint: the operations inside are kept or thrown away as a whole.
         model_map scratch;
         uint64_t  scratch_sum = rs.sum;
         model_map& target     = discard ? (scratch = rs.model) : rs.model;
         uint64_t&  tsum       = discard ? scratch_sum : rs.sum;
         bool       done       = false;
         if (h)
         {
            if constexpr (requires { h->sub_transaction(); })
            {
               auto frame = h->sub_transaction();
               run_ops(h, ws, target, tsum, rng, root, k, 4, 1000, fixed, false, p);
               if (discard)
                  frame.abort();
               else
                  frame.commit();
               done = true;
            }
         }
         if (!done)  // no savepoints on this kind of handle: thrown-away work is not done
            run_ops(discard ? static_cast<H*>(nullptr) : h, ws, target, tsum, rng, root, k, 4, 1000,
                    fixed, false, p);
      }
      run_ops(h, ws, rs.model, rs.sum, rng, root, k, ops - first, 2000, fixed, false, p);

      auto book = [&](const char* key, uint64_t v) {
         auto s = std::make_shared<std::string>(reinterpret_cast<const char*>(&v), 8);
         if (h)
            h->upsert(key, std::string_view(*s));
         rs.model[key] = entry{std::move(s), nullptr};
      };
      book(seq_key, k);
      book(sum_key, rs.sum);
      rs.version = k;
      if (stash_scheduled(k))
      {
         rs.prev_stash         = std::move(rs.stash);
         rs.prev_stash_version = rs.stash_version;
         rs.stash              = rs.model;
         rs.stash_version      = k;
      }
   }
   struct no_handle
   {
      // the shape run_ops needs of a handle, for the parent's model-only replay
      void     upsert(std::string_view, std::string_view) {}
      int      remove(std::string_view) { return 0; }
      uint64_t remove_range_counted(std::string_view, std::string_view) { return 0; }
      struct sub
      {
         void upsert(std::string_view, std::string_view) {}
      };
      sub& open_subtree(std::string_view)
      {
         static sub s;
         return s;
      }
      template <class T>
      void upsert_subtree(std::string_view, T&&) {}
   };
   struct no_session
   {
      struct wt_t
      {
         void upsert(std::string_view, std::string_view) {}
         int  get_tree() { return 0; }
      };
      int  create_temporary_tree() { return 0; }
      wt_t start_write_transaction(int) { return {}; }
   };
   void replay(root_state& rs, uint64_t seed, uint32_t root, uint64_t to, const profile& p)
   {
      while (rs.version < to)
         apply_publication(static_cast<no_handle*>(nullptr), static_cast<no_session*>(nullptr), rs,
                           seed, root, rs.version + 1, p);
   }

   // ── reading the store ───────────────────────────────────────────────────
   template <class Cursor>
   submap scan_subtree(Cursor& c)
   {
      submap out;
      auto   sub = c.subtree().snapshot_cursor();
      for (bool ok = sub.seek_begin(); ok; ok = sub.next())
      {
         auto v = sub.template value<std::string>();
         if (!v)
            throw std::runtime_error("a subtree key without a readable value");
         out.emplace(std::string(sub.key()), std::move(*v));
      }
      return out;
   }
   // The cursor's position against the model entry it should be on.
   template <class Cursor>
   std::string differs(Cursor& c, const model_map::value_type& want)
   {
      if (c.key() != want.first)
         return "key differs: found " + std::to_string(c.key().size()) + " bytes '" +
                std::string(c.key().substr(0, 24)) + "', model '" + want.first.substr(0, 24) + "'";
      if (want.second.sub)
      {
         if (!c.is_subtree())
            return "model has a subtree at '" + want.first.substr(0, 24) + "', store a value";
         if (scan_subtree(c) != *want.second.sub)
            return "subtree at '" + want.first.substr(0, 24) + "' differs";
         return {};
      }
      if (c.is_subtree())
         return "store has a subtree at '" + want.first.substr(0, 24) + "', model a value";
      auto v = c.template value<std::string>();
      if (!v)
         return "no readable value at '" + want.first.substr(0, 24) + "'";
      if (*v != *want.second.value)
         return "value at '" + want.first.substr(0, 24) + "' differs (" + std::to_string(v->size()) +
                " bytes, model " + std::to_string(want.second.value->size()) + ")";
      return {};
   }
   // Forward scan, reverse scan and point lookups of one tree against a model.
   template <class Cursor>
   std::string compare_tree(Cursor& c, const model_map& m, uint64_t seed)
   {
      auto it = m.begin();
      for (bool ok = c.seek_begin(); ok; ok = c.next(), ++it)
      {
         if (it == m.end())
            return "forward scan: the store has keys the model does not ('" +
                   std::string(c.key().substr(0, 24)) + "')";
         if (auto d = differs(c, *it); !d.empty())
            return "forward scan: " + d;
      }
      if (it != m.end())
         return "forward scan: the model has keys the store does not ('" + it->first.substr(0, 24) +
                "'), " + std::to_string(std::distance(it, m.end())) + " missing";
      auto rit = m.rbegin();
      for (bool ok = c.seek_last(); ok; ok = c.prev(), ++rit)
      {
         if (rit == m.rend())
            return "reverse scan: the store has keys the model does not";
         if (auto d = differs(c, *rit); !d.empty())
            return "reverse scan: " + d;
      }
      if (rit != m.rend())
         return "reverse scan: the model has keys the store does not";
      for (auto& [key, e] : m)
      {
         if (e.sub)
         {
            if (!c.find(key) || !c.is_subtree())
               return "lookup: no subtree at '" + key.substr(0, 24) + "'";
            continue;
         }
         auto v = c.template get<std::string>(key);
         if (!v || *v != *e.value)
            return "lookup of '" + key.substr(0, 24) + "' differs";
      }
      std::mt19937_64 rng(seed);
      for (int i = 0; i < 64; ++i)
      {
         std::string key = key_of(uint32_t(rng() % 4000)) + (i % 2 ? "x" : "");
         if (!m.count(key) && c.template get<std::string>(key))
            return "lookup: the store has '" + key.substr(0, 24) + "', the model does not";
      }
      return {};
   }
   template <class Cursor>
   uint64_t stored_u64(Cursor& c, const char* key)
   {
      auto v = c.template get<std::string>(key);
      if (!v)
         return 0;
      if (v->size() != 8)
         throw std::runtime_error(std::string("bookkeeping key ") + key + " is not 8 bytes");
      uint64_t out;
      std::memcpy(&out, v->data(), 8);
      return out;
   }

   // ── the audit ───────────────────────────────────────────────────────────
   // Every control block's count against the references found by walking all root slots; no
   // reachable object freed or outside its segment's allocated part; nothing allocated that
   // no root reaches; the shared branch-set counters; the segments' free-space accounting.
   struct audit_result
   {
      uint64_t    objects = 0, data_segments = 0, bytes = 0, shared_counters = 0;
      std::string error;
   };
   template <class N>
   bool walk_dense(sal::allocator& alloc, const sal::alloc_header* obj,
                   std::unordered_map<uint32_t, uint32_t>& counters,
                   const std::function<void(sal::ptr_address)>& ref)
   {
      using namespace sal_art_ref;
      if (obj->type() != N::type_id)
         return false;
      const N* n = reinterpret_cast<const N*>(obj);
      if (!n->rc_has_ctrs())
         return false;
      if (const auto tv = n->term_value(); tv.is_pointer())
         ref(tv.as_pointer());
      bool first[range_count];
      for (int r = 0; r < range_count; ++r)
         first[r] = counters[*n->rc_ctr(r)]++ == 0;
      n->for_each_entry([&](uint8_t byte, smart_address sa) {
         if (sa.is_pointer() && first[range_of(byte)])
            ref(sa.as_pointer());
      });
      (void)alloc;
      return true;
   }
   // expect_no_leak false: the store was opened with deferred cleanup, which keeps the dead
   // owner's counts. Then a count may be higher than the references found, never lower, and
   // nothing is said about leaks or segment space.
   audit_result audit(database& db, bool expect_no_leak)
   {
      using namespace sal_art_ref;
      using namespace sal_art_ref::radix;
      audit_result out;
      auto&        alloc = db.underlying_allocator();
      db.wait_for_compactor(std::chrono::milliseconds(20000));
      alloc.stop_background_threads();
      struct restart
      {
         sal::allocator& a;
         ~restart() { a.start_background_threads(); }
      } restart_threads{alloc};

      std::unordered_map<uint32_t, uint32_t> expected, counters;
      std::vector<sal::ptr_address>          todo;
      std::string                            error;
      const std::function<void(sal::ptr_address)> ref = [&](sal::ptr_address a) {
         if (a == sal::null_ptr_address)
            return;
         if (expected[*a]++ == 0)
            todo.push_back(a);
      };
      auto& roots = alloc.root_objects();
      for (uint32_t i = 0; i < roots.size(); ++i)
         ref(roots[i].load(std::memory_order_relaxed).root);
      const auto                                 dump = alloc.dump();
      std::vector<uint64_t>                      live(dump.segments.size(), 0);
      const uint32_t usable = sal::segment_size - sal::mapped_memory::segment_footer_size;
      while (!todo.empty() && error.empty())
      {
         const auto a = todo.back();
         todo.pop_back();
         auto [obj, loc] = alloc.resolve(a);
         if (!obj)
         {
            error = "reachable address " + std::to_string(*a) +
                    " does not resolve to an object (freed, zero count, or another object there)";
            break;
         }
         const uint32_t seg = *loc.segment();
         if (seg >= dump.segments.size())
         {
            error = "reachable object " + std::to_string(*a) + " lies in segment " +
                    std::to_string(seg) + ", beyond the store";
            break;
         }
         const auto& info = dump.segments[seg];
         if (info.is_free)
            error = "reachable object " + std::to_string(*a) + " lies in FREE segment " +
                    std::to_string(seg);
         else if (loc.segment_offset() + obj->size() > info.alloc_pos || obj->size() == 0 ||
                  obj->size() > usable)
            error = "reachable object " + std::to_string(*a) + " (offset " +
                    std::to_string(loc.segment_offset()) + ", size " + std::to_string(obj->size()) +
                    ") is outside the allocated part of segment " + std::to_string(seg) +
                    " (alloc_pos " + std::to_string(info.alloc_pos) + ")";
         live[seg] += obj->size();
         out.bytes += obj->size();
         if (!walk_dense<bitset_r2_node>(alloc, obj, counters, ref) &&
             !walk_dense<bitset_r4_node>(alloc, obj, counters, ref) &&
             !walk_dense<full_radix_node>(alloc, obj, counters, ref))
            alloc.type_ops(obj).visit_children(obj, ref);
      }
      out.objects = expected.size();
      if (!error.empty())
      {
         out.error = error;
         return out;
      }
      // every allocated control block against the walk
      uint64_t seen = 0, leaked = 0, zero = 0, wrong = 0;
      std::string first_wrong;
      {
         auto ses = alloc.create_session();
         ses->for_each_live_object(
             [&](sal::ptr_address adr, uint32_t refs, const sal::alloc_header*) {
                auto it = expected.find(*adr);
                if (it == expected.end())
                {
                   refs == 0 ? ++zero : ++leaked;
                   return;
                }
                ++seen;
                if (expect_no_leak ? refs != it->second : refs < it->second)
                {
                   if (wrong++ == 0)
                      first_wrong = "object " + std::to_string(*adr) + " has count " +
                                    std::to_string(refs) + ", the roots hold " +
                                    std::to_string(it->second) + " references";
                }
             });
      }
      if (wrong)
         out.error = std::to_string(wrong) + " reference counts differ from the walk; " + first_wrong;
      else if (seen != expected.size())
         out.error = std::to_string(expected.size() - seen) +
                     " reachable objects have no allocated control block";
      else if (expect_no_leak && (leaked || zero))
         out.error = std::to_string(leaked) + " objects are allocated that no root reaches, " +
                     std::to_string(zero) + " control blocks are allocated with a zero count";
      else if (expect_no_leak && alloc.total_allocated_objects() != expected.size())
         out.error = "the allocator counts " + std::to_string(alloc.total_allocated_objects()) +
                     " objects, the roots reach " + std::to_string(expected.size());
      if (!out.error.empty())
         return out;
      // the shared branch-set counters
      for (auto& [ctr, n] : counters)
         if (const auto have = alloc.iref().count(sal::ptr_address(ctr));
             expect_no_leak ? have != n : have < n)
         {
            out.error = "branch-set counter " + std::to_string(ctr) + " is " +
                        std::to_string(alloc.iref().count(sal::ptr_address(ctr))) + ", " +
                        std::to_string(n) + " nodes share it";
            return out;
         }
      if (expect_no_leak && alloc.iref().used() != counters.size())
      {
         out.error = "the branch-set counter pool has " + std::to_string(alloc.iref().used()) +
                     " counters in use, the roots reach " + std::to_string(counters.size());
         return out;
      }
      out.shared_counters = counters.size();
      // segments
      uint32_t orphans = 0;
      for (size_t s = 0; s < dump.segments.size(); ++s)
      {
         const auto& info = dump.segments[s];
         if (live[s])
            ++out.data_segments;
         if (info.is_free)
            continue;  // (a reachable object in a free segment was reported above)
         if (info.is_read_only)
         {
            // What a frozen segment holds that is not live is what the compactor may reclaim.
            if (expect_no_leak && uint64_t(info.freed_bytes) + live[s] != usable)
            {
               out.error = "segment " + std::to_string(s) + ": freed " +
                           std::to_string(info.freed_bytes) + " + live " + std::to_string(live[s]) +
                           " is not its size " + std::to_string(usable);
               return out;
            }
         }
         else if (info.alloc_pos != 0)
            ++orphans;  // written by a session; no session of this process has written yet
      }
      if (expect_no_leak && orphans)
         out.error = std::to_string(orphans) +
                     " segments hold data but belong to no session and are not compactable";
      return out;
   }

   // ── one life ────────────────────────────────────────────────────────────
   struct life_plan
   {
      uint64_t           seed = 1;
      profile            prof;
      uint32_t           txns    = 40;
      uint32_t           readers = 2;
      std::string        point;  // empty: not armed
      uint64_t           at        = 0;
      bool               counting  = false;
      bool               workload  = true;
      bool               clean_end = true;  // close the store when the workload ends
      sal::recovery_mode recovery  = sal::recovery_mode::none;  // none: the engine decides
      bool               expect_clean = false;                  // the store was closed cleanly
      bool               leaks_allowed = false;                 // after a deferred-cleanup open
      bool (*special)()            = nullptr;
   };

   bool arms_before_open(const std::string& point)
   {
      return point.starts_with("recover:") || point.starts_with("open:");
   }

   void throttle(database& db)
   {
      for (int i = 0; i < 2000; ++i)
      {
         const auto d    = db.dump();
         uint32_t   busy = 0;
         for (auto& s : d.segments)
            busy += !s.is_free;
         if (busy < segment_limit_for_writers)
            return;
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
   }

   // Open, verify every root against the campaign state inherited from the parent, audit.
   // Reports what it found; `state` is advanced to it.
   std::shared_ptr<database> open_and_verify(const fs::path& dir, const life_plan& plan,
                                             campaign_state& state)
   {
      auto db = database::open(dir, psitri::open_mode::create_or_open, store_config(),
                               plan.recovery);
      send(m_opened, 0, uint64_t(db->recovery_performed()), uint64_t(db->prior_shutdown()));
      {
         auto rs = db->start_read_session();
         for (uint32_t r = 0; r < data_roots; ++r)
         {
            auto&          st    = state[r];
            auto           c     = rs->create_cursor(r);
            const uint64_t found = stored_u64(c, seq_key);
            // Every acknowledged publication is present; the one in flight is wholly there
            // or wholly absent; nothing else.
            if (found != st.acked && found != st.acked + 1)
               child_fail("root " + std::to_string(r) + " is at version " + std::to_string(found) +
                          ", the last acknowledged publication is " + std::to_string(st.acked));
            replay(st, plan.seed, r, found, plan.prof);
            if (auto d = compare_tree(c, st.model, mix(plan.seed, r, found)); !d.empty())
               child_fail("root " + std::to_string(r) + " at version " + std::to_string(found) +
                          ": " + d);
            st.acked = found;
            send(m_root, r, found);
         }
         if (state[pair_b].version > state[pair_a].version)
            child_fail("the second root of a two-root commit is ahead of the first");
         for (uint32_t r = 0; r < data_roots; ++r)
         {
            auto&          st    = state[r];
            auto           c     = rs->create_cursor(stash_base + r);
            const uint64_t found = stored_u64(c, seq_key);
            const model_map* want = nullptr;
            if (found == st.stash_version)
               want = &st.stash;
            else if (found == st.prev_stash_version && st.version == st.stash_version &&
                     st.stash_acked < st.stash_version)
               want = &st.prev_stash;  // died between the commit and its stash
            if (!want)
               child_fail("stash of root " + std::to_string(r) + " is at version " +
                          std::to_string(found) + ", expected " + std::to_string(st.stash_version) +
                          " (acknowledged " + std::to_string(st.stash_acked) + ")");
            if (auto d = compare_tree(c, *want, mix(plan.seed, r, found)); !d.empty())
               child_fail("stash of root " + std::to_string(r) + " at version " +
                          std::to_string(found) + ": " + d);
            send(m_stash, r, found);
         }
      }
      const auto a = audit(*db, !plan.leaks_allowed);
      if (!a.error.empty())
         child_fail("audit: " + a.error);
      send(m_audit, uint32_t(a.shared_counters), a.objects, a.data_segments);
      return db;
   }

   template <class Session>
   void stash_now(Session& ws, uint32_t r, uint64_t version)
   {
      ws.set_root(stash_base + r, ws.get_root(r));  // two roots now share one tree
      send(m_ack, stash_base + r, version);
   }

   void transaction_writer(database& db, const life_plan& plan, uint32_t r, root_state& st,
                           std::atomic<bool>& stop)
   {
      auto ws = db.start_write_session();
      for (uint32_t n = 0; n < plan.txns && !stop.load(std::memory_order_relaxed); ++n)
      {
         const uint64_t k = st.version + 1;
         if (mix(plan.seed ^ 0xD00D, r, k) % 5 == 0 && !reverts(r, k))
         {
            // work that is thrown away: nothing of it may reach the root
            root_state scratch = st;
            auto       tx      = ws->start_transaction(r);
            apply_publication(&tx, ws.get(), scratch, plan.seed ^ 0xD00D, r, k, plan.prof);
            if (k % 2)
               tx.abort();
         }
         {
            auto tx = ws->start_transaction(r);
            if (reverts(r, k))
               (void)tx.primary().set_root(ws->get_root(stash_base + r));
            apply_publication(&tx, ws.get(), st, plan.seed, r, k, plan.prof);
            tx.commit();
         }
         send(m_ack, r, k);  // after commit() returned
         if (stash_scheduled(k))
            stash_now(*ws, r, k);
         if (n % 8 == 7)
            throttle(db);
      }
   }
   void pair_writer(database& db, const life_plan& plan, root_state& a, root_state& b,
                    std::atomic<bool>& stop)
   {
      auto ws = db.start_write_session();
      if (b.version < a.version)
      {
         // The last life died between the two root stores of one commit, which is allowed:
         // roots are published one after another. Bring the second root level.
         auto tx = ws->start_transaction(pair_b);
         apply_publication(&tx, ws.get(), b, plan.seed, pair_b, b.version + 1, plan.prof);
         tx.commit();
         send(m_ack, pair_b, b.version);
         if (stash_scheduled(b.version))
            stash_now(*ws, pair_b, b.version);
      }
      for (uint32_t n = 0; n < plan.txns && !stop.load(std::memory_order_relaxed); ++n)
      {
         const uint64_t k = a.version + 1;
         {
            auto  tx    = ws->start_transaction(pair_a);
            auto& other = tx.open_root(pair_b);
            apply_publication(&tx, ws.get(), a, plan.seed, pair_a, k, plan.prof);
            apply_publication(&other, ws.get(), b, plan.seed, pair_b, k, plan.prof);
            tx.commit();  // publishes pair_a, then pair_b
         }
         send(m_ack, pair_a, k);
         send(m_ack, pair_b, k);
         if (stash_scheduled(k))
         {
            stash_now(*ws, pair_a, k);
            stash_now(*ws, pair_b, k);
         }
         if (n % 8 == 7)
            throttle(db);
      }
   }
   // The node-local root: the session builds a tree outside any transaction, publishes it
   // with set_root, and goes on editing the tree it holds.
   void keep_writer(database& db, const life_plan& plan, root_state& st, std::atomic<bool>& stop)
   {
      auto ws   = db.start_write_session();
      auto mine = ws->get_root(keep_root);
      // A tree of the session's own that is never published and is edited across the
      // session's publications: what it wrote before a publication is frozen by it, so the
      // next edit of a node only this tree owns has to move the node.
      auto                               side = ws->create_temporary_tree();
      std::map<std::string, std::string> side_model;
      std::mt19937_64                    side_rng(mix(plan.seed, 0x51DE, st.version));
      for (uint32_t n = 0; n < plan.txns && !stop.load(std::memory_order_relaxed); ++n)
      {
         const uint64_t k = st.version + 1;
         {
            auto wt = ws->start_write_transaction(std::move(side));
            for (int i = 0; i < 6; ++i)
            {
               const std::string key = key_of(uint32_t(side_rng() % 300));
               if (side_rng() % 4 == 0)
               {
                  wt.remove(key);
                  side_model.erase(key);
               }
               else
               {
                  const std::string v(size_t(side_rng() % 400), char('a' + i));
                  wt.upsert(key, v);
                  side_model[key] = v;
               }
            }
            side = wt.get_tree();
         }
         if (mix(plan.seed ^ 0xD00D, keep_root, k) % 5 == 0 && !reverts(keep_root, k))
         {
            root_state scratch = st;
            auto       wt      = ws->start_write_transaction(mine);  // a copy: edits are dropped
            apply_publication(&wt, ws.get(), scratch, plan.seed ^ 0xD00D, keep_root, k, plan.prof);
         }
         auto wt = reverts(keep_root, k)
                       ? ws->start_write_transaction(ws->get_root(stash_base + keep_root))
                       : k % 2 ? ws->start_write_transaction(std::move(mine))
                               : ws->start_write_transaction(mine);
         apply_publication(&wt, ws.get(), st, plan.seed, keep_root, k, plan.prof);
         mine = wt.get_tree();
         ws->set_root(keep_root, mine);  // the session keeps `mine`
         send(m_ack, keep_root, k);
         if (stash_scheduled(k))
            stash_now(*ws, keep_root, k);
         if (n % 8 == 7)
            throttle(db);
      }
      auto     c = side.snapshot_cursor();
      auto     it = side_model.begin();
      for (bool ok = c.seek_begin(); ok; ok = c.next(), ++it)
         if (it == side_model.end() || c.key() != it->first ||
             c.template value<std::string>() != it->second)
            child_fail("the session's unpublished tree differs from what was written to it");
      if (it != side_model.end())
         child_fail("the session's unpublished tree lost keys");
   }
   // The node's main root: a tree built outside, put in place of the root by a transaction.
   void replace_writer(database& db, const life_plan& plan, root_state& st, std::atomic<bool>& stop)
   {
      auto ws = db.start_write_session();
      for (uint32_t n = 0; n < plan.txns && !stop.load(std::memory_order_relaxed); ++n)
      {
         const uint64_t k  = st.version + 1;
         auto           wt = ws->start_write_transaction(
             ws->get_root(reverts(replace_root, k) ? stash_base + replace_root : replace_root));
         apply_publication(&wt, ws.get(), st, plan.seed, replace_root, k, plan.prof);
         {
            auto tx = ws->start_transaction(replace_root);
            (void)tx.primary().set_root(wt.get_tree());
            tx.commit();
         }
         send(m_ack, replace_root, k);
         if (stash_scheduled(k))
            stash_now(*ws, replace_root, k);
         if (n % 8 == 7)
            throttle(db);
      }
   }
   // A reader: snapshots of any root, some held while the writers move on. A snapshot is
   // checked against the sum its own publication stored in it.
   void reader(database& db, uint64_t seed, std::atomic<bool>& stop)
   {
      auto            rs = db.start_read_session();
      std::mt19937_64 rng(seed);
      std::deque<engine::shared_tree> held;
      auto check = [&](auto& c, const char* what) {
         uint64_t sum = 0, n = 0;
         const bool backwards = rng() % 4 == 0;
         for (bool ok = backwards ? c.seek_last() : c.seek_begin(); ok;
              ok      = backwards ? c.prev() : c.next())
         {
            const std::string key(c.key());
            if (key[0] == '~')
               continue;
            ++n;
            if (c.is_subtree())
               sum ^= entry_hash(key, entry{nullptr, std::make_shared<submap>()});
            else
            {
               auto v = c.template value<std::string>();
               if (!v)
                  child_fail(std::string(what) + ": a key without a readable value");
               sum ^= entry_hash(key, entry{std::make_shared<std::string>(std::move(*v)), nullptr});
            }
         }
         if (sum != stored_u64(c, sum_key))
            child_fail(std::string(what) + ": a snapshot of " + std::to_string(n) +
                       " keys does not match the sum its publication stored");
      };
      while (!stop.load(std::memory_order_relaxed))
      {
         const uint32_t root = uint32_t(rng() % (2 * data_roots));
         {
            auto c = rs->create_cursor(root);
            check(c, "reader");
         }
         held.push_back(rs->get_root(root));
         if (held.size() > 3)
         {
            auto c = rs->create_cursor(held.front());
            check(c, "reader of a held snapshot");
            held.pop_front();
         }
         std::this_thread::sleep_for(std::chrono::microseconds(200 + rng() % 2000));
      }
   }

   void run_child(const fs::path& dir, const life_plan& plan, campaign_state state)
   {
      crash::install();
      crash::counting.store(plan.counting);
      crash::special.store(plan.special);
      if (arms_before_open(plan.point))
         crash::arm(plan.point, plan.at);
      {
         auto db = open_and_verify(dir, plan, state);
         if (plan.workload)
         {
            {
               // a stash whose commit was the last thing the previous life did
               auto ws = db->start_write_session();
               auto rs = db->start_read_session();
               for (uint32_t r = 0; r < data_roots; ++r)
               {
                  auto c = rs->create_cursor(stash_base + r);
                  if (stored_u64(c, seq_key) != state[r].stash_version)
                     stash_now(*ws, r, state[r].stash_version);
               }
            }
            if (!arms_before_open(plan.point))
               crash::arm(plan.point, plan.at);
            send(m_ready);
            std::atomic<bool>        stop{false};
            std::vector<std::thread> writers, readers;
            auto guard = [](auto fn) {
               return [fn]() {
                  try
                  {
                     fn();
                  }
                  catch (const std::exception& e)
                  {
                     child_fail(std::string("exception in the workload: ") + e.what());
                  }
               };
            };
            for (uint32_t r = 0; r < 4; ++r)
               writers.emplace_back(
                   guard([&, r] { transaction_writer(*db, plan, r, state[r], stop); }));
            writers.emplace_back(
                guard([&] { pair_writer(*db, plan, state[pair_a], state[pair_b], stop); }));
            writers.emplace_back(guard([&] { keep_writer(*db, plan, state[keep_root], stop); }));
            writers.emplace_back(
                guard([&] { replace_writer(*db, plan, state[replace_root], stop); }));
            for (uint32_t i = 0; i < plan.readers; ++i)
               readers.emplace_back(guard([&, i] { reader(*db, mix(plan.seed, 77, i), stop); }));
            for (auto& t : writers)
               t.join();
            stop.store(true);
            for (auto& t : readers)
               t.join();
            send(m_done);
         }
         if (!plan.clean_end)
         {
            // the workload is over and the process just stops, as a killed node does
            send(m_dying, 0, 0, 0, "(end of life)");
            ::kill(::getpid(), SIGKILL);
            for (;;)
               ::pause();
         }
      }
      send(m_closed);
      if (plan.counting)
         for (auto& [name, n] : crash::counts())
            send(m_count, 0, n, 0, name);
      ::_exit(0);
   }

   // ── the parent ──────────────────────────────────────────────────────────
   void wait_for_store_lock(const fs::path& dir)
   {
      // A killed process's descriptors are closed as part of its exit, which need not be
      // complete when waitpid returns; the store's files are locked without waiting.
      const auto file = dir / "ptrs" / "header.bin";
      if (!fs::exists(file))
         return;
      for (int attempt = 0; attempt < 10000; ++attempt)
      {
         const int fd = ::open(file.c_str(), O_RDONLY);
         if (fd < 0)
            return;
         const bool unlocked = ::flock(fd, LOCK_EX | LOCK_NB) == 0;
         if (unlocked)
            ::flock(fd, LOCK_UN);
         ::close(fd);
         if (unlocked)
            return;
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      throw std::runtime_error("the dead child's lock on the store was never released");
   }

   struct life_result
   {
      bool                            opened = false, ready = false, done = false, closed = false;
      bool                            killed_by_parent = false, timed_out = false;
      int                             exit_code   = -1;  // -1: died of a signal
      int                             term_signal = 0;
      uint64_t                        recovery = 0, prior = 0;
      std::string                     died_at;  // the crash point the child reported dying at
      std::string                     failure;  // from the child, or about the child
      std::map<std::string, uint64_t> counts;
      uint64_t                        acks = 0, roots_verified = 0, audited_objects = 0;
      uint64_t                        audited_counters = 0;
      bool                            audited = false;
   };

   // Run one life. `state` follows what the child reports: verified versions, then acks.
   // kill_after_ready_us >= 0: the parent kills the child that long after it reports ready.
   life_result run_life(const fs::path& dir, const life_plan& plan, campaign_state& state,
                        int64_t kill_after_ready_us = -1, int timeout_s = 120)
   {
      wait_for_store_lock(dir);
      int fds[2];
      if (::pipe(fds) != 0)
         throw std::runtime_error("pipe failed");
      std::fflush(stdout);
      std::fflush(stderr);
      const pid_t pid = ::fork();
      if (pid < 0)
         throw std::runtime_error("fork failed");
      if (pid == 0)
      {
         ::close(fds[0]);
         g_out = fds[1];
         try
         {
            run_child(dir, plan, state);
         }
         catch (const std::exception& e)
         {
            child_fail(std::string("exception: ") + e.what());
         }
         ::_exit(0);
      }
      ::close(fds[1]);
      life_result res;
      const auto  start    = std::chrono::steady_clock::now();
      auto        deadline = start + std::chrono::seconds(timeout_s);
      std::optional<std::chrono::steady_clock::time_point> kill_at;
      auto handle = [&](const message& m) {
         switch (m.type)
         {
            case m_opened:
               res.opened   = true;
               res.recovery = m.a;
               res.prior    = m.b;
               break;
            case m_root:
               replay(state[m.root], plan.seed, m.root, m.a, plan.prof);
               state[m.root].acked = m.a;
               ++res.roots_verified;
               break;
            case m_stash:
               ++res.roots_verified;
               break;
            case m_audit:
               res.audited         = true;
               res.audited_objects = m.a;
               res.audited_counters = m.root;
               break;
            case m_ready:
               res.ready = true;
               if (kill_after_ready_us >= 0)
                  kill_at = std::chrono::steady_clock::now() +
                            std::chrono::microseconds(kill_after_ready_us);
               break;
            case m_ack:
               ++res.acks;
               if (m.root < data_roots)
               {
                  // (the model itself is advanced by the next life's verification)
                  if (m.a != state[m.root].acked + 1 && res.failure.empty())
                     res.failure = "harness: acknowledgement out of order";
                  state[m.root].acked = m.a;
               }
               else
                  state[m.root - stash_base].stash_acked = m.a;
               break;
            case m_dying:
               res.died_at = m.text;
               break;
            case m_count:
               res.counts[m.text] += m.a;
               break;
            case m_fail:
               if (res.failure.empty())
                  res.failure = m.text;
               break;
            case m_done:
               res.done = true;
               break;
            case m_closed:
               res.closed = true;
               break;
            default:
               break;
         }
      };
      for (;;)
      {
         auto now  = std::chrono::steady_clock::now();
         auto wake = deadline;
         if (kill_at && *kill_at < wake)
            wake = *kill_at;
         if (now >= wake)
         {
            if (!(kill_at && now >= *kill_at))
               res.timed_out = true;
            res.killed_by_parent = true;
            ::kill(pid, SIGKILL);
            kill_at.reset();
            deadline = now + std::chrono::seconds(30);
            if (res.timed_out)
               break;
            continue;
         }
         pollfd     p{fds[0], POLLIN, 0};
         const auto ms =
             std::chrono::duration_cast<std::chrono::milliseconds>(wake - now).count();
         const int ready = ::poll(&p, 1, int(std::max<int64_t>(1, ms)));
         if (ready <= 0)
            continue;
         message       m;
         const ssize_t n = ::read(fds[0], &m, sizeof(m));
         if (n == 0)
            break;  // the child is gone and everything it wrote has been read
         if (n == ssize_t(sizeof(m)))
            handle(m);
      }
      ::close(fds[0]);
      int status = 0;
      ::waitpid(pid, &status, 0);
      if (WIFEXITED(status))
         res.exit_code = WEXITSTATUS(status);
      else if (WIFSIGNALED(status))
         res.term_signal = WTERMSIG(status);
      if (res.timed_out && res.failure.empty())
         res.failure = std::string("the child did not finish in ") + std::to_string(timeout_s) +
                       " s (" + (res.opened ? "after opening the store" : "the open did not return") +
                       ")";
      if (res.failure.empty() && res.term_signal && res.term_signal != SIGKILL)
         res.failure = std::string("the child died of signal ") + std::to_string(res.term_signal) +
                       (res.opened ? "" : " while opening the store");
      if (res.failure.empty() && res.exit_code > 0)
         res.failure = "the child exited with code " + std::to_string(res.exit_code);
      return res;
   }

   struct campaign
   {
      std::unique_ptr<temporary_directory> dir;
      campaign_state                       state;
      uint64_t                             seed;
      profile                              prof;
      bool                                 dirty = false;  // the last life did not close the store
      bool unknown = false;  // it died where the test cannot tell which side of the record
      // totals
      uint64_t lives = 0, deaths = 0, deaths_in_recovery = 0, clean_closes = 0;
      uint64_t publications = 0, roots_verified = 0, audits = 0, stores = 0;
      uint64_t objects_audited = 0, max_objects = 0, max_counters = 0;
      std::map<std::string, uint64_t> deaths_at;
      std::vector<std::string>        failures;
      bool                            verbose = env_u64("LGE_CRASH_VERBOSE", 0) != 0;

      explicit campaign(uint64_t s) : seed(s) { fresh_store(); }
      void fresh_store()
      {
         dir = std::make_unique<temporary_directory>();
         state = {};
         dirty   = false;
         unknown = false;
         ++stores;
      }
      life_plan plan() const
      {
         life_plan p;
         p.seed = seed;
         p.prof = prof;
         p.txns = uint32_t(env_u64("LGE_CRASH_TXNS", 60));
         return p;
      }
      // Run a life and check what it reported against how the store was left before it.
      life_result life(life_plan p, int64_t kill_after_us = -1)
      {
         const bool was_dirty = dirty;
         auto       res       = run_life(dir->path, p, state, kill_after_us);
         ++lives;
         publications += res.acks;
         roots_verified += res.roots_verified;
         audits += res.audited;
         objects_audited += res.audited_objects;
         max_objects  = std::max(max_objects, res.audited_objects);
         max_counters = std::max(max_counters, res.audited_counters);
         const bool died = res.term_signal == SIGKILL;
         if (died)
         {
            ++deaths;
            if (!res.opened)
               ++deaths_in_recovery;
            if (!res.died_at.empty())
               ++deaths_at[res.died_at];
         }
         if (res.closed)
            ++clean_closes;
         std::string failure = res.failure;
         if (failure.empty() && res.opened && p.recovery == sal::recovery_mode::none && !unknown)
         {
            const auto want = was_dirty ? sal::recovery_mode::app_crash : sal::recovery_mode::none;
            if (sal::recovery_mode(res.recovery) != want)
               failure = "the open ran recovery mode " + std::to_string(res.recovery) + ", expected " +
                         std::to_string(int(want)) + " (prior shutdown state " +
                         std::to_string(res.prior) + ")";
         }
         if (verbose || !failure.empty())
            std::printf("  life %llu: point '%s' n=%llu %s -> %s%s%s, %llu acks%s%s\n",
                        (unsigned long long)lives, p.point.c_str(), (unsigned long long)p.at,
                        was_dirty ? "(after a death)" : "(after a clean close)",
                        died ? "died" : res.closed ? "closed" : "ended",
                        res.died_at.empty() ? "" : " at ", res.died_at.c_str(),
                        (unsigned long long)res.acks, failure.empty() ? "" : "  FAILURE: ",
                        failure.c_str());
         if (!failure.empty())
         {
            failures.push_back("life " + std::to_string(lives) + " (point '" + p.point + "' n=" +
                               std::to_string(p.at) + (was_dirty ? ", after a death" : "") +
                               "): " + failure);
            fresh_store();  // what the store holds now is not known: start another
            res.failure = failure;
            return res;
         }
         // How the store was left. The shutdown record is written as the open's last step
         // and the clean record as the close's last, so a death right around either can
         // have fallen on either side of it.
         const bool was_unknown = unknown;
         unknown                = false;
         if (res.closed || res.died_at == "close:clean_record_written")
            dirty = false;
         else if (!res.opened)
         {
            if (res.died_at == "open:record_written" || res.died_at == "open:threads_started")
               dirty = true;
            else if (!was_dirty || was_unknown)
               unknown = res.died_at.empty() || was_unknown;  // a clean store stays clean
         }
         else if (res.done && res.died_at.empty())
            unknown = true;  // killed by the parent somewhere in the close
         else
            dirty = true;
         return res;
      }
      void absorb(const campaign& o)
      {
         lives += o.lives, deaths += o.deaths, deaths_in_recovery += o.deaths_in_recovery;
         clean_closes += o.clean_closes, publications += o.publications;
         roots_verified += o.roots_verified, audits += o.audits, stores += o.stores;
         objects_audited += o.objects_audited;
         max_objects  = std::max(max_objects, o.max_objects);
         max_counters = std::max(max_counters, o.max_counters);
         for (auto& [name, n] : o.deaths_at)
            deaths_at[name] += n;
         failures.insert(failures.end(), o.failures.begin(), o.failures.end());
      }
      void summary(const char* title) const
      {
         std::printf("%s: %llu lives on %llu stores, %llu deaths (%llu before the open returned), "
                     "%llu clean closes, %llu publications acknowledged, %llu root checks, %llu "
                     "audits (%llu object counts checked, largest store %llu objects, %llu "
                     "branch-set counters), %zu failures\n",
                     title, (unsigned long long)lives, (unsigned long long)stores,
                     (unsigned long long)deaths, (unsigned long long)deaths_in_recovery,
                     (unsigned long long)clean_closes, (unsigned long long)publications,
                     (unsigned long long)roots_verified, (unsigned long long)audits,
                     (unsigned long long)objects_audited, (unsigned long long)max_objects,
                     (unsigned long long)max_counters, failures.size());
         for (auto& f : failures)
            std::printf("  FAILURE %s\n", f.c_str());
         std::fflush(stdout);
      }
   };
}  // namespace

TEST_CASE("large engine: a workload with no death verifies and audits", "[large-crash][smoke]")
{
   campaign c(env_u64("LGE_CRASH_SEED", 1));
   for (int i = 0; i < 3; ++i)
   {
      auto p = c.plan();
      auto r = c.life(p);
      REQUIRE(r.failure == "");
      REQUIRE(r.closed);
   }
   // and one that just stops
   auto p      = c.plan();
   p.clean_end = false;
   auto r      = c.life(p);
   REQUIRE(r.failure == "");
   p           = c.plan();
   p.workload  = false;
   r           = c.life(p);
   REQUIRE(r.failure == "");
   REQUIRE(sal::recovery_mode(r.recovery) == sal::recovery_mode::app_crash);
   c.summary("no-death workload");
   REQUIRE(c.failures.empty());
}

TEST_CASE("large engine: death at every crash point", "[large-crash][points]")
{
   const uint64_t seed   = env_u64("LGE_CRASH_SEED", 1);
   const uint64_t rounds = env_u64("LGE_CRASH_ROUNDS", 3);
   const auto     only   = env_str("LGE_CRASH_POINT");
   campaign       c(seed);
   std::mt19937_64 rng(mix(seed, 4242));

   // How often each point is passed: in a life that follows a death (so the open recovers),
   // runs the workload and closes.
   std::map<std::string, uint64_t> passes;
   {
      auto p      = c.plan();
      p.clean_end = false;
      REQUIRE(c.life(p).failure == "");
      p          = c.plan();
      p.counting = true;
      auto r     = c.life(p);
      REQUIRE(r.failure == "");
      REQUIRE(r.closed);
      passes = r.counts;
      if (env_u64("LGE_CRASH_VERBOSE", 0))
         for (auto& [name, n] : passes)
            std::printf("  passes of %s in one life: %llu\n", name.c_str(), (unsigned long long)n);
   }
   std::vector<std::string> points;
   for (const char* name : all_points)
      if (only.empty() || only == name)
         points.push_back(name);
   std::vector<std::string> recovery_points;
   for (auto& name : points)
      if (name.starts_with("recover:"))
         recovery_points.push_back(name);

   std::set<std::string> never_passed;
   for (uint64_t round = 0; round < rounds; ++round)
      for (auto& name : points)
      {
         const uint64_t seen = passes.count(name) ? passes[name] : 0;
         if (seen == 0)
            never_passed.insert(name);
         // the first pass, then anywhere in what a life passes
         const uint64_t at = round == 0 || seen <= 1 ? 1 : 1 + rng() % seen;
         auto           p  = c.plan();
         p.point           = name;
         p.at              = at;
         if (name.starts_with("recover:") && !c.dirty)
         {
            // recovery has to have something to recover: a death first
            auto q      = c.plan();
            q.clean_end = false;
            c.life(q, int64_t(rng() % 150000));
         }
         p.counting = true;  // (a life that closes reports what it passed)
         auto r     = c.life(p);
         for (auto& [point, n] : r.counts)
            passes[point] = std::max(passes[point], n);
         if (r.failure.empty() && r.died_at != name && round == 0)
         {
            // Not reached on a store that has been written to for a while: some points
            // are only passed while a store is new (its file growing, its first open).
            campaign young(seed + 7);
            young.verbose = c.verbose;
            auto q        = young.plan();
            q.point       = name;
            q.at          = 1;
            auto y        = young.life(q);
            if (y.failure.empty() && young.dirty)
            {
               q          = young.plan();
               q.workload = false;
               young.life(q);
            }
            c.absorb(young);
         }
         if (r.failure.empty() && !r.opened && rng() % 2 == 0 && !recovery_points.empty())
         {
            // it died while recovering: die once more in the recovery of that
            auto q  = c.plan();
            q.point = recovery_points[rng() % recovery_points.size()];
            q.at    = 1 + rng() % 3;
            c.life(q);
         }
      }
   // A second chance for the points no life has died at yet: the store has grown since
   // their turn (wide routers, long release queues), and more is known about how often
   // each is passed.
   for (int chance = 0; chance < 3; ++chance)
      for (auto& name : points)
      {
         if (c.deaths_at.count(name) || !passes.count(name) || passes[name] == 0)
            continue;
         if (name.starts_with("recover:") && !c.dirty)
         {
            auto q      = c.plan();
            q.clean_end = false;
            c.life(q, int64_t(rng() % 150000));
         }
         auto p     = c.plan();
         p.point    = name;
         p.at       = 1 + rng() % std::max<uint64_t>(1, passes[name] / 2);
         p.counting = true;
         auto r     = c.life(p);
         for (auto& [point, n] : r.counts)
            passes[point] = std::max(passes[point], n);
      }
   // whatever the last death left is recovered and checked
   {
      auto p     = c.plan();
      p.workload = false;
      c.life(p);
   }
   c.summary("death at every crash point");
   std::vector<std::string> not_died;
   for (auto& name : points)
      if (!c.deaths_at.count(name))
         not_died.push_back(name);
   std::printf("crash points: %zu, died at %zu; never died at:", points.size(),
               points.size() - not_died.size());
   (void)never_passed;
   for (auto& name : not_died)
      std::printf(" %s%s", name.c_str(),
                  passes.count(name) && passes[name] ? "" : "(never passed)");
   std::printf("\n");
   if (env_u64("LGE_CRASH_VERBOSE", 0))
      for (auto& [name, n] : c.deaths_at)
         std::printf("  deaths at %s: %llu\n", name.c_str(), (unsigned long long)n);
   REQUIRE(c.failures.empty());
}

TEST_CASE("large engine: death at random moments", "[large-crash][random]")
{
   const uint64_t  seed  = env_u64("LGE_CRASH_SEED", 1);
   const uint64_t  kills = env_u64("LGE_CRASH_RANDOM", 60);
   campaign        c(seed + 1000);
   std::mt19937_64 rng(mix(seed, 99));
   for (uint64_t i = 0; i < kills; ++i)
   {
      auto p = c.plan();
      p.txns = uint32_t(env_u64("LGE_CRASH_TXNS", 120));
      if (i % 9 == 8)
         c.life(p);  // now and then a life that closes the store
      else
      {
         p.clean_end = false;  // (a life the kill comes too late for stops by itself)
         c.life(p, int64_t(rng() % 3 == 0 ? rng() % 3000 : rng() % 500000));
      }
   }
   auto p     = c.plan();
   p.workload = false;
   c.life(p);
   c.summary("death at random moments");
   REQUIRE(c.failures.empty());
}

namespace
{
   // Run `fn` in a forked child; what went wrong, or empty. A failure in the child (an
   // abort, a fault, an exception, a failed check) must not take the test process with it.
   template <class Fn>
   std::string in_child(Fn&& fn, int timeout_s = 120)
   {
      int fds[2];
      if (::pipe(fds) != 0)
         throw std::runtime_error("pipe failed");
      std::fflush(stdout);
      std::fflush(stderr);
      const pid_t pid = ::fork();
      if (pid == 0)
      {
         ::close(fds[0]);
         g_out = fds[1];
         try
         {
            fn();
         }
         catch (const std::exception& e)
         {
            child_fail(std::string("exception: ") + e.what());
         }
         ::_exit(0);
      }
      ::close(fds[1]);
      std::string failure;
      const auto  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
      for (;;)
      {
         if (std::chrono::steady_clock::now() > deadline)
         {
            ::kill(pid, SIGKILL);
            failure = "the child did not finish in " + std::to_string(timeout_s) + " s";
            break;
         }
         pollfd p{fds[0], POLLIN, 0};
         if (::poll(&p, 1, 100) <= 0)
            continue;
         message       m;
         const ssize_t n = ::read(fds[0], &m, sizeof(m));
         if (n == 0)
            break;
         if (n == ssize_t(sizeof(m)) && m.type == m_fail && failure.empty())
            failure = m.text;
      }
      ::close(fds[0]);
      int status = 0;
      ::waitpid(pid, &status, 0);
      if (failure.empty() && WIFSIGNALED(status))
         failure = "the child died of signal " + std::to_string(WTERMSIG(status));
      if (failure.empty() && WIFEXITED(status) && WEXITSTATUS(status) != 0)
         failure = "the child exited with code " + std::to_string(WEXITSTATUS(status));
      return failure;
   }
   void check(bool ok, const std::string& what)
   {
      if (!ok)
         child_fail(what);
   }
}  // namespace

// A subtree stored at a key whose part after the page's common prefix does not fit a page
// record goes into an external record. When the page is full, the insert is retried after a
// split; the first attempt's record must not take the subtree with it.
TEST_CASE("large engine: a subtree at a long key survives the page split that stores it",
          "[large-crash][subtree]")
{
   temporary_directory dir;
   const auto failure = in_child([&] {
      auto db = database::open(dir.path, psitri::open_mode::create_only, store_config());
      std::map<std::string, submap> want;
      for (uint32_t round = 0; round < 4; ++round)
      {
         auto ws = db->start_write_session();
         auto tx = ws->start_transaction(0);
         for (uint32_t i = 0; i < 120; ++i)
         {
            char name[16];
            std::snprintf(name, sizeof(name), "s/%03u", (i * 7 + round) % 200);
            const std::string key = name + std::string(140 + i % 90, 'q');
            submap            s;
            for (uint32_t j = 0; j < 3 + i % 5; ++j)
               s["sub/" + std::to_string(j)] = std::string(20 + i, char('a' + j));
            auto wt = ws->start_write_transaction(ws->create_temporary_tree());
            for (auto& [k, v] : s)
               wt.upsert(k, v);
            tx.upsert_subtree(key, wt.get_tree());
            want[key] = std::move(s);
         }
         tx.commit();
      }
      {
         auto rs = db->start_read_session();
         auto c  = rs->create_cursor(0);
         auto it = want.begin();
         for (bool ok = c.seek_begin(); ok; ok = c.next(), ++it)
         {
            check(it != want.end() && c.key() == it->first, "the keys read back differ");
            check(c.is_subtree(), "a subtree key reads back as a value");
            check(scan_subtree(c) == it->second, "a subtree reads back with other contents");
         }
         check(it == want.end(), "subtree keys are missing");
      }
      const auto a = audit(*db, true);
      check(a.error.empty(), "audit: " + a.error);
   });
   REQUIRE(failure == "");
}

// ── deaths inside operations of the segment provider that offer no marker ──────────────
// The provider's free-segment map and its queue of prepared segments are structures of
// another library, updated by several stores each. The tests below stop at the marker just
// before such an update, make the update's first store themselves through a mapping of the
// allocator's state file, and die: the state a death between two stores of the update
// leaves.
namespace
{
   struct provider_layout
   {
      size_t free_level0 = 0;  // offset of the free map's lowest level in the state file
      size_t queue_bits = 0, queue_produced = 0;  // of the prepared-segment queue's words
      static size_t word_with(const void* object, size_t size, uint64_t value, const char* what)
      {
         size_t found = size, n = 0;
         for (size_t off = 0; off + 8 <= size; off += 8)
         {
            uint64_t w;
            std::memcpy(&w, static_cast<const char*>(object) + off, 8);
            if (w == value)
               found = off, ++n;
         }
         if (n != 1)
            throw std::runtime_error(std::string("cannot locate ") + what);
         return found;
      }
      provider_layout()
      {
         using state = sal::mapped_memory::allocator_state;
         using prov  = sal::mapped_memory::segment_provider;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
         const size_t provider = offsetof(state, _segment_provider);
         const size_t free_map = provider + offsetof(prov, free_segments);
         const size_t queue    = provider + offsetof(prov, ready_pinned_segments);
#pragma GCC diagnostic pop
         {
            // one bit set: word 5 of the lowest level holds 8, word 0 of the next holds 32
            auto map = std::make_unique<ucc::hierarchical_bitmap<sal::max_segment_count>>();
            map->set(5 * 64 + 3);
            free_level0 = free_map + word_with(map.get(), sizeof(*map), 8, "the free map") - 40;
         }
         {
            auto q = std::make_unique<ucc::poly_buffer<sal::segment_number>>();
            q->push(sal::segment_number(9));
            q->push(sal::segment_number(9));
            queue_bits     = queue + word_with(q.get(), sizeof(*q), 3, "the queue's slot bits");
            queue_produced = queue + word_with(q.get(), sizeof(*q), 2, "the queue's push count");
         }
      }
   };
   char* g_state_file = nullptr;  // the allocator's state file, mapped by the child
   const provider_layout& layout()
   {
      static const provider_layout l;
      return l;
   }
   void map_state_file(const fs::path& dir)
   {
      const int fd = ::open((dir / "header").c_str(), O_RDWR);
      if (fd < 0)
         child_fail("cannot open the allocator's state file");
      struct stat st;
      ::fstat(fd, &st);
      void* p = ::mmap(nullptr, size_t(st.st_size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (p == MAP_FAILED)
         child_fail("cannot map the allocator's state file");
      g_state_file = static_cast<char*>(p);
   }
   // At "provider:take_free": the provider is about to take the lowest free segment. When
   // that is the last one of its 64, clearing its bit is followed by clearing the word's
   // bit one level up. Do the first and die.
   bool tear_free_map()
   {
      auto* level0 = reinterpret_cast<std::atomic<uint64_t>*>(g_state_file + layout().free_level0);
      for (uint32_t w = 0; w < 64; ++w)
         if (const uint64_t bits = level0[w].load())
         {
            if (std::popcount(bits) != 1)
               return false;
            level0[w].store(0);
            return true;
         }
      return false;
   }
   // At "provider:marked_queued": the provider is about to push a prepared segment: store
   // it in its slot, set the slot's bit, then count it. Die between the last two.
   bool tear_queue_push()
   {
      auto* bits     = reinterpret_cast<std::atomic<uint64_t>*>(g_state_file + layout().queue_bits);
      auto* produced = reinterpret_cast<std::atomic<uint64_t>*>(g_state_file + layout().queue_produced);
      bits->fetch_or(uint64_t(1) << (produced->load() & 63));
      return true;
   }

   // Lives with the special hook until one dies of it, then one that must open, verify,
   // audit and work on.
   std::string death_inside(const char* point, bool (*special)(), bool close_first)
   {
      campaign c(env_u64("LGE_CRASH_SEED", 1) + 31);
      static const fs::path* dir;
      dir = &c.dir->path;
      for (int attempt = 0; attempt < 40; ++attempt)
      {
         auto p = c.plan();
         if (close_first && attempt == 0)
         {
            // a store that was closed: the death then falls in the open that follows
            c.life(p);
            continue;
         }
         p.point   = point;
         p.at      = 1;
         // (the hook maps the state file on first use, then runs the tear)
         static bool (*tear)();
         tear      = special;
         p.special = +[] {
            if (!g_state_file)
               map_state_file(*dir);
            return tear();
         };
         p.txns = 400;
         auto r = c.life(p);
         if (!r.failure.empty())
            return r.failure;
         if (r.died_at != point)
            continue;
         for (int after = 0; after < 2; ++after)
         {
            auto q = c.plan();
            r      = run_life(c.dir->path, q, c.state, -1, 10);
            if (!r.failure.empty())
               return "after the death: " + r.failure;
         }
         return {};
      }
      return "the state to tear never came up";
   }
}  // namespace

TEST_CASE("large engine: death between the stores that take a free segment",
          "[large-crash][provider]")
{
   REQUIRE(death_inside("provider:take_free", &tear_free_map, false) == "");
}
TEST_CASE("large engine: death between the stores that queue a prepared segment",
          "[large-crash][provider]")
{
   REQUIRE(death_inside("provider:marked_queued", &tear_queue_push, false) == "");
}

// ── is anything a published root reaches ever changed before the next root store? ─────
// The model stands on this: a root in the root table, and everything it reaches, stays as
// it is until that slot is stored again. The engine edits in place what only the writer
// owns, and the compactor moves objects behind their control blocks. The test below takes,
// for each published root, the logical content (through a reader's cursor) and the bytes of
// every object the root reaches, and takes them again after each kind of activity that
// must not change them.
namespace
{
   struct object_image
   {
      uint64_t location = 0;  // cacheline of the object's bytes
      uint32_t size     = 0;
      uint64_t bytes    = 0;  // hash of them
      uint16_t session  = 0;  // the session that wrote the segment they are in
   };
   struct root_image
   {
      uint64_t                                  content = 0;  // keys, values, subtrees
      uint64_t                                  keys    = 0;
      std::unordered_map<uint32_t, object_image> objects;
   };
   template <class Cursor>
   uint64_t content_hash(Cursor& c, uint64_t& keys)
   {
      uint64_t h = 0x1234;
      for (bool ok = c.seek_begin(); ok; ok = c.next())
      {
         ++keys;
         h = mix(h, hash_bytes(c.key()));
         if (c.is_subtree())
         {
            auto     sub = c.subtree().snapshot_cursor();  // trees inside trees
            uint64_t n   = 0;
            h            = mix(h, content_hash(sub, n), 2);
         }
         else
         {
            auto v = c.template value<std::string>();
            if (!v)
               child_fail("a key without a readable value");
            h = mix(h, hash_bytes(*v), 1);
         }
      }
      return h;
   }
   root_image image_of(database& db, uint32_t root)
   {
      using namespace sal_art_ref;
      using namespace sal_art_ref::radix;
      root_image out;
      {
         auto rs     = db.start_read_session();
         auto c      = rs->create_cursor(root);
         out.content = content_hash(c, out.keys);
      }
      auto& alloc = db.underlying_allocator();
      auto  ses   = alloc.create_session();
      auto  pin   = ses->lock();  // no segment read below is recycled meanwhile
      std::unordered_map<uint32_t, uint32_t>      counters;
      std::vector<sal::ptr_address>               todo;
      const std::function<void(sal::ptr_address)> ref = [&](sal::ptr_address a) {
         if (a != sal::null_ptr_address && out.objects.emplace(*a, object_image{}).second)
            todo.push_back(a);
      };
      ref(alloc.root_objects()[root].load(std::memory_order_relaxed).root);
      while (!todo.empty())
      {
         const auto a = todo.back();
         todo.pop_back();
         auto [obj, loc] = alloc.resolve(a);
         if (!obj)
            child_fail("an object a published root reaches does not resolve");
         auto& img    = out.objects[*a];
         img.location = loc.cacheline();
         img.size     = obj->size();
         img.bytes =
             hash_bytes({reinterpret_cast<const char*>(obj), size_t(obj->size())});
         img.session = *reinterpret_cast<const sal::mapped_memory::segment*>(
                            reinterpret_cast<const char*>(obj) - loc.segment_offset())
                            ->_session_id;
         if (!walk_dense<bitset_r2_node>(alloc, obj, counters, ref) &&
             !walk_dense<bitset_r4_node>(alloc, obj, counters, ref) &&
             !walk_dense<full_radix_node>(alloc, obj, counters, ref))
            alloc.type_ops(obj).visit_children(obj, ref);
      }
      return out;
   }
   struct image_stats
   {
      uint64_t objects = 0, unmoved = 0, moved = 0;
   };
   // `writers`: the session numbers of every session that writes in this test. An object
   // found at another place must have been put there by a session that is none of them
   // (the compactor's).
   void same_image(const root_image& before, const root_image& after, const std::string& when,
                   const std::set<uint16_t>& writers, image_stats& stats)
   {
      check(before.content == after.content && before.keys == after.keys,
            when + ": the content read through a snapshot cursor changed");
      check(before.objects.size() == after.objects.size(),
            when + ": the set of objects the root reaches changed");
      for (auto& [addr, was] : before.objects)
      {
         auto it = after.objects.find(addr);
         check(it != after.objects.end(), when + ": an object the root reached is gone");
         const auto& now = it->second;
         ++stats.objects;
         if (now.location == was.location)
         {
            ++stats.unmoved;
            check(now.size == was.size && now.bytes == was.bytes,
                  when + ": object " + std::to_string(addr) + " was changed where it lies");
         }
         else
         {
            ++stats.moved;
            check(!writers.count(now.session),
                  when + ": object " + std::to_string(addr) +
                      " was moved by a writer (session " + std::to_string(now.session) +
                      "): a published object was copied for writing and redirected");
         }
      }
   }

   constexpr uint32_t inplace_roots = 5;  // 0..4 are published and watched; 6 is churned
   template <class H>
   void fill(H& h, uint32_t root, uint64_t gen, uint32_t keys)
   {
      for (uint32_t i = 0; i < keys; ++i)
      {
         const size_t size = i % 13 == 0 ? 200 + i % 3000 : i % 97 == 0 ? 20000 + i * 7 : i % 60;
         h.upsert(key_of(i), std::string_view(*make_value(root, gen, i, size)));
      }
   }
   template <class WS>
   auto small_tree(WS& ws, uint64_t gen)
   {
      auto wt = ws.start_write_transaction(ws.create_temporary_tree());
      for (uint32_t j = 0; j < 40; ++j)
         wt.upsert("sub/" + std::to_string(j), std::string(size_t(10 + j * 5), char('a' + gen % 26)));
      return wt.get_tree();
   }
   // Edits of every kind on a handle whose result is not published.
   template <class H, class WS>
   void scribble(H& h, WS& ws, uint64_t gen, bool subtrees)
   {
      for (uint32_t i = 0; i < 700; ++i)
      {
         const uint32_t k = uint32_t(mix(gen, i) % 1300);
         switch (i % 7)
         {
            case 0:
            case 1:
               h.upsert(key_of(k), std::string(size_t(mix(gen, i, 3) % 90), 'x'));  // same or other size
               break;
            case 2:
               h.upsert(key_of(k), std::string(size_t(150 + mix(gen, i, 5) % 9000), 'y'));
               break;
            case 3:
               h.remove(key_of(k));
               break;
            case 4:
               h.upsert(key_of(k), std::string_view(*make_value(9, gen, i, 100 + i % 50)));
               break;
            case 5:
               if (i % 35 == 5)
                  h.remove_range_counted(key_of(k), key_of(k) + "\xff");
               else
                  h.upsert(key_of(k + 5000), "new");
               break;
            case 6:
               if (subtrees && i % 70 == 6)
                  h.upsert_subtree("s/" + std::to_string(i % 5), small_tree(ws, gen + i));
               break;
         }
      }
      h.upsert("k/BIG", std::string(size_t((2u << 20) + 300000), char('A' + gen % 26)));
   }
}  // namespace

TEST_CASE("large engine: what a published root reaches does not change before the next store "
          "of that root",
          "[large-crash][in-place]")
{
   temporary_directory dir;
   uint64_t            contents[inplace_roots] = {};
   int                 pipe_fds[2];
   REQUIRE(::pipe(pipe_fds) == 0);
   // 1: a process that builds the roots, checks them through every kind of unpublished work
   // and through compaction, and dies with work in progress.
   auto failure = in_child([&] {
      crash::install();
      crash::counting.store(true);
      auto               db = database::open(dir.path, psitri::open_mode::create_only, store_config());
      auto               ws = db->start_write_session();
      auto               other = db->start_write_session();
      std::set<uint16_t> writers{uint16_t(*ws->get_root(0).session()->get_session_num())};
      {
         auto tx = ws->start_transaction(0);
         fill(tx, 0, 1, 1300);
         tx.upsert("k/BIG", std::string(size_t((2u << 20) + 500000), 'B'));
         for (uint32_t j = 0; j < 5; ++j)
            tx.upsert_subtree("s/" + std::to_string(j), small_tree(*ws, j));
         tx.commit();
      }
      ws->set_root(1, ws->get_root(0));  // root 1 shares everything with root 0
      auto mine = [&] {
         auto wt = ws->start_write_transaction(ws->create_temporary_tree());
         fill(wt, 2, 2, 900);
         return wt.get_tree();
      }();
      ws->set_root(2, mine);  // the session keeps `mine`
      {
         // root 3: its own keys, and subtrees that are root 0's and root 2's trees
         auto tx = ws->start_transaction(3);
         fill(tx, 3, 3, 300);
         tx.upsert_subtree("s/whole0", ws->get_root(0));
         tx.upsert_subtree("s/whole2", ws->get_root(2));
         tx.upsert_subtree("s/part", ws->get_root(0).get_subtree("s/1"));
         tx.commit();
      }
      {
         // root 4: written by another session, never touched again
         auto tx = other->start_transaction(4);
         fill(tx, 4, 4, 600);
         tx.commit();
      }
      writers.insert(uint16_t(*other->get_root(4).session()->get_session_num()));

      root_image   base[inplace_roots];
      image_stats  stats;
      auto take = [&] {
         for (uint32_t r = 0; r < inplace_roots; ++r)
            base[r] = image_of(*db, r);
      };
      auto unchanged = [&](const std::string& when, int except = -1) {
         for (uint32_t r = 0; r < inplace_roots; ++r)
            if (int(r) != except)
               same_image(base[r], image_of(*db, r), when + ", root " + std::to_string(r), writers,
                          stats);
      };
      take();
      unchanged("nothing done");

      {  // a transaction's work, while it is open and after it is aborted
         auto tx = ws->start_transaction(0);
         scribble(tx, *ws, 10, true);
         tx.open_subtree("s/2").upsert("sub/3", "edited in place");
         unchanged("during an open transaction on root 0");
         tx.abort();
      }
      unchanged("after an aborted transaction on root 0");
      {  // savepoints inside a transaction that is then dropped
         auto tx = ws->start_transaction(0);
         scribble(tx, *ws, 11, false);
         {
            auto frame = tx.sub_transaction();
            scribble(tx, *ws, 12, true);
            unchanged("inside a savepoint");
            frame.abort();
         }
         {
            auto frame = tx.sub_transaction();
            scribble(tx, *ws, 13, false);
            frame.commit();
         }
         scribble(tx, *ws, 14, false);
         unchanged("after savepoints, transaction still open");
      }
      unchanged("after a transaction with savepoints was dropped");
      {  // a tree taken from a root, edited, never published
         auto wt = ws->start_write_transaction(ws->get_root(1));
         scribble(wt, *ws, 15, true);
         unchanged("a tree taken from root 1 edited");
         auto t = wt.get_tree();
         auto w2 = other->start_write_transaction(other->get_root(3));
         scribble(w2, *other, 16, true);
         unchanged("a tree taken from root 3 edited by another session");
      }
      {  // the tree the session published with set_root and kept
         auto wt = ws->start_write_transaction(std::move(mine));
         scribble(wt, *ws, 17, true);
         unchanged("the session's kept tree edited after set_root (handle moved in)");
         mine = wt.get_tree();
         auto w2 = ws->start_write_transaction(mine);
         scribble(w2, *ws, 18, false);
         unchanged("the session's kept tree edited after set_root (handle copied)");
      }
      {  // a subtree a published root owns, opened and edited
         auto tx = ws->start_transaction(3);
         tx.open_subtree("s/whole0").upsert(key_of(3), "through root 3");
         tx.open_subtree("s/part").upsert("sub/1", "through root 3");
         scribble(tx.open_subtree("s/whole2"), *ws, 19, false);
         unchanged("subtrees of root 3 opened and edited");
         tx.abort();
      }
      unchanged("after all unpublished work");

      // Publish on root 0 and root 2: the others, which share with them, do not move.
      {
         auto tx = ws->start_transaction(0);
         scribble(tx, *ws, 20, true);
         tx.commit();
      }
      unchanged("after publishing root 0", 0);
      base[0] = image_of(*db, 0);
      ws->set_root(2, mine);
      unchanged("after publishing root 2 with set_root", 2);
      base[2] = image_of(*db, 2);

      // Compaction: churn another root until the compactor has moved objects, some of them
      // objects the watched roots reach.
      const auto moved_before = stats.moved;
      for (uint32_t round = 0; round < 400 && stats.moved < moved_before + 200; ++round)
      {
         for (uint32_t n = 0; n < 60; ++n)
         {
            auto tx = ws->start_transaction(6);
            for (uint32_t i = 0; i < 12; ++i)
               tx.upsert(key_of(uint32_t(mix(round, n, i) % 500)),
                         std::string(size_t(mix(round, n, i) % 2000), 'c'));
            tx.commit();
         }
         throttle(*db);
         unchanged("while the compactor works (round " + std::to_string(round) + ")");
      }
      check(stats.moved >= moved_before + 200,
            "the compactor moved too few objects of the watched roots to say anything (" +
                std::to_string(stats.moved - moved_before) + ")");
      check(crash::counts()["compact:redirected"] > 0, "the compactor did not run");

      for (uint32_t r = 0; r < inplace_roots; ++r)
         if (::write(pipe_fds[1], &base[r].content, 8) != 8)
            child_fail("cannot report");
      std::printf("in-place: %llu object comparisons, %llu at the same place and byte-identical, "
                  "%llu moved by the compactor; compactor redirects %llu, allocator "
                  "copy-on-write redirects %llu\n",
                  (unsigned long long)stats.objects, (unsigned long long)stats.unmoved,
                  (unsigned long long)stats.moved,
                  (unsigned long long)crash::counts()["compact:redirected"],
                  (unsigned long long)crash::counts()["cow:redirected"]);
      std::fflush(stdout);
      // die with a transaction open, a kept tree being edited and a snapshot held
      auto tx = ws->start_transaction(0);
      scribble(tx, *ws, 30, true);
      auto wt = other->start_write_transaction(other->get_root(4));
      scribble(wt, *other, 31, false);
      ::kill(::getpid(), SIGKILL);
   });
   REQUIRE(failure == "the child died of signal 9");
   for (uint32_t r = 0; r < inplace_roots; ++r)
      REQUIRE(::read(pipe_fds[0], &contents[r], 8) == 8);
   wait_for_store_lock(dir.path);

   // 2: after the death. Opened without recovery of the counts first (deferred cleanup),
   // then, after another death, with the light pass. Each time: the roots hold what they
   // held, a writer's unpublished work does not change them, and what it publishes on one
   // root leaves the others alone.
   for (const auto mode : {sal::recovery_mode::deferred_cleanup, sal::recovery_mode::none})
   {
      failure = in_child([&] {
         auto db = database::open(dir.path, psitri::open_mode::open_existing, store_config(), mode);
         check(db->recovery_performed() ==
                   (mode == sal::recovery_mode::none ? sal::recovery_mode::app_crash : mode),
               "unexpected recovery mode");
         auto               ws = db->start_write_session();
         std::set<uint16_t> writers{uint16_t(*ws->get_root(0).session()->get_session_num())};
         root_image         base[inplace_roots];
         image_stats        stats;
         for (uint32_t r = 0; r < inplace_roots; ++r)
         {
            base[r] = image_of(*db, r);
            check(base[r].content == contents[r],
                  "root " + std::to_string(r) + " does not hold what it held before the death");
         }
         auto unchanged = [&](const std::string& when, int except = -1) {
            for (uint32_t r = 0; r < inplace_roots; ++r)
               if (int(r) != except)
                  same_image(base[r], image_of(*db, r), when + ", root " + std::to_string(r),
                             writers, stats);
         };
         {
            auto tx = ws->start_transaction(0);
            scribble(tx, *ws, 40, true);
            tx.open_subtree("s/2").upsert("sub/3", "edited in place");
            unchanged("after the death: during a transaction on root 0");
            tx.abort();
         }
         {
            auto wt = ws->start_write_transaction(ws->get_root(2));
            scribble(wt, *ws, 41, false);
            auto tx = ws->start_transaction(3);
            scribble(tx.open_subtree("s/whole0"), *ws, 42, false);
            unchanged("after the death: a tree of root 2 and a subtree of root 3 edited");
         }
         {
            auto tx = ws->start_transaction(4);
            scribble(tx, *ws, 43, false);
            tx.commit();
         }
         unchanged("after the death: root 4 published", 4);
         {
            // put root 4 back as it was (root 1's tree would do as well: any content)
            auto tx = ws->start_transaction(4);
            for (bool more = true; more;)
            {
               auto c = tx.snapshot_cursor();
               more   = c.seek_begin();
               if (more)
                  tx.remove_range_counted(std::string(c.key()), "\xff\xff\xff\xff");
            }
            fill(tx, 4, 4, 600);
            tx.commit();
         }
         uint64_t keys = 0;
         {
            auto rs = db->start_read_session();
            auto c  = rs->create_cursor(4);
            check(content_hash(c, keys) == contents[4], "root 4 was not rebuilt as it was");
         }
         ::kill(::getpid(), SIGKILL);
      });
      REQUIRE(failure == "the child died of signal 9");
      wait_for_store_lock(dir.path);
   }
   // 3: and a last open that audits everything (the deferred open's leaks are reclaimed by
   // the light pass that followed it).
   failure = in_child([&] {
      auto db = database::open(dir.path, psitri::open_mode::open_existing, store_config());
      check(db->recovery_performed() == sal::recovery_mode::app_crash, "unexpected recovery mode");
      for (uint32_t r = 0; r < inplace_roots; ++r)
      {
         uint64_t keys = 0;
         auto     rs   = db->start_read_session();
         auto     c    = rs->create_cursor(r);
         check(content_hash(c, keys) == contents[r],
               "root " + std::to_string(r) + " does not hold what it held");
      }
      const auto a = audit(*db, true);
      check(a.error.empty(), "audit: " + a.error);
   });
   REQUIRE(failure == "");
   ::close(pipe_fds[0]);
   ::close(pipe_fds[1]);
}


TEST_CASE("large engine: the published-root check notices a change", "[large-crash][in-place]")
{
   // The test above is only worth something if it fails when a published object changes.
   for (int kind = 0; kind < 2; ++kind)
   {
      temporary_directory dir;
      const auto failure = in_child([&] {
         auto db = database::open(dir.path, psitri::open_mode::create_only, store_config());
         auto ws = db->start_write_session();
         {
            auto tx = ws->start_transaction(0);
            fill(tx, 0, 1, 400);
            tx.commit();
         }
         ws->set_root(1, ws->get_root(0));
         const auto  before = image_of(*db, 1);
         image_stats stats;
         if (kind == 0)
         {
            // one byte of one object root 1 reaches, changed where it lies
            auto [obj, loc] = db->underlying_allocator().resolve(
                sal::ptr_address(before.objects.begin()->first));
            const_cast<char*>(reinterpret_cast<const char*>(obj))[obj->size() - 1] ^= 1;
         }
         else
         {
            auto tx = ws->start_transaction(1);
            tx.upsert(key_of(1), "changed");
            tx.commit();
         }
         same_image(before, image_of(*db, 1), "negative control", {}, stats);
      });
      INFO(failure);
      REQUIRE(failure.starts_with("negative control"));
   }
}

TEST_CASE("large engine: deaths around opens that defer cleanup", "[large-crash][deferred]")
{
   const uint64_t  seed = env_u64("LGE_CRASH_SEED", 1);
   campaign        c(seed + 500);
   std::mt19937_64 rng(mix(seed, 501));
   for (int round = 0; round < int(env_u64("LGE_CRASH_DEFERRED", 8)); ++round)
   {
      auto p      = c.plan();
      p.clean_end = false;
      c.life(p, int64_t(rng() % 120000));  // a death
      // opened without recounting: everything published is there, counts are never too
      // low, the store is written to; then it dies or closes
      p               = c.plan();
      p.recovery      = sal::recovery_mode::deferred_cleanup;
      p.leaks_allowed = true;
      p.clean_end     = round % 3 == 0;
      c.life(p, p.clean_end ? -1 : int64_t(rng() % 120000));
      if (round % 2)
      {
         p               = c.plan();  // twice in a row
         p.recovery      = sal::recovery_mode::deferred_cleanup;
         p.leaks_allowed = true;
         p.clean_end     = false;
         c.life(p, int64_t(rng() % 120000));
      }
      // the light pass, asked for by name (the store may have been closed): exact again
      p          = c.plan();
      p.recovery = sal::recovery_mode::app_crash;
      c.life(p);
   }
   c.summary("deaths around deferred-cleanup opens");
   REQUIRE(c.failures.empty());
}

// A value of several continuation chunks, each nearly a segment: death while it is built,
// while it is published, and while the version it replaced is released.
TEST_CASE("large engine: death while a value of several chunks is written and replaced",
          "[large-crash][large-value]")
{
   temporary_directory dir;
   const size_t        huge = (68u << 20) + 12345;
   auto value_of = [&](char fill) {
      std::string v(huge, fill);
      for (size_t at = 0; at + 8 <= v.size(); at += 1 << 20)
         std::memcpy(v.data() + at, &at, 8);
      return v;
   };
   // What root 0 must hold after each step: 0 = no value, else the fill byte.
   struct step
   {
      const char* point;
      uint64_t    at;
      char        write;     // the value this life tries to publish
      const char* allowed;   // fill bytes the next open may find ('-' = none)
   };
   const step steps[] = {
       {"large:head_allocated", 1, 'A', "-"},   {"large:chunk_allocated", 1, 'A', "-"},
       {"large:chunk_linked", 2, 'A', "-"},     {"root:store_before", 1, 'A', "-"},
       {"root:store_after", 1, 'A', "A"},       {"large:chunk_linked", 1, 'B', "A"},
       {"commit:old_root_held", 1, 'B', "B"},   {"release:children_released", 3, 'C', "BC"},
       {"final_release:decremented", 1, 'D', "CD"}, {"", 0, 'E', "E"},
   };
   std::string expect = "-";
   for (const auto& st : steps)
   {
      const auto failure = in_child([&] {
         crash::install();
         auto db = database::open(dir.path, psitri::open_mode::create_or_open, store_config());
         {
            auto rs = db->start_read_session();
            auto c  = rs->create_cursor(0);
            auto v  = c.get<std::string>("huge");
            check(expect.find(v ? (*v)[9] : '-') != std::string::npos,
                  std::string("the value found is not one the last life could have left: ") +
                      (v ? (*v)[9] : '-'));
            if (v)
               check(*v == value_of((*v)[9]), "the value read back is damaged");
            check(c.get<std::string>("small") ==
                      (v ? std::optional<std::string>("kept") : std::nullopt),
                  "the small key published with the value is not in the state the value is in");
         }
         const auto a = audit(*db, true);
         check(a.error.empty(), "audit: " + a.error);
         auto ws = db->start_write_session();
         auto rs = db->start_read_session();
         auto held = rs->create_cursor(0);  // the old value stays referenced for a while
         crash::arm(st.point, st.at);
         {
            auto tx = ws->start_transaction(0);
            tx.upsert("huge", value_of(st.write));
            tx.upsert("small", "kept");
            tx.commit();
         }
         held = rs->create_cursor(0);  // drops the last reference to the old version
         db->wait_for_compactor();
         check(std::string(st.point).empty(), "the crash point was not reached");
      });
      if (std::string(st.point).empty())
         REQUIRE(failure == "");
      else
         REQUIRE(failure == "the child died of signal 9");
      wait_for_store_lock(dir.path);
      expect = st.allowed;
   }
}

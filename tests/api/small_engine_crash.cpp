// Small Engine: death of the process at chosen points.
//
// The engine's failure model is that the PROCESS can stop at any instruction
// (killed, aborted, crashed in any thread) while the operating system keeps
// running and keeps every byte the process wrote to its shared mappings. A
// commit is the store of the new root into the mapped root slot. Power loss
// and operating-system failure are not defended against.
//
// This file is compiled with START_REF_CRASH_POINTS, which turns the
// START_REF_CRASH_POINT("name") markers in the engine into calls of a hook.
// A forked child runs writers on several roots, readers and snapshot holders,
// and kills itself with SIGKILL the n-th time a chosen marker is passed (or
// the parent kills it at a random moment). The parent then reopens the store
// with the ordinary engine and checks it against a model, audits every
// reference count and the free map against an independent walk of the
// published roots, and the next child carries on writing to the same store.
//
// The second part asks the question the model stands on: is any byte
// reachable from a published root ever changed before the next root store?
#include <psitri/api.hpp>
#include <psitri/engines/small.hpp>

#include <catch2/catch_test_macros.hpp>

#include <dlfcn.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef START_REF_CRASH_POINTS
#error "this file must be compiled with START_REF_CRASH_POINTS"
#endif

namespace
{
   using engine   = psitri::small_engine;
   using database = psitri::database_for<engine>;

   struct temporary_directory
   {
      std::filesystem::path path;
      temporary_directory()
      {
         std::string pattern =
             (std::filesystem::temp_directory_path() / "psitri-small-crash-XXXXXX").string();
         auto* result = ::mkdtemp(pattern.data());
         if (!result)
            throw std::runtime_error("mkdtemp failed");
         path = result;
      }
      ~temporary_directory()
      {
         std::error_code ignored;
         std::filesystem::remove_all(path, ignored);
      }
   };

   uint64_t env_u64(const char* name, uint64_t fallback)
   {
      const char* v = std::getenv(name);
      return v ? std::strtoull(v, nullptr, 10) : fallback;
   }

   // ── the crash hook ──────────────────────────────────────────────────────
   namespace crash
   {
      char                  point[64] = "";
      std::atomic<uint64_t> countdown{0};
      std::atomic<bool>     armed{false};

      void hook(const char* name)
      {
         if (!armed.load(std::memory_order_relaxed) || std::strcmp(name, point) != 0)
            return;
         if (countdown.fetch_sub(1, std::memory_order_relaxed) == 1)
         {
            // SIGKILL: no handlers, no destructors, no flushing; the other
            // threads are wherever they are.
            ::kill(::getpid(), SIGKILL);
            for (;;)
               ::pause();
         }
      }
      void arm(const std::string& name, uint64_t at)
      {
         std::snprintf(point, sizeof(point), "%s", name.c_str());
         countdown.store(at, std::memory_order_relaxed);
         armed.store(!name.empty(), std::memory_order_release);
         start_ref::testing::crash_hook.store(&hook, std::memory_order_release);
      }
   }  // namespace crash

   // ── flush failure injection (see the last test) ────────────────────────
   std::atomic<int> fail_msync_in{0};  // > 0: the n-th msync from now fails with EIO
}  // namespace

extern "C" int msync(void* address, size_t size, int flags)
{
   static auto real = reinterpret_cast<int (*)(void*, size_t, int)>(::dlsym(RTLD_NEXT, "msync"));
   if (fail_msync_in.load(std::memory_order_relaxed) > 0 &&
       fail_msync_in.fetch_sub(1, std::memory_order_relaxed) == 1)
   {
      errno = EIO;
      return -1;
   }
   return real(address, size, flags);
}

namespace
{
   // ── the model ───────────────────────────────────────────────────────────
   // Each root has its own sequence of transactions. Transaction k of root r
   // is a pure function of (seed, r, k) and of the root's contents before it,
   // so a child that finds the root at version k simply continues with k+1,
   // and the parent can say exactly what any version holds.
   using value_ptr = std::shared_ptr<const std::string>;
   using model_map = std::map<std::string, value_ptr>;

   constexpr const char* seq_key = "m/seq";
   constexpr const char* log_key = "m/log";
   constexpr size_t      log_versions = 256;  // versions the log remembers
   constexpr size_t      log_record   = 9;    // kind + base version

   enum txn_kind : char
   {
      plain  = 'p',  // the generated operations on the previous version
      shrink = 's',  // the store was full: removes only
      revert = 'r',  // the root replaced by an older snapshot, then the operations
   };

   struct profile
   {
      uint32_t keyspace     = 1500;
      uint32_t max_ops      = 24;
      bool     fill         = false;  // values sized to fill the store
      uint32_t chunk_every  = 40;     // root 0 writes a chunked value this often (0: never)
   };

   uint64_t mix(uint64_t a, uint64_t b, uint64_t c)
   {
      uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0xBF58476D1CE4E5B9ull) * 0x94D049BB133111EBull ^
                   (c + 0x2545F4914F6CDD1Dull) * 0xD6E8FEB86659FD93ull;
      x ^= x >> 31;
      x *= 0x9E3779B97F4A7C15ull;
      return x ^ (x >> 29);
   }

   std::string key_of(uint32_t i)
   {
      char num[16];
      std::snprintf(num, sizeof(num), "%06u", i);
      if (i % 7 == 0)
      {
         // long keys with long shared prefixes: prefix nodes of several lines
         std::string k = "L/";
         k.append(120 + (i % 3) * 260, char('p' + i % 3));
         if (i % 21 == 0)
            k.append(300, 'z');
         return k + num;
      }
      // two scrambled bytes ahead of the number: every router width
      const uint32_t h = uint32_t(i * 2654435761u) >> 16;
      std::string    k = "k/";
      k.push_back(char(h >> 8));
      k.push_back(char(h));
      return k + num;
   }

   value_ptr make_value(uint32_t root, uint64_t k, uint32_t op, size_t size)
   {
      auto v = std::make_shared<std::string>(size, char('a' + (k + op) % 26));
      const uint64_t stamp[2] = {k, (uint64_t(root) << 32) | op};
      std::memcpy(v->data(), stamp, std::min<size_t>(size, sizeof(stamp)));
      if (size > 64)  // something at the far end too
         std::memcpy(v->data() + size - 8, &k, 8);
      return v;
   }

   uint64_t seq_of(const model_map& m)
   {
      uint64_t   seq = 0;
      const auto it  = m.find(seq_key);
      if (it != m.end() && it->second->size() == 8)
         std::memcpy(&seq, it->second->data(), 8);
      return seq;
   }
   std::string log_of(const model_map& m)
   {
      const auto it = m.find(log_key);
      return it == m.end() ? std::string() : *it->second;
   }

   // Apply transaction k of `root` to the model `m` (which holds the version
   // the transaction starts from) and, when tx is not null, to the store.
   // `lineage_log` is the log of the version before k (which for a revert is
   // not the version the contents start from).
   template <class Tx>
   void apply_txn(Tx* tx, model_map& m, const std::string& lineage_log, uint64_t seed,
                  uint32_t root, uint64_t k, txn_kind kind, uint64_t base_version,
                  const profile& p)
   {
      auto put = [&](const std::string& key, value_ptr v) {
         if (tx)
            tx->upsert(key, std::string_view(*v));
         m[key] = std::move(v);
      };
      auto drop = [&](const std::string& key) {
         if (tx && tx->remove(key) != 0)
            throw std::runtime_error("model: a key the model holds is not in the store");
         m.erase(key);
      };
      auto data_key = [&](uint64_t r) -> model_map::iterator {
         // the first model key at or after a generated key that is not bookkeeping
         auto it = m.lower_bound(key_of(uint32_t(r % p.keyspace)));
         while (it != m.end() && it->first.starts_with("m/"))
            ++it;
         if (it == m.end())
         {
            it = m.begin();
            while (it != m.end() && it->first.starts_with("m/"))
               ++it;
         }
         return it;
      };
      std::mt19937_64 rng(mix(seed, root, k));
      if (kind == shrink)
      {
         for (uint32_t n = 0; n < 400; ++n)
         {
            auto it = data_key(rng());
            if (it == m.end())
               break;
            drop(std::string(it->first));
         }
      }
      else
      {
         const uint32_t ops = 1 + uint32_t(rng() % p.max_ops);
         for (uint32_t op = 0; op < ops; ++op)
         {
            const uint32_t pick = uint32_t(rng() % 100);
            const uint64_t r    = rng();
            if (p.fill ? pick < 88 : pick < 74)
            {
               size_t size;
               if (p.fill)
                  size = 300 + r % 2700;
               else if (pick < 55)
                  size = r % 91;  // inside one line with its key
               else if (pick < 70)
                  size = 100 + r % 5900;  // a leaf of several lines
               else
                  size = 9000 + r % 190000;  // the large-value arena
               put(key_of(uint32_t((r >> 20) % p.keyspace)), make_value(root, k, op, size));
            }
            else if (pick < 95)
            {
               auto it = data_key(r);
               if (it != m.end())
                  drop(std::string(it->first));
            }
            else
            {
               // a short range
               auto it = data_key(r);
               if (it == m.end())
                  continue;
               const std::string low = it->first;
               std::string       high;
               uint32_t          n = 0;
               auto              end = it;
               while (end != m.end() && n < 5 && !end->first.starts_with("m/"))
                  ++end, ++n;
               if constexpr (requires { tx->remove_range_counted(low, high); })
               {
                  if (end != m.end() && !end->first.starts_with("m/"))
                  {
                     high = end->first;
                     if (tx && tx->remove_range_counted(low, high) != n)
                        throw std::runtime_error("model: range remove count differs");
                     m.erase(it, end);
                     continue;
                  }
               }
               for (uint32_t j = 0; j < n; ++j)
               {
                  const std::string key = it->first;
                  ++it;
                  drop(key);
               }
            }
         }
         if (root == 0 && p.chunk_every && k % p.chunk_every == 7)
            put("big/chunked", make_value(root, k, 9999, (2u << 20) + 300000 + (k % 5) * 500000));
         if (root == 0 && p.chunk_every && k % p.chunk_every == 27)
            if (m.count("big/chunked"))
               drop("big/chunked");
      }
      put(seq_key, std::make_shared<std::string>(reinterpret_cast<const char*>(&k), 8));
      std::string log = lineage_log;
      log.push_back(char(kind));
      log.append(reinterpret_cast<const char*>(&base_version), 8);
      if (log.size() > log_versions * log_record)
         log.erase(0, log.size() - log_versions * log_record);
      put(log_key, std::make_shared<std::string>(std::move(log)));
   }

   template <class Cursor>
   model_map scan(Cursor& c)
   {
      model_map m;
      for (c.seek_begin(); !c.is_end(); c.next())
      {
         auto v = c.template value<std::string>();
         if (!v)
            throw std::runtime_error("scan: a key without a readable value");
         m.emplace(std::string(c.key()), std::make_shared<std::string>(std::move(*v)));
      }
      return m;
   }

   // roots: 0..3 single writers, 4 and 5 written together by one writer,
   // 8..11 the stashed snapshots of roots 0..3
   constexpr uint32_t single_roots = 4;
   constexpr uint32_t pair_a = 4, pair_b = 5;
   constexpr uint32_t stash_base = 8;

   struct life_plan
   {
      uint64_t       seed    = 1;
      profile        prof;
      engine::config cfg;
      uint32_t       writers = single_roots;  // single-root writers
      bool           pair    = true;           // the two-root writer
      uint32_t       readers = 2;
      uint32_t       txns    = 60;  // per writer, per life
      std::string    point;         // empty: not armed
      uint64_t       at = 0;
   };

   struct ack_record
   {
      uint32_t root;
      uint32_t reserved;
      uint64_t version;
   };

   [[noreturn]] void child_fail(int code, const char* what)
   {
      std::fprintf(stderr, "crash child (pid %d): %s\n", int(::getpid()), what);
      ::_exit(code);
   }

   void ack(int fd, uint32_t root, uint64_t version)
   {
      const ack_record r{root, 0, version};
      if (::write(fd, &r, sizeof(r)) != ssize_t(sizeof(r)))
         ::_exit(3);
   }

   // One single-root writer: the life of root r.
   void single_writer(database& db, const life_plan& plan, uint32_t r, int ack_fd)
   {
      auto      ws = db.start_write_session();
      model_map cur, stash;
      {
         auto rs = db.start_read_session();
         {
            auto c = rs->create_cursor(r);
            cur    = scan(c);
         }
         auto c = rs->create_cursor(stash_base + r);
         stash  = scan(c);
      }
      uint64_t k = seq_of(cur);
      ack(ack_fd, r, k);  // what the store held when this life began
      for (uint32_t n = 0; n < plan.txns; ++n)
      {
         ++k;
         if (k % 13 == 0)
         {
            // Keep the current version as a second root: the two slots then
            // share every node.
            try
            {
               auto tx = ws->start_transaction(stash_base + r);
               tx.primary().set_root(ws->get_root(r));
               tx.commit();
               stash = cur;
            }
            catch (const engine::storage_full&)
            {
            }
         }
         if (k % 11 == 3)
         {
            // a transaction that is rolled back
            model_map scratch = cur;
            try
            {
               auto tx = ws->start_transaction(r);
               apply_txn(&tx, scratch, log_of(cur), plan.seed ^ 0xdec0de, r, k, plain, 0,
                         plan.prof);
               tx.abort();
            }
            catch (const engine::storage_full&)
            {
            }
         }
         model_map next;
         bool      done = false;
         try
         {
            auto tx = ws->start_transaction(r);
            next    = cur;
            txn_kind kind = plain;
            uint64_t base = 0;
            if (k % 13 == 6 && !stash.empty())
            {
               // Write something, then throw it away by replacing the root
               // with the stashed (older) snapshot, and write on top of that.
               for (uint32_t j = 0; j < 5; ++j)
                  tx.upsert("junk/" + std::to_string(j), std::string(40 + j * 300, 'j'));
               tx.primary().set_root(ws->get_root(stash_base + r));
               next = stash;
               kind = revert;
               base = seq_of(stash);
            }
            apply_txn(&tx, next, log_of(cur), plan.seed, r, k, kind, base, plan.prof);
            tx.commit();
            done = true;
         }
         catch (const engine::storage_full&)
         {
         }
         if (!done)
         {
            // The store is full: this version removes instead.
            try
            {
               auto tx = ws->start_transaction(r);
               next    = cur;
               apply_txn(&tx, next, log_of(cur), plan.seed, r, k, shrink, 0, plan.prof);
               tx.commit();
            }
            catch (const engine::storage_full& e)
            {
               child_fail(5, e.what());  // removes must not be refused
            }
         }
         cur = std::move(next);
         ack(ack_fd, r, k);
      }
   }

   // The writer of two roots in one transaction. The roots are published one
   // after the other (docs/api/operations.md), so a stop between the two
   // stores leaves the first one transaction ahead; this writer brings the
   // second up to date before going on.
   void pair_writer(database& db, const life_plan& plan, int ack_fd)
   {
      auto      ws = db.start_write_session();
      model_map a, b;
      {
         auto rs = db.start_read_session();
         {
            auto c = rs->create_cursor(pair_a);
            a      = scan(c);
         }
         auto c = rs->create_cursor(pair_b);
         b      = scan(c);
      }
      uint64_t k = seq_of(a);
      if (seq_of(b) + 1 == k)
      {
         // the kind of transaction k is in root a's log
         const std::string la = log_of(a);
         const txn_kind    kind = txn_kind(la[la.size() - log_record]);
         auto              tx   = ws->start_transaction(pair_b);
         apply_txn(&tx, b, log_of(b), plan.seed, pair_b, k, kind, 0, plan.prof);
         tx.commit();
      }
      else if (seq_of(b) != k)
         child_fail(6, "the two roots written together are more than one transaction apart");
      ack(ack_fd, pair_a, k);
      ack(ack_fd, pair_b, k);
      for (uint32_t n = 0; n < plan.txns; ++n)
      {
         ++k;
         model_map na, nb;
         bool      done = false;
         for (txn_kind kind : {plain, shrink})
         {
            try
            {
               auto tx = ws->start_transaction(pair_a);
               auto hb = tx.open_root(pair_b);
               na = a, nb = b;
               apply_txn(&tx, na, log_of(a), plan.seed, pair_a, k, kind, 0, plan.prof);
               apply_txn(&hb, nb, log_of(b), plan.seed, pair_b, k, kind, 0, plan.prof);
               tx.commit();
               done = true;
               break;
            }
            catch (const engine::storage_full& e)
            {
               if (kind == shrink)
                  child_fail(5, e.what());
            }
         }
         if (!done)
            child_fail(5, "pair writer made no progress");
         a = std::move(na), b = std::move(nb);
         const ack_record both[2] = {{pair_a, 0, k}, {pair_b, 0, k}};
         if (::write(ack_fd, both, sizeof(both)) != ssize_t(sizeof(both)))
            ::_exit(3);
      }
   }

   [[noreturn]] void run_child(const std::filesystem::path& dir, const life_plan& plan, int ack_fd)
   {
      try
      {
         crash::arm(plan.point, plan.at);  // before open: recovery is part of a life
         auto db = database::open(dir, psitri::open_mode::create_or_open, plan.cfg);
         std::atomic<bool>        stop{false};
         std::atomic<int>         failed{0};
         std::vector<std::thread> threads, watchers;
         auto guarded = [&](auto body) {
            return [&, body] {
               try
               {
                  body();
               }
               catch (const std::exception& e)
               {
                  std::fprintf(stderr, "crash child thread: %s\n", e.what());
                  failed.store(4);
               }
            };
         };
         for (uint32_t r = 0; r < plan.writers; ++r)
            threads.emplace_back(guarded([&, r] { single_writer(*db, plan, r, ack_fd); }));
         if (plan.pair)
            threads.emplace_back(guarded([&] { pair_writer(*db, plan, ack_fd); }));
         for (uint32_t i = 0; i < plan.readers; ++i)
            watchers.emplace_back(guarded([&, i] {
               std::mt19937_64 rng(mix(plan.seed, 77, i));
               auto            rs = db->start_read_session();
               auto            ws = db->start_write_session();  // for owned snapshots
               while (!stop.load(std::memory_order_relaxed))
               {
                  const uint32_t root = uint32_t(rng() % 12);
                  if (rng() % 16 == 0)
                  {
                     // A transaction that changes nothing, through a session
                     // whose working tree is versions behind: it has to let
                     // go of the old version and take the current one.
                     auto tx = ws->start_transaction(root % 6);
                     if (rng() % 2)
                        tx.commit();
                     else
                        tx.abort();
                  }
                  else if (rng() % 4 == 0)
                  {
                     // an owned snapshot, kept across other writers' commits
                     auto     held = ws->get_root(root);
                     auto     c    = held.snapshot_cursor();
                     uint64_t n    = 0;
                     for (c.seek_begin(); !c.is_end() && n < 50; c.next())
                        n += c.value<std::string>()->size() != SIZE_MAX;
                     std::this_thread::sleep_for(std::chrono::microseconds(rng() % 2000));
                  }
                  else
                  {
                     auto     c = rs->create_cursor(root);
                     uint64_t n = 0;
                     for (c.lower_bound(key_of(uint32_t(rng() % 1500))); !c.is_end() && n < 100;
                          c.next())
                        n += c.value<std::string>()->size() != SIZE_MAX;
                     if (rng() % 8 == 0)
                        std::this_thread::sleep_for(std::chrono::microseconds(rng() % 3000));
                  }
               }
            }));
         for (auto& t : threads)
            t.join();
         stop = true;
         for (auto& t : watchers)
            t.join();
         if (failed.load())
            ::_exit(failed.load());
         db.reset();  // a clean close, with its own crash points
         ::_exit(0);
      }
      catch (const std::exception& e)
      {
         child_fail(2, e.what());
      }
      ::_exit(2);
   }

   struct life_result
   {
      int      status = 0;
      bool     any_ack[16] = {};
      uint64_t acked[16]   = {};
      bool     killed() const { return WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL; }
      bool     finished() const { return WIFEXITED(status) && WEXITSTATUS(status) == 0; }
   };

   // Run one life. kill_after_us >= 0: the parent kills it that long after
   // its first acknowledgement.
   life_result run_life(const std::filesystem::path& dir, const life_plan& plan,
                        int64_t kill_after_us = -1)
   {
      int fds[2];
      REQUIRE(::pipe(fds) == 0);
      std::fflush(nullptr);
      const pid_t pid = ::fork();
      REQUIRE(pid >= 0);
      if (pid == 0)
      {
         ::close(fds[0]);
         run_child(dir, plan, fds[1]);
      }
      ::close(fds[1]);
      life_result       r;
      std::atomic<bool> first{false};
      std::thread       killer;
      if (kill_after_us >= 0)
         killer = std::thread([&] {
            const auto t0 = std::chrono::steady_clock::now();
            while (!first.load() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(20))
               std::this_thread::sleep_for(std::chrono::microseconds(100));
            std::this_thread::sleep_for(std::chrono::microseconds(kill_after_us));
            ::kill(pid, SIGKILL);
         });
      ack_record rec;
      while (::read(fds[0], &rec, sizeof(rec)) == ssize_t(sizeof(rec)))
      {
         if (rec.root < 16)
         {
            r.any_ack[rec.root] = true;
            r.acked[rec.root]   = std::max(r.acked[rec.root], rec.version);
         }
         first.store(true);
      }
      first.store(true);
      // The killer is joined before the child is reaped: until then the pid
      // is still the child's, and a late kill cannot reach anything else.
      if (killer.joinable())
         killer.join();
      ::waitpid(pid, &r.status, 0);
      ::close(fds[0]);
      return r;
   }

   // ── the independent audit ───────────────────────────────────────────────
   // Walks every published root and counts, for every line and every grain,
   // how many references the tree holds on it. The engine's reference counts
   // and free maps must say exactly that: a count too low frees live data, a
   // count too high or a line that is neither referenced nor free is a leak.
   struct census
   {
      start_ref::arena&                      lines;
      start_ref::large_arena&                grains;
      std::unordered_map<uint32_t, uint32_t> line_refs, grain_refs;

      void check_line(uint32_t i) const { lines.check_index(i); }
      void check_large(uint32_t g) const { grains.check_index(g); }
      bool line(uint32_t i) { return ++line_refs[i] == 1; }
      void tail(uint32_t i, uint32_t n)
      {
         lines.check_index(i, n);
         for (uint32_t k = 0; k < n; ++k)
            if (++line_refs[i + k] != 1)
               throw std::runtime_error("audit: a node's tail line is used twice");
      }
      bool large(uint32_t g, uint32_t units)
      {
         grains.check_index(g, units);
         if (++grain_refs[g] != 1)
            return false;
         for (uint32_t k = 1; k < units; ++k)
            if (++grain_refs[g + k] != 1)
               throw std::runtime_error("audit: a large value's grain is used twice");
         return true;
      }
      void leaf_value(std::string_view slot)
      {
         namespace lv = start_ref::large_value_detail;
         if (slot.size() < 2 || !lv::chunked(slot))
            return;
         const lv::head_view head(slot);
         for (size_t i = 0; i < head.entries().size(); ++i)
            large(head.entries()[i].grain, lv::units(head.length(i)));
      }
      [[noreturn]] void unsupported() { throw std::runtime_error("audit: unsupported node"); }
   };

   census walk_roots(database& db)
   {
      census                   c{db.store(), db.large_store(), {}, {}};
      start_ref::kv::engine_t  view{start_ref::part_alloc<start_ref::kv::bval>(&db.store())};
      for (uint32_t i = 0; i < start_ref::kv::kMaxRoots; ++i)
      {
         const uint64_t raw = db.store().get_root(i);
         if (raw)
            view.recover_mark(start_ref::kv::decode_root(raw), c);
      }
      return c;
   }

   struct audit_result
   {
      uint64_t lines = 0, grains = 0;
   };
   // Only on a store with no session, no snapshot and nothing queued.
   template <class Arena>
   uint64_t audit_arena(Arena& a, const std::unordered_map<uint32_t, uint32_t>& expect,
                        const char* what, uint64_t held_unreferenced)
   {
      uint64_t       used = 0, unreferenced = 0;
      const uint32_t per  = start_ref::lines_per_segment;
      for (uint32_t s = 0; s < a.num_segments(); ++s)
         for (uint32_t l = 0; l < per; ++l)
         {
            const uint32_t i    = s * per + l;
            const auto     it   = expect.find(i);
            const uint32_t want = it == expect.end() ? 0 : it->second;
            const bool     free = a.freeset_of(s).is_free(l);
            const uint32_t have = a.ref(i);
            if (want)
            {
               ++used;
               if (free || have != want)
               {
                  INFO(what << " " << i << ": the roots hold " << want
                            << " references, the store says " << have << (free ? ", free" : ""));
                  REQUIRE(have == want);
                  REQUIRE_FALSE(free);
               }
            }
            else if (!free)
               ++unreferenced;
            else if (have != 0)
            {
               INFO(what << " " << i << " is free with a count of " << have);
               REQUIRE(have == 0);
            }
         }
      INFO(what << ": " << unreferenced << " units neither referenced nor free");
      REQUIRE(unreferenced == held_unreferenced);
      REQUIRE(expect.size() == used);
      return used;
   }
   audit_result audit(database& db)
   {
      db.flush_releases();
      const census c = walk_roots(db);
      audit_result r;
      // at the ceiling the reserve holds lines out of the map without counting them
      r.lines  = audit_arena(db.store(), c.line_refs, "line", db.store().reserve_free_lines());
      r.grains = audit_arena(db.large_store(), c.grain_refs, "grain", 0);
      const auto u = db.usage();
      REQUIRE(u.used_bytes == r.lines * start_ref::line_bytes);
      REQUIRE(u.large_used_bytes == r.grains * uint64_t(start_ref::large_grain_bytes));
      return r;
   }

   // ── the parent's knowledge of the store ─────────────────────────────────
   struct root_history
   {
      uint64_t                      version = 0;
      std::map<uint64_t, model_map> versions;  // the last few, by version
   };
   struct store_model
   {
      root_history roots[6];
      uint64_t     torn_pairs = 0, verified = 0;
      store_model()
      {
         for (auto& r : roots)
            r.versions[0] = {};
      }
   };

   void require_equal(const model_map& found, const model_map& want, uint32_t root)
   {
      if (found.size() == want.size())
      {
         auto a = found.begin();
         auto b = want.begin();
         for (; a != found.end(); ++a, ++b)
            if (a->first != b->first || *a->second != *b->second)
               break;
         if (a == found.end())
            return;
         INFO("root " << root << ": first difference at key '" << a->first.substr(0, 40)
                      << "' (model key '" << b->first.substr(0, 40) << "'), value sizes "
                      << a->second->size() << " and " << b->second->size());
         REQUIRE(false);
      }
      INFO("root " << root << " holds " << found.size() << " keys, the model " << want.size());
      REQUIRE(found.size() == want.size());
   }

   // Open the store, check every root against the model and audit the
   // counts. `lives` are the lives run since the last verification.
   void verify(const std::filesystem::path& dir, const life_plan& plan, store_model& model,
               const std::vector<life_result>& lives, audit_result* audited = nullptr)
   {
      auto db = database::open(dir, psitri::open_mode::create_or_open, plan.cfg);
      auto rs = db->start_read_session();
      // what was acknowledged: the highest version any of the lives reported
      uint64_t acked[16] = {};
      for (const auto& life : lives)
         for (uint32_t r = 0; r < 16; ++r)
            acked[r] = std::max(acked[r], life.acked[r]);
      uint64_t found_version[6] = {};
      for (uint32_t r = 0; r < 6; ++r)
      {
         model_map found;
         {
            auto c = rs->create_cursor(r);
            found  = scan(c);
         }
         root_history&  h = model.roots[r];
         const uint64_t k = seq_of(found);
         found_version[r] = k;
         INFO("root " << r << ": found version " << k << ", acknowledged " << acked[r]
                      << ", last verified " << h.version);
         // Everything acknowledged is there; at most the one transaction
         // that was in flight is ahead of it, wholly.
         REQUIRE(k >= acked[r]);
         REQUIRE(k >= h.version);
         REQUIRE(k <= std::max(acked[r], h.version) + 1);
         REQUIRE(k - h.version < log_versions);
         const std::string log = log_of(found);
         for (uint64_t v = h.version + 1; v <= k; ++v)
         {
            REQUIRE(log.size() >= (k - v + 1) * log_record);
            const size_t   at = log.size() - (k - v + 1) * log_record;
            const txn_kind kind = txn_kind(log[at]);
            uint64_t       base;
            std::memcpy(&base, log.data() + at + 1, 8);
            model_map         m    = h.versions.at(v - 1);
            const std::string prev = log_of(m);
            if (kind == revert)
            {
               REQUIRE(h.versions.count(base));
               m = h.versions.at(base);
            }
            else
               REQUIRE((kind == plain || kind == shrink));
            apply_txn(static_cast<engine::transaction*>(nullptr), m, prev, plan.seed, r, v, kind,
                      base, plan.prof);
            h.versions[v] = std::move(m);
         }
         h.version = k;
         while (h.versions.size() > 48)
            h.versions.erase(h.versions.begin());
         require_equal(found, h.versions.at(k), r);
      }
      // The two roots written together: published one after the other.
      REQUIRE(found_version[pair_a] >= found_version[pair_b]);
      REQUIRE(found_version[pair_a] <= found_version[pair_b] + 1);
      model.torn_pairs += found_version[pair_a] != found_version[pair_b];
      // The stashed snapshots: each is some version of its root, whole.
      for (uint32_t r = 0; r < single_roots; ++r)
      {
         model_map found;
         {
            auto c = rs->create_cursor(stash_base + r);
            found  = scan(c);
         }
         if (found.empty())
            continue;
         const uint64_t k = seq_of(found);
         INFO("stash of root " << r << " holds version " << k);
         REQUIRE(k <= model.roots[r].version);
         REQUIRE(model.roots[r].versions.count(k));
         require_equal(found, model.roots[r].versions.at(k), stash_base + r);
      }
      rs.reset();
      const audit_result a = audit(*db);
      if (audited)
         *audited = a;
      ++model.verified;
   }

   // Every marker in the engine. A run reports which of them it reached.
   const char* const all_points[] = {
       "open.before_unclean_mark", "open.after_unclean_mark", "open.before_trim",
       "open.after_trim", "large.open.before_unclean_mark", "large.open.after_unclean_mark",
       "large.open.before_trim", "large.open.after_trim",
       "recover.begin", "recover.reset_map", "recover.before_reset_counts",
       "recover.between_resets", "large.recover.reset_map", "large.recover.before_reset_counts",
       "recover.mark", "large.recover.mark", "recover.after_root", "recover.finish_map",
       "recover.between_ends", "large.recover.finish_map", "recover.done",
       "commit.before_flush", "commit.after_flush", "root.before_store", "root.after_store",
       "commit.after_root", "commit.before_drain", "commit.after_drain", "commit.published",
       "retire.before_release", "retire.after_release", "releaser.before_release",
       "releaser.after_release", "refresh.before_release",
       "abort.before_release", "abort.after_release",
       "set_root.before_adopt", "set_root.after_adopt", "set_root.release_held",
       "alloc", "free", "large.alloc", "large.free",
       "grow.before_data", "grow.after_data", "grow.after_map", "grow.after_counts",
       "grow.before_count", "grow.after_count", "large.grow.before_data", "large.grow.after_data",
       "large.grow.after_map", "large.grow.after_counts", "large.grow.before_count",
       "large.grow.after_count",
       "segment.claim", "segment.sweep_claim", "segment.release", "segment.resume",
       "large.segment.resume", "reserve.draw", "reserve.repay",
       "close.after_releaser_stop", "close.release_retired", "close.before_arenas",
       "close.before_reserve_return", "close.after_reserve_return", "close.between_map_syncs",
       "close.before_clean_mark", "close.after_clean_mark", "large.close.before_reserve_return",
       "large.close.between_map_syncs", "large.close.before_clean_mark",
       "large.close.after_clean_mark"};

   struct point_state
   {
      uint64_t next_at = 1, deaths = 0, misses = 0;
   };

   struct campaign_stats
   {
      uint64_t lives = 0, point_deaths = 0, timed_kills = 0, clean = 0, recovery_deaths = 0;
      std::map<std::string, point_state> points;
   };

   // Crash the same store again and again; verify after every one to three lives.
   void campaign(life_plan plan, uint64_t cycles, uint64_t seed, const char* name,
                 std::vector<std::string> must_reach, bool empty_at_end = true)
   {
      temporary_directory tmp;
      store_model         model;
      campaign_stats      stats;
      std::mt19937_64     rng(seed);
      std::vector<std::string> names(std::begin(all_points), std::end(all_points));
      for (const auto& n : names)
         stats.points[n];
      const auto t0 = std::chrono::steady_clock::now();
      size_t     next_point = 0;
      audit_result last_audit;
      for (uint64_t cycle = 0; cycle < cycles; ++cycle)
      {
         std::vector<life_result> lives;
         std::string              story;
         const uint32_t           chain = 1 + uint32_t(rng() % 3);
         for (uint32_t link = 0; link < chain; ++link)
         {
            plan.seed  = seed;
            plan.point.clear();
            int64_t        kill_us = -1;
            const uint32_t pick    = uint32_t(rng() % 100);
            const bool     unclean = !database::closed_cleanly(tmp.path);
            if (link > 0 && unclean && pick < 70)
            {
               // die again while the store is being recovered
               static const char* const recovery[] = {
                   "recover.begin", "recover.reset_map", "recover.before_reset_counts",
                   "recover.between_resets", "large.recover.reset_map", "recover.mark",
                   "recover.mark", "recover.mark", "large.recover.mark", "recover.after_root",
                   "recover.finish_map", "recover.between_ends", "large.recover.finish_map",
                   "recover.done", "open.after_unclean_mark", "open.before_trim",
                   "open.after_trim"};
               plan.point = recovery[rng() % std::size(recovery)];
            }
            else if (pick < 62)
               plan.point = names[next_point++ % names.size()];
            else if (pick < 90)
               kill_us = int64_t(rng() % 40000);
            if (!plan.point.empty())
            {
               auto& p = stats.points[plan.point];
               // mostly walk outwards from the first hit; sometimes anywhere below
               plan.at = rng() % 4 ? p.next_at : 1 + rng() % p.next_at;
            }
            const life_result life = run_life(tmp.path, plan, kill_us);
            ++stats.lives;
            INFO(name << " cycle " << cycle << " link " << link << " point '" << plan.point
                      << "' at " << plan.at << " kill_us " << kill_us << " status "
                      << life.status);
            REQUIRE((life.killed() || life.finished()));
            if (!plan.point.empty())
            {
               auto& p = stats.points[plan.point];
               if (life.killed())
               {
                  ++p.deaths;
                  ++stats.point_deaths;
                  stats.recovery_deaths += plan.point.find("recover.") != std::string::npos;
                  p.next_at = p.next_at * 2 + rng() % 3;
               }
               else
               {
                  ++p.misses;
                  p.next_at = 1;
               }
            }
            else if (kill_us >= 0)
               stats.timed_kills += life.killed();
            stats.clean += life.finished();
            lives.push_back(life);
            story += " [" + (plan.point.empty() ? std::string(kill_us >= 0 ? "timed kill" : "unarmed")
                                                : plan.point + "@" + std::to_string(plan.at)) +
                     (life.killed() ? ": killed]" : ": finished]");
            if (life.finished())
               break;  // a clean close: verify what a clean open finds
         }
         INFO(name << " cycle " << cycle << ":" << story);
         verify(tmp.path, plan, model, lives, &last_audit);
      }
      uint64_t versions = 0;
      for (const auto& r : model.roots)
         versions += r.version;
      const double secs =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("crash campaign %s: %llu lives (%llu died at a marker, %llu of those inside "
                  "recovery or open; %llu killed at a random time; %llu closed cleanly), %llu "
                  "verifications, %llu transactions kept, two-root commits found half "
                  "published %llu times; store %.1f MiB lines, %.1f MiB large; %.1fs\n",
                  name, (unsigned long long)stats.lives, (unsigned long long)stats.point_deaths,
                  (unsigned long long)stats.recovery_deaths,
                  (unsigned long long)stats.timed_kills, (unsigned long long)stats.clean,
                  (unsigned long long)model.verified, (unsigned long long)versions,
                  (unsigned long long)model.torn_pairs, last_audit.lines * 128 / 1048576.0,
                  last_audit.grains * 32768 / 1048576.0, secs);
      std::string reached, never;
      for (const auto& [n, p] : stats.points)
         (p.deaths ? reached : never) += " " + n + (p.deaths ? "x" + std::to_string(p.deaths) : "");
      std::printf("  died at:%s\n  never reached:%s\n", reached.c_str(), never.c_str());
      std::fflush(stdout);
      for (const auto& n : must_reach)
      {
         INFO("marker " << n << " was never reached in campaign " << name);
         REQUIRE(stats.points[n].deaths > 0);
      }
      if (!empty_at_end)
         return;
      // Nothing is left behind: remove every key of every root, and the
      // arenas drain.
      auto db = database::open(tmp.path, psitri::open_mode::open_existing, plan.cfg);
      {
         auto ws = db->start_write_session();
         for (uint32_t r = 0; r < 16; ++r)
         {
            auto tx = ws->start_transaction(r);
            tx.primary().set_root({});
            tx.commit();
         }
      }
      db->flush_releases();
      const auto u = db->usage();
      INFO("after dropping every root: " << u.used_bytes << " bytes, large "
                                         << u.large_used_bytes);
      REQUIRE(u.used_bytes <= 64 * 128);
      REQUIRE(u.large_used_bytes == 0);
   }

   life_plan base_plan(psitri::sync_mode sync, uint32_t max_segments)
   {
      life_plan p;
      p.cfg.sync_mode    = sync;
      p.cfg.max_segments = max_segments;
      return p;
   }
}  // namespace

// Writers on six roots (two of them written together), stashed snapshots that
// share every node with a live root, reverts to old snapshots, rolled-back
// transactions, values in one line, several lines, the large-value arena and
// in chunks, readers and owned snapshots: killed at every marker and at
// random times, recovered (and killed again while recovering), verified and
// continued, on one store. PSITRI_SMALL_CRASH_CYCLES scales it.
TEST_CASE("Small Engine: a process killed at any marked step loses nothing",
          "[small][crash][steps]")
{
   const uint64_t cycles = env_u64("PSITRI_SMALL_CRASH_CYCLES", 260);
   campaign(base_plan(psitri::sync_mode::none, 1024), cycles,
            env_u64("PSITRI_SMALL_CRASH_SEED", 20261128), "none",
            {"root.before_store", "root.after_store", "commit.after_root", "commit.before_flush",
             "commit.published", "recover.mark", "recover.begin", "recover.done", "alloc", "free",
             "large.alloc", "large.free", "abort.before_release", "set_root.after_adopt",
             "retire.before_release", "close.before_clean_mark", "close.after_clean_mark",
             "releaser.before_release", "refresh.before_release"});
}

// The same under a mode that flushes at commit: nothing about recovery may
// depend on it.
TEST_CASE("Small Engine: a process killed at any marked step loses nothing when commits flush",
          "[small][crash][flush]")
{
   const uint64_t cycles = env_u64("PSITRI_SMALL_CRASH_CYCLES", 260) / 4;
   auto           plan   = base_plan(psitri::sync_mode::msync_sync, 1024);
   plan.txns             = 30;
   campaign(plan, cycles, env_u64("PSITRI_SMALL_CRASH_SEED", 20261128) + 1, "msync_sync",
            {"root.before_store", "root.after_store", "recover.mark"});
}

// A store that grows to its ceiling and stays there: growth in two steps,
// refused transactions that roll back, removes that draw on the reserve,
// segments taken from idle sessions, all with deaths in the middle.
TEST_CASE("Small Engine: a process killed while the store grows or is full loses nothing",
          "[small][crash][ceiling]")
{
   const uint64_t cycles = env_u64("PSITRI_SMALL_CRASH_CYCLES", 260) / 2;
   auto           plan   = base_plan(psitri::sync_mode::none, 6);
   plan.prof.fill        = true;
   plan.prof.keyspace    = 30000;
   plan.prof.chunk_every = 0;
   plan.writers          = 3;
   plan.txns             = 50;
   campaign(plan, cycles, env_u64("PSITRI_SMALL_CRASH_SEED", 20261128) + 2, "ceiling",
            {"grow.after_data", "grow.before_count", "grow.after_count", "segment.claim",
             "reserve.draw", "root.after_store", "recover.mark"},
            /*empty_at_end=*/true);
}

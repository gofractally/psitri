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
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
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

   // ── the store's file locks ──────────────────────────────────────────────
   // Every file of a store is locked (flock, exclusive, non-blocking) by the
   // process that has it open. A killed process is reaped before the kernel
   // has necessarily closed its files, so an open straight after waitpid can
   // still find a lock held (seen on Linux as EWOULDBLOCK from open).
   const char* const store_files[] = {"data.part",       "free.part",       "refs.part",
                                      "meta.part",       "large/data.part", "large/free.part",
                                      "large/refs.part", "large/meta.part"};
   // -1: no such file; 0: not locked; otherwise the errno of the flock.
   int lock_state(const std::filesystem::path& file)
   {
      const int fd = ::open(file.c_str(), O_RDONLY);
      if (fd < 0)
         return -1;
      int err = 0;
      if (::flock(fd, LOCK_EX | LOCK_NB) == 0)
         ::flock(fd, LOCK_UN);
      else
         err = errno;
      ::close(fd);
      return err;
   }
   void wait_for_store_locks(const std::filesystem::path& dir)
   {
      for (const char* name : store_files)
         for (int attempt = 0; lock_state(dir / name) > 0; ++attempt)
         {
            if (attempt == 20000)
               throw std::runtime_error("the lock of a dead process on " + (dir / name).string() +
                                        " was never released");
            ::usleep(1000);
         }
   }
   // For a child whose open failed: which file, and what holds it.
   void report_open_failure(const std::filesystem::path& dir, const std::exception& e)
   {
      std::fprintf(stderr, "crash child (pid %d): open of %s failed: %s\n", int(::getpid()),
                   dir.c_str(), e.what());
      for (const char* name : store_files)
      {
         const int st = lock_state(dir / name);
         std::fprintf(stderr, "   %-16s %s%s\n", name,
                      st < 0 ? "absent" : st == 0 ? "present, not locked" : "flock(LOCK_EX|LOCK_NB): ",
                      st > 0 ? std::strerror(st) : "");
      }
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
      shrink = 's',  // the store was full: nothing but the bookkeeping, after removes
      revert = 'r',  // the root replaced by an older snapshot, then the operations
   };

   struct profile
   {
      uint32_t keyspace     = 1500;
      uint32_t max_ops      = 24;
      bool     fill         = false;  // values sized to fill the store
      uint32_t chunk_every  = 40;     // root 0 writes a chunked value this often (0: never)
      bool     stash        = true;   // keep and put back snapshots through other root slots
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
      if (kind != shrink)
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

   // The store refused transaction k as full: a transaction of removes only
   // (it changes no bookkeeping, so the root stays at version k - 1 with
   // fewer keys). `step` counts such transactions before version k. A
   // refused remove is what the engine promises not to do at its ceiling.
   template <class Tx>
   void apply_removes(Tx* tx, model_map& m, uint64_t seed, uint32_t root, uint64_t k,
                      uint32_t step, const profile& p)
   {
      std::mt19937_64 rng(mix(seed ^ 0x5eed0fu, root, k * 16 + step));
      for (uint32_t n = 0; n < 400; ++n)
      {
         auto it = m.lower_bound(key_of(uint32_t(rng() % p.keyspace)));
         while (it != m.end() && it->first.starts_with("m/"))
            ++it;
         if (it == m.end())
            for (it = m.begin(); it != m.end() && it->first.starts_with("m/");)
               ++it;
         if (it == m.end())
            return;
         if (tx && tx->remove(it->first) != 0)
            throw std::runtime_error("model: a key the model holds is not in the store");
         m.erase(it);
      }
   }
   constexpr uint32_t max_remove_steps = 64;

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
      // A second session on the same root, used now and then: each time it
      // finds the root moved on and has to let go of the version it held.
      // (Not at the ceiling, where a version held back is storage held back.)
      auto      other = plan.prof.fill ? nullptr : db.start_write_session();
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
      ack(ack_fd, stash_base + r, seq_of(stash));
      for (uint32_t n = 0; n < plan.txns; ++n)
      {
         ++k;
         if (plan.prof.stash && k % 13 == 0)
         {
            // Keep the current version as a second root: the two slots then
            // share every node.
            try
            {
               auto tx = ws->start_transaction(stash_base + r);
               tx.primary().set_root(ws->get_root(r));
               tx.commit();
               stash = cur;
               ack(ack_fd, stash_base + r, seq_of(stash));
            }
            catch (const engine::storage_full&)
            {
            }
         }
         if (other && k % 7 == 5)
         {
            auto tx = other->start_transaction(r);
            tx.abort();
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
            if (plan.prof.stash && k % 13 == 6 && !stash.empty())
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
         // The store is full. Transactions of removes only, each followed by
         // an attempt at a version that writes just its bookkeeping (which
         // is an ordinary write and may be refused in its turn).
         for (uint32_t step = 0; !done; ++step)
         {
            if (step == max_remove_steps)
               child_fail(8, "still no room for two small values after 64 batches of removes");
            // What the removes freed comes back when the version before them
            // is released, and a reader or snapshot on that version holds it
            // for as long as it stays there: give them time to move on.
            if (step)
               std::this_thread::sleep_for(std::chrono::milliseconds(2 * step));
            try
            {
               auto tx = ws->start_transaction(r);
               apply_removes(&tx, cur, plan.seed, r, k, step, plan.prof);
               tx.commit();
            }
            catch (const engine::storage_full& e)
            {
               child_fail(5, (std::string("a transaction of removes only was refused: ") +
                              e.what()).c_str());
            }
            try
            {
               auto tx = ws->start_transaction(r);
               next    = cur;
               apply_txn(&tx, next, log_of(cur), plan.seed, r, k, shrink, step + 1, plan.prof);
               tx.commit();
               done = true;
            }
            catch (const engine::storage_full&)
            {
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
         try
         {
            auto tx = ws->start_transaction(pair_a);
            auto hb = tx.open_root(pair_b);
            na = a, nb = b;
            apply_txn(&tx, na, log_of(a), plan.seed, pair_a, k, plain, 0, plan.prof);
            apply_txn(&hb, nb, log_of(b), plan.seed, pair_b, k, plain, 0, plan.prof);
            tx.commit();
            done = true;
         }
         catch (const engine::storage_full& e)
         {
            // the campaign at the ceiling runs without this writer
            child_fail(9, e.what());
         }
         if (!done)
            child_fail(9, "pair writer made no progress");
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
         std::shared_ptr<database> db;
         try
         {
            db = database::open(dir, psitri::open_mode::create_or_open, plan.cfg);
         }
         catch (const std::exception& e)
         {
            report_open_failure(dir, e);
            ::_exit(10);
         }
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
               // A write session keeps a reference to the version of each
               // root it last worked on, for as long as it lives or until
               // its next transaction on that root: an idle one keeps that
               // whole version's storage from being freed. Elsewhere that is
               // wanted here (the session below falls versions behind); on a
               // store kept at its ceiling it would turn every remove into
               // one that frees nothing, so there the session is made anew
               // each time round.
               auto            kept_session = db->start_write_session();
               while (!stop.load(std::memory_order_relaxed))
               {
                  auto ws = plan.prof.fill ? db->start_write_session() : kept_session;
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
                     // A snapshot is one version: its keys come in order, every
                     // value reads, and its sequence number does not change
                     // while other writers commit.
                     auto              c      = rs->create_cursor(root);
                     const auto        before = c.get<std::string>(seq_key);
                     uint64_t          n      = 0;
                     std::string       last;
                     for (c.lower_bound(key_of(uint32_t(rng() % 1500))); !c.is_end() && n < 100;
                          c.next(), ++n)
                     {
                        const std::string key(c.key());
                        if (n && !(last < key))
                           throw std::runtime_error("reader: keys of a snapshot out of order");
                        if (!c.value<std::string>())
                           throw std::runtime_error("reader: a key of a snapshot has no value");
                        last = key;
                     }
                     if (c.get<std::string>(seq_key) != before)
                        throw std::runtime_error("reader: a snapshot changed under its reader");
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
      uint64_t acked[16]   = {};  // the highest version acknowledged, per root
      uint64_t first[16]   = {};  // the first: what the life found in the store
      bool     hung        = false;  // killed by the parent's deadline
      bool     killed() const { return WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL; }
      bool     finished() const { return WIFEXITED(status) && WEXITSTATUS(status) == 0; }
   };

   // Run one life. kill_after_us >= 0: the parent kills it that long after
   // its first acknowledgement.
   life_result run_life(const std::filesystem::path& dir, const life_plan& plan,
                        int64_t kill_after_us = -1, bool from_start = false)
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
            // from_start: the clock runs from the fork, so the death can
            // fall inside open and recovery
            const auto t0 = std::chrono::steady_clock::now();
            while (!from_start && !first.load() &&
                   std::chrono::steady_clock::now() - t0 < std::chrono::seconds(20))
               std::this_thread::sleep_for(std::chrono::microseconds(100));
            std::this_thread::sleep_for(std::chrono::microseconds(kill_after_us));
            ::kill(pid, SIGKILL);
         });
      // A child that stops making progress is killed, not waited for: the
      // deadline is far beyond what a life takes.
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::seconds(env_u64("PSITRI_SMALL_CRASH_LIFE_SECONDS", 600));
      ack_record rec;
      for (;;)
      {
         pollfd waiting{fds[0], POLLIN, 0};
         if (::poll(&waiting, 1, 1000) == 0)
         {
            if (std::chrono::steady_clock::now() < deadline)
               continue;
            r.hung = true;
            ::kill(pid, SIGKILL);
            break;
         }
         if (::read(fds[0], &rec, sizeof(rec)) != ssize_t(sizeof(rec)))
            break;
         if (rec.root < 16)
         {
            if (!r.any_ack[rec.root])
               r.first[rec.root] = rec.version;
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
      wait_for_store_locks(dir);
      INFO("a child did not finish within its deadline; armed marker '"
           << plan.point << "' at " << plan.at);
      REQUIRE_FALSE(r.hung);
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

   // `versions`: walk these root words instead of the sixteen root slots.
   census walk_roots(database& db, const std::vector<uint64_t>* versions = nullptr)
   {
      census                   c{db.store(), db.large_store(), {}, {}};
      start_ref::kv::engine_t  view{start_ref::part_alloc<start_ref::kv::bval>(&db.store())};
      std::vector<uint64_t>    slots;
      if (!versions)
      {
         for (uint32_t i = 0; i < start_ref::kv::kMaxRoots; ++i)
            slots.push_back(db.store().get_root(i));
         versions = &slots;
      }
      for (const uint64_t raw : *versions)
         if (raw)
            view.recover_mark(start_ref::kv::decode_root(raw), c);
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
      uint64_t     torn_pairs = 0, verified = 0, refused = 0, reverted = 0, removes_ahead = 0;
      // The highest version ever acknowledged or verified, per root slot.
      uint64_t     floor[16] = {};
      store_model()
      {
         for (auto& r : roots)
            r.versions[0] = {};
      }
   };

   // A life's first acknowledgement for a root is the version it found in the
   // store. If that is below what an earlier life had acknowledged, the store
   // lost an acknowledged transaction, whatever the life went on to commit
   // again. Returns what was lost, or nothing.
   std::string regression(const life_result& life, const uint64_t floor[16])
   {
      for (uint32_t r = 0; r < 16; ++r)
         if (life.any_ack[r] && life.first[r] < floor[r])
            return "root " + std::to_string(r) + " came back at version " +
                   std::to_string(life.first[r]) + " after version " + std::to_string(floor[r]) +
                   " had been acknowledged";
      return {};
   }
   // Run a life of a store the parent keeps a model of.
   life_result live(const std::filesystem::path& dir, const life_plan& plan, store_model& model,
                    int64_t kill_after_us = -1, bool from_start = false)
   {
      const life_result life = run_life(dir, plan, kill_after_us, from_start);
      const std::string lost = regression(life, model.floor);
      INFO(lost);
      REQUIRE(lost.empty());
      for (uint32_t r = 0; r < 16; ++r)
         model.floor[r] = std::max(model.floor[r], life.acked[r]);
      return life;
   }

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
            model.refused += kind == shrink;
            model.reverted += kind == revert;
            if (kind == revert)
            {
               REQUIRE(h.versions.count(base));
               m = h.versions.at(base);
            }
            else
            {
               REQUIRE((kind == plain || kind == shrink));
               REQUIRE(base <= max_remove_steps);  // the removes-only transactions before it
               REQUIRE((kind == shrink) == (base > 0));
               for (uint32_t step = 0; step < base; ++step)
                  apply_removes(static_cast<engine::transaction*>(nullptr), m, plan.seed, r, v,
                                step, plan.prof);
            }
            apply_txn(static_cast<engine::transaction*>(nullptr), m, prev, plan.seed, r, v, kind,
                      base, plan.prof);
            h.versions[v] = std::move(m);
         }
         h.version      = k;
         model.floor[r] = std::max(model.floor[r], k);
         while (h.versions.size() > 48)
            h.versions.erase(h.versions.begin());
         // Removes-only transactions of the next version may have been
         // committed (each one whole) before the process died: the root is
         // then version k with those keys gone, and the next life goes on
         // from exactly that.
         if (plan.prof.fill)
         {
            model_map m = h.versions.at(k);
            for (uint32_t step = 0; step < max_remove_steps && found.size() != m.size(); ++step)
            {
               apply_removes(static_cast<engine::transaction*>(nullptr), m, plan.seed, r, k + 1,
                             step, plan.prof);
               if (found.size() == m.size())
               {
                  h.versions[k] = m;
                  model.removes_ahead += 1;
               }
            }
         }
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
         const uint64_t k = seq_of(found);
         INFO("stash of root " << r << " holds version " << k << ", acknowledged "
                               << acked[stash_base + r]);
         // a stash whose commit returned is there, and never an older one
         REQUIRE(k >= acked[stash_base + r]);
         REQUIRE(k >= model.floor[stash_base + r]);
         model.floor[stash_base + r] = k;
         if (found.empty())
            continue;
         REQUIRE(plan.prof.stash);
         REQUIRE(k <= model.roots[r].version);
         REQUIRE(model.roots[r].versions.count(k));
         require_equal(found, model.roots[r].versions.at(k), stash_base + r);
      }
      // Root slots nothing writes to stay empty.
      for (uint32_t r : {6u, 7u, 12u, 13u, 14u, 15u})
      {
         auto c = rs->create_cursor(r);
         INFO("root " << r << " should be empty");
         REQUIRE_FALSE(c.seek_begin());
      }
      if (!plan.pair)
      {
         REQUIRE(found_version[pair_a] == 0);
         REQUIRE(found_version[pair_b] == 0);
      }
      rs.reset();
      const audit_result a = audit(*db);
      if (audited)
         *audited = a;
      ++model.verified;
   }

   // Every marker compiled into the engine: the macro registers each one
   // before main runs (start_ref/crash_point.hpp), so this list cannot fall
   // behind the engine.
   std::vector<std::string> engine_markers()
   {
      std::vector<std::string> names(start_ref::testing::markers().begin(),
                                     start_ref::testing::markers().end());
      std::sort(names.begin(), names.end());
      names.erase(std::unique(names.begin(), names.end()), names.end());
      return names;
   }

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
   //
   // `first` are markers every life passes (or, for recovery and open, every
   // open of a store that was not closed): the campaign begins by dying at
   // each of them once, so that those deaths do not depend on which marker
   // the random part happens to arm while the store is in the right state.
   // Which of the other markers a run reached is printed, not required.
   // `until_refused`: go on until some transaction has been refused because
   // the store was full.
   void campaign(life_plan plan, uint64_t cycles, uint64_t seed, const char* name,
                 std::vector<std::string> first, bool until_refused = false)
   {
      temporary_directory tmp;
      store_model         model;
      campaign_stats      stats;
      std::mt19937_64     rng(seed);
      const std::vector<std::string> names = engine_markers();
      for (const auto& n : names)
         stats.points[n];
      const auto t0 = std::chrono::steady_clock::now();
      size_t     next_point = 0;
      INFO("campaign " << name << ", seed " << seed << " (PSITRI_SMALL_CRASH_SEED sets the base)");
      audit_result last_audit;
      plan.seed = seed;
      for (const auto& marker : first)
      {
         std::vector<life_result> lives;
         if (marker.find("recover.") != std::string::npos ||
             marker.find("open.") != std::string::npos)
         {
            // these need a store that was not closed
            plan.point = "commit.published";
            plan.at    = 7;
            lives.push_back(live(tmp.path, plan, model));
            REQUIRE(lives.back().killed());
         }
         plan.point = marker;
         plan.at    = 1;
         lives.push_back(live(tmp.path, plan, model));
         INFO(name << ": no death at marker " << marker << ", status " << lives.back().status);
         REQUIRE(lives.back().killed());
         ++stats.lives, ++stats.point_deaths;
         ++stats.points[marker].deaths;
         verify(tmp.path, plan, model, lives, &last_audit);
      }
      for (uint64_t cycle = 0;
           cycle < cycles || (until_refused && !model.refused && cycle < 4 * cycles + 400); ++cycle)
      {
         std::vector<life_result> lives;
         std::string              story;
         const uint32_t           chain = 1 + uint32_t(rng() % 3);
         for (uint32_t link = 0; link < chain; ++link)
         {
            plan.seed  = seed;
            plan.point.clear();
            int64_t        kill_us = -1;
            bool           from_start = false;
            const uint32_t pick    = uint32_t(rng() % 100);
            const bool     unclean = !database::closed_cleanly(tmp.path);
            bool           certain = false;
            if (link > 0 && plan.prof.fill)
            {
               // At the ceiling a life that writes is verified before the
               // next one writes (see removes_ahead), so the further links
               // of a chain are deaths inside recovery and nothing else:
               // markers every recovery passes, at their first passing.
               if (!unclean)
                  break;
               static const char* const always[] = {
                   "recover.begin", "recover.reset_map", "recover.before_reset_counts",
                   "recover.between_resets", "recover.mark", "recover.after_root",
                   "recover.finish_map", "recover.between_ends", "recover.done",
                   "open.after_unclean_mark"};
               plan.point = always[rng() % std::size(always)];
               certain    = true;
            }
            else if (link > 0 && unclean && pick < 70)
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
            {
               kill_us    = int64_t(rng() % 40000);
               from_start = rng() % 2;  // half of them can fall inside open and recovery
            }
            if (!plan.point.empty())
            {
               auto& p = stats.points[plan.point];
               // mostly walk outwards from the first hit; sometimes anywhere below
               plan.at = rng() % 4 ? p.next_at : 1 + rng() % p.next_at;
               if (certain)
                  plan.at = plan.point == "recover.mark" ? 1 + rng() % 50 : 1;
            }
            const life_result life = live(tmp.path, plan, model, kill_us, from_start);
            ++stats.lives;
            INFO(name << " cycle " << cycle << " link " << link << " point '" << plan.point
                      << "' at " << plan.at << " kill_us " << kill_us << " status "
                      << life.status);
            REQUIRE((life.killed() || life.finished()));
            if (certain)
               REQUIRE(life.killed());
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
                  "verifications, %llu transactions kept (%llu of them on a root put back to an "
                  "older snapshot, %llu after the store refused the write as full), two-root "
                  "commits found half published %llu times; store %.1f MiB lines, %.1f MiB "
                  "large; %.1fs\n",
                  name, (unsigned long long)stats.lives, (unsigned long long)stats.point_deaths,
                  (unsigned long long)stats.recovery_deaths,
                  (unsigned long long)stats.timed_kills, (unsigned long long)stats.clean,
                  (unsigned long long)model.verified, (unsigned long long)versions,
                  (unsigned long long)model.reverted, (unsigned long long)model.refused,
                  (unsigned long long)model.torn_pairs, last_audit.lines * 128 / 1048576.0,
                  last_audit.grains * 32768 / 1048576.0, secs);
      std::string reached, never;
      for (const auto& [n, p] : stats.points)
         (p.deaths ? reached : never) += " " + n + (p.deaths ? "x" + std::to_string(p.deaths) : "");
      std::printf("  died at:%s\n  never reached:%s\n", reached.c_str(), never.c_str());
      std::fflush(stdout);
      if (until_refused)
         REQUIRE(model.refused > 0);
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
             "retire.before_release", "close.before_clean_mark", "close.after_clean_mark"});
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
   plan.prof.stash       = false;  // these run in the other campaigns; here every
   plan.pair             = false;  // refused write is followed by removes
   plan.writers          = 3;
   plan.txns             = 50;
   campaign(plan, cycles, env_u64("PSITRI_SMALL_CRASH_SEED", 20261128) + 2, "ceiling",
            {"root.after_store", "recover.mark"}, /*until_refused=*/true);
}

// Growth extends three files and then records the new segment count; a
// creation writes the header of a file it has just sized. A death between
// any two of those steps, at creation and when a store that holds committed
// data grows again, must leave a store that opens and has lost nothing.
TEST_CASE("Small Engine: a process killed while a store is created or grows loses nothing",
          "[small][crash][growth]")
{
   const uint64_t seed = env_u64("PSITRI_SMALL_CRASH_SEED", 20261128) + 3;
   static const char* const creation[] = {
       "create.before_header", "create.in_header", "large.create.before_header",
       "large.create.in_header", "grow.before_data", "grow.after_data", "grow.after_map",
       "grow.after_counts", "grow.before_count", "grow.after_count", "large.grow.before_data",
       "large.grow.after_data", "large.grow.after_map", "large.grow.after_counts",
       "large.grow.before_count", "large.grow.after_count"};
   uint64_t deaths = 0, trims = 0;
   for (const char* point : creation)
      for (uint64_t at : {1, 3})
      {
         const bool header = std::string(point).find("create.") != std::string::npos;
         // once per arena or per growth, the others once per new segment
         if ((header || std::string(point).ends_with("_count")) && at != 1)
            continue;
         temporary_directory tmp;
         store_model         model;
         auto                plan = base_plan(psitri::sync_mode::none, 1024);
         plan.seed  = seed;
         plan.txns  = 40;
         plan.point = point;
         plan.at    = at;
         INFO("creation: " << point << " at " << at);
         std::vector<life_result> lives{live(tmp.path, plan, model)};
         REQUIRE(lives.back().killed());
         ++deaths;
         // the next open finds files longer than the count and cuts them: die there too
         if (!header)
            for (const char* trim : {"open.before_trim", "open.after_trim", "large.open.before_trim",
                                     "large.open.after_trim"})
            {
               plan.point = trim;
               plan.at    = 1 + at % 3;
               lives.push_back(live(tmp.path, plan, model));
               REQUIRE((lives.back().killed() || lives.back().finished()));
               trims += lives.back().killed();
            }
         plan.point.clear();
         lives.push_back(live(tmp.path, plan, model));
         REQUIRE(lives.back().finished());
         verify(tmp.path, plan, model, lives);
      }
   REQUIRE(trims > 0);

   // A store with committed data in every segment it has, growing again.
   uint64_t regrow_deaths = 0;
   for (const char* point : {"grow.before_data", "grow.after_data", "grow.after_map",
                             "grow.after_counts", "grow.before_count", "grow.after_count"})
   {
      temporary_directory tmp;
      store_model         model;
      auto                plan = base_plan(psitri::sync_mode::none, 1024);
      plan.seed             = seed;
      plan.prof.fill        = true;
      plan.prof.keyspace    = 60000;
      plan.prof.chunk_every = 0;
      plan.txns             = 60;
      const uint64_t first_batch = 4 * uint64_t(start_ref::segment_bytes);
      audit_result   a;
      for (int life = 0; life < 200 && a.lines * 128 < first_batch * 3 / 4; ++life)
      {
         std::vector<life_result> lives{live(tmp.path, plan, model)};
         REQUIRE(lives.back().finished());
         verify(tmp.path, plan, model, lives, &a);
      }
      INFO("regrowth: " << point << ", " << a.lines * 128 << " bytes in use before");
      REQUIRE(a.lines * 128 >= first_batch * 3 / 4);
      plan.point = point;
      plan.at    = std::string(point).ends_with("_count") ? 1 : 1 + regrow_deaths % 2;
      bool died  = false;
      for (int life = 0; life < 40 && !died; ++life)
      {
         std::vector<life_result> lives{live(tmp.path, plan, model)};
         REQUIRE((lives.back().killed() || lives.back().finished()));
         died = lives.back().killed();
         if (died)
         {
            // and once more while the longer files are being cut back
            plan.point = regrow_deaths % 2 ? "open.before_trim" : "open.after_trim";
            plan.at    = 1;
            lives.push_back(live(tmp.path, plan, model));
            REQUIRE((lives.back().killed() || lives.back().finished()));
         }
         verify(tmp.path, plan, model, lives, &a);
      }
      REQUIRE(died);
      ++regrow_deaths;
      // and it still grows afterwards
      plan.point.clear();
      for (int life = 0; life < 60 && a.lines * 128 < first_batch + (4u << 20); ++life)
      {
         std::vector<life_result> lives{live(tmp.path, plan, model)};
         REQUIRE(lives.back().finished());
         verify(tmp.path, plan, model, lives, &a);
      }
      REQUIRE(a.lines * 128 >= first_batch);
   }
   std::printf("crash growth: %llu deaths during creation, %llu while trimming, %llu during "
               "regrowth\n",
               (unsigned long long)deaths, (unsigned long long)trims,
               (unsigned long long)regrow_deaths);
}

// ── is committed data ever written in place? ───────────────────────────────
// The model stands on one property: between two root stores, no byte that a
// published root reaches is changed. The writer does change nodes in place
// when it holds the only reference to them, so the property is that a node
// (or one line of a node of several lines) reachable from a root slot is
// never in that state. `frozen` records a checksum of every line and every
// large-value grain reachable from the sixteen root slots; intact() says
// whether all of them still read the same.
namespace
{
   uint64_t checksum(const void* p, size_t n)
   {
      const auto* b = static_cast<const unsigned char*>(p);
      uint64_t    h = 0x9E3779B97F4A7C15ull;
      for (size_t i = 0; i + 8 <= n; i += 8)
      {
         uint64_t w;
         std::memcpy(&w, b + i, 8);
         h = (h ^ w) * 0xFF51AFD7ED558CCDull;
         h ^= h >> 32;
      }
      return h;
   }
   struct frozen
   {
      database*                              db = nullptr;
      std::vector<std::pair<uint32_t, uint64_t>> lines, grains;
      std::string                            damage;
      uint64_t                               looks = 0;  // how often intact() was asked

      // `versions`: these root words (versions held by snapshots and
      // readers) instead of the root slots
      void take(database& d, const std::vector<uint64_t>* versions = nullptr)
      {
         db = &d;
         lines.clear(), grains.clear();
         const census c = walk_roots(d, versions);
         for (const auto& [i, refs] : c.line_refs)
            lines.emplace_back(i, checksum(d.store().get(i), start_ref::line_bytes));
         for (const auto& [g, refs] : c.grain_refs)
            grains.emplace_back(g, checksum(d.large_store().get(g), start_ref::large_grain_bytes));
      }
      bool intact()
      {
         ++looks;
         for (const auto& [i, sum] : lines)
            if (checksum(db->store().get(i), start_ref::line_bytes) != sum)
            {
               damage = "line " + std::to_string(i);
               return false;
            }
         for (const auto& [g, sum] : grains)
            if (checksum(db->large_store().get(g), start_ref::large_grain_bytes) != sum)
            {
               damage = "large-value grain " + std::to_string(g);
               return false;
            }
         return true;
      }
   };

   // A transaction (or root handle) that checks after every single operation.
   template <class Tx>
   struct watched
   {
      Tx&         tx;
      frozen&     ice;
      std::string where;
      void        look(const char* op, std::string_view key)
      {
         if (!ice.intact())
            throw std::runtime_error("committed data changed in place: " + ice.damage +
                                     " during " + op + " of '" + std::string(key.substr(0, 24)) +
                                     "' (" + where + ")");
      }
      void upsert(std::string_view k, std::string_view v)
      {
         tx.upsert(k, v);
         look("upsert", k);
      }
      int remove(std::string_view k)
      {
         const int r = tx.remove(k);
         look("remove", k);
         return r;
      }
      uint64_t remove_range_counted(std::string_view a, std::string_view b)
         requires requires(Tx& t) { t.remove_range_counted(a, b); }
      {
         const uint64_t n = tx.remove_range_counted(a, b);
         look("range remove", a);
         return n;
      }
   };

   // Inside commit: called by the engine's markers. Just before each root
   // store the bytes every slot reaches, and the versions snapshots and
   // readers hold, must still be as they were; just after it the slots'
   // baseline is taken again, so that the second store of a two-root commit
   // is watched like the first.
   namespace watching
   {
      frozen*     slots = nullptr;
      frozen*     held  = nullptr;
      std::string violation;
      uint64_t    root_stores = 0;
      void hook(const char* name)
      {
         if (!slots)
            return;
         if (std::strcmp(name, "root.before_store") == 0)
         {
            ++root_stores;
            if (!slots->intact())
               violation = slots->damage + " (reachable from a root slot) changed inside commit";
            else if (!held->intact())
               violation = held->damage + " (of a version a snapshot or reader holds) changed "
                                          "inside commit";
         }
         else if (std::strcmp(name, "root.after_store") == 0)
            slots->take(*slots->db);
      }
   }  // namespace watching

   struct in_place_stats
   {
      uint64_t commits = 0, aborts = 0, stashes = 0, reverts = 0, old_snapshots = 0, pairs = 0,
               checks = 0, reopened_clean = 0, reopened_killed = 0, root_stores = 0;
   };

   // `rounds` transactions of every kind on an open store, checking that the
   // published bytes stand still until each root store.
   void write_and_watch(database& db, uint64_t seed, uint64_t rounds, const profile& prof,
                        in_place_stats& st)
   {
      std::mt19937_64 rng(seed);
      auto            sessions = {db.start_write_session(), db.start_write_session()};
      auto            rs       = db.start_read_session();
      frozen          ice, held_ice;
      ice.take(db);
      // versions kept alive by owned snapshots and readers, as root words
      std::vector<uint64_t> held_versions;
      held_ice.take(db, &held_versions);
      watching::slots = &ice;
      watching::held  = &held_ice;
      watching::violation.clear();
      start_ref::testing::crash_hook.store(&watching::hook, std::memory_order_release);
      {
         // the detector detects: one byte of one published line, put back
         if (!ice.lines.empty())
         {
            auto* p = static_cast<unsigned char*>(db.store().get(ice.lines.front().first)) + 77;
            *p ^= 0x40;
            REQUIRE_FALSE(ice.intact());
            *p ^= 0x40;
         }
         REQUIRE(ice.intact());
      }
      auto model_of = [&](uint32_t root) {
         auto c = rs->create_cursor(root);
         return scan(c);
      };
      struct kept_tree
      {
         uint32_t            first;   // the root it is a snapshot of
         start_ref::kv::tree second;
         uint64_t            version;  // its root word
      };
      struct reader
      {
         std::unique_ptr<engine::cursor> cursor;
         uint64_t                        version;
      };
      std::vector<kept_tree> kept;  // old owned snapshots
      std::vector<reader>    readers;
      // The held set changed: what stays must be intact, then it is the new baseline.
      auto rehold = [&] {
         INFO(held_ice.damage << " (of a held version) changed");
         REQUIRE(held_ice.intact());
         held_versions.clear();
         for (const auto& k : kept)
            held_versions.push_back(k.version);
         for (const auto& r : readers)
            held_versions.push_back(r.version);
         held_ice.take(db, &held_versions);
      };
      for (uint64_t round = 0; round < rounds; ++round)
      {
         auto&          ws   = *(sessions.begin() + rng() % 2);
         const uint32_t r    = uint32_t(rng() % single_roots);
         const uint32_t what = uint32_t(rng() % 100);
         const std::string where = "round " + std::to_string(round) + ", root " + std::to_string(r);
         INFO(where << ", kind " << what);
         auto still = [&](const char* when) {
            if (!ice.intact())
            {
               INFO(ice.damage << " changed " << when);
               REQUIRE(false);
            }
            if (!held_ice.intact())
            {
               INFO(held_ice.damage << " (of a version a snapshot or reader holds) changed "
                                    << when);
               REQUIRE(false);
            }
            INFO(watching::violation);
            REQUIRE(watching::violation.empty());
         };
         // readers come and go; old owned snapshots are kept for a while
         bool changed = false;
         if (rng() % 5 == 0)
         {
            const uint32_t slot = uint32_t(rng() % 12);
            readers.push_back({std::make_unique<engine::cursor>(rs->create_cursor(slot)),
                               db.store().get_root(slot)});
            changed = true;
         }
         if (readers.size() > 6 || (!readers.empty() && rng() % 7 == 0))
         {
            readers.erase(readers.begin() + rng() % readers.size());
            changed = true;
         }
         if (rng() % 6 == 0)
         {
            kept.push_back({r, ws->get_root(r), db.store().get_root(r)});
            changed = true;
         }
         if (kept.size() > 8)
         {
            kept.erase(kept.begin() + rng() % kept.size());
            changed = true;
         }
         if (changed)
            rehold();

         model_map cur = model_of(r);
         uint64_t  k   = seq_of(cur) + 1;
         if (what < 10)
         {
            // rolled back
            model_map scratch = cur;
            auto      tx      = ws->start_transaction(r);
            watched   w{tx, ice, where};
            apply_txn(&w, scratch, log_of(cur), seed ^ round, r, k, plain, 0, prof);
            still("before an abort");
            tx.abort();
            still("by an abort");
            ++st.aborts;
         }
         else if (what < 20)
         {
            // a second slot takes the same tree
            auto tx = ws->start_transaction(stash_base + r);
            tx.primary().set_root(ws->get_root(r));
            still("by set_root of the live tree into another slot");
            tx.commit();
            still("by the commit that stored the root");
            ++st.stashes;
         }
         else if (what < 32)
         {
            // write, replace the root by the stashed or a kept old snapshot, write again
            model_map base = model_of(stash_base + r);
            auto      tx   = ws->start_transaction(r);
            watched   w{tx, ice, where};
            for (uint32_t j = 0; j < 4; ++j)
               w.upsert("junk/" + std::to_string(j), std::string(30 + j * 400, 'j'));
            bool old = false;
            for (auto it = kept.begin(); it != kept.end(); ++it)
               if (it->first == r && rng() % 2)
               {
                  {
                     auto c = it->second.snapshot_cursor();
                     base   = scan(c);
                  }
                  tx.primary().set_root(std::move(it->second));
                  kept.erase(it);
                  rehold();
                  old = true;
                  break;
               }
            if (!old)
               tx.primary().set_root(ws->get_root(stash_base + r));
            still("by set_root of an old snapshot");
            apply_txn(&w, base, log_of(cur), seed ^ round, r, k, revert, seq_of(base), prof);
            still("by writes on top of an old snapshot");
            tx.commit();
            still("by the commit that stored the root");
            ++(old ? st.old_snapshots : st.reverts);
         }
         else if (what < 44)
         {
            // two roots in one transaction; the first root store ends the
            // watch (the second root is published a moment later)
            model_map a = model_of(pair_a), b = model_of(pair_b);
            const uint64_t pk = seq_of(a) + 1;
            auto           tx = ws->start_transaction(pair_a);
            auto           hb = tx.open_root(pair_b);
            watched        wa{tx, ice, where};
            watched        wb{hb, ice, where};
            apply_txn(&wa, a, log_of(a), seed ^ round, pair_a, pk, plain, 0, prof);
            apply_txn(&wb, b, log_of(b), seed ^ round, pair_b, pk, plain, 0, prof);
            still("by a two-root transaction");
            tx.commit();
            still("by the commit that stored the two roots");
            ++st.pairs;
         }
         else
         {
            auto tx = ws->start_transaction(r);
            if (round % 3 == 0)
            {
               watched w{tx, ice, where};  // after every operation
               apply_txn(&w, cur, log_of(cur), seed ^ round, r, k, plain, 0, prof);
            }
            else
               apply_txn(&tx, cur, log_of(cur), seed ^ round, r, k, plain, 0, prof);
            still("by a transaction before its commit");
            tx.commit();
            still("by the commit that stored the root");
            ++st.commits;
         }
      }
      readers.clear();
      kept.clear();
      start_ref::testing::crash_hook.store(nullptr, std::memory_order_release);
      watching::slots = watching::held = nullptr;
      st.checks += ice.looks + held_ice.looks;
      st.root_stores = watching::root_stores;
   }
}  // namespace

// Every kind of transaction, with roots that share nodes, old snapshots put
// back as roots, readers and owned snapshots held and dropped, two sessions
// taking turns on the same roots, on a store that is reopened cleanly and
// reopened after being killed (so that the counts the writer consults were
// rebuilt by recovery). After every operation that could write: every line
// and grain the root slots reach still has the bytes it had at the last
// root store. PSITRI_SMALL_INPLACE_ROUNDS scales it.
TEST_CASE("Small Engine: nothing a published root reaches is written before the next root store",
          "[small][crash][inplace]")
{
   const uint64_t      rounds = env_u64("PSITRI_SMALL_INPLACE_ROUNDS", 1200);
   const uint64_t      seed   = env_u64("PSITRI_SMALL_CRASH_SEED", 20261128) + 4;
   temporary_directory tmp;
   in_place_stats      st;
   auto                plan = base_plan(psitri::sync_mode::none, 1024);
   plan.seed                = seed;
   plan.prof.keyspace       = 400;  // small trees, checked often; every node kind still occurs
   plan.prof.chunk_every    = 25;
   plan.txns                = 25;
   std::mt19937_64 rng(seed);
   const uint64_t  phases = 12;
   for (uint64_t phase = 0; phase < phases; ++phase)
   {
      {
         auto db = database::open(tmp.path, psitri::open_mode::create_or_open, plan.cfg);
         write_and_watch(*db, seed + phase, rounds / phases, plan.prof, st);
         db->flush_releases();
      }
      if (phase % 2 == 0)
      {
         // Other writers, then a kill: the next phase runs on counts that
         // recovery rebuilt from the root slots.
         const life_result life = run_life(tmp.path, plan, int64_t(rng() % 20000));
         REQUIRE((life.killed() || life.finished()));
         st.reopened_killed += life.killed();
      }
      else
         ++st.reopened_clean;
      auto db = database::open(tmp.path, psitri::open_mode::open_existing, plan.cfg);
      audit(*db);
   }
   std::printf("in place: %llu commits, %llu aborts, %llu shared-slot stashes, %llu reverts to "
               "the stash, %llu to older owned snapshots, %llu two-root commits; %llu checks of "
               "every published byte; reopened after a kill %llu times, cleanly %llu\n",
               (unsigned long long)st.commits, (unsigned long long)st.aborts,
               (unsigned long long)st.stashes, (unsigned long long)st.reverts,
               (unsigned long long)st.old_snapshots, (unsigned long long)st.pairs,
               (unsigned long long)st.checks, (unsigned long long)st.reopened_killed,
               (unsigned long long)st.reopened_clean);
   REQUIRE(st.reverts > 0);
   REQUIRE(st.old_snapshots > 0);
   REQUIRE(st.root_stores > st.commits);  // the checks inside commit ran
   REQUIRE(st.reopened_killed > 0);
}

// ── not gating under the process-death model, but not left silently wrong ──

// A flush that fails. Every other flush of the store already ends the
// process (sal's rule for durable I/O: exit code 74, no acknowledgement);
// the msync of the data pages ignored its result and the commit went on to
// store the root.
TEST_CASE("Small Engine: a failed flush of data pages ends the process before the root is stored",
          "[small][crash][flush-error]")
{
   temporary_directory tmp;
   auto                cfg = base_plan(psitri::sync_mode::msync_sync, 64).cfg;
   int                 fds[2];
   REQUIRE(::pipe(fds) == 0);
   std::fflush(nullptr);
   const pid_t pid = ::fork();
   REQUIRE(pid >= 0);
   if (pid == 0)
   {
      ::close(fds[0]);
      ::close(STDERR_FILENO);  // the diagnostic is expected
      try
      {
         auto db = database::open(tmp.path, psitri::open_mode::create_only, cfg);
         auto ws = db->start_write_session();
         for (uint64_t k = 1; k <= 2; ++k)
         {
            auto tx = ws->start_transaction(0);
            tx.upsert("version", std::string(1, char('0' + k)));
            tx.upsert("payload", std::string(3000, char('a' + k)));
            if (k == 2)
               fail_msync_in.store(1);  // the next msync: this commit's data pages
            tx.commit();
            ack(fds[1], 0, k);
         }
      }
      catch (...)
      {
         ::_exit(3);  // never back into the test runner
      }
      ::_exit(0);
   }
   ::close(fds[1]);
   uint64_t   acked = 0;
   ack_record rec;
   while (::read(fds[0], &rec, sizeof(rec)) == ssize_t(sizeof(rec)))
      acked = rec.version;
   ::close(fds[0]);
   int status = 0;
   ::waitpid(pid, &status, 0);
   wait_for_store_locks(tmp.path);
   REQUIRE(WIFEXITED(status));
   REQUIRE(WEXITSTATUS(status) == sal::durable_io_failure_exit_code);
   REQUIRE(acked == 1);
   auto db  = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);
   REQUIRE(db->recovered_on_open());
   REQUIRE(cur.get<std::string>("version") == std::optional<std::string>("1"));
   REQUIRE(cur.get<std::string>("payload") == std::optional<std::string>(std::string(3000, 'b')));
}

// Settings the engine does not implement are refused at open, with nothing created.
TEST_CASE("Small Engine: page protection and checksums are refused, not ignored",
          "[small][crash][config]")
{
   temporary_directory tmp;
   const auto          dir = tmp.path / "store";
   auto refused = [&](engine::config cfg, const char* word) {
      std::string said;
      try
      {
         auto db = database::open(dir, psitri::open_mode::create_or_open, cfg);
      }
      catch (const std::invalid_argument& e)
      {
         said = e.what();
      }
      INFO("said: " << said);
      REQUIRE(said.find(word) != std::string::npos);
      REQUIRE(said.find("not implemented") != std::string::npos);
      REQUIRE_FALSE(std::filesystem::exists(dir));
   };
   engine::config a, b, c;
   a.sync_mode               = psitri::sync_mode::mprotect;
   b.write_protect_on_commit = true;
   c.checksum_on_commit      = true;
   refused(a, "mprotect");
   refused(b, "write_protect_on_commit");
   refused(c, "checksum_on_commit");
}

namespace
{
   // A typed value that refers to something outside the store.
   struct ticket
   {
      static constexpr uint16_t type_id = 91;
      struct t_pod
      {
         uint64_t id;
         static void retain(void*, const t_pod&) {}
         static void release(void*, const t_pod&) {}
      };
      uint64_t id = 0;
      t_pod    take_pod() && { return {id}; }
      static ticket make_from_pod(const t_pod& p, bool) { return {p.id}; }
   };
}  // namespace

// Typed values after an unclean stop: every committed one is there and still
// reads as its type, and the open reports how many it walked over, so an
// application whose types count references outside the store knows it has
// to reconcile them (recovery calls no hooks; see
// database::typed_values_found_by_recovery).
TEST_CASE("Small Engine: recovery keeps typed values and says that it found them",
          "[small][crash][typed]")
{
   temporary_directory tmp;
   auto                cfg = base_plan(psitri::sync_mode::none, 64).cfg;
   std::fflush(nullptr);
   const pid_t pid = ::fork();
   REQUIRE(pid >= 0);
   if (pid == 0)
   {
      try
      {
         auto db = database::open(tmp.path, psitri::open_mode::create_only, cfg);
         db->register_value_types<ticket>();
         auto ws = db->start_write_session();
         {
            auto tx = ws->start_transaction(0);
            for (uint64_t i = 0; i < 500; ++i)
               tx.upsert(key_of(uint32_t(i)), ticket{i});
            tx.upsert("plain", "bytes");
            auto r1 = tx.open_root(1);
            tx.commit();
         }
         {
            // a second root that shares the first one's tree
            auto tx = ws->start_transaction(1);
            tx.primary().set_root(ws->get_root(0));
            tx.commit();
         }
         auto tx = ws->start_transaction(0);  // never committed
         for (uint64_t i = 500; i < 600; ++i)
            tx.upsert(key_of(uint32_t(i)), ticket{i});
         ::kill(::getpid(), SIGKILL);
      }
      catch (...)
      {
      }
      ::_exit(1);
   }
   int status = 0;
   ::waitpid(pid, &status, 0);
   REQUIRE(WIFSIGNALED(status));
   wait_for_store_locks(tmp.path);
   auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   REQUIRE(db->recovered_on_open());
   REQUIRE(db->typed_values_found_by_recovery() == 500);  // shared by two roots, counted once
   db->register_value_types<ticket>();
   auto ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      for (uint64_t i = 0; i < 500; ++i)
      {
         const auto v = tx.get_any(key_of(uint32_t(i)));
         REQUIRE(v.is<ticket>());
         REQUIRE(v.as<ticket>().id == i);
      }
      REQUIRE_FALSE(tx.get_any(key_of(550)));
      tx.abort();
   }
   ws.reset();
   audit(*db);
   db.reset();
   // a clean close and open reports nothing to reconcile
   db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   REQUIRE_FALSE(db->recovered_on_open());
   REQUIRE(db->typed_values_found_by_recovery() == 0);
}

// The harness's own check, checked: a store that comes back older than what
// was acknowledged is caught at the next life's first acknowledgement, even
// though that life then commits the lost versions again.
TEST_CASE("Small Engine crash harness: a store that lost acknowledged versions is caught",
          "[small][crash][selftest]")
{
   temporary_directory tmp;
   store_model         model;
   auto                plan = base_plan(psitri::sync_mode::none, 1024);
   plan.seed                = 99;
   plan.txns                = 40;
   std::vector<life_result> lives{live(tmp.path, plan, model)};
   verify(tmp.path, plan, model, lives);
   REQUIRE(model.roots[0].version == 40);
   // Put root 0 back to its stashed snapshot (version 38) behind the model's back.
   std::fflush(nullptr);
   const pid_t pid = ::fork();
   REQUIRE(pid >= 0);
   if (pid == 0)
   {
      try
      {
         auto db = database::open(tmp.path, psitri::open_mode::open_existing, plan.cfg);
         auto ws = db->start_write_session();
         auto tx = ws->start_transaction(0);
         tx.primary().set_root(ws->get_root(stash_base));
         tx.commit();
      }
      catch (...)
      {
         ::_exit(3);
      }
      ::_exit(0);
   }
   int status = 0;
   ::waitpid(pid, &status, 0);
   REQUIRE((WIFEXITED(status) && WEXITSTATUS(status) == 0));
   wait_for_store_locks(tmp.path);
   const life_result next = run_life(tmp.path, plan);
   REQUIRE(next.finished());
   REQUIRE(next.acked[0] > 40);  // it went on and passed the old version again
   const std::string lost = regression(next, model.floor);
   INFO("regression said: " << lost);
   REQUIRE(lost.find("root 0 came back at version 38 after version 40") != std::string::npos);
   // The comparison with the model alone does not see it: the versions that
   // were lost have been committed again, identically.
   std::vector<life_result> after{next};
   verify(tmp.path, plan, model, after);
}

// Every marker in the engine is died at, at least once, in every run of this
// test: a marker is tried on a new store, then on stores prepared in the ways
// that lead to the steps a new store does not take, and the test fails if a
// marker is left that is not on the short list of ones no test can reach.
TEST_CASE("Small Engine: the process is killed at every marker the engine has",
          "[small][crash][markers]")
{
   // Markers of the large-value arena that need more data than a test writes.
   static const std::pair<const char*, const char*> out_of_reach[] = {
       {"large.segment.release",
        "a large-value segment is given up only when it cannot place a run: 2 GiB of large values"},
       {"large.segment.sweep_claim",
        "the large-value arena's ceiling is at least one 2 GiB segment of large values"},
   };
   const uint64_t seed = env_u64("PSITRI_SMALL_CRASH_SEED", 20261128) + 5;
   auto           normal = base_plan(psitri::sync_mode::none, 1024);
   normal.seed           = seed;
   normal.txns           = 30;
   auto full             = base_plan(psitri::sync_mode::none, 6);
   full.seed             = seed;
   full.prof.fill        = true;
   full.prof.keyspace    = 30000;
   full.prof.chunk_every = 0;
   full.prof.stash       = false;
   full.pair             = false;
   full.writers          = 3;
   full.txns             = 50;

   std::vector<std::string> todo = engine_markers();
   REQUIRE(todo.size() > 60);
   for (const auto& [marker, why] : out_of_reach)
   {
      INFO(marker << " is listed as out of reach but is not a marker of the engine");
      REQUIRE(std::erase(todo, std::string(marker)) == 1);
   }
   struct prepared
   {
      const char* name;
      const char* first;  // a death that sets the store up, or null
      uint64_t    at;
      bool        clean_life;  // one life that closes, before anything else
   };
   static const prepared ways[] = {
       {"a new store", nullptr, 0, false},
       {"a store closed once", nullptr, 0, true},
       {"a store left unclean", "commit.published", 40, true},
       {"a store killed while it grew", "grow.after_data", 1, false},
       {"a store killed while its large-value arena grew", "large.grow.after_data", 1, false},
   };
   std::map<std::string, std::string> reached_by;
   uint64_t                           lives_run = 0;
   for (const prepared& way : ways)
      for (const std::string& marker : std::vector<std::string>(todo))
      {
         temporary_directory      tmp;
         store_model              model;
         auto                     plan = normal;
         std::vector<life_result> lives;
         if (way.clean_life)
         {
            lives.push_back(live(tmp.path, plan, model));
            REQUIRE(lives.back().finished());
         }
         if (way.first)
         {
            plan.point = way.first;
            plan.at    = way.at;
            lives.push_back(live(tmp.path, plan, model));
            REQUIRE(lives.back().killed());
         }
         plan.point = marker;
         plan.at    = 1;
         lives.push_back(live(tmp.path, plan, model));
         lives_run += lives.size();
         INFO("marker " << marker << " on " << way.name);
         REQUIRE((lives.back().killed() || lives.back().finished()));
         if (lives.back().killed())
         {
            reached_by[marker] = way.name;
            std::erase(todo, marker);
         }
         verify(tmp.path, plan, model, lives);
      }
   // What is left needs a store at its ceiling: one such store for all of them.
   if (!todo.empty())
   {
      temporary_directory tmp;
      store_model         model;
      auto                plan = full;
      for (int life = 0; life < 400 && !model.refused; ++life)
      {
         std::vector<life_result> lives{live(tmp.path, plan, model)};
         verify(tmp.path, plan, model, lives);
         ++lives_run;
      }
      REQUIRE(model.refused > 0);
      for (const std::string& marker : std::vector<std::string>(todo))
         for (int attempt = 0; attempt < 6; ++attempt)
         {
            plan.point = marker;
            plan.at    = 1;
            std::vector<life_result> lives{live(tmp.path, plan, model)};
            ++lives_run;
            REQUIRE((lives.back().killed() || lives.back().finished()));
            const bool died = lives.back().killed();
            verify(tmp.path, plan, model, lives);
            if (died)
            {
               reached_by[marker] = "a store at its ceiling";
               std::erase(todo, marker);
               break;
            }
         }
   }
   std::map<std::string, std::string> by_way;
   for (const auto& [marker, way] : reached_by)
      if (way != std::string(ways[0].name))
         by_way[way] += " " + marker;
   std::printf("crash markers: %zu markers died at in %llu lives; %zu on a new store",
               reached_by.size(), (unsigned long long)lives_run,
               reached_by.size() - [&] { size_t n = 0; for (auto& [w, m] : by_way) n += size_t(std::count(m.begin(), m.end(), ' ')); return n; }());
   for (const auto& [way, markers] : by_way)
      std::printf("\n  on %s:%s", way.c_str(), markers.c_str());
   std::printf("\n  listed as out of reach:");
   for (const auto& [marker, why] : out_of_reach)
      std::printf(" %s (%s)", marker, why);
   std::printf("\n");
   std::string left;
   for (const auto& m : todo)
      left += " " + m;
   INFO("markers never died at:" << left);
   REQUIRE(todo.empty());
}

// Files left by a process that died while creating a store are not a store:
// create_only finishes the job, open_existing says there is none.
TEST_CASE("Small Engine: a store whose creation was cut short can be created, not opened",
          "[small][crash][create]")
{
   for (const char* point : {"create.before_header", "create.in_header"})
   {
      temporary_directory tmp;
      auto                plan = base_plan(psitri::sync_mode::none, 64);
      plan.point               = point;
      plan.at                  = 1;
      REQUIRE(run_life(tmp.path, plan).killed());
      REQUIRE(std::filesystem::exists(tmp.path / "meta.part"));
      std::string said;
      try
      {
         auto db = database::open(tmp.path, psitri::open_mode::open_existing, plan.cfg);
      }
      catch (const std::runtime_error& e)
      {
         said = e.what();
      }
      INFO(point << ": open_existing said: " << said);
      REQUIRE(said.find("does not exist") != std::string::npos);
      auto db = database::open(tmp.path, psitri::open_mode::create_only, plan.cfg);
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.upsert("first", "value");
      tx.commit();
   }
}

// Small Engine operating limits and failure behaviour, at the level an
// embedder uses it: psitri::database_for<psitri::small_engine>.
//
// The workload throughout is the one the engine is meant for: a table of small
// rows keyed by identity plus an index ordered by last-seen time, both under
// one root, rewritten in small batches by one writer.
#include <psitri/api.hpp>
#include <psitri/engines/small.hpp>

#include <catch2/catch_test_macros.hpp>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

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
             (std::filesystem::temp_directory_path() / "psitri-small-XXXXXX").string();
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

   // ── the accounting model ────────────────────────────────────────────────
   // Transaction s (1-based) touches rows (s*batch + j) % rows for j < batch.
   // A row's value records the transaction that last wrote it; the index holds
   // exactly one entry per row, keyed by that transaction. The whole database
   // is therefore a function of the number of committed transactions, so a
   // reopened store can be checked exactly.
   struct model
   {
      uint64_t rows;
      uint64_t batch;
      // Also keep two values that live in the large-value arena, rewritten
      // every `big_every` transactions (0: none): one that fits a single
      // large allocation and one that spans several.
      uint64_t big_every = 0;

      static constexpr const char* mid_key = "b/mid";
      static constexpr const char* big_key = "b/big";
      static std::string big_value(uint64_t seq, size_t base)
      {
         std::string v(base + (seq % 7) * 4096, char('A' + seq % 26));
         std::memcpy(v.data(), &seq, 8);
         return v;
      }
      uint64_t last_big(uint64_t seq) const { return big_every ? seq / big_every * big_every : 0; }

      // Two scrambled bytes ahead of the number spread the keys over every
      // branch width the tree has, up to 256-way nodes.
      static std::string id_part(uint64_t id)
      {
         const uint32_t h = uint32_t(id * 2654435761u) >> 16;
         char           b[24];
         std::snprintf(b, sizeof(b), "%012llu", (unsigned long long)id);
         std::string out;
         out.push_back(char(h >> 8));
         out.push_back(char(h));
         return out + b;
      }
      static uint64_t parse_id(std::string_view part)
      {
         return std::strtoull(std::string(part.substr(2)).c_str(), nullptr, 10);
      }
      static std::string row_key(uint64_t id) { return "r/" + id_part(id); }
      static std::string index_key(uint64_t seq, uint64_t id)
      {
         char b[24];
         std::snprintf(b, sizeof(b), "t/%012llu/", (unsigned long long)seq);
         return b + id_part(id);
      }
      static std::string row_value(uint64_t seq, uint64_t id)
      {
         std::string v(96, char('a' + id % 26));
         std::memcpy(v.data(), &seq, 8);
         std::memcpy(v.data() + 8, &id, 8);
         return v;
      }
      static uint64_t value_seq(std::string_view v)
      {
         uint64_t s = 0;
         std::memcpy(&s, v.data(), 8);
         return s;
      }
      static constexpr const char* seq_key = "m/seq";

      template <typename Tx>
      void apply(Tx& tx, uint64_t seq) const
      {
         for (uint64_t j = 0; j < batch; ++j)
         {
            const uint64_t id = (seq * batch + j) % rows;
            // Thrown, not REQUIREd: this also runs in forked writer children,
            // which must exit on a failure and not carry on as a test runner.
            if (auto old = tx.template get<std::string>(row_key(id)))
               if (tx.remove(index_key(value_seq(*old), id)) != 0)
                  throw std::runtime_error("model: a row's index entry is missing");
            tx.upsert(row_key(id), row_value(seq, id));
            tx.upsert(index_key(seq, id), std::string_view());
         }
         if (big_every && seq % big_every == 0)
         {
            tx.upsert(mid_key, big_value(seq, 300u << 10));
            tx.upsert(big_key, big_value(seq, 5u << 20));
         }
         tx.upsert(seq_key, std::string_view(reinterpret_cast<const char*>(&seq), 8));
      }

      // The last transaction <= seq that wrote row `id`, or 0.
      uint64_t last_writer(uint64_t seq, uint64_t id) const
      {
         // s*batch + j == id (mod rows), 0 <= j < batch, largest s <= seq
         for (uint64_t s = seq; s >= 1 && seq - s <= rows / batch + 1; --s)
         {
            const uint64_t first = (s * batch) % rows;
            const uint64_t off   = (id + rows - first) % rows;
            if (off < batch)
               return s;
         }
         return 0;
      }

      // Full forward scan of a snapshot; returns the committed sequence number
      // after checking every row and index entry against the model.
      template <typename Cursor>
      uint64_t verify(Cursor& c) const
      {
         uint64_t seq = 0;
         if (auto v = c.template get<std::string>(seq_key))
            std::memcpy(&seq, v->data(), 8);
         uint64_t                 n_rows = 0, n_index = 0, n_other = 0, n_big = 0;
         std::vector<uint64_t>    row_seq(rows, 0);
         bool                     ok = true;
         for (c.seek_begin(); !c.is_end(); c.next())
         {
            const std::string k(c.key());
            if (k.starts_with("r/"))
            {
               const uint64_t id = parse_id(std::string_view(k).substr(2));
               const auto     v  = c.template value<std::string>();
               ok &= id < rows && v && v->size() == 96;
               if (!ok)
                  break;
               row_seq[id] = value_seq(*v);
               ok &= *v == row_value(row_seq[id], id);
               ok &= row_seq[id] == last_writer(seq, id);
               ++n_rows;
            }
            else if (k.starts_with("t/"))
            {
               const uint64_t s  = std::strtoull(k.c_str() + 2, nullptr, 10);
               const uint64_t id = parse_id(std::string_view(k).substr(15));
               // rows sort before the index, so the row has been seen already
               ok &= id < rows && row_seq[id] == s;
               ++n_index;
            }
            else if (k.starts_with("b/"))
            {
               const auto v = c.template value<std::string>();
               ok &= v && *v == big_value(last_big(seq), k == mid_key ? 300u << 10 : 5u << 20);
               ++n_big;
            }
            else
               ++n_other;
         }
         REQUIRE(n_big == (last_big(seq) ? 2u : 0u));
         const uint64_t expect = std::min(rows, seq * batch);
         INFO("seq " << seq << " rows " << n_rows << " index " << n_index << " expect " << expect);
         REQUIRE(ok);
         REQUIRE(n_rows == expect);
         REQUIRE(n_index == expect);
         REQUIRE(n_other == (seq ? 1u : 0u));
         return seq;
      }
   };

   engine::config small_config(psitri::sync_mode sync, uint32_t max_segments = 64)
   {
      engine::config cfg;
      cfg.sync_mode    = sync;
      cfg.max_segments = max_segments;
      return cfg;
   }

   const char* name_of(psitri::sync_mode m)
   {
      switch (m)
      {
         case psitri::sync_mode::none:        return "none";
         case psitri::sync_mode::mprotect:    return "mprotect";
         case psitri::sync_mode::msync_async: return "msync_async";
         case psitri::sync_mode::msync_sync:  return "msync_sync";
         case psitri::sync_mode::fsync:       return "fsync";
         case psitri::sync_mode::full:        return "full";
      }
      return "?";
   }

   // Child body: continue the model from the store's own sequence number,
   // reporting each sequence number after its commit returns. Never returns.
   [[noreturn]] void writer_child(const std::filesystem::path& dir, engine::config cfg,
                                  const model& m, int report_fd)
   {
      try
      {
         auto db = database::open(dir, psitri::open_mode::create_or_open, cfg);
         auto ws = db->start_write_session();
         uint64_t seq = 0;
         {
            auto tx = ws->start_transaction(0);
            if (auto v = tx.get<std::string>(model::seq_key))
               std::memcpy(&seq, v->data(), 8);
            tx.abort();
         }
         for (;;)
         {
            ++seq;
            auto tx = ws->start_transaction(0);
            m.apply(tx, seq);
            tx.commit();
            if (::write(report_fd, &seq, 8) != 8)
               ::_exit(3);
         }
      }
      catch (const std::exception& e)
      {
         std::fprintf(stderr, "writer child: %s\n", e.what());
      }
      ::_exit(2);
   }

   struct kill_result
   {
      uint64_t acknowledged = 0;  // last sequence number the child reported
      int      status       = 0;
   };

   // Run a writer child and SIGKILL it after `run_us` microseconds.
   kill_result run_and_kill(const std::filesystem::path& dir, engine::config cfg, const model& m,
                            uint64_t run_us)
   {
      int fds[2];
      REQUIRE(::pipe(fds) == 0);
      std::fflush(nullptr);
      const pid_t pid = ::fork();
      REQUIRE(pid >= 0);
      if (pid == 0)
      {
         ::close(fds[0]);
         writer_child(dir, cfg, m, fds[1]);
      }
      ::close(fds[1]);
      std::atomic<uint64_t> last{0};
      std::thread           drain([&] {
         uint64_t seq;
         while (::read(fds[0], &seq, 8) == 8)
            last.store(seq, std::memory_order_relaxed);
      });
      // Let the child get going before the clock starts, so short delays land
      // inside transactions rather than inside open().
      const auto t0 = std::chrono::steady_clock::now();
      while (last.load() == 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30))
         std::this_thread::sleep_for(std::chrono::microseconds(200));
      std::this_thread::sleep_for(std::chrono::microseconds(run_us));
      ::kill(pid, SIGKILL);
      kill_result r;
      ::waitpid(pid, &r.status, 0);
      drain.join();
      ::close(fds[0]);
      r.acknowledged = last.load();
      return r;
   }
// Remove everything and report what the arenas still hold. Reference counts
// that are too high leave storage behind here; counts that are too low free
// live nodes and show up as damage long before.
start_ref::kv::usage_info usage_after_emptying(const std::filesystem::path& dir, engine::config cfg)
{
   auto db = database::open(dir, psitri::open_mode::open_existing, cfg);
   {
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.remove_range_counted(std::string_view(), std::string_view("\xff\xff", 2));
      tx.commit();
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE_FALSE(cur.seek_begin());
   }
   db->flush_releases();
   return db->usage();
}

void kill_repeatedly(psitri::sync_mode sync, const model& m, uint64_t rounds, uint64_t max_run_us)
{
   std::mt19937_64     rng(env_u64("PSITRI_SMALL_KILL_SEED", 12345));
   temporary_directory tmp;
   const auto          cfg = small_config(sync);
   uint64_t            last_seq = 0, settled_used = 0, used = 0, large_settled = 0, large_used = 0;
   const auto          t0 = std::chrono::steady_clock::now();
   for (uint64_t round = 0; round < rounds; ++round)
   {
      const auto killed = run_and_kill(tmp.path, cfg, m, rng() % max_run_us);
      REQUIRE(WIFSIGNALED(killed.status));
      REQUIRE(WTERMSIG(killed.status) == SIGKILL);

      REQUIRE_FALSE(database::closed_cleanly(tmp.path));
      auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
      REQUIRE(db->recovered_on_open());
      auto           rs  = db->start_read_session();
      auto           cur = rs->create_cursor(0);
      const uint64_t seq = m.verify(cur);
      INFO("sync " << name_of(sync) << " round " << round << " acknowledged "
                   << killed.acknowledged << " found " << seq);
      // Everything acknowledged is there; at most the one transaction that
      // was committed but not yet reported is ahead of it.
      REQUIRE(seq >= killed.acknowledged);
      REQUIRE(seq <= killed.acknowledged + 1);
      REQUIRE(seq >= last_seq);
      last_seq   = seq;
      used       = db->usage().used_bytes;
      large_used = db->usage().large_used_bytes;
      // Once every row exists the live set has a constant size.
      if (!settled_used && seq * m.batch >= 2 * m.rows && (!m.big_every || m.last_big(seq)))
      {
         settled_used  = used;
         large_settled = large_used;
      }
   }
   // A clean close after all that is reported as clean and needs no recovery.
   REQUIRE(database::closed_cleanly(tmp.path));
   {
      auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
      REQUIRE_FALSE(db->recovered_on_open());
   }
   // Killed transactions leave nothing behind: use stays where it settled.
   if (settled_used)
   {
      REQUIRE(used < settled_used + settled_used / 4 + (256u << 10));
      REQUIRE(large_used <= large_settled + (8u << 20));
   }
   const auto emptied = usage_after_emptying(tmp.path, cfg);
   const double secs =
       std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
   std::printf("kill test: sync=%s big=%d kills=%llu transactions=%llu used: settled %llu, "
               "after last kill %llu, after removing every key %llu (large %llu) bytes; %.1fs\n",
               name_of(sync), int(m.big_every != 0), (unsigned long long)rounds,
               (unsigned long long)last_seq, (unsigned long long)settled_used,
               (unsigned long long)used, (unsigned long long)emptied.used_bytes,
               (unsigned long long)emptied.large_used_bytes, secs);
   REQUIRE(emptied.used_bytes <= 64 * 128);
   REQUIRE(emptied.large_used_bytes == 0);
}
}  // namespace

// A writer is killed at a random moment, over and over, on the same store.
// After every kill the store must report that it was not closed, reopen with
// every acknowledged transaction present, and hold exactly the tree the model
// says. PSITRI_SMALL_KILL_ROUNDS raises the number of kills per sync mode.
TEST_CASE("Small Engine: a killed writer loses no acknowledged transaction", "[small][kill]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_KILL_ROUNDS", 40);
   const model    m{2000, 20};
   for (auto sync : {psitri::sync_mode::none, psitri::sync_mode::msync_async,
                     psitri::sync_mode::msync_sync, psitri::sync_mode::fsync,
                     psitri::sync_mode::full})
      kill_repeatedly(sync, m, rounds, sync == psitri::sync_mode::full ? 150000 : 30000);
}

// The same with values in the large-value arena, which has its own free map
// and reference counts to rebuild.
TEST_CASE("Small Engine: a killed writer of large values loses nothing", "[small][kill][large]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_KILL_ROUNDS", 40) / 4 + 1;
   const model    m{2000, 20, 8};
   for (auto sync : {psitri::sync_mode::none, psitri::sync_mode::fsync})
      kill_repeatedly(sync, m, rounds, 60000);
}

// Two root slots that share most of their nodes, an owned snapshot, and an
// unfinished transaction, all abandoned without closing. Recovery has to count
// a node once per parent, not once per path to it.
TEST_CASE("Small Engine: recovery counts shared nodes exactly", "[small][kill]")
{
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none);
   const model         m{5000, 50};
   std::fflush(nullptr);
   const pid_t pid = ::fork();
   REQUIRE(pid >= 0);
   if (pid == 0)
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      for (uint64_t seq = 1; seq <= 150; ++seq)
      {
         auto tx = ws->start_transaction(0);
         m.apply(tx, seq);
         tx.commit();
      }
      {
         // root 1 starts as root 0's tree and then diverges a little
         auto tx = ws->start_transaction(0);
         auto r1 = tx.open_root(1);
         r1.set_root(ws->get_root(0));
         r1.upsert("only-in-root-1", "x");
         tx.commit();
      }
      auto held = ws->get_root(0);  // a snapshot nobody releases
      auto tx   = ws->start_transaction(0);
      m.apply(tx, 151);  // never committed
      ::_exit(0);
   }
   int status = 0;
   ::waitpid(pid, &status, 0);
   REQUIRE(WIFEXITED(status));
   REQUIRE(WEXITSTATUS(status) == 0);
   REQUIRE_FALSE(database::closed_cleanly(tmp.path));
   {
      auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
      REQUIRE(db->recovered_on_open());
      auto rs = db->start_read_session();
      {
         auto c0 = rs->create_cursor(0);
         REQUIRE(m.verify(c0) == 150);
      }
      auto c1 = rs->create_cursor(1);
      REQUIRE(c1.get<std::string>("only-in-root-1") == std::optional<std::string>("x"));
      uint64_t n = 0;
      for (c1.seek_begin(); !c1.is_end(); c1.next())
         ++n;
      REQUIRE(n == 2 * 5000 + 2);
   }
   {
      auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
      REQUIRE_FALSE(db->recovered_on_open());
      {
         auto ws = db->start_write_session();
         auto tx = ws->start_transaction(0);
         tx.remove_range_counted(std::string_view(), std::string_view("\xff\xff", 2));
         tx.open_root(1).set_root({});
         tx.commit();
      }
      db->flush_releases();
      REQUIRE(db->usage().used_bytes <= 64 * 128);
   }
}

// Space reuse under churn at constant live size: the engine never moves or
// compacts a node, so freed lines have to be found again where they are.
// Every update rewrites a row in place and moves its index entry to a new key.
// A reader thread takes short-lived snapshots throughout, as the consumer's
// query threads will. PSITRI_SMALL_CHURN_UPDATES / _ROWS scale the run.
TEST_CASE("Small Engine: storage stays bounded under churn at constant live size",
          "[small][churn]")
{
   const uint64_t updates = env_u64("PSITRI_SMALL_CHURN_UPDATES", 400000);
   const model    m{env_u64("PSITRI_SMALL_CHURN_ROWS", 20000), env_u64("PSITRI_SMALL_CHURN_BATCH", 100)};
   const bool     reader_on = env_u64("PSITRI_SMALL_CHURN_READER", 1) != 0;
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 1024));

   std::atomic<bool>     stop{false};
   std::atomic<uint64_t> snapshots{0};
   std::thread           reader;
   if (reader_on)
      reader = std::thread([&] {
         auto rs = db->start_read_session();
         while (!stop.load(std::memory_order_relaxed))
         {
            auto     cur = rs->create_cursor(0);
            uint64_t n   = 0;
            for (cur.lower_bound("t/"); !cur.is_end() && n < 200; cur.next())
               ++n;
            snapshots.fetch_add(1, std::memory_order_relaxed);
         }
      });

   auto           ws        = db->start_write_session();
   const uint64_t txns      = updates / m.batch;
   const uint64_t fill_txns = m.rows / m.batch;
   const uint64_t sample    = std::max<uint64_t>(1, txns / 10);
   uint64_t       settled_mapped = 0, settled_used = 0, peak_used = 0;
   const auto     t0 = std::chrono::steady_clock::now();
   std::printf("churn: rows=%llu batch=%llu updates=%llu reader=%d\n",
               (unsigned long long)m.rows, (unsigned long long)m.batch,
               (unsigned long long)updates, int(reader_on));
   for (uint64_t seq = 1; seq <= txns; ++seq)
   {
      {
         auto tx = ws->start_transaction(0);
         m.apply(tx, seq);
         tx.commit();
      }
      if (seq % 64 == 0)
         peak_used = std::max(peak_used, db->usage().used_bytes);
      if (seq == 3 * fill_txns)  // every row rewritten twice since the fill
      {
         settled_mapped = db->usage().mapped_bytes;
         settled_used   = db->usage().used_bytes;
      }
      if (seq % sample == 0 || seq == txns)
      {
         const auto u = db->usage();
         std::printf("churn: updates=%10llu used=%7.1f MiB mapped=%7.1f MiB\n",
                     (unsigned long long)(seq * m.batch), u.used_bytes / 1048576.0,
                     u.mapped_bytes / 1048576.0);
      }
   }
   const double secs =
       std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
   stop = true;
   if (reader.joinable())
      reader.join();
   db->flush_releases();
   const auto u = db->usage();
   std::printf("churn: done in %.1fs (%.0f updates/s), %llu reader snapshots; used=%.1f MiB "
               "peak used=%.1f MiB mapped=%.1f MiB (settled: used %.1f, mapped %.1f)\n",
               secs, double(txns * m.batch) / secs, (unsigned long long)snapshots.load(),
               u.used_bytes / 1048576.0, peak_used / 1048576.0, u.mapped_bytes / 1048576.0,
               settled_used / 1048576.0, settled_mapped / 1048576.0);
   std::fflush(stdout);
   {
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE(m.verify(cur) == txns);
   }
   REQUIRE(txns > 4 * fill_txns);  // otherwise the run says nothing about steady state
   // Live size is constant, so use and the high-water mark must level off.
   REQUIRE(u.used_bytes <= settled_used + settled_used / 4 + (1u << 20));
   REQUIRE(u.mapped_bytes <= 2 * settled_mapped);
}

// The storage ceiling is max_segments. Reaching it in the middle of a
// transaction must be an ordinary failure: an exception, the transaction
// rolled back, everything committed before still there, and the store usable.
TEST_CASE("Small Engine: reaching the storage limit fails the transaction, not the store",
          "[small][limit]")
{
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 4);  // 32 MiB
   const model         m{4000, 40};
   uint64_t            committed = 0;
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_only, cfg);
      REQUIRE(db->usage().limit_bytes == 4ull * 8 * 1024 * 1024);
      auto ws = db->start_write_session();
      for (uint64_t seq = 1; seq <= 300; ++seq)
      {
         auto tx = ws->start_transaction(0);
         m.apply(tx, seq);
         tx.commit();
         committed = seq;
      }
      const auto before = db->usage();
      REQUIRE(before.used_bytes > 0);
      REQUIRE(before.used_bytes <= before.mapped_bytes);
      REQUIRE(before.mapped_bytes <= before.limit_bytes);

      // Several times over, so that a failure at one particular point in a
      // tree operation is not the only one exercised.
      for (int attempt = 0; attempt < 6; ++attempt)
      {
         bool        threw = false;
         uint64_t    wrote = 0;
         std::string what;
         try
         {
            auto              tx = ws->start_transaction(0);
            const std::string value(64 + attempt * 37, 'v');
            for (uint64_t i = 0;; ++i)
            {
               char k[40];
               std::snprintf(k, sizeof(k), "fill/%d/%012llu", attempt,
                             (unsigned long long)(i * 2654435761u % 100000007u));
               tx.upsert(k, value);
               ++wrote;
               REQUIRE(i < 2'000'000);  // the limit has to arrive
            }
         }
         catch (const psitri::small_engine::storage_full& e)
         {
            threw = true;
            what  = e.what();
         }
         INFO("attempt " << attempt << " wrote " << wrote << ": " << what);
         REQUIRE(threw);
         REQUIRE(what.find("max_segments") != std::string::npos);
         REQUIRE(wrote > 1000);

         // rolled back, and the committed tree is exactly what it was
         {
            auto rs  = db->start_read_session();
            auto cur = rs->create_cursor(0);
            REQUIRE(m.verify(cur) == committed);
         }
         db->flush_releases();
         REQUIRE(db->usage().used_bytes <= before.used_bytes + (256u << 10));
         REQUIRE(db->usage().mapped_bytes <= db->usage().limit_bytes);

         // still writable
         auto tx = ws->start_transaction(0);
         m.apply(tx, committed + 1);
         tx.commit();
         ++committed;
      }
   }
   auto db  = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);
   REQUIRE(m.verify(cur) == committed);
}

// The documented key limit is 1,024 bytes, and it is the enforced one.
TEST_CASE("Small Engine: keys up to 1,024 bytes are stored and longer ones are refused",
          "[small][keys]")
{
   static_assert(psitri::small_engine::max_key_size == 1024);
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto ws = db->start_write_session();

   const std::string at_limit(1024, 'k');
   const std::string over(1025, 'k');
   const std::string far_over(70000, 'k');
   const std::string big_value(3u << 20, 'B');  // takes the chunked-value path
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("before", "1");
      tx.upsert(at_limit, "at the limit");
      tx.upsert(at_limit.substr(0, 1023) + "z", big_value);
      for (const auto& key : {over, far_over})
      {
         REQUIRE_THROWS_AS(tx.upsert(key, "small"), std::invalid_argument);
         REQUIRE_THROWS_AS(tx.upsert(key, big_value), std::invalid_argument);
         REQUIRE_THROWS_AS(tx.primary().upsert(key, "small"), std::invalid_argument);
         // reads and removes of a key that cannot exist are ordinary misses
         REQUIRE_FALSE(tx.get<std::string>(key));
         REQUIRE(tx.remove(key) == -1);
      }
      try
      {
         tx.upsert(over, "small");
      }
      catch (const std::invalid_argument& e)
      {
         REQUIRE(std::string(e.what()).find("1024") != std::string::npos);
      }
      tx.upsert("after", "2");
      tx.commit();
   }
   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);
   REQUIRE(cur.get<std::string>(at_limit) == std::optional<std::string>("at the limit"));
   REQUIRE(cur.get<std::string>(at_limit.substr(0, 1023) + "z") ==
           std::optional<std::string>(big_value));
   REQUIRE_FALSE(cur.get<std::string>(over));
   std::vector<size_t> sizes;
   for (cur.seek_begin(); !cur.is_end(); cur.next())
      sizes.push_back(cur.key().size());
   REQUIRE(sizes == std::vector<size_t>{5, 6, 1024, 1024});
   REQUIRE(cur.lower_bound(over));  // positions on the 1,024-byte ...kz key
   REQUIRE(cur.key().size() == 1024);

   auto tx = ws->start_transaction(0);
   REQUIRE(tx.remove(at_limit) == 0);
   tx.commit();
}

// What a session can detect about its own misuse is reported as a caller
// error and leaves the open transaction and the store untouched. (A root held
// through ANOTHER session is waited for, whichever thread asks: the holder may
// be finished by a different thread, so there is nothing to detect.)
TEST_CASE("Small Engine: a session's second transaction and misordered roots are refused",
          "[small][threads]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto first  = db->start_write_session();
   auto second = db->start_write_session();
   {
      auto tx = first->start_transaction(0);
      tx.upsert("a", "1");
      // same session, any root
      REQUIRE_THROWS_AS(first->start_transaction(0), std::logic_error);
      REQUIRE_THROWS_AS(first->start_transaction(5), std::logic_error);
      // additional roots: once each, ascending
      auto r3 = tx.open_root(3);
      r3.upsert("in-3", "x");
      REQUIRE_THROWS_AS(tx.open_root(3), std::logic_error);
      REQUIRE_THROWS_AS(tx.open_root(2), std::logic_error);
      REQUIRE_THROWS_AS(tx.open_root(0), std::logic_error);
      REQUIRE_THROWS_AS(tx.open_root(16), std::out_of_range);
      // a different root from another session is fine on the same thread
      {
         auto other = second->start_transaction(1);
         other.upsert("b", "2");
         other.commit();
      }
      tx.commit();
      REQUIRE_THROWS_AS(tx.commit(), std::logic_error);
   }
   // the refusals released nothing they should not have, and held nothing
   {
      auto tx = second->start_transaction(0);
      REQUIRE(tx.get<std::string>("a") == std::optional<std::string>("1"));
      tx.upsert("c", "3");
      tx.commit();
   }
   {
      auto tx = first->start_transaction(3);
      REQUIRE(tx.get<std::string>("in-3") == std::optional<std::string>("x"));
      tx.abort();
   }
   // Another thread asking for a held root waits, as before, and gets it.
   {
      auto              tx = first->start_transaction(0);
      std::atomic<bool> got{false};
      std::thread       waiter([&] {
         auto ws    = db->start_write_session();
         auto inner = ws->start_transaction(0);
         got        = true;
         REQUIRE(inner.get<std::string>("d") == std::optional<std::string>("4"));
         inner.abort();
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      REQUIRE_FALSE(got.load());
      tx.upsert("d", "4");
      tx.commit();
      waiter.join();
      REQUIRE(got.load());
   }
}

// A write session is not tied to the thread that made it: it may be handed
// from thread to thread as long as only one uses it at a time.
TEST_CASE("Small Engine: sessions may move between threads when use is serialized",
          "[small][threads]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto ws = db->start_write_session();
   auto rs = db->start_read_session();
   for (int round = 0; round < 50; ++round)
   {
      std::thread worker([&] {
         auto tx = ws->start_transaction(0);
         tx.upsert("k" + std::to_string(round), std::to_string(round));
         tx.commit();
         auto cur = rs->create_cursor(0);
         REQUIRE(cur.get<std::string>("k" + std::to_string(round)) ==
                 std::optional<std::string>(std::to_string(round)));
      });
      worker.join();
      auto tx = ws->start_transaction(0);
      REQUIRE(tx.get<std::string>("k" + std::to_string(round)));
      tx.abort();
   }
   auto     cur = rs->create_cursor(0);
   uint64_t n   = 0;
   for (cur.seek_begin(); !cur.is_end(); cur.next())
      ++n;
   REQUIRE(n == 50);
}

// There are 64 reader slots. Running out is an exception that names the
// limit; nothing else is disturbed and a freed slot is usable again.
TEST_CASE("Small Engine: the 65th reader is refused with a clear error", "[small][readers]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("k", "v0");
      tx.commit();
   }
   auto expect_refused = [](auto&& make) {
      try
      {
         make();
         FAIL("a 65th reader slot was handed out");
      }
      catch (const std::runtime_error& e)
      {
         REQUIRE(std::string(e.what()).find("64 reader slots") != std::string::npos);
      }
   };
   {
      std::vector<std::shared_ptr<engine::read_session>> sessions;
      for (int i = 0; i < 64; ++i)
         sessions.push_back(db->start_read_session());
      expect_refused([&] { return db->start_read_session(); });
      // each session's first cursor uses the session's own slot ...
      std::vector<engine::cursor> cursors;
      for (auto& s : sessions)
         cursors.push_back(s->create_cursor(0));
      // ... a second cursor on any of them needs a slot of its own
      expect_refused([&] { return sessions[0]->create_cursor(0); });
      // the writer is unaffected, and existing snapshots stay what they were
      for (int i = 1; i <= 20; ++i)
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("k", "v" + std::to_string(i));
         tx.commit();
      }
      for (auto& c : cursors)
         REQUIRE(c.get<std::string>("k") == std::optional<std::string>("v0"));
      cursors.clear();  // cursors before the sessions they came from
   }
   // every slot is free again
   {
      auto                        one = db->start_read_session();
      std::vector<engine::cursor> cursors;
      for (int i = 0; i < 64; ++i)
         cursors.push_back(one->create_cursor(0));
      expect_refused([&] { return one->create_cursor(0); });
      REQUIRE(cursors.back().get<std::string>("k") == std::optional<std::string>("v20"));
   }
   db->flush_releases();
   auto another = db->start_read_session();
   auto cur     = another->create_cursor(0);
   REQUIRE(cur.get<std::string>("k") == std::optional<std::string>("v20"));
}

// What a reader can see of a transaction that commits two roots. The roots
// are published one after the other. A reader that takes its two snapshots
// through a read session can land between them and see the first root new
// and the second old; whole commits can also fall between any two snapshots.
// Each root on its own is always a committed state. The test records how
// often the torn view was seen; it asserts only what always holds.
TEST_CASE("Small Engine: roots committed together are published one at a time",
          "[small][multiroot]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto counter = [](std::optional<std::string> v) {
      uint64_t n = 0;
      if (v)
         std::memcpy(&n, v->data(), 8);
      return n;
   };
   std::atomic<bool>     stop{false};
   std::atomic<uint64_t> looks{0}, torn{0}, ahead{0}, impossible{0}, owned_looks{0},
       owned_behind{0};
   std::thread reader([&] {
      auto rs = db->start_read_session();
      auto ws = db->start_write_session();  // for owned snapshots
      while (!stop.load(std::memory_order_relaxed))
      {
         {
            uint64_t a, b;
            {
               auto c0 = rs->create_cursor(0);
               a       = counter(c0.get<std::string>("n"));
            }
            {
               auto c1 = rs->create_cursor(1);
               b       = counter(c1.get<std::string>("n"));
            }
            ++looks;
            if (b + 1 == a)
               ++torn;  // root 0 of commit n, root 1 still of commit n-1
            else if (b > a)
               ++ahead;  // one or more whole commits in between
            else if (b != a)
               ++impossible;
         }
         {
            // Owned snapshots are taken under the publication lock, so the
            // second can be later than the first but never behind it.
            auto t0 = ws->get_root(0);
            auto t1 = ws->get_root(1);
            ++owned_looks;
            if (counter(t1.get<std::string>("n")) < counter(t0.get<std::string>("n")))
               ++owned_behind;
         }
      }
   });
   auto     ws = db->start_write_session();
   uint64_t n  = 0;
   const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
   while (std::chrono::steady_clock::now() < until)
   {
      ++n;
      auto tx = ws->start_transaction(0);
      tx.upsert("n", std::string_view(reinterpret_cast<const char*>(&n), 8));
      tx.open_root(1).upsert("n", std::string_view(reinterpret_cast<const char*>(&n), 8));
      tx.commit();
   }
   stop = true;
   reader.join();
   std::printf("multi-root: %llu commits; read-session pairs: %llu, torn (root 0 new, root 1 "
               "old) %llu, second later %llu; owned-snapshot pairs: %llu, second behind %llu\n",
               (unsigned long long)n, (unsigned long long)looks.load(),
               (unsigned long long)torn.load(), (unsigned long long)ahead.load(),
               (unsigned long long)owned_looks.load(), (unsigned long long)owned_behind.load());
   REQUIRE(impossible.load() == 0);
   REQUIRE(owned_behind.load() == 0);
   REQUIRE(looks.load() > 0);
}

// Read-only open is not implemented. It must say so, and must not create or
// change anything.
TEST_CASE("Small Engine: read-only open is refused with a clear error", "[small][open]")
{
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 16);
   auto refused = [&] {
      try
      {
         database::open(tmp.path, psitri::open_mode::read_only, cfg);
         FAIL("read-only open returned a database");
      }
      catch (const std::runtime_error& e)
      {
         const std::string what = e.what();
         REQUIRE(what.find("read-only") != std::string::npos);
         REQUIRE(what.find("Small Engine") != std::string::npos);
      }
   };
   refused();
   REQUIRE(std::filesystem::is_empty(tmp.path));  // nothing was created
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.upsert("k", "v");
      tx.commit();
   }
   refused();
   REQUIRE(database::closed_cleanly(tmp.path));  // and the refusal did not touch it
   auto db  = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);
   REQUIRE(cur.get<std::string>("k") == std::optional<std::string>("v"));
}

// A root line's reference count has a ceiling: ordinary references stop at
// 190 and owned snapshots may take it to 254. A reference that cannot be
// taken must be an error where it is asked for, never a count that is later
// given back without having been taken.
TEST_CASE("Small Engine: many owned snapshots of one version do not unbalance its count",
          "[small][refs]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto ws = db->start_write_session();
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 500; ++i)
         tx.upsert("key" + std::to_string(i), std::string(40, 'v'));
      tx.commit();
   }
   {
      std::vector<psitri::snapshot_for<engine>> held;
      for (int i = 0; i < 200; ++i)
         held.push_back(ws->get_root(0));
      // More sessions and aborted transactions than the count has room for.
      for (int i = 0; i < 40; ++i)
      {
         try
         {
            auto extra = db->start_write_session();
            auto tx    = extra->start_transaction(0);
            tx.upsert("scratch", "x");
            tx.abort();
         }
         catch (const std::runtime_error&)
         {
            // refusing is fine; miscounting is not
         }
      }
      for (auto& snapshot : held)
         REQUIRE(snapshot.get<std::string>("key7") == std::optional<std::string>(std::string(40, 'v')));
   }
   // Every snapshot is gone; the committed tree must still be whole.
   for (int round = 0; round < 3; ++round)
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 500; ++i)
         REQUIRE(tx.get<std::string>("key" + std::to_string(i)) ==
                 std::optional<std::string>(std::string(40, 'v')));
      tx.upsert("more" + std::to_string(round), std::string(300, 'm'));
      tx.commit();
   }
   {
      auto     rs  = db->start_read_session();
      auto     cur = rs->create_cursor(0);
      uint64_t n   = 0;
      for (cur.seek_begin(); !cur.is_end(); cur.next())
         ++n;
      REQUIRE(n == 503);
   }  // a live cursor would keep the version it reads
   {
      auto tx = ws->start_transaction(0);
      tx.remove_range_counted(std::string_view(), std::string_view("\xff\xff", 2));
      tx.commit();
   }
   ws.reset();
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

// One owner at a time: the store's files are locked while it is open.
TEST_CASE("Small Engine: a second open of a store that is already open is refused",
          "[small][open]")
{
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 16);
   auto db = database::open(tmp.path, psitri::open_mode::create_only, cfg);
   {
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.upsert("k", "v");
      tx.commit();
   }
   REQUIRE_THROWS_AS(database::open(tmp.path, psitri::open_mode::open_existing, cfg),
                     std::runtime_error);
   std::fflush(nullptr);
   const pid_t pid = ::fork();
   REQUIRE(pid >= 0);
   if (pid == 0)
   {
      try
      {
         auto other = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
         ::_exit(1);  // a second owner got in
      }
      catch (const std::exception&)
      {
         ::_exit(0);
      }
   }
   int status = 0;
   ::waitpid(pid, &status, 0);
   REQUIRE(WIFEXITED(status));
   REQUIRE(WEXITSTATUS(status) == 0);
   // The refused opens changed nothing for the owner.
   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);
   REQUIRE(cur.get<std::string>("k") == std::optional<std::string>("v"));
}

// If the tree cannot be walked after an unclean stop, open must fail, and
// keep failing: a failed recovery must not leave the store marked clean, or
// the next open would trust a half-rebuilt free map and hand out live lines.
TEST_CASE("Small Engine: a store that cannot be recovered is refused every time",
          "[small][kill][damage]")
{
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 16);
   const model         m{1000, 20};
   const auto          killed = run_and_kill(tmp.path, cfg, m, 20000);
   REQUIRE(WIFSIGNALED(killed.status));
   REQUIRE_FALSE(database::closed_cleanly(tmp.path));

   // Point root slot 0 far outside the arena, as a torn or lost write could.
   const auto    meta        = tmp.path / "meta.part";
   const long    root_offset = 32;  // start_ref::arena_t::meta_hdr::roots[0]
   std::uint64_t original    = 0;
   {
      std::FILE* f = std::fopen(meta.c_str(), "r+b");
      REQUIRE(f);
      REQUIRE(std::fseek(f, root_offset, SEEK_SET) == 0);
      REQUIRE(std::fread(&original, 8, 1, f) == 1);
      REQUIRE(original != 0);
      const std::uint64_t damaged =
          (original & 0xF) | (std::uint64_t(0x0FFFFFF0) << 4) | (std::uint64_t(1) << 40);
      REQUIRE(std::fseek(f, root_offset, SEEK_SET) == 0);
      REQUIRE(std::fwrite(&damaged, 8, 1, f) == 1);
      std::fclose(f);
   }
   for (int attempt = 0; attempt < 3; ++attempt)
   {
      try
      {
         database::open(tmp.path, psitri::open_mode::open_existing, cfg);
         FAIL("a store with a root outside the arena was opened");
      }
      catch (const std::runtime_error& e)
      {
         INFO(e.what());
         REQUIRE(std::string(e.what()).find("not closed cleanly") != std::string::npos);
      }
      REQUIRE_FALSE(database::closed_cleanly(tmp.path));
   }
   // With the root slot restored the same files recover completely.
   {
      std::FILE* f = std::fopen(meta.c_str(), "r+b");
      REQUIRE(f);
      REQUIRE(std::fseek(f, root_offset, SEEK_SET) == 0);
      REQUIRE(std::fwrite(&original, 8, 1, f) == 1);
      std::fclose(f);
   }
   {
      auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
      REQUIRE(db->recovered_on_open());
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE(m.verify(cur) >= killed.acknowledged);
   }
   REQUIRE(usage_after_emptying(tmp.path, cfg).used_bytes <= 64 * 128);
}


// commit() publishes all of a transaction's roots or none. A root the
// transaction opened but did not change needs no new reference, however many
// owned snapshots of it exist.
TEST_CASE("Small Engine: commit is all-or-nothing when a root reference is refused",
          "[small][refs][commit]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto ws = db->start_write_session();
   auto read = [&](uint32_t root, std::string_view key) {
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(root);
      return cur.get<std::string>(key);
   };
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("a", "0");
      tx.open_root(1).upsert("b", "0");
      tx.open_root(2).upsert("c", "0");
      tx.commit();
   }
   {
      std::vector<psitri::snapshot_for<engine>> held;
      for (int i = 0; i < 195; ++i)
         held.push_back(ws->get_root(1));
      {
         // root 1 opened and left alone
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "1");
         REQUIRE(tx.open_root(1).get<std::string>("b") == std::optional<std::string>("0"));
         REQUIRE_NOTHROW(tx.commit());
      }
      REQUIRE(read(0, "a") == std::optional<std::string>("1"));
      {
         // nothing changed at all
         auto tx = ws->start_transaction(1);
         REQUIRE_NOTHROW(tx.commit());
      }
      // reading through a snapshot at that count needs no further reference
      for (int i = 0; i < 57; ++i)
         held.push_back(ws->get_root(1));  // 252 snapshots
      REQUIRE(held.back().get<std::string>("b") == std::optional<std::string>("0"));
      bool seen = false;
      REQUIRE(held.front().get("b", [&](std::string_view v) { seen = v == "0"; }));
      REQUIRE(seen);
   }
   {
      // A root that cannot be published: root 1 is given a tree whose
      // version is already at the writer ceiling. Root 0, earlier in the
      // order, must not be published either.
      std::vector<psitri::snapshot_for<engine>> held;
      for (int i = 0; i < 195; ++i)
         held.push_back(ws->get_root(2));
      {
         // Past the ceiling the tree cannot even be assigned: set_root takes
         // a writer reference of its own. Nothing changes.
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "half");
         REQUIRE_THROWS_AS(tx.open_root(1).set_root(ws->get_root(2)), std::runtime_error);
         REQUIRE(tx.open_root(2).get<std::string>("c") == std::optional<std::string>("0"));
         tx.abort();
      }
      REQUIRE(read(0, "a") == std::optional<std::string>("1"));
      REQUIRE(read(1, "b") == std::optional<std::string>("0"));
      // One short of it: 186 snapshots, the slot's reference, this session's
      // working reference and the tree being assigned make 189; set_root's
      // own reference is the 190th and commit's is refused.
      held.resize(186);
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("a", "half");
         REQUIRE_NOTHROW(tx.open_root(1).set_root(ws->get_root(2)));
         REQUIRE_THROWS_AS(tx.commit(), std::runtime_error);
         REQUIRE(read(0, "a") == std::optional<std::string>("1"));
         REQUIRE(read(1, "b") == std::optional<std::string>("0"));
         // the transaction is still open and still holds its changes
         REQUIRE(tx.get<std::string>("a") == std::optional<std::string>("half"));
         tx.abort();
      }
      REQUIRE(read(0, "a") == std::optional<std::string>("1"));
      held.clear();
      auto tx = ws->start_transaction(0);
      tx.upsert("a", "2");
      tx.open_root(1).set_root(ws->get_root(2));
      tx.commit();
   }
   REQUIRE(read(0, "a") == std::optional<std::string>("2"));
   REQUIRE(read(1, "c") == std::optional<std::string>("0"));
   REQUIRE_FALSE(read(1, "b"));
   {
      auto tx = ws->start_transaction(0);
      tx.remove("a");
      tx.open_root(1).set_root({});
      tx.open_root(2).set_root({});
      tx.commit();
   }
   ws.reset();
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

namespace
{
   std::string scattered_key(const char* prefix, uint64_t i)
   {
      char k[48];
      std::snprintf(k, sizeof(k), "%s/%010llu", prefix,
                    (unsigned long long)(i * 2654435761u % 1000000007u));
      return k;
   }
   template <typename Cursor>
   void require_matches(Cursor& cur, const std::map<std::string, std::string>& expect)
   {
      uint64_t n = 0, bad = 0;
      for (cur.seek_begin(); !cur.is_end(); cur.next())
      {
         auto it = expect.find(std::string(cur.key()));
         if (it == expect.end() || cur.template value<std::string>() != it->second)
            ++bad;
         ++n;
      }
      REQUIRE(bad == 0);
      REQUIRE(n == expect.size());
   }
}  // namespace

// The session that hit the limit must be able to carry on after aborting,
// as a new session on the same database can.
TEST_CASE("Small Engine: the session that reached the storage limit can write again",
          "[small][limit]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   auto ws = db->start_write_session();
   std::map<std::string, std::string> expect;
   uint64_t                           n = 0;
   bool                               full = false;
   while (!full)
   {
      std::vector<std::pair<std::string, std::string>> batch;
      try
      {
         auto tx = ws->start_transaction(0);
         for (int j = 0; j < 200; ++j, ++n)
         {
            batch.emplace_back(scattered_key("s", n), std::string(100, char('a' + n % 26)));
            tx.upsert(batch.back().first, batch.back().second);
         }
         tx.commit();
         for (auto& kv : batch)
            expect[kv.first] = kv.second;
      }
      catch (const engine::storage_full&)
      {
         full = true;
      }
      REQUIRE(n < 5'000'000);
   }
   db->flush_releases();
   const auto u = db->usage();
   std::printf("limit: insert-only 100-byte values reached the limit at %llu keys, %.1f of %.1f "
               "MiB used (%.0f%%)\n",
               (unsigned long long)expect.size(), u.used_bytes / 1048576.0,
               u.limit_bytes / 1048576.0, 100.0 * u.used_bytes / u.limit_bytes);
   // Full means full: most of the store holds live data when it is refused.
   REQUIRE(u.used_bytes * 100 >= u.limit_bytes * 75);
   for (int round = 0; round < 3; ++round)
   {
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("tiny", std::to_string(round));
         tx.commit();
         expect["tiny"] = std::to_string(round);
      }
      {
         auto tx        = ws->start_transaction(0);
         const auto key = expect.begin()->first == "tiny" ? std::next(expect.begin())->first
                                                           : expect.begin()->first;
         REQUIRE(tx.remove(key) == 0);
         tx.commit();
         expect.erase(key);
      }
   }
   {
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      require_matches(cur, expect);
   }
   // Room made by removing is room that can be used.
   {
      auto tx = ws->start_transaction(0);
      uint64_t removed = 0;
      for (auto it = expect.begin(); it != expect.end() && removed < 20000;)
      {
         REQUIRE(tx.remove(it->first) == 0);
         it = expect.erase(it);
         ++removed;
      }
      tx.commit();
   }
   for (uint64_t i = 0; i < 10000; i += 200)
   {
      auto tx = ws->start_transaction(0);
      for (uint64_t j = i; j < i + 200; ++j)
      {
         tx.upsert(scattered_key("again", j), std::string(100, 'z'));
         expect[scattered_key("again", j)] = std::string(100, 'z');
      }
      tx.commit();
   }
   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);
   require_matches(cur, expect);
}

// Multi-line allocations on a fragmented store that is at its limit: the
// failure is storage_full, like any other way of running out.
TEST_CASE("Small Engine: running out of room for a large node is storage_full",
          "[small][limit]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   auto ws = db->start_write_session();
   std::map<std::string, std::string> expect;
   uint64_t                           seeded = 0;
   while (db->usage().used_bytes <= db->usage().limit_bytes * 6 / 10)
   {
      auto tx = ws->start_transaction(0);
      for (uint64_t j = seeded; j < seeded + 200; ++j)
         tx.upsert(scattered_key("s", j), std::string(900, char('a' + j % 26)));
      tx.commit();
      for (uint64_t j = seeded; j < seeded + 200; ++j)
         expect[scattered_key("s", j)] = std::string(900, char('a' + j % 26));
      seeded += 200;
   }
   {
      auto tx = ws->start_transaction(0);
      for (uint64_t j = 0; j < seeded; j += 3)
      {
         REQUIRE(tx.remove(scattered_key("s", j)) == 0);
         expect.erase(scattered_key("s", j));
      }
      tx.commit();
   }
   db->flush_releases();
   for (int attempt = 0; attempt < 4; ++attempt)
   {
      bool        full = false;
      std::string other;
      uint64_t    wrote = 0;
      try
      {
         auto tx = ws->start_transaction(0);
         for (uint64_t j = 0; j < 100000; ++j, ++wrote)
            tx.upsert(scattered_key("big", j), std::string(7900, char('A' + attempt)));
      }
      catch (const engine::storage_full&)
      {
         full = true;
      }
      catch (const std::exception& e)
      {
         other = e.what();
      }
      INFO("attempt " << attempt << " wrote " << wrote << " other: " << other);
      REQUIRE(other.empty());
      REQUIRE(full);
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      require_matches(cur, expect);
   }
   auto tx = ws->start_transaction(0);
   tx.upsert("tiny", "x");
   tx.commit();
}

namespace
{
   // A typed value whose release is slow when it runs on the background
   // releaser, which makes "the writer is ahead of the releaser" a certainty
   // instead of a matter of timing.
   struct slow_value
   {
      static constexpr uint16_t type_id = 77;
      struct t_pod
      {
         uint64_t n;
         static void retain(void*, const t_pod&) {}
         static void release(void*, const t_pod&)
         {
            if (std::this_thread::get_id() != writer.load())
            {
               std::this_thread::sleep_for(std::chrono::microseconds(20));
               released_in_background.fetch_add(1, std::memory_order_relaxed);
            }
         }
      };
      uint64_t n = 0;
      t_pod    take_pod() && { return {n}; }
      static slow_value make_from_pod(const t_pod& p, bool) { return {p.n}; }
      static inline std::atomic<std::thread::id> writer{};
      static inline std::atomic<uint64_t>        released_in_background{0};
   };
}  // namespace

TEST_CASE("Small Engine: a slow releaser makes the writer wait instead of failing",
          "[small][limit][releaser][slow-releaser]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   db->register_value_types<slow_value>();
   slow_value::writer = std::this_thread::get_id();
   slow_value::released_in_background = 0;
   const uint64_t upserts = 160000;
   {
      auto     ws = db->start_write_session();
      uint64_t n  = 0;
      while (n < upserts)
      {
         auto tx = ws->start_transaction(0);
         for (int j = 0; j < 200; ++j, ++n)
            tx.upsert(scattered_key("s", n % 20000), slow_value{n});
         REQUIRE_NOTHROW(tx.commit());
      }
   }
   db->flush_releases();
   const auto u = db->usage();
   std::printf("slow releaser: %llu upserts, %llu values released in the background, settled "
               "%.1f of %.1f MiB\n",
               (unsigned long long)upserts,
               (unsigned long long)slow_value::released_in_background.load(),
               u.used_bytes / 1048576.0, u.limit_bytes / 1048576.0);
   REQUIRE(slow_value::released_in_background.load() > 0);
   REQUIRE(u.used_bytes * 4 < u.limit_bytes);
}

// A writer that overwrites a small live set faster than the background
// releaser frees the old versions must be made to wait for it, not be told
// the store is full. PSITRI_SMALL_OVERWRITE_UPSERTS scales the run.
TEST_CASE("Small Engine: a writer ahead of the releaser waits instead of reporting full",
          "[small][limit][releaser]")
{
   const uint64_t upserts = env_u64("PSITRI_SMALL_OVERWRITE_UPSERTS", 6'000'000);
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   auto     ws        = db->start_write_session();
   uint64_t n         = 0, peak = 0;
   const auto t0 = std::chrono::steady_clock::now();
   while (n < upserts)
   {
      auto tx = ws->start_transaction(0);
      for (int j = 0; j < 200; ++j, ++n)
         tx.upsert(scattered_key("s", n % 20000), std::string(100, char('a' + n % 26)));
      tx.commit();
      if (n % 20000 == 0)
         peak = std::max(peak, db->usage().used_bytes);
   }
   const double secs =
       std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
   db->flush_releases();
   const auto u = db->usage();
   std::printf("overwrite: %llu upserts of 20,000 keys in %.1fs; peak used %.1f MiB, settled "
               "%.1f MiB of %.1f MiB\n",
               (unsigned long long)n, secs, peak / 1048576.0, u.used_bytes / 1048576.0,
               u.limit_bytes / 1048576.0);
   auto     rs  = db->start_read_session();
   auto     cur = rs->create_cursor(0);
   uint64_t keys = 0;
   for (cur.seek_begin(); !cur.is_end(); cur.next())
      ++keys;
   REQUIRE(keys == 20000);
}

// A transaction may be handed from the thread that started it to another,
// as long as only one of them uses it at a time. The root's writer lock goes
// with the transaction, not with a thread: the first thread asking for the
// same root through another session waits for the hand-over's commit.
// Worker threads only record what they saw; every check is made here.
TEST_CASE("Small Engine: a transaction handed to another thread keeps its root",
          "[small][threads][handover]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto a = db->start_write_session();
   auto b = db->start_write_session();
   for (int round = 0; round < 20; ++round)
   {
      // Started here, finished by another thread, while this thread asks for
      // the same root through session b.
      auto tx = std::make_unique<engine::transaction>(a->start_transaction(0));
      tx->upsert("k", "from-starter-" + std::to_string(round));
      std::promise<void> asking;        // this thread is about to ask for the root
      std::atomic<bool>  committing{false};
      std::string        finisher_error;
      std::thread finisher([&, handed = std::move(tx), go = asking.get_future()]() mutable {
         try
         {
            go.wait();
            handed->upsert("k", "from-finisher-" + std::to_string(round));
            committing = true;  // set before the root can possibly be released
            handed->commit();
            handed.reset();
         }
         catch (const std::exception& e)
         {
            finisher_error = e.what();
         }
      });
      asking.set_value();
      bool                       refused = false, waited = false;
      std::optional<std::string> seen;
      try
      {
         auto mine = b->start_transaction(0);  // must wait for the finisher's commit
         waited    = committing.load();
         seen      = mine.get<std::string>("k");
         mine.abort();
      }
      catch (const std::logic_error&)
      {
         refused = true;
      }
      finisher.join();
      REQUIRE(finisher_error.empty());
      REQUIRE_FALSE(refused);
      REQUIRE(waited);
      REQUIRE(seen == std::optional<std::string>("from-finisher-" + std::to_string(round)));
   }
   for (int round = 0; round < 20; ++round)
   {
      // Started elsewhere, finished here, while a third thread waits for the
      // root. The waiter announces itself, then blocks in start_transaction;
      // it can only get through after the commit below, so the value it reads
      // proves the order without any timing.
      std::unique_ptr<engine::transaction> tx;
      std::thread starter([&] {
         tx = std::make_unique<engine::transaction>(a->start_transaction(0));
         tx->upsert("r", "uncommitted-" + std::to_string(round));
      });
      starter.join();
      std::promise<void>         about_to_ask;
      std::optional<std::string> seen;
      std::string                waiter_error;
      std::thread waiter([&] {
         try
         {
            about_to_ask.set_value();
            auto mine = b->start_transaction(0);
            seen      = mine.get<std::string>("r");
            mine.abort();
         }
         catch (const std::exception& e)
         {
            waiter_error = e.what();
         }
      });
      about_to_ask.get_future().wait();
      tx->upsert("r", "committed-" + std::to_string(round));
      tx->commit();  // on a thread that did not start it
      tx.reset();
      waiter.join();
      REQUIRE(waiter_error.empty());
      REQUIRE(seen == std::optional<std::string>("committed-" + std::to_string(round)));
   }
}

// Not run by default: the cost of starting and committing a transaction,
// which is where the root's writer lock is taken and released.
TEST_CASE("Small Engine: transaction start/commit throughput", "[.bench]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 16));
   auto           ws = db->start_write_session();
   const uint64_t n  = env_u64("PSITRI_SMALL_BENCH_TXNS", 3'000'000);
   for (int rep = 0; rep < 5; ++rep)
   {
      const auto t0 = std::chrono::steady_clock::now();
      for (uint64_t i = 0; i < n; ++i)
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("counter", std::string_view(reinterpret_cast<const char*>(&i), 8));
         tx.commit();
      }
      const double secs =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("bench: %llu one-upsert transactions, %.0f tx/s\n", (unsigned long long)n,
                  double(n) / secs);
   }
}


namespace
{
   // Fill a 4-segment store with one-upsert transactions until it refuses.
   std::vector<std::string> fill_to_the_brim(database& db, engine::write_session& ws)
   {
      std::vector<std::string> keys;
      for (uint64_t n = 0;; ++n)
      {
         try
         {
            auto tx = ws.start_transaction(0);
            tx.upsert(scattered_key("s", n), std::string(100, char('a' + n % 26)));
            tx.commit();
            keys.push_back(scattered_key("s", n));
         }
         catch (const engine::storage_full&)
         {
            break;
         }
         if (n > 5'000'000)
            throw std::runtime_error("the limit never arrived");
      }
      db.flush_releases();
      return keys;
   }
}  // namespace

// A store filled until it refuses, with nothing aborted to make room, must
// still let keys be removed: a remove copies the path to its key before the
// old version is freed, and that copy comes out of a reserve that ordinary
// writes cannot touch.
TEST_CASE("Small Engine: a brim-full store can be emptied one remove at a time",
          "[small][limit][brim]")
{
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 4);
   auto db   = database::open(tmp.path, psitri::open_mode::create_only, cfg);
   auto ws   = db->start_write_session();
   auto keys = fill_to_the_brim(*db, *ws);
   const auto full = db->usage();
   std::printf("brim: %zu keys; used %.2f of %.2f MiB, reserve %.2f MiB\n", keys.size(),
               full.used_bytes / 1048576.0, full.limit_bytes / 1048576.0,
               full.reserve_bytes / 1048576.0);
   REQUIRE(full.used_bytes + full.reserve_bytes + (64u << 10) >= full.limit_bytes);
   // ordinary writes are still refused, from this session and from a new one
   // (a session can have a few dozen lines of its own left, so not
   // necessarily the very first of them)
   auto refused = [](engine::write_session& w) {
      try
      {
         auto tx = w.start_transaction(0);
         for (int j = 0; j < 200; ++j)
            tx.upsert("more/" + std::to_string(j), std::string(100, 'x'));
      }
      catch (const engine::storage_full&)
      {
         return true;
      }
      return false;
   };
   REQUIRE(refused(*ws));
   {
      auto fresh = db->start_write_session();
      REQUIRE(refused(*fresh));
   }
   // every key goes, one transaction each, alternating sessions
   auto     other   = db->start_write_session();
   uint64_t removed = 0;
   for (const auto& key : keys)
   {
      auto tx = (removed % 2 ? other : ws)->start_transaction(0);
      REQUIRE(tx.remove(key) == 0);
      tx.commit();
      ++removed;
      if (removed == keys.size() / 2)
      {
         // half way: room made by removing is room for writing
         auto fill = ws->start_transaction(0);
         for (int j = 0; j < 1000; ++j)
            fill.upsert("again/" + std::to_string(j), std::string(100, 'z'));
         fill.commit();
         auto drop = ws->start_transaction(0);
         for (int j = 0; j < 1000; ++j)
            REQUIRE(drop.remove("again/" + std::to_string(j)) == 0);
         drop.commit();
      }
   }
   {
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE_FALSE(cur.seek_begin());
   }
   ws.reset();
   other.reset();
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

// Two other ways out of a full store: drop the whole root, or reopen with a
// higher limit.
TEST_CASE("Small Engine: a brim-full store can drop its root or be reopened larger",
          "[small][limit][brim]")
{
   {
      temporary_directory tmp;
      auto db   = database::open(tmp.path, psitri::open_mode::create_only,
                                 small_config(psitri::sync_mode::none, 4));
      auto ws   = db->start_write_session();
      auto keys = fill_to_the_brim(*db, *ws);
      {
         auto tx = ws->start_transaction(0);
         tx.primary().set_root({});
         REQUIRE_NOTHROW(tx.commit());
      }
      db->flush_releases();
      REQUIRE(db->usage().used_bytes <= 64 * 128);
      auto tx = ws->start_transaction(0);
      for (int j = 0; j < 1000; ++j)
         tx.upsert(keys[j], "back");
      tx.commit();
   }
   {
      temporary_directory tmp;
      size_t              count = 0;
      {
         auto db = database::open(tmp.path, psitri::open_mode::create_only,
                                  small_config(psitri::sync_mode::none, 4));
         auto ws = db->start_write_session();
         count   = fill_to_the_brim(*db, *ws).size();
      }
      auto db = database::open(tmp.path, psitri::open_mode::open_existing,
                               small_config(psitri::sync_mode::none, 8));
      REQUIRE(db->usage().limit_bytes == 8ull * 8 * 1024 * 1024);
      auto ws = db->start_write_session();
      for (int round = 0; round < 50; ++round)
      {
         auto tx = ws->start_transaction(0);
         for (int j = 0; j < 200; ++j)
            tx.upsert("grown/" + std::to_string(round * 200 + j), std::string(100, 'g'));
         tx.commit();
      }
      auto     rs  = db->start_read_session();
      auto     cur = rs->create_cursor(0);
      uint64_t n   = 0;
      for (cur.seek_begin(); !cur.is_end(); cur.next())
         ++n;
      REQUIRE(n == count + 10000);
   }
}

// A version kept alive by a reader is released when the reader lets go, but
// the release itself runs at a commit. A writer that needs the room before
// its next commit must get it: retired versions nobody reads any more are
// drained before the store is declared full.
TEST_CASE("Small Engine: room held by a version no reader uses any more is found",
          "[small][limit][retired]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   auto           ws   = db->start_write_session();
   const uint64_t keys = 75000;
   auto rewrite = [&](char fill) {
      auto tx = ws->start_transaction(0);
      for (uint64_t j = 0; j < keys; ++j)
         tx.upsert(scattered_key("s", j), std::string(100, fill));
      tx.commit();
   };
   for (uint64_t j = 0; j < keys; j += 200)
   {
      auto tx = ws->start_transaction(0);
      for (uint64_t k = j; k < j + 200; ++k)
         tx.upsert(scattered_key("s", k), std::string(100, 'a'));
      tx.commit();
   }
   db->flush_releases();
   const auto one_version = db->usage().used_bytes;
   REQUIRE(one_version * 2 < db->usage().limit_bytes - db->usage().reserve_bytes);
   REQUIRE(one_version * 3 > db->usage().limit_bytes);
   for (int round = 0; round < 5; ++round)
   {
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(0);  // keeps the current version
         rewrite(char('b' + round));       // two versions now
         REQUIRE(cur.get<std::string>(scattered_key("s", 7)).has_value());
      }
      // the reader is gone; nothing has committed since
      REQUIRE_NOTHROW(rewrite(char('B' + round)));
   }
}

// Sessions that are not in a transaction do not hold storage hostage.
TEST_CASE("Small Engine: idle write sessions do not use up the store", "[small][limit][idle]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   std::vector<std::shared_ptr<engine::write_session>> sessions;
   for (int i = 0; i < 12; ++i)
   {
      sessions.push_back(db->start_write_session());
      auto tx = sessions.back()->start_transaction(0);
      REQUIRE_NOTHROW(tx.upsert("key" + std::to_string(i), "v"));
      tx.commit();
   }
   // and each of them can still write afterwards
   for (int i = 0; i < 12; ++i)
   {
      auto tx = sessions[i]->start_transaction(0);
      tx.upsert("again" + std::to_string(i), "v");
      tx.commit();
   }
   auto     rs  = db->start_read_session();
   auto     cur = rs->create_cursor(0);
   uint64_t n   = 0;
   for (cur.seek_begin(); !cur.is_end(); cur.next())
      ++n;
   REQUIRE(n == 24);
}

namespace
{
   // A typed value two kilobytes wide whose release is slow on the background
   // releaser: a few thousand dead ones are megabytes the writer cannot have
   // until the releaser gets to them.
   struct wide_slow_value
   {
      static constexpr uint16_t type_id = 78;
      struct t_pod
      {
         uint64_t n;
         char     filler[2000];
         static void retain(void*, const t_pod&) {}
         static void release(void*, const t_pod&)
         {
            if (std::this_thread::get_id() != writer.load())
            {
               std::this_thread::sleep_for(std::chrono::microseconds(100));
               released_in_background.fetch_add(1, std::memory_order_relaxed);
            }
         }
      };
      t_pod pod{};
      t_pod take_pod() && { return pod; }
      static wide_slow_value make_from_pod(const t_pod& p, bool) { return {p}; }
      static inline std::atomic<std::thread::id> writer{};
      static inline std::atomic<uint64_t>        released_in_background{0};
   };
}  // namespace

// The wait itself: dead versions queued for a releaser that is far behind
// are most of the store. Only waiting for the releaser gets the room back;
// taking free lines from other segments cannot, because there are none.
TEST_CASE("Small Engine: a writer waits for queued releases before reporting full",
          "[small][limit][releaser][wait]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   db->register_value_types<wide_slow_value>();
   wide_slow_value::writer                 = std::this_thread::get_id();
   wide_slow_value::released_in_background = 0;
   const uint64_t live = 2000, upserts = 40000;  // 4 MiB live, 80 MiB written
   {
      auto     ws = db->start_write_session();
      uint64_t n  = 0;
      while (n < upserts)
      {
         auto tx = ws->start_transaction(0);
         for (int j = 0; j < 200; ++j, ++n)
         {
            wide_slow_value v;
            v.pod.n = n;
            tx.upsert(scattered_key("w", n % live), std::move(v));
         }
         REQUIRE_NOTHROW(tx.commit());
      }
   }
   db->flush_releases();
   std::printf("wait: %llu wide upserts, %llu released in the background, settled %.1f MiB\n",
               (unsigned long long)upserts,
               (unsigned long long)wide_slow_value::released_in_background.load(),
               db->usage().used_bytes / 1048576.0);
   REQUIRE(db->usage().used_bytes * 4 < db->usage().limit_bytes);
}

// Two writers on two roots at the ceiling, each overwriting its own keys and
// both feeding the releaser. When one is refused it aborts and removes a
// batch of its keys to make room; those removes must always go through.
// Also the releaser queue's only multi-writer exercise: at the end every key
// is accounted for and, once both roots are emptied, so is every line.
TEST_CASE("Small Engine: two writers at the ceiling recover by removing",
          "[small][limit][releaser][two-writers]")
{
   const uint64_t upserts = env_u64("PSITRI_SMALL_TWO_WRITER_UPSERTS", 1'200'000);
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   struct outcome
   {
      std::map<std::string, std::string> expect;
      uint64_t                           refused = 0;
      std::string                        error;
   };
   outcome results[2];
   auto writer = [&](uint32_t root) {
      outcome& out = results[root];
      try
      {
         auto           ws   = db->start_write_session();
         const uint64_t span = 100000;  // more than half the store each, if all were live
         uint64_t       n    = 0;
         while (n < upserts)
         {
            std::vector<std::pair<std::string, std::string>> batch;
            try
            {
               auto tx = ws->start_transaction(root);
               for (int j = 0; j < 200; ++j, ++n)
               {
                  batch.emplace_back(scattered_key(root ? "b" : "a", n % span),
                                     std::string(100, char('a' + n % 26)));
                  tx.upsert(batch.back().first, batch.back().second);
               }
               tx.commit();
               for (auto& kv : batch)
                  out.expect[kv.first] = kv.second;
            }
            catch (const engine::storage_full&)
            {
               ++out.refused;
               // make room: drop a tenth of what this writer has, one
               // transaction per 100 keys; a remove is never refused
               uint64_t drop = out.expect.size() / 10 + 1;
               while (drop && !out.expect.empty())
               {
                  auto tx = ws->start_transaction(root);
                  for (int j = 0; j < 100 && drop && !out.expect.empty(); ++j, --drop)
                  {
                     auto it = out.expect.begin();
                     if (tx.remove(it->first) != 0)
                        throw std::runtime_error("a committed key was missing");
                     out.expect.erase(it);
                  }
                  tx.commit();
               }
            }
         }
      }
      catch (const std::exception& e)
      {
         out.error = e.what();
      }
   };
   std::thread t0(writer, 0), t1(writer, 1);
   t0.join();
   t1.join();
   std::printf("two writers: refused %llu and %llu times; %zu and %zu keys live\n",
               (unsigned long long)results[0].refused, (unsigned long long)results[1].refused,
               results[0].expect.size(), results[1].expect.size());
   REQUIRE(results[0].error.empty());
   REQUIRE(results[1].error.empty());
   for (uint32_t root = 0; root < 2; ++root)
   {
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(root);
      require_matches(cur, results[root].expect);
   }
   {
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.primary().set_root({});
      tx.open_root(1).set_root({});
      tx.commit();
   }
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

// The releaser's queue is fed by every writer thread. Aborting a large
// transaction hands its private subtrees to the queue outside any shared
// lock, so two writers on two roots aborting at once push concurrently. A
// queue that loses or duplicates an entry leaks storage or frees it twice.
TEST_CASE("Small Engine: two writers feed the release queue at once",
          "[small][release-queue]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_QUEUE_ROUNDS", 400);
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 64));
   std::string errors[2];
   auto writer = [&](uint32_t root) {
      try
      {
         auto ws = db->start_write_session();
         {
            auto tx = ws->start_transaction(root);
            for (int j = 0; j < 500; ++j)
               tx.upsert(scattered_key("keep", j), std::string(60, 'k'));
            tx.commit();
         }
         for (uint64_t r = 0; r < rounds; ++r)
         {
            auto tx = ws->start_transaction(root);
            for (int j = 0; j < 3000; ++j)
               tx.upsert(scattered_key("gone", r * 3000 + j), std::string(60, 'g'));
            tx.abort();  // thousands of private nodes die here, most via the queue
         }
      }
      catch (const std::exception& e)
      {
         errors[root] = e.what();
      }
   };
   std::thread t0(writer, 0), t1(writer, 1);
   t0.join();
   t1.join();
   REQUIRE(errors[0].empty());
   REQUIRE(errors[1].empty());
   db->flush_releases();
   for (uint32_t root = 0; root < 2; ++root)
   {
      auto     rs  = db->start_read_session();
      auto     cur = rs->create_cursor(root);
      uint64_t n   = 0;
      for (cur.seek_begin(); !cur.is_end(); cur.next())
      {
         REQUIRE(cur.key().starts_with("keep/"));
         REQUIRE(cur.value<std::string>() == std::optional<std::string>(std::string(60, 'k')));
         ++n;
      }
      REQUIRE(n == 500);
   }
   {
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.primary().set_root({});
      tx.open_root(1).set_root({});
      tx.commit();
   }
   db->flush_releases();
   // every line is accounted for: nothing lost in the queue, nothing freed twice
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

namespace
{
   std::string file_bytes(const std::filesystem::path& p)
   {
      std::string out;
      if (std::FILE* f = std::fopen(p.c_str(), "rb"))
      {
         char   buf[1 << 16];
         size_t n;
         while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
            out.append(buf, n);
         std::fclose(f);
      }
      return out;
   }
}  // namespace

// Lowering max_segments below what a store already occupies cannot be
// honoured (nothing is ever moved). The open must be refused before any of
// the store is mapped, with the two sizes in the message, and must leave
// every file exactly as it was.
TEST_CASE("Small Engine: reopening with a limit below the store's size is refused untouched",
          "[small][open][limit][shrink]")
{
   temporary_directory tmp;
   const model         m{20000, 100};
   uint64_t            mapped = 0;
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_only,
                               small_config(psitri::sync_mode::none, 16));
      auto ws = db->start_write_session();
      for (uint64_t seq = 1; seq <= 400; ++seq)
      {
         auto tx = ws->start_transaction(0);
         m.apply(tx, seq);
         tx.commit();
      }
      mapped = db->usage().mapped_bytes;
      REQUIRE(mapped > 2ull * 8 * 1024 * 1024);
   }
   const char* files[] = {"meta.part", "free.part", "refs.part", "large/meta.part"};
   std::vector<std::string> before;
   for (auto f : files)
      before.push_back(file_bytes(tmp.path / f));
   const auto data_size = std::filesystem::file_size(tmp.path / "data.part");

   for (int attempt = 0; attempt < 2; ++attempt)
   {
      std::string what;
      try
      {
         database::open(tmp.path, psitri::open_mode::open_existing,
                        small_config(psitri::sync_mode::none, 2));
         FAIL("a store of more than two segments was opened with max_segments = 2");
      }
      catch (const std::runtime_error& e)
      {
         what = e.what();
      }
      INFO(what);
      REQUIRE(what.find("max_segments") != std::string::npos);
      REQUIRE(what.find(std::to_string(mapped / (8 * 1024 * 1024)) + " segments") !=
              std::string::npos);
      REQUIRE(what.find("2 segments") != std::string::npos);
      REQUIRE(database::closed_cleanly(tmp.path));
      for (size_t i = 0; i < std::size(files); ++i)
         REQUIRE(file_bytes(tmp.path / files[i]) == before[i]);
      REQUIRE(std::filesystem::file_size(tmp.path / "data.part") == data_size);
   }
   // exactly at the store's size is allowed; so is the original limit
   for (uint32_t limit : {uint32_t(mapped / (8 * 1024 * 1024)), 16u})
   {
      auto db = database::open(tmp.path, psitri::open_mode::open_existing,
                               small_config(psitri::sync_mode::none, limit));
      REQUIRE_FALSE(db->recovered_on_open());
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE(m.verify(cur) == 400);
   }
}

// The same when it is the large-value arena that does not fit. It is opened
// second; the refusal must come before the ordinary arena has been opened and
// had its clean mark cleared, so that nothing at all is touched.
TEST_CASE("Small Engine: a limit below the large-value arena's size is refused untouched",
          "[small][open][limit][shrink][shrink-large]")
{
   temporary_directory tmp;
   // One ordinary segment and no large one to begin with ...
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_only,
                               small_config(psitri::sync_mode::none, 1));
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.upsert("small", "value");
      tx.commit();
      REQUIRE(db->usage().mapped_bytes == 8ull * 1024 * 1024);
      REQUIRE(db->usage().large_mapped_bytes == 0);
   }
   // ... then large values, which make the large arena grow by four segments.
   const std::string big(300u << 10, 'L'), chunked(3u << 20, 'C');
   uint64_t          large_segments = 0;
   {
      auto db = database::open(tmp.path, psitri::open_mode::open_existing,
                               small_config(psitri::sync_mode::none, 16));
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.upsert("big", big);
      tx.upsert("chunked", chunked);
      tx.commit();
      const auto u = db->usage();
      REQUIRE(u.mapped_bytes == 8ull * 1024 * 1024);
      large_segments = u.large_mapped_bytes / (65536ull * 32 * 1024);
      REQUIRE(large_segments == 4);
   }
   const char* files[] = {"meta.part",       "free.part",       "refs.part",
                          "large/meta.part", "large/free.part", "large/refs.part"};
   const char* sized[] = {"data.part", "large/data.part"};
   std::vector<std::string>                     before;
   std::vector<std::filesystem::file_time_type> written;
   std::vector<std::uintmax_t>                  sizes;
   for (auto f : files)
   {
      before.push_back(file_bytes(tmp.path / f));
      written.push_back(std::filesystem::last_write_time(tmp.path / f));
   }
   for (auto f : sized)
   {
      sizes.push_back(std::filesystem::file_size(tmp.path / f));
      written.push_back(std::filesystem::last_write_time(tmp.path / f));
   }
   auto untouched = [&] {
      size_t w = 0;
      for (size_t i = 0; i < std::size(files); ++i, ++w)
      {
         INFO(files[i]);
         REQUIRE(file_bytes(tmp.path / files[i]) == before[i]);
         REQUIRE((std::filesystem::last_write_time(tmp.path / files[i]) == written[w]));
      }
      for (size_t i = 0; i < std::size(sized); ++i, ++w)
      {
         INFO(sized[i]);
         REQUIRE(std::filesystem::file_size(tmp.path / sized[i]) == sizes[i]);
         REQUIRE((std::filesystem::last_write_time(tmp.path / sized[i]) == written[w]));
      }
   };
   // max_segments 2 covers the ordinary arena (1 segment) but not the large one (4).
   for (int attempt = 0; attempt < 2; ++attempt)
   {
      std::string what;
      try
      {
         database::open(tmp.path, psitri::open_mode::open_existing,
                        small_config(psitri::sync_mode::none, 2));
         FAIL("a store with four large-value segments was opened with max_segments = 2");
      }
      catch (const std::runtime_error& e)
      {
         what = e.what();
      }
      INFO(what);
      REQUIRE(what.find("max_segments") != std::string::npos);
      REQUIRE(what.find("4 segments") != std::string::npos);
      REQUIRE(what.find("large") != std::string::npos);
      REQUIRE(database::closed_cleanly(tmp.path));
      untouched();
   }
   for (uint32_t limit : {4u, 16u})
   {
      auto db = database::open(tmp.path, psitri::open_mode::open_existing,
                               small_config(psitri::sync_mode::none, limit));
      REQUIRE_FALSE(db->recovered_on_open());
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE(cur.get<std::string>("small") == std::optional<std::string>("value"));
      REQUIRE(cur.get<std::string>("big") == std::optional<std::string>(big));
      REQUIRE(cur.get<std::string>("chunked") == std::optional<std::string>(chunked));
   }
}

// A session keeps its segment from one transaction to the next, whichever
// segment that is. It used to keep it only while the segment was the highest
// it had ever written to: after its first move to a lower-numbered one,
// every transaction started by abandoning the segment and claiming again,
// under the claim mutex, with a scan of all segments.
TEST_CASE("Small Engine: a session keeps its segment across transactions after reusing one",
          "[small][segments]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 64));
   auto        ws = db->start_write_session();
   const auto& st = db->store().stats();
   const std::string value(16, 'x');
   uint64_t          t = 0;
   auto run = [&](uint64_t txns) {
      for (uint64_t end = t + txns; t < end; ++t)
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("acct/" + std::to_string((t * 7919) % 20000), value);
         tx.commit();
      }
   };
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 20000; ++i)
         tx.upsert("acct/" + std::to_string(i), value);
      tx.commit();
   }
   // churn until the session has gone back to a segment used before
   const uint64_t reused_at_start = st.segments_reused.load();
   while (st.segments_reused.load() == reused_at_start)
   {
      run(10000);
      REQUIRE(t < 2'000'000);
   }
   const uint64_t reused_after = t;
   const uint64_t switches     = st.range_switches.load();
   const uint64_t claims       = st.segments_reused.load() + st.segments_grown.load();
   const auto t0 = std::chrono::steady_clock::now();
   run(100000);
   const double secs =
       std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
   const uint64_t more_switches = st.range_switches.load() - switches;
   const uint64_t more_claims = st.segments_reused.load() + st.segments_grown.load() - claims;
   std::printf("segments: first reuse within %llu transactions; over the next 100,000: %llu "
               "segment switches, %llu claims, %.0f tx/s; mapped %.0f MiB\n",
               (unsigned long long)reused_after, (unsigned long long)more_switches,
               (unsigned long long)more_claims, 100000 / secs,
               db->usage().mapped_bytes / 1048576.0);
   // a switch happens when a segment fills, a few times in 100,000 small
   // transactions; not once per transaction
   REQUIRE(more_switches < 200);
   REQUIRE(more_claims < 200);
}

// set_root() with a snapshot that nobody else holds. The snapshot's version
// can share storage with the live tree in a way its own root's reference
// count does not show: a wide router that a later version copied has its
// tail lines retained by the copy. A transaction working on such a tree
// must treat it as shared; it used to write into those lines in place, so
// readers of the committed root saw uncommitted data, and an aborted insert
// stayed.
TEST_CASE("Small Engine: set_root of an old snapshot does not write through to the live tree",
          "[small][set-root]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 64));
   auto ws = db->start_write_session();
   auto key = [](int first, int n) {
      std::string k;
      k.push_back(char(first));
      return k + "key" + std::to_string(n);
   };
   auto live = [&](const std::string& k) {
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      return cur.get<std::string>(k);
   };
   {
      // version 1: every first byte, three keys each: a 256-way root router
      auto tx = ws->start_transaction(0);
      for (int n = 0; n < 3; ++n)
         for (int i = 0; i <= 255; ++i)
            tx.upsert(key((i * 97) % 256, n), "v1");
      tx.commit();
   }
   auto snapshot = ws->get_root(0);
   {
      // version 2 copies the root router and shares version 1's lines
      auto tx = ws->start_transaction(0);
      tx.upsert(key(5, 0), "v2");
      tx.commit();
   }
   {
      // one more commit: now only `snapshot` refers to version 1's root
      auto tx = ws->start_transaction(0);
      tx.upsert(key(6, 0), "v3");
      tx.commit();
   }
   db->flush_releases();
   REQUIRE(live(key(150, 1)) == std::optional<std::string>("v1"));

   // a held cursor on the committed root, as a concurrent reader would have
   auto reader = db->start_read_session();
   auto held   = reader->create_cursor(0);
   {
      auto tx = ws->start_transaction(0);
      tx.primary().set_root(std::move(snapshot));
      tx.upsert(key(150, 1), "XX");       // same size as the committed value
      tx.upsert(key(150, 7), "new-key");
      REQUIRE(tx.get<std::string>(key(150, 1)) == std::optional<std::string>("XX"));
      // nothing of this is committed: readers must not see it
      REQUIRE(held.get<std::string>(key(150, 1)) == std::optional<std::string>("v1"));
      REQUIRE_FALSE(held.get<std::string>(key(150, 7)));
      REQUIRE(live(key(150, 1)) == std::optional<std::string>("v1"));
      REQUIRE_FALSE(live(key(150, 7)));
      tx.abort();
   }
   db->flush_releases();
   REQUIRE(live(key(150, 1)) == std::optional<std::string>("v1"));
   REQUIRE_FALSE(live(key(150, 7)));
   REQUIRE(live(key(5, 0)) == std::optional<std::string>("v2"));
   REQUIRE(live(key(6, 0)) == std::optional<std::string>("v3"));

   // and committed this time, from another old snapshot nobody else holds
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("later", "v4");
      tx.commit();
   }
   auto old_version = ws->get_root(0);  // has "later"; about to be the only holder
   {
      auto tx = ws->start_transaction(0);
      tx.upsert(key(8, 0), "v5");
      tx.commit();
      auto tx2 = ws->start_transaction(0);
      tx2.upsert(key(9, 0), "v6");
      tx2.commit();
   }
   db->flush_releases();
   auto before = ws->get_root(0);
   {
      auto tx = ws->start_transaction(0);
      tx.primary().set_root(std::move(old_version));
      tx.upsert(key(150, 1), "YY");
      REQUIRE(tx.remove(key(7, 2)) == 0);
      REQUIRE(held.get<std::string>(key(150, 1)) == std::optional<std::string>("v1"));
      REQUIRE(before.get<std::string>(key(7, 2)) == std::optional<std::string>("v1"));
      tx.commit();
   }
   REQUIRE(live(key(150, 1)) == std::optional<std::string>("YY"));
   REQUIRE_FALSE(live(key(7, 2)));
   REQUIRE_FALSE(live(key(9, 0)).value_or("") == "v6");  // root 0 is the old version again
   // what was live before, and the reader's cursor, are what they were
   REQUIRE(before.get<std::string>(key(150, 1)) == std::optional<std::string>("v1"));
   REQUIRE(before.get<std::string>(key(7, 2)) == std::optional<std::string>("v1"));
   REQUIRE(before.get<std::string>(key(9, 0)) == std::optional<std::string>("v6"));
   REQUIRE(held.get<std::string>(key(150, 1)) == std::optional<std::string>("v1"));
   REQUIRE(held.get<std::string>(key(6, 0)) == std::optional<std::string>("v3"));
}

// The reference a transaction keeps for a tree given to set_root is given
// back when the transaction ends, whichever way it ends.
TEST_CASE("Small Engine: trees given to set_root are released with the transaction",
          "[small][set-root]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 64));
   {
      auto ws = db->start_write_session();
      for (int round = 0; round < 40; ++round)
      {
         {
            auto tx = ws->start_transaction(0);
            for (int j = 0; j < 300; ++j)
               tx.upsert(scattered_key("k", round * 300 + j), std::string(50, char('a' + round % 26)));
            tx.commit();
         }
         auto old = ws->get_root(0);
         {
            auto tx = ws->start_transaction(0);
            tx.upsert(scattered_key("k", round), "changed");
            tx.commit();
         }
         auto tx = ws->start_transaction(0);
         auto r1 = tx.open_root(1);
         r1.set_root(std::move(old));           // root 1 := the older version of root 0
         r1.upsert("only-in-1", std::to_string(round));
         auto displaced = tx.primary().set_root(ws->get_root(0));  // a no-op assignment
         if (round % 3 == 0)
            tx.abort();
         else
            tx.commit();
      }
      auto tx = ws->start_transaction(0);
      tx.primary().set_root({});
      tx.open_root(1).set_root({});
      tx.commit();
   }
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

namespace
{
   // A typed value that counts how often the engine duplicates and destroys
   // its bytes: a copy of a leaf shows up as a retain.
   struct counted_value
   {
      static constexpr uint16_t type_id = 78;
      struct t_pod
      {
         uint64_t n;
         static void retain(void*, const t_pod&) { retains.fetch_add(1, std::memory_order_relaxed); }
         static void release(void*, const t_pod&) { releases.fetch_add(1, std::memory_order_relaxed); }
      };
      uint64_t n = 0;
      t_pod    take_pod() && { return {n}; }
      static counted_value make_from_pod(const t_pod& p, bool) { return {p.n}; }
      static inline std::atomic<uint64_t> retains{0};
      static inline std::atomic<uint64_t> releases{0};
   };
}  // namespace

// gofractally/psitri#33. A value that is about to be removed or replaced is
// not copied first. The copy was made of the whole leaf, only to be freed in
// the same call, so on a full store a remove needed a free run as long as the
// value it was deleting and could be refused.
TEST_CASE("Small Engine: a full store of multi-line values can be emptied one remove at a time",
          "[small][limit][brim][multiline]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 2));
   auto ws = db->start_write_session();
   std::mt19937_64                                rng(33);
   std::vector<std::pair<std::string, size_t>>    rows;
   for (uint64_t n = 0;; ++n)
   {
      const size_t size = 900 + rng() % 2200;  // 8 to 25 lines
      try
      {
         auto tx = ws->start_transaction(0);
         tx.upsert(scattered_key("m", n), std::string(size, char('a' + n % 26)));
         tx.commit();
         rows.emplace_back(scattered_key("m", n), size);
      }
      catch (const engine::storage_full&)
      {
         break;
      }
      REQUIRE(n < 1'000'000);
   }
   db->flush_releases();
   const auto full = db->usage();
   std::printf("multi-line brim: %zu values; used %.2f of %.2f MiB, reserve %.2f MiB\n",
               rows.size(), full.used_bytes / 1048576.0, full.limit_bytes / 1048576.0,
               full.reserve_bytes / 1048576.0);
   std::shuffle(rows.begin(), rows.end(), rng);
   uint64_t refused = 0;
   for (const auto& [key, size] : rows)
   {
      auto tx = ws->start_transaction(0);
      try
      {
         REQUIRE(tx.remove(key) == 0);
         tx.commit();
      }
      catch (const engine::storage_full&)
      {
         ++refused;
         tx.abort();
      }
   }
   REQUIRE(refused == 0);
   ws.reset();
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

TEST_CASE("Small Engine: replacing or removing a value does not copy the old one first",
          "[small][cow][no-clone]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   db->register_value_types<counted_value>();
   auto ws = db->start_write_session();
   const std::string at_leaf = "k/leaf";  // a leaf of its own
   const std::string at_term = "k";       // ends where a router begins: the router's term
   {
      auto tx = ws->start_transaction(0);
      for (int i = 0; i < 200; ++i)
         tx.upsert("k/pad/" + std::to_string(i), std::string(40, 'p'));
      tx.upsert(at_leaf, counted_value{1});
      tx.upsert(at_term, counted_value{2});
      tx.commit();
   }
   db->flush_releases();
   auto settle = [&] {
      counted_value::retains  = 0;
      counted_value::releases = 0;
   };
   // every transaction below starts on a tree it shares with the committed
   // version, so the value's leaf is shared when the write reaches it
   for (const std::string& key : {at_leaf, at_term})
   {
      INFO("key " << key);
      settle();
      {  // same size: the old leaf used to be cloned and the clone overwritten
         auto tx = ws->start_transaction(0);
         tx.upsert(key, counted_value{10});
         CHECK(counted_value::retains.load() == 0);
         CHECK(counted_value::releases.load() == 0);  // the committed version still holds it
         tx.commit();
      }
      db->flush_releases();
      CHECK(counted_value::retains.load() == 0);
      CHECK(counted_value::releases.load() == 1);  // the replaced value, once
      settle();
      {  // another size: plain bytes replace the typed value
         auto tx = ws->start_transaction(0);
         tx.upsert(key, std::string(300, 'b'));
         CHECK(counted_value::retains.load() == 0);
         tx.commit();
      }
      db->flush_releases();
      CHECK(counted_value::retains.load() == 0);
      CHECK(counted_value::releases.load() == 1);
      {
         auto tx = ws->start_transaction(0);
         tx.upsert(key, counted_value{20});
         tx.commit();
      }
      db->flush_releases();
      settle();
      {  // remove: the leaf used to be cloned and the clone freed
         auto tx = ws->start_transaction(0);
         REQUIRE(tx.remove(key) == 0);
         CHECK(counted_value::retains.load() == 0);
         CHECK(counted_value::releases.load() == 0);
         tx.commit();
      }
      db->flush_releases();
      CHECK(counted_value::retains.load() == 0);
      CHECK(counted_value::releases.load() == 1);
      {  // an aborted remove and an aborted overwrite leave the value alone
         auto put = ws->start_transaction(0);
         put.upsert(key, counted_value{30});
         put.commit();
         db->flush_releases();
         settle();
         auto tx = ws->start_transaction(0);
         REQUIRE(tx.remove(key) == 0);
         tx.abort();
         auto tx2 = ws->start_transaction(0);
         tx2.upsert(key, std::string(50, 'c'));
         tx2.abort();
         db->flush_releases();
         CHECK(counted_value::retains.load() == 0);
         CHECK(counted_value::releases.load() == 0);
         auto rs = db->start_read_session();
         auto cur = rs->create_cursor(0);
         REQUIRE(cur.template get<counted_value>(key).has_value());
      }
   }
   // a split that moves a shared value into a new node duplicates it: one
   // retain, balanced when the old version dies
   settle();
   {
      auto tx = ws->start_transaction(0);
      tx.upsert(at_leaf + "/deeper", std::string(10, 'd'));  // at_leaf's leaf becomes a term
      tx.commit();
   }
   db->flush_releases();
   CHECK(counted_value::retains.load() == counted_value::releases.load());
   {
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      auto a   = cur.template get<counted_value>(at_leaf);
      auto b   = cur.template get<counted_value>(at_term);
      REQUIRE(a.has_value());
      REQUIRE(b.has_value());
      CHECK(a->n == 30);
      CHECK(b->n == 30);
   }
   ws.reset();
   db.reset();
   // every stored value was released exactly as often as it was stored or copied
}

// Not part of the suite: `small-engine-tests "[.bench][overwrite]"`.
TEST_CASE("Small Engine: overwrite throughput by value size", "[.bench][overwrite]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 64));
   auto ws = db->start_write_session();
   for (size_t size : {size_t(100), size_t(4000), size_t(200) << 10, size_t(1536) << 10})
   {
      const int   keys   = 16;
      const int   rounds = size > (64u << 10) ? 400 : 20000;
      std::string value(size, 'v');
      {
         auto tx = ws->start_transaction(0);
         for (int k = 0; k < keys; ++k)
            tx.upsert("o/" + std::to_string(k), value);
         tx.commit();
      }
      for (int same = 1; same >= 0; --same)
      {
         const auto t0 = std::chrono::steady_clock::now();
         for (int r = 0; r < rounds; ++r)
         {
            auto tx = ws->start_transaction(0);
            if (!same)
               value.resize(size + (r % 2 ? 64 : 0), 'w');
            tx.upsert("o/" + std::to_string(r % keys), value);
            tx.commit();
         }
         const double us = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - t0)
                               .count() /
                           rounds;
         std::printf("overwrite %8zu bytes, %s size: %9.2f us per transaction\n", size,
                     same ? "same   " : "changed", us);
         value.resize(size, 'v');
      }
      db->flush_releases();
   }
}

// gofractally/psitri#35 item 3. A typed value's reference becomes the
// engine's when the value is handed to upsert. An upsert that is refused must
// give it back, wherever on the way to the leaf the refusal happens.
TEST_CASE("Small Engine: a refused typed upsert gives the value's reference back",
          "[small][typed][limit]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   db->register_value_types<counted_value>();
   auto ws   = db->start_write_session();
   auto keys = fill_to_the_brim(*db, *ws);
   counted_value::retains  = 0;
   counted_value::releases = 0;
   uint64_t stored = 0, refused = 0;
   for (uint64_t round = 0; round < 40; ++round)
   {
      auto tx = ws->start_transaction(0);
      try
      {
         for (uint64_t j = 0; j < 2000; ++j)
         {
            tx.upsert(scattered_key("typed", round * 2000 + j), counted_value{j});
            ++stored;
         }
      }
      catch (const engine::storage_full&)
      {
         ++refused;
      }
      tx.abort();  // the stored values die with the aborted transaction
   }
   db->flush_releases();
   std::printf("typed upserts on a full store: %llu stored then aborted, %llu refused\n",
               (unsigned long long)stored, (unsigned long long)refused);
   REQUIRE(refused > 0);
   // every value handed over was released exactly once
   CHECK(counted_value::releases.load() == stored + refused);
   CHECK(counted_value::retains.load() == 0);
}

// gofractally/psitri#35 item 6. A write session keeps the lines it frees
// during a transaction in a private cache, by size, to reuse them without
// going back to the shared free map. Up to a megabyte per size can sit there,
// and it used to be handed back only by a commit or by destroying the session:
// not by an abort, and not even to the session's own request for another size.
// So a store could report full while that room existed.
namespace
{
   // Leaves lines of one size parked in the session's cache: values created
   // and removed inside one open transaction on `root`.
   template <typename Tx>
   void park_lines(Tx& tx, int count)
   {
      for (int i = 0; i < count; ++i)
         tx.upsert("park/" + std::to_string(i), std::string(200, 'p'));  // two-line leaves
      for (int i = 0; i < count; ++i)
         REQUIRE(tx.remove("park/" + std::to_string(i)) == 0);
   }
   bool writes_fit(engine::write_session& ws, uint32_t root, const char* prefix)
   {
      try
      {
         auto tx = ws.start_transaction(root);
         for (int j = 0; j < 400; ++j)
            tx.upsert(std::string(prefix) + std::to_string(j), std::string(40, 'w'));  // one line
         tx.commit();
         return true;
      }
      catch (const engine::storage_full&)
      {
         return false;
      }
   }
}  // namespace

TEST_CASE("Small Engine: lines a session parked are room again after it aborts",
          "[small][limit][parked]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   auto parker = db->start_write_session();
   auto other  = db->start_write_session();
   // fill until a little over a megabyte is left above the reserve
   for (uint64_t n = 0;; n += 500)
   {
      db->flush_releases();  // count only what is really in use
      const auto u = db->usage();
      if (u.used_bytes + u.reserve_bytes + (1300u << 10) >= u.limit_bytes)
         break;
      auto tx = parker->start_transaction(1);
      for (uint64_t j = 0; j < 500; ++j)
         tx.upsert(scattered_key("f", n + j), std::string(40, 'f'));
      tx.commit();
      REQUIRE(n < 2'000'000);
   }
   auto wants = [&](const char* prefix) {  // about three quarters of a megabyte
      try
      {
         auto tx = other->start_transaction(0);
         for (int j = 0; j < 6000; ++j)
            tx.upsert(std::string(prefix) + std::to_string(j), std::string(40, 'w'));
         tx.commit();
         return true;
      }
      catch (const engine::storage_full&)
      {
         return false;
      }
   };
   {
      auto held = parker->start_transaction(1);
      // most of a megabyte, taken and let go inside the transaction (the
      // room left is spread over all four segments, and not all of it is in
      // two-line runs)
      park_lines(held, 3200);
      REQUIRE_FALSE(wants("during/"));
      held.abort();
   }
   db->flush_releases();
   // the transaction is gone and with it every line it took
   REQUIRE(wants("after/"));
}

TEST_CASE("Small Engine: a session uses the lines it parked before reporting full",
          "[small][limit][parked]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   auto ws = db->start_write_session();
   for (uint64_t n = 0;; n += 500)
   {
      db->flush_releases();  // count only what is really in use
      const auto u = db->usage();
      if (u.used_bytes + u.reserve_bytes + (1300u << 10) >= u.limit_bytes)
         break;
      auto tx = ws->start_transaction(0);
      for (uint64_t j = 0; j < 500; ++j)
         tx.upsert(scattered_key("f", n + j), std::string(40, 'f'));
      tx.commit();
      REQUIRE(n < 2'000'000);
   }
   auto tx = ws->start_transaction(0);
   park_lines(tx, 3200);  // two-line runs, now in this session's cache
   // the same transaction wants one-line leaves: three quarters of a megabyte
   // of them fit in what it has just let go
   REQUIRE_NOTHROW([&] {
      for (int j = 0; j < 6000; ++j)
         tx.upsert("own/" + std::to_string(j), std::string(40, 'o'));
   }());
   tx.commit();
}

// gofractally/psitri#35 item 1. Growing the store extends three files and only
// then records the new segment count. A process killed in between leaves the
// files longer than the count says. The next open must treat the unrecorded
// tail as never having existed: otherwise the first growth near the limit
// asks the files for a block past their maximum and fails with an error that
// is not storage_full, and the store never notices that it is at its ceiling.
TEST_CASE("Small Engine: files left longer than the segment count by a stop during growth",
          "[small][limit][growth]")
{
   namespace fs = std::filesystem;
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 8);
   uintmax_t           data_block = 0;
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      for (int j = 0; j < 100; ++j)
         tx.upsert("seed/" + std::to_string(j), std::string(40, 's'));
      tx.commit();
      ws.reset();
      db->flush_releases();
      data_block = db->usage().mapped_bytes;  // what exists so far
   }
   // what a kill inside growth leaves behind: each file extended to the
   // maximum, the recorded count unchanged
   const uintmax_t data_size = fs::file_size(tmp.path / "data.part");
   REQUIRE(data_block > 0);
   REQUIRE(data_size == data_block);
   const uintmax_t segments = data_size / (8u << 20);
   REQUIRE(segments >= 1);
   REQUIRE(segments < 8);
   for (const char* name : {"data.part", "free.part", "refs.part"})
   {
      const auto path = tmp.path / name;
      fs::resize_file(path, fs::file_size(path) / segments * 8);
   }
   auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   REQUIRE(db->usage().mapped_bytes == data_size);  // the tail is not part of the store
   auto ws = db->start_write_session();
   std::vector<std::string> keys;
   REQUIRE_NOTHROW(keys = fill_to_the_brim(*db, *ws));  // ends in storage_full, nothing else
   const auto full = db->usage();
   REQUIRE(full.mapped_bytes == full.limit_bytes);
   REQUIRE(full.used_bytes + full.reserve_bytes + (64u << 10) >= full.limit_bytes);
   // at the ceiling the reserve works: every key can be removed again
   for (const auto& key : keys)
   {
      auto tx = ws->start_transaction(0);
      REQUIRE(tx.remove(key) == 0);
      tx.commit();
   }
}

// gofractally/psitri#35 item 5. A write refused for lack of room part-way
// through building its nodes must give back everything it had built. Routers
// of every width, shared with a held snapshot so that each write copies its
// path, and a store kept at its limit so that refusals land everywhere.
TEST_CASE("Small Engine: writes refused at the limit leak nothing", "[small][limit][leak]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 2));
   std::mt19937_64                    rng(env_u64("PSITRI_SMALL_SEED", 5));
   std::map<std::string, std::string> model;
   auto make_key = [&] {
      std::string k;
      switch (rng() % 4)
      {
         case 0:  // a wide root and a wide second level
            k.push_back(char(rng() % 256));
            k.push_back(char(rng() % 256));
            k += std::to_string(rng() % 8);
            break;
         case 1:  // medium fan-out under one prefix
            k = "m/" + std::string(1, char('a' + rng() % 20)) + std::string(1, char('a' + rng() % 20));
            break;
         case 2:  // nested prefixes: keys that end where routers begin
            k = std::string(1 + rng() % 30, 'n');
            break;
         default:
            k = "s/" + std::to_string(rng() % 40000);
      }
      return k;
   };
   auto make_value = [&] {
      const auto r = rng() % 100;
      return std::string(r < 60 ? rng() % 60 : r < 94 ? rng() % 900 : rng() % 6000,
                         char('A' + rng() % 26));
   };
   uint64_t refused = 0, committed = 0;
   {
      auto ws = db->start_write_session();
      const uint64_t rounds = env_u64("PSITRI_SMALL_LEAK_ROUNDS", 12000);
      for (uint64_t r = 0; r < rounds; ++r)
      {
         // every third round a snapshot of the committed tree is held across
         // the transaction, so its routers are shared and must be copied
         std::optional<decltype(ws->get_root(0))> held;
         if (r % 3 == 0)
            held.emplace(ws->get_root(0));
         auto tx   = ws->start_transaction(0);
         bool full = false;
         std::vector<std::pair<std::string, std::optional<std::string>>> undo;
         auto remember = [&](const std::string& key) {
            auto it = model.find(key);
            undo.emplace_back(key, it == model.end() ? std::nullopt
                                                     : std::optional<std::string>(it->second));
         };
         try
         {
            const int ops = 1 + int(rng() % 40);
            for (int i = 0; i < ops; ++i)
            {
               const auto key = make_key();
               if (rng() % 100 < 75)
               {
                  const auto value = make_value();
                  tx.upsert(key, value);
                  remember(key);
                  model[key] = value;
               }
               else
               {
                  const bool had = model.count(key) > 0;
                  REQUIRE((tx.remove(key) == 0) == had);
                  remember(key);
                  model.erase(key);
               }
            }
         }
         catch (const engine::storage_full&)
         {
            full = true;
         }
         if (full || rng() % 16 == 0)
         {
            refused += full;
            tx.abort();
            for (auto it = undo.rbegin(); it != undo.rend(); ++it)
               if (it->second)
                  model[it->first] = *it->second;
               else
                  model.erase(it->first);
         }
         else
         {
            tx.commit();
            ++committed;
         }
      }
      INFO("committed " << committed << ", refused " << refused);
      REQUIRE(refused > 100);
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(0);
         require_matches(cur, model);
      }
      // empty it, one remove per transaction
      for (const auto& [key, value] : model)
      {
         auto tx = ws->start_transaction(0);
         REQUIRE(tx.remove(key) == 0);
         tx.commit();
      }
   }
   db->flush_releases();
   const auto u = db->usage();
   std::printf("refused-write leak check: %llu committed, %llu refused, left %llu + %llu bytes\n",
               (unsigned long long)committed, (unsigned long long)refused,
               (unsigned long long)u.used_bytes, (unsigned long long)u.large_used_bytes);
   REQUIRE(u.used_bytes == 0);
   REQUIRE(u.large_used_bytes == 0);
}

// ── model test ───────────────────────────────────────────────────────────
// Written for the review of the segment-reuse and set_root changes
// (gofractally/psitri#31, #35) and kept: random transactions against a model
// of what must be stored.
namespace rfm
{
   uint64_t h64(std::string_view s, uint64_t seed = 1469598103934665603ull)
   {
      uint64_t h = seed;
      for (unsigned char c : s)
         h = (h ^ c) * 1099511628211ull;
      h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 32;
      return h;
   }
   uint64_t kvh(std::string_view k, std::string_view v) { return h64(v, h64(k) | 1); }
   const std::string man_key = "~manifest";
   struct manifest { uint64_t seq = 0, count = 0, sum = 0; };
   std::string enc(const manifest& m) { return std::string(reinterpret_cast<const char*>(&m), sizeof(m)); }

   std::string key_of(uint32_t root, uint64_t id)
   {
      std::string k = scattered_key(root == 0 ? "a" : root == 1 ? "b" : "c", id);
      k.append(id % 37, char('k' + id % 5));  // varied key lengths
      return k;
   }
   std::string value_of(std::mt19937_64& rng)
   {
      const uint32_t r = rng() % 100;
      size_t n;
      static const uint64_t small_only = env_u64("RFM_SMALL", 0);
      if (small_only) n = rng() % small_only;
      else if (r < 45) n = rng() % 64;
      else if (r < 78) n = 64 + rng() % 600;
      else if (r < 97) n = 600 + rng() % 2600;
      else n = 20000 + rng() % 150000;
      std::string v(n, char('A' + rng() % 26));
      const uint64_t x = rng();
      for (size_t i = 0; i + 8 <= n; i += 61) std::memcpy(v.data() + i, &x, 8);
      return v;
   }
   // scan a cursor: returns "" if content is self-consistent with its manifest
   template <typename Cursor>
   std::string scan(Cursor& cur, std::map<std::string, std::string>* out, manifest* mo = nullptr)
   {
      manifest m, got;
      bool     have = false;
      for (cur.seek_begin(); !cur.is_end(); cur.next())
      {
         std::string k(cur.key());
         auto        v = cur.template value<std::string>();
         if (!v) return "value missing for a key the cursor is on";
         if (k == man_key)
         {
            if (v->size() != sizeof(m)) return "manifest size";
            std::memcpy(&m, v->data(), sizeof(m));
            have = true;
            continue;
         }
         ++got.count;
         got.sum ^= kvh(k, *v);
         if (out) (*out)[k] = std::move(*v);
      }
      if (mo) *mo = m;
      if (mo) mo->count = got.count, mo->sum = got.sum;
      if (!have) return "";
      if (got.count != m.count) return "count " + std::to_string(got.count) + " != manifest " + std::to_string(m.count) + " at seq " + std::to_string(m.seq);
      if (got.sum != m.sum) return "content hash differs from manifest at seq " + std::to_string(m.seq);
      return "";
   }
   template <typename Cursor>
   std::string compare(Cursor& cur, const std::map<std::string, std::string>& expect)
   {
      auto it = expect.begin();
      uint64_t n = 0;
      for (cur.seek_begin(); !cur.is_end(); cur.next())
      {
         std::string k(cur.key());
         if (k == man_key) continue;
         if (it == expect.end()) return "extra key " + k;
         if (k != it->first) return "key mismatch: got " + k + " want " + it->first;
         auto v = cur.template value<std::string>();
         if (!v || *v != it->second) return "value mismatch at " + k;
         ++it; ++n;
      }
      if (it != expect.end()) return "missing key " + it->first;
      return "";
   }

   struct params
   {
      uint32_t max_segments = 3;
      uint64_t keyspace     = 4000;
      uint64_t txns         = 1500;   // per writer per epoch
      uint32_t writers      = 3;
      bool     threads      = true;
      bool     readers      = true;
      uint64_t seed         = 1;
   };
   struct writer_state
   {
      std::map<std::string, std::string> expect;
      manifest                           man;
      uint64_t commits = 0, aborts = 0, fulls = 0, purges = 0;
      uint64_t remove_refused = 0;
      bool     absent = false;  // the manifest key is not in the store
      std::string error;
   };

   // one writer: `txns` transactions on `root` using session ws
   template <typename DB, typename WS>
   void run_writer(DB& db, WS& ws, uint32_t root, const params& p, writer_state& st, uint64_t seed,
                   std::atomic<bool>* stop = nullptr, int report_fd = -1)
   {
      std::mt19937_64 rng(seed);
      struct snap { decltype(ws->get_root(0)) t; std::vector<std::pair<std::string, std::optional<std::string>>> samples; };
      std::vector<snap> snaps;
      auto fail = [&](std::string e) { if (st.error.empty()) st.error = "root " + std::to_string(root) + ": " + e; };
      auto verify = [&] {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(root);
         auto e   = compare(cur, st.expect);
         if (!e.empty()) fail("after commit " + std::to_string(st.man.seq) + ": " + e);
         if (auto mv = cur.template get<std::string>(man_key))
         {
            manifest m; std::memcpy(&m, mv->data(), sizeof(m));
            if (st.absent || m.seq != st.man.seq || m.count != st.man.count || m.sum != st.man.sum) fail("manifest differs from model");
         }
         else if (st.man.seq && !st.absent) fail("manifest missing");
      };
      auto purge = [&] {
         ++st.purges;
         uint64_t drop = st.expect.size() / 8 + 1;
         while (drop && !st.expect.empty())
         {
            auto     tx = ws->start_transaction(root);
            manifest m  = st.man;
            std::vector<std::string> gone;
            auto it = st.expect.lower_bound(key_of(root, rng() % p.keyspace));
            static const int per_txn = int(env_u64("RFM_PURGE", 60));
            for (int j = 0; j < per_txn && drop; ++j, --drop)
            {
               if (it == st.expect.end()) it = st.expect.begin();
               if (it == st.expect.end() || std::find(gone.begin(), gone.end(), it->first) != gone.end()) break;
               if (tx.remove(it->first) != 0) { fail("purge: committed key missing"); return; }
               --m.count; m.sum ^= kvh(it->first, it->second);
               gone.push_back(it->first);
               ++it;
            }
            // removes only (never refused): the manifest goes too, and comes
            // back with the next ordinary transaction
            tx.remove(man_key);
            m.seq = 0;
            tx.commit();
            for (auto& k : gone) st.expect.erase(k);
            st.man.count = m.count; st.man.sum = m.sum; st.absent = true;
            ++st.commits;
            if (gone.empty()) break;
         }
      };
      for (uint64_t t = 0; t < p.txns && st.error.empty() && !(stop && stop->load()); ++t)
      {
         std::map<std::string, std::optional<std::string>> pending;
         manifest m = st.man;
         auto cur_val = [&](const std::string& k) -> std::optional<std::string> {
            if (auto i = pending.find(k); i != pending.end()) return i->second;
            if (auto i = st.expect.find(k); i != st.expect.end()) return i->second;
            return std::nullopt;
         };
         bool full = false;
         try
         {
            auto tx = ws->start_transaction(root);
            const uint32_t ops = 1 + rng() % (rng() % 8 == 0 ? 150 : 30);
            for (uint32_t o = 0; o < ops; ++o)
            {
               const std::string k   = key_of(root, rng() % p.keyspace);
               const auto        old = cur_val(k);
               const uint32_t    r   = rng() % 100;
               if (r < 62)
               {
                  std::string v = value_of(rng);
                  tx.upsert(k, v);
                  if (old) { m.sum ^= kvh(k, *old); } else ++m.count;
                  m.sum ^= kvh(k, v);
                  pending[k] = std::move(v);
               }
               else if (r < 92)
               {
                  const int rc = tx.remove(k);
                  if ((rc == 0) != bool(old)) { fail("remove result wrong for " + k); break; }
                  if (old) { m.sum ^= kvh(k, *old); --m.count; pending[k] = std::nullopt; }
               }
               else
               {
                  auto got = tx.template get<std::string>(k);
                  if (got != old) { fail("in-transaction read wrong for " + k); break; }
               }
            }
            ++m.seq;
            tx.upsert(man_key, enc(m));
            static const uint64_t abort_every = env_u64("RFM_ABORT", 4);
            if (abort_every && rng() % abort_every == 0)
            {
               tx.abort();
               ++st.aborts;
            }
            else
            {
               tx.commit();
               for (auto& [k, v] : pending)
                  if (v) st.expect[k] = *v; else st.expect.erase(k);
               st.man = m; st.absent = false;
               ++st.commits;
               if (report_fd >= 0) (void)!::write(report_fd, &m.seq, 8);
            }
         }
         catch (const engine::storage_full&)
         {
            full = true;
            ++st.fulls;
         }
         catch (const std::exception& e)
         {
            fail(std::string("exception: ") + e.what());
         }
         if (!st.error.empty()) break;
         try
         {
            if (full)
            {
               try { purge(); }
               catch (const engine::storage_full& e)
               {
                  ++st.remove_refused;
                  if (env_u64("RFM_STRICT_REMOVE", 0)) fail(std::string("a remove-only transaction was refused: ") + e.what());
                  // last way out: drop the root
                  auto tx = ws->start_transaction(root);
                  tx.primary().set_root({});
                  tx.commit();
                  st.expect.clear(); st.man.count = 0; st.man.sum = 0; st.absent = true;
               }
            }
            verify();
            // owned snapshots of old versions, checked and dropped later
            if (rng() % 16 == 0 && snaps.size() < 4)
            {
               snap s{ws->get_root(root), {}};
               for (int i = 0; i < 12; ++i)
               {
                  std::string k = key_of(root, rng() % p.keyspace);
                  auto i2 = st.expect.find(k);
                  s.samples.emplace_back(k, i2 == st.expect.end() ? std::nullopt : std::optional<std::string>(i2->second));
               }
               snaps.push_back(std::move(s));
            }
            if (!snaps.empty() && rng() % 24 == 0)
            {
               const size_t i = rng() % snaps.size();
               for (auto& [k, v] : snaps[i].samples)
                  if (snaps[i].t.template get<std::string>(k) != v) fail("owned snapshot changed at " + k);
               snaps.erase(snaps.begin() + i);
            }
         }
         catch (const std::exception& e)
         {
            fail(std::string("exception (post): ") + e.what());
         }
      }
      for (auto& s : snaps)
         for (auto& [k, v] : s.samples)
            if (s.t.template get<std::string>(k) != v) fail("owned snapshot changed at " + k);
   }

   // one epoch on an open database; returns first error
   template <typename DB>
   std::string epoch(DB& db, const params& p, std::vector<writer_state>& st, uint64_t epoch_no,
                     std::atomic<bool>* stop = nullptr, int report_fd = -1)
   {
      std::atomic<bool> readers_stop{false};
      std::string       reader_error[3];
      std::vector<std::thread> readers;
      std::atomic<uint64_t> scans{0};
      if (p.readers)
         for (uint32_t r = 0; r < p.writers; ++r)
            readers.emplace_back([&, r] {
               auto rs = db->start_read_session();
               while (!readers_stop.load() && reader_error[r].empty())
               {
                  auto cur = rs->create_cursor(r);
                  reader_error[r] = scan(cur, nullptr);
                  ++scans;
               }
            });
      if (p.threads)
      {
         std::vector<std::thread> ws;
         for (uint32_t w = 0; w < p.writers; ++w)
            ws.emplace_back([&, w] {
               auto s = db->start_write_session();
               run_writer(db, s, w, p, st[w], p.seed * 1000003 + epoch_no * 131 + w, stop, report_fd);
            });
         for (auto& t : ws) t.join();
      }
      else
      {
         // sessions take turns, and change root every slice
         std::vector<decltype(db->start_write_session())> ses;
         for (uint32_t w = 0; w < p.writers; ++w) ses.push_back(db->start_write_session());
         params slice = p;
         slice.txns   = 7;
         for (uint64_t done = 0, turn = 0; done < p.txns * p.writers; done += slice.txns, ++turn)
         {
            const uint32_t s = turn % p.writers, root = (turn / p.writers + turn) % p.writers;
            run_writer(db, ses[s], root, slice, st[root], p.seed * 7919 + epoch_no * 1000 + turn, stop, report_fd);
            if (!st[root].error.empty()) break;
         }
      }
      readers_stop = true;
      for (auto& t : readers) t.join();
      for (auto& s : st) if (!s.error.empty()) return s.error;
      for (uint32_t r = 0; r < 3; ++r) if (!reader_error[r].empty()) return "reader of root " + std::to_string(r) + ": " + reader_error[r];
      return "";
   }
}  // namespace rfm

namespace rfm
{
   // Random transactions on three roots by up to three writers with readers
   // scanning, checked against a model, over `epochs` closes and reopens;
   // then every root is emptied and the arenas must drain.
   void run_model(const params& p, uint64_t epochs, psitri::sync_mode sync)
   {
   temporary_directory tmp;
   std::vector<rfm::writer_state> st(p.writers);
   for (uint64_t e = 0; e < epochs; ++e)
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_or_open, small_config(sync, p.max_segments));
      REQUIRE_FALSE(db->recovered_on_open());
      // after reopen: everything is what the model says
      for (uint32_t r = 0; r < p.writers; ++r)
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(r);
         REQUIRE(rfm::compare(cur, st[r].expect) == "");
         REQUIRE(rfm::scan(cur, nullptr) == "");
      }
      const auto& as = db->store().stats();
      const std::string err = rfm::epoch(db, p, st, e);
      std::printf("epoch %llu: commits %llu/%llu/%llu aborts %llu full %llu/%llu/%llu keys %zu/%zu/%zu switches %llu reused %llu grown %llu used %.1f MiB mapped %.0f MiB large used %.1f MiB remove-refused %llu\n",
                  (unsigned long long)e, (unsigned long long)st[0].commits, (unsigned long long)st[1].commits, (unsigned long long)st[2].commits,
                  (unsigned long long)(st[0].aborts + st[1].aborts + st[2].aborts),
                  (unsigned long long)st[0].fulls, (unsigned long long)st[1].fulls, (unsigned long long)st[2].fulls,
                  st[0].expect.size(), st[1].expect.size(), st[2].expect.size(),
                  (unsigned long long)as.range_switches.load(), (unsigned long long)as.segments_reused.load(), (unsigned long long)as.segments_grown.load(),
                  db->usage().used_bytes / 1048576.0, db->usage().mapped_bytes / 1048576.0, db->usage().large_used_bytes / 1048576.0, (unsigned long long)(st[0].remove_refused + st[1].remove_refused + st[2].remove_refused));
      std::fflush(stdout);
      REQUIRE(err == "");
   }
   // nothing leaked: empty every root, the arenas drain
   auto db = database::open(tmp.path, psitri::open_mode::open_existing, small_config(sync, p.max_segments));
   {
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.primary().set_root({});
      tx.open_root(1).set_root({});
      tx.open_root(2).set_root({});
      tx.commit();
   }
   db->flush_releases();
   std::printf("drained: used %llu bytes, large used %llu bytes\n", (unsigned long long)db->usage().used_bytes, (unsigned long long)db->usage().large_used_bytes);
   REQUIRE(db->usage().used_bytes <= 64 * 128);
   REQUIRE(db->usage().large_used_bytes == 0);
}
}  // namespace rfm

// A short run in every suite: three writers on a store small enough that
// segments are reused and the limit is reached.
TEST_CASE("Small Engine model: random transactions over reused segments", "[small][model]")
{
   rfm::params p;
   p.max_segments = 3;
   p.keyspace     = 4000;
   p.txns         = 300;
   p.threads      = 1;
   p.readers      = 1;
   p.seed         = 1;
   rfm::run_model(p, 2, psitri::sync_mode::none);
}

// The same, sized by the environment (RFM_SEGS, RFM_KEYS, RFM_TXNS,
// RFM_THREADS, RFM_READERS, RFM_SEED, RFM_EPOCHS, RFM_SYNC):
// `small-engine-tests "[.rfmodel]"`.
TEST_CASE("Small Engine model: random transactions, sized by the environment", "[.rfmodel]")
{
   rfm::params p;
   p.max_segments = uint32_t(env_u64("RFM_SEGS", 3));
   p.keyspace     = env_u64("RFM_KEYS", 4000);
   p.txns         = env_u64("RFM_TXNS", 1500);
   p.threads      = env_u64("RFM_THREADS", 1);
   p.readers      = env_u64("RFM_READERS", 1);
   p.seed         = env_u64("RFM_SEED", 1);
   rfm::run_model(p, env_u64("RFM_EPOCHS", 4), psitri::sync_mode(env_u64("RFM_SYNC", 0)));
}

TEST_CASE("Small Engine model: killed mid-transaction, recovered, continued", "[.rfkill]")
{
   rfm::params p;
   p.max_segments = uint32_t(env_u64("RFM_SEGS", 3));
   p.keyspace     = env_u64("RFM_KEYS", 4000);
   p.txns         = ~0ull;
   p.threads      = env_u64("RFM_THREADS", 1);
   p.readers      = env_u64("RFM_READERS", 1);
   p.seed         = env_u64("RFM_SEED", 1);
   const uint64_t kills = env_u64("RFM_KILLS", 12);
   const auto     sync  = psitri::sync_mode(env_u64("RFM_SYNC", 0));
   temporary_directory tmp;
   std::mt19937_64     rng(p.seed);
   uint64_t            last_seq[3] = {0, 0, 0};
   for (uint64_t k = 0; k < kills; ++k)
   {
      int fds[2];
      REQUIRE(::pipe(fds) == 0);
      const pid_t pid = ::fork();
      REQUIRE(pid >= 0);
      if (pid == 0)
      {
         ::close(fds[0]);
         try
         {
            auto db = database::open(tmp.path, psitri::open_mode::create_or_open, small_config(sync, p.max_segments));
            std::vector<rfm::writer_state> st(p.writers);
            for (uint32_t r = 0; r < p.writers; ++r)
            {
               auto rs  = db->start_read_session();
               auto cur = rs->create_cursor(r);
               auto e   = rfm::scan(cur, &st[r].expect, &st[r].man);
               if (!e.empty()) { std::fprintf(stderr, "child: after recovery root %u: %s\n", r, e.c_str()); ::_exit(41); }
               st[r].absent = st[r].man.seq == 0;
               if (st[r].absent) st[r].man.seq = (k + 1) << 32;  // keeps rising across kills
            }
            const uint64_t ready = ~0ull;
            (void)!::write(fds[1], &ready, 8);
            auto e = rfm::epoch(db, p, st, k, nullptr, -1);
            std::fprintf(stderr, "child: %s\n", e.c_str());
         }
         catch (const std::exception& e) { std::fprintf(stderr, "child exception: %s\n", e.what()); }
         ::_exit(42);
      }
      ::close(fds[1]);
      uint64_t ready = 0;
      REQUIRE(::read(fds[0], &ready, 8) == 8);
      ::usleep(useconds_t(20000 + rng() % 1500000));
      ::kill(pid, SIGKILL);
      int status = 0;
      ::waitpid(pid, &status, 0);
      ::close(fds[0]);
      INFO("kill " << k << " status " << status);
      REQUIRE(WIFSIGNALED(status));
      // reopen: recovered, self-consistent, never older than before
      auto db = database::open(tmp.path, psitri::open_mode::open_existing, small_config(sync, p.max_segments));
      REQUIRE(db->recovered_on_open());
      uint64_t seqs[3];
      for (uint32_t r = 0; r < p.writers; ++r)
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(r);
         rfm::manifest m;
         REQUIRE(rfm::scan(cur, nullptr, &m) == "");
         if (m.seq)
         {
            REQUIRE(m.seq >= last_seq[r]);
            last_seq[r] = m.seq;
         }
         seqs[r] = m.seq;
      }
      std::printf("kill %llu: seq %llu/%llu/%llu used %.1f MiB mapped %.0f MiB\n", (unsigned long long)k,
                  (unsigned long long)seqs[0], (unsigned long long)seqs[1], (unsigned long long)seqs[2],
                  db->usage().used_bytes / 1048576.0, db->usage().mapped_bytes / 1048576.0);
      std::fflush(stdout);
   }
   auto db = database::open(tmp.path, psitri::open_mode::open_existing, small_config(sync, p.max_segments));
   {
      auto ws = db->start_write_session();
      auto tx = ws->start_transaction(0);
      tx.primary().set_root({});
      tx.open_root(1).set_root({});
      tx.open_root(2).set_root({});
      tx.commit();
   }
   db->flush_releases();
   std::printf("drained: used %llu bytes, large used %llu bytes\n", (unsigned long long)db->usage().used_bytes, (unsigned long long)db->usage().large_used_bytes);
   REQUIRE(db->usage().used_bytes <= 64 * 128);
   REQUIRE(db->usage().large_used_bytes == 0);
}

// A full store whose free lines are scattered singles (gofractally/psitri#39).
// Fill to refusal, remove every third key (the freed leaves are single lines
// spread through both segments), fill to refusal again so that those scattered
// lines are all that is left, and then remove every key, one transaction
// each. A remove needs runs of two or more lines for some of the nodes it
// copies; they must come from room that is contiguous by construction.
TEST_CASE("Small Engine: a fragmented full store can be emptied one remove at a time",
          "[small][limit][brim][fragmented]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 2));
   auto ws   = db->start_write_session();
   auto keys = fill_to_the_brim(*db, *ws);
   std::vector<std::string> live;
   for (size_t i = 0; i < keys.size();)
   {
      auto tx = ws->start_transaction(0);
      for (int j = 0; j < 300 && i < keys.size(); ++j, ++i)
      {
         if (i % 3 == 0)
            REQUIRE(tx.remove(keys[i]) == 0);
         else
            live.push_back(keys[i]);
      }
      tx.commit();
   }
   db->flush_releases();
   // top up with single upserts until refused again
   for (uint64_t n = 0;; ++n)
   {
      try
      {
         auto tx = ws->start_transaction(0);
         tx.upsert(scattered_key("t", n), std::string(100, 'T'));
         tx.commit();
         live.push_back(scattered_key("t", n));
      }
      catch (const engine::storage_full&)
      {
         break;
      }
      REQUIRE(n < 2'000'000);
   }
   db->flush_releases();
   const auto u = db->usage();
   std::printf("fragmented: %zu keys; used %.2f of %.2f MiB\n", live.size(),
               u.used_bytes / 1048576.0, u.limit_bytes / 1048576.0);
   // deterministic but not in insertion order
   std::mt19937_64 rng(7);
   std::shuffle(live.begin(), live.end(), rng);
   uint64_t removed = 0;
   for (const auto& key : live)
   {
      INFO("remove " << removed << " of " << live.size());
      auto tx = ws->start_transaction(0);
      REQUIRE(tx.remove(key) == 0);
      tx.commit();
      ++removed;
   }
   {
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE_FALSE(cur.seek_begin());
   }
   ws.reset();
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

// The shape the review found it with: two segments, three sessions taking
// turns, values under 200 bytes, the store kept at its limit. Each round
// removes two keys in one transaction and then writes until refused again.
// A remove transaction is never refused. PSITRI_SMALL_FRAG_ROUNDS scales it.
TEST_CASE("Small Engine: removes are never refused on a store kept at its limit",
          "[small][limit][brim][kept-full]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_FRAG_ROUNDS", 12000);
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 2));
   std::shared_ptr<engine::write_session> sessions[3] = {
       db->start_write_session(), db->start_write_session(), db->start_write_session()};
   std::mt19937_64                    rng(20261125);
   std::map<std::string, std::string> expect;
   std::vector<std::string>           order;  // keys, for picking victims
   uint64_t                           next = 0, turn = 0, refused_writes = 0;
   auto value = [&] { return std::string(20 + rng() % 170, char('a' + rng() % 26)); };
   auto write_until_refused = [&] {
      for (;;)
      {
         auto&             ws = *sessions[turn++ % 3];
         const std::string k  = scattered_key("k", next), v = value();
         try
         {
            auto tx = ws.start_transaction(0);
            tx.upsert(k, v);
            tx.commit();
         }
         catch (const engine::storage_full&)
         {
            ++refused_writes;
            return;
         }
         ++next;
         expect[k] = v;
         order.push_back(k);
      }
   };
   write_until_refused();
   const size_t at_full = expect.size();
   uint64_t     round   = 0;
   std::string  failure;
   for (; round < rounds && failure.empty(); ++round)
   {
      auto& ws = *sessions[turn++ % 3];
      try
      {
         auto tx = ws.start_transaction(0);
         for (int j = 0; j < 2 && !order.empty(); ++j)
         {
            const size_t pick = rng() % order.size();
            if (tx.remove(order[pick]) != 0)
               throw std::runtime_error("a committed key was missing");
            expect.erase(order[pick]);
            order[pick] = order.back();
            order.pop_back();
         }
         tx.commit();
      }
      catch (const std::exception& e)
      {
         failure = e.what();
         break;
      }
      write_until_refused();
   }
   std::printf("kept full: %zu keys at the limit; %llu two-remove transactions, %llu refused "
               "writes; %zu keys at the end\n",
               at_full, (unsigned long long)round, (unsigned long long)refused_writes,
               expect.size());
   INFO("after " << round << " remove transactions: " << failure);
   REQUIRE(failure.empty());
   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);
   require_matches(cur, expect);
}

// The shape the defect was found with (gofractally/psitri#39): two segments,
// three writers at once on three roots, values under 200 bytes. Each writer
// writes until it is refused, then removes two keys in one transaction, over
// and over. There are more writers than segments, and the store is at its
// limit throughout; a remove transaction is never refused, and no writer is
// shut out. PSITRI_SMALL_FRAG_ROUNDS scales it.
TEST_CASE("Small Engine: three writers on two segments at the limit can always remove",
          "[small][limit][brim][three-writers]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_FRAG_ROUNDS", 12000) / 6;
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 2));
   struct outcome
   {
      std::map<std::string, std::string> expect;
      uint64_t                           removes = 0, refused_writes = 0, written = 0;
      std::string                        error;
   };
   outcome results[3];
   auto writer = [&](uint32_t root) {
      outcome& out = results[root];
      try
      {
         auto                     ws = db->start_write_session();
         std::mt19937_64          rng(1000 + root);
         std::vector<std::string> order;
         uint64_t                 next = 0;
         const char*              prefix = root == 0 ? "a" : root == 1 ? "b" : "c";
         for (uint64_t round = 0; round < rounds; ++round)
         {
            // write until refused (in small transactions, so that the other
            // writers get their turns)
            for (bool full = false; !full;)
            {
               std::vector<std::pair<std::string, std::string>> batch;
               try
               {
                  auto tx = ws->start_transaction(root);
                  for (int j = 0; j < 20; ++j)
                  {
                     batch.emplace_back(scattered_key(prefix, next + j),
                                        std::string(20 + rng() % 170, char('a' + rng() % 26)));
                     tx.upsert(batch.back().first, batch.back().second);
                  }
                  tx.commit();
                  next += 20;
                  out.written += 20;
                  for (auto& kv : batch)
                  {
                     out.expect[kv.first] = kv.second;
                     order.push_back(kv.first);
                  }
               }
               catch (const engine::storage_full&)
               {
                  ++out.refused_writes;
                  full = true;
               }
            }
            // two removes in one transaction: never refused
            auto tx = ws->start_transaction(root);
            for (int j = 0; j < 2 && !order.empty(); ++j)
            {
               const size_t pick = rng() % order.size();
               if (tx.remove(order[pick]) != 0)
                  throw std::runtime_error("a committed key was missing");
               out.expect.erase(order[pick]);
               order[pick] = order.back();
               order.pop_back();
            }
            tx.remove("never-there");  // a remove that finds nothing allocates too
            tx.commit();
            ++out.removes;
         }
      }
      catch (const std::exception& e)
      {
         out.error = e.what();
      }
   };
   std::thread t0(writer, 0), t1(writer, 1), t2(writer, 2);
   t0.join();
   t1.join();
   t2.join();
   for (uint32_t root = 0; root < 3; ++root)
   {
      std::printf("three writers: root %u wrote %llu keys, refused %llu times, %llu remove "
                  "transactions, %zu keys at the end\n",
                  root, (unsigned long long)results[root].written,
                  (unsigned long long)results[root].refused_writes,
                  (unsigned long long)results[root].removes, results[root].expect.size());
      INFO("root " << root << " after " << results[root].removes
                   << " remove transactions: " << results[root].error);
      REQUIRE(results[root].error.empty());
      REQUIRE(results[root].removes == rounds);
      REQUIRE(results[root].written > 0);  // nobody was shut out
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(root);
      require_matches(cur, results[root].expect);
   }
   // and from there everything can be removed, one key per transaction
   {
      auto ws = db->start_write_session();
      for (uint32_t root = 0; root < 3; ++root)
         for (auto& kv : results[root].expect)
         {
            auto tx = ws->start_transaction(root);
            REQUIRE(tx.remove(kv.first) == 0);
            tx.commit();
         }
   }
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

// ── the remove reserve over time (gofractally/psitri#39) ───────────────────
namespace
{
   struct reserve_run
   {
      uint64_t    rounds = 0, refused_removes = 0, mixed_refused = 0;
      uint64_t    start_room = 0, least_room = UINT64_MAX, keys = 0;
      std::string error;
   };
   std::string reserve_key(bool binary, const char* prefix, uint64_t n)
   {
      if (!binary)
         return scattered_key(prefix, n);
      // eight scrambled bytes: every branch width, no shared prefixes to speak of
      uint64_t h = (n + 1) * 0x9E3779B97F4A7C15ull;
      h ^= h >> 29;
      h *= 0xBF58476D1CE4E5B9ull;
      h ^= uint64_t(uint8_t(prefix[0])) << 56;
      return std::string(reinterpret_cast<const char*>(&h), 8);
   }
   // The store at `dir` (two segments) is kept at its limit. Each round
   // removes one randomly chosen key in a transaction of its own and then
   // writes new keys, one per transaction, until refused. After the releases
   // of each round have settled the reserve must hold what it held at the
   // start, and no remove may ever be refused. `mixed` adds, every tenth
   // round, a transaction that removes two keys and then tries an upsert.
   void reserve_rounds(database& db, reserve_run& out, uint64_t rounds, uint64_t seed, bool binary,
                       bool mixed)
   {
      auto                     ws = db.start_write_session();
      std::mt19937_64          rng(seed);
      std::vector<std::string> keys;
      {
         auto rs  = db.start_read_session();
         auto cur = rs->create_cursor(0);
         for (cur.seek_begin(); !cur.is_end(); cur.next())
            keys.emplace_back(cur.key());
      }
      uint64_t next = seed << 32;  // fresh keys every call
      auto     fill = [&](const char* prefix) {
         for (;;)
         {
            const std::string k = reserve_key(binary, prefix, next);
            try
            {
               auto tx = ws->start_transaction(0);
               tx.upsert(k, std::string(20 + rng() % 171, char('a' + rng() % 26)));
               tx.commit();
            }
            catch (const engine::storage_full&)
            {
               return;
            }
            ++next;
            keys.push_back(k);
         }
      };
      auto room = [&] {
         db.flush_releases();
         // what the reserve holds plus what it is owed is always its size
         if (db.store().reserve_free_lines() + db.store().reserve_owed_lines() !=
             db.store().config().reserve_lines)
            out.error = "the reserve's lines do not add up";
         return uint64_t(db.store().reserve_free_lines());
      };
      fill("c");
      if (!out.start_room)
         out.start_room = room();
      for (uint64_t r = 0; r < rounds && out.error.empty(); ++r, ++out.rounds)
      {
         auto take = [&] {
            const size_t      pick = rng() % keys.size();
            const std::string k    = keys[pick];
            keys[pick]             = keys.back();
            keys.pop_back();
            return k;
         };
         try
         {
            auto tx = ws->start_transaction(0);
            if (tx.remove(take()) != 0)
               out.error = "a committed key was missing";
            if (mixed && r % 10 == 9)
            {
               if (tx.remove(take()) != 0)
                  out.error = "a committed key was missing";
               const std::string k = reserve_key(binary, "m", next++);
               try
               {
                  tx.upsert(k, std::string(150, 'm'));  // ordinary: may be refused, may not
                  keys.push_back(k);
               }
               catch (const engine::storage_full&)
               {
                  ++out.mixed_refused;
                  tx.abort();  // an ordinary write was refused: the transaction is the caller's to end
                  continue;    // (its two removes are undone; the keys are simply forgotten here)
               }
            }
            tx.commit();
         }
         catch (const engine::storage_full& e)
         {
            ++out.refused_removes;
            out.error = std::string("a remove was refused in round ") + std::to_string(out.rounds) +
                        ": " + e.what();
            break;
         }
         fill("r");
         const uint64_t now = room();
         out.least_room     = std::min(out.least_room, now);
         if (now < out.start_room)
            out.error = "reserve room fell to " + std::to_string(now) + " lines from " +
                        std::to_string(out.start_room) + " in round " + std::to_string(out.rounds);
      }
      out.keys = keys.size();
   }
   void report(const char* what, const reserve_run& run)
   {
      std::printf("%s: %llu rounds, %llu keys; reserve room at the start %llu lines, least seen "
                  "after a round %llu; removes refused %llu\n",
                  what, (unsigned long long)run.rounds, (unsigned long long)run.keys,
                  (unsigned long long)run.start_room, (unsigned long long)run.least_room,
                  (unsigned long long)run.refused_removes);
      std::fflush(stdout);
   }
}  // namespace

// The review's reproduction: digit keys, values of 20 to 190 bytes, one
// session, a clean close and reopen every 300 rounds. The reserve used to
// lose a little at every open until nothing was left (by round 12,327) and
// every remove was refused. PSITRI_SMALL_RESERVE_ROUNDS scales it.
TEST_CASE("Small Engine: the remove reserve is whole after every round, across clean reopens",
          "[small][limit][reserve][reserve-reopen]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_RESERVE_ROUNDS", 30000);
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 2);
   reserve_run         run;
   for (uint64_t epoch = 0; run.rounds < rounds && run.error.empty(); ++epoch)
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_or_open, cfg);
      REQUIRE_FALSE(db->recovered_on_open());
      reserve_rounds(*db, run, std::min<uint64_t>(300, rounds - run.rounds), epoch + 1, false, false);
   }
   report("reserve, clean reopens", run);
   INFO(run.error);
   REQUIRE(run.error.empty());
   REQUIRE(run.refused_removes == 0);
   REQUIRE(run.start_room == 2048);  // a sixty-fourth of two segments
   REQUIRE(run.least_room >= run.start_room);
}

// The same with every epoch ending in SIGKILL in the middle of a
// transaction, so that every open is a recovery.
TEST_CASE("Small Engine: the remove reserve is whole after every round, across kills",
          "[small][limit][reserve][reserve-kill]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_RESERVE_ROUNDS", 30000);
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 2);
   reserve_run         total;
   for (uint64_t epoch = 0; total.rounds < rounds && total.error.empty(); ++epoch)
   {
      int fds[2];
      REQUIRE(::pipe(fds) == 0);
      std::fflush(nullptr);
      const pid_t pid = ::fork();
      REQUIRE(pid >= 0);
      if (pid == 0)
      {
         ::close(fds[0]);
         reserve_run run;
         run.start_room = total.start_room;
         try
         {
            auto db = database::open(tmp.path, psitri::open_mode::create_or_open, cfg);
            if (epoch && !db->recovered_on_open())
               run.error = "the open after a kill did not recover";
            reserve_rounds(*db, run, 250, epoch + 1, false, false);
            // die with a transaction half done
            auto ws = db->start_write_session();
            auto tx = ws->start_transaction(0);
            tx.upsert("half-done", std::string(100, 'h'));
            uint64_t numbers[5] = {run.rounds, run.refused_removes, run.start_room,
                                   run.least_room, run.keys};
            (void)!::write(fds[1], numbers, sizeof(numbers));
            (void)!::write(fds[1], run.error.data(), run.error.size());
            ::kill(::getpid(), SIGKILL);
         }
         catch (const std::exception& e)
         {
            std::fprintf(stderr, "reserve kill child: %s\n", e.what());
         }
         ::_exit(3);
      }
      ::close(fds[1]);
      uint64_t numbers[5] = {};
      REQUIRE(::read(fds[0], numbers, sizeof(numbers)) == ssize_t(sizeof(numbers)));
      char    text[512];
      ssize_t n;
      while ((n = ::read(fds[0], text, sizeof(text))) > 0)
         total.error.append(text, size_t(n));
      ::close(fds[0]);
      int status = 0;
      ::waitpid(pid, &status, 0);
      REQUIRE(WIFSIGNALED(status));
      total.rounds += numbers[0];
      total.refused_removes += numbers[1];
      total.start_room = numbers[2];
      total.least_room = std::min(total.least_room, numbers[3]);
      total.keys       = numbers[4];
   }
   report("reserve, kills", total);
   INFO(total.error);
   REQUIRE(total.error.empty());
   REQUIRE(total.refused_removes == 0);
   REQUIRE(total.least_room >= total.start_room);
   // and what is left is a whole store
   auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   REQUIRE(db->recovered_on_open());
   REQUIRE(db->store().reserve_free_lines() >= total.start_room);
}

// Eight-byte binary keys: a store like this reports full with thousands of
// scattered lines free, which is exactly when a count of free lines says
// nothing about whether a remove can get a run.
TEST_CASE("Small Engine: the remove reserve is whole after every round with binary keys",
          "[small][limit][reserve][reserve-binary]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_RESERVE_ROUNDS", 30000) / 3;
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::none, 2);
   reserve_run         run;
   for (uint64_t epoch = 0; run.rounds < rounds && run.error.empty(); ++epoch)
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_or_open, cfg);
      reserve_rounds(*db, run, std::min<uint64_t>(500, rounds - run.rounds), epoch + 1, true, false);
      if (epoch == 0)
      {
         db->flush_releases();
         const auto u = db->usage();
         std::printf("reserve, binary keys: full at %.2f of %.2f MiB used, reserve %.2f MiB\n",
                     u.used_bytes / 1048576.0, u.limit_bytes / 1048576.0,
                     u.reserve_bytes / 1048576.0);
      }
   }
   report("reserve, binary keys", run);
   INFO(run.error);
   REQUIRE(run.error.empty());
   REQUIRE(run.refused_removes == 0);
   REQUIRE(run.least_room >= run.start_room);
   // and from there the store can be emptied, one key per transaction
   auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   {
      auto                     ws = db->start_write_session();
      std::vector<std::string> keys;
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(0);
         for (cur.seek_begin(); !cur.is_end(); cur.next())
            keys.emplace_back(cur.key());
      }
      for (const auto& k : keys)
      {
         auto tx = ws->start_transaction(0);
         REQUIRE(tx.remove(k) == 0);
         tx.commit();
      }
   }
   db->flush_releases();
   REQUIRE(db->usage().used_bytes <= 64 * 128);
}

// Transactions that remove and then write. Lines a remove drew from the
// reserve and then let go again inside the transaction must not be picked up
// by its ordinary writes: after every such transaction, committed or
// aborted, the reserve is whole.
TEST_CASE("Small Engine: a transaction's ordinary writes get nothing from the reserve",
          "[small][limit][reserve][reserve-mixed]")
{
   const uint64_t rounds = env_u64("PSITRI_SMALL_RESERVE_ROUNDS", 30000) / 6;
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 2));
   reserve_run run;
   reserve_rounds(*db, run, rounds, 1, false, true);
   report("reserve, removes and writes mixed", run);
   std::printf("reserve, mixed: %llu of the mixed transactions had their write refused\n",
               (unsigned long long)run.mixed_refused);
   INFO(run.error);
   REQUIRE(run.error.empty());
   REQUIRE(run.refused_removes == 0);
   REQUIRE(run.least_room >= run.start_room);
}

// A remove at the ceiling claims no segment, sweeps nothing and waits for
// nothing, so a writer sitting in a transaction on another root cannot hold
// it up; and a write that has to be refused is refused at once.
TEST_CASE("Small Engine: a remove at the ceiling is not held up by another writer",
          "[small][limit][reserve][no-wait]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 2));
   auto a = db->start_write_session();
   auto b = db->start_write_session();
   {
      auto tx = a->start_transaction(1);
      for (int j = 0; j < 500; ++j)
         tx.upsert(scattered_key("a", j), std::string(100, 'a'));
      tx.commit();
   }
   std::vector<std::string> keys;
   for (uint64_t n = 0;; ++n)
   {
      try
      {
         auto tx = b->start_transaction(0);
         tx.upsert(scattered_key("b", n), std::string(100, 'b'));
         tx.commit();
         keys.push_back(scattered_key("b", n));
      }
      catch (const engine::storage_full&)
      {
         break;
      }
   }
   // writer A sits in a transaction on root 1, in its segment, for the duration
   auto held = a->start_transaction(1);
   REQUIRE(held.remove(scattered_key("a", 7)) == 0);

   using clock = std::chrono::steady_clock;
   auto ms = [](clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
   std::vector<double> removes, refusals;
   for (int k = 0; k < 400; ++k)
   {
      const auto t0 = clock::now();
      auto       tx = b->start_transaction(0);
      REQUIRE(tx.remove(keys[size_t(k) * 37 % keys.size()]) >= -1);
      tx.commit();
      removes.push_back(ms(clock::now() - t0));
      if (k % 40 == 0)
      {
         const auto t1 = clock::now();
         bool       refused = false;
         try
         {
            auto tx2 = b->start_transaction(0);
            for (int j = 0; j < 400; ++j)
               tx2.upsert("more/" + std::to_string(j), std::string(1500, 'x'));
         }
         catch (const engine::storage_full&)
         {
            refused = true;
         }
         if (refused)
            refusals.push_back(ms(clock::now() - t1));
      }
   }
   held.abort();
   std::sort(removes.begin(), removes.end());
   std::sort(refusals.begin(), refusals.end());
   double sum = 0;
   for (size_t i = 0; i < removes.size() * 9 / 10; ++i)
      sum += removes[i];
   const double typical = sum / double(removes.size() * 9 / 10);
   std::printf("no wait: 400 single-key removes beside an open transaction: fastest nine tenths "
               "average %.3f ms, median %.3f ms, slowest %.3f ms; %zu refused writes, quickest "
               "%.3f ms\n",
               typical, removes[removes.size() / 2], removes.back(), refusals.size(),
               refusals.empty() ? 0.0 : refusals.front());
   // with the 20 ms wait these were 5 ms on average and 21 ms at the least
   REQUIRE(typical < 1.0);
   REQUIRE_FALSE(refusals.empty());
   REQUIRE(refusals.front() < 10.0);
}

// A stop while the files were being grown, or trimmed after such a stop, can
// leave the free map and the count file longer than the recorded segment
// count even when the data file is not. All three are mapped whole, so all
// three count towards the size the limit must cover.
TEST_CASE("Small Engine: map files longer than the segment count are counted against the limit",
          "[small][open][limit][shrink][trim]")
{
   namespace fs = std::filesystem;
   temporary_directory tmp;
   const model         m{2000, 20};
   uint64_t            segments = 0;
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_only,
                               small_config(psitri::sync_mode::none, 8));
      auto ws = db->start_write_session();
      for (uint64_t seq = 1; seq <= 150; ++seq)
      {
         auto tx = ws->start_transaction(0);
         m.apply(tx, seq);
         tx.commit();
      }
      segments = db->usage().mapped_bytes / (8ull * 1024 * 1024);
      REQUIRE(segments == 4);
   }
   // two more blocks on the free map and on the count file, as a stop would leave them
   uintmax_t sizes[2];
   int       k = 0;
   for (const char* name : {"free.part", "refs.part"})
   {
      sizes[k]              = fs::file_size(tmp.path / name);
      const uintmax_t block = sizes[k] / segments;
      fs::resize_file(tmp.path / name, sizes[k] + 2 * block);
      ++k;
   }
   // at exactly the recorded count the longer files would be mapped past the reservation
   REQUIRE_THROWS_AS(database::open(tmp.path, psitri::open_mode::open_existing,
                                    small_config(psitri::sync_mode::none, 4)),
                     std::runtime_error);
   REQUIRE(database::closed_cleanly(tmp.path));
   REQUIRE(fs::file_size(tmp.path / "free.part") > sizes[0]);  // refused untouched
   {
      // a limit that covers them opens the store and cuts them back
      auto db = database::open(tmp.path, psitri::open_mode::open_existing,
                               small_config(psitri::sync_mode::none, 8));
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE(m.verify(cur) == 150);
   }
   REQUIRE(fs::file_size(tmp.path / "free.part") == sizes[0]);
   REQUIRE(fs::file_size(tmp.path / "refs.part") == sizes[1]);
   auto db = database::open(tmp.path, psitri::open_mode::open_existing,
                            small_config(psitri::sync_mode::none, 4));
   auto rs  = db->start_read_session();
   auto cur = rs->create_cursor(0);
   REQUIRE(m.verify(cur) == 150);
}

// ── long runs (gofractally/psitri#39, third review) ────────────────────────
// A store at its ceiling holds, among small keys, 60 pairs of keys behind a
// 900-byte common prefix. Removing one of a pair copies nodes of many lines
// (the prefix alone is eight). The reserve must be able to hand out such runs
// however many short draws came before: after one transaction that removes
// `batch` small keys (committed, or refused and aborted), or after the same
// removes one per transaction while a reader holds the version before them.
namespace
{
   std::string long_prefix_key(int i, char last)
   {
      uint64_t h = (uint64_t(i) + 1) * 0x9E3779B97F4A7C15ull;
      h ^= h >> 29;
      h *= 0xBF58476D1CE4E5B9ull;
      h ^= h >> 32;
      std::string k = "Q";
      k.append(reinterpret_cast<const char*>(&h), 4);
      k.append(900, 'p');
      k.push_back(last);
      return k;
   }
   struct long_run_result
   {
      uint64_t control_ok = 0, long_ok = 0, long_refused = 0, small_ok = 0, small_refused = 0;
      uint64_t batch_done = 0;
      bool     batch_refused = false;
      uint32_t longest_before = 0, longest_after = 0, long_runs_after = 0;
   };
   long_run_result long_runs_after_short_draws(bool under_reader, uint64_t batch, bool reopen,
                                               psitri::sync_mode sync = psitri::sync_mode::none)
   {
      long_run_result     out;
      temporary_directory tmp;
      const auto          cfg = small_config(sync, 2);
      auto                db  = database::open(tmp.path, psitri::open_mode::create_only, cfg);
      auto                ws  = db->start_write_session();
      {
         auto tx = ws->start_transaction(0);
         for (int i = 0; i < 60; ++i)
         {
            tx.upsert(long_prefix_key(i, 'a'), std::string(20, 'W'));
            tx.upsert(long_prefix_key(i, 'b'), std::string(20, 'S'));
         }
         tx.commit();
      }
      std::vector<std::string> keys;
      uint64_t                 next = 0;
      std::mt19937_64          rng(99);
      auto refill = [&] {
         for (int tiny = 0; tiny < 2; ++tiny)
            for (;;)
            {
               try
               {
                  auto tx = ws->start_transaction(0);
                  tx.upsert(scattered_key("k", next),
                            tiny ? std::string(1, 't') : std::string(20 + rng() % 170, 'v'));
                  tx.commit();
               }
               catch (const engine::storage_full&)
               {
                  break;
               }
               keys.push_back(scattered_key("k", next));
               ++next;
            }
      };
      auto remove_long = [&](int from, int to) {
         uint64_t ok = 0;
         for (int i = from; i < to; ++i)
         {
            try
            {
               auto tx = ws->start_transaction(0);
               if (tx.remove(long_prefix_key(i, 'a')) != 0)
                  throw std::runtime_error("a long-prefix key was missing");
               tx.commit();
               ++ok;
            }
            catch (const engine::storage_full&)
            {
               ++out.long_refused;
            }
         }
         return ok;
      };
      refill();
      out.control_ok = remove_long(0, 5);
      refill();
      db->flush_releases();
      out.longest_before = db->store().reserve_longest_run();
      std::shuffle(keys.begin(), keys.end(), rng);
      if (under_reader)
      {
         auto rs  = db->start_read_session();
         auto cur = rs->create_cursor(0);
         for (; out.batch_done < batch; ++out.batch_done)
         {
            try
            {
               auto tx = ws->start_transaction(0);
               tx.remove(keys.back());
               tx.commit();
               keys.pop_back();
            }
            catch (const engine::storage_full&)
            {
               out.batch_refused = true;
               break;
            }
         }
      }
      else
      {
         auto tx = ws->start_transaction(0);
         try
         {
            for (; out.batch_done < batch; ++out.batch_done)
               tx.remove(keys[keys.size() - 1 - out.batch_done]);
            tx.commit();
            keys.resize(keys.size() - out.batch_done);
         }
         catch (const engine::storage_full&)
         {
            out.batch_refused = true;
            tx.abort();
         }
      }
      db->flush_releases();
      refill();
      if (reopen)
      {
         ws.reset();
         db.reset();
         db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
         ws = db->start_write_session();
      }
      db->flush_releases();
      REQUIRE(db->store().reserve_free_lines() + db->store().reserve_owed_lines() ==
              db->store().config().reserve_lines);
      out.longest_after   = db->store().reserve_longest_run();
      out.long_runs_after = uint32_t(db->usage().reserve_long_runs);
      out.long_refused    = 0;
      out.long_ok         = remove_long(5, 45);
      for (int j = 0; j < 200 && !keys.empty(); ++j)
      {
         try
         {
            auto tx = ws->start_transaction(0);
            tx.remove(keys.back());
            tx.commit();
            ++out.small_ok;
         }
         catch (const engine::storage_full&)
         {
            ++out.small_refused;
         }
         keys.pop_back();
      }
      return out;
   }
   void report(const char* what, const long_run_result& r)
   {
      std::printf("%s: the batch did %llu removes%s; reserve longest run %u lines before, %u after "
                  "(%u runs of nine or more); long-prefix removes afterwards %llu of 40, refused "
                  "%llu; small removes %llu, refused %llu\n",
                  what, (unsigned long long)r.batch_done, r.batch_refused ? " and was refused" : "",
                  r.longest_before, r.longest_after, r.long_runs_after,
                  (unsigned long long)r.long_ok, (unsigned long long)r.long_refused,
                  (unsigned long long)r.small_ok, (unsigned long long)r.small_refused);
      std::fflush(stdout);
   }
   void require_long_runs_kept(const long_run_result& r)
   {
      REQUIRE(r.control_ok == 5);
      REQUIRE(r.long_refused == 0);
      REQUIRE(r.long_ok == 40);
      REQUIRE(r.small_refused == 0);
      REQUIRE(r.longest_after >= 16);
   }
}  // namespace

TEST_CASE("Small Engine: long removes still work after one large batch of short ones",
          "[small][limit][reserve][reserve-long]")
{
   for (bool reopen : {false, true})
      for (uint64_t batch : {uint64_t(1040), uint64_t(1500)})
      {
         const auto r = long_runs_after_short_draws(false, batch, reopen);
         report(reopen ? "long runs, one batch, reopened" : "long runs, one batch", r);
         require_long_runs_kept(r);
      }
}

TEST_CASE("Small Engine: long removes still work after many short ones under a reader",
          "[small][limit][reserve][reserve-long]")
{
   for (bool reopen : {false, true})
   {
      const auto r = long_runs_after_short_draws(true, 1500, reopen);
      report(reopen ? "long runs, under a reader, reopened" : "long runs, under a reader", r);
      require_long_runs_kept(r);
   }
}

// A transaction too large for the reserve is refused and rolled back. What
// it drew before that must not have cost the reserve its long runs either.
TEST_CASE("Small Engine: a refused and aborted batch does not cost the reserve its long runs",
          "[small][limit][reserve][reserve-long]")
{
   const auto r = long_runs_after_short_draws(false, 20000, false);
   report("long runs, batch refused and aborted", r);
   REQUIRE(r.batch_refused);
   require_long_runs_kept(r);
}

// The same with a sync mode that flushes at every commit.
TEST_CASE("Small Engine: the reserve behaves the same with commits that sync",
          "[small][limit][reserve][reserve-sync]")
{
   const auto r = long_runs_after_short_draws(false, 1500, true, psitri::sync_mode::msync_sync);
   report("long runs, one batch, msync_sync, reopened", r);
   require_long_runs_kept(r);
   temporary_directory tmp;
   const auto          cfg = small_config(psitri::sync_mode::fsync, 2);
   reserve_run         run;
   for (uint64_t epoch = 0; run.rounds < 600 && run.error.empty(); ++epoch)
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_or_open, cfg);
      reserve_rounds(*db, run, 200, epoch + 1, false, false);
   }
   report("reserve, fsync, clean reopens", run);
   INFO(run.error);
   REQUIRE(run.error.empty());
   REQUIRE(run.refused_removes == 0);
   REQUIRE(run.least_room >= run.start_room);
}

// Removes whose paths are made of long nodes, in the shapes that could be
// expected to leave the reserve short of long runs: two long key prefixes one
// after the other with a key branching off between them (shape 0), a long
// prefix above a leaf with a long key suffix (shape 1), and a long prefix
// above leaves with values of several lines (shape 2). Each remove draws runs
// of three to seven lines. After every one of them, once its old version is
// released, nothing is owed: every run drawn has been repaid with a run as
// long, and the guaranteed stock of every length is complete. 450 groups of
// three keys, every key removed in a transaction of its own.
TEST_CASE("Small Engine: removes through long nodes are repaid run for run",
          "[small][limit][reserve][reserve-long-nodes]")
{
   for (int shape = 0; shape < 3; ++shape)
   {
      temporary_directory tmp;
      auto db = database::open(tmp.path, psitri::open_mode::create_only,
                               small_config(psitri::sync_mode::none, 2));
      auto ws = db->start_write_session();
      const int groups = 150;
      auto key = [&](int g, int which) {
         char head[16];
         std::snprintf(head, sizeof(head), "G%05d", g);
         const std::string p = head + std::string(444, 'p');
         if (which == 3)
            return p + "b";
         if (shape == 0)
            return p + "a" + std::string(450, 'q') + char('0' + which);
         if (shape == 1)
            return p + (which == 1 ? "a" : "c") + std::string(450, 'q');
         return p + "a" + char('0' + which);
      };
      {
         auto tx = ws->start_transaction(0);
         for (int g = 0; g < groups; ++g)
            for (int which : {1, 2, 3})
               tx.upsert(key(g, which), std::string(shape == 2 ? 700 : 20, 'v'));
         tx.commit();
      }
      for (uint64_t n = 0;; ++n)
      {
         try
         {
            auto tx = ws->start_transaction(0);
            tx.upsert(scattered_key("k", n), std::string(60, 's'));
            tx.commit();
         }
         catch (const engine::storage_full&)
         {
            break;
         }
      }
      db->flush_releases();
      REQUIRE(db->usage().reserve_owed_bytes == 0);
      uint64_t most_drawn = 0, longest_drawn = 0;
      for (int g = 0; g < groups; ++g)
         for (int which : {3, 1, 2})
         {
            uint32_t before[65], during[65];
            db->store().reserve_histogram(before);
            auto tx = ws->start_transaction(0);
            REQUIRE(tx.remove(key(g, which)) == 0);
            db->store().reserve_histogram(during);
            most_drawn = std::max<uint64_t>(most_drawn, db->store().reserve_owed_lines());
            for (uint32_t k = 2; k <= 64; ++k)
               if (during[k] < before[k])
                  longest_drawn = std::max<uint64_t>(longest_drawn, k);
            tx.commit();
            db->flush_releases();
            INFO("shape " << shape << " group " << g << " key " << which);
            REQUIRE(db->usage().reserve_owed_bytes == 0);
         }
      uint32_t hist[65];
      db->store().reserve_histogram(hist);
      for (uint32_t k = 2; k <= 16; ++k)
         REQUIRE(hist[k] >= db->store().reserve_quota(k));
      std::printf("long nodes, shape %d: %d removes, at most %llu lines drawn by one, runs of up "
                  "to %llu lines taken from stock; nothing owed after any of them; longest run "
                  "now %u\n",
                  shape, groups * 3, (unsigned long long)most_drawn,
                  (unsigned long long)longest_drawn, db->store().reserve_longest_run());
   }
}

// A store opened with more reserve than it has free lines comes up short.
// It says so through usage(), ordinary writes are refused, removes work, and
// the lines they free go to the reserve until it is whole; only then is there
// room for writes again.
TEST_CASE("Small Engine: a store opened too full to collect its reserve fills it from frees",
          "[small][limit][reserve][reserve-short-open]")
{
   temporary_directory tmp;
   auto                cfg = small_config(psitri::sync_mode::none, 2);
   std::vector<std::string> keys;
   {
      auto db = database::open(tmp.path, psitri::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      keys    = fill_to_the_brim(*db, *ws);
      REQUIRE(db->usage().reserve_bytes == 2048 * 128);
   }
   cfg.reserve_bytes = 1u << 20;  // 8,192 lines, of which about 2,048 are free
   auto db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   auto ws = db->start_write_session();
   auto u  = db->usage();
   std::printf("short open: reserve %llu lines wanted, %llu found, %llu owed; longest run %llu "
               "lines, %u runs of nine or more\n",
               (unsigned long long)u.reserve_bytes / 128, (unsigned long long)u.reserve_free_bytes / 128,
               (unsigned long long)u.reserve_owed_bytes / 128,
               (unsigned long long)u.reserve_longest_run_bytes / 128, unsigned(u.reserve_long_runs));
   REQUIRE(u.reserve_bytes == 8192 * 128);
   REQUIRE(u.reserve_free_bytes < u.reserve_bytes / 2);
   REQUIRE(u.reserve_free_bytes + u.reserve_owed_bytes >= u.reserve_bytes);
   REQUIRE(u.reserve_longest_run_bytes >= 16 * 128);  // the long runs were collected first
   {
      auto tx = ws->start_transaction(0);
      REQUIRE_THROWS_AS([&] {
         for (int j = 0; j < 200; ++j)
            tx.upsert("more/" + std::to_string(j), std::string(100, 'x'));
      }(), engine::storage_full);
   }
   // removes work, and what they free is the reserve's first: the flexible
   // stock is filled from the single lines, the guaranteed runs from frees
   // that are long enough
   size_t         removed = 0;
   uint64_t       owed    = u.reserve_owed_bytes;
   const uint64_t free0   = u.reserve_free_bytes;
   while (removed < keys.size() * 6 / 10)
   {
      auto tx = ws->start_transaction(0);
      for (int j = 0; j < 50 && removed < keys.size(); ++j, ++removed)
         REQUIRE(tx.remove(keys[removed]) == 0);
      tx.commit();
      db->flush_releases();
      const uint64_t now = db->usage().reserve_owed_bytes;
      REQUIRE(now <= owed);
      owed = now;
   }
   u = db->usage();
   std::printf("short open: after %zu removes the reserve holds %llu lines, %llu still owed (runs "
               "of lengths nothing freed was long enough for)\n",
               removed, (unsigned long long)u.reserve_free_bytes / 128,
               (unsigned long long)u.reserve_owed_bytes / 128);
   REQUIRE(u.reserve_free_bytes >= free0 + 4000 * 128);
   // the flexible stock is full: freed lines are ordinary room again
   {
      auto tx = ws->start_transaction(0);
      for (int j = 0; j < 100; ++j)
         tx.upsert("more/" + std::to_string(j), std::string(100, 'x'));
      tx.commit();
   }
   // and with room in the store, the next open collects all of it
   ws.reset();
   db.reset();
   db = database::open(tmp.path, psitri::open_mode::open_existing, cfg);
   u  = db->usage();
   REQUIRE(u.reserve_owed_bytes == 0);
   REQUIRE(u.reserve_free_bytes == u.reserve_bytes);
}

// What a remove copies, and so the longest run it can need. It copies the
// routers and key prefixes on the path to its key; it never copies a leaf,
// its own or a neighbour's, so the size of the values does not come into it.
// Here every removed key has a neighbour holding a value of several
// kilobytes (24 to 63 lines), on a store at its ceiling: no remove draws a
// run of more than 16 lines, each draws a handful of lines in all, and all of
// it is repaid.
TEST_CASE("Small Engine: a remove does not copy its neighbour's value, however large",
          "[small][limit][reserve][reserve-big-values]")
{
   for (int value_bytes : {3000, 6000, 8000})
   {
      temporary_directory tmp;
      auto db = database::open(tmp.path, psitri::open_mode::create_only,
                               small_config(psitri::sync_mode::none, 2));
      auto ws = db->start_write_session();
      auto key = [](int g, char last) {
         char b[32];
         std::snprintf(b, sizeof(b), "L%05d", g);
         return std::string(b) + last;
      };
      const int pairs = 120;
      {
         for (int g = 0; g < pairs; g += 20)
         {
            auto tx = ws->start_transaction(0);
            for (int k = g; k < g + 20; ++k)
            {
               tx.upsert(key(k, 'a'), std::string(20, 'v'));
               tx.upsert(key(k, 'b'), std::string(size_t(value_bytes), 'B'));
            }
            tx.commit();
         }
      }
      for (uint64_t n = 0;; ++n)
      {
         try
         {
            auto tx = ws->start_transaction(0);
            tx.upsert(scattered_key("k", n), std::string(60, 's'));
            tx.commit();
         }
         catch (const engine::storage_full&)
         {
            break;
         }
      }
      db->flush_releases();
      const uint64_t reserve = db->store().config().reserve_lines;
      uint64_t       most = 0, longest = 0;
      for (int g = 0; g < pairs; ++g)
         for (char which : {'a', 'b'})
         {
            uint32_t before[65], during[65];
            db->store().reserve_histogram(before);
            auto tx = ws->start_transaction(0);
            REQUIRE(tx.remove(key(g, which)) == 0);
            db->store().reserve_histogram(during);
            most = std::max<uint64_t>(most, db->store().reserve_owed_lines());
            for (uint32_t k = 2; k <= 64; ++k)
               if (during[k] < before[k])
                  longest = std::max<uint64_t>(longest, k);
            tx.commit();
            db->flush_releases();
            INFO("value " << value_bytes << " pair " << g << " key " << which);
            REQUIRE(db->store().reserve_owed_lines() == 0);
            REQUIRE(db->store().reserve_free_lines() == reserve);
         }
      std::printf("big values (%d bytes, %d lines): %d removes, at most %llu lines drawn by one, "
                  "the longest run taken from stock %llu lines\n",
                  value_bytes, (value_bytes + 8 + 127) / 128, pairs * 2, (unsigned long long)most,
                  (unsigned long long)longest);
      REQUIRE(most <= 16);
      REQUIRE(longest <= 16);
   }
}

// A remove that the reserve cannot serve waits, once and for a bounded time,
// for releases that were already queued. Another writer that keeps the
// release queue busy for ever must not be able to hold it longer than that.
TEST_CASE("Small Engine: a writer that keeps queueing releases cannot hold up a remove",
          "[small][limit][reserve][bounded-wait]")
{
   temporary_directory tmp;
   auto db = database::open(tmp.path, psitri::open_mode::create_only,
                            small_config(psitri::sync_mode::none, 4));
   auto ws = db->start_write_session();
   std::vector<std::string> keys;
   for (uint64_t n = 0;; n += 100)
   {
      try
      {
         auto tx = ws->start_transaction(0);
         for (uint64_t j = n; j < n + 100; ++j)
            tx.upsert(scattered_key("k", j), std::string(60, 's'));
         tx.commit();
         for (uint64_t j = n; j < n + 100; ++j)
            keys.push_back(scattered_key("k", j));
      }
      catch (const engine::storage_full&)
      {
         break;
      }
   }
   // The other writer: root 1, transactions of a few hundred keys built in
   // the reserve's shadow and aborted, over and over. Every abort hands
   // subtrees to the releaser, so its queue is never empty for long.
   std::atomic<bool>     stop{false};
   std::atomic<uint64_t> aborts{0};
   std::thread           busy([&] {
      auto other = db->start_write_session();
      while (!stop.load(std::memory_order_relaxed))
      {
         auto tx = other->start_transaction(1);
         try
         {
            for (int j = 0; j < 3000; ++j)
               tx.upsert(scattered_key("o", j), std::string(40, 'o'));
         }
         catch (const engine::storage_full&)
         {
         }
         tx.abort();
         aborts.fetch_add(1, std::memory_order_relaxed);
      }
   });
   using clock = std::chrono::steady_clock;
   std::vector<double> took;
   uint64_t            refused = 0, removed = 0;
   // transactions of many removes: they run the reserve down, so that draws
   // miss and take the path that waits
   for (int round = 0; round < 30 && removed + 2000 < keys.size(); ++round)
   {
      auto tx = ws->start_transaction(0);
      try
      {
         for (int j = 0; j < 2000; ++j)
         {
            const auto t0 = clock::now();
            const int  rc = tx.remove(keys[removed + size_t(j)]);
            took.push_back(std::chrono::duration<double, std::milli>(clock::now() - t0).count());
            if (rc != 0)
               throw std::runtime_error("a committed key was missing");
         }
         tx.commit();
         removed += 2000;
      }
      catch (const engine::storage_full&)
      {
         took.push_back(0);  // the refused call was timed just above
         ++refused;
         tx.abort();
         removed += 2000;  // forget them; the next round takes other keys
      }
   }
   stop = true;
   busy.join();
   std::sort(took.begin(), took.end());
   std::printf("bounded wait: %zu removes beside a writer that aborted %llu transactions; %llu "
               "transactions refused; slowest remove %.3f ms, 99.9th percentile %.3f ms\n",
               took.size(), (unsigned long long)aborts.load(), (unsigned long long)refused,
               took.back(), took[took.size() * 999 / 1000]);
   REQUIRE(aborts.load() > 0);
   // one bounded wait (half a millisecond) plus the drain of retired
   // versions; generous for a loaded machine, far below an unbounded wait
   REQUIRE(took.back() < 250.0);
}

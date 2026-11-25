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
            if (auto old = tx.template get<std::string>(row_key(id)))
               REQUIRE(tx.remove(index_key(value_seq(*old), id)) == 0);
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

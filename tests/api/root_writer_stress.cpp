// More than one writer, more than one root, more versions than usual
// (gofractally/psibase#2268).
//
// 1. A second root with its own writer thread, the way an embedder keeps
//    rebuildable data beside the chain's root: every change continues from the
//    published root (get_root), edits ONE key in a detached write transaction
//    and publishes the result (set_root), with no flush; now and then the root
//    is replaced by an empty tree. Other threads hand the writer their changes
//    and read the root through their own read sessions; another write session
//    publishes root 0 the whole time.
// 2. The same with large values on one side and, on the other, a chain's way of
//    publishing: trees built detached, handed around as shared references,
//    stored as subtrees of a dense top tree under consecutive keys, published
//    in a transaction, the previous version's reference dropped for the
//    engine's release thread to finish.
// 3. One writer, one tree, more live versions of it than a dense node's
//    per-range share counter (one byte) can count. This is the deterministic
//    form of what 2 runs into whenever versions are dropped faster than the
//    release thread finishes them: before the counter refused to pass its
//    ceiling it wrapped to zero, the range's branches were released while
//    still in use and released again later ("reference count was already
//    zero").
//
// 4. One thread and a model: single-key publications of a root with a dense
//    top, snapshots taken and dropped at random, every key compared.
//
// Sizes by environment, for long runs:
//   PSITRI_ROOT_STRESS_ROUNDS   repetitions of case 1           (default 1)
//   PSITRI_ROOT_STRESS_THREADS  storing threads                 (default 12)
//   PSITRI_ROOT_STRESS_OWN      keys each thread stores         (default 40)
//   PSITRI_ROOT_STRESS_BIG_MB   size of the occasional large value, MiB (default 3)
//   PSITRI_ROOT_STRESS_CHAIN    1 = root 0 is written at the same time (default 1)
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <unistd.h>

#include <psitri/engines/big.hpp>

namespace
{
   using engine = psitri::big_engine;
   constexpr uint32_t cache_root = 2;

   unsigned env(const char* name, unsigned fallback)
   {
      const char* text = std::getenv(name);
      return text && *text ? unsigned(std::strtoul(text, nullptr, 10)) : fallback;
   }

   std::string key(uint32_t value)
   {
      std::string result(32, '\0');
      result[0] = char(uint8_t(value * 37));
      std::memcpy(result.data() + 1, &value, sizeof(value));
      return result;
   }
   std::string artifact(uint32_t value, size_t size)
   {
      std::string result(size, '\0');
      for (size_t i = 0; i < size; ++i)
         result[i] = char(value * 131 + i * 7 + (i >> 8));
      return result;
   }

   // The one thread that writes the root: it creates the write session, uses
   // it and destroys it. Callers hand over a change and wait for it.
   class root_writer
   {
     public:
      using change = std::function<sal_art_ref::radix::tree(engine::write_session&)>;
      explicit root_writer(engine::database& db) : _db(db), _thread([this] { run(); }) {}
      ~root_writer()
      {
         {
            std::lock_guard lock(_mutex);
            _stopping = true;
         }
         _ready.notify_all();
         _thread.join();
      }
      void apply(const change& c)
      {
         request          r{&c};
         std::unique_lock lock(_mutex);
         _queue.push_back(&r);
         _ready.notify_one();
         _finished.wait(lock, [&] { return r.done; });
      }
      void put(std::string_view k, std::string_view v)
      {
         apply([&](engine::write_session& w)
               {
                  auto edit = w.start_write_transaction(w.get_root(cache_root));
                  edit.upsert(k, v);
                  return edit.get_tree();
               });
      }
      void erase(std::string_view k)
      {
         apply([&](engine::write_session& w)
               {
                  auto edit = w.start_write_transaction(w.get_root(cache_root));
                  edit.remove(k);
                  return edit.get_tree();
               });
      }
      void clear()
      {
         apply([&](engine::write_session& w) { return w.create_temporary_tree(); });
      }

     private:
      struct request
      {
         const change* c;
         bool          done = false;
      };
      void run()
      {
         auto             session = _db.start_write_session();
         std::unique_lock lock(_mutex);
         for (;;)
         {
            _ready.wait(lock, [&] { return _stopping || !_queue.empty(); });
            if (_queue.empty())
               break;
            request* r = _queue.front();
            _queue.pop_front();
            lock.unlock();
            session->set_root(cache_root, (*r->c)(*session));
            lock.lock();
            r->done = true;
            _finished.notify_all();
         }
         lock.unlock();
         session.reset();
      }
      engine::database&       _db;
      std::mutex              _mutex;
      std::condition_variable _ready, _finished;
      std::deque<request*>    _queue;
      bool                    _stopping = false;
      std::thread             _thread;
   };

   std::optional<std::string> load(engine::read_session& reader, std::string_view k)
   {
      std::optional<std::string> result;
      auto                       cursor = reader.create_cursor(cache_root);
      cursor.get(k, [&](std::string_view v) { result.emplace(v); });
      return result;
   }
}  // namespace

TEST_CASE("A second root is written key by key by its own thread while others read it",
          "[root_writer]")
{
   const unsigned rounds  = env("PSITRI_ROOT_STRESS_ROUNDS", 1);
   const unsigned threads = env("PSITRI_ROOT_STRESS_THREADS", 12);
   const unsigned own     = env("PSITRI_ROOT_STRESS_OWN", 40);
   const unsigned big_mb  = env("PSITRI_ROOT_STRESS_BIG_MB", 3);
   const bool     chain   = env("PSITRI_ROOT_STRESS_CHAIN", 1) != 0;
   constexpr unsigned shared = 16;

   for (unsigned round = 0; round < rounds; ++round)
   {
      const auto dir = std::filesystem::temp_directory_path() /
                       ("psitri_root_writer_stress_" + std::to_string(::getpid()));
      std::filesystem::remove_all(dir);
      std::atomic<uint32_t> wrong{0}, missing{0};
      {
         engine::config cfg;
         cfg.enable_read_cache        = false;
         cfg.max_pinned_cache_size_mb = 0;
         auto db = engine::database::open(dir, psitri::open_mode::create_only, cfg);
         // The chain's writer: its own session, root 0.
         auto chain_session = db->start_write_session();
         {
            auto tx = chain_session->start_transaction(0);
            tx.upsert("identity", "root-writer-stress");
            tx.commit();
         }
         root_writer              writer(*db);
         std::atomic<bool>        go{false}, done{false};
         std::vector<std::thread> workers;
         for (uint32_t t = 0; t < threads; ++t)
            workers.emplace_back(
                [&, t]
                {
                   auto reader = db->start_read_session();
                   while (!go)
                      std::this_thread::yield();
                   for (uint32_t i = 0; i < own; ++i)
                   {
                      const auto mine = 1000 * (t + 1) + i;
                      writer.put(key(mine), artifact(mine, 2000 + 997 * i));
                      const auto common = i % shared;
                      writer.put(key(common), artifact(common, 300000));
                      if (auto loaded = load(*reader, key(mine)))
                         wrong += *loaded != artifact(mine, 2000 + 997 * i);
                      else
                         ++missing;
                      if (auto loaded = load(*reader, key(common)))
                         wrong += *loaded != artifact(common, 300000);
                      else
                         ++missing;
                      const auto other = 1000 * ((t + 1) % threads + 1) + i;
                      if (auto loaded = load(*reader, key(other)))
                         wrong += *loaded != artifact(other, 2000 + 997 * i);
                      // Now and then: a large value, a removal of it, a key
                      // that is replaced by a value of another size.
                      if (big_mb && i % 13 == t % 13)
                      {
                         const auto large = 500000 + t;
                         writer.put(key(large), artifact(large, (size_t(big_mb) << 20) + 17 * t));
                         if (auto loaded = load(*reader, key(large)))
                            wrong += *loaded != artifact(large, (size_t(big_mb) << 20) + 17 * t);
                         else
                            ++missing;
                         writer.erase(key(large));
                      }
                      if (i % 7 == 3)
                         writer.put(key(mine), artifact(mine, 2000 + 997 * i));
                   }
                });
         std::thread chain_thread;
         if (chain)
            chain_thread = std::thread(
                [&]
                {
                   // Root 0 is written and published by another session the
                   // whole time, as the chain is.
                   for (uint32_t n = 0; !done; ++n)
                   {
                      auto edit = chain_session->start_write_transaction(chain_session->get_root(0));
                      edit.upsert("block-" + std::to_string(n % 64), artifact(n, 100 + n % 4000));
                      chain_session->set_root(0, edit.get_tree());
                   }
                });
         go = true;
         for (auto& worker : workers)
            worker.join();
         done = true;
         if (chain_thread.joinable())
            chain_thread.join();
         CHECK(wrong == 0);
         CHECK(missing == 0);
         // Everything stored is there; then the root is emptied and filled again.
         {
            auto     reader = db->start_read_session();
            uint32_t found  = 0;
            for (uint32_t t = 0; t < threads; ++t)
               for (uint32_t i = 0; i < own; ++i)
                  found += load(*reader, key(1000 * (t + 1) + i)) ==
                           artifact(1000 * (t + 1) + i, 2000 + 997 * i);
            CHECK(found == threads * own);
            writer.clear();
            CHECK(!load(*reader, key(1001)));
            for (uint32_t i = 0; i < shared; ++i)
               writer.put(key(i), artifact(i, 300000));
            for (uint32_t i = 0; i < shared; ++i)
               CHECK(load(*reader, key(i)) == artifact(i, 300000));
         }
      }
      std::filesystem::remove_all(dir);
   }
}

// Case 2. Before the fix this aborted in about half of the runs on an idle
// machine and in every run when the release thread was slowed down.
//   PSITRI_TWO_ROOTS_ROUNDS   repetitions            (default 1)
//   PSITRI_TWO_ROOTS_STORES   stores per storer      (default 50)
//   PSITRI_TWO_ROOTS_STORERS  storing threads        (default 4)
//   PSITRI_TWO_ROOTS_BYTES    size of a stored value (default 1 MiB)
//   PSITRI_TWO_ROOTS_READ     1 = read back each store (default 1)
//   PSITRI_TWO_ROOTS_SHARED   0 = root 0 is published with get_root / set_root instead
TEST_CASE("Two write sessions publish two roots at the same time", "[two_roots]")
{
   const unsigned rounds  = env("PSITRI_TWO_ROOTS_ROUNDS", 1);
   const unsigned stores  = env("PSITRI_TWO_ROOTS_STORES", 50);
   const unsigned storers = env("PSITRI_TWO_ROOTS_STORERS", 4);
   const unsigned bytes   = env("PSITRI_TWO_ROOTS_BYTES", 1u << 20);
   const bool     read    = env("PSITRI_TWO_ROOTS_READ", 1) != 0;
   const bool     shared_chain = env("PSITRI_TWO_ROOTS_SHARED", 1) != 0;

   for (unsigned round = 0; round < rounds; ++round)
   {
      const auto dir = std::filesystem::temp_directory_path() /
                       ("psitri_two_roots_" + std::to_string(::getpid()));
      std::filesystem::remove_all(dir);
      std::atomic<uint32_t> wrong{0};
      const auto            module = artifact(1, bytes);
      {
         engine::config cfg;
         cfg.enable_read_cache        = false;
         cfg.max_pinned_cache_size_mb = 0;
         auto db            = engine::database::open(dir, psitri::open_mode::create_only, cfg);
         auto chain_session = db->start_write_session();
         root_writer       writer(*db);
         std::atomic<bool> done{false};
         std::thread       chain(
             [&]
             {
                // What an embedder's chain does for a block: build the block's trees
                // detached, hand them around as shared references, store them as
                // subtrees of the top tree, publish that in a transaction of root 0,
                // and let go of the references to the previous ones (which the
                // engine's release thread finishes).
                auto&               cs = *chain_session;
                engine::shared_tree top;
                if (shared_chain)
                {
                   // A top tree wide enough for a dense root, as a chain's is.
                   auto edit = cs.start_write_transaction(cs.create_temporary_tree());
                   for (unsigned i = 0; i < 256 * 4; ++i)
                   {
                      std::string k(12, 'b');
                      k[0] = char(uint8_t(i));
                      k[1] = char('a' + i / 256);
                      edit.upsert(k, artifact(i, 300));
                   }
                   top = cs.share(edit.get_tree());
                }
                // Revisions are stored under a few neighbouring branches of the root,
                // as consecutive block numbers are.
                auto revision_key = [](uint32_t n)
                {
                   std::string k = "\x01revision-";
                   k += std::to_string(n);
                   k[0] = char(1 + n % 4);
                   return k;
                };
                for (uint32_t n = 0; !done; ++n)
                {
                   if (!shared_chain)
                   {
                      auto edit = cs.start_write_transaction(cs.get_root(0));
                      edit.upsert("contract", "block");
                      edit.upsert("status-" + std::to_string(n % 256), artifact(n, 200));
                      cs.set_root(0, edit.get_tree());
                      continue;
                   }
                   auto service = cs.start_write_transaction(cs.create_temporary_tree());
                   service.upsert("contract", "block");
                   auto service_root = cs.share(service.get_tree());
                   auto native = cs.start_write_transaction(cs.create_temporary_tree());
                   native.upsert("status", artifact(n, 200));
                   auto native_root = cs.share(native.get_tree());
                   auto list = cs.start_write_transaction(cs.create_temporary_tree());
                   list.upsert("count", "2");
                   list.upsert_subtree("0", cs.adopt(service_root));
                   list.upsert_subtree("1", cs.adopt(native_root));
                   auto edit = cs.start_write_transaction(top ? cs.adopt(top)
                                                              : cs.create_temporary_tree());
                   edit.upsert_subtree(revision_key(n), list.get_tree());
                   edit.upsert("head", std::to_string(n));
                   if (n >= 8)
                      edit.remove(revision_key(n - 8));
                   auto updated = cs.share(edit.get_tree());
                   {
                      auto tx = cs.start_transaction(0);
                      tx.primary().set_root(cs.adopt(updated));
                      tx.commit();
                   }
                   top = std::move(updated);
                }
             });
         std::vector<std::thread> workers;
         for (uint32_t t = 0; t < storers; ++t)
            workers.emplace_back(
                [&, t]
                {
                   auto reader = db->start_read_session();
                   for (uint32_t i = 0; i < stores; ++i)
                   {
                      const auto id = 200000 + 1000 * t + i;
                      writer.put(key(id), module);
                      if (read)
                         wrong += load(*reader, key(id)) != module;
                   }
                });
         for (auto& worker : workers)
            worker.join();
         done = true;
         chain.join();
         CHECK(wrong == 0);
         auto     reader = db->start_read_session();
         uint32_t found  = 0;
         for (uint32_t t = 0; t < storers; ++t)
            for (uint32_t i = 0; i < stores; ++i)
               found += load(*reader, key(200000 + 1000 * t + i)) == module;
         CHECK(found == storers * stores);
      }
      std::filesystem::remove_all(dir);
   }
}

// More live versions of one tree than a dense node's per-range share counter
// can count (it is one byte). Every version is a small change to the same
// tree, kept alive by a shared reference, as an embedder keeps revisions or
// as happens by itself when references are dropped faster than the engine's
// release thread finishes them.
//   PSITRI_VERSIONS  how many versions are alive at once (default 600)
TEST_CASE("More versions of a dense tree are alive than a range counter can count", "[versions]")
{
   const unsigned versions = env("PSITRI_VERSIONS", 600);
   const auto     dir      = std::filesystem::temp_directory_path() /
                             ("psitri_versions_" + std::to_string(::getpid()));
   std::filesystem::remove_all(dir);
   {
      engine::config cfg;
      cfg.enable_read_cache        = false;
      cfg.max_pinned_cache_size_mb = 0;
      auto db     = engine::database::open(dir, psitri::open_mode::create_only, cfg);
      auto writer = db->start_write_session();
      auto reader = db->start_read_session();
      // Keys that differ in their first byte across all 256 values, several
      // per value: the root is a dense node, with leaf pages under it.
      constexpr unsigned keys = 256 * 6;
      auto               name = [](unsigned i)
      {
         std::string k(16, 'k');
         k[0] = char(uint8_t(i));
         k[1] = char('a' + i / 256);
         return k;
      };
      auto value = [](unsigned i, unsigned version)
      { return artifact(i * 7919 + version, 600 + (i + version) % 900); };
      {
         auto edit = writer->start_write_transaction(writer->create_temporary_tree());
         for (unsigned i = 0; i < keys; ++i)
            edit.upsert(name(i), value(i, 0));
         writer->set_root(0, edit.get_tree());
      }
      std::vector<engine::shared_tree> alive;
      std::vector<unsigned>            changed;  // key changed by version v+1
      alive.push_back(writer->share(writer->get_root(0)));
      for (unsigned v = 1; v <= versions; ++v)
      {
         // Always under the same few branches of the root, so the other
         // branches stay shared by every version.
         const unsigned i    = ((v / 32) % 6) * 256 + (v * 5) % 32;
         auto           edit = writer->start_write_transaction(writer->adopt(alive.back()));
         edit.upsert(name(i), value(i, v));
         changed.push_back(i);
         alive.push_back(writer->share(edit.get_tree()));
      }
      // Every version still reads as itself.
      auto check = [&](unsigned v)
      {
         std::vector<unsigned> at(keys, 0);
         for (unsigned u = 1; u <= v; ++u)
            at[changed[u - 1]] = u;
         auto     cursor = reader->create_cursor(alive[v]);
         unsigned wrong  = 0;
         for (unsigned i = 0; i < keys; ++i)
         {
            std::optional<std::string> got;
            cursor.get(name(i), [&](std::string_view s) { got.emplace(s); });
            wrong += got != value(i, at[i]);
         }
         return wrong;
      };
      for (unsigned v : {0u, 1u, versions / 2, versions})
         CHECK(check(v) == 0);
      // Let go of every other version, oldest first, and look again.
      for (unsigned v = 0; v + 1 < alive.size(); v += 2)
         alive[v].reset();
      REQUIRE(db->wait_for_compactor(std::chrono::milliseconds(20000)));
      for (unsigned v = 1; v < alive.size(); v += 2)
         CHECK(check(v) == 0);
      // And the rest.
      const auto last = alive.size() - 1;
      for (unsigned v = 1; v < last; ++v)
         alive[v].reset();
      REQUIRE(db->wait_for_compactor(std::chrono::milliseconds(20000)));
      CHECK(check(unsigned(last)) == 0);
      writer->set_root(0, writer->adopt(alive[last]));
      alive.clear();
      REQUIRE(db->wait_for_compactor(std::chrono::milliseconds(20000)));
      auto     cursor = reader->create_cursor(0);
      unsigned found  = 0;
      for (unsigned i = 0; i < keys; ++i)
         found += cursor.get(name(i), [](std::string_view) {});
      CHECK(found == keys);
   }
   std::filesystem::remove_all(dir);
}

// The same ceiling met while versions are being dropped: a window of live
// versions slides forward, so the release thread destroys old versions (giving
// their shares back, and releasing a range's branches when it is the last
// owner) at the same time as the writer clones the dense node, is refused on
// the ranges every version shares after taking shares on the range it has
// just renewed, and gives those back. Readers check versions inside the window.
//   PSITRI_WINDOW          live versions kept           (default 300)
//   PSITRI_WINDOW_VERSIONS versions written in all      (default 3000)
TEST_CASE("A window of versions slides past the range counter's ceiling", "[versions]")
{
   const unsigned window   = env("PSITRI_WINDOW", 300);
   const unsigned versions = env("PSITRI_WINDOW_VERSIONS", 3000);
   const auto     dir      = std::filesystem::temp_directory_path() /
                             ("psitri_window_" + std::to_string(::getpid()));
   std::filesystem::remove_all(dir);
   {
      engine::config cfg;
      cfg.enable_read_cache        = false;
      cfg.max_pinned_cache_size_mb = 0;
      auto               db     = engine::database::open(dir, psitri::open_mode::create_only, cfg);
      auto               writer = db->start_write_session();
      constexpr unsigned keys   = 256 * 4;
      auto               name   = [](unsigned i)
      {
         std::string k(16, 'k');
         k[0] = char(uint8_t(i));
         k[1] = char('a' + i / 256);
         return k;
      };
      // What key i holds in version v: the last version up to v that wrote it.
      // Version u writes key (u % 4) * 256 + (u * 5) % 32: always under the
      // first 32 branches of the root.
      auto written = [](unsigned u) { return (u % 4) * 256 + (u * 5) % 32; };
      auto value   = [](unsigned i, unsigned version)
      { return artifact(i * 7919 + version, 300 + (i + version) % 500); };

      std::mutex                                             mutex;
      std::deque<std::pair<unsigned, engine::shared_tree>>   alive;
      {
         auto edit = writer->start_write_transaction(writer->create_temporary_tree());
         for (unsigned i = 0; i < keys; ++i)
            edit.upsert(name(i), value(i, 0));
         alive.emplace_back(0, writer->share(edit.get_tree()));
      }
      std::atomic<bool>     done{false};
      std::atomic<uint32_t> wrong{0}, checked{0};
      std::vector<std::thread> readers;
      for (unsigned t = 0; t < 3; ++t)
         readers.emplace_back(
             [&, t]
             {
                auto reader = db->start_read_session();
                for (unsigned n = t; !done; n += 7)
                {
                   unsigned            v;
                   engine::shared_tree tree;
                   {
                      std::lock_guard lock(mutex);
                      auto& entry = alive[n % alive.size()];
                      v           = entry.first;
                      tree        = entry.second;
                   }
                   auto cursor = reader->create_cursor(tree);
                   // The keys the last few versions wrote, and some never written.
                   for (unsigned back = 0; back < 40 && back <= v; ++back)
                   {
                      const unsigned i = written(v - back);
                      unsigned       last = 0;
                      for (unsigned u = v; u >= 1; --u)
                         if (written(u) == i)
                         {
                            last = u;
                            break;
                         }
                      std::optional<std::string> got;
                      cursor.get(name(i), [&](std::string_view s) { got.emplace(s); });
                      wrong += got != value(i, last);
                   }
                   for (unsigned i = 40 + (n % 200); i < keys; i += 211)
                   {
                      if (i % 256 < 32)
                         continue;  // a key some version writes
                      std::optional<std::string> got;
                      cursor.get(name(i), [&](std::string_view s) { got.emplace(s); });
                      wrong += got != value(i, 0);
                   }
                   ++checked;
                }
             });
      for (unsigned v = 1; v <= versions; ++v)
      {
         engine::shared_tree base;
         {
            std::lock_guard lock(mutex);
            base = alive.back().second;
         }
         auto edit = writer->start_write_transaction(writer->adopt(base));
         edit.upsert(name(written(v)), value(written(v), v));
         auto                next = writer->share(edit.get_tree());
         engine::shared_tree dropped;
         {
            std::lock_guard lock(mutex);
            alive.emplace_back(v, std::move(next));
            if (alive.size() > window)
            {
               dropped = std::move(alive.front().second);
               alive.pop_front();
            }
         }
         // `dropped` and `base` let go here, outside the lock: the release
         // thread finishes them.
      }
      done = true;
      for (auto& reader : readers)
         reader.join();
      CHECK(wrong == 0);
      CHECK(checked > 0);
      // Publish the newest, let go of everything else, and read it all.
      writer->set_root(0, writer->adopt(alive.back().second));
      alive.clear();
      REQUIRE(db->wait_for_compactor(std::chrono::milliseconds(30000)));
      auto     reader = db->start_read_session();
      auto     cursor = reader->create_cursor(0);
      unsigned bad    = 0;
      for (unsigned i = 0; i < keys; ++i)
      {
         unsigned last = 0;
         for (unsigned u = versions; u >= 1; --u)
            if (written(u) == i)
            {
               last = u;
               break;
            }
         std::optional<std::string> got;
         cursor.get(name(i), [&](std::string_view s) { got.emplace(s); });
         bad += got != value(i, last);
      }
      CHECK(bad == 0);
   }
   std::filesystem::remove_all(dir);
}

// One thread, no timing: single-key publications of a root whose top is
// dense, with snapshots taken and let go at random and the release thread's
// work either waited for or not. After every few changes every key is
// compared with a model, through a fresh snapshot and through the held ones.
//   PSITRI_MODEL_CHANGES  changes made    (default 4000)
//   PSITRI_MODEL_SEED     random seed     (default 1)
TEST_CASE("Single-key publications of a dense root agree with a model", "[model]")
{
   const unsigned changes = env("PSITRI_MODEL_CHANGES", 4000);
   const unsigned seed    = env("PSITRI_MODEL_SEED", 1);
   const auto     dir     = std::filesystem::temp_directory_path() /
                            ("psitri_model_" + std::to_string(::getpid()));
   std::filesystem::remove_all(dir);
   {
      engine::config cfg;
      cfg.enable_read_cache        = false;
      cfg.max_pinned_cache_size_mb = 0;
      auto db     = engine::database::open(dir, psitri::open_mode::create_only, cfg);
      auto writer = db->start_write_session();
      auto reader = db->start_read_session();

      uint64_t state = 0x9E3779B97F4A7C15ull * (seed + 1);
      auto     next  = [&]
      {
         state ^= state << 13;
         state ^= state >> 7;
         state ^= state << 17;
         return state;
      };
      constexpr unsigned ids = 700;
      // id -> version of its value (0 = absent)
      std::vector<unsigned> model(ids, 0);
      auto size_of = [](unsigned id, unsigned version)
      { return id < 16 ? 300000u : 2000u + 997u * ((id + version) % 40); };
      auto value_of = [&](unsigned id, unsigned version)
      { return artifact(id * 31 + version, size_of(id, version)); };
      struct held
      {
         engine::shared_tree   tree;
         std::vector<unsigned> model;
      };
      std::vector<held> snapshots;
      auto compare = [&](engine::cursor cursor, const std::vector<unsigned>& expect)
      {
         unsigned bad = 0;
         for (unsigned id = 0; id < ids; ++id)
         {
            std::optional<std::string> got;
            cursor.get(key(id < 16 ? id : 1000 + id), [&](std::string_view s) { got.emplace(s); });
            if (expect[id] == 0)
               bad += got.has_value();
            else
               bad += got != value_of(id, expect[id]);
         }
         return bad;
      };
      unsigned bad = 0;
      for (unsigned change = 1; change <= changes && bad == 0; ++change)
      {
         const auto     roll = next();
         const unsigned id   = unsigned(roll % 100 < 15 ? (roll >> 8) % 16 : (roll >> 8) % ids);
         const auto     k    = key(id < 16 ? id : 1000 + id);
         if ((roll >> 40) % 10 == 0 && model[id])
         {
            auto edit = writer->start_write_transaction(writer->get_root(cache_root));
            edit.remove(k);
            writer->set_root(cache_root, edit.get_tree());
            model[id] = 0;
         }
         else if ((roll >> 40) % 400 == 1)
         {
            writer->set_root(cache_root, writer->create_temporary_tree());
            std::fill(model.begin(), model.end(), 0);
         }
         else
         {
            auto edit = writer->start_write_transaction(writer->get_root(cache_root));
            edit.upsert(k, value_of(id, change));
            writer->set_root(cache_root, edit.get_tree());
            model[id] = change;
         }
         // A reader looks at what was just published, sometimes keeps it.
         {
            auto cursor = reader->create_cursor(cache_root);
            std::optional<std::string> got;
            cursor.get(k, [&](std::string_view s) { got.emplace(s); });
            bad += model[id] ? got != value_of(id, model[id]) : got.has_value();
            if ((roll >> 20) % 3 == 0)
               snapshots.push_back({reader->get_root(cache_root), model});
         }
         if (!snapshots.empty() && (roll >> 24) % 2 == 0)
         {
            const auto at = (roll >> 28) % snapshots.size();
            snapshots.erase(snapshots.begin() + long(at));
         }
         if (snapshots.size() > 40)
            snapshots.erase(snapshots.begin(), snapshots.begin() + 20);
         if ((roll >> 50) % 7 == 0)
            REQUIRE(db->wait_for_compactor(std::chrono::milliseconds(30000)));
         if (change % 50 == 0)
         {
            bad += compare(reader->create_cursor(cache_root), model);
            for (auto& snapshot : snapshots)
               bad += compare(reader->create_cursor(snapshot.tree), snapshot.model);
            INFO("change " << change);
            CHECK(bad == 0);
         }
      }
      CHECK(bad == 0);
      snapshots.clear();
   }
   std::filesystem::remove_all(dir);
}

// The transaction types in <psitri/transactions.hpp>, on every engine.
//
// Each case drives a transaction and a plain std::map with the same random
// operations and requires every read to agree: point reads, bounds, scans in
// both directions, range removes and nested savepoints, and requires a cursor
// to keep showing the state it was created on. A read that ignored the
// transaction's own changes, in any of those forms, fails here. The engines'
// own transactions are held to the same reads.

#include <psitri/api.hpp>
#include <psitri/transactions.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace
{
   struct temporary_directory
   {
      std::filesystem::path path;
      temporary_directory()
      {
         std::string pattern =
             (std::filesystem::temp_directory_path() / "psitri-tx-XXXXXX").string();
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

   template <psitri::engine E>
   typename E::config config()
   {
      typename E::config cfg;
      cfg.sync_mode = psitri::sync_mode::none;
      if constexpr (std::same_as<E, psitri::small_engine>)
         cfg.max_segments = 32;
      else
         cfg.max_pinned_cache_size_mb = 128;
      return cfg;
   }

   using model = std::map<std::string, std::string, std::less<>>;

   // Keys cluster under a few prefixes so that ranges and neighbours matter.
   std::string random_key(std::mt19937& rng)
   {
      static const char* prefixes[] = {"", "a", "a/", "ab", "b/", "b/x/", "zz"};
      std::string        key        = prefixes[rng() % 7];
      auto               n          = rng() % 40;
      if (n < 36)
         key += std::to_string(n);
      if (rng() % 9 == 0)
         key.push_back('\0');
      return key;
   }
   std::string random_value(std::mt19937& rng)
   {
      return std::string(rng() % 5 == 0 ? 300 + rng() % 400 : rng() % 12, char('A' + rng() % 26));
   }

   template <class Cursor>
   void require_same_scan(Cursor cursor, const model& expected)
   {
      auto it = expected.begin();
      for (bool more = cursor.seek_begin(); more; more = cursor.next(), ++it)
      {
         REQUIRE(it != expected.end());
         REQUIRE(cursor.key() == it->first);
         REQUIRE(cursor.template value<std::string>() == std::optional<std::string>{it->second});
      }
      REQUIRE(it == expected.end());
      REQUIRE(cursor.is_end());
      auto rit = expected.rbegin();
      for (bool more = cursor.seek_last(); more; more = cursor.prev(), ++rit)
      {
         REQUIRE(rit != expected.rend());
         REQUIRE(cursor.key() == rit->first);
      }
      REQUIRE(rit == expected.rend());
      REQUIRE(cursor.is_rend());
   }

   template <class Tx>
   concept has_savepoints = requires(Tx& tx) { tx.save(); };

   // Savepoints where the transaction type has them.
   template <class Tx>
   struct savepoints
   {
      void save(Tx&, const model&) {}
      bool close(Tx&, model&, bool) { return false; }
      void release_all(Tx&) {}
   };
   template <has_savepoints Tx>
   struct savepoints<Tx>
   {
      std::vector<std::pair<typename Tx::savepoint, model>> open;
      void save(Tx& tx, const model& expected)
      {
         if (open.size() < 4)
            open.emplace_back(tx.save(), expected);
      }
      bool close(Tx& tx, model& expected, bool rollback)
      {
         if (open.empty())
            return false;
         if (rollback)
         {
            tx.rollback(open.back().first);
            expected = open.back().second;
         }
         else
            tx.release(open.back().first);
         open.pop_back();
         return true;
      }
      void release_all(Tx& tx)
      {
         while (!open.empty())
         {
            tx.release(open.back().first);
            open.pop_back();
         }
      }
   };

   // Applies `steps` random operations to both and checks every read.
   template <class Tx>
   void exercise(Tx& tx, model& expected, std::mt19937& rng, int steps)
   {
      savepoints<Tx> saved;
      for (int step = 0; step < steps; ++step)
      {
         auto key = random_key(rng);
         switch (rng() % 14)
         {
            case 0:
            case 1:
            case 2:
            case 3:
            {
               auto value = random_value(rng);
               tx.upsert(key, value);
               expected[key] = value;
               break;
            }
            case 4:
            case 5:
            {
               int removed = tx.remove(key);
               REQUIRE(removed == (expected.erase(key) ? 0 : -1));
               break;
            }
            case 6:
            {
               auto found = expected.find(key);
               REQUIRE(tx.template get<std::string>(key) ==
                       (found == expected.end() ? std::nullopt
                                                : std::optional<std::string>{found->second}));
               break;
            }
            case 7:
            {
               // First key at or after.
               auto cursor = tx.snapshot_cursor();
               auto found  = expected.lower_bound(key);
               REQUIRE(cursor.lower_bound(key) == (found != expected.end()));
               if (found != expected.end())
               {
                  REQUIRE(cursor.key() == found->first);
                  REQUIRE(cursor.template value<std::string>() ==
                          std::optional<std::string>{found->second});
               }
               else
                  REQUIRE(cursor.is_end());
               break;
            }
            case 8:
            {
               // Last key at or before.
               auto cursor = tx.snapshot_cursor();
               auto found  = expected.upper_bound(key);
               bool any    = found != expected.begin();
               REQUIRE(cursor.upper_bound(key) == any);
               if (any)
                  REQUIRE(cursor.key() == std::prev(found)->first);
               else
                  REQUIRE(cursor.is_rend());
               break;
            }
            case 9:
            {
               // Walk a few steps each way from a bound, changing direction.
               auto cursor = tx.snapshot_cursor();
               auto found  = expected.lower_bound(key);
               if (!cursor.lower_bound(key))
                  break;
               for (int i = 0; i < 6; ++i)
               {
                  if (rng() % 2)
                  {
                     ++found;
                     REQUIRE(cursor.next() == (found != expected.end()));
                     if (found == expected.end())
                        break;
                  }
                  else
                  {
                     bool any = found != expected.begin();
                     REQUIRE(cursor.prev() == any);
                     if (!any)
                        break;
                     --found;
                  }
                  REQUIRE(cursor.key() == found->first);
                  REQUIRE(cursor.template value<std::string>() ==
                          std::optional<std::string>{found->second});
               }
               break;
            }
            case 10:
            {
               // A cursor keeps showing the state it was created on.
               auto  cursor = tx.snapshot_cursor();
               model frozen = expected;
               for (int i = 0; i < 3; ++i)
               {
                  auto other = random_key(rng);
                  auto value = random_value(rng);
                  if (rng() % 2)
                  {
                     tx.upsert(other, value);
                     expected[other] = value;
                  }
                  else
                  {
                     tx.remove(other);
                     expected.erase(other);
                  }
               }
               require_same_scan(std::move(cursor), frozen);
               break;
            }
            case 11:
            {
               auto upper = random_key(rng);
               if (upper < key)
                  std::swap(key, upper);
               if (key.empty() || upper.empty() || key == upper)
                  break;
               auto          first = expected.lower_bound(key);
               auto          last  = expected.lower_bound(upper);
               std::uint64_t count = std::distance(first, last);
               expected.erase(first, last);
               REQUIRE(tx.remove_range_counted(key, upper) == count);
               break;
            }
            case 12:
               saved.save(tx, expected);
               break;
            case 13:
               saved.close(tx, expected, rng() % 2);
               break;
         }
         if (step % 25 == 0)
            require_same_scan(tx.snapshot_cursor(), expected);
      }
      saved.release_all(tx);
      require_same_scan(tx.snapshot_cursor(), expected);
   }

   template <class Session>
   model fill(Session& writer, std::uint32_t root, std::mt19937& rng, int count)
   {
      model contents;
      auto  tx = writer.start_transaction(root);
      for (int i = 0; i < count; ++i)
      {
         auto key      = random_key(rng);
         contents[key] = random_value(rng);
         tx.upsert(key, contents[key]);
      }
      tx.commit();
      return contents;
   }
}  // namespace

TEMPLATE_TEST_CASE("Transactions: a transient transaction reads its own changes and leaves the root alone",
                   "[api][transactions]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());
   auto                writer = db->start_write_session();

   for (std::uint32_t seed = 1; seed <= 12; ++seed)
   {
      std::mt19937 rng(seed);
      auto         committed = fill(*writer, 1, rng, seed == 1 ? 0 : 60);
      // Start from whatever earlier rounds committed as well.
      {
         auto cursor = writer->get_root(1).snapshot_cursor();
         committed.clear();
         for (bool more = cursor.seek_begin(); more; more = cursor.next())
            committed[std::string(cursor.key())] = *cursor.template value<std::string>();
      }
      psitri::transient_transaction tx(writer->get_root(1));
      auto                          expected = committed;
      exercise(tx, expected, rng, 400);
      // Nothing reached the root.
      require_same_scan(writer->get_root(1).snapshot_cursor(), committed);
      tx.reset();
      require_same_scan(tx.snapshot_cursor(), committed);
   }
}

TEMPLATE_TEST_CASE("Transactions: a delta transaction commits exactly what it showed",
                   "[api][transactions]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());
   auto                writer = db->start_write_session();
   model               committed;

   for (std::uint32_t seed = 1; seed <= 12; ++seed)
   {
      std::mt19937 rng(seed * 7919);
      bool         keep = seed % 3 != 0;
      {
         psitri::delta_transaction<E> tx(*writer, 2);
         auto                         expected = committed;
         exercise(tx, expected, rng, 400);
         // Until commit the root is unchanged for everyone else.
         std::thread(
             [&]
             {
                auto reader = db->start_read_session();
                auto cursor = reader->create_cursor(2);
                auto it     = committed.begin();
                bool same   = true;
                for (bool more = cursor.seek_begin(); more; more = cursor.next(), ++it)
                   if (it == committed.end() || cursor.key() != it->first)
                   {
                      same = false;
                      break;
                   }
                REQUIRE(same);
                REQUIRE(it == committed.end());
             })
             .join();
         if (keep)
         {
            tx.commit();
            committed = std::move(expected);
         }
         else
            tx.abort();
      }
      require_same_scan(writer->get_root(2).snapshot_cursor(), committed);
   }
}

TEMPLATE_TEST_CASE("Transactions: transient transactions run in parallel on one snapshot",
                   "[api][transactions]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());
   model               committed;
   {
      auto         writer = db->start_write_session();
      std::mt19937 rng(42);
      committed = fill(*writer, 0, rng, 200);
   }
   std::atomic<bool>        failed{false};
   std::vector<std::thread> threads;
   for (std::uint32_t t = 0; t < 4; ++t)
   {
      threads.emplace_back(
          [&, t]
          {
             // Each thread reads through its own read session.
             struct reader_snapshot
             {
                std::shared_ptr<typename E::read_session> session;
                auto snapshot_cursor() const { return session->create_cursor(0); }
             };
             reader_snapshot base{db->start_read_session()};
             std::mt19937    rng(1000 + t);
             for (int round = 0; round < 20 && !failed; ++round)
             {
                psitri::transient_transaction tx(base);
                model                         expected = committed;
                for (int i = 0; i < 200; ++i)
                {
                   auto key = random_key(rng);
                   if (rng() % 3)
                   {
                      auto value = random_value(rng);
                      tx.upsert(key, value);
                      expected[key] = value;
                   }
                   else
                   {
                      tx.remove(key);
                      expected.erase(key);
                   }
                }
                auto cursor = tx.snapshot_cursor();
                auto it     = expected.begin();
                for (bool more = cursor.seek_begin(); more; more = cursor.next(), ++it)
                   if (it == expected.end() || cursor.key() != it->first ||
                       cursor.template value<std::string>() != std::optional<std::string>{it->second})
                   {
                      failed = true;
                      break;
                   }
                if (it != expected.end())
                   failed = true;
             }
          });
   }
   for (auto& thread : threads)
      thread.join();
   REQUIRE_FALSE(failed);
}

TEMPLATE_TEST_CASE("Transactions: an engine's own transaction reads its own changes",
                   "[api][transactions]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());
   auto                writer = db->start_write_session();
   model               committed;

   for (std::uint32_t seed = 1; seed <= 12; ++seed)
   {
      std::mt19937 rng(seed * 104729);
      bool         keep = seed % 3 != 0;
      {
         auto tx       = writer->start_transaction(4);
         auto expected = committed;
         exercise(tx, expected, rng, 400);
         if (keep)
         {
            tx.commit();
            committed = std::move(expected);
         }
         else
            tx.abort();
      }
      require_same_scan(writer->get_root(4).snapshot_cursor(), committed);
   }
}

TEST_CASE("Transactions: changes kept aside are applied to a tree that is in no root",
          "[api][transactions][detached]")
{
   using E  = psitri::big_engine;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());
   auto                writer = db->start_write_session();
   std::mt19937        rng(7);

   // A tree of its own, never installed in a root.
   model contents;
   auto  base = [&]
   {
      auto edit = writer->start_write_transaction(writer->create_temporary_tree());
      for (int i = 0; i < 300; ++i)
      {
         auto key      = random_key(rng);
         contents[key] = random_value(rng);
         edit.upsert(key, contents[key]);
      }
      return edit.get_tree();
   }();

   for (int round = 0; round < 8; ++round)
   {
      // Work on it without touching it...
      psitri::transient_transaction tx(base);
      auto                          expected = contents;
      exercise(tx, expected, rng, 300);
      require_same_scan(base.snapshot_cursor(), contents);
      if (round % 3 == 2)
         continue;  // ...and sometimes throw the work away.
      // ...then write the changes into a copy of it in one pass.
      auto edit = writer->start_write_transaction(base);
      tx.apply_to(edit);
      auto next = edit.get_tree();
      require_same_scan(base.snapshot_cursor(), contents);  // the old tree is as it was
      require_same_scan(next.snapshot_cursor(), expected);
      base     = std::move(next);
      contents = std::move(expected);
   }
}

TEMPLATE_TEST_CASE("Transactions: a nested transaction's changes fold into the outer one",
                   "[api][transactions]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());
   auto                writer = db->start_write_session();
   std::mt19937        rng(11);
   model               committed = fill(*writer, 5, rng, 200);
   {
      auto cursor = writer->get_root(5).snapshot_cursor();
      committed.clear();
      for (bool more = cursor.seek_begin(); more; more = cursor.next())
         committed[std::string(cursor.key())] = *cursor.template value<std::string>();
   }

   // The block: one delta transaction. Each transaction in it works on its
   // own changes over the block so far and is folded in, or dropped, whole.
   psitri::delta_transaction<E> block(*writer, 5);
   model                        expected = committed;
   for (int n = 0; n < 30; ++n)
   {
      struct block_so_far
      {
         psitri::delta_transaction<E>* block;
         auto snapshot_cursor() const { return block->snapshot_cursor(); }
      };
      psitri::transient_transaction tx(block_so_far{&block});
      model                         inner = expected;
      exercise(tx, inner, rng, 60);
      if (n % 4 == 3)
         continue;  // this transaction failed: nothing of it remains
      tx.apply_to(block);
      expected = std::move(inner);
      require_same_scan(block.snapshot_cursor(), expected);
   }
   require_same_scan(writer->get_root(5).snapshot_cursor(), committed);  // nothing published yet
   block.commit();
   require_same_scan(writer->get_root(5).snapshot_cursor(), expected);
}

TEST_CASE("Transactions: a store of changes can be read as it was at each mark",
          "[transactions][delta_store]")
{
   // An embedder that folds its changes into a tree while savepoints are open
   // needs the changes as of each savepoint, not only as they are now.
   psitri::delta_store                              store;
   std::vector<psitri::delta_store::mark>           marks;
   std::vector<std::map<std::string, std::string>>  expected;  // "" stands for removed
   std::map<std::string, std::string>               model;
   std::mt19937                                     rng(7);
   REQUIRE(store.bytes() == 0);
   for (int level = 0; level < 6; ++level)
   {
      marks.push_back(store.save());
      expected.push_back(model);
      for (int i = 0; i < 200; ++i)
      {
         std::string key = "k" + std::to_string(rng() % 300);
         if (rng() % 4 == 0)
         {
            store.erase(key);
            model[key] = "";
         }
         else
         {
            std::string value = "v" + std::to_string(level) + "-" + std::to_string(i);
            store.put(key, value);
            model[key] = value;
         }
      }
   }
   auto contents = [](psitri::delta_store::view view)
   {
      std::map<std::string, std::string> result;
      for (auto it = view.begin(), end = view.end(); it != end; ++it)
         result[std::string(it.key())] =
             it.value().removed ? std::string() : std::string(it.value().value());
      return result;
   };
   for (std::size_t i = 0; i < marks.size(); ++i)
      REQUIRE(contents(store.at(marks[i])) == expected[i]);
   REQUIRE(contents(store.freeze()) == model);

   // Rolling back leaves the earlier marks readable and does not shrink the region.
   const auto used = store.bytes();
   REQUIRE(used > 0);
   store.restore(marks[3]);
   REQUIRE(contents(store.freeze()) == expected[3]);
   for (std::size_t i = 0; i <= 3; ++i)
      REQUIRE(contents(store.at(marks[i])) == expected[i]);
   REQUIRE(store.bytes() >= used);
   store.clear();
   REQUIRE(store.bytes() == 0);
   REQUIRE(store.bytes_held() >= used);  // the region is kept for reuse
   REQUIRE(store.empty());
}

TEMPLATE_TEST_CASE("Cursors: an empty root is both ended and reverse-ended after a failed seek",
                   "[api][cursors]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());
   auto                writer = db->start_write_session();
   auto                cursor = writer->get_root(3).snapshot_cursor();
   REQUIRE_FALSE(cursor.seek_begin());
   CHECK(cursor.is_end());
   REQUIRE_FALSE(cursor.seek_last());
   CHECK(cursor.is_rend());
   REQUIRE_FALSE(cursor.lower_bound("a"));
   CHECK(cursor.is_end());
   REQUIRE_FALSE(cursor.upper_bound("a"));
   CHECK(cursor.is_rend());
   {
      auto tx = writer->start_transaction(3);
      tx.upsert("m", "v");
      tx.commit();
   }
   auto one = writer->get_root(3).snapshot_cursor();
   REQUIRE_FALSE(one.lower_bound("n"));
   CHECK(one.is_end());
   REQUIRE_FALSE(one.upper_bound("l"));
   CHECK(one.is_rend());
   REQUIRE(one.seek_last());
   REQUIRE_FALSE(one.prev());
   CHECK(one.is_rend());
   REQUIRE(one.seek_begin());
   REQUIRE_FALSE(one.next());
   CHECK(one.is_end());
}

// Not a correctness case: how long a small transaction that is thrown away
// takes, kept aside in a transient transaction versus made in the engine's own
// transaction and aborted. Run with "[.][bench]".
TEMPLATE_TEST_CASE("Bench: a small discarded transaction, kept aside versus in the engine",
                   "[.][bench]",
                   psitri::small_engine,
                   psitri::big_engine)
{
   using E  = TestType;
   using DB = psitri::database_for<E>;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, config<E>());
   auto                writer = db->start_write_session();
   constexpr int       base_keys = 200000;
   auto                name = [](int i)
   {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "acct/%08d/bal", i);
      return std::string(buffer);
   };
   {
      auto tx = writer->start_transaction(0);
      for (int i = 0; i < base_keys; ++i)
         tx.upsert(name(i), "0000000000000000");
      tx.commit();
   }
   constexpr int rounds = 20000;
   auto          work   = [&](auto& tx, std::mt19937& rng)
   {
      std::size_t found = 0;
      for (int i = 0; i < 16; ++i)
      {
         auto key = name(rng() % base_keys);
         found += tx.template get<std::string>(key).has_value();
         tx.upsert(key, "1111111111111111");
         found += tx.template get<std::string>(key).has_value();
      }
      auto cursor = tx.snapshot_cursor();
      if (cursor.lower_bound(name(rng() % base_keys)))
         for (int i = 0; i < 4 && cursor.next(); ++i)
            ++found;
      return found;
   };
   using clock = std::chrono::steady_clock;
   std::size_t sink = 0;

   std::mt19937 rng1(1);
   auto         t0   = clock::now();
   auto         base = writer->get_root(0);
   for (int r = 0; r < rounds; ++r)
   {
      psitri::transient_transaction tx(base);
      sink += work(tx, rng1);
   }
   auto t1 = clock::now();
   std::mt19937 rng2(1);
   for (int r = 0; r < rounds; ++r)
   {
      auto tx = writer->start_transaction(0);
      sink += work(tx, rng2);
      tx.abort();
   }
   auto t2 = clock::now();
   auto us = [&](auto a, auto b)
   { return std::chrono::duration<double, std::micro>(b - a).count() / rounds; };
   std::printf("BENCH discarded transaction (16 writes, 32 reads, 1 short scan): kept aside %.2f us, "
               "in the engine %.2f us  [%zu]\n",
               us(t0, t1), us(t1, t2), sink);
   SUCCEED();
}

// The transaction types in <psitri/transactions.hpp>, on every engine.
//
// Each case drives a transaction and a plain std::map with the same random
// operations and requires every read to agree: point reads, bounds, scans in
// both directions, a cursor held across writes, range removes and nested
// savepoints. A read that ignored the transaction's own changes, in any of
// those forms, fails here.

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

   // Applies `steps` random operations to both and checks every read.
   template <class Tx>
   void exercise(Tx& tx, model& expected, std::mt19937& rng, int steps)
   {
      using savepoint = typename Tx::savepoint;
      std::vector<std::pair<savepoint, model>> saved;
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
               // A cursor held while the transaction is written to.
               auto cursor = tx.snapshot_cursor();
               if (!cursor.lower_bound(key))
                  break;
               std::string at(cursor.key());
               auto        other = random_key(rng);
               auto        value = random_value(rng);
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
               auto after = expected.upper_bound(at);
               REQUIRE(cursor.next() == (after != expected.end()));
               if (after != expected.end())
                  REQUIRE(cursor.key() == after->first);
               break;
            }
            case 11:
            {
               auto upper = random_key(rng);
               if (upper < key)
                  std::swap(key, upper);
               if (rng() % 6 == 0)
                  upper.clear();
               auto first = key.empty() ? expected.begin() : expected.lower_bound(key);
               auto last  = upper.empty() ? expected.end() : expected.lower_bound(upper);
               std::uint64_t count = 0;
               if (upper.empty() || key < upper)
               {
                  count = std::distance(first, last);
                  expected.erase(first, last);
               }
               REQUIRE(tx.remove_range(key, upper) == count);
               break;
            }
            case 12:
            {
               if (saved.size() < 4)
                  saved.emplace_back(tx.save(), expected);
               break;
            }
            case 13:
            {
               if (saved.empty())
                  break;
               if (rng() % 2)
               {
                  tx.rollback(saved.back().first);
                  expected = saved.back().second;
               }
               else
                  tx.release(saved.back().first);
               saved.pop_back();
               break;
            }
         }
         if (step % 25 == 0)
            require_same_scan(tx.snapshot_cursor(), expected);
      }
      while (!saved.empty())
      {
         tx.release(saved.back().first);
         saved.pop_back();
      }
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

// The writer, root and reader rules, as test cases. Include after defining
// WRITER_ENGINES (the engine types to run on) and writer_config<E>(bool
// small_store = false), which returns the configuration to open E with; a
// small store is one whose freed space is reused within a few commits.
// WRITER_ROOT0_ENGINES, when defined, lists the engines for the cases that need
// only root 0; it defaults to WRITER_ENGINES.
#ifndef WRITER_ROOT0_ENGINES
#define WRITER_ROOT0_ENGINES WRITER_ENGINES
#endif


TEMPLATE_TEST_CASE("Writers: sessions on different roots commit concurrently",
                   "[api][writers]",
                   WRITER_ENGINES)
{
   using E  = TestType;
   using DB = typename E::database;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, writer_config<E>());

   constexpr std::uint32_t roots  = 4;
   constexpr std::uint32_t rounds = 40;
   constexpr std::uint32_t batch  = 250;

   std::atomic<bool>        go{false}, done{false}, torn{false}, foreign{false};
   std::atomic<std::size_t> snapshots{0};
   std::vector<std::thread> threads;
   for (std::uint32_t root = 0; root < roots; ++root)
   {
      threads.emplace_back(
          [&, root]
          {
             auto writer = db->start_write_session();
             while (!go)
                std::this_thread::yield();
             for (std::uint32_t round = 0; round < rounds; ++round)
             {
                auto tx = writer->start_transaction(root);
                for (std::uint32_t i = 0; i < batch; ++i)
                   tx.upsert(key(root, round, i), key(root, round, i));
                tx.commit();
             }
          });
   }
   // Readers: every state they see is a whole number of committed batches of
   // that root's own keys.
   for (std::uint32_t root = 0; root < roots; ++root)
   {
      threads.emplace_back(
          [&, root]
          {
             auto reader = db->start_read_session();
             while (!go)
                std::this_thread::yield();
             while (!done)
             {
                auto cursor = reader->create_cursor(root);
                bool other  = false;
                auto n      = count_own_keys(cursor, root, other);
                if (other)
                   foreign = true;
                if (n % batch != 0)
                   torn = true;
                ++snapshots;
             }
          });
   }
   go = true;
   for (std::uint32_t root = 0; root < roots; ++root)
      threads[root].join();
   done = true;
   for (std::uint32_t i = roots; i < threads.size(); ++i)
      threads[i].join();

   REQUIRE_FALSE(torn);
   REQUIRE_FALSE(foreign);
   REQUIRE(snapshots > 0);
   auto reader = db->start_read_session();
   for (std::uint32_t root = 0; root < roots; ++root)
   {
      auto cursor = reader->create_cursor(root);
      bool other  = false;
      REQUIRE(count_own_keys(cursor, root, other) == std::size_t{rounds} * batch);
      REQUIRE_FALSE(other);
   }
}

TEMPLATE_TEST_CASE("Writers: a second writer on a root waits and no commit is lost",
                   "[api][writers]",
                   WRITER_ROOT0_ENGINES)
{
   using E  = TestType;
   using DB = typename E::database;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, writer_config<E>());

   constexpr std::uint32_t writers = 4;
   constexpr std::uint32_t rounds  = 200;
   std::atomic<bool>        go{false};
   std::atomic<int>         open{0};
   std::atomic<bool>        overlapped{false};
   std::vector<std::thread> threads;
   for (std::uint32_t w = 0; w < writers; ++w)
   {
      threads.emplace_back(
          [&, w]
          {
             auto writer = db->start_write_session();
             while (!go)
                std::this_thread::yield();
             for (std::uint32_t round = 0; round < rounds; ++round)
             {
                auto tx = writer->start_transaction(0);
                // Only one transaction may be open on the root at a time.
                if (open.fetch_add(1) != 0)
                   overlapped = true;
                tx.upsert(key(w, round, 0), "v");
                std::this_thread::yield();
                open.fetch_sub(1);
                tx.commit();
             }
          });
   }
   go = true;
   for (auto& thread : threads)
      thread.join();

   REQUIRE_FALSE(overlapped);
   auto        reader = db->start_read_session();
   auto        cursor = reader->create_cursor(0);
   std::size_t n      = 0;
   for (cursor.seek_begin(); !cursor.is_end(); cursor.next())
      ++n;
   REQUIRE(n == std::size_t{writers} * rounds);
}

TEMPLATE_TEST_CASE("Writers: a held cursor is unchanged by commits and aborts are invisible",
                   "[api][writers]",
                   WRITER_ROOT0_ENGINES)
{
   using E  = TestType;
   using DB = typename E::database;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, writer_config<E>());

   auto writer = db->start_write_session();
   {
      auto tx = writer->start_transaction(0);
      for (std::uint32_t i = 0; i < 100; ++i)
         tx.upsert(key(0, 0, i), "first");
      tx.commit();
   }
   std::atomic<int>  stage{0};
   std::atomic<bool> wrong{false};
   std::thread       reader_thread(
       [&]
       {
          auto reader = db->start_read_session();
          auto held   = reader->create_cursor(0);
          stage       = 1;
          while (stage != 2)
             std::this_thread::yield();
          // The cursor created before the writer's work still shows that state.
          std::size_t n = 0;
          for (held.seek_begin(); !held.is_end(); held.next())
          {
             if (held.template value<std::string>() != std::optional<std::string>{"first"})
                wrong = true;
             ++n;
          }
          if (n != 100)
             wrong = true;
          // A new cursor shows the committed work and none of the aborted work.
          auto        fresh = reader->create_cursor(0);
          std::size_t m     = 0;
          for (fresh.seek_begin(); !fresh.is_end(); fresh.next())
          {
             if (fresh.template value<std::string>() != std::optional<std::string>{"second"})
                wrong = true;
             ++m;
          }
          if (m != 150)
             wrong = true;
       });
   while (stage != 1)
      std::this_thread::yield();
   {
      auto tx = writer->start_transaction(0);
      for (std::uint32_t i = 0; i < 150; ++i)
         tx.upsert(key(0, 0, i), "second");
      tx.commit();
   }
   {
      auto tx = writer->start_transaction(0);
      for (std::uint32_t i = 0; i < 300; ++i)
         tx.upsert(key(0, 0, i), "aborted");
      tx.abort();
   }
   stage = 2;
   reader_thread.join();
   REQUIRE_FALSE(wrong);
}

TEMPLATE_TEST_CASE("Writers: the eight portable roots are separate keyspaces",
                   "[api][writers]",
                   WRITER_ENGINES)
{
   using E  = TestType;
   using DB = typename E::database;
   temporary_directory temp;
   auto                db = DB::open(temp.path / "db", psitri::open_mode::create_only, writer_config<E>());

   auto writer = db->start_write_session();
   for (std::uint32_t root = 0; root < 8; ++root)
   {
      auto tx = writer->start_transaction(root);
      for (std::uint32_t i = 0; i <= root; ++i)
         tx.upsert(key(root, 0, i), "v");
      tx.commit();
   }
   auto reader = db->start_read_session();
   for (std::uint32_t root = 0; root < 8; ++root)
   {
      auto cursor = reader->create_cursor(root);
      bool other  = false;
      REQUIRE(count_own_keys(cursor, root, other) == root + 1);
      REQUIRE_FALSE(other);
   }
}

// Large Engine extension: trees that are not installed in any root. Each
// writer edits its own copy of one shared ancestor at the same time; the
// ancestor and the other writers' copies are unaffected.
TEMPLATE_TEST_CASE("Readers: every cursor of a read session stays on its own snapshot",
                   "[api][writers][readers]",
                   WRITER_ROOT0_ENGINES)
{
   using E  = TestType;
   using DB = typename E::database;
   temporary_directory temp;
   // A small store, so that space freed by one commit is reused by the next
   // ones and a snapshot that is not protected is soon overwritten.
   auto cfg = writer_config<E>(/*small_store=*/true);
   auto db     = DB::open(temp.path / "db", psitri::open_mode::create_only, cfg);
   auto writer = db->start_write_session();
   auto reader = db->start_read_session();

   auto write_generation = [&](std::uint32_t generation)
   {
      auto tx = writer->start_transaction(0);
      // Every key is rewritten, so nothing of an older generation is shared.
      for (std::uint32_t i = 0; i < 2000; ++i)
         tx.upsert(key(0, 0, i), "generation " + std::to_string(generation) + std::string(400, '.'));
      tx.commit();
   };
   auto all_are = [](auto& cursor, std::uint32_t generation)
   {
      const std::optional<std::string> expected{"generation " + std::to_string(generation) +
                                                std::string(400, '.')};
      std::size_t                      n = 0;
      for (bool more = cursor.seek_begin(); more; more = cursor.next(), ++n)
         if (cursor.template value<std::string>() != expected)
            return false;
      return n == 2000;
   };

   write_generation(1);
   auto first = reader->create_cursor(0);
   write_generation(2);
   auto second = reader->create_cursor(0);
   write_generation(3);
   auto third = reader->create_cursor(0);
   // Enough further commits that anything not held would be reclaimed and reused.
   for (std::uint32_t generation = 4; generation < 300; ++generation)
      write_generation(generation);

   CHECK(all_are(first, 1));
   CHECK(all_are(second, 2));
   CHECK(all_are(third, 3));
   auto latest = reader->create_cursor(0);
   CHECK(all_are(latest, 299));
   CHECK(all_are(first, 1));
}

TEMPLATE_TEST_CASE("Sessions: a session keeps its database open",
                   "[api][writers][lifetime]",
                   WRITER_ROOT0_ENGINES)
{
   using E  = TestType;
   using DB = typename E::database;
   temporary_directory temp;
   const auto          path = temp.path / "db";
   {
      auto db     = DB::open(path, psitri::open_mode::create_only, writer_config<E>());
      auto writer = db->start_write_session();
      auto reader = db->start_read_session();
      // The last handle to the database goes away while its sessions are in use.
      db.reset();
      {
         auto tx = writer->start_transaction(0);
         for (std::uint32_t i = 0; i < 500; ++i)
            tx.upsert(key(0, 0, i), "kept");
         tx.commit();
      }
      auto        cursor = reader->create_cursor(0);
      std::size_t n      = 0;
      for (bool more = cursor.seek_begin(); more; more = cursor.next())
         ++n;
      REQUIRE(n == 500);
   }
   // Everything was closed in order: the store opens again with the data.
   auto        db     = DB::open(path, psitri::open_mode::open_existing, writer_config<E>());
   auto        reader = db->start_read_session();
   auto        cursor = reader->create_cursor(0);
   std::size_t n      = 0;
   for (bool more = cursor.seek_begin(); more; more = cursor.next())
      ++n;
   REQUIRE(n == 500);
}

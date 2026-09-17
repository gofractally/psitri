#include <psitri/api.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

// The same application builds for either engine; no backend headers or
// private include directories are needed by the application code below.
#ifdef PSITRI_EXAMPLE_SMALL
using engine_type = psitri::small_engine;
#else
using engine_type = psitri::big_engine;
#endif

template <psitri::engine Engine>
void example(const std::filesystem::path& directory)
{
   typename Engine::config cfg;
   cfg.sync_mode = psitri::sync_mode::fsync;
   auto db = Engine::database::open(directory, psitri::open_mode::create_only, cfg);
   auto writer = db->start_write_session();
   {
      auto tx = writer->start_transaction(0);
      tx.upsert("account/alice", "100");
      tx.upsert("account/bob", "50");
      tx.commit();
   } // destroy the transaction before starting another

   auto before = writer->get_root(0); // owned, frozen snapshot
   {
      auto tx = writer->start_transaction(0);
      tx.upsert("account/alice", "125");
      tx.commit();
   }
   {
      auto tx = writer->start_transaction(0);
      tx.upsert("account/bob", "0");
      // Leaving this scope without commit aborts the changes.
   }

   auto old = before.snapshot_cursor();
   if (old.template get<std::string>("account/alice") != "100")
      throw std::runtime_error("snapshot changed");

   auto reader = db->start_read_session();
   auto cursor = reader->create_cursor(0); // snapshot of the latest commit
   const auto alice = cursor.template get<std::string>("account/alice");
   if (alice != "125")
      throw std::runtime_error("committed value missing");

   // Optional capability: older engines compile this same application without
   // range methods. Selection happens entirely at compile time.
   if constexpr (psitri::range_engine<Engine>)
   {
      if (cursor.template get<std::string>("account/alice", 1, 1) != "2")
         throw std::runtime_error("value range mismatch");
   }

   // Views passed to callbacks are valid only for that call. Copy to retain.
   cursor.get("account/bob", [](std::string_view value) {
      std::cout << "Bob: " << value << '\n';
   });

   // Keys are ordered bytes. lower_bound is inclusive.
   for (bool found = cursor.lower_bound("account/");
        found && cursor.key().starts_with("account/"); found = cursor.next())
   {
      std::cout << cursor.key() << " = " << *cursor.template value<std::string>() << '\n';
   }
   // Reverse bounds use psitri's historical floor convention: <=, not STL >.
   if (!cursor.upper_bound("account/alice") || cursor.key() != "account/alice")
      throw std::runtime_error("floor lookup failed");
   // Locals are destroyed in reverse order: cursors/snapshots, sessions, db.
}

int main(int argc, char** argv)
{
   if (argc != 2)
   {
      std::cerr << "Usage: " << argv[0] << " NEW_DATABASE_DIRECTORY\n";
      return 2;
   }
   try
   {
      example<engine_type>(argv[1]);
   }
   catch (const std::exception& e)
   {
      std::cerr << e.what() << '\n';
      return 1;
   }
}

// Big-engine extension: persist revisions as owned subtrees under one
// published manifest root. This is not part of the portable byte-only API.
#include <psitri/engines/big.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>

void revisions(const std::filesystem::path& path)
{
   using E = psitri::big_engine;
   E::config cfg;
   cfg.sync_mode = psitri::sync_mode::fsync;
   {
      auto db = E::database::open(path, psitri::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      {
         auto tx = ws->start_transaction(0);
         auto& state = tx.create_subtree("state");
         state.upsert("alice", "100");
         tx.commit();
      }
      auto manifest = ws->get_root(0);
      auto revision = manifest.get_subtree("state");
      {
         auto tx = ws->start_transaction(0);
         tx.upsert_subtree("history/1", revision); // database retains the old state
         auto& state = tx.open_subtree("state");
         state.upsert("alice", "125");
         tx.commit(); // publishes the manifest and its children through root 0
      }
   }
   {
      auto db = E::database::open(path, psitri::open_mode::open_existing, cfg);
      auto ws = db->start_write_session();
      auto manifest = ws->get_root(0);
      auto saved = manifest.get_subtree("history/1");
      if (saved.get<std::string>("alice") != "100" ||
          manifest.get_subtree("state").get<std::string>("alice") != "125")
         throw std::runtime_error("persisted revision mismatch");
      {
         auto tx = ws->start_transaction(0);
         tx.upsert_subtree("state", saved); // restore the old state by ownership
         tx.commit();
      }
      auto restored = ws->get_root(0);
      if (restored.get_subtree("state").get<std::string>("alice") != "100")
         throw std::runtime_error("revision restore failed");
      std::cout << "Saved, reopened, and restored revision 1\n";
   }
}

int main(int argc, char** argv)
{
   if (argc != 2)
   {
      std::cerr << "Usage: " << argv[0] << " NEW_DATABASE_DIRECTORY\n";
      return 2;
   }
   try { revisions(argv[1]); }
   catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

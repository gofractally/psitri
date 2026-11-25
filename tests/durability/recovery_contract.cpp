// Large Engine unclean-shutdown and read-only-open contract.
//
//   recovery_contract <workdir>                 run every scenario, each in its own process
//   recovery_contract <workdir> <scenario>      run one scenario in this process
//
// Writer children are re-executions of this binary that are SIGKILLed while they
// still hold roots, detached trees, and an open transaction. Nothing here injects
// faults into the library: the process simply dies, as psinode would.
#include <psitri/engines/big.hpp>
#include <sal/alloc_header.hpp>
#include <sal/mapped_memory/segment.hpp>

#include <fcntl.h>
#include <sys/file.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
   namespace fs = std::filesystem;
   using Engine = psitri::big_engine;

   constexpr unsigned rootCount  = 8;
   constexpr unsigned keyCount   = 6000;
   constexpr unsigned childCount = 300;
   // Sorts after every generated key except those whose first byte is 0xff.
   const std::string  subtreeKey = "\xff\xff\xffsubtree";

   bool           durable = true;
   Engine::config config()
   {
      Engine::config result;
      result.sync_mode = durable ? psitri::sync_mode::fsync : psitri::sync_mode::none;
      result.enable_read_cache        = false;
      result.max_pinned_cache_size_mb = 0;
      result.max_database_size        = 1024LL * 1024 * 1024;
      return result;
   }
   void require(bool value, const std::string& message)
   {
      if (!value)
         throw std::runtime_error(message);
   }
   // Two leading hash bytes give the upper tree levels full fan-out, so the store
   // contains the dense node tiers that share per-range reference counters.
   std::string makeKey(unsigned index)
   {
      const std::uint32_t hash = index * 2654435761u;
      char                buffer[32];
      std::snprintf(buffer, sizeof(buffer), "xx-%06u", index);
      std::string result(buffer);
      result[0] = char(hash >> 24);
      result[1] = char(hash >> 16);
      return result;
   }
   // Keys in engine (unsigned bytewise) order.
   const std::vector<std::string>& orderedKeys(unsigned count)
   {
      static std::map<unsigned, std::vector<std::string>> cache;
      auto&                                               result = cache[count];
      if (result.empty())
      {
         for (unsigned i = 0; i < count; ++i)
            result.push_back(makeKey(i));
         std::sort(result.begin(), result.end(), [](const std::string& a, const std::string& b) {
            return std::lexicographical_compare(
                a.begin(), a.end(), b.begin(), b.end(),
                [](char x, char y) { return static_cast<unsigned char>(x) < static_cast<unsigned char>(y); });
         });
      }
      return result;
   }
   std::string key(unsigned index)
   {
      return makeKey(index);
   }
   // Same length for every generation: overwriting keeps the tree shape, so the
   // reachable object count is a property of the key set alone.
   std::string value(unsigned root, unsigned index, unsigned generation)
   {
      const std::size_t size = index % 97 == 0 ? 6000 : 48;
      std::string       result(size, '\0');
      std::uint64_t     state = (std::uint64_t(generation) << 40) ^ (std::uint64_t(root) << 32) ^ index;
      for (auto& c : result)
      {
         state = state * 6364136223846793005ULL + 1442695040888963407ULL;
         c     = char(state >> 56);
      }
      return result;
   }

   // The adapter's shape: edit a retained copy, then publish it as the root.
   void writeGeneration(Engine::write_session& writer, unsigned generation)
   {
      for (unsigned root = 0; root < rootCount; ++root)
      {
         auto child = writer.start_write_transaction(writer.create_temporary_tree());
         for (unsigned i = 0; i < childCount; ++i)
            child.upsert(key(i), value(root + 100, i, generation));
         auto base = writer.get_root(root);
         auto edit = writer.start_write_transaction(base ? base : writer.create_temporary_tree());
         for (unsigned i = 0; i < keyCount; ++i)
            edit.upsert(key(i), value(root, i, generation));
         edit.upsert_subtree(subtreeKey, child.get_tree());
         auto tx = writer.start_transaction(root);
         tx.primary().set_root(edit.get_tree());
         tx.commit();
      }
   }
   unsigned indexOf(const std::string& key)
   {
      return static_cast<unsigned>(std::stoul(key.substr(3)));
   }
   void verifyGeneration(Engine::database& database, unsigned generation)
   {
      const auto& keys      = orderedKeys(keyCount);
      const auto& childKeys = orderedKeys(childCount);
      auto        reader    = database.start_read_session();
      for (unsigned root = 0; root < rootCount; ++root)
      {
         auto     cursor = reader->create_cursor(root);
         unsigned seen   = 0;
         bool     child  = false;
         for (bool found = cursor.lower_bound(std::string_view{}); found; found = cursor.next())
         {
            if (cursor.key() == subtreeKey)
            {
               require(cursor.is_subtree(), "subtree entry lost its type");
               auto     nested = cursor.subtree_cursor();
               unsigned inner  = 0;
               for (bool f = nested.lower_bound(std::string_view{}); f; f = nested.next(), ++inner)
                  require(inner < childCount && nested.key() == childKeys[inner] &&
                              nested.value<std::string>() ==
                                  value(root + 100, indexOf(childKeys[inner]), generation),
                          "subtree content mismatch in root " + std::to_string(root));
               require(inner == childCount, "subtree key count mismatch");
               child = true;
               continue;
            }
            require(seen < keyCount && cursor.key() == keys[seen],
                    "unexpected key in root " + std::to_string(root));
            require(cursor.value<std::string>() == value(root, indexOf(keys[seen]), generation),
                    "value mismatch in root " + std::to_string(root) + " at position " +
                        std::to_string(seen) + " (expected generation " +
                        std::to_string(generation) + ")");
            ++seen;
         }
         require(seen == keyCount && child, "root " + std::to_string(root) + " is incomplete");
      }
   }

   struct Usage
   {
      std::uint64_t objects  = 0;
      std::uint64_t counters = 0;
      std::uint64_t bytes    = 0;
   };
   Usage settle(Engine::database& database)
   {
      Usage previous;
      for (unsigned stable = 0, attempt = 0; attempt < 400; ++attempt)
      {
         require(database.wait_for_compactor(std::chrono::seconds(30)), "release queue did not drain");
         Usage current{database.engine_for_testing().object_count(),
                       database.underlying_allocator().iref().used(), 0};
         stable = current.objects == previous.objects && current.counters == previous.counters
                      ? stable + 1
                      : 0;
         previous = current;
         if (stable == 5)
            break;
         std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      return previous;
   }
   std::uint64_t segmentBytes(const fs::path& database)
   {
      return fs::file_size(database / "segs");
   }

   struct File
   {
      std::uint64_t size;
      std::uint64_t hash;
      std::int64_t  mtime;
      bool          operator==(const File&) const = default;
   };
   std::map<std::string, File> fingerprint(const fs::path& directory)
   {
      std::map<std::string, File> result;
      for (const auto& entry : fs::recursive_directory_iterator(directory))
      {
         if (!entry.is_regular_file())
            continue;
         std::ifstream     file(entry.path(), std::ios::binary);
         std::uint64_t     hash = 1469598103934665603ULL, size = 0;
         std::vector<char> buffer(1 << 20);
         while (file.read(buffer.data(), buffer.size()) || file.gcount())
         {
            for (std::streamsize i = 0; i < file.gcount(); ++i)
               hash = (hash ^ static_cast<unsigned char>(buffer[i])) * 1099511628211ULL;
            size += file.gcount();
         }
         struct stat status;
         require(::stat(entry.path().c_str(), &status) == 0, "stat failed");
#ifdef __APPLE__
         const std::int64_t mtime = status.st_mtimespec.tv_sec * 1000000000LL + status.st_mtimespec.tv_nsec;
#else
         const std::int64_t mtime = status.st_mtim.tv_sec * 1000000000LL + status.st_mtim.tv_nsec;
#endif
         result[fs::relative(entry.path(), directory).string()] = {size, hash, mtime};
      }
      return result;
   }
   std::string difference(const std::map<std::string, File>& before,
                          const std::map<std::string, File>& after)
   {
      std::string result;
      for (const auto& [name, file] : after)
         if (!before.contains(name) || before.at(name) != file)
            result += " " + name;
      for (const auto& [name, file] : before)
         if (!after.contains(name))
            result += " -" + name;
      return result;
   }

   // ── writer child ──────────────────────────────────────────────────────────
   // Publishes generations [first, last], keeping everything a live node keeps,
   // then waits to be killed. With `pause`, it stops after `first` until the
   // parent writes a byte, so the parent can copy never-fsynced files.
   int writerChild(const fs::path& path, unsigned first, unsigned last, bool pause)
   {
      auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
      auto writer   = database->start_write_session();
      std::vector<psitri::snapshot_for<Engine>> held;
      for (unsigned generation = first; generation <= last; ++generation)
      {
         writeGeneration(*writer, generation);
         for (unsigned root = 0; root < rootCount; ++root)
            held.push_back(writer->get_root(root));
         auto detached = writer->start_write_transaction(writer->create_temporary_tree());
         for (unsigned i = 0; i < 400; ++i)
            detached.upsert(key(i), value(200, i, generation));
         held.push_back(detached.get_tree());
         if (pause && generation == first)
         {
            std::cout << "SNAPSHOT\n" << std::flush;
            char byte;
            if (::read(STDIN_FILENO, &byte, 1) != 1)
               return 3;
         }
      }
      auto unfinished = writer->start_write_transaction(writer->get_root(0));
      for (unsigned i = 0; i < 200; ++i)
         unfinished.upsert(key(i), value(300, i, last));
      std::cout << "READY\n" << std::flush;
      for (;;)
         ::pause();
   }

   std::string self;
   struct Child
   {
      pid_t pid    = -1;
      int   output = -1;
      int   input  = -1;
      void  expect(const std::string& line)
      {
         std::string text;
         char        c;
         while (::read(output, &c, 1) == 1 && c != '\n')
            text += c;
         require(text == line, "writer child reported '" + text + "', expected '" + line + "'");
      }
      void resume() { require(::write(input, "g", 1) == 1, "cannot resume writer child"); }
      ~Child()
      {
         if (pid > 0)
         {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
         }
      }
      void kill()
      {
         ::kill(pid, SIGKILL);
         int status = 0;
         ::waitpid(pid, &status, 0);
         pid = -1;
         ::close(output);
         ::close(input);
         require(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL, "writer child was not killed");
      }
   };
   std::unique_ptr<Child> spawnWriter(const fs::path& path, unsigned first, unsigned last, bool pause)
   {
      int out[2], in[2];
      require(::pipe(out) == 0 && ::pipe(in) == 0, "pipe failed");
      const auto a = std::to_string(first), b = std::to_string(last);
      const pid_t pid = ::fork();
      require(pid >= 0, "fork failed");
      if (pid == 0)
      {
         ::dup2(out[1], STDOUT_FILENO);
         ::dup2(in[0], STDIN_FILENO);
         for (int fd : {out[0], out[1], in[0], in[1]})
            ::close(fd);
         ::execl(self.c_str(), self.c_str(), path.c_str(), "writer-child", a.c_str(), b.c_str(),
                 pause ? "pause" : "run", static_cast<char*>(nullptr));
         std::_Exit(96);
      }
      ::close(out[1]);
      ::close(in[0]);
      return std::unique_ptr<Child>(new Child{pid, out[0], in[1]});
   }
   // A killed process is reaped before the kernel has necessarily finished closing its
   // files (the last release of a dying task's files can run just after its parent's
   // wait returns), and the store's files are locked without waiting. So after killing a
   // writer, wait until its lock on the store is really gone before opening it.
   void waitForStoreLock(const fs::path& path)
   {
      const auto file = path / "ptrs" / "header.bin";
      for (int attempt = 0; attempt < 5000; ++attempt)
      {
         const int fd = ::open(file.c_str(), O_RDONLY);
         require(fd >= 0, "cannot open " + file.string());
         const bool unlocked = ::flock(fd, LOCK_EX | LOCK_NB) == 0;
         if (unlocked)
            ::flock(fd, LOCK_UN);
         ::close(fd);
         if (unlocked)
            return;
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      throw std::runtime_error("the killed writer's lock on the store was never released");
   }
   void crashWriter(const fs::path& path, unsigned first, unsigned last)
   {
      auto child = spawnWriter(path, first, last, false);
      child->expect("READY");
      child->kill();
      waitForStoreLock(path);
   }
   void create(const fs::path& path, unsigned generation)
   {
      fs::remove_all(path);
      auto database = Engine::database::open(path, psitri::open_mode::create_only, config());
      auto writer   = database->start_write_session();
      writeGeneration(*writer, generation);
   }

   // ── scenarios ─────────────────────────────────────────────────────────────

   // Finding 1, scenario A: the process dies, the OS survives. Every reference
   // the dead process held must be forgotten, and what it leaked reclaimed.
   void crashLoop(const fs::path& path)
   {
      create(path, 1);
      Usage baseline;
      {
         // A clean writer that overwrites every key gives the steady-state shape.
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         {
            auto writer = database->start_write_session();
            writeGeneration(*writer, 2);
            writeGeneration(*writer, 3);
         }
         verifyGeneration(*database, 3);
         baseline = settle(*database);
      }
      std::cout << "baseline objects=" << baseline.objects << " counters=" << baseline.counters
                << " segment_bytes=" << segmentBytes(path) << '\n';
      std::vector<std::uint64_t> sizes;
      for (unsigned cycle = 1; cycle <= 6; ++cycle)
      {
         const unsigned last = 10 * cycle + 3;
         crashWriter(path, 10 * cycle + 1, last);
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         verifyGeneration(*database, last);
         const auto usage = settle(*database);
         database.reset();
         sizes.push_back(segmentBytes(path));
         std::cout << "cycle " << cycle << " objects=" << usage.objects
                   << " counters=" << usage.counters << " segment_bytes=" << sizes.back() << '\n';
         require(usage.objects == baseline.objects,
                 "crash leaked objects: " + std::to_string(usage.objects) + " live after cycle " +
                     std::to_string(cycle) + ", baseline " + std::to_string(baseline.objects));
         require(usage.counters == baseline.counters,
                 "crash leaked range counters: " + std::to_string(usage.counters) + ", baseline " +
                     std::to_string(baseline.counters));
      }
      require(sizes.back() <= sizes[1], "store grows with every crash: " +
                                            std::to_string(sizes[1]) + " -> " +
                                            std::to_string(sizes.back()) + " segment bytes");
   }

   // After recovery the store must keep working: further writes, a clean close,
   // and a clean reopen that needs no second recovery.
   void usableAfterRecovery(const fs::path& path)
   {
      create(path, 1);
      crashWriter(path, 2, 4);
      {
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         verifyGeneration(*database, 4);
         auto writer = database->start_write_session();
         writeGeneration(*writer, 5);
         writeGeneration(*writer, 6);
      }
      const auto closed = fingerprint(path);
      {
         auto database = Engine::database::open(path, psitri::open_mode::read_only, config());
         verifyGeneration(*database, 6);
      }
      require(fingerprint(path) == closed, "read-only open after recovery modified the store");
      auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
      verifyGeneration(*database, 6);
   }

   // Recovery writes, so a read-only open of an unclean store must refuse
   // instead of reading through whatever control blocks the dead writer left.
   void readOnlyRefusesUnclean(const fs::path& path)
   {
      create(path, 1);
      crashWriter(path, 2, 3);
      const auto before = fingerprint(path);
      bool       refused = false;
      try
      {
         auto database = Engine::database::open(path, psitri::open_mode::read_only, config());
      }
      catch (const std::exception& error)
      {
         refused = std::string(error.what()).find("not shut down cleanly") != std::string::npos;
         require(refused, std::string("unexpected read-only failure: ") + error.what());
      }
      require(refused, "read-only open accepted a store that was not shut down cleanly");
      require(fingerprint(path) == before, "refused read-only open modified the store");
      auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
      verifyGeneration(*database, 3);
   }

   // A restart gives the kernel a new boot session. Model it by making the record
   // name a session that is not this one.
   void simulateRestart(const fs::path& path)
   {
      std::fstream file(path / "shutdown", std::ios::in | std::ios::out | std::ios::binary);
      require(bool(file), "store has no shutdown record");
      std::string record((std::istreambuf_iterator<char>(file)), {});
      const auto  begin = record.find("boot=");
      require(begin != std::string::npos && record.find("state=open") != std::string::npos,
              "shutdown record does not say the store is open");
      const auto end = record.find('\n', begin);
      for (auto i = begin + 5; i < end; ++i)
         record[i] = '0';
      file.seekp(0);
      file.write(record.data(), static_cast<std::streamsize>(record.size()));
   }

   // Finding 1, scenario B: power is lost. `segs` and `roots` are flushed by every
   // publication; `ptrs`, `iref`, and `header` never are. Model the files that were
   // never flushed as (stale) the copy from before the last publications, or
   // (missing) never having reached the disk at all. Without `restarted` the record
   // still claims this boot's page cache is intact: recovery must notice that the
   // control blocks contradict the roots instead of freeing what they do not list.
   void powerLoss(const fs::path& path, bool missing, bool restarted)
   {
      create(path, 1);
      const fs::path saved  = path.string() + ".unflushed";
      const auto     volatileFiles = {"ptrs", "iref", "header"};
      fs::remove_all(saved);
      fs::create_directories(saved);
      auto child = spawnWriter(path, 2, 5, true);
      child->expect("SNAPSHOT");
      for (const char* name : volatileFiles)
         fs::copy(path / name, saved / name, fs::copy_options::recursive);
      child->resume();
      child->expect("READY");
      child->kill();
      waitForStoreLock(path);
      for (const char* name : volatileFiles)
      {
         fs::remove_all(path / name);
         if (!missing)
            fs::copy(saved / name, path / name, fs::copy_options::recursive);
      }
      if (restarted)
         simulateRestart(path);
      {
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         require(database->recovery_performed() == sal::recovery_mode::power_loss,
                 "store was not rebuilt from its flushed segments");
         verifyGeneration(*database, 5);
         auto writer = database->start_write_session();
         writeGeneration(*writer, 6);
         verifyGeneration(*database, 6);
         settle(*database);
      }
      auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
      verifyGeneration(*database, 6);
   }

   // What a power cut leaves after the last flushed commit: pages that were written
   // but never flushed. Model the worst of them, a newer unpublished version of a live
   // object, by appending damaged duplicates of flushed objects after each segment's
   // last flushed sync header and extending the footer over them. A rebuild that reads
   // past the flushed prefix prefers these (same sequence, later position).
   unsigned tearTails(const fs::path& path)
   {
      using sal::mapped_memory::segment;
      const int fd = ::open((path / "segs").c_str(), O_RDWR);
      require(fd >= 0, "cannot open segment file");
      const auto count  = fs::file_size(path / "segs") / sal::segment_size;
      auto*      buffer = static_cast<char*>(std::aligned_alloc(4096, sal::segment_size));
      unsigned   torn   = 0;
      for (std::uint64_t index = 0; index < count; ++index)
      {
         const off_t base = static_cast<off_t>(index * sal::segment_size);
         require(::pread(fd, buffer, sal::segment_size, base) == ssize_t(sal::segment_size),
                 "short segment read");
         auto*         seg = reinterpret_cast<segment*>(buffer);
         std::uint32_t              pos = 0, flushed = 0;
         std::vector<std::uint32_t> starts;
         while (pos + 64 <= seg->end_pos())
         {
            const auto* header = reinterpret_cast<const sal::alloc_header*>(seg->data + pos);
            if (header->size() == 0 || header->size() > seg->end_pos() - pos)
               break;
            starts.push_back(pos);
            pos += header->size();
            if (header->type() == sal::header_type::sync_head)
               flushed = pos;
         }
         if (flushed == 0)
            continue;
         // The most recently flushed whole objects (the live ones), as many as fit.
         const std::uint32_t room   = std::min<std::uint32_t>(seg->end_pos() - flushed, 4u << 20);
         std::uint32_t       source = flushed;
         for (auto start = starts.rbegin(); start != starts.rend(); ++start)
            if (*start < flushed && flushed - *start <= room)
               source = *start;
         const std::uint32_t length = flushed - source;
         if (length == 0)
            continue;
         std::memcpy(seg->data + flushed, seg->data + source, length);
         for (std::uint32_t offset = 0; offset < length;)
         {
            auto* header = reinterpret_cast<sal::alloc_header*>(seg->data + flushed + offset);
            for (std::uint32_t i = sizeof(sal::alloc_header) + 16; i < header->size(); ++i)
               seg->data[flushed + offset + i] ^= 0x5a;
            offset += header->size();
         }
         seg->set_alloc_pos(flushed + length);
         require(::pwrite(fd, buffer, sal::segment_size, base) == ssize_t(sal::segment_size),
                 "short segment write");
         ++torn;
      }
      std::free(buffer);
      ::close(fd);
      return torn;
   }
   void tornTail(const fs::path& path)
   {
      create(path, 1);
      Usage baseline;
      {
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         auto writer   = database->start_write_session();
         writeGeneration(*writer, 2);
         writeGeneration(*writer, 3);
         writer.reset();
         baseline = settle(*database);
      }
      crashWriter(path, 4, 7);
      simulateRestart(path);
      require(tearTails(path) != 0, "no segment had a flushed prefix to tear");
      {
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         require(database->recovery_performed() == sal::recovery_mode::power_loss,
                 "store was not rebuilt from its flushed segments");
         verifyGeneration(*database, 7);
         const auto usage = settle(*database);
         require(usage.objects == baseline.objects && usage.counters == baseline.counters,
                 "rebuild kept unflushed objects: " + std::to_string(usage.objects) +
                     " objects, baseline " + std::to_string(baseline.objects));
         auto writer = database->start_write_session();
         writeGeneration(*writer, 8);
      }
      auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
      verifyGeneration(*database, 8);
   }

   // A store with no shutdown record (written before the record existed, or with the
   // record lost) has unknown history and is rebuilt from its segments. On a store
   // that was in fact closed cleanly, the rebuild must reproduce it exactly.
   void rebuildIsExact(const fs::path& path)
   {
      // Enough rewrites that segments are compacted and recycled: the rebuild then
      // has to pick the live copy of every relocated object.
      const unsigned churn = 120;
      create(path, 1);
      Usage clean;
      {
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         auto writer   = database->start_write_session();
         for (unsigned generation = 2; generation <= churn; ++generation)
            writeGeneration(*writer, generation);
         clean = settle(*database);
         const auto dump = database->dump();
         std::cout << "churn segments=" << dump.total_segments << " free_space=" << dump.total_free_space
                   << " objects=" << clean.objects << '\n';
      }
      {
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         require(database->recovery_performed() == sal::recovery_mode::none,
                 "a cleanly closed store was recovered");
      }
      fs::remove(path / "shutdown");
      for (unsigned pass = 0; pass < 2; ++pass)
      {
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         require(database->recovery_performed() ==
                     (pass == 0 ? sal::recovery_mode::power_loss : sal::recovery_mode::none),
                 "unexpected recovery on pass " + std::to_string(pass));
         verifyGeneration(*database, churn);
         const auto usage = settle(*database);
         require(usage.objects == clean.objects && usage.counters == clean.counters,
                 "rebuild changed the store: " + std::to_string(usage.objects) + " objects, " +
                     std::to_string(usage.counters) + " counters; clean close had " +
                     std::to_string(clean.objects) + ", " + std::to_string(clean.counters));
      }
   }

   // Durability is opt-in by sync level. A store that does not flush on commit never
   // flushes for the shutdown record either, and is never charged a rebuild from
   // segments: every unclean open gets the light pass, even after a restart.
   void nonDurableStaysLight(const fs::path& path)
   {
      durable = false;
      create(path, 1);
      Usage baseline;
      {
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         auto writer   = database->start_write_session();
         writeGeneration(*writer, 2);
         writeGeneration(*writer, 3);
         writer.reset();
         baseline = settle(*database);
      }
      for (bool restarted : {false, true})
      {
         crashWriter(path, 4, 6);
         if (restarted)
            simulateRestart(path);
         auto database = Engine::database::open(path, psitri::open_mode::open_existing, config());
         require(database->recovery_performed() == sal::recovery_mode::app_crash,
                 "a non-durable store was charged a rebuild from segments");
         verifyGeneration(*database, 6);
         const auto usage = settle(*database);
         require(usage.objects == baseline.objects && usage.counters == baseline.counters,
                 "non-durable store leaked after a crash");
      }
      crashWriter(path, 7, 8);
      auto database = Engine::database::open(path, psitri::open_mode::open_existing, config(),
                                             sal::recovery_mode::deferred_cleanup);
      require(database->recovery_performed() == sal::recovery_mode::deferred_cleanup,
              "deferred recovery was not honored");
      verifyGeneration(*database, 8);
      auto writer = database->start_write_session();
      writeGeneration(*writer, 9);
      verifyGeneration(*database, 9);
   }

   // Finding 2: two read-only openers at once, and no file changes while
   // read-only cursors and retained roots are live.
   void readOnlyShared(const fs::path& path)
   {
      create(path, 1);
      const auto before = fingerprint(path);
      {
         auto first        = Engine::database::open(path, psitri::open_mode::read_only, config());
         auto firstReader  = first->start_read_session();
         std::vector<Engine::cursor> cursors;
         for (unsigned root = 0; root < rootCount; ++root)
         {
            cursors.push_back(firstReader->create_cursor(root));
            require(cursors.back().lower_bound(std::string_view{}), "read-only root is empty");
         }
         std::shared_ptr<Engine::database> second;
         try
         {
            second = Engine::database::open(path, psitri::open_mode::read_only, config());
         }
         catch (const std::exception& error)
         {
            throw std::runtime_error(std::string("second concurrent read-only open failed: ") +
                                     error.what());
         }
         verifyGeneration(*second, 1);
         verifyGeneration(*first, 1);
         const auto live = difference(before, fingerprint(path));
         require(live.empty(), "live read-only sessions modified:" + live);
         bool writerRefused = false;
         try
         {
            Engine::database::open(path, psitri::open_mode::open_existing, config());
         }
         catch (const std::exception&)
         {
            writerRefused = true;
         }
         require(writerRefused, "a writer opened the store while read-only openers held it");
      }
      const auto closed = difference(before, fingerprint(path));
      require(closed.empty(), "read-only sessions modified:" + closed);
   }

   // Finding 2: a store on read-only media or without write permission.
   void readOnlyMedia(const fs::path& path)
   {
      create(path, 1);
      if (::geteuid() == 0)
      {
         std::cout << "SKIP read-only-media permission check: running as root\n";
         return;
      }
      auto protect = [&](bool writable)
      {
         std::vector<fs::path> entries{path};
         for (const auto& entry : fs::recursive_directory_iterator(path))
            entries.push_back(entry.path());
         for (const auto& entry : entries)
            fs::permissions(entry, fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write,
                            writable ? fs::perm_options::add : fs::perm_options::remove);
         if (writable)
            fs::permissions(path, fs::perms::group_write | fs::perms::others_write,
                            fs::perm_options::remove);
      };
      protect(false);
      std::string failure;
      try
      {
         auto database = Engine::database::open(path, psitri::open_mode::read_only, config());
         verifyGeneration(*database, 1);
      }
      catch (const std::exception& error)
      {
         failure = error.what();
      }
      protect(true);
      require(failure.empty(), "read-only open of a non-writable store failed: " + failure);
   }

   const std::vector<std::pair<std::string, void (*)(const fs::path&)>> scenarios = {
       {"crash-loop-reclaims", crashLoop},
       {"usable-after-recovery", usableAfterRecovery},
       {"read-only-refuses-unclean", readOnlyRefusesUnclean},
       {"power-loss-stale-control", [](const fs::path& p) { powerLoss(p, false, true); }},
       {"power-loss-missing-control", [](const fs::path& p) { powerLoss(p, true, true); }},
       {"stale-control-same-boot", [](const fs::path& p) { powerLoss(p, false, false); }},
       {"missing-control-same-boot", [](const fs::path& p) { powerLoss(p, true, false); }},
       {"power-loss-torn-tail", tornTail},
       {"rebuild-is-exact", rebuildIsExact},
       {"non-durable-stays-light", nonDurableStaysLight},
       {"read-only-shared", readOnlyShared},
       {"read-only-media", readOnlyMedia},
   };
}  // namespace

int main(int argc, char** argv)
{
   if (argc < 2)
   {
      std::cerr << "usage: recovery_contract <workdir> [scenario]\n";
      return 2;
   }
   self = fs::absolute(argv[0]).string();
   const fs::path work = fs::absolute(argv[1]);
   try
   {
      if (argc >= 6 && std::string(argv[2]) == "writer-child")
         return writerChild(work, std::stoul(argv[3]), std::stoul(argv[4]),
                            std::string(argv[5]) == "pause");
      if (argc == 3)
      {
         for (const auto& [name, run] : scenarios)
            if (name == argv[2])
            {
               run(work / name);
               return 0;
            }
         std::cerr << "unknown scenario " << argv[2] << '\n';
         return 2;
      }
   }
   catch (const std::exception& error)
   {
      std::cout << std::flush;
      std::cerr << "FAILED: " << error.what() << '\n';
      return 1;
   }
   fs::create_directories(work);
   unsigned failures = 0;
   for (const auto& [name, run] : scenarios)
   {
      std::cout << "=== " << name << '\n' << std::flush;
      const pid_t pid = ::fork();
      if (pid == 0)
      {
         ::execl(self.c_str(), self.c_str(), work.c_str(), name.c_str(), static_cast<char*>(nullptr));
         std::_Exit(96);
      }
      int status = 0;
      ::waitpid(pid, &status, 0);
      const bool passed = WIFEXITED(status) && WEXITSTATUS(status) == 0;
      failures += !passed;
      std::cout << (passed ? "PASS " : "FAIL ") << name;
      if (WIFSIGNALED(status))
         std::cout << " (terminated by signal " << WTERMSIG(status) << ')';
      else if (!passed)
         std::cout << " (exit " << WEXITSTATUS(status) << ')';
      std::cout << '\n' << std::flush;
   }
   return failures ? 1 : 0;
}

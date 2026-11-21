#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <psitri/engines/big.hpp>
#include <sal/block_allocator.hpp>
#include <sal/mapping.hpp>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{
   enum class Operation
   {
      none,
      msync,
      fsync,
      full
   };
   std::atomic<bool>     armed{};
   std::atomic<unsigned> calls{};
   Operation             selected  = Operation::none;
   unsigned              target    = 0;
   bool                  permanent = false;
   int                   failure   = EIO;
   pthread_t             selectedThread{};
   bool                  inject(Operation operation)
   {
      if (!armed.load(std::memory_order_acquire) || operation != selected)
         return false;
      if (!pthread_equal(pthread_self(), selectedThread))
         return false;
      const auto call = ++calls;
      if (call == target || (permanent && call > target))
      {
         errno = failure;
         return true;
      }
      return false;
   }
   template <class F>
   F resolve(const char* name)
   {
      auto function = reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
      if (!function)
         std::_Exit(98);
      return function;
   }
   void arm(const std::string& mode, const std::string& operation, unsigned number)
   {
      selected       = operation == "msync"   ? Operation::msync
                       : operation == "fsync" ? Operation::fsync
                                              : Operation::full;
      permanent      = mode.starts_with("error");
      failure        = mode == "eintr"          ? EINTR
                       : mode == "error-enospc" ? ENOSPC
                       : mode == "error-erofs"  ? EROFS
                       : mode == "error-ebadf"  ? EBADF
                                                : EIO;
      target         = number;
      calls          = 0;
      selectedThread = pthread_self();
      armed.store(true, std::memory_order_release);
   }
   void ack()
   {
      armed = false;
      std::cout << "ACK calls=" << calls.load() << '\n' << std::flush;
   }
   void require(bool value, const char* message)
   {
      if (!value)
         throw std::runtime_error(message);
   }
   using Engine = psitri::big_engine;
   Engine::config config(bool full = false)
   {
      Engine::config result;
      result.sync_mode                = full ? psitri::sync_mode::full : psitri::sync_mode::fsync;
      result.enable_read_cache        = false;
      result.max_pinned_cache_size_mb = 0;
      result.max_database_size        = 512LL * 1024 * 1024;
      return result;
   }
}  // namespace
extern "C" int msync(void* address, std::size_t size, int flags)
{
   static auto real = resolve<int (*)(void*, std::size_t, int)>("msync");
   if (inject(Operation::msync))
      return -1;
   return real(address, size, flags);
}
extern "C" int fsync(int fd)
{
   static auto real = resolve<int (*)(int)>("fsync");
   if (inject(Operation::fsync))
      return -1;
   return real(fd);
}
#ifdef __APPLE__
extern "C" int fcntl(int fd, int command, ...)
{
   static auto real = resolve<int (*)(int, int, ...)>("fcntl");
   if (command == F_FULLFSYNC)
   {
      if (inject(Operation::full))
         return -1;
      return real(fd, command);
   }
   if (command == F_GETFD || command == F_GETFL || command == F_GETOWN)
      return real(fd, command);
   va_list arguments;
   va_start(arguments, command);
   if (command == F_SETFD || command == F_SETFL || command == F_SETOWN || command == F_DUPFD ||
       command == F_NOCACHE)
   {
      const auto arg = va_arg(arguments, int);
      va_end(arguments);
      return real(fd, command, arg);
   }
   if (command == F_GETLK || command == F_SETLK || command == F_SETLKW || command == F_GETPATH)
   {
      const auto arg = va_arg(arguments, void*);
      va_end(arguments);
      return real(fd, command, arg);
   }
   va_end(arguments);
   std::_Exit(97);  // unsupported fixture command; never guess its variadic type
}
#endif
int main(int argc, char** argv)
{
   try
   {
      if (argc == 3 && std::string(argv[1]) == "reopen")
      {
         auto       db = Engine::database::open(argv[2], psitri::open_mode::read_only, config());
         auto       session    = db->start_read_session();
         auto       cursor     = session->create_cursor(0);
         const auto generation = cursor.get<std::string>("generation");
         require(generation && (*generation == "old" || *generation == "new"),
                 "missing or torn root generation");
         const auto expected = *generation == "old" ? 'o' : 'n';
         require(cursor.get<std::string>("payload") == std::string(65536, expected),
                 "mixed root payload");
         for (unsigned i = 0; i < 100; ++i)
            require(
                cursor.get<std::string>("key" + std::to_string(i)) == std::string(128, expected),
                "mixed root entries");
         std::cout << "CONSISTENT generation=" << *generation << '\n';
         return 0;
      }
      if (argc != 7)
         return 2;
      const std::string           kind = argv[1], mode = argv[3], operation = argv[4];
      const auto                  number = unsigned(std::stoul(argv[5]));
      const bool                  full   = std::string(argv[6]) == "full";
      const std::filesystem::path path   = argv[2];
      std::filesystem::create_directories(path);
      if (kind == "mapping")
      {
         sal::mapping map(path / "mapping", sal::access_mode::read_write);
         map.resize(16384);
         std::memset(map.data(), 'x', map.size());
         arm(mode, operation, number);
         map.sync(full ? sal::sync_type::full : sal::sync_type::fsync);
         ack();
         return 0;
      }
      if (kind == "readonly-truncate")
      {
         {
            sal::block_allocator block(path / "blocks", 16384, 16);
            block.resize(1);
            *block.get<char>(sal::block_allocator::block_number(0)) = 'x';
            block.fsync();
         }
         const auto           size = std::filesystem::file_size(path / "blocks");
         sal::block_allocator readonly(path / "blocks", 16384, 16, sal::access_mode::read_only);
         bool                 rejected = false;
         try
         {
            readonly.truncate(0);
         }
         catch (const std::runtime_error&)
         {
            rejected = true;
         }
         require(rejected && readonly.num_blocks() == 1 &&
                     std::filesystem::file_size(path / "blocks") == size &&
                     *readonly.get<char>(sal::block_allocator::block_number(0)) == 'x',
                 "read-only truncate changed mapping or file");
         ack();
         return 0;
      }
      if (kind == "block" || kind == "fd0" || kind == "block-grow")
      {
         if (kind == "fd0")
            require(close(STDIN_FILENO) == 0, "close stdin failed");
         {
            sal::block_allocator block(path / "blocks", 16384, 16);
            block.resize(1);
            *block.get<char>(sal::block_allocator::block_number(0)) = 'x';
            if (kind == "block-grow")
            {
               for (unsigned i = 2; i <= 8; ++i)
               {
                  block.truncate(i);
                  require(block.num_blocks() == i &&
                              *block.get<char>(sal::block_allocator::block_number(0)) == 'x',
                          "truncate growth lost block count or prefix");
               }
               block.truncate(1);
               require(block.num_blocks() == 1 &&
                           *block.get<char>(sal::block_allocator::block_number(0)) == 'x',
                       "truncate shrink lost prefix");
            }
            arm(mode, operation, number);
            require(block.fsync(full), "block fsync reported false");
            ack();
         }
         if (kind == "fd0")
            require(fcntl(STDIN_FILENO, F_GETFD) == -1 && errno == EBADF,
                    "descriptor zero leaked at destruction");
         return 0;
      }
      auto engineWork = [&]
      {
         auto db     = Engine::database::open(path, psitri::open_mode::create_only, config(full));
         auto writer = db->start_write_session();
         {
            auto tx = writer->start_transaction(0);
            tx.upsert("generation", "old");
            tx.upsert("payload", std::string(65536, 'o'));
            for (unsigned i = 0; i < 100; ++i)
               tx.upsert("key" + std::to_string(i), std::string(128, 'o'));
            tx.commit();
         }
         auto tx = writer->start_transaction(0);
         tx.upsert("generation", "new");
         tx.upsert("payload", std::string(65536, 'n'));
         for (unsigned i = 0; i < 100; ++i)
            tx.upsert("key" + std::to_string(i), std::string(128, 'n'));
         arm(mode, operation, number);
         tx.commit();
         ack();
         // A failure must never reach acknowledgement or allow later work.
         auto next = writer->start_transaction(0);
         next.upsert("after_ack", "yes");
         next.commit();
      };
      if (kind == "engine-thread")
      {
         std::exception_ptr failure;
         std::thread        thread(
             [&]
             {
                try
                {
                   engineWork();
                }
                catch (...)
                {
                   failure = std::current_exception();
                }
             });
         thread.join();
         if (failure)
            std::rethrow_exception(failure);
      }
      else
         engineWork();
      return 0;
   }
   catch (const std::exception& error)
   {
      std::cerr << error.what() << '\n';
      return 3;
   }
}

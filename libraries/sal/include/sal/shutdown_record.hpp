#pragma once
// Whether the last read-write owner of a store closed it, and if it did not,
// whether the operating system has restarted since.
//
// The commit path flushes segment data and roots only. Control blocks, implicit
// refcounts, and allocator state are rebuilt by recovery, so an engine must know
// when to run it:
//
//   clean            the owner flushed everything and closed; nothing to do.
//   process_stopped  the owner died but this boot's page cache still holds every
//                    byte it wrote; only reference counts and allocator
//                    bookkeeping are stale.
//   system_stopped   the system restarted (power loss, kernel crash), or there is
//                    no usable record. Unflushed files hold arbitrary old state.
//
// The record is a small file written with write() and never mapped, so marking
// a store open is durable before the first mutation, independent of the mapped
// state it describes.
#include <sal/config.hpp>

#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace sal
{
   enum class shutdown_state
   {
      clean,
      process_stopped,
      system_stopped
   };

   class shutdown_record
   {
     public:
      static constexpr const char* filename = "shutdown";

      static shutdown_state read(const std::filesystem::path& dir)
      {
         std::ifstream file(dir / filename, std::ios::binary);
         if (!file)
            return shutdown_state::system_stopped;
         char        buffer[record_size + 1] = {};
         file.read(buffer, record_size);
         const std::string text(buffer, static_cast<std::size_t>(file.gcount()));
         if (text.size() != record_size || !text.starts_with(magic))
            return shutdown_state::system_stopped;
         const auto state = field(text, "state=");
         if (state == "clean")
            return shutdown_state::clean;
         const auto boot = boot_session();
         if (state == "open" && !boot.empty() && field(text, "boot=") == boot)
            return shutdown_state::process_stopped;
         return shutdown_state::system_stopped;
      }

      /// Record that a writer owns the store. Throws if the record cannot be
      /// written: without it a later unclean stop would go undetected.
      static void mark_open(const std::filesystem::path& dir, sync_type sync)
      {
         if (!write(dir, "open", sync))
            throw std::system_error(errno, std::generic_category(),
                                    "cannot record store ownership in " +
                                        (dir / filename).generic_string());
      }

      /// Record a clean close. Call only after every store file is flushed to
      /// the level `sync` promises. On failure the store stays marked open and
      /// the next writer recovers it.
      static void mark_clean(const std::filesystem::path& dir, sync_type sync) noexcept
      {
         try
         {
            (void)write(dir, "clean", sync);
         }
         catch (...)
         {
         }
      }

      /// Identifies the running kernel instance; empty when the platform offers none.
      static std::string boot_session()
      {
#if defined(__APPLE__)
         char        buffer[64] = {};
         std::size_t size       = sizeof(buffer) - 1;
         if (::sysctlbyname("kern.bootsessionuuid", buffer, &size, nullptr, 0) != 0)
            return {};
         return trim(buffer);
#elif defined(__linux__)
         std::ifstream file("/proc/sys/kernel/random/boot_id");
         std::string   id;
         std::getline(file, id);
         return trim(id);
#else
         return {};
#endif
      }

     private:
      static constexpr std::string_view magic       = "sal-shutdown-record-v1\n";
      static constexpr std::size_t      record_size = 128;  // one sector: never torn

      static std::string trim(std::string value)
      {
         while (!value.empty() && (value.back() == '\n' || value.back() == ' ' || value.back() == '\0'))
            value.pop_back();
         return value.size() <= 64 ? value : std::string{};
      }
      static std::string field(const std::string& text, std::string_view name)
      {
         const auto begin = text.find(name);
         if (begin == std::string::npos)
            return {};
         const auto end = text.find('\n', begin);
         return text.substr(begin + name.size(), end - begin - name.size());
      }
      static bool flush(int descriptor, sync_type sync)
      {
         if (sync < sync_type::fsync)
            return true;
         int result;
         do
         {
#ifdef __APPLE__
            result = sync == sync_type::full ? ::fcntl(descriptor, F_FULLFSYNC) : ::fsync(descriptor);
#else
            result = ::fsync(descriptor);
#endif
         } while (result != 0 && errno == EINTR);
         return result == 0;
      }
      static bool write(const std::filesystem::path& dir, std::string_view state, sync_type sync)
      {
         std::string record(magic);
         record += "state=" + std::string(state) + "\nboot=" + boot_session() + "\n";
         record.resize(record_size, '\n');

         const auto path    = dir / filename;
         const bool created = !std::filesystem::exists(path);
         const int  fd      = ::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
         if (fd < 0)
            return false;
         std::size_t offset = 0;
         bool        ok     = true;
         while (ok && offset < record.size())
         {
            const auto written = ::pwrite(fd, record.data() + offset, record.size() - offset,
                                          static_cast<off_t>(offset));
            if (written < 0 && errno == EINTR)
               continue;
            ok = written > 0;
            if (ok)
               offset += static_cast<std::size_t>(written);
         }
         ok              = ok && flush(fd, sync);
         const int error = errno;
         ::close(fd);
         if (ok && created)
         {
            const int directory = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
            if (directory >= 0)
            {
               ok = flush(directory, sync);
               ::close(directory);
            }
         }
         if (!ok)
            errno = error;
         return ok;
      }
   };
}  // namespace sal

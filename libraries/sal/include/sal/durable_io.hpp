#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace sal
{
   // A flush failure can occur after a root or its recovery record has become
   // visible to the kernel. It is not a recoverable transaction rejection:
   // acknowledging it or continuing to write would claim unknown durability.
   inline constexpr int durable_io_failure_exit_code = 74;

   [[noreturn]] inline void fail_durable_io(const char* operation, int error) noexcept
   {
      char       message[256];
      const auto size = std::snprintf(message, sizeof(message),
                                      "SAL durability failure: %s errno=%d; store durability is "
                                      "unknown, terminating without acknowledgement\n",
                                      operation, error);
      if (size > 0)
      {
         const auto length = static_cast<std::size_t>(size) < sizeof(message)
                                 ? static_cast<std::size_t>(size)
                                 : sizeof(message) - 1;
         // Best-effort diagnostic, with no C++ logging locks or exit handlers.
         // Never unwind through a partially published durable transaction.
         [[maybe_unused]] const auto written = ::write(STDERR_FILENO, message, length);
      }
      std::_Exit(durable_io_failure_exit_code);
   }

   template <class Function>
   inline void checked_durable_io(const char* operation, Function&& function) noexcept
   {
      int result;
      do
      {
         result = function();
      } while (result != 0 && errno == EINTR);
      if (result != 0)
         fail_durable_io(operation, errno);
   }

   inline void durable_msync(void* address, std::size_t size, int flags) noexcept
   {
      checked_durable_io("msync", [&] { return ::msync(address, size, flags); });
   }

   inline void durable_fsync(int descriptor, bool full = false) noexcept
   {
#ifdef __APPLE__
      if (full)
      {
         checked_durable_io("F_FULLFSYNC", [&] { return ::fcntl(descriptor, F_FULLFSYNC); });
         return;
      }
#else
      (void)full;
#endif
      checked_durable_io("fsync", [&] { return ::fsync(descriptor); });
   }
}  // namespace sal

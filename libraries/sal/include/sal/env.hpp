#pragma once
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef __APPLE__
#  include <crt_externs.h>
#  define PSITRI_ENVIRON (*_NSGetEnviron())
#else
#  include <unistd.h>
#  define PSITRI_ENVIRON environ
#endif

namespace sal
{
   // ───────────────────────────────────────────────────────────────────────────
   // Process-wide environment snapshot, read ONCE at first use.
   //
   // getenv() takes the C library's environ lock and linear-scans environ on EVERY
   // call; probed in a hot path it cost ~20% (__findenv_locked) of Silkworm sync.
   // env snapshots the whole environment into an immutable static structure the
   // first time it's touched, so every later lookup is a lock-free scan of our own
   // copy. ALL env reads — runtime config AND diagnostics — go through here; there
   // is no getenv() anywhere else in the engine.
   //
   // The snapshot allocates once at startup (not a hot path), and is immutable
   // afterward (so it's race-free across threads, and setenv() after first use is
   // intentionally not observed — config/diagnostic flags are set before main).
   // ───────────────────────────────────────────────────────────────────────────
   class env
   {
     public:
      /// Raw value of `name`, or nullptr if unset (like getenv, but cached).
      static const char* get(const char* name) noexcept { return cache().lookup(name); }

      /// True if `name` is set to a non-empty value.
      static bool flag(const char* name) noexcept
      {
         const char* v = get(name);
         return v && *v;
      }

      /// Integer value of `name`, or `def` if unset/empty.
      static long num(const char* name, long def) noexcept
      {
         const char* v = get(name);
         return (v && *v) ? std::atol(v) : def;
      }

     private:
      struct snapshot
      {
         std::vector<std::string> kv;  // "KEY=VALUE" copies, owned (startup-only alloc)

         snapshot()
         {
            for (char** e = PSITRI_ENVIRON; e && *e; ++e)
               kv.emplace_back(*e);
         }

         const char* lookup(const char* name) const noexcept
         {
            const size_t n = std::strlen(name);
            for (const std::string& entry : kv)
               if (entry.size() > n && entry[n] == '=' &&
                   std::memcmp(entry.data(), name, n) == 0)
                  return entry.c_str() + n + 1;
            return nullptr;
         }
      };

      static const snapshot& cache() noexcept
      {
         static const snapshot s;  // thread-safe one-time init (C++11)
         return s;
      }
   };
}  // namespace sal

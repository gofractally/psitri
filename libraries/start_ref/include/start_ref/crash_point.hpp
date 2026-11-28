// Crash-injection points for the process-death tests (tests/api/small_engine_crash.cpp).
//
// The Small Engine's failure model is the death of the process at any
// instruction, with the operating system keeping every byte already written
// to the shared mappings. START_REF_CRASH_POINT("name") marks a step at which
// the state in the files changes shape; a test build installs a hook that
// ends the process there.
//
// Only a translation unit compiled with START_REF_CRASH_POINTS defined gets
// any code from these: everywhere else the macro expands to nothing, so a
// release build carries no test, no load and no branch for them.
#pragma once

#ifdef START_REF_CRASH_POINTS
#include <atomic>

namespace start_ref::testing
{
   // Called with the point's name each time one is passed, from whatever
   // thread passes it. Null: nothing happens.
   using crash_hook_fn = void (*)(const char* point);
   inline std::atomic<crash_hook_fn> crash_hook{nullptr};
}  // namespace start_ref::testing

#define START_REF_CRASH_POINT(name)                                                        \
   do                                                                                      \
   {                                                                                       \
      if (auto start_ref_crash_hook_ =                                                     \
              ::start_ref::testing::crash_hook.load(std::memory_order_relaxed))            \
         start_ref_crash_hook_(name);                                                      \
   } while (0)
#else
#define START_REF_CRASH_POINT(name) ((void)0)
#endif

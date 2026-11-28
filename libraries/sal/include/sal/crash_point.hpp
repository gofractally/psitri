// Crash-injection points for the process-death tests (tests/api/large_engine_crash.cpp).
//
// The failure model of the Large Engine and of this allocator is the death of
// the PROCESS at any instruction, in any thread, with the operating system
// keeping every byte the process already wrote to its shared mappings.
// SAL_CRASH_POINT("area:step") marks a step at which the state in the files
// changes shape; a test build installs a hook that ends the process there.
//
// Only a translation unit compiled with SAL_CRASH_POINTS defined gets any code
// from these: everywhere else the macro expands to nothing, so a release build
// carries no test, no load and no branch for them. A binary must not mix
// translation units compiled with and without it (the points sit in inline
// functions); the test binary compiles the allocator's sources itself.
#pragma once

#ifdef SAL_CRASH_POINTS
#include <atomic>

namespace sal::testing
{
   // Called with the point's name each time one is passed, from whatever
   // thread passes it. Null: nothing happens.
   using crash_hook_fn = void (*)(const char* point);
   inline std::atomic<crash_hook_fn> crash_hook{nullptr};
   // For the test that shows what an open did before it repaired the segment provider's
   // state (gofractally/psitri#45): true makes the allocator skip that repair.
   inline std::atomic<bool> skip_provider_repair{false};
}  // namespace sal::testing

#define SAL_CRASH_POINT(name)                                                              \
   do                                                                                      \
   {                                                                                       \
      if (auto sal_crash_hook_ = ::sal::testing::crash_hook.load(std::memory_order_relaxed)) \
         sal_crash_hook_(name);                                                            \
   } while (0)
#else
#define SAL_CRASH_POINT(name) ((void)0)
#endif

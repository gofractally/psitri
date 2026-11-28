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
//
// The markers sit in inline functions, so every translation unit of one
// program must agree on START_REF_CRASH_POINTS: two that disagree would
// define the same functions with different bodies. The classes that carry
// markers are therefore declared with START_REF_CRASH_ABI, which in a marker
// build puts a tag into the mangled names of their members (and of every
// template instantiated over them): a unit built with markers and one built
// without then have different symbols instead of silently sharing one body.
#pragma once

#ifdef START_REF_CRASH_POINTS
#include <atomic>
#include <vector>

#define START_REF_CRASH_ABI [[gnu::abi_tag("crash_points")]]

namespace start_ref::testing
{
   // Called with the point's name each time one is passed, from whatever
   // thread passes it. Null: nothing happens.
   using crash_hook_fn = void (*)(const char* point);
   inline std::atomic<crash_hook_fn> crash_hook{nullptr};

   // Every marker compiled into this program, whether or not it has been
   // passed: each use of the macro registers its name before main runs. A
   // test can therefore require a death at every marker that exists, and a
   // marker added to the engine cannot be left out of it.
   inline std::vector<const char*>& markers()
   {
      static std::vector<const char*> all;
      return all;
   }
   inline int register_marker(const char* name)
   {
      markers().push_back(name);
      return 0;
   }
   template <class Name>
   struct registered
   {
      static inline const int id = register_marker(Name{}());
   };
}  // namespace start_ref::testing

#define START_REF_CRASH_POINT(name)                                                        \
   do                                                                                      \
   {                                                                                       \
      (void)&::start_ref::testing::registered<decltype([] { return name; })>::id;          \
      if (auto start_ref_crash_hook_ =                                                     \
              ::start_ref::testing::crash_hook.load(std::memory_order_relaxed))            \
         start_ref_crash_hook_(name);                                                      \
   } while (0)
#else
#define START_REF_CRASH_ABI
#define START_REF_CRASH_POINT(name) ((void)0)
#endif

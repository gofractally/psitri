#pragma once
// Typed-value lifecycle registry (the LOCKED adb API), engine-side and
// sal-free (the hard rule: no sal changes for an engine feature). EVERY
// stored value is a [u16 LE typenum][payload] envelope; typenum 0 = raw
// bytes (the null-hook builtin). Registered ids >= 1 carry retain/release
// callbacks; the node lifecycle (leaf_page clone/destroy, etc.) fires them
// as value bytes are physically duplicated or die — INCLUDING on sal's GC
// threads, so the callbacks must be thread-safe and non-blocking.
//
// The registry is a process-global keyed by typenum: typenums are globally
// stable by design (the psio schema-persistence contract), and the
// callbacks are pure functions of (typenum, payload), so one table serves
// every database in the process. Registration happens at startup, before any
// value of that type is stored; the store's happens-before covers the GC
// read.
#include <cstdint>
#include <cstring>
#include <string_view>

namespace sal_art_ref
{
   inline constexpr uint16_t kRawType = 0;

   struct value_type_entry
   {
      void (*retain)(void* session, const char* pod, size_t n)  = nullptr;
      void (*release)(void* session, const char* pod, size_t n) = nullptr;
   };

   // 64Ki slots: one per possible u16 typenum. Trivial, zero-initialized.
   inline value_type_entry g_value_types[65536] = {};

   inline uint16_t env_typenum(std::string_view slot) noexcept
   {
      if (slot.size() < 2)
         return kRawType;
      uint16_t t;
      std::memcpy(&t, slot.data(), 2);
      return t;
   }

   // Fire retain/release for ONE stored value slot (envelope included). A raw
   // (typenum 0) slot or an unregistered typenum is a no-op. session may be
   // null (GC-thread frees have no civilian session).
   inline void fire_value_retain(std::string_view slot, void* session = nullptr) noexcept
   {
      const uint16_t t = env_typenum(slot);
      if (t == kRawType)
         return;
      if (auto f = g_value_types[t].retain)
         f(session, slot.data() + 2, slot.size() - 2);
   }
   inline void fire_value_release(std::string_view slot, void* session = nullptr) noexcept
   {
      const uint16_t t = env_typenum(slot);
      if (t == kRawType)
         return;
      if (auto f = g_value_types[t].release)
         f(session, slot.data() + 2, slot.size() - 2);
   }
}  // namespace sal_art_ref

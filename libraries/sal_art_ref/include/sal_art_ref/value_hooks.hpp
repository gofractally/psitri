#pragma once
// Typed-value lifecycle registry (the LOCKED adb API), engine-side and
// sal-free (the hard rule: no sal changes for an engine feature).
//
// A stored value is either the caller's bytes, untouched and never inspected
// (inline_value::vt_bytes), or a TYPED value (inline_value::vt_typed): a type
// number and a small trivially copyable token, written by the engine as
// [u16 LE typenum][token]. The kind is the record's own metadata; the type
// number is never part of, and never read from, a caller's bytes.
//
// Registered type numbers carry retain/release callbacks. The node lifecycle
// fires them as a typed value's bytes are physically duplicated (a leaf page
// is cloned or split) or die (overwritten, removed, the page or the external
// record destroyed) — INCLUDING on sal's background threads, so the callbacks
// must be thread-safe and non-blocking.
//
// The registry is a process-global keyed by typenum: typenums are globally
// stable by design (the psio schema-persistence contract), and the
// callbacks are pure functions of (typenum, token), so one table serves
// every database in the process. Registration happens at startup, before any
// value of that type is stored; the store's happens-before covers the
// background read.
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sal_art_ref
{
   /// Type number of plain bytes. Never stored: a raw value has no type number.
   inline constexpr uint16_t kRawType = 0;
   /// Largest token of a typed value. Small enough that a typed value is never
   /// a chunked large value.
   inline constexpr uint32_t max_typed_token_size = 256;

   /// A value was read as something it is not: a typed value as bytes, bytes
   /// as a typed value, or one type as another.
   struct invalid_type : std::runtime_error
   {
      invalid_type(uint16_t stored, uint16_t asked)
          : std::runtime_error("sal_art_ref: value is type " + std::to_string(stored) +
                               ", read as type " + std::to_string(asked)),
            stored_type(stored),
            asked_type(asked)
      {
      }
      uint16_t stored_type;
      uint16_t asked_type;
   };

   struct value_type_entry
   {
      void (*retain)(void* session, const char* pod, size_t n)  = nullptr;
      void (*release)(void* session, const char* pod, size_t n) = nullptr;
   };

   // 64Ki slots: one per possible u16 typenum. Trivial, zero-initialized.
   inline value_type_entry g_value_types[65536] = {};

   /// The type number of a TYPED value's stored bytes ([u16 typenum][token]).
   /// Only ever applied to bytes the engine wrote for a vt_typed record.
   inline uint16_t typed_typenum(std::string_view stored) noexcept
   {
      uint16_t t = kRawType;
      if (stored.size() >= 2)
         std::memcpy(&t, stored.data(), 2);
      return t;
   }
   inline std::string_view typed_token(std::string_view stored) noexcept
   {
      return stored.size() >= 2 ? stored.substr(2) : std::string_view{};
   }

   // Fire retain/release for ONE typed value's stored bytes. An unregistered
   // typenum is a no-op. session may be null (background frees have no
   // civilian session). Callers pass vt_typed records only.
   inline void fire_typed_retain(std::string_view stored, void* session = nullptr) noexcept
   {
      if (auto f = g_value_types[typed_typenum(stored)].retain)
         f(session, stored.data() + 2, stored.size() - 2);
   }
   inline void fire_typed_release(std::string_view stored, void* session = nullptr) noexcept
   {
      if (auto f = g_value_types[typed_typenum(stored)].release)
         f(session, stored.data() + 2, stored.size() - 2);
   }
}  // namespace sal_art_ref

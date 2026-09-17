#pragma once
#include <cstdint>
#include <sal_hybrid/alloc/env.hpp>  // sal_hybrid::env — the cached, lock-free env reader (no getenv at runtime)

// ─────────────────────────────────────────────────────────────────────────────
// Compile-gated diagnostics. Default OFF → zero code, zero env reads, zero work.
//
//   -DPSITRI_STATS       : lightweight perf counters (e.g. cow_counters). Reads
//                          yield 0 and every mutation inlines away when undefined.
//   -DPSITRI_DEBUG_BUILD : heavy diagnostics (op-logs, query logs, tree dumps,
//                          traces, invariant checks). Compiled out when undefined.
//
// Usage at a glance:
//   sal_hybrid::stat_counter c{};                 // counter field: ++c / c += n compile to
//                                          //   nothing under -USTATS (reads → 0).
//   PSITRI_DEBUG( stmt; stmt; )            // statement(s) present only in a debug
//                                          //   build — NO #ifdef at the call site.
//   if (sal_hybrid::env::flag("PG_OPS")) ...      // env reads are cached & lock-free; never
//                                          //   call getenv() directly (it locks +
//                                          //   scans environ on every call).
//
// Why PSITRI_DEBUG_BUILD (not PSITRI_DEBUG) is the build flag: PSITRI_DEBUG is the
// function-like wrapper macro below, so the enabling -D flag must use a distinct
// name (a macro cannot be both object-like and function-like).
//
// PSITRI_DEBUG(...) wraps STATEMENTS in otherwise-live functions. Whole debug-only
// SUBSYSTEMS (function/namespace-scope definitions, especially ones with top-level
// commas) still use a plain `#ifdef PSITRI_DEBUG_BUILD ... #endif` block.
// ─────────────────────────────────────────────────────────────────────────────

namespace sal_hybrid
{
#ifdef PSITRI_STATS
   using stat_counter = uint64_t;
#else
   // Zero-cost stand-in for a perf counter: every mutation inlines to nothing;
   // every read yields 0. Same call sites (`++x`, `x += n`, comparisons) compile
   // unchanged, so no per-increment macro is needed at the ~dozen hot-path sites.
   struct stat_counter
   {
      constexpr stat_counter(uint64_t = 0) noexcept {}
      constexpr stat_counter& operator++() noexcept { return *this; }
      constexpr stat_counter  operator++(int) noexcept { return *this; }
      constexpr stat_counter& operator+=(uint64_t) noexcept { return *this; }
      constexpr               operator uint64_t() const noexcept { return 0; }
   };
#endif
}  // namespace sal_hybrid

#ifdef PSITRI_DEBUG_BUILD
#  define PSITRI_DEBUG(...) __VA_ARGS__
#else
#  define PSITRI_DEBUG(...)
#endif

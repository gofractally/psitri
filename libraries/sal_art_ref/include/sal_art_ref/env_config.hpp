#pragma once
#include <cstdint>
#include <sal/env.hpp>  // cached, lock-free env reader (no getenv at runtime)

namespace sal_art_ref
{
   // Single-init snapshot of the sal_art_ref engine's environment knobs.
   //
   // Read ONCE at first use (the static-local in get() is initialized exactly
   // once, thread-safely, by the C++ runtime) and never re-read. Every engine
   // call site reads `env_config::get().field` — there is no scattered getenv,
   // and no hand-rolled `static const x = getenv(...)` to forget to make static.
   //
   // These are RUNTIME tunables (not diagnostics), so they are always present —
   // unlike the PSITRI_STATS / PSITRI_DEBUG gated machinery in <sal/diag.hpp>.
   struct env_config
   {
      // HEART_PAGE_BYTES — COW-unit leaf_page size (clamped 512..8192; u16 tail
      // offsets + the 0xFF00 extern-code floor keep it well below 64KB).
      uint32_t page_bytes = 2048;

      // HEART_PINNED_MB — RAM budget for pinned (mlock'd) cache segments, in MB.
      // 0 means "leave sal's runtime_config default" (8 GB).
      uint64_t pinned_mb = 0;

      // HEART_READ_CACHE — enable MFU read-cache promotion. When on, read-hot,
      // read-only objects are promoted by the compactor into mlock'd pinned
      // segments so the active working set stays resident — this is the lever for
      // swap + TLB pressure once the cold base outgrows RAM. Default OFF (the
      // historical hardcode), but now a real runtime toggle.
      bool read_cache = false;

      // HEART_READ_CACHE_SEC — MFU decay window in seconds (0 = sal default, 5h).
      uint64_t read_cache_window_sec = 0;

      static const env_config& get() noexcept
      {
         static const env_config c = load();
         return c;
      }

     private:
      static env_config load() noexcept
      {
         env_config  c;
         const long  pb = sal::env::num("HEART_PAGE_BYTES", 0);
         if (pb >= 512 && pb <= 8192)
            c.page_bytes = uint32_t(pb);
         const long pm = sal::env::num("HEART_PINNED_MB", 0);
         if (pm > 0)
            c.pinned_mb = uint64_t(pm);
         c.read_cache  = sal::env::flag("HEART_READ_CACHE");
         const long rw = sal::env::num("HEART_READ_CACHE_SEC", 0);
         if (rw > 0)
            c.read_cache_window_sec = uint64_t(rw);
         return c;
      }
   };
}  // namespace sal_art_ref

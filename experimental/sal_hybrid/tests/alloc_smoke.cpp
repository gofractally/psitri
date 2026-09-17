// Standalone smoke test for the SAL allocator copied into namespace sal_hybrid
// (Phase 1: near-verbatim copy, version narrowed 48-bit -> 32-bit).
//
// Proves the copied allocator subsystem is self-consistent under `sal_hybrid`:
//   open allocator -> register a CB-backed object type -> start bg threads ->
//   create a session -> smart_alloc -> read it back via get_ref -> release ->
//   confirm the live-object count returns to baseline.
//
// No Catch2 — just a main() that returns 0 on success so CTest can run it.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <sal_hybrid/alloc/allocator_session_impl.hpp>
#include <sal_hybrid/alloc/atomic_ptr.hpp>
#include <sal_hybrid/alloc/smart_ptr_impl.hpp>

namespace fs = std::filesystem;

namespace
{
   int g_failures = 0;

#define SMOKE_CHECK(cond)                                                  \
   do                                                                      \
   {                                                                       \
      if (!(cond))                                                         \
      {                                                                    \
         std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
         ++g_failures;                                                     \
      }                                                                    \
   } while (0)

   // Minimal user object: one 128-byte allocation grain, a tag payload.
   struct probe : sal_hybrid::alloc_header
   {
      static constexpr sal_hybrid::header_type type_id = sal_hybrid::header_type::start_user_type;
      uint64_t                           tag;
      static uint32_t alloc_size(uint64_t) noexcept { return sal_hybrid::cacheline_size; }
      probe(uint32_t asize, sal_hybrid::ptr_address_seq seq, uint64_t t) noexcept
          : alloc_header(asize, type_id, seq), tag(t)
      {
      }
   };

   uint64_t live_count(const sal_hybrid::allocator_session_ptr& ses)
   {
      uint64_t n = 0;
      ses->for_each_live_object([&](auto, auto, auto*) { ++n; });
      return n;
   }

   fs::path temp_dir(std::string name)
   {
      auto path = fs::temp_directory_path() / name;
      fs::remove_all(path);
      fs::create_directories(path);
      return path;
   }
}  // namespace

int main()
{
   // The 32-bit version narrow, asserted at the source of truth.
   static_assert(sizeof(sal_hybrid::version_t) == 4, "sal_hybrid version_t must be 32-bit");
   static_assert(sizeof(sal_hybrid::ptr_address_seq) == 8, "sal_hybrid ptr_address_seq = address(4)+version(4)");
   static_assert(sizeof(sal_hybrid::alloc_header) == 14, "sal_hybrid alloc_header shrinks with 32-bit version");

   auto path = temp_dir("sal_hybrid_alloc_smoke");

   {
      sal_hybrid::allocator alloc(path, sal_hybrid::runtime_config{}, /*start_threads_now=*/false);
      alloc.register_type_ops<probe>();  // ops BEFORE threads (sal contract)
      alloc.start_background_threads();  // provider ON: get_new_segment waits on it otherwise

      auto       ses  = alloc.create_session();
      const auto base = live_count(ses);

      // alloc a CB-backed object
      auto p   = ses->smart_alloc<probe>(uint64_t{0xABCDEF});
      auto adr = p.address();
      SMOKE_CHECK(adr != sal_hybrid::null_ptr_address);
      SMOKE_CHECK(ses->get_ref<probe>(adr).ref() == 1);

      // get it: payload round-trips, and the allocation sequence (the 32-bit version)
      // is non-degenerate.
      {
         auto r = ses->get_ref<probe>(adr);
         SMOKE_CHECK(r->tag == 0xABCDEF);
         SMOKE_CHECK(r->address() == adr);
         // sequence() returns the 32-bit alloc seq widened to uint64_t; must fit in 32 bits.
         SMOKE_CHECK(r->sequence() <= 0xFFFFFFFFull);
      }

      SMOKE_CHECK(live_count(ses) == base + 1);

      // free it: dropping the only smart_ptr returns the live count to baseline
      p.release();
      SMOKE_CHECK(live_count(ses) == base);
   }

   fs::remove_all(path);

   if (g_failures == 0)
   {
      std::printf("sal_hybrid-alloc-smoke: OK (alloc/get/free round-trip; 32-bit version)\n");
      return 0;
   }
   std::fprintf(stderr, "sal_hybrid-alloc-smoke: %d FAILURES\n", g_failures);
   return 1;
}

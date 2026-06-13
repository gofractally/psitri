#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>

#include <sal_art_ref/compact_addresses.hpp>
#include <sal_art_ref/smart_address.hpp>

using sal_art_ref::compact_addresses;
using sal_art_ref::smart_address;
using sal::ptr_address;

namespace
{
   // Helper: build a node-pointer branch with a given cacheline group and slot.
   smart_address addr(uint32_t cacheline, uint32_t slot)
   {
      return smart_address{(cacheline << 4) | (slot & 0xF)};
   }
}  // namespace

TEST_CASE("compact_addresses append + read back", "[compact_addresses]")
{
   compact_addresses<128> ca;
   ca.init();
   REQUIRE(ca.size() == 0);
   REQUIRE(ca.base_count() == 0);

   ca.insert(addr(0x10, 1), 0);
   ca.insert(addr(0x10, 2), 1);
   ca.insert(addr(0x20, 7), 2);
   REQUIRE(ca.size() == 3);

   REQUIRE(ca.at(0) == addr(0x10, 1));
   REQUIRE(ca.at(1) == addr(0x10, 2));
   REQUIRE(ca.at(2) == addr(0x20, 7));

   // Two distinct cachelines -> exactly two bases (cacheline dedup worked).
   REQUIRE(ca.base_count() == 2);
}

TEST_CASE("compact_addresses insert in the middle shifts positions", "[compact_addresses]")
{
   compact_addresses<128> ca;
   ca.init();
   ca.insert(addr(0x10, 0), 0);
   ca.insert(addr(0x10, 2), 1);
   // insert between them
   ca.insert(addr(0x10, 1), 1);

   REQUIRE(ca.size() == 3);
   REQUIRE(ca.at(0) == addr(0x10, 0));
   REQUIRE(ca.at(1) == addr(0x10, 1));
   REQUIRE(ca.at(2) == addr(0x10, 2));
   REQUIRE(ca.base_count() == 1);  // all share one cacheline
}

TEST_CASE("compact_addresses remove is lazy: base lingers as dead, not dropped", "[compact_addresses]")
{
   compact_addresses<128> ca;
   ca.init();
   ca.insert(addr(0x10, 0), 0);  // base A
   ca.insert(addr(0x20, 0), 1);  // base B
   ca.insert(addr(0x10, 1), 2);  // base A again
   REQUIRE(ca.base_count() == 2);
   REQUIRE(ca.live_base_count() == 2);

   // remove the lone member of base B -> returns the address that was there
   REQUIRE(ca.remove(1) == addr(0x20, 0));
   REQUIRE(ca.size() == 2);
   REQUIRE(ca.base_count() == 2);       // slot lingers (lazy) ...
   REQUIRE(ca.live_base_count() == 1);  // ... but is now dead
   REQUIRE(ca.at(0) == addr(0x10, 0));
   REQUIRE(ca.at(1) == addr(0x10, 1));

   // canonicalize reclaims the dead slot
   ca.canonicalize();
   REQUIRE(ca.base_count() == 1);
   REQUIRE(ca.live_base_count() == 1);
   REQUIRE(ca.at(0) == addr(0x10, 0));
   REQUIRE(ca.at(1) == addr(0x10, 1));
}

TEST_CASE("compact_addresses canonicalize renumbers branches after dropping a middle base",
          "[compact_addresses]")
{
   compact_addresses<128> ca;
   ca.init();
   ca.insert(addr(0x10, 0), 0);  // base index 0
   ca.insert(addr(0x20, 0), 1);  // base index 1
   ca.insert(addr(0x30, 0), 2);  // base index 2
   REQUIRE(ca.base_count() == 3);

   // Remove the only user of the middle base (0x20). It lingers dead until canon.
   REQUIRE(ca.remove(1) == addr(0x20, 0));
   REQUIRE(ca.base_count() == 3);
   REQUIRE(ca.live_base_count() == 2);
   REQUIRE(ca.at(0) == addr(0x10, 0));
   REQUIRE(ca.at(1) == addr(0x30, 0));  // still resolves correctly while dead lingers

   // canonicalize drops base 0x20 and renumbers the branch that pointed at base 2.
   ca.canonicalize();
   REQUIRE(ca.base_count() == 2);
   REQUIRE(ca.at(0) == addr(0x10, 0));
   REQUIRE(ca.at(1) == addr(0x30, 0));
}

TEST_CASE("compact_addresses reuses a dead slot before growing when tight", "[compact_addresses]")
{
   // data_size = 31; each distinct-cacheline base+branch costs 5 bytes.
   compact_addresses<33> ca;
   ca.init();
   for (uint32_t cl = 0; cl < 6; ++cl)  // fills to free == 1
      ca.insert(addr(0x10 + cl, 0), ca.size());
   REQUIRE(ca.base_count() == 6);
   REQUIRE(ca.free_bytes() == 1);

   // Free one in the middle -> its base goes dead but stays (no room to grow).
   REQUIRE(ca.remove(2) == addr(0x12, 0));
   REQUIRE(ca.base_count() == 6);
   REQUIRE(ca.live_base_count() == 5);
   REQUIRE(ca.free_bytes() == 2);  // a branch byte was freed, base byte lingers

   // A brand-new cacheline cannot grow (needs 5, has 2) so it must reuse the dead
   // slot in place — base_count stays 6, no canonicalize needed.
   REQUIRE(ca.can_insert(addr(0x99, 0)));
   ca.insert(addr(0x99, 0), ca.size());
   REQUIRE(ca.base_count() == 6);
   REQUIRE(ca.live_base_count() == 6);
   REQUIRE(ca.at(ca.size() - 1) == addr(0x99, 0));
}

TEST_CASE("compact_addresses set replaces in place", "[compact_addresses]")
{
   compact_addresses<128> ca;
   ca.init();
   ca.insert(addr(0x10, 0), 0);
   ca.insert(addr(0x20, 0), 1);
   ca.insert(addr(0x10, 1), 2);
   REQUIRE(ca.base_count() == 2);

   // same-cacheline replace: in place, no base churn
   REQUIRE(ca.set(0, addr(0x10, 9)));
   REQUIRE(ca.at(0) == addr(0x10, 9));
   REQUIRE(ca.base_count() == 2);

   // different-cacheline replace of the lone 0x20 member: re-homes onto 0x10,
   // freeing base 0x20.
   REQUIRE(ca.set(1, addr(0x10, 5)));
   REQUIRE(ca.at(1) == addr(0x10, 5));
   REQUIRE(ca.live_base_count() == 1);
   // surrounding entries intact
   REQUIRE(ca.at(0) == addr(0x10, 9));
   REQUIRE(ca.at(2) == addr(0x10, 1));
}

TEST_CASE("compact_addresses set can fail when re-home needs a base and node is full",
          "[compact_addresses]")
{
   // data_size = 31. Fill exactly: cacheline 0x10 has TWO members (base 4 + 2
   // branches = 6 bytes), then five distinct cachelines (5 * 5 = 25). 6+25 = 31,
   // free == 0. Crucially 0x10's base keeps a second sharer, so removing entry 0
   // does NOT free a base slot.
   compact_addresses<33> ca;
   ca.init();
   ca.insert(addr(0x10, 0), 0);
   ca.insert(addr(0x10, 1), 1);
   for (uint32_t cl = 1; cl <= 5; ++cl)
      ca.insert(addr(0x10 + cl, 0), ca.size());
   REQUIRE(ca.size() == 7);
   REQUIRE(ca.base_count() == 6);
   REQUIRE(ca.free_bytes() == 0);

   // Re-homing entry 0 to a brand-new cacheline needs a fresh base; removing the
   // old entry only frees a branch byte (0x10 still has its other member), so it
   // can't fit -> fail, unchanged.
   REQUIRE_FALSE(ca.set(0, addr(0x99, 0)));
   REQUIRE(ca.at(0) == addr(0x10, 0));
   REQUIRE(ca.size() == 7);

   // But re-homing onto an already-present cacheline succeeds.
   REQUIRE(ca.set(0, addr(0x11, 7)));
   REQUIRE(ca.at(0) == addr(0x11, 7));
}

TEST_CASE("compact_addresses get_hint returns cacheline bases", "[compact_addresses]")
{
   compact_addresses<128> ca;
   ca.init();
   ca.insert(addr(0x10, 3), 0);
   ca.insert(addr(0x20, 9), 1);
   ca.insert(addr(0x10, 5), 2);

   auto hint = ca.get_hint();
   REQUIRE(hint.size() == 2);
   // Each hint resolves to a base cacheline via >> 4 (low nibble carries the
   // share count, which SAL ignores). We assert on the cacheline.
   std::vector<uint32_t> got{*hint[0] >> 4, *hint[1] >> 4};
   std::sort(got.begin(), got.end());
   REQUIRE(got == std::vector<uint32_t>{0x10u, 0x20u});
}

TEST_CASE("compact_addresses one cacheline holds up to 16 sharers", "[compact_addresses]")
{
   compact_addresses<128> ca;
   ca.init();
   for (uint32_t slot = 0; slot < 16; ++slot)
      ca.insert(addr(0x10, slot), ca.size());
   REQUIRE(ca.size() == 16);
   REQUIRE(ca.base_count() == 1);  // slot is 4 bits -> 16 fit in a single base
   for (uint32_t slot = 0; slot < 16; ++slot)
      REQUIRE(ca.at(slot) == addr(0x10, slot));
}

TEST_CASE("compact_addresses can_insert reflects real capacity", "[compact_addresses]")
{
   // Capacity chosen so distinct-cacheline fills leave exactly a 1-byte gap:
   // data_size = 31, each distinct cacheline costs 5B (4 base + 1 branch).
   compact_addresses<33> ca;  // data_size = 31
   ca.init();
   int inserted = 0;
   for (uint32_t cl = 0; cl < 64; ++cl)
   {
      const auto a = addr(0x10 + cl, 0);
      if (!ca.can_insert(a))
         break;
      ca.insert(a, ca.size());
      ++inserted;
   }
   // 6 * 5 = 30 bytes used, 1 byte free.
   REQUIRE(inserted == 6);
   REQUIRE(ca.base_count() == 6);
   REQUIRE(ca.free_bytes() == 1);
   REQUIRE_FALSE(ca.can_insert(addr(0x10 + 99, 0)));  // a new base needs 5 bytes

   // A same-cacheline insert only needs the 1 free branch byte, so it still fits.
   REQUIRE(ca.can_insert(addr(0x10, 1)));
   ca.insert(addr(0x10, 1), ca.size());
   REQUIRE(ca.free_bytes() == 0);
}

TEST_CASE("compact_addresses SIMD free-slot scan reuses a null slot (real-sized node)",
          "[compact_addresses]")
{
   // data_size = 66 (>= 64) so find_free_slot takes the ucc::find_u32x16 SIMD path.
   compact_addresses<68> ca;
   ca.init();
   int n = 0;
   for (uint32_t cl = 0; ca.can_insert(addr(0x40 + cl, 0)); ++cl)  // distinct cachelines
   {
      ca.insert(addr(0x40 + cl, 0), ca.size());
      ++n;
   }
   REQUIRE(n >= 8);  // enough bases that the SIMD scan is meaningful
   const int filled = ca.base_count();
   REQUIRE(ca.free_bytes() < 5);  // no room to just grow

   // Free a middle entry -> its base goes null; the SIMD scan must find that slot.
   const auto victim = addr(0x40 + 3, 0);
   REQUIRE(ca.remove(3) == victim);
   REQUIRE(ca.live_base_count() == filled - 1);

   REQUIRE(ca.can_insert(addr(0xABC, 0)));
   ca.insert(addr(0xABC, 0), ca.size());
   REQUIRE(ca.base_count() == filled);  // reused the null slot, did not grow
   REQUIRE(ca.live_base_count() == filled);
   REQUIRE(ca.at(ca.size() - 1) == addr(0xABC, 0));
}

TEST_CASE("compact_addresses fuzz against reference vector", "[compact_addresses][fuzz]")
{
   std::mt19937 rng(2024);
   for (int trial = 0; trial < 300; ++trial)
   {
      compact_addresses<128> ca;
      ca.init();
      std::vector<smart_address> ref;

      const int ops = 40;
      for (int op = 0; op < ops; ++op)
      {
         const bool do_insert = ref.empty() || (rng() & 1);
         if (do_insert)
         {
            // Keep cacheline spread small so dedup actually triggers and we
            // stay within the 16-base limit.
            const uint32_t cl   = rng() % 6;
            const uint32_t slot = rng() % 16;
            const auto     a    = addr(0x10 + cl, slot);
            const int      pos  = rng() % (ref.size() + 1);
            if (ca.can_insert(a))
            {
               ca.insert(a, pos);
               ref.insert(ref.begin() + pos, a);
            }
         }
         else
         {
            const int pos = rng() % ref.size();
            // remove returns the address that was stored at pos
            REQUIRE(ca.remove(pos) == ref[pos]);
            ref.erase(ref.begin() + pos);
         }

         REQUIRE(ca.size() == (int)ref.size());
         for (int i = 0; i < (int)ref.size(); ++i)
            REQUIRE(ca.at(i) == ref[i]);
      }
   }
}

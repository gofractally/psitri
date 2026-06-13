#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <new>

#include <sal_art_ref/compact_addresses.hpp>
#include <sal_art_ref/paged.hpp>
#include <sal_art_ref/radix_full.hpp>
#include <sal_art_ref/smart_address.hpp>

using sal_art_ref::smart_address;
using sal::ptr_address;

TEST_CASE("compact_addresses round-trips a mix of node pointers and tail tags",
          "[compact_addresses][smart_address]")
{
   sal_art_ref::compact_addresses<128> ca;
   ca.init();

   const auto node1 = smart_address::from_pointer(ptr_address(0x1234));
   const auto node2 = smart_address::from_pointer(ptr_address(0x1235));  // same cacheline as node1
   const auto tail1 = smart_address::from_tail(1);    // a local (inline) allocation
   const auto tail2 = smart_address::from_tail(500);

   ca.insert(node1, 0);
   ca.insert(tail1, 1);
   ca.insert(node2, 2);
   ca.insert(tail2, 3);

   // every entry round-trips bit-exact, and re-tags correctly
   REQUIRE(ca.at(0) == node1);
   REQUIRE(ca.at(0).is_pointer());
   REQUIRE(ca.at(1) == tail1);
   REQUIRE(ca.at(1).is_tail());
   REQUIRE(ca.at(1).tail_offset() == 1);
   REQUIRE(ca.at(2) == node2);
   REQUIRE(ca.at(3).tail_offset() == 500);

   // get_hint: node bases are usable hints; tail bases are reserved (SAL skips them)
   bool saw_node_hint = false, saw_reserved_hint = false;
   for (auto h : ca.get_hint())
   {
      if (sal::is_reserved_ptr_address(h))
         saw_reserved_hint = true;
      else
         saw_node_hint = true;
   }
   REQUIRE(saw_node_hint);      // node1/node2 cacheline
   REQUIRE(saw_reserved_hint);  // tail tag cachelines
}

TEST_CASE("compact_addresses tail tags near the null sentinel stay safe",
          "[compact_addresses][smart_address]")
{
   // tail offsets 1..15 all map to cacheline 0x0FFFFFFF — the only cacheline whose
   // base word could collide with the 0xFFFFFFFF free-slot sentinel. Offset 0 is
   // null and is never stored, so the base tops out at 15 sharers (word 0xFFFFFFFE).
   sal_art_ref::compact_addresses<128> ca;
   ca.init();
   for (uint16_t off = 1; off <= 15; ++off)
      ca.insert(smart_address::from_tail(off), ca.size());

   REQUIRE(ca.size() == 15);
   REQUIRE(ca.base_count() == 1);       // all one (reserved) cacheline
   REQUIRE(ca.live_base_count() == 1);  // base is NOT mistaken for free
   for (uint16_t off = 1; off <= 15; ++off)
      REQUIRE(ca.at(off - 1).tail_offset() == off);
}

TEST_CASE("radix_full holds node pointers and tail tags interchangeably",
          "[radix_full][smart_address]")
{
   sal_art_ref::radix_full<256> rf;
   rf.init();
   rf.set_branch(10, smart_address::from_pointer(ptr_address(0x2000)));
   rf.set_branch(20, smart_address::from_tail(7));  // local allocation

   REQUIRE(rf.find(10).is_pointer());
   REQUIRE(rf.find(20).is_tail());
   REQUIRE(rf.find(20).tail_offset() == 7);
   REQUIRE_FALSE(rf.contains(11));  // a real absent slot is null (== tail offset 0)
}

TEST_CASE("a smart_address branch resolves to a node ptr or a local tail allocation",
          "[smart_address][paged]")
{
   alignas(128) unsigned char buf[256];
   auto* page = new (buf) sal_art_ref::paged();
   page->init_page(sizeof(buf), sal::header_type::start_user_type, sal::ptr_address_seq{});

   // a LOCAL branch: an inline value written to this page's tail heap
   char* blk = page->alloc_tail(4);
   std::memcpy(blk, "abcd", 4);
   const smart_address local = smart_address::from_tail(page->tail_pos<>().offset_from_tail);

   // a NODE branch: a pointer to another node
   const smart_address node = smart_address::from_pointer(ptr_address(0x9999));

   // resolve a local branch -> live pointer into THIS page (relocation-stable)
   REQUIRE(local.is_tail());
   REQUIRE((void*)page->resolve(local) == (void*)blk);
   REQUIRE(std::memcmp(page->resolve(local), "abcd", 4) == 0);

   // resolve a node branch -> the child address (followed via the allocator)
   REQUIRE(node.is_pointer());
   REQUIRE(*node.as_pointer() == 0x9999u);
}

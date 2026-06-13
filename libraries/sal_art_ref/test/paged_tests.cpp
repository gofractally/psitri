#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <new>

#include <sal_art_ref/paged.hpp>

TEST_CASE("paged tail arena alloc + resolve", "[paged]")
{
   alignas(128) unsigned char buf[256];
   auto* p = new (buf) sal_art_ref::paged();
   p->init_page(sizeof(buf), sal::header_type::start_user_type, sal::ptr_address_seq{});

   REQUIRE(p->size() == 256);
   REQUIRE((void*)p->tail() == (void*)(buf + 256));
   REQUIRE(p->tail_used() == 0);
   REQUIRE((void*)p->tail_top() == (void*)(buf + 256));

   const void* head_end = buf + sizeof(sal_art_ref::paged);
   REQUIRE(p->free_bytes(head_end) == 256 - sizeof(sal_art_ref::paged));
   REQUIRE(p->can_alloc_tail(10, head_end));

   char* b1 = p->alloc_tail(10);
   auto  r1 = p->tail_pos<char>();
   REQUIRE((void*)b1 == (void*)(buf + 246));
   REQUIRE(p->tail_used() == 10);
   REQUIRE((void*)p->resolve(r1) == (void*)b1);

   char* b2 = p->alloc_tail(20);
   auto  r2 = p->tail_pos<char>();
   REQUIRE((void*)b2 == (void*)(buf + 226));
   REQUIRE(p->tail_used() == 30);
   REQUIRE((void*)p->resolve(r1) == (void*)b1);  // earlier ref still valid
   REQUIRE((void*)p->resolve(r2) == (void*)b2);

   b1[0] = 0x42;
   REQUIRE(p->resolve(r1)[0] == 0x42);
}

TEST_CASE("paged tail_ptr survives relocation", "[paged]")
{
   alignas(128) unsigned char buf[256];
   auto* p = new (buf) sal_art_ref::paged();
   p->init_page(sizeof(buf), sal::header_type::start_user_type, sal::ptr_address_seq{});

   char* blk = p->alloc_tail(8);
   auto  ref = p->tail_pos<char>();
   std::memset(blk, 0xAB, 8);

   // Simulate SAL relocating the allocation to a different address.
   alignas(128) unsigned char buf2[256];
   std::memcpy(buf2, buf, sizeof(buf));
   auto* moved = reinterpret_cast<sal_art_ref::paged*>(buf2);

   // The same tail_ptr resolves to the block at its NEW absolute address.
   REQUIRE((void*)moved->resolve(ref) == (void*)(buf2 + 256 - 8));
   REQUIRE(moved->resolve(ref)[0] == (char)0xAB);
}

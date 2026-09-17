#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <new>
#include <array>

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

TEST_CASE("paged clone copies every tail alignment without touching free space", "[paged][regression]")
{
   for (uint16_t used = 64; used < 128; ++used)
   {
      CAPTURE(used);
      alignas(128) std::array<unsigned char, 512> source{};
      alignas(128) std::array<unsigned char, 384> destination;
      destination.fill(0xCD);
      auto* src = new (source.data()) sal_art_ref::paged();
      auto* dst = new (destination.data()) sal_art_ref::paged();
      src->init_page(512, sal::header_type::start_user_type, sal::ptr_address_seq{});
      dst->init_page(256, sal::header_type::start_user_type, sal::ptr_address_seq{});
      auto* bytes = src->alloc_tail(used);
      for (uint16_t i = 0; i < used; ++i) bytes[i] = char(i + 1);
      std::memset(source.data() + sizeof(sal_art_ref::paged), 0xAB,
                  80 - sizeof(sal_art_ref::paged));
      src->blit_head_tail(dst, 80);
      REQUIRE(dst->size() == 256);
      REQUIRE(dst->tail_used() == used);
      REQUIRE(std::memcmp(dst->tail_top(), bytes, used) == 0);
      REQUIRE(std::memcmp(destination.data() + sizeof(sal::alloc_header),
                          source.data() + sizeof(sal::alloc_header),
                          80 - sizeof(sal::alloc_header)) == 0);
      for (size_t i = 80; i < 256 - used; ++i) REQUIRE(destination[i] == 0xCD);
      for (size_t i = 256; i < destination.size(); ++i) REQUIRE(destination[i] == 0xCD);
   }
}

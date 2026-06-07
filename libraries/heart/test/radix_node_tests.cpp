#include <catch2/catch_test_macros.hpp>
#include <new>

#include <heart/radix_nodes.hpp>
#include <heart/inline_value.hpp>
#include <heart/leaf_node.hpp>
#include <heart/radix_node.hpp>
#include <heart/version_history.hpp>
#include <heart/version_node.hpp>
#include <heart/versioned.hpp>

using heart::smart_address;

namespace
{
   // A minimal Derived to exercise the CRTP layout. The fixed-field struct is a
   // separate, complete type passed as the second template argument.
   struct example_fixed
   {
      uint16_t nstems;
   };
   struct example_node : heart::radix_impl<example_node, example_fixed>
   {
      static constexpr heart::dispatch_shape tier = heart::dispatch_shape::r1;
   };
   static_assert(sizeof(example_node) == 128, "radix_impl must fill exactly one cacheline");
   static_assert(example_node::prefix_capacity == 128 - int(sizeof(heart::radix_base)) - 2);
}  // namespace

TEST_CASE("radix_impl layout + common accessors", "[radix_node]")
{
   alignas(128) unsigned char buf[128];
   auto* n = new (buf) example_node();
   n->init_page(sizeof(buf), sal::header_type::start_user_type,
                sal::ptr_address_seq{sal::ptr_address(7), 12345});

   n->set_rtype(heart::radix_node_type::setlist_u8);
   n->set_version_history(sal::null_ptr_address);
   n->set_term_value(smart_address::from_pointer(sal::ptr_address(42)));
   n->fixed.nstems = 5;

   const uint8_t pfx[] = {0xAA, 0xBB, 0xCC};
   n->set_prefix(std::span<const uint8_t>(pfx, 3));

   REQUIRE(n->rtype() == heart::radix_node_type::setlist_u8);

   // node version reuses the SAL alloc_header 48-bit field (no separate storage)
   REQUIRE(n->created_version() == 12345);
   REQUIRE(n->created_version() == n->sequence());

   REQUIRE_FALSE(n->has_version_history());
   REQUIRE(n->has_term_value());
   REQUIRE(*n->term_value().as_pointer() == 42);

   REQUIRE(n->prefix_length() == 3);
   REQUIRE(n->prefix().size() == 3);
   REQUIRE(n->prefix()[0] == 0xAA);
   REQUIRE(n->prefix()[2] == 0xCC);

   REQUIRE(n->fixed.nstems == 5);

   // a prefix forces the r1 dispatch shape even though the node's tier is r1 here
   REQUIRE(n->stamp_shape() == heart::dispatch_shape::r1);

   // it's still a versioned page: the tail arena is usable past the header cacheline
   REQUIRE((void*)n->tail() == (void*)(buf + 128));
   REQUIRE(n->tail_used() == 0);
}

TEST_CASE("node family: sizes and dispatch tiers", "[radix_node]")
{
   using namespace heart;
   // total cachelines: 1 / 3 / 5 / 9 = 128 / 384 / 640 / 1152
   CHECK(sizeof(suffix_value_node) == 128);
   CHECK(sizeof(prefix_next_node) == 128);
   CHECK(sizeof(setlist_tiny_node) == 128);
   CHECK(sizeof(setlist_node<uint8_t>) == 128);
   CHECK(sizeof(setlist_node<uint32_t>) == 128);
   CHECK(sizeof(bitset1_node) == 128);
   CHECK(sizeof(bitset_r2_node) == 384);
   CHECK(sizeof(bitset_r4_node) == 640);
   CHECK(sizeof(full_radix_node) == 1152);

   CHECK(setlist_tiny_node::tier == dispatch_shape::r1);
   CHECK(bitset1_node::tier == dispatch_shape::r1);
   CHECK(bitset_r2_node::tier == dispatch_shape::r2);
   CHECK(bitset_r4_node::tier == dispatch_shape::r4);
   CHECK(full_radix_node::tier == dispatch_shape::r8);

   // the bitset family carries the refcount-tree roots; sparse nodes do not
   CHECK(bitset1_node::refcount_block_count == 1);
   CHECK(bitset_r2_node::refcount_block_count == 2);
   CHECK(bitset_r4_node::refcount_block_count == 4);
   CHECK(full_radix_node::refcount_block_count == 4);
}

TEST_CASE("full_radix_node is 9 cachelines with an r8 tier", "[radix_node]")
{
   static_assert(sizeof(heart::full_radix_node) == 1152);
   REQUIRE(heart::full_radix_node::tier == heart::dispatch_shape::r8);

   // with no prefix, the stamped shape is the tier (r8); a prefix would force r1
   alignas(128) static unsigned char buf[1152];
   auto* n = new (buf) heart::full_radix_node();
   n->init_page(sizeof(buf), sal::header_type::start_user_type, sal::ptr_address_seq{});
   n->set_rtype(heart::radix_node_type::full256);
   REQUIRE(n->stamp_shape() == heart::dispatch_shape::r8);
   n->branches.init();
   n->branches.set_branch(200, smart_address::from_pointer(sal::ptr_address(0x77)));
   REQUIRE(*n->branches.find(200).as_pointer() == 0x77);
}

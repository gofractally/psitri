#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <new>
#include <set>
#include <vector>

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

TEST_CASE("wide setlist_node<u32> grows forward across cachelines", "[radix_node]")
{
   // A wide (stem_width==4) setlist uses the growable forward layout: stems then
   // branches, both extending past CL0, tail heap from the end. u8 is untouched.
   using wide = heart::setlist_node<uint32_t>;
   alignas(64) unsigned char buf[1024];
   const uint8_t pfx[] = {1, 2, 3, 4, 5};
   auto* n = new (buf) wide(1024u, sal::ptr_address_seq{sal::ptr_address(1), 1},
                            std::span<const uint8_t>(pfx, 5));
   REQUIRE(n->stem_width == 4);
   REQUIRE(n->prefix_length() == 5);

   const int K = 100;  // 100 * (4+4) = 800 bytes of index -> spans many cachelines
   for (int i = 0; i < K; ++i)
   {
      uint32_t stem = 0x01000000u + uint32_t(i) * 0x010101u;  // distinct big-endian u32
      REQUIRE(n->set_branch(stem, smart_address::from_pointer(sal::ptr_address(1000 + i))));
   }
   REQUIRE(n->nstems() == K);
   for (int i = 0; i < K; ++i)  // every stem still resolves to its branch (append-memmove ok)
   {
      uint32_t stem = 0x01000000u + uint32_t(i) * 0x010101u;
      auto     sa   = n->find(stem);
      REQUIRE(sa.is_pointer());
      REQUIRE(*sa.as_pointer() == uint32_t(1000 + i));
   }
   REQUIRE(n->find(0xDEADBEEFu).is_null());

   // reprefix (shrink): stems+branches move as one block, all entries preserved
   const uint8_t shorter[] = {4, 5};
   n->reprefix(std::span<const uint8_t>(shorter, 2));
   REQUIRE(n->prefix_length() == 2);
   REQUIRE(n->nstems() == K);
   for (int i = 0; i < K; ++i)
   {
      uint32_t stem = 0x01000000u + uint32_t(i) * 0x010101u;
      auto     sa   = n->find(stem);
      REQUIRE(sa.is_pointer());
      REQUIRE(*sa.as_pointer() == uint32_t(1000 + i));
   }
}

namespace
{
   // Abstract contract for the SORTED setlist: regardless of insertion order, stems end
   // up ascending, find() resolves, and lower_bound_index/upper_bound_index agree with
   // std::lower_bound/upper_bound over the inserted set. Exercised for every stem width.
   template <class ST>
   void exercise_sorted_setlist(unsigned char* buf, uint32_t bufsize,
                                const std::vector<uint32_t>& cells)
   {
      using node_t = heart::setlist_node<ST>;
      auto* n      = new (buf) node_t(bufsize, sal::ptr_address_seq{sal::ptr_address(1), 1},
                                      std::span<const uint8_t>{});
      std::set<uint32_t> ins;
      for (uint32_t c : cells)  // insert in the given (non-sorted) order
      {
         const ST s = ST(c);
         REQUIRE(n->set_branch(s, smart_address::from_pointer(sal::ptr_address(1000 + uint32_t(s)))));
         ins.insert(uint32_t(s));
         for (int i = 1; i < n->nstems(); ++i)  // invariant after EVERY insert: ascending
            REQUIRE(n->stem_at(i - 1) < n->stem_at(i));
      }
      REQUIRE(n->nstems() == int(ins.size()));
      for (uint32_t c : ins)  // every key resolves; re-set is an update, not an append
      {
         auto sa = n->find(ST(c));
         REQUIRE(sa.is_pointer());
         REQUIRE(*sa.as_pointer() == uint32_t(1000 + c));
      }
      REQUIRE(n->set_branch(ST(*ins.begin()),
                            smart_address::from_pointer(sal::ptr_address(7))));  // update existing
      REQUIRE(n->nstems() == int(ins.size()));
      REQUIRE(*n->find(ST(*ins.begin())).as_pointer() == 7u);

      std::vector<uint32_t> sorted(ins.begin(), ins.end());
      const uint32_t        maxc = uint32_t(ST(~0u));
      const uint32_t        step = maxc / 257 + 1;
      for (uint64_t q = 0; q <= maxc; q += step)
      {
         const int lb = int(std::lower_bound(sorted.begin(), sorted.end(), uint32_t(q)) - sorted.begin());
         const int ub = int(std::upper_bound(sorted.begin(), sorted.end(), uint32_t(q)) - sorted.begin());
         REQUIRE(n->lower_bound_index(ST(q)) == lb);
         REQUIRE(n->upper_bound_index(ST(q)) == ub);
      }
   }
}  // namespace

TEST_CASE("setlist_node is a SORTED set: ordered stems + lower/upper_bound_index", "[radix_node]")
{
   // Build scattered, distinct cells (a cheap hash) so insertion order is NOT ascending —
   // this is what proves set_branch sorts rather than appends.
   auto scatter = [](int count, uint32_t mask) {
      std::set<uint32_t>    seen;
      std::vector<uint32_t> out;
      for (uint32_t i = 0; out.size() < size_t(count); ++i)
      {
         uint32_t c = ((i * 2654435761u) ^ (i << 13)) & mask;
         if (seen.insert(c).second)
            out.push_back(c);
      }
      return out;
   };

   SECTION("u8")
   {
      alignas(128) unsigned char buf[128];
      exercise_sorted_setlist<uint8_t>(buf, sizeof(buf), scatter(15, 0xFF));
   }
   SECTION("u16")
   {
      alignas(64) unsigned char buf[2048];
      exercise_sorted_setlist<uint16_t>(buf, sizeof(buf), scatter(200, 0xFFFF));
   }
   SECTION("u32")
   {
      alignas(64) unsigned char buf[2048];
      exercise_sorted_setlist<uint32_t>(buf, sizeof(buf), scatter(200, 0xFFFFFFFFu));
   }
}

namespace
{
   // Abstract contract for the segmented bitset tier (bitset_rN_node template): a byte
   // routes to seg(byte>>seg_shift), find resolves, and for_each_entry yields ascending
   // byte order. Run for both the r2 and r4 instantiations of the SAME template.
   template <class NodeT>
   void exercise_bitset_node(unsigned char* buf, uint32_t bufsize)
   {
      auto* n = new (buf) NodeT(bufsize, sal::ptr_address_seq{sal::ptr_address(1), 1},
                                std::span<const uint8_t>{});
      std::vector<int> in;
      for (int b = 3; b < 256; b += 7)  // spread across every segment, sparse per segment
         in.push_back(b);
      for (int b : in)
         REQUIRE(n->set_branch(uint8_t(b), smart_address::from_pointer(sal::ptr_address(2000 + b))));
      for (int b : in)
      {
         auto sa = n->find(uint8_t(b));
         REQUIRE(sa.is_pointer());
         REQUIRE(*sa.as_pointer() == uint32_t(2000 + b));
         REQUIRE(n->seg(b >> NodeT::seg_shift).find(b & NodeT::seg_mask) == sa);  // routing
      }
      REQUIRE(n->find(uint8_t(4)).is_null());
      std::vector<int> seen;
      n->for_each_entry([&](uint8_t b, smart_address) { seen.push_back(b); });
      REQUIRE(seen == in);  // `in` is ascending → bitset enumerates in byte order
   }
}  // namespace

TEST_CASE("bitset_rN_node template: routing + ordered enumeration (r2 and r4)", "[radix_node]")
{
   SECTION("r2")
   {
      alignas(128) unsigned char buf[heart::bitset_r2_node::nseg * 128 + 128];
      exercise_bitset_node<heart::bitset_r2_node>(buf, sizeof(buf));
   }
   SECTION("r4")
   {
      alignas(128) unsigned char buf[heart::bitset_r4_node::nseg * 128 + 128];
      exercise_bitset_node<heart::bitset_r4_node>(buf, sizeof(buf));
   }
}

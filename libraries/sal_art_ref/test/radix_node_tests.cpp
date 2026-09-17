#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <map>
#include <new>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <sal_art_ref/bucket_node.hpp>
#include <sal_art_ref/radix_nodes.hpp>
#include <sal_art_ref/inline_value.hpp>
#include <sal_art_ref/leaf_node.hpp>
#include <sal_art_ref/radix_node.hpp>
#include <sal_art_ref/version_history.hpp>
#include <sal_art_ref/version_node.hpp>
#include <sal_art_ref/versioned.hpp>

using sal_art_ref::smart_address;

namespace
{
   // A minimal Derived to exercise the CRTP layout. The fixed-field struct is a
   // separate, complete type passed as the second template argument.
   struct example_fixed
   {
      uint16_t nstems;
   };
   struct example_node : sal_art_ref::radix_impl<example_node, example_fixed>
   {
      static constexpr sal_art_ref::dispatch_shape tier = sal_art_ref::dispatch_shape::r1;
   };
   static_assert(sizeof(example_node) == 128, "radix_impl must fill exactly one cacheline");
   static_assert(example_node::prefix_capacity == 128 - int(sizeof(sal_art_ref::radix_base)) - 2);
}  // namespace

TEST_CASE("radix_impl layout + common accessors", "[radix_node]")
{
   alignas(128) unsigned char buf[128];
   auto* n = new (buf) example_node();
   n->init_page(sizeof(buf), sal::header_type::start_user_type,
                sal::ptr_address_seq{sal::ptr_address(7), 12345});

   n->set_rtype(sal_art_ref::radix_node_type::setlist_u8);
   n->set_version_history(sal::null_ptr_address);
   n->set_term_value(smart_address::from_pointer(sal::ptr_address(42)));
   n->fixed.nstems = 5;

   const uint8_t pfx[] = {0xAA, 0xBB, 0xCC};
   n->set_prefix(std::span<const uint8_t>(pfx, 3));

   REQUIRE(n->rtype() == sal_art_ref::radix_node_type::setlist_u8);

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
   REQUIRE(n->stamp_shape() == sal_art_ref::dispatch_shape::r1);

   // it's still a versioned page: the tail arena is usable past the header cacheline
   REQUIRE((void*)n->tail() == (void*)(buf + 128));
   REQUIRE(n->tail_used() == 0);
}

TEST_CASE("node family: sizes and dispatch tiers", "[radix_node]")
{
   using namespace sal_art_ref;
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

   // the bitset family carries per-range refcounts (range_ctrs); sparse nodes do not. Every big
   // tier now has RANGE_COUNT (8) per-range counters — refcount_block_count is the is_big marker.
   CHECK(bitset1_node::refcount_block_count == sal_art_ref::range_count);
   CHECK(bitset_r2_node::refcount_block_count == sal_art_ref::range_count);
   CHECK(bitset_r4_node::refcount_block_count == sal_art_ref::range_count);
   CHECK(full_radix_node::refcount_block_count == sal_art_ref::range_count);
}

TEST_CASE("full_radix_node is 9 cachelines with an r8 tier", "[radix_node]")
{
   static_assert(sizeof(sal_art_ref::full_radix_node) == 1152);
   REQUIRE(sal_art_ref::full_radix_node::tier == sal_art_ref::dispatch_shape::r8);

   // with no prefix, the stamped shape is the tier (r8); a prefix would force r1
   alignas(128) static unsigned char buf[1152];
   auto* n = new (buf) sal_art_ref::full_radix_node();
   // Skip-header readers address dense branches directly from byte 128. A
   // nested header gaining platform-specific alignment breaks that path.
   REQUIRE(reinterpret_cast<unsigned char*>(&n->fixed) - buf == 29);
   REQUIRE(reinterpret_cast<unsigned char*>(&n->branches) - buf ==
           sal_art_ref::full_radix_node::branch_offset);
   n->init_page(sizeof(buf), sal::header_type::start_user_type, sal::ptr_address_seq{});
   n->set_rtype(sal_art_ref::radix_node_type::full256);
   REQUIRE(n->stamp_shape() == sal_art_ref::dispatch_shape::r8);
   n->branches.init();
   n->branches.set_branch(200, smart_address::from_pointer(sal::ptr_address(0x77)));
   REQUIRE(*n->branches.find(200).as_pointer() == 0x77);
}

TEST_CASE("wide setlist_node<u32> grows forward across cachelines", "[radix_node]")
{
   // A wide (stem_width==4) setlist uses the growable forward layout: stems then
   // branches, both extending past CL0, tail heap from the end. u8 is untouched.
   using wide = sal_art_ref::setlist_node<uint32_t>;
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
      using node_t = sal_art_ref::setlist_node<ST>;
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
      alignas(128) unsigned char buf[sal_art_ref::bitset_r2_node::nseg * 128 + 128];
      exercise_bitset_node<sal_art_ref::bitset_r2_node>(buf, sizeof(buf));
   }
   SECTION("r4")
   {
      alignas(128) unsigned char buf[sal_art_ref::bitset_r4_node::nseg * 128 + 128];
      exercise_bitset_node<sal_art_ref::bitset_r4_node>(buf, sizeof(buf));
   }
}

TEST_CASE("bucket_node: 2-3-4 leaf bucket — find/lower_bound/in-order vs std::map",
          "[radix_node]")
{
   using sal_art_ref::bucket_node;
   using sal_art_ref::inline_value;
   alignas(128) static unsigned char buf[bucket_node::page_bytes];
   auto* b = new (buf) bucket_node(bucket_node::page_bytes,
                                   sal::ptr_address_seq{sal::ptr_address(1), 1});

   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(99);
   int                                inserted = 0;
   for (;;)  // fill until the page overflows
   {
      std::string k(1 + int(rng() % 18), 0);
      for (auto& c : k) c = char(uint8_t(rng()));
      std::string v(1 + int(rng() % 20), char(uint8_t(rng())));
      int r = b->insert(reinterpret_cast<const uint8_t*>(k.data()), int(k.size()),
                        reinterpret_cast<const uint8_t*>(v.data()), int(v.size()));
      if (r == 0) break;  // full -> caller would split to radix
      oracle[k] = v;
      ++inserted;
   }
   REQUIRE(inserted > 50);  // a 4 KB page holds many variable-length keys

   for (auto& [k, v] : oracle)  // find every key
   {
      const inline_value* iv = b->find(reinterpret_cast<const uint8_t*>(k.data()), int(k.size()));
      REQUIRE(iv != nullptr);
      REQUIRE(std::string(reinterpret_cast<const char*>(iv->data) + iv->suffix_len, iv->value_len) == v);
   }

   auto it  = oracle.begin();  // in-order == oracle order
   int  cnt = 0;
   b->for_each([&](const inline_value& iv) {
      REQUIRE(it != oracle.end());
      REQUIRE(std::string(reinterpret_cast<const char*>(iv.data), iv.suffix_len) == it->first);
      ++it;
      ++cnt;
   });
   REQUIRE(cnt == int(oracle.size()));

   auto lb = [&](const std::string& q) {  // lower_bound vs std::map (returns entry index)
      int  i  = b->lower_bound(reinterpret_cast<const uint8_t*>(q.data()), int(q.size()));
      auto oi = oracle.lower_bound(q);
      if (i >= b->nent)
      {
         REQUIRE(oi == oracle.end());
         return;
      }
      REQUIRE(oi != oracle.end());
      const inline_value* iv = b->entry(i);
      REQUIRE(std::string(reinterpret_cast<const char*>(iv->data), iv->suffix_len) == oi->first);
   };
   for (auto& [k, v] : oracle)
   {
      lb(k);
      if (!k.empty()) lb(k.substr(0, k.size() - 1));
      lb(k + char(uint8_t(rng())));
   }
   for (int i = 0; i < 500; ++i)
   {
      std::string q(int(rng() % 16), 0);
      for (auto& c : q) c = char(uint8_t(rng()));
      lb(q);
   }
}

TEST_CASE("bucket_node: COW/compact blit copies head+tail, skips the free hole", "[radix_node]")
{
   using sal_art_ref::bucket_node;
   using sal_art_ref::inline_value;
   alignas(128) static unsigned char src[bucket_node::page_bytes];
   auto* b = new (src) bucket_node(bucket_node::page_bytes,
                                   sal::ptr_address_seq{sal::ptr_address(1), 1});

   std::map<std::string, std::string> oracle;
   std::mt19937_64                    rng(424242);
   int                                inserted = 0;
   for (;;)
   {
      std::string k(1 + int(rng() % 18), 0);
      for (auto& ch : k) ch = char(uint8_t(rng()));
      std::string v(1 + int(rng() % 20), char(uint8_t(rng())));
      int         r = b->insert(reinterpret_cast<const uint8_t*>(k.data()), int(k.size()),
                                reinterpret_cast<const uint8_t*>(v.data()), int(v.size()));
      if (r == 0) break;
      oracle[k] = v;
      if (++inserted >= 60) break;  // stop half-full so there's a free hole to skip/reclaim
   }
   REQUIRE(inserted == 60);

   // compact_size must actually shrink the page (it isn't full of head+tail) and copy proof:
   const uint32_t cs = b->compact_size();
   REQUIRE(cs <= bucket_node::page_bytes);
   REQUIRE(cs < bucket_node::alloc_size());  // the free hole is reclaimed

   auto verify = [&](const bucket_node* p) {
      REQUIRE(int(p->nent) == inserted);
      for (auto& [k, v] : oracle)  // every key still resolves to its value
      {
         const inline_value* iv =
             p->find(reinterpret_cast<const uint8_t*>(k.data()), int(k.size()));
         REQUIRE(iv != nullptr);
         REQUIRE(std::string(reinterpret_cast<const char*>(iv->data) + iv->suffix_len,
                             iv->value_len) == v);
      }
      auto it  = oracle.begin();  // in-order preserved
      int  cnt = 0;
      p->for_each([&](const inline_value& iv) {
         REQUIRE(std::string(reinterpret_cast<const char*>(iv.data), iv.suffix_len) == it->first);
         ++it;
         ++cnt;
      });
      REQUIRE(cnt == inserted);
   };

   // compact_to: blit into a TIGHTLY sized destination (the compactor's job).
   alignas(128) static unsigned char cbuf[bucket_node::page_bytes];
   auto* c = new (cbuf) bucket_node(cs, sal::ptr_address_seq{sal::ptr_address(2), 1});
   b->compact_to(c);
   REQUIRE(c->size() == cs);
   verify(c);

   // copy_to: blit (COW) into a full working page; it must still accept inserts in place.
   alignas(128) static unsigned char wbuf[bucket_node::page_bytes];
   auto* w = new (wbuf) bucket_node(bucket_node::alloc_size(),
                                    sal::ptr_address_seq{sal::ptr_address(3), 1});
   b->copy_to(w);
   REQUIRE(w->size() == bucket_node::alloc_size());
   verify(w);

   // round-trip: a compacted page COW-expanded back to a working page is identical and writable.
   auto* w2 = new (src) bucket_node(bucket_node::alloc_size(),  // reuse src storage
                                    sal::ptr_address_seq{sal::ptr_address(4), 1});
   c->copy_to(w2);
   verify(w2);
   std::string nk = "\x01zzznew", nv = "v";
   REQUIRE(w2->insert(reinterpret_cast<const uint8_t*>(nk.data()), int(nk.size()),
                      reinterpret_cast<const uint8_t*>(nv.data()), int(nv.size())) == 1);
}

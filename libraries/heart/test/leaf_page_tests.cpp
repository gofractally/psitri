// Standalone leaf_page tests — raw aligned buffers, no allocator. The page is the
// 2KB COW terminal tier: SoA fingerprint/prefix/vals columns, tail-heap records,
// cbase-coded external values, split-time common-prefix factoring.
#include <catch2/catch_test_macros.hpp>

#include <heart/leaf_page.hpp>

#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using heart::inline_value;
using heart::leaf_page;
using sal::ptr_address;

namespace
{
   using bytes = std::span<const uint8_t>;
   // LIFETIME: the span aliases s. An op built from a TEMPORARY string must be
   // consumed within the same full expression — never stored and applied later.
   bytes b(const std::string& s)
   {
      return {reinterpret_cast<const uint8_t*>(s.data()), s.size()};
   }

   // RAII aligned page buffer (heap is fine in tests).
   struct page_buf
   {
      std::unique_ptr<char[]> mem;
      leaf_page*              pg = nullptr;

      explicit page_buf(uint32_t asize = leaf_page::page_bytes())
          : mem(new char[asize + 64])
      {
         char* base = mem.get();
         char* al   = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(base) + 63) & ~63ull);
         pg         = new (al) leaf_page(asize, sal::ptr_address_seq{sal::ptr_address(1), 1});
      }
      page_buf(uint32_t asize, const auto& op) : mem(new char[asize + 64])
      {
         char* base = mem.get();
         char* al   = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(base) + 63) & ~63ull);
         pg = new (al) leaf_page(asize, sal::ptr_address_seq{sal::ptr_address(1), 1}, op);
      }
      leaf_page* operator->() const { return pg; }
      leaf_page& operator*() const { return *pg; }
   };

   leaf_page::insert_op ins_bytes(const std::string& k, const std::string& v)
   {
      return {.key = b(k), .value = b(v), .vtype = inline_value::vt_bytes};
   }
   leaf_page::insert_op ins_ext(const std::string& k, ptr_address a)
   {
      return {.key = b(k), .value = {}, .vtype = inline_value::vt_extern, .ext = a};
   }

   std::string get_key(const leaf_page& p, uint32_t i)
   {
      uint8_t        buf[520];
      const uint32_t n = p.full_key(i, buf);
      return std::string(reinterpret_cast<const char*>(buf), n);
   }
   std::string get_val(const leaf_page& p, uint32_t i)
   {
      auto v = p.value(i);
      return std::string(reinterpret_cast<const char*>(v.data()), v.size());
   }

   // Insert preferring in-place; falls back to a rebuilding clone into `buf`.
   // Returns false when the page reports FULL both ways (caller would split).
   bool insert_any(page_buf& buf, const leaf_page::insert_op& op)
   {
      if (buf->can_apply(op) == leaf_page::can_apply_mode::modify)
      {
         buf->apply(op);
         return true;
      }
      if (!buf->rebuilt_fits(op))
         return false;
      page_buf next(leaf_page::page_bytes(), leaf_page::clone_insert_op{buf.pg, op});
      buf = std::move(next);
      return true;
   }

   void check_oracle(const leaf_page& p, const std::map<std::string, std::string>& m)
   {
      REQUIRE(p.num_keys() == m.size());
      REQUIRE(p.validate_invariants());
      uint32_t i = 0;
      for (auto& [k, v] : m)
      {
         REQUIRE(get_key(p, i) == k);  // in-order iteration matches the map
         REQUIRE(get_val(p, i) == v);
         const int fi = p.find(reinterpret_cast<const uint8_t*>(k.data()), uint32_t(k.size()));
         REQUIRE(fi == int(i));
         REQUIRE(p.lower_bound(reinterpret_cast<const uint8_t*>(k.data()), uint32_t(k.size())) ==
                 int(i));
         ++i;
      }
   }
}  // namespace

TEST_CASE("leaf_page empty init", "[leaf_page]")
{
   page_buf p;
   REQUIRE(p->num_keys() == 0);
   REQUIRE(p->capacity() % 4 == 0);
   REQUIRE(p->num_cbases() == 0);
   REQUIRE(p->common_prefix().empty());
   REQUIRE(p->validate_invariants());
   REQUIRE(p->find(reinterpret_cast<const uint8_t*>("x"), 1) == -1);
   REQUIRE(p->lower_bound(reinterpret_cast<const uint8_t*>("x"), 1) == 0);
}

TEST_CASE("leaf_page in-place insert find lower_bound vs oracle", "[leaf_page]")
{
   page_buf                           p;
   std::map<std::string, std::string> m;
   std::mt19937                       rng(42);
   for (int i = 0; i < 120; ++i)
   {
      std::string k = "key" + std::to_string(rng() % 100000);
      std::string v = "val" + std::to_string(i);
      if (m.count(k))
         continue;
      auto op = ins_bytes(k, v);
      if (p->can_apply(op) != leaf_page::can_apply_mode::modify)
         break;
      p->apply(op);
      m[k] = v;
   }
   REQUIRE(m.size() > 50);  // a 2KB page holds plenty of short entries
   check_oracle(*p, m);

   // misses: absent keys land at the right lower_bound and find() == -1
   for (int i = 0; i < 200; ++i)
   {
      std::string q = "key" + std::to_string(rng() % 200000) + "x";
      REQUIRE(p->find(reinterpret_cast<const uint8_t*>(q.data()), uint32_t(q.size())) == -1);
      auto      it       = m.lower_bound(q);
      const int expected = int(std::distance(m.begin(), it));
      REQUIRE(p->lower_bound(reinterpret_cast<const uint8_t*>(q.data()), uint32_t(q.size())) ==
              expected);
   }
}

TEST_CASE("leaf_page equal-prefix runs and zero bytes", "[leaf_page]")
{
   page_buf                           p;
   std::map<std::string, std::string> m;
   // keys sharing the first two bytes (one pfx-column run) + embedded NULs
   std::vector<std::string> keys = {std::string("ab"),
                                    std::string("ab\0", 3),
                                    std::string("ab\0x", 4),
                                    std::string("abc"),
                                    std::string("abcd"),
                                    std::string("abd"),
                                    std::string("\0\0", 2),
                                    std::string("\0a", 2),
                                    std::string("")};
   int                      i    = 0;
   for (auto& k : keys)
   {
      const std::string v  = "v" + std::to_string(i++);
      auto              op = ins_bytes(k, v);
      REQUIRE(p->can_apply(op) == leaf_page::can_apply_mode::modify);
      p->apply(op);
      m[k] = v;
   }
   check_oracle(*p, m);
}

TEST_CASE("leaf_page capacity stays multiple of 4 across growth", "[leaf_page]")
{
   page_buf p;
   uint16_t last_cap = p->capacity();
   for (int i = 0; i < 100; ++i)
   {
      const std::string k = "k" + std::to_string(1000 + i);
      const std::string v = "v";
      auto              op = ins_bytes(k, v);
      if (p->can_apply(op) != leaf_page::can_apply_mode::modify)
         break;
      p->apply(op);
      REQUIRE(p->capacity() % 4 == 0);
      REQUIRE(p->capacity() >= p->num_keys());
      last_cap = p->capacity();
   }
   REQUIRE(last_cap >= 100);
   REQUIRE(p->validate_invariants());
}

TEST_CASE("leaf_page clone insert splice equals in-place build", "[leaf_page]")
{
   // Build A in place; build B by chaining rebuild-clones; same content must result.
   page_buf                           a;
   page_buf                           bb;
   std::map<std::string, std::string> m;
   std::mt19937                       rng(7);
   for (int i = 0; i < 60; ++i)
   {
      std::string k = "node" + std::to_string(rng() % 10000);
      if (m.count(k))
         continue;
      std::string v = "v" + std::to_string(i);
      m[k]          = v;
      auto op       = ins_bytes(k, v);
      REQUIRE(a->can_apply(op) == leaf_page::can_apply_mode::modify);
      a->apply(op);
      REQUIRE(bb->rebuilt_fits(op));
      page_buf next(leaf_page::page_bytes(), leaf_page::clone_insert_op{bb.pg, op});
      bb = std::move(next);
   }
   check_oracle(*a, m);
   check_oracle(*bb, m);
   // the clone chain compacts as it goes: no dead bytes, capacity right-sized
   REQUIRE(bb->capacity() == leaf_page::round_cap(bb->num_keys()));
}

TEST_CASE("leaf_page update in place and via clone", "[leaf_page]")
{
   page_buf                           p;
   std::map<std::string, std::string> m;
   for (int i = 0; i < 40; ++i)
   {
      std::string k = "upd" + std::to_string(i);
      std::string v(8, char('a' + i % 26));
      p->apply(ins_bytes(k, v));
      m[k] = v;
   }
   check_oracle(*p, m);

   // shrink in place
   {
      const int             idx = p->find(reinterpret_cast<const uint8_t*>("upd7"), 4);
      REQUIRE(idx >= 0);
      std::string           nv = "tiny";
      leaf_page::update_op  op{.index = uint16_t(idx), .value = b(nv)};
      REQUIRE(p->can_apply(op) == leaf_page::can_apply_mode::modify);
      p->apply(op);
      m["upd7"] = nv;
      check_oracle(*p, m);
   }
   // grow via tail realloc in place
   {
      const int            idx = p->find(reinterpret_cast<const uint8_t*>("upd9"), 4);
      std::string          nv(60, 'Z');
      leaf_page::update_op op{.index = uint16_t(idx), .value = b(nv)};
      REQUIRE(p->can_apply(op) == leaf_page::can_apply_mode::modify);
      p->apply(op);
      m["upd9"] = nv;
      check_oracle(*p, m);
   }
   // clone-update rebuild drops the dead bytes
   {
      const int            idx = p->find(reinterpret_cast<const uint8_t*>("upd9"), 4);
      std::string          nv = "after-clone";
      leaf_page::update_op op{.index = uint16_t(idx), .value = b(nv)};
      page_buf             q(leaf_page::page_bytes(), leaf_page::clone_update_op{p.pg, op});
      m["upd9"] = nv;
      check_oracle(*q, m);
   }
}

TEST_CASE("leaf_page remove in place and via clone", "[leaf_page]")
{
   page_buf                           p;
   std::map<std::string, std::string> m;
   for (int i = 0; i < 30; ++i)
   {
      std::string k = "rm" + std::to_string(i);
      p->apply(ins_bytes(k, "v" + std::to_string(i)));
      m[k] = "v" + std::to_string(i);
   }
   // in-place removes
   for (int i = 0; i < 30; i += 3)
   {
      std::string k   = "rm" + std::to_string(i);
      const int   idx = p->find(reinterpret_cast<const uint8_t*>(k.data()), uint32_t(k.size()));
      REQUIRE(idx >= 0);
      p->apply(leaf_page::remove_op{uint16_t(idx)});
      m.erase(k);
      REQUIRE(p->validate_invariants());
   }
   check_oracle(*p, m);
   // clone-remove (shared mode path)
   {
      const int idx = p->find(reinterpret_cast<const uint8_t*>("rm1"), 3);
      REQUIRE(idx >= 0);
      page_buf q(leaf_page::page_bytes(), leaf_page::clone_remove_op{p.pg, uint16_t(idx)});
      m.erase("rm1");
      check_oracle(*q, m);
   }
}

TEST_CASE("leaf_page external values cbase coding", "[leaf_page]")
{
   page_buf p;
   // clustered addresses: 32 externals across exactly 2 cachelines of control blocks
   std::map<std::string, uint32_t> ext;
   for (uint32_t i = 0; i < 32; ++i)
   {
      const uint32_t    addr = 0x1000u + (i / 16) * 16 + (i % 16);
      const std::string k    = "ext" + std::to_string(100 + i);
      auto              op   = ins_ext(k, ptr_address(addr));
      REQUIRE(p->can_apply(op) == leaf_page::can_apply_mode::modify);
      p->apply(op);
      ext[k] = addr;
   }
   REQUIRE(p->num_cbases() == 2);
   REQUIRE(p->validate_invariants());
   for (auto& [k, addr] : ext)
   {
      const int i = p->find(reinterpret_cast<const uint8_t*>(k.data()), uint32_t(k.size()));
      REQUIRE(i >= 0);
      REQUIRE(p->is_external(uint32_t(i)));
      REQUIRE(*p->external_addr(uint32_t(i)) == addr);
   }
   // for_each_child enumerates every external exactly once
   std::vector<uint32_t> seen;
   p->for_each_child([&](ptr_address a) { seen.push_back(*a); });
   REQUIRE(seen.size() == ext.size());

   // scattered addresses exhaust the 16-cbase budget -> page reports FULL (split early)
   page_buf q;
   for (uint32_t i = 0; i < 16; ++i)
   {
      const std::string k  = "scat" + std::to_string(i);
      auto              op = ins_ext(k, ptr_address(0x10000u + i * 64));
      REQUIRE(q->can_apply(op) == leaf_page::can_apply_mode::modify);
      q->apply(op);
   }
   REQUIRE(q->num_cbases() == 16);
   const std::string k17  = "scat16";
   auto              op17 = ins_ext(k17, ptr_address(0x90000u));
   REQUIRE(q->can_apply(op17) == leaf_page::can_apply_mode::none);
   REQUIRE_FALSE(q->can_add_address(ptr_address(0x90000u)));
   REQUIRE_FALSE(q->rebuilt_fits(op17));  // clone can't help: still 17 live bases
   // ...but an address on an existing cacheline still fits
   const std::string ks      = "scat_same";
   auto              op_same = ins_ext(ks, ptr_address(0x10000u + 5));
   REQUIRE(q->can_apply(op_same) == leaf_page::can_apply_mode::modify);
}

TEST_CASE("leaf_page subtree values", "[leaf_page]")
{
   page_buf          p;
   const uint32_t    root = 0xABCDu;
   const std::string sk   = "sub1";
   uint8_t           rootb[4];
   std::memcpy(rootb, &root, 4);
   leaf_page::insert_op op{.key   = b(sk),
                           .value = std::span<const uint8_t>(rootb, 4),
                           .vtype = inline_value::vt_subtree};
   REQUIRE(p->can_apply(op) == leaf_page::can_apply_mode::modify);
   p->apply(op);
   const int i = p->find(reinterpret_cast<const uint8_t*>("sub1"), 4);
   REQUIRE(i >= 0);
   REQUIRE(p->value_type(uint32_t(i)) == inline_value::vt_subtree);
   REQUIRE(*p->subtree_root(uint32_t(i)) == root);
   std::vector<uint32_t> seen;
   p->for_each_child([&](ptr_address a) { seen.push_back(*a); });
   REQUIRE(seen == std::vector<uint32_t>{root});
   REQUIRE(p->validate_invariants());
}

TEST_CASE("leaf_page split divider and halves", "[leaf_page]")
{
   page_buf                           p;
   std::map<std::string, std::string> m;
   std::mt19937                       rng(99);
   // fill the page to (near) capacity
   while (true)
   {
      std::string k = "word" + std::to_string(rng() % 1000000);
      if (m.count(k))
         continue;
      const std::string v  = "payload-" + std::to_string(rng() % 1000);
      auto              op = ins_bytes(k, v);
      if (p->can_apply(op) != leaf_page::can_apply_mode::modify)
         break;
      p->apply(op);
      m[k] = v;
   }
   REQUIRE(p->num_keys() >= 16);

   const auto  sp  = p->get_split_pos();
   std::string div(reinterpret_cast<const char*>(sp.div), sp.div_len);
   // divider strictly separates the sorted keys: last left < divider <= first right
   REQUIRE(sp.index > 0);
   REQUIRE(sp.index < p->num_keys());
   REQUIRE(get_key(*p, sp.index - 1) < div);
   REQUIRE(div <= get_key(*p, sp.index));
   // bounded skew: split point stays in the middle window
   REQUIRE(sp.index >= p->num_keys() * 3 / 8);
   REQUIRE(sp.index <= p->num_keys() * 5 / 8 + 1);

   page_buf left(leaf_page::page_bytes(), leaf_page::split_op{p.pg, 0, sp.index});
   page_buf right(leaf_page::page_bytes(), leaf_page::split_op{p.pg, sp.index, p->num_keys()});
   REQUIRE(left->validate_invariants());
   REQUIRE(right->validate_invariants());
   REQUIRE(left->num_keys() + right->num_keys() == p->num_keys());

   // halves partition the oracle in order, and every key still finds
   uint32_t i = 0;
   for (auto& [k, v] : m)
   {
      const leaf_page& half = i < sp.index ? *left : *right;
      const uint32_t   hi   = i < sp.index ? i : i - sp.index;
      REQUIRE(get_key(half, hi) == k);
      REQUIRE(get_val(half, hi) == v);
      REQUIRE(half.find(reinterpret_cast<const uint8_t*>(k.data()), uint32_t(k.size())) ==
              int(hi));
      ++i;
   }
   // split is where factoring tightens: both halves share at least src's cpfx ("word")
   REQUIRE(left->common_prefix().size() >= 4);
   REQUIRE(right->common_prefix().size() >= 4);
}

TEST_CASE("leaf_page divider snaps to power-of-2 byte", "[leaf_page]")
{
   page_buf p;
   p->apply(ins_bytes("apple", "1"));
   p->apply(ins_bytes("azzz", "2"));
   const auto  sp = p->get_split_pos();
   std::string div(reinterpret_cast<const char*>(sp.div), sp.div_len);
   // shortest separator of (apple, azzz) is "a?" with ? in (p, z]; 0x78 has the most
   // trailing zeros in (0x70, 0x7A]
   REQUIRE(div == std::string("a\x78"));
   REQUIRE(sp.index == 1);
}

TEST_CASE("leaf_page reprefix on diverging insert", "[leaf_page]")
{
   // build a page whose keys share "team", split so the half factors the prefix out
   page_buf src;
   for (int i = 0; i < 20; ++i)
      src->apply(ins_bytes("team" + std::to_string(100 + i), "v" + std::to_string(i)));
   page_buf half(leaf_page::page_bytes(), leaf_page::split_op{src.pg, 0, src->num_keys()});
   REQUIRE(half->common_prefix().size() >= 4);  // "team1.." factored
   std::map<std::string, std::string> m;
   for (uint32_t i = 0; i < half->num_keys(); ++i)
      m[get_key(*half, i)] = get_val(*half, i);

   // a key that diverges from the factored prefix can't apply in place...
   const std::string dk = "tax";
   const std::string dv = "diverges";
   auto              op = ins_bytes(dk, dv);
   REQUIRE(half->can_apply(op) == leaf_page::can_apply_mode::none);
   REQUIRE(half->rebuilt_fits(op));
   // ...but the rebuilding clone re-expands suffixes under a shorter cpfx
   page_buf re(leaf_page::page_bytes(), leaf_page::clone_insert_op{half.pg, op});
   m["tax"] = "diverges";
   check_oracle(*re, m);
   REQUIRE(re->common_prefix().size() == 1);  // "t" ("tax" vs "team..." diverge at byte 1)
}

TEST_CASE("leaf_page compact roundtrip", "[leaf_page]")
{
   page_buf                           p;
   std::map<std::string, std::string> m;
   for (int i = 0; i < 50; ++i)
   {
      std::string k = "cmp" + std::to_string(i * 7);
      p->apply(ins_bytes(k, std::string(20, char('a' + i % 26))));
      m[k] = std::string(20, char('a' + i % 26));
   }
   // churn: removes + updates leave dead tail bytes behind
   for (int i = 0; i < 50; i += 5)
   {
      std::string k   = "cmp" + std::to_string(i * 7);
      const int   idx = p->find(reinterpret_cast<const uint8_t*>(k.data()), uint32_t(k.size()));
      p->apply(leaf_page::remove_op{uint16_t(idx)});
      m.erase(k);
   }
   {
      const int            idx = p->find(reinterpret_cast<const uint8_t*>("cmp7"), 4);
      std::string          nv(40, 'Q');
      leaf_page::update_op op{.index = uint16_t(idx), .value = b(nv)};
      p->apply(op);
      m["cmp7"] = nv;
   }
   REQUIRE(p->validate_invariants());

   const uint32_t csize = p->compact_size();
   REQUIRE(csize < leaf_page::page_bytes());  // churn means real shrink
   // compact destination the way SAL builds one: a header sized csize, content rebuilt
   page_buf dst(csize);  // empty-page ctor stamps the alloc_header at the compact size
   p->compact_to(dst.pg);
   REQUIRE(dst->size() == csize);
   check_oracle(*dst, m);
   REQUIRE(dst->capacity() == leaf_page::round_cap(dst->num_keys()));
}

TEST_CASE("leaf_page fuzz vs std::map", "[leaf_page][fuzz]")
{
   std::mt19937                       rng(1337);
   page_buf                           p;
   std::map<std::string, std::string> m;
   auto rand_key = [&]() -> std::string
   {
      // skewed: heavy shared-prefix population + some divergent strays
      if (rng() % 8)
         return "fuzz/" + std::to_string(rng() % 400);
      return std::string(1, char('a' + rng() % 26)) + std::to_string(rng() % 50);
   };
   for (int step = 0; step < 4000; ++step)
   {
      const std::string k    = rand_key();
      const uint8_t*    kp   = reinterpret_cast<const uint8_t*>(k.data());
      const uint32_t    kn   = uint32_t(k.size());
      const int         roll = int(rng() % 10);
      const int         idx  = p->find(kp, kn);
      REQUIRE((idx >= 0) == (m.count(k) > 0));
      if (roll < 5)  // upsert
      {
         std::string v = "v" + std::to_string(rng() % 100000);
         if (idx >= 0)
         {
            leaf_page::update_op op{.index = uint16_t(idx), .value = b(v)};
            if (p->can_apply(op) == leaf_page::can_apply_mode::modify)
            {
               p->apply(op);
               m[k] = v;
            }
            else if (p->rebuilt_fits(op))
            {
               page_buf next(leaf_page::page_bytes(), leaf_page::clone_update_op{p.pg, op});
               p    = std::move(next);
               m[k] = v;
            }
            // else: page full -> engine would split first
         }
         else if (insert_any(p, ins_bytes(k, v)))
            m[k] = v;
         // page full -> engine would split; the fuzz keeps the working set small enough
      }
      else if (roll < 8)  // remove (alternate in-place / clone)
      {
         if (idx >= 0)
         {
            if (rng() % 2)
               p->apply(leaf_page::remove_op{uint16_t(idx)});
            else
            {
               page_buf next(leaf_page::page_bytes(),
                             leaf_page::clone_remove_op{p.pg, uint16_t(idx)});
               p = std::move(next);
            }
            m.erase(k);
         }
      }
      else  // lower_bound spot-check
      {
         auto      it       = m.lower_bound(k);
         const int expected = int(std::distance(m.begin(), it));
         REQUIRE(p->lower_bound(kp, kn) == expected);
      }
      if (step % 256 == 0)
      {
         REQUIRE(p->validate_invariants());
         REQUIRE(p->num_keys() == m.size());
      }
   }
   check_oracle(*p, m);
}

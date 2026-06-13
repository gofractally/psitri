// COW correctness: privatize-then-mutate over the verbatim hkv tree (SPEC §2.7,
// §7, §9). Generational snapshots must stay bit-stable while the live tree
// mutates; releasing snapshots cascades old versions back to the arena; at the
// end the arena drains to exactly empty (no leak, no double free).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <random>
#include <string>

#include <start_ref/hkv/tree.hpp>
#include <start_ref/tree_alloc.hpp>

using namespace start_ref;
namespace fs = std::filesystem;

using tree_t =
    start_ref::hkv::kv_tree<std::string_view, uint64_t, start_ref::hkv::mode::none, part_alloc<uint64_t>>;
using oracle_t = std::map<std::string, uint64_t>;

namespace
{
   fs::path tmp_dir(const char* tag)
   {
      static std::mt19937_64 rng(std::random_device{}());
      auto p = fs::temp_directory_path() /
               (std::string("part_cow_") + tag + "_" + std::to_string(rng()));
      fs::remove_all(p);
      return p;
   }

   void verify_exact(tree_t& t, const oracle_t& o)
   {
      uint64_t out = 0;
      for (auto& [k, v] : o)
      {
         REQUIRE(t.find(std::string_view(k), out));
         REQUIRE(out == v);
      }
      auto oit = o.begin();
      for (auto it = t.begin(); it != t.end(); ++it, ++oit)
      {
         REQUIRE(oit != o.end());
         REQUIRE((*it).first == oit->first);
         REQUIRE((*it).second == oit->second);
      }
      REQUIRE(oit == o.end());
   }

}  // namespace

TEST_CASE("snapshots stay stable across COW mutation; releases drain exactly")
{
   arena_config c;
   c.dir = tmp_dir("gen");
   arena a(c);
   {
      auto   ses = a.start_session();
      auto   tx  = ses.start_transaction(0);
      tree_t t{part_alloc<uint64_t>(&ses)};

      std::mt19937_64          rng(123);
      std::vector<std::string> keys;
      for (int i = 0; i < 8000; ++i)
      {
         const uint64_t r = rng();
         keys.emplace_back(reinterpret_cast<const char*>(&r), 8);
      }
      for (int i = 0; i < 8000; ++i)
         keys.push_back("acct/" + std::to_string(rng() % 50000));

      // gen 0
      oracle_t o0;
      uint64_t v = 1;
      for (auto& k : keys)
      {
         t.cow_insert(std::string_view(k), v);
         o0[k] = v;
         ++v;
      }
      auto r0 = t.root_handle();
      REQUIRE(t.cow_retain(r0));  // snapshot 0 pinned

      // gen 1: overwrite half, add fresh keys
      oracle_t o1 = o0;
      size_t   n  = 0;
      for (auto& k : keys)
      {
         if (++n % 2)
            continue;
         t.cow_insert(std::string_view(k), v);
         o1[k] = v;
         ++v;
      }
      for (int i = 0; i < 2000; ++i)
      {
         std::string k = "fresh/" + std::to_string(i);
         t.cow_insert(std::string_view(k), v);
         o1[k] = v;
         ++v;
      }
      auto r1 = t.root_handle();
      REQUIRE(r1.raw() != r0.raw());  // the root was path-copied
      REQUIRE(t.cow_retain(r1));      // snapshot 1 pinned

      // gen 2: more churn on top
      oracle_t o2 = o1;
      for (int i = 0; i < 2000; ++i)
      {
         std::string k = "acct/" + std::to_string(i * 7 % 50000);
         t.cow_insert(std::string_view(k), v);
         o2[k] = v;
         ++v;
      }

      // Live tree sees gen 2; snapshots see their exact generations.
      verify_exact(t, o2);
      {
         tree_t s0{part_alloc<uint64_t>(&ses)};
         s0.adopt_root(r0);
         verify_exact(s0, o0);
         s0.adopt_root(decltype(r0)::null());
      }
      {
         tree_t s1{part_alloc<uint64_t>(&ses)};
         s1.adopt_root(r1);
         verify_exact(s1, o1);
         s1.adopt_root(decltype(r1)::null());
      }

      // Release snapshot 0: snapshot 1 and live stay intact.
      t.cow_release(r0);
      {
         tree_t s1{part_alloc<uint64_t>(&ses)};
         s1.adopt_root(r1);
         verify_exact(s1, o1);
         s1.adopt_root(decltype(r1)::null());
      }
      verify_exact(t, o2);

      t.cow_release(r1);
      verify_exact(t, o2);
      // tree dtor releases the live root's exclusive nodes via deallocate...
      // but the live root itself carries refcounts now: release it the COW way
      // and hand the dtor an empty tree.
      t.cow_release(t.root_handle());
      t.adopt_root(decltype(r0)::null());
   }
   // Every line came back: no leak, no double free, across all generations.
   uint64_t free_total = 0;
   for (uint32_t s = 0; s < a.num_segments(); ++s)
      free_total += a.freeset_of(s).free_lines();
   REQUIRE(free_total == uint64_t(a.num_segments()) * lines_per_segment);
}

TEST_CASE("cow on an unshared tree degenerates to in-place (no clones)")
{
   arena_config c;
   c.dir = tmp_dir("inplace");
   arena a(c);
   {
      auto   ses = a.start_session();
      auto   tx  = ses.start_transaction(0);
      tree_t t{part_alloc<uint64_t>(&ses)};
      for (int i = 0; i < 5000; ++i)
         t.cow_insert(std::string_view("k" + std::to_string(i)), uint64_t(i));
      const auto allocated_before = a.stats().bytes_allocated.load() + 0;
      const auto root_before      = t.root_handle().raw();
      // Overwrites on a ref-1 path: no privatization, root identity unchanged.
      for (int i = 0; i < 5000; ++i)
         t.cow_insert(std::string_view("k" + std::to_string(i)), uint64_t(i) + 9'000'000);
      REQUIRE(t.root_handle().raw() == root_before);
      (void)allocated_before;
      uint64_t out = 0;
      REQUIRE(t.find(std::string_view("k123"), out));
      REQUIRE(out == 9'000'123);
      t.cow_release(t.root_handle());
      t.adopt_root(decltype(t.root_handle())::null());
   }
   uint64_t free_total = 0;
   for (uint32_t s = 0; s < a.num_segments(); ++s)
      free_total += a.freeset_of(s).free_lines();
   REQUIRE(free_total == uint64_t(a.num_segments()) * lines_per_segment);
}

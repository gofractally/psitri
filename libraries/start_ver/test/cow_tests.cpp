// COW correctness under STAMP LIVENESS (docs/unified_tier_design.md):
// privatize-then-mutate over the verbatim hkv tree, with versions instead of
// refcounts. Generational snapshots must stay bit-stable while the live tree
// mutates; draining a version's clone-log frees exactly that generation's
// garbage; at the end the arena drains to exactly empty (no leak, no double
// free). The test drives the tree-level protocol directly — version eras are
// switched by hand (the kv layer does the same per transaction).
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <random>
#include <string>

#include <start_ver/hkv/tree.hpp>
#include <start_ver/tree_alloc.hpp>

using namespace start_ver;
namespace fs = std::filesystem;

using tree_t =
    start_ver::hkv::kv_tree<std::string_view, uint64_t, start_ver::hkv::mode::none, part_alloc<uint64_t>>;
using oracle_t = std::map<std::string, uint64_t>;

namespace
{
   fs::path tmp_dir(const char* tag)
   {
      static std::mt19937_64 rng(std::random_device{}());
      auto p = fs::temp_directory_path() /
               (std::string("vpart_cow_") + tag + "_" + std::to_string(rng()));
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

   // "commit" an era: detach the clone-log chain (it becomes the committed
   // version's supersession record). Returns the chain head.
   uint32_t era_commit(arena&, vlog_state& vl)
   {
      const uint32_t chain = vl.clone_head;
      vl.reset();
      return chain;
   }

   // drain one committed chain: free every superseded node it records
   // (shallow = the husk only; deep = cascade), then the chain lines.
   void drain_chain(arena& a, uint32_t head)
   {
      tree_t v{part_alloc<uint64_t>(&a, /*reclaim=*/true)};
      uint32_t line = head;
      while (line != vlog_state::npos)
      {
         const uint32_t* w    = static_cast<const uint32_t*>(a.get(line));
         const uint32_t  prev = w[0], n = w[1];
         for (uint32_t i = 0; i < n; ++i)
            v.vdrain_entry(w[2 + i * 2], w[3 + i * 2]);
         a.release_run(line, 1);
         line = prev;
      }
   }

   // teardown: cascade-free the live tree (reclaim context frees everything)
   void reclaim_tree(arena& a, tree_t& t)
   {
      tree_t v{part_alloc<uint64_t>(&a, /*reclaim=*/true)};
      v.cow_release(t.root_handle());
      t.adopt_root(decltype(t.root_handle())::null());
   }

   uint64_t free_lines_total(arena& a)
   {
      uint64_t free_total = 0;
      for (uint32_t s = 0; s < a.num_segments(); ++s)
         free_total += a.freeset_of(s).free_lines();
      return free_total;
   }
}  // namespace

TEST_CASE("snapshots stay stable across COW mutation; drains free exactly")
{
   arena_config c;
   c.dir = tmp_dir("gen");
   arena a(c);
   {
      auto       ses = a.start_session();
      auto       tx  = ses.start_transaction(0);
      vlog_state vl;
      tree_t     t{part_alloc<uint64_t>(&ses)};
      t.allocator_mut().vlog = &vl;
      t.allocator_mut().vtxn = 1;  // era 1

      std::mt19937_64          rng(123);
      std::vector<std::string> keys;
      for (int i = 0; i < 8000; ++i)
      {
         const uint64_t r = rng();
         keys.emplace_back(reinterpret_cast<const char*>(&r), 8);
      }
      for (int i = 0; i < 8000; ++i)
         keys.push_back("acct/" + std::to_string(rng() % 50000));

      // era 1: build gen 0. Everything superseded during the build is
      // era-born -> freed inline; the clone chain stays empty.
      oracle_t o0;
      uint64_t v = 1;
      for (auto& k : keys)
      {
         t.cow_insert(std::string_view(k), v);
         o0[k] = v;
         ++v;
      }
      const uint32_t chain1 = era_commit(a, vl);
      REQUIRE(chain1 == vlog_state::npos);  // no committed state existed yet
      auto r0 = t.root_handle();            // snapshot 0 = the era-1 boundary

      // era 2: overwrite half, add fresh keys
      t.allocator_mut().vtxn = 2;
      oracle_t o1            = o0;
      size_t   n             = 0;
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
      const uint32_t chain2 = era_commit(a, vl);
      auto           r1     = t.root_handle();  // snapshot 1 = the era-2 boundary
      REQUIRE(r1.raw() != r0.raw());            // the root was path-copied
      REQUIRE(chain2 != vlog_state::npos);      // gen-0 husks were logged

      // era 3: more churn on top
      t.allocator_mut().vtxn = 3;
      oracle_t o2            = o1;
      for (int i = 0; i < 2000; ++i)
      {
         std::string k = "acct/" + std::to_string(i * 7 % 50000);
         t.cow_insert(std::string_view(k), v);
         o2[k] = v;
         ++v;
      }
      const uint32_t chain3 = era_commit(a, vl);
      t.allocator_mut().vtxn = 0;

      // Live tree sees gen 2; snapshots see their exact generations.
      verify_exact(t, o2);
      {
         tree_t s0{part_alloc<uint64_t>(&a)};
         s0.adopt_root(r0);
         verify_exact(s0, o0);
         s0.adopt_root(decltype(r0)::null());
      }
      {
         tree_t s1{part_alloc<uint64_t>(&a)};
         s1.adopt_root(r1);
         verify_exact(s1, o1);
         s1.adopt_root(decltype(r1)::null());
      }

      // "Release snapshot 0" = drain era 2's chain (deaths at version 2 are
      // exactly what snapshot 0 could still see): snapshot 1 and live intact.
      drain_chain(a, chain2);
      {
         tree_t s1{part_alloc<uint64_t>(&a)};
         s1.adopt_root(r1);
         verify_exact(s1, o1);
         s1.adopt_root(decltype(r1)::null());
      }
      verify_exact(t, o2);

      // "Release snapshot 1" = drain era 3's chain: live intact.
      drain_chain(a, chain3);
      verify_exact(t, o2);

      // teardown: the live tree drains back to the arena
      reclaim_tree(a, t);
   }
   // Every line came back: no leak, no double free, across all generations.
   REQUIRE(free_lines_total(a) == uint64_t(a.num_segments()) * lines_per_segment);
}

TEST_CASE("cow within one era degenerates to in-place (no clones)")
{
   arena_config c;
   c.dir = tmp_dir("inplace");
   arena a(c);
   {
      auto       ses = a.start_session();
      auto       tx  = ses.start_transaction(0);
      vlog_state vl;
      tree_t     t{part_alloc<uint64_t>(&ses)};
      t.allocator_mut().vlog = &vl;
      t.allocator_mut().vtxn = 1;
      for (int i = 0; i < 5000; ++i)
         t.cow_insert(std::string_view("k" + std::to_string(i)), uint64_t(i));
      const auto root_before = t.root_handle().raw();
      // Overwrites inside the SAME era: stamp == txn -> no privatization,
      // root identity unchanged.
      for (int i = 0; i < 5000; ++i)
         t.cow_insert(std::string_view("k" + std::to_string(i)), uint64_t(i) + 9'000'000);
      REQUIRE(t.root_handle().raw() == root_before);
      uint64_t out = 0;
      REQUIRE(t.find(std::string_view("k123"), out));
      REQUIRE(out == 9'000'123);
      const uint32_t chain = era_commit(a, vl);
      REQUIRE(chain == vlog_state::npos);  // nothing old existed to supersede
      t.allocator_mut().vtxn = 0;
      reclaim_tree(a, t);
   }
   REQUIRE(free_lines_total(a) == uint64_t(a.num_segments()) * lines_per_segment);
}

TEST_CASE("abort semantics: the window walk frees the diverged era exactly")
{
   arena_config c;
   c.dir = tmp_dir("abort");
   arena a(c);
   {
      auto       ses = a.start_session();
      auto       tx  = ses.start_transaction(0);
      vlog_state vl;
      tree_t     t{part_alloc<uint64_t>(&ses)};
      t.allocator_mut().vlog = &vl;

      // era 1: committed base
      t.allocator_mut().vtxn = 1;
      oracle_t o0;
      for (int i = 0; i < 4000; ++i)
      {
         std::string k = "base/" + std::to_string(i);
         t.cow_insert(std::string_view(k), uint64_t(i));
         o0[k] = uint64_t(i);
      }
      (void)era_commit(a, vl);
      const auto r0 = t.root_handle();

      // era 2: diverge, then ABORT — the window walk frees everything era 2
      // created, the clone chain drops unpublished, the boundary root is
      // restored bit-stable.
      t.allocator_mut().vtxn = 2;
      for (int i = 0; i < 2000; ++i)
         t.cow_insert(std::string_view("base/" + std::to_string(i)), uint64_t(i) + 7'000'000);
      for (int i = 0; i < 500; ++i)
         t.cow_insert(std::string_view("new/" + std::to_string(i)), uint64_t(i));
      REQUIRE(t.root_handle().raw() != r0.raw());

      // free the diverged window by WALKING it: recursion stops at era-1
      // nodes (shared with the restored root) — no create-log exists at all
      t.vabort_release(t.root_handle());
      t.adopt_root(r0);
      {  // drop the clone chain unpublished (the superseded nodes LIVE ON)
         uint32_t line = vl.clone_head;
         while (line != vlog_state::npos)
         {
            const uint32_t prev = *static_cast<const uint32_t*>(a.get(line));
            a.release_run(line, 1);
            line = prev;
         }
      }
      vl.reset();
      t.allocator_mut().vtxn = 0;

      // the committed generation is bit-stable after the rollback
      verify_exact(t, o0);
      reclaim_tree(a, t);
   }
   REQUIRE(free_lines_total(a) == uint64_t(a.num_segments()) * lines_per_segment);
}

TEST_CASE("wide routers carry inline prefixes (see-header), never prefix_nodes")
{
   arena_config c;
   c.dir = tmp_dir("wpfx");
   arena a(c);
   {
      auto       ses = a.start_session();
      auto       tx  = ses.start_transaction(0);
      vlog_state vl;
      tree_t     t{part_alloc<uint64_t>(&ses)};
      t.allocator_mut().vlog = &vl;
      t.allocator_mut().vtxn = 1;

      oracle_t o;
      auto     put = [&](const std::string& k, uint64_t v)
      {
         t.cow_insert(std::string_view(k), v);
         o[k] = v;
      };
      // a shared prefix with fanout >16 forces setlist -> c2 widening UNDER
      // the prefix: the prefix must ride the wide node's CL0 (K::hdr), and a
      // prefix_node here is the bug Dan named.
      for (int i = 0; i < 40; ++i)
         put(std::string("shared/") + char('A' + i) + "tail", uint64_t(i));
      // grow one branch deeper so iteration crosses the hdr node repeatedly
      for (int i = 0; i < 30; ++i)
         put(std::string("shared/Adeep/") + std::to_string(i), uint64_t(1000 + i));

      uint64_t hdr_nodes = 0, pfx_nodes = 0;
      t.census([&](auto row) {
         if (unsigned(row.kind) == 12) ++hdr_nodes;          // K::hdr
         if (unsigned(row.kind) == 2)  ++pfx_nodes;          // K::prefix_node
      });
      REQUIRE(hdr_nodes >= 1);  // the widened router carries the prefix itself
      REQUIRE(pfx_nodes == 0);  // short prefixes never spawn prefix_nodes
      verify_exact(t, o);

      // era 2: COW across the boundary — clone of an hdr node, privatize
      // through its prefix, and a SPLIT INSIDE the wide prefix
      const uint32_t chain1 = era_commit(a, vl);
      REQUIRE(chain1 == vlog_state::npos);
      auto r0 = t.root_handle();
      auto o0 = o;
      t.allocator_mut().vtxn = 2;
      for (int i = 0; i < 40; i += 2)
         put(std::string("shared/") + char('A' + i) + "tail", uint64_t(5000 + i));
      put("shaq", 42);  // diverges INSIDE "shared/" -> split_hdr_prefix_
      put("sh",   43);  // key ends inside the prefix -> term inside split router
      const uint32_t chain2 = era_commit(a, vl);
      t.allocator_mut().vtxn = 0;
      verify_exact(t, o);
      {
         tree_t s0{part_alloc<uint64_t>(&a)};
         s0.adopt_root(r0);
         verify_exact(s0, o0);  // the boundary is bit-stable across hdr clones
         s0.adopt_root(decltype(r0)::null());
      }
      drain_chain(a, chain2);
      verify_exact(t, o);

      // era 3: remove most branches -> de-widen carries the prefix back down
      t.allocator_mut().vtxn = 3;
      for (int i = 4; i < 40; ++i)
      {
         std::string k = std::string("shared/") + char('A' + i) + "tail";
         if (t.cow_remove(std::string_view(k)))
            o.erase(k);
      }
      const uint32_t chain3 = era_commit(a, vl);
      t.allocator_mut().vtxn = 0;
      verify_exact(t, o);
      drain_chain(a, chain3);
      verify_exact(t, o);

      reclaim_tree(a, t);
   }
   REQUIRE(free_lines_total(a) == uint64_t(a.num_segments()) * lines_per_segment);
}

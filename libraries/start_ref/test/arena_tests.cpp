// SPEC §15 tests 3-6 — arena: growth/claiming/qualifying, txn floor semantics,
// commit msync + write-amp stats, refcount bands + read-session slots, and the
// commit-ordering protocol under SIGKILL (root flipped => new data, not
// flipped => old tree intact).
#include <catch2/catch_test_macros.hpp>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#include <start_ref/arena.hpp>

using namespace start_ref;
namespace fs = std::filesystem;

namespace
{
   fs::path tmp_dir(const char* tag)
   {
      static std::mt19937_64 rng(std::random_device{}());
      auto p = fs::temp_directory_path() / (std::string("part_") + tag + "_" +
                                            std::to_string(rng()));
      fs::remove_all(p);
      return p;
   }
   arena_config cfg(const fs::path& dir, sync_type st = sync_type::none,
                    uint32_t grow_batch = 1)
   {
      arena_config c;
      c.dir        = dir;
      c.grow_batch = grow_batch;
      c.sync_mode  = st;
      return c;
   }
}  // namespace

TEST_CASE("fresh arena grows on demand and fills bottom-up")
{
   arena a(cfg(tmp_dir("grow")));
   REQUIRE(a.num_segments() == 0);
   auto ses = a.start_session();
   auto tx  = ses.start_transaction(0);
   REQUIRE(tx.alloc_lines(1) == 0);
   REQUIRE(a.num_segments() == 1);
   REQUIRE(tx.alloc_lines(1) == 1);
   // Ascending and word-confined; the session word-window serves multi-line
   // runs from the first contiguous fit (contiguity, not class alignment).
   const line_idx r8 = tx.alloc_lines(8);
   REQUIRE(r8 == 2);
   REQUIRE((r8 % 64) + 8 <= 64);
   REQUIRE(a.ref(0) == 1);
   REQUIRE(a.stats().segments_grown.load() == 1);
}

TEST_CASE("refcount bands: writer 190, pins to 254, sentinel throws at 255")
{
   arena a(cfg(tmp_dir("refs")));
   auto  ses = a.start_session();
   auto  tx  = ses.start_transaction(0);
   const line_idx i = tx.alloc_lines(1);
   REQUIRE(a.ref(i) == 1);

   uint32_t retained = 0;
   while (a.retain(i))
      ++retained;
   REQUIRE(retained == writer_clone_at - 1);  // 1 -> 190, refused at 190
   REQUIRE(a.ref(i) == writer_clone_at);
   REQUIRE(a.stats().clones_forced.load() == 1);

   for (uint32_t k = writer_clone_at; k < ref_ceiling; ++k)
      a.add_ref_or_throw(i);  // pins: 190 -> 254
   REQUIRE(a.ref(i) == ref_ceiling);
   REQUIRE_THROWS_AS(a.add_ref_or_throw(i), ref_overflow);
   REQUIRE(a.ref(i) == ref_ceiling);  // count undamaged by the throw

   bool died = false;
   for (uint32_t k = ref_ceiling; k >= 1; --k)
      died = a.release(i);
   REQUIRE(died);
   REQUIRE(tx.alloc_lines(1) == i);  // the freed line is the lowest hole again
}

TEST_CASE("txn floor: holes behind the floor page are invisible until recycle")
{
   arena a(cfg(tmp_dir("floor")));
   auto  ses = a.start_session();
   {
      auto tx = ses.start_transaction(0);
      for (int k = 0; k < 25; ++k)
         tx.alloc_lines(8);  // lines 0..199, fill_end = 200
      tx.commit();           // floor -> page 1 (line 128)
   }
   a.release(5);    // hole on page 0 — behind next txn's floor
   a.release(130);  // hole on page 1 — at/after the floor
   auto tx = ses.start_transaction(0);
   REQUIRE(tx.alloc_lines(1) == 130);  // first-fit from the floor, not from 0
   REQUIRE(tx.alloc_lines(1) == 200);
   REQUIRE(a.freeset_of(0).is_free(5));  // the behind-floor hole survives untouched
}

TEST_CASE("mid-txn frees at/after the floor are recycled immediately")
{
   arena a(cfg(tmp_dir("hot")));
   auto  ses = a.start_session();
   auto  tx  = ses.start_transaction(0);
   const line_idx x = tx.alloc_lines(4);
   tx.alloc_lines(4);
   a.release(x);  // refcount 1 -> 0: COW churn of a this-txn node
   a.release(x + 1);
   a.release(x + 2);
   a.release(x + 3);
   REQUIRE(tx.alloc_lines(4) == x);  // hot hole reused at zero sync cost
}

TEST_CASE("read-session slots: 64 live, 65th refused, slot recycles")
{
   arena a(cfg(tmp_dir("slots")));
   {
      std::vector<read_session> rs;
      for (int k = 0; k < 64; ++k)
         rs.push_back(a.start_read_session());
      REQUIRE_THROWS(a.start_read_session());
      rs.pop_back();
      rs.push_back(a.start_read_session());  // freed slot is reacquirable
   }
   auto again = a.start_read_session();  // all returned on scope exit
}

TEST_CASE("qualifying: every-other-line 50% free never qualifies; 8-contig does")
{
   arena a(cfg(tmp_dir("qual")));
   auto  ses = a.start_session();
   auto  tx  = ses.start_transaction(0);
   auto  fill_segment = [&] {
      for (uint32_t k = 0; k < lines_per_segment / 64; ++k)
         tx.alloc_lines(64);
   };
   fill_segment();  // seg 0
   fill_segment();  // seg 1 (grow + claim)
   REQUIRE(a.num_segments() == 2);

   // Free every other line of segment 0: 50% free, zero free 8-groups.
   for (line_idx i = 0; i < lines_per_segment; i += 2)
      a.release(i);
   REQUIRE(a.freeset_of(0).free_lines() == lines_per_segment / 2);
   REQUIRE(a.freeset_of(0).free_groups8() == 0);

   fill_segment();  // seg 1 exhausts mid-way; must GROW (seg 2), not claim seg 0
   REQUIRE(a.num_segments() == 3);

   // Now free the other half: fully free, qualifies.
   for (line_idx i = 1; i < lines_per_segment; i += 2)
      a.release(i);
   REQUIRE(a.freeset_of(0).free_groups8() == lines_per_segment / 8);

   fill_segment();  // exhausts seg 2 -> claims seg 0 instead of growing
   REQUIRE(a.num_segments() == 3);
   REQUIRE(a.stats().segments_reused.load() >= 1);
}

TEST_CASE("commit syncs only dirty runs: fresh-append write amp ~ 1.0")
{
   arena a(cfg(tmp_dir("amp"), sync_type::msync_sync));
   auto  ses = a.start_session();
   auto  tx  = ses.start_transaction(0);
   for (int k = 0; k < 1000; ++k)
   {
      const line_idx i = tx.alloc_lines(1);
      std::memset(a.get(i), 0xAB, line_bytes);
   }
   tx.commit();
   const auto& st = a.stats();
   REQUIRE(st.bytes_synced.load() == 8 * disk_page_bytes);  // 1000 lines = 7.8 pages
   REQUIRE(st.sync_calls.load() == 1);                      // one coalesced run
   REQUIRE(a.stats().write_amp() < 1.1);
}

TEST_CASE("durability protocol: root flipped => data present; not flipped => old root")
{
   const auto dir = tmp_dir("kill");
   for (int flip_root : {0, 1})
   {
      fs::remove_all(dir);
      int pipefd[2];
      REQUIRE(pipe(pipefd) == 0);
      const pid_t pid = fork();
      REQUIRE(pid >= 0);
      if (pid == 0)
      {  // child: write a recognizable record, commit, optionally flip the root
         arena a(cfg(dir, sync_type::full));
         auto  ses = a.start_session();
         auto  tx  = ses.start_transaction(0);
         const line_idx i = tx.alloc_lines(1);
         std::memset(a.get(i), 0x5A, line_bytes);
         tx.commit();
         if (flip_root)
            a.set_root(0, 0xBEEF0000ull | i);
         char ready = 1;
         (void)!write(pipefd[1], &ready, 1);
         for (;;)
            pause();  // parent SIGKILLs us — no clean shutdown, no dtor
      }
      char ready = 0;
      REQUIRE(read(pipefd[0], &ready, 1) == 1);
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
      close(pipefd[0]);
      close(pipefd[1]);

      arena a(cfg(dir, sync_type::full));
      REQUIRE_FALSE(a.was_clean());  // SIGKILL skipped the clean-shutdown path
      if (flip_root)
      {
         const uint64_t root = a.get_root(0);
         REQUIRE((root >> 16) == 0xBEEF);
         const line_idx i = line_idx(root & 0xFFFF);
         const auto*    p = static_cast<const uint8_t*>(a.get(i));
         for (uint32_t b = 0; b < line_bytes; ++b)
            REQUIRE(p[b] == 0x5A);
      }
      else
         REQUIRE(a.get_root(0) == 0);  // commit without set_root leaves the old root
   }
}

TEST_CASE("reopen after clean close trusts the freeset; rebuild reconstructs it")
{
   const auto dir = tmp_dir("reopen");
   line_idx   keep = 0;
   {
      arena a(cfg(dir, sync_type::msync_sync));
      auto  ses = a.start_session();
      auto  tx  = ses.start_transaction(0);
      keep = tx.alloc_lines(3);
      tx.alloc_lines(1);   // lands at keep+3 (ascending fill)
      a.release(keep + 3); // and is freed again
      tx.commit();
   }  // clean dtor: clean flag set
   {
      arena a(cfg(dir, sync_type::msync_sync));
      REQUIRE(a.was_clean());
      REQUIRE(a.num_segments() == 1);
      REQUIRE_FALSE(a.freeset_of(0).is_free(0));
      REQUIRE(a.freeset_of(0).is_free(3));
      // Synthetic recovery: only `keep`'s 3 lines are reachable.
      a.rebuild_begin();
      a.mark_line(keep);
      a.mark_line(keep + 1);
      a.mark_line(keep + 2);
      a.mark_line(keep);  // second parent: refcount bumps, no re-claim
      a.rebuild_end();
      REQUIRE(a.ref(keep) == 2);
      REQUIRE(a.ref(keep + 1) == 1);
      REQUIRE_FALSE(a.freeset_of(0).is_free(keep));
      REQUIRE(a.freeset_of(0).is_free(3));
      REQUIRE(a.freeset_of(0).free_lines() == lines_per_segment - 3);
   }
}

namespace
{
   // One segment, every line allocated, then every other line freed: plenty
   // of free lines and not one run of two.
   void fragment_whole_arena(arena& a, session& ses)
   {
      auto tx = ses.start_transaction(0);
      for (uint32_t k = 0; k < lines_per_segment; ++k)
         tx.alloc_lines(1);
      tx.commit();
      for (line_idx i = 0; i < lines_per_segment; i += 2)
         a.release(i);
   }
   arena_config one_segment(const fs::path& dir)
   {
      arena_config c  = cfg(dir);
      c.max_segments  = 1;
      c.reserve_lines = 0;
      return c;
   }
}  // namespace

TEST_CASE("a session whose claim failed holds no segment", "[arena][limit]")
{
   arena a(one_segment(tmp_dir("noclaim")));
   auto  first = a.start_session();
   fragment_whole_arena(a, first);
   {
      auto tx = first.start_transaction(0);
      REQUIRE_THROWS_AS(tx.alloc_lines(2), arena_full);  // no run of two anywhere
      // Another session now takes the only segment and works in it.
      auto second = a.start_session();
      auto tx2    = second.start_transaction(0);
      tx2.alloc_lines(1);
      REQUIRE(a.segment_state(0) == arena::SEG_OWNED);
      // The first session ending its transaction, or going away, must not
      // touch a segment it no longer holds.
      tx.close();
      REQUIRE(a.segment_state(0) == arena::SEG_OWNED);
      tx2.alloc_lines(1);
      tx2.commit();
   }
}

TEST_CASE("an allocation that can never fit gives up even if reclaim keeps reporting work",
          "[arena][limit]")
{
   arena a(one_segment(tmp_dir("spin")));
   auto  ses = a.start_session();
   fragment_whole_arena(a, ses);
   static uint64_t calls;
   calls = 0;
   // A reclaim hook that always says it freed something, as it would while
   // another writer keeps queueing releases, and frees nothing this
   // allocation can use.
   a.set_reclaim([](void*, bool) noexcept { ++calls; return true; }, nullptr);
   auto tx = ses.start_transaction(0);
   REQUIRE_THROWS_AS(tx.alloc_lines(2), arena_full);
   REQUIRE(calls > 0);
   REQUIRE(calls < 1000);
   REQUIRE(tx.alloc_lines(1) != null_idx);  // single lines still fit
   a.set_reclaim(nullptr, nullptr);
}

TEST_CASE("the reserve is kept from ordinary allocations and given to reclaiming ones",
          "[arena][limit]")
{
   arena_config c  = cfg(tmp_dir("reserve"));
   c.max_segments  = 1;
   c.reserve_lines = 1000;
   arena a(c);
   auto  ses = a.start_session();
   auto  tx  = ses.start_transaction(0);
   uint32_t got = 0;
   try
   {
      for (;; ++got)
         tx.alloc_lines(1);
   }
   catch (const arena_full&)
   {
   }
   // ordinary allocation stopped at the reserve, give or take one window
   REQUIRE(got >= lines_per_segment - 1000 - 64);
   REQUIRE(got <= lines_per_segment - 1000 + 64);
   REQUIRE_THROWS_AS(tx.alloc_lines(1), arena_full);
   {
      auto use = ses.use_reserve();
      for (int k = 0; k < 500; ++k)
         REQUIRE(tx.alloc_lines(1) != null_idx);
   }
   // without the guard: at most the session's private window, then refused
   uint32_t extra = 0;
   try
   {
      for (; extra < 1000; ++extra)
         tx.alloc_lines(1);
   }
   catch (const arena_full&)
   {
   }
   REQUIRE(extra < 64);
}

TEST_CASE("a segment left idle by one session can be taken by another at the ceiling",
          "[arena][limit]")
{
   arena_config c  = cfg(tmp_dir("idle"));
   c.max_segments  = 2;
   c.reserve_lines = 0;
   arena a(c);
   auto  s1 = a.start_session();
   auto  s2 = a.start_session();
   {
      auto t1 = s1.start_transaction(0);
      t1.alloc_lines(1);
      t1.commit();
      auto t2 = s2.start_transaction(0);
      t2.alloc_lines(1);
      t2.commit();
   }
   REQUIRE(a.num_segments() == 2);
   auto s3 = a.start_session();
   auto t3 = s3.start_transaction(0);
   line_idx got = null_idx;
   REQUIRE_NOTHROW(got = t3.alloc_lines(1));
   t3.commit();
   t3.close();  // idle again: three sessions, two segments, two at a time
   // the sessions it was taken from carry on, somewhere
   auto t1 = s1.start_transaction(0);
   auto t2 = s2.start_transaction(0);
   const line_idx l1 = t1.alloc_lines(1), l2 = t2.alloc_lines(1);
   REQUIRE(l1 != l2);
   REQUIRE(l1 != got);
   REQUIRE(l2 != got);
}

TEST_CASE("a run recycled from a segment the session has left does not move its fill position",
          "[arena][segments]")
{
   arena_config c  = cfg(tmp_dir("fillpos"));
   c.max_segments  = 2;
   c.reserve_lines = 0;
   arena a(c);
   auto  ses = a.start_session();
   {
      // fill segment 0, spill into segment 1
      auto tx = ses.start_transaction(0);
      for (uint32_t k = 0; k < lines_per_segment + 64; ++k)
         tx.alloc_lines(1);
      tx.commit();
   }
   REQUIRE(a.num_segments() == 2);
   for (line_idx i = 0; i < lines_per_segment; ++i)
      a.release(i);  // segment 0 is empty again and qualifies for reuse
   line_idx parked = null_idx;
   {
      auto tx = ses.start_transaction(0);
      parked  = tx.alloc_lines(1);
      REQUIRE(a.segment_of(parked) == 1);
      ses.release_run(parked, 1);  // freed this transaction: parked in the hot-hole cache
      // use up segment 1 in pairs (the parked single stays parked), moving to segment 0
      line_idx g = 0;
      do
         g = tx.alloc_lines(2);
      while (a.segment_of(g) == 1);
      REQUIRE(a.segment_of(g) == 0);
      // the parked line, in segment 1, comes back out while the session is in segment 0
      REQUIRE(tx.alloc_lines(1) == parked);
      tx.commit();
   }
   // The next transaction carries on in segment 0. If the recycled line had
   // dragged the fill position up into segment 1, its floor would lie outside
   // segment 0 and the first allocation would abandon the segment.
   const uint64_t switches = a.stats().range_switches.load();
   auto           tx       = ses.start_transaction(0);
   for (int k = 0; k < 200; ++k)
      REQUIRE(a.segment_of(tx.alloc_lines(2)) == 0);
   REQUIRE(a.stats().range_switches.load() == switches);
}


// gofractally/psitri#39: the reserve is contiguous room, not a count of free
// lines. Every ordinary line taken and every other one freed again leaves
// thousands of free lines and not one run of two; a reclaiming allocation of
// two or more lines still succeeds, out of the reserve.
TEST_CASE("the reserve holds runs when the rest of the arena has none", "[arena][limit][reserve]")
{
   arena_config c  = cfg(tmp_dir("rsvrun"));
   c.max_segments  = 1;
   c.reserve_lines = 1024;
   arena a(c);
   auto  ses = a.start_session();
   auto  tx  = ses.start_transaction(0);
   std::vector<line_idx> taken;
   try
   {
      for (;;)
         taken.push_back(tx.alloc_lines(1));
   }
   catch (const arena_full&)
   {
   }
   REQUIRE(taken.size() == lines_per_segment - 1024);  // all of it, and none of the reserve
   for (line_idx i : taken)
      REQUIRE(i < lines_per_segment - 1024);
   tx.commit();
   tx.close();
   for (size_t k = 0; k < taken.size(); k += 2)
      a.release(taken[k]);
   REQUIRE(a.used_bytes() == uint64_t(taken.size() / 2) * line_bytes);
   REQUIRE(a.reserve_free_lines() == 1024);

   auto tx2 = ses.start_transaction(0);
   REQUIRE_THROWS_AS(tx2.alloc_lines(2), arena_full);  // ordinary: no run of two anywhere
   REQUIRE(tx2.alloc_lines(1) != null_idx);            // single lines there are
   std::vector<std::pair<line_idx, uint32_t>> lent;
   {
      auto use = ses.use_reserve();
      for (uint32_t n : {2u, 3u, 5u, 8u, 16u, 64u, 2u, 7u})
      {
         const line_idx g = tx2.alloc_lines(n);
         REQUIRE(g >= lines_per_segment - 1024);  // from the reserve
         REQUIRE((g % 64) + n <= 64);             // one map word, like any allocation
         lent.emplace_back(g, n);
      }
   }
   REQUIRE(a.reserve_free_lines() == 1024 - 107);
   tx2.commit();
   tx2.close();
   // lines of the reserve go back to the reserve, not to the ordinary map
   const uint64_t ordinary_used = a.used_bytes();
   for (auto [g, n] : lent)
      a.release_run(g, n);
   REQUIRE(a.reserve_free_lines() == 1024);
   REQUIRE(a.used_bytes() == ordinary_used - 107 * line_bytes);
   auto tx3 = ses.start_transaction(0);
   REQUIRE_THROWS_AS(tx3.alloc_lines(2), arena_full);
}

namespace
{
   // One segment, every ordinary line taken in runs of `run`, committed.
   std::vector<line_idx> take_everything(session& ses, uint32_t run)
   {
      std::vector<line_idx> taken;
      auto                  tx = ses.start_transaction(0);
      try
      {
         for (;;)
            taken.push_back(tx.alloc_lines(run));
      }
      catch (const arena_full&)
      {
      }
      tx.commit();
      tx.close();
      return taken;
   }
   arena_config one_segment_with_reserve(const fs::path& dir)
   {
      arena_config c  = cfg(dir);
      c.max_segments  = 1;
      c.reserve_lines = 1024;
      return c;
   }
}  // namespace

// What a remove draws from the reserve is owed, and the next frees repay it
// before the free map sees them: with a run of the same length when one is
// freed, and line for line when only shorter ones are.
TEST_CASE("what the reserve lends is repaid by the next frees", "[arena][limit][reserve]")
{
   arena a(one_segment_with_reserve(tmp_dir("repay")));
   auto  ses   = a.start_session();
   auto  taken = take_everything(ses, 4);  // ordinary data in runs of four
   REQUIRE(a.reserve_free_lines() == 1024);
   REQUIRE(a.reserve_owed_lines() == 0);
   REQUIRE(a.reserve_longest_run() == 64);
   const uint64_t used = a.used_bytes();

   auto tx = ses.start_transaction(0);
   std::vector<std::pair<line_idx, uint32_t>> drawn;
   {
      auto use = ses.use_reserve();
      for (uint32_t n : {2u, 3u, 4u, 1u, 2u})
         drawn.emplace_back(tx.alloc_lines(n), n);
   }
   REQUIRE(a.reserve_free_lines() == 1024 - 12);
   REQUIRE(a.reserve_owed_lines() == 12);
   REQUIRE(a.used_bytes() == used + 12 * line_bytes);  // what was drawn is ordinary data now
   tx.commit();
   tx.close();

   // Three ordinary four-line runs are freed (the "old versions"). The
   // reserve takes exactly what it is owed, 4 + 3 + 2 + 2 + 1, and the
   // free map gets nothing.
   for (int k = 0; k < 3; ++k)
      a.release_run(taken[k], 4);
   REQUIRE(a.reserve_free_lines() == 1024);
   REQUIRE(a.reserve_owed_lines() == 0);
   REQUIRE(a.used_bytes() == used);
   {
      auto tx2 = ses.start_transaction(0);
      REQUIRE_THROWS_AS(tx2.alloc_lines(1), arena_full);  // none of it reached the free map
   }
   // nothing owed: the next free is an ordinary free
   a.release_run(taken[3], 4);
   REQUIRE(a.reserve_free_lines() == 1024);
   REQUIRE(a.used_bytes() == used - 4 * line_bytes);
   {
      auto tx2 = ses.start_transaction(0);
      REQUIRE(tx2.alloc_lines(4) == taken[3]);
   }
   // what the reserve can lend is what it could before: the same runs again
   {
      auto tx2 = ses.start_transaction(0);
      auto use = ses.use_reserve();
      for (uint32_t n : {2u, 3u, 4u, 1u, 2u})
         REQUIRE(tx2.alloc_lines(n) != null_idx);
      REQUIRE(tx2.alloc_lines(64) != null_idx);  // and whole words are still there
   }
}

// Only shorter runs come free than the one that was lent: the reserve's line
// count still comes back, and the debt is settled in pieces.
TEST_CASE("the reserve's line count is restored by shorter runs too", "[arena][limit][reserve]")
{
   arena a(one_segment_with_reserve(tmp_dir("pieces")));
   auto  ses   = a.start_session();
   auto  taken = take_everything(ses, 1);
   {
      auto tx  = ses.start_transaction(0);
      auto use = ses.use_reserve();
      REQUIRE(tx.alloc_lines(8) != null_idx);
      tx.commit();
   }
   REQUIRE(a.reserve_owed_lines() == 8);
   for (int k = 0; k < 16; k += 2)  // eight single lines, none adjacent
      a.release(taken[k]);
   REQUIRE(a.reserve_owed_lines() == 0);
   REQUIRE(a.reserve_free_lines() == 1024);
}

// A run drawn from the reserve and freed again inside the same transaction
// goes back to the reserve. It must not wait in the session's private cache,
// where the next ordinary allocation of that size would take it.
TEST_CASE("a reserve run freed in its own transaction is not handed to an ordinary allocation",
          "[arena][limit][reserve]")
{
   arena a(one_segment_with_reserve(tmp_dir("samettx")));
   auto  ses   = a.start_session();
   auto  taken = take_everything(ses, 1);
   auto  tx    = ses.start_transaction(0);
   line_idx g = null_idx;
   {
      auto use = ses.use_reserve();
      g        = tx.alloc_lines(2);
   }
   REQUIRE(a.reserve_free_lines() == 1022);
   ses.release_run(g, 2);  // e.g. the node was replaced later in the transaction
   REQUIRE(a.reserve_free_lines() == 1024);
   REQUIRE(a.reserve_owed_lines() == 0);
   REQUIRE_THROWS_AS(tx.alloc_lines(2), arena_full);  // an ordinary allocation gets nothing of it
}

// Nothing about the reserve is kept across an open: it goes back to the free
// map at a clean close and is collected again, longest runs first. A store
// too full to fill it says so, and the first lines freed go to it.
TEST_CASE("the reserve is collected again at every open", "[arena][limit][reserve]")
{
   const auto            dir = tmp_dir("reopen-reserve");
   std::vector<line_idx> taken;
   line_idx              lent = null_idx;
   {
      arena a(one_segment_with_reserve(dir));
      auto  ses = a.start_session();
      taken     = take_everything(ses, 1);
      auto tx   = ses.start_transaction(0);
      auto use  = ses.use_reserve();
      lent      = tx.alloc_lines(40);  // still lent at the close: 40 lines owed
      tx.commit();
      REQUIRE(a.reserve_owed_lines() == 40);
   }
   {
      arena a(one_segment_with_reserve(dir));
      REQUIRE(a.was_clean());
      // 984 lines were free; that is all there is, and the rest is owed
      REQUIRE(a.reserve_free_lines() == 984);
      REQUIRE(a.reserve_owed_lines() == 40);
      REQUIRE(a.reserve_longest_run() == 64);
      auto ses = a.start_session();
      {
         auto tx = ses.start_transaction(0);
         REQUIRE_THROWS_AS(tx.alloc_lines(1), arena_full);
      }
      // Fifty single lines come free, none adjacent. The flexible stock's
      // share of the shortfall is repaid; what the guaranteed runs are short
      // of stays owed, since single lines cannot make a run; the rest is
      // ordinary room.
      for (int k = 0; k < 100; k += 2)
         a.release(taken[k]);
      const uint64_t still_owed = a.reserve_owed_lines();
      REQUIRE(still_owed < 40);
      REQUIRE(a.reserve_free_lines() == 1024 - still_owed);
      auto     tx       = ses.start_transaction(0);
      uint32_t ordinary = 0;
      try
      {
         for (;; ++ordinary)
            tx.alloc_lines(1);
      }
      catch (const arena_full&)
      {
      }
      REQUIRE(ordinary == 10 + still_owed);
   }
   {
      // 984 lines in the runs the reserve gave back plus what the frees
      // repaid: fewer than 1,024 are free, so it is still short, by exactly
      // what ordinary allocation took
      arena a(one_segment_with_reserve(dir));
      REQUIRE(a.reserve_free_lines() + a.reserve_owed_lines() == 1024);
      (void)lent;
   }
}

// The property the reserve is built for: no number of short draws takes away
// its ability to serve a longer one. Single lines are drawn until refused;
// after that every length from 2 to 16 can still be drawn as often as its
// quota says, and one run of 64.
TEST_CASE("short draws cannot use up the reserve's longer runs", "[arena][limit][reserve]")
{
   arena_config c  = cfg(tmp_dir("classes"));
   c.max_segments  = 1;
   c.reserve_lines = 2048;
   arena a(c);
   auto  ses   = a.start_session();
   auto  taken = take_everything(ses, 1);
   REQUIRE(a.reserve_free_lines() == 2048);
   REQUIRE(a.reserve_quota(2) == 16);
   REQUIRE(a.reserve_quota(8) == 4);
   REQUIRE(a.reserve_quota(14) == 2);
   REQUIRE(a.reserve_quota(16) == 2);
   REQUIRE(a.reserve_big_quota() == 1);

   auto tx  = ses.start_transaction(0);
   auto use = ses.use_reserve();
   uint32_t singles = 0;
   try
   {
      for (;; ++singles)
         tx.alloc_lines(1);
   }
   catch (const arena_full&)
   {
   }
   // everything flexible, and nothing else: a little over three quarters
   REQUIRE(singles > 1500);
   REQUIRE(singles < 1600);
   REQUIRE(a.reserve_free_lines() == 2048 - singles);
   REQUIRE(a.reserve_longest_run() == 64);
   uint32_t h[max_alloc_lines + 1];
   a.reserve_histogram(h);
   REQUIRE(h[1] == 0);
   for (uint32_t k = 2; k <= 16; ++k)
   {
      INFO("length " << k);
      REQUIRE(h[k] == a.reserve_quota(k));
      for (uint32_t q = 0; q < a.reserve_quota(k); ++q)
         REQUIRE(tx.alloc_lines(k) != null_idx);
      REQUIRE_THROWS_AS(tx.alloc_lines(k), arena_full);  // its own quota, and no other length's
   }
   REQUIRE(tx.alloc_lines(40) != null_idx);  // the long run, for anything over 16
   REQUIRE(a.reserve_longest_run() == 24);   // what was left of it is flexible now
   REQUIRE_THROWS_AS(tx.alloc_lines(30), arena_full);
}

// A guaranteed run's place is refilled only by a freed run at least as long:
// short frees top up the flexible stock and leave the longer debt standing.
TEST_CASE("a guaranteed run is repaid only by a run as long", "[arena][limit][reserve]")
{
   arena_config c  = cfg(tmp_dir("inkind"));
   c.max_segments  = 1;
   c.reserve_lines = 1024;
   arena a(c);
   auto  ses    = a.start_session();
   auto  singles = take_everything(ses, 1);
   {
      auto tx  = ses.start_transaction(0);
      auto use = ses.use_reserve();
      try  // use up the flexible stock
      {
         for (;;)
            tx.alloc_lines(64);
      }
      catch (const arena_full&)
      {
      }
      try
      {
         for (;;)
            tx.alloc_lines(1);
      }
      catch (const arena_full&)
      {
      }
      REQUIRE(a.reserve_quota(9) == 1);
      REQUIRE(tx.alloc_lines(9) != null_idx);            // the guaranteed nine-line run
      REQUIRE_THROWS_AS(tx.alloc_lines(9), arena_full);
      tx.commit();
   }
   const uint64_t owed = a.reserve_owed_lines();
   const uint32_t long_runs = a.reserve_runs_at_least(9);  // lengths 10 to 16, untouched
   // eighteen single lines come free, none adjacent: the flexible stock takes
   // them, the nine-line debt stands
   for (int k = 0; k < 36; k += 2)
      a.release(singles[k]);
   REQUIRE(a.reserve_owed_lines() == owed - 18);
   REQUIRE(a.reserve_runs_at_least(9) == long_runs);
   {
      auto tx  = ses.start_transaction(0);
      auto use = ses.use_reserve();
      REQUIRE_THROWS_AS(tx.alloc_lines(9), arena_full);
      REQUIRE(tx.alloc_lines(1) != null_idx);
   }
   // twelve adjacent lines come free as one run (lines 100..111 were
   // allocated one at a time; release_run frees them together): now it is repaid
   a.release_run(singles[100], 12);
   REQUIRE(a.reserve_runs_at_least(9) == long_runs + 1);
   auto tx  = ses.start_transaction(0);
   auto use = ses.use_reserve();
   REQUIRE(tx.alloc_lines(9) != null_idx);
}

namespace
{
   // what the reserve holds plus what it is owed is always its size
   void require_reserve_adds_up(arena& a)
   {
      REQUIRE(a.reserve_free_lines() + a.reserve_owed_lines() == a.config().reserve_lines);
   }
}  // namespace

// The long part of the reserve: a 64-line run for draws over 16 lines. What
// a draw leaves of it stays in the long part, and a run that comes back is
// joined to what it was cut from, so the 64 lines are whole again. Lines are
// never lost: after every step, held plus owed is the reserve's size.
TEST_CASE("the reserve's 64-line run is whole again when what was cut from it returns",
          "[arena][limit][reserve]")
{
   arena_config c  = cfg(tmp_dir("longpart"));
   c.max_segments  = 1;
   c.reserve_lines = 2048;
   arena a(c);
   auto  ses   = a.start_session();
   auto  taken = take_everything(ses, 1);
   REQUIRE(a.reserve_big_quota() == 1);
   require_reserve_adds_up(a);
   auto     tx  = ses.start_transaction(0);
   auto     use = ses.use_reserve();
   // use up the flexible stock's long runs, so that long draws reach the long part
   uint32_t flexible = 0;
   while (a.reserve_runs_at_least(17) > 1)
   {
      tx.alloc_lines(64);
      flexible += 64;
   }
   require_reserve_adds_up(a);
   REQUIRE(a.reserve_longest_run() == 64);

   const line_idx first = tx.alloc_lines(17);
   REQUIRE(first != null_idx);
   REQUIRE(a.reserve_longest_run() == 47);  // the rest of the run, still in the long part
   REQUIRE(a.reserve_owed_lines() == flexible + 17);
   require_reserve_adds_up(a);
   a.release_run(first, 17);                // it comes back: joined, 64 again
   REQUIRE(a.reserve_longest_run() == 64);
   REQUIRE(a.reserve_owed_lines() == flexible);
   require_reserve_adds_up(a);

   const line_idx second = tx.alloc_lines(40);
   REQUIRE(second == first);
   REQUIRE(a.reserve_longest_run() == 24);
   require_reserve_adds_up(a);
   a.release_run(second, 40);
   REQUIRE(a.reserve_longest_run() == 64);
   REQUIRE(a.reserve_owed_lines() == flexible);
   require_reserve_adds_up(a);

   // draws of 64, 48 and 24 lines all still succeed, one after the other
   for (uint32_t n : {64u, 48u, 24u})
   {
      const line_idx g = tx.alloc_lines(n);
      REQUIRE(g != null_idx);
      require_reserve_adds_up(a);
      a.release_run(g, n);
      require_reserve_adds_up(a);
   }
   REQUIRE(a.reserve_longest_run() == 64);

   // Repaid from somewhere else instead (the usual case: the old version of
   // the node, not the copy): the lines are all there, in two pieces.
   const line_idx copy = tx.alloc_lines(20);
   REQUIRE(copy != null_idx);
   tx.commit();
   tx.close();
   // twenty adjacent ordinary lines, freed as one run
   a.release_run(taken[640], 20);
   REQUIRE(a.reserve_owed_lines() == flexible);
   REQUIRE(a.reserve_longest_run() == 44);
   require_reserve_adds_up(a);
}

// A draw the reserve cannot meet asks the reclaim hook once, not sixteen times.
TEST_CASE("a reserve miss asks for queued releases once", "[arena][limit][reserve]")
{
   arena a(one_segment_with_reserve(tmp_dir("oncehook")));
   auto  ses   = a.start_session();
   auto  taken = take_everything(ses, 1);
   static uint64_t calls;
   calls = 0;
   a.set_reclaim([](void*, bool) noexcept { ++calls; return true; }, nullptr);
   auto tx  = ses.start_transaction(0);
   auto use = ses.use_reserve();
   try  // every nine-line run the reserve can give, flexible or guaranteed
   {
      for (;;)
         tx.alloc_lines(9);
   }
   catch (const arena_full&)
   {
   }
   calls = 0;
   REQUIRE_THROWS_AS(tx.alloc_lines(9), arena_full);
   // once for the reserve, then the ordinary path's own bounded rounds
   REQUIRE(calls >= 1);
   REQUIRE(calls <= 1 + 16 + 1);
   a.set_reclaim(nullptr, nullptr);
}

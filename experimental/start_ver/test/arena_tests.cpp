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

#include <start_ver/arena.hpp>

using namespace start_ver;
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

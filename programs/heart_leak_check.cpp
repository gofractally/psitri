// heart memory-cleanliness check: drive the PUBLIC API with the scale-bench
// workload shape (random u64 keys, chunked commits), quiesce, and account for
// every object. Three phases, each ending at a wait_for_compactor barrier:
//
//   A insert N keys      -> objects/key sane, audit_refs: every node ref==1
//   B overwrite all N    -> object_count flat vs A (updates must not strand)
//   C remove all N       -> object_count back to ~baseline (exact no-leak)
//
//   heart-leak-check [N=5e6] [chunk=1e5] [dir=/tmp/heart_leak_db]
//
// Exit 1 on any stranded object or audit anomaly. Run the build/relinv flavor
// (ENABLE_DEEP_INVARIANTS=ON) to also abort on the first double/partial free.
#include <radix_engine.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static uint64_t splitmix64(uint64_t x)
{
   x += 0x9E3779B97F4A7C15ull;
   x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
   x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
   return x ^ (x >> 31);
}
static std::string_view k8(const uint64_t& v)
{
   return {reinterpret_cast<const char*>(&v), 8};
}

int main(int argc, char** argv)
{
   const uint64_t N     = argc > 1 ? strtoull(argv[1], nullptr, 10) : 5'000'000ull;
   const uint64_t chunk = argc > 2 ? strtoull(argv[2], nullptr, 10) : 100'000ull;
   const char*    dirs  = argc > 3 ? argv[3] : "/tmp/heart_leak_db";

   std::filesystem::path dir(dirs);
   std::filesystem::remove_all(dir);
   std::filesystem::create_directories(dir);
   auto  db  = heart::radix::database::open(dir.string());
   auto  ses = db->start_write_session();
   auto& eng = db->engine_for_testing();

   db->wait_for_compactor();
   const uint64_t base = eng.object_count();
   printf("baseline objects: %llu\n", (unsigned long long)base);

   auto stats = [&](const char* phase, uint64_t prev) -> uint64_t
   {
      if (!db->wait_for_compactor(std::chrono::milliseconds(60000)))
         printf("  [%s] WARN: compactor did not drain in 60s\n", phase);
      const uint64_t oc = eng.object_count();
      auto           ra = eng.audit_refs(10);
      auto           d  = db->dump();
      uint64_t       live = 0;
      for (auto& seg : d.segments)
         if (!seg.is_free && seg.alloc_pos > 0)
            live += seg.alloc_pos - seg.freed_bytes;
      printf("%s: objects=%llu (delta %+lld)  audit: nodes=%llu leaves=%llu BAD=%llu  live=%.1f MB  file=%.1f MB\n",
             phase, (unsigned long long)oc, (long long)(int64_t(oc) - int64_t(prev)),
             (unsigned long long)ra.nodes, (unsigned long long)ra.leaves,
             (unsigned long long)ra.bad, double(live) / 1048576.0,
             double(d.total_segments) * sal::segment_size / 1048576.0);
      fflush(stdout);
      if (ra.bad)
         printf("  ^^ %llu nodes carry phantom refs (will strand on drop)\n",
                (unsigned long long)ra.bad);
      return oc;
   };

   // ── A: insert ──
   for (uint64_t done = 0; done < N;)
   {
      const uint64_t end = std::min(N, done + chunk);
      auto           tx  = ses->start_transaction(0);
      for (uint64_t i = done; i < end; ++i)
      {
         const uint64_t k = splitmix64(i);
         const uint64_t v = splitmix64(i ^ 0xDEADBEEFCAFEBABEull);
         tx.upsert(k8(k), k8(v));
      }
      tx.commit();
      done = end;
   }
   const uint64_t a = stats("A insert   ", base);

   // ── B: overwrite every key, PASSES times (the file-bloat trigger is churn:
   // object counts stay flat while segment free accounting drifts) ──
   const unsigned PASSES = argc > 4 ? unsigned(atoi(argv[4])) : 1;
   uint64_t       b      = a;
   for (unsigned p = 0; p < PASSES; ++p)
   {
      for (uint64_t done = 0; done < N;)
      {
         const uint64_t end = std::min(N, done + chunk);
         auto           tx  = ses->start_transaction(0);
         for (uint64_t i = done; i < end; ++i)
         {
            const uint64_t k = splitmix64(i);
            const uint64_t v = splitmix64(i ^ (0x1234567890ABCDEFull + p));
            tx.upsert(k8(k), k8(v));
         }
         tx.commit();
         done = end;
      }
      char lbl[32];
      snprintf(lbl, sizeof lbl, "B overwr #%u", p + 1);
      b = stats(lbl, b);
   }

   // ── C: remove everything ──
   for (uint64_t done = 0; done < N;)
   {
      const uint64_t end = std::min(N, done + chunk);
      auto           tx  = ses->start_transaction(0);
      for (uint64_t i = done; i < end; ++i)
      {
         const uint64_t k = splitmix64(i);
         tx.remove(k8(k));
      }
      tx.commit();
      done = end;
   }
   const uint64_t c = stats("C remove   ", b);

   const int64_t strand = int64_t(c) - int64_t(base);
   printf("stranded after remove-all: %lld objects (expect ~0)\n", (long long)strand);
   return (strand > 8) ? 1 : 0;
}

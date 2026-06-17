// mdbx_bank_bench — the SAME bank workload (N accounts, NTX transfers, balance
// conservation, R concurrent readers) run on REAL libmdbx, for a head-to-head vs
// the psitri adb engines (bank_compare_bench). One writer; MVCC read txns.
// Durability matched to the psitri engines (no per-commit fsync): SAFE_NOSYNC.
//   mdbx-bank-bench [accounts=1e6] [transactions=2e6] [batch=1000] [readers=4] [dir]
#include "mdbx.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
static constexpr uint64_t INIT = 1'000'000ull, AMOUNT = 1;

static uint64_t pick(std::mt19937_64& rng, uint64_t n)  // triangular (hot accounts) — matches bank_compare
{ std::uniform_int_distribution<uint64_t> d(0, n / 2 - 1); return d(rng) + d(rng); }

static MDBX_val mv(const void* p, size_t n) { MDBX_val v; v.iov_base = const_cast<void*>(p); v.iov_len = n; return v; }
#define CK(call) do { int _rc = (call); if (_rc != MDBX_SUCCESS) { std::printf("MDBX err %d (%s) @ %s\n", _rc, mdbx_strerror(_rc), #call); std::exit(2); } } while (0)

int main(int argc, char** argv)
{
   const uint64_t N       = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1'000'000ull;
   const uint64_t NTX     = argc > 2 ? strtoull(argv[2], nullptr, 10) : 2'000'000ull;
   const uint64_t BATCH   = argc > 3 ? strtoull(argv[3], nullptr, 10) : 1'000ull;
   const uint64_t READERS = argc > 4 ? strtoull(argv[4], nullptr, 10) : 4ull;
   const std::string dir  = argc > 5 ? argv[5] : "/tmp/bank_mdbx";
   std::filesystem::remove_all(dir);
   std::filesystem::create_directories(dir);

   MDBX_env* env = nullptr;
   CK(mdbx_env_create(&env));
   CK(mdbx_env_set_geometry(env, -1, -1, int64_t(8) << 30, -1, -1, -1));  // up to 8 GB
   // WRITEMAP + SAFE_NOSYNC: no per-commit fsync (matches the psitri engines' in-memory commit)
   CK(mdbx_env_open(env, dir.c_str(), MDBX_WRITEMAP | MDBX_SAFE_NOSYNC, 0664));
   MDBX_dbi dbi;
   { MDBX_txn* t; CK(mdbx_txn_begin(env, nullptr, MDBX_TXN_READWRITE, &t)); CK(mdbx_dbi_open(t, nullptr, MDBX_DB_DEFAULTS, &dbi)); CK(mdbx_txn_commit(t)); }

   auto put = [&](MDBX_txn* t, uint64_t k, uint64_t v) { MDBX_val K = mv(&k, 8), V = mv(&v, 8); CK(mdbx_put(t, dbi, &K, &V, MDBX_UPSERT)); };

   // bulk load
   double load_s;
   { auto t0 = Clock::now(); uint64_t i = 0;
     while (i < N) { MDBX_txn* t; CK(mdbx_txn_begin(env, nullptr, MDBX_TXN_READWRITE, &t));
        const uint64_t end = std::min(N, i + BATCH); for (; i < end; ++i) put(t, i, INIT); CK(mdbx_txn_commit(t)); }
     load_s = secs(t0, Clock::now()); }

   auto writer = [&](std::atomic<bool>* stop) -> uint64_t {
      std::mt19937_64 rng(0xBA11C0DE); uint64_t done = 0;
      while (done < NTX && (!stop || !stop->load(std::memory_order_relaxed))) {
         MDBX_txn* t; CK(mdbx_txn_begin(env, nullptr, MDBX_TXN_READWRITE, &t));
         const uint64_t bend = std::min(NTX, done + BATCH);
         for (; done < bend; ++done) {
            const uint64_t s = pick(rng, N), d = pick(rng, N); if (s == d) continue;
            MDBX_val Ks = mv(&s, 8), Kd = mv(&d, 8), Vs, Vd;
            if (mdbx_get(t, dbi, &Ks, &Vs) != MDBX_SUCCESS || mdbx_get(t, dbi, &Kd, &Vd) != MDBX_SUCCESS) continue;
            uint64_t sb; std::memcpy(&sb, Vs.iov_base, 8); uint64_t db_; std::memcpy(&db_, Vd.iov_base, 8);
            if (sb < AMOUNT) continue;
            put(t, s, sb - AMOUNT); put(t, d, db_ + AMOUNT);
         }
         CK(mdbx_txn_commit(t));
      }
      return done;
   };
   double wtx;
   { auto t0 = Clock::now(); uint64_t n = writer(nullptr); wtx = double(n) / secs(t0, Clock::now()); }

   auto reader = [&](std::atomic<bool>* stop, std::atomic<uint64_t>* count) {
      std::mt19937_64 rng(0xF00D); uint64_t local = 0;
      while (!stop->load(std::memory_order_relaxed)) {
         MDBX_txn* rt; CK(mdbx_txn_begin(env, nullptr, MDBX_TXN_RDONLY, &rt));
         for (int b = 0; b < 1000; ++b) { uint64_t k = pick(rng, N); MDBX_val K = mv(&k, 8), V; if (mdbx_get(rt, dbi, &K, &V) == MDBX_SUCCESS) ++local; }
         mdbx_txn_abort(rt);
      }
      count->fetch_add(local, std::memory_order_relaxed);
   };

   double rd_ww, wtx_ww;
   { std::atomic<bool> stop{false}; std::atomic<uint64_t> rc{0}; std::vector<std::thread> rt;
     for (uint64_t i = 0; i < READERS; ++i) rt.emplace_back(reader, &stop, &rc);
     auto t0 = Clock::now(); uint64_t wn = writer(&stop); stop.store(true); for (auto& t : rt) t.join();
     const double dt = secs(t0, Clock::now()); rd_ww = double(rc.load()) / dt; wtx_ww = double(wn) / dt; }

   double rd_solo;
   { std::atomic<bool> stop{false}; std::atomic<uint64_t> rc{0}; std::vector<std::thread> rt;
     auto t0 = Clock::now(); for (uint64_t i = 0; i < READERS; ++i) rt.emplace_back(reader, &stop, &rc);
     std::this_thread::sleep_for(std::chrono::seconds(2)); stop.store(true); for (auto& t : rt) t.join();
     rd_solo = double(rc.load()) / secs(t0, Clock::now()); }

   // balance conservation
   bool pass; uint64_t total = 0;
   { MDBX_txn* rt; CK(mdbx_txn_begin(env, nullptr, MDBX_TXN_RDONLY, &rt));
     for (uint64_t i = 0; i < N; ++i) { MDBX_val K = mv(&i, 8), V; if (mdbx_get(rt, dbi, &K, &V) == MDBX_SUCCESS) { uint64_t b; std::memcpy(&b, V.iov_base, 8); total += b; } }
     mdbx_txn_abort(rt); pass = (total == N * INIT); }

   // footprint: live = used B-tree pages × page size; disk = the data file high-water
   double live_mb = 0, disk_mb = 0;
   { MDBX_txn* rt; CK(mdbx_txn_begin(env, nullptr, MDBX_TXN_RDONLY, &rt)); MDBX_stat st; CK(mdbx_env_stat_ex(env, rt, &st, sizeof(st)));
     live_mb = double(uint64_t(st.ms_leaf_pages + st.ms_branch_pages + st.ms_overflow_pages) * st.ms_psize) / (1024.0 * 1024.0);
     mdbx_txn_abort(rt); }
   { std::error_code ec; uint64_t b = 0; for (auto& e : std::filesystem::recursive_directory_iterator(dir, ec)) if (e.is_regular_file(ec)) b += e.file_size(ec); disk_mb = double(b) / (1024.0 * 1024.0); }
   mdbx_env_close(env);

   std::printf("\nmdbx (real libmdbx, WRITEMAP|SAFE_NOSYNC): %llu accts x %llu transfers, batch %llu, %llu readers\n",
               (unsigned long long)N, (unsigned long long)NTX, (unsigned long long)BATCH, (unsigned long long)READERS);
   std::printf("%-20s %12s %12s %14s %12s %14s %9s %9s  %s\n", "engine", "load/s", "wtx/s", "rd/s(+wr)", "wtx/s(+rd)", "rd/s(solo)", "live(MB)", "disk(MB)", "balance");
   std::printf("%-20s %12.0f %12.0f %14.0f %12.0f %14.0f %9.0f %9.0f  %s\n", "mdbx", N / load_s, wtx, rd_ww, wtx_ww, rd_solo, live_mb, disk_mb, pass ? "PASS" : "FAIL");
   return 0;
}

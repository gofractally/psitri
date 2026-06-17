# bank: gcart::st (spec engine) vs all psitri engines vs REAL libmdbx

Head-to-head of the SAME bank workload after the dying-sentinel reclaim fix
(commit 961cdb4) — confirms the 3.4x write regression is gone and the
reader-vs-writer race is closed (balance conservation PASS, TSan-clean).

## Conditions
- Workload: 1,000,000 accounts @ 1,000,000 init each; 2,000,000 balance-conserving
  transfers (triangular hot-account distribution); batch 1,000 txns; 4 reader threads.
- Phases per engine: bulk load, write-only, readers+writer, readers-solo (2s),
  full balance-conservation sweep, then drain()+live_bytes()+on-disk high-water.
- Durability: psitri engines commit in-memory (no per-commit fsync); libmdbx run
  with WRITEMAP|SAFE_NOSYNC to match (no per-commit fsync).
- Commands:
    ./build/bin/bank-compare-bench 1000000 2000000 1000 4 /tmp/bank_cmp
    DYLD_LIBRARY_PATH=$HOME/arbtrie/external/libmdbx/build \
      ./build/bin/mdbx-bank-bench 1000000 2000000 1000 4 /tmp/bank_mdbx
- Build: Release -O3 -DNDEBUG, clang (Apple), git 7a3c8a5 on branch sal2.
- Machine: Apple M5 Max, 18 cores, macOS (darwin 25.3).
- libmdbx: arbtrie vendored, WRITEMAP|SAFE_NOSYNC, geometry up to 8 GB.

## Headline (gcart_spec vs real libmdbx, same workload)
| metric        | gcart_spec | mdbx    | gcart_spec / mdbx |
|---------------|-----------:|--------:|------------------:|
| load/s        |  2,869,963 | 588,685 |             4.9x  |
| wtx/s         |  1,138,597 |  77,214 |            14.7x  |
| rd/s (+wr)    | 36,102,343 |10,431,636|            3.5x  |
| rd/s (solo)   | 42,982,589 |11,624,799|            3.7x  |
| live (MB)     |         39 |      36 |  ~par (settled data) |
| disk (MB)     |        154 |      64 |  2.4x (COW high-water) |
| balance       |       PASS |    PASS |                   |

gcart_spec's SETTLED live footprint (39 MB) is ~par with libmdbx's B-tree (36 MB);
the 2.4x on-disk gap is un-reclaimed COW garbage + segment over-allocation
high-water — the known reclaim-lag lever, not data bloat. On throughput it is
4.9x/14.7x/3.5x/3.7x ahead of real MDBX.

## Files
- bank_compare.log — all 7 psitri engines (tiered, start_ver, start_ref, sal_art_ref, gcart, gcart_spec)
- mdbx.log — real libmdbx baseline

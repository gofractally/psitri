# heart garbage-burst A/B — 2 KB vs 1 KB pages @ 100k-op commits, 170M keys

**Date:** 2026-06-10 (evening; each run ~260 s, runs sequential, solo).
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1.
**Build:** `build/release` clang Release; NOTE: `debug_compactor=true` temporarily
compiled in for these runs (stderr-only skip-counter logging; perf impact nil —
reverted after, never committed).
**Git:** `b876375f` + the temp debug flip.
**Commands:**
- `bench_2kb_pages.log`: `HEART_PINNED_MB=2048 heart-scale-bench 170000000 100000 …`
- `bench_1kb_pages.log`: `HEART_PINNED_MB=2048 HEART_PAGE_BYTES=1024 heart-scale-bench 170000000 100000 …`
**Workload:** shared scale workload, **100k-op commits** (the burst trigger).

## Results

| keys | 2KB file GB (free%) | 1KB file GB (free%) |
|---|---|---|
| 100M | 4.19 (32) | 4.31 (29) |
| 130M | 6.88 (51) | 5.41 (29) |
| 150M | 12.34 (69) | 6.28 (28.5) |
| 170M | 16.72 (73) | **7.38 (29.4)** |

1KB perf vs 2KB at 170M: reads 450 vs 468-523ns (equal/better), inserts ~equal,
iteration 21 vs 15.6ns (~30% worse), live +20% (30.6 vs 25.5 B/key).

## Findings

- The burst is **generation-rate vs compactor copy-bandwidth** (garbage itself
  costs the compactor only a skip — Dan's framing; the cost is copying live
  survivors to turn segments over). At 100k-op commits + 2KB pages the writer's
  dead-byte production crosses the single compactor's bandwidth at ~100M keys
  (compactor was 100% busy through the burst: only 9 idle polls in 260s, all
  pre-burst — see the skip-counter lines in stderr of the original run).
- Halving the COW unit halves generation → 1KB run never crosses → no burst.
- 1M-op commits achieve the same at 2KB pages (see heart-scale-1B): repeat
  page-touches within a commit mutate in place.
- The burst is self-limiting either way (deeper tree throttles the writer back
  under the line): the earlier killed 2KB/100k 1B-run sat flat at 51.16GB from
  350M→800M.
- psitri control on the identical workload never bursts (1.26× live, monotone
  free%): see psitri-scale-262M-control.

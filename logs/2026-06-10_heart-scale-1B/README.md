# heart progressive scale — 1B keys, clean run

**Date:** 2026-06-10 (~22:00, 23.1 min total wall)
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1, db on /tmp (APFS NVMe). Solo run.
**Build:** `build/release`, clang `-O3 -march=native -std=c++23`, Release.
**Git:** `b876375f` (post SAL fix `9b6a52e3` + `HEART_PINNED_MB` `389e51fd`).
**Command:** `HEART_PINNED_MB=2048 heart-scale-bench 1000000000 1000000 /tmp/heart_scale_db`
**Config:** 2 KB pages (default), 1M-op commits, sync=none, 2 GB pinned budget.
**Workload:** shared scale workload (logs/README.md).

## Results

| keys | ins ns | rd ns | iter ns | file GB | free% |
|---|---|---|---|---|---|
| 1M | ~200 | ~90 | 9 | 0.3 | — |
| 50M | 911 | 650 | 17.5 | 3.5 | 25.3 |
| 100M | 1193 | 519 | 16.8 | 6.1 | 25.5 |
| 250M | 1450 | 495 | 15.0 | 11.0 | 23.5 |
| 500M | 1393 | 609 | 17.1 | 21.3 | 27.4 |
| 750M | 1417 | 533 | 17.4 | 32.9 | 27.8 |
| 1000M | 1383 | 566 | 16.1 | 40.0 | 27.9 |

**total wall 1385.3 s** including ~100M point reads + 100M iteration steps of probing. hit% 100 throughout.

## Findings

- Insert cost PLATEAUS (~1.38µs from 100M on) — depth cost saturates; reads
  plateau 520-610ns; **iteration 15-18ns/step flat across three orders of
  magnitude** (the page-locality headline). ~29 B/key live, 1.39× gross.
- free% locked 23-28% for the whole run: at 1M-op commits, COW garbage
  generation stays under the single compactor's copy bandwidth (cf.
  heart-burst-ab for what happens at 100k-op commits).
- Context at part's 392M address wall: heart was at 19 GB file, ins 1.34µs,
  iter 17ns (part: 32 GB, ~330ns reads, iter ~40ns — part keeps the point-read
  edge, heart everything else).
- Preconditions for this result, fixed earlier the same day: SAL spill
  fall-through undead-control bug (`9b6a52e3`) and the alloc-to-pinned 8 GB
  default budget (knob `389e51fd`) — without those this run bloats unboundedly.

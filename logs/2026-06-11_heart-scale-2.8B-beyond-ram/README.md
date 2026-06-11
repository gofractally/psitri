# heart progressive scale — 2.8B keys, the beyond-RAM baseline

**Date:** 2026-06-10 22:35 → 2026-06-11 01:05 (killed at 2.8B on Dan's call — asymptote established; target was 4B)
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1, db on /tmp (APFS, internal NVMe). Solo run (nothing else on the machine).
**Build:** `build/release` — clang (homebrew LLVM), `-O3 -march=native -std=c++23`, Release, asserts off.
**Git:** `b876375f` (includes the SAL undead-control fix `9b6a52e3` and `HEART_PINNED_MB` knob `389e51fd`).
**Command:** `HEART_PINNED_MB=2048 heart-scale-bench 4000000000 1000000 /tmp/heart_scale_db`
**Config:** 2 KB pages (engine default), **1M-op commits**, sal sync=none, pinned-cache budget 2 GB.
**Workload:** shared scale workload (see logs/README.md) — uniform-random unique u64→u64, per-1M-chunk probes.
**Side log:** `pagestats.log` — 30s sampler: per-process faults/pageins (`top -l 2 -pid`, NB: `pgrep -x` not `-f`, which matches the zsh wrapper) + `vm_stat` cumulatives (pageins/pageouts/swapins/swapouts/compressor).

## Results

| keys | ins ns | rd ns | iter ns | file GB | RSS GB |
|---|---|---|---|---|---|
| 1M | 207 | 149 | 12 | 0.34 | 0.4 |
| 1B | 1402 | 540 | 16 | 40.0 | 40.3 |
| 1.5B | 2026 | 882 | 20 | 65.8 | 66.4 |
| 2B | 2407 | 1127 | 21 | 86.4 | 87.3 |
| 2.5B | 3451 | 1765 | 28 | 98.8 | 99.6 |
| 2.75B | 15748 | 13770 | 153 | 109.9 | 104.4 |
| 2.8B (kill) | ~20000 | ~19500 | ~240 | 113.0 | 104.4 |

- `hit%` 100.0 on every chunk (≈280M point reads + 280M iteration steps total).
- `free%` 28-29% the entire run — no garbage burst at 1M-op commits (cf. heart-burst-ab).
- ~29 B/key live, 1.39× gross; 1B keys loaded in 23 min, 2.5B in ~85 min.

## Findings

1. **Thrash onset at ~100 GB RSS (78% of 128 GB)** — the kernel defends ~28 GB
   (free margin, compressor, other residents). First symptoms at ~83 GB
   (intermittent 4-8µs chunks), full collision at 2.53B keys/100 GB.
2. **The floor is the read-before-write**: one synchronous QD1 16 KB fault per
   uniform-random insert ≈ 20µs. Pageins ran 40-48 GB/5min (~160 MB/s sustained
   random reads) vs pageouts ~40 MB/s — read-bound, NOT write-bound (writes are
   sequential through the 32 MB bump allocator; the LSM out-of-core advantage
   is blind puts only).
3. Iteration degraded worst in thrash (16→240ns, 15×): every page boundary on
   an evicted page is a 16 KB fault.
4. Numbers are batched: 1M-op commits amortize commit cost; COW batch
   amortization vanishes once batch ≪ page count (99% distinct pages at 2.7B).
5. Agreed next direction from this data: DWAL delta-tree pipeline (foreground
   temp trees at shallow-tree speed, background key-sort-order merge using the
   radix as a prefetch oracle, which also re-clusters the base key-sequentially).

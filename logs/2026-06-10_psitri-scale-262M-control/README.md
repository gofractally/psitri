# psitri progressive scale — control leg, killed at 262M

**Date:** 2026-06-10 (late evening). **CAVEAT: contended** — ran concurrently
with the heart 1B run (and briefly a third run) on the same machine; latency
columns carry contention noise; the FILE/free% columns (the purpose of the
control) are robust. Killed externally at 262M of a 300M target.
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1.
**Build:** `build/release` clang Release. **Git:** `b876375f` (bench added in that commit).
**Command:** `psitri-scale-bench 300000000 100000 /tmp/psitri_scale_db`
**Config:** psitri defaults, sync=none, **100k-op commits** (the config that
makes heart burst — that's the point).
**Workload:** shared scale workload (logs/README.md).

## Results

| keys | file GB | free% | rd ns (contended) |
|---|---|---|---|
| 50M | 1.94 | 36.4 | 759 |
| 100M | 3.22 | 25.5 | 819 |
| 150M | 4.34 | 20.4 | 789 |
| 200M | ~5.6 | ~18 | ~850 |
| 262M (kill) | 7.16 | 14.5 | 1022 |

## Findings

- **free% monotonically falling** the whole run; file = 1.26× live at 150M
  (heart at the same mark, same workload: 12.75GB at 70% free = 3.4×).
  Same SAL, same single compactor thread → the compactor was never heart's
  bottleneck; heart's 2KB-page COW generation rate was (Dan called this).
- psitri live ≈ 23 B/key vs heart ≈ 25 B/key — at-rest space nearly identical;
  the difference is entirely in-flight garbage.
- Open question carried forward: psitri COWs big nodes whole (it is NOT a
  node-granularity story) yet generates less dead-byte volume per insert —
  keys-per-leaf density and split cadence are the candidate explanations.

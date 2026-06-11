# hkv (standalone ~/myartlib) progressive scale — in-memory, killed at 550M

**Date:** 2026-06-11 (~4 min run). Solo.
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1. NOTE:
swap/compressor state from the same-night heart 2.8B run may have lowered the
effective ceiling (pressure onset at ~88 GB RSS; swapfile 94% full at kill).
**Build:** standalone clang `-O3 -DNDEBUG -std=c++23 -march=native -I ~/myartlib/src`
(bench: `programs/hkv_scale_bench.cpp`).
**Source:** ~/myartlib hkv header-only (basis of part), default `std::allocator`
→ native **u48 packed_ptr** (6-byte direct pointers). Pure in-memory: no SAL,
no transactions, no sync — `kv_tree<std::string_view, uint64_t>` (mode::none).
**Command:** `hkv-scale-bench 700000000 1000000` (target was 2B — see findings).
**Workload:** shared scale workload (logs/README.md), no commits (plain inserts).

## Results

| keys | ins ns | rd ns | iter ns | rss GB |
|---|---|---|---|---|
| 50M | 194 | 104 | 61 | 7.7 |
| 100M | 189 | 133 | 75 | 14.2 |
| 200M | 216 | 141 | 71 | 28.7 |
| 300M | 231 | 157 | 95 | 54.1 |
| 400M | 206 | 167 | 102 | 76.5 |
| 450M | 224 | 235 | 142 | 84.4 |
| 500M | 287 | 672 | 547 | 87.8 |
| 550M (kill) | 516 | 1507 | 1100 | 87.8 (peak; compressor active) |

hit% 100 throughout.

## Findings

- **In-RAM perf is the best of anything measured this campaign**: inserts
  ~190-230ns and reads ~130-235ns essentially FLAT to 450M keys — at 100M,
  reads are 2× part-on-SAL (133 vs 255ns) and 4× heart (519ns). The u48
  same-cacheline descend at full strength with zero persistence overhead.
- **Footprint is the wall: ~160-190 B/key** for u64→u64 (peaked at ~270 B/key
  marginal during a router-promotion wave at 250-350M, amortized back to ~158).
  Mode matrix at 50M (none/buckets/adaptive/dense_tiers/±pool): all within
  165-169 B/key — modes don't change layout on uniform-random u64s, and 8-byte
  values can't inline in 6-byte handles (≤5 B only). The 2B target needs
  330 GB — impossible in 128 GB; the run was capped at the honest RAM limit.
- Memory-pressure knee at ~88 GB RSS: compressor took over (6.8M pages stored
  ≈ 104 GB compressed at ~6.4:1), reads 235→672→1507ns over 100M keys,
  swapfile 94% full at the kill. Anonymous memory degrades far harder than
  the mmap'd engines' file-backed eviction (no cheap clean-page reclaim).
- Cross-engine footprint context: heart lives at **29 B/key** live (pages
  amortize per-key overhead 5.7×); part/hkv-on-SAL ~167 B/key live. Density,
  not raw speed, is what lets the persistent page engine go 5× deeper in the
  same RAM.

# part progressive scale — stops at 392M keys (32 GB address wall)

**Date:** 2026-06-10 (846.3 s total). Mostly solo (a part 48-bit A/B overlapped
part of the window; file/stop numbers unaffected).
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1.
**Build:** standalone clang `-O3 -march=native -std=c++23 -flto=thin` against a
HEAD snapshot of part headers (repo tree.hpp was mid-edit in a parallel
session) + repo `libsal.a`.
**Git:** bench at `5d02175c`; part headers at `18b3e5b4`.
**Command:** `part-scale-bench 1000000000 1000000 /tmp/part_scale_db`
**Config:** part::kv defaults, sync=none, max_segments=4096 (= the honest
ceiling: `detail::line_ptr` = 4-bit tag + 28-bit line index → 2^28 × 128 B =
**32 GB hard**, independent of config — raising max_segments past it silently
truncates pointers in release builds).
**Workload:** shared scale workload, 1M-op commits.

## Results

| keys | ins ns | rd ns | iter ns | file GB | free% |
|---|---|---|---|---|---|
| 1M | 130 | 80 | 19.5 | 0.09 | 32 |
| 50M | 1268 | 235 | 89 | 12.4 | 48 |
| 100M | 3131 | 255 | 54 | 14.3 | 30 |
| 200M | 3426 | 297 | 39 | 16.2 | 30 |
| 392M | ~1700-2800 | ~330 | ~42 | 32.0 | 37.7 |

**STOP at 392,000,000 committed keys: "part: arena is full (max_segments)".**
hit% 100 throughout. Live ≈54 B/key, gross ≈87 B/key at the wall; 37.7% of
lines free but too fragmented for word-confined reuse.

## Findings

- The wall is the 4-byte handle, not config. Lifting it: see part-ptr48-ab-20M.
- 128 GB RAM never stressed (file capped at 32 GB) — decay here is pure
  DRAM/TLB depth decay.
- part keeps the best point reads of the three engines at depth (~330ns at
  392M vs heart ~537ns) but iteration (~40ns) and inserts (~1.7-2.8µs) trail
  heart (17ns / 1.34µs at the same mark).

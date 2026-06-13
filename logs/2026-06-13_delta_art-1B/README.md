# delta_art 1B progressive — start_ver hot delta → sal_art_ref cold base (#28)

First 1-billion-key run of the bounded-delta tiered pipeline, after the
per-root-floor GC fix (logs/2026-06-13_start_ver-per-root-floor). Verbatim
per-10M-key probe table in `run.log`.

## Conditions
- **engine**: `delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>`
  (hot delta = start_ver / version-stamp ART; cold base = heart dwal engine).
- **cmd**: `DELTA_HI_WATER_MB=256 delta_art-scale-bench 1000000000 1000 10000000 /tmp/da1b_db`
- **workload**: 1e9 splitmix64 u64 keys → u64 values, batched 1000/txn. After
  every 10M inserts: 100k random point reads + 100k iteration steps from a
  random lower_bound, on the LATEST merged view (delta ∪ base).
- **knobs**: hi_water 256 MB (flush trigger); single logical writer; the
  merger flushes on a background thread (sorted bulk, one txn/chunk).
- **git**: gofractally/psitri @ dd2967c
- **build**: Release, clang -O3 -DNDEBUG -march=native
- **machine**: Apple M5 Max, 18 cores, 128 GB RAM, darwin/arm64
- **contention**: writer thread + background merger + start_ver drain/reaper
  threads + heart's SAL release thread (all on the one machine).

## Result — 1,000,000,000 keys in 10.7 min, 100% hit throughout
```
      keys    ins ns    rd ns   hit%   iter ns   file GB   rss GB   chunk s
  10000000       278      247  100.0%    84.5      2.39      2.30      2.82
 100000000       406      627  100.0%    37.0     11.24     11.15      4.13
 200000000       511      798  100.0%    28.1     17.69     17.60      5.19
 300000000       696      822  100.0%    26.1     23.99     23.91      7.04
 400000000       495      887  100.0%    28.2     29.71     29.63      5.04
 500000000       535      995  100.0%    27.4     34.28     34.19      5.46
 600000000       679     1024  100.0%    26.5     36.34     36.25      6.90
 700000000       591     1047  100.0%    27.3     40.09     40.00      6.02
 800000000       688     1035  100.0%    29.1     51.36     51.27      6.99
 900000000       690     1509  100.0%    34.0     56.33     56.24      7.06
1000000000       657     1157  100.0%    27.9     58.17     58.08      6.69
delta arena: 5.14 GB used / 15.17 GB total (34% used)
delta segs:  grown=1808  reused=20,130,288  (reuse 100%)
DONE: 1,000,000,000 keys in 10.7 min
```
Final tiers on disk: **delta 14 GB, base 36 GB** (du).

## vs the raw-heart 1B baseline (memory: project_heart_1b_milestone)
| | time | RSS @ 1B | thrash | hit% |
|---|---|---|---|---|
| heart alone | ~23 min | ~100 GB (wall) | yes (cold random read-before-write) | — |
| **delta_art** | **10.7 min** | **58 GB** | **none** | 100% |

**~2.1× faster, ~half the RAM, no thrash.** The hot delta absorbs the random
writes; the sorted bulk merge re-clusters them into heart, so the base never
does the cold random faulting that walled raw heart. The tiering thesis at
full scale.

## Reading the curve
- **insert ns** drifts 278 → ~650 across the run: the *base* tree deepens
  (each flush COWs a taller base path) and RSS climbs — not the delta
  degrading. No cliff; flat-ish in the second half.
- **read ns** 247 → ~1100: the 2/3-way merged read over an ever-deeper base +
  cold pages at 58 GB resident. Still sub-1.2 µs at 1B.
- **file/rss track each other** (no over-mapping); rss never approaches the
  128 GB ceiling → no thrash, unlike raw heart's ~100 GB wall.

## Remaining lever — drain throughput at extreme scale
The per-root floor holds **reuse at 100%** (20M reuses) and bounds the delta
well at 100M (3.7 GB). At 1B the delta high-water still climbs to ~15 GB
(34% used): the base is deeper → each flush window is longer → more hot COW
garbage accrues before the SINGLE drain thread frees it. This is the *second
half* of audit #31 (faster / parallel drain), distinct from the floor-gating
(fixed). It does NOT thrash, leak, or hit the address wall — the delta
recycles cleanly, just at a higher plateau. Flattening it = parallelize the
drain or shorten the flush window.

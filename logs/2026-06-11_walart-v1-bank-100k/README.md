# walart v1 vs dwal — bank @100k accounts, first head-to-head

**Date:** 2026-06-11. Solo runs, sequential.
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1.
**Build:** build/release clang Release. **Git:** walart v1 @ `fad828a4` (spec docs/pdwal_design.md @ 5c76a328).
**Commands:** `bank-bench-{walart,dwal} --num-accounts 100000 --num-transactions 200000 [--batch-size 100]`
**Config:** names keys, sync=none, WALART_STEAL_KEYS=256000 (defaults), read-mode latest.

## Results (write-only tx/s | solo reads/s)

| batch | walart | dwal (incumbent) |
|---|---|---|
| 1 | 460,920 \| 5,130,707 | 754,453 \| 5,618,748 |
| 100 | 720,072 | 805,858 |

Both validate exactly (balance + log counts, 100% reader found).
Context, same config (uniform table): part native 484k, heart native 290k,
psitri 158k. walart batch=1 ≈ native part minus ~5% pipeline overhead.

## Findings

- walart v1 = part-speed writes with continuous background merge into heart;
  1.6× heart-native and 2.9× psitri-native writes at batch=1.
- dwal still wins raw small-scale throughput (-39% at batch=1, -11% at
  batch=100): its RW delta is a bump-arena art_map + buffered WAL append vs
  part's full COW — the structural price of a durable, snapshottable,
  spillable delta. The bank does not measure walart's claimed wins
  (O(1) savepoints, replay-free recovery, per-txn snapshot granularity,
  scale beyond RAM-bound deltas, heart-base scaling vs psitri-base).
- Implementation landmines (fixed, documented in fad828a4): part reclamation
  is writer-thread-only; moved-from hkv views tear down pinned snapshots;
  thread_local sessions vs exit() ordering.
- Next levers: steal-size sweep (grow-bias unexplored), larger scale (1M+
  accounts where heart-base beats psitri-base), fsync-mode comparison
  (part group-commit vs WAL fsync), part delta-insert fast path.

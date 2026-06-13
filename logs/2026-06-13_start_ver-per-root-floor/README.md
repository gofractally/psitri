# start_ver per-root version floors — bound the delta_art hot arena

Fix for the GC problem (logs/2026-06-12_delta_art-threshold-sweep finding 3):
the merger's warm snapshot pinned the GLOBAL version floor at V_warm for the
whole flush, starving the HOT delta's drain (root-0 garbage with death > V_warm
can't drain, even though the V_warm reader reads only root 1). Result was the
delta arena climbing to 10-13GB vs a 256MB flush target (37% used / 63% free,
grown 1680).

## Fix
- per-pin root attribution: arena `_reader_root_idx[slot]` set before a cursor
  publishes its pin (create_cursor / get_root snapshot). floor_for_(R) = oldest
  version pinned by a reader OF root R (+ gates-all pins); a pin on a different
  specific root no longer gates R.
- per-chain root tag: a single-root txn's supersession chain is tagged with its
  root (the hot write = root 0); a multi-root txn (rotation/clear) = global.
- drain_ consumes the whole ring (eat eligible at floor_for_(chain.root), spill
  gated) — no head-of-line blocking; `_drain_active` flag closes the
  flush_releases in-flight race (the require_drained leak check).

## Result (100M u64 scale, DELTA_HI_WATER_MB=256)
```
                 delta GB   arena used/total      grown   reuse
before (global)   10.0      5.07 / 13.52 (37%)    1680    97%
after  (per-root)  4.2      2.35 /  4.46 (53%)     532   100% (2.38M reuses)
```
100M in 0.8 min, flat insert, 100% hit. base 7.2GB (~1.4x live, unchanged).
start_ver unit 4/4, full battery 5/5 (the rare cross-engine ledger flake is a
test-harness global-counter timing issue, not an engine bug).

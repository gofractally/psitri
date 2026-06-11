# dwal Engine templating + Heart-DWAL first run + delta-map A/B

**Date:** 2026-06-12 (overnight session). Solo runs.
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1.
**Git:** `4d16490e` (dwal kv_engine concept), heart adapter heart_dwal_engine.hpp.

## Bank @100k accounts / 200k tx, names, sync=none, batch=1

| engine | write-only tx/s | rd solo | validation |
|---|---|---|---|
| PsiTri-DWAL (post-templating) | 716-745k (= pre-templating baseline) | 5.6M | PASS |
| **Heart-DWAL (first ever)** | **633,634** | 5.05M | PASS |
| walart (same night, bval port) | 441k | 5.17M | PASS |

Heart-DWAL conditions: merge_threads=1 (heart is single-writer), bulk_load via
dwal transactions, database constructed on a throwaway thread (SAL get_session
shares sessions per-thread via TLS — a main-thread open hands heart's engine
session to main-thread READERS while the merge worker WRITES through it; both
crash modes are documented in the commit).

## Delta-map A/B (delta_map_ab.cpp): art_map vs hkv on dwal's RW op mix

Bank-shaped generations (100k entries: upserts + ~7% erases + hit & miss
probes), then the sorted merge-drain; 20 cycles, identical work both sides.

| | mixed foreground ns/op | drain ns/entry |
|---|---|---|
| art_map<btree_value> | 120.0 | **6.8** |
| hkv kv_tree<btree_value> | **102.6** | 13.6 |

**Verdict:** hkv wins the foreground delta path by ~17%; art_map drains 2×
faster (background thread). Projected bank impact of a full RW-layer swap:
~5-8% — and hkv has no seq-COW, so dwal's lock-free latest/fresh read modes
would degrade to buffered (the cow_coordinator integration is exactly 7 sites,
all mapped). The "blows art_map out of the water" impression comes from
point-read-heavy workloads (hash24: 2-4× reads); dwal's delta is write+drain
shaped. Recommendation: not worth the integration at this payoff; revisit if
the delta role grows read-heavy or hkv gains a seq-COW mode.

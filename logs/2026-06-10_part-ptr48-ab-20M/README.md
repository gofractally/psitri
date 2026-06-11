# part 48-bit handle impact A/B — 20M keys per side

**Date:** 2026-06-10. Some background contention (a part 1B scale attempt was
running); both sides equally affected — the differential is what matters.
**Machine:** Mac17,6 — Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.3.1.
**Build:** standalone clang `-O3 -march=native -flto=thin` against a HEAD
shadow of part headers (`18b3e5b4`) **with the width-adaptive cb_hdr patch**
(see below) + repo `libsal.a`. Bench source: `programs/part_ptr48_bench.cpp`
(committed `5d02175c`).
**Command:** `part-ptr48-bench 20000000 1000000`
**Setup:** the SAME hkv tree instantiated twice via a bench-local allocator:
4-byte `line_ptr` (28-bit index, 32 GB ceiling) vs 6-byte `line_ptr6` (44-bit
index, 2 PB). Identical harness, keys, and probe pattern for both.

## Results (at 20M keys)

| | 4-byte | 6-byte | delta |
|---|---|---|---|
| tree footprint | 211.6 B/key | 213.3 B/key | **+0.8%** |
| insert (final chunk) | 773 ns | 760 ns | ~equal |
| point read | 164 ns | 182 ns | **+11-15%** |
| iteration step | 50.7 ns | 56.8 ns | +12% |

Both sides 100% found; censuses identical except 255 cx routers shift to
full_x (smaller cnode segment caps at 6 B); node_full grows 9→14 lines.

## Findings

- Memory is a non-issue (cbuckets + leaves dominate and are width-neutral);
  the read cost concentrates in the dense routers spanning more lines.
- **The ONE blocker:** `cb_hdr` fills its 128-byte line exactly at 4 B with a
  hardcoded `node_alloc_(128)`, no sizeof, no static_assert — at 6 B it is
  164 B and every header write tramples the neighbor line. Fix (validated,
  byte-identical at 4 B): `page_line[sizeof(Ptr)>4 ? 8 : 12]`, computed prefix
  cap (36→24), CB_PAGES to match, `static_assert(sizeof(cb_hdr)==128)`, and
  the two hardcoded 36s (cb_make2_ memcpy + the `c <= 36` formation gate).
  Full recipe in the header comment of `programs/part_ptr48_bench.cpp`.
- **5-byte handle is the sweet spot if 1 TB suffices**: 36-bit index, cb_hdr =
  90 B + prefix[38] (keeps CB_PAGES=12 AND prefix≥36 in one line), node_full
  11 lines → roughly half the 6-byte read penalty. Note arena's u32 line_idx
  caps practice at 512 GB until widened.

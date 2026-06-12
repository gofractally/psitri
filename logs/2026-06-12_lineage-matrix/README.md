# Full lineage: tiered vs its base (heart), legacy dwal (both bases), raw psitri

Conditions: 2026-06-12 ~09:37-09:50, load avg 2.3-2.6 (absolutes ~5% low;
within-table comparisons valid). Release -O3, 1M accounts / 10M txns,
batch=1, sync=none, 1 reader. part/vpart/tiered rows from the same-morning
keymode-matrix (same conditions). All 15 runs validate exact.

## xxh64 keys (8-byte hashes — Dan's baseline band)
| engine                  | write-only | w+r       | reads w/ writer |
|-------------------------|------------|-----------|-----------------|
| psitri raw              | 160,001    | 153,311   | 1,391,952       |
| heart raw (tiered base) | 297,093    | 271,025   | 3,592,534       |
| part                    | 450,490    | —         | 5,752,518       |
| vpart raw               | 529,897    | —         | 7,820,164       |
| TIERED (vpart->heart)   | 526,222    | —         | 6,463,101       |
| dwal over heart         | 684,629    | 655,585   | 4,081,509       |
| dwal over psitri        | 763,347    | 762,775   | 1,767,493       |

## names keys (16-char strings)
| engine                  | write-only | reads w/ writer |
|-------------------------|------------|-----------------|
| psitri raw              | 141,707    | 1,212,374       |
| heart raw               | 263,514    | 2,718,887       |
| part                    | 361,899    | —               |
| vpart raw               | 438,254    | —               |
| TIERED (vpart->heart)   | 439,077*   | 4,987,939*      |  *last night, quiet machine
| dwal over heart         | 529,568    | 3,137,019       |
| dwal over psitri        | 614,925    | 1,431,433       |

## Findings
1. TIERED vs ITS BASE: +77% writes (526k vs 297k) and +80% concurrent reads
   (6.46M vs 3.59M) over raw heart at xxh64 — the pipeline pays for itself
   on BOTH axes.
2. TIERED vs LEGACY DWAL-on-heart: dwal absorbs writes +30% faster (685k vs
   526k) — its delta is a TRANSIENT HEAP art_map + WAL-file appends vs
   tiered's fully persistent COW delta. But tiered reads +58% faster (6.46M
   vs 4.08M: delta-first point gets on a real tree vs dwal's merge path),
   recovery = adopt roots (no WAL replay protocol), the delta supports MVCC
   snapshots/cursors, and there is no 4 GB bump-arena ceiling (artmap DNF'd
   the 1M-versioned bank). The write gap is the price of delta durability.
3. dwal-on-psitri writes highest (763k) but reads LOWEST of all pipelines
   (1.77M — psitri base reads dominate misses).
4. Key-mode spread is uniform (~15-20% across every engine) — relative
   rankings identical in both modes.

# rcart — status & TODO

`rcart` is the **refcount** version of the spec engine (the frozen reference;
byte-identical fork of the original gcart). 3-region node taxonomy (FIX
cl1/cl2/cl9, SAL var_allocator<32> leaves, BIG big_allocator<64>), node-level
COW, frame-stack cursor, reopen recovery. On `sal2` (per-slot u8 refcount +
dying-sentinel reclaim).

Functional status: **green** — gcart-spec / gcart-spec-kv / gcart-kv style
oracles pass; the refcount latch + reclaim are correct.

## OPEN ISSUE (blocking Silkworm-scale) — refcount reclaim doesn't bound the SAL high-water

**Symptom.** Tiered `start_ver → (refcount cold base)` on Silkworm mainnet
(2026-06-17) **CRASHED at block 6.65M**:

```
CRIT Silkworm terminating due to exception: sal2: var_allocator full
```

The cold-base SAL region (`var_allocator<32>`, the ≤2KB suffix_value leaves)
hit its **hard ceiling**:

| chaindata/base | size | region |
|---|---|---|
| **sal.dat** | **128 G** | leaves — THE WALL |
| cl1.dat | 97 G | setlist / nested routing |
| cl9.dat | 17 G | full nodes |
| cl2.dat | 2.3 G | double_setlist |

`sal.dat = exactly 128 G` is the **u32 grain-index limit**: 16384 segments ×
8 MB = 2^32 grains for `var_allocator<32>`. `GCART_SAL_SEGS` cannot go higher
without a wider grain index. Milestones before the crash: 5M @2254s (64 G RSS,
115 G disk), 6M @4403s (67.6 G, 210 G). RSS tracked native (~68 G); disk blew
past native's 134 GB-at-8M long before reaching 8M.

**Root cause (class, not a bug).** This is COW-garbage **high-water**: each
write supersedes nodes; refcount reclaim frees them, but the bump+recycle
allocator only recycles a segment when it's >50% free, so partially-dead
segments accumulate and the *peak* (live + not-yet-recycled garbage) climbs
unbounded with churn until it hits the region cap. Same class as the prior
sal_art_ref 334 GB runaway ([[project_silkworm_scale_perf]]); here it's a hard
*crash* because gcart/rcart's SAL has a fixed ceiling sal_art_ref didn't.

**What it is NOT.** Not a correctness bug, not a leak (refcount drains to 0 in
the unit oracles), and **not fixable by the liveness model** — the version-
liveness GC (vcart) produces the *same* COW garbage; if anything its deferred
reclaim can hold garbage a touch longer. Disk/high-water is a separate axis
(plan §7': GC buys speed, not disk).

## TODO (the levers, in priority order)

1. **High-water / compaction (the real fix).** Trigger compaction on
   `freed_space + shrinkable-slack` (not just freed), or a shrink-on-seal pass,
   so the cold base stays near live density and segments recycle before the cap.
   This is what unblocks an honest 8M run. Ties to the movable-region compactor
   (sal2_design §6.2) and the sal_art_ref density findings
   ([[project_silkworm_scale_perf]]: ~2x density gap vs MDBX, leaf/value
   packing, DUPSORT key-duplication).
2. **Raise the ceiling (stopgap).** Wider SAL grain index (or larger grain) to
   push the 128 GB cap up — gets a full curve vs native but papers over (1).
3. **Memory-recovery bar for vcart.** Whatever vcart's version-liveness GC does,
   it must recover memory *at least as well as* rcart's refcount (drain-to-zero
   in the oracles; bounded high-water at scale) — see the vcart work.

Native 8M baseline (the yardstick): 14881s, 68 G RSS, 134 G disk
(`~/eth-bench/lmdbx_8m_baseline.tsv`). Tiered milestones:
`~/eth-bench/tiered_8m.tsv`.

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

**Measured breakdown of the 128 GB SAL (post-mortem on the .rc arrays, 2026-06-17).**
NOT "4 billion keys" — `sal.rc` is one refcount byte per 32 B grain, so the
4.29 B = 2^32 figure is the **grain index** (the u32 ceiling we exhausted), not
the object count. Counting non-zero rc:

| region | slots | live (rc>0) | dead/free | live % |
|---|---|---|---|---|
| sal (leaves) | 4.29 B grains | **1.545 B object-starts** | — | 36% of *grains* |
| cl1 (setlist/nested) | 811 M | 508 M | 303 M | **62.7%** |
| cl9 (full) | 17.9 M | 9.3 M | 8.5 M | **52.3%** |
| cl2 (double) | 9.4 M | 5.2 M | 4.2 M | **55.4%** |

So **TWO things are true, and both matter:**

1. **The live data is genuinely enormous — ~1.545 B live leaf objects** (avg
   ~2–3 grains each) + ~508 M live routing nodes. That is *real ETH archive
   data*, not garbage: the psitri-optimized layout keeps Erigon-style
   history/changesets (every account/storage change per block → unique,
   never-overwritten keys), and the prior sal_art_ref audit measured **3.0 B
   keys at 8 M** ([[project_silkworm_scale_perf]]) — ~1.5 B at 6.65 M is
   consistent. So a big chunk of the 128 GB is simply real.
2. **~37–48% of every region is dead-but-not-recycled** (freed nodes still
   occupying committed segments). The fixed regions sit at 52–63% *live* =
   37–48% *free* — **just under the >50%-free segment-recycle threshold**, so
   those segments never recycle and the garbage sticks. That's a
   recycle-threshold / compaction problem.

**So the wall = real-archive-scale live data (1.5 B keys) + a fat recycle
backlog, both pressing on a u32 grain index that's simply too small for archive
ETH.** Even at perfect density, ~1.5 B leaves × ~1.7 grains ≈ 2.6 B grains =
60% of the 4.29 B cap *with live data alone*; at 8 M (3 B keys) it overflows the
u32 index regardless of garbage.

**What it is NOT.** Not "mostly garbage" (my first read — wrong; it's ~60% live)
and **not fixable by the liveness model** — the version-liveness GC (gcart)
frees the *same* nodes and faces the *same* recycle threshold; if anything its
deferred reclaim holds garbage a touch longer. Liveness is a speed axis, not a
capacity axis (plan §7').

## ROOT CAUSE — the designed movable region was never built

Both symptoms (128 GB cap + 37–48% recycle backlog) are ONE gap: `sal2`'s
`var_allocator` shipped as a plain **direct-addressed bump allocator** — the
branch pointer (`ptr_address`) IS a global 32-byte-grain index
(`addr = base + slot·Grain`, var_allocator.hpp:16-17,100) — instead of the
**control-block / relocatable movable region** `sal2_design.md §4.3` specifies.
There is **no `sal2/control_block.hpp`**; nothing references relocation / `.segs`
/ `zone` / `offset`. The designed CB is `{ ref:16, active, pending_cache,
zone:2, offset:44 }` → **u32 ptr_address = 4 B objects (control blocks)**, each
**`offset:44` × 32 B grain = 512 TB/zone**, **relocated by CAS-rewriting only
`offset`** (so the compactor reclaims by moving live objects), leaves in `.segs`.

Consequence of the direct-addressed shortcut:
- `ptr_address` conflated with grain index ⇒ cap = 2^32 *grains* × 32 B = **128 GB**
  (not 2^32 *objects* × 512 TB). 1.545 B live objects × ~2.8 grains ≈ the whole
  index ⇒ crash.
- no indirection ⇒ **non-relocatable** ⇒ no compaction ⇒ can only whole-segment
  recycle at >50% free ⇒ the 37–48% stuck backlog.

The designed SAL would not have crashed: 1.5 B objects << 4 B CB budget, ~128 GB
data << 512 TB.

## TODO (the levers, in priority order)

1. **Build the movable region the design already specifies (§4.3).** CB
   indirection (`ptr_address` → control_block → `zone`+`offset`) + relocation +
   the compactor reclaiming by moving live objects. This is the capacity fix
   (128 GB → 512 TB, 4 B objects) AND the compaction fix (kills the recycle
   backlog) in one. Supersedes the band-aids below.
2. **Density (independent win).** gcart/rcart store each leaf as its OWN movable
   slot (~89 B incl. rounding); sal_art_ref page-PACKS leaves (leaf_page) and is
   ~2× denser. cl1 routing alone is 97 GB / 508 M live nodes. Fewer/denser
   objects shrink the load on the movable region regardless of (1).
3. **Memory-recovery bar for gcart.** gcart's version-liveness GC must reclaim
   *at least as well as* rcart's refcount here — match, not beat (capacity is (1)).

Native 8M baseline (the yardstick): 14881s, 68 G RSS, 134 G disk
(`~/eth-bench/lmdbx_8m_baseline.tsv`). Tiered milestones:
`~/eth-bench/tiered_8m.tsv`.

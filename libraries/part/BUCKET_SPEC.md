# part bucket — composed, line-refcounted last-mile collector (SPEC §18 draft)

Status: DRAFT for sign-off, spec before impl. Replaces the monolithic
bucket2k sketch: the bucket is a COMPOSITION of independently refcounted
lines (the full_x/cx lesson applied to terminals), so COW copies only the
lines that change and everything else is shared by retaining line refs.

## 0. Why (measured)

Terminals dominate live bytes (167B/key vs ~50B of user data; 77% of live
is leaf lines). A 60-entry bucket at ~32B/entry costs ~22 lines (~47B/entry
all-in) vs 60 leaf lines — ~2.7x denser, entries sized to user data
(prefix + postfix + valuetype + value in one object), and dense ordered
iteration fixes the 3.3x scan deficit vs heart. The monolithic tradeoff
(2KB COW per touch, 16-contig allocs hostile to per-line refcounts,
update-in-place page churn) is dissolved by composition.

## 1. Composition (every piece is a separate refcounted allocation)

| piece | lines | alloc class | shareable |
|---|---|---|---|
| CL0 header | 1 | 1 | clone-on-write |
| FP line (fingerprints) | 1 | 1 | yes |
| SP lines (stems+refs) ×4 | 1 each | 1 | yes, individually |
| sub-pages (small payload heap) ×≤12 | 2 each (256B) | 2 | yes, individually |
| dedicated value lines (entries ≥64B) | 1..n | 1..n | yes |

No 16-contiguous allocation is required anywhere, and there is NO implicit
variant: the sub-page table in CL0 makes the explicit composed form the
default and only bucket (Dan, 2026-06-11). One tag: K::bucket (9),
redefined as this composed form.

## 2. CL0 — the header cacheline (byte map, 128 B)

```
off  sz   field
  0   2   node_hdr      {has_term:1, nbranch:9 (≤64), prefix_len:6}
                        router_hdr-compatible at offset 0: term update and
                        set_or_update_term reuse verbatim
  2   4   term          line_ptr, value when the key ends at this bucket
  6   1   open_spi      the ONE open (bump-allocating) sub-page slot
  7   1   open_off8     bump position within it, 8B units (the "global
                        alloc_pos": advancing past a page folds its
                        untouched tail into that page's free_space)
  8   4   cnt[4]  u8    entries in each SP line (lines are RAGGED: entry's
                        global rank = prefix-sum(cnt) + slot — removal
                        touches ONE SP line, no cross-line shifting)
                        npages/live-slot mask derivable: page_line[i]!=null
 12   8   stem_lo[4]u16 first stem of each SP line — the header-resident
                        line index: 3 compares pick the ONE SP line to
                        load (heart's stem lower_bound trick)
 20   1   nbig          dedicated-line entries
 21   3   rsv
 24   4   fp_line       line_ptr → FP line
 28  16   sp_line[4]    line_ptr → SP lines
 44  48   page_line[12] line_ptr → sub-page (2 lines) or dedicated value
                        lines; null = free table slot (reusable for a
                        FRESH page); low spare bits tag {subpage | bigval}
 92  36   prefix[36]    common prefix of the collapsed radix tail
                        (longer prefixes keep a prefix_node above)
                                                                  = 128
```

## 3. SP line (stems + local refs) and FP line

```
SP line:  u16 stems[16]   sorted within the line   bytes   0..31
          u16 lref[16]    parallel local refs      bytes  32..63
          64 B reserved   (future: aux per entry / wider stems)
FP line:  u8 fp[4][16]    parallel to SP (line, slot) — loaded only to
                          disambiguate equal stems / pre-reject before
                          touching payload (never needed for stem scans)
```

16 per line keeps slot→line a shift+mask; stem_lo in CL0 means a point
lookup loads exactly one SP line. ~60 entries ⇒ 4 SP lines, matching
heart's observed occupancy.

## 4. lref — inline bits implicitly name an address RANGE

```
u16 lref = [big:1][spi:4][off:5][len:6]
  spi  → range id 0..15: CL0 maps it to a physical cline
         (page_line[spi]) — or arithmetic (base + 6 + 2*spi) when
         implicit_layout — Dan's "implicit range → explicit cline" rule
  big=0: payload cell in sub-page spi at byte off*8, allocated length
         len*8 (immutable; the cell's live vlen may shrink below it).
         ≤64B entries never cross the 2-line sub-page boundary
  big=1: page_line[spi] is a DEDICATED 1..len-line allocation owning the
         whole entry — values ≥64B waste <50% (line-rounded), never bloat
         sub-pages
```

Payload cell (the design5 leaf, finally variable):
`[klen u8][vlen u8][vtype u8][suffix klen][value vlen]`, padded to 8B.
Sub-page header: `[free_space8 u8][rsv 7B]` then bump-packed cells.
free_space is a DEAD-BYTE LEDGER, not an allocator: deleting a cell adds
its allocated size; closing the page (the bump moved past it) adds the
untouched tail. Nothing ever reallocates INTO it — COW regenerates nodes
constantly, so holes are not worth chasing. Its one job: when
free_space == capacity, the page is dead → the bucket releases its line
ref and nulls page_line[spi]; the slot is reused for a FRESH page.

## 5. Operations

**find(K)**: CL0 (prefix check, stem from next 2 bytes) → stem_lo ladder
picks SP line → SIMD lower_bound over 16 stems → equal-stem run: FP line
disambiguates → load payload via lref. 3–4 line loads total — same as the
router+leaf path it replaces (the bucket absorbs 1–2 router levels).

**insert**: rank by stem → target SP line; if its cnt=16, shift one entry
to a neighbor line (touches 2 SP lines, rare) or SPLIT at 64 total
(existing bkt_split semantics: burst into routers). Payload: bump into
open_page if it fits, else next page with room, else new sub-page
(class-2 alloc), else dedicated lines (≥64B or no room).

**remove**: delete slot in ONE SP line (ragged lines — no global shift),
clear FP slot, credit the cell's allocated size to its page's free_space.
When a non-open page's free_space reaches capacity it is RELEASED (line
ref dropped, table slot nulled). No intra-page reclamation ever: dead
space rides until the page empties or the bucket splits/rebuilds —
bounded by pages × 248B, and COW churn regenerates buckets often enough
in practice (the no-realloc rule keeps lrefs stable so clones never have
to rewrite SP lines).

**update value**: new_len <= allocated len AND line private this txn →
write in place, update cell vlen (page already dirty; no amplification;
free_space NOT credited — the cell stays live at its allocated size).
Growing → bump-alloc a new cell in the open page (old cell's allocated
size → free_space), rewrite lref in the (private) SP line.

## 6. COW (the whole point)

First touch per generation (bucket shared): clone CL0 (128B memcpy) +
retain its ≤17 referenced lines — one refcount cacheline of bumps, the
fx/cx pattern. Then per mutation, clone ONLY:

| mutation | lines copied (beyond CL0) |
|---|---|
| value update | 1 sub-page (2 lines) or 1 dedicated line |
| insert | 1 SP line (+FP line) (+payload as above) |
| remove | 1 SP line (+FP line) |

Typical COW touch = 2–4 line copies + ≤17 retains vs 16-line monolithic
copy, and vs leaf-model's router+leaf clone chain it is a near-wash on
writes while being ~2.7x denser and far faster to scan. Release is
line-wise (dying sub-pages/SP/FP lines free independently; shared ones
survive) — identical to fx/cx destroy rules. Clones are verbatim 256B
memcpys (no compaction — lrefs stay valid, SP lines untouched).

## 7. Allocator interplay

Only class-1 and class-2 allocations on the hot path (dedicated values
1..n). Per-line refcounts of one bucket's pieces land near each other
(allocated together by the same session) ⇒ retains/releases of a bucket
generation touch 1–2 refcount cachelines. The per-bucket page allocator
is the CL0 table itself: one open page bumping, dead pages released and
their slots reused for fresh pages — never a 16-contig alloc, no
freeset changes needed.

## 8. Open questions for sign-off

1. CL0 prefix cap 36B (longer keeps a prefix_node above) — enough?
2. 12 sub-pages × 256B + bigs: cap the bucket's total footprint (split
   when index full OR pages exhausted)? Proposed: split at 64 entries or
   12 pages, whichever first.
3. FP function: 1 byte of XXH3(suffix) — or suffix length byte (cheaper,
   weaker)? Proposed: hash byte.
4. Reserved 64B per SP line: leave reserved, or 21×6B stem+lref pairs
   (capacity 84) at the cost of the shift/mask indexing? Proposed: keep 16
   + reserved (indexing simplicity wins; capacity comes from splitting).
5. (resolved 2026-06-11: explicit composed form is the only bucket; no
   implicit variant, no allocator tier.)
```

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

No 16-contiguous allocation is required anywhere. (Optional later: a fully
IMPLICIT bucket — one contiguous run where fp/sp/page addresses are
computed from base+k, no CL0 table loads — created by bulk load and
clone-forward re-promotion, demoted to the explicit form on first COW:
the same implicit↔explicit cycle as node_full↔full_x. If implicit-16
creation proves common, the freeset grows a free_lines_16 hint tier; the
allocator already counts 16s as adjacent-8 pairs.)

Tags: implicit bucket = existing K::bucket (9); explicit composed bucket =
K::bucket_x (12). Both ≤ 0xE.

## 2. CL0 — the header cacheline (byte map, 128 B)

```
off  sz   field
  0   2   node_hdr      {has_term:1, nbranch:9 (≤64), prefix_len:6}
                        router_hdr-compatible at offset 0: term update and
                        set_or_update_term reuse verbatim
  2   4   term          line_ptr, value when the key ends at this bucket
  6   1   flags         {implicit_layout:1, open_page:4, rsv:3}
  7   1   npages        live sub-pages (≤12)
  8   4   cnt[4]  u8    entries in each SP line (lines are RAGGED: entry's
                        global rank = prefix-sum(cnt) + slot — removal
                        touches ONE SP line, no cross-line shifting)
 12   8   stem_lo[4]u16 first stem of each SP line — the header-resident
                        line index: 3 compares pick the ONE SP line to
                        load (heart's stem lower_bound trick)
 20   1   dead8         dead payload bytes /8 (compaction-at-clone hint)
 21   1   nbig          dedicated-line entries
 22   2   rsv
 24   4   fp_line       line_ptr → FP line      (computed, not loaded,
 28  16   sp_line[4]    line_ptr → SP lines      when implicit_layout=1)
 44  48   page_line[12] line_ptr → sub-page (2 lines) or dedicated value
                        lines; low spare bits tag {subpage | bigval}
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
  big=0: payload cell in sub-page spi at byte off*8, length len*8 (≤64B
         entries: never cross the 2-line sub-page boundary by construction)
  big=1: page_line[spi] is a DEDICATED 1..len-line allocation owning the
         whole entry — values ≥64B waste <50% (line-rounded), never bloat
         sub-pages
```

Payload cell (the design5 leaf, finally variable):
`[klen u8][vlen u8][vtype u8][suffix klen][value vlen]`, padded to 8B.
Sub-page header: `[used8 u8][dead8 u8][rsv 6B]` then bump-packed cells.

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
clear FP slot, add entry size to dead8/sub-page dead counter. Payload
space reclaims lazily: the next CLONE of that sub-page skips dead cells
(COW is the compactor); a bucket whose dead8 crosses a threshold compacts
fully on its next privatize.

**update value, same vlen, all-private**: write in place (pages already
dirty this txn — no amplification). Different vlen: new cell (old marked
dead), lref rewritten in the (private) SP line.

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
survive) — identical to fx/cx destroy rules. Cloning a sub-page compacts
it for free.

## 7. Allocator interplay

Only class-1 and class-2 allocations on the hot path (dedicated values
1..n). Per-line refcounts of one bucket's pieces land near each other
(allocated together by the same session) ⇒ retains/releases of a bucket
generation touch 1–2 refcount cachelines. The "16-bit free slot index"
(free_lines_16 hint tier) is OPTIONAL and only if/when the implicit
contiguous bucket fast path is added.

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
5. bucket_x only at first (created by bkt creation under COW); implicit
   K::bucket re-promotion deferred with full_x/cx re-promotion work.
```

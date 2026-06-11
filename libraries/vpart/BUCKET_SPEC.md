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
  spi  → range id 0..15: the implicitly-named address range; CL0's
         page_line[spi] maps it to the physical cline (the one and only
         translation — the table IS the per-bucket allocator)
  big=0: payload cell in sub-page spi at byte off*8, allocated length
         len*8 (immutable; the cell's live vlen may shrink below it).
         ≤64B entries never cross the 2-line sub-page boundary
  big=1: page_line[spi] is a DEDICATED 1..len-line allocation owning the
         whole entry — values ≥64B waste <50% (line-rounded), never bloat
         sub-pages
```

Payload cell (the design5 leaf, finally variable):
`[klen u8][vlen u8][vtype u8][suffix klen][value vlen]`, padded to 8B.
Shrunk cells stay self-describing (page-local walks step cell→cell by
ALLOCATED extent, and re-grow-in-place needs the headroom): when a value
shrinks below its allocation, vtype.has_cap is set and the byte at
value[vlen] records the original byte capacity — the record lives inside
the very gap it describes, so never-shrunk cells pay nothing. (Dan's
sign-trick equivalent — vlen<0, capacity at abs(vlen)+1 — adopted with a
vtype flag bit instead of the sign so shrink-to-empty has no -0 hole.)
Invariant: lref.len == roundup8(3 + klen + capacity)/8; lref.len is
authoritative for free_space accounting (remove has it in hand), the cell
record for walks and in-place re-growth; debug asserts tie them.
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

**update value**: new_len <= capacity AND line private this txn → write
in place: set vlen, and if new_len < capacity set vtype.has_cap +
value[new_len] = capacity (clear has_cap when re-grown back to exactly
capacity). free_space NOT credited — the cell stays live at its allocated
size. Growing past capacity → bump-alloc a new cell in the open page (old
cell's allocated size → free_space), rewrite lref in the (private) SP line.

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


# v2 — heart's diver rule adopted (Dan, 2026-06-10): pages + divider radix

Supersedes the v1 FORMATION rule (leaf-collision cb_make2_) and the v1
burst-by-lead-byte split. The composed line-refcounted LAYOUT (§1–§7)
stays verbatim; what changes is WHERE buckets come from, how they split,
and how the radix above them routes.

## v2.1 The diver rule (formation)

Keys terminate in a bucket from the FIRST insert: an empty tree's root
is one bucket; every insert dives into exactly one bucket (floor-routed,
v2.3) and lands there. Buckets therefore only ever appear by SPLITTING a
full bucket — every bucket starts at ~half capacity and fills until it
splits. No leaf terminals exist in page mode (router terms hold buckets;
big values still ride dedicated lines inside the bucket per §4).

Buckets are SELF-CONTAINED: an entry's full key = bucket.prefix +
cell.suffix, where bucket.prefix = the LCP of its entries (<=36B, longer
LCPs simply truncate and ride the suffixes). The radix path ABOVE a
bucket is its divider, NOT its prefix — key reconstruction never reads
the radix path. A key that lands in a bucket without sharing its current
LCP re-prefixes the bucket (rebuild with the shorter LCP — heart's
"factoring only ever loosens").

## v2.2 The split rule (shortest round divider near the median)

On overflow (65th entry or page table exhausted), entries are already
totally ordered (cb_place_). Candidates are the boundaries in the middle
window [3n/8, 5n/8]; for the boundary between entries a,b with shared
prefix p, the separator is full_key(b)[0..p] + v where v ∈ (a[p], b[p]]
is chosen to MAXIMIZE TRAILING ZEROS (snap toward a power of two). Pick
the candidate with (shortest separator, then most trailing zeros, then
closest to the median). Entries [0,i) rebuild as bucket L, [i,n) as R
(each with its own LCP); the old slot points at L and the divider D is
inserted into the radix mapping to R. The root case grows a setlist
whose TERM is L (the "" divider) — the radix root always carries the ""
divider. Dividers are short and round by construction: radix keys stay
short (Dan's constraint), and the radix above pages stays shallow.

## v2.3 Floor routing (the lower-bound radix walk)

The radix maps divider -> bucket with FLOOR semantics: route(key) = the
bucket of the largest divider <= key. The walk (heart's route_to_page):
descend the equality path; at each node remember the deepest fallback
(a node passed with a term or a branch below the taken one). On byte
mismatch: take the largest branch BELOW the key byte and descend to that
subtree's MAX divider; if none, the node's term; else unwind to the
remembered fallback. Key exhausting inside a prefix routes below every
divider through it (fallback); key above a node's whole prefix routes to
the subtree's max. Point ops floor-route then run the §5 bucket ops with
FULL keys (prefix verified in the bucket). The iterator walks the radix
in lex order (terms before branches — divider order IS bucket order) and
yields self-contained bucket entries; ++ crosses buckets exactly like
v1 frames.

## v2.4 Remove / collapse

Bucket empties: non-root bucket -> remove its divider from the radix
(ordinary term removal + rm_shrink_ collapse) and release the bucket.
The "" bucket may sit empty while other dividers remain (floor still
routes to it; finds miss). When the LAST divider goes, a root reduced to
a term-only router with an EMPTY "" bucket nulls the root — remove-all
drains the arena to zero, same gate as v1.

## v2.5 COW

Unchanged from v1 §6: privatize the radix path to the bucket slot
(floor-aware walk), then the bucket self-privatizes per piece (cb_own_).
Splits run on the privatized bucket; L/R are fresh; the divider insert
privatizes the (already private) path. Line-granular sharing, demotions,
and the occupancy ladder (phase 1) apply to the divider radix verbatim.

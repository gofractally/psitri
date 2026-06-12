# vpart L1a first head-to-head: vpart vs part vs heart (bank, 1M/10M)

Conditions: 2026-06-11, macOS arm64 (M-series), Release -O3, names key mode,
batch=1, sync=none, 1 reader thread, runs sequential (no co-tenancy).
Cmd: bank-bench-{vpart,part,heart} --num-accounts 1000000 --num-transactions 10000000
Git: vpart L1a stamp-liveness (this commit); part/heart at same tree.

| Engine | Write-only tx/s | Write+Read tx/s | Reads w/ writer | Reads solo | file MB | live MB |
|---|---|---|---|---|---|---|
| vpart  | 344,300 | 352,358 | 4,959,100 | 6,554,589 | 3038 | 1836 |
| part   | 372,338 | 358,376 | 4,572,180 | 5,762,576 | 2391 | 1843 |
| heart  | 263,209 | 239,560 | 2,775,390 | 4,182,685 | 1120 |  665 |

vpart vs HEART: writes +31%, write+read +47%, concurrent reads +79%, solo +57%.
vpart vs PART:  reads +8.5% concurrent / +13.7% solo (no fullx/cx demotion —
the hot top of the tree stays implicit skip-CL0); writes −7.5% (node-grain
clones memcpy whole multi-line runs; per-line stamps are the recorded L1c
follow-up), high-water +27% (drain lag + 10-line husks — same follow-up).
All runs validate: balances, log entries, account counts exact.

## Round 2 (same evening): variance-controlled medians + attribution

3 alternated runs each (rounds/ dir), create-log removed from vpart's alloc path
(abort = window walk now — zero log traffic on allocation):

| Engine (median of 3) | Write-only tx/s | Reads solo |
|---|---|---|
| vpart  | 328,780 | 5,745,513 |
| part   | 356,562 | 4,652,108 |
| artmap (transient heap, Dan's baseline class, 1 run) | 1,317,358 | 3,269,536 |

**Attribution (census + alloc-volume evidence, 1M/2M runs):**
- READS vpart +23%: identical tree shape (12,590 wide routers each), but part's
  are 100% DEMOTED to explicit full_x/cx (census: full_x=12590, node_full=0 —
  every wide router demotes on first COW and stays demoted; each descent pays
  CL0→line indirection). vpart's are 100% implicit node_full (skip-CL0,
  single-load dispatch). Structural: refcount COW requires the explicit form
  for sub-line sharing; stamp COW does not.
- WRITES part +8%: vpart clones implicit wide routers WHOLESALE — 31.25 GB
  allocated vs part's 10.89 GB on the same 2M-txn workload (2.9x). part's
  demoted clone = 1-2 lines + ~18 refcount RMWs; vpart's implicit clone =
  10 lines + 1.25 KB memcpy + drain.
- ONE trade, two faces: explicit = cheap COW / slow reads; implicit = fast
  reads / 10-line COW. part is stuck explicit after first COW; vpart sits
  implicit. On EITHER side vpart >= part (no refcount traffic), the configs
  just sit on opposite sides today. The designed resolution is COW-time
  placement + background re-promotion (spec §3 / mover) — clone cheap,
  read fast.
- artmap reference: 1.32M tx/s writes = the in-memory ceiling (no durability,
  no freeset, bump-only); reads BELOW both (std::string leaf values indirect).

## Round 3: honest artmap layout + the versioning-cost ladder

ArtMapEngine fixed to INLINE value bytes (upsert_inline; same-size updates
overwrite in place — the std::string layout paid a heap hop per read and a
heap alloc per update). ARTMAP_VERSIONED=1 mints a COW boundary per batch
(isolates versioning cost; no reclamation exists, so it cannot run 10M txns —
the bump arena exhausts at ~2.5M: reclamation is not optional at bank scale).

WRITES (tx/s)                                   READS solo (reads/s)
  artmap inline, NO versioning   1,679,023        vpart           5,745,513
  artmap versioned (transient,     ~680-830k      artmap inline   5,411,585
    no reclaim, OOMs at 2.5M)                     part            4,652,108
  part                             356,562        heart           4,182,685
  vpart                            328,780
  heart                            263,209

Reading the ladder: persistence is the OS (mmap, sync=none) — NOT the gap.
Version-per-txn COW costs ~2x even on a pure bump arena (1.68M -> ~750k);
the durable allocator + clone-log + drain + implicit whole-node clones cost
the next ~2.2x (750k -> 329k). vpart READS exceed the transient in-memory
art_map — the dispatch contract is more than met; the write levers are the
allocator path and the wide-router clone volume (placement/L1c).

## Round 4: art_map<btree_value> — the dwal-exact Value

ArtMapEngine now stores psitri::dwal::btree_value (kind data|subtree|tombstone
+ string_view data + tree_id), value bytes inline after the struct, view
fixed up exactly as btree_layer::store does. Same-size in-place overwrite kept
(standalone bump arena has no swaps).

  art_map<btree_value>:  1,529,285 tx/s write-only   5,080,324 reads/s solo
  (vs bv{u32 len}:       1,679,023                   5,411,585 — the 32B-vs-4B
   Value struct costs ~9% writes / ~6% reads at bank value sizes)

LANDMINE (confirmed by validation failure + fix): btree_layer's pattern of
storing an ABSOLUTE pointer in bv->data and trusting it on read breaks when a
structural op relocates a leaf — the bank run drifted +3.65B in balances until
val_of_ derived the pointer positionally ((char*)v + sizeof(bv)). In dwal the
same stale pointer would dangle into a SWAPPED (freed) arena. Needs a dwal
repro/audit: either art leaves never relocate in dwal's usage (then why did
the bank's?) or btree_layer should derive the pointer positionally too.

## Round 5: artmap versioning aligned with commit_batch (env knob removed)

commit_batch() now seals a COW generation (bump_cow_seq) — same commit
semantics as every engine, scaling with --batch-size; the in-place same-size
hack removed (sealed generations must never be scribbled; always-realloc is
dwal-exact btree_layer behavior). Consequence measured honestly: standalone
art_map cannot run bank at 1M-account scale AT ANY batch size (u32 bump
arena = 4 GB lifetime budget, no reclamation, ~100B+ realloc per update).
Survivable envelope 100k accounts / 200k txns @ batch=1:

  art_map<btree_value>, commit-aligned: 711,953 tx/s   6,296,790 reads/s solo
  (vs vpart at the same 100k scale: 474,661 tx/s — and vpart runs 10M+ txns
   indefinitely; reclamation is the difference, not the tree)

## Correction (Dan): the sweep claim ignored artmap — honest four-engine table

At 1M/10M artmap is DNF (generation-per-commit on the 4 GB bump arena
exhausts ~2.5M txns) — but DNF is a row, not an excuse. At its envelope
(100k accounts / 200k txns, rounds/bank_small_table.txt):

  artmap   713,309* w   761,648* w+r   (reads n/a: single-threaded)  live 1,133.6 MB
  vpart    386,251      470,479        7.03M* w/wr  8.35M* solo      live 59.2
  part     442,358      449,907        6.02M        7.09M            live 58.3
  heart    307,072      270,265        3.66M        4.96M            live 17.9*

Readings: (1) the transient no-reclaim ceiling is 1.6-1.8x the durable
engines' writes — that headroom is what bottom-up COW (#23), per-line
stamps (L1c) and bump-region allocation are chasing; artmap pays 19x
vpart's live bytes for it and cannot scale or serve readers. (2) at this
small/cachey scale part out-writes vpart (442 vs 386) — the 16-line full
clones cost relatively more when memory bandwidth isn't the constraint;
the 1M-scale ordering (vpart 380 > part 349) is the realistic one. The
"sweep" claim holds only among the durable versioned engines at scale.

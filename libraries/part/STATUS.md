# part — end-to-end build status (2026-06-11, branch `part-lib`)

Everything below was built and measured overnight per the three gates:
near-libart reads/unique-writes, MATCH hkv on read-only, beat heart on bank.
All numbers: M5, one process at a time, contestants back-to-back per rep.

## What exists

| layer | file(s) | state |
|---|---|---|
| freeset (SPEC §6) | `include/part/freeset.hpp` | l0 authority + g2/g4/g8 + meta/top tiers, exact via responsible-toggler; classes, 16=8-pair, odd sizes via guaranteed-tier navigation + in-word runs; anti-split; low_water |
| arena (SPEC §5,7-11) | `include/part/arena.hpp` | data/free/refs/meta files on sal block_allocator; qualifying = free-8-contig; longest-run + continuity claiming; txn floor + dirty masks + coalesced msync; u8 refcount bands (clone@190, pins 191..254, 255 throws); 64 reader slots; recovery hooks; session word-window + hot-hole cache (flushed at commit — txn-boundary state exact) |
| tree | `include/part/hkv/*` | hkv VERBATIM (same layout, same descent) over `part_alloc` (4-byte indexed handle = 28-bit line + 4-bit kind) |
| COW | tree.hpp part-extension | privatize-then-mutate; `full_x`/`cx` = design5 implicit→explicit demotion (line-granular sharing, children ride the lines); overflow re-promotes to implicit; line-wise destroy |
| surface | `include/part/kv.hpp` | psitri-shaped database→write_session→transaction→tree→cursor; commit = msync → retain+set_root → release old; snapshot cursors pin under the root mutex |
| shape | `include/part/db_shape.hpp` | concepts; kv-compat static_asserts heart+psitri+part |
| tests | `test/*` | 1.07M assertions, 24 cases; TSan suite; fork+SIGKILL durability; kv battery mirroring kv-compat |

## Gate results

**Read-only (dict) — gate: MATCH hkv.** MET: hit 0.95x, miss 0.96x vs hkv
(and 0.92x/0.73x vs libart — libart loads a header per hop, we don't).

**Reads + unique writes — gate: near libart.** MET-ish: u64 hit 0.87x / miss
0.82x of libart (faster); u64 insert 1.12x, dict insert 1.20x libart —
inserts now match/beat the hkv gold standard itself (0.97-0.98x), and hkv
trails libart on inserts by design (radix split work).

**Bank vs heart (1M accounts / 10M transfers, batch=1, sync none):**

| metric | part | heart | ratio |
|---|---|---|---|
| bulk load | 6.27M ops/s | 3.27M | **1.92x** |
| write-only transfers | 386.7k tx/s | 268.0k | **1.44x** |
| write+read transfers | 229.2k tx/s | 240.6k | 0.95x |
| reader (with writer) | 1.62M reads/s | 2.91M | 0.56x |
| reader (solo) | 5.83M reads/s | 4.44M | **1.31x** |
| file size at end | 4.9 GB | 1.1 GB | 4.4x worse |

At 50k-account scale part wins write+read 1.27x; at 1M it is par. The
demotion work moved write+read from 0.58x → 0.95x; profile-confirmed the
remaining interference is generation lifecycle (clone/destroy churn + the
reader's per-batch unpin cascades contending the freeset).

## NEXT SESSION, FIRST: variable-length leaf values (Dan's directive)

pval must NOT be size-limited: the leaf is sized to the user data —
prefix + postfix(suffix) + valuetype + value in ONE object (the design5
leaf). The current fixed-80B pval is interim. Implementation map (the
fixed-T model is the only thing in the way; the SUFFIX is already
variable):
- leaf layout -> [slen u16][pfx u16][vlen u16][value bytes][suffix/key bytes];
  introduce marker type `hkv::bytes_value{string_view}` and gate with
  `raw_vals = is_same_v<T, bytes_value>` in: leaf_value_off/leaf_size/
  leaf_alloc_size (+leaf_vlen), leaf_val/leaf_str, make_value_at,
  split_leaf (byte-copy not move), the Assign overwrite paths (insert_
  leaf case + set_or_update_term: same-vlen memcpy in place, else
  REBUILD leaf + replace slot/term handle + free old), find/val_of_/
  copy_value (view out; part::kv copies to std::string immediately),
  iterator value() (mutable bytes_value cache instead of set_ptr_),
  free_all_/cow paths already byte-copy/size via leaf_alloc_size.
- bucket code paths must compile but stay unreachable (mode::none);
  gate bkt_val uses with if constexpr.
- then part::kv drops pval entirely: upsert takes value_view straight in,
  valuetype byte reserved next to vlen for future typed values.

## Phase 1: no-bucket + occupancy dense ladder (2026-06-10, gate default 0)

Dan's size directive, phase 1. The grow ladder is now
setlist(16) -> {c2|cx1} -> {c4|cx2} -> {c8|cx3} -> {node_full|full_x},
the form at each rung chosen by TARGET occupancy: implicit monolithic
(skip-CL0 1-load descent) only when <=1 of its range lines would be
empty, else the explicit per-line form where empty ranges are NULL lines
(materialized on first insert, released the moment a remove empties
them). COW demotion (implicit -> explicit on first shared touch) also
nulls empty ranges now, so write-hot subtrees converge to the shareable
explicit form and read-mostly ones keep the direct path.

**Latent bug found by this work: cnode_t::cl0pad was a hardcoded 120**
(= 128 - the 6-byte handle's 8-byte router_hdr). Under part's 4-byte
handle the header is 6 bytes -> segs sat at offset 126 and ALL
line-boundary machinery (demote from_off, line-wise destroy, per-line
refcounts) was off by 2 bytes. Struct-relative readers masked it; the
first line-granular clone read garbage. Pad is now derived from the
handle width + per-instantiation layout asserts (node_full always had
them — why the full tier never broke).

Census (fresh loads, gate 0):
- dict 236k: the 404 sparse fulls (10 lines, 20% occupancy) are now
  403 implicit c2 (3 lines, 100%) + 1 cx. Sparse-full waste gone.
- bank-shape 3M: 9.1k dense log routers = 1.3k implicit full + 7.8k
  full_x (90,989/90,990 lines used). Totals unchanged (32MB / 300MB) —
  leaves and 2.5-branch setlists dominate, as the census always said.

Perf (same-session A/B vs pre-ladder, gate 0 both):
| metric | pre | post |
|---|---|---|
| dict insert | 6.39M/s | **7.39M/s (+16%)** |
| dict get | 20.4M/s | 17.5M/s (−14%) |
| dict scan | 43.5M/s | 47.4M/s (+9%) |
| bank write-only / w+r | 363.5k / 345.9k | 353.9k / 348.4k (noise) |
| bank bulk load | 6.80M/s | 5.65M/s (−17%, ladder churn) |
| bank scan | 13.24M/s | 12.56M/s (−5%, fx +1 load) |
| bank file / live end | 2359 / 1837MB | 2391 / 1843MB |

Levers noted: (a) re-promote full_x -> implicit node_full when it
reaches near-256 (the 7.8k log fx pay +1 dependent load forever — the
bank scan/load deltas); (b) cnode_step's bitmap rank-decode costs ALU vs
full's direct index on hot top routers (the dict get −14%); a
direct-when-dense top tier or offtab-style c2 would close it. (c) bulk
load ladder churn: sequential fill walks every rung — could jump
setlist -> full_x when the insert pattern is recognizably dense.

## Phase 2 prep: heart's diver rule (studied 2026-06-10)

Heart is btree-pages-with-a-radix-index, NOT radix-with-bucket-leaves:
- bucket_node = a 4KB page of lex-sorted (suffix, value) SoA entries
  with a SIMD u16 divider index; **keys terminate in a page; the page
  FILLS UNTIL SPLIT** (so pages sit between half and full — avg ~60
  entries) and the split is what creates radix structure above.
- full_radix_node (bucket mode) stores single-key terminals INLINE in
  its own tail heap — sequential/dense terminal bytes collapse into the
  router page instead of exploding into per-key objects.
- the compactor shrinks rarely-filled pages back (cow_size grows to the
  working page, compact_size trims).
part's cbucket inverted this: buckets formed at 2-key leaf collisions
(cb_make2_) and averaged 6 entries over 5+ lines. Phase 2 = flip the
formation rule: terminals land in a bucket page from the start, the
bucket fills to its working capacity, and overflow bursts into a router
OF buckets (cb_build_ already does the burst); plus dense-terminal
inlining on full routers. The composed line-refcounted layout stays.

## Composed bucket — CORRECT and MEASURED (2026-06-10, gate=1 default at the time; now 0)

Full suite green with the gate on: 30 cases / 1.08M assertions + TSan.
Two final husk-class bugs (keys stored unreachable) plus two iterator bugs:

1. **split-retry depth rewind**: the burst re-encodes the bucket prefix in
   a prefix_node wrapper, so insert_'s retry must `depth -= pl` before
   re-descending (was landing keys pl bytes off — find-unreachable husk).
2. **term rides the burst rebuild**: a term bucket CAN burst into a
   prefix_node top (term + 64 entries sharing the next byte — dictionary
   keys hit it 343/1M); the post-hoc term reattach asserted there and
   under NDEBUG dropped the key + leaked a setlist. Now the term joins
   cb_build_ as the empty-suffix entry (an empty suffix also stops the
   shared-prefix strip, so the bad shape can't form). Regression case in
   kv_tests.
3. **suffix-as-full-key leaves**: cb-built leaves used make_value_at(sfx,0)
   which records the SUFFIX as the leaf's zero-copy FULL key → scans
   returned truncated/empty keys. cb_leaf_ clears LEAF_FULL
   (layout-identical at plen==0); iterator reconstructs from the stack.
4. **iterator self-pointer**: bucket-cell values materialize into the
   iterator's cvalbuf_; val_ pointed there and dangled under the
   DEFAULTED copy/move (kv cursor does `_it = view.begin()`), visible
   only under -flto=thin stack reuse. val_local_ flag; value() rebases
   off `this`.

**Measured (same-session A/B, M5, one process at a time):**

Bank 1M acct / 10M transfers, batch=1 (bucket ON vs OFF / vs heart):
| metric | ON | OFF | heart |
|---|---|---|---|
| bulk load | 6.23M ops/s | 7.21M | 2.94M |
| write-only | 361.3k tx/s | 383.6k | 259.8k |
| write+read | 362.1k tx/s | 372.3k | 233.2k |
| reader w/ writer | 3.84M/s | 3.97M | 2.68M |
| validation scan | 13.67M/s | 13.20M | 31.2M |
| file end / live end | 2747 / 1792MB | 2359 / 1837MB | 1120 / 663MB |
| file load / live load | 130 / 109MB | 194 / 154MB | 320 / 31MB |

Dictionary probe (236k words, value=word, ×3 get sweeps, ×5 scans),
AFTER the ordered-storage change (2026-06-10, no std::sort anywhere on
part's paths — cb_place_ keeps (stem,suffix) total order at insert; the
iterator walks rank->(line,slot) off the cnt bytes; bursts consume
cb_for_each_ directly; the missing descend_lb_ cbucket arm added):
| metric | ON | OFF |
|---|---|---|
| insert | 4.85M/s | 6.39M/s |
| point get | 18.7M/s | 20.4M/s |
| scan | 38.0M/s (was 25.6 with the sort) | 43.5M/s |
| live | **33.7MB** | 56.1MB |
Bank re-run: scan 13.57M vs 13.20M OFF (+2.5%), writes ~flat
(346.6k/343.9k ON vs 363.5k/345.9k OFF), live-at-load −29%, end file
still +16% (COW write-amp, not iteration).

**Reading:** the space win at rest is real (−40% dict live, −29% bank
load live). Everything else went the wrong way, and the original
motivation has shifted under us: the leaf model's scan (45.9M/s) now
BEATS heart's 31.2M — the 3.3x scan deficit that justified the bucket
predates LEAF_FULL zero-copy keys + the inline-frame iterator. The bank
scan barely moves (+4%) because 93% of scanned keys are log entries
(sequential, never bucketed). End-of-bank file is +16% WORSE with
buckets: a bucket COW touch retires ~5 lines (CL0+FP+SP+subpage) vs 1–2
for leaf+router, so batch=1 churn carries more outstanding garbage.

**Done: the sort lever** — ordered storage took the dict scan 25.6 →
38.0M/s; the remaining −13% vs the leaf model is per-yield work, and the
ordering change also FIXED a latent cross-line misorder (cb_pick_line_
stopped at remove-emptied lines and appended out of order — masked by
the read-time sort) and lower_bound-on-bucket (no descend_lb_ arm).

**Remaining levers:**
- yield: lazy value materialization (store cell ptr/len; from_bytes only
  when value() is read) — scans that only touch keys skip the 80B copy.
- COW: clone CL0+SP+FP as one 3-line ALLOC (1 alloc + 1 refcount line
  instead of 3) and/or revisit writer_clone_at for bucket lines — this is
  the end-file +16% under batch=1 churn.
Decision pending: space (−30..40% at rest) vs insert −24%/get −8%/scan
−13% (dict); bank ~flat on everything but load (−15%) and end file.

## Composed bucket — correctness session 2 findings (gate still 0, HISTORICAL)

Unit suite (test/cbucket_tests.cpp): GREEN — 20k-step oracle fuzz of the
raw ops (insert/update/remove/find, shrink/regrow-in-place, page
lifecycle, multi-line equal-stem runs incl. the duplicate-update shape).
Tree-level single-txn: GREEN to 2048 keys (the earlier WRONGVAL was the
REPRO's bug: oracle pre-filled from a 20k tape, verified after 2k).
Three more integration bugs fixed this session:
  - cb_build_ single-entry-with-term orphaned a leaf (alloc then
    reassign without free)
  - burst targets now scale setlist->node_full (ignored setlist_set
    return silently dropped branches past CAP=16 groups)
  - cb_split_ recursion never drops oversized groups
REMAINING: a 2-line leak under GENERATIONS (commit + remove-all):
deterministic repro = 1200 "acct/N" keys, one bulk txn+commit, remove-all
txn+commit -> exactly 2 lines leak (a leaf + its prefix_node parent,
threshold = the 64-entry split; /tmp/cbl.cpp pattern, PART_TRACE_LINE
env hooks now in arena.hpp print a line's full refcount lifecycle).
Forensic state: the leaked line's trace shows a clean death during the
LAST remove, yet the ref byte reads 1 at dump with freeset agreeing —
i.e. an untraced re-allocation/write inside the rm-1199/commit window;
every arena ref-write site is hooked, so suspicion is on a path that
bypasses finish_alloc_ or a bulk write (hardware watchpoint showed the
granule going 0x0101010101010101). NEXT: software-watch ref(1) inside
commit's sub-steps (flush_fcache_/flush_window_/cascade) to bisect the
window, or instrument seg_freeset::alloc directly for the line.

## (prior notes)

All machinery is in tree.hpp and compiled (structures, lref, sub-page
lifecycle, self-privatizing ops, recursive burst, iterator frames, COW
arms, pval bytes-bridge = variable-length values). Oracle bisection fixed
three integration bugs (run-spanning equal-stem lookups; recursive split
with prefix_node wrappers — setlist PREFIX_CAP=8 smash; burst targets
scale setlist->node_full — dropped-branch bug). REMAINING: duplicate-key
update under churn inserts a second entry (WRONGVAL @ ~2k keys, repro:
/tmp/cbr*.cpp pattern). Next: the BUCKET_SPEC TDD pass (cell/page/SP unit
tests vs oracle), then flip PART_CBUCKET=1 and measure (expected: file
~psitri-class, scan gap closed).

## Found in OTHER engines by the generic suite

- heart: silent SIGABRT after a 1-key page split at nest=200 / ~580B keys
  (bounds reduced in kv_compat with a note).
- psitri: SEGV in cursor::find_impl during the adversarial phase —
  REPRODUCIBLE as of d84abf08 (layout-sensitive: identical inputs passed
  in earlier binaries). Blocks kv-compat x3 (heart PASS, psitri crash,
  part unreached; part's own kv battery mirrors these phases and passes).

## Open levers (next session)

1. **Reader interference / write+read**: per-batch pin→unpin frees a whole
   generation on the reader thread; batch the cascade (defer to an epoch
   list the writer drains, or pin per-N-batches in the bench adapter the
   way heart's read_session amortizes). Also: privatize currently
   re-descends per upsert (privatize+insert are two walks); fusing them
   into one descent saves ~30% of the write path under sharing.
2. **Space (4.9GB)**: survivors still block requalification at 0.125;
   options: per-size-class qualifying (free_count[0] for 1-line-heavy
   reuse), or clone-forward sweep of stragglers when a segment is nearly
   reclaimable (the design5 "stale data keeps moving forward" story —
   refcounts know exactly which lines are stragglers).
3. **Durable-mode bank** (sync_every): wired (commit msyncs dirty runs;
   write-amp counters in arena_stats) but unmeasured vs heart sync modes.
4. cow_remove fusion-under-sharing: rm_shrink_ leaves full_x/cx unshrunk
   (safe); de-widen for them not implemented (space-only).
5. asm-gate: the read descent gained two jump-table cases (full_x/cx);
   ratios show no regression, but the per-function asm diff discipline
   from hkv hasn't been applied to the new kinds.

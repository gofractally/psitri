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

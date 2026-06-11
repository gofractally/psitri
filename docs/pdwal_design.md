# PDWAL — part-backed Delta WAL over a heart base

Status: DESIGN (2026-06-11). No code. Replaces the dwal delta machinery
(`art::art_map` RW layer + arena + op-log WAL file) with **part** (COW,
refcounted, persistent) and the base tier with **heart** (page model).
Grounded in the measured numbers from `logs/` (2026-06-10/11 campaign) and a
full inventory of `psitri/dwal/` (21 headers, 4.4k lines).

## 1. The pipeline

Three refcounted roots per logical tree, all in SAL root slots:

```
            W (writer part root)         S (stolen part root)        H (heart root)
 writer ──► part COW txn lineage ──┐
            commit = publish W     │ steal at txn boundary
                                   ▼
                        background merger: key-sort-order
                        merge S into heart working tree ──► publish H', release S
```

- **Writer thread**: each transaction COWs from the current W lineage; commit
  publishes the new part root (W). The next transaction COWs from that same
  root — the delta *accumulates* across transactions.
- **Steal**: when the merger is free (and a trigger fires — §6), it takes a
  ref on the latest committed W at a transaction boundary and marks the
  lineage stolen (S). The writer's next transaction starts an **empty** part
  tree.
- **Merge**: the merger iterates S in key order, applying into heart's
  working tree (upserts, tombstone deletes), possibly across many internal
  heart transactions, and publishes the heart root **once, at completion**
  (atomic delta application — readers never see a torn merge). Then releases
  S (part's arena recycles it) and goes back for the next steal.
- **Reader views** (maps onto dwal's `read_mode`):
  | view | tiers | dwal equivalent |
  |---|---|---|
  | base | H | `trie` |
  | durable | S ∪ H | `buffered` |
  | latest | W ∪ S ∪ H | `latest` / `fresh` |
  Reads are at most **3-tier always** (single stolen generation policy).
  Precedence: W shadows S shadows H; tombstones in a higher tier hide lower
  entries. The writer's in-transaction view is its uncommitted working tree ∪
  S ∪ H (read-your-writes for free — the working tree IS a part tree).

## 2. Why part beats the art_map + WAL file (justification)

1. **The log and the memtable collapse into one structure.** Today every op
   is written twice (WAL record bytes + ART node) and recovered by replay
   (header scan, per-entry xxh3 validation, torn-entry stop, multi-tx
   completeness filtering — `wal_reader`/`dwal_database_impl::recover`, ~300
   lines of protocol). A committed part root *is* the durable, already-applied
   state: recovery = adopt the W/S/H roots from SAL's root table and resume
   the merge. No replay, no WAL format, no truncation logic (releasing S
   recycles its segments automatically).
2. **The delta spills.** The RW arena is RAM-bound; dwal carries three
   backpressure mechanisms (max_rw_arena wait, adaptive throttle_sleep,
   max_rw_entries) because an oversized delta is fatal. A part delta is
   mmap-backed — a burst that outruns the merger degrades to disk speed
   instead of stalling; throttling becomes a policy choice, not survival.
3. **Savepoints become O(1).** dwal's `undo_log` (235 lines, 6 entry types,
   arena string copies, frame replay on abort) exists because the ART layer
   can't snapshot. part is COW + refcounted: savepoint = retain the current
   working root (ref>1 flips subsequent mutations to COW automatically);
   rollback = re-adopt the pinned root and release the abandoned lineage.
   Nested transactions, statement rollback, and full abort are all the same
   O(1) operation. The undo log is deleted, not ported.
4. **Snapshot machinery unifies.** dwal's cow_coordinator/COWART head/prev +
   epoch registry exist to give readers stable views of a non-COW structure.
   Every PDWAL tier is a refcounted COW root: a reader view is a set of root
   pins (part already ships hazard-slot publish-validate for `latest`).
5. **Crash-safe mid-merge by construction.** S is persistent and the merge is
   idempotent (upsert/tombstone semantics): if the merger dies, reopen finds
   S and an unpublished heart working lineage, discards the latter, re-merges.
   Parity with RO-WAL replay, minus the WAL.
6. **Measured costs favor it.** The delta is small by construction (drained
   continuously): part at ≤10M keys inserts at 130-200ns and reads at
   ~80-130ns (logs/2026-06-10_part-scale…, early rows) — log-append-
   competitive on the write path, and the read tax for `latest` is two
   shallow hot probes (~+200-300ns over heart's 566ns base read at 1B keys).
7. **Heart as the base tier** brings 29 B/key live density (5.7× the node
   engines), 16ns/step iteration, and the merge's key-sorted page visits
   re-cluster the base into key-sequential segments every cycle —
   compounding locality the current psitri base never gets.
8. **One structure family.** art_map + bump arena + range_tombstone_list +
   wal_format/reader/writer/status all retire; part/heart are already
   shipped, tested (kv-compat: one test body drives both), and benched.

## 3. What today's dwal does that this spec must address (the gap list)

Ordered by risk:

1. **Crash-consistency proof for part.** The WAL's per-entry checksums and
   torn-write semantics are battle-understood; part's claim to replace them
   rests on SAL's sync + part's root-publish ordering. REQUIRED before any
   cutover: a kill-9 crash-fuzz harness for part (the psitri `[recovery]`
   suite shape): commit N, kill at random points (including mid-msync),
   reopen, verify exactly the committed prefix is visible. Includes the
   `checksum_on_commit` story and segment-replay semantics.
2. **Range tombstones** (`remove_range`, used by sqlite table clears; dwal
   has a dedicated `range_tombstone_list` with O(log R) shadowing, merge-on-
   insert, split-on-insert). part has no range-delete representation.
   Options: (a) sidecar range list carried per delta root (ports today's
   structure, consulted by the 3-tier cursor and applied at merge via heart
   range-remove); (b) first-class range-tombstone VALUE in part. Spec choice:
   **(a)** — smallest part surface change, proven structure; revisit (b) if
   sidecar/root lifecycle gets awkward.
3. **Subtree values** (`btree_value::kind::subtree`, `sal::tree_id` payload,
   refcount cascade through layer destruction). PDWAL stores subtree refs as
   part values with a discriminator (heart already has `value_type` +
   first-class subtree values). OPEN: the delta (part arena) holding refs
   into the base (heart arena) is a **cross-allocator refcount edge** —
   needs an explicit protocol (retain-on-store in the owner arena, release
   routed back on delta-entry death/overwrite/merge), or a restriction that
   subtree values live only in the base and the delta stores them by id +
   pin table. Must be decided before sqlite parity.
4. **Multi-root atomic commit.** dwal: 512 independent roots, sorted lock
   acquisition, multi-tx WAL framing (tx_id + participant count) so a
   cross-root commit recovers all-or-nothing. PDWAL options:
   (a) **one part delta, table-id-prefixed composite keys** — a single root
   publish is atomic across all tables by construction; merge demultiplexes
   by prefix into per-table heart roots. Kills the multi-tx protocol
   entirely. Cost: one merge stream (shard by prefix range if needed), and
   per-table generation tracking moves into the merger.
   (b) per-table part deltas + a tiny commit-record root.
   Spec choice: **(a)** — atomicity by construction beats reconstructing
   multi-tx framing; revisit if per-table merge parallelism is proven needed.
5. **Read-mode parity.** `latest` (writer-coordinated) and `buffered` map
   directly. `fresh` — dwal's zero-coordination slightly-stale read via the
   prev_root — needs part to publish a **prev** slot alongside head (trivial:
   keep one extra retained root). `trie` = heart root. Keep all four names.
6. **Steal/merge triggers + backpressure knobs**: keep `max_rw_entries` →
   max delta keys, `max_wal_bytes` → max delta bytes (part arena size),
   `max_freshness_delay` (force a steal so readers' `durable` view advances),
   idle flush. Adaptive throttle survives as an optional knob with disk as
   the new ceiling. **Hard guard**: writer halts if the un-stolen delta
   approaches part's 32 GB address ceiling (merger death scenario); at any
   sane drain rate this is unreachable, but the guard must exist (or part
   gets the 5-byte/1TB handle first — independent track,
   logs/2026-06-10_part-ptr48-ab-20M).
7. **Heart merge API.** Today's drain uses psitri `upsert_sorted` with
   sibling prefetch. heart needs the equivalent: a sorted-merge entry that
   routes once per page, applies the run of deltas belonging to that page in
   one COW, prefetches the next K target pages via the radix (the QD lever —
   measured QD1 fault floor ~20µs/op vs batched fault bandwidth ~5-15×
   better), and carries internal-transaction chunking with **single root
   publish at the end** (heart's write_session already keeps a private
   working tree across transactions; 1M-op txns are proven —
   logs/2026-06-10_heart-scale-1B).
8. **Lazy `remove_result`** (answer "did it exist?" without lower-tier lookup
   unless asked) — trivial to keep, just port the wrapper.
9. **Observability**: `wal_status` shared-mmap counters (layer sizes, merge
   wall/cpu, throttle, fsync counts) — keep the file, retarget the fields
   (delta keys/bytes, steal count, merge lag keys/seconds, heart publish
   count). The scale-bench/log-archive workflow consumes these.
10. **Sync ladder** for psitri-sqlite parity: PRAGMA synchronous OFF →
    part commit with sal sync none; NORMAL → msync_async on part commit;
    FULL → fsync; fullfsync → F_FULLFSYNC. The ack point is the part root
    publish at the chosen level. (Heart publishes need no fsync urgency —
    worst case the merge re-runs from S.)
11. **Thread contracts**: writer txn objects single-thread (unchanged);
    merger owns its SAL sessions on its own thread (heart bank-bench already
    proved the session-per-thread pattern); reader sessions per-thread with
    hazard slots (part proven).

## 4. Open questions (decide before building)

- Composite-key single delta (§3.4a): confirm sqlite's table count / key
  shapes don't blow part's key length or hot-path comparisons.
- Cross-arena subtree refs (§3.3): protocol vs restriction.
- Merge chunk size and whether the merger should prioritize key ranges that
  the writer is currently cold on (avoids stealing the writer's cache).
- Does `durable`-view iteration need S's range tombstones materialized, or
  is cursor-time shadowing (today's model) kept verbatim? (Spec: keep.)
- part crash-fuzz harness scope (§3.1) — this is the gating workstream.

## 5. Measured-number cheat sheet (from logs/)

| quantity | value | source |
|---|---|---|
| part insert/read, small tree | 130-200ns / 80-133ns | part-scale early rows |
| heart base read @1B | 566ns | heart-scale-1B |
| heart iteration @1B | 16ns/step | heart-scale-1B |
| heart live density | 29 B/key | heart-scale-1B |
| heart 1M-op txn ingest | 1.38µs/key sustained @1B | heart-scale-1B |
| cold uniform-random fault floor (QD1) | ~20µs/op | heart-scale-2.8B |
| part address ceiling | 32 GB (4-byte line_ptr) | part-scale-392M |
| 48-bit part handle cost | +0.8% mem, +11-15% reads | part-ptr48-ab |

# vpart version-lifecycle audit (Dan's pause, 2026-06-12)

Trigger: with NO flushing and NO background pipeline (tiered knobs disabled,
pure writer + probes), inserts degrade 164 ns -> 5.3 us by 61M keys and the
arena holds 260 B/key (16 GB for ~6 GB of live tree). The writer is being
blocked by vpart itself. Separately, the codebase has grown TWO parallel
lifecycle mechanisms where the design calls for one.

## 1. What exists today (two mechanisms, neither is the spec'd model)

**A. Published-version pins (readers).** A cursor publishes
`(version<<32|root)` into a registry slot (seq_cst publish-validate), the
drain POLLS all slots to compute a floor = min pinned version, and frees
supersession chains with death <= floor. Slot scan per drain pass; one
parked reader blocks every younger chain; floor computation is O(slots)
polling, not event-driven.

**B. Frozen-root handles (the new tree_root).** shared_ptr-counted handle
over {version, root}; last drop pushes the root to a REAPER thread that
deep-cascades the whole tree. Event-driven and correct, but it bypasses the
version sequence entirely: the reaper frees by reachability (cascade), not
by version, and coexists awkwardly with A (a warm-leg cursor holds BOTH a
pin slot and a handle).

## 2. The spec'd model (Dan)

ONE logical object per published version — the smart_ptr/tree_root
equivalent, not sal's smart_ptr but logically similar:

    version_handle { version seq, root, atomic refcount }

- Every holder (reader cursor, writer's current root, merger's frozen warm,
  crash-slot) owns a ref. Copy = ref++, drop = ref--.
- Commit publishes version v+1 and DROPS the writer's ref on v.
- refcount -> 0 pushes the VERSION (not the root) onto the GC queue.
- GC (one thread, event-fed, in version order): when version v dies and all
  versions < v are already dead+processed, release "everything possible with
  that version gone" — exactly v's supersession chain (the husks superseded
  AT v are visible only to pins < v... corrected: chain v is releasable when
  no LIVE version < v remains). A frozen tree that was never superseded is
  the degenerate case: its handle death releases the whole tree through the
  same queue (the reaper lane becomes a GC case, not a separate mechanism).

Properties: exact, event-driven (no slot polling), one mechanism for hot
versions and frozen trees, and the version sequence gives the floor for free
(a dead-version min-heap / ordered ring instead of slot scans).

## 3. Why the writer degrades ALONE (the no-flush 5 us)

Measured chain of causes, in order of likely contribution:

1. **Drain throughput vs batch-1000 production.** At ~2.5M inserts/s the
   writer produces ~1 chain line per ~31 ops plus husk traffic; the drain is
   one thread doing per-entry `vdrain_entry` + per-run frees + ring pushes.
   The arena's 260 B/key says ~10 GB of chains were waiting at 61M keys: the
   drain LOST the race long before any pin existed. Throughput, not floors.
2. **Arena growth poisons claim_.** claim_ is O(num_segments) under one
   mutex per window claim (most-empty scan, two passes with the fallback
   bar). At 16 GB = 2000 segments; at 26 GB (the tiered 1B attempt) = 3300.
   Once the arena is big, every window refill is us-class — the 4-6 us
   steady-state. Growth and slow claims feed each other.
3. **Recycle-ring capacity.** The writer adopts <= 512 recycled runs per
   txn from one SPSC ring; at batch=1000 the writer consumes lines faster
   than it adopts them back, so even drained memory returns slowly,
   pushing allocation onto fresh segments (growth -> cause 2).

## 4. Proposed rework (pending sign-off — no code until then)

1. **version_handle table** (Section 2) replacing both mechanisms: a small
   ring of {seq, root_raw, atomic<uint32_t> refs} indexed by seq window.
   Reader cursors hold handles (slot publishing retired or kept only as the
   drain's fast floor hint); tree_root becomes an alias of version_handle.
2. **GC unification**: one GC thread consuming the dead-version queue in
   order; per version: release its chain; frozen-tree death = cascade. The
   reaper lane and chain drain merge.
3. **GC throughput**: batch chain processing per pass; the cascade already
   bypasses per-key work; if one thread still trails, shard by segment.
4. **claim_ index**: replace the O(nseg) scan with a small free-space
   ordered index (e.g., 16-bucket histogram of segments by free_groups8 ->
   pop from the emptiest bucket; updated on free/claim) — O(1) claims at
   any arena size.
5. **Ring sizing/adoption**: adopt budget proportional to batch size, or a
   second consumer pass at window refill.

## 5. Unit tests required (Dan)

- version_handle: copy/drop across threads; refcount exactness under
  contention; commit drops the writer's ref on the prior version.
- GC order: version v's chain is NOT released while any handle < v lives;
  released exactly once after; frozen-tree degenerate case.
- Leak gates: free-lines accounting returns to baseline after (a) pure
  churn, (b) freeze/flush/drop cycles, (c) parked-reader release.
- Writer-isolation: batch-1000 insert-only run holds vpart-native ins ns to
  150M+ keys with a bounded arena (THE regression gate for this audit —
  fails today at ~46M).

## 6. Concept extension (LOCKED after Dan's review, 2026-06-12)

psitri's multi-root transaction design REVIEWED AND ADOPTED: ascending-index
lock acquisition enforced by assert (deadlock-free by construction), ONE
tree_handle abstraction for primary/additional roots and subtrees, abort
rolls back all roots, ordered (not atomic) cross-root publish — the concept
documents "ordered durability"; rotation makes the crash window idempotent
by publishing warm before hot.

The exchange primitive (Dan: swap_trees makes no sense):

    tree handle.set_root(tree)   // assign, RETURN the displaced tree

with `tree` the engine's owned-tree value type (psitri::tree exists; heart
has tree values; vpart/part implement as an owning root handle — in vpart
this IS the refcounted version-root: dropping the returned tree feeds the
version GC, unifying set_root with the standalone lifecycle).

Rotation, entirely in concept vocabulary, one multi-root transaction:

    auto h0 = tx.primary();              // hot
    auto h1 = tx.open_root(1);           // warm slot (empty in steady state)
    h1.set_root(h0.set_root(tree{}));    // hot reborn empty; warm = frozen
    tx.commit();                         // publish order: warm, then hot

Enforcement (Dan): tiered is templated on the concept ONLY and its test
suite instantiates heart/heart, vpart/heart, part/heart. vpart passes the
full engine conformance body standalone. No engine customization for tiered.

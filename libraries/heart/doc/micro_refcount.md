# Micro-refcount pool — per-range branch-set refcounts for COW-to-root

## Why
COW-to-root clones every node on the update path. For a dense (≤256-wide) node, the
canonical clone retains all ~256 children and destroy releases them — `O(fan-out)`
atomics per clone/destroy, the dominant refcount traffic at high fan-out.

The fix is a **second level of refcounting at RANGE granularity**: the 256 branches
are split into N ranges (e.g. 8×32 or 16×16); each range has one refcount that counts
**how many node-versions share that range's exact branch-set**. Then:

- clone a node = `+1` to each range counter (N atomics, not 256),
- destroy a node = `-1` to each range counter; the version that drives a counter to 0
  releases that range's branch pointers (from its own node array) — `O(fan-out)` only
  at the genuine last reference,
- diverge one branch = that range's set changed → allocate a fresh counter for it
  (retain the kept branches as that counter's base), release the old counter.

A range counter is a **smart pointer whose pointee is implicit**: it does not store a
pointer; it counts the branch ptr_addresses in its range, which live in the node array
and are identical across every version that shares the counter.

## The counter is its own lifetime (no separate free tracking)
The counter VALUE is the allocation state:

- `value 0` ⇒ slot is free.
- `alloc()` = find a 0 slot, `CAS 0→1`, return handle.
- clone node = `fetch_add(1)` each of the node's counters.
- destroy node = `fetch_sub(1)` each; the thread whose `fetch_sub` returns 1 (slot now
  0) releases that range's branch pointers AND the slot is thereby back in the pool.

Race-clean:
- alloc only claims 0 slots; clone only touches counters already ≥1 → never collide.
- alloc-vs-alloc resolved by the CAS.
- "was I the last?" is the `fetch_sub` RETURN value (atomic), and the branch release
  reads the *node's* array (not the slot), so a slot freed-then-reused by another
  node's `alloc` cannot cross-contaminate — the slot identity is implicit per node.

## Storage: 1-byte atomics, co-located, addressed by region-base + small offset
- A full `control_block` (8 B + its own allocation) is far too heavy for a counter.
  A counter is **one byte** (`std::atomic<uint8_t>`; x86 `LOCK XADD`, ARMv8.1 `LDADDB`).
  255 headroom ⇒ forks/overflow are astronomically rare (vs 15 for a 4-bit nibble).
- A node needs N (8–16) counters. Storing N×4-byte ptr_addresses blows the header.
  Instead the allocator **co-locates** a node's N counters in one region so the node
  stores one region-base + N small offsets (1 B offset if region ≤256 slots → 4 + 16 =
  20 B for 16 counters; fits the leftover header cacheline).
- clone copies the base+offsets (shares the counters), `fetch_add` each — no alloc.
- region full mid-divergence: spill the one new counter to a 2nd region (bigger handle)
  or, fallback, re-home the node's counters to a fresh region. NOTE: only counters this
  node UNIQUELY owns (value 1) can be moved; shared counters cannot (other versions hold
  the handle) → co-location is best-effort, never a correctness constraint.

## Two hard requirements (these set the architecture)
1. **Capacity ≥ the control-block space.** 8–16 counters per big node ⇒ the pool is a
   dedicated **pinned-RAM region**, sized at least like the CB space, deterministically
   addressed (region id → fixed location) so a persisted handle resolves after restart.
   This is a CB-sized SAL subsystem, parallel to `control_block_alloc`, not a slab.
2. **Counts are reconstructed from node data alone.** The counter VALUES are ephemeral
   (like CB refcounts and the node's `type`/presence hint): zeroed on open, re-tallied
   by walking the live roots. Only the HANDLE (region-base + offsets) is persistent, in
   the node. Recovery: zero the pool, walk every node reachable from the durable roots,
   `fetch_add(1)` the slot each (node,range) handle names; afterwards every slot holds
   its exact live share-count. Dead/unreachable versions are never walked, so a
   freed-then-reused slot stays consistent within the live set.

## Open problems (the real engineering)
- **Co-location vs sharing**: a clone's counters are shared (value>1) and cannot be
  moved, so a region that fills can't be compacted for live counters. Placement is
  best-effort; need a region-selection policy that keeps a node's set together at birth.
- **Cross-node fragmentation / recycle**: many nodes' counters interleave in a region;
  it recycles only when all slots reach 0. Likely wants a clean-region recycler (same
  shape as SAL's clean-page eviction, at slot granularity).
- **Region-full-mid-divergence fallback**: spill-to-2nd-region (variable handle) vs
  re-home vs fork. Pick one; measure.
- **Deterministic addressing**: region layout must survive restart so persisted handles
  resolve — fixed region map, like CB indices.

## Implementation vehicle: parameterize control_block_alloc
This is NOT a new allocator — it is `sal::control_block_alloc` with `T = u8` (1-byte
counter) and 128 slots per 128-byte line instead of 16. That allocator already provides
everything requirement #1 needs: zones-as-regions (`ptrs_per_zone`), the per-zone free
bitmap, least-filled-line placement, the `alloc_hint = span<ptr_address>` co-location
path (its doc literally targets "update 256 refcounts touching 32 not 256 cachelines"),
mlock pinning, and deterministic 32-bit indices that survive restart. Change-set:

- **element size 8→1**: zone byte size + mmap shrink 8×.
- **slot-within-line 4→7 bits**: affinity arithmetic `cacheline = ptr>>4 / slot = ptr&15`
  → `>>7 / &127` (16→128 slots per 128B line).
- **per-line occupancy counter width**: must hold 0..128 → 8 bits (was ≤16 → 4–5).
- **hint fallback policy**: prefer the hint's shared line, then SAME ZONE (not global
  least-filled), so a node's 8–16 counters stay co-located for small-offset handles.

Prefer templating control_block_alloc on (element, slots_per_line) so the CB pool and the
micro-refcount pool share one tested implementation; clone only if templating is too
invasive. The `retain`/`release`-to-0-frees semantics are the counter's own value (this
doc's lifetime model), layered on top of the alloc/free slot management.

### Free-slot tracking: atomic bitmap (DECIDED), not value-scan
Keep control_block_alloc's **per-slot atomic free-bitmap** for ALL instantiations,
including `colo_alloc<u8>`. The tempting "value 0 = free, SIMD-scan the counters" is a
**data race**: a non-atomic vector load of bytes other threads are atomically RMW-ing is
UB and **TSan flags it** (the build enables TSan). The bitmap keeps the scannable state
in `atomic<uint64_t>` words — found via relaxed *atomic* word loads + ffs/popcount, claimed
by CAS — so it's TSan-clean and the lazy-read→CAS race is benign (stale read → CAS fails →
retry, exactly as today). Cost: 1 bit/slot = **~12% overhead on the u8 pool — accepted**,
in exchange for TSan-cleanliness and keeping `colo_alloc<control_block>` byte-for-byte
comparable to the baseline (honest differential gate). The counter `atomic<uint8_t>` is
then touched ONLY by retain/release, never scanned.
A later, measured-only alternative if the 12% ever matters: pack counters into
`atomic<uint64_t>` words and free-find via the SWAR has-zero-byte trick
`((w-0x0101010101010101) & ~w & 0x8080808080808080)` with word-CAS ops — TSan-clean and
bitmap-free, but it reintroduces packed-word contention + can't-move-one-range divergence,
so only on real numbers, not speculatively.

### Migration strategy (zero risk to live code)
1. **Clone** control_block_alloc → generic `colo_alloc<T>`; `sizeof(T)` derives
   `slots_per_line = 128 / sizeof(T)` and the slot-within-line shift/mask (no magic 16).
2. **Keep zones a fixed PTR-COUNT (4M), not fixed bytes** — so `ptr_address→(zone,offset)`
   math is T-independent (handles resolve uniformly across restart) and only the per-zone
   mmap bytes scale with `sizeof(T)` (32MB for control_block, 4MB for u8). `colo_alloc<u8>`
   then spans the full 2^32 address space → ≥ the CB space (requirement #1, for free).
3. **Differential test (the gate)**: drive the SAME alloc/retain/release/hint op-stream
   through `colo_alloc<control_block>` and the original `control_block_alloc` side by side;
   assert identical conservation, comparable co-location quality, no perf regression.
4. Once proven, `colo_alloc<implicit_ref_count>` (1-byte atomic, retain/release→was_last)
   is just the narrower case. Existing code untouched the whole time.
5. **Only after colo_alloc is reliable for a long time**: redefine
   `control_block_alloc = colo_alloc<control_block>` and relegate the original to a
   differential-comparison harness. Not before.

## Build order (prove each layer before the next)
1. **Standalone SAL micro-refcount pool** (no engine): `alloc()→handle`,
   `retain(handle)`, `release(handle)→bool was_last`, region co-location, self-free at 0,
   deterministic addressing. Concurrent conservation test: N threads
   alloc/retain/release millions of slots; assert conservation, no slot double-handed,
   no lost free. Earn trust exactly like the segment allocator did.
2. **Recovery re-tally**: build pool from a set of "live roots" walks; assert the
   rebuilt counts equal the live counts. Test independent of the engine.
3. **Engine wiring** (replaces packed nibbles + settle + fork-on-overflow on
   `heart-rr-wip`): node header = region-base + range offsets; birth allocs, clone
   retains-each, destroy releases-each (last → release range branches), diverge allocs a
   fresh slot. NO settle, NO materialized-vs-deferred ambiguity, async-safe by
   construction.
4. **Validate**: `rrstress` (concurrent pinned snapshots, divergent write+remove
   lineages off shared ancestors, common-ancestor release, model-checked count+checksum)
   + 1M `rand`, both under build-di (free_range_tracker). Then measure vs the 1690 ns
   packed-insert number and psitri 2960.

## ROI caveat (decide before step 1)
After child-retain batching, the worst-case COW-to-root bottleneck is the node MEMMOVE
(copying the dense children array), not counter atomics — and at `share==1` the counters
barely move. So this subsystem's payoff is **correctness + clean async-safe construction
+ the heavy-multi-snapshot-divergence regime**, NOT throughput on the simple worst case
(the lever there is shrinking the dense node, psitri-style packed pointers). Worth
building for the architecture; don't expect it to beat 1690 ns on its own.

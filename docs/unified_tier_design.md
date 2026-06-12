# Unified Tier Design — version-stamp liveness, tagged placement, generational tiers

Status: DRAFT for sign-off. Distills the 2026-06 design conversation (art_map
seq-COW → stamp liveness → fork pins → tagged dual allocation → lazy tiers)
into a buildable, layer-by-layer plan. Layer 1 (`vpart`) is copy+modify from
`libraries/part`, not a rewrite.

---

## 0. Thesis

Replace per-node/per-line reference counting with **version-stamp liveness**:
every allocation is stamped with the `(fork, seq)` window that created it;
supersession is recorded in per-window **clone-logs**; reclamation is an
interval check against a small **live-version set** (snapshots ∪ fork branch
points), drained by a background scan. Refcounts survive only at genuine
DAG points (subtree-value handles) and as one coarse count per sealed
fork-base — never per line, never per child, never on the clone path.

What the measured history says this buys (receipts):

| Evidence | Implication |
|---|---|
| bounded-inline release: ~9x single-update win came from **release locality/contention**, not refcount atomics | the cascade walk is the cost center; stamps replace it with a sequential log drain that never touches live nodes |
| clone-fusion measured <1% end-to-end | don't oversell clone-side atomic savings; the win is release-side + O(fanout)→O(1) bookkeeping + correctness |
| heart COW-mode-latch bug (mutating snapshot-shared ref==1 children) | `ref==1` is not "private"; `stamp == current window` **is** — the bug class is structurally eliminated |
| async-release bug tally (take_for_destroy CAS, `_release_in_flight` phantom, pending_release_count miss) | the deferred-cascade machinery this design deletes has been a recurring defect source |
| part 48-bit handle A/B: +11–15% reads vs 32-bit; indirection ≈ same class | pointer width and indirection hops are real in-RAM costs → both must be *modes*, not constants |
| heart 1.65×-live high-water fight; 11KB/commit COW garbage | unrelocatable cold data and reclamation pacing are first-order; the design must answer both |
| dwal A/B: hkv +17% foreground, art_map 2× drain; verdict blocked on losing lock-free `latest`/`fresh` | stamp liveness is what lets the fast tree keep lock-free read modes — the blocker dissolves |

## 1. High-level data flow

```
                         writers (one per root; forks = one per fork head)
                              │ upsert/remove (in-place when stamp==window,
                              │  COW + clone-log append when shared)
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  HOT TIER ("W") — part-style: direct pointers, fixed addresses  │
   │  • fork deltas: one small tree per live fork over shared trunk  │
   │  • (fork,seq) stamps; per-window clone-logs + alloc-logs        │
   │  • snapshots/forks = pins in the live-version set               │
   │  • generational reclaim: logs retire wholesale as pins advance  │
   └───────────────┬─────────────────────────────────────────────────┘
                   │ DEMOTER (background)         policy, in priority order:
                   │  1. drop dead forks wholesale (their logs ARE the free list)
                   │  2. migrate aging pins → cold-tier COW roots (pin table CAS)
                   │  3. demote settled trunk versions — ONLY under memory
                   │     pressure in the fixed region (lazy, not eager)
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  COLD TIER ("H") — heart/SAL-style: indirected, relocatable     │
   │  • control-block handles (move = one CAS, all versions follow)  │
   │  • same stamp liveness in control blocks (no refcounts)         │
   │  • mover: MFU clustering + segment evacuation, reader-epoch     │
   │    grace instead of residency refcounts                         │
   └─────────────────────────────────────────────────────────────────┘

   READS
   • point lookup: probe fork delta → ancestor deltas → trunk → cold.
     A HIT short-circuits (latest version is authoritative; deletion
     tombstones shadow lower tiers). A MISS pays full depth.
   • lower_bound / ordered scan: N-way merge cursor across ALL tiers,
     always (no shortcut exists for order).
   • snapshot read: resolve pin → (tier, realization) → single-rooted
     read in whichever tier currently holds it.
```

Fork lifecycle (blockchain-shaped, the driving workload):
fork-from-any-snapshot is O(1) — add a branch-point pin, allocate a fork id,
open a delta. Losing forks drop wholesale (logs + delta arena), never having
cost the cold tier a write. The winner's settled prefix becomes trunk and is
demoted only past finality and only under pressure. "Trunk" is never
privileged — when the fork wins, the old trunk's post-branch writes are just
another fork id whose logs drop.

**Two operating modes for vpart** (same engine, one behavioral switch):

- **Standalone engine**: vpart is the whole database. Delete is a real
  delete — no tombstones exist in this mode. This is the L1 deliverable:
  conforms to the kv_engine concept, runs the bank tests by itself, and is
  complete without any cold tier behind it.
- **Diff layer over heart** (Phase 2): vpart's state is *relative* to
  heart's. Deletes of heart-resident keys become **tombstones** (heart has
  `x`; txn1 deletes `x` → vpart stores tomb(x); txn2 upserts `x` → the COW
  upsert replaces the tombstone with the value — restored). **Heart never
  holds tombstones**: the background push applies vpart versions to heart
  **in order**, deleting where it sees tombs, so heart's state is always a
  clean materialization. If vpart carries a blockchain fork, the merge
  creates a **parallel fork in heart** (a separate heart root), and every
  downstream query on that fork flows to that same fork's heart root.

## 2. Liveness model (the substrate everything shares)

### 2.1 Stamps — where the version lives (no look-aside on hot paths)

Requirement: the version must never cost a look-aside cacheline miss on the
write path. Two encodings, by tier:

- **Cold tier (SAL): in-header.** The 16 B `alloc_header` already carries a
  48-bit `version_t` — the stamp lives inside the object's own cacheline.
  Proposed split: 16-bit fork, 32-bit seq.
- **Hot tier (part/vpart): in-node, placed in existing slack.** (A
  positional/structural encoding was considered and rejected: first-fit-
  forward interleaves this-window allocations with old live lines at/after
  the floor, so position cannot answer privacy. A per-line side table is
  rejected too: 4× refs.part's footprint and a look-aside miss on the
  write path.) The stamp lives inside the node's own cachelines — the
  privacy test reads memory the writer is about to touch anyway. The key
  layout fact: at part's 4 B `line_ptr`, the router arrays are
  **LANES-capped (16), not byte-capped** — the hkv "fills the cacheline"
  comments are for the 6 B `packed_ptr`; the per-instantiation
  static_asserts only bind that alias. Placement per kind
  (router_hdr_t<line_ptr> = 6 B):

  | Kind | Layout @ 4 B ptr | Stamp placement | Cost |
  |---|---|---|---|
  | setlist_u8 | 6+16+64+8 = 94 B used of 128 | trailing slack (34 B free) | **zero** — bytes[]/branch[] offsets unchanged |
  | setlist_u16 | 6+32+64 = 102 B + 26 B pad | tail pad | **zero** |
  | node_full | CL0 = 6 B hdr + 122 B pad; 9 direct lines (31 ptr/line) | own + 9 per-line stamps (40 B) in CL0 pad | **zero** |
  | full_x / cx | CL0 ≤ 42 B used | own + per-line/per-seg stamps in CL0 | **zero** |
  | c2/c4/c8 | CL0 = 6 B + pad; segs CL1+ | own + per-seg stamps in CL0 | **zero** |
  | prefix_node | [plen u16][ptr][prefix…] | after ptr | −4 B prefix capacity per line |
  | bucket (512 B) | — | — | **dropped in vpart** (rectification checklist §7-L1) |
  | leaf (value_ptr) | [suffix][T], sized exactly | 4 B prefix | +4 B alloc; line-rounding absorbs ~97% (~3% of leaves gain a line) |
  | value_inline | in-handle | none needed | zero (no identity) |
  | cbucket | — | — | **dropped in vpart** (rectification checklist §7-L1); its stamp-budget problem is moot |

  Branch lines shared between router versions (full/cx demotions) carry
  their birth in the *referencing parent's* CL0; the 128 B CL0 memcpy at
  clone time copies the stamps with it (birth is immutable, so every
  referencing parent agrees). **Read-path byte-identity is the revised L1
  contract**: stamps sit in slack/pad/tail only, so descent dispatch, SIMD
  scan offsets, and skip-CL0 behavior are unchanged for every router kind;
  only leaf/prefix_node pay the 4 B, and census must report per-kind node
  counts so the A/B attribution stays honest.
- **The version IS the transaction: write direct, one boundary per
  commit.** Writers COW the trunk directly; the version counter increments
  at every commit, which publishes the new boundary. The in-place fast
  path is *within-transaction only*: anything stamped by an earlier
  transaction is COW'd on touch — which is precisely what makes the last
  committed root stable for readers with **zero coordination** (no
  reader/writer CAS dance, no waiting: the stamp discipline alone
  guarantees nothing reachable from a published boundary is ever mutated).
  Cross-commit always-COW matches the measured history — psitri got
  *faster* when the cross-txn in-place path was removed (MVCC-removal),
  because release locality dominates. Undo is symmetric with reclaim:
  abort = free everything in the transaction's create-log (all
  window-born → instant free) and drop its clone-log chain unpublished
  (the superseded trunk nodes were never dead). Until commit, readers are
  pointed at the last committed boundary.
- **Stamp width: 32 bits — the tiers make it sufficient.** The hazard is
  modular aliasing of the equality privacy test, and the safety quantity
  is not elapsed versions but the **live-stamp span**: if every live vpart
  stamp lies within a window < 2³² behind the counter, the only live stamp
  ≡ cur (mod 2³²) is cur itself — modular comparison is exact. The tier
  architecture enforces a narrow span by construction: the demoter
  continuously pulls aged data out of vpart into heart (which
  **resequences** onto its own in-header 48-bit `version_t` and **batches**
  many vpart versions per cold-tier commit, so heart's counter advances at
  batch rate and never remotely wraps). Hot data re-stamps itself by being
  rewritten; cold data leaves the arena. vpart never carries hours of
  versions, let alone 2³² of them. Backstop for the one gap — a
  rarely-touched node surviving in a no-pressure arena while the counter
  races: **aging joins pressure as a demotion trigger** (demote-or-restamp
  nodes approaching the horizon; stamp 0 is reserved as "ancient" =
  permanently shared, the conservative re-stamp target; counter skips 0 on
  wrap), alarmed in stats.
- **Privacy test** (the write path's only check): stamp == my transaction
  id → private, mutate in place; anything else → shared, COW. Local,
  latch-free. No unique→shared mode latch exists or is needed. How fork
  identity rides the stamp at L2 (fork bits vs one global txn-id counter +
  a tiny id→lineage map) is open decision §8.6 — per-commit versioning
  means fork bits eat seq lifespan, so global ids are favored.
- Stamps/watermarks are plain memory (single writer per fork; readers never
  consult them — they consume roots published with release/acquire).

### 2.2 Logs

- **clone-log** per transaction: append the old address when a line/node is
  superseded (COW'd away, removed, grown out of). Entries are the 4 B tagged
  raw pointer (kind rides in the tag — the reclaimer needs it to expand, and
  the kind derives the line count, so multi-line nodes and value spills are
  ONE entry each; no separate run encoding). Birth is read from the dying
  node's in-node stamp at reclaim time (content stays intact until freed).
  Commit publishes the chain under the new version; abort drops it
  unpublished.
- **The log is itself a part node type**: a linked list of 128 B lines,
  push-to-front — 31 × 4 B entries + one 4 B prev pointer per line. Log
  lines are ordinary session allocations, so they land in the writer's
  *active segments* (dirty-anyway pages): log writes cost zero additional
  sync under part's write-amp model. The chain head is per-transaction
  session state; a retired version's chain frees wholesale after its
  entries drain. Log lines are never referenced by tree branches, so they
  need no K tag (statically typed — "see header" by default).
- **create-log** per transaction: the nodes this transaction allocated.
  Undo is symmetric with reclaim: abort = free every entry (all
  transaction-born → instant free, no interval check) + drop the
  clone-log chain. At commit the create-log retires (its lines, being
  transaction-born, recycle immediately).
- **Ownership rule: your log owns what you bore.** Superseding a *foreign*
  node (stamp.fork ≠ mine) appends an owning entry only in v1's conservative
  mode if the node is trunk-owned and I am trunk; cross-fork supersession is
  recorded as tombstone traffic (deferred layer, §2.5), never as a second
  owning entry. One owner ⇒ no double-free.
- **Removal cascades become lazy expansion**: removing a subtree logs only
  its root (with its kind tag). When the reclaimer frees it (interval
  expired), the node content is still intact — the reclaimer invokes a
  tree-registered expand hook (same shape as part's §19 destroy callback)
  which logs the children at the *same death version*, recursively, level by
  level. The cascade still happens, but: sequential, background, read-only
  on content, zero touches of shared survivors' metadata.

### 2.3 Live-version set and reclamation

- The live set = active snapshot pins ∪ branch-point pins of live forks.
  Small (≤ pin-slot cap), enumerable, in shared memory.
- **Where version roots live**: a runtime published-boundary table (≤ 64
  slots): (root_ptr, version, sharer count). The writer publishes the new
  boundary at every commit; **readers never mint version roots** — a
  reader pin shares the most recent published slot, i.e. the end of the
  last committed transaction, never an in-flight state. Only boundaries
  someone actually pinned occupy a slot beyond the latest; zero-sharer
  slots recycle. Slots are derived state; recovery starts with exactly one
  (the durable root).
- **Reclaim rule**: an owned log entry with birth `b` (from the stamp) and
  death `d` (the log's version) frees iff **no live pin lies in [b, d)**
  on the owner's lineage path. The common case is a continuous drain:
  **every version from the oldest in the arena up to the oldest running
  reader reclaims**, except versions held by an explicit snapshot (e.g. a
  block boundary), whose pins block exactly the entries with
  `b ≤ pin < d` — what that snapshot can see — and nothing born after it.
  A fork's branch-point pin behaves identically.
- Same-transaction birth+death (created and orphaned inside one
  transaction) frees immediately, inline — the fast path part has today
  for unshared frees.
- Snapshot release = drop a sharer count. O(1). No walk.

### 2.4 What keeps a refcount

1. **Subtree-value handles** (true DAG points: a stored subtree is
   referenceable from places with no version-tree relationship). Coarse,
   rare, at the handle boundary heart already treats as the cascade seam.
2. **Sealed fork bases** in v1: one count per branch point = number of live
   descendant lineages (it's just the pin; releasing the pin is the
   decrement). Nothing per line.

### 2.5 Deferred layer: tombstone counts (designed, not built)

For long-lived divergent forks where v1's branch-point pin over-retains:
a count materialized at *first* cross-branch supersession ("live branches
that can still see this minus one"), incremented by later forks-in-scope
(fork is rare; it can afford to walk pending tombstones), decremented by
other branches' supersessions/deaths; zero → normal reclaim. On a linear
history the layer is bit-for-bit dormant. Build only when measurement shows
pinned-base amplification (§7 gates). The stamp format already accommodates
it — nothing in v1 has to change.

### 2.6 Cost accounting vs refcounts (quantified, not assumed)

The metadata cost *moves* from per-REFERENCE-event to per-DEATH-event. The
honest ledger (hot tier, per the §2.1 encoding):

| Event | refcounts (part) | versions (vpart) |
|---|---|---|
| share (COW of a referencing parent) | 1 atomic RMW, look-aside cline | **zero** |
| unshare (release, not last) | 1 atomic RMW, look-aside cline | **zero** |
| death of one line | 1 RMW + free-bit, inline cascade | 4 B sequential log append; interval check + free-bit at reclaim |
| death of an n-line contiguous run | n RMWs (contiguous counters) | **one** (ptr, n) entry |
| static storage | 1 B/line = 0.78%, permanent, separate file | in-node slack (routers: free; terminals: −/+4 B) + log, **transient** |
| steady-state log footprint | — | death-rate × reclaim latency |

The headline trade Dan flagged: a 64-line region dying line-by-line costs
refcounts one 64 B counter cacheline of RMWs, versions ~256 B of log (4 B
entries) — **4× the metadata bytes when death is bulk and scattered**. The
compensations: (a) log writes are sequential stores while refcount RMWs
scatter with the death pattern (the bounded-inline 9x lesson: locality of
release is the lever, not op count); (b) run-length entries collapse
contiguous death (multi-line nodes, `hkv_free_run`) to 8 B regardless of n;
(c) sharing traffic — the dominant event class for hot upper-tree lines —
goes to zero; every line dies at most once but is shared many times; (d) the
log is transient and recycles, refs.part is permanent. Concrete scale: at
heart's measured ~11 KB/commit garbage (~86 lines), 8 B entries ≈ 688 B/commit
of sequential log vs ~86 B of scattered RMWs — both trivial in bandwidth;
the difference that matters is scatter vs stream.

Where versions genuinely lose this trade: high scattered single-line death
rate **plus** a long-held pin (log accumulates at death-rate × pin age while
reclaim is blocked). That configuration — bank-rate churn under an
hours-old snapshot — is exactly what the L1 bench gate must report
(high-water vs live, log depth over time), not hand-wave.

### 2.7 Recovery

- Stamps, logs, pins, freesets are **derived state** (part SPEC §2.3
  principle preserved): never synced on commit, never trusted unclean.
- Unclean open: exactly one live version exists (the durable root). Rebuild
  = part's existing `rebuild_begin / mark_line / rebuild_end` reachability
  walk, *simpler than today*: marking claims the line and sets stamp 0
  (= ancient = shared = COW-on-next-write); re-marks from other parents are
  no-ops instead of refcount bumps. Logs are discarded. Current seq starts
  at 1.
- Clean shutdown: drain logs (reclaim to quiescence), persist stamps + L0,
  set clean flag — same dance as today's refcounts, same files, new payload.

### 2.8 Stored values are encodings, not structs (Dan)

API value types (e.g. dwal's `btree_value`: kind data|subtree|tombstone +
view + tree_id) are NEVER memcpy'd into nodes as literal structs. The node
stores a compact encoded form — `[tag|len][payload bytes]` — and the API
type materializes at the read boundary with views derived POSITIONALLY
from the stored bytes. Two reasons, both measured in the bank: (a) the
literal struct wastes ~28 B/leaf at small value sizes (~9% writes / ~6%
reads); (b) a stored POINTER (string_view) goes stale when a structural op
relocates a leaf — the bank drifted +3.65 B in balances until the view was
derived positionally, and in dwal the same stale view dangles into a
SWAPPED arena (read-after-free). AUDIT: dwal's btree_layer stores the
literal struct and trusts the stored view today. walart's bval
`[tag][payload]` staging is the pattern of record.

## 3. Memory model — tagged dual allocation

- **Two allocators, one tree, one tag bit per branch**: direct fixed-address
  lines (part-style) and indirected relocatable handles (SAL-style control
  blocks). The branch's tag says which; descent pays one predictable branch.
- **Static default carries the policy weight**: bulk bytes (buckets / leaf
  pages) allocate *relocatable*; routing nodes allocate *fixed*. Frozen-cold
  data in fixed memory never re-decides (no COW event ever fires on it), so
  the default must keep the bulk evacuable or fixed segments become the new
  1.65×-live leak. Heat only ever *promotes* into fixed.
- **Dynamic refinement at COW time, free**: the writer reallocating a node
  already holds `current_seq − stamp` (write-age) and the MFU bits
  (read-heat). Placement decision costs zero extra loads at the only moment
  it can be acted on.
- **MFU: 2-bit advisory saturating counters**, sampled (1-in-N descents),
  racy by design — off the correctness path, unlike refcounts. Storage:
  the bits refcounts vacate. Indirect side: spare bits in the 8-byte control
  block entry (48-bit ptr + 2-bit MFU + move-epoch + spare). Direct side:
  presence-bits-style side bitmap by grain.
- **Moves (cold tier only)**: one CAS on the control block; every version's
  parents follow the handle. Physical safety = segment-level reader epochs
  (enter at the handle hop; mover republishes, waits for drain, reuses) —
  replaces residency-by-refcount. Parked cursors epoch-stamp and refresh on
  stale (the cursor `refresh` capability already in the kv_engine contract).
- **Capacity decision to lock before v1 persists anything** (§8): 4-byte
  tagged branches cap the relocatable space at 2³¹ handles (~½ TB at 256 B
  objects) — under the 1 TB ambition. Options: 5-byte branches (part A/B
  says +0.8% mem, +11–15% reads) or accept the cap per deployment.
- **Graceful degradation is policy, not architecture**: fully provisioned →
  promotion threshold low → converges to part (direct everywhere).
  Constrained → fixed hot core + relocatable MFU-clustered body → converges
  to SAL. Same binary, same format, one knob.

## 4. Pin contract — pins are relocatable

- A pin is a **logical coordinate** `(fork, seq)`, held as an opaque pin id
  resolved through a **pin table**: pin id → realization (W coordinates | H
  root). Resolution happens at cursor create/refresh only.
- **Migration protocol**: materialize → CAS the realization entry → drain
  reader epoch → retire the W versions. The one ordering rule: W reclaim of
  the migrated versions waits for readers that resolved before the repoint.
- **The clone-logs are the diff stream**: consecutive pins differ by exactly
  the log entries between their seqs. Migrate the oldest pin as a full
  materialization; derive each next as a COW child by replaying the log
  delta. Cost ∝ diffs; cold-tier structural sharing keeps the chain compact.
- The cross-tier version map (which H root embodies which W version) is
  dwal's existing `ro_base_root`/`tri_root` correspondence, kept per fork.
- Contract honesty: **identical content, not identical latency**. A migrated
  pin reads the same bytes slower — correct for old snapshots, observable,
  documented.
- The whole contract is differentially testable: enumerate through the W
  realization, migrate, enumerate through the H root, byte-compare
  (kv-compat pattern). This test gates the migration layer (§7 L4).
- Consequence: **pins never block demotion**. The demoter may drain the
  fixed region unconditionally; pins follow the data. Fork-prefix demotion
  and snapshot lowering are the same operation (migrate-and-repoint).

## 5. Tier policy

- **Lazy, pressure-driven demotion** replaces eager merge: data stays in the
  fixed tier while RAM permits. Dying-young data (lost forks, hot rewrites)
  never reaches the cold tier — write amplification falls by the death rate.
  Safe to be lazy because part is a durable *tree*, not a log: recovery is
  mmap + root validation regardless of how much is parked (no replay-time
  penalty for laziness).
- **Victim policy**: settled-first (stamps tell the demoter what's past the
  branch points / finality), then coldest-first (MFU bits). Both signals
  already exist in the substrate. Demotion **resequences**: heart assigns
  its own in-header 48-bit versions and batches many vpart versions per
  cold-tier commit, so the cold counter advances at batch rate.
- **Demotion triggers: pressure OR aging.** Pressure: dwal's adaptive
  throttle loop with a new setpoint — begin demoting at X% fixed-region
  utilization, hard-throttle writers at Y% (headroom matters: demotion is
  a durable merge with merge latency, not an eviction). Aging: nodes
  approaching the stamp horizon are demoted-or-restamped regardless of
  pressure — this is what keeps vpart's live-stamp span < 2³² (§2.1) in a
  no-pressure arena with a racing commit counter.
- **Read-path contracts**: point-lookup hit short-circuits, miss pays full
  depth (each level an in-RAM probe; only cold risks IO). Ordered ops always
  N-way merge — the lazy policy makes W bigger, which makes the **3-tier
  merge cursor load-bearing infrastructure** (it was already walart's
  queued gap; this design promotes it to a required layer).
- Range-scan-heavy roots (TATP/sqlite-shaped) may want eager demotion per
  root — keep the policy per-root tunable through the existing dwal config
  hooks.

## 6. Where today's code maps

| Component | Becomes |
|---|---|
| `libraries/part` (arena, freeset, sessions, hkv tree) | `libraries/vpart` Layer 1 host — geometry/files/freeset/msync model verbatim; refcount machinery swapped for stamps+logs+pins |
| part §19 background releaser (ring + callback + clean-shutdown drain) | the reclaimer thread: same skeleton, drains per-version logs gated on pins instead of a cascade ring |
| `refs.part` (u8/line, derived, rebuildable, never synced) | deleted — stamps live in-node inside data pages, durable with commits; no rebuild on unclean open |
| part recovery (`rebuild_begin/mark_line/rebuild_end`) | identical walk, simpler marks (claim + stamp 0; no count bumps) |
| SAL control blocks + compactor moves + MOVE-OF-DEAD diagnostics | cold tier: same move machinery; liveness question answered by stamps; freed refcount bits → MFU + move-epoch |
| dwal cow_coordinator / read modes | W-tier snapshot machinery per fork (head/prev publication, collision-only COW) |
| dwal merge_pool + adaptive throttle | the demoter + pressure controller |
| walart W/S/H composition | the end-state shape; vwalart = walart on vpart+vheart with lazy policy |
| walart 3-tier merge cursor (queued task) | required layer (L5) |
| kv_engine concept + unified `open()` | unchanged contract all of this sits beneath |

## 7. Build layers — each one shippable, tested, benchmarked before the next

Process per layer: copy + modify (never rewrite); the donor's full test suite
must pass on the copy; add the layer's new differential tests; A/B bench
against the donor and archive logs per the bench-log convention.

### L1 — `vpart`: stamps replace refcounts, linear history

**part → vpart rectification checklist** (general defects in part's impl,
identified during the stamp-placement investigation; vpart corrects them):

1. **Buckets failed — straight radix.** Tests showed the bucket design
   loses to plain adaptive radix, and this architecture leans on point
   lookups (tier hits short-circuit) while lower_bound is structurally
   expensive regardless (3-tree merge, slowest tree included, tombstones
   skipped). Optimize the op the design wins: vpart is **radix-only** — no
   `mode::buckets`/`mode::adaptive` bucket terminals, `K::bucket` and
   `K::cbucket` retired. (Side effect: the cbucket stamp-budget problem
   vanishes.)
2. **Header slack becomes common-prefix storage — and a prefix_node above
   a node whose buffer could hold the prefix is a BUG (Dan).** prefix_node
   exists ONLY for exceptionally long common prefixes, never the common
   case. Today the violation is structural: widen() LIFTS a setlist's
   inline prefix into a prefix_node because the wide tiers have no buffer
   (census: ~105 prefix_nodes per engine, parked above the hottest wide
   routers). Every node kind with spare header capacity carries an inline
   cprefix (part wastes this slack today and pays prefix_node hops for
   it); the deep-validate/census walk asserts the invariant. Presence is signaled by **one
   reserved tag code — "see header"** — not per-kind variants: tagged
   pointers exist for *dynamically typed* dispatch, so any node whose
   header must be read anyway carries tag = see-header and the header
   states the full type (a type byte beside the version in CL0 slack;
   `node_hdr`'s 16 bits are already full). Prefix-less dense routers keep
   their native tags, so skip-CL0 dispatch is untouched where it pays.
   Statically-typed references never need a real tag — the caller knows
   the type — so they default to see-header. **The version field takes
   4 B of cprefix capacity** — its real cost in routers (remaining caps:
   node_full ~82 B, c2/c4/c8 110/102/86 B, fullx ~82 B, setlist16 22 B —
   all ≥ the 63 B `prefix_len:6` can express except setlist16, which
   keeps its short-prefix role). Bucket retirement leaves the freed
   K-slots free.
3. **Leaves (suffix+value, optional prefix if space): no issue** — exact-fit
   allocations have line-rounding slack in most cases (~97% absorb the 4 B
   stamp; ~3% gain a line).
   **Setlist prefix capacity is DERIVATIVE, not a constant (Dan)**: branch
   arrays take precedence; the prefix uses whatever the current branch
   count leaves free (~34 B of slack at full fanout at the 4 B ptr, far
   more when narrow), and overflow is always legal via a prefix_node —
   which is exactly what makes "prefix_node = exceptionally long only"
   achievable. Implementation queued: tail-anchored prefix bytes +
   evict-to-prefix_node when a branch insert needs the room.
4. **The clone-log is a first-class node type** (§2.2): 128 B lines,
   push-to-front, 31 × 4 B entries + 4 B prev — allocated through the
   session so all log writes land in active segments (dirty-anyway pages,
   zero extra sync). Log lines carry no K tag (the reclaimer walks them
   with static type knowledge; tag defaults to see-header if ever stored
   in a tagged slot).
5. **Write direct; the version is the transaction** (§2.1): writers COW
   the trunk directly, one boundary published per commit; the in-place
   fast path is within-transaction only, so the last committed root is
   stable for readers with zero coordination. Undo = free the create-log
   (all transaction-born → instant) + drop the clone-log unpublished.

**Build phases** — L1a lands the liveness swap with read-path byte-identity
(the clean A/B vs part: same dispatch, same scans, stamps in slack only;
bucket modes simply not instantiated on either side so the comparison is
apples-to-apples). L1b lands checklist item 2 (universal cprefix + tag
variants — a read-path change, so it is A/B'd separately to keep
attribution honest).

- **Copy** `libraries/part` → `libraries/vpart` (namespace `vpart`).
  `hkv/` tree sources copied; the L1a contract is **read-path
  byte-identity** (§2.1): stamps land in slack/pad/tail only, so descent
  dispatch, SIMD offsets, and skip-CL0 are unchanged — node structs gain
  stamp fields and the ~35 COW-hook sites edit mechanically, but no scan
  or dispatch code moves. (Full tree-verbatim died with the structural
  encoding: stamp location is kind-specific, so the tree must locate it.)
- **The seam**: a tree-side `stamp_of(p)` accessor (kind-dispatched address
  of the stamp — the call sites hold the tagged ptr, so the kind is free)
  plus allocator-side window machinery:
  - shared-test: `alloc_.is_old(stamp_of(p))` — one compare against the
    current window, reading a cacheline the writer is about to mutate
    anyway (the donor pays a refs.part look-aside at the same sites).
  - `hkv_retain(line)` → `return true` (sharing is free). The
    `writer_clone_at=190` forced-clone-forward behavior disappears — its
    emergent stale-data-moves-forward locality is **a measured risk** (§8).
  - `hkv_release(line, destroy)` → stamp==window: free inline + run destroy
    (today's unshared fast path, preserved). Else: append tagged raw to the
    window's clone-log; destroy is NOT run (lazy expansion at reclaim).
  - `hkv_defer(raw)` → subsumed: log append *is* the deferral.
  - `hkv_free_run` → same split (inline if window-born, else log).
- **arena.hpp**: `refs.part` is **deleted with no replacement file** —
  stamps ride inside data pages, durable with the data they describe (no
  stamp rebuild on unclean open; only the freeset walk remains).
  retain/release/pin
  family → stamp read/write + log append; the 64-slot session bitmap stays
  as the **pin table** (pin = version, not per-line band); §19 releaser ring
  → reclaimer draining logs gated on `min/interval` over pins.
- **kv.hpp**: direct-write transactions (§2.1) — commit = publish boundary
  + version increment; abort = create-log free + clone-log drop (kv-compat
  4.7b's "diverged generation released" becomes exactly this). Snapshot
  cursor = share the latest published boundary slot (O(1) acquire/release,
  §2.3); readers never mint version roots and never wait on writers.
- **Test gates**: all part suites green on vpart (arena/cow/tree/kv/freeset);
  census tool extended: after fuzz with snapshots held/released in *random
  order*, assert live-lines == reachable-lines (no leak, no early free);
  crash-fuzz reuses part's harness (recovery is the same walk).
- **Scope**: L1 vpart is the **standalone engine** (§1 modes) — delete is a
  real delete, no tombstone machinery anywhere. It must be complete and
  bank-testable with no cold tier behind it. Phase 2 (the demoter pushing
  versions to heart in order, tombstones in diff mode, fork-parallel heart
  roots) builds *on top of* a working standalone vpart, never into it.
- **Bench gate**: part vs vpart on names + bank (writes, point reads, scan,
  concurrent-reader config); release-side cost and reclaim pacing
  (high-water vs live) explicitly reported. This A/B is the whole thesis
  measured: if vpart doesn't win (or at minimum tie with massively simpler
  invariants), stop and re-think before L2.
- **Expected profile** (the predictions the gate confirms or refutes):
  - reads ≈ hkv — descent unchanged (stamps never read on the read path),
    possibly faster once L1b cprefix removes prefix_node hops;
  - writes ≈ art_map or faster — same COW shape, but alloc goes bump →
    O(1) freeset bitmap and free goes noop → O(1) bitmap update, traded
    against zero share/unshare traffic and durable placement;
  - version expiration = a background thread calling release (the
    reclaimer draining chains to the oldest running reader);
  - point queries of recently-written data stay fast for the whole
    window before space/seq pressure forces demotion (Phase 2: and after
    demotion, heart-tier queries are as fast as today — maybe faster
    with cprefix and no refcounts);
  - Phase 2 expectations recorded now: heart's insert rate jumps from
    small-batch random to **large-batch sequential** (the demoter pushes
    in version order), many writes are **deduped in vpart** and never
    reach heart's slower write path at all, and heart's own writes get
    faster with refcounts gone.

### L2 — vpart forks: (fork, seq), branch-point pins, wholesale abandon
- Stamp gains the fork field; per-fork logs; fork-from-any-snapshot =
  pin + fork id + delta tree; ownership rule active; fork-id recycling
  guarded (no reuse until stamps swept).
- Tests: fork fuzz vs N shadow std::maps (one per fork, random fork/abandon/
  commit schedules); abandon-leak census; trunk-flip symmetry (winner
  becomes trunk, loser's logs drop).
- Bench: blockchain-shaped churn (forks of a few-thousand blocks, hours-scale
  pins simulated), measure abandon cost vs part's cascade.

### L3 — vheart: SAL version-mode (cold tier)
- SAL control blocks answer liveness by stamp interval instead of refcount;
  compactor/mover keeps its move machinery, gains reader-epoch grace;
  freed bits → MFU + move-epoch. heart runs on top unchanged at the API.
- Gates: sal-tests + heart-tests + kv-compat + deep-invariants toolkit;
  the COW-latch bug's repro becomes a regression test that passes by
  construction; bank with concurrent readers (the 949k/3.9M config).

### L4 — pin table + migration (W → H)
- Pin table, materialize-from-log-diff-stream, CAS repoint, epoch drain,
  W retire. Gates: the migration differential test (§4) under fuzz; pins
  surviving migration mid-iteration (cursor refresh path).

### L5 — vwalart: lazy tiered composition
- walart shape on vpart (W) + vheart (H); demoter with settled-first +
  pressure setpoints; **3-tier merge cursor** (closes walart's kv_engine
  cursor gap → the engine-recursion static_assert is the finish line).
- Gates: kv-compat over the composition; bank + TATP A/B vs dwal and
  walart-today; write-amp accounting (bytes to cold tier / bytes written).

### L6 — tagged dual allocation (deferred until L1–L5 prove the substrate)
- The per-branch direct/indirect hybrid + COW-time placement. Requires the
  branch-width capacity decision (§8) locked first.

## 8. Open decisions (lock at sign-off or first use)

1. ~~cbucket stamp budget~~ **RESOLVED by rectification checklist item 1**:
   buckets retired in vpart (measured loss vs straight radix; the
   architecture optimizes point lookups). No cbucket, no budget problem.
2. ~~Log entry width~~ **RESOLVED: 4 B entries, 31 + prev per 128 B line**
   (§2.2). Birth reads the dying node's in-node stamp at reclaim; kind tag
   derives multi-line run lengths, so one entry per node regardless of
   size. The possibly-cold page touch per entry at reclaim is accepted
   (the expand hook reads dying content anyway) and batched by the
   reclaimer.
3. **Stamp wraparound**: handled by the tier flow, not stamp width (§2.1).
   Safety invariant = live-stamp span < 2³² (modular equality then exact);
   enforced by demotion draining aged data to heart (which resequences and
   batches) plus **aging as a demotion trigger** (§5) for the no-pressure
   gap. Stamp 0 reserved as "ancient"/always-shared; counter skips 0 on
   wrap; horizon proximity alarmed in stats.
4. **Branch width for the relocatable space** (gates L6, not L1): 4 B tagged
   (½ TB cap) vs 5 B (1 TB+, +11–15% reads per the part A/B).
5. **Loss of clone-forward (`writer_clone_at`)**: emergent locality where
   popular old lines re-materialize into fresh segments. If L1's bench shows
   read-locality regression on aged data, re-add deliberately as an
   MFU-driven promote-on-COW (the §3 mechanism) rather than a count
   threshold.
6. **Fork identity in the stamp** (L2): fork bits would shrink the modular
   live-span margin (16-bit fork + 16-bit seq is hopeless; even 8+24 is
   tight under bank rates). Favored: **one global txn-id counter across
   all forks** (the 32-bit stamp is pure id; privacy and intervals stay
   fork-oblivious) + a tiny id→lineage map in the version-tree metadata
   for visibility checks — forks just make the id sequence non-contiguous
   per lineage, which the map already handles. Fork-id recycling
   discipline rides this decision.
7. **Tombstone layer trigger** (§2.5): the pinned-base amplification number
   from L2's blockchain bench that justifies building it.
8. **Pin slot capacity**: 64 (one u64 bitmap, part idiom) — enough for
   snapshots + forks? Blockchain says forks ≤ dozens live; long-lived
   snapshots are few. Revisit if L2 fuzz wants more.

## 9. What is explicitly out of scope here

- No changes to `libraries/part` itself (donor stays pristine as the A/B
  baseline) — same for SAL until L3 sign-off (per the no-SAL-changes rule,
  L3 is the explicit permission grant).
- No engine API changes: everything lands beneath the kv_engine concept and
  the unified `open(path, open_mode, config)` shape.
- art_map/dwal stay as-is; dwal's RW layer adopting any of this is a
  separate decision after L1's A/B (the hkv-delta swap verdict gets
  re-evaluated on vpart numbers).

# part — Persistent ART, Layer 1: Memory Layout & Allocator

Status: DRAFT for sign-off. Supersedes the allocation sections of `design5.md`;
node formats in design5 remain the (non-binding) sketch for the tree layer.

## 1. Goal

A file-backed, non-relocating, cacheline-granular allocator whose placement
policy minimizes msync write amplification. It is the foundation the tree
layer (tagged-pointer ART) builds on. This layer knows nothing about node
formats — it deals in *lines* (128B units), refcounts, transactions, and
segments.

Non-goals for this layer: node types, tree operations, MVCC, compaction
(memory never moves — that is the premise), background threads.

## 2. Principles (from design5 + review)

1. Memory never relocates. Reads dereference `base + index*128` directly; no
   control-block indirection. Type dispatch lives in the pointer tag (tree
   layer).
2. Every disk page is in one of two states: **synced & logically read-only**,
   or **dirty & will be msync'd at next commit**. All placement policy serves
   this model: never dirty a read-only page for allocation when a
   dirty-anyway page can serve.
3. Refcounts and the freeset are **derived state** — never part of consensus
   state, never required to be crash-consistent, rebuildable by a
   reachability scan. They live in separate files that are not synced on
   commit.
4. Minimize syscalls: one msync per contiguous dirty run at commit.
5. Allocation = refcount-slot allocation. Same index addresses both, and
   the unit of both is the **line, not the node**: every cacheline of a
   multi-line node is independently refcounted and independently freeable
   (§9). This is the headline difference from sal/heart — it is what makes
   COW of one branch of a full node a 128B copy instead of a 1KB+ copy.
6. No background threads.
7. **The read path keeps hkv's headerless dispatch** — the property that
   beat libart. Node kind rides in the tagged pointer, and *implicit* node
   forms locate their branch lines by address arithmetic (node index +
   fixed offset): descent never loads a header to decide how to dispatch,
   and never chases a pointer to find an implicit node's own lines. A COW
   deliberately trades this away **locally and temporarily**: it demotes
   implicit → explicit (one 128 B line copy + ≤ 8 contiguous line retains,
   zero per-child refcount traffic) instead of copying the whole group —
   reduced copies and fewer refcounts in exchange for one extra dependent
   load on descent through that node. Clone-forward rewrites (§9)
   re-materialize the implicit form in fresh segments. Fast reads are the
   steady state; the explicit form is the transient.

## 3. Geometry — one constant, everything derives

All compile-time, in `part/config.hpp`, overridable via `-D` (same pattern as
`HKV_POOL_BLOCK_BYTES`):

| Constant | Default | Constraint |
|---|---|---|
| `PART_SEGMENT_BYTES` | 8 MB | power of 2, 4–16 MB intended range |
| `PART_LINE_BYTES` | 128 | fixed in practice (M-series CL, x86 prefetch pair) |
| `PART_DISK_PAGE_BYTES` | 16 KB | power of 2, ≥ OS page size |
| `PART_MAX_READ_SESSIONS` | 64 | ≤ 64 (one u64 slot bitmap); derives `writer_clone_at = 254 − it = 190` (255 = throw sentinel, §9) |

Derived (shown for the defaults):

| Quantity | Formula | @ defaults |
|---|---|---|
| lines / segment | SEG / LINE | 65,536 |
| disk pages / segment | SEG / PAGE | 512 |
| lines / disk page | PAGE / LINE | 128 |
| L0 freeset / segment | lines / 8 | 8 KB (1024 u64) |
| hint levels L2/L4/L8 | halved each | 4 KB / 2 KB / 1 KB |
| word indexes (meta_k) | 1 bit per u64 word of each array | 128 / 64 / 32 / 16 B |
| top summaries (top_k) | 1 u64 over each meta_k | 4 × 8 B |
| refcounts / segment | lines × 1B | 64 KB (0.78% of data) |
| dirty mask / segment | pages / 64 | 8 u64 |

`static_assert` the power-of-2 and divisibility relations. A u64 word of L0
covers 64 lines = 8 KB of data.

## 4. Addressing

- **`line_idx` = plain `uint32_t`**, global line index. Address =
  `mapped_base + line_idx * 128`. Segment = `line_idx / lines_per_segment`
  (shift). Max addressable: 2^32 × 128B = 512 GB.
- **The top 2^16 values are reserved sentinels — never valid indexes.**
  Values in `0xFFFF0000..0xFFFFFFFF` encode non-index meanings, so every
  ptr-shaped field is *either* a global index *or* a local reference,
  never ambiguously both, distinguished by one range check
  (`is_sentinel(p) = p >= 0xFFFF0000`). Default band layout (tree layer
  owns the semantics): `0xFFFFFFFF` = null; the remaining values carry a
  16-bit local payload in the low half (`k = p & 0xFFFF`, k ≠ 0xFFFF) —
  **local tail pointers** for node building (e.g. bucket2k's internal
  16-bit tail-ptr allocations, references to a spot within the same node
  group). Cost: 8 MB of the 512 GB space (0.0015%). Layer-1 obligations
  only: never mint an index in the band, and validate at open that
  `max_segments × lines_per_segment ≤ 0xFFFF0000`. Interpretation of the
  payloads is entirely tree-layer.
- The tree's 28+4 tagged `ptr<T>` (32 GB cap) is a *tree-layer encoding
  decision*, deferred. This layer is unaffected by however the tree packs
  it (the same top-of-range reservation trick scales to whatever index
  width the tree chooses).
- An allocation is `1..64` consecutive lines that **do not cross a u64 word
  boundary of L0** (an 8 KB-aligned region). This makes every alloc/free a
  single-word CAS. Practical max node today is 16 lines (bucket2k).
- Fast-path size classes `{1, 2, 4, 8, 16}` lines, naturally aligned
  (16 = an aligned pair of 8s). Arbitrary run sizes are supported via
  class-mask composition (§6) at the same O(1) cost (e.g. 3 = 2+1, 5 =
  4+1, or the 9-line run of a full implicit node: 1 header + 8 child-ptr
  lines, contiguity being what makes its branch addresses computable —
  §2 principle 7); the tree layer may still prefer to round to a class.
  Both exposed; measurement decides what the tree uses.

## 5. Files (one directory)

| File | Contents | Synced? | On reopen |
|---|---|---|---|
| `data.part` | segments (block_allocator, block = SEGMENT) | msync at commit (dirty runs only) + fsync | authoritative |
| `refs.part` | u8 refcount per line | never | trusted iff clean-flag, else rebuilt |
| `free.part` | L0 + L2 + L4 + L8 freesets per segment | never | L0 trusted iff clean-flag, else rebuilt; **hints always rebuilt** |
| `meta.part` | magic, format version, geometry echo, clean flag, root slots (`max_roots` × u64, u32-indexed like psitri's root_index), segment directory | fsync at commit | validated (geometry mismatch = hard error) |

Segment directory (in `meta.part`): per segment — four atomic counters
`free_count[k]`, k ∈ {1,2,4,8} = number of set bits in `free_lines_k`
(free aligned k-groups; inclusive — a free 8-group also counts in 4/2/1),
`atomic<u8> state` (FREE / QUALIFIED / OWNED / RETIRED-ish), owner session
id; plus the **qualifying bitset** (1 bit per segment) used for run
selection. The qualifying bit is driven by `free_count[8]` (§8). All of
this is derived state too (rebuilt on unclean open).

All four are `block_allocator` instances (copied from sal) sharing the
contiguous-VA-reservation property, so `get(line_idx)` is one shift+add.
`refs.part` / `free.part` / segment directory get `mlock` best-effort
(pinned, per design5 §4).

## 6. Freeset: one authority + XOR-settling hints + word indexes

Per segment, per size class k ∈ {1, 2, 4, 8}, three tiers:

- **`free_lines_k` — the direct bitmaps.** k=1: 1 bit per line, `1 = free` —
  **the single authority**; every alloc/free is a CAS on exactly one of its
  words. k=2/4/8: 1 bit per naturally aligned k-line group, set *intending*
  "group entirely free" — navigational hints.
- **`meta_k` — word index.** 1 bit per u64 word of `free_lines_k`, set ⇒
  that word has ≥ 1 set bit and is worth scanning. This is what makes alloc
  O(1) instead of O(N): a nearly-full segment would otherwise degrade to a
  linear sweep of the direct bitmap (1024 words @ 8 MB). We pay a small
  index-maintenance price on every free to buy that.
- **`top_k` — one u64 over the words of `meta_k`** (only meaningful when
  `meta_k` exceeds one word; @ 8 MB, meta_1 is 16 words → 16 bits of top_1).

Generically: each `free_lines_k` array is indexed by a fanout-64
hierarchical bitmap whose depth derives from the word count — ≤ 64 words
needs one index level, > 64 needs two, **> 64×64 (4096) words needs a
third**, and so on (⌈log64⌉). The concept is already demonstrated in
`ucc::hierarchical_bitmap` (compile-time depth via `calc_level`,
specialized descents per depth); part copies it as the basis and adds what
our use needs: an **atomic level-0** (the CAS authority for k=1), upper
levels as XOR-settle hints rather than exactly-maintained parents, and
**find-first-set-at-or-after(pos)** via the masked descent — ucc's only
searches from 0. At the default 4–16 MB geometry two index levels always
suffice; the third appears automatically past 32 MB segments.

**Maintenance (XOR-settle, all tiers above the authority):** after a
successful authoritative transition, the mutating thread re-reads the
affected word, computes the desired bit one level up, and XOR-toggles on
mismatch — order-invariance lets concurrent toggles converge with 1–2
retries; any later observer of a stale bit fixes it the same way.
Propagation only fires on boundary transitions (word 0 ↔ nonzero, group
all-free ↔ not), so the steady-state cost per free is O(1) amortized.

**Alloc of class k = masked find-first-set descent.** All tiers are ordered
identically to memory, so first-fit-forward-from-floor (§7) is: mask off
bits below the floor position, `countr_zero` down top_k → meta_k →
free_lines_k, then the authoritative CAS of the k-bit mask in free_lines_1.
On CAS failure or a discovered stale hint: fix the lying bit, continue
forward from there.

**Odd sizes = class-mask composition, not linear scan.** An arbitrary run
decomposes largest-component-first (3 = 2+1, 5 = 4+1, 6 = 4+2, 9 = 8+1, …).
The descent navigates the **largest** component's index tiers to a
candidate word, then finds aligned-neighbor placements by combining the
class bitmaps — e.g. a 9-run starting at 8-group g requires
`L8 bit g AND L1 bit 8(g+1)`: expand/shift one word, AND against the
other, `countr_zero` the result. All word-parallel ops, so odd-size
allocation has the same O(1) shape as the class path — no n-consecutive-bit
scanning. The composed mask is still cleared by one CAS on the single
authoritative free_lines_1 word (word confinement guarantees a composition
never straddles words). Compositions inherit the anti-split preference
below where the word offers alternatives.

**Anti-split placement preference.** A class-k alloc prefers a free k-group
whose parent 2k-group is *not* entirely free — i.e. fill lone holes before
carving up larger contiguous runs, so the `free_count[8]` that qualified the
segment isn't consumed by small allocs that had alternatives. The check is
one extra mask op during the same descent (expand the level-2k bits covering
the current word and prefer `free_k & ~expanded`; for k=8 the "parent" is
the implicit aligned 16-pair — check the buddy bit). The preference is
**bounded to the current word**: never skip forward just to avoid a split —
floor locality wins; fall back to splitting when the word offers nothing
else.

**Per-class counters.** Each segment maintains `free_count[k]`, k ∈
{1,2,4,8} (relaxed atomics, §5). The ±1 updates ride the same boundary
transitions the XOR-settle propagation already detects (group
all-free ↔ not), so no additional transition detection is needed.
`free_count[8]` crossing the reuse threshold flips the segment's qualifying
bit (§8).

Hints, indexes, and counters are never persisted as truth: rebuilt from
free_lines_1 on every open (linear, single-threaded, trivial).

## 7. Sessions, transactions, fill policy

A **session** is an explicitly-created per-writer-thread handle (sal-style,
no hidden thread_locals). Sessions own the write path; reads need no session.

Each session owns an **active range**: one or more *contiguous* qualifying
segments it has claimed. Within the active range:

- `txn_floor` — the byte offset of the start of the **disk page on which the
  current transaction began writing**.
- **Allocation = first-fit forward from `txn_floor`**, guided by the hint
  levels. Holes behind the floor are invisible until the segment recycles
  through the pool. Holes at/after the floor are fair game: pages there are
  either dirty already or about to be — filling them costs no extra sync.
- On **commit** (`transaction::commit()`):
  1. msync the dirty disk-page runs of the active range (and any ranges in
     segments abandoned mid-txn), coalescing runs separated by < G clean
     pages into one syscall (G tunable, default 1);
  2. fsync `data.part` (`F_FULLFSYNC` on macOS behind the existing
     `sync_type` knob);
  3. advance `txn_floor` to the page containing the current fill position
     (mid-page commit ⇒ that page will be re-dirtied and re-synced next
     commit — inherent, accepted).

The canonical tree-layer commit sequence is then:

```
txn.commit();                   // data durable
arena.set_root(slot, new_root); // meta write + fsync
release(old_root_ref);          // cascade frees old-version lines
```

### Frees are immediate (crash safety is the COW contract's job)

`release` applies **immediately**: per-line refcount decrement; on 1→0 the
destroy hook runs, then the L0 bit is set and the line is reusable. No
pending queue. This is safe because of the COW discipline the tree layer
already follows:

1. A transaction begins by retaining the durable root (ref ≥ 2). The COW
   rule — never modify a line reachable from a ref≥2 node — makes all
   committed data read-only for the life of the txn; modifications
   path-copy toward the root, retaining siblings on the way. Therefore the
   only lines that can reach ref 0 *mid-txn* are lines allocated by this
   txn — never durable, so reusing them is crash-irrelevant.
2. Old-version lines reach ref 0 only via the old root's release cascade,
   which the tree triggers strictly **after** `set_root` made the new root
   durable. By then the durable root no longer references them.
3. Torn-page analysis: msync of a partially-rewritten page only endangers
   bytes that changed, and by (1)+(2) every byte changed since the last
   commit lives in a line unreachable from the durable root.

The allocator does not enforce this ordering — it documents it, and the
commit API above makes the right order the natural one.

Write-amp note: mid-txn frees are typically lines allocated this txn, hence
at/after the floor — the first-fit scan reuses them at zero additional sync
cost (the page is dirty anyway). This is the COW-churn fast path and it
falls out of the policy with no extra machinery.

## 8. Segment lifecycle & leveling

- **Qualifying bit** = segment is unowned AND `free_count[8] ≥
  reuse_threshold × (lines_per_segment / 8)` — the threshold is measured in
  **free 8-line contiguous groups**, not raw free lines (runtime-config,
  default 50% ⇒ ≥ 4 MB of 8-contig space @ 8 MB segments). Rationale: a
  segment can be 50% free on every other line — claiming it is risky
  because any multi-line alloc fails immediately. Counting only 8-contig
  space guarantees a claimed range can absorb allocations of every class.
  Fragmented segments are not stranded: release cascades free old paths in
  contiguous runs, so holes coalesce back into 8-groups over time, the
  count rises, and the segment re-qualifies — at which point the sweep's
  smaller sizes pick up the odd holes (with the anti-split preference of
  §6 keeping the 8-groups intact for the larger classes).
- 16-line allocations (an adjacent pair of free 8s) are not guaranteed by
  the qualifying test; a 16-alloc the range cannot place triggers the
  normal switch below. Accepted: bucket2k is the rare pathological case.
- When a session exhausts its active range (first-fit can't place the request
  before range end):
  1. Prefer **continuity**: if the next adjacent segment is qualified and
     unowned, claim it and keep filling (lazy claim, one segment at a time —
     no hogging of long runs).
  2. Else scan the qualifying bitset for the **longest contiguous run** of
     qualified segments; tie-break nearest-forward from current position
     (clock sweep). Claim its first segment, set fill position to its bottom.
  3. Else (no qualified segment): **grow** `data.part` by
     `grow_batch_segments` (default 4) fresh segments; claim the first.
- Claiming/owner transitions happen under a small mutex — this is a
  once-per-~8MB-written slow path; no lock-free heroics. The qualifying
  bitset is maintained lock-free (set/cleared on threshold crossings) and is
  itself only a hint; the per-segment `state` byte is the authority during a
  claim.
- A request the active range cannot satisfy (no N-line run ahead of floor)
  triggers the same switch; the abandoned segment keeps its true `free_lines`
  and re-qualifies naturally for smaller allocations later.
- Fresh segments from growth are trivially qualified (100% free), so the file
  tail is the natural "longest run" when reuse is poor.

## 9. Refcounts — strictly per line

- `atomic<uint8_t>` per line, in `refs.part`, indexed by the same
  `line_idx`. **Every line is an independently shareable, independently
  refcounted resource.** There is no head-line convention: an allocation of
  n lines initializes **all n counters to 1**, and each line is retained,
  released, and freed on its own.
- Why: sub-node COW granularity. A full node (implicit or explicit) has each
  of its cachelines independently refcounted. To COW one branch of an
  implicit full node, you do **not** copy the whole group: allocate one new
  line (copy+modify 128B), retain the other 7 branch lines individually,
  and write a new *explicit* node whose pointers reference 7 shared old
  lines + 1 new line. The children themselves are untouched: a full node's
  255 unmodified child refcounts see **zero** traffic, because child
  references live inside the shared lines and follow the line's count —
  this is what replaces heart's per-child refcount machinery. Releasing an
  implicit full node releases its header line and each of its implicit
  branch lines — even though they were allocated as one run; lines still
  shared by explicit nodes survive, and only when a line itself hits 0
  does its destroy hook release the ≤ 32 children packed inside it.
- The implicit ↔ explicit cycle (§2 principle 7): implicit = fast
  headerless dispatch, explicit = cheap COW; clone-forward rewrites
  re-promote to implicit. Layer 1 supports the cycle with contiguous
  arbitrary-run allocation (a full implicit node is a 1+8-line run, §4),
  `retain_range` with per-line refusal reporting, and per-line frees — no
  allocator change is ever needed to switch a node between forms.
- Locality: implicit groups are allocated contiguously, so their refcount
  atomics are contiguous too. The word-confinement rule (§4) caps any
  allocation at 64 lines ⇒ its refcounts span ≤ 64 B; an 8-line implicit
  group's counters are 8 contiguous bytes. Halving the counter width also
  halves `refs.part` (0.78% of data) and doubles how much of it stays hot
  in cache.
- **Overflow is unreachable by design, and detected by sentinel when a bug
  reaches for it — nothing is ever pinned.** Valid counts are 0..254;
  **255 is reserved as the overflow sentinel and is never a stored count**:
  any increment that would reach 255 throws `part::ref_overflow`. The
  designed failure mode for programmer error — e.g. a thread creating or
  storing many smart_ptr copies instead of moving — is an immediate,
  attributable exception at the offending copy site, not a silent pin or
  a corrupted count. This throw is the only exception in the refcount
  path, and it uniformly covers every increment source (state retains,
  pins, handle copies). Within the valid range, two bands:
  `writer_clone_at = 254 − max_read_sessions = 190`.
  - **Writers clone at 190.** `retain(idx) → bool` (the state-reference
    entry point, writer-only) succeeds iff the observed count < 190, so
    state references never exceed 190. On false the writer — which always
    has a transaction in scope — clones the line (a 128 B copy into its
    own active range) and references the clone (ref 1). Structural sharing
    in a COW tree is an optimization, never a semantic requirement, so
    clone-instead-of-retain is always correct. Emergent property:
    genuinely popular old lines re-materialize *forward* into fresh
    segments while the hot original's refs drain away with dying versions
    — stale data keeps moving toward the writers. (Reader pins inflate
    the observed count, so writers may clone slightly early — harmless,
    and it drains the hot line sooner.)
  - **Readers are capped at 64 concurrent** (`max_read_sessions = 64`, one
    u64 slot bitmap in the arena — same idiom as sal's session table).
    `start_read_session()` acquires a slot; pins (`pin(idx)`/`unpin(idx)`
    through the session) use the reserved 191..254 band. Discipline: ≤ 1
    pin per line per session — the tree layer aggregates same-version
    snapshot handles behind one pin, so 64 sessions ⇒ ≤ 64 pins per line ⇒
    count ≤ 190 + 64 = 254, and the sentinel throw never fires on
    disciplined use.
- `release(idx, destroy)` = exact `fetch_sub(acq_rel)`, always — no sticky
  no-op cases. On 1→0 the destroy hook runs while the line content is
  still valid (the tree releases whatever the line references — for a
  branch line, its child pointers), then the L0 free bit is set
  immediately (§7). Range conveniences (`retain_range`, `release_range`)
  loop; `retain_range` reports the first refused line so the caller can
  clone exactly that one and continue.
- Never synced, never trusted after unclean shutdown; rebuild recounts
  actual parents.

## 10. Recovery

- Clean shutdown writes L0 + refcounts + directory, sets clean flag, fsyncs.
  Clean open trusts them, rebuilds hints + qualifying bits (linear scan).
- Unclean open: this layer provides the primitives; the *tree* layer drives
  the walk (it knows node formats):
  - `rebuild_begin()` — zero refcounts, mark all lines free, clear directory;
  - `mark_line(idx)` — called per reachable *line* from the root scan
    (per-line, matching §9): first mark claims the line (clears free bit,
    ref=1), later marks from other parents just bump the refcount;
  - `rebuild_end()` — recompute `free_lines`, hints, qualifying bits.
- Phase-1 test drives this with a synthetic object graph (no tree needed).

## 11. Stats (always on, relaxed counters)

Per-arena: `bytes_allocated`, `bytes_freed`, `lines_in_use`, `segments_grown`,
`segments_reused`, `range_switches`. Per-commit and cumulative:
`pages_synced`, `sync_syscalls`, `bytes_synced`. **Write amp =
bytes_synced / bytes_allocated** — the number this whole design exists to
minimize; the perf test prints it per workload.

## 12. Public API

### 12.1 Core sketch

```cpp
namespace part {

struct config {
   std::filesystem::path dir;
   uint64_t  max_segments     = 4096;   // × 8MB = 32GB default ceiling;
                                        // validated: total lines ≤ 0xFFFF0000
                                        // (top 2^16 indexes are sentinels, §4)
   uint32_t  max_roots        = 16;     // u64 root slots in meta.part
   double    reuse_threshold  = 0.50;   // fraction of capacity, measured in
                                        // free 8-line contig groups (§8)
   uint32_t  grow_batch       = 4;      // segments per file growth
   uint32_t  msync_gap_pages  = 1;      // coalesce dirty runs across ≤ G clean pages
   sync_type sync_mode        = sync_type::msync;  // same enum family as
                                        // sal/psitri runtime_config; ::none
                                        // = no msync machinery (in-memory)
};

class arena {
  public:
   explicit arena(config);
   ~arena();                            // clean shutdown path

   session  start_session();            // one per writer thread

   void*       get(line_idx i) noexcept;        // base + i*128
   const void* get(line_idx i) const noexcept;

   // per-line refcounts (§9); frees are immediate (§7); any thread
   [[nodiscard]] bool retain(line_idx i) noexcept;   // state refs (writers):
                                        // false at count ≥ 190 ⇒ clone the
                                        // line instead (§9)
   uint32_t retain_range(line_idx i, uint32_t n) noexcept;  // returns count
                                        // retained; < n ⇒ line i+count refused
                                        // (clone that one, continue)

   // reader pins: ≤ 64 concurrent read sessions (u64 slot bitmap), pins
   // use the reserved 191..254 band, ≤ 1 pin per line per session (§9)
   read_session start_read_session();
   template <typename Destroy>                  // destroy(line_idx) runs on 1→0,
   bool release(line_idx i, Destroy&&) noexcept;  // then the line is freed
   template <typename Destroy>
   void release_range(line_idx i, uint32_t n, Destroy&&) noexcept;

   // durable root slots, indexed like psitri's root_index (tree layer owns semantics)
   void     set_root(uint32_t slot, uint64_t v);  // meta write + fsync
   uint64_t get_root(uint32_t slot) const noexcept;

   // recovery
   void rebuild_begin();
   void mark_line(line_idx i);          // claim on first mark, add_ref after
   void rebuild_end();

   const arena_stats& stats() const noexcept;
};

class session {                          // movable, not copyable
  public:
   transaction start_transaction(uint32_t root_slot);  // one open txn at a time
};

class transaction {                      // movable; first-class, harness-shaped
  public:
   line_idx alloc_lines(uint32_t nlines);  // 1..64 word-confined; all refs = 1
   void     commit();   // msync dirty runs + fsync, advance floor (§7);
                        // the tree layer's commit composes:
                        //   txn.commit() → arena.set_root(slot, new) → release(old)
   void     abort() noexcept;  // layer 1: nothing to undo — the tree layer
                        // releases its working root and the cascade frees
                        // every line this txn allocated (immediate frees
                        // make abort free)
};

class read_session {                     // ≤ 64 live arena-wide; dtor frees the slot
  public:
   bool pin(line_idx i);                 // +1 in the reserved 191..254 band;
                                         // ≤ 1 pin per line per session (§9);
                                         // throws ref_overflow at the sentinel
   void unpin(line_idx i) noexcept;      // exact decrement
};

}  // namespace part
```

(Exact signatures will evolve in TDD; the shape — dumb line-granular core,
typed sugar on top, explicit sessions, transactions as objects — is the
spec.)

### 12.2 smart_ptr: the ownership discipline

**Raw indexes exist in exactly two places:** (a) inside node state in the
mapped file, and (b) transiently borrowed under a live pin (below). Every
other reference — stack, application heap, containers — is a `smart_ptr`
with automatic retain/release and a pointer to the allocator to make it
happen:

```cpp
template <typename T>      // T supplies destroy(arena&, line_idx) for the 1→0 hook
class smart_ptr {
   arena* _a;              // who retains/releases
   ptr<T> _p;              // raw index (+ tree-layer tag bits, opaque here)
  public:
   // copy = retain; move = steal; dtor = release (running T::destroy on 1→0).
   // Copy may THROW part::ref_overflow: the increment that would reach the
   // 255 sentinel (§9) — i.e. a thread piling up copies instead of moving —
   // fails at the offending copy site. Move is noexcept and free: prefer it.
   void   give(ptr<T>);    // adopt ownership WITHOUT retaining (+1 came from elsewhere)
   ptr<T> take();          // surrender ownership WITHOUT releasing
};
template <typename T, typename... Args>
smart_ptr<T> alloc(transaction&, uint32_t nlines, Args&&...);  // construct at the lines
```

Crossing the state boundary never double-counts: loading a child from a
node into an owning local = copy (retain); installing a freshly allocated
or cloned child into a node = `take()` (the +1 from alloc transfers into
state); evicting a child from a node = `give()` (state's ref rides out in
the handle and dies with it).

**Borrow-under-pin:** a cursor or algorithm holding a `smart_ptr` to a
(snapshot) root may keep raw indexes for its descent path. §7's COW
contract guarantees the pinned root transitively keeps every reachable
line alive and immutable, so the borrow is safe — and read paths stay free
of refcount traffic. This is the deliberate exception to "everything on
the stack is a smart_ptr": the pin *is* the smart_ptr; the path entries
are borrows under it.

### 12.3 Harness-shape compatibility (psitri / heart)

The stack part builds toward must drop into the templated test/bench
harness in `~/psitri-merge` — one templated body, parameterized on each
engine's NATIVE database type. Reference points:

- `programs/kv_compat_test.cpp` — the conformance driver
  (`run<heart::kv::database>` / `run<psitri::database>` → eventually
  `run<part::database>`).
- `libraries/heart/test/heart_kv.hpp` — the canonical minimal mirror of
  the psitri shape; part's layer-2 surface copies this shape, not its
  implementation.

Required surface (different backend, entirely different types — same SHAPE):

```cpp
part::database::create(dir, runtime_config = {}) → std::shared_ptr<database>;
db->start_write_session()                        → std::shared_ptr<write_session>;
ws->start_transaction(uint32_t root_index)       → transaction;
    tx.insert / tx.upsert(key_view, value_view);
    tx.remove(key_view) → int;  tx.commit();  tx.abort();
    tx.get(key_view, fn(value_view)) → bool;
ws->get_root(uint32_t root_index)                → tree;
tree.snapshot_cursor()                           → cursor;
cursor: seek_begin / lower_bound / upper_bound / find / next / is_end /
        key() / value<T>() / get_value;
runtime_config: sync_mode (sal::sync_type-compatible),
                write_protect_on_commit, checksum_on_commit;
```

**The shape is enforced by a C++ concept, not by convention.** A
standalone, dependency-free header `part/include/part/db_shape.hpp`
defines the contract; `psitri::database` and `heart::kv::database` must
pass it (which validates the concept against ground truth), and part's
classes `static_assert` it at their definition sites so a shape drift is a
compile error, not a harness surprise:

```cpp
namespace part::shape {

template <class C>
concept kv_cursor = requires(C c, std::string_view k) {
   c.seek_begin();  c.lower_bound(k);  c.upper_bound(k);  c.find(k);
   c.next();
   { c.is_end() } -> std::convertible_to<bool>;
   { c.key() }    -> std::convertible_to<std::string_view>;
   c.template value<std::string>();          // optional-like result
};

template <class T>
concept kv_tree = requires(const T t) {
   { t.snapshot_cursor() } -> kv_cursor;
   { static_cast<bool>(t) };
};

template <class TX>
concept kv_transaction = requires(TX tx, std::string_view k, std::string_view v) {
   tx.insert(k, v);  tx.upsert(k, v);
   { tx.remove(k) } -> std::convertible_to<int>;
   tx.commit();  tx.abort();
   { tx.get(k, [](std::string_view) {}) } -> std::convertible_to<bool>;
};

template <class WS>
concept kv_write_session = requires(WS ws, uint32_t root) {
   { ws.start_transaction(root) } -> kv_transaction;
   { ws.get_root(root) }          -> kv_tree;
};

template <class DB>
concept kv_database = requires(DB& db, std::filesystem::path dir) {
   { DB::create(dir) } -> std::convertible_to<std::shared_ptr<DB>>;
   requires kv_write_session<
       typename decltype(db.start_write_session())::element_type>;
};

}  // namespace part::shape
```

(Sketch — exact metaprogramming hammered in TDD against all three
engines; per-class `static_assert`s of the sub-concepts keep failures
localized to the offending class instead of one opaque `kv_database`
failure.)

Layer 1 is named so layer 2 is a thin composition, not an adaptation:

| harness type | part composition |
|---|---|
| `database` | `arena` + root slots + tree dispatch |
| `write_session` | `session` (active range, floor) |
| `transaction` | `transaction` (floor epoch) + working-root `smart_ptr` |
| `tree` | a pinned root `smart_ptr` handle |
| `cursor` | path stack of raw indexes borrowed under the root pin (§12.2) |

`kv_compat_test.cpp` becomes runnable only when the tree layer exists; it
is the layer-2 acceptance gate, recorded here so layer-1 names and
transaction/session/root shapes don't have to move when we get there.

## 13. Copied from sal (into `namespace part`, paths flattened)

| sal file | fate |
|---|---|
| `block_allocator.{hpp,cpp}` | copy near-verbatim (it already does contiguous VA reservation, growth, mlock, punch_hole, fsync incl. F_FULLFSYNC) |
| `mapping.{hpp,cpp}` | copy (meta header mapping + sync_type) |
| `ucc/typed_int.hpp` | copy into `part/typed_int.hpp` |
| `ucc/hierarchical_bitmap.hpp` | copy as the basis of the freeset index; part derives a concurrent variant (atomic level 0 + XOR-settle upper levels + masked at-or-after descent, §6) |
| `config.hpp` / `debug.hpp` | strip to the ~50 lines we need (PART_WARN etc.) |
| `control_block_alloc` | **not copied** — superseded by this design; its alloc-scan idiom (pick a 64B chunk of bits, SIMD/popcount scan, CAS) is reused inside the freeset |
| `simd_utils.hpp` | copy iff the freeset scan ends up wanting `max_pop_cnt8_index64`; start scalar |

## 14. Directory & build

```
part/
  CMakeLists.txt          # static lib `part` + test executables
  include/part/           # config.hpp, typed_int.hpp, block_allocator.hpp,
                          # mapping.hpp, freeset.hpp, segment.hpp, arena.hpp,
                          # session.hpp, smart_ptr.hpp, debug.hpp,
                          # db_shape.hpp (shape concepts, §12.3 — zero deps)
  src/                    # block_allocator.cpp, mapping.cpp, arena.cpp, session.cpp
  tests/                  # see §15
```

Top-level `CMakeLists.txt` gains one line: `add_subdirectory(part)`.
Flags match house style: `-O3 -DNDEBUG -Wall -Werror=switch` for perf
targets, `-O1/-O2 -g` + ASan/TSan variants for test targets. Tests are
plain executables returning non-zero on failure (house pattern), runnable
via ctest.

## 15. Test plan (TDD order)

1. **freeset_tests** — single segment, single thread: class allocs land
   aligned; arbitrary runs never cross words; first-fit-from-floor ordering
   (masked descent returns the lowest eligible slot ≥ floor); exhaustion;
   free + re-alloc; alloc cost stays O(1) on a nearly-full segment (probe
   counter, not wall clock) — asserted for class sizes AND odd sizes
   (3/5/9 via mask composition, §6); composed placements match an oracle
   that brute-scans free_lines_1; hint + meta + top convergence after
   randomized op tapes (every tier compared against truth recomputed from
   free_lines_1 at quiescence).
2. **freeset_concurrency_tests** (TSan build) — N threads hammer
   alloc/free on one segment: no double-allocation (ownership ledger), L0
   popcount == ledger, hints converge at quiescence, free_lines counter
   exact.
3. **arena_tests** — multi-segment: active-range claim, continuity claim,
   longest-run selection (synthetic qualifying patterns), growth when none
   qualify, request-too-big-for-range switch, volatile mode. Reserved
   band: no alloc ever returns an index ≥ 0xFFFF0000 (fill a max-config
   arena to the brim and assert), and open rejects a config whose
   line count would cross into the band. Qualifying
   semantics: an every-other-line 50%-free segment never qualifies; per-class
   `free_count[k]` match truth recomputed from free_lines_1; anti-split
   preference (lone hole chosen over splitting an 8-group in the same word,
   split still taken when the word offers nothing else).
4. **txn_tests** — frees apply immediately; mid-txn alloc+release lines are
   reused by the floor scan (zero-extra-sync recycle); floor advances per
   commit; holes behind floor untouched; holes ≥ floor filled.
5. **sync_tests** — dirty-mask exactness; commit msyncs only dirty runs;
   syscall count and bytes_synced vs a process-fork + kill durability check
   (write, commit, set_root, SIGKILL child at each ordering point, reopen:
   root flipped ⇒ new data intact, root not flipped ⇒ old tree intact).
   Write-amp ratio asserted for two workloads: fresh-append (~1.0×) vs
   scattered-reuse (bounded, printed).
6. **refcount_tests** (TSan) — per-line independence: the implicit→explicit
   demotion scenario from §9 as a model test (alloc 8-line group, share 7
   lines into a second owner, release the group, verify exactly the
   unshared line frees); retain/release races free exactly once; saturation
   writer-clone threshold: drive a line to 190, verify retain reports
   false while pins still succeed in the 191..254 band, clone, verify the
   original drains to free as holders release while the clone lives on;
   read-session slots: 64 acquired, 65th refused, slot freed on dtor and
   reacquirable; release is exact at every count (no sticky no-ops
   anywhere); sentinel: a copy-proliferation loop throws `ref_overflow`
   exactly at the increment that would reach 255, count stays 254, and
   every prior copy still releases cleanly. **smart_ptr semantics**:
   copy/move/dtor retain–release
   balance (net zero on a tape of handle churn), give/take cross the state
   boundary with zero refcount traffic, destroy hook fires exactly once at
   1→0, borrow-under-pin (raw borrows valid while the pin lives — asserted
   via the ledger, not UB).
7. **recovery_tests** — synthetic graph → rebuild_* → freeset/refcounts/
   qualifying bits match a model; clean-flag fast path.
8. **part_perf** — alloc/free throughput single & multi-thread vs malloc
   baseline; write-amp per workload. M5 perf hygiene: ratio-based gates,
   one perf process at a time.
9. **shape_check** — compile-only target: `static_assert`s
   `part::shape::kv_database` (and each sub-concept) on `psitri::database`
   and `heart::kv::database`, gated on a `PSITRI_MERGE_DIR` CMake option
   (present on this machine, optional elsewhere). `part::database` joins
   the assert list when layer 2 lands; until then the concept is kept
   honest by the two engines that already exist.

Model-based testing throughout: a `std::map`/bitset oracle mirrors every op.

## 16. Open items folded into this draft (veto on review)

1. Default `PART_SEGMENT_BYTES` = **8 MB** (your range was 4–16).
2. Run selection = longest run, tie-break nearest-forward; **lazy claim**
   one segment at a time with adjacent-continuation preference (no
   multi-segment hogging). Is that what "next-longest continuous range"
   intended?
3. Reuse threshold = free 8-contig groups ≥ 50% of capacity, runtime config.
4. u8 refcounts, no pinned-stuck lines ever: writers clone at 190, readers
   capped at 64 concurrent sessions with ≤ 1 pin per line per session
   (aggregated same-version handles, tree layer), and any increment that
   would reach the 255 sentinel throws `ref_overflow` (programmer-error
   backstop: copy-proliferation instead of move). Fencepost note: reserving
   255 as the throw sentinel makes the ceiling 254, hence clone-at
   = 254 − 64 = **190**, one less than the 191 you quoted — veto if you'd
   rather keep 191 and cap pins at 63 (§9).
5. Size classes {1,2,4,8,16} + arbitrary-run escape hatch (§4) — tree layer
   free to use either.
6. Root slots: `max_roots` generic u64s in `meta.part` (default 16),
   u32-indexed to match the harness's `root_index`; semantics owned by the
   tree layer.
7. No background threads — nothing in this layer spawns one, and nothing in
   the design assumes one exists later (a compactor cannot exist at all:
   memory never moves).
8. Borrow-under-pin (§12.2): cursors/descents hold raw indexes covered by a
   root `smart_ptr` rather than per-hop smart_ptrs — the one sanctioned
   exception to "everything off-state is a smart_ptr", chosen to keep read
   paths free of refcount traffic.
9. One open transaction per session at a time (matches the harness usage;
   simplifies the floor bookkeeping).
10. Sentinel-band layout (§4): null = `0xFFFFFFFF`; the other 65,535 values
    carry 16-bit local payloads (`k = p & 0xFFFF`, k ≠ 0xFFFF) for the
    tree layer's local tail pointers. Layer 1 treats the whole band as
    opaque non-indexes.

## 17. Implementation phasing (after sign-off)

A. Copy + rename sal files, `part/CMakeLists.txt`, smoke-build;
   `db_shape.hpp` + shape_check passing on psitri & heart (locks the
   target shape before any part code exists).
B. `freeset` (L0 + hints) with tests 1–2.
C. `arena` segments/directory/claiming with test 3.
D. `session` txn/floor/pending-frees with test 4.
E. dirty tracking + commit/msync with test 5; refcounts with test 6.
F. recovery hooks + test 7; stats; perf harness (test 8).

Each phase lands compiling + tested before the next (TDD-bugs-first house
rule applies).

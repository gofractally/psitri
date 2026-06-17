# gcart → version-liveness GC — implementation plan

Status: planning (2026-06-17). Branch `sal2`. Companion to `docs/sal2_design.md`
(§4–§5 are the normative spec) and `docs/sal2_gcart_status.md` (M5 = "the deferred
perf lever").

## 0. Where we are vs. where this goes

What actually got built (refcount-first, per Dan's "refcount for now"):
- **`sal2`** = bump/slab allocators with **per-slot u8 refcount** liveness +
  dying-sentinel reclaim (`fixed_allocator<128/256/1152>`, `var_allocator<32>`,
  `big_allocator<64>`). Reopen recovery shipped (root table mmap + `adopt_existing_`).
- **`rcart`** = the refcount engine (the proven one — frame-stack cursor, 3-region
  node taxonomy, tiered-capable). **Frozen as the reference; keeps using sal2.**
- **`gcart`** = currently a byte-identical copy of rcart. **This plan rewrites it**
  to the version-liveness GC model of `sal2_design.md` §4–§5.

Why bother (the design's two claims, to be proven by this work):
1. **Perf.** Refcount COW retains *every shared child* on each clone + walks them on
   release — cold-cacheline traffic on the hot path. Version-GC touches **only the
   dead** (one `death` stamp per superseded husk) and **never a refcount on COW**.
   `sal2_gcart_status.md` names this the architectural gap to start_ver.
2. **Reclaim correctness at MVCC scale** without per-COW durable write-amp (death is
   derived from reachability; only the root flip is durable).

**start_ver is the working reference for version-stamp liveness** — it already does
vstamp births + a clone-log (random-access death) + held-version GC. Mine its
`hkv`/`kv` for the model; gcart-GC is "start_ver's liveness on rcart's node taxonomy
+ 3-region allocators."

## 1. The model in one paragraph (normative: design §5)

Every object has **birth B** (the committing txn's version, written at alloc) and
**death D** (default `ALIVE`). A COW commit at version `V`: alloc new/cloned nodes
with `birth=V`; stamp each superseded husk `death=V` (SHALLOW = clone husks on the
path; DEEP = dropped subtrees); flip the durable root; publish `current=V`. Readers
take a snapshot by **holding a root** (a `smart_ptr` / pinned root version S) and
traverse with **zero liveness checks** — nothing reachable from a held root is ever
freed. Liveness is consulted **only by the GC**: an object is reclaimable iff it is
**death-stamped** *and* **no held-root version lies in `[B, D)`**. No refcount on the
COW path; no version registry (held roots *are* the live set); no death certificates
(death is reconstructible by reachability, §5.6).

## 2. Allocator strategy — INDEPENDENT COPY (Dan, 2026-06-17)

**Decision: an independent copy, not a shared/templated allocator.** The refcount
and version-liveness engines have **genuinely different memory layouts** — the GC
engine carries the **object version (birth) IN the node header** (design §4.2),
which the refcount nodes do not. Straddling two layouts with a `liveness_policy`
template or `#ifdef`s makes both dirty; two clean copies stay readable and let each
optimize its own bytes. So:

- **rcart + sal2** stay exactly as they are (refcount, no node version field).
- **gcart gets its own version-liveness allocator** — a fork of sal2
  (`libraries/sal2gc/`, copy-and-evolve, the same Phase-0 move as sal→sal2):
  - **drop** the per-slot refcount array (`.rc`) + dying-sentinel + `retain`/
    `release_one` machinery;
  - **add** a per-slot `death_ver[]` array (`.dv`, one `u32`, default `ALIVE`;
    mmap'd, not force-synced, rebuilt on crash) + a GC-facing `free(slot)`;
  - keep all the proven slab infra (segments, freeset/bump, sessions, recycle,
    **reopen recovery / `adopt_existing_`** — adapted: rebuild `death` not refcount);
  - a monotonic commit counter lives in `.meta`.
- **gcart's nodes carry `birth` in their header** (a `u32` version field, design
  §4.2). This costs ~4 B/node and changes the node structs + `alloc_size`/ctor — but
  it's exactly why the copies are independent: the version IS part of this engine's
  layout. Birth is written once at alloc (in the bytes the mutator already writes,
  §5.2 — no extra cacheline). Death stays a parallel array (it's stamped post-hoc).

Cost owned: ~one allocator's worth of duplicated infra to maintain. Bought:
clean, independently-optimizable layouts; zero blast-radius on rcart; the COW path
touches **no** refcount at all (the design's whole point), because there isn't one.

## 3. Reference code to mirror (don't reinvent)

- **`libraries/start_ver/`** — version-stamp liveness in production: vstamp birth,
  clone-log (random-access death), held-version GC, the `template<cow_mode>`
  birth-latch. THE model. Read `start_ver/{kv.hpp, hkv/tree.hpp}` first.
- **`libraries/rcart/`** — the node taxonomy, 3-region allocators, frame-stack
  cursor, kv adapter, reopen recovery. gcart-GC keeps this shape; swaps the
  *liveness* (refcount → version) underneath.
- **`art/cow_coordinator.hpp`** — latest-reader-vs-writer collision/notify (design
  §5.3) for consistent "latest" reads under a concurrent commit.
- **`sal` recovery** (`reset_reference_counts`/`recover`) — the mark-from-roots pass
  to adapt for death/ref reconstruction (§5.6/§10).
- **`[[project_heart_cow_mode_latch]]`** (memory) — the EXACT bug to avoid: a missing
  unique→shared birth-latch mutates a snapshot-shared node in place and corrupts
  readers. The latch is a hard requirement (§5.2).

## 4. Phases (each independently landable + gated; mirror design §11)

### GC-1 — version core in the allocator layer
- Add per-region **`birth_ver[]` + `death_ver[]`** mmap arrays (sibling files,
  `.bv`/`.dv`, same geometry as the slab; not force-synced; flushed on clean close;
  rebuilt on crash). `death_ver` default `ALIVE` (0xFFFFFFFF).
- A monotonic **commit counter** in `.meta` (`begin_commit()`/`publish_commit(V)`).
- alloc stamps `birth_ver[slot]=V`, baseline (no retain); add a GC-facing
  `free(slot)` that bypasses the refcount assert path. Keep the slot refcount field
  **untouched on COW** (it stays for a future `smart_ptr`, or is vestigial in v1).
- **Gate:** unit test — alloc stamps birth; death-stamp + a held-version check makes
  an object GC-eligible iff no held version ∈ `[B,D)`; reopen restores birth (force-
  synced via the slab data) + tolerates stale death (safe).

### GC-2 — engine COW path: birth-latch + death stamps (the core)
**Status (2026-06-17): GC-2a/b/c DONE (the birth-latch), GC-2d NEXT (death+GC).**
- GC-2a/b — **birth (u32) in all 5 node layouts** (commits 5fa7986, bf118a6):
  setlist / nested_setlist / full headers (cl1/cl2 @ +4, full @ +8;
  double_setlist+big_nested inherit), and a body PREFIX on standalone sv_leaf /
  big_leaf (inline-in-nested leaves are birth-FREE — they share the nested node's
  header birth; `leaf_*_body` standalone readers skip the prefix). Direct aligned
  u32 access (slot base is grain-aligned), not memcpy.
- GC-2c — **birth-latch drives the COW decision** (commit 425145e): the 3 routing
  latch sites (upsert/remove/promote_nested) + 2 inline-eligibility checks now test
  `node_birth==_V` instead of refcount==1; `birth==V`⟹in place, `birth<V`⟹clone.
  `tree::_V` (write version) is stamped at EVERY node creation (make_sv, pack_*,
  reprefix clone, de-inline). spec_kv assigns each txn a fresh version from a
  **monotonic counter persisted in roots.dat slot 64** (births never collide across
  reopen). This is the validated HYBRID — latch is pure version-liveness, reclaim
  still refcount. Gate met: spec-tree oracle 833637/0 (in-place); spec-kv 417730/0
  incl. concurrent 2.6M reads / 0 rfail (the clone-path / snapshot-isolation gate).
- gcart is now **single-root** (commit c48d020): the multi-root `_extra`
  std::vector + open_root were rcart-fork dead weight + a heap-rule violation;
  removed (delta_art needs multi-root only on W, gcart is always B; shim uses one
  root). adb::kv_engine<gcart::kv::engine> still holds (kv_transaction, not
  multi_root); delta_art::tiered_kv<start_ver, gcart> compiles.
- **GC-2d (NEXT) — remove refcount, switch reclaim to version-liveness:** on
  supersession stamp `death_ver[husk]=V` (SHALLOW husks on the path; DEEP for
  dropped subtrees) instead of release_ref/release_husk/release_subtree; **remove
  all retain/release** from the COW + snapshot paths; move the engine sal2→sal2gc
  (death-based allocators). Needs GC-3 (held-version watermark) to keep the
  concurrent test green (reclaim must not free a held snapshot's version), so
  GC-2d+GC-3+GC-4(minimal) land together.
- Abort = nodes born at the unpublished V are unreachable → free immediately.
- **Gate:** oracle (vs std::map) 0 fails incl. churn + drain; **snapshot isolation**
  test (reader holding root@V sees exactly keys ≤ V under a concurrent writer);
  TSAN-clean concurrent bank balance PASS. This is the riskiest phase — the latch
  bug corrupts readers silently; test snapshots hard.

### GC-3 — snapshots = held root versions; read path checkless
- Snapshot acquire = read session pins the current root (pointer + version S);
  maintain an **oldest-held-version watermark** (min over the root table + active
  read sessions) — that, not a registry, is what the GC consults. Read traversal
  does **no** liveness checks (design §5.3); "latest" reads use `cow_coordinator`.
- **Gate:** concurrent readers at staggered versions each see their consistent
  snapshot; watermark advances as readers drop.

### GC-4 — the GC / reclaim pass
- Epoch-poked (on root supersession / last-snapshot-drop): scan `death_ver[]`;
  reclaim each slot that is death-stamped **and** `[B,D)` is covered by **no** held
  version (i.e. `D ≤ oldest_held` ∨ no held ∈ `[B,D)`) → `free(slot)`. FIX frees via
  the freeset; movable via the compactor/relocate path (later).
- 32-bit **wrap**: bounded-span modular compare (design §5.4 #2) — assert
  `current − oldest_held < 2^31` in `begin_commit`; rebase-on-GC (#1) deferred.
- **Gate:** GC reclaims a death-stamped object once no held root covers `[B,D)`,
  **never before**; 100 M-op fuzz (alloc/COW/snapshot/release) vs oracle —
  **0 use-after-free, 0 premature free**; ASAN + TSAN clean. (Design §4.3 acceptance.)

### GC-5 — recovery (crash) + clean reopen
- Clean close: flush `birth`(already durable in slab)/`death`/watermark; reopen
  reuses them (the rcart `adopt_existing_` pattern already does the slab side).
- Crash: rebuild `death` (and any ref) in **one mark-from-all-retained-roots pass**
  (§5.6) — reset death=ALIVE, enumerate retained roots ascending, set death to the
  next-newer-root-that-stops-reaching; over-approximation is safe. Or the cheap
  variant: reset suspect death=ALIVE, let the next GC reclaim leaks.
- **Gate:** kill-9 mid-commit, reopen, tree == last durable commit (oracle replay),
  no leaked/double-allocated slots (design §10 acceptance).

### GC-6 — conformance, bench, Silkworm
- `static_assert(adb::kv_engine<gcart::kv::engine>)` + the recovery conformance test
  (already required for all engines) + full kv_compat battery.
- Bench vs rcart (refcount) + start_ver + native: the headline is **gcart-GC ≥ rcart
  on read/insert** (no refcount-COW traffic) **at ≤ rcart memory**, and the
  oplog/bank/Silkworm replays byte-exact.
- **Gate:** Silkworm op-stream byte-exact; bank balance PASS; perf ≥ rcart.

## 5. Risks & mitigations
- **Birth-latch correctness (GC-2)** — the known snapshot-corruption trap
  ([[project_heart_cow_mode_latch]]). Mitigation: copy start_ver's proven
  `template<cow_mode>` verbatim; snapshot-isolation test before any perf work.
- **GC trigger correctness (GC-4)** — freeing an object a reader still sees. The
  whole [B,D)-vs-held-version test must be exact for runtime stamps (volatile reader
  can pin any version). Fuzz-vs-oracle is the backstop.
- **death array not force-synced** — a lost death only makes a dead object look alive
  (leak), never the reverse; recovery rebuilds. Safe by construction.
- **Coexistence with rcart/sal2** — Option C keeps blast-radius in gcart-GC; if sal2
  needs changes that risk rcart, gate them behind the policy template (B) instead.

## 6. Decided / open
**Decided (Dan):** independent copy, not templated (§2); **birth in the node header**
(§2/§4.2), not a parallel array; death stays a parallel `.dv` array.

**Open (resolve at GC-1):**
- 32-bit `version_t` (design default; 4 B commits, bounded-span wrap) vs wider.
  **Recommend 32-bit** per the design; wrap handled by bounded-span compare.
- Keep a vestigial slot `ref` for a future `smart_ptr`, or pure version-liveness in
  v1 (snapshots = held root pointer + watermark, no per-slot ref). **Recommend pure
  version-liveness v1** (simplest; add `smart_ptr` ref later if a use needs it).
- How gcart-GC and rcart coexist in the shim/bench: both as separate backends
  (`PSITRIMDBX_BACKEND_GCART` = GC, add `PSITRIMDBX_BACKEND_RCART` = refcount) + both
  rows in bank/oplog/scan benches for a direct refcount-vs-GC comparison.

## 7'. Progress bar + what GC does / doesn't buy (Dan, 2026-06-17)
The acceptance bar is comparative: **as long as the engine is faster than native
libmdbx and uses less disk than our prior impls** (delta_art ~334 GB / native 134 GB
at 8M Silkworm), it's progress.

**What version-GC buys: speed, NOT disk (Dan).** GC eliminates the refcount
COW traffic — the cold-cacheline retain-on-every-shared-child cost; that is the
perf lever (≈ the gap to start_ver). It does **not** reduce disk: the COW garbage a
write produces is identical whether liveness is a refcount or a version, so the
disk high-water is the *same* (if anything GC's deferred reclaim can hold garbage a
touch longer). **Disk is a separate problem** — solved by reclaim timing /
compaction (the movable-region compactor, design §6.2), not by the liveness model.
So gcart-GC's job is the SPEED bar; the disk bar is owned by compaction/reclaim
work, tracked separately. Silkworm-scale evidence: the tiered start_ver→gcart run
held ~115 GB at 5M (≈ gcart-direct), confirming tiering didn't densify either — disk
needs its own lever.

## 7. Sequencing note
Land GC-1..GC-2 behind the existing gcart tests staying green at each step
(gcart-spec / gcart-spec-kv / gcart-kv), pause for review after GC-2 (the liveness
rewrite, design's review checkpoint), then GC-3..GC-6. rcart stays the fallback the
whole way; the shim's stable backend can point at rcart until gcart-GC passes GC-6.

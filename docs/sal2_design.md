# SAL 2.0 — a movable + fixed multi-region memory system (design + build plan)

**Status:** design, ready to build. **Repo:** `@gofractally/psitri`.
**Branch:** `sal2`. **New libraries:** `libraries/sal2/` (the allocator) and
`libraries/gcart/` (the GC-ART engine built on it).
**Basis:** Dan's rulings of 2026-06-16 (recorded verbatim in §3.1), applied to the
existing machinery in `libraries/{sal,start_ref,start_ver,art}`.

This document is **self-contained**: everything needed to build sal2 lives here
and in the named in-repo libraries — no external documents or repositories are
required. It is concrete on purpose: exact before/after struct layouts, exact
file/type names, exact acceptance gates per phase. Where a number is a knob, it
says so.

> **Engine map** (all in `libraries/`): `start_ref` = static ART with refcount
> liveness; `start_ver` = static ART with version-stamp liveness; `sal_art_ref` =
> ART on the movable `sal` allocator; `delta_art` = a composed hot-delta + base
> engine. The database *concepts* live in `libraries/adb` (namespace `adb::`).
> These engines are untouched A/B comparators for sal2.

---

## 0. Thesis (read this first)

> Cold cache lines are the #1 cause of RAM-bound slowness. A *movable* node costs
> one extra indirection (control block → location) at **every** level of the
> tree, doubling cold-cacheline loads on descent. Reference counting adds more
> cold-cacheline traffic on every COW. **SAL 2.0 keeps the routing layer (inner
> nodes) in a *fixed*, directly-addressed region with no control block and no
> refcount, and keeps only the *leaves* (the key/value data that actually
> benefits from hot/cold clustering) in the movable region.** Liveness everywhere
> becomes a version-interval test `[birth, death)` against a set of pinned
> snapshot versions — no per-object reference counting at all.

SAL 2.0 is **not a clean slate**. It is `libraries/sal/` *copied and evolved*.
All hard-won SAL machinery (32 MB segments, the compactor that clusters hot/cold,
write sessions, mmap reserve+commit address stability, checksums, sync levels,
power-loss recovery) is preserved. Four targeted mutations (§3.1) turn refcount
liveness into version liveness and make the grain configurable; three regions
(§6) and a redesigned pointer (§7) are added around that core.

All existing engines (`art`, `start_ref`, `start_ver`, `sal_art_ref`) **stay
untouched** as A/B comparison points.

The engine that ships on top of this allocator is **`gcart`** (the GC-ART
library, a new `libraries/gcart/` that depends on `sal2`) — a **first-class
`adb` engine**: it must pass the compile-time `adb` concepts and be a **drop-in**
for any library written against `adb` — delta_art, walart, dwal, the 3rd-party
shims, multiindex — so they can template on it unchanged (§3.2). Two new
libraries result: **`sal2`** (the allocator, §4–§10) and **`gcart`** (the engine
on it, §3.2 / Phase 3).

---

## 1. The problem, precisely

Measured in the `start_ref`/`start_ver`/`sal_art_ref` line of work:

1. **Indirection doubling.** SAL today addresses every object through a
   `control_block`: `ptr_address (4B)` → `control_block (8B, cold)` →
   `location`. On a 7-hop dictionary descent that is 7 *extra* cold-cacheline
   loads purely for indirection. The movable indirection earns its keep **only**
   for data the compactor wants to relocate to separate hot from cold — which is
   the **leaves**, not the routing layer.
2. **Refcount cold traffic.** COW clones touch refcounts on the clone *and* on
   every shared child whose count changes; release walks them again. Each touch
   is a cold atomic on a cold line.
3. **Non-power-of-2 pointer latency.** Interleaving a tag into the offset makes
   each branch pointer an odd width (5 B), which measurably raised per-node CPU
   latency in prior tests. Tags must live **separate** from offsets.
4. **The routing layer is a minority of state but is always on the critical
   path.** Inner nodes (setlist / c-nodes / full) are a small fraction of bytes;
   the leaves dominate. So: make the small, always-traversed routing layer as
   cheap as physically possible (fixed, direct-indexed, headerless dispatch),
   and pay the movable price only at the leaves.

---

## 2. Best-of-everything / avoid-the-worst (what we take, from where)

| Idea (KEEP)                                                                                       | Source in-tree                                         | Why |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | --- |
| 32 MB segments, mmap reserve(PROT_NONE)+commit, address stability                                 | `sal/block_allocator.hpp`, `sal/mapping.hpp`           | Pointers never relocate; copy verbatim. |
| Background **compactor** that clusters hot/cold by write recency                                  | `sal/src/allocator.cpp` compactor_loop                 | The whole reason the *movable* region exists. |
| Write **sessions** owning an active segment + touched-page set for sync                           | `sal/allocator_session.hpp`                            | Per-region session model generalizes cleanly. |
| Sync levels none/msync_async/msync/fsync/F_FULLFSYNC + per-page dirty-run msync                   | `sal/config.hpp`, `sal` write-session touched-page set | Durability, with dirty-run merging to fight fsync amplification. |
| Power-loss recovery, checksums, torn-tail truncation                                              | `sal/allocator.hpp::recover*`, `alloc_header` checksum | Crash safety; adapt to 3 regions + version model. |
| **Hierarchical free bitmap** (l0 lines + g2/g4/g8 + meta/top, lock-free, O(1), first-fit-forward) | `start_ref/freeset.hpp::seg_freeset`                   | The **intra-segment free bitmap** — one of `FixedAllocator<N>`'s four parts (§6.1); the file/segment/session/death layers wrap it. Port the bit-twiddling; don't reinvent. |
| **Version-stamp liveness** (no refcounts), DEEP/SHALLOW COW supersession                          | `start_ver` vstamp + clone log                         | Proves Dan's ruling #3/#4 model works at scale. |
| Headerless **tag-dispatch** taxonomy (setlist/c2/c4/c8/full, skip-CL0)                            | `art`, `start_ver` `tagged_ptr`/`cnode`                | The routing layer sal2's FIX region serves. |

| Idea (AVOID)                             | Why it's the worst                                    | What we do instead |
| ---------------------------------------- | ----------------------------------------------------- | --- |
| Control block on **inner** nodes         | doubles cold clines on the always-hot path            | FIX region: 4-byte direct line index, no CB. |
| Per-object **reference counting**        | cold-cacheline traffic on every COW/release           | version `[birth,death)` + refcounted *snapshots*. |
| Tag interleaved into offset (5-byte ptr) | non-pow2 width → CPU latency                          | 4-byte offset array + separate **3-bit-packed** tag array. |
| One contiguous region for everything     | can't separate hot/cold OR fragments 9-cl full nodes  | 3 regions: FIX (cl1/cl2/cl9), movable SAL, BIG. |
| Relocating ≥4KB leaves                   | already at OS page size; relocation is pure write-amp | BIG region: fixed, never moved. |

---

## 3. Architecture overview

```
                        ┌───────────────────────────────────────────┐
   key descent  ───────▶│  FIX region   (routing layer, the majority │  ← no control block
   (inner nodes)        │  of *hops*, the minority of *bytes*)        │  ← 4B direct index
                        │  .cl1 setlist · .cl2 c2/double · .cl9 full  │  ← version liveness
                        └───────────────┬───────────────────────────┘
                                        │ leaf branch (tagged)
                        ┌───────────────▼───────────────────────────┐
   suffix+value  ──────▶│  MOVABLE SAL region  (the data — clustered  │  ← control block (relocation)
   (< 4KB leaves)       │  hot/cold by the compactor)   .segs         │  ← version liveness
                        └───────────────┬───────────────────────────┘
                                        │ overflow ≥ 4KB
                        ┌───────────────▼───────────────────────────┐
   big leaves    ──────▶│  BIG region  (≥4KB, fixed, never moved)     │  ← control block, zone=big
   (≥ 4KB)              │  boost-shmem-malloc style free lists  .big  │  ← version liveness
                        └─────────────────────────────────────────────┘

   versions: held roots are the pins (root table + smart_ptr; root embeds version)
   death:    parallel u32 death_ver[ptr_address]  (.ctrl-adjacent)  — supersession stamp
```

The **version model is unified** across all three regions; only the *addressing*
differs (direct line index for FIX/BIG vs control-block indirection for movable
SAL). Readers pin a root version and traverse without touching any liveness
state; liveness is consulted **only** by the GC/compactor.

### 3.1. Dan's rulings (verbatim, 2026-06-16)

> sal2 will be a new allocator library that needs to **copy sal and evolve**
> rather than be a totally clean slate. A lot of work went into sal, and we do
> not want to lose it. The main thing we are changing in sal is:
> 1. **32 bit obj version**
> 2. **static const config to change the default byte alignment from 128 to 32**
>    (or whatever other alignment we want to test)
> 3. **control block is losing its ref count field and a parallel u32 death_ver
>    array will be maintained instead**
> 4. **version numbers need to be first class objects that are reference counted**
>    so that we can quickly test whether an object created at V is visible from
>    any snapshot in the range `[birth, death)`

> v1 scope: **all three regions, staged.**

These four mutations are specified exactly in §4; the version model they imply is
specified in §5.

### 3.2. Engine conformance: an `adb` drop-in (hard requirement)

`gcart` is not a throwaway test vehicle — it is a production engine that
**conforms to the `adb` concept** and is **interchangeable** with the existing
engines. The contract has two faces; `gcart` satisfies **both**:

1. **The compile-time shape concept** — `adb::kv_database`
   (`libraries/adb/include/adb/concepts.hpp`, namespace `adb`). It composes
   `kv_cursor` → `kv_tree` → `kv_transaction` → `kv_write_session` →
   `kv_database`, plus the locked extensions `storable_type`/`typed_values` (POD
   token take/make), `multi_root_transaction` (ascending open_root, set_root
   exchange), and `tree_snapshots` (O(1) snapshot of a root). The engine's
   classes `static_assert` these (as `start_ref`/`start_ver`/`sal_art_ref`/
   `psitri` do — see `tests/kv_compat_test.cpp` and
   `libraries/adb/tests/concept_check.cpp`), so a drift is a compile error
   naming the class, not a runtime surprise.

2. **The traits-bundle concept** — `psitri::dwal::kv_engine`
   (`libraries/psitri/include/psitri/dwal/kv_engine.hpp`): five member types
   `{database, write_session, read_session, transaction, cursor, config}` with
   unified construction `database::open(path, open_mode, config) ->
   shared_ptr<database>` and the full cursor surface
   (`seek_begin/seek_last/lower_bound/upper_bound/next/prev/is_end/is_rend/
   is_subtree/key/value<T>/get<T>`). This is the form that lets a higher library
   write `template<class Engine> ... Engine::database` and accept `gcart` as a
   drop-in base — exactly how `dwal`/`delta_art` take an engine today
   (`psitri::dwal::psitri_engine`). `gcart` provides `gcart::engine` as the
   traits bundle.

**Concretely, `gcart` exposes a native, psitri-shaped API** mirroring
`start_ref::kv` / `start_ver::kv` / `sal_art_ref::kv`:
```cpp
namespace gcart::kv {
  class database;          // ::open(path, mode, config) -> shared_ptr<database>
  class write_session;     // start_transaction(root_idx) -> transaction
  class read_session;      // create_cursor(root_idx) -> cursor
  class transaction;       // upsert/remove/commit/abort, get<T>, get_any, get(k, fn)
  class cursor;            // the full ordered-iteration surface above
  // typed values: register_value_types<T>(); multi-root: open_root(idx)/primary();
  // snapshots: snapshot(root_idx)
}
static_assert(adb::kv_database<gcart::kv::database>);   // and the extensions it supports
```

**Drop-in proof obligations** (acceptance, §12): `gcart` is added as the **5th
engine** in `tests/kv_compat_test.cpp`, passing the identical templated battery
(including the `if constexpr`-gated typed_values / multi_root / snapshot batteries
for the features it implements), and at least one engine-templated consumer (the
simplest: `delta_art` or a `kv_engine`-templated smoke test) is instantiated on
`gcart::engine` and run green — demonstrating real interchangeability, not just
concept satisfaction.

**Tests are authored in `adb` form (test-authoring rule).** `tests/` is "the
behavioral battery, written against `adb`, instantiated on every engine's native
database type." So: **when `gcart` needs a new test — or when something fails —
express the test against the `adb` concept** (templated on `Engine::database`,
not on `gcart` directly) and instantiate it on every engine. The existing engines
(`start_ref`, `start_ver`, `sal_art_ref`, `psitri`) are **differential oracles**:
any behavior `gcart` must exhibit, they already exhibit, so a divergence between
`gcart` and the others on the same `adb` battery localizes the bug — use them
alongside `std::map` (which stays the oracle for plain ordered-map semantics;
the engines are the oracle for engine-specific behavior like snapshots, typed
values, and multi-root). Avoid `gcart`-only tests except for sal2-internal
mechanics (allocator/version/region) that have no `adb` surface.

---

## 4. The four SAL mutations — exact before/after

Work happens in `libraries/sal2/` (a copy of `libraries/sal/`, namespace
`sal`→`sal2`). The "before" snippets are today's `sal`; the implementer reads the
live files for ground truth before editing.

### 4.1. Configurable grain (`cacheline_size`)

**Before** — `sal/config.hpp`:
```cpp
constexpr uint32_t cacheline_size  = 128;            // hard-coded
constexpr uint32_t cacheline_shift = (cacheline_size == 128) ? 7 : 6;
```
`location::from_cacheline()` shifts by 7 (×128). Object size field, segment
math, and `max_database_size` all derive from 128.

**After** — `sal2/config.hpp`:
```cpp
// grain is the persistent format's allocation quantum; a build-time knob so we
// can A/B 32 vs 64 vs 128. Changing it is an on-disk format break (guarded by a
// geometry field in the segment/meta header — reopening with a different grain
// must throw — a geometry stamp in each region's segment header guards reopen).
constexpr uint32_t grain_size  = SAL2_GRAIN_SIZE;     // default 32; was 128
static_assert((grain_size & (grain_size - 1)) == 0);
constexpr uint32_t grain_shift = std::countr_zero(grain_size);
```
- Rename `cacheline_size`→`grain_size` everywhere (the concept is "allocation
  quantum", which is no longer tied to a CPU cacheline). Provide
  `-DSAL2_GRAIN_SIZE=32` (default), overridable to 64/128 for benches.
- **Scope:** the grain knob is the **movable region's** quantum (Dan's ruling #2:
  32-byte alignment instead of 128, for movable leaves). The **FIX regions** allocate
  in **128-byte cacheline slots** (cl1/cl2/cl9 = 1/2/9 cachelines) — routing nodes
  are cacheline-designed (skip-CL0 dispatch), and their per-slot death overhead
  (§6) is computed in those 128-byte units.
- Derive every dependent constant from `grain_size`/`grain_shift`
  (`location` shift, `alloc_header::_size` width, `max_database_size` for a
  given offset width). **Audit every literal `128`/`<< 7`/`7`** in the copied
  tree — there are several (`memcpy_aligned_64byte`, prefetch helpers,
  `location.hpp`). Each must route through the knob or be justified as a true
  CPU-cacheline concern (e.g. prefetch granularity) that is *independent* of the
  allocation grain.
- **Consequence on offset width:** the offset is in grain units, so its reach is
  `2^offset_bits × grain`. The CB budget after dropping `type` leaves `offset:44`
  (§4.3) → 2^44 × 32 B = **512 TB per zone** at 32 B grain (and each zone is
  independent, §4.3). Keep the offset width a derived constant sized so
  `offset_bits + grain_shift` reaches the configured `max_database_size`; the bits
  freed by removing `type` pay for it.

**Acceptance:** sal2 builds and `sal2-tests` pass at grain 32, 64, and 128 (a CTest
matrix); a stored DB created at one grain refuses to reopen at another with a
clear geometry error.

### 4.2. 32-bit object version

**Before** — `sal/control_block_alloc.hpp`, `sal/alloc_header.hpp`:
```cpp
struct version_t { uint8_t bytes[6]; };               // 48-bit
static_assert(sizeof(version_t) == 6);
class ptr_address_seq { ptr_address address; version_t sequence; };  // 10 B
static_assert(sizeof(ptr_address_seq) == 10);
// alloc_header: uint16_t _checksum; ptr_address_seq _address; _size:25; _type:7  → 16 B
```

**After** — `sal2`:
```cpp
using version_t = uint32_t;                           // 32-bit birth/death/snapshot id
class ptr_address_seq { ptr_address address; version_t version; };   // 8 B
static_assert(sizeof(ptr_address_seq) == 8);
// alloc_header shrinks: uint16_t _checksum; ptr_address_seq _address(8); _size/_type(4) → 14 B
//   (pad/repack to a clean 16 B or tighten to 14 B; pick one and static_assert it)
```
- The object header's `version` field is the object's **birth version** (§5).
  This reuses the existing "sequence" slot — it already lives in the header and
  is written at alloc; we are narrowing it and giving it MVCC meaning.
- 32 bits ⇒ 4 B commits before wrap. Wrap handling: §5.4.

**Acceptance:** header/seq static_asserts hold; recovery still reads `_checksum`
at offset 0; a round-trip writes and reads back birth versions correctly.

### 4.3. Control block: keep refcount (for `smart_ptr`), add parallel `u32 death_ver[]`

**Before** — `sal/control_block.hpp` (the 64-bit packed CB):
```cpp
uint64_t ref : 18;              // KEEP — but the COW mutation path must NEVER touch it (§4.4)
uint64_t type : 6;              // ← REMOVE (the parent branch tag carries node type/region, §7)
uint64_t cacheline_offset : 38; // keep
uint64_t active : 1;            // keep (MFU read sampling)
uint64_t pending_cache : 1;     // keep
```
`retain()`/`release()` stay (they back `smart_ptr` shared-pointer use), but the
**per-node refcount traffic on the COW path is removed**: cloning no longer
retains shared children and releasing no longer walks them — COW stamps `death`
instead (§4.4 / §5.2). So `ref` moves only on an explicit `smart_ptr` acquire/drop.

**After** — `sal2/control_block.hpp`. The CB is **`{ ref, MFU bits, zone, offset }`**.
`type` is gone (the parent branch tag supplies node type for read routing, §7);
`ref` stays but is **mutation-path-free** — only `smart_ptr` moves it:
```cpp
// 64-bit control block — every field sized to its real range; static_assert == 8 B:
uint64_t ref           : 16;  // smart_ptr pin count, baseline 1; ample for concurrent
                              //   snapshots; COW NEVER touches it (§4.4). Overflow spills
                              //   to a parallel wide-count (§15), not a hard cap.
uint64_t active        : 1;   // MFU: object was read (read-cache sampling)
uint64_t pending_cache : 1;   // MFU: pending promotion to the pinned/hot cache
uint64_t zone          : 2;   // region of `offset`: movable | big | 2 reserved
uint64_t offset        : 44;  // zone-relative location: 2^44 × grain per zone
                              //   (512 TB @ 32 B grain); steal ref bits to widen if needed
```
(Widths are range-reasoned defaults — adjust and `static_assert(sizeof(control_block)
== 8)`. `ref:16` covers far more concurrent pins than any object sees; `zone:2`
covers the two v1 regions plus headroom; the rest is `offset`.)
**Why `zone` lives in the CB.** The *read* path does **not** consult it — the
branch tag (§7) already routes a reference to its region (`suffix_value` ⟹
movable, `big_suffix_value` ⟹ BIG). But the **MFU read-cache promoter** has no
tag: the `active`/`pending_cache` bits live in the CB, and promotion runs later
from a bare `ptr_address` (the rcache queue / a CB scan). To promote a hot object
it must **(a)** tell **movable** (relocatable → copy into a pinned/hot segment)
from **BIG** (never moves → skip), and **(b)** resolve the object's address to copy
it — both need the zone. The compactor wants the same when it works from a bare
back-reference. So the zone rides **with the MFU bits, in the CB**; it is not
redundant with the tag, because these consumers never see the tag.

**`offset` is zone-relative** — it counts the zone's alignment units from the
zone's base: `addr = zone_base[zone] + offset × zone_stride[zone]` (movable: 32 B
grain; BIG: page alignment; FIX per-class strides, §7.2). The same `offset` value
names *different* objects in different zones — `zone` disambiguates — so **zones
cost no shared address space**: each gets the *full* offset range at its own grain,
not a sub-range of one global space. `resolve(zone, ptr_address)` reads the CB's
`zone`+`offset`; the relocation CAS rewrites only `offset`. v1 zones: `movable`
(`.segs`) and `big` (`.big`, never moved — `.big` is not a compacted region);
remaining `zone` values reserved (a second movable tier, a stored-subtree-root
reference) — not in v1.
**The death record** — a **new parallel array**, one `u32` per control block,
alongside the control-block array (a sibling region/file, dense for GC scan
locality):
```cpp
// sal2/death_ver_alloc.hpp  (parallel to control_block_alloc, same zone geometry)
std::atomic<uint32_t> death_ver[/* one per ptr_address */];
//   death_ver[a] == k_alive (e.g. 0xFFFFFFFF) → object still live
//   death_ver[a] == V       → superseded at commit version V; live for [birth, V)
```
- **Liveness is split between the version interval and the `smart_ptr` count.**
  An object's structural life is the half-open interval `[birth, death)` (`birth`
  in the header §4.2, `death` in `death_ver[addr]`, default alive); `ref` adds
  explicit pins on top. `retain`/`release` stay (for `smart_ptr`), but the COW
  path uses neither — it stamps `death`.
- **COW supersession** (the only writer of `death_ver`): when node N is replaced
  by N′ in the transaction committing at version V, the writer stamps
  `death_ver[N] = V`. Children N keeps sharing are **not** touched (SHALLOW);
  children N drops get `death_ver = V` too (DEEP). This is start_ver's clone-log
  made random-access. **The `death_ver` array is an mmap'd file that is *not
  force-synced*** — it is a cache of a fact fully derivable from the persistent
  root structure (`death` = first version that stops *reaching* the object). The
  durable truth is the root flip; death stamps are never on the per-commit fsync
  path, so COW adds **zero** durable write amplification. The file is flushed at
  clean shutdown (instant reopen); after a **crash** its entries may be stale, so
  recovery either reconstructs them by traversal (§5.6, §10) or tolerates the
  residue as **leaks** — a *lost* death stamp only makes a dead object look alive
  (safe; reclaimed by a later GC), never the reverse. No death certificates are
  logged.
- **Compactor & GC**: when an object is **death-stamped** and **no held root
  covers `[birth, death)`** (the live set is the held-root versions, §4.4 — not a
  registry), the GC calls **`release()` once**; release-to-zero frees it now (no
  `smart_ptr`) or at the holder's final release (`smart_ptr` out). The GC is just
  another reference holder — it doesn't inspect the count. `cas_move()` stays for
  relocation (CASes `offset`).
- **FIX region** has *no* control block: its death lives in a **per-slot**
  `death_ver[slot]` (one u32 per node) in a sibling array, indexed by slot, not
  ptr_address (§6.1). **MOVABLE and BIG both keep a control block** and share the
  CB-indexed `death_ver[ptr_address]` array. The CB's `zone` distinguishes them for
  the tag-less consumers (the MFU promoter and the compactor): `movable` relocates
  (compactor may `cas_move`); `big` never moves (`.big` is not a compacted region).
  The read path uses the branch tag for this instead (§7). Same `[birth,death)`
  semantics everywhere; only the index space differs.

**Acceptance:** a unit test proves an object is GC-eligible iff no held-root
version lies in `[birth, death)`; the compactor relocates exactly the still-visible
objects and drops the rest; ASAN/TSAN clean; a fuzz harness (random
alloc/COW/snapshot/release) matched against an oracle shows zero
use-after-free and zero premature free across 100 M ops.

### 4.4. Snapshots pin versions implicitly — no registry

**There is no version registry and no `pin`/`unpin` API.** The machinery that
already exists *is* the pin:
- **Top roots** — the durable root table holds the published roots (one per tree).
- **`smart_ptr`** — an owning reference; holding one to a root **is** a snapshot.
- **`smart_ref`** — the threaded-by-reference traversal handle (no per-node refcount).
- **The root of every tree is a SAL object with its version (`birth`) embedded** in
  its header (§4.2).

So the **set of alive versions = the versions of the roots currently referenced**:
the top-root table entries, plus every `smart_ptr` a reader holds to a current or
older (snapshot) root. A reader takes a snapshot by acquiring a `smart_ptr` to the
current root — one `retain` on that root's control block — and traverses via
`smart_ref`; dropping the `smart_ptr` (one `release`) ends it. The root stays alive
while held, and **everything reachable from a held root stays alive**, because the
GC reclaims by reachability from the live root set (§5.5/§5.6). This replaces the
registry's `any_pinned_in`/`min_pinned`: the held roots' versions *are* the
"pinned versions."

**Where the refcount went** (revises the strict reading of ruling #3): the CB
**keeps a `ref`** so a node can be used as a `smart_ptr` shared pointer — but **the
COW path never touches it** (§4.3), so the per-node refcount traffic that the
redesign set out to kill is gone; `ref` moves only on an explicit `smart_ptr`
acquire/drop (a snapshot on a root — rare). Baseline `ref == 1` is the object's
**structural reference**, owned by the GC/tree; each `smart_ptr` adds one.

**GC reclaim is one `release()`.** When the GC determines an object is
**death-stamped** and **no held root covers `[birth, death)`** (invisible to every
live snapshot), it calls `release()` **once** — relinquishing the baseline
structural reference. Ordinary release-to-zero does the rest:
- if no `smart_ptr` was ever taken (`ref` was 1), that release is the final one →
  **freed immediately**;
- if a `smart_ptr` is out (`ref` > 1), the GC's release just decrements and the
  object is freed at the holder's **final** release.

So the GC never special-cases the count — it is **just another reference holder
dropping its own reference**, and `smart_ptr` callers get correct shared-pointer
semantics for free. (The GC's trigger — death-stamped + uncovered by any held
root — is the reachability/version test of §5.5/§5.6; the `release()` is SAL's
existing path.)

- **Persistence:** only **`birth` is force-persistent** (in the object header,
  synced with the object). **`death` and `ref` live in mmap'd files but are *not
  force-synced*** — flushed at clean shutdown (instant reopen), but after a crash
  their entries may be stale, so recovery reconstructs them by one
  mark-from-all-roots pass (§5.6, §10) or tolerates the residue as leaks (safe
  direction). `ref` is rebuilt by counting references exactly as SAL rebuilds
  control-block counts on recovery.

**Acceptance:** snapshot isolation (a reader holding a `smart_ptr` to root@V sees
exactly the keys committed ≤ V under concurrent writers); GC reclaims a
death-stamped, `ref == 1` object once no held root covers `[birth, death)`, never
before.

---

## 5. The version-liveness model in full

**The whole model in one analogy.** In a single-user, non-shared, non-versioned
tree you would simply `new` and `delete` nodes as the current version needs them.
SAL 2.0 is that same tree with exactly one change: **every node you *would have*
deleted, you instead stamp with a death version.** Nothing is freed at mutation
time; a node is freed only once no alive snapshot can still see the version
interval it belonged to. Births come for free (the committing version, already in
the header), deaths are the stamp, and reclamation is deferred to the GC (§5.5).

### 5.1. Definitions
- Every object has **birth** B (header, written at alloc = the committing
  transaction's version) and **death** D (`death_ver[id]`, default `ALIVE`).
- A **snapshot** is a **held root** at version S (a `smart_ptr` to that root).
- **Visibility:** object is visible from snapshot S ⟺ `B ≤ S < D`.
- **Garbage:** object is reclaimable ⟺ no held-root version S satisfies
  `B ≤ S < D` ⟺ **no held root covers `[B, D)`** (the held roots are the live set,
  §4.4).

### 5.2. Write path (COW commit at version V)
1. `V = begin_commit()` — advance the monotonic commit counter (a plain atomic in
   `.meta`, not a registry of pins).
2. Allocate new/cloned nodes with `birth = V`. Stamp superseded husks
   `death_ver = V` (SHALLOW for clone husks, DEEP for dropped subtrees).
3. Flip the root (durable root publish, §9), then `publish_commit(V)` (the current
   version becomes V).
4. Aborts are cheap: nodes born at an unpublished V are unreachable, so their
   slots are freed immediately (the session knows exactly what it allocated this
   txn), with the GC epoch as the backstop.

**Unique vs shared — the COW mode latch (`upsert<cow_mode = shared>`).** On the COW
descent the mutator reads each node's **birth** (already in the header it loads to
route) and compares it to the txn version `V`:
- `birth == V` ⟹ **unique** — allocated or cloned earlier in *this* txn; mutate in
  place.
- `birth < V` ⟹ **shared** — committed in a prior version, so potentially visible
  to a pinned snapshot; it **must be cloned** before any change, or the snapshot
  corrupts.

The descent is templated on a **scoped enum `cow_mode { shared, unique }`, default
`shared`** (an enum, *not* a `bool` — named modes avoid the true/false ambiguity
and leave room for additional descent modes). A shared node's children were all
born no later than it, so **a shared node implies every node below it is shared.**
The descent therefore *latches*: `upsert<unique>` checks `birth` per node; the
moment it meets a shared node it clones and recurses as `upsert<shared>`, which
clones unconditionally and **never checks `birth` again**. The public
`upsert(k,v)` defaults to `<shared>` — always safe — and the recursion drops to
`<unique>` only where `birth == V` proves it; so the check is paid only on the
unique prefix (the path already touched this txn). This **birth-based latch — not
refcounts** — is what makes in-place mutation safe, and it is a **hard
requirement** for the gcart mutation path (the proven `template<cow_mode>` pattern
from the existing engines); omitting it mutates a snapshot-shared node in place and
corrupts live readers.

**Mutator liveness footprint (the whole point).** A COW commit touches exactly
two pieces of liveness state:
1. **the version counter** — advanced once at `begin_commit` and read to stamp; a
   single hot value, in cache;
2. **death stamps** — `death_ver[x] = V`, one `u32` write per object the commit
   *deletes* (each SHALLOW husk on the COW path ≈ one per level, plus each DEEP-
   dropped subtree object). This is the only liveness-specific *extra* write.

**`birth = V` rides in the node bytes** the mutator is already writing, so it costs
no separate cacheline. And the mutator touches **no reference counts of any kind**:
shared children carried into a clone are *not* stamped (they stay alive), and the
CB `ref` is a `smart_ptr`/snapshot concern (§4.4) — the GC moves it later, never
the writer. That is strictly less cold-cacheline traffic than refcount COW, which
had to touch the husk *and* every shared child — this design touches only the dead.

### 5.3. Read path
Take a `smart_ptr` to the current root (one `retain` on its CB) → traverse via
`smart_ref` from that root → drop the `smart_ptr` (one `release`). Holding the
root *is* the snapshot at version S (its embedded `birth`). Because GC never frees
anything reachable from a held root, traversal needs no per-node liveness checks.
A "latest" reader colliding with a writer uses the existing `cow_coordinator`
collision/notify protocol (port from `art/cow_coordinator.hpp`) so it sees a
consistent committed root.

### 5.4. 32-bit wrap (the cost of ruling #1)
4 B commits per DB lifetime before `version_t` wraps. Mitigations, in order of
preference:
1. **Rebase on GC.** When the system is quiescent or the **oldest held-root
   version** advances far, a background pass can renumber live births down toward
   0 (births are only compared, never arithmetic). Cheapest during compaction (the
   compactor already rewrites objects).
2. **Epoch high-bit.** Treat comparisons as modular within a window guaranteed
   smaller than 2^31 by bounding the span between the **oldest held-root version**
   and `current` (a snapshot is a held root, and one can't outlive that span).
   Wrap is then a non-event as long as no held root is older than 2^31 commits — a
   safe assumption.
3. Document the ceiling and revisit if a workload approaches it.

Pick **#2 as the default invariant** (modular compare with a bounded live span),
with **#1 available** as the compaction-time cleanup. Encode the bounded-span
assumption as an assert in `begin_commit` (`current − oldest_held_root_version <
2^31`).

### 5.5. Garbage collection (the held roots are the live set)
The live versions are exactly the **versions of the held roots** (§4.4) — the
top-root table plus reader `smart_ptr`s — so there is no separate alive-set to
maintain. When the GC determines an object is **death-stamped** and **no held root
covers `[birth, death)`**, it calls **`release()` once** (§4.4): the object's
baseline structural reference goes away, and SAL's release-to-zero frees it now
(if no `smart_ptr` was taken) or at the holder's final release (if one is out).
The GC is **just another reference holder** — it never inspects the count.

GC is *epoch-poked*: dropping the last `smart_ptr` to a root, or superseding a root
in the table, retires that root's version; the next scan releases (once) the
objects only that version kept alive — freed through the hierarchical bitmap (FIX)
or the compactor's release path (movable). Nothing is freed eagerly at mutation
time (per the opening analogy) — only here, and only when the object is provably
invisible to every held root.

**Persistence split.** Only **`birth` is force-persistent** (in the object header,
synced with the object — free). The CB **`ref`** and the **`death` numbers** live
in **mmap'd files that are *not* force-synced** — flushed at clean shutdown,
rebuildable after a crash (§5.6). The durable substrate is just: the persistent
**tree structure** (parent→child branches), each object's persistent **birth**,
and the set of durable **retained roots** (root table + stored-snapshot subtree
values). Every allocated object has a death number at all times — *implicit*
`ALIVE` (the array's default fill) or *explicit* `V`.

### 5.6. Death is reconstructible — no death certificates
`death` is **not an independent fact**; it is a derived property of reachability:

> `death(O)` = the smallest version `V > birth(O)` for which `O` is **not reachable**
> from `root@V`. In linear history that is *(the last version whose root reaches O) + 1*.

So the `death_ver` array is a **cache of a derivable quantity** (an mmap'd file,
not force-synced); a crash that leaves it stale loses nothing durable. Recovery
rebuilds it (and the CB `ref` counts) in **one multi-root mark pass** — the same
walk SAL already does to rebuild refcounts — or, cheaply, conservatively resets
suspect entries to `ALIVE` and lets a later GC reclaim the leaks (safe direction):
1. reset all `death_ver = ALIVE`, `ref = 0`;
2. enumerate durable retained roots ascending (root table **+ every stored-snapshot
   subtree value** — a stored snapshot *is* a persistent root), mark reachability
   from each and count per-version references;
3. object reachable from current → `death = ALIVE`; reachable from retained `v_i`
   but not the next-newer retained root → `death` = that newer root's version;
   reachable from **no** retained root → garbage, free the slot.

**Precision & safety.** The runtime stamp is *exact* (the supersession commit
version) — needed because a **volatile reader** can pin any version. Recovery can
only reconstruct death to **retained-root granularity**, which is always an
**over-approximation** (`reconstructed ≥ true`, since every retained root older
than true-death still reaches the object). Over-approximating death only *widens*
`[birth, death)` → strictly more conservative GC → **never a premature free**, at
worst a sliver reclaimed on the next sweep. The lost sub-retained precision is
moot: no snapshot survives a crash *between* retained roots, so nothing needs it.

**Why not a durable death log.** Logging a "death certificate" per supersession
would re-impose per-COW durable write amplification — the exact cost SAL 2.0
removed. The root flip is already the single durability point; death rides
downstream of it for free. **Linchpin:** every *retained* version must have a
durable, traversable root (durable snapshots satisfy this; volatile reader pins
legitimately vanish on crash, and anything kept alive only by such a pin is
genuinely garbage afterward). Because reconstruction is pure reachability, it also
handles rollback/branching (making an old snapshot current) with no special case.

---

## 6. The three regions

| Region         | File(s)                                | Addressing                             | Liveness index                               | Movable?        | Holds |
| -------------- | -------------------------------------- | -------------------------------------- | -------------------------------------------- | --------------- | --- |
| **FIX**        | `.cl1`,`.cl2`,`.cl9` + per-class `.dv` | 4B slot index → `base + slot·(N·128B)` | **per-slot** `death_ver[slot]` (1×u32/node)  | no              | inner/routing nodes (setlist→cl1, c2/double→cl2, full/node256/c8→cl9) |
| **MOVABLE**    | `.segs` + `.ctrl` + `death_ver[]`      | `ptr_address`→CB→location              | `death_ver[ptr_address]`                     | yes (compactor) | suffix+value leaves < 4KB |
| **BIG**        | `.big` (traditional alloc) + `.ctrl`   | `ptr_address`→CB→location              | `death_ver[ptr_address]` (shared w/ movable) | no              | leaves ≥ 4KB |
| commit counter | in `.meta`                             | —                                      | —                                            | —               | monotonic version; **held roots are the pins** (no registry) |

**Per-entry death overhead.** Every live entry carries one `u32` death number in a
parallel array — a sibling `.dv` mapping for FIX, the CB-indexed
`death_ver[ptr_address]` for movable + BIG. These are **mmap'd files but not
force-synced** (no fsync on commit → no write-amp; flushed at clean shutdown,
rebuilt-or-leaked on crash, §5.6), so the overhead below is the array's footprint,
not a sync cost. Because the FIX number is **per node, not per cacheline**, the
wide classes amortize it away:

| class              | entry size     | death/entry    | overhead |
| ------------------ | -------------- | -------------- | --- |
| cl1 (setlist)      | 128 B          | 4 B            | **3.1%** |
| cl2 (c2/double)    | 256 B          | 4 B            | **1.5%** |
| cl9 (full/node256) | 1152 B         | 4 B            | **0.33%** |
| movable leaf       | ≥ grain (32 B) | 4 B (CB array) | ≤ 12.5%, but only on leaves the compactor already indirects |
| BIG (≥4 KB)        | ≥ 4096 B       | 4 B (CB array) | ≤ 0.1% |

Region of a pointer is selected by its **3-bit tag** (§7), not by inspecting the
address. The size→file ceiling map (`.cl1` ≤512 GB, `.cl2` ≤1 TB, `.cl9` ≤4.5 TB,
`.segs` ≤64 TB, `.big` ≤16 TB, `.ctrl` ≤32 GB) is the target ceiling table;
actual ceilings derive from grain + offset width (§4.1).

### 6.1. FIX region = `FixedAllocator<N>` per size class
`FixedAllocator<N>` is a **complete allocator** for one size class. The
hierarchical free bitmap is only **one of its four parts** — the intra-segment
free-line bookkeeping:

1. **Backing store** — an mmap'd `.cl{N}` file of 32 MB segments, block-backed and
   grown on demand (`sal`'s `block_allocator` + `mapping`).
2. **Intra-segment free bitmap** — per-segment `start_ref/freeset.hpp::seg_freeset`
   (a lock-free hierarchical "64×64×64 bits + index"): tracks free slots *within*
   one 32 MB segment. For a fixed-size slab it collapses to **one bit per slot** —
   port `seg_freeset`'s navigation index (l0 authority + meta + top, lock-free,
   O(1) find-first-free) and drop the variable-run tiers (g2/g4/g8); a single-size
   slab never allocates a multi-slot run, so the slab allocator is just
   `find-first-free-bit` + `clear-bit-on-free`.
3. **Segment/session layer** — sal2 **inherits this from `sal`** (it is a copy of
   `sal`, §4): the movable and FIX regions share `sal`'s write-session,
   segment-provider, and sync machinery. The FIX region plugs the slab bitmap in
   where the movable region uses a bump allocator + compactor; everything around
   it (a per-session active segment, growth, and a free-**segment** index for
   O(1) "find a reusable segment") is `sal`'s. The single FIX-specific knob: a
   segment **qualifies for reuse** when its free-slot count ≥ a threshold, and a
   freed slot returns to the slab bitmap **immediately** (slots are never stranded
   — a fixed slab + version-GC makes reclaim a single `set-bit`).
4. **Per-slot death array** — a `u32` per slot in a sibling `.cl{N}.dv` mmap, **not
   force-synced** (§5.6); slot index = byte_offset / (N·128); GC frees a slot when
   no held root covers `[birth, death)`. Birth lives in the node header (the existing
   `node_header::cow_seq`/vstamp slot — repurpose as birth version).

**Size classes** (128-byte cacheline units, independent of the movable grain,
§4.1): `FixedAllocator<1>` = 128 B (setlist), `<2>` = 256 B (c2/double-setlist),
`<9>` = 1152 B (full/node256). A node rounds up to the smallest class that fits;
full nodes in their own slab never fragment the small slab and never fail to fit.
Add an intermediate class (e.g. cl4) later if round-up waste warrants. Per-slot
death overhead: 3.1% / 1.5% / 0.33% (§6 table).

### 6.2. MOVABLE region = today's SAL, evolved
Copy `sal`'s segment + compactor + session machinery wholesale. The only changes
are the four mutations (§4). The compactor's relocate/drop decision switches from
"`ref > 0`" to "death-stamped **and** no held root covers `[birth, death)`" — and
drop = the GC's single `release()` (§4.4), not a bespoke path.

### 6.3. BIG region
- ≥4KB objects, fixed, never relocated (already ≥ OS page). A **traditional
  allocator** (classic free-list / size-class malloc, boost-shmem style) over its
  own mmap'd file (`.big`) hands out the object.
- **Big objects still go through a `control_block`** that points to the object,
  and each control block carries its death number in the **same CB-indexed
  `death_ver[ptr_address]` array as the movable region**, with `zone = big` in the
  CB. BIG never relocates: `.big` is not a compacted region, and the CB's
  `zone = big` tells the MFU promoter / compactor to skip it. (So the control block +
  death array are shared by movable and BIG; only FIX is control-block-free.) The
  CB gives big objects a stable 4-byte `ptr_address` identity for parent branches
  and for recovery, without paying relocation.
- fsync must include `.big` dirty pages. Leaves that overflow the movable-region
  object cap (or that the engine knows are large) allocate here; the leaf's tag
  marks it BIG so dispatch resolves it through the CB without consulting the
  compactor's relocation path.

---

## 7. Pointer & tag redesign

Key insight: **separate the tag from the offset** so the offset stays
a power-of-2 width (aligned, SWAR-friendly), and the tag lives in its own array.

### 7.1. Per-branch storage in an inner node
```
inner node (one grain line, e.g. setlist):
  [ node_header (birth version, num_children, prefix_len, …) ]
  [ keys[n]    : u8 ]                 ← branch bytes (SWAR scan)
  [ types[n]   : 3 bits, BIT-PACKED ] ← the dispatch tag: ⌈3·n/8⌉ bytes, NOT a byte/branch
  [ offsets[n] : u32 ]               ← 4-byte, aligned, power-of-2
  [ prefix … ]
```
**The per-branch tag is a 3-bit `type` — store it bit-packed, never a `u8`.** The
type is the dispatch selector, one of 8 kinds:
```cpp
enum class node_type : uint8_t {    // 3 bits, 8 values
  full = 0, setlist, nested_setlist, double_setlist, big_nested,  // FIX (cl1/cl2/cl9)
  suffix_value,                                                   // MOVABLE
  big_suffix_value,                                              // BIG
  null                                                          // empty
};
// stored as a packed 3-bit-per-branch array: types_bytes = (3*n + 7) / 8.
//
// BRANCHLESS read — no straddle, no branch (the over-read+pad trick from the SIMD
// node scans). A 3-bit field starts at intra-byte offset 0..7, so it ends by bit
// ≤10 — always inside ONE unaligned u16 load, so it never spans two reads:
//   uint16_t w = load_u16_unaligned(&packed[(3*i) >> 3]);   // wider load covers it
//   node_type t = node_type( (w >> ((3*i) & 7)) & 0x7 );     // shift to LSBs, mask
// The load may over-read ≤1 byte past the tag array; the following offsets[] (or a
// 1-byte pad) absorbs it — same "read past n into the next in-node array"
// convention the padded SIMD scans use. No `if`, no two-byte recombine.
```
- `type` ⇒ which **region** + which **node kind**: `full`/`setlist`/`nested`/
  `double`/`big_nested` ⇒ FIX (cl1/cl2/cl9 chosen by kind); `suffix_value` ⇒
  MOVABLE; `big_suffix_value` ⇒ BIG; `null` ⇒ empty.
- **No per-branch `header`/`zones` bits.** Whether a kind is headerless-dispatchable
  (skip-CL0) and its cacheline layout/prefetch shape are **properties of the kind**
  — a small `constexpr` table keyed by the 3-bit `type` — not extra bits paid per
  branch. (For the rare wide router that needs an inline-prefix variant, the *kind*
  carries it; still 3 bits per branch.)
- A 3-bit packed array of `n` tags costs ⌈3n/8⌉ bytes vs `n` bytes — e.g. 16
  branches: **6 bytes, not 16**. Single-tag reads are the branchless u16 over-read
  above; bulk passes unpack 8 tags from 3 bytes at once (e.g. to mask "which
  branches are `suffix_value`"), still no per-tag branch.

**Full node (cl9) — the 3-bit packing is load-bearing here.** A full node routes
all 256 byte values. Packing its per-branch types at 3 bits makes the type header
`256 × 3 / 8 = `**96 bytes** — it fits in the node's **first cacheline** (≈32 B
left for `node_header` + a short common prefix). The 256 `u32` offsets are
`256 × 4 = 1024 B = 8 cachelines`. Total = **9 cachelines = the cl9 class**. Because
the offsets are a flat `u32[256]`, branch `byte`'s offset is in cacheline
`byte >> 5` — a **power-of-2 shift, not a multiply/divide by 20–28** — so dispatch
computes the target offset cacheline straight from the routing byte and
**prefetches it before reading the 96-byte type header**, hiding the offset load
behind the header read + type decode. A byte-per-tag header (256 B = 2 cachelines)
would push the node to 10 cachelines, leave no room for a prefix, and break the
clean power-of-2 offset indexing — so the 3-bit packing is what *makes* the full
node a 9-cacheline, prefetchable design.

### 7.2. dynamic_dispatch
```cpp
// next_byte: the key byte we're routing on; type: the unpacked 3-bit tag; off: u32 offset
inline void* dispatch(uint8_t next, node_type type, uint32_t off) {
  if (type < node_type::suffix_value) {       // FIX region: direct, no CB
    // off is a SLOT index; stride is the class slot size (128/256/1152 B —
    // shift for the pow2 classes cl1/cl2, multiply for cl9).
    void* p = fix_base[region] + size_t(off) * fix_slot_bytes[region];
    switch (type) {
      case full:          // prefetch branch `next`'s offset cacheline (next>>5, a
                          // shift) BEFORE reading the 96-byte 3-bit type header —
                          // the offset load hides behind the header read (§7.1)
                          prefetch(p + 128 + ((next >> 5) << 7)); return as<full_node>(p);
      case setlist:       return as<setlist>(p);
      case nested_setlist:return as<nested_setlist>(p);
      case double_setlist:prefetch_2(p);       return as<double_setlist>(p);
      case big_nested:    return as<prefix>(p);
    }
  } else if (type == suffix_value) {           // MOVABLE: through control block
    return as<prefix>(sal_resolve(ptr_address(off)));
  } else if (type == big_suffix_value) {       // BIG
    return as<prefix>(big_resolve(ptr_address(off)));
  } else { /* null */ return nullptr; }
}
```
This makes the routing-layer hop a **single** cold line (the child) instead of
two (CB then child), for the majority of descent.

### 7.3. Movable/BIG offsets are still `ptr_address`
For movable and BIG, the 4-byte `offset` is a `ptr_address` (control-block
index), preserving relocation. For FIX, it's a direct line index. The tag tells
dispatch which interpretation applies — so the same 4-byte column carries both,
and alloc-hints from a mixed column are filtered by tag (ignore out-of-range
hints; false hints cost only locality, never correctness).

---

## 8. FixedAllocator<N> requirements

The acceptance contract for `FixedAllocator<N>` — the **whole** allocator (§6.1's
four parts), not just the bitmap. Reqs 1–3 are the **intra-segment free bitmap**
(port `start_ref/freeset.hpp::seg_freeset`); reqs 4–7 are the **segment/session
layer**, inherited from `sal`'s session + segment-provider machinery (sal2 is a
copy of `sal`, §4):
1. Thread-safe + lock-free (CAS/fetch_or per op).
2. O(1) alloc and free (probe-counted, not wall-clocked — port
   `g_freeset_probes`).
3. First-fit-forward slab allocation: a per-session cursor hands out the
   first free slot at-or-after it; freed slots become reusable immediately (no
   stranding). For a single-size slab this is just find-first-free-bit.
4. Write sessions handed out like SAL alloc sessions; each tracks current 32 MB
   segment, next-alloc cursor, and touched pages for fsync.
5. Each 32 MB segment owned by one session until full, then claim another that
   qualifies (free-slot count ≥ threshold).
6. Per-segment atomic free-slot counter; crossing the threshold marks the segment
   reusable in the free-segment index.
7. On exhaustion, adopt the lowest/leftmost segment meeting the 50% criterion.

**Robustness gate (Phase 1 of the build plan):** a standalone fuzz that hammers each
`FixedAllocator<1|2|9>` with random alloc/free interleavings and verifies: no two
live allocations overlap, every freed run is fully reclaimable, free-count
invariant (`alloc_pos == freed + live`) holds, and probe counts stay O(1).

---

## 9. Sessions, sync, and fsync amplification

The cost to manage: splitting writes across FIX/MOVABLE/BIG amplifies fsync if
naive. Mitigations build on `sal`'s existing session sync:
- A **write session** tracks, per region, the active segment and a **touched-page
  set** since last commit (a per-segment dirty-page bitmap). Commit syncs **only
  dirty pages**, merging runs separated by ≤ `msync_gap_pages` clean pages into
  one syscall.
- Commit order (durability contract): sync touched data pages of all regions →
  fsync data files (if level ≥ fsync) → flip durable root in `.meta` →
  fsync `.meta`. The root flip is the linearization point; a crash before it
  leaves the previous root authoritative.
- `sync_type` knob identical to SAL (none/msync_async/msync/fsync/F_FULLFSYNC).
  `none` makes multi-region a non-issue.
- Measure write-amplification per region and report it in the bench; the
  acceptance bar is "multi-region fsync cost within a small constant of
  single-region for the same logical commit."

---

## 10. Recovery across three regions

Port SAL's recovery and extend:
- Each region's segment headers carry a geometry stamp (grain, segment size,
  offset width, version width) — reopening with a mismatch throws.
- On **clean** shutdown the `death`/`ref` mmaps are flushed, so reopen reuses them
  as-is — no reconstruction.
- On **unclean** shutdown (crash) those not-force-synced mmaps may be stale:
  validate per-page checksums, truncate torn tails per region, then **rebuild the
  derived liveness in one multi-root mark pass** (§5.6) — the same walk SAL already
  does in `recover()`/`reset_reference_counts()`. Death numbers, CB `ref` counts,
  free bitmaps (FIX), and control blocks (movable/BIG) are **all** rebuilt by it;
  only `birth` (in object headers) and the tree structure are read from disk. **No
  death certificates are logged** — death is derived from reachability. (A cheaper
  alternative resets suspect entries to `ALIVE` and lets a later GC reclaim the
  resulting leaks — safe, since a lost death only delays reclaim, never frees a
  live object.)
- The pass: reset `death_ver = ALIVE` and `ref = 0`; enumerate durable retained
  roots ascending (root table **+ stored-snapshot subtree values**); mark
  reachability, setting each reachable object's baseline `ref = 1` (+1 per
  structural reference, as SAL's recovery counts); set `death` to the next-newer
  retained root that stops reaching each object (`ALIVE` if reachable from
  current). `current = max birth seen + 1`. Volatile **reader** `smart_ptr` pins do
  not survive a restart, so any object reachable from **no** retained root is
  reclaimed immediately. Reconstructed death is a safe over-approximation of true
  death (§5.6) — never premature-free.

**Acceptance:** kill -9 mid-commit across a workload, reopen, verify the tree
equals the last durably-committed state (oracle replay), no leaked lines, no
double-allocated lines, all three region files internally consistent.

---

## 11. Build plan — phased, each phase independently landable + gated

> Cadence: land one phase per branch checkpoint, run its gate, archive logs
> (`logs/YYYY-MM-DD_sal2-<phase>/` per the bench-log convention), then proceed.
> Pause for review after Phase 1 (the liveness rewrite is the riskiest change)
> and after Phase 3 (first end-to-end).
>
> **Working-tree caution:** the `sal2` branch currently carries unrelated
> in-flight edits (`start_ref/`, `sal_art_ref/`, `delta_art/`, `psitrimdbx/`,
> `benchmarks/`). Do **not** revert or sweep them into sal2 commits; stage only
> sal2 files.

### Phase 0 — Bootstrap the copy
- `cp -r libraries/sal libraries/sal2`; rename namespace `sal`→`sal2`, include
  guards, and CMake target (`sal2`, `sal2-tests`). Copy the sal test sources →
  sal2 tests. Add `add_subdirectory(libraries/sal2)` to the top-level
  `CMakeLists.txt` (after `libraries/sal`).
- **Gate:** `sal2-tests` pass identically to the sal tests (a faithful copy). No
  behavior change yet.

### Phase 1 — The four mutations (version core) ⟵ review checkpoint
- 1a configurable grain (§4.1); CTest matrix grain∈{32,64,128}.
- 1b 32-bit object version (§4.2).
- 1c remove CB refcount; add `death_ver[]` parallel array; rewrite
  retain/release/compactor-liveness as `[birth,death)` (§4.3).
- 1d commit counter + implicit root-pinning (held roots = live versions; **no
  registry**); GC reclaim = one `release()` on death-stamped + uncovered objects (§4.4).
- **Gate:** version-visibility unit tests; snapshot-isolation test; compactor
  relocate/drop-by-version test; 100 M-op alloc/COW/snapshot/release fuzz vs
  oracle, ASAN + TSAN clean; `sal2-tests` green at all three grains.

### Phase 2 — Pointer/tag + FIX region (FixedAllocator)
- 2a port `start_ref/freeset.hpp::seg_freeset` as `FixedAllocator<N>`'s
  **intra-segment free bitmap** (grain-parameterized, one-bit-per-slot); standalone
  random alloc/free robustness fuzz (§8 reqs 1-3).
- 2b the rest of `FixedAllocator<N>` (the wrapper around the bitmap, §6.1 parts
  1/3/4): `.cl{N}` file + 32 MB segment backing, write sessions (active segment +
  next-alloc cursor + touched-page set), and threshold-based cross-segment reuse —
  all via `sal`'s inherited session/segment-provider machinery (§8 reqs 4-7) —
  plus the per-slot `.cl{N}.dv` death array.
- 2c 3-bit-packed `type` tags + multi-region `dispatch` skeleton + `resolve` per region
  (§7); round-trip tests.
- **Gate:** FixedAllocator fuzz (no overlap/leak, O(1) probes); dispatch
  round-trips for all tag types; FIX-region durability (touched-page msync
  correctness).

### Phase 3 — `gcart`: the `adb`-conforming engine (FIX inner nodes + MOVABLE leaves) ⟵ review checkpoint
- Build **`libraries/gcart/`** — a **real `adb` engine** (§3.2), not a test stub,
  depending on `sal2`. Start from the node taxonomy in `art` (setlist + node256)
  but wire allocation so inner nodes land in FIX (direct line index) and
  suffix+value leaves land in MOVABLE SAL (control block); version-stamped COW
  (birth on new, death on husk). Model the public surface on `start_ref::kv` /
  `start_ver::kv` (copy their adapter *shape*, not their allocator) so the concept
  passes with minimal new surface. **`sal2` and all existing engines stay
  untouched.**
- **Mutation uses the `upsert<cow_mode = shared>` COW mode latch (§5.2)** — a scoped
  enum (not bool), birth-version unique/shared check on descent, latching to shared
  on the first shared node.
  This is correctness-critical (the snapshot-isolation gate must catch its
  absence) and the only safe basis for in-place mutation.
- Provide the traits bundle `gcart::engine` (`database/write_session/
  read_session/transaction/cursor/config` + `database::open`) for the
  `kv_engine` form.
- **Gate (conformance + correctness):**
  - `static_assert(adb::kv_database<gcart::kv::database>)` and the extension
    concepts it implements; add a `concept_check`-style TU for `gcart`.
  - Add `gcart` as the **5th engine** in `tests/kv_compat_test.cpp`; it passes the
    identical templated battery (point CRUD, ordered cursor forward/back,
    lower/upper bound, snapshot isolation, and the `if constexpr` typed_values /
    multi_root / snapshot batteries for implemented features) — cross-checked
    against `std::map` and **the other engines as differential oracles** (§3.2).
  - **Drop-in proof:** instantiate one `kv_engine`-templated consumer
    (`delta_art` or a minimal `kv_engine` smoke harness) on `gcart::engine`
    and run it green.
  - Scale 1 M & 10 M with the inner/leaf region split; MVCC reads under
    concurrent writers.
  - **Headline A/B:** cold-cacheline count + ns/op on the routing path vs
    `start_ref`/`start_ver`/`sal_art_ref` (fewer cold lines per descent — SAL 2.0's
    core promise, §0).

### Phase 4 — BIG region
- Size-class free-list allocator over `.big`; `zone = big` in the CB (the MFU
  promoter / compactor skip it) and `.big` is not a compacted region, so big
  objects never relocate; leaves ≥4KB route here; fsync includes `.big`.
- **Gate:** large-value (4KB…MBs) round-trip + recovery; overflow-from-movable
  routing correctness; sync correctness across all three regions.

### Phase 5 — Hardening + bench
- Crash-fuzz / power-loss recovery across 3 regions (§10 gate).
- Write-amplification measurement per region (§9); 32-bit-wrap invariant assert
  exercised; GC epoch timing test.
- Headline benchmark vs existing engines, logs archived per convention; update
  this doc's "Results" section and the reorg plan if sal2 graduates.

---

## 12. Acceptance testing plan (global)

| Gate                    | What it proves             | How |
| ----------------------- | -------------------------- | --- |
| **Copy fidelity**       | sal2 ≡ sal at Phase 0      | sal2 tests mirror the sal tests, green. |
| **Grain matrix**        | grain is a real knob       | CTest builds/tests at 32/64/128; cross-grain reopen throws. |
| **Version visibility**  | `[birth,death)` is correct | unit: visible iff `B≤S<D`; GC-eligible iff no held-root version in `[B,D)`. |
| **Snapshot isolation**  | MVCC correctness           | reader@V sees exactly keys committed ≤V under concurrent writers. |
| **Liveness fuzz**       | no UAF / no premature free | 100M random alloc/COW/snapshot/release vs oracle; ASAN+TSAN. |
| **FixedAllocator fuzz** | allocator robust           | random alloc/free: no overlap, full reclaim, free-count invariant, O(1) probes. |
| **Dispatch round-trip** | pointer/tag correct        | every tag type resolves to the right region/object. |
| **adb conformance**     | engine is a drop-in        | `static_assert(adb::kv_database<gcart::kv::database>)` + extensions; `gcart` is the 5th engine in `tests/kv_compat_test.cpp`, passes the templated battery vs std::map + the other engines as oracles; one `kv_engine`-templated consumer runs green on `gcart::engine`. |
| **End-to-end kv**       | engine works               | behavioral battery vs std::map; 1M/10M scale. |
| **Crash/recovery**      | durability                 | kill -9 mid-commit × N, reopen, equals last durable commit; no leaks/double-allocs across regions. |
| **Headline A/B**        | the design pays off        | cold-cacheline count + ns/op on routing path beats single-region engines; write-amp bounded. |

**Test-authoring rule.** Tests live in `tests/`, written against `adb` and
instantiated per engine. **Author every new behavioral test in `adb` form**
(templated on `Engine::database`) so it runs on `gcart` *and* the existing
engines; when something fails, the existing engines (`start_ref`, `start_ver`,
`sal_art_ref`, `psitri`) are **differential oracles** that localize the bug, used
alongside `std::map`. Reserve `gcart`-specific or `sal2`-specific tests for
mechanics with no `adb` surface — the allocator (FixedAllocator fuzz), the
version/liveness model, region dispatch, and crash/recovery internals. Reuse the
`ctest` targets and the bench-log archive convention; every scale/bench run writes
`logs/YYYY-MM-DD_sal2-<thing>/` with a conditions README (cmd, env, git sha,
machine) + verbatim logs + an index line in `logs/README.md`.

---

## 13. Risks & mitigations

| Risk                                                                                  | Severity  | Mitigation |
| ------------------------------------------------------------------------------------- | --------- | --- |
| Liveness rewrite (refcount→version) introduces UAF                                    | high      | Phase-1 review checkpoint; 100M-op oracle fuzz under ASAN/TSAN before any engine wiring. |
| fsync amplification from 3 regions                                                    | med       | per-region touched-page sets, dirty-run merging, single root-flip linearization; measure write_amp as a gate. |
| 32-bit version wrap                                                                   | med       | bounded-live-span modular compare (§5.4 #2) + compaction-time rebase (#1); assert `current − oldest-held-root-version < 2^31`. |
| 9-grain full nodes fragment the small region                                          | med       | dedicated `FixedAllocator<9>` region (`.cl9`). |
| 4 B object cap (every leaf still needs a CB in movable region)                        | low/known | acknowledged; leaf-bucketing (group multiple pairs behind one CB) raises it later — out of v1 scope, noted. |
| `death_ver` array cold during GC scans                                                | low       | dense u32 arrays scanned block-at-a-time by the epoch GC; not on the read path. |
| Grain change is an on-disk format break                                               | low       | geometry stamp + reopen guard; default 32, knob for benches. |
| Scope creep (3 regions at once)                                                       | med       | strict phase gates; Phases 1-3 deliver the core win; BIG (4) and hardening (5) follow. |
| `adb` conformance surface is large (typed values, multi-root, snapshots, full cursor) | med       | copy the adapter *shape* from `start_ref::kv`/`start_ver::kv` (proven conformant); gate features with the `if constexpr` batteries so partial support still passes the core concept; land core CRUD+cursor first, then extensions. |

---

## 14. Autonomous execution protocol

- **Repo/branch:** all work in `~/gofractally/psitri` on the `sal2` branch.
  Commit per phase with the gate result in the message; **stage only sal2 files**
  (the branch carries unrelated WIP — never `git add -A`); do not push or open a
  PR unless asked.
- **Build:** the repo uses `build/` and `build-debug/`. Add `sal2` / `sal2-tests`
  targets and build them there; do not reconfigure existing build dirs
  destructively.
- **Test:** run the sal2/gcart test binaries directly and via `ctest`; ASAN/TSAN
  builds for the liveness fuzz. Run the existing `kv-compat-test` target once
  `gcart` is added as an engine — author new behavioral tests in `adb` form so
  they run on every engine and use the others as oracles (§12 test-authoring
  rule).
- **Don't touch** `libraries/{sal,art,start_ref,start_ver,sal_art_ref,delta_art}`,
  `adb`, `ucc`, or `psio` except to *read* — they are untouched comparators /
  oracles / dependencies (HARD RULE: no sal/ucc changes without permission). The
  only new code is the two new libraries: `libraries/sal2/` (copy of `sal`,
  evolved) and `libraries/gcart/` (engine on sal2), plus adding `gcart` to
  `tests/kv_compat_test.cpp`.
- **Verify compiler assumptions** (HARD RULE): any claim of inlining/folding for
  the dispatch hot path must be backed by `nm`/`otool` disassembly, not asserted.
- **No heap allocations in hot paths** (HARD RULE): SAL alloc + stack/thread-local
  scratch only.
- **Pack to the data's actual range** (HARD RULE): use bitfields sized to the real
  value range — never a whole byte for a flag, never a field wider than its domain.
  Examples already in this doc: the CB's `zone:2` / `ref:16`; the per-branch tag is
  a **3-bit `type`, bit-packed at ⌈3n/8⌉ bytes** (not a `u8`/branch, §7.1);
  `alloc_header`'s `size`/`type` are bit-packed. Size every field from its range and
  **`static_assert` every on-disk/control struct size** so a layout drift is a
  compile error.
- **Checkpoints:** pause for review after Phase 1 and Phase 3 (the two riskiest
  boundaries); otherwise proceed phase to phase, archiving logs each time.
- **When blocked on a genuine fork** (e.g. a taxonomy choice for the richer FIX
  node variants deferred from v1, §15), ask; otherwise pick the documented default
  and proceed.

---

## 15. Open decisions / deferred (not in v1)

- **Richer FIX node taxonomy.** Candidate future variants that pack several
  sparse sub-nodes into one cacheline (nested / double / range setlists) can fill
  the cl2 class. v1 uses the proven `art` taxonomy — setlist + node256 (cl1 + cl9),
  with cl2 reserved — and adds variants once the region plumbing is proven.
- **Leaf bucketing** to exceed the 4 B-object control-block cap (group multiple
  key/value pairs behind one control block).
- **`ref` field width / externalizing it.** The CB keeps a `ref` for `smart_ptr`
  pins (§4.4); since COW never touches it and pins are rare (mostly on roots), a
  narrow field suffices. Externalizing `ref` to a parallel array (like `death_ver`)
  to widen `offset` is a possible later optimization; deferred.
- **`adb` extension coverage.** v1 must pass the **core** concept
  (`adb::kv_database`: CRUD + ordered cursor + snapshot isolation). The locked
  extensions — `typed_values` (POD take/make), `multi_root_transaction`,
  `tree_snapshots` — are gated by `if constexpr` in the battery, so `gcart` can
  land core-first and add each extension in a follow-up while still being a valid
  drop-in for libraries that use only the core. Target all three by end of
  Phase 5; treat any not-yet-implemented extension as an explicit, logged gap
  (no silent caps).
- **walart/dwal** layering on `gcart` — beyond the allocator+engine; but because
  `gcart` satisfies `kv_engine`, `dwal`/`delta_art` can template on it with no
  changes when that work is scheduled.

---

### Appendix A — file/type cross-reference (where each piece comes from)

| piece                                                   | lib   | copy/port from                                                                          | notes |
| ------------------------------------------------------- | ----- | --------------------------------------------------------------------------------------- | --- |
| segments, mmap, sessions, compactor, sync, recovery     | sal2  | `libraries/sal/*`                                                                       | copy wholesale, namespace rename |
| `control_block` (no ref), `death_ver[]`                 | sal2  | `sal/control_block.hpp` + new                                                           | §4.3 |
| commit counter + implicit root pinning                  | sal2  | version counter in `.meta`; held roots (table + `smart_ptr`) are the pins               | §4.4 (no registry) |
| `FixedAllocator<N>` intra-segment bitmap (1 of 4 parts) | sal2  | `start_ref/freeset.hpp::seg_freeset`                                                    | port + grain-parameterize, §6.1/§8 |
| segment/session layer (FIX + movable)                   | sal2  | `sal` session + segment-provider + sync (inherited via the `sal` copy)                  | per-session active segment, free-segment index, touched-page sync; §6.1 part 3 / §8 reqs 4-7 |
| 3-bit packed `type` tags + `dispatch`                   | gcart | `art`/`start_ver` `tagged_ptr`/`cnode` + §7                                             | tag/offset separation; ⌈3n/8⌉ B/node |
| node taxonomy (setlist+node256)                         | gcart | `art/include/art/{node,art_map,cow_coordinator}.hpp`                                    | starting point for the FIX-region routers |
| `gcart::kv` engine adapter *shape*                      | gcart | `start_ref/kv.hpp`, `start_ver/kv.hpp`, `sal_art_ref/test/heart_kv.hpp`                 | copy the public API shape (proven `adb`-conformant), back it with the sal2 allocator |
| `adb` concepts to satisfy                               | gcart | `libraries/adb/include/adb/concepts.hpp` (namespace `adb`), `psitri/dwal/kv_engine.hpp` | `static_assert` both |
| conformance harness + oracles                           | tests | `tests/kv_compat_test.cpp`, `libraries/adb/tests/concept_check.cpp`                     | add `gcart` as the 5th engine; other engines are differential oracles |
| bench/log conventions                                   | —     | `logs/` archive convention                                                              | reuse as-is |

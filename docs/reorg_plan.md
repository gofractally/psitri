# Reorg plan — psitri-merge → @gofractally/psitri (+ siblings)

Refines `~/myartlib/ThePlan.md`. Revision 2 — Dan's annotations of 2026-06-12
are folded in as rulings (§6 records them). This file is the working copy for
the psitri portion: target tree, rename map, review findings, the
carried-forward TODO ledger mapped to its new homes, and migration mechanics.
Mechanics are copy-paste + find-replace renames — NO rewrites.

## 0. Sibling repos

- **@gofractally/libart++ — substantially DONE.** The repo exists
  (`~/gofractally/libart++`): hkv → `artpp::map`, header-only, optional psio
  key codec (`artpp/psio_codec.hpp`, ARTPP_WITH_PSIO), bench suite vs
  absl::btree / std::map / upstream libart, GitHub Pages site with charts.
  Remaining: one-time bug-class diff sweep vs the part/vpart hkv descendants,
  standardized third-party comparison suites if any exist, publish cadence.
- **@gofractally/ucc — NEW repo (Dan).** ucc extracted via the composition
  process (psio dependency-pattern: FetchContent + FETCHCONTENT_SOURCE_DIR_UCC
  co-dev override), **hash merged into it** (headers move to `ucc/hash/…`,
  the separate `hash` target dies). Header-only + its test suite. Consumed by
  psitri as vendor/ucc exactly like vendor/psio.
- **@gofractally/psitri-multiindex** — exists at `~/psitri-multiindex`; move
  under `~/gofractally/`, port to the adb transaction concept, audit
  heterogeneous lookup (literal / string_view queries must not materialize
  temporary std::string). Consumes adb from the psitri repo (ruling A).

## 1. Target tree for @gofractally/psitri

```
libraries/adb          abstract database concepts (today: part/db_shape.hpp
                       + the kv_engine concept) — namespace adb::
                       independently consumable subdir (psio dependency-
                       pattern) so multiindex and others can FetchContent it
libraries/sal          smart alloc — unchanged
libraries/start_ref    static ART, refcount liveness   (today: part)
libraries/start_ver    static ART, version/stamp       (today: vpart)
libraries/sal_art_ref  ART on sal                      (today: heart)
libraries/psitri       legacy sal btree — unchanged name
libraries/delta_art    two adb engines composed: hot delta tree, batch
                       merge into materialized base, adb on the surface
                       (today: tiered) [Dan: "tiered db becomes delta_art"]
libraries/walart       WAL + art_map write-set engine — REMAINS for now,
                       as does dwal (rides with sal_art_ref today); both
                       pending a use-case review (later decision); dwal can
                       exist and use adb
shims/psi-sqlite       sqlite btree.c replacement      (today: psitri-sqlite)
shims/psi-duckdb       (today: psitri-duckdb)
shims/psi-rocksdb      (today: psitrirocks)
shims/psi-leveldb      (today: psitrileveldb)
shims/psi-mdbx         (today: psitrimdbx)
                       shims are FIRST-CLASS PRODUCTS: we will eventually
                       mimic every 3rd-party API over our backend — the
                       onboarding path for existing code bases. ALL move.
vendor/psio            pssz internal representation (FetchContent wired;
                       co-dev override; SOURCE_DIR external/ → vendor/)
vendor/ucc             @gofractally/ucc (same pattern; hash now inside)
tests/                 the behavioral battery, written against adb,
                       instantiated per engine (kv_compat + crash/recovery +
                       scale gates + quick_gate). TEST-SHIMS — wrappers that
                       make 3rd-party DBs (mdbx, rocksdb, libart, tidesdb…)
                       satisfy the battery/bench drivers — live HERE; they
                       are not products.
benchmarks/            against adb, per-engine instantiation: bank, dict,
                       random/scale, pipeline, TATP; drives the test-shims
                       for third-party comparison rows
docs/                  design records (vpart_version_lifecycle_audit,
                       unified_tier_design, pdwal_design, node_design_patterns,
                       this file, BUGS.md convention)
logs/                  bench archive convention carries over verbatim
scripts/               run_tests.sh, quick_gate.sh, watchdog, bench matrices
```

## 2. Rename map (find-replace pass)

Namespace rule (Dan): **namespace == file structure**; private implementation
details go in a nested `detail{}` namespace (e.g. `vhkv::` →
`start_ver::detail`).

| today | new path | namespace today → new |
|---|---|---|
| libraries/part | libraries/start_ref | `part::` → `start_ref::` |
| libraries/vpart | libraries/start_ver | `vpart::`/`vhkv::` → `start_ver::` / `start_ver::detail` |
| libraries/heart | libraries/sal_art_ref | `heart::` → `sal_art_ref::` |
| libraries/tiered | libraries/delta_art | `tiered::` → `delta_art::` |
| part/db_shape.hpp | libraries/adb | `part::shape::` → `adb::` |
| libraries/psitri-sqlite, -duckdb, psitrirocks, psitrileveldb, psitrimdbx | shims/psi-* | unchanged |
| libraries/ucc + libraries/hash | @gofractally/ucc repo → vendor/ucc | `<hash/x.h>` → `<ucc/hash/x.h>`; `hash` target dies |
| libraries/mdbxrocks, third_party/libart, tidesdb wrapper | tests/ (test-shims) | comparators, not products |
| libraries/art (our art_map — psitri dwal dependency) | libraries/art | unchanged (plan correction) |
| programs/kv_compat_test.cpp | tests/ | — |
| programs/*_bench* | benchmarks/ | — |
| external/psio | vendor/psio | — |

## 3. Review findings (resolved where annotated)

1. **ucc + hash were unlisted.** RULING: ucc becomes its own repo via the
   composition process; hash merges into it; psitri consumes vendor/ucc.
2. **adb cross-repo consumption.** RULING (A): in-repo subdir, designed
   independently consumable (psio dependency-pattern).
3. **Shims vs comparators.** RULING: psi-* foreign-API faces are first-class
   onboarding products (eventually every 3rd-party API). Wrappers that exist
   to run other DBs through OUR tests/benches are test-shims under tests/ —
   not products.
4. **Shims currently bind the legacy psitri engine** (sqlite's DWAL backend).
   Future: bind adb so delta_art can back sqlite. Not a move blocker.
5. **Three permanent hkv descendants** — artpp (heap, single-thread),
   start_ref (arena, refcount), start_ver (arena, stamps). Intentional
   divergence; one-time bug-class diff sweep at extraction, then stop
   syncing.
6. **Test split rule**: behavioral battery + concept phases + crash/recovery
   + census + scale gates → tests/ (adb-typed). Engine-INTERNAL tests
   (freeset, arena, node layout, TSan, private-invariant fuzz) stay in
   libraries/X/tests.
7. **Repo/library name shadow** (repo psitri ⊃ libraries/psitri): accepted.
8. **CI from day one**: clang/gcc matrix, Debug+asserts, TSan job, the
   battery as merge gate.
9. **walart vs wal_art collision.** RULING (C): tiered → delta_art; walart
   keeps its name and remains for now alongside dwal; whether they still
   serve a use case is a LATER decision (walart ≈ dwal's role; dwal can be
   made to use adb).
10. **adb concept incomplete** — land the §4 concept items (schema
    persistence API, psio storable adapter, owned-tree values) as part of
    the move, not after.

## 4. Carried-forward TODO ledger (mapped to new homes)

**libraries/adb**
- [x] storable_type / typed_values / any_value / three-outcome reads (LOCKED,
      §9–§12 of docs/vpart_version_lifecycle_audit.md)
- [x] multi_root_transaction + root_handle/set_root exchange
- [x] tree_snapshots (snapshot(root) = O(1) owned shared tree)
- [ ] schema persistence (§14): register_value_types persists each type's
      canonical psio schema (psch); on load, stored typenums bind to runtime
      callbacks by canonical-schema COMPARISON (id drift detected/remapped,
      unknown stored types fail loudly, pssz evolution rules apply)
- [ ] psio pssz storable adapter: PSIO_REFLECT'd types as the default
      storable shape; zero-copy reads via view<T> with the engine read lock
      as the Store policy; strings/vectors stay null-hook builtins
- [ ] owned-tree values: storing a tree_root/subtree AS a typed value
      (the smart_ptr storable — retain/release = root refcount)
- [ ] concept: cursor typed surface (value_type()/is_type<T>()) — start_ver
      has it; promote to the concept when a second engine lands it

**libraries/start_ver — first GREEN engine (1d97f950)**
- [x] full kv-compat PASS: 152219 checks, typed values + multi-root +
      snapshot + exact drain census; unit suite green; bank 676k tx/s
- [ ] GC/writer-isolation rework: O(1) bucketed claim_ (today O(nseg) scans
      poison once the arena grows), bigger ring adoption, drain throughput
      vs batch-1000 production. GATE: 150M no-flush writer isolation
      (today degrades 164ns → 5.3µs by ~46M)

**libraries/start_ref, sal_art_ref, psitri — typed values + multi-root + snapshot**
- [x] start_ref typed values: refcount-liveness hooks; typed-values phase GREEN
- [x] sal_art_ref typed values: sal-type-ops leaf_page lifecycle + engine-side
      global registry (no sal change); typed-values phase GREEN
- [ ] psitri typed values: SAME envelope+registry pattern, but added directly
      to psitri's native transaction/cursor (no kv adapter) and fired through
      the tree_ops/tree_context COW (leaf destroy = release; the inline-value
      on_copy site inside the leaf-clone path; in-place overwrite/remove =
      release). Largest of the three — legacy COW + the open shared_smart_ptr
      SIGSEGV (docs/BUGS.md). Its own focused pass.
- [x] sal_art_ref multi-root + snapshot: native tree_handle::set_root +
      heart_kv primary/open_root/set_root; get_root IS the frozen snapshot
      (Dan's 'a tree is a snapshot' — dropped the redundant snapshot()/
      owned_tree). sal_art_ref now PASSES the FULL battery.
- [x] adb cleanup: tree IS a snapshot — one owned-tree type per engine;
      get_root the snapshot primitive; tree_snapshots requires get_root.
      start_ver get_root now returns the frozen tree_root (was a live handle).
- [ ] start_ref multi-root + set_root + frozen get_root (typed values done;
      still 2 battery failures)
- [ ] psitri multi-root + set_root + typed values + invalid-type-on-wrong-pod
      (2 battery failures)

**benchmarks/ (start_ver + sal_art_ref composition)**
- [x] delta_art-scale-bench: delta_art<start_ver hot, sal_art_ref dwal cold>
      benching (~270ns ins / ~128ns rd, rotation drain active)
- [x] walart-scale-bench: walart (sal_art_ref base + start_ref delta) benching
      (~360ns ins / ~205ns rd)
- [ ] remaining bench migration: bank (multi-engine), dict, pipeline, TATP +
      comparator test-shims

**libraries/delta_art (tiered) — #32 DONE (concept-only)**
- [x] concept-only rotation via multi-root set_root (h1.set_root(h0.set_root
      ({}))); warm read via get_root(1) frozen snapshot; clear via set_root
      ({}); crash recovery = re-drain root 1 on open. ALL W contraband shims
      dropped from delta_art's requirements (freeze_and_reset/clear_warm_slots
      /pin_at/pending_warm/committed_version/make_orphan_root/footprint).
      20M-key forced-flush stress = 100% hit through every rotation.
- [ ] still TODO: DELETE the now-unused shims from start_ver's public surface
      (delta_art no longer calls them); add the sal_art_ref/sal_art_ref +
      start_ref/sal_art_ref instantiations to the test (need start_ref
      multi-root first)
- [ ] snapshot migration (§8 identity model): ttree = {hot ref, cold ref |
      resolved cold ref}; persistent lock-free bitset registry; merger swaps
      identities — never copy-constructs a tree; ordered boundary
      materialization; subtree values opaque to the hot engine
- [ ] 1B scale run vs sal_art_ref baseline (#28) — blocked on start_ver GC

**tests/**
- [x] battery semantics locked: take_pod counts +1 (custody enters the pod);
      quiescent census = no live sessions (session fcache holds claimed lines
      legitimately)
- [ ] schema-persistence battery case: register, store, reopen with permuted
      type ids, read back by schema match
- [ ] crash-fuzz gate for walart (was still gating when parked)

**other carried items**
- [ ] walart + dwal use-case review (Dan: later decision); dwal → adb if kept
- [ ] dwal btree_layer compact stored values + stale-view audit (#20)
- [ ] sal_art_ref point-op gap, sample-driven (#22, 1.5–3.4x vs part family)
- [ ] psitri-multiindex: port to adb concept + heterogeneous lookup audit

## 5. Migration mechanics

1. Freeze: tag psitri-merge (`reorg-baseline`); psitri-merge becomes the
   read-only archive — branches, logs/, history stay here.
2. Extract @gofractally/ucc first (ucc + hash merged, standalone tests green).
3. Scaffold @gofractally/psitri: target tree, root CMake, CI, vendor/psio +
   vendor/ucc.
4. Move in dependency order, ONE library per commit, full battery + that
   library's unit suite green before the next: sal → adb → start_ref →
   start_ver → sal_art_ref → psitri → delta_art/walart → shims →
   tests/benchmarks last (they compile against everything).
5. Renames are mechanical find-replace per §2 table; no logic edits ride
   along. Anything broken by a rename gets fixed in the SAME commit.
6. Does not move: build dirs, scratch testdbs, dead experiments living on
   other branches (wtree2, big_index, fast_index v4 stay archived in their
   repos/branches).
7. After the move: psitri-merge gets a README pointer to the new repo and
   the path-mapping table.

## 6. Rulings record (Dan, 2026-06-12)

A. adb: in-repo subdir, independently consumable.
B. Namespaces match the file structure; private impl details in `detail{}`.
C. tiered → **delta_art**. walart keeps its name and remains for now, along
   with dwal — use-case review is a later decision; dwal can use adb.
D. Shims: keep them ALL — first-class onboarding products. Test-shims
   (3rd-party DB wrappers for our tests/benches) live under tests/.
E. ucc: own repo via the composition process; hash merged into it.

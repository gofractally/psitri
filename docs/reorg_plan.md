# Reorg plan — psitri-merge → @gofractally/psitri (+ siblings)

Refines `~/myartlib/ThePlan.md` (Dan, 2026-06). This file is the working copy
for the psitri portion: target tree, rename map, gaps found during review,
the carried-forward TODO ledger mapped to its new homes, and migration
mechanics. Mechanics are copy-paste + find-replace renames — NO rewrites.

## 0. Status of the sibling plan items

- **@gofractally/libart++ — substantially DONE.** The repo exists
  (`~/gofractally/libart++`): hkv → `artpp::map`, header-only, optional psio
  key codec (`artpp/psio_codec.hpp`, ARTPP_WITH_PSIO), bench suite vs
  absl::btree / std::map / upstream libart, GitHub Pages site with charts.
  Remaining: one-time fix-sync sweep vs the part/vpart hkv descendants
  (bug-class diffs only — e.g. growable insert paths; part/artpp lack vpart's
  recorded-descent so the `path_[72]` smash class does not apply to them,
  verified by the 1500-byte big-key battery), standardized third-party
  comparison suites if any exist, publish cadence.
- **@gofractally/psitri-multiindex** — exists at `~/psitri-multiindex`;
  move under `~/gofractally/`, port to the adb transaction concept, audit
  heterogeneous lookup (literal / string_view queries must not materialize
  temporary std::string). Depends on where adb lands (decision A).

## 1. Target tree for @gofractally/psitri

```
libraries/adb          abstract database concepts (today: part/db_shape.hpp
                       + the kv_engine concept) — namespace adb::
libraries/sal          smart alloc — unchanged
libraries/ucc          SIMD unsigned-char ops          [MISSING FROM PLAN]
libraries/hash         xxhash wrapper                  [MISSING FROM PLAN]
libraries/start_ref    static ART, refcount liveness   (today: part)
libraries/start_ver    static ART, version/stamp       (today: vpart)
libraries/sal_art_ref  ART on sal                      (today: heart)
libraries/psitri       legacy sal btree — unchanged name
libraries/wal_art      two adb engines composed: hot delta tree, batch
                       merge into materialized base, adb on the surface
                       (today: tiered)
libraries/walart       WAL + art_map write-set engine  [NAMING — decision C]
shims/psi-sqlite       sqlite btree.c replacement      (today: psitri-sqlite)
shims/psi-duckdb       (today: psitri-duckdb)
shims/psi-rocksdb      (today: psitrirocks)
shims/psi-leveldb      (today: psitrileveldb)
shims/psi-mdbx         (today: psitrimdbx)
vendor/psio            pssz internal representation (FetchContent already
                       wired; co-dev override; move SOURCE_DIR external/ →
                       vendor/)
tests/                 the behavioral battery, written against adb,
                       instantiated per engine (today: programs/kv_compat_test
                       + crash/recovery + scale gates + quick_gate.sh)
benchmarks/            against adb, per-engine instantiation: bank, dict,
                       random/scale, pipeline, TATP; vendored COMPARATORS
                       (libart, rocksdb, mdbxrocks, tidesdb, absl) live here
                       behind CMake gates — they are not shims
docs/                  design records (vpart_version_lifecycle_audit,
                       unified_tier_design, pdwal_design, node_design_patterns,
                       this file, BUGS.md convention)
logs/                  bench archive convention carries over verbatim
scripts/               run_tests.sh, quick_gate.sh, watchdog, bench matrices
```

## 2. Rename map (find-replace pass)

| today | new path | namespace today → new |
|---|---|---|
| libraries/part | libraries/start_ref | `part::` → `start_ref::` (OPEN — decision B) |
| libraries/vpart | libraries/start_ver | `vpart::`/`vhkv::` → `start_ver::` (OPEN) |
| libraries/heart | libraries/sal_art_ref | `heart::` → ? (OPEN) |
| libraries/tiered | libraries/wal_art | `tiered::` → `wal_art::` |
| part/db_shape.hpp | libraries/adb | `part::shape::` → `adb::` |
| libraries/psitri* (API shims) | shims/psi-* | unchanged |
| libraries/mdbxrocks, libraries/art | benchmarks/vendor | comparators, not shims |
| programs/kv_compat_test.cpp | tests/ | — |
| programs/*_bench* | benchmarks/ | — |
| external/psio | vendor/psio | — |

## 3. Gaps found in the plan (review, 2026-06-12)

1. **ucc + hash are load-bearing and unlisted.** sal, part, vpart, heart all
   depend on ucc; hash is used across benches/engines. Keep both as
   libraries/.
2. **adb cross-repo consumption** — psitri-multiindex must see the adb
   concepts. Options: (a) adb stays a psitri subdir but is designed
   independently consumable (psio dependency-pattern: FetchContent +
   SOURCE_SUBDIR), (b) its own tiny repo. → decision A.
3. **Two kinds of "foreign API" code are conflated.** psitri-behind-foreign-
   API (rocksdb/leveldb/sqlite/duckdb/mdbx faces) are shims/. Foreign-engine-
   behind-our-bench (mdbxrocks, vendored libart, tidesdb, rocksdb itself) are
   benchmark comparators → benchmarks/vendor, CMake-gated. The plan's
   "shims/psi-lmdbx" reads as the former; mdbxrocks is the latter.
4. **Shims currently bind the legacy psitri engine specifically** (sqlite's
   DWAL backend, etc.). Long-term they should bind adb so wal_art can back
   sqlite. Carry as future work; do NOT block the move on it.
5. **Three permanent hkv descendants** — artpp (heap, single-thread),
   start_ref (arena, refcount), start_ver (arena, stamps). The divergence is
   intentional (different liveness models); record it, do a one-time
   bug-class diff sweep at extraction time, then stop trying to keep them in
   sync.
6. **Test split rule needs to be explicit**: behavioral battery + concept
   phases + crash/recovery + drain census + scale gates → tests/ (adb-typed,
   per-engine instantiation). Engine-INTERNAL tests (freeset, arena, node
   layout, TSan suites, fuzz of private invariants) stay in
   libraries/X/tests. "Vast majority against adb" = the behavioral set.
7. **Repo-name/library-name shadow**: repo psitri contains libraries/psitri.
   Accepted (it is the eponymous engine) — just noting it.
8. **CI not mentioned**: new repo needs the build matrix (clang/gcc,
   Debug+asserts, TSan job for the MT suites) and the battery as the merge
   gate from day one — the gates below assume it.
9. **walart vs wal_art near-collision** → decision C.
10. **adb concept is not yet complete** — the carried TODO ledger (§4) has
    concept-level items (schema persistence API, psio storable adapter,
    subtree-value storage); land them in adb as part of the move, not after.

## 4. Carried-forward TODO ledger (this session's context, mapped to new homes)

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
- [ ] concept: cursor typed surface (value_type()/is_type<T>()) — vpart has
      it; promote to the concept when a second engine lands it

**libraries/start_ver (vpart) — first GREEN engine (1d97f950)**
- [x] full kv-compat PASS: 152219 checks, typed values + multi-root +
      snapshot + exact drain census; unit suite green; bank 676k tx/s
- [ ] GC/writer-isolation rework: O(1) bucketed claim_ (today O(nseg) scans
      poison once the arena grows), bigger ring adoption, drain throughput
      vs batch-1000 production. GATE: 150M no-flush writer isolation
      (today degrades 164ns → 5.3µs by ~46M)

**libraries/start_ref (part), sal_art_ref (heart), psitri — green-ups, in order**
- [ ] each: typed values (envelope+hooks per its liveness model), multi-root
      + set_root, snapshot; battery flips phase-by-phase, no adapter layers
- [ ] psitri additionally: wrong-pod read must be invalid_type, never
      not-found (it has open_root/primary already; missing set_root)

**libraries/wal_art (tiered) — #32 rewrite**
- [ ] concept-only: templated on two adb engines; THREE instantiations under
      test (heart/heart, vpart/heart, part/heart)
- [ ] rotation expressed via multi-root set_root (h1.set_root(h0.set_root({})))
      — DELETE vpart's contraband shims (freeze_and_reset, clear_warm_slots,
      pending_warm) from the engine's public surface
- [ ] snapshot migration (§8 identity model): ttree = {hot ref, cold ref |
      resolved cold ref}; persistent lock-free bitset registry; merger swaps
      identities — never copy-constructs a tree; ordered boundary
      materialization; subtree values opaque to the hot engine
- [ ] 1B scale run vs heart baseline (#28) — blocked on start_ver GC gate

**tests/**
- [x] battery semantics locked: take_pod counts +1 (custody enters the pod);
      quiescent census = no live sessions (session fcache holds claimed lines
      legitimately)
- [ ] schema-persistence battery case: register, store, reopen with permuted
      type ids, read back by schema match
- [ ] crash-fuzz gate for walart (was still gating when parked)

**other carried items**
- [ ] dwal btree_layer compact stored values + stale-view audit (#20)
- [ ] heart point-op gap, sample-driven (#22, 1.5–3.4x vs part family)
- [ ] psitri-multiindex: port to adb concept + heterogeneous lookup audit

## 5. Migration mechanics

1. Freeze: tag psitri-merge (`reorg-baseline`); psitri-merge becomes the
   read-only archive — branches, logs/, history stay here.
2. Scaffold @gofractally/psitri: target tree, root CMake, CI, vendor/psio.
3. Move in dependency order, ONE library per commit, full battery + that
   library's unit suite green before the next: ucc, hash → sal → adb →
   start_ref → start_ver → sal_art_ref → psitri → wal_art/walart → shims →
   tests/benchmarks last (they compile against everything).
4. Renames are mechanical find-replace per §2 table; no logic edits ride
   along. Anything broken by a rename gets fixed in the SAME commit.
5. Does not move: build dirs, scratch testdbs, dead experiments living on
   other branches (wtree2, big_index, fast_index v4 stay archived in their
   repos/branches).
6. After the move: psitri-merge gets a README pointer to the new repo and
   the path-mapping table.

## 6. Decisions needed (Dan)

A. **adb consumption**: in-repo subdir consumable standalone (recommended,
   psio-pattern) vs own tiny repo?
B. **Namespaces**: start_ref::/start_ver:: as written? heart's namespace →
   sal_art_ref:: or keep heart::? (Paths and namespaces can rename in the
   same pass either way.)
C. **walart vs wal_art**: keep both (they are different architectures —
   WAL+write-set vs delta-tree rotation)? If both, one should be renamed —
   the near-collision will bite.
D. **Shim survivors**: sqlite + rocksdb earn their keep (TATP, benches).
   leveldb / mdbx / duckdb faces — move, or archive in psitri-merge?

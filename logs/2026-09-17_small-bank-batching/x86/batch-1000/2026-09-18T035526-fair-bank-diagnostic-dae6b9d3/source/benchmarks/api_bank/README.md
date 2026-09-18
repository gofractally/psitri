# Same-API bank comparisons

`api-bank-v1` compiles the shared bank driver against an original library and its
psitri implementation. Compare pairs within a machine, revision, workload and
durability cohort. ARM versus x86 also changes the CPU, OS, page size and storage;
the architecture selector is a machine selector, not an isolated ISA experiment.

The packaged MDBX shim and the separate RocksDB/SQLite integration source use
separate build directories to avoid conflicting library symbols. The tested
source pins are in [`sources.json`](sources.json). Fetch the integration branch
from the psitri repository and check out the exact pinned commit in a separate
worktree; pass that path as `API_BANK_INTEGRATIONS`. Pin every source revision. The latter integrations are not the maintained engine profiles.

```sh
cmake -S benchmarks/api_bank -B build/api-mdbx -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DFETCHCONTENT_SOURCE_DIR_PSIO=/path/to/clean/psio \
  -DFETCHCONTENT_SOURCE_DIR_UCC=/path/to/clean/ucc \
  -DFAIR_MDBX_INCLUDE_DIR=/path/to/libmdbx \
  -DFAIR_MDBX_LIBRARY=/path/to/libmdbx.a \
  -DPSITRIMDBX_BACKEND=sal_art_ref
cmake --build build/api-mdbx --target fair-bank-mdbx fair-bank-psitrimdbx

cmake -S benchmarks/api_bank -B build/api-integrations -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DFETCHCONTENT_SOURCE_DIR_PSIO=/path/to/clean/psio \
  -DAPI_BANK_INTEGRATIONS=/path/to/pinned/integrations \
  -DAPI_BANK_ABSEIL=/path/to/pinned/abseil \
  -DFAIR_ROCKSDB_INCLUDE_DIR=/path/to/rocksdb/include \
  -DFAIR_ROCKSDB_LIBRARY=/path/to/librocksdb.a
cmake --build build/api-integrations --target fair-bank-rocksdb \
  fair-bank-psitrirocks fair-bank-sqlite fair-bank-psitri_sqlite

python3 scripts/fair_bank.py --suite api --build build/api-mdbx \
  --machine my-runner --profile smoke --engines mdbx psitrimdbx --durability none
python3 scripts/fair_bank.py --suite api --build build/api-integrations \
  --machine my-runner --profile smoke --engines rocksdb psitrirocks --durability none
python3 scripts/fair_bank.py --suite api --build build/api-integrations \
  --machine my-runner --profile smoke --engines sqlite psitri_sqlite --durability none
```

The example explicitly selects the direct `sal_art_ref` MDBX backend. The shim's
CMake default is the tiered `delta_art` backend; these are different engine
configurations, not interchangeable measurements of a single shim. Use separate
build directories and retain the backend choice with each comparison.

After smoke validation, investigate unexpected results with `--profile diagnostic`.
It uses the performance workload but runs each selected engine exactly once,
retains correctness checks and the idle gate, and never qualifies as a publishable
performance series. Change one variable at a time; for example, compare the
default 1,000-transfer batch with `--batch 1` before attributing a difference to
the engine. This does not recreate the April workload, which also used different
keys, account selection, transfer logging, and SQLite schema.

```sh
python3 scripts/fair_bank.py --suite api --build build/api-mdbx \
  --machine my-runner --profile diagnostic --engines mdbx psitrimdbx \
  --durability none --batch 1
```

Inspect reader progress as well as throughput. In particular, a mixed phase with
only one read batch per reader can pass data-correctness checks while exposing
reader starvation; it does not establish useful concurrent progress. Diagnose
snapshot admission and cursor work before collecting more repetitions.

## Direct engine batching diagnostic

The packaged build also provides `fair-bank-large`, using the native Large Engine
API with the same driver and compiler settings as the MDBX pair. The `direct`
suite compares native MDBX, the direct MDBX shim, and native psitri. It requires
`PSITRIMDBX_BACKEND=sal_art_ref`, disables the shim's optional read-promotion cache
to match the native adapter, and gives both psitri paths the same pinned-cache
budget. The manifest retains those shim environment settings. The native API
sets its database capacity from `--map-gib`; the shim retains its allocator's
default capacity limit, so this resident-data diagnostic is not a capacity test.

```sh
cmake --build build/api-mdbx --target fair-bank-mdbx fair-bank-psitrimdbx fair-bank-large
python3 scripts/fair_bank.py --suite direct --build build/api-mdbx \
  --machine my-runner --profile diagnostic --durability none \
  --accounts 100000 --transfers 1000000 --batch 1
```

Sweep batches 1, 10, 100, 1000, 10000, and 100000 sequentially, with one diagnostic
per configuration. Keep the account count, transfer trace, warmup, reader settings,
and durability fixed. `write` measures the transfers without concurrent readers;
mixed-phase metrics are retained separately. This measures transaction grouping,
not persistent subtree creation or a database larger than RAM.
`scripts/direct_bank_sweep.py --build build/api-mdbx --output NEW_DIRECTORY
--machine my-runner` runs this sweep after a smoke check, stopping on a failed
cohort and retaining each invocation and its evidence.

Once the workload, backend selection, and any anomalies are understood, use
`--profile performance --repeats 5` for each pair. Or collect all three pairs in
sequence, measuring only the pairs that pass smoke:

```sh
python3 scripts/api_bank_matrix.py --machine my-runner \
  --build-mdbx build/api-mdbx --build-integrations build/api-integrations \
  --output logs --repeats 5 --idle-timeout 1800
```

The matrix retains failures and all repetitions. It does not publish results.
Finish all builds first. The runner takes a machine-wide advisory lock and
requires ten consecutive seconds below 3% aggregate CPU activity before each
process. Linux I/O wait counts as busy. Idle samples are archived. A timeout fails
without launching a benchmark; `--idle-timeout` changes only the waiting limit.
The lock coordinates cooperating runners; do not launch unrelated benchmarks or
builds during collection. Use a dedicated idle host, with the same workload and
source/dependency revisions on both architectures. Run one benchmark at a time.

All pairs keep rollback, held-snapshot, read-your-writes, exact final account
state, balance, cardinality and reopen checks. Unsupported calls or semantics are
failures, not a reason to remove a check for the shim. Only complete passing pairs
belong in a performance chart. The API suite is separate from `fair-bank-v2`:
RocksDB uses the supported `WriteBatch` API plus the same application-side pending
map on both implementations, instead of `WriteBatchWithIndex`. Include that
overhead on both sides. SQLite executes identical SQL and PRAGMA calls and
verifies their results; a shim that cannot satisfy them fails validation.

Record machine, OS, page size, source and dependency revisions, compiler flags,
effective durability, every repetition, and failed attempts. Retain build logs
alongside run evidence. Do not mix the April integration workload with this
workload or treat smoke results as performance evidence.

## Functional regression targets

Configure with `-DAPI_BANK_BUILD_REGRESSION_TESTS=ON` to build
`api-storage-regressions` (Catch2 3 required). It includes page-copy and allocator
lifetime checks, DWAL snapshot/isolation/iteration checks, and the SQLite key and
transaction regressions appropriate to that build directory.

`api-contract-<engine>` targets run the same bank API correctness checks as the
benchmark, including rollback, held snapshots, read-your-writes, exact balances,
cardinality, concurrent readers, and reopen. They write `api-contract-v1` results
without throughput metrics. Use these while debugging; they are functional tests,
not performance evidence. Passing them does not establish complete API coverage.
Performance collection still requires a clean source build, the idle gate, and
exclusive sequential execution.

## DWAL batching diagnostic

Configure a separate build with the same dependencies and compiler flags, replacing
`-DPSITRIMDBX_BACKEND=sal_art_ref` with `-DPSITRIMDBX_BACKEND=dwal`. Build
`fair-bank-mdbx`, `fair-bank-psitrimdbx`, and `fair-bank-dwal`, then run:

```sh
python3 scripts/direct_bank_sweep.py --suite dwal --build build/api-dwal \
  --machine my-runner --output logs/my-dwal-sweep
```

This compares native MDBX, its shim over DWAL, and native DWAL over the Large
Engine at the same six batch sizes. The adapter is an experimental byte-value
binding, not a new public `psitri::engine` profile. Both DWAL paths use the same
buffer thresholds, two merge workers, committed snapshots, and disabled read
promotion. They submit a complete WAL record to the OS at every commit without
per-commit fsync. Rotation and clean close retain DWAL's existing full barriers.
The adapter currently accepts only the `none` durability tier; it makes no
power-loss durability claim. Close/drain time, allocated storage, and peak RSS
are retained alongside foreground throughput.

`basic_dwal_transaction::commit(sal::sync_type::none)` submits WAL bytes before
releasing writer admission; the existing no-argument commit still buffers them.
WAL format 2 uses a 32-bit operation count, so a 100,000-transfer transaction's
200,000 updates can be replayed as one atomic entry. Readers also accept format
1's 25-byte header; writers reject incompatible files instead of rewriting them.

## Small Engine batching diagnostic

The packaged build also provides `fair-bank-small` (direct Small Engine) and
`fair-bank-dwal_small` (DWAL over Small Engine). Both use `api-bank-v1`, the same
native byte API adapter, and the same trace and validation as the Large Engine
comparison. Build both targets, then collect one isolated run per batch size:

```sh
cmake --build build/api-dwal --target fair-bank-small fair-bank-dwal_small
python3 scripts/direct_bank_sweep.py --suite small --build build/api-dwal \
  --machine my-runner --output logs/my-small-engine-sweep
```

The `small` suite has no MDBX shim target. Its DWAL binding is experimental and
uses the same WAL submission, merge thresholds, and close barriers as DWAL over
Large Engine. Small Engine retains its default fixed main arena capacity of
4096 × 8 MiB (32 GiB), with background release enabled. `--map-gib` does not resize
that arena, and `--cache-mib` is not a Small Engine cache limit. This resident
bank dataset does not exercise the separate large-value arena or establish
performance beyond RAM. Compare reader throughput, memory, and close/drain time
alongside writes; large DWAL commits delay fresh snapshot admission.

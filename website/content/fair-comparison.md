# Common bank comparison

`fair-bank` runs one workload through six compile-time adapters: the Small Engine,
Large Engine, real libmdbx, real RocksDB, system SQLite, and native DuckDB. It is
an application-level transactional point-access comparison. SQLite and DuckDB
execute prepared SQL, so their SQL execution overhead is included. This is not
a SQL analytics benchmark or a measurement of a psitri compatibility wrapper.

The benchmark adds no requirements to the public `psitri::engine` concept.
Experimental engines and foreign-API replacements remain separate configurations.

## Build

Supply development headers and compiled libraries for MDBX, RocksDB, SQLite, and
DuckDB. Pin the versions and retain their build settings. All four are required
when enabling the suite; CMake must not silently skip a competitor. A Release
build and C++23 compiler are required. On macOS use a recent Homebrew LLVM.

```sh
cmake -S . -B build/fair -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
  -DPSITRI_BUILD_EXAMPLES=OFF -DPSITRI_FAIR_BANK=ON \
  -DFAIR_MDBX_INCLUDE_DIR=/path/to/libmdbx \
  -DFAIR_MDBX_LIBRARY=/path/to/libmdbx.dylib \
  -DFAIR_ROCKSDB_INCLUDE_DIR=/path/to/rocksdb/include \
  -DFAIR_ROCKSDB_LIBRARY=/path/to/librocksdb.a \
  -DFAIR_DUCKDB_INCLUDE_DIR=/path/to/duckdb/include \
  -DFAIR_DUCKDB_LIBRARY=/path/to/libduckdb.dylib \
  -DFAIR_ROCKSDB_SYNC_LEVEL=full -DFAIR_DUCKDB_SYNC_LEVEL=fsync
cmake --build build/fair --target fair-bank -j 6
```

Use `.so` or `.a` paths as appropriate. `find_package(SQLite3)` supplies SQLite;
override `SQLite3_INCLUDE_DIR` and `SQLite3_LIBRARY` if needed. A static RocksDB
build may also need `FAIR_ROCKSDB_EXTRA_LIBRARIES` for its compression libraries.
The two `SYNC_LEVEL` arguments above describe the verified macOS validation
builds, not assumptions about every build. Set each to `fsync` or `full` only
after checking that dependency's platform flush path. The default `unknown`
excludes it from synchronous cohorts; hashes alone do not establish the barrier.
The adapter disables RocksDB compression explicitly. The other engines use
their storage formats' own encoding/compression behavior.

For the initial macOS validation, DuckDB's official `v1.5.5`
`libduckdb-osx-universal.zip` was used, SHA-256
`7b5b8915cc382d0708636fe6385c0cdad5a61c9ff8ba2638b3e2141640783155`.
Obtain binaries from the [official release](https://github.com/duckdb/duckdb/releases/tag/v1.5.5)
and verify their checksums. This is a reproducible dependency option, not a
requirement that future comparisons use that version.

## Validate before measuring

Python 3.11+ is required. From the repository root:

```sh
python3 scripts/fair_bank.py --machine my-runner --profile smoke \
  --engines small large sqlite duckdb --durability fsync
```

The runner starts each engine in a separate process with a fresh database.
It retains raw output, process failures/timeouts, exact arguments, settings,
runtime versions, executable and library hashes, compiler commands, source
revision/dirty state, source snapshots, machine data, and each repetition's JSON.
Archives are unique directories under `logs/`. Database files are removed only
after a successful validated run; use `--keep-databases` to retain them too.
Failed runs retain their database for diagnosis.

Smoke results validate the harness and adapters. They are not product speed
evidence, regardless of how attractive their timings look.

## Matched workload

Every adapter receives the same:

- Eight-byte big-endian binary account keys and byte values. The first eight
  value bytes encode the balance; larger values use explicit `0xa5` padding.
- Specified SplitMix64 generator, seed, uniform account selection, and transfer
  order. Source and destination always differ. No skipped work counts as a
  transfer. The initial balance prevents insufficient-funds skips.
- Two reads and two updates per transfer, read-your-writes within the batch,
  and one commit per configured batch, including the final partial batch.
- One writer and the requested number of independent reader sessions, one
  snapshot per `reads_per_snapshot` lookups. Readers copy full values.
- Bulk load, contract checks, warm-up transfers, write-only, concurrent reads
  and writes, read-only, and clean close/reopen phases.

An independent in-memory model replays the transfer stream outside timed regions.
Validation checks cardinality, every account's exact balance and payload, total
balance using a 128-bit accumulator, and the final state after reopening.
Contract checks exercise rollback, repeated updates, read-your-writes, and a
reader snapshot held across a commit on the writer thread. Exposed database
errors are fatal; a missing row, failed SQL step, failed commit, or absent metric
cannot become a successful result. The runner also enables SAL error logging
and rejects native runs that log a storage error, including flush failures.

Only timed phases produce throughput. Validation warms the dataset; these are
warm-workload measurements, not cold-cache measurements. The read-only phase
performs a fixed amount of work. Mixed readers run until the fixed writer stream
finishes, then drain their current snapshots. Mixed rates share that full phase
interval; `mixed_write_active` separately measures the writer's active interval.
Batch p50/p99 latency includes transaction work and commit, not commit alone.

## Choose the durability contract

`--durability` is required. There is no preferred tier: select the application's
failure and recovery model, then compare matching tiers. In a blockchain with
an authoritative block log and replication, database state can be reconstructed;
forcing every derived-state transaction through a device-cache flush can add a
cost the application does not need. Record block-log persistence, replication
acknowledgments, and the permitted replay window as separate responsibilities.

| Tier | Requested work at each commit | Current adapters |
|---|---|---|
| `none` | No per-commit synchronization barrier | Small Engine, Large Engine, MDBX, RocksDB, SQLite |
| `async` | Request asynchronous mapped-page writeback without waiting | Small Engine, Large Engine, MDBX |
| `msync` | Wait for mapped-page writeback (`MS_SYNC`) | Small Engine, Large Engine |
| `fsync` | Wait for ordinary filesystem synchronization | Small Engine, Large Engine, SQLite; MDBX/RocksDB/DuckDB when their builds match |
| `full` | Request device-cache flush (`F_FULLFSYNC` on macOS) | Small Engine, Large Engine, SQLite; MDBX/RocksDB/DuckDB when their builds match |

On macOS, `fsync` sends writes through the filesystem to the drive, but does not
make the same device-cache-flush request as `F_FULLFSYNC`. It is more than merely
copying data into the OS cache. On other systems the distinction and available
barriers depend on the platform. Neither label proves recovery correctness or
that a storage device honors the request.

The adapters expose the actual choices:

- Small Engine / Large Engine: `none`, `msync_async`, `msync_sync`, `fsync`, `full`.
- MDBX: `WRITEMAP` with `UTTERLY_NOSYNC` for `none`, `SAFE_NOSYNC` for `async`,
  and `SYNC_DURABLE` for synchronous tiers. On macOS its build option
  `MDBX_APPLE_SPEED_INSTEADOF_DURABILITY` selects `fsync` (1) or `full` (0).
- RocksDB: WAL remains enabled; `WriteOptions.sync=false` for `none`, `true`
  for its verified build barrier. Turning off synchronization does not turn off
  logging or make the database purely in memory.
- SQLite: WAL with `synchronous=OFF` for `none`, `FULL` for `fsync`/`full`;
  `fullfsync` is enabled only for `full`. The adapter verifies these settings.
- DuckDB: persistent database with its verified synchronous WAL barrier. This
  adapter does not expose a matching no-sync or asynchronous mode.

No library is silently omitted or assigned a stronger tier. Inspect each binary
with `--describe`. For the validated macOS dependencies, use these separate cohorts:

```sh
# Derived state without a per-commit sync barrier.
python3 scripts/fair_bank.py --machine my-runner --profile smoke \
  --engines small large mdbx rocksdb sqlite --durability none

# Asynchronous mapped-page writeback.
python3 scripts/fair_bank.py --machine my-runner --profile smoke \
  --engines small large mdbx --durability async

# Ordinary filesystem synchronization (the supplied DuckDB binary uses fsync).
python3 scripts/fair_bank.py --machine my-runner --profile smoke \
  --engines small large sqlite duckdb --durability fsync

# Device-cache flush (these MDBX and RocksDB builds use F_FULLFSYNC).
python3 scripts/fair_bank.py --machine my-runner --profile smoke \
  --engines small large mdbx rocksdb sqlite --durability full
```

A tier describes the requested barrier, not identical recovery guarantees. For
example, MDBX's `UTTERLY_NOSYNC`, RocksDB's unsynchronized WAL, and SQLite's
`synchronous=OFF` have different crash behavior. Clean reopen is tested here;
process-kill, disk-full, I/O faults, and power-loss tests are separate qualification.
Never combine different tiers into one speed ranking.

### DWAL and tiered storage: separate acknowledgment from consolidation

Flush strength and flush timing are separate axes. A comparison must state when
commit returns, when the WAL reaches the OS/device, and when buffered changes
merge into the base tree. An application may synchronize every transaction,
every block, after a byte/time threshold, or rely on an external replay log.
Record that cadence and the maximum tolerated replay window.

DWAL buffers transaction entries and provides `flush_wal(sync_type)` for explicit
flush strength. Tiered storage batches hot updates before merging them into its
base. Evaluating these configurations requires foreground latency, sustained
throughput, WAL/merge backlog, backpressure, close/drain cost, and recovery time.
A short run that finishes with growing merge debt is not sustained throughput.

The six adapters above measure maintained native engines and real competitors.
DWAL and experimental tiered configurations are **not yet adapters in this
common suite**; their separate benchmarks must state flush cadence, reader
freshness, and snapshot semantics before joining a matched cohort. API wrappers
need the same treatment. An asynchronous hint is not a timed group-commit policy,
and neither is interchangeable with a no-sync run.

References: [SQLite WAL synchronization](https://www.sqlite.org/wal.html),
[RocksDB write synchronization](https://github.com/facebook/rocksdb/wiki/Basic-Operations),
[DuckDB transaction durability](https://duckdb.org/2024/10/30/analytics-optimized-concurrent-transactions).

## Collect a performance series

Commit the source and use a quiet, dedicated runner with stable power settings.
Retain the dependency build recipes as well as the hashes recorded automatically:
a prebuilt library's hash cannot recover its compiler flags. The runner requires
a clean source checkout and identified, clean native dependencies for this profile.

```sh
python3 scripts/fair_bank.py --machine dedicated-runner-01 \
  --profile performance --repeats 5 \
  --engines small large mdbx rocksdb sqlite \
  --accounts 1000000 --transfers 2000000 --warmup 100000 \
  --batch 1000 --readers 4 --reads-per-snapshot 100 \
  --read-batches 1000 --value-bytes 8 --durability none \
  --timeout 7200 --notes 'Record storage device, power policy, background load, dependency builds.'
```

Repeat with batches 1, 100, and 1,000 and value sizes relevant to the application.
The default performance profile has smaller dimensions and at least three
repetitions. Engines run sequentially, with a seeded randomized order rotated
between repetitions. Never discard a slow run. `summary.json` contains median,
minimum, maximum, and all samples. It does not produce a ranking if any requested
run failed, a repetition is missing, the settings differ, or final state hashes
disagree. A completed performance series still needs review before publication.

Memory and space are part of the result. Peak RSS includes the process, driver,
oracle model, database, and validation. Logical and allocated file bytes are
recorded before and after close; close time includes each engine's close work.
These are named checkpoints, not continuous disk high-water measurements.
No forced RocksDB compaction or extra SQL checkpoint is hidden outside the timer.

`cache_mib` is an engine-specific tuning amount, **not a shared RSS limit**:
the Large Engine uses it for pinned segments, RocksDB for its block cache plus
two 64 MiB write buffers, SQLite per connection, and DuckDB for its memory-limit
setting. MDBX relies on the OS cache; the Small Engine relies on resident fixed
storage. `map_gib` controls the Large Engine and MDBX address-space ceilings.
DuckDB query execution uses one worker; the common reader count controls client
concurrency. Native background maintenance remains enabled.

For beyond-RAM work, select the Large Engine and disk-backed competitors, hold
physical/container memory and storage conditions constant, and verify actual
resident and stored sizes. Large padded values may compress: logical payload
size alone cannot establish that a dataset exceeded RAM. The Small Engine is
the resident-data profile. Do not present a short warm-cache run as evidence of
sustained compaction, reclamation, recovery, or beyond-RAM behavior.

## API-wrapper comparisons

Compare a real library with its psitri-backed replacement through the **same
foreign API driver**. The separate integration suite's RocksDB/SQLite bank and
DuckDB TATP drivers serve that purpose, but their results must pass the same
error, exact-state, durability, and provenance requirements before publication.
They do not currently use these maintained engine adapters. Do not label the
native Small Engine/Large Engine results as drop-in-wrapper performance.

For a blockchain state comparison, also match the number and lifetime of retained
forks, speculative transactions, fork-switch depth, and persistent revision
recovery. Include any undo-log, revision-index, copying, and replay work required
by the competing implementation. The common bank workload measures transactional
point access; it does not yet measure that complete versioned-state workload.

# Fair-bank validation, 2026-09-17

These are correctness/harness smoke runs from a shared development machine and
uncommitted development snapshots. They are **not publication-grade throughput,
crash-recovery, or beyond-RAM evidence**. No new timings are added to public charts.

Final `fair-bank-v2` runs use 257 accounts, 1,025 transfers in each timed write
phase, 33 warmup transfers, batches of 13 (including partial batches), two readers,
17 reads per snapshot, three read-only batches, 128-byte values, and three
repetitions per engine. All 57 runs passed exact account cardinality/payload/state,
balance conservation, rollback, repeated-key updates, read-your-writes, held
snapshots across writer commits, and clean reopen. Final hashes match across
engines in each cohort.

| Policy | Archive | Runs |
|---|---|---:|
| none | [5f8cf8c0](../2026-09-17T163211-fair-bank-smoke-5f8cf8c0/manifest.json) | 15 |
| async | [a3c93dc6](../2026-09-17T163213-fair-bank-smoke-a3c93dc6/manifest.json) | 9 |
| msync | [4049b5a6](../2026-09-17T163214-fair-bank-smoke-4049b5a6/manifest.json) | 6 |
| fsync | [deeba65c](../2026-09-17T163215-fair-bank-smoke-deeba65c/manifest.json) | 12 |
| full | [643f3a61](../2026-09-17T163220-fair-bank-smoke-643f3a61/manifest.json) | 15 |

Expected preflight failures are retained too:
[DuckDB cannot match full](../2026-09-17T163254-fair-bank-smoke-76917944/manifest.json)
and [DuckDB cannot match none](../2026-09-17T163254-fair-bank-smoke-574c9971/manifest.json).

## Dependencies and durability

Apple M5 Max / Mac17,6, 18 logical CPUs, 128 GiB RAM, Darwin 25.3.0 arm64;
Homebrew Clang 22.1.2 Release. Per-run manifests retain binary/library hashes,
compiler commands, local native dependency revisions, source patches/snapshots,
machine/storage information, settings, exact commands, stdout/stderr and counts.

- RocksDB 9.9.3: source revision `14d3046a53dda588ef5887ee2c80e93104caf079`;
  static build with `HAVE_FULLFSYNC`. [Build command excerpt](rocksdb-io-build.txt).
  Both `PosixWritableFile::Sync` and `Fsync` request `F_FULLFSYNC` in this build.
- DuckDB v1.5.5: official macOS universal release archive SHA-256
  `7b5b8915cc382d0708636fe6385c0cdad5a61c9ff8ba2638b3e2141640783155`.
  The arm64 binary's [LocalFileSystem::FileSync](duckdb-filesync-arm64.txt)
  directly calls `_fsync`, with no `F_FULLFSYNC` branch. Merely seeing conditional
  full-flush code in source would have misclassified this binary.
- MDBX runtime `v0.14.1-477-g051b7437`: runtime build options/compiler/flags are
  retained in each result. `MDBX_APPLE_SPEED_INSTEADOF_DURABILITY=0` selects full
  flushing. The adjacent source checkout is currently at
  `f6bc81f35ca33dd698be5e2dbfaddc60662de736`, **not the prebuilt library revision**.
  Do not infer a reproducible dependency build from that checkout alone.
- SQLite 3.53.0: source ID
  `4525003a53a7fc63ca75c59b22c79608659ca12f0131f52c18637f829977f20b`.
  WAL, synchronous and fullfsync settings are checked on each connection.

These prebuilt dependencies suffice for adapter validation. A publishable series
must preserve complete dependency build recipes and run on a dedicated host.
The recorded policy is a requested barrier, not a power-loss qualification.

## Retained earlier development runs

The four `fair-bank-v1` archives at 16:17:23, 16:18:03, 16:21:04 and 16:23:32
contain 48 successful development runs, including batch-one and 10,000-account
cases. They used a coarse `sync` label that did not distinguish macOS ordinary
fsync from device-cache flush. They must not be used as matched-durability speed
comparisons. They remain intact with the source versions that generated them.

The 16:28:34 `cfd47026` archive retains 15 passes and three rejected Large Engine
runs. The new log guard found an informational allocator-size message incorrectly
using `SAL_ERROR`; its severity was corrected to debug. No storage corruption or
failed flush was observed. The rejected database files remain locally under that
archive (gitignored); raw errors, result files and failure records are committed.
The archive also predates the final per-library sync capability checks.

A final [five-engine no-sync run](../2026-09-17T163455-fair-bank-smoke-8be1bec2/manifest.json)
also passed after adding raw output retention for capability inspection.

The automated guard suite additionally tests process failures/timeouts, missing
metrics/checks, changed settings/versions, mixed tiers, and incomplete cohorts.

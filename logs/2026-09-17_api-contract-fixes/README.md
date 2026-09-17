# API contract and storage bug fixes

These are functional regression runs, **not performance results**. Debugging did
not use the idle gate and no functional timing supports a speedup claim. The
performance collectors were stopped while fixing the failures. Bank correctness
checks remain enabled on both sides of every API pair.

Corrections:

- The MDBX shim's SAL ART page clone called an aligned AVX-512 copy on an unaligned
  tail. Copy the short leading fragment normally, then the aligned suffix. Tests
  cover every tail alignment and dense binary account updates/rollback.
- RocksDB snapshots omitted committed data still in DWAL's RW and RO layers.
  Snapshots now retain fixed COW roots, range tombstones, and all layer owners.
- Single-root DWAL transactions did not hold the root transaction lock. Snapshot
  capture could observe half a transaction. Writers now hold that lock until
  commit/abort; snapshot reads release it after capture. Existing multi-root
  transaction lock ownership is preserved.
- SQLite encoded WITHOUT ROWID payload columns into its primary key, causing
  duplicate logical rows after updates. Key encoding now uses KeyInfo.nKeyField.
  Table counts are bounded by the selected table prefix. Read transactions retain
  a snapshot, close aborts uncommitted writes, and writer cursors refresh after
  mutation. Sync policy belongs to the connection. Journal-mode reporting comes
  from the actual DWAL backend instead of the unrelated dummy SQLite pager.
- The allocator's utility session was destroyed after the mapped allocator state.
  It now releases its session slot while the mapping is still alive.
- Switching a merged cursor's iteration direction could skip an exhausted layer.
  Direction changes now reposition all layers; same-direction scans retain their
  incremental iteration. Live writer views remain live, while read snapshots use
  fixed roots.
- Interrupting an idle wait no longer starts the next benchmark pair.

The common DWAL and allocator fixes are applied both to the main library and to
the separately pinned RocksDB/SQLite integration sources. These tests cover the
bank workload contract, not every SQLite or RocksDB API feature or durability mode.

The red runs, debugger traces, incremental patches, and green runs are retained.
Earlier patch captures omit then-untracked new files; the final source commits
and the pinned integration revision provide the complete implementation.
Generated databases are excluded. Build logs and commands record the platforms.

Final validation:

- Mac main-library suite: **174 cases, 24,675 assertions passed**.
- Mac integration suite: **154 cases, 2,564 assertions passed**.
- x86 main-library suite: **174 cases, 24,675 assertions passed**.
- x86 integration suite: **154 cases, 2,564 assertions passed**.
- All three full shim bank contracts passed on both platforms. The final hash is
  `12150392390243270099` for every shim, with the same smoke-sized configuration.
- Python website/collector suite: **31 tests passed**.

The integration source is committed as
[`cb3eccaa9af41b6606680fbe59359537b2f4ab20`](https://github.com/gofractally/psitri/commit/cb3eccaa9af41b6606680fbe59359537b2f4ab20)
on `codex/api-compatibility-fixes` in the psitri repository. Main-library fixes and
these retained regressions are committed on `sal2`. See
[`benchmarks/api_bank/sources.json`](../../benchmarks/api_bank/sources.json) for
the dependency pins used for the next clean-build performance collection.

Fresh performance collection remains pending idle runners. Unrelated Rust builds
and desktop activity on the Mac and a psizam workload on x86 were left running.
No fresh performance numbers were published from these debugging runs.

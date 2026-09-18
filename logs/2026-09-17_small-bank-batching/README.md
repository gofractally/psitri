# Small Engine and DWAL bank batching comparison

The interactive report in `comparison.html` adds **Small Engine · direct** and
**Small Engine · DWAL** to the five retained MDBX/Large Engine series. Engine
checkboxes and eight measurements let readers compare writes, concurrent reader
progress, memory, storage, latency, and close/drain cost.

## Results in this diagnostic

All 12 measured runs and both smokes passed. Measured revision:
`43c9c05c` (full hash in `provenance.json`); every measured configuration produced
final account hash `1346279285150323400`.

- At one transfer per commit, direct Small Engine delivered 361K transfers/s
  versus direct Large Engine's 91.5K. DWAL Small Engine delivered 1.023M versus
  DWAL Large Engine's 998K. These are separate single observations.
- At 100,000 transfers per commit, direct Small Engine reached 2.184M transfers/s,
  compared with native MDBX's 1.359M, direct Large Engine's 1.202M, and DWAL Small
  Engine's 1.693M.
- The DWAL drop at 100,000 persists with Small Engine: 2.324M at 10,000 falls to
  1.693M at 100,000 (27%). Large Engine DWAL falls from 2.328M to 1.640M (30%).
  Changing the base does not remove this observed drop; the cause is not profiled
  by this sweep.
- At 100,000, direct Small Engine sustains 2.087M mixed transfers/s alongside
  9.356M reads/s. DWAL Small Engine sustains 1.614M mixed transfers/s alongside
  173K reads/s. The fresh-snapshot admission tradeoff remains visible.
- Small Engine direct used about 71–86 MiB peak RSS with 4–5 ms close time.
  DWAL Small Engine used about 140–1,091 MiB with 35–111 ms close time across
  these batches. Include this buffering cost with the foreground speed.
- Small Engine direct has a non-monotonic point at batch 100: 241K writes/s,
  below batch 10's 512K. This observation is retained unchanged; it has not been
  profiled or replaced by extra repetitions.

## Method

- x86: the same AMD EPYC-Turin cloud host, 16 logical CPUs, approximately 121.5 GiB
  RAM, Linux 6.17.0-22-generic, Clang 20.1.8 Release, `-march=native`.
- `api-bank-v1`: 100,000 accounts, one million transfers in each of the solo and
  mixed write phases, 10,000 warmup transfers, eight-byte keys/values, seed 12345.
- Commit sizes: 1, 10, 100, 1,000, 10,000, 100,000. The selected commit size also
  controls initial loading and warmup.
- Four readers, 100 reads per snapshot, 100 read-only batches per reader. Mixed
  readers run until the writer finishes. A larger DWAL transaction blocks fresh
  snapshot admission longer; mixed writer speed must be read with reader speed.
- No per-commit synchronization for either path. DWAL submits WAL bytes to the OS
  per commit, with its existing full barriers at log rotation and clean close.
  Direct Small Engine uses `sync_mode=none`. These are not power-loss tests.
- Small Engine: fixed main arena, default 4096 × 8 MiB = 32 GiB capacity;
  background release enabled. The shared `map_gib=64` and `cache_mib=256` workload
  settings do not change its arena capacity or impose a cache cap. Eight-byte
  account values do not exercise its separate large-value arena.
- DWAL: 100,000-entry RW threshold, 64 MiB WAL threshold, 1 GiB RW arena threshold,
  two merge workers. The same experimental byte adapter and thresholds are used
  for Small Engine and Large Engine.
- Exactly one diagnostic per configuration after fixes. A machine-wide advisory
  lock serializes cooperating runners; every process waits for ten consecutive
  seconds at or below 3% aggregate CPU activity, counting Linux I/O wait as busy.
- The Mac stayed busy (20.8–29.2% CPU in `local/arm-availability.json`), so there
  are no ARM performance bars. Local timings are functional evidence only.

Prior MDBX, shim, and Large Engine bars come from
[`../2026-09-17_dwal-bank-batching`](../2026-09-17_dwal-bank-batching/README.md)
and its preceding direct sweep. They are separate sequential observations on the
same machine, not simultaneous pairs or repeated distributions. The report
checks hardware, workload settings, and final hashes before combining them.
The MDBX shim bars still use Large Engine, directly or under DWAL; this addition
measures the two **native** Small Engine paths.

This dataset fits in RAM. These single observations do not establish a release
performance claim, results beyond RAM, or architecture-independent rankings.

## Correctness fixes found before charting

`x86-attempt-01/` retains the initial failed cohort at revision `4314570`.
Direct Small Engine passed; DWAL Small Engine failed exact account validation
following mixed writes. The sweep stopped at batch 1.

Small Engine reused a writer session's old working root even after another
session committed. DWAL's two merge workers exposed lost updates. The fixed
transaction start locks the root and adopts its latest committed version before
reading or mutating. The rule also applies to additional roots opened in a
transaction. A deterministic two-session test failed before the fix and passes
after it, covering retained snapshots, rollback, multiple roots, and reopen.
The full Small Engine suite passes 39 tests / 1,098,734 assertions.

The new DWAL adapter tests also found that the close path skipped buffers
containing only range tombstones. It now tests whether the whole layer is empty,
so those deletions reach the base before the WAL is removed. Tests cover both
Small Engine and Large Engine. Core DWAL tests passed 159 cases / 602,375
assertions; the adapter and session regressions passed 6 cases / 92 assertions;
all 38 Python evidence/website tests passed.

The failed database bytes remain outside Git at
`build/small-dwal-failed-database-01` locally and in the original cloud
`small-evidence/x86/batch-1/.../r1-dwal_small/database` directory. Raw errors,
commands, idle samples, manifests, and build information are committed here.

## Reproduce and inspect

`provenance.json` identifies the measured revision, binary hashes, and exact
cloud build commands. `x86/sweep.json` records every invocation and manifest.
Each run retains raw output, validation results, configuration, build flags,
dependency information, and idle samples. `audit_results.py` verifies the
archive; `render_report.py` regenerates the derived chart and JSON. These
scripts do not publish the public website.

```sh
python3 logs/2026-09-17_small-bank-batching/audit_results.py
python3 logs/2026-09-17_small-bank-batching/render_report.py logs/2026-09-17_small-bank-batching
```

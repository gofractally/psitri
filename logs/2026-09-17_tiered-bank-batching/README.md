# Tiered Small Engine + Large Engine bank comparison

The interactive report in `comparison.html` adds **Tiered · Small Engine + Large
Engine** to the seven retained native MDBX, shim, direct, and DWAL series. All
six new measurements and the smoke passed. Every measured configuration
produced final account hash `1346279285150323400`, matching the previous sweeps.
The measured source is `87a63d029ccf6f6f2ccb0a4285b61986e084b703`.

## Observations

At 100,000 transfers per commit, the tiered pair delivered **3.705M foreground
transfers/s**, versus the retained direct Small Engine result of 2.184M and DWAL
Small Engine result of 1.693M. With four concurrent readers, tiered delivered
2.007M transfers/s alongside 10.770M reads/s. These are single diagnostic
observations from separate sequential sweeps on the same host.

The tiered write-only measurements for batches 1, 10, 100, 1,000, 10,000, and
100,000 were approximately 357K, 508K, 772K, 372K, 898K, and 3.705M transfers/s.
The non-monotonic batch-1,000 observation is retained; no extra repetitions were
used to select a faster result. Peak RSS was 390–405 MiB and close took 23–31 ms.

**Foreground throughput does not imply completed cold migration.** At the first
post-write validation check, batches 1 and 10 had completed two base commits.
Batches 100 through 100,000 had completed none at that check. By the post-mixed
check every run had completed one to four base commits. Background migration
can continue during the untimed model/validation work between phases. These
measurements do not establish the sustained throughput of a fully drained
pipeline. The chart makes this visible when foreground writes are selected.

## Configuration and matched work

- Native `psitri::experimental::tiered_engine<>`, which composes
  `psitri::small_engine` (`start_ref::kv::engine`) with `psitri::big_engine`
  (`sal_art_ref::radix::engine`). This is not the MDBX shim's separate tiered
  default using `start_ver`.
- Reads use the merged **latest** view (hot, warm if a flush is active, and cold
  base), not the base-only stable view. The latest view retains normal bank
  correctness checks including held snapshots and read-your-writes.
- Both tiers use `sync_mode=none`; there is no WAL. No per-commit fsync is added.
- **16 MiB bytes-written flush trigger**, explicitly selected below the engine's
  512 MiB default so the fixed trace exercises cold migration. This is not a
  16 MiB memory cap. Background flush is enabled; each base transaction may
  contain up to 10 million keys. Small Engine keeps its 32 GiB default main
  arena limit and background release. Large Engine has read promotion disabled,
  a 256 MiB pinned-cache budget, and the same 64 GiB capacity setting.
- 100,000 accounts; one million transfers in each timed solo/mixed write phase;
  10,000 warmup transfers; eight-byte keys and values; seed 12345. Batch size
  also controls loading and warmup. Four readers, 100 reads per snapshot,
  100 read-only batches per reader; mixed readers run until the writer finishes.
- Same x86 host: AMD EPYC-Turin, 16 logical CPUs, approximately 121.5 GiB RAM,
  Linux 6.17.0-22-generic, Clang 20.1.8 Release with `-march=native`.
- One process at a time under the machine-wide lock; ten final consecutive idle
  samples at or below 3% CPU activity before each invocation. I/O wait counts
  as busy. One diagnostic per batch size, not a repeated release series.
- Mac availability samples were 15.5–21.9% busy. No ARM timings were collected;
  local test timings are correctness evidence only.

Close joins an active flush but preserves remaining committed hot data in Small
Engine. Reopen reads that hot data together with the cold base. DWAL instead
drains its overlays into the base on close. The chart's close metric therefore
includes each engine's actual lifecycle, not a common forced full-drain barrier.
The dataset fits in RAM and does not test capacity or performance beyond RAM.
The tiered composition remains experimental, without portable owned composite
subtree snapshots.

## Validation and evidence

The pair-specific tests passed 2 cases / 26 assertions: explicit migration,
snapshots across migrations, tombstone masking, rollback, hot/cold reopen, and
atomic paired reads during background migration with one-key base commits.
The common bank smoke passed all six checks. All 38 Python tests passed.
Every full x86 run passed rollback, held snapshots, read-your-writes, exact
accounts, balance conservation, cardinality, and clean-close/reopen validation.

`base_commit_count()` exposes the existing base-generation counter, with no new
per-key instrumentation. The adapter logs it outside the timed write phases.
`provenance.json` retains the counter after each validation point; it resets on
reopen. The audit requires nonzero completed migration in every measured run.
The chart also retains this phase-by-phase evidence in `comparison.json`.

Raw output, commands, source/dependency revisions, binary hashes, compiler
settings, build logs, storage information, and idle samples are under `x86/`.
`provenance.json` includes exact build commands. The previous seven bars come
from [`../2026-09-17_small-bank-batching`](../2026-09-17_small-bank-batching/README.md)
and its pinned predecessors; the renderer verifies matching hardware, workload
settings, and final hashes before combining them. No baseline was rerun here.

```sh
python3 logs/2026-09-17_tiered-bank-batching/audit_results.py
python3 logs/2026-09-17_tiered-bank-batching/render_report.py logs/2026-09-17_tiered-bank-batching
```

Browser checks cover all eight engine controls, six batch groups, eight metrics,
empty/specific selections, and mobile layout. This artifact is a local diagnostic
report; the public website was not changed or published.

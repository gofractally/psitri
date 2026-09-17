# API-pair performance investigation

The investigation found and fixed SQLite reader starvation, and isolated a
substantial batching effect. It used source inspection, targeted functional
traces, and three single-run diagnostic measurements after the hosts' other
timed work finished. No repeated performance cohort was collected. Existing
results and failed attempts remain intact; diagnostics are not published as
performance-series evidence.

## Evidence and source identities

- April archive: [`../2026-04-06_integration-bank/`](../2026-04-06_integration-bank/).
  Its recorded measured revision is
  `ec383f2b976586d15c8e0000fd1b23614caa9329`. The bank driver at that revision was
  inspected in the integration repository, rather than assuming today's driver
  described the archived run.
- September archive:
  [`../2026-09-17_api-pairs-x86/2026-09-17T222003-api-matrix-66a5d3a1/`](../2026-09-17_api-pairs-x86/2026-09-17T222003-api-matrix-66a5d3a1/).
  Measured main source: `5ba8186a9bb6c99f92262b9cc3cf77bc13c92e5b`;
  integration source: `cb3eccaa9af41b6606680fbe59359537b2f4ab20`.
- Audit checkout: `85f5bf0533e8e170bebb96b6ad2ed101495a2662`.
- Integration fix: `e62126d5aa2d4c73d1c7f244eb7911190efc85e8`, pushed to
  `codex/api-compatibility-fixes` and pinned in `benchmarks/api_bank/sources.json`.
  `admission-fix/source-validation.json` verifies that the local engine and test
  sources match the files tested on x86.
- Both archives identify an AMD EPYC-Turin, 16 logical CPUs and approximately
  121 GiB of RAM. Kernel versions differ. The April subset lacks complete build
  flags, dependency versions and repetitions, so it cannot establish a controlled
  regression baseline.

## Confirmed comparison differences

| Choice | April integration driver | September API-pair driver |
| --- | --- | --- |
| Accounts / transfer attempts | 1,000,000 / 1,000,000 | 100,000 / 100,000 |
| Transfers per transaction | 1 | 1,000 |
| Keys | Variable-length account names | Eight-byte big-endian integers |
| Account distribution | Triangular / hot accounts | Uniform |
| Transfer writes | Two balances plus an appended transfer log | Two balances |
| Skipped transfers | Counted in reported attempt rate | No skipped work permitted |
| Concurrent readers | One | Four |
| SQLite table | TEXT primary key, rowid table plus primary-key index | BLOB primary key, WITHOUT ROWID |
| SQLite update SQL | INSERT OR REPLACE | ON CONFLICT DO UPDATE |

These are different workloads. In particular, batching amortizes transaction
costs, and the SQLite schema removes the separate rowid-table/index combination.
The single-run batching diagnostic below isolates transaction grouping; the
schema and key-distribution effects have not been individually measured.
Neither the old relative win nor the new relative loss quantifies a code regression.

The packaged MDBX shim was explicitly built with
`PSITRIMDBX_BACKEND=sal_art_ref`, as recorded in the September CMake cache. That is
the direct backend, not the default `delta_art` configuration combining a
`start_ver` hot tier and `sal_art_ref` cold tier. The README build example also
selects the direct backend. This result cannot represent the tiered backend's
performance. It is not yet established which backend produced any earlier
successful MDBX win; the April MDBX pair in this archive did not complete.

Release settings are present in the September build evidence: Clang 20.1.8,
`-O3 -DNDEBUG`, native CPU tuning. A Debug build is not the explanation for this
cohort.

## What the saved measurements actually show

September medians, successful transfers per second:

| API | Original library | psitri shim | Shim / original |
| --- | ---: | ---: | ---: |
| MDBX, direct sal_art_ref | 589,564 | 256,578 | 0.435× |
| RocksDB | 395,806 | 576,781 | 1.457× |
| SQLite | 353,764 | 68,711 | 0.194× |

The RocksDB shim still leads on this workload; its mixed reader rate is also
2.20× the original library's. The April shim write attempt rates were 369,446 for
RocksDB and 52,835 for SQLite. Today's absolute write numbers are higher, but the
changed work and counting rules prevent treating that as an engine speedup.
The April native SQLite run failed balance conservation and cannot support a
valid SQLite speedup claim.

## Confirmed SQLite reader starvation and fix

Across the five September repetitions, the SQLite shim completed only
**400, 500, 400, 400, 500 mixed reads**. Each snapshot contains 100 reads and there
are four reader threads: only four or five reader batches complete in the whole
mixed phase. Standalone read throughput is approximately one million reads/s;
the mixed median is only 275 reads/s.

The snapshot correction in `cb3eccaa` introduced an exclusive acquisition of
`root.tx_mutex` in `libraries/psitri/include/psitri/dwal/read_snapshot.hpp`.
`start_write_transaction()` in `dwal_database_impl.hpp` takes that same lock, and
the write transaction retains it through commit/abort. SQLite's read-transaction
entry constructs a `read_snapshot`. Thus a new snapshot must wait for an active
writer. Existing snapshots release the lock after capture and remain readable.

The writer runs 100 consecutive transactions of 1,000 transfers. A targeted
[x86 lock-admission trace](x86-admission-trace/) confirmed that readers waited
1.13 seconds while **76–77 write transactions** committed, then waited another
0.34 seconds through **23 more commits**. Instrumentation counted actual commits
and measured time acquiring `tx_mutex`, without tracing SQL statements. The host
passed the ten-second idle gate and ran this diagnostic alone.

The default lock policy now uses FIFO shared/exclusive admission. Queued snapshot
captures cannot be overtaken by repeatedly starting writers, and future readers
cannot overtake a queued writer. Consecutive shared transactions can still overlap.
This preserves transaction isolation, sorted multi-root lock ordering, the engine
concept, and custom fiber lock policies. Already captured snapshots retain no
transaction lock. New captures can still wait behind the active writer and
requests already queued; this is not a claim of nonblocking snapshot acquisition.

The common driver's `mixed_reads > 0` check only establishes some reader work.
A batch acquired near or after the last commit still contributes to that count.
Passing data/snapshot correctness therefore does not establish useful reader
progress during the write phase. That check alone was insufficient to catch
this scheduling defect; admission now also has targeted regression coverage.

The same integration change also destroys writer cursors after every insert and
delete in `libraries/psitri-sqlite/src/btree_psitri.cpp`. Recreating a cursor copies
layer owners and creates a base-tree cursor. The updated DWAL cursor already uses
a live writer view on reseek. This was investigated as a candidate for avoidable
work; the functional trace below does not support it as the cause of this bank
workload's slowdown.

## Targeted functional traces on the Mac

These are `api-contract-psitri_sqlite` executions with tracing, not throughput
measurements. The temporary instrumentation patches, commands, raw output, binary
hashes, configurations and result hashes are retained. Tracing changes scheduling;
the runs do not establish performance on either architecture. Instrumentation was
removed and the ordinary functional binary rebuilt afterward. These traces
preceded the FIFO fix.

- [`sqlite-cursor-trace/`](sqlite-cursor-trace/): 32 accounts, 64 transfers per
  phase, batches of 16. All correctness checks passed. Every merge-cursor
  construction followed a SQLite `BtreeCursor` opening; **zero** constructions
  followed a mutation reset within the same SQLite cursor lifetime. The write
  phase plus validation opened and constructed 290 cursors each: 256 for the
  four SQL statements per transfer, plus validation/setup. Removing mutation
  resets would not avoid those statement-level constructions. No speculative
  cursor-reuse change was made.
- [`sqlite-admission-trace/`](sqlite-admission-trace/): 128 accounts, 1,000
  transfers per phase, batches of 10, four readers. All correctness checks passed.
  The trace recorded 148 snapshot acquisitions in the mixed phase, including
  connection setup and validation. Requests sometimes spanned subsequent writer
  commit events, with at most two intervening commits in this run. It did **not**
  reproduce the x86 large-batch starvation. The smaller batches, different lock
  implementation and intrusive statement logging prevent treating this as
  evidence that the published x86 anomaly is fixed.

The x86 diagnostic subsequently used the original batch size and minimal
lock/commit instrumentation after the other timed cohort finished.

## Single-run diagnostics after the fix

Every run passed rollback, held-snapshot, read-your-writes, exact account state,
balance, cardinality and reopen checks. All produced final hash
`2427670804914841226`. Each ran alone after the idle gate. These are dirty-build
diagnostics with retained source patches, not release performance series.

- [Batch 1,000, shim only](admission-fix/x86/diagnostic/): **40,000 mixed reads**
  completed, compared with 400–500 in the original cohort. Write-only throughput
  was 68,457 transfers/s; mixed throughput was 66,707 transfers/s and 26,683
  reads/s. Snapshot acquisition still waits at transaction boundaries, which
  limits snapshot frequency with these long batches.
- [Batch 1, original and shim](admission-fix/x86/batch-one/): all other workload
  settings remained the same. One run per implementation:

  | Metric | SQLite | psitri shim |
  | --- | ---: | ---: |
  | Write-only transfers/s | 147,200 | 65,444 |
  | Mixed transfers/s | 25,030 | 46,546 |
  | Mixed reads/s | 882,773 | 1,026,609 |

The mixed-write ranking reverses with batching. The write-only result still
favors SQLite. This isolates why changing transaction grouping materially
changes this comparison; it does not reproduce April's different schema,
keys, distribution, transfer logging or skipped-work accounting.

## Validation and retained source

- The new admission regression fails against the old x86 lock implementation.
- Final Mac and x86 main suites: **176 cases, 24,712 assertions passed** each.
- Final Mac and x86 integration suites: **156 cases, 2,601 assertions passed** each.
- Python collector/website validation: **38 tests passed**.
- The full x86 suite exposed an existing leak-test error: 500 keys grew to 2,500,
  while the bound required strictly fewer than five times the initial object
  count. It failed at 2,527 versus a 2,525 limit. The test now updates a fixed
  500-key dataset across ten merged generations, checks each refreshed value,
  requires zero pending releases and at most the original live-object count.
  The failed run is retained in `admission-fix/x86/build-integrations-tests.log`.
- Final engine/runner patches are in
  `admission-fix/x86/psitri-final.patch` and
  `admission-fix/x86/integrations-final.patch`, relative to the measured source
  pins above. The integration patch preserves that branch's distinct cursor
  semantics and temporary-directory setup. Build logs and source archives are
  retained alongside the diagnostic manifests.

## Next diagnostic steps

1. Use one diagnostic invocation per hypothesis, on an idle host and sequentially.
   The new `--profile diagnostic` uses the performance-sized workload exactly
   once per engine, retains all checks and raw evidence, and cannot qualify for
   publication as a repeated performance series.
2. To further improve SQLite mixed reads with large batches, investigate
   acquiring an immutable committed snapshot during an active write. FIFO
   admission fixes starvation but does not remove the snapshot-boundary wait.
3. Compare explicitly named MDBX direct and tiered builds under the same contract
   checks and workload. Do not silently replace the direct measurement with the
   faster configuration. Profile point lookups, write-set bookkeeping and commit
   costs if the configured shim remains unexpectedly slow.
4. Resume repeated collection only after these issues are understood and fixes
   pass targeted correctness tests. Do not discard the present measurements or
   weaken their validation to recover a favorable ratio.

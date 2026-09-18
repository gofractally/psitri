# Direct versus DWAL: bank transfers per commit

One diagnostic run per engine and commit size on the x86 server, with the
preceding direct-engine sweep retained as a reference. All 18 measured
processes and three smoke processes passed. Source:
`2f10f14f89560489cf8c7d7d969c232797d605a3`. Every measured run produced final
hash `1346279285150323400`.

Native DWAL led native MDBX in write-only throughput at all six commit sizes:
998K transfers/s at batch 1, peaking at 2.33M at batch 10,000, and reaching
1.64M at batch 100,000. The corresponding MDBX results were 36K, 1.22M, and
1.36M. The DWAL MDBX shim led MDBX at batches 1 through 1,000; MDBX led the
shim at 10,000 and 100,000. At batch 100, the shim reached 1.11M transfers/s
versus MDBX's 285K (3.90×).

Fresh-snapshot admission is a substantial tradeoff. At batch 10,000, native
DWAL's mixed writer reached 2.03M transfers/s, while its readers delivered
101K reads/s; MDBX delivered 859K transfers/s and 8.26M reads/s. These writer
numbers must be read alongside the different reader progress. Snapshot capture
waits for the active DWAL writer; reads from an already captured snapshot do
not retain that lock.

Compared with the prior direct run, native DWAL improves write-only throughput
at every measured batch. At batch 1, however, both DWAL paths have lower mixed
writer throughput than their direct counterparts, and peak RSS rises from
about 388 MiB to 1.33 GiB. At batch 100,000, the native DWAL write advantage over
direct is about 36%; the shim advantage is only about 6%. These are single-run
comparisons, and the separately rerun MDBX mixed results also varied materially.
The native MDBX executable is byte-identical to the prior sweep. The report
retains the prior MDBX measurements in `comparison.json`.

Native MDBX's measured peak RSS and allocated storage were approximately
61 GiB at batch 1, falling to about 61 MiB RSS and 64 MiB allocated storage
at batch 100,000 under this configuration. DWAL's close/drain time was
45–152 ms, versus roughly 9–29 ms for the prior direct psitri paths. These
resource measurements are part of the comparison, not just the foreground
transfer rates.

Each path executes `api-bank-v1`: 100,000 accounts, one million transfers in
each timed write phase, 10,000 warmup transfers, eight-byte keys and balances,
and seed 12345. Each transfer reads two balances and writes both. There is no
appended transfer-history table. Batch sizes are 1, 10, 100, 1,000, 10,000, and
100,000; loading and warmup use the selected batch size too. The mixed phase
adds four readers, each taking a new committed snapshot every 100 reads.

Native MDBX uses WRITEMAP | UTTERLY_NOSYNC. The native DWAL and MDBX-shim DWAL
paths use the same Large Engine base, a 100,000-entry ART threshold, a 64 MiB
WAL threshold, a 1 GiB arena backpressure threshold, and two merge workers.
Both submit WAL bytes to the OS at every commit without a per-commit fsync.
Existing full barriers during WAL rotation and clean close remain enabled.
The base uses sync=none. No power-loss persistence claim is made for this tier.
The no-argument DWAL commit, which can buffer WAL bytes in process memory, is
not the mode used here.

All psitri paths disable read promotion and use a 256 MiB pinned-cache budget.
The native adapters set a 64 GiB database capacity limit; the shim retains its
allocator's default capacity. The dataset fits in RAM. These tests do not
measure larger-than-memory scaling, subtree snapshot creation, or replication.
Close/drain time, peak RSS, and allocated storage are retained with foreground
throughput so deferred work and memory costs remain visible.

A machine-wide recorder lock serializes engines; each process starts after ten
consecutive seconds at or below 3% aggregate CPU activity. The x86 benchmark
host was reserved from other builds and timings. The Mac remained busy
(17.6–19.2% in the availability check), so it has only functional checks here,
not performance measurements. Machine selection also changes OS, compiler,
and storage and should not be interpreted as an isolated ARM/x86 comparison.

The implementation fixes a reproduced WAL recovery failure: a 16-bit operation
count wrapped for a 100,000-transfer transaction's 200,000 updates. Format 2
uses a 32-bit count. Regression tests verify complete replay and rejection of
a torn large transaction, format-1 reading, and non-destructive refusal to
append an incompatible format. The rotation predicate also now counts
buffered WAL bytes. Explicit no-sync commit submits its record while it still
owns writer admission. The byte-value DWAL adapter is experimental; it does
not expand the public `psitri::engine` concept or expose persistent subtree
snapshots.

Validation before timing: 159 DWAL regression cases (602,375 assertions), a
Large Engine/DWAL snapshot-and-merge regression (20 assertions), both native
and shim functional bank checks, and 38 Python evidence/website checks passed.
The failed pre-fix recovery test and intermediate compiler errors are retained.
Each measured process also checks rollback, read-own-writes, held snapshots,
exact balances and account count, and clean-close/reopen state. Clean-close
reopen does not establish crash durability.

[Interactive comparison](comparison.html) · [Structured results](comparison.json)

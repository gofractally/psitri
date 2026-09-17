# The evidence behind the page

Compare databases under the conditions that matter to your application:
workload, data size, concurrency, memory, and durability. Each result below
identifies its tested configuration and revision. Benchmark numbers apply to
that build; use the same workload to evaluate the revision you plan to deploy.

[Download the structured data](../data/benchmarks.json). All numbers on the landing
page derive from this file and the archived records linked below.

## Bank transfers versus real libmdbx

**Date:** June 17, 2026. **psitri revision:** `7a3c8a5`.
**Hardware:** Apple M5 Max, 18 cores. **OS:** Darwin 25.3.
**Build:** Release, Apple clang, `-O3 -DNDEBUG`.

The same bank workload runs against native psitri engines and real libmdbx:

- 1,000,000 accounts, each initialized to 1,000,000 units.
- 2,000,000 balance-conserving transfers, triangular hot-account distribution.
- Batch size 1,000; four reader threads.
- Separate bulk load, write-only, readers-plus-writer, and readers-only phases.
- Readers-only phase lasts two seconds. Balance conservation is checked, then
  the engine drains and reports live storage and disk high-water.

**Durability is asynchronous in this comparison.** psitri commits in memory,
without per-commit fsync. libmdbx uses `WRITEMAP | SAFE_NOSYNC`, also without
per-commit fsync. These settings do not establish identical crash semantics.
They do not measure per-commit synchronous durability performance.

<!-- BANK_CHART -->

Balance conservation passed for every measured engine. Use the checkboxes to choose
the engines shown and the metric buttons to compare throughput or storage. The Large Engine and Small Engine chart labels
identify the `sal_art_ref` and `start_ref` configurations at revision `7a3c8a5`.
Experimental bars describe optional configurations, separate from the public
shared engine API contract.

### How to read the result

`551010 / 77214 = 7.14`: the Large Engine’s write throughput is 7.14× MDBX in this
run. Its disk high-water is `542 / 64 = 8.47×`. The Small Engine records 11.33×
write throughput and 7.14× disk high-water. Higher throughput is preferable;
lower disk high-water is preferable. The chart’s ratios always mean **the
reported value divided by MDBX**, including storage metrics.

Disk high-water includes allocation and retained copy-on-write storage. It is
not equivalent to settled live payload. A `0` live-byte report means that measurement was unavailable; the structured data represents
those entries as `null`, not a zero-size database.

### Reproduction and limits

Original command arguments, with the MDBX library path omitted:

```sh
bank-compare-bench 1000000 2000000 1000 4 /tmp/bank_cmp
mdbx-bank-bench 1000000 2000000 1000 4 /tmp/bank_mdbx
```

MDBX used map geometry up to 8 GB. Its exact source revision was not recorded. RAM capacity, exact compiler
version, CPU affinity, warm-up policy, repetitions, and variability were not
recorded here. There is one retained output per command, so no error bars are
shown. These measurements exercise native engines directly. Application results
through the MDBX compatibility layer require a separate measurement.

Raw outputs: [psitri engines](../evidence/bank_compare.log),
[real libmdbx](../evidence/mdbx.log).

## Additional benchmark backends

### Keep the API: original libraries and psitri shims

Compare **libmdbx with the psitri MDBX shim**, **RocksDB with the psitri RocksDB
shim**, and **SQLite with psitri-backed SQLite**. Each API gets a pair of adjacent
bars, so the comparison includes the cost of keeping that API's workflow.
Choose the architecture and API pairs to focus the view.

<!-- API_CHART -->

The validated x86 RocksDB pair records **2.34× write-only throughput** and
**3.05× write throughput with a reader**, with **2.91× reported file size**.
These ratios describe this run and API integration. Translation overhead and
the workflows imposed by an existing API can limit the optimization paths
available through psitri's native interface.

The ARM view has no published same-API pair yet. Native-engine measurements on
Apple Silicon do not include shim overhead. New comparisons use the same
[`api-bank-v1` driver](https://github.com/gofractally/psitri/tree/sal2/benchmarks/api_bank)
on both architectures, retaining every repetition and checking the resulting
state before admitting either side to the chart. Runs wait for an idle machine
and execute one benchmark process at a time.

The MDBX, RocksDB and SQLite shims pass the `api-bank-v1` functional contract
checks on ARM64 macOS and x86-64 Linux: rollback, held snapshots, read-your-writes,
exact balances, cardinality, concurrent readers and reopen. These functional runs
do not supply performance bars. Fresh paired measurements require idle hosts and
complete passing repetitions. See the [tested source pins and regression evidence](https://github.com/gofractally/psitri/blob/sal2/logs/2026-09-17_api-contract-fixes/README.md).

### Common native-engine workload

The [common bank comparison suite](fair-comparison.html) runs the maintained
Small Engine and Large Engine alongside real MDBX, RocksDB, SQLite, and DuckDB
with one workload and explicit durability settings. Its validation runs establish
that the adapters agree on the resulting state; they are not performance claims.

**SQLite and RocksDB participate in the same bank transfer test**, using the same
account data, transfer sequence, transaction batches, and correctness checks as
the Small Engine and Large Engine. Both passed the retained no-sync and full-flush
validation cohorts. Run the five-engine no-sync comparison with:

```sh
python3 scripts/fair_bank.py --machine my-runner --profile smoke \
  --engines small large mdbx rocksdb sqlite --durability none
```

Use `--profile performance --repeats 5` on a clean build and dedicated runner to
collect comparison results. The [comparison protocol](fair-comparison.html)
documents build dependencies, durability tiers, and the checks required before
publishing timings. The June chart above contains its recorded MDBX cohort;
RocksDB and SQLite need results from a new matched performance series to appear
alongside the maintained engines.

The separate integration suite runs bank workloads through real and psitri-backed
RocksDB/SQLite APIs, plus a TATP telecom workload through DuckDB and its psitri
storage extension. The retained integration records below measure those source
configurations.

| Interface | Benchmark targets | Retained evidence |
|---|---|---|
| RocksDB | `bank-bench-rocksdb`, `bank-bench-psitrirocks` | April 6 bank logs; both passed balance conservation |
| SQLite | `bank-bench-system-sqlite`, `bank-bench-psitri-sqlite` | April 6 bank logs; psitri-backed run passed, system SQLite run failed balance conservation |
| DuckDB | `tatp-bench --engine duckdb`, `tatp-bench --engine psitri` | Benchmark implementation exists; no retained result is included here |

These integrations use a separate source configuration from the maintained
Small Engine and Large Engine profiles. Their measurements do not belong in the
June native-engine chart: the April workload uses one million accounts, one
million transfer attempts, batch size 1, one concurrent reader, and no per-commit
sync on AMD EPYC-Turin. The June workload uses different hardware, keys,
transaction work, batching, and reader counts.

The April machine record identifies revision `ec383f2`, a clean Release build
with `clang++-20`, 16 logical CPUs, 121 GB RAM, Linux, and ext4. Full compiler and
dependency versions, repeated measurements, and variability are not recorded in
this archive. The throughput counters include skipped transfer attempts.

RocksDB and PsiTriRocks both passed this run's final balance check. The system
SQLite result **failed validation**, so its timings do not support a SQLite
speedup claim. The failure's cause is not established by the saved output.
The cohort status also records failed or incomplete runs for other configurations.

Raw evidence: [RocksDB](../evidence/integration-bank-rocksdb.log),
[PsiTriRocks](../evidence/integration-bank-psitrirocks.log),
[psitri-backed SQLite](../evidence/integration-bank-psitri-sqlite.log),
[system SQLite, failed](../evidence/integration-bank-system-sqlite.log),
[machine settings](../evidence/integration-bank-machine.json),
[cohort status](../evidence/integration-bank-progress.txt), and
[archive provenance and checksums](../evidence/integration-bank-provenance.json).

The [bank driver](https://github.com/gofractally/arbtrie/blob/ec383f2b976586d15c8e0000fd1b23614caa9329/programs/bank_bench.cpp)
and [DuckDB TATP driver](https://github.com/gofractally/arbtrie/blob/ec383f2b976586d15c8e0000fd1b23614caa9329/libraries/psitri-duckdb/benchmarks/tatp_bench.cpp)
identify the tested interfaces. Source access may be restricted.

## ART lookups

**Measured configuration:** four-byte / 32-bit direct-address handles, a 32 GiB
ordinary arena, Apple M5, June 11, 2026. This is a handle width, not a requirement
for a 32-bit process. The supplied Small Engine profile uses five-byte handles; the
ratios below do not measure that profile and the four-byte layout is not exposed
as a selectable public engine type.

<!-- ART_CHART -->

The record says runs were back-to-back with one process at a time. It does not
preserve absolute timings, an exact measured revision for these ratios, or a repeat
count. We publish the ratios as recorded and do not invent confidence intervals.
This is evidence of competitive point-read performance, not a claim to win every
ART operation. [Source measurement excerpt](../evidence/start_ref-art-excerpt.txt).

## Billion-key scale test

On June 13, 2026, revision `dd2967c` ran an experimental tiered engine:
`start_ver` hot delta feeding a `sal_art_ref` cold base. It inserted one billion
splitmix64 u64 keys and u64 values in **10.7 minutes**, with 100% probe hits.

The M5 Max machine had **128 GB RAM**. Final sampled RSS was **58.08 GB**.
The dataset was **RAM-resident**. This is a scale measurement of the experimental
tiered configuration, separate from the native Large Engine and beyond-RAM testing.

- Release, clang `-O3 -DNDEBUG -march=native`.
- Batch 1,000; delta high-water trigger 256 MB.
- One logical writer plus background merge, reaper, and release work.
- Every 10 million inserts: 100,000 random reads and 100,000 iteration steps.
- Last sample: 1,157 ns/read; 657 ns/insert for the last chunk, not the whole run.

```sh
DELTA_HI_WATER_MB=256 delta_art-scale-bench \
  1000000000 1000 10000000 /tmp/da1b_db
```

[Full probe log](../evidence/scale-1b.log). The landing page’s RSS line shows the
10-million-key sample followed by the 100-million-key intervals; the full series
is retained in the downloadable data.

## Applying these results

The bank comparison measures native engines without per-commit fsync. Match
those settings when reproducing it, or choose a synchronous policy on both sides
when that is your application's requirement. The ART chart measures the
four-byte configuration identified above. The tiered scale test uses 128 GB RAM.

The RocksDB and SQLite integration runs above are separate from the native
Small Engine/Large Engine comparison. Neither an API-compatibility claim nor a
point-read result predicts end-to-end application performance. Test your calls, data distribution, snapshot lifetimes, and failure
model using the [recording workflow](benchmarking.html).

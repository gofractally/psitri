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

| Measured configuration | Load accounts/s | Transfers/s | Reads/s with writer | Transfers/s with readers | Reads/s alone | Disk high-water MB |
|---|---:|---:|---:|---:|---:|---:|
| libmdbx | 588,685 | 77,214 | 10,431,636 | 67,695 | 11,624,799 | 64 |
| psitri Big: sal_art_ref | 3,005,255 | 551,010 | 15,233,451 | 479,190 | 22,483,905 | 542 |
| psitri Small: start_ref | 6,747,858 | 874,524 | 46,403,149 | 520,717 | 48,631,120 | 457 |
| Experimental: delta_art(sv→sar) | 1,002,327 | 97,167 | 47,025,679 | 80,570 | 45,759,946 | 675 |
| Experimental: delta_art(sr→sar) | 3,210,994 | 464,275 | 42,691,076 | 466,937 | 45,095,819 | 839 |
| Experimental: start_ver | 10,008,858 | 2,268,805 | 49,750,317 | 2,085,093 | 52,199,001 | 293 |
| Experimental: gcart | 3,405,586 | 1,143,136 | 55,639,758 | 522,395 | 63,810,691 | 186 |
| Experimental: gcart_spec | 2,869,963 | 1,138,597 | 36,102,343 | 553,568 | 42,982,589 | 154 |

Balance conservation passed for every row. The Big and Small chart labels
identify the `sal_art_ref` and `start_ref` configurations at revision `7a3c8a5`.
Experimental rows describe optional configurations, separate from the public
Small/Big API contract.

### How to read the result

`551010 / 77214 = 7.14`: Big’s write throughput is 7.14× MDBX in this
run. Its disk high-water is `542 / 64 = 8.47×`. Small records 11.33×
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

## ART lookups

**Measured configuration:** four-byte / 32-bit direct-address handles, a 32 GiB
ordinary arena, Apple M5, June 11, 2026. This is a handle width, not a requirement
for a 32-bit process. The supplied Small profile uses five-byte handles; the
ratios below do not measure that profile and the four-byte layout is not exposed
as a selectable public engine type.

| Case | psitri latency / libart latency | Interpretation |
|---|---:|---|
| Integer hit | 0.87× | 13% lower latency |
| Integer miss | 0.82× | 18% lower latency |
| Dictionary hit | 0.92× | 8% lower latency |
| Dictionary miss | 0.73× | 27% lower latency |
| Integer insert | 1.12× | 12% higher latency |
| Dictionary insert | 1.20× | 20% higher latency |

The record says runs were back-to-back with one process at a time. It does not
preserve absolute timings, an exact measured revision for this table, or a repeat
count. We publish the ratios as recorded and do not invent confidence intervals.
This is evidence of competitive point-read performance, not a claim to win every
ART operation. [Source measurement excerpt](../evidence/start_ref-art-excerpt.txt).

## Billion-key scale test

On June 13, 2026, revision `dd2967c` ran an experimental tiered engine:
`start_ver` hot delta feeding a `sal_art_ref` cold base. It inserted one billion
splitmix64 u64 keys and u64 values in **10.7 minutes**, with 100% probe hits.

The M5 Max machine had **128 GB RAM**. Final sampled RSS was **58.08 GB**.
The dataset was **RAM-resident**. This is a scale measurement of the experimental
tiered configuration, separate from the native Big engine and beyond-RAM testing.

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
when that is your application's requirement. The ART table measures the
four-byte configuration identified above. The tiered scale test uses 128 GB RAM.

A controlled RocksDB comparison is not available in this dataset. Neither an
API-compatibility claim nor a point-read result predicts end-to-end application
performance. Test your calls, data distribution, snapshot lifetimes, and failure
model using the [recording workflow](benchmarking.html).

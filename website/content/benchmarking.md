# Benchmarks that can be compared later

Keep each benchmark run in `logs/`. Use `scripts/benchmark.py`, a Python 3.10+ standard-library
tool that writes structured metadata beside raw stdout and stderr. Every run
gets a unique directory; rerunning never overwrites earlier evidence.

## Compare different libraries

Use the [common comparison suite](fair-comparison.html) for a matched bank
workload across the Small Engine, Large Engine, MDBX, RocksDB, SQLite, and DuckDB.
It checks exact account state, snapshot isolation, rollback, and clean reopen,
then retains each process and repetition. Start with:

```sh
python3 scripts/fair_bank.py --machine my-runner --profile smoke
```

The build instructions, explicit durability tiers and matched cohorts, resource accounting,
and API-wrapper comparison boundaries are in that guide. The recorder below
remains useful for other workloads and regressions within one configuration.

## Record a run

Build the benchmark in Release mode, then run from the repository root:

```sh
python3 scripts/benchmark.py record \
  --label bank-big-small \
  --workload bank-1m-2m-batch1000-readers4 \
  --machine dedicated-m5max-01 \
  --build 'Release -O3 -DNDEBUG' \
  --compiler 'Apple clang: record the exact --version output' \
  --durability 'no per-commit fsync' \
  --param accounts=1000000 --param transfers=2000000 \
  --param batch=1000 --param readers=4 \
  --param competitor_revision=not-applicable \
  --notes 'Record background load, warm-up policy, and data preparation here.' \
  --parser bank -- \
  ./build/bin/bank-compare-bench 1000000 2000000 1000 4 /tmp/bank_cmp
```

Replace the example compiler text with the actual compiler version. `--machine`
is your stable identity for the physical runner, not merely a processor family.
The tool also records the OS release, architecture, CPU, core count, and memory
where available. Record the actual comparison library revision in a parameter
for competitor runs. Use the same recorder with `mdbx-bank-bench` and `--parser bank`.

Each directory contains:

| File | Purpose |
|---|---|
| `run.json` | Schema version, command, cwd, Git revision, branch, dirty state, tracked-diff digest, untracked paths, hardware, build, compiler, durability, parameters, selected environment, timing, exit status, metrics |
| `stdout.log` | Raw standard output, streamed to disk while the process runs |
| `stderr.log` | Raw standard error, including errors from a failed run |
| `README.md` | Human-readable command and notes |

The metadata starts with `status: running` and is finalized after exit. A failed
command, failed balance check, or malformed metrics produces a failed record
and a nonzero exit. A hard kill may leave a `running` record, which cannot pass
a comparison. A dirty run is identified, not presented as a clean revision;
the diff digest does not replace committing the source needed to reproduce it.

Only `CC`, `CXX`, `CFLAGS`, `CXXFLAGS`, `LDFLAGS`, and `DELTA_HI_WATER_MB` are
automatically retained from the environment. Record other performance-relevant
settings with `--param`. The tool does not dump credentials or the full
environment. Raw output, commands, notes, and paths still require review before
publication. Capture does not automatically upload or commit a result.

## Parse other workloads

`--parser bank` understands the existing bank table, including its correctness
column. Missing live-byte instrumentation (`0` in the bank output) is omitted,
not treated as zero storage. Record repetitions separately.

`--parser json` expects stdout to be one JSON list:

```json
[
  {"engine": "big", "name": "lookup_p99", "value": 812,
   "unit": "ns/op", "direction": "lower"},
  {"engine": "big", "name": "write_rate", "value": 510000,
   "unit": "operations/s", "direction": "higher"}
]
```

Metric names must be unique within an engine, and values must be finite and
nonnegative. Send diagnostic logging to stderr. `--parser none` retains all
output and metadata for a workload without a parser; such runs cannot yet be
used for automated metric comparisons.

## Compare a baseline and a candidate

```sh
python3 scripts/benchmark.py compare \
  logs/BASELINE_RUN logs/CANDIDATE_RUN \
  --threshold 5 --report comparison.json
```

Exit `0` means all matched metrics stayed within the threshold; `1` means at
least one regression exceeded it; `2` means the evidence could not be compared.
Higher-is-better throughput and lower-is-better latency/storage are handled in
opposite directions. The default threshold is 5%, a policy choice rather than
a statistical significance test.

Comparisons require matching workload, machine identity, captured hardware/OS,
build, compiler, durability, parameters, selected environment, parser, and
command arguments. The executable basename must match; its build directory
may differ. Dataset paths are part of the arguments and must match too. Revision
and dirty status may differ because measuring a code change is the purpose.
Missing metrics, changed units, or changed direction fail closed.

Do not edit metadata to force mismatched runs to compare. If an engine, library,
compiler, or workload changes intentionally, start a clearly identified new
series. This tool compares regressions within a series, not arbitrary competing
engines. For head-to-head comparisons, retain each competitor’s own complete
record and publish the conditions together.

## A repeatable performance habit

1. Use a dedicated runner with controlled background load and stable power settings.
2. Commit the source and pin dependencies. Record the dataset and preparation.
3. Select durability and memory budgets explicitly. Measure correctness first.
4. Run several independent baseline/candidate repetitions. Record every run;
   summarize distributions as well as throughput, tail latency, RSS, and disk.
5. Keep the raw archives in `logs/` and link the result from the relevant change.
6. Review a representative series before promoting it to public benchmark data.

The GitHub workflow tests the recorder and website; it does not publish noisy
shared-runner timings as product evidence. The public site only copies a small
explicit evidence allowlist, never the entire repository or all raw logs.

# Benchmark evidence

When running performance benchmarks, retain the results. Use
`python3 scripts/benchmark.py record ... -- BENCHMARK_COMMAND` as documented in
`website/content/benchmarking.md`, or preserve equivalent evidence when adapting
a workload the recorder cannot parse.

- Keep raw stdout/stderr, failures, exact command, source/dependency revisions,
  build/compiler settings, hardware, durability, dataset/seed, thread counts,
  batching, and relevant environment settings under `logs/`.
- Record each repetition separately. Do not overwrite an earlier run or discard
  a slow result. Label smoke runs and dirty builds so they are not mistaken for
  release evidence.
- Use matched conditions for regression comparisons. Keep correctness checks,
  memory, storage, and durability tradeoffs alongside speed measurements.
- Commit meaningful benchmark evidence with the work it supports, after checking
  logs for credentials and unrelated personal data. Recording a run does not
  authorize publishing it on the public website.
- Public results are curated in `website/data/benchmarks.json` and explained in
  `website/content/benchmarks.md`. Preserve revision, workload, and experimental
  status when adding a result; do not generalize an archived win to all engines.

Website validation: `python3 -m unittest discover -s tests/website -v`, then
`python3 scripts/build_site.py`. The Pages workflow publishes only the generated
site artifact. See `website/README.md` for its publication allowlist.

# Public documentation

Write the site and API guides for developers evaluating psitri as a standalone
library. Focus on present capabilities and operating requirements. Keep internal
development history and application-specific adoption plans out of public copy.
Retain benchmark revisions, dates, and configuration limits in methodology.
Use “practically lock-free” with accurate synchronization boundaries, and explain
compatibility-layer overhead alongside “Keep the API. Rethink the engine.”

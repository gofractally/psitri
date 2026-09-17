# psitri marketing site

A static, responsive GitHub Pages site with no client-side framework, tracking,
third-party font service, or runtime package dependency. JavaScript enhances the
snapshot diagram, engine example, and benchmark explorer. Core content and
benchmark methodology remain usable without JavaScript.

## Build and preview

```sh
python3 -m pip install -r website/requirements.txt
python3 -m unittest discover -s tests/website -v
python3 scripts/build_site.py
python3 -m http.server 8080 --directory build
```

Open `http://localhost:8080/site/`. Relative links also work at the project Pages
path `https://gofractally.github.io/psitri/`. The builder checks local links,
fragments, duplicate IDs, image alt attributes, and root-relative asset paths.
For visual review, check desktop and mobile, every benchmark metric, the experimental-engine
toggle, the snapshot illustration, example switching/copy, menu, and details.

## Publication boundary

`scripts/build_site.py` copies only:

- `website/index.html`, `website/assets`, and curated `website/data`;
- the listed API Markdown guides, rendered as public HTML;
- the three curated methodology/integration guides in `website/content`;
- `examples/api/basic.cpp` and `examples/api/revisions.cpp`;
- the curated native bank and scale logs, integration bank logs with their
  correctness status and provenance, and a benchmark-only ART measurement excerpt.

It does **not** upload the checkout, all source code, all raw logs, private runner
metadata, or newly recorded benchmark runs. Review each new public asset. A
private repository can have a public Pages site when its GitHub plan supports
it; Pages access and repository visibility are separate choices. Source links
may require repository access. No license or public-source availability is
implied by this page.

## GitHub Pages setup

In repository **Settings → Pages**, choose **GitHub Actions** as the build source.
The `github-pages` deployment environment must allow the publishing branch.
`.github/workflows/pages.yml` builds, tests, and deploys changes from `sal2` (the
current development branch) and `main`. Once development moves to `main`, remove
`sal2` from the trigger and restrict the environment to `main`. PRs build and
validate without deployment permissions. Do not use concurrent publishing
branches for divergent versions of the site.

The deploy job uses only the generated `build/site` artifact. It needs
`pages: write` and `id-token: write`; repository contents are read-only.
If Pages is unavailable for the private repository’s plan, keep the generated
artifact and resolve hosting/access explicitly; do not change repository
visibility as an incidental deployment workaround.

## Updating evidence

`website/data/benchmarks.json` retains curated numeric data, provenance, and
unknown fields explicitly. The bank chart reads it directly. Keep the static
fallback, methodology tables, ART cases, and scale summary consistent when
updating it. Do not promote research measurements as current engine results.
The recorder is documented in [the benchmark guide](content/benchmarking.md).

Claims that must remain explicit until evidence changes:

- The portable API has one serialized writer; independent-root writes are a
  native SAL capability, while the Large Engine adapter shares its write session.
- The Large Engine’s nested savepoints have a narrower scope than arbitrary nested subtree
  transactions. Read-driven hot-object promotion is opt-in.
- The Large Engine is the disk-backed profile. Comprehensive crash/power-loss and sustained
  beyond-RAM validation is incomplete. The billion-key tiered run fit in RAM.
- The libart result measures four-byte handles, not the supplied five-byte Small Engine
  profile; keep it in configuration-specific methodology, not current-profile
  marketing. Inserts were slower in that measurement.
- The MDBX bank comparison identifies its measured revision, uses no per-commit
  fsync, and has a
  larger psitri disk high-water. Separate RocksDB/SQLite bank records use a
  different configuration and workload; do not merge them into that chart. The
  retained system SQLite run failed validation and cannot support a speedup claim.
  DuckDB has a TATP driver, without a retained result in the public dataset.

## Public audience and terminology

Write for developers evaluating a standalone database library. Describe present
capabilities, operating limits, and integration tradeoffs. Keep internal project
history, renaming rationale, and application-specific adoption plans out of the
landing page and public API guides. Benchmark dates, revisions, and tested
configurations belong in the methodology because they are necessary evidence.

Use **practically lock-free** for the concurrency design, with the precise
boundary in the operating guide: acquired snapshots traverse without node
mutexes; root publication/acquisition and some allocator/maintenance paths
synchronize. SAL's read lock is an atomic residency marker, not a blocking mutex.

Keep **“Keep the API. Rethink the engine.”** and explain the cost alongside it:
foreign API workflows add translation overhead and can limit optimization paths
available through the native API. Do not apply native benchmark results directly
to a compatibility layer.

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
For visual review, check desktop and mobile, every benchmark metric, the research
toggle, the snapshot illustration, example switching/copy, menu, and details.

## Publication boundary

`scripts/build_site.py` copies only:

- `website/index.html`, `website/assets`, and curated `website/data`;
- the listed API Markdown guides, rendered as public HTML;
- the three curated methodology/integration guides in `website/content`;
- `examples/api/basic.cpp` and `examples/api/revisions.cpp`;
- three raw logs and a benchmark-only excerpt of the historical ART status record.

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
  native storage capability under integration qualification.
- Big’s nested savepoints have a narrower scope than arbitrary nested subtree
  transactions. Read-driven hot-object promotion is opt-in.
- Big is the beyond-RAM integration candidate, not a fully release-qualified
  claim. The billion-key archived run fit in its 128 GB machine’s RAM.
- The libart result used historical four-byte handles; current Small defaults
  to five bytes. The archived inserts were slower.
- The MDBX bank comparison has no per-commit fsync, is historical, and has a
  larger psitri disk high-water. No current RocksDB speedup is asserted.

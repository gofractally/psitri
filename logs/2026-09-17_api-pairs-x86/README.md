# Repeated same-API bank comparisons on x86-64

Collection started September 17, 2026 on the cloud x86 runner. The other
compiler benchmark workload stopped at a result-publication boundary before
this collection began. No timed compiler child was suspended. The collection
uses one exclusive runner and requires ten continuous seconds at no more than
3% aggregate CPU activity before each benchmark process. Every idle sample is
retained with its repetition.

## Reproduction

From the clean cloud psitri checkout:

```sh
python3 scripts/api_bank_matrix.py --machine cloud-x86 \
  --build-mdbx ../build-mdbx --build-integrations ../build-integrations \
  --output ../results/approved-window --repeats 5 --idle-timeout 1800
```

The matrix runs a smoke pair before each performance pair, sequentially.
Performance uses 100,000 accounts, 100,000 completed transfers per phase,
10,000 warm-up transfers, batches of 1,000, four readers, eight-byte values,
seed 12345, a 256 MiB cache budget, a 64 GiB map limit, and durability `none`.
That policy requests no per-commit synchronization; it does not establish
identical power-loss semantics across libraries.

## Source and build provenance

- psitri and common driver: `5ba8186a9bb6c99f92262b9cc3cf77bc13c92e5b`.
- RocksDB/SQLite integrations: `cb3eccaa9af41b6606680fbe59359537b2f4ab20`.
- Release builds with Clang 20.1.8. Per-pair archives retain the exact binaries'
  hashes, descriptions, compiler output, CMake cache, compile commands,
  dependency identities, source snapshots, and full result settings.
- Clean-build logs and dependency provenance are also retained in
  `../2026-09-17_verified-api-collection/` and
  `../2026-09-17_api-contract-fixes/`.

All repetitions and smoke runs are retained, including any failure. Successfully
validated generated database directories are removed by the collector; their
logical and allocated sizes before and after close remain in each result.
Public bars are admitted only by `scripts/api_bank_evidence.py` after validation
of the complete paired series. They use all five repetitions' medians and show
minimum–maximum ranges. Storage bars use allocated bytes after close, in MiB.

## Outcome

All three smoke pairs and all 30 performance repetitions passed. All six
implementations produced final account hash `2427670804914841226` on every
performance repetition. The other cloud workload was released to resume after
collection and archive transfer completed.

Median shim/original ratios (larger is better for throughput; smaller is better
for storage):

| API | Write | Writes + readers | Reads + writer | Allocated storage |
|---|---:|---:|---:|---:|
| MDBX | 0.435× | 0.414× | 0.655× | 0.280× |
| RocksDB | 1.457× | 1.241× | 2.203× | 15.144× |
| SQLite | 0.194× | 0.243× | 0.000093× | 63.153× |

These results do not support a general compatibility-layer speedup claim. The
SQLite shim's mixed reader throughput is particularly low; passing correctness
checks does not establish competitive performance. Every result, including the
slow repetitions and storage overhead, is included in the public chart data.

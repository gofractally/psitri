# Matched API-pair collection dispatch

Driver/collector revision: `230e49b`. Both platforms use the same driver sources,
no forced sync, identical per-profile parameters, and five performance repetitions
per passing pair. MDBX native version on both hosts is `v0.13.11-0-gb862eb08`.
The matrix runs libmdbx/psitrimdbx, RocksDB/PsiTriRocks, then SQLite/PsiTri-SQLite.
It runs a smoke pair first and collects performance repetitions only if it passes.

Each host requires ten seconds at or below 3% aggregate CPU activity before
every child process, holding `/tmp/psitri-benchmark.lock` throughout collection.
Other builds were active at dispatch. Earlier attempts were held by the idle
gate and produced no timing samples; they remain in the adjacent archives.
The x86 preliminary attempt was interrupted while waiting so the complete matrix
could replace it. The current collection waits up to one hour for each idle window.

Dispatch paths (raw results remain on each runner until collection completes):

- Mac: `build/api-collection/mac/2026-09-17T192038-api-matrix-a307dff5/matrix.json`.
- Cloud: `psitri-api-bench-20260917/results/2026-09-17T192039-api-matrix-99f63327/matrix.json`.

No fresh performance claim is supported by this dispatch record. Review completed
matrices and all raw errors, preserve their metadata and repetitions, and curate
only complete validated pairs into the public architecture views. Do not publish
a single-sided win or replace missing measurements with native-engine numbers.

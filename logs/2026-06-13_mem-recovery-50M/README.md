# memory + recovery — delta_art vs standalone parts (50M u64)

ONE engine per process (peak RSS is process-wide). Load 50M splitmix64 u64
keys (batched 1000/txn), record peak RSS + on-disk file, then CLEAN close and
REOPEN, timing recovery + verifying a 1000-key sample survived.

## Conditions
- cmd: `mem-recovery-bench <engine> 50000000 /tmp/memrec50`
- git: gofractally/psitri @ 4b9c20b ; build Release -O3 -march=native
- machine: Apple M5 Max, 18 cores, 128 GB RAM, darwin/arm64
- engines: start_ver (stamp ART, standalone), sal_art_ref (heart, standalone),
  delta_art<start_ver hot, sal_art_ref dwal cold> (the tiered pipeline)

## Result
```
engine        load 50M     peak RSS   file      reopen    verify
start_ver     103.1 s      12.77 GB   12.78 GB  0.008 s   OK
sal_art_ref    68.9 s       8.72 GB    8.81 GB  0.155 s   OK
delta_art      21.8 s       8.35 GB    8.44 GB  0.084 s   OK
```

## Read
- **delta_art is BOTH the fastest to load AND the lowest memory**: 21.8 s vs
  start_ver's 103.1 s (4.7x) and sal_art_ref's 68.9 s; 8.35 GB vs 12.77 GB.
- **start_ver ALONE degrades at this scale**: 50M is past its no-flush knee
  (~46M keys) — the unbounded delta's COW garbage + O(nseg) claim_ scans
  bloat it to 12.8 GB and crater load to 103 s. delta_art rescues it: the
  delta stays bounded and the sorted bulk merge re-clusters into the compact
  base, so the tiered pipeline is faster AND smaller than its own hot engine
  run standalone. The tiering thesis on memory + load.
- **recovery is fast for all (8-155 ms) — no WAL**: the committed root is the
  durable state, so reopen = mmap + root-table read (no replay). start_ver's
  STAMP model reopens fastest (8 ms: the version is one root-table slot);
  delta_art (84 ms) opens both tiers + checks the warm slot; sal_art_ref
  (155 ms) reattaches heart's SAL state. Clean-close data verified intact.

## Note on start_ref
Excluded from this run: start_ref's per-line refcounts are "derived, never
synced" (arena.hpp), so a true crash needs a refcount REBUILD from the
committed roots (not implemented) — a different recovery path than start_ver's
stamp-in-root-table. The liveness model (stamp vs refcount) IS the recovery
story: stamps reopen instantly; refcounts need a rebuild.

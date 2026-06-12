# Benchmark run archive

One directory per run: `YYYY-MM-DD_<engine>-<bench>-<scope>/` containing a
`README.md` (exact conditions: command, env, build, git commit, machine, plus
the headline numbers and findings) and the raw log(s). The point is tracking
evolution over time — when a number moves, the conditions that produced the
old number are right here.

Conventions:
- Raw bench output is committed verbatim (`bench.log`, plus any side logs like
  `pagestats.log`). One status row per commit-chunk; columns:
  `keys, ins ns, rd ns, hit%, iter ns, file GB, free%, rss GB, chunk s`.
- The shared workload (`*-scale-bench` family, all three engines emit the
  identical format so curves overlay): keys = `splitmix64(i)` (bijective u64 →
  unique, uniformly random order), values = 8 bytes of `splitmix64(i ^ C)`;
  inserts committed in chunks; after EVERY chunk, 100k random point reads over
  everything inserted so far + 100k iteration steps from a random lower_bound
  (wrapping at end). `hit%` must be 100.0 — anything else is a correctness bug.
- Record the git commit of the BUILD, env knobs (`HEART_PINNED_MB`,
  `HEART_PAGE_BYTES`, …), and whether anything else ran on the machine.

## Index

| date | run | headline |
|---|---|---|
| 2026-06-10 | [part-scale-392M-address-wall](2026-06-10_part-scale-392M-address-wall/) | part stops at 392M keys: 4-byte line_ptr = 28-bit line index = 32 GB hard ceiling |
| 2026-06-10 | [part-ptr48-ab-20M](2026-06-10_part-ptr48-ab-20M/) | 48-bit handle costs +0.8% memory, +11-15% point reads; cb_hdr is the one layout blocker |
| 2026-06-10 | [heart-burst-ab-170M](2026-06-10_heart-burst-ab-170M/) | 2KB pages @100k-op commits burst 4.2→16.7GB; 1KB pages: no burst, 29% free flat |
| 2026-06-10 | [heart-scale-1B](2026-06-10_heart-scale-1B/) | 1B keys in 23.1 min; ins 1.38µs / rd 566ns / iter 16ns flat; file 40GB = 1.39× live |
| 2026-06-10 | [psitri-scale-262M-control](2026-06-10_psitri-scale-262M-control/) | psitri on the identical workload: 1.26× live, free% monotone — compactor never the bottleneck |
| 2026-06-11 | [heart-scale-2.8B-beyond-ram](2026-06-11_heart-scale-2.8B-beyond-ram/) | thrash onset ~100GB RSS (78% of RAM); floor ~20µs = QD1 read-before-write fault; 160MB/s pageins |
| 2026-06-11 | [hkv-scale-550M-in-memory](2026-06-11_hkv-scale-550M-in-memory/) | u48 hkv in-memory: rd 130-235ns flat to 450M (fastest measured); ~165B/key caps 128GB RAM at ~550M — 2B needs 330GB |
| 2026-06-11 | [walart-v1-bank-100k](2026-06-11_walart-v1-bank-100k/) | walart v1 validates: 461k tx/s b=1 (≈part native, 1.6× heart); dwal incumbent still ahead (754k) on bump-arena delta |
| 2026-06-12 | [dwal-engine-templating](2026-06-12_dwal-engine-templating/) | dwal templated on kv_engine: Heart-DWAL first run 633k tx/s PASS; delta A/B: hkv +17% foreground, art_map 2× drain — swap not worth it |
- 2026-06-11_vpart-bank-1m/ — vpart L1a stamp-liveness first matrix: beats heart on every metric (+31% writes, +79% conc reads); part A/B reads +9..14%, writes −7.5%
- 2026-06-11_dict-matrix/ — 5-engine load/find/iterate: heart flagged 1.5-3.4x off (point-op gap, needs sample review); artmap iterate best-in-class; dispatch trade measured

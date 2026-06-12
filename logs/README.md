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
- 2026-06-11_dense-ab/ — dense-tiers A/B: no static winner; dense +19.5% on cache-resident bank, straight +9-21% dict reads; placement design owns the choice
- 2026-06-12_part-backport-ab/ — part back-port A/B (identical engineering): 100k writes PAR, 1M vpart +7.7%w/+17%r-solo, dict part load 132.9->98.2; deltas now attribute to liveness scheme + form choice
- 2026-06-12_vpart-space-census/ — vpart +50% space EXPLAINED (straight-widen form, not stamps: 4B/leaf sub-line only); dense default flipped; 6-row matrix + census; bank PROVISIONAL (co-tenant load 4.5+)
- 2026-06-12_pipeline-bench-v1/ — L5 prototype: 1GB delta flushes to heart in 4.6s (2.2M keys/s, 5x heart random rate); writer 537k tx/s concurrent; retire 4.2M keys/s
- 2026-06-12_sustained-matrix/ — sustained throughput: clean baseline 677-743k (537k was co-tenancy); 512MB delta +10%; PACING INVERTS (burst beats trickle — L5 hysteresis validated); qos useless, repin free
- 2026-06-12_tiered-bank/ — L5 pipeline gate PASSED: tiered 439k tx/s @1M = 103.8% of raw vpart, live bursts + retire pump, validation exact
- 2026-06-12_keymode-matrix/ — key-mode disambiguation: Dan's 484k = xxh64 band (part 450k, vpart 530k, tiered 526k today); all prior reports were names mode
- 2026-06-12_lineage-matrix/ — full lineage: tiered +77%w/+80%r over raw heart; legacy dwal +30% writes over tiered (heap+WAL delta vs persistent COW delta) but -37% reads; psitri raw 160k floor

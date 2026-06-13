# delta_art 1B — start_ver hot delta -> sal_art_ref cold base (#28)

First 1B run of the bounded-delta tiered pipeline (post per-root-floor GC fix).

- cmd: `DELTA_HI_WATER_MB=256 delta_art-scale-bench 1000000000 1000 10000000 /tmp/da1b_db`
  (1B splitmix64 u64 keys -> u64 vals, batched 1000/txn; 100k rd + 100k iter
  probe every 10M on the LATEST merged view)
- machine: darwin/arm64, Release -O3
- engine: delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>

## Result
```
1,000,000,000 keys in 10.7 min, 100% hit throughout
insert ~250-990ns (climbs as the base tree deepens); read ~250ns->1.2us
final file/RSS 58.2 GB ; tiers: delta 14 GB, base 36 GB
delta arena: 5.14 GB used / 15.17 GB total (34% used)
delta segs:  grown=1808  reused=20,130,288 (reuse 100%)
```

## vs the raw-heart 1B baseline ([[project_heart_1b_milestone]])
- heart alone: 1B in ~23 min, thrash onset ~100 GB RSS (cold random
  read-before-write faults).
- delta_art:   1B in 10.7 min (~2.1x faster), 58 GB RSS (~half), NO thrash —
  the hot delta absorbs writes; the sorted bulk merge re-clusters into heart.
  The tiering thesis at scale.

## Remaining lever
The per-root floor holds reuse at 100% (20M reuses) and bounds the delta well
at 100M (3.7 GB). At 1B the delta high-water still climbs to ~15 GB (34% used)
— the DRAIN THROUGHPUT now lags during the longer flush windows at scale (the
base tree is deeper -> each flush takes longer -> more hot garbage accrues
before the single drain frees it). That's the second half of audit #31 (faster
/ parallel drain), not the floor-gating (fixed). It does NOT thrash or hit the
address wall — the delta recycles, just at a higher plateau.

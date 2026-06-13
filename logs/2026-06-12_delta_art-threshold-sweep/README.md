# delta_art threshold sweep — does keeping the hot delta small win?

Dan's thesis: a big hot delta degrades writes (COW churn, locality, claim_
scans), so flushing early/often (small delta) should be faster.

## Finding 1 — overwrite workload (bank, 1M accts / 10M transfers)
Throughput FLAT across 16-512 MB (~2.0-2.26M tx/s). The live set is bounded
(same 1M accounts), so the delta never grows; the threshold only changes
WRITE-AMP: small marks flush often -> bigger base file (16MB -> 4.8GB), big
marks never flush at this scale (512MB -> 0.66GB ~= start_ver standalone).
=> for overwrite-heavy workloads a LARGER delta is better (less write-amp).

## Finding 2 — growing keys (50M distinct), marks 32-1024 MB
ins ns flat & noise-dominated (289-672 ss), file 12-24 GB. None of these
marks lets the delta reach the degradation regime: from the no-flush
isolation result start_ver only degrades past ~46M live keys (~2+ GB), but a
1 GB high-water flushes every ~20M keys. So all marks keep the delta "small
enough"; the variance is background-merger flush timing, not delta size.
=> the threshold is a MEMORY/write-amp knob here, not a throughput knob.

## To surface the cliff
The threshold must let the delta grow past ~46M keys (multi-GB high-water),
OR compare delta_art@small-mark vs start_ver STANDALONE (unbounded delta) at
100M+ keys where the standalone delta degrades and delta_art stays flat.
(see the high-threshold 80M-key run in this dir / the scale bench)

## Finding 3 — the delta footprint is a GC problem, not a threshold problem
100M scale run @256MB hi_water (delta_art-scale-bench): 0.8 min, ~480ns flat
insert, 100% hit. Per-tier file: DELTA 10GB, BASE 7.3GB.
- base = 7.3GB / ~5GB live ~= 1.46x: heart reclaim healthy; the merge already
  does N sorted COW upserts in ONE txn per chunk (Dan: that's fine).
- delta = 10GB for a 256MB flush TARGET. Occupancy probe @30M: 5.07GB used /
  13.52GB total = 37% used, 63% FREE.
  => reclaim WORKS (63% free) but the file high-water climbs because the
     writer grabs NEW segments instead of reusing freed ones. reuse_threshold
     is NOT the lever (0.25 == 0.80, both ~9GB) — the claim_ path isn't reusing
     freed segments (the O(nseg) claim_ scan poisoning = audit #31 GC rework).
  => plus a drain backlog: 5GB "used" >> 256MB live window = drain/reaper
     behind the supersession + reaped-warm-tree volume.
FIX = the #31 start_ver GC rework (O(1) bucketed claim_ so freed segments are
reused; faster drain) — "the reclaim GC engine should be more aggressive".

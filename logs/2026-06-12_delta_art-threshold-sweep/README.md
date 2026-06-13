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

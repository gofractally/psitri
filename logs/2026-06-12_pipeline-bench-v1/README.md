# pipeline-bench v1: vpart hot delta -> heart sorted bulk flush (L5 prototype)

Conditions: 2026-06-12 ~02:30, macOS arm64, Release -O3, machine QUIET
(parallel sessions ended; load < 1.5 during runs). Git ff5f1ca6.
Cmd: pipeline-bench --keys K --sustain S [--matrix]. Keys = 16-hex xxh64
(uniform random order), values 64 B. vpart dense default; heart per
radix_engine.hpp; flush = ONE pinned read-session cursor (one consistent
version for the whole iteration) -> heart txns of --chunk keys.

## 10M keys (~1 GB delta — Dan's flush scenario)
Phase A  vpart delta writes :  902,380 tx/s   (heart direct: 410,276 = 2.2x)
Phase B  bulk flush -> heart: 2,173,451 keys/s  (10M keys in 4.60s)
Phase C  sustained writer   :  536,561 tx/s   (flusher concurrent 2,783,258 keys/s)
Phase D  retire (sorted rm) : 4,187,613 keys/s (10M in 2.39s, batch=4096)
Phase E  stable read (heart):  810.6 ns/op    (latest post-retire 824.1)

## 1M keys + chunk matrix
A: vpart 1,416,462 tx/s vs heart direct 472,780 (3.0x)
B: chunk 10k = 2.46M | 100k = 3.15M | 1M = 3.14M | 10M = 3.16M keys/s
   -> flush plateaus at chunk >= 100k (matches Dan's psitri batch-100k note;
      the chunk x base matrix incl. psitri is the recorded follow-up)
C: writer 874,053 tx/s + flusher 3,228,510 keys/s concurrent
D: retire 4,597,337 keys/s   E: heart 425.3 ns, fall-through 376.5 ns

## Verdict
- Flush >= 4-5x heart's random direct write rate (dedup did not even engage:
  every key written once; overwrite-heavy workloads only improve this).
- Keep-up margin ~5x: flusher drains far faster than the writer fills.
- Retire ~ 1/3 the cost of a write, sorted batches.
- Integrity: post-flush probes 200k/200k found in heart, both scales.
Known gaps in v1: sustained-mode retire (needs remove-if-stamp<=F engine
primitive), tombstone pass-through, latest-view merge cursor, psitri base.

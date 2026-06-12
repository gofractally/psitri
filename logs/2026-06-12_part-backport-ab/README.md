# part back-port A/B: identical engineering, isolate the liveness scheme

Conditions: 2026-06-12, macOS arm64 (M-series), Release -O3, per-process runs
(no co-tenancy), names key mode, batch=1, sync=none, 1 reader thread.
Git: f0c6357e (part back-port: freeset ctz + alloc_c<N> chain + fcache stats
symmetry + template<Assign,Shared> insert with fused per-line fx/cx COW).
vpart at 926a7ca7 (template<Shared>). Cmd: bank-bench-{part,vpart}
--num-accounts {100000,1000000} --num-transactions {200000,10000000};
dict-bench --engine E --keys dict|rand.

## What the port did to part

| metric                      | pre-port  | post-port | delta  |
|-----------------------------|-----------|-----------|--------|
| bank 100k/200k write-only   | 442,358   | 462,631 (med 2) | +4.6% |
| bank 1M/10M write-only      | 356,562 (med 3) | 357,418 | ~0 |
| dict load ns                | 132.9     | 98.2      | -26%   |
| dict hit ns                 | 133.6     | 132.8     | ~0 (reads untouched) |
| rand load ns                | 35.4      | 31.8      | -10%   |

The dict-load win is the fused walk: privatize+insert was TWO descents per
op, now ONE (template<Shared>: Unique zero-work, Shared eager-clone with the
retain-latch). At 1M bank the write path is clone-volume-bound, so the
descent saving vanishes; at 100k (compute-bound) it pays.

## Remaining deltas = the isolated variables (same engineering both sides)

| metric (today, same machine)   | part      | vpart     | vpart vs part |
|--------------------------------|-----------|-----------|---------------|
| bank 100k write-only (med 2)   | 462,631   | 462,155   | PAR           |
| bank 1M write-only             | 357,418   | 384,858   | +7.7%         |
| bank 1M write+read             | 340,949   | 378,460   | +11.0%        |
| bank 1M reads w/ writer        | 4,199,936 | 4,509,515 | +7.4%         |
| bank 1M reads solo             | 4,759,391 | 5,586,178 | +17.4%        |
| bank 1M live MB                | 1833.0    | 1833.4    | PAR           |
| bank 1M file MB (high-water)   | 2391      | 2973      | -24% (part)   |
| dict load / hit / miss ns      | 98.2 / 132.8 / 48.8 | 92.6 / 112.1 / 37.8 | +6% / +18% / +29% |
| rand load / hit / miss ns      | 31.8 / 48.5 / 7.6   | 32.3 / 40.3 / 5.0   | PAR / +20% / +52% |
| rand live MB / iter ns         | 35.5 / 9.1 | 57.0 / 13.0 | part wins (density) |

Attribution (now clean):
1. READS: vpart +18-20% hit = part's fx/cx demotion (explicit lines add a
   dependent load). Demotion exists BECAUSE refcounts make implicit-node
   clones cost 256 child retains; stamps make the straight 10-line memcpy
   free of that. This is the liveness scheme's read cost, not impl slop.
2. WRITES at scale: vpart +7.7% = retain/release traffic + per-line COW
   ref checks in the clone path (clone-volume-bound regime).
3. WRITES cache-resident: PAR — the schemes cost the same when the working
   set fits; the fused walk was the actual gap, now equalized.
4. MEMORY: live PAR. part's file high-water 24% lower (refcount cascades
   free eagerly; vpart drains on the version floor — known GC lag).
   rand live: part 35.5 vs vpart 57 MB = dense ladder packs low-fanout
   routers; vpart straight-to-full trades footprint for hit latency.

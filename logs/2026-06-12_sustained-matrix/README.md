# Sustained-throughput matrix: delta size x repin x qos x pace (Dan's named goal)

Conditions: 2026-06-12 ~03:00, machine QUIET (load <1.5), Release -O3, git
3b7a14a3. pipeline-bench --keys K --sustain K [levers], values 64 B,
key=16-hex xxh64. Phase C = writer commits K txns WHILE flusher streams the
K-key snapshot to heart (chunk=10M default / 1M when paced).

## Results (Phase C sustained writer tx/s; flusher keys/s constant ~2.9-3.1M)

| config            | 5M keys (~512 MB) | 10M keys (~1 GB) |
|-------------------|-------------------|-------------------|
| solo (Phase A)    | 1,154,575-1,257,881 | 972,395-1,140,909 |
| base              |   716,289         |   677,266         |
| repin             |   743,090         |   671,165         |
| qos               |   733,673         |   553,349*        |
| repin+qos         |   738,541         |   602,808         |
| repin+pace 800k   |   443,035 (flusher 800k)  | — |
| repin+pace 1.2M   |   552,577 (flusher 1.2M)  | — |
*that whole run was slow (Phase A 972k) — variance, not a qos effect.

## Findings
1. CLEAN BASELINE: sustained = 677-743k tx/s (the earlier 537k was co-tenant
   pollution). 512 MB delta (Dan's hypothesis) confirmed: +8% solo, +10%
   sustained vs 1 GB.
2. PACING INVERTS: throttled flushers LOWER writer average. Flat-out, the
   flusher finishes its 5M keys in ~1.6s and the writer runs SOLO for the
   rest; paced, the writer co-runs the whole window. Paced rows measure the
   TRUE continuous co-run rate (~550k at flusher 1.2M) — the equilibrium if
   merge ran as a trickle.
3. THEREFORE: BURST scheduling maximizes sustained throughput — exactly the
   L5 hysteresis policy (flush 1 GB flat-out at hi-water, idle between).
   Steady state = writer ~1.2M solo between bursts, ~550k during short
   bursts; duty-cycled average ~716-743k at 512 MB deltas.
4. repin (per-chunk pin) ~free, tiny win at 5M — keep it (drain advances).
   qos: no benefit. The co-run tax is memory bandwidth, not scheduling, and
   not the pin (repin would have moved it otherwise).
5. Lever ranking: burst scheduling >> delta size (512 MB) > repin > qos(0).

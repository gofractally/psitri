# Tiered scale-workload hi-water sweep — found the real bottleneck (retire tax)

Conditions: 2026-06-12 ~10:05-10:30, post claim_-two-tier + flush-repin fixes
(git 4ab40... see parent commits), u64->u64 splitmix keys, batch=1000, probes
per 1M. 100M keys per point at TIERED_HI_WATER_MB in {256,512,1024,2048}.

## The 512MB curve (completed; others equivalent up to their knees)
  0-29M : ins ~480ns steady, rd ~100ns, file growing (PURE DELTA — no flush:
          hi_water counts RAW KV BYTES (17B/key -> 30M keys) while the arena
          footprint is ~15x that. The knob measures the wrong thing.)
  32-38M: flush#1 co-running: ins 1.6-2.2us (the known co-run tax)
  41-47M: RETIRE PUMP WINDOW: ins 12us(!) — every 1000-insert commit pays a
          4096-remove pump txn, and each remove COWs a path like an insert.
  50M+  : steady state 3.3-4.5us ins / 320-470ns rd — continuous
          flush+retire cycling. ~250k inserts/s sustained = LOSES to raw
          heart at scale (1.38us). The pipeline thesis holds only until the
          first flush; per-key retirement is the structural bottleneck.
  256MB : died SILENTLY at 16M keys — second occurrence, reproducible,
          uninvestigated (no exception text; needs a foreground run).

## Conclusions
1. Hi-water dial-in is MOOT until retirement is restructured: per-key
   removes make retirement cost ~= insertion cost (2x write amplification
   inherent). The L5 v2 answer is SUBTREE retirement: max-child-stamp hint
   in router slack -> unlink whole all-<=F subtrees in one path mutation +
   cascade release (insert-only workloads retire ~everything wholesale).
2. hi_water must track ARENA FOOTPRINT (segments held), not raw kv bytes.
3. retire_batch (4096) vs user batch (1000) = 4-10x pump tax per commit；
   must scale relative to user ops, and ideally run on merge-idle windows.
4. 256MB silent death: open bug.

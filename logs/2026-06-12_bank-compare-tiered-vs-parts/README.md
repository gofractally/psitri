# bank compare — delta_art<start_ver,sal_art_ref> vs each part alone

Same transfer workload + balance check on the composed pipeline and on each
constituent engine standalone.

- cmd: `DELTA_HI_WATER_MB=64 bank-compare-bench 1000000 5000000 1000 4 /tmp/bank_cmp`
  (1M accts x 1,000,000 initial, 5M transfers, batch 1000, 4 readers)
- machine: darwin/arm64, Release -O3
- note: at 1M accts the hot delta fits in RAM, so start_ver standalone is the
  unconstrained CEILING; delta_art flushes ~8 cycles at 64MB high-water.

## Result (all balance PASS)
```
engine                  load/s        wtx/s     rd/s(+wr)  wtx/s(+rd)   rd/s(solo)  balance
delta_art(sv->sar)    10,159,865    2,003,351   32,546,657  1,749,971   20,801,327  PASS
start_ver             10,010,173    2,313,798   55,468,122  2,379,319   65,260,125  PASS
sal_art_ref            3,200,915      564,533   15,856,826    540,746   24,217,144  PASS
```

## Read
- start_ver alone = ceiling (pure hot, fits in RAM): 2.31M wtx/s, 65M rd/s.
- sal_art_ref alone = direct-to-durable-base floor: 0.56M wtx/s (4.1x slower
  writes than start_ver — heart COW-to-root + sal alloc per write).
- delta_art = 2.00M wtx/s: 87% of start_ver's write speed AND 3.6x faster than
  writing straight to sal_art_ref, while streaming durably into sal_art_ref.
  The merge tax shows in solo reads (20.8M, the 2/3-way cursor) but with a
  concurrent writer it still beats sal_art_ref (32.5M vs 15.9M).

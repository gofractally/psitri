# delta_art bank bench — start_ver (hot) -> sal_art_ref dwal (cold)

First bank/transfer workload over the concept-only delta_art pipeline.

- cmd: `DELTA_HI_WATER_MB=64 delta_art-bank-bench 1000000 5000000 1000 4 /tmp/da_bank`
  (1M accounts x 1,000,000 initial balance, 5M transfers, batch 1000, 4 readers)
- env: DELTA_HI_WATER_MB=64 (forces several rotation/flush cycles)
- git: gofractally/psitri @ delta_art bank bench commit
- machine: darwin/arm64 (Release -O3)
- engine: delta_art::tiered_kv<start_ver::kv::engine, sal_art_ref::radix::dwal_engine>

## Result (PASS — balance conserved through flushes)
```
bulk load:            9,988,197 accts/sec
write-only:           2,221,987 tx/sec
reader (with writer): 34,254,587 reads/sec  [writer 1,795,766 tx/sec]
reader (solo):        23,346,981 reads/sec
balance: 1,000,000,000,000 == expected over 1,000,000/1,000,000 accts
```

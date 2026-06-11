# vpart L1a first head-to-head: vpart vs part vs heart (bank, 1M/10M)

Conditions: 2026-06-11, macOS arm64 (M-series), Release -O3, names key mode,
batch=1, sync=none, 1 reader thread, runs sequential (no co-tenancy).
Cmd: bank-bench-{vpart,part,heart} --num-accounts 1000000 --num-transactions 10000000
Git: vpart L1a stamp-liveness (this commit); part/heart at same tree.

| Engine | Write-only tx/s | Write+Read tx/s | Reads w/ writer | Reads solo | file MB | live MB |
|---|---|---|---|---|---|---|
| vpart  | 344,300 | 352,358 | 4,959,100 | 6,554,589 | 3038 | 1836 |
| part   | 372,338 | 358,376 | 4,572,180 | 5,762,576 | 2391 | 1843 |
| heart  | 263,209 | 239,560 | 2,775,390 | 4,182,685 | 1120 |  665 |

vpart vs HEART: writes +31%, write+read +47%, concurrent reads +79%, solo +57%.
vpart vs PART:  reads +8.5% concurrent / +13.7% solo (no fullx/cx demotion —
the hot top of the tree stays implicit skip-CL0); writes −7.5% (node-grain
clones memcpy whole multi-line runs; per-line stamps are the recorded L1c
follow-up), high-water +27% (drain lag + 10-line husks — same follow-up).
All runs validate: balances, log entries, account counts exact.

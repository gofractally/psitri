# Tiered (vpart->heart) bank gate: the L5 pipeline through ONE kv_engine

Conditions: 2026-06-12 ~03:40, quiet machine, Release -O3, git 2d96eddd+bank.
Cmd: bank-bench-tiered --num-accounts {100000,1000000} --num-transactions
{200000,10000000}. Knobs: hi_water 512MB, flush_chunk 10M, retire_batch 4096,
background merger ON (live bursts during the 1M run). names keys, batch=1.

| metric             | tiered 100k | tiered 1M | raw vpart 1M (same day) |
|--------------------|-------------|-----------|--------------------------|
| write-only tx/s    | 590,007     | 439,077   | 422,859                  |
| write+read tx/s    | 727,952     | 438,438   | 425,960                  |
| reads w/ writer    | 6,936,525   | 4,987,939 | 4,466,284                |
| reads solo         | 7,806,460   | 6,244,899 | —                        |
| file MB (total)    | 480         | 4054 (delta+base) | 3038            |
| validation         | PASS exact  | PASS exact (13,264,724 log entries, balance to the unit, 113.7M reads 100% found) | PASS |

GATE: >=90% of raw vpart required; measured 103.8% at 1M (the retire pump
keeps the delta SHALLOW — Dan's small-tree effect pays for the pipeline) and
~103% at 100k (no flush triggered: pure delta speed, envelope cost invisible).
Reads BEAT raw vpart at 1M: hot keys hit a small delta, misses land in heart.
First run ever with merger bursts + retire pump live under load: correct.

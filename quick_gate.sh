#!/bin/bash
# quick_gate.sh — sub-minute rotation gate for the dev loop (Dan: test runs
# must stay under ~1 min). Small N + SMALL HI-WATER so rotations fire MANY
# times: it's the machinery (freeze/flush/release/3-way reads) under test,
# not the scale. Full-scale runs are milestone-only.
set -e
B=${B:-$(pwd)/build/release/bin}
OUT=/tmp/qgate; rm -rf $OUT; mkdir -p $OUT; cd /tmp
# bank: 100k accts / 1M txns, hi_water 8MB -> dozens of rotations mid-load
d=/tmp/qg_bank; rm -rf $d; mkdir -p $d; cd $d
TIERED_HI_WATER_MB=8 $B/bank-bench-tiered --num-accounts 100000 \
  --num-transactions 1000000 --key-mode xxh64 > $OUT/bank.log 2>&1 || true
cd /tmp; rm -rf $d
# scale: 20M keys, hi_water 64MB -> ~5 rotations; probes per 1M
rm -rf /tmp/qg_sc
TIERED_HI_WATER_MB=64 $B/tiered-scale-bench 20000000 1000 1000000 \
  /tmp/qg_sc > $OUT/scale.log 2>&1 || true
rm -rf /tmp/qg_sc
echo "── bank ──"
grep -E "Write-only:|Reader:     |Total balance|PASS|FAIL" $OUT/bank.log | sed 's/^ *//'
echo "── scale (last 4 rows) ──"
tail -5 $OUT/scale.log

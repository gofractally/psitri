# Local functional checks

This busy Mac is used for correctness, not timing evidence.

```
cmake --build build/api-dwal --target fair-bank-tiered api-tiered-regressions -j4
SAL_LOG_LEVEL=ERROR build/api-dwal/bin/api-tiered-regressions
python3 logs/2026-09-17_tiered-bank-batching/local/run_functional.py
python3 -m unittest discover -s tests/website -v
```

`pair-tests-02.log`: 2 cases / 26 assertions passed. Tests cover snapshots across
explicit cold-base flushes, tombstone masking, abort, persisted hot and cold data
on reopen, and complete reader transactions during background migration with
one-key base commits. The concurrent test asserts that base commits occurred.
`pair-tests-01.log` retains the first run before adding that migration assertion.
The common bank driver's exact commands and binary hash are in `functional.json`;
all six checks passed. All 38 Python tests passed. Source was a dirty development
checkout; these logs do not qualify as performance evidence.

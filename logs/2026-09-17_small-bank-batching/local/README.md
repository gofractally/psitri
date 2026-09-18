# Functional checks on the Mac

These are correctness checks on a busy development machine, not performance evidence.
The exact bank commands and binary hashes are in `functional.json`.
Both Small Engine direct and DWAL over Small Engine passed all six checks and
produced the same final hash. `run_functional.py` recreates these checks; preserve
new runs in a separate directory rather than overwriting these observations.

Build commands (Homebrew Clang 22 Release, existing `build/api-dwal` configuration):

```
cmake --build build/api-dwal --target fair-bank-small fair-bank-dwal_small api-storage-regressions -j4
SAL_LOG_LEVEL=ERROR build/api-dwal/bin/api-storage-regressions '[dwal-byte]'
SAL_LOG_LEVEL=ERROR build/api-dwal/bin/api-storage-regressions '[dwal-byte],[dwal]'
python3 logs/2026-09-17_small-bank-batching/local/run_functional.py
python3 -m unittest discover -s tests/website -v
```

`byte-tests-01.log` retains the failing range-only buffer drain regression.
The fix changes the close-path emptiness check to include range tombstones.
`regressions-02.log`: 164 cases / 602,439 assertions passed.
The strengthened populated-base snapshot test then passed 5 cases / 72 assertions
in `byte-tests-03.log`. It exercises independent cursors, retained base roots,
rollback, background merges, range deletion, and reopen on both engines.

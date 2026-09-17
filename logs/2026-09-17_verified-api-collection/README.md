# Clean-build collection after API bug fixes

Main source: `5ba8186a9bb6c99f92262b9cc3cf77bc13c92e5b`.
Integration source: `cb3eccaa9af41b6606680fbe59359537b2f4ab20`.
Both machines rebuilt all six benchmark executables from these committed sources.
The functional regressions and corrected contract checks are in the adjacent
`2026-09-17_api-contract-fixes` archive.

The Mac retry timed out waiting for ten seconds below 3% aggregate CPU activity.
No benchmark child was launched. Other desktop work remained active and was not
interrupted. This archive retains the idle samples, exact build metadata and
commands. The collector stopped without advancing to another pair.

The x86 retry also timed out without launching a benchmark. Unrelated compiler
work consumed the machine during the idle wait. Both collectors have exited;
there is no background performance run or automatic retry. All checks are kept,
and no idle threshold was relaxed. These archives contain **zero new throughput
samples** and are not chart inputs.

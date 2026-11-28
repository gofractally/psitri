Small Engine gate (gofractally/psitri#42), 2026-11-28, macOS arm64, Release.
Counts taken from the stress logs of that day; the logs themselves stayed in the build directory.

Before the fixes of cf8e98e (main b637805), small-engine-tests under CPU load:
  release.counts.txt            full suite, 8 concurrent copies + 12 busy loops: 14 ok, 29 failed
  release.failing-assertions.txt  where those 29 failed (a run can fail at more than one place).
                                  Eight further runs were in flight when the loop was stopped and
                                  ended on SIGTERM; they are not counted.
  churn-release.counts.txt      "[churn]" only, 16 copies + 40 busy loops: 2,704 ok, 496 failed,
                                all at small_engine.cpp:565
After them:
  churn-fixed.counts.txt        the same churn conditions: 3,200 ok, 0 failed
  release-fixed.counts.txt      full suite, 4 copies + 10 busy loops, sanitizer jobs running
                                beside it: 56 ok, 0 failed (stopped there; lighter than before)
  crash-long-2000-cycles.txt    small-engine-crash-tests "[steps],[ceiling],[inplace]" with
                                PSITRI_SMALL_CRASH_CYCLES=2000 PSITRI_SMALL_INPLACE_ROUNDS=6000 at
                                commit e70b173. Its one failure is the harness fault corrected in
                                2540628 (a refused bookkeeping write after removes), not the engine.

Figures in commit messages on claude/small-engine-gate, and what stands behind them:
  cf8e98e  "29 failing runs of 43", "496 of 3,200": measured; the counts above.
  a47c0ea  "2,400 transactions and about 12,000 checks": NOT measured. The test then ran 2,400
           rounds and its counter did not include the per-operation checks. Measured since:
           1,200 rounds (the default) made about 13,600 checks and 6,000 rounds 68,817 at
           7a39719; since de3d6f8 the test also checks inside commit and the versions held
           by snapshots and readers, and a default run printed 20,381.
  7a39719  "13,649 checks for the default 1,200 rounds": measured in one run; the count varies
           by a few hundred from run to run with what the killed lives leave behind.
  2540628  "1,000-cycle run", "600 cycles passed afterwards": measured (crash-long-2000-cycles.txt
           is the first; the second was PSITRI_SMALL_CRASH_CYCLES=1200 "[ceiling]", output not kept).
  5faab80  "eight batches of removes": the limit in the code at that commit's parent was eight;
           that commit raised it to 64, which is what the code says.
  7b1f41a  "8,000 ms before": one run of the test with the fix disabled (8,000.8 ms).
Later harness changes (a5578cf, de3d6f8) changed what the campaigns do; counts printed by a run
of the current test are the ones to quote.

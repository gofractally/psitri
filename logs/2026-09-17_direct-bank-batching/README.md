# Direct engine batching diagnostic

One run per configuration; these are diagnostic measurements, not repeated release benchmarks.

On x86, native psitri and the direct MDBX shim lead at one transfer per commit. Native MDBX leads both at every measured batch from 10 through 100,000. At 100,000, native psitri reaches 1.202M transfers/s versus MDBX at 1.376M (87% of MDBX), while the shim reaches 662K. The gap narrows at the largest batches, but this sweep does not demonstrate a large-batch crossover.

The ARM sweep was interrupted before its first benchmark because the Mac never satisfied the idle gate. Its waiting log and interrupted manifest are retained under `arm/`; there are no ARM timings in this report.

100,000 accounts; 1,000,000 successful transfers in each timed write phase; 10,000 warmup transfers; eight-byte keys and balances; seed 12345. Each transfer reads two balances and writes both, without an appended transfer log. The selected batch size also governs loading and warmup. The mixed phase adds four readers, each refreshing its snapshot every 100 reads.

All paths use no per-commit sync. MDBX and its shim receive WRITEMAP | UTTERLY_NOSYNC. Both psitri paths use direct sal_art_ref, read promotion disabled, and a 256 MiB pinned-cache budget. Native psitri sets a 64 GiB capacity limit; the shim retains the allocator default. This resident dataset does not test capacity, persistent subtree creation, or crash durability.

Each process runs alone after ten consecutive seconds below 3% aggregate CPU activity. The runner retains idle samples, commands, binary hashes, source revisions, compiler flags, all raw output, and validation. Errors and interrupted attempts are retained separately. ARM/x86 comparisons also change CPU, OS, compiler and storage; they do not isolate instruction set.

[Interactive bar charts](comparison.html) · [Machine-readable comparison](comparison.json)

## X86: AMD EPYC-Turin Processor

Sweep status: passed. Source: `83cfcd044a6f0284127ba771016c933bdccdc850`. All completed configurations produced final hash `1346279285150323400`.

Write-only throughput, transfers/second:

| Transfers/commit | Native MDBX | MDBX shim | Native psitri | Shim/MDBX | Native/MDBX |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39,399 | 49,076 | 91,488 | 1.25× | 2.32× |
| 10 | 188,536 | 126,123 | 143,673 | 0.67× | 0.76× |
| 100 | 283,048 | 168,535 | 178,045 | 0.60× | 0.63× |
| 1,000 | 593,397 | 257,829 | 288,462 | 0.43× | 0.49× |
| 10,000 | 1,220,181 | 558,361 | 847,987 | 0.46× | 0.69× |
| 100,000 | 1,376,041 | 662,143 | 1,202,322 | 0.48× | 0.87× |

With four concurrent readers, writer throughput (active writer interval), transfers/second:

| Transfers/commit | Native MDBX | MDBX shim | Native psitri |
| ---: | ---: | ---: | ---: |
| 1 | 26,484 | 40,124 | 69,841 |
| 10 | 102,642 | 91,857 | 108,046 |
| 100 | 167,010 | 124,474 | 141,849 |
| 1,000 | 323,336 | 191,967 | 221,957 |
| 10,000 | 765,660 | 327,705 | 532,824 |
| 100,000 | 945,174 | 459,149 | 690,899 |

Each run checks read-your-writes, rollback, held snapshot stability, exact account state, conservation of balances, account count, and clean-close/reopen state. Reopen validation does not establish power-failure recovery.

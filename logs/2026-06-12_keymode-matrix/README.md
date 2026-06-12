# Key-mode disambiguation: part vs vpart vs tiered, names vs xxh64 (Dan's 484k)

Conditions: 2026-06-12 ~09:15, load avg 2.0-2.5 (not fully quiet — absolute
numbers ~5%-low; runs ALTERNATED so relative deltas hold). Release -O3, 1M
accounts / 10M txns, batch=1, sync=none. Cmd: bank-bench-E --key-mode M.
ALL PRIOR REPORTS THIS WEEK WERE names MODE (16-char strings).

| engine  | key mode | write-only tx/s     | reads solo |
|---------|----------|---------------------|------------|
| part    | names    | 361,899             | —          |
| vpart   | names    | 438,254             | —          |
| part    | xxh64    | 451,490 / 448,922   | 7,365,600  |
| vpart   | xxh64    | 529,897 / 529,610   | 8,640,753  |
| tiered  | xxh64    | 526,222 (99.3% of raw vpart, merger+pump live) | — |

Dan's prior baseline "part batch=1 = 484k" sits in the XXH64 band (vs 450k
measured today under load; not the 362k names number). Under xxh64:
vpart +17.6% writes / +17% reads over part; names: vpart +21% writes.
All validations exact (7/7 PASS).

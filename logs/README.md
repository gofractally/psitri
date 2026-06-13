# Benchmark log archive

One directory per run: `YYYY-MM-DD_<what>/` with a `README.md` (conditions:
cmd, env, git, machine, contention + result) and any verbatim `*.log`. Newest
first.

| date | run | headline |
|---|---|---|
| 2026-06-13 | [delta_art-1B](2026-06-13_delta_art-1B/) | 1B keys in 10.7 min, 100% hit, 58 GB RSS — ~2.1× faster than raw-heart 1B, no thrash |
| 2026-06-13 | [start_ver-per-root-floor](2026-06-13_start_ver-per-root-floor/) | per-root version floors: delta 10 GB → 4.2 GB @ 100M, reuse 97% → 100% |
| 2026-06-12 | [delta_art-threshold-sweep](2026-06-12_delta_art-threshold-sweep/) | flush threshold is a memory/write-amp knob until ~46M keys/cycle; GC, not threshold, was the lever |
| 2026-06-12 | [bank-compare-tiered-vs-parts](2026-06-12_bank-compare-tiered-vs-parts/) | bank: delta_art keeps 87% of start_ver's write speed, 3.6× over direct-to-sal_art_ref |
| 2026-06-12 | [delta_art-bank-start_ver-sal_art_ref](2026-06-12_delta_art-bank-start_ver-sal_art_ref/) | bank transfers + balance PASS through rotation/flush cycles |

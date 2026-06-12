# dict matrix: load/find/iterate across hkv/artmap/part/vpart/heart

2026-06-11, macOS arm64, Release -O3, sequential single-process runs.
Cmd: dict-bench --engine E --keys dict|rand. dict = /usr/share/dict/words
(236k var-len words, prefix-rich); rand = 1M x 8-byte binary. hkv/artmap =
raw in-memory baselines; part/vpart/heart = full engines (batch=8192
commits; finds via tx/cursor get<string> — string materialization makes
raw-vs-engine find-hit absolutes imperfectly comparable; within-group
relative numbers are clean). All engines cross-validated: identical
counts + key-byte checksums.

ns/op           dict:  load  hit  miss  iter   rand:  load  hit  miss  iter
  hkv (raw)            87.5  106   39   20.1          27.9   43   6.3  13.5
  artmap (raw)         96.6  154   60   12.2          36.6   41   5.1   2.7
  part                125.5  129   49   21.2          36.1   57   7.6   9.2
  vpart               128.1  124   48   21.5          37.4   51   6.5   9.4
  heart               291.4  193   76   22.3         124.8  145  31.4  10.4

FINDINGS (the matrix's purpose — technique wins + unreported gaps):
1. HEART has the pathological gap: 2.3-3.4x slower loads, 1.5-2.8x slower
   finds than part-family AT THE SAME ENGINE SURFACE, far off its own
   historical names-bench wins. Iterate is competitive (22.3/10.4) — the
   gap is point-op-shaped: suspect per-descent pin/get_ref machinery on
   the read path (the read_lock-pins-residency work) and write-path SAL
   per-node costs. NEEDS A SAMPLE-DRIVEN REVIEW like vpart's size-class
   find (which this kind of matrix caught).
2. artmap ITERATE is best-in-class (2.7 ns rand, 12.2 dict): fat node256 +
   dense sorted child arrays. Lesson: iteration loves big dense nodes.
3. Dispatch trade measured: header-first (artmap) loses on prefix-rich
   var-len keys (154 vs hkv's 106 hit) but WINS on dense fixed 8B keys
   (40.9 vs 43.1) — header cost amortizes when node256 direct-indexing
   dominates. Pointer-tag dispatch is the right default for mixed keys.
4. vpart out-reads part on BOTH keysets (124 vs 129 dict, 51 vs 57 rand):
   batch-boundary COW during load already demotes part's wide routers to
   the explicit form; vpart stays implicit (census-proven on bank).
5. find-miss is cheap everywhere (divergence early-out works in all five).

## v2 (same evening): fused privatize + u64 values + live/reserve columns

Changes since v1: (a) vpart's privatize pass FUSED into insert/remove
descents (clone-if-shared per hop, one walk instead of two) — dict load
128->98 ns (-23%), now 25% faster than part and within 8% of raw hkv;
(b) zero-copy callback finds for all engines (the get<string> per-hit
materialization was most of the "query gap" — with it gone vpart WINS
rand find-hit outright at 38.4 ns, ahead of raw hkv 41.5 and artmap 38.8);
(c) u64 values everywhere; (d) live bytes (tree walk) + reserve bytes
(allocator footprint) columns; (e) single-binary --engine all with an
aligned, color-coded, winner-marked table (tables: table_{dict,rand}_v2.txt).

Memory findings: HEART has the densest live tree (12.3 MB dict — its
design goal showing; the cost is indirection latency + 32MB-segment
reserve). part-family live is ~half of raw hkv at rand (35.5/39.3 vs 67.7
— line packing + leaf density). artmap pays 2-3.8x live bytes for its
iterate crown (fat node256 + rounded leaves). vpart live is +5% over part
(stamps + size-class rounding) — the price of versioning measured at last:
five percent of bytes, zero query cost.

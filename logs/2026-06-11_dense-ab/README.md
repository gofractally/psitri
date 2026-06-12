# vpart dense-tiers A/B (Dan: "show me with and without dense nodes")

Same build (358b0f36 + VPART_DENSE_TIERS flag), per-process runs.
straight = mode::none (widen straight to node_full); dense = mode::dense_tiers
(c2/c4/c8 ladder). Bottom-up COW + exact sizes + recycle ring in both.

DICT 236k:            load    hit   miss   iter  liveMB
  straight            98.3  116.7*  38.1*  21.9   17.5
  dense              102.2  127.5   48.0   21.8   17.2
RAND 1M:
  straight            33.9   41.1    5.2*  13.3   57.0
  dense               33.8   40.7*   6.2    9.1*  39.3*  (-31% live, -32% iter)
BANK 1M/10M:          writes    reads-solo   file MB   live MB
  straight            374,594   5,208,863    2,973     1,833
  dense               376,357   5,137,625    3,038     1,833    (wash)
BANK 100k/200k (cache-resident, the real-workflow shape):
  straight            441,783
  dense               527,990   (+19.5% — and past part's 442k)

VERDICT: no static winner — Dan's point proven. Micro-bench latency
(dict, tree-is-the-only-working-set) favors straight-to-full (+9-21%
reads); the cache-resident REAL workload favors dense (+19.5% writes at
100k; -31% live at rand). At big working sets it's a wash. The form
choice is genuinely per-node heat/density territory (the placement/mover
design), not a compile-time constant. Also feeds the allocator-waste
question: dense nodes (3/5/9 lines) pack L0 words better than 10-line
fulls (word confinement strands up to 4/64 lines mixing 10s).

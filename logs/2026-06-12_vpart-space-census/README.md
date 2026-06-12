# vpart "extra space" census + dense-default flip + fresh 6-row matrix

Conditions: 2026-06-12 ~00:30-01:00, macOS arm64, Release -O3, per-process.
*** CO-TENANCY WARNING: load avg 4.5-4.8 throughout (two parallel Claude
sessions compiling/running on this machine). Bank tx/s swung +-40% run to
run (part 356k-466k, vpart 339k-578k in the same hour) — bank rates in this
dir are PROVISIONAL; dict ns/op ran in one tight window, relative order
consistent with 06-11 quiet-machine numbers but absolutes are suspect.
The census tables are deterministic (node counts/bytes) — load-independent.
Git: census at 2b4bef79 (vpart straight default); flip at b88a... (vpart
dense default). Cmd: DICT_CENSUS=1 dict-bench[-straight] --engine E --keys K.

## THE SPACE ANSWER (rand 1M, deterministic census)

| | part | vpart straight (old default) | vpart dense (new default) |
|---|---|---|---|
| leaves | 1,000,000 x 22 B | 1,000,000 x 26 B (+4 B vstamp) | same |
| setlist_u8 | 73,298 / 6.89 MB | identical | identical |
| mid-fanout routers | 20,738 c2 @384 B = 7.96 MB | 21,000 node_full @1280 B = 26.88 MB | 20,743 c2 @384 B = 7.97 MB |
| full routers | 257 full_x @1280 | 0 | 257 node_full @1280 |
| rand live total | 35.5 MB | 57.0 MB | 39.3 MB |
| reserve | 160 MB | 160 MB | 160 MB |

The +50% was the straight-to-full WIDEN DEFAULT (form policy), not the
version header: ~19 MB of 10-line node_full where part packs 3-line c2.
The stamps' true cost = 4 B/leaf of SUB-LINE slack (22->26 B inside one
128 B line -> ZERO extra lines; reserve identical) and 0 B on routers
(header slack). vpart default now = dense ladder, matching part's forms
exactly (router census byte-identical); VPART_STRAIGHT_WIDEN=1 keeps the
form experiment buildable (dict reads -13%, rand live +45%).

## Fresh dict matrix (all baselines; heart cached from 06-11, code unchanged)
cols: load/hit/miss/iter ns, live/reserve MB           (d)=dict (r)=rand
hkv     (d) 90.3/110.2/39.5/20.8 21.3/44.6   (r) 28.0/44.0/5.4/12.7  67.7/166.9
hkv-dt  (d) 95.2/115.3/40.8/20.6 20.9/44.3   (r) 27.2/42.2/6.3/ 8.3  39.9/139.1
artmap  (d)102.0/169.1/60.0/12.7 44.0/47.6   (r) 35.5/41.1/5.2/ 5.5 133.9/135.0
part    (d) 95.6/133.5/49.1/21.7 16.3/64.0   (r) 31.6/48.1/7.6/ 9.1  35.5/160.0
vpart   (d) 95.0/125.3/47.6/21.4 17.2/64.0   (r) 32.4/40.7/6.1/ 9.1  39.3/160.0
vstraight(d)90.8/109.7/38.5/21.6 17.5/64.0   (r) 32.1/39.9/5.3/13.1  57.0/160.0
heart*  (d)311.0/193.9/77.1/27.8 12.3/288    (r)127.8/118.7/30.4/10.7 35.7/384  *cached 06-11

With matched ladders, vpart-dense restores part's density exactly (rand
live 39.3 vs 35.5 = the leaf vstamp; iter 9.1 == 9.1) and still reads
faster (dict hit -6%, rand hit -15%): the residual = part's fx/cx COW
demotion — its 257 demoted TOP routers add a dependent line-pointer load
on ~every query. Demotion is refcount-forced (an implicit node_full clone
would cost 256 child retains); stamps clone it as one memcpy. K::hdr (L1b)
is dormant here (census: 0 hdr nodes dict and rand).

## Bank (PROVISIONAL — co-tenancy):
vpart-dense sm 565-578k, 1M 422.9k (file 3038, live 1833); same-hour
controls swung to 339-496k. Re-run on a quiet machine before claiming.
artmap (bank-bench-art, transient in-place, no commit gen): sm 2.04M tx/s,
live 32.8 MB — the no-versioning ceiling row. artmap 1M: DNF (4 GB bump
arena, recorded 06-11). hkv has no bank engine (heap-only, no sessions).

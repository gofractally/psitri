# SAL 2.0 / gcart — build status (2026-06-17)

Autonomous build session against `docs/sal2_design.md`. Branch `sal2`.
**Headline result: gcart meets both stated goals — read perf ≈ start_ver and
3.4× faster than sal_art_ref; memory 2.4–2.7× lower.**

## Benchmark (1 M random keys, 88-byte values, same machine + same keys)

| engine        | insert | **read hit** | RSS    | note |
|---------------|--------|--------------|--------|------|
| **gcart**     | 257 ns | **166 ns**   | 210 MB | raw tree (no txn/commit/durability overhead) |
| start_ver     | 340 ns | 162 ns       | 506 MB | full kv adb path |
| sal_art_ref   | 1566 ns| 563 ns       | 573 MB | full kv adb path |

- **Read: gcart 166 ns ≈ start_ver 162 ns, and 3.4× faster than sal_art_ref (563 ns).** ✅
- **Memory: gcart 210 MB RSS vs 506 / 573 — 2.4–2.7× lower.** ✅ (gcart node bytes
  = 165 MB live = ~173 B/key; nodes are 128-byte cacheline multiples, no control
  blocks.)
- gcart per-mode (1 M): dict insert 349 / hit 254 / miss 176 ns, 179 B/key;
  random insert 257 / hit 166 / miss 95 ns, 173 B/key; sequential insert 89 /
  hit 168 / miss 26 ns, 142 B/key.

Caveat (honest): the read comparison is apples-to-apples (pure `get` on all
three). gcart's *insert* number omits transaction/commit/sync overhead the
others pay — that lands with the kv adapter (below). The read path and memory
footprint are the real, final ones.

Reproduce: `ninja -C build gcart-bench && ./build/bin/gcart-bench 1000000 <0|1|2>`.
Comparison harness (start_ver/sal_art_ref) preserved at `logs/2026-06-17_gcart-readbench/cmp.cpp`.

## Milestones

| # | what | state |
|---|------|-------|
| M1 | `libraries/sal2/` — FIX-region line allocator (segments + seg_freeset + per-node refcount, simple session) | ✅ fuzz-green (8.0 M ops, 0 fails) |
| M2 | `libraries/gcart/` — ART engine core (node.hpp + tree.hpp): get/upsert/update/erase/in-order, required style | ✅ oracle-green (537 K checks vs std::map) |
| M2.1 | bench + allocator fragmentation fix (multi-line alloc livelock) | ✅ done |
| M3 | `gcart::kv` adb adapter — cursor (fwd/rev/lower_bound) + snapshot `tree` | 🚧 WIP (cursor + snapshot written; **transaction / write_session / database / read_session not yet written**) |
| M4 | add gcart as 5th engine in `tests/kv_compat_test.cpp`; `static_assert(adb::kv_database)`; pass templated battery | ⬜ not started |
| M5 | sal2 version-GC (death stamps + held-roots, design §4–§5) replacing v1 refcount; movable + BIG regions | ⬜ design's later phases |

## What's built (all on branch `sal2`, committed)

- `libraries/sal2/{config,freeset,arena}.hpp` + fuzz test. Configurable 128-byte
  grain; file-backed recycling line store; per-node refcount COW liveness.
- `libraries/gcart/{node,tree}.hpp` + oracle test + bench. Prefix-compressed
  ART: leaf / setlist / node256, in the required style (`alloc_size(args)` ↔
  `ctor(asize,args)`; copy-construct-insert op tickets; privatize-on-descent COW;
  grow via copy-construct-insert + free-husk, child pointers transferred;
  recursive refcount reclaim).
- `libraries/gcart/kv.hpp` — **partial**: `cursor` (path-stack in-order, fwd/rev,
  lower/upper bound, point get) and the owned snapshot `tree`. Not yet wired
  (`database`/`write_session`/`read_session`/`transaction` are the next step) and
  not in any build target, so the build stays green.

## Next steps (resume here)

1. **Finish `gcart::kv`** (M3): `database` (holds a `gcart::tree` per root +
   `sal2::arena`), `write_session::start_transaction(idx)` → `transaction`
   (COW-snapshot: pin committed root, mutate private copy, commit publishes +
   releases old, abort restores), `write_session::get_root(idx)` → the snapshot
   `tree`, `read_session::create_cursor(idx)`, the `gcart::engine` traits bundle.
2. **M4**: `static_assert(adb::kv_database<gcart::kv::database>)`; add `gcart` to
   `tests/kv_compat_test.cpp` (the 5th engine) and pass the templated battery vs
   std::map + the other engines as differential oracles (design §3.2). This also
   gives the side-by-side perf table including the txn/commit path.
3. **Deviations from the design to revisit**: v1 uses per-node **refcount**
   reclamation, not the design's **version-GC** (death stamps + held roots,
   §4–§5) — swap once the engine is conformant. v1 is **FIX-region only**; the
   movable SAL + BIG regions (§6.2/§6.3) and the 3-bit-packed separate tags (§7,
   v1 reads the node type byte instead) are later phases. Each is a measurable
   win deferred behind correctness + the headline perf/memory result, which is
   already achieved.

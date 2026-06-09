# Task 24 handoff — wire implicit_ref_pool into the heart radix engine

**Goal:** replace heart's broken packed-nibble range-refcount integration with the proven,
TSan-clean `sal::implicit_ref_pool` (per-range self-freeing 1-byte counters). This is the final
step that makes COW-to-root child retain/release O(#ranges)=8 instead of O(fan-out)=256, *and*
retires the overflow/snapshot-corruption that the packed scheme hit on the remove path.

Read first, in order: `libraries/heart/doc/micro_refcount.md` (the full contract — model, the
two architecture requirements, free-policy, header layout, ROI caveat). This doc is the
build-it checklist on top of that contract.

---

## Branch / commit map

- **`heart`** = the shipping base. Contains: COW correctness (insert latch `6dc426c`, remove latch
  `ba5514b`), the full SAL foundation (`44bf988` colo_alloc, `ebcfc62` implicit_ref_addr,
  `075aa62` implicit_ref_pool — all tested + TSan-clean), and all the design docs. **Green.**
  Does NOT contain any engine range-refcount wiring (the rr engine code is only on the wip branch).
- **`heart-rr-wip`** (`88f1c5b`) = the PACKED-nibble integration to be REPLACED. Branched from
  `6dc426c` (so it has the insert latch but NOT the remove latch `ba5514b`, NOT the SAL
  foundation). Insert path works (snaptest ISOLATED, 1690 ns/txn) but remove/1M-rand OVERFLOW.
  Use it ONLY as a reference for *where* the rr hooks go in the engine. Its `rrstress` harness +
  `adopt_root`/`sig_at` engine hooks are worth salvaging into the new branch.

**Recommendation:** start a FRESH branch off `heart` (call it `heart-rr2`). `heart` already has the
SAL foundation + remove-latch fix + locked header format; `heart-rr-wip` has neither and is too
diverged to merge. Cherry-pick the *ideas* (hook locations, rrstress) not the code.

---

## What the SAL foundation gives you (all on `heart`, done + tested + TSan-clean)

`#include <sal/implicit_ref_pool.hpp>` →
- `sal::implicit_ref_pool` on `colo_alloc<std::atomic<uint8_t>>`:
  - `ptr_address alloc_fresh()` — a fresh counter = 1 (anywhere). Use for the FIRST counter of a
    node (picks the region).
  - `std::optional<ptr_address> alloc_in_region(const implicit_ref_hint&)` — co-located sibling
    counter; `nullopt` only if the region is full (caller re-homes — see below).
  - `void retain(ptr_address)` — ++ (clone shares the range). asserts `1 ≤ prev < 255`.
  - `[[nodiscard]] bool release(ptr_address)` — --; returns **true iff was last** (slot is then
    freed; caller must release that range's branch pointers).
  - `uint8_t count(ptr_address)`, `uint64_t used()`.
  - recovery: `clear_all()`, `retally(ptr_address)` (claim-on-first-sight + ++).
- `sal::implicit_ref_addr` = u16 offset; `sal::implicit_ref_region` = u16; `combine(region,off)` →
  full `ptr_address`; `region_of(addr)` / `offset_of(addr)` split it.
- `sal::implicit_ref_hint { region; std::span<const implicit_ref_addr> within; }`.

Compact handle: full counter address = `combine(region, range_ctr[r])`. Counts are EPHEMERAL
(rebuilt by `retally` on recovery); region + offsets are PERSISTENT in the node.

---

## Step 1 — pool ownership (SAL, do on `heart` first, then branch)

The engine reaches the pool through the session, mirroring how `retain`/`release` wrap
`_ptr_alloc` today.

1. `sal::allocator` (libraries/sal/include/sal/allocator.hpp + .cpp): own an
   `implicit_ref_pool` (its own dir, e.g. `<db>/iref/`), constructed alongside the
   `control_block_alloc`. A second pinned-RAM pool — see allocator.cpp ctor where `_ptr_alloc`
   (control_block_alloc) is created.
2. `allocator_session` (allocator_session.hpp / allocator_session_impl.hpp): add thin forwarders
   `rr_alloc_fresh()`, `rr_alloc_in_region(hint)`, `rr_retain(adr)`, `rr_release(adr)→bool`,
   `rr_count(adr)` that call `_sega`'s pool. (Look at how `retain(ptr_address)` /
   `release(ptr_address)` forward to `_ptr_alloc` — same shape.)
3. Recovery: when the engine/db rebuilds (the existing control-block recovery path), call
   `pool.clear_all()` then walk live roots and `pool.retally(combine(region, range_ctr[r]))` for
   every big node's 8 ranges. (Wire into whatever recovery walk psitri/heart already has; if heart
   has none yet, this can be a later add — the pool works without it for non-crash runs.)

---

## Step 2 — node header (radix_node.hpp)

Replace `refcount_roots<K>` / the `large_radix` `blocks[]` with the locked layout
(`micro_refcount.md` "Node header layout"):

```cpp
struct range_ctrs {                          // 18 bytes, fixed-header for all big tiers
   sal::implicit_ref_region region;          // u16
   sal::implicit_ref_addr   range_ctr[8];    // u16 each; 8 ranges (range_of(byte)=byte>>5)
};
```

`large_radix<Derived>` (drop the K param — always 8 ranges) uses `range_ctrs` as its FixedHeader.
Add accessors: `region()`, `ctr(r)` (→ `combine(region, range_ctr[r])`), `set_ctr(r, addr)`.
Construction nulls them (region=0xFFFF sentinel / offsets = a reserved "absent" marker) so a
half-built node / missing-birth reads "no counters" and the destroy falls back safely.

Rewrite `large_radix::destroy(session)`: for each range r in 0..7, `was_last =
s->rr_release(node.ctr(r))`; if `was_last`, release that range's branch pointers — i.e.
`self().for_each_entry([&](byte, sa){ if (sa.is_pointer() && range_of(byte)==r) s->release(sa); })`.
Then `term_value` if pointer (held directly, not a range). NO nibble math, NO custom_cb.

---

## Step 3 — engine lifecycle (radix_engine.hpp). Reference: heart-rr-wip line numbers below.

Detect big tiers with the existing `is_big<N> = requires { N::refcount_block_count; }` (keep a
marker on `large_radix`). Replace each packed hook:

- **`rr_birth<N>(n)`** (wip:640; call sites: `make_node` wip:601, widen/`grow_and_set` wip:2405 +
  the second widen, `rebuild_to_full` ~wip:2860): alloc the 8 counters. First via
  `_ses->rr_alloc_fresh()` → that sets `region`. Remaining 7 via `_ses->rr_alloc_in_region({region,
  placed_offsets})`; if any returns `nullopt` (region full), RE-HOME: pick a new region
  (`rr_alloc_fresh` again) and move the already-placed unique counters (they're count==1 at birth,
  safe to move: alloc new, copy nothing-needed since fresh, free old). Store region + 8 offsets.
  Each counter starts at 1 (this version).
- **clone** (wip:743 — the `if constexpr(is_big) rr_clone_share` in `clone<N>`): copy
  `range_ctrs` from src (blit already copies the header bytes), then `for r in 0..7:
  _ses->rr_retain(nn.ctr(r))`. 8 retains. Plus `term_value` retain if pointer (as today).
  **Delete `rr_clone_share`'s fork path** — no fork; if a counter would hit 255, that's the
  genuine 255-version ceiling (assert for now; real fork is a later, rare add).
- **`children_shared<N>(cn)`** (wip:652; used by the COW mode latch at router_insert wip:1724 and
  remove_router wip:1837): a big node's children are co-owned iff ANY range counter > 1. Replace
  the `custom_cb_share(blk)>1` check with `for r: if (_ses->rr_count(cn.ctr(r)) > 1) return true;`.
  **KEEP the latch** — `if (ref.ref()>1 || children_shared<N>(cn)) _shared=true;` — it's load-bearing
  for snapshot isolation (deferred counts hide sharing from `cb.ref`; this is the bug snaptest
  caught). Also KEEP the remove-path latch + `_shared=false` reset in `remove()` (these are the
  `ba5514b` fix; heart-rr-wip does NOT have them — re-apply).
- **divergence / `rr_settle<N>(n, range_of(b))`** (the ~9 mutation sites: wip:1788,1804,1815 in
  remove_router; 1870,1899,1912,1951 in router_insert; 2170,2266 in the full_radix add paths):
  REPLACE settle with proper range-COW. When a node mutates range r's branch-set (add/drop/
  re-point a branch in range r) AND `rr_count(ctr(r)) > 1` (shared):
    1. retain the KEPT pointer-children in range r (the ~31 unchanged) so they survive under both
       the old and the new counter — `for_each_entry` filtered to range r;
    2. `was_last = _ses->rr_release(ctr(r))` (drop this version from the old set; was_last can't be
       true here since count was >1, but handle defensively);
    3. `ctr(r) = _ses->rr_alloc_fresh()` or `alloc_in_region` (the new, count==1, this-version's
       range-r set).
  If `rr_count(ctr(r)) == 1` (already unique to this version) the mutation is in-place, no COW of
  the counter. This is the inherent ~32-retain divergence cost — but clean (no nibble ambiguity,
  no settle-vs-async hazard). NOTE: the engine's existing `_ses->release(old_child)` at re-point
  sites stays — that's the branch-level op; the range-COW above is the counter-level op.

---

## Step 4 — validate (the gate; do not declare done without all of these)

Bring the `rrstress` harness + `adopt_root`/`sig_at`/`count_keys_at`/`take_snapshot`/`find_at`
engine hooks over from heart-rr-wip (kv_u64_bench.cpp `rrstress` mode + the radix_tree debug
methods). Then, under **build-di** (RelWithDebInfo + ENABLE_DEEP_INVARIANTS — the ONLY thing that
sees SAL-arena UAF/double-free/leak; ASAN can't):

```
ninja -C build-di kv-u64-bench heart-tests
./build-di/bin/heart-tests                          # all 81 cases, [remove] [fuzz]
./build-di/bin/kv-u64-bench 100000 snaptest         # both checks ISOLATED (incl. remove isolation)
./build-di/bin/kv-u64-bench 4000   rrstress         # 0 MISMATCHES, no overflow  ← the one that broke
./build-di/bin/kv-u64-bench 1000000 rand            # completes, [OK], no "reference count exceeded"
```
Release perf: `build/release/bin/kv-u64-bench 1000000 worst` — expect heart ≤ ~1690 ns/txn
(packed insert number) and NOW correct on remove. Compare psitri 2960.

---

## Gotchas & hard-won lessons (DO NOT relearn these)

- **The mode latch is mandatory and subtle.** Deferred counts make `cb.ref(child)` understate true
  sharing, so the unique→shared COW latch MUST also fire on `children_shared` (any range counter
  >1), not just `ref()>1`. Missing it → a unique node with shared children mutates a
  snapshot-reachable child in place → snapshot corruption (off-by-N keys). snaptest's count check
  catches it; `find_at` does NOT (it's `seek` = first-key-≥-q, gives false positives — use
  `count_keys_at`/`sig_at`, never `find_at`, for membership).
- **Remove path needs the latch too.** `remove()` must `_shared=false` at top; `remove_router` must
  latch `if (ref()>1 || children_shared) _shared=true`. heart-rr-wip is missing this (it predates
  `ba5514b`). This was a real corruption bug.
- **The overflow that killed the packed scheme was settle-vs-async**, NOT atomics. The new pool has
  NO settle and is async-safe by construction (clone +1 / destroy -1 conserved). If you see
  "reference count exceeded limits" again, it's a retain/release imbalance in the wiring, not an
  inherent flaw — trace which `rr_retain` lacks a matching `rr_release`.
- **divergence still costs ~32** (retain the kept children for the new range counter). That's
  inherent to per-range granularity; the pool only makes the COUNTER cheap, not the branch retains.
- **Perf reality**: after batching, the worst-case wall is the node MEMMOVE, not counter atomics
  (and settle/divergence no-ops at share==1). So this won't beat 1690 on the simple worst case;
  its payoff is correctness + the multi-snapshot regime. The throughput lever below 1690 is
  shrinking the dense node (psitri-style packed pointers) — a SEPARATE effort.
- **Build dirs**: build/release (-O3), build/debug, build-di (deep invariants — USE THIS for rr
  validation), build/tsan (ENABLE_SANITIZER=ON — run the SAL pool tests here).
- **Don't `git add -A`** (sweeps build dirs). Commit named files. Commit/push only when asked.
  Messages end `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Definition of done
`rrstress` + 1M `rand` + `snaptest` all clean under build-di (no overflow, no MISMATCH, ISOLATED),
heart-tests 81/81 green, worst-case ≤ ~1690 ns/txn and correct on remove. Then the packed scheme
(`heart-rr-wip`, `range_refcount.hpp` nibble math) can be retired.

# Range refcounts — batched child retain/release for COW-to-root

## Problem

COW-to-root clones every node on the update path. heart's clone currently does
`for_each_child(retain)` — one atomic CB increment per child — because the clone
co-parents all of the original's children. On the symmetric side, releasing the old
version releases every child. For a node of fan-out `F` that is `O(F)` atomics per
clone and per release; on a depth-`D` path the worst-case (commit-per-update)
transaction pays `≈ Σ(fan-out)` atomic refcount operations. Measured: this is a large
slice of the ~1750 ns/txn heart number and the ~2920 ns/txn psitri number.

Goal: make the bulk child retain/release on a clone `O(1)` in the steady state, paying
the per-child `O(F)` cost only at genuinely necessary, rare lifecycle points.

## Mechanism

Each big-tier node (bitset_r2/r4/full — the high-fan-out tiers; the small/sparse tiers
keep their byte-pinned layout and their cheap per-child retain) stores **one extra
`ptr_address`** in its CL0 header: a pointer to a shared **nibble block** `B`.

`B` is a bare SAL control_block (no backing data) flagged by a `type` code:
- its 18-bit `ref` **is the share count** — how many node-versions point at it;
- its 38-bit `cacheline_offset` is reinterpreted as **8 nibbles** `n[0..7]` (32 bits,
  6 spare), one per branch range (256 branches / 8 = 32 per range — also one nibble per
  branch-cacheline of `full_radix`);
- the whole thing is a single u64, mutated by one CAS.

Children are no longer anchored by a single node. The invariant:

```
cb.ref(c)         = number of live nibble-blocks that cover c   (each block holds ONE base retain per child)
n_i[r]            = B_i.share - 1                                (the block's base covers version #1; the nibble covers the rest)
true_ref(c)       = Σ_i (1 + n_i[r]) = Σ_i B_i.share = total versions referencing c
```

The point of `cb.ref = #blocks`: a child is freed (SAL's `cb.ref → 0`) exactly when the
*last* block covering it dies; the deferred per-version counts live in the nibbles and
never strand, because **a block never overflows** (we fork before it would) so
`n_i[r] = share_i − 1` holds for the entire life of every block.

## Block lifecycle

- **Clone node V→V′ (shares all children), within a block** (`n[r] < 15` for every range):
  `retain(B)` (share++) and `n[r]++` for each uniform range — **one CAS**, no child
  touched.
- **Fork** (a bump would hit 16 on any nibble): allocate `B′`; `cb.retain(c) ∀c` (B′'s
  base — "increment everything by 1"); `B′.share=1, n=0`; `V′ → B′`. The **old block is
  left full at 15** and the old versions keep pointing at it. `O(F)`, but only once per
  16 stacked live versions.
- **Release node V, within a block** (`n[r] > 0`): `n[r]--`, `share--` — **one CAS**, no
  child touched. Because every block is full-or-draining, a release is *always* this
  cheap decrement until the block's last version.
- **Block death** (last version releases → `share→0`, all `n=0`): `cb.release(c) ∀c`,
  free `B`. `O(F)`, necessary (the group's children lose their anchor).

Releases never branch on "is this materialized" and old versions never learn that a
fork happened — a releaser touches only **its own block** + **its own branch array**.
All the info is implicit.

## Divergence (re-point a branch to a NEW child)

A freshly placed child `L′` cannot inherit the range's nibble: `true_ref(L′) =
cb.ref(L′) + n[r]` would need `cb.ref(L′) = 1 − n[r] < 0`. So re-pointing into a range
with `n[r] > 0` must **settle that one range first**:
`cb.retain(c) += n[r] ∀ live c∈r; n[r] = 0` (invariant-preserving — pushes deferred into
real `cb.ref`, safe even on a shared block). Then place `L′` with `cb.ref=1`. The other
7 ranges stay deferred and shared. In COW steady state `n[r]` is 0–1 so the settle is
cheap, and only the one range the path passes through is affected per node.

Fork handles "too many versions"; settle handles "a branch changed." Orthogonal; both
preserve `n_i[r] = share_i − 1`.

## Verified overflow trace (16 versions share B0: share=16, n=15, cb.ref(c)=1)

| event | B0 share/n | B1 share/n | cb.ref(c) | true_ref(c) |
|---|---|---|---|---|
| clone N15→N16 (overflow → fork) | 16 / 15 | 1 / 0, `cb.retain ∀c` | 2 | 17 ✓ |
| clone N16→N17 (in B1) | 16 / 15 | 2 / 1 | 2 | 18 ✓ |
| release N17 | 16 / 15 | 1 / 0 | 2 | 17 ✓ |
| release N1..N15 (in B0) | 1 / 0 | 1 / 0 | 2 | 2 ✓ |
| release N0 (B0 death → `cb.release ∀c`) | freed | 1 / 0 | 1 | 1 ✓ |
| release N16 (B1 death → `cb.release ∀c`) | — | freed | 0 → c freed | 0 ✓ |

Every step holds `true_ref = cb.ref + Σn`. The only `O(F)` touches are fork (block
birth) and block death — both rare and necessary. In the COW steady state `share=1`: no
block ever forks, the single block oscillates `n: 0↔1`, and children are touched only at
node birth/death (the COW itself). `cb.ref(c)` grows by 1 per 16 simultaneously-pinned
versions — bounded and tiny.

## Cost summary vs today

| | today | with range refcounts |
|---|---|---|
| clone a big node (steady state) | `O(F)` atomics | **1 CAS** |
| release a big node (steady state) | `O(F)` atomics | **1 CAS** |
| diverged range on the path | included in `O(F)` | `O(32)` settle (one range) |
| fork / block death | — | `O(F)`, 1 per 16 stacked versions |
| node footprint | — | +1 `ptr_address` in CL0 (big tiers only) |

## Measured ceiling (why this is worth it)

`kv-u64-bench … worst` with the per-child retain (clone) + release (reclaim) compiled
out (`HEART_SKIP_CHILD_REFCOUNT`, leaks — measurement only):

| N | baseline ns/txn | atomics removed | atomics' share |
|---|---|---|---|
| 300k | 3,121 | 128 | ~96% |
| 1M | 1,802 | 155 | ~91% |

Child-refcount atomics are **~90–96% of all COW-to-root latency**. Realistic capture
with 8-ish nibbles is bounded by the divergence-settle (~6–8×) → COW-to-root ~1,800 →
~300–500 ns. This is the single biggest remaining lever.

## Substrate (DONE)

- `control_block::cas_user_value(expect, desired)` — CAS the packed nibble word,
  preserving the share count (`ref`). Composes with concurrent retain/release of the
  block (those change `ref`; the CAS retries).
- `allocator_session::custom_cb_share(adr)` / `custom_cb_cas_value(adr, expect, desired)`
  + existing `alloc_custom_cb` / `read_custom_cb` / `retain` / `release`. A range block is
  just a custom CB: `ref` = share, location bits = 8 nibbles.

## Engine integration (TODO — the large piece)

The block must *hold* the base child retains, so every site that takes/drops a ref on a
big-tier **branch pointer** routes through the block. Surface (radix_engine.hpp unless
noted):

1. **Node birth that builds children** (make_node / promote setlist→bitset, re_stride,
   prefix_split building a big node): alloc the block, `cb.retain` each child (base),
   nibbles=0.
2. **Clone** (`555`, `2656`): blit already copies `blocks[]` (clone shares the block) →
   `retain(block)` + bump 8 nibbles, or **fork** on overflow. Replaces `for_each_child(retain)`.
3. **Destroy** (`radix_node.hpp:151`): `release(block)`; on block death, `cb.release` each
   covered child. Replaces `for_each_child(release)`.
4. **Structural mutation of a big-tier branch set** — `set_branch` (add), `remove_branch`
   (drop), re-point/replace, collapse/demote (`~1518–2809`): **settle the affected range**
   first (`cb.retain += n[r] ∀c∈r; n[r]=0`), then the individual add/drop on `cb.ref`.
5. Small/sparse tiers (setlist*, single-child, suffix_value) keep direct per-child refs —
   no block.

Validate every increment under `build-di` (free_range_tracker is the only thing that sees
SAL-arena UAF/double-free/leak), plus `[remove]` fuzz and kv-compat, then measure `worst`.
The correctness surface is item 4 (range-aware settle at each mutation site); that is the
bulk of the risk and effort.

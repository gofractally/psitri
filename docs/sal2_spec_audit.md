# sal2 — spec-conformance audit (2026-06-17)

> Historical design record. Current code uses `experimental/sal_hybrid`
> (formerly `sal2`) and `experimental/sal_gc` (formerly `sal2gc`).
> See the [allocator rename map](../experimental/README.md#allocator-names).

Audit of `libraries/sal2/` against `docs/sal2_design.md` (the normative spec).
**The only sanctioned deviation was: refcount liveness instead of
death_ver + version-interval GC** (Dan, "refcount for now"). Everything else was
supposed to follow the spec. This lists every deviation, sanctioned or not.

Legend: ✅ conforms · 🟡 sanctioned deviation (refcount-for-now) · ❌ **unsanctioned
deviation** (must fix).

## Summary

The FIX region (`fixed_allocator<128/256/1152>` + `freeset`) largely conforms.
**The movable (SAL leaf) and BIG regions were built WITHOUT the control-block /
relocation layer the spec mandates** — that is the root unsanctioned deviation,
and it is exactly what crashed the Silkworm run (128 GB cap + no compaction).
The refcount-instead-of-version was sanctioned; **dropping the control block,
relocation, and the whole version substrate was not.**

| # | Spec | Requirement | Status | Evidence |
|---|---|---|---|---|
| D1 | §3, §4.3, §6.2, §7.3 | MOVABLE leaves addressed `ptr_address → control_block → location`, **relocatable** by the compactor | ❌ | `var_allocator.hpp:16` "slot index … is a GLOBAL grain index: addr = base + slot·Grain" — direct, no CB, no relocation |
| D2 | §4.3 | `sal2/control_block.hpp` = `{ref:16, active:1, pending_cache:1, zone:2, offset:44}` | ❌ | file does not exist; nothing references it |
| D3 | §6.3 | BIG objects go **through a control block**, `zone=big`, sharing the CB-indexed death array | ❌ | `big_allocator.hpp` is direct-indexed (`get(g)=base+g·Grain`), inline `[ngrains|ref]` header, no CB/zone |
| D4 | §4.1 | movable **grain default 32**; CTest matrix at 32/64/128; cross-grain reopen throws | 🟡/❌ | movable grain IS 32 (`var_allocator<32>`), but `config.hpp:11` knob defaults 128 and there is **no grain matrix / geometry-stamp reopen guard** |
| D5 | §4.2, §4.4, §5 | 32-bit object **birth** in headers (force-persistent), commit counter in `.meta`, held-roots-as-pins, `[birth,death)` | 🟡 | refcount-for-now is sanctioned, BUT birth was **dropped entirely** (`arena.hpp:7`, `fixed_allocator.hpp:12` "created/birth dropped") — even the version groundwork is gone, so gcart had to re-add birth to node headers itself (GC-2a) |
| D6 | §4.3, §6.2 | `death_ver[]` parallel mmap (not force-synced), supersession stamp, reconstruct-on-crash | 🟡 | none — refcount `.rc`/`.refs` arrays instead (sanctioned) |
| D7 | §6.1 part 3 / §8 | FIX session/segment layer **inherited from sal** | 🟡 | `arena.hpp:4` deliberately rejects sal's session for a custom simple one — defensible, but a documented divergence |
| D8 | §6.1 | one FixedAllocator<N> per class | ✅ | `fixed_allocator<128/256/1152>` |
| D9 | §6.1 part 2 / §8 | per-segment `seg_freeset` hierarchical bitmap, lock-free O(1) | ✅ | `freeset.hpp` ports start_ref's seg_freeset |
| D10 | §6.1 | two parallel FIX impls | 🟡 | `arena.hpp` (single `data.sal2` global 128 B-line store) backs the **legacy** engine (`gcart/rcart tree.hpp/node.hpp/kv.hpp`); the **spec** engine (`spec_tree`/`spec_kv`) uses the per-class `fixed_allocator<128/256/1152>`. Not dead — but a single-region design that doesn't match the spec's 3-class FIX; legacy-only, fine to leave, don't extend |
| D11 | §7.1 | engine inner nodes pack the per-branch `type` at **3 bits** | ✅ | gcart `tag3_get/tag3_set`, `tag3_bytes(n)=(3n+7)/8` |
| D12 | §5.2 | engine COW `template<cow_mode{shared,unique}>` birth latch | ✅ | gcart `cow_mode` + birth-latch (GC-2c) — though it rides on the dropped-then-re-added birth (D5) |

## The root deviation (D1/D2/D3), in detail

Spec (§3 diagram, §6 table, §7.3): the movable region's branch pointer is a
**`ptr_address` = control-block index**; `resolve()` reads the CB's `{zone,
offset}` and returns `zone_base[zone] + offset × stride`; the compactor relocates
an object by **CAS-rewriting only `offset`** in its CB, so the u32 `ptr_address`
in every parent branch stays valid. The CB's `offset:44 × 32 B = 512 TB/zone`,
and the u32 `ptr_address` gives **4 billion relocatable objects**.

What shipped: `var_allocator`'s "ptr_address" **is** the 32-byte grain index
(`addr = base + slot·Grain`). No indirection ⇒ **not relocatable** ⇒ no
compaction ⇒ segments only whole-recycle at >50% free ⇒ the recycle backlog. And
the index is a **grain** index, so the region caps at `2^32 grains × 32 B = 128
GB`, not `2^32 objects × 512 TB`. BIG is the same shape (direct, no CB).

**This single deviation produced both Silkworm failures** (the 128 GB `sal.dat`
wall + the 37–48% un-recyclable backlog). The designed region would not have
crashed: 1.5 B live objects << 4 B CB budget; 128 GB data << 512 TB.

## Fix plan (keep the sanctioned refcount; restore everything else)

The CB **keeps a `ref`** in the spec anyway (§4.3, for `smart_ptr`), so the
sanctioned "refcount-for-now" maps cleanly onto the designed control block — v1
uses `ref` for liveness, the `death_ver` array is the later swap. So the fix is
NOT a rewrite away from refcount; it is **building the control-block / relocation
layer the spec always required**, with `ref` as the (sanctioned) liveness field.

1. **`sal2/control_block.hpp`** — the 8-byte CB `{ref:16, active:1,
   pending_cache:1, zone:2, offset:44}` + a control-block allocator (the
   `ptr_address` space, a CB-index slab) and `death_ver[ptr_address]` sibling
   (stubbed to `ALIVE` under refcount-v1, real under version-GC later). (§4.3)
2. **MOVABLE region = `var_allocator` behind the CB.** alloc returns a
   `ptr_address` (CB index); the CB holds `{zone=movable, offset}`; `resolve(pa)`
   = `seg_base + cb[pa].offset × grain`. The grain store stays; the *pointer* gains
   one indirection. (§6.2, §7.3)
3. **Compactor relocation** — relocate a live object, `cas_move()` rewrites only
   `cb[pa].offset`; parent branches (holding `pa`) are untouched. This is the
   capacity fix (frees grains by densifying) AND removes the recycle backlog. (§6.2)
4. **BIG region behind the CB** — `zone=big`, shares `death_ver`, never relocated.
   (§6.3)
5. **Birth back in object headers** (§4.2) — gcart already re-added it (GC-2a/b);
   fold that into the sal2 header so it's force-persistent per spec, not engine-bolted.
6. **Grain matrix + geometry-stamp reopen guard** (§4.1). Leave `arena.hpp`
   (legacy-engine store, D10) alone.

Sequence: 1→2→3 unblocks Silkworm scale (the crash); 4 (BIG) and 5/6 follow.
The engine (`gcart`) changes only where a branch pointer's *meaning* flips from
"grain index" to "ptr_address (CB index)" — `resolve()` gains the CB hop for
`suffix_value`/`big_suffix_value` (it already dispatches by tag, §7.2).

# nested_setlist implementation plan (workflow-synthesized 2026-06-17)

I have everything I need. The maps are accurate against the live code. Let me verify the two key build-fit numbers (max prefix that lets nested fit, and inner setlist byte cost) so the CREATE condition is concrete.

The setlist header is 3 bytes (`meta`, `nbranch`, `cap`). An inner setlist with prefix `pl`, `n` byte-branches + optional term needs `branches_off_(pl, c) + c*4` bytes per the existing `bytes_for_cap`. For a 3-branch, 0-prefix, no-term inner: `branches_off_(0,3) = ((3+0+3+tag3_bytes(3))+3)&~3 = ((3+3+2)+3)&~3 = (8+3)&~3 = 8`, then `+3*4 = 12` → 20 bytes. Matches the spec's "~20 bytes" exactly. Good — the existing `setlist::bytes_for_cap` is the correct sizing primitive for inner nodes too.

Now I'll produce the plan.

---

# Implementation plan: `nested_setlist` in `gcart::st`

All line numbers are against the live `spec_tree.hpp` I just read. The design model is settled by the live dispatch tables and the spec, and they agree: **`nested_setlist` is a standalone, refcounted, 128‑byte cl1 slot** (groups A #1–#6 already route it to cl1; `resolve`/`unique`/`retain`/`release_one` all treat it like a cl1 node). The `node.hpp:37` comment ("inlined in this node's tail-heap") describes the *inner* layout, not how the *outer* node is allocated — the outer IS a cl1 slot whose tail-heap holds the inner setlists. There is no contradiction once you read "inlined in tail-heap" as "the inner sub-setlists live in the same 128B line," which is exactly the §7 NestedSetlistNode. **Proceed with the standalone-cl1 model.**

The single most important invariant: a `nested_setlist` is **fully self-contained in one 128B line**. Outer "branch pointers" are u8 byte-offsets *within the line*; only the inner setlists' u32 branch entries are real cross-region pointers. This is what makes COW a single memcpy and release a single `cl1.free`.

---

## STAGE 0 — Decode contract (decide before writing a byte)

`build_node` (line 638) is the ONLY tag producer for routing nodes and it will never emit `nested_setlist`. Therefore on ANY structural COW edit (`rebuild`, `reprefix` drop>0), an existing nested would be flattened. **Decision (the simple, correct one): nested_setlist is a CREATE-time packing that NORMALIZES back to a flat `setlist`/children on any rebuild/reprefix.** Concretely:

- Read paths (`find_in`, `read_branches`, `node_term`, `node_child`, `node_prefix`) MUST decode nested correctly (Stage 2).
- `rebuild`/`reprefix-drop>0` flatten nested → `read_branches` yields each outer-byte's child as a *materialized* child node, which means a nested rebuild must first **promote** (Stage 4) the inner setlists into real cl1 slots so the scratch `br[]` holds real u32 pointers. This keeps the write path uniform and avoids teaching `build_node` to re-pack nested. The COW clone hot path (`reprefix` drop==0, line 865) preserves nested via verbatim memcpy + a nested-aware retain (Stage 5).
- In-place maintenance for a *private* nested (Stage 3) keeps it nested as long as it fits; the moment it doesn't, promote (Stage 4).

This makes nested **inert until create fires** (Stage 6 gate), and every other path either already handles it or is taught a single explicit `t == bt::nested_setlist` arm.

---

## 1. `struct nested_setlist` C++ layout (insert after `struct setlist`, ~line 263)

Mirror setlist's cap-based style. The outer is a setlist-of-u8-offsets; inner nodes are header-less setlists packed from the tail.

```cpp
// ── nested_setlist (cl1, 128B): an OUTER setlist whose branches are u8
//    tail-offsets to INNER header-less setlists packed in the SAME 128B line.
// Two trie levels in one cacheline (no second cold load). Self-contained: the
// only cross-region pointers are the inner setlists' u32 branch entries.
// layout: [u8 meta(term:1,prefixlen:7)] [u8 nbranch] [u8 tail_used]
//         prefix[prefixlen]
//         okeys[nbranch]          (the outer byte for each inner)
//         ooff[nbranch + has_term] (u8 offset-from-END for each inner; term slot last)
//         ... free gap ...
//         [inner setlists packed contiguously, growing DOWN from base()+slot_bytes]
//
// inner(i) starts at base() + slot_bytes - ooff[i]; its size is
//   ooff[i] - ooff[i-1]  (offsets are DESCENDING in i: ooff[0] is largest =
//   the inner nearest the tail-end; the LAST inner's low edge is at
//   slot_bytes - tail_used).  We store offset-from-END so a memcpy clone keeps
//   them valid.  An inner is an `inner_setlist` (setlist bytes WITHOUT the 3B
//   header — meta/nbranch/cap are recomputed; actually we reuse `setlist`'s body
//   but place the 3B header inside the packed region too, see below).
struct nested_setlist
{
   uint8_t meta;       // term:1 (bit7), prefixlen:7  (the OUTER term, if any)
   uint8_t nbranch;    // outer byte-branch count
   uint8_t tail_used;  // bytes consumed by packed inner nodes (from the tail)

   static constexpr uint32_t slot_bytes = 128;
   static constexpr uint32_t hdr        = 3;

   bool     has_term() const noexcept { return meta & 0x80; }
   uint32_t prefixlen() const noexcept { return meta & 0x7f; }
   uint32_t nslot() const noexcept { return nbranch + (has_term() ? 1u : 0u); }

   char*       base() noexcept { return reinterpret_cast<char*>(this); }
   const char* base() const noexcept { return reinterpret_cast<const char*>(this); }
   char*       prefix_ptr() noexcept { return base() + hdr; }
   const char* prefix_ptr() const noexcept { return base() + hdr; }
   sv          prefix() const noexcept { return {prefix_ptr(), prefixlen()}; }

   uint8_t*       okeys() noexcept { return reinterpret_cast<uint8_t*>(base() + hdr + prefixlen()); }
   const uint8_t* okeys() const noexcept { return reinterpret_cast<const uint8_t*>(base() + hdr + prefixlen()); }
   // one u8 offset-from-END per inner (byte-branches + a term inner if has_term).
   uint8_t*       ooff() noexcept { return okeys() + nbranch; }
   const uint8_t* ooff() const noexcept { return okeys() + nbranch; }

   // header area size (everything before the packed-inner tail-heap).
   uint32_t head_bytes() const noexcept { return hdr + prefixlen() + nbranch + nslot(); }
   // bytes free between the head area and the packed tail.
   uint32_t free_bytes() const noexcept { return slot_bytes - head_bytes() - tail_used; }

   // INNER ACCESS: inner(i) is a `setlist` whose bytes live at base()+slot_bytes-ooff[i].
   //   ooff is DESCENDING (ooff[0] largest). inner(i).size = ooff[i]-ooff[i+1]
   //   (ooff[nslot] := slot_bytes-tail_used's complement => for the last inner use
   //   the low watermark). We store the END offset; the inner's start = end - size.
   // For simplicity inner nodes are FULL `setlist` structs (3B hdr + body) so we
   // can reuse every setlist accessor verbatim. They differ from a cl1 setlist
   // ONLY in that their `cap == nbranch+term` (tight, no growth slack).
   setlist* inner(uint32_t i) noexcept
   { return reinterpret_cast<setlist*>(base() + slot_bytes - ooff()[i]); }
   const setlist* inner(uint32_t i) const noexcept
   { return reinterpret_cast<const setlist*>(base() + slot_bytes - ooff()[i]); }

   // outer term occupies inner-slot index `nbranch` (key-less): a setlist with
   // an empty prefix carrying just the term, OR (optimization) a direct
   // (tag,ptr). We carry it as inner(nbranch) = a 0-branch term-only setlist so
   // the decode is uniform. (See term handling note in §7.)
   bool   has_inner_term() const noexcept { return has_term(); }

   int find_index(uint8_t byte) const noexcept
   {
      int pos = int(ucc::lower_bound_padded(okeys(), nbranch, byte));
      return (pos < int(nbranch) && okeys()[pos] == byte) ? pos : -1;
   }

   // bytes an inner header-less... — actually inner uses setlist::bytes_for_cap.
   static uint32_t inner_bytes(uint32_t in_nb, uint32_t in_pl, bool in_term) noexcept
   { return setlist::bytes_for_cap(in_nb + (in_term ? 1u : 0u), in_pl); }

   // FIT: would an outer of (nb outer branches + oterm) whose inner nodes total
   // `inner_total` bytes fit one 128B line for this outer prefixlen?
   static bool fits(uint32_t nb, uint32_t opl, bool oterm, uint32_t inner_total) noexcept
   {
      const uint32_t nslot = nb + (oterm ? 1u : 0u);
      const uint32_t head  = hdr + opl + nb + nslot;
      return head + inner_total <= slot_bytes;
   }
};
static_assert(sizeof(nested_setlist) == 3);
```

Key offset facts (concrete):
- Header is 3 bytes at offset 0. Prefix at offset 3. `okeys` at `3+prefixlen`. `ooff` immediately after (`3+prefixlen+nbranch`), one byte per inner slot (`nbranch + has_term`).
- Inner setlists grow **down** from `base()+128`. `ooff[i]` = offset-from-END of inner(i)'s **start** (so `inner(i) = base()+128-ooff[i]`). Store offsets so they survive a verbatim memcpy unchanged.
- An inner setlist is a **complete `setlist`** (3B hdr + body, `cap` tight). This is the decisive simplification: every setlist accessor (`prefix/keys/types/branches/btype/branch/term_*/find_index`) works on `inner(i)` with **zero new code**. The "header-less" idea from the spec is dropped in favor of reuse — the 3 extra header bytes per inner cost ~9 bytes across 3 inners, well within budget, and eliminate an entire class of offset bugs.

---

## 2. Dispatch sites — the EXACT ordered list to edit (decode arms)

Add a `t == bt::nested_setlist` arm to each. Ordered so the engine stays compilable + green after each (nested is unreachable until Stage 6).

**Read decoders (Stage 2):**

1. **`find_in`** — `spec_tree.hpp:413` (before `const setlist* n = ...`). Add:
   ```cpp
   if (t == bt::nested_setlist) { /* outer prefix → outer key → inner setlist → inner key → child; see body below */ }
   ```
2. **`node_prefix`** — `spec_tree.hpp:626-627`. Add a nested arm returning the OUTER prefix:
   ```cpp
   if (t == bt::nested_setlist) return static_cast<const nested_setlist*>(resolve(e,t,p))->prefix();
   ```
3. **`node_child`** — `spec_tree.hpp:619` (before the `setlist` cast). Add nested: find outer index; if absent return false; the child is the **inner setlist as a pointer** — but inner has no slot. **Resolution:** `node_child` on a nested returns false-equivalent for the write path because nested children aren't independent (the write path promotes nested before editing — Stage 4). So in the write path, nested never reaches `node_child` un-promoted. For SAFETY add an arm that promotes-or-asserts. Concretely the write path calls `node_child` only inside `edit_node`, and Stage 3/4 intercept nested *before* `node_child`. So this arm is `assert(false)` / unreachable, but add it defensively returning the inner's first-real-child is wrong — leave it `return false;` with a comment, since promotion guarantees non-arrival.
4. **`node_term`** — `spec_tree.hpp:607` (before `setlist` cast). Add nested arm reading the outer term inner.
5. **`read_branches`** — `spec_tree.hpp:595` (the `else`). Add nested arm BEFORE the setlist else. This is the **most important** one (drives release + retain + rebuild flatten). It must enumerate **each inner setlist's real u32 children**, NOT the outer offsets. See body in Stage 5.

**Region/free selectors — VERIFY (already correct, no edit needed):**

6. `resolve` line 311 — nested→cl1 ✓ (already present)
7. `unique` line 332 — nested→cl1 ✓
8. `try_retain`/`retain`/`release_one` lines 343/354/366 — nested→cl1 ✓
9. `free_node` line 633 — nested falls to `else`→cl1 ✓
10. `release_subtree` line 447 / `release_async_` line 484 — nested falls to `cl1.free` else ✓ — BUT their child-iteration (lines 444-446 / 481-483) cast to `setlist` and iterate `nbranch` outer offsets as if u32 pointers. **These MUST be taught nested** (Stage 5). Add a nested arm in BOTH before the `setlist` else.

**Release/COW (Stage 5):**

11. **`release_subtree`** — `spec_tree.hpp:442` (`else`). Add nested arm: walk inner u32 children, then `cl1.free(p)`.
12. **`release_async_`** — `spec_tree.hpp:479` (`else`). Mirror #11 with budget.
13. **`release_ref`** (tree method) — `spec_tree.hpp:576-579`. Uses `read_branches`/`node_term`/`free_node`; once #5 + #4 handle nested, this works unchanged. **No edit** (relies on #5/#4).
14. **`reprefix` drop==0** — `spec_tree.hpp:872` (`else`). Already memcpies 128B and preserves `out={t,s}` ✓. Its retain loop at line 875 uses `read_branches` → fixed by #5. **No edit** beyond #5.

**Construction (Stage 4):**

15. **`build_node`** — `spec_tree.hpp:638`. Add the nested-create clause (Stage 4).
16. **`edit_node`** — `spec_tree.hpp:817-830`. Add a `wt == bt::nested_setlist` arm (Stage 3/4).

**spec_kv.hpp:** no edits — the root tag is carried opaquely through `pack/rtag/rptr` and every helper it calls (try_retain/retain/release_subtree/find_in) is fixed above. (Confirmed in maps #30–#34.)

---

## 3. CREATE strategy + in-place maintenance

### When `build_node` emits a nested_setlist (line 638, FIRST check)

The nested tier sits **before** the flat-setlist tier in the ladder. `build_node` receives a flat `br[]` (byte → (tag,u32)). It can emit nested only when **every** child branch is itself a *small routing setlist* that can be inlined. But `build_node`'s `brs` are already-materialized children (pointers to existing nodes), not sub-key-groups. So **the natural create site is NOT `build_node`** — it's the place where a *flat setlist would otherwise grow past cap*, i.e. `edit_node`'s grow branch and `rebuild`. There, the children are `suffix_value` leaves and small setlists whose **suffixes share leading bytes** — those leading bytes are what a nested groups on.

**Concrete create rule (the simple, correct version for v1):** Create a nested_setlist when, in `build_node`, the `(prefix, brs[], term)` would NOT fit a flat `setlist` (128) but the branch set partitions by **leading key byte** into groups each of which is a tiny setlist, and the sum of inner setlist bytes + outer header fits 128. Implement as:

```cpp
std::pair<bt,uint32_t> build_node(sv prefix, const br* brs, uint32_t nb, bool term, bt tt, uint32_t tp)
{
   const uint32_t pl = uint32_t(prefix.size());
   if (setlist::fits(nb, pl, term, 128))
      return {bt::setlist, pack_setlist<128>(_s1, prefix, brs, nb, term, tt, tp)};
   // NESTED: try to pack as 2-level if children are sparse small setlists in one line.
   uint32_t s;
   if (try_pack_nested(prefix, brs, nb, term, tt, tp, s))
      return {bt::nested_setlist, s};
   if (setlist::fits(nb, pl, term, 256))
      return {bt::double_setlist, pack_setlist<256>(_s2, prefix, brs, nb, term, tt, tp)};
   return {bt::full, pack_full(prefix, brs, nb, term, tt, tp)};
}
```

**`try_pack_nested`** (new method): the inputs are already individual byte-branches (flat). To form a *2-level* node from a flat list, group `brs[]` by leading byte is meaningless (each `br` is already one byte). The genuine nesting opportunity is: the flat list has > ~23 branches (doesn't fit cl1 flat) but they **cluster** — many share a common second-level prefix that the current flat node can't express. **This is not reconstructable from `build_node`'s flat input.** 

**Therefore the create site is `edit_node`'s child-recursion result (REP) + a periodic compaction, OR the leaf-split.** For v1, the tractable, high-value create is at the **leaf-split / prefix-split** (lines 764, 782) and at **rebuild-from-a-flat-setlist-that-just-overflowed**, where we re-examine the child set: if the overflowing flat setlist's children are all `suffix_value` leaves whose suffixes share leading bytes, repartition into inner setlists keyed by that byte.

Given the complexity, **v1 scope = create nested ONLY in `try_pack_nested` driven by a re-grouping of `brs[]` where each `br` is a `suffix_value` whose suffix is non-empty**: pull the first suffix byte up as the outer key, demote the rest. Implement `try_pack_nested` as:

```cpp
// Group brs[] by a SYNTHETIC second level: each child is a suffix_value leaf;
// use its current outer byte as the outer key and pack the leaf into the inner
// setlist for that byte. (v1: only fires when all children are leaves and the
// resulting 2-level node fits one cl1 line.) Returns false → caller falls through.
bool try_pack_nested(sv prefix, const br* brs, uint32_t nb, bool term, bt tt, uint32_t tp, uint32_t& slot_out);
```

**Honest assessment for v1:** the cleanest first-create that actually fits the existing flat `br[]` is: when a flat setlist overflows 128 but the **outer prefix is long** and children are few-and-leafy, a nested buys nothing over double_setlist. The real win (per spec) is *sparse multi-byte fanout*. **Recommendation: implement the layout + all decode/COW/release (Stages 1,2,5) first and keep `try_pack_nested` returning `false` (nested inert), validate green, THEN implement a focused create in a follow-up once the read/COW/release machinery is proven.** This is exactly the staged gate (Stage 6). The create heuristic is the one genuinely hard design call and should be measured, not guessed.

When create IS implemented, **`pack_nested_setlist`** (new, model on `pack_setlist`):
1. Partition the sorted `brs[]` into groups; each group → one inner `setlist` (use `pack_setlist`-style fill but into the tail region with tight `cap`).
2. Lay inner setlists from `base()+128` downward; record `ooff[i]` = `128 - inner_start`.
3. Fill `okeys[i]` = group's outer byte; set `meta`/`nbranch`/`tail_used`.
4. The outer term (if any) → inner(nbranch) = a 0-branch term-only setlist, `ooff[nbranch]` recorded.

### In-place maintenance (private nested, `edit_node` line 817 nested arm)

```cpp
if (wt == bt::nested_setlist)
{
   nested_setlist* n = static_cast<nested_setlist*>(resolve(_e, wt, wp));
   const int oi = n->find_index(byte_at_or_term);
   // 1. byte present at outer level → recurse into inner(oi)'s child for key[depth+?]
   //    BUT inner branches are u32 → recurse upsert_rec on the inner's real child.
   // 2. byte absent at outer level → must add a new inner setlist (1 leaf) into the
   //    tail. If free_bytes() >= new inner bytes + 1 (okeys) + 1 (ooff): pack the
   //    new inner at the tail, shift okeys/ooff to keep sorted, ++nbranch, update
   //    tail_used. Else → PROMOTE (Stage 4).
   // 3. inner present but inner needs a new branch and inner is tight-cap: rebuild
   //    inner in place if free_bytes() covers the delta; else PROMOTE.
}
```

Because inner setlists are tight-cap (`cap == nslot`), **any insert into an inner requires re-laying that inner** (no slack). The in-place op for nested is therefore: rebuild the one affected inner into the free gap, repoint its `ooff[oi]`, bump `tail_used`, and if no room → promote. This is more involved than flat-setlist in-place; **v1 may legitimately PROMOTE on every nested edit** (treat nested as read-only-after-create + promote-on-write) to ship correctness first, then add in-place inner maintenance as an optimization. State this explicitly in the code comment.

---

## 4. OVERFLOW / PROMOTE strategy

When a nested can no longer be edited in place (insert doesn't fit the line), **promote it to a flat structure**: unpack each inner setlist into its own **standalone cl1 `setlist`** and build a flat OUTER setlist (or double_setlist/full) whose branches point to those new cl1 nodes. Implement as **`promote_nested(bt t, uint32_t p) -> std::pair<bt,uint32_t>`**:

```cpp
// Unpack a nested_setlist into a flat outer setlist + one standalone cl1 setlist
// per inner. The flat node goes through build_node; each inner becomes a real
// pack_setlist<128> node retained as a child. Frees the nested slot.
std::pair<bt,uint32_t> promote_nested(bt t, uint32_t p)
{
   const nested_setlist* n = static_cast<const nested_setlist*>(resolve(_e,t,p));
   br scratch[257]; uint32_t nb = 0;
   for (uint32_t i = 0; i < n->nbranch; ++i)
   {
      const setlist* in = n->inner(i);
      // re-emit the inner as a STANDALONE cl1 setlist (copy its branches as-is —
      // its u32 children are already real and get RETAINED via build/clone rules)
      uint32_t s = pack_setlist_copy(_s1, in);   // verbatim body copy into a fresh cl1
      // retain the inner's real u32 children (they're now referenced by the new node)
      for (uint32_t j=0;j<in->total();++j) retain(_e, in->btype(j), in->branch(j));
      scratch[nb++] = br{ n->okeys()[i], bt::setlist, s };
   }
   bool oterm=false; bt tt=bt::null; uint32_t tp=0;
   if (n->has_term()) { oterm=true; const setlist* it=n->inner(n->nbranch);
      tt=it->term_type(); tp=it->term_branch(); retain(_e,tt,tp); }
   auto out = build_node(n->prefix(), scratch, nb, oterm, tt, tp);
   release_ref(t, p);   // drops the nested slot; its inner u32 children get released,
                        // exactly cancelling the retains above (net: transferred)
   return out;
}
```

Then `edit_node` does: `if (no room) { auto pr = promote_nested(wt,wp); return edit_node<unique>(pr.first, pr.second, key, depth, val); }`. After promotion the node is a normal flat setlist/double/full and the standard path inserts. This **never grows nested→big_nested in v1** (big_nested is a later tier); promotion always lands on existing flat tiers.

**Alternative (deferred):** grow to `big_nested` (cl2, 2 cachelines, u16 prefix) keeping the 2-level shape. Defer — it's a second layout with its own decode set. v1 promotes to flat.

---

## 5. COW + RELEASE (exact code shapes)

### `read_branches` nested arm (line 595, the linchpin) — enumerate REAL u32 children

```cpp
if (t == bt::nested_setlist)
{
   const nested_setlist* nn = static_cast<const nested_setlist*>(resolve(e, t, p));
   for (uint32_t i = 0; i < nn->nbranch; ++i)
   {
      const setlist* in = nn->inner(i);
      const uint8_t obyte = nn->okeys()[i];
      for (uint32_t j = 0; j < in->nbranch; ++j)
         f(br{ /*byte synthesized*/ in->keys()[j], in->btype(j), in->branch(j) });
      // NOTE: a flattened nested loses the 2-level structure — see Stage 3 decision.
   }
   return;
}
```

**Critical subtlety:** `read_branches` is used by `rebuild`/`reprefix` to produce a flat `br[]` for `build_node`, AND by `release_ref`/`reprefix-clone` to retain/release children. A nested's "byte-branch" at the OUTER level maps to an inner setlist, whose own branches have inner bytes — flattening to a single byte stream is **lossy** (you'd merge two levels of bytes). 

**Therefore DO NOT make `read_branches` flatten nested for rebuild.** Instead: **`rebuild` and `reprefix-drop>0` must call `promote_nested` first** (Stage 4) so they only ever see flat nodes. `read_branches` for nested is used ONLY by the retain/release walks, where the contract is "visit every real u32 child once for refcounting" — and for that, the byte value is irrelevant. So give `read_branches` a nested arm that visits inner u32 children (byte value cosmetic), and **guarantee rebuild/reprefix never receive nested** by promoting upstream.

Cleaner split: add a dedicated **`for_each_nested_child(p, f)`** used by release/retain, and keep `read_branches` asserting `t != nested_setlist` (or promoting). Pick ONE:
- **Chosen:** `read_branches` gets the nested arm above (visits real u32 children, synthetic byte) — safe for retain/release. `rebuild`/`reprefix-drop>0` are guarded with `if (t==nested) { auto pr=promote_nested(...); t=pr.first; p=pr.second; }` at their top so they never build *from* nested.

### `release_subtree` nested arm (line 442 `else`, before setlist)

```cpp
if (t == bt::nested_setlist)
{
   nested_setlist* n = static_cast<nested_setlist*>(resolve(e, t, p));
   for (uint32_t i = 0; i < n->nbranch; ++i)
   {
      setlist* in = n->inner(i);
      for (uint32_t j = 0; j < in->nbranch; ++j) release_subtree(e, in->btype(j), in->branch(j));
      if (in->has_term()) release_subtree(e, in->term_type(), in->term_branch());
   }
   if (n->has_term())
   { setlist* it = n->inner(n->nbranch); release_subtree(e, it->term_type(), it->term_branch()); }
   e.cl1.free(p);   // ONE free for the whole 2-level structure
   return;
}
```

(The `release_one(e,t,p)` guard at line 432 already ran and hit zero before reaching here — nested routes through cl1.release_one ✓. So this arm goes AFTER the `if(!release_one)return` and inside the post-zero block, structured like the `full`/`else` arms.)

### `release_async_` nested arm (line 479) — identical with `--budget` per node + `release_async_` recursion. Mirror verbatim.

### COW clone — `reprefix` drop==0 (line 872 `else`): **already correct.**
- `else` branch memcpies 128B (line 873) and sets `out = {t, s}` → tag preserved as nested ✓.
- The inner setlists, being tail-relative (`ooff` = offset-from-END), are byte-identical in the copy and remain valid — **this is why we store offset-from-END.**
- Line 875 retain loop calls `read_branches(out, ...)` → with the nested arm (above), it retains every real inner u32 child exactly once. ✓
- Line 877 `release_ref(t,p)` drops the source nested (its inner u32 children get one release each, balancing). ✓
- **No edit to reprefix-drop==0 needed** beyond the `read_branches` nested arm.

### `release_ref` (line 564) — no edit; relies on `read_branches`(#5) + `node_term`(#4) + `free_node`(#9, →cl1) which now handle nested.

---

## 6. Staged build order with a green gate after each

Validation gate after EVERY stage:
```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B /tmp/gcart_build /Users/dlarimer/gofractally/psitri
ninja -C /tmp/gcart_build gcart-spec-tests gcart-spec-kv-tests
/tmp/gcart_build/.../gcart-spec-tests      # expect "0 fails"
/tmp/gcart_build/.../gcart-spec-kv-tests   # expect pass
```
(Confirm exact binary path from CMake; targets are `gcart-spec-tests`, `gcart-spec-kv-tests`, `gcart-kv-tests`.)

- **Stage A — layout only.** Add `struct nested_setlist` (Stage 1) + `static_assert`. No dispatch wired. GATE: compiles, both oracles green (nested unreferenced).
- **Stage B — decode arms, inert.** Add nested arms to `find_in`, `node_prefix`, `node_term`, `read_branches`, `node_child` (Stage 2). Add nested arms to `release_subtree`/`release_async_` (Stage 5). Keep `try_pack_nested` → `false` (no create). GATE: both oracles green — nested still never created, so every arm is dead code but compiles and the existing flat paths are untouched.
- **Stage C — promote + COW guards.** Implement `promote_nested` (Stage 4); guard `rebuild` + `reprefix-drop>0` to promote-on-entry if nested. Verify `reprefix-drop==0` retain via the nested `read_branches` arm. GATE: green (still no create, so promote is dead but compiled).
- **Stage D — create, behind a knob.** Implement `pack_nested_setlist` + `try_pack_nested`; gate it behind `if (getenv("GCART_NESTED"))` (use `sal::env`, not raw getenv per the env-convention rule). GATE: oracle green with knob OFF; turn knob ON → oracle MUST stay 0-fails (this exercises create→read→COW→promote→release end to end). bytes/key should drop vs flat. Then make ON the default once green.
- **Stage E — in-place inner maintenance (optional perf).** Replace promote-on-every-write with true in-place inner insert (Stage 3). GATE: green + write throughput check.

Each stage is independently revertible; nested is inert (Stages A–C) until create fires (D).

---

## 7. Top risks / easy-to-miss traps

1. **Offset direction (the #1 bug).** `ooff` MUST be offset-from-END (`inner = base+128-ooff[i]`), NOT from-base. Offset-from-base would be invalidated by `reprefix-drop>0`'s rebuild and is fragile; offset-from-END survives the verbatim 128B memcpy (the whole point). The `inner(i).size = ooff[i]-ooff[i+1]` ordering (descending) must match how `pack_nested_setlist` lays them — get the sentinel for the last inner right (`tail_used` is the low watermark).

2. **`read_branches` flattening is lossy for rebuild.** A nested has TWO byte levels; `read_branches` emitting a single byte stream is correct ONLY for refcount walks, NOT for reconstructing a node. **Guard `rebuild`/`reprefix-drop>0` to `promote_nested` first** so they never build from a nested. If you skip this guard, rebuild will silently merge outer+inner bytes and corrupt the trie. This is the subtle correctness trap the maps flagged as "build_node downgrade."

3. **Missed dispatch site = silent wrong-region.** The five decode arms (`find_in`, `node_prefix`, `node_child`, `node_term`, `read_branches`) and the two release walks (`release_subtree`, `release_async_`) are the complete set that cast non-full to `setlist`. Miss any one and a nested gets read as a flat setlist (its `okeys`/`ooff` misread as setlist `keys`/`types`). The region selectors (`resolve`/`free_node`/refcount helpers) are ALREADY correct for nested→cl1 — do not "fix" them.

4. **COW retain double-count / under-count.** On `reprefix-drop==0`, retain each inner u32 child EXACTLY once (the nested `read_branches` arm). The inner *setlist bytes* are NOT refcounted (no slot) — only the inner's u32 branches are. Retaining the outer offsets as if pointers (the bug if you forget the arm) corrupts cl1 refcounts. On `promote_nested`, the retain-then-release_ref must net to transfer (retain inner children, then `release_ref(nested)` releases them once → net zero extra), matching `rebuild`'s is_unique transfer discipline.

5. **Release double-free.** A nested is ONE cl1 slot; its inners are NOT separate slots. The release arm must walk inner u32 children for the cascade but call `cl1.free(p)` **exactly once** for the whole structure — never `free` an inner. Conversely `promote_nested` creates real cl1 inners that DO get freed individually later; don't free them during promote (they're live children of the new flat node).

6. **Term handling — two terms.** A nested has an OUTER term (key ends at the outer prefix) AND each inner can have its own term (key ends at the inner prefix). Carry the outer term as `inner(nbranch)` (a 0-branch term-only setlist) so `node_term(nested)` reads it uniformly, and the release/retain walks must visit BOTH the outer term inner AND each inner's own `term_branch()`. Forgetting the inner terms leaks; forgetting to skip a missing outer term reads garbage at `inner(nbranch)`.

7. **`cap` semantics differ.** Inner setlists are tight-cap (`cap == nslot`, no growth slack) to maximize packing; a flat cl1 setlist uses `derive_cap` slack. Any in-place insert into an inner (Stage E) therefore CANNOT use `setlist_insert_inplace` (which assumes `total()<cap`) — it must re-lay the inner. v1's promote-on-write sidesteps this; if you add Stage E, this is the trap.

8. **`tag3` over-read at inner boundaries.** `tag3_get` does a 2-byte load and may over-read 1 byte past the packed types array. Inside a nested, an inner's `types[]` is followed by its `branches[]` or the next inner — the over-read is absorbed (same as flat). But the LAST inner (nearest base, lowest address... actually nearest tail) — verify its `types[]` tail doesn't over-read past `base()+128`. The 4-byte-aligned `branches_off_` padding in each inner setlist absorbs it; confirm `pack_nested_setlist` lays a full `setlist::bytes_for_cap` per inner (including that pad) so no inner's last tag3 read crosses `base()+128`.

9. **`find_in` prefetch.** The nested arm reads two setlist levels from ONE already-prefetched cacheline (per spec: `dispatch<NestedSetlist>` issues no extra prefetch). Don't add a child prefetch between outer and inner — they're co-resident; only prefetch the *escaping* u32 child (`resolve(e, t, p)`) after selecting the inner branch, mirroring the flat setlist tail at line 421.

---

Files to edit (both absolute): `/Users/dlarimer/gofractally/psitri/libraries/gcart/include/gcart/spec_tree.hpp` (all of Stages 1–6); `/Users/dlarimer/gofractally/psitri/libraries/gcart/include/gcart/spec_kv.hpp` (NO edits — root tag is opaque). Validation binaries: `gcart-spec-tests` (oracle vs std::map, 200k mixed keys, must print `0 fails`) and `gcart-spec-kv-tests`. The single hardest, defer-and-measure decision is the **create heuristic in `try_pack_nested`** (Stage D) — ship the layout/decode/COW/release/promote machinery green-and-inert first (Stages A–C), then turn on create behind a `sal::env` knob and confirm 0-fails + bytes/key improvement before defaulting it on.
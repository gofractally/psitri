# Radix-shape fix: build the canonical/minimal radix in sal_art_ref

**Goal (data-independent invariants of an ideal radix):**
1. **No single-child internal nodes** — every internal router is a true branch point (≥2 children); a single-child path lives as a node *prefix*, not a separate node+hop.
2. **Variable-length prefixes** — a node's prefix is sized to the full shared prefix, not capped. Every prefix byte is free height reduction.

**Why (measured):** on a 5M synthetic base, **58% of setlist_u8 are 1-entry** (`prefix=tail, term→page, 0 stems`) — redundant indirection. At 8M Erigon: 63.4M setlists / 15.3GB, ~58% redundant ≈ ~9GB + extra hops (read latency). This is the biggest structural lever, and it impacts perf, not just size.

## Root cause
`make_divider_tail` / `make_prefix_chain` / `divider_prefix_split` materialize a divider's unique tail as a **chain of single-branch setlist_u8**, each capped at `divider_prefix_cap = bitset_r2::prefix_capacity = 65`. The cap is a **SIGBUS workaround**: `try_upgrade` copies a node's prefix into a wider tier whose CL0 prefix slot is smaller (dense tiers put branches at FIXED CL offsets), so a >65 prefix overruns CL1 → bus error. Chunking both caps prefixes AND manufactures single-child nodes.

## Precedents in-tree
- `make_sized` (radix_engine.hpp:5182) already sizes `alloc<N>(sizeof(N)+pfx+extra+headroom)` and writes the prefix into CL0's variable region.
- Wide setlist (`stem_width>1`) already grows prefix+stems+branches past CL0 with a variable `head_size()` + `index_cap()` (radix_nodes.hpp:119,369).
- `leaf_page` stores its cpfx as a variable tail-heap region.

## EXECUTION WRINKLE (found during impl)
`setlist_u8` (stem_width==1) uses a **fixed 1-CL index** (`var_capacity`, branches end-anchored) — it canNOT hold a >65 prefix as-is. So a long-prefix node must be either a **wide setlist** (variable head, already supports it) or `setlist_u8` widened to variable. Decision: long-prefix collapse nodes use the variable-layout tier; `setlist_u8` stays fixed for the dense common case.

## Steps (smallest-risk-first, each conformance-gated)
- **Step 0 — baseline:** build `sal_art_ref-tests` (2,281,199 assertions), `kv-compat-test` (sal_art_ref PASS), `tree-shape-audit` (records 1-entry setlist count). Re-run all 3 after every step.
- **Step 1 — variable-prefix layout (behavior-neutral w/ cap in place):** `radix_impl::alloc_size(prefix)` sizes to prefix; `prefix()/set_prefix()` use runtime capacity; `head_size()` prefix-aware. Inert until Step 3 (all prefixes ≤65 today → byte-identical). Lands alone.
- **Step 2 — upgrade capability gate (behavior-neutral):** `try_upgrade`/`try_upgrade_c1`/`rebuild_into` DECLINE (ok=false) when `cn->prefix().size() > To::prefix_capacity`; `grow_and_set` falls back to staying setlist. Replaces the SIGBUS assert. Inert with cap in place.
- **Step 3 — single-child collapse (the change; needs 1+2):** `make_divider_tail`/`make_prefix_chain` → ONE prefixed node (or direct page ptr); `divider_prefix_split` np takes full common prefix; mid-router carries shared continuation; delete `divider_prefix_cap`. Long-prefix nodes use the variable tier. Re-audit → 1-entry → ~0.
- **Step 4 — debug invariant:** assert no published internal node has exactly one child WHERE the child is a router and there's no term (exempt root + single-edge-to-page/leaf, which are legit terminal edges).

## Gates
sal_art_ref-tests (debug, asserts on) → kv-compat (differential + long-key SIGBUS regression) → tree-shape-audit (single-child→~0) → TSan on the deep-nest case (COW-to-root, immutable-shared).

## ATTEMPT 1 RESULT — Option 2 (cpfx-discrimination) is UNSOUND (reverted)

Implemented (a) make_divider_tail→page-direct, (b) route_to_page page-case discriminates vs
`common_prefix()`, (c) is_page divergence recovers tail from cpfx. **`sal_art_ref-tests` caught it:**
find misses + iteration-count mismatch + a hard `assert("divider already present")` on divider
`"log11"`. Root cause — the plan's premise is BACKWARDS:

- A page's **divider** (lower-bound separator) is `get_split_pos`'s output = `cpfx + b[0..p] + v`,
  which is `R.min`-ish but with `v` rounded down for trailing zeros (leaf_page.hpp:639-657). The
  **cpfx is a PREFIX of the divider** (`cpfx ⊆ divider`), NOT `cpfx ⊇ divider`. The single-child
  setlist holds the divider's distinguishing tail `b[0..p]+v` — bytes BEYOND the cpfx that the page
  cannot reconstruct (cpfx is shorter; `v` may equal no key byte). So discriminating by cpfx routes
  e.g. key `"log11"` (which lives in an earlier page) into a later page whose cpfx is `"log11"` but
  whose min key is `"log110"` → false negatives + divider collisions.
- **The single-child setlist is NOT redundant** — it carries the genuine separator. It is only
  "wasteful" as a separate 64-byte SAL node (one hop / one cache miss per affected page).

## CORRECTED understanding + the real fix
- Dividers are ALREADY minimal (get_split_pos: `cpfx + p + 1` bytes, `v` rounded). The per-page
  distinguishing tail is short (~1-2 bytes for random keys); the cpfx part of the divider is already
  consumed by the parent path, so the setlist's prefix is just `b[0..p]+v`.
- The remaining overhead is the NODE COUNT (≈1 short single-child node per page), from (1)
  make_divider_tail when the tail is non-empty AND (2) divider_prefix_split's demoted dnode (a
  reprefixed single-child page-holder).
- **The only correct collapse = store the divider's distinguishing tail ON the leaf_page** (a small
  variable-length field, e.g. `_div_tail` / its length), attach the page directly under the branch
  byte, and have route_to_page's page-case discriminate against the page-stored divider tail (same
  bytes the setlist held → behavior-identical, hence CORRECT). Maintain it across split_op /
  divider_prefix_split / COW init_from. This is a structural change to the HOTTEST node + read path.
  Decision point for the user: meatier hot-node change for ~1 hop / page saved on 16-58% of pages.

## ATTEMPT 2 — SHIPPED (page-stored divider) ✅
The CORRECT fix, landed + gated green in the canonical tree.

**Design:** store the page's full divider (lower-bound separator) ON the leaf_page as a tail-heap
field `_div` (`_div_len`/`_div_pos`, mirrors `_cpfx`). The page attaches DIRECTLY under its branch
byte (no single-child setlist); route_to_page's page-case discriminates `key[depth..]` vs
`_div[depth..]` (root/leftmost = no-fallback catch-all); divider_insert's is_page branch reads the
existing page's `_div` to build a true ≥2-child divergence router (np capped at divider_prefix_cap,
overflow → make_prefix_chain). `_div` is set once at split (right half = `get_split_pos().divider()`,
left half = inherited `pg->divider()`, leftmost/root = empty), copied verbatim by every COW/rebuild.

**Edits:** leaf_page.hpp — field decl, init_empty zero, init_from `div` param + tail write,
clone_insert/update/remove + compact_to pass `src->_div` verbatim, split_op `div` field + ctor,
`divider()`/`div_data()` accessors, validate_invariants `live += _div_len`, sizeof 162→166.
radix_engine.hpp — make_divider_tail returns page-direct, route_to_page page-case + take_fallback
lambda, divider_insert_in is_page np-divergence via `_div`, split_op `.div` wiring.

**Results (synthetic, tree-shape-audit AUDIT_BUILD):**
- 1M klen=20: single-child setlists 5472→0, total nodes 39215→33743 (−14%), depth-3 pages 5506→36
  (99.9% reached in 2 hops). root = flat full_radix → pages direct.
- 5M klen=20: 0 single-child (64789 setlists all ≥2-child routers; depth-3 = inherent 256-way fanout).
- 500k klen=200 (long dividers, >65 = the SIGBUS-prone path): 0 single-child, no crash, count exact.

**Gates green:** sal_art_ref-tests 2,281,205 (debug asserts + release), kv-compat sal_art_ref 0
failures, count_keys exact at 1M/5M/500k, downstream adb+delta_art+psitrimdbx compile.

**NEXT:** mirror these edits to the Silkworm submodule (~/silkworm-psitri/third_party/psitri) +
re-run the honest sync-vs-lmdbx perf (the overarching goal). divider_prefix_cap KEPT (np + np-cap
still need it; cap deletion is separate variable-prefix work, now lower priority since shape is fixed).

## Bounds where canonical can't hold
- Dense tier + >65 prefix: impossible (fixed branch offsets) → node forced to stay setlist (Step 2 fallback). Single-child invariant still holds; only tier constrained. Near-impossible case.
- `_prefix_length` is 12-bit (4095); keys ≤1024 → length never the bound. Add `static_assert(max_key_len < 4096)`.
- Legit single-edge-to-page (root above one page) is NOT the bug — Step 4 assertion exempts it (audit's `sl1_childptr`=router is the bug; single-edge-to-page/leaf is fine).

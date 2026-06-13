# heart

An alternative radix-tree storage engine for PsiTri, built directly on SAL.

**Design source:** `~/myartlib/design3.md` ("Headerless ART Database Design").

`heart` depends only on `sal` (for `ptr_address` / allocator types) and `ucc`.
It does **not** depend on `libraries/psitri` — it is a parallel engine, not a
patch on the existing B-tree-style trie.

The guiding idea: make the *pointer* a micro-header. A tagged `control_block`
type tells the reader the node's dispatch shape so descent jumps straight to the
cacheline it needs without first reading a node header (the header is only
consulted on upsert — for refcount/segment ownership — and on MVCC descent).

## Phase 1 — utility classes (this directory)

These are the leaf primitives that later compose the node hierarchy. Each is
built and vetted in isolation before any node type is written.

| Header | Purpose |
|--------|---------|
| `include/heart/bitset.hpp` | Fixed-width `bitset<N>` (N a multiple of 64). `rank()` (a.k.a. `lower_bound`) maps a present key byte to its dense slot in a parallel branch array; plus `count/test/set/reset/next_set/first_set`. |
| `include/heart/smart_address.hpp` | `smart_address` / `tail_ptr<T>` — **the canonical branch type**: a 32-bit slot that is *either* a `ptr_address` (a child node) *or* an inline tail offset (a local allocation on the page), distinguished by value (top 2^16 = tail offsets; `0xFFFFFFFF` = null in both interpretations). |
| `include/heart/compact_addresses.hpp` | `compact_addresses<ByteCapacity>` — a dense `smart_address` array that deduplicates the shared 28-bit cacheline base (16 control blocks per 128 B cacheline) and stores only a 4-bit slot per branch. `insert/remove/at/set/get_hint`. |
| `include/heart/radix_bitset.hpp` | `radix_bitset<Bits, Size>` — composes a `bitset<Bits>` (`present`) with `compact_addresses<Size - sizeof(present)>` (`branches`); maps key byte → `smart_address` via `present.rank(byte)`. A generic container (no node logic), `Size` bytes total. `split`/`merge` compose it from two half-width sets; `expand`/`contract_from` convert to/from `radix_full`. |
| `include/heart/radix_full.hpp` | `radix_full<Bits>` — the dense tier: a byte-indexed `smart_address[Bits]` (Bits=256 → 1024 B = 8 cachelines), null `smart_address` where absent. Branchless `branches[byte]` routing. |


### Node hierarchy (CRTP layering; each layer adds fields, all packed)

`alloc_header(16) → versioned(+4) → paged(+2) → radix_base(+7) → radix_impl<Derived,FixedHeader> → Derived`

| header | role |
|--------|------|
| `include/heart/versioned.hpp` | `versioned : sal::alloc_header` — MVCC layer: adds `version_history` ptr; node version reuses the `alloc_header` 48-bit field (`created_version()`). (20 B) |
| `include/heart/paged.hpp` | `paged : versioned` — arena: tail heap from `alloc_header::tail()`, blocks addressed by relocation-stable `tail_ptr`; `alloc_tail`/`resolve`/`free_bytes`/`resolve(smart_address)`. (22 B) |
| `include/heart/radix_node.hpp` | `radix_base : paged` (rtype, term_value smart_address, prefix_length; `stamp_shape()` = `dispatch_shape{r1,r2,r4,r8}` from prefix+tier) + CRTP `radix_impl<Derived,FixedHeader>` filling CL0 with FixedHeader + inline prefix. radix_base = 29 B. |
| `include/heart/radix_node.hpp` (cont.) | `refcount_roots<K>` + `large_radix<Derived,K>` — base for the dense family; `FixedHeader` = K packed-count blocks (the 4-bit refcount-tree roots, group-counter COW). |
| `include/heart/radix_nodes.hpp` | the concrete taxonomy (ART N4/N16/N48/N256): `suffix_value`/`prefix_next` (1-child, R1), `setlist_tiny` (<4), `setlist<StemType>` (≤16), `bitset1` (K=1, 1 CL), `bitset_r2` (K=2, 3 CL), `bitset_r4` (K=4, 5 CL), `full_radix` (K=4, 9 CL). Sizes byte-pinned to 1/3/5/9 cachelines; `tier` (dispatch_shape) is a per-node `static constexpr`. |
| `include/heart/leaf_node.hpp` | `leaf_node : versioned` + `inline_value` — a versioned (MVCC) key→value leaf; not paged, not radix. |
| `include/heart/inline_value.hpp` | `inline_value` — `suffix_len/value_len/value_type + data[]`; the inline object a leaf embeds and a radix branch tail-allocates (≤64 B). |
| `include/heart/version_history.hpp` | `version_history<N>` — a radix node's table of older versions: `{version, frozen node (owning), version_node handle (non-owning)}` rows; N≈7 at 128 B. |
| `include/heart/version_node.hpp` | `version_node : sal::alloc_header` — per-txn version handle; its lifetime *is* the version's lifetime (gates dead-version GC). |

Tests: `test/*_tests.cpp` (Catch2). Build and run:

```bash
ninja -C build/release heart-tests
./build/release/bin/heart-tests
```

## SAL tag range (resolved)

`smart_address` packs a 16-bit inline tail offset into the top **2^16** of the
`ptr_address` space. SAL reserves exactly that range
(`sal::reserved_ptr_address_count == 1<<16`,
`sal::first_reserved_ptr_address == 0xFFFF0000`): its alloc-hint path skips
reserved values and its deref/prefetch helpers reject them, so a branch column
holding tail tags can double as a SAL alloc-hint source without SAL ever treating
a tag as a real address. A `static_assert` in `smart_address.hpp` ties the tail
range to SAL's reserved range so they can't silently drift apart.

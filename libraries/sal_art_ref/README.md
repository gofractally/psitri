# sal_art_ref — the Large Engine

A copy-on-write key/value engine on the SAL allocator (`libraries/sal`). It
depends on `sal`, `ucc` and `adb` (the engine concepts) and on nothing else in
this repository. The engine-independent API over it is `psitri::big_engine`
(`libraries/psitri_api`); what an embedder may rely on is in `docs/api/`.

## Shape

Keys live in 2 KB **leaf pages**, sorted, with their common prefix factored out.
Above the pages is a **radix of dividers**: each page is reached through the
shortest key that separates it from the page before it, and a lookup routes to
the last divider not above the key. A page that cannot take a key is split, or
the key gets a page of its own; values too large for a page record live in
their own allocations. Nothing is modified in place once another version can
see it: a writer copies the page and the routers on its path, and versions
share everything else by reference count.

Keys are at most 1,024 bytes. Roots are independent: one writer at a time per
root, any number of roots written at once, readers never wait.

## Files (`include/sal_art_ref/`)

| Header | What it holds |
|---|---|
| `engine.hpp` | the engine as the `adb` concept sees it |
| `radix_engine.hpp` | `radix_tree`: routing, the page operations, copy-on-write, roots, transactions, savepoints, cursors, sessions, the database |
| `leaf_page.hpp` | the 2 KB page: columns, records, common prefix and divider, split planning |
| `inline_value.hpp` | a page record: suffix, value kind, payload |
| `key_val_node.hpp`, `large_value.hpp`, `value_read.hpp` | values (and long keys) stored outside a page, in one allocation or chunked |
| `value_hooks.hpp` | typed values: retain/release callbacks by type number |
| `radix_node.hpp`, `radix_nodes.hpp`, `radix_c1.hpp`, `radix_bitset.hpp`, `radix_full.hpp` | the router nodes: sparse (`setlist`) and dense (`bitset_c1`, `bitset_r2`, `bitset_r4`, `full_radix`), and the per-range share counters of the dense ones |
| `bitset.hpp`, `compact_addresses.hpp`, `flat_addresses.hpp`, `smart_address.hpp` | building blocks of the nodes |
| `paged.hpp`, `versioned.hpp` | the allocation header layers under pages and nodes |
| `radix_type_ids.hpp` | the SAL type id of every object kind |
| `env_config.hpp` | settings read once from the environment |
| `leaf_node.hpp`, `bucket_node.hpp` | terminals of the byte-radix layout that predates leaf pages |

`doc/micro_refcount.md` describes the per-range share counters.

## Tests

`test/*_tests.cpp` (Catch2), target `sal_art_ref-tests`. `review_fixes_tests.cpp`
holds one reproduction per defect found by review. The cross-engine suites are
under `tests/` at the top of the repository.

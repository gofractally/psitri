# Engine profiles and Psibase

## Concrete types

| Type | Implementation | Intended use | Current API status |
|---|---|---|---|
| `psitri::small_engine` | `start_ref::kv::engine` | Resident data, direct-address read performance | Version 1 contract and shared behavioral suite |
| `psitri::big_engine` | `sal_art_ref::radix::engine` | Data beyond RAM; movable, compactable storage with refcounts | Version 1 contract and shared behavioral suite; Psibase integration candidate |
| `psitri::default_engine` | Alias of Big | Default for new generic examples | Source alias, not disk migration |
| `psitri::experimental::tiered_engine<Hot, Base>` | `delta_art::engine<Hot, Base>` | Hot/warm delta rotation and background base merging | Experimental; missing the common owned composite snapshot contract |
| `psitri::experimental::write_engine` | `tiered_engine<small_engine, big_engine>` | Refcount-only write-throughput candidate | Experimental; not `psitri::engine` |

The names describe the workload: **Small** is intended for data that fits in
RAM; **Big** is intended to scale beyond RAM. They do not specify pointer width,
a fixed payload threshold, or the maximum size of an individual value. The
current Small implementation uses five-byte handles, while Big addresses SAL
control blocks with 32-bit IDs. `engine32` and `engine64` would therefore
misdescribe these profiles. The historical 32-bit fixed layout remains a
separate benchmark result, not the current Small profile.

The initial `fixed_engine` and `compact_engine` spellings remain compatibility
aliases of `small_engine` and `big_engine`. Their old profile headers and CMake
targets (`psitri::fixed`, `psitri::compact`) also forward to the new names.
New applications use `psitri::small` / `<psitri/engines/small.hpp>` and
`psitri::big` / `<psitri/engines/big.hpp>`.

The tiered aliases are opt-in: include `<psitri/experimental/tiered.hpp>` and link
`psitri::experimental`. There is intentionally no supported `psitri::write_engine`
alias yet. Its transaction removal result and root semantics differ, it lacks
`write_session::get_root`, and coordinated durability/recovery requires separate
qualification. The strong common concept has not been weakened to admit it.

Header names, CMake targets, and source namespaces for historical code remain
available. `sal_art_ref::radix::dwal_engine` is a compatibility alias for the
public native `engine`. `<radix_engine.hpp>` and `<heart_dwal_engine.hpp>` in the
old test include directory forward to public headers; new consumers use
`<psitri/api.hpp>` or `<sal_art_ref/engine.hpp>`.

## Capabilities and limits

| Property | Small | Big |
|---|---|---|
| Liveness | Refcount + deferred release | Refcount + deferred release |
| Object location | Fixed allocation-line position | Movable through SAL control blocks |
| Committed snapshots / byte CRUD | Yes | Yes |
| Native reverse traversal | Forward rescan; reverse operations are O(n) | Yes |
| Root slots | 8 | 1,024 |
| Native subtree ownership | No | Yes |
| Primary-root savepoints | No common extension | Native `sub_transaction()`; restrictions below |
| Read-only open | Rejected | Supported with separate-process locking restrictions |
| Ordinary arena | Default 32 GiB; representable ceiling just under 512 GiB | SAL default storage ceiling 32 TiB, subject to platform limits |
| Logical value limit | 4 GiB − 1 byte; tested through 512 MiB | 4 GiB − 1 byte; tested through 512 MiB |
| Large-value storage | Fixed-address chunks, at most 2 MiB per allocation | Segment-aware chunks, almost 32 MiB per allocation |
| Clean reopen test | Passing | Passing |
| Full crash / beyond-RAM release gate | Not established by this pass | Not established by this pass |

The Small arena ceiling counts nodes and metadata, not payload. Current handles
are five bytes; the ordinary allocator uses 128-byte lines and a 32-bit line
index with reserved high indices. A separate large-value arena adds its own
reservation: up to `min(max_segments, 512)` segments of 2 GiB. That is
address-space capacity, with data pages touched as needed. The 2 MiB physical
allocation ceiling no longer limits logical value size; see
[chunked-value storage](chunked-values.md). Capacity and address-space reservation are not
RAM usage; keep the actual working set, snapshots, fragmentation, and release
backlog within the available memory for this profile's intended performance.

The historical **32-bit-handle** fixed implementation beat libart in published
in-RAM point lookup measurements: u64 hit/miss ratios 0.87/0.82 and dictionary
hit/miss 0.92/0.73. The same record shows slower inserts. That layout had a
28-bit line index plus a four-bit tag and a **32 GiB** ordinary arena ceiling.
See [the recorded results](../../libraries/start_ref/STATUS.md). These results
are not a new measurement of the current five-byte layout, and the current API
does not invent a `fixed32_engine` alias for an implementation it no longer
selects. Recovering that precise profile would require a separate supported
configuration and benchmarks.

## What Psibase should use

Target **`psitri::big_engine`**, with an explicit durable sync policy, as the
first native integration. It matches the capacity requirement and supplies
owned subtrees and copy-on-write snapshots needed for revision history. Keep
Psibase's revision/rollback adapter independent of the concrete engine where
the portable contract suffices; express the extra subtree requirements
explicitly. Small cannot supply those native subtree semantics by merely
changing the type alias.

Represent a revision's related database trees beneath **one manifest root**.
Attach owned subtree handles to that manifest and publish it once. Avoid
placing parts of one atomic revision in independently published root slots.
The [revision example](../../examples/api/revisions.cpp) demonstrates saving an
old state subtree, updating the active state, reopening, and restoring the old
state. Handles belong to their originating database and must never be copied
as raw bytes or installed into another database.

Big's savepoint frames snapshot the primary working root and the count of
opened children. They are not a general nested transaction over all roots or
all already-open child editors. Until that stronger behavior is verified, build
rollback around owned roots/manifest replacement and test Psibase's exact
nested-session semantics. Do not assume a `sub_transaction()` spelling alone
matches Psibase's undo stack.

Before replacing the deployed database, validate Psibase's value-size needs
with its own configured limits, revision retention and deletion, forks and
rollback, exceptions and allocation failure, process-kill recovery, and a
sustained workload exceeding physical RAM. Logical API portability does not
make on-disk formats interchangeable or remove the need to qualify an upgrade.
No Psibase backend is switched by this change. The MDBX shim also retains its
existing `start_ver`/Heart tiered default.

## Experimental inventory

| Source directory / namespace | Status |
|---|---|
| `experimental/gcart`, `gcart::*` | GC research; reference/benchmarks, not a release profile |
| `experimental/sal2gc`, `sal2gc::*` | GC allocator research |
| `experimental/rcart`, `rcart::*` | Hybrid refcount research using sal2 |
| `experimental/sal2`, `sal2::*` | Research fixed/movable allocator |
| `experimental/start_ver`, `start_ver::*` | Version/stamp liveness and reclamation experiments |
| `experimental/walart`, `walart::*` | WAL/write-set research |
| `psitri::experimental::*` | New opt-in experiment aliases, currently tiered/write |

The directory move deliberately preserved native namespaces to avoid an
unnecessary source rewrite. A namespace without the word `experimental` is not
a release endorsement. `delta_art` remains reusable library machinery, but its
current tiered profiles are experimental. The legacy DWAL concept is an
internal composition protocol, not an alternative public conformance standard.
GC is still a research direction. Ethereum/MDBX sync remains a useful benchmark;
API semantics, recovery, and scalable storage determine the release decision.

# Choose an engine

Use the **Small Engine** when your data and retained versions fit in RAM. Use the **Large Engine** for
persistent data expected to outgrow RAM, or when you need owned subtrees and
native savepoints. Both implement the same C++23 byte-database interface.

## Public types

| Type | Intended use | Include / CMake target |
|---|---|---|
| `psitri::small_engine` | Small Engine: resident data; direct-address ART | `<psitri/engines/small.hpp>` / `psitri::small` |
| `psitri::big_engine` | Large Engine: disk-backed, movable, compactable ART | `<psitri/engines/big.hpp>` / `psitri::big` |
| `psitri::default_engine` | Alias of the Large Engine | `<psitri/api.hpp>` / `psitri::api` |
| `psitri::database_for<E>` | Database type for a selected engine | `<psitri/api.hpp>` / `psitri::api` |

```cpp
#include <psitri/api.hpp>
using Engine = psitri::big_engine;
static_assert(psitri::engine<Engine>);
using Database = psitri::database_for<Engine>;
```

The engine is selected at compile time, without virtual dispatch. Pin an
explicit engine type for persistent application data. The Small Engine and Large Engine names
identify their storage model, not a limit on the size of an individual value.

## Capabilities and limits

| Property | Small Engine | Large Engine |
|---|---|---|
| Storage ownership | Reference counting + deferred release | Reference counting + deferred release |
| Object location | Fixed allocation-line position | Movable through SAL control blocks |
| Snapshot tree traversal | No node mutexes | No node mutexes; atomic residency tracking |
| Committed snapshots / byte CRUD | Yes | Yes |
| Reverse traversal | Forward rescan; reverse operations are O(n) | Native reverse traversal |
| Root slots | 16 | 1,024 |
| Native subtree ownership | No | Yes |
| Primary-root savepoints | No | Native `sub_transaction()`; scope below |
| Read-only open | Rejected with an error that says so | Supported with separate-process locking restrictions |
| Key limit | 1,024 bytes, enforced on every write | 1,024 bytes, enforced on every write |
| Reader slots | 64 (see below) | No such table |
| Unclean stop | Reported; free map and reference counts rebuilt from the roots at open | Reported; recovery at open |
| Storage ceiling | `max_segments`; exceeding it throws `storage_full` | `max_database_size` |
| Ordinary arena | Default 32 GiB; addressable ceiling just under 512 GiB | Default storage ceiling 32 TiB, subject to platform limits |
| Logical value limit | 4 GiB − 1 byte | 4 GiB − 1 byte |
| Tested allocated value size | Through 512 MiB | Through 512 MiB |
| Large-value storage | Fixed-address chunks, at most 2 MiB per allocation | Segment-aware chunks, almost 32 MiB per allocation |
| Read-driven hot-object promotion | Not applicable | Optional; disabled by default |

Both engines support parallel snapshot readers and **one writer per root at a
time, with writers on different roots running concurrently**. Each write
session has its own allocation state. See the
[concurrency contract](operations.md#concurrency-contract) before sharing objects
between threads.

## Small Engine: budget for the whole working set

The Small Engine follows fixed arena addresses without resolving movable-object control
blocks. Its five-byte handles use a 32-bit index of 128-byte allocation lines,
with high indices reserved. The addressable ceiling counts nodes and metadata,
not just application payload.

The default ordinary arena reserves 32 GiB. A separate large-value arena
reserves up to `min(max_segments, 512)` segments of 2 GiB each. These are virtual
address reservations; data pages are touched as needed. The 2 MiB allocation
size does not limit a logical value: [large values](chunked-values.md) span
multiple owned chunks.

For resident-data performance, budget for live data, snapshots, metadata,
fragmentation, and the release backlog within available RAM. A large address
reservation does not make the Small Engine a substitute for the Large Engine's relocation and compaction.

### Bounding and watching storage

`max_segments` is the ceiling: the ordinary arena never maps more than
`max_segments` × 8 MiB, and its data files never grow past that. Segments are
mapped as they are first needed and are never returned, so the mapped size is a
high-water mark. `db->usage()` reports, for the ordinary arena and the
large-value arena separately:

| Field | Meaning |
|---|---|
| `used_bytes` | Allocated 128-byte lines: live nodes, nodes kept by snapshots, nodes waiting for the releaser |
| `mapped_bytes` | Segments created so far: file size and the most that can be resident |
| `limit_bytes` | `max_segments` × segment size |

| `reserve_bytes` | Part of the limit kept for removes once the store cannot grow |
| `reserve_free_bytes` | Of the reserve, what a remove can draw right now |
| `reserve_owed_bytes` | Of the reserve, what it is waiting to get back (lent to removes, or not found free at open) |

It may be called from any thread. A write that would take the store past its
limit throws `psitri::small_engine::storage_full` (a `std::runtime_error`)
from the write call. Abort the transaction: it rolls back completely and what
was committed before is untouched.

What "full" means: once the store has all its segments, a sixty-fourth of it
(between 128 KiB and 1 MiB; 512 KiB of a 32 MiB store) is held out of ordinary
allocation as the **reserve**: free runs of lines, taken out of the free map,
counted in `limit_bytes` and never in `used_bytes`. An ordinary write is
refused when no segment has a free run of lines long enough for the
allocation. Before refusing, the writer releases versions that no reader holds
any more, waits for the background releaser to finish what is queued, and
looks for free lines in every segment, including segments held by sessions
that are between transactions. It does not wait for a segment that another
writer is in: with more writers in transactions at once than the store has
segments, the extra writers' ordinary writes are refused. In the tests a store
of 100-byte values took 174,000 keys and was 98% used (everything but the
reserve) when it first refused. A node that needs many lines in a row (a value
of several KiB) can be refused earlier on a fragmented store.

A full store is never stuck:

- **Remove keys.** A remove has to copy the path to its key before the old
  path is freed. On a store at its ceiling, `remove` and
  `remove_range_counted` take that room from the session's own segment if it
  has some and otherwise from the reserve. They claim no segment and wait for
  nothing, so they are not held up by other writers and do not depend on how
  many there are.

  What a remove draws from the reserve is owed back, and every run of lines
  freed anywhere in the store is offered to the reserve first while anything
  is owed. A debt is repaid with a run of the same length when a freed run is
  long enough (the old versions of the nodes the remove copied are such
  runs), and line for line when only shorter runs come free. Nothing about
  the reserve is kept on storage or remembered across an open: it returns to
  the free map at a clean close and is collected again, longest runs first,
  at every open and after every recovery.

  **Guaranteed:** after a transaction that only removes keys has committed
  and the versions it replaced have been released, the reserve holds at least
  as many lines as before it, indefinitely, across clean reopens and across
  recoveries, whatever else the store is doing; and ordinary writes never
  get lines of the reserve. `tests/api/small_engine.cpp` checks this after
  each of 30,000 rounds of "remove one key, refill until refused" with a
  reopen every 300 rounds, the same with a kill every 250 rounds, and with
  binary keys; no remove is refused.

  **The boundary:** the guarantee is in lines. A remove that needs a run of
  several lines (a node on the path that spans lines: a long key prefix, a
  leaf rebuilt next to the removed key whose value fills more than one line)
  gets it while the reserve has a run that long. The reserve starts as whole
  64-line runs and is repaid with runs as long as the ones it lent whenever
  the freed nodes are that long, so in practice it keeps runs of every size
  removes ask for; but it is not guaranteed for a node longer than any run
  the reserve has at that moment, and one transaction that removes very many
  keys can draw more than the reserve holds before it commits (commit in
  batches; a refused transaction rolls back and can be retried smaller).
  Values above 8 KiB live in the other arena and are not copied by a remove.
  `usage().reserve_free_bytes` is what removes can draw right now and
  `usage().reserve_owed_bytes` what the reserve is waiting to get back; a
  store opened too full to collect its reserve shows the shortfall there, and
  the first lines freed go to it.
- **Drop the root.** `tx.primary().set_root({})` followed by `commit()`
  allocates nothing and frees the whole tree.
- **Raise the limit.** Close the store and open it with a larger
  `max_segments`; the same files are used and nothing is rewritten.
  Lowering it is accepted only down to the number of segments the store
  already has (`usage().mapped_bytes`, and for the large-value arena
  `usage().large_mapped_bytes` against `min(max_segments, 512)` of its 2 GiB
  segments); below that, `open` throws `std::runtime_error` naming both
  sizes. Both arenas are checked before either is opened, so a refused open
  has not mapped, locked or written anything. The size that counts is the
  larger of the recorded segment count and what the data file holds: the
  file is what gets mapped. After a stop in the middle of growing, the file
  can be a few segments longer than the count; such a store is refused for a
  limit between the two until it has been opened once with a limit that
  covers the file, which trims it.

Overwriting a value is an ordinary write and is refused like an insert. Freed
space is reused in place; nothing is moved or compacted. Under the workload
the engine is meant for (rows rewritten in place plus an index entry moved to
a new key per update, short reader snapshots throughout)
`tests/api/small_engine.cpp` measured use and mapped size level from the first
sample to the last: 31 MiB used and 96 MiB mapped for 100,000 rows over 10
million updates, 316 MiB used and 928 MiB mapped for 1 million rows over 6
million updates. Plan for mapped size of about three times live size, more if
snapshots are held for long, and set `max_segments` with that in mind: a limit
close to the live size leaves a store that is always near full.

Values above about 8 KiB live in the large-value arena, which is bounded only
by its own reservation of `min(max_segments, 512)` segments of 2 GiB. A
deployment that needs a tight memory bound should keep values small or
account for that arena separately.

### Reader slots

There are 64 reader slots. Each read session holds one for its lifetime, and
each cursor a read session has open beyond its first holds one more while it
lives. When none is free, `start_read_session()` or `create_cursor()` throws
`std::runtime_error` naming the limit; nothing else is affected, and a slot is
reusable as soon as its holder is destroyed. A live cursor keeps the version
it reads, and everything that version shares with nothing newer, allocated.

## Large Engine: keep the working set close

The Large Engine uses stable 32-bit object IDs with movable storage. Background compaction
reclaims and rearranges segments. Optional access-frequency tracking promotes
hot objects and groups them together, improving locality without requiring the
entire dataset to remain resident.

Configure `max_pinned_cache_size_mb` and `enable_read_cache` before opening the
database. The pinned-segment budget is one part of process memory; allow for
metadata, writer buffers, retained versions, OS cache, and application memory.
See [configuration](operations.md#configuration) for an example.

The Large Engine's configured capacity ceiling is an address-space/storage limit, not a
measured throughput or scale guarantee. Both profiles pass the shared API and
clean-reopen tests; comprehensive crash/power-loss and sustained beyond-RAM
validation is incomplete. The [durability guide](operations.md#durability)
describes the supported policies and current validation coverage.

## Model application revisions with owned trees

The Large Engine can store a tree as a value in another tree. Put a revision's related tables
beneath **one manifest root**, attach owned subtree handles, and publish that
manifest once. Publishing independent root slots is not an atomic multi-root
transaction.

The [revision example](../../examples/api/revisions.cpp) saves a state subtree,
updates the active state, closes and reopens the database, and restores the
saved version. Handles belong to their originating database; do not copy their
raw IDs into another database.

The Large Engine's `sub_transaction()` saves the primary working root and the count of opened
children. Aborting restores that root and discards children opened in the frame.
It does not rewind all other roots or every child editor already open when the
frame began. Use an owned manifest replacement when your rollback boundary
covers a complete hierarchy of application state.

## Optional configurations

Experimental engines are separate from the shared source contract:

| Configuration | Purpose | Availability |
|---|---|---|
| `psitri::experimental::tiered_engine<Hot, Base>` | Hot delta with background merging into a base engine | `<psitri/experimental/tiered.hpp>`, target `psitri::experimental` |
| `psitri::experimental::write_engine` | Refcounted tiering with the Small Engine over the Large Engine | Same explicit experimental target |
| `gcart`, `sal_gc` | Garbage-collected storage | `experimental/` sources |
| `rcart`, `sal_hybrid` | Hybrid reference-counted storage and allocation | `experimental/` sources |
| `start_ver`, `walart` | Version-based reclamation and write-set storage | `experimental/` sources |

The tiered aliases do not satisfy `psitri::engine`: their removal results and root
semantics differ, and they lack its owned composite snapshot interface. Use the
maintained Small Engine and Large Engine profiles for applications targeting the shared contract.
A source namespace alone does not establish API support or production readiness.

## Data portability

The common API is a **source contract**. Changing engine types does not convert
files, and raw object IDs are not portable between databases. Move data through
logical export/import or a separately validated converter. Validate snapshot
retention, rollback, errors, recovery, and your workload's memory requirements
when adopting a different engine or upgrading a persisted format.

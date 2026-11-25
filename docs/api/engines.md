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

It may be called from any thread. A write that would take the store past its
limit throws `psitri::small_engine::storage_full` (a `std::runtime_error`)
from the write call. Abort the transaction: it rolls back completely and what
was committed before is untouched.

What "full" means: the reserve is a place, not an amount. It is the top of the
store's last possible segment (a sixty-fourth of the store, between 128 KiB
and 1 MiB; 512 KiB of a 32 MiB store), it is counted in `limit_bytes` but
never in `used_bytes`, and ordinary writes are never put there. An ordinary
write is refused when no segment has a free run of lines long enough for the
allocation. Before refusing, the writer releases versions that no reader holds
any more, waits for the background releaser to finish what is queued, looks
for free lines in every segment, including segments held by sessions that are
between transactions. It does not wait for a segment that another writer is
in: with more writers in transactions at once than the store has segments,
the extra writers' ordinary writes are refused. In the tests a store of 100-byte values took 174,000
keys and was 98% used (everything but the reserve) when it first refused. A
node that needs many lines in a row (a value of several KiB) can be refused
earlier on a fragmented store.

A full store is never stuck:

- **Remove keys.** `remove` and `remove_range_counted` first look for room
  like any write and, when there is none, take it from the reserve: a remove
  has to copy the path to its key before the old path is freed. The reserve
  is contiguous by construction, so this does not depend on how scattered
  the store's free lines are, and it needs no segment of the session's own,
  so it does not depend on how many writers are active. Lines of the reserve
  go back to the reserve when the node that used them is next rewritten or
  removed.

  What is guaranteed: on a store that has reported `storage_full`, a
  transaction that only removes keys succeeds as long as the nodes it has to
  copy fit in what is free in the reserve at that moment. One remove copies
  the nodes on the path to its key, a few lines for keys of ordinary length
  whatever the size of the values (values are not copied by a remove, and
  values above 8 KiB live in the other arena). The reserve holds at least
  1,024 lines, enough for hundreds of such paths before any of it has come
  back. `tests/api/small_engine.cpp` fills a two-segment store to refusal,
  scatters its free lines, refills it, and removes every key one transaction
  at a time down to empty; and runs three writers on two segments at the limit
  for thousands of remove transactions without one being refused.

  What is not guaranteed: a single transaction that removes very many keys
  can need more than the reserve before it commits (commit in batches; a
  refused transaction rolls back and can be retried smaller); and keys near
  the 1,024-byte limit have paths of hundreds of nodes, so a remove of such a
  key can need a large part of a small reserve. To keep the reserve from
  being used up by removes and refills taking turns at the brim, ordinary
  writes are refused a little early (at four reserves' worth of free lines
  instead of none) while more than half the reserve is lent out.
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

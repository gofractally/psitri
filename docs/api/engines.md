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

It may be called from any thread. A write that needs a segment beyond the
limit throws `psitri::small_engine::storage_full` (a `std::runtime_error`)
from the write call. Abort the transaction: it rolls back completely, what was
committed before is untouched, and the same session can go on writing as soon
as there is room (removing keys makes room). The limit is reported only when
no segment has a free run of lines long enough for the allocation: before
giving up, the writer waits for the background releaser to finish freeing
versions that are already dead, and takes free lines from any segment. In the
tests a store of 100-byte values filled to 100% of its limit before the first
refusal. A node that needs many lines in a row (a value of several KiB) can be
refused earlier on a fragmented store. Freed space is reused in place; nothing
is moved or compacted. Under the workload the engine
is meant for (rows rewritten in place plus an index entry moved to a new key
per update, short reader snapshots throughout) `tests/api/small_engine.cpp`
measured use and mapped size level from the first sample to the last: 31 MiB
used and 96 MiB mapped for 100,000 rows over 10 million updates, 316 MiB used
and 928 MiB mapped for 1 million rows over 6 million updates. Plan for mapped
size of about three times live size, more if snapshots are held for long.

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

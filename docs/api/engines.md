# Choose an engine

Use **Small** when your data and retained versions fit in RAM. Use **Big** for
persistent data expected to outgrow RAM, or when you need owned subtrees and
native savepoints. Both implement the same C++23 byte-database interface.

## Public types

| Type | Intended use | Include / CMake target |
|---|---|---|
| `psitri::small_engine` | Resident data; direct-address ART | `<psitri/engines/small.hpp>` / `psitri::small` |
| `psitri::big_engine` | Disk-backed, movable, compactable ART | `<psitri/engines/big.hpp>` / `psitri::big` |
| `psitri::default_engine` | Alias of Big | `<psitri/api.hpp>` / `psitri::api` |
| `psitri::database_for<E>` | Database type for a selected engine | `<psitri/api.hpp>` / `psitri::api` |

```cpp
#include <psitri/api.hpp>
using Engine = psitri::big_engine;
static_assert(psitri::engine<Engine>);
using Database = psitri::database_for<Engine>;
```

The engine is selected at compile time, without virtual dispatch. Pin an
explicit engine type for persistent application data. The Small and Big names
identify their storage model, not a limit on the size of an individual value.

## Capabilities and limits

| Property | Small | Big |
|---|---|---|
| Storage ownership | Reference counting + deferred release | Reference counting + deferred release |
| Object location | Fixed allocation-line position | Movable through SAL control blocks |
| Snapshot tree traversal | No node mutexes | No node mutexes; atomic residency tracking |
| Committed snapshots / byte CRUD | Yes | Yes |
| Reverse traversal | Forward rescan; reverse operations are O(n) | Native reverse traversal |
| Root slots | 8 | 1,024 |
| Native subtree ownership | No | Yes |
| Primary-root savepoints | No | Native `sub_transaction()`; scope below |
| Read-only open | Rejected | Supported with separate-process locking restrictions |
| Ordinary arena | Default 32 GiB; addressable ceiling just under 512 GiB | Default storage ceiling 32 TiB, subject to platform limits |
| Logical value limit | 4 GiB − 1 byte | 4 GiB − 1 byte |
| Tested allocated value size | Through 512 MiB | Through 512 MiB |
| Large-value storage | Fixed-address chunks, at most 2 MiB per allocation | Segment-aware chunks, almost 32 MiB per allocation |
| Read-driven hot-object promotion | Not applicable | Optional; disabled by default |

Both engines support parallel snapshot readers. The **portable API uses one
serialized writer**. Native SAL supports independent root writers, but the Big
API adapter shares a write session. See the
[concurrency contract](operations.md#concurrency-contract) before sharing objects
between threads.

## Small: budget for the whole working set

Small follows fixed arena addresses without resolving movable-object control
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
reservation does not make Small a substitute for Big's relocation and compaction.

## Big: keep the working set close

Big uses stable 32-bit object IDs with movable storage. Background compaction
reclaims and rearranges segments. Optional access-frequency tracking promotes
hot objects and groups them together, improving locality without requiring the
entire dataset to remain resident.

Configure `max_pinned_cache_size_mb` and `enable_read_cache` before opening the
database. The pinned-segment budget is one part of process memory; allow for
metadata, writer buffers, retained versions, OS cache, and application memory.
See [configuration](operations.md#configuration) for an example.

Big's configured capacity ceiling is an address-space/storage limit, not a
measured throughput or scale guarantee. Both profiles pass the shared API and
clean-reopen tests; comprehensive crash/power-loss and sustained beyond-RAM
validation is incomplete. The [durability guide](operations.md#durability)
describes the supported policies and current validation coverage.

## Model application revisions with owned trees

Big can store a tree as a value in another tree. Put a revision's related tables
beneath **one manifest root**, attach owned subtree handles, and publish that
manifest once. Publishing independent root slots is not an atomic multi-root
transaction.

The [revision example](../../examples/api/revisions.cpp) saves a state subtree,
updates the active state, closes and reopens the database, and restores the
saved version. Handles belong to their originating database; do not copy their
raw IDs into another database.

Big's `sub_transaction()` saves the primary working root and the count of opened
children. Aborting restores that root and discards children opened in the frame.
It does not rewind all other roots or every child editor already open when the
frame began. Use an owned manifest replacement when your rollback boundary
covers a complete hierarchy of application state.

## Optional configurations

Experimental engines are separate from the supported Small/Big source contract:

| Configuration | Purpose | Availability |
|---|---|---|
| `psitri::experimental::tiered_engine<Hot, Base>` | Hot delta with background merging into a base engine | `<psitri/experimental/tiered.hpp>`, target `psitri::experimental` |
| `psitri::experimental::write_engine` | Refcounted Small/Big tiering | Same explicit experimental target |
| `gcart`, `sal_gc` | Garbage-collected storage | `experimental/` sources |
| `rcart`, `sal_hybrid` | Hybrid reference-counted storage and allocation | `experimental/` sources |
| `start_ver`, `walart` | Version-based reclamation and write-set storage | `experimental/` sources |

The tiered aliases do not satisfy `psitri::engine`: their removal results and root
semantics differ, and they lack its owned composite snapshot interface. Use the
maintained Small/Big profiles for applications targeting the shared contract.
A source namespace alone does not establish API support or production readiness.

## Data portability

The common API is a **source contract**. Changing engine types does not convert
files, and raw object IDs are not portable between databases. Move data through
logical export/import or a separately validated converter. Validate snapshot
retention, rollback, errors, recovery, and your workload's memory requirements
when adopting a different engine or upgrading a persisted format.

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
| `reserve_free_bytes` | Lines the reserve holds now, in whatever runs |
| `reserve_longest_run_bytes` | The longest single run it can hand to a remove now |
| `reserve_long_runs` | How many runs of nine lines (1,152 bytes) or more it can hand out now |
| `reserve_owed_bytes` | What it is waiting to get back (lent to removes, or not found free at open) |

It may be called from any thread. A write that would take the store past its
limit throws `psitri::small_engine::storage_full` (a `std::runtime_error`)
from the write call. Abort the transaction: it rolls back completely and what
was committed before is untouched.

What "full" means: once the store has all its segments, a part of it is held
out of ordinary allocation as the **reserve** (by default a sixty-fourth of
the store, between 128 KiB and 1 MiB: 512 KiB of a 32 MiB store;
`config::reserve_bytes` sets it, up to 4 MiB). It is counted in `limit_bytes`
and never in `used_bytes`. An ordinary write is refused when no segment has a
free run of lines long enough for the allocation. Before refusing, the writer
releases versions that no reader holds any more, waits for the background
releaser to finish what is queued, and looks for free lines in every segment,
including segments held by sessions that are between transactions. It does not
wait for a segment that another writer is in: with more writers in
transactions at once than the store has segments, the extra writers' ordinary
writes are refused. In the tests a store of 100-byte values took 174,000 keys
and was 98% used (everything but the reserve) when it first refused. A node
that needs many lines in a row (a value of several KiB) can be refused earlier
on a fragmented store.

A full store is never stuck:

- **Remove keys.** A remove has to copy the path to its key before the old
  path is freed. On a store at its ceiling, `remove` and
  `remove_range_counted` take that room from the session's own segment if it
  has some and otherwise from the reserve. They claim no segment and wait for
  no other writer.

  A remove needs *runs* of consecutive 128-byte lines, one per node it
  copies, and how long depends on the node:

  | Node on the path | Lines |
  |---|---|
  | Router with few branches, or one line of a wide router | 1 |
  | Compact routers of 2, 4 or 8 lines of branches | 3, 5, 9 |
  | Full 256-way router copied whole | up to 14 |
  | Key prefix of p bytes | (p + 7) / 128, rounded up: 9 for a 1,024-byte key |

  A remove never copies a leaf, its own or a neighbour's, so the size of the
  values does not come into it: the longest run a remove can need is 14
  lines, whatever the keys and values.

  So the reserve is kept as runs, in two parts:

  - **Guaranteed runs.** For each length from 2 to 16 lines, a fixed number
    of runs of exactly that length. A guaranteed run is handed only to a draw
    of its own length, and its place is refilled only by a freed run at least
    that long. No number of shorter draws can use them up. Separately, a few
    runs of 64 lines are kept for draws of more than 16 lines; no remove
    needs one as the tree stands, and they are there as margin. What a draw
    leaves of such a run stays with it, and a run that comes back is joined
    to what it was cut from.
  - **Flexible room**: the rest, a little over three quarters. Any draw may
    cut a piece from it, and it is refilled line for line with whatever is
    freed, so short draws turn it into short runs over time.

  | Reserve | Flexible | Runs of 2 | of 4 | of 8 | of 9 | of 14 | of 16 | of 64 |
  |---|---|---|---|---|---|---|---|---|
  | 256 KiB (a 16 MiB store) | 194 KiB | 16 | 8 | 4 | 3 | 2 | 2 | 1 |
  | 512 KiB (32 MiB) | 383 KiB | 32 | 16 | 8 | 7 | 4 | 4 | 2 |
  | 1 MiB (64 MiB and up) | 759 KiB | 64 | 32 | 16 | 14 | 9 | 8 | 4 |

  (For a length k from 2 to 16 the number is the reserve's lines divided by
  64k, at least one.) A draw uses flexible room while that still has a run
  long enough, and a guaranteed run otherwise.

  What is drawn is owed back. Every run of lines freed anywhere in the store
  is offered to the reserve first while it is owed something that run can
  repay, and a node of several lines is freed as the one run it was. Nothing
  about the reserve is kept on storage or remembered across an open: it
  returns to the free map at a clean close and is collected again, longest
  free runs first, at every open and after every recovery. Collection takes
  one pass over the free map when whole free 64-line words are to be had (any
  store that is not nearly full) and at most seven otherwise; a pass reads one
  word per 8 KiB of store, about four million for a 32 GiB store.

  **Guaranteed, on a store at its ceiling:**
  - One transaction that only removes keys can draw at least the flexible
    room in runs of any length up to 64 lines while that room is fresh, and
    always, whatever was drawn before or earlier in the same transaction, the
    guaranteed number of runs of each length from 2 to 16. Single lines come
    from the flexible room only.
  - After that transaction has committed and the versions it replaced have
    been released, every run it drew has been repaid with a run of the same
    length, provided the versions released contained one, which they do for
    every node a remove copies. The reserve then holds what it held before:
    indefinitely, across clean reopens and across recoveries. What it holds
    and what it is owed always add up to its size.
  - Ordinary writes never get lines of the reserve.

  `tests/api/small_engine.cpp` checks the reserve after each of 30,000 rounds
  of "remove one key, refill until refused" with a reopen every 300 rounds,
  with a kill every 250, with binary keys, and with a syncing commit mode;
  checks that 40 removes behind a 900-byte key prefix succeed after one
  transaction of 1,500 short removes, after 1,500 removes under a reader, and
  after a batch that was refused and rolled back; and that removes through
  long nodes of several shapes leave nothing owed.

  **Not guaranteed:**
  - A transaction that removes more than the reserve can lend before it
    commits is refused; it rolls back and nothing is lost. In the tests one
    transaction removed about 9,800 small keys from a full two-segment store
    before that happened. Commit removes in batches, or raise
    `config::reserve_bytes`: the numbers above scale with it. This matters
    when one unit of work is one transaction, however large.
  - A remove that has to build a node longer than every run it frees leaves
    that length owed until a run that long is freed anywhere in the store.
    None of the shapes tested does this.
  - While a reader stays on a version that removes have replaced, what they
    drew stays owed: the old nodes cannot be freed under it. A reader twenty
    rounds behind was seen holding nearly the whole reserve owed.

  When the reserve has no run of the length a remove needs, the remove has
  the releases that are already queued carried out, waiting at most half a
  millisecond for the background releaser and never for another writer, and
  looks once more; after that it looks for a free run in the whole store like
  an ordinary write, and is refused if there is none.

  `usage()` says where things stand: `reserve_free_bytes` is how many lines
  the reserve holds, which is not what a remove can get;
  `reserve_longest_run_bytes` is the longest run it can hand out now, and a
  remove that must copy a longer node gets nothing from the reserve for it; `reserve_long_runs` is how
  many runs of nine lines or more it has; `reserve_owed_bytes` is what it is
  waiting to get back. A store opened too full to collect its reserve shows
  the shortfall as owed, gives the flexible room its share of what was found,
  and fills the rest from frees; guaranteed runs it could not find stay owed
  until runs that long are freed or the store is reopened with room.
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

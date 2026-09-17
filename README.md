# psitri

A standalone embedded database library built around copy-on-write adaptive radix
trees. Retain snapshots without copying the database, let readers traverse a
stable view while writes proceed, and choose a storage engine for resident or
disk-backed data through one C++23 source interface.

The **practically lock-free** design keeps node mutexes off snapshot traversal.
Atomic residency and lifetime tracking let readers operate in parallel with the
writer's tree work. Synchronization is concentrated at root acquisition and
publication, writer coordination, and allocator/maintenance boundaries.

See the [website](https://gofractally.github.io/psitri/),
[developer guide](docs/api/README.md), and [quickstart](docs/api/quickstart.md).

## One API, two storage profiles

```cpp
#include <psitri/api.hpp>
using Engine = psitri::big_engine; // or psitri::small_engine
static_assert(psitri::engine<Engine>);
```

Link `psitri::api`. Use `database_for<Engine>` for a database type selected at
compile time, without virtual dispatch.

- The **Small Engine** follows fixed addresses for data and retained versions that fit in
  RAM. Its default ordinary arena is 32 GiB; five-byte handles address just under
  512 GiB. Nodes, metadata, fragmentation, and snapshots consume capacity. Large
  values use a separate arena.
- The **Large Engine** uses movable reference-counted storage, segment compaction, and optional
  access-frequency-based hot-object promotion. Choose it for disk-backed data
  expected to outgrow RAM, owned subtrees, and native savepoints.

Both profiles provide byte CRUD, ordered scans, commit/abort, owned snapshots,
and range reads. Logical values may be up to 4 GiB − 1 byte; mixed allocation/free
is tested through 512 MiB. The Large Engine packs values into segment-aware chunks; the Small Engine uses
fixed-address chunks. The shared API permits parallel readers and one serialized
writer. Native SAL supports independent root writers.

Read the [capability matrix](docs/api/engines.md) and
[operating contract](docs/api/operations.md) for exact limits. The common API and
clean-reopen suites pass; comprehensive crash/power-loss and sustained beyond-RAM
validation is incomplete. Choose durability explicitly and validate your
application's workload and failure model.

## Keep the API. Rethink the engine.

The MDBX C/C++ compatibility layer lets existing applications evaluate psitri
through a familiar interface. Translating a foreign API's transaction, cursor,
and ownership workflow adds overhead and can restrict optimization paths exposed
by the native interface. Use native psitri for direct access to its ownership and
tree operations; measure compatible APIs through your actual application.

See [integration status and tradeoffs](website/content/compatibility.md).
API compatibility does not imply interchangeable database files. Migrate stored
data through logical export/import or a validated converter.

## Benchmarks

The [published methodology](website/content/benchmarks.md) identifies each
measurement's revision, hardware, workload, durability, and configuration.
Use the [benchmark recorder](scripts/benchmark.py) and
[comparison workflow](website/content/benchmarking.md) to retain raw output and
metadata and detect regressions under matched conditions. Keep results in `logs/`.

## Repository layout

| Directory | Role |
|---|---|
| `libraries/adb` | C++ database concepts and shared behavioral checks |
| `libraries/psitri_api` | Public profile names and consumer CMake targets |
| `libraries/sal` | Movable, refcounted storage and compaction |
| `libraries/sal_art_ref` | Large Engine: refcounted ART with packed pages on SAL |
| `libraries/start_ref` | Small Engine: fixed-address, refcounted ART |
| `libraries/delta_art` | Hot-delta/base composition; supplied tiered profiles are experimental |
| `libraries/psitri`, `libraries/art` | DWAL implementation and write-set machinery |
| `libraries/psitrimdbx`, `shims/` | Foreign-API integration code and integration directory |
| `experimental/` | Optional engines and allocators for evaluation and benchmarks |
| `tests/`, `benchmarks/`, `logs/` | Behavioral checks, comparison workloads, and retained evidence |
| `website/` | Public website source and deployment guide |

The public API target includes only maintained engine dependencies. Optional
configurations require explicit experimental headers and targets; their interface
and validation status are separate from the public engine contract.

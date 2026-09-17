# The psitri C++ API

psitri is a standalone embedded database library for transactional key-value
storage, inexpensive snapshots, and ordered traversal. Choose
`psitri::small_engine` when your data and retained versions fit in RAM, or
`psitri::big_engine` for disk-backed storage expected to outgrow memory.
Both implement the same C++23 byte-database contract.

```cpp
#include <psitri/api.hpp>
using Engine = psitri::big_engine;
static_assert(psitri::engine<Engine>);
using Database = psitri::database_for<Engine>;
```

Link the public CMake target `psitri::api`. Engine selection happens at compile
time, with no adapter object or virtual dispatch. The target includes only
maintained engine dependencies. `psitri::default_engine` aliases Big;
applications that own persisted data should select an explicit engine.

## Start here

1. [Build and first database](quickstart.md): CMake integration, transactions,
   reads, snapshots, and executable examples.
2. [API reference](reference.md): return values, cursor bounds, byte encoding,
   open modes, limits, and error handling.
3. [Lifetimes, concurrency, and durability](operations.md): ownership, mutex-free
   snapshot traversal, synchronization boundaries, configuration, and shutdown.
4. [Choose an engine](engines.md): concrete types, capabilities, capacity,
   optional configurations, and data migration.
5. [Implementing and validating an engine](conformance.md): the concept,
   behavioral checks, compatibility policy, and optional capabilities.
6. [Large values in Small and Big](chunked-values.md): storage layouts,
   segment-aware chunks, ownership, and allocation tests through 512 MiB.
7. [Range reads](range-reads.md): offsets and lengths, EOF semantics, borrowed
   views, and transparent owned fallback across chunks.

## The source contract

`psitri::engine<E>` is the version 1 source interface. It builds on
`adb::openable_engine<E, sal::open_mode>` and checks engine role types,
construction, sync configuration, rollback, owned snapshots, and bidirectional
reads. `adb` is independently usable and standard-library-only.
`psitri::api_version` is `1`.

Use `database_for<Engine>` for generic application code. The separate
`psitri::database` implementation is not the generic engine alias. The source
contract covers Small and Big in their supplied layouts; it does not promise
C++ ABI stability, interchangeable disk formats, or experimental internals.

The [basic example](../../examples/api/basic.cpp) builds for both profiles.
The [revision example](../../examples/api/revisions.cpp) demonstrates Big's
additional subtree ownership API, including close, reopen, and restore.
See [durability](operations.md#durability) for validation coverage and deployment
considerations.

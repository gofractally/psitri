# The psitri C++ API

Use `psitri::big_engine` for storage expected to outgrow RAM. Use
`psitri::small_engine` when the database and its retained versions fit in RAM.
Both implement the same C++23 byte-database contract and have compiling examples.
`big_engine` is the recommended **Psibase integration candidate**; API
conformance is not a claim of completed crash or beyond-RAM release qualification.

```cpp
#include <psitri/api.hpp>
using Engine = psitri::big_engine;
static_assert(psitri::engine<Engine>);
using Database = psitri::database_for<Engine>;
```

The public CMake target is `psitri::api`. There are no test-directory includes,
required adapter objects, virtual dispatch, or experimental engine dependencies
on this target. `psitri::default_engine` currently aliases `big_engine`.
Applications that own persistent data should select the engine explicitly.

## Read in this order

1. [Build and first database](quickstart.md): third-party CMake, transactions,
   reads, snapshots, and executable examples.
2. [API reference](reference.md): exact return values, cursor bounds, byte
   encoding, open modes, limits, and error handling.
3. [Lifetimes, concurrency, and durability](operations.md): the rules that
   make a correct application, including configuration and shutdown.
4. [Engine selection and Psibase](engines.md): concrete types, capabilities,
   capacity, experimental namespaces, and migration requirements.
5. [Implementing and validating an engine](conformance.md): the concept,
   runtime checks, compatibility policy, and extending the API.
6. [Large-value impact assessment](large-values.md): proposed 24-bit lengths,
   Psibase/Triedent requirements, format compatibility, and allocation limits.
7. [Large values in Small and Big](chunked-values.md): storage diagrams,
   direct-address and segment-aware chunks, ownership, and tests through 512 MiB.
8. [Range reads](range-reads.md): optional offset/length capability, EOF semantics,
   borrowed views, and transparent owned fallback across chunks.

## What is stable here?

`psitri::engine<E>` is the version 1 **source interface** to target in new
code. It builds on the canonical `adb::openable_engine<E, sal::open_mode>`
concept and requires a writable sync setting, with engine role types, construction, rollback, owned snapshots, and
bidirectional reads checked together. `adb` remains independently usable and
standard-library-only. `psitri::api_version` is `1`.

The maintained profiles are Small and Big in their default layouts. The
existing `psitri::database` name still denotes the legacy implementation; use
`database_for<Engine>` for new generic code. Disk formats, C++ ABI compatibility,
allocator internals, experimental types, and legacy layout switches are outside
this source contract. Changing an alias does not convert database files.

The [basic example](../../examples/api/basic.cpp) is built for both profiles.
The [revision example](../../examples/api/revisions.cpp) demonstrates Big's
additional subtree ownership API, including close, reopen, and restore.

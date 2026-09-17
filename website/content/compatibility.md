# Keep the interface. Evaluate the implementation.

psitri’s compatibility work puts a different storage engine behind APIs that
applications already use. The goal is a **near drop-in source integration for
the supported subset**, followed by an honest comparison using the same
application, calls, and benchmark workload.

This is different from the native C++ engine concept. The concept gives new
applications a shared source interface across maintained psitri engines.
Foreign-API integrations translate another database’s semantics into psitri.

## Integration status

| Interface | Current status | What to evaluate |
|---|---|---|
| Native C++23 | Maintained Small Engine and Large Engine profiles in this repository | Shared byte API, lifetime rules, sync policy, engine-specific extensions |
| MDBX C / C++ | Implementation in `libraries/psitrimdbx` | The calls, flags, comparators, duplicate handling, cursor and transaction semantics your application uses |
| SQLite | Separate storage-engine implementation; not included here | Replaces SQLite’s B-tree layer while retaining its SQL parser and virtual machine; port and revalidate against the maintained engines |
| RocksDB | Separate API wrapper; not included here | Supported API subset, snapshots, batches, transactions, options, and migration |
| LevelDB | Separate API wrapper; not included here | Supported byte API and batch/snapshot behavior |
| DuckDB | Separate storage extension; not included here | Catalog, scan, insert, and transactional integration |

The packaged integration in this repository is MDBX. SQLite, RocksDB, LevelDB,
and DuckDB implementations do not currently target its maintained engine
engine API and are not available as packages from this checkout. Evaluate the
supported subset before choosing an integration.

## What “near drop-in” does and does not mean

**The opportunity:** rebuild an application against a compatible API, preserve
its workload, and measure whether a different storage architecture improves it.
The same benchmark driver is especially useful: it reduces the chance that an
API redesign, different batching, or an easier workload creates the apparent win.

**The boundary:** a compatible API is not a compatible file format. Use a logical
export/import or an application migration. An existing MDBX, SQLite, or RocksDB
file cannot simply be opened as a psitri database.

Native features also do not automatically pass through a shim. The MDBX
implementation preserves one-writer-per-environment behavior. Its current
backend-specific value budget is approximately 512 KiB for non-GC/RC paths,
separate from the maintained native engines’ 4 GiB − 1 byte logical value limit.
The shim selects its own tiered/backend configurations; its behavior
must be validated independently of the shared engine conformance suite.

## Compatibility and performance

**Keep the API. Rethink the engine.** A compatibility layer can reduce migration
work, but it is not performance-neutral. Another API imposes a transaction,
cursor, and ownership workflow that may differ from psitri's native design.
Translating that workflow adds overhead and can restrict optimization paths
available through the native interface.

For example, the MDBX layer preserves MDBX's one-writer-per-environment semantics.
Foreign byte-oriented APIs also do not automatically expose the Large Engine's owned subtrees,
manifest-root publication, or native tree operations. Applications can miss those
opportunities even when individual key-value calls are fast.

Use the native API when you can shape the application around psitri's ownership,
snapshot, and transaction model. Use a compatible API when reducing application
changes matters more. The tradeoff depends on the workload: measure end-to-end
performance through the interface you will deploy. Native-engine benchmark
numbers do not include compatibility-layer overhead.

## A useful evaluation sequence

1. Pin the application, compatibility layer, psitri, and comparison database revisions.
2. Exercise the API subset you actually use. Include missing keys, empty values,
   duplicate values, bounds, abort, reopen, and error paths where applicable.
3. Match data, batching, readers, durability requirements, and memory limits.
4. Run the existing application benchmark against both implementations. Record
   throughput, latency, disk high-water, memory, and correctness checks.
5. Repeat on an otherwise quiet machine. Follow with recovery tests and sustained
   workloads larger than the intended memory budget.

The [benchmark recorder](benchmarking.html) retains both raw output and structured
metadata. The [published evidence](benchmarks.html) measures native bank workloads
against real MDBX. Measure the compatibility layer in your application
independently.

## Choose an integration path

For a new application, the native C++ API exposes owned snapshots and backend
selection directly. The Large Engine adds owned subtrees and savepoints. For an existing
application, an API compatibility layer can reduce code changes while you test
whether psitri's storage model improves its workload.

Start with the [native quickstart](quickstart.html) or inspect the integration
source and run the supported API subset before planning a migration.
[Repository access](https://github.com/gofractally/psitri) may be restricted.

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
| Native C++23 | Maintained Small and Big profiles in this repository | Shared byte API, lifetime rules, sync policy, engine-specific extensions |
| MDBX C / C++ | Implementation in `libraries/psitrimdbx` | The calls, flags, comparators, duplicate handling, cursor and transaction semantics your application uses |
| SQLite | Earlier storage-engine integration; not migrated here | Replaces SQLite’s B-tree layer while retaining its SQL parser and virtual machine; port and revalidate against the maintained engines |
| RocksDB | Earlier API wrapper; not migrated here | Supported API subset, snapshots, batches, transactions, options, and migration |
| LevelDB | Earlier API wrapper; not migrated here | Supported byte API and batch/snapshot behavior |
| DuckDB | Earlier storage extension; not migrated here | Catalog, scan, insert, and transactional integration |

The earlier integrations are recorded in the project’s repository reorganization
plan. They are not advertised as available packages in this checkout. No complete
SQLite, RocksDB, or DuckDB compatibility certification is claimed.

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
The shim currently selects historical/tiered engine combinations; its behavior
must be validated independently of the native Small/Big conformance suite.

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
metadata. The [published evidence](benchmarks.html) currently includes native bank
workloads against real MDBX, not a current end-to-end API-shim speedup claim.

## Why we keep exploring

Ethereum synchronization through the MDBX abstraction has been a motivating
integration workload. A good microbenchmark result does not establish faster
end-to-end synchronization. Keeping the native API stable lets applications
adopt a better engine when correctness and workload evidence support it.

For new code, start with the [native quickstart](quickstart.html). For existing
applications, inspect the integration source and run the supported subset before
planning a migration. [Repository access](https://github.com/gofractally/psitri)
may be restricted.

# psitri

Persistent, transactional key-value engines, a shared C++ database interface,
and common behavioral tests and benchmarks.

The product goal is a reliable database that scales beyond RAM. Psibase must
be able to adopt faster implementations through a stable API while preserving
revision, snapshot, ownership, rollback, and durability semantics. Ethereum
sync through the MDBX compatibility layer is a useful integration workload;
winning that benchmark is not a prerequisite for the Psibase API.

## Public C++ API

Link `psitri::api` and include `<psitri/api.hpp>`. The version 1 contract is
`psitri::engine<E>`; use `psitri::big_engine` for the beyond-RAM integration
candidate or `psitri::small_engine` for resident data. `default_engine` aliases
Big. `database_for<E>` names the database without changing the legacy
`psitri::database` type.

Start with the [developer guide](docs/api/README.md),
[compiling quickstart](docs/api/quickstart.md), and
[engine/Psibase recommendation](docs/api/engines.md). Both maintained profiles
pass the common API suite. Crash recovery and sustained beyond-RAM release
qualification remain separate gates. Both profiles support values up to
4 GiB − 1 byte, with mixed allocation/free tests through 512 MiB. Big uses
segment-aware chunks; Small uses fixed-address chunks. The write-oriented
tiered alias is explicitly experimental.

## Repository layout

| Directory | Role |
|---|---|
| `libraries/adb` | C++ database concepts and shared behavioral checks; the foundation of the public API. |
| `libraries/psitri_api` | Public profile names, API contract, and consumer CMake targets. |
| `libraries/sal` | Movable, refcounted storage and compaction. |
| `libraries/sal_art_ref` | Heart: refcounted ART with packed pages on SAL; the primary engine to qualify for Psibase's beyond-RAM storage. |
| `libraries/start_ref` | Part: fixed-address ART with refcounts; the RAM-resident performance option and a candidate hot tier. |
| `libraries/delta_art` | Reusable hot-delta/base composition for sustained writes. The chosen component engines determine whether a configuration is experimental. |
| `libraries/psitri` | Existing psitri implementation and DWAL API, retained for compatibility. |
| `libraries/art` | ART write-set machinery used by DWAL. |
| `libraries/psitrimdbx`, `shims/` | Foreign-API integrations. |
| `experimental/` | Research engines and allocators retained for benchmarks and reference. |
| `tests/`, `benchmarks/`, `logs/` | Shared checks, comparison workloads, and historical evidence. |

Placement in `libraries/` identifies the code being maintained toward the
supported API; it does not certify every configuration as release-qualified.
The experimental inventory and unchanged build behavior are documented in
[experimental/README.md](experimental/README.md).

## Small-memory performance and capacity

The historical fixed-address, **32-bit-handle** configuration beat libart on
measured in-RAM point lookups. The recorded M5 results include u64 hit/miss
latencies of 0.87x/0.82x libart and dictionary hit/miss latencies of
0.92x/0.73x. These are workload-specific read results; the same record reports
slower inserts. See the [original measurements](libraries/start_ref/STATUS.md).

That layout uses a 28-bit index plus a four-bit tag, addressing **32 GiB** of
128-byte allocation lines. This is an arena limit, not 32 GiB of user payload
or a requirement to run a 32-bit process. Current `start_ref` uses five-byte
handles, retains a 32 GiB default ordinary arena, and its allocator permits
just under 512 GiB in that arena. Large values have a separate arena. Neither
the historical result nor the representable limit is a new benchmark or scale
qualification of the current wider-handle implementation. RAM sizing must
include metadata, fragmentation, retained snapshots, and pending reclamation.

## Compatibility during reorganization

The layout move preserves engine targets, public include spellings, backend
selections, and database formats. A subsequent naming cleanup gives the
experimental allocators descriptive names, `sal_hybrid` and `sal_gc`, while
`sal` remains the canonical maintained library. See the
[allocator rename map](experimental/README.md#allocator-names) for research
namespace, header, and target changes. Experimental engines remain in the
existing build and comparison suite. In particular, the MDBX
shim's existing `delta_art` default still uses experimental `start_ver` over
`sal_art_ref`; the directory move does not switch existing users' engines.

The API normalization adds public Heart headers and the checked factory,
transaction, cursor, and owned-snapshot contract over
[adb](libraries/adb/include/adb/concepts.hpp). It normalizes callback miss
behavior, forwards Big's allocator configuration, and rejects values that
would overflow its persisted length field. Big uses a unified external-value
format with an embedded continuation directory and a 4 GiB − 1 byte value limit;
its development-era disk layouts are not retained.
The older DWAL concept remains an internal composition protocol; experimental
engines do not acquire public API status merely by satisfying it.

Migrated from psitri-merge at tag `reorg-baseline`. The earlier rename map and
carried TODO ledger remain in [docs/reorg_plan.md](docs/reorg_plan.md). Historical
documents and logs retain their original paths; for the six research modules,
read `libraries/<name>` as `experimental/<name>` in the current checkout,
then apply the allocator rename map above.

# Engine authors, validation, and compatibility

## The contract

The standard-only [adb concepts](../../libraries/adb/include/adb/concepts.hpp)
separate cursor, transaction, session, snapshot, database, and optional feature
requirements. `adb::openable_engine<E, Mode>` combines them with construction
and exact role relationships. `<psitri/concepts.hpp>` binds the mode to SAL and
provides `psitri::engine<E>` as the public contract.

An engine must declare these roles:

```cpp
struct my_engine {
    using database = my_database;
    using config = my_config;
    using write_session = my_write_session;
    using read_session = my_read_session;
    using transaction = my_transaction;
    using cursor = my_cursor;
};
static_assert(psitri::engine<my_engine>);
```

Factories must return matching shared session/database handles; the session
transaction and cursor types must match the declared roles. Snapshots must be
copyable and create the same bidirectional cursor type. Byte reads return
`std::optional<std::string>`, deletion returns `int`, callback reads return a
boolean, and transactions provide explicit abort. The engine config must
support the common `sync_mode` setting. The complete operation semantics are
in [the reference](reference.md), including the non-STL floor bound convention.

Preserve the existing adb concepts when adding optional capabilities. Range reads use
the optional `adb::range_reader`, `adb::range_cursor`, and `adb::range_engine`
concepts, exposed for public profiles as `psitri::range_engine`. Existing
experimental engines do not need to implement those methods to keep satisfying
their original concepts. Explicit abort is required by `openable_engine`, not
the older `kv_transaction` concept. Capability selection is compile-time only;
no runtime adapter, interface object, or renamed read methods are required.

A concept cannot prove isolation, ownership, callback lifetimes, durable
publication, or correct rollback. Do not advertise a new profile solely because
its `static_assert` compiles. Conversely, do not weaken the concept to allow a
missing snapshot method or a void deletion return. Keep such a profile under
`experimental` until its behavior matches.

## Run the checks

```sh
cmake -S . -B build/api-check -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF -DPSITRI_BUILD_EXAMPLES=ON
cmake --build build/api-check --target \
  adb-concept-check api-conformance-tests start_ref-tests sal_art_ref-tests \
  psitri-example-big psitri-example-small psitri-example-revisions
ctest --test-dir build/api-check -R '^(adb-concept-check|api-conformance-tests)$' \
  --output-on-failure
build/api-check/bin/start_ref-tests
build/api-check/bin/sal_art_ref-tests
```

Use the compiler options from [the quickstart](quickstart.md) on macOS. Add a
profile to the shared template tests in
[tests/api/conformance.cpp](../../tests/api/conformance.cpp); run its native
suite as well. Keep a separate consumer build of `examples/api`: it catches
private include-path leaks, missing transitive targets, and embedding mistakes
that an in-repository build can conceal.

The common suite exercises create/open rejection, read-only support/rejection,
read-your-writes, empty and
binary keys/values, 65,535-byte records, explicit/implicit rollback,
remove status, snapshot copies across publication, readers using their own
thread sessions, both scan directions, floor/inclusive bounds, callback hit/miss
behavior, and clean reopen. Additional checks cover Big config forwarding
and rejection of oversized keys without changing existing values. Negative
compile assertions reject role mismatches, missing factories, absent abort, and
configurations without the common sync setting. The multi-root extension tests
verify rollback and root exchange without assuming atomic cross-root publication.
The experimental tiered alias is intentionally asserted *not* to pass.

Range tests additionally cover offset/length and EOF behavior, overload
selection, borrowed addresses, callback exceptions, and snapshot stability in
both profiles. The adb self-check includes a byte-only engine without
range methods and verifies that it still meets the original engine concept.

Big's large-value tests exercise full and range reads through 512 MiB, mixing
small objects with allocation, replacement, deletion, snapshots, and rollback.
They verify exact segment packing/free-space totals, the embedded directory and
first payload, reference-count rebuild, explicit recovery, and actual relocation
during concurrent snapshot reads. Placement arithmetic covers `UINT32_MAX`; an
inaccessible 4 GiB virtual mapping verifies rejection before input bytes are read.
A 128 MiB pinned-cache setting does not bound process RSS or the OS page cache.
Small's separate large-value suite exercises mixed allocation/free through
512 MiB, exact physical-allocation boundaries, direct borrowing and crossing
reads, holder lifetimes, single-leaf snapshots, saturated continuation refcounts,
concurrent readers, multi-root reads, rollback, durable reopen, and allocation
failure cleanup. It checks the same 4 GiB rejection before reading input bytes.

The revision example verifies persisted owned subtrees and restore, rather than
claiming that all optional subtree/undo semantics follow from the base concept.
The broader `kv-compat-test` includes experimental engines. Evaluate those
results separately from conformance of the maintained public profiles.

## Source compatibility policy

API version 1 concerns source-level application code rebuilt against a tested
revision. Preserve existing spellings and semantics when adding capabilities.
New required expressions or changed return meanings require an explicit API
version decision and migration notes. An optional extension must have a named
capability check and a test demonstrating its lifetime and failure semantics.
Do not make an engine's private layout, allocator pointer, test helper, or
benchmark switch a common requirement.

Use the named profile headers and `database_for<E>` in generic applications.
The separate `psitri::database` implementation and DWAL composition interfaces
do not establish conformance to `psitri::engine` version 1. The source compatibility target does not promise a stable
C++ ABI, serialized raw C++ objects, or storage-format interchangeability.

The API target links only maintained engine dependencies. Experimental headers
require their explicit target and namespace. Changing a default alias must not
silently reopen another engine's files; persistent applications pin an explicit
profile and carry out a validated data migration when needed.

## Validation recorded for this normalization

On 2026-09-17, macOS arm64 with Homebrew LLVM 22, Debug assertions enabled:

| Check | Result |
|---|---|
| `adb-concept-check` | Passed |
| `api-conformance-tests` | 21 cases passed, including both profiles through 512 MiB, embedded first payloads, 4 GiB rejection, and Small allocation-failure cleanup |
| `start_ref-tests` | 38 cases, 1,098,714 assertions passed |
| `sal_art_ref-tests` | 104 cases, 2,279,600 assertions passed |
| `sal-tests` | 49 cases, 101,634,787 assertions passed |
| `delta_art-tests` | 6 cases, 55,632 assertions passed |
| Standalone consumer examples | Small, Big, and revision reopen/restore passed |
| Existing consumer compilation | MDBX shim, WAL compile check, shared compatibility driver, bank and scan benchmarks built |

This records API and regression evidence for the current checkout, not a new
performance result, Linux validation, or a crash/beyond-RAM certification.

# Build and first database

Requires C++23, a supported Unix platform (Linux or macOS), and CMake 3.20+.
On macOS use Homebrew LLVM; the system Apple Clang may lack required C++23
support. Catch2 3 is needed for the test suites, not for a consumer application.

## Consume from another CMake project

Check out a tested psitri revision under `third_party/psitri`, then:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_database_app LANGUAGES C CXX)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(PSITRI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/psitri EXCLUDE_FROM_ALL)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE psitri::api)
```

`psitri::api` supplies include paths, C++23, and transitive dependencies. Do not
add any `test/` directory to your application's include path. To include only
one profile, link `psitri::big` with `<psitri/engines/big.hpp>`, or
`psitri::small` with `<psitri/engines/small.hpp>`. The std-only concepts target is
`adb::adb`; `<psitri/concepts.hpp>` additionally uses SAL's open and sync enums.
A packaged `find_package(psitri)` installation is not provided yet.

The repository fetches `psio` and `ucc`, defaulting to local sibling checkouts
under `$HOME/gofractally` when available, otherwise their `main` branches.
For reproducible builds, pin all three checkouts and pass
`FETCHCONTENT_SOURCE_DIR_PSIO` and `FETCHCONTENT_SOURCE_DIR_UCC` explicitly.
The consumer test below exercises the same `add_subdirectory` integration.

## Create and write

This is the core of the [complete, runnable example](../../examples/api/basic.cpp):

```cpp
#include <psitri/api.hpp>
using Engine = psitri::big_engine;
Engine::config cfg;
cfg.sync_mode = psitri::sync_mode::fsync;
auto db = Engine::database::open("accounts", psitri::open_mode::create_only, cfg);
auto writer = db->start_write_session();
{
    auto tx = writer->start_transaction(0);
    tx.upsert("account/alice", "100");
    tx.upsert("account/bob", "50");
    tx.commit();
}
```

Root `0` is the ordinary application root. `create_only` prevents this example
from reopening an existing database; use `open_existing` when that is your
intent. Neither mode wipes an existing database. The default sync policy is
`none`; choose the durability policy explicitly for persisted application data.
A successful synchronous flush is only one part of crash safety; see
[durability and recovery qualification](operations.md#durability).

A transaction reads its own writes. `upsert` inserts or replaces, and copies
the supplied bytes. `remove` returns `0` when it removed a key and `-1` when
absent. `commit()` publishes changes. Explicit `abort()` or destruction without
commit discards them. End the transaction's scope before starting another;
a committed transaction is not a reusable object.

## Read, scan, and keep a snapshot

```cpp
auto before = writer->get_root(0); // frozen, owning reference
{
    auto tx = writer->start_transaction(0);
    tx.upsert("account/alice", "125");
    tx.commit();
}
auto old = before.snapshot_cursor();
auto original = old.get<std::string>("account/alice"); // optional{"100"}

auto reader = db->start_read_session();
auto cur = reader->create_cursor(0); // snapshot at cursor creation
auto current = cur.get<std::string>("account/alice"); // optional{"125"}
for (bool found = cur.lower_bound("account/");
     found && cur.key().starts_with("account/"); found = cur.next())
{
    auto value = cur.value<std::string>(); // owned copy
    // Use cur.key() and *value here.
}
```

A missing key returns `std::nullopt`. An existing empty value is an engaged
optional containing an empty string. Keys and values may include zero bytes;
construct a `std::string_view(data, length)` rather than a null-terminated view
when passing binary input. Portable applications use keys up to 1,024 bytes
and keep values within their chosen profile's capacity. Both profiles support
[large values](chunked-values.md) up to 4 GiB − 1 byte, tested through 512 MiB.

To borrow contiguous values or ranges, use a callback. Both profiles assemble
ranges that cross chunks and still call the callback once:

```cpp
std::string retained;
bool found = cur.get("account/alice", [&](std::string_view bytes) {
    retained.assign(bytes); // the view itself must not escape this call
});
```

Keep the database and sessions alive until their transactions, snapshots, and
cursors are destroyed. Use a separate read session for every simultaneously
live read cursor. Create and use a reader's session on that reader thread.
[The lifetime guide](operations.md) explains these restrictions.

## Build and run the examples

From a checkout, with a suitable compiler on PATH:

```sh
cmake -S examples/api -B build/api-consumer -G Ninja \
  -DPSITRI_SOURCE="$PWD" -DCMAKE_BUILD_TYPE=Debug
cmake --build build/api-consumer --target \
  psitri-example-big psitri-example-small psitri-example-revisions
sample_dir=$(mktemp -d)
build/api-consumer/psitri-example-big "$sample_dir/big"
build/api-consumer/psitri-example-small "$sample_dir/small"
build/api-consumer/psitri-example-revisions "$sample_dir/revisions"
```

On macOS add
`-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang` and
`-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++` (or your Homebrew prefix).
Each command expects a new database directory. The basic examples print Bob's
balance and the updated account scan. The revision example persists two tree
versions, closes the database, reopens it, and restores the older version.
The examples leave their databases available for inspection.

# Experimental engines and allocators

These implementations remain available for benchmarks, differential tests,
and design reference. Their namespaces do not carry the supported API or
on-disk compatibility commitment being developed for the main libraries.

| Directory / namespace | Research role |
|---|---|
| `gcart` | Version-based COW and reclamation with fixed routing and movable leaves. |
| `rcart` | Refcount counterpart to gcart for layout and liveness comparisons. |
| `sal2` | Experimental allocator family, including the copied SAL leaf allocator. |
| `sal2gc` | Version-liveness allocator work. |
| `start_ver` | Vpart: fixed-address ART with version-based reclamation; used in historical tiered scale runs. |
| `walart` | Earlier refcounted delta/base pipeline, retained for comparison. |

## Build and API preservation

The root CMake file adds these directories in the same dependency order as
before. Targets, include paths such as `<gcart/kv.hpp>`, namespaces, compile
options, tests, and benchmark names remain unchanged. Target include directories
follow the new locations through `CMAKE_CURRENT_SOURCE_DIR`; consumers linking
the existing targets do not need source edits. Consumers that manually hardcode
`libraries/<name>/include` must update that filesystem path.

There is deliberately no new build gate in this relocation: the existing
cross-engine tests and benchmark sources use research engines directly. Their
existing `BUILD_TESTS`, `BUILD_BENCHMARKS`, and per-library options still apply.
Backend selection also remains unchanged. The MDBX shim's default tiered
configuration uses `start_ver` here; a refcount-only tiered configuration uses
`start_ref` from the main libraries instead.

For example, from the repository root with the project's compiler configured:

```sh
cmake -S . -B build/research -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build build/research --target bank-compare-bench rcart-kv-tests
```

The relocation preserves pre-existing source changes and known experimental
test failures. It does not repair algorithms, alter storage formats, or claim
release qualification. Historical design notes and logs retain their original
`libraries/<name>` paths as records of the builds they describe.

The opt-in `<psitri/experimental/tiered.hpp>` header adds
`psitri::experimental::tiered_engine<Hot, Base>` and `write_engine` (Small over
Big). Link `psitri::experimental`. These configurations do not yet satisfy
the public owned-snapshot contract. See the [API guide](../docs/api/README.md)
and [profile status](../docs/api/engines.md).

# Experimental engines and allocators

These implementations remain available for benchmarks, differential tests,
and design reference. Their namespaces do not carry the supported API or
on-disk compatibility commitment being developed for the main libraries.

| Directory / namespace | Research role |
|---|---|
| `gcart` | Version-based COW and reclamation with fixed routing and movable leaves. |
| `rcart` | Refcount counterpart to gcart for layout and liveness comparisons. |
| `sal_hybrid` | Experimental allocator family, including the copied SAL leaf allocator. |
| `sal_gc` | Version-liveness allocator work. |
| `start_ver` | Vpart: fixed-address ART with version-based reclamation; used in historical tiered scale runs. |
| `walart` | Earlier refcounted delta/base pipeline, retained for comparison. |

## Allocator names

`libraries/sal`, namespace `sal`, is the maintained allocator used by Big.
The research allocators coexist with it and have descriptive names:

| Former name | Directory / namespace | CMake targets / headers |
|---|---|---|
| `sal2` | `experimental/sal_hybrid`, `sal_hybrid::*` | `sal_hybrid`, `sal_hybrid_alloc`; `<sal_hybrid/...>` |
| `sal2gc` | `experimental/sal_gc`, `sal_gc::*` | `sal_gc`; `<sal_gc/...>` |

The hybrid fork combines direct-address fixed routing with a modified SAL
leaf allocator (32-byte grains and 32-bit allocation versions). The GC fork
contains version-liveness experiments. These are separate implementations;
using the canonical `sal` namespace for them would conflict in the comparison
tools that compile maintained and experimental engines together.

The rename covers headers, namespaces, CMake targets and test names, and
configuration macros (`SAL_HYBRID_*` and `SAL_GC_*`). Update external research
consumers to those spellings. Public `psitri` engine names and maintained `sal`
are unchanged. No aliases retain the numbered allocator names.

Existing experimental arena filenames (`data.sal2`, `free.sal2`, `refs.sal2`
and their `.sal2gc` counterparts) are preserved; renaming source code does not
change the storage layout or redirect existing files. Historical notes and
benchmark logs retain their original names and measurements.

## Build and API preservation

The root CMake file adds these directories in the same dependency order.
Engine targets, include paths such as `<gcart/kv.hpp>`, and backend selections
remain unchanged. Allocator include directories follow their new locations
through `CMAKE_CURRENT_SOURCE_DIR`. Research consumers with hardcoded source
paths must update those paths as well as the renamed allocator targets.

The existing `BUILD_TESTS`, `BUILD_BENCHMARKS`, and per-library options still
apply. Cross-engine tests and benchmark sources continue to use research
engines directly. The MDBX shim's default tiered configuration uses `start_ver`
here; a refcount-only tiered configuration uses `start_ref` from the main
libraries instead.

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

## Validation of the allocator rename

A fresh Debug build with LLVM 22 on macOS arm64 compiles both allocator
families, gcart/rcart test targets, the shared compatibility driver, the
comparison benchmark, and the MDBX library. The maintained API's 21 cases and
the concept self-check pass. A source comparison confirms that all 77 affected
C++ files differ only by allocator identifiers, diagnostic strings, and comments.

The fixed, variable, arena, and control-block tests for `sal_hybrid`, all three
`sal_gc` allocator tests, and `rcart-kv-tests` pass before and after the rename.
The variable-allocation recycle test fails identically before and after:
drifting sizes leave 307 segments allocated against a 32-segment bound.

The original `sal2-alloc-smoke` and `gcart-kv-tests` executables hit the existing
`size % 64 == 0` assertion in the copied allocator before the rename. Their
renamed sources compile; these two failing runtime tests were not rerun.
These research failures remain separate from the maintained Small/Big release
gates. The naming cleanup does not fix allocator algorithms or qualify GC.

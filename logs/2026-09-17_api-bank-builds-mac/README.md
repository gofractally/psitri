# API-pair benchmark build preparation

Mac builds of the shared `api-bank-v1` driver: packaged MDBX shim with the
`sal_art_ref` backend, and separate RocksDB/SQLite integrations pinned at
`e4c5ebbddd5b179f0f15abec9e890f386a94d104`.

Compiler: Homebrew LLVM 22.1.2; Release, C++23, `-O3 -DNDEBUG -march=native`.
psio: `ff114b6` (clean detached worktree); ucc: `2c124eb` (clean detached worktree);
Abseil: `4447c7562e3bc702ade25105912dce503f0c4010`.

The initial integration configuration referenced an unrelated libart path; the
standalone build now includes only the required libraries. The initial native
RocksDB compilation needed explicit default initialization of WriteBatch. The
SQLite integration required its database template definitions in the C++ TU;
the harness supplies the existing header with a compiler include. Failures and
subsequent successful build logs are retained here.

No performance result follows from a successful build. The first smoke attempt
was prevented from starting by the idle gate because other workloads were busy.
Its raw record is in `../2026-09-17T190447-fair-bank-smoke-0cc6f146/`.

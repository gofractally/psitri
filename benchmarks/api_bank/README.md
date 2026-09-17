# Same-API bank comparisons

`api-bank-v1` compiles the shared bank driver against an original library and its
psitri implementation. Compare pairs within a machine, revision, workload and
durability cohort. ARM versus x86 also changes the CPU, OS, page size and storage;
the architecture selector is a machine selector, not an isolated ISA experiment.

The packaged MDBX shim and the separate RocksDB/SQLite integration source use
separate build directories to avoid conflicting library symbols. Pin every
source revision. The latter integrations are not the maintained engine profiles.

```sh
cmake -S benchmarks/api_bank -B build/api-mdbx -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DFETCHCONTENT_SOURCE_DIR_PSIO=/path/to/clean/psio \
  -DFETCHCONTENT_SOURCE_DIR_UCC=/path/to/clean/ucc \
  -DFAIR_MDBX_INCLUDE_DIR=/path/to/libmdbx \
  -DFAIR_MDBX_LIBRARY=/path/to/libmdbx.a \
  -DPSITRIMDBX_BACKEND=sal_art_ref
cmake --build build/api-mdbx --target fair-bank-mdbx fair-bank-psitrimdbx

cmake -S benchmarks/api_bank -B build/api-integrations -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DFETCHCONTENT_SOURCE_DIR_PSIO=/path/to/clean/psio \
  -DAPI_BANK_INTEGRATIONS=/path/to/pinned/integrations \
  -DAPI_BANK_ABSEIL=/path/to/pinned/abseil \
  -DFAIR_ROCKSDB_INCLUDE_DIR=/path/to/rocksdb/include \
  -DFAIR_ROCKSDB_LIBRARY=/path/to/librocksdb.a
cmake --build build/api-integrations --target fair-bank-rocksdb \
  fair-bank-psitrirocks fair-bank-sqlite fair-bank-psitri_sqlite

python3 scripts/fair_bank.py --suite api --build build/api-mdbx \
  --machine my-runner --profile smoke --engines mdbx psitrimdbx --durability none
python3 scripts/fair_bank.py --suite api --build build/api-integrations \
  --machine my-runner --profile smoke --engines rocksdb psitrirocks --durability none
python3 scripts/fair_bank.py --suite api --build build/api-integrations \
  --machine my-runner --profile smoke --engines sqlite psitri_sqlite --durability none
```

After smoke validation, use `--profile performance --repeats 5` for each pair.
Or collect all three pairs in sequence, measuring only the pairs that pass smoke:

```sh
python3 scripts/api_bank_matrix.py --machine my-runner \
  --build-mdbx build/api-mdbx --build-integrations build/api-integrations \
  --output logs --repeats 5 --idle-timeout 1800
```

The matrix retains failures and all repetitions. It does not publish results.
Finish all builds first. The runner takes a machine-wide advisory lock and
requires ten consecutive seconds below 3% aggregate CPU activity before each
process. Linux I/O wait counts as busy. Idle samples are archived. A timeout fails
without launching a benchmark; `--idle-timeout` changes only the waiting limit.
The lock coordinates cooperating runners; do not launch unrelated benchmarks or
builds during collection. Use a dedicated idle host, with the same workload and
source/dependency revisions on both architectures. Run one benchmark at a time.

All pairs keep rollback, held-snapshot, read-your-writes, exact final account
state, balance, cardinality and reopen checks. Unsupported calls or semantics are
failures, not a reason to remove a check for the shim. Only complete passing pairs
belong in a performance chart. The API suite is separate from `fair-bank-v2`:
RocksDB uses the supported `WriteBatch` API plus the same application-side pending
map on both implementations, instead of `WriteBatchWithIndex`. Include that
overhead on both sides. SQLite executes identical SQL and PRAGMA calls and
verifies their results; a shim that cannot satisfy them fails validation.

Record machine, OS, page size, source and dependency revisions, compiler flags,
effective durability, every repetition, and failed attempts. Retain build logs
alongside run evidence. Do not mix the April integration workload with this
workload or treat smoke results as performance evidence.

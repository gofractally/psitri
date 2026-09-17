# API-pair benchmark build preparation on x86

Cloud runner: AMD EPYC-Turin, Linux, Clang 20.1.8, C++23, Release, `-march=native`.
The clean source bundles retain the original git history and revisions:

- Driver source: `5e646e24dd8f14249f52836533dd2b52b72c24bc`, plus the radix-header packing fix.
- Integrations: `e4c5ebbddd5b179f0f15abec9e890f386a94d104`.
- psio: `ff114b6a8c9265e6b5da5ffb85df1617b2b43514`.
- ucc: `2c124eba978e7bcf20e252eec7fd5b2ea3635295`.
- Abseil: `4447c7562e3bc702ade25105912dce503f0c4010`.
- RocksDB: source checkout `14d3046a53dda588ef5887ee2c80e93104caf079`.
- MDBX: source checkout `1d115d3df69cc07afbe0157763fce741f44909d2`.

The prebuilt RocksDB/MDBX archives contain LLVM bitcode. Linking requires
`-DCMAKE_EXE_LINKER_FLAGS=-flto=thin` so Clang loads its LLVMgold linker plugin.
Initial link failures and subsequent successful builds are retained.

The packaged MDBX shim's `sal_art_ref` backend exposed x86 padding of the nested
`range_ctrs` header, failing the existing node-size assertions. Switching C++
standard libraries did not fix the layout. Packing `range_ctrs` explicitly keeps
its alignment at one and the dense branch array at byte 128. Build 3 uses the
default Linux C++ standard library and passes all compile-time layout assertions.

All six API-pair executables now build. Builds are not performance measurements.
The benchmark runner waits for an idle host before starting any workload.

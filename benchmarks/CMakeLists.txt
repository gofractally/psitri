# benchmarks/ — written against adb, instantiated per engine. The composed
# pipeline benches (delta_art = hot delta over cold base) drive the
# start_ver + sal_art_ref instantiation.

option(PSITRI_FAIR_BANK "Build the common bank workload against all comparison libraries" OFF)
if(PSITRI_FAIR_BANK)
    add_subdirectory(fair_bank)
endif()

# delta_art scale bench: start_ver (hot delta) -> sal_art_ref dwal (cold base).
# The cold engine is sal_art_ref::radix::engine (sal_art_ref/engine.hpp,
# under sal_art_ref/test); it uses psitri's dwal::kv_engine concept.
add_executable(delta_art-scale-bench delta_art_scale_bench.cpp)
target_link_libraries(delta_art-scale-bench PRIVATE delta_art start_ver sal_art_ref psitri)
target_include_directories(delta_art-scale-bench PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)  # test-side adapters (heart_kv.hpp)
target_compile_definitions(delta_art-scale-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(delta_art-scale-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(delta_art-scale-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# walart scale bench: WAL + art_map write-set (sal_art_ref base + start_ref delta).
add_executable(walart-scale-bench walart_scale_bench.cpp)
target_link_libraries(walart-scale-bench PRIVATE walart sal_art_ref start_ref psitri)
target_compile_definitions(walart-scale-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(walart-scale-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(walart-scale-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# delta_art bank bench: transfer workload + balance conservation over
# start_ver (hot delta) -> sal_art_ref dwal (cold base).
add_executable(delta_art-bank-bench delta_art_bank_bench.cpp)
target_link_libraries(delta_art-bank-bench PRIVATE delta_art start_ver sal_art_ref psitri)
target_include_directories(delta_art-bank-bench PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(delta_art-bank-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(delta_art-bank-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(delta_art-bank-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# bank comparison: the same workload on delta_art<start_ver,sal_art_ref> vs
# standalone start_ver vs standalone sal_art_ref.
add_executable(bank-compare-bench bank_compare_bench.cpp)
target_link_libraries(bank-compare-bench PRIVATE delta_art start_ver start_ref sal_art_ref psitri gcart rcart)
target_include_directories(bank-compare-bench PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(bank-compare-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(bank-compare-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(bank-compare-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# focused gcart load profile (one engine, for `sample`/perf attribution)
add_executable(st-load-bench st_load_bench.cpp)
target_link_libraries(st-load-bench PRIVATE gcart)
target_compile_options(st-load-bench PRIVATE -O3 -DNDEBUG -fno-omit-frame-pointer)
set_target_properties(st-load-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# focused inline-leaf READ microbench (nested_setlist: reads that hit inline leaves)
add_executable(st-inline-read-bench st_inline_read_bench.cpp)
target_link_libraries(st-inline-read-bench PRIVATE gcart)
target_compile_options(st-inline-read-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(st-inline-read-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# gcart oplog replay: read+write VALIDATION of gcart::kv on a captured Silkworm
# op-stream (reads cross-checked against Silkworm's recorded results).
add_executable(gcart-oplog-replay gcart_oplog_replay.cpp)
target_link_libraries(gcart-oplog-replay PRIVATE gcart)
target_compile_definitions(gcart-oplog-replay PRIVATE XXH_INLINE_ALL)
target_compile_options(gcart-oplog-replay PRIVATE -O3 -DNDEBUG)
set_target_properties(gcart-oplog-replay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# oplog comparison: replay one captured Silkworm op-stream through each adb engine.
add_executable(oplog-compare-bench oplog_compare_bench.cpp)
target_link_libraries(oplog-compare-bench PRIVATE gcart start_ver start_ref sal_art_ref psitri)
target_include_directories(oplog-compare-bench PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(oplog-compare-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(oplog-compare-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(oplog-compare-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# delta_art threshold sweep: same sustained transfer workload at a range of
# flush high-water marks (Dan: smaller hot delta = faster).
add_executable(delta_art-threshold-sweep delta_art_threshold_sweep.cpp)
target_link_libraries(delta_art-threshold-sweep PRIVATE delta_art start_ver sal_art_ref psitri)
target_include_directories(delta_art-threshold-sweep PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(delta_art-threshold-sweep PRIVATE XXH_INLINE_ALL)
target_compile_options(delta_art-threshold-sweep PRIVATE -O3 -DNDEBUG)
set_target_properties(delta_art-threshold-sweep PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# memory + recovery comparison (ONE engine per process; peak RSS is process-wide)
add_executable(mem-recovery-bench mem_recovery_bench.cpp)
target_link_libraries(mem-recovery-bench PRIVATE delta_art start_ver sal_art_ref psitri)
target_include_directories(mem-recovery-bench PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(mem-recovery-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(mem-recovery-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(mem-recovery-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# tree-shape-audit: read-only leak + tree-shape audit of a sal_art_ref cold base.
add_executable(tree-shape-audit tree_shape_audit.cpp)
target_link_libraries(tree-shape-audit PRIVATE sal_art_ref psitri)
target_include_directories(tree-shape-audit PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(tree-shape-audit PRIVATE XXH_INLINE_ALL)
target_compile_options(tree-shape-audit PRIVATE -O3 -DNDEBUG)
set_target_properties(tree-shape-audit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# churn-leak: reproduce cold-base bloat via controlled overwrite/remove churn.
add_executable(churn-leak churn_leak.cpp)
target_link_libraries(churn-leak PRIVATE delta_art start_ver start_ref sal_art_ref psitri)
target_include_directories(churn-leak PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(churn-leak PRIVATE XXH_INLINE_ALL)
target_compile_options(churn-leak PRIVATE -O3 -DNDEBUG)
set_target_properties(churn-leak PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# oplog-replay: replay a captured Silkworm op-stream into delta_art, track footprint.
add_executable(oplog-replay oplog_replay.cpp)
target_link_libraries(oplog-replay PRIVATE delta_art start_ver start_ref sal_art_ref psitri)
target_include_directories(oplog-replay PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(oplog-replay PRIVATE XXH_INLINE_ALL)
target_compile_options(oplog-replay PRIVATE -O3 -DNDEBUG)
set_target_properties(oplog-replay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# delta_art reopen-race repro (TSan): reopen a populated tiered DB + write.
add_executable(delta_art-reopen-repro delta_art_reopen_repro.cpp)
target_link_libraries(delta_art-reopen-repro PRIVATE delta_art start_ver sal_art_ref psitri)
target_include_directories(delta_art-reopen-repro PRIVATE ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(delta_art-reopen-repro PRIVATE XXH_INLINE_ALL)

# sal_art_ref cold-base corruption repro: sorted bulk drains of vt_extern/large values
add_executable(delta_art-largeval-repro delta_art_largeval_repro.cpp)
target_link_libraries(delta_art-largeval-repro PRIVATE delta_art start_ver sal_art_ref psitri)
target_include_directories(delta_art-largeval-repro PRIVATE ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(delta_art-largeval-repro PRIVATE XXH_INLINE_ALL)
target_compile_options(delta_art-largeval-repro PRIVATE -O2 -g)

# scan / lower_bound comparison across the adb engines (ordered cursor).
add_executable(scan-lower-bound-bench scan_lower_bound_bench.cpp)
target_link_libraries(scan-lower-bound-bench PRIVATE gcart start_ver start_ref sal_art_ref psitri)
target_include_directories(scan-lower-bound-bench PRIVATE ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)
target_compile_definitions(scan-lower-bound-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(scan-lower-bound-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(scan-lower-bound-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Real libmdbx bank baseline (head-to-head vs the psitri engines' bank_compare).
# Optional: only built when a prebuilt libmdbx is found (default: arbtrie's vendored
# copy). Override with -DLIBMDBX_DIR=/path/to/libmdbx (expects <dir>/mdbx.h and a
# libmdbx.{dylib,so,a} under <dir>/build). Run with the .dylib on the loader path.
set(LIBMDBX_DIR "$ENV{HOME}/arbtrie/external/libmdbx" CACHE PATH "libmdbx source dir for mdbx-bank-bench")
find_library(LIBMDBX_LIB NAMES mdbx libmdbx PATHS "${LIBMDBX_DIR}/build" NO_DEFAULT_PATH)
if(EXISTS "${LIBMDBX_DIR}/mdbx.h" AND LIBMDBX_LIB)
   add_executable(mdbx-bank-bench mdbx_bank_bench.cpp)
   target_include_directories(mdbx-bank-bench PRIVATE "${LIBMDBX_DIR}")
   target_link_libraries(mdbx-bank-bench PRIVATE "${LIBMDBX_LIB}")
   target_compile_options(mdbx-bank-bench PRIVATE -O3 -DNDEBUG)
   set_target_properties(mdbx-bank-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
   message(STATUS "mdbx-bank-bench: using ${LIBMDBX_LIB}")
else()
   message(STATUS "mdbx-bank-bench: skipped (no libmdbx at ${LIBMDBX_DIR}; set -DLIBMDBX_DIR=...)")
endif()

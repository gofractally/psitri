# PsiTriMDBX: MDBX-compatible drop-in API. The generic shim (dbi mapping,
# DUPSORT composite-key encoding, cursor ops, txn lifecycle) is engine-agnostic;
# the backend now rides the new fast model — the delta_art tiered engine
# (start_ver hot delta over a sal_art_ref dwal cold base).
add_library(psitrimdbx
    src/mdbx_impl.cpp
)

target_include_directories(psitrimdbx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(psitrimdbx PUBLIC psitri delta_art start_ver start_ref sal_art_ref gcart rcart)
target_compile_features(psitrimdbx PUBLIC cxx_std_20)

# Backend selection (default: delta_art tiered). The shim source picks the engine
# from a PSITRIMDBX_BACKEND_* macro; this maps a friendly string option to it so an
# embedder (Silkworm: -DPSITRIMDBX_BACKEND=gcart) can choose without editing source.
set(PSITRIMDBX_BACKEND "delta_art" CACHE STRING
    "psitrimdbx storage backend: delta_art|delta_gcart|start_ver|start_ref|gcart|rcart|sal_art_ref|psitri")
if(PSITRIMDBX_BACKEND STREQUAL "gcart")
    target_compile_definitions(psitrimdbx PUBLIC PSITRIMDBX_BACKEND_GCART=1)
elseif(PSITRIMDBX_BACKEND STREQUAL "rcart")
    target_compile_definitions(psitrimdbx PUBLIC PSITRIMDBX_BACKEND_RCART=1)
elseif(PSITRIMDBX_BACKEND STREQUAL "delta_gcart")
    target_compile_definitions(psitrimdbx PUBLIC PSITRIMDBX_BACKEND_DELTA_GCART=1)
elseif(PSITRIMDBX_BACKEND STREQUAL "start_ver")
    target_compile_definitions(psitrimdbx PUBLIC PSITRIMDBX_BACKEND_START_VER=1)
elseif(PSITRIMDBX_BACKEND STREQUAL "start_ref")
    target_compile_definitions(psitrimdbx PUBLIC PSITRIMDBX_BACKEND_START_REF=1)
elseif(PSITRIMDBX_BACKEND STREQUAL "sal_art_ref")
    target_compile_definitions(psitrimdbx PUBLIC PSITRIMDBX_BACKEND_SAL_ART_REF=1)
elseif(PSITRIMDBX_BACKEND STREQUAL "psitri")
    target_compile_definitions(psitrimdbx PUBLIC PSITRIMDBX_BACKEND_PSITRI=1)
elseif(NOT PSITRIMDBX_BACKEND STREQUAL "delta_art")
    message(FATAL_ERROR "Unknown PSITRIMDBX_BACKEND='${PSITRIMDBX_BACKEND}'")
endif()
message(STATUS "psitrimdbx backend: ${PSITRIMDBX_BACKEND}")

# Shim debug tracing (PSITRI_NF_LOG last-NOTFOUND dump, etc.). Off by default;
# enable to diagnose engine integration (e.g. a fatal MDBX_NOTFOUND at startup).
option(PSITRIMDBX_DEBUG "Enable psitrimdbx debug tracing (PSITRI_DEBUG_BUILD)" OFF)
if(PSITRIMDBX_DEBUG)
    target_compile_definitions(psitrimdbx PUBLIC PSITRI_DEBUG_BUILD=1)
    message(STATUS "psitrimdbx: debug tracing ON")
endif()

option(PSITRIMDBX_BUILD_TESTS "Build the psitrimdbx test suite" ${BUILD_TESTS})
if(PSITRIMDBX_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(psitrimdbx-tests tests/mdbx_tests.cpp)
        target_link_libraries(psitrimdbx-tests PRIVATE psitrimdbx Catch2::Catch2WithMain)
        set_target_properties(psitrimdbx-tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        add_test(NAME psitrimdbx-tests COMMAND psitrimdbx-tests)
    else()
        message(STATUS "psitrimdbx: Catch2 not found, tests disabled")
    endif()
endif()

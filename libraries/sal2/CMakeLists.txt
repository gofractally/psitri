# sal2 — SAL 2.0 FIX-region line allocator (header-only; depends on sal + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(sal2 INTERFACE)
add_library(sal2::sal2 ALIAS sal2)
target_include_directories(sal2 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(sal2 INTERFACE sal ucc)
target_compile_features(sal2 INTERFACE cxx_std_23)

# ── sal2_alloc: SAL 1.0 allocator subsystem copied into namespace sal2 ──────────
# (Phase 1 of the SAL 2.0 port: near-verbatim copy of SAL 1.0's proven allocator
#  — background compactor, MFU read-cache, segment threads, lock-free CB alloc,
#  sessions — under include/sal2/alloc/, with the 48-bit version narrowed to 32.
#  Lands ALONGSIDE the from-scratch sal2 allocators; nothing here is wired to
#  gcart/rcart yet. Phase 2 will retire movable_region in favor of this.)
add_library(sal2_alloc
    src/block_allocator.cpp
    src/allocator.cpp
    src/mapping.cpp
    src/control_block_alloc.cpp
    src/allocator_session.cpp
    src/segment_thread.cpp
)
add_library(sal2::alloc ALIAS sal2_alloc)
target_compile_definitions(sal2_alloc PUBLIC XXH_INLINE_ALL)
target_include_directories(sal2_alloc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(sal2_alloc PUBLIC ucc)
target_compile_features(sal2_alloc PUBLIC cxx_std_23)
target_compile_options(sal2_alloc PRIVATE -Wno-vla-cxx-extension)

if(BUILD_TESTS)
    enable_testing()
    add_executable(sal2-alloc-smoke tests/alloc_smoke.cpp)
    target_link_libraries(sal2-alloc-smoke PRIVATE sal2_alloc)
    set_target_properties(sal2-alloc-smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-alloc-smoke COMMAND sal2-alloc-smoke)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_executable(sal2-tests tests/fuzz_test.cpp)
    target_link_libraries(sal2-tests PRIVATE sal2)
    set_target_properties(sal2-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-tests COMMAND sal2-tests)

    add_executable(sal2-fa-tests tests/fixed_allocator_test.cpp)
    target_link_libraries(sal2-fa-tests PRIVATE sal2)
    set_target_properties(sal2-fa-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-fa-tests COMMAND sal2-fa-tests)

    add_executable(sal2-va-tests tests/var_allocator_test.cpp)
    target_link_libraries(sal2-va-tests PRIVATE sal2)
    set_target_properties(sal2-va-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-va-tests COMMAND sal2-va-tests)

    # Recycle / high-water regression: steady-state churn must keep committed
    # segments bounded. Reproduces the Silkworm "var_allocator full" stranding
    # (single-class bounds; size-drift strands per-class segments). See task #32.
    add_executable(sal2-va-recycle-tests tests/var_allocator_recycle_test.cpp)
    target_link_libraries(sal2-va-recycle-tests PRIVATE sal2)
    set_target_properties(sal2-va-recycle-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-va-recycle-tests COMMAND sal2-va-recycle-tests)

    add_executable(sal2-cb-tests tests/control_block_test.cpp)
    target_link_libraries(sal2-cb-tests PRIVATE sal2)
    set_target_properties(sal2-cb-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-cb-tests COMMAND sal2-cb-tests)
endif()

# sal_hybrid — experimental fixed and movable allocators (depends on sal + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(sal_hybrid INTERFACE)
add_library(sal_hybrid::sal_hybrid ALIAS sal_hybrid)
target_include_directories(sal_hybrid INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(sal_hybrid INTERFACE sal ucc)
target_compile_features(sal_hybrid INTERFACE cxx_std_23)

# The movable leaf allocator is a fork of SAL in namespace sal_hybrid, with
# 32-byte grains and 32-bit allocation versions. gcart uses it for both leaf
# sizes; rcart uses it for small leaves and keeps its standalone big_allocator.
# The fixed routing allocators above stay direct-addressed.
add_library(sal_hybrid_alloc
    src/block_allocator.cpp
    src/allocator.cpp
    src/mapping.cpp
    src/control_block_alloc.cpp
    src/allocator_session.cpp
    src/segment_thread.cpp
)
add_library(sal_hybrid::alloc ALIAS sal_hybrid_alloc)
target_compile_definitions(sal_hybrid_alloc PUBLIC XXH_INLINE_ALL)
target_include_directories(sal_hybrid_alloc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(sal_hybrid_alloc PUBLIC ucc)
target_compile_features(sal_hybrid_alloc PUBLIC cxx_std_23)
target_compile_options(sal_hybrid_alloc PRIVATE -Wno-vla-cxx-extension)

if(BUILD_TESTS)
    enable_testing()
    add_executable(sal_hybrid-alloc-smoke tests/alloc_smoke.cpp)
    target_link_libraries(sal_hybrid-alloc-smoke PRIVATE sal_hybrid_alloc)
    set_target_properties(sal_hybrid-alloc-smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_hybrid-alloc-smoke COMMAND sal_hybrid-alloc-smoke)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_executable(sal_hybrid-tests tests/fuzz_test.cpp)
    target_link_libraries(sal_hybrid-tests PRIVATE sal_hybrid)
    set_target_properties(sal_hybrid-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_hybrid-tests COMMAND sal_hybrid-tests)

    add_executable(sal_hybrid-fa-tests tests/fixed_allocator_test.cpp)
    target_link_libraries(sal_hybrid-fa-tests PRIVATE sal_hybrid)
    set_target_properties(sal_hybrid-fa-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_hybrid-fa-tests COMMAND sal_hybrid-fa-tests)

    add_executable(sal_hybrid-va-tests tests/var_allocator_test.cpp)
    target_link_libraries(sal_hybrid-va-tests PRIVATE sal_hybrid)
    set_target_properties(sal_hybrid-va-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_hybrid-va-tests COMMAND sal_hybrid-va-tests)

    # Recycle / high-water regression: steady-state churn must keep committed
    # segments bounded. Reproduces the Silkworm "var_allocator full" stranding
    # (single-class bounds; size-drift strands per-class segments). See task #32.
    add_executable(sal_hybrid-va-recycle-tests tests/var_allocator_recycle_test.cpp)
    target_link_libraries(sal_hybrid-va-recycle-tests PRIVATE sal_hybrid)
    set_target_properties(sal_hybrid-va-recycle-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_hybrid-va-recycle-tests COMMAND sal_hybrid-va-recycle-tests)

    add_executable(sal_hybrid-cb-tests tests/control_block_test.cpp)
    target_link_libraries(sal_hybrid-cb-tests PRIVATE sal_hybrid)
    set_target_properties(sal_hybrid-cb-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_hybrid-cb-tests COMMAND sal_hybrid-cb-tests)
endif()

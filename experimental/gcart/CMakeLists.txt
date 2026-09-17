# gcart — GC-ART engine on sal_hybrid (header-only; depends on sal_hybrid + adb + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(gcart INTERFACE)
add_library(gcart::gcart ALIAS gcart)
target_include_directories(gcart INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
# Phase 2: gcart's LEAF region rides the copied SAL allocator (sal_hybrid_alloc, a
# compiled library — the from-scratch sal_hybrid header-only FIX allocators still drive
# routing). Link both: sal_hybrid (FIX routing) + sal_hybrid_alloc (SAL leaf).
target_link_libraries(gcart INTERFACE sal_hybrid sal_hybrid_alloc adb ucc)
target_compile_features(gcart INTERFACE cxx_std_23)

if(BUILD_TESTS)
    enable_testing()
    add_executable(gcart-kv-tests tests/kv_test.cpp)
    target_link_libraries(gcart-kv-tests PRIVATE gcart)
    set_target_properties(gcart-kv-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME gcart-kv-tests COMMAND gcart-kv-tests)

    add_executable(gcart-tree-tests tests/tree_test.cpp)
    target_link_libraries(gcart-tree-tests PRIVATE gcart)
    set_target_properties(gcart-tree-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME gcart-tree-tests COMMAND gcart-tree-tests)

    add_executable(gcart-big-leaf-tests tests/big_leaf_test.cpp)
    target_link_libraries(gcart-big-leaf-tests PRIVATE gcart)
    set_target_properties(gcart-big-leaf-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME gcart-big-leaf-tests COMMAND gcart-big-leaf-tests)

    # nested-directly-mutable invariant: create == unique-in-place == shared-clone
    # produce BYTE-IDENTICAL nested lines (lay_nested_into_), + content/UAF/leak guards.
    add_executable(gcart-nested-invariant-tests tests/nested_invariant_test.cpp)
    target_link_libraries(gcart-nested-invariant-tests PRIVATE gcart)
    set_target_properties(gcart-nested-invariant-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME gcart-nested-invariant-tests COMMAND gcart-nested-invariant-tests)

    # A/B perf: gcart (CB-movable leaves) vs rcart (direct-grain leaves).
    add_executable(gcart-bench tests/bench.cpp)
    target_link_libraries(gcart-bench PRIVATE gcart rcart)
    set_target_properties(gcart-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

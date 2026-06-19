# rcart — refcount-baseline GC-ART engine on sal2 (same node taxonomy as gcart;
# header-only; depends on sal2 + adb + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(rcart INTERFACE)
add_library(rcart::rcart ALIAS rcart)
target_include_directories(rcart INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
# rcart's SMALL leaf region (suffix_value) rides the copied SAL 1.0 allocator
# (sal2_alloc, a compiled library — the from-scratch sal2 header-only FIX allocators
# still drive routing, and big_suffix_value stays on the header-only big_allocator).
# Link both: sal2 (FIX routing + BIG) + sal2_alloc (SAL 1.0 leaf).
target_link_libraries(rcart INTERFACE sal2 sal2_alloc adb ucc)
target_compile_features(rcart INTERFACE cxx_std_23)

if(BUILD_TESTS)
    enable_testing()
    add_executable(rcart-kv-tests tests/kv_test.cpp)
    target_link_libraries(rcart-kv-tests PRIVATE rcart)
    set_target_properties(rcart-kv-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-kv-tests COMMAND rcart-kv-tests)

    add_executable(rcart-tree-tests tests/tree_test.cpp)
    target_link_libraries(rcart-tree-tests PRIVATE rcart)
    set_target_properties(rcart-tree-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-tree-tests COMMAND rcart-tree-tests)

    # nested_setlist byte-identity invariant: a node's bytes are a pure function of
    # (prefix, entries, term) — identical via the unique in-place edit AND the shared
    # clone+edit paths (the directly-mutable-nested / kill-promote-on-write property).
    add_executable(rcart-nested-invariant-tests tests/nested_invariant_test.cpp)
    target_link_libraries(rcart-nested-invariant-tests PRIVATE rcart)
    set_target_properties(rcart-nested-invariant-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-nested-invariant-tests COMMAND rcart-nested-invariant-tests)
endif()

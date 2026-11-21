cmake_minimum_required(VERSION 3.16)
project(sal_art_ref VERSION 0.1.0 LANGUAGES CXX)

# sal_art_ref is an alternative radix-tree engine to libraries/psitri. It depends only
# on sal (for ptr_address / allocator types) and ucc; it must NOT depend on psitri.
#
# Phase 1 is header-only utility classes (bitset, compact_addresses,
# smart_address) that later compose the node hierarchy. Keep it INTERFACE until a
# .cpp is genuinely needed.
add_library(sal_art_ref INTERFACE)

target_include_directories(sal_art_ref INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(sal_art_ref INTERFACE cxx_std_23)
target_link_libraries(sal_art_ref INTERFACE sal ucc adb)

# Tests
option(SAL_ART_REF_BUILD_TESTS "Build the sal_art_ref test suite" OFF)
if(SAL_ART_REF_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 REQUIRED)

    add_executable(sal_art_ref-tests
        test/bitset_tests.cpp
        test/compact_addresses_tests.cpp
        test/smart_address_tests.cpp
        test/radix_bitset_tests.cpp
        test/radix_full_tests.cpp
        test/paged_tests.cpp
        test/leaf_page_tests.cpp
        test/radix_node_tests.cpp
        test/branch_smart_address_tests.cpp
        test/radix_tree_tests.cpp
        test/sal_radix_tree_tests.cpp
        test/radix_engine_tests.cpp
        test/shared_tree_tests.cpp
    )
    target_link_libraries(sal_art_ref-tests
        PRIVATE
            sal_art_ref
            Catch2::Catch2WithMain
    )
    target_compile_options(sal_art_ref-tests PRIVATE -Wall -Wextra)
    # The unit tests assert on cow_counters (e.g. restrides >= 1 after a split), so
    # they must build with the perf counters live. PSITRI_DEBUG is left OFF here:
    # the gated paths (op-log/validate/dump) are compile-checked separately, and the
    # validate path aborts on the first structurally-odd node, which the tests neither
    # want nor need on the normal path.
    target_compile_definitions(sal_art_ref-tests PRIVATE PSITRI_STATS)
    set_target_properties(sal_art_ref-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    include(CTest)
    include(Catch)
    catch_discover_tests(sal_art_ref-tests)
endif()

# Sanitizer / coverage propagation to match sibling libraries.
if(ENABLE_SANITIZER AND TARGET sal_art_ref-tests)
    target_compile_options(sal_art_ref-tests PUBLIC -fsanitize=thread -g)
    target_link_options(sal_art_ref-tests PUBLIC -fsanitize=thread -g)
endif()
if(ENABLE_COVERAGE AND TARGET sal_art_ref-tests)
    target_compile_options(sal_art_ref-tests PUBLIC -fprofile-arcs -ftest-coverage -g -O1 -DNDEBUG)
    target_link_options(sal_art_ref-tests PUBLIC -fprofile-arcs -ftest-coverage)
endif()

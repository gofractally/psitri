cmake_minimum_required(VERSION 3.16)
project(heart VERSION 0.1.0 LANGUAGES CXX)

# heart is an alternative radix-tree engine to libraries/psitri. It depends only
# on sal (for ptr_address / allocator types) and ucc; it must NOT depend on psitri.
#
# Phase 1 is header-only utility classes (bitset, compact_addresses,
# smart_address) that later compose the node hierarchy. Keep it INTERFACE until a
# .cpp is genuinely needed.
add_library(heart INTERFACE)

target_include_directories(heart INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(heart INTERFACE cxx_std_23)
target_link_libraries(heart INTERFACE sal ucc)

# Tests
option(HEART_BUILD_TESTS "Build the heart test suite" OFF)
if(HEART_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 REQUIRED)

    add_executable(heart-tests
        test/bitset_tests.cpp
        test/compact_addresses_tests.cpp
        test/smart_address_tests.cpp
        test/radix_bitset_tests.cpp
        test/radix_full_tests.cpp
        test/paged_tests.cpp
        test/radix_node_tests.cpp
        test/branch_smart_address_tests.cpp
        test/radix_tree_tests.cpp
        test/sal_radix_tree_tests.cpp
        test/radix_engine_tests.cpp
    )
    target_link_libraries(heart-tests
        PRIVATE
            heart
            Catch2::Catch2WithMain
    )
    target_compile_options(heart-tests PRIVATE -Wall -Wextra)
    set_target_properties(heart-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    include(CTest)
    include(Catch)
    catch_discover_tests(heart-tests)
endif()

# Sanitizer / coverage propagation to match sibling libraries.
if(ENABLE_SANITIZER AND TARGET heart-tests)
    target_compile_options(heart-tests PUBLIC -fsanitize=thread -g)
    target_link_options(heart-tests PUBLIC -fsanitize=thread -g)
endif()
if(ENABLE_COVERAGE AND TARGET heart-tests)
    target_compile_options(heart-tests PUBLIC -fprofile-arcs -ftest-coverage -g -O1 -DNDEBUG)
    target_link_options(heart-tests PUBLIC -fprofile-arcs -ftest-coverage)
endif()

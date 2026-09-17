cmake_minimum_required(VERSION 3.16)
project(start_ref VERSION 0.1.0 LANGUAGES CXX)

# start_ref — static ART with refcount liveness (formerly `part`; see
# SPEC.md). Layer 1 is a non-relocating, line-granular (128 B) allocator
# whose placement policy minimizes msync write amplification. Depends on
# sal + ucc + the standard-only adb helpers — never on psitri.
add_library(start_ref INTERFACE)

target_include_directories(start_ref INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(start_ref INTERFACE cxx_std_23)
target_link_libraries(start_ref INTERFACE sal ucc adb)

option(START_REF_BUILD_TESTS "Build the start_ref test suite" ${BUILD_TESTS})
if(START_REF_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(start_ref-tests
            test/freeset_tests.cpp
            test/arena_tests.cpp
            test/tree_tests.cpp
            test/cow_tests.cpp
            test/kv_tests.cpp
            test/cbucket_tests.cpp
        )
        target_link_libraries(start_ref-tests PRIVATE start_ref Catch2::Catch2WithMain)
        target_compile_options(start_ref-tests PRIVATE -O1 -g -UNDEBUG -Wall -Werror=switch)
        set_target_properties(start_ref-tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME start_ref-tests COMMAND start_ref-tests)

        # Concurrency suite under ThreadSanitizer (own binary: TSan must not mix
        # with the plain-build objects).
        add_executable(start_ref-tests-tsan
            test/freeset_mt_tests.cpp
        )
        target_link_libraries(start_ref-tests-tsan PRIVATE start_ref Catch2::Catch2WithMain)
        target_compile_options(start_ref-tests-tsan PRIVATE -O1 -g -fsanitize=thread -Wall)
        target_link_options(start_ref-tests-tsan PRIVATE -fsanitize=thread)
        set_target_properties(start_ref-tests-tsan PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME start_ref-tests-tsan COMMAND start_ref-tests-tsan)
    else()
        message(STATUS "start_ref: Catch2 not found, tests disabled")
    endif()
endif()

# The part-vs-libart A/B bench stayed in the psitri-merge archive; the
# cross-engine benchmark suite (benchmarks/) supersedes it.

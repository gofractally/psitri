cmake_minimum_required(VERSION 3.16)
project(start_ver VERSION 0.1.0 LANGUAGES CXX)

# part — persistent ART (see SPEC.md). Layer 1 is a non-relocating, line-granular
# (128 B) allocator whose placement policy minimizes msync write amplification.
# Like heart, it depends only on sal + ucc and must NOT depend on psitri.
add_library(start_ver INTERFACE)

target_include_directories(start_ver INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(start_ver INTERFACE cxx_std_23)
target_link_libraries(start_ver INTERFACE sal ucc)

option(START_VER_BUILD_TESTS "Build the part test suite" ${BUILD_TESTS})
if(START_VER_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(start_ver-tests
            test/freeset_tests.cpp
            test/arena_tests.cpp
            test/tree_tests.cpp
            test/cow_tests.cpp
            test/kv_tests.cpp
            test/large_value_tests.cpp
        )
        target_link_libraries(start_ver-tests PRIVATE start_ver Catch2::Catch2WithMain)
        target_compile_options(start_ver-tests PRIVATE -O1 -g -UNDEBUG -Wall -Werror=switch)
        set_target_properties(start_ver-tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME start_ver-tests COMMAND start_ver-tests)

        # Concurrency suite under ThreadSanitizer (own binary: TSan must not mix
        # with the plain-build objects).
        add_executable(start_ver-tests-tsan
            test/freeset_mt_tests.cpp
        )
        target_link_libraries(start_ver-tests-tsan PRIVATE start_ver Catch2::Catch2WithMain)
        target_compile_options(start_ver-tests-tsan PRIVATE -O1 -g -fsanitize=thread -Wall)
        target_link_options(start_ver-tests-tsan PRIVATE -fsanitize=thread)
        set_target_properties(start_ver-tests-tsan PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME start_ver-tests-tsan COMMAND start_ver-tests-tsan)
    else()
        message(STATUS "part: Catch2 not found, tests disabled")
    endif()
endif()

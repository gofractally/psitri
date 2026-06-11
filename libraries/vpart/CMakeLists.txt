cmake_minimum_required(VERSION 3.16)
project(vpart VERSION 0.1.0 LANGUAGES CXX)

# part — persistent ART (see SPEC.md). Layer 1 is a non-relocating, line-granular
# (128 B) allocator whose placement policy minimizes msync write amplification.
# Like heart, it depends only on sal + ucc and must NOT depend on psitri.
add_library(vpart INTERFACE)

target_include_directories(vpart INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(vpart INTERFACE cxx_std_23)
target_link_libraries(vpart INTERFACE sal ucc)

option(VPART_BUILD_TESTS "Build the part test suite" ON)
if(VPART_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(vpart-tests
            test/freeset_tests.cpp
            test/arena_tests.cpp
            test/tree_tests.cpp
            test/cow_tests.cpp
            test/kv_tests.cpp
        )
        target_link_libraries(vpart-tests PRIVATE vpart Catch2::Catch2WithMain)
        target_compile_options(vpart-tests PRIVATE -O1 -g -UNDEBUG -Wall -Werror=switch)
        set_target_properties(vpart-tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME vpart-tests COMMAND vpart-tests)

        # Concurrency suite under ThreadSanitizer (own binary: TSan must not mix
        # with the plain-build objects).
        add_executable(vpart-tests-tsan
            test/freeset_mt_tests.cpp
        )
        target_link_libraries(vpart-tests-tsan PRIVATE vpart Catch2::Catch2WithMain)
        target_compile_options(vpart-tests-tsan PRIVATE -O1 -g -fsanitize=thread -Wall)
        target_link_options(vpart-tests-tsan PRIVATE -fsanitize=thread)
        set_target_properties(vpart-tests-tsan PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME vpart-tests-tsan COMMAND vpart-tests-tsan)
    else()
        message(STATUS "part: Catch2 not found, tests disabled")
    endif()
endif()

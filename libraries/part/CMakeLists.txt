cmake_minimum_required(VERSION 3.16)
project(part VERSION 0.1.0 LANGUAGES CXX)

# part — persistent ART (see SPEC.md). Layer 1 is a non-relocating, line-granular
# (128 B) allocator whose placement policy minimizes msync write amplification.
# Like heart, it depends only on sal + ucc and must NOT depend on psitri.
add_library(part INTERFACE)

target_include_directories(part INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(part INTERFACE cxx_std_23)
target_link_libraries(part INTERFACE sal ucc)

option(PART_BUILD_TESTS "Build the part test suite" ON)
if(PART_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(part-tests
            test/freeset_tests.cpp
        )
        target_link_libraries(part-tests PRIVATE part Catch2::Catch2WithMain)
        target_compile_options(part-tests PRIVATE -O1 -g -Wall -Werror=switch)
        set_target_properties(part-tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME part-tests COMMAND part-tests)

        # Concurrency suite under ThreadSanitizer (own binary: TSan must not mix
        # with the plain-build objects).
        add_executable(part-tests-tsan
            test/freeset_mt_tests.cpp
        )
        target_link_libraries(part-tests-tsan PRIVATE part Catch2::Catch2WithMain)
        target_compile_options(part-tests-tsan PRIVATE -O1 -g -fsanitize=thread -Wall)
        target_link_options(part-tests-tsan PRIVATE -fsanitize=thread)
        set_target_properties(part-tests-tsan PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME part-tests-tsan COMMAND part-tests-tsan)
    else()
        message(STATUS "part: Catch2 not found, tests disabled")
    endif()
endif()

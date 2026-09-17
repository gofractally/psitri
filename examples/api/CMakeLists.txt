# This directory also serves as a standalone third-party consumer check:
# cmake -S examples/api -B build/consumer -DPSITRI_SOURCE=/path/to/psitri
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(psitri_api_examples LANGUAGES C CXX)
    if(NOT PSITRI_SOURCE)
        message(FATAL_ERROR "Set PSITRI_SOURCE to the psitri checkout")
    endif()
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(PSITRI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${PSITRI_SOURCE}" psitri)
endif()

add_executable(psitri-example-big basic.cpp)
target_link_libraries(psitri-example-big PRIVATE psitri::api)
add_executable(psitri-example-small basic.cpp)
target_compile_definitions(psitri-example-small PRIVATE PSITRI_EXAMPLE_SMALL)
target_link_libraries(psitri-example-small PRIVATE psitri::api)

add_executable(psitri-example-revisions revisions.cpp)
target_link_libraries(psitri-example-revisions PRIVATE psitri::big)

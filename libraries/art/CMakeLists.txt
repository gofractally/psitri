cmake_minimum_required(VERSION 3.16)
add_library(art INTERFACE)
target_include_directories(art INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(art INTERFACE ucc)

option(ART_BUILD_TESTS "Build the art test suite" ON)
if(ART_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(art-tests
            tests/art_map_tests.cpp
            tests/cow_coordinator_tests.cpp
        )
        target_link_libraries(art-tests PRIVATE art Catch2::Catch2WithMain)
        target_compile_options(art-tests PRIVATE -O1 -g -UNDEBUG -Wall)
        set_target_properties(art-tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME art-tests COMMAND art-tests)
    endif()
endif()

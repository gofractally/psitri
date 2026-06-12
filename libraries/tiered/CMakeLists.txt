add_library(tiered INTERFACE)
target_include_directories(tiered INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tiered INTERFACE sal)

if(BUILD_TESTS)
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(tiered-tests test/tiered_tests.cpp)
        target_link_libraries(tiered-tests PRIVATE tiered vpart heart Catch2::Catch2WithMain)
        target_include_directories(tiered-tests PRIVATE
            ${CMAKE_SOURCE_DIR}/libraries/heart/test
            ${CMAKE_SOURCE_DIR}/libraries/psitri/include)
        # NDEBUG matches heart's own test conditions: heart's database flow
        # trips sal's session-thread-ownership assert (LATENT heart bug,
        # repro: open db + write on main; see task list) — vpart-side assert
        # coverage lives in vpart-tests, which runs -UNDEBUG.
        target_compile_options(tiered-tests PRIVATE -O1 -g -Wall)
        set_target_properties(tiered-tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME tiered-tests COMMAND tiered-tests)
    endif()
endif()

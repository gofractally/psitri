add_library(delta_art INTERFACE)
target_include_directories(delta_art INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(delta_art INTERFACE sal adb)

if(BUILD_TESTS)
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(delta_art-tests test/delta_art_tests.cpp)
        target_link_libraries(delta_art-tests PRIVATE delta_art start_ver sal_art_ref Catch2::Catch2WithMain)
        target_include_directories(delta_art-tests PRIVATE
            ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test
            ${PSITRI_SOURCE_DIR}/libraries/psitri/include)
        # NDEBUG matches sal_art_ref's own test conditions: sal_art_ref's database flow
        # trips sal's session-thread-ownership assert (LATENT sal_art_ref bug,
        # repro: open db + write on main; see task list) — start_ver-side assert
        # coverage lives in start_ver-tests, which runs -UNDEBUG.
        target_compile_options(delta_art-tests PRIVATE -O1 -g -Wall)
        set_target_properties(delta_art-tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME delta_art-tests COMMAND delta_art-tests)
    endif()
endif()

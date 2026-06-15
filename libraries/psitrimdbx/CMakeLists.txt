# PsiTriMDBX: MDBX-compatible drop-in API. The generic shim (dbi mapping,
# DUPSORT composite-key encoding, cursor ops, txn lifecycle) is engine-agnostic;
# the backend now rides the new fast model — the delta_art tiered engine
# (start_ver hot delta over a sal_art_ref dwal cold base).
add_library(psitrimdbx
    src/mdbx_impl.cpp
)

target_include_directories(psitrimdbx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # heart_dwal_engine.hpp / radix_engine.hpp (the cold base instantiation)
        ${CMAKE_SOURCE_DIR}/libraries/sal_art_ref/test
)

target_link_libraries(psitrimdbx PUBLIC psitri delta_art start_ver start_ref sal_art_ref)
target_compile_features(psitrimdbx PUBLIC cxx_std_20)

option(PSITRIMDBX_BUILD_TESTS "Build the psitrimdbx test suite" ON)
if(PSITRIMDBX_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(psitrimdbx-tests tests/mdbx_tests.cpp)
        target_link_libraries(psitrimdbx-tests PRIVATE psitrimdbx Catch2::Catch2WithMain)
        set_target_properties(psitrimdbx-tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        add_test(NAME psitrimdbx-tests COMMAND psitrimdbx-tests)
    else()
        message(STATUS "psitrimdbx: Catch2 not found, tests disabled")
    endif()
endif()

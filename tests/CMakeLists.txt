# tests/ — the behavioral battery, written against adb, instantiated on every
# engine's NATIVE database type. Test-shims (3rd-party DB wrappers for these
# drivers) will also live here — they are not products.
find_package(Catch2 3 QUIET)

add_executable(kv-compat-test kv_compat_test.cpp)
target_link_libraries(kv-compat-test PRIVATE adb sal_art_ref start_ref start_ver gcart)
target_include_directories(kv-compat-test PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)  # heart_kv.hpp (hoist later)
target_compile_definitions(kv-compat-test PRIVATE XXH_INLINE_ALL)
set_target_properties(kv-compat-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME kv-compat-test COMMAND kv-compat-test)

if(Catch2_FOUND)
    add_executable(api-conformance-tests api/conformance.cpp api/large_values.cpp api/small_large_values.cpp api/writers.cpp api/transactions.cpp)
    target_link_libraries(api-conformance-tests PRIVATE psitri::api psitri::transactions psitri::experimental Catch2::Catch2WithMain)
    set_target_properties(api-conformance-tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME api-conformance-tests COMMAND api-conformance-tests)

    # A second root written key by key by its own thread (gofractally/psibase#2268).
    add_executable(root-writer-stress-tests api/root_writer_stress.cpp)
    target_link_libraries(root-writer-stress-tests PRIVATE psitri::api Catch2::Catch2WithMain)
    set_target_properties(root-writer-stress-tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME root-writer-stress-tests COMMAND root-writer-stress-tests)

    # Small Engine operating limits: process kill, space reuse, storage ceiling.
    add_executable(small-engine-tests api/small_engine.cpp)
    target_link_libraries(small-engine-tests PRIVATE psitri::api psitri::small Catch2::Catch2WithMain)
    set_target_properties(small-engine-tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME small-engine-tests COMMAND small-engine-tests)

    # Large Engine: death of the process at chosen points (gofractally/psitri#44).
    # The crash points are compiled in only here: this binary builds the allocator's
    # sources itself with SAL_CRASH_POINTS, and does not link the sal library, so that
    # every translation unit in it agrees about the markers.
    get_target_property(_lge_sal_sources sal SOURCES)
    get_target_property(_lge_sal_dir sal SOURCE_DIR)
    set(_lge_sources api/large_engine_crash.cpp)
    foreach(_src ${_lge_sal_sources})
        list(APPEND _lge_sources "${_lge_sal_dir}/${_src}")
    endforeach()
    # Every marker name in the sources, for "which points did a campaign never die at".
    file(GLOB_RECURSE _lge_marked CONFIGURE_DEPENDS
        "${PSITRI_SOURCE_DIR}/libraries/sal/include/sal/mapped_memory/*.hpp"
        "${PSITRI_SOURCE_DIR}/libraries/sal/include/sal/alloc*.hpp"
        "${PSITRI_SOURCE_DIR}/libraries/sal/include/sal/control_block*.hpp"
        "${PSITRI_SOURCE_DIR}/libraries/sal/src/*.cpp"
        "${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/include/*.hpp")
    set(_lge_points "")
    foreach(_file ${_lge_marked})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
        file(READ "${_file}" _text)
        string(REGEX MATCHALL "SAL_CRASH_POINT\\(\"[a-z_:]+\"" _found "${_text}")
        foreach(_match ${_found})
            string(REGEX REPLACE "SAL_CRASH_POINT\\(\"([a-z_:]+)\"" "\\1" _name "${_match}")
            list(APPEND _lge_points "${_name}")
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _lge_points)
    list(SORT _lge_points)
    set(_lge_inc "")
    foreach(_name ${_lge_points})
        string(APPEND _lge_inc "\"${_name}\",\n")
    endforeach()
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/lge_crash/large_engine_crash_points.inc"
         CONTENT "${_lge_inc}")
    add_executable(large-engine-crash-tests ${_lge_sources})
    target_compile_definitions(large-engine-crash-tests PRIVATE SAL_CRASH_POINTS XXH_INLINE_ALL)
    target_include_directories(large-engine-crash-tests PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/lge_crash"
        "${PSITRI_SOURCE_DIR}/libraries/sal/include"
        "${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/include"
        "${PSITRI_SOURCE_DIR}/libraries/psitri_api/include")
    target_link_libraries(large-engine-crash-tests PRIVATE adb ucc Catch2::Catch2WithMain)
    target_compile_features(large-engine-crash-tests PRIVATE cxx_std_23)
    set_target_properties(large-engine-crash-tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME large-engine-crash-tests COMMAND large-engine-crash-tests)
    set_tests_properties(large-engine-crash-tests PROPERTIES TIMEOUT 1800)

    # The same writer and reader cases on the engines layered over others.
    if(TARGET psitri)
        add_executable(layered-conformance-tests api/layered.cpp)
        target_link_libraries(layered-conformance-tests PRIVATE psitri psitri::api psitri::experimental Catch2::Catch2WithMain)
        set_target_properties(layered-conformance-tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME layered-conformance-tests COMMAND layered-conformance-tests)
    endif()
endif()

if(UNIX AND NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(durability)
endif()

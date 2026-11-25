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

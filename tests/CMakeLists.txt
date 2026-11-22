# tests/ — the behavioral battery, written against adb, instantiated on every
# engine's NATIVE database type. Test-shims (3rd-party DB wrappers for these
# drivers) will also live here — they are not products.
find_package(Catch2 3 QUIET)

add_executable(kv-compat-test kv_compat_test.cpp)
target_link_libraries(kv-compat-test PRIVATE adb psitri sal_art_ref start_ref start_ver gcart)
target_include_directories(kv-compat-test PRIVATE
    ${PSITRI_SOURCE_DIR}/libraries/sal_art_ref/test)  # heart_kv.hpp (hoist later)
target_compile_definitions(kv-compat-test PRIVATE XXH_INLINE_ALL)
set_target_properties(kv-compat-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME kv-compat-test COMMAND kv-compat-test)

if(Catch2_FOUND)
    add_executable(api-conformance-tests api/conformance.cpp api/large_values.cpp api/small_large_values.cpp api/writers.cpp api/transactions.cpp)
    target_link_libraries(api-conformance-tests PRIVATE psitri::api psitri::experimental Catch2::Catch2WithMain)
    set_target_properties(api-conformance-tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME api-conformance-tests COMMAND api-conformance-tests)
endif()

if(UNIX AND NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(durability)
endif()

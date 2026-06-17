# gcart — GC-ART engine on sal2 (header-only; depends on sal2 + adb + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(gcart INTERFACE)
add_library(gcart::gcart ALIAS gcart)
target_include_directories(gcart INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(gcart INTERFACE sal2 adb ucc)
target_compile_features(gcart INTERFACE cxx_std_23)

if(BUILD_TESTS)
    enable_testing()
    add_executable(gcart-tree-tests tests/tree_test.cpp)
    target_link_libraries(gcart-tree-tests PRIVATE gcart)
    set_target_properties(gcart-tree-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME gcart-tree-tests COMMAND gcart-tree-tests)

    add_executable(gcart-spec-tests tests/spec_tree_test.cpp)
    target_link_libraries(gcart-spec-tests PRIVATE gcart)
    set_target_properties(gcart-spec-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME gcart-spec-tests COMMAND gcart-spec-tests)

    add_executable(gcart-kv-tests tests/kv_test.cpp)
    target_link_libraries(gcart-kv-tests PRIVATE gcart)
    set_target_properties(gcart-kv-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME gcart-kv-tests COMMAND gcart-kv-tests)
endif()

if(BUILD_BENCHMARKS)
    add_executable(gcart-bench tests/bench.cpp)
    target_link_libraries(gcart-bench PRIVATE gcart)
    set_target_properties(gcart-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

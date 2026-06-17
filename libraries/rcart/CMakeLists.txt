# rcart — GC-ART engine on sal2 (header-only; depends on sal2 + adb + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(rcart INTERFACE)
add_library(rcart::rcart ALIAS rcart)
target_include_directories(rcart INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(rcart INTERFACE sal2 adb ucc)
target_compile_features(rcart INTERFACE cxx_std_23)

if(BUILD_TESTS)
    enable_testing()
    add_executable(rcart-tree-tests tests/tree_test.cpp)
    target_link_libraries(rcart-tree-tests PRIVATE rcart)
    set_target_properties(rcart-tree-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-tree-tests COMMAND rcart-tree-tests)

    add_executable(rcart-spec-kv-tests tests/spec_kv_test.cpp)
    target_link_libraries(rcart-spec-kv-tests PRIVATE rcart)
    set_target_properties(rcart-spec-kv-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-spec-kv-tests COMMAND rcart-spec-kv-tests)

    add_executable(rcart-spec-tests tests/spec_tree_test.cpp)
    target_link_libraries(rcart-spec-tests PRIVATE rcart)
    set_target_properties(rcart-spec-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-spec-tests COMMAND rcart-spec-tests)

    add_executable(rcart-kv-tests tests/kv_test.cpp)
    target_link_libraries(rcart-kv-tests PRIVATE rcart)
    set_target_properties(rcart-kv-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-kv-tests COMMAND rcart-kv-tests)
endif()

if(BUILD_BENCHMARKS)
    add_executable(rcart-bench tests/bench.cpp)
    target_link_libraries(rcart-bench PRIVATE rcart)
    set_target_properties(rcart-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

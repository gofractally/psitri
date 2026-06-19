# rcart — refcount-baseline GC-ART engine on sal2 (same node taxonomy as gcart,
# direct-grain leaves; header-only; depends on sal2 + adb + ucc).
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
    add_executable(rcart-kv-tests tests/kv_test.cpp)
    target_link_libraries(rcart-kv-tests PRIVATE rcart)
    set_target_properties(rcart-kv-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-kv-tests COMMAND rcart-kv-tests)

    add_executable(rcart-tree-tests tests/tree_test.cpp)
    target_link_libraries(rcart-tree-tests PRIVATE rcart)
    set_target_properties(rcart-tree-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME rcart-tree-tests COMMAND rcart-tree-tests)
endif()

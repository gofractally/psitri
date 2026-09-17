# sal_gc — experimental version-liveness allocators (depends on sal + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(sal_gc INTERFACE)
add_library(sal_gc::sal_gc ALIAS sal_gc)
target_include_directories(sal_gc INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(sal_gc INTERFACE sal ucc)
target_compile_features(sal_gc INTERFACE cxx_std_23)

if(BUILD_TESTS)
    enable_testing()
    add_executable(sal_gc-tests tests/fuzz_test.cpp)
    target_link_libraries(sal_gc-tests PRIVATE sal_gc)
    set_target_properties(sal_gc-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_gc-tests COMMAND sal_gc-tests)

    add_executable(sal_gc-fa-tests tests/fixed_allocator_test.cpp)
    target_link_libraries(sal_gc-fa-tests PRIVATE sal_gc)
    set_target_properties(sal_gc-fa-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_gc-fa-tests COMMAND sal_gc-fa-tests)

    add_executable(sal_gc-va-tests tests/var_allocator_test.cpp)
    target_link_libraries(sal_gc-va-tests PRIVATE sal_gc)
    set_target_properties(sal_gc-va-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal_gc-va-tests COMMAND sal_gc-va-tests)
endif()

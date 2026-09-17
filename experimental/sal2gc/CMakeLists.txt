# sal2gc — SAL 2.0 FIX-region line allocator (header-only; depends on sal + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(sal2gc INTERFACE)
add_library(sal2gc::sal2gc ALIAS sal2gc)
target_include_directories(sal2gc INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(sal2gc INTERFACE sal ucc)
target_compile_features(sal2gc INTERFACE cxx_std_23)

if(BUILD_TESTS)
    enable_testing()
    add_executable(sal2gc-tests tests/fuzz_test.cpp)
    target_link_libraries(sal2gc-tests PRIVATE sal2gc)
    set_target_properties(sal2gc-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2gc-tests COMMAND sal2gc-tests)

    add_executable(sal2gc-fa-tests tests/fixed_allocator_test.cpp)
    target_link_libraries(sal2gc-fa-tests PRIVATE sal2gc)
    set_target_properties(sal2gc-fa-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2gc-fa-tests COMMAND sal2gc-fa-tests)

    add_executable(sal2gc-va-tests tests/var_allocator_test.cpp)
    target_link_libraries(sal2gc-va-tests PRIVATE sal2gc)
    set_target_properties(sal2gc-va-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2gc-va-tests COMMAND sal2gc-va-tests)
endif()

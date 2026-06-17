# sal2 — SAL 2.0 FIX-region line allocator (header-only; depends on sal + ucc).
cmake_minimum_required(VERSION 3.20)

add_library(sal2 INTERFACE)
add_library(sal2::sal2 ALIAS sal2)
target_include_directories(sal2 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(sal2 INTERFACE sal ucc)
target_compile_features(sal2 INTERFACE cxx_std_23)

if(BUILD_TESTS)
    enable_testing()
    add_executable(sal2-tests tests/fuzz_test.cpp)
    target_link_libraries(sal2-tests PRIVATE sal2)
    set_target_properties(sal2-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-tests COMMAND sal2-tests)

    add_executable(sal2-fa-tests tests/fixed_allocator_test.cpp)
    target_link_libraries(sal2-fa-tests PRIVATE sal2)
    set_target_properties(sal2-fa-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-fa-tests COMMAND sal2-fa-tests)

    add_executable(sal2-va-tests tests/var_allocator_test.cpp)
    target_link_libraries(sal2-va-tests PRIVATE sal2)
    set_target_properties(sal2-va-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME sal2-va-tests COMMAND sal2-va-tests)
endif()

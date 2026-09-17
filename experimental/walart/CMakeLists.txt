# walart — WAL + art_map write-set engine over the heart base (formerly
# validated via bank-bench-walart). REMAINS pending the use-case review
# (reorg ruling C); crash-fuzz gate is a carried TODO before default-on.
add_library(walart INTERFACE)
target_include_directories(walart INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(walart INTERFACE sal_art_ref start_ref art)

if(BUILD_TESTS)
    add_executable(walart-compile-check test_compile.cpp)
    target_link_libraries(walart-compile-check PRIVATE walart)
    target_compile_definitions(walart-compile-check PRIVATE XXH_INLINE_ALL)
endif()

# benchmarks/ — written against adb, instantiated per engine. The composed
# pipeline benches (delta_art = hot delta over cold base) drive the
# start_ver + sal_art_ref instantiation.

# delta_art scale bench: start_ver (hot delta) -> sal_art_ref dwal (cold base).
# The cold engine is sal_art_ref::radix::dwal_engine (heart_dwal_engine.hpp,
# under sal_art_ref/test); it uses psitri's dwal::kv_engine concept.
add_executable(delta_art-scale-bench delta_art_scale_bench.cpp)
target_link_libraries(delta_art-scale-bench PRIVATE delta_art start_ver sal_art_ref psitri)
target_include_directories(delta_art-scale-bench PRIVATE
    ${CMAKE_SOURCE_DIR}/libraries/sal_art_ref/test)  # radix_engine + heart_dwal_engine
target_compile_definitions(delta_art-scale-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(delta_art-scale-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(delta_art-scale-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# walart scale bench: WAL + art_map write-set (sal_art_ref base + start_ref delta).
add_executable(walart-scale-bench walart_scale_bench.cpp)
target_link_libraries(walart-scale-bench PRIVATE walart sal_art_ref start_ref psitri)
target_compile_definitions(walart-scale-bench PRIVATE XXH_INLINE_ALL)
target_compile_options(walart-scale-bench PRIVATE -O3 -DNDEBUG)
set_target_properties(walart-scale-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

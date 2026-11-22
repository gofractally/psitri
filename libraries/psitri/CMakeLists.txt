add_library( psitri
             src/database.cpp
             src/node/leaf.cpp
             src/node/inner.cpp
             src/dwal/wal_writer.cpp
             src/dwal/wal_status.cpp
             src/dwal/wal_reader.cpp
             src/dwal/dwal_transaction.cpp
             src/dwal/dwal_database.cpp
             src/dwal/transaction.cpp
             src/dwal/dwal_read_session.cpp
             src/dwal/merge_cursor.cpp
             src/dwal/merge_pool.cpp
         )

target_include_directories( psitri
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

# Define XXH_INLINE_ALL to use xxhash as header-only
target_compile_definitions(psitri PUBLIC XXH_INLINE_ALL)

target_link_libraries( psitri PUBLIC sal ucc art psitri_api_transactions)

option(PSITRI_BUILD_TESTS "Build the psitri test suite" ${BUILD_TESTS})
if(PSITRI_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(psitri-tests
            tests/test_main.cpp
            tests/leaf_node_test.cpp
            tests/leaf_node_transform_tests.cpp
            tests/inner_node_tests.cpp
            tests/inner_node_util_tests.cpp
            tests/leaf_node_split_test.cpp
            tests/trie_tests.cpp
            tests/tree_context_tests.cpp
            tests/database_test.cpp
            tests/public_api_tests.cpp
            tests/count_keys_tests.cpp
            tests/range_remove_tests.cpp
            tests/subtree_tests.cpp
            tests/tree_handle_tests.cpp
            tests/multi_writer_tests.cpp
            tests/recovery_tests.cpp
            tests/fuzz_tests.cpp
            tests/cursor_tests.cpp
            tests/integrity_tests.cpp
            tests/freed_space_tests.cpp
            tests/edge_case_tests.cpp
            tests/update_value_tests.cpp
            tests/zip_tests.cpp
            tests/deep_recovery_tests.cpp
            tests/coverage_gap_tests.cpp
            tests/tree_ops_tests.cpp
            tests/truncate_tests.cpp
            tests/smart_ptr_tests.cpp
            tests/dwal_tests.cpp
            tests/dwal_snapshot_tests.cpp
            tests/write_buffer_tests.cpp
            tests/value_node_tests.cpp
            tests/live_range_map_tests.cpp
            # MVCC removed (single version 0 everywhere): disabled, not deleted.
            # tests/mvcc_tests.cpp
            # tests/per_txn_version_tests.cpp
            tests/test_dwal_cursor_visibility.cpp
        )
        target_link_libraries(psitri-tests PRIVATE psitri Catch2::Catch2)
        target_compile_definitions(psitri-tests PUBLIC XXH_INLINE_ALL)
        set_target_properties(psitri-tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        add_test(NAME psitri-tests COMMAND psitri-tests)
    endif()
endif()

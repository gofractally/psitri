# adb — abstract database concepts (header-only, std-only).
#
# Independently consumable (ruling A): builds standalone from this directory
# or as a subdirectory / FetchContent target, so external consumers
# (psitri-multiindex) can depend on the concepts without the engines.

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(adb VERSION 0.1.0 LANGUAGES CXX)
    set(ADB_STANDALONE ON)
else()
    set(ADB_STANDALONE OFF)
endif()

add_library(adb INTERFACE)
add_library(adb::adb ALIAS adb)

target_include_directories(adb INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(adb INTERFACE cxx_std_23)

option(ADB_ENABLE_TESTS "Build the adb concept self-checks" ${ADB_STANDALONE})
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND BUILD_TESTS)
    set(ADB_ENABLE_TESTS ON)
endif()
if(ADB_ENABLE_TESTS)
    enable_testing()
    add_executable(adb-concept-check tests/concept_check.cpp)
    target_link_libraries(adb-concept-check PRIVATE adb)
    add_test(NAME adb-concept-check COMMAND adb-concept-check)
endif()

# psitri — persistent transactional key-value engines over the adb concept.
#
# Structure (docs/reorg_plan.md):
#   libraries/adb           abstract database concepts
#   libraries/sal           smart alloc (movable memory, persistent storage)
#   libraries/start_ref     static ART, refcount liveness
#   libraries/start_ver     static ART, version/stamp liveness
#   libraries/sal_art_ref   ART on sal
#   libraries/psitri        legacy sal btree
#   libraries/delta_art     two adb engines composed (hot delta -> base)
#   shims/psi-*             foreign-API products over our engines
#   vendor/{psio,ucc}       FetchContent deps (co-dev override pattern)
#   tests/ benchmarks/      written against adb, instantiated per engine
cmake_minimum_required(VERSION 3.20)
project(psitri VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# When embedded as a subdirectory (e.g. Silkworm's psitrimdbx backend), default
# the dev-only targets OFF so the consumer gets just the libraries it links.
# (PROJECT_IS_TOP_LEVEL needs CMake 3.21; compute it portably for 3.20.)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PSITRI_IS_TOP ON)
else()
    set(PSITRI_IS_TOP OFF)
endif()

option(BUILD_TESTS "Build test suites" ${PSITRI_IS_TOP})
option(BUILD_BENCHMARKS "Build benchmarks" ${PSITRI_IS_TOP})
if(BUILD_TESTS)
    enable_testing()
endif()

# ── vendored deps: psio + ucc (dependency-pattern §6b, co-dev defaults) ──────
include(FetchContent)
if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_PSIO AND EXISTS "$ENV{HOME}/gofractally/psio/cpp/CMakeLists.txt")
    set(FETCHCONTENT_SOURCE_DIR_PSIO "$ENV{HOME}/gofractally/psio")
    message(STATUS "psio: co-developing against ${FETCHCONTENT_SOURCE_DIR_PSIO}")
endif()
FetchContent_Declare(psio
    GIT_REPOSITORY https://github.com/gofractally/psio.git
    GIT_TAG        main
    SOURCE_DIR     ${CMAKE_SOURCE_DIR}/vendor/psio
    SOURCE_SUBDIR  cpp)
if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_UCC AND EXISTS "$ENV{HOME}/gofractally/ucc/CMakeLists.txt")
    set(FETCHCONTENT_SOURCE_DIR_UCC "$ENV{HOME}/gofractally/ucc")
    message(STATUS "ucc: co-developing against ${FETCHCONTENT_SOURCE_DIR_UCC}")
endif()
FetchContent_Declare(ucc
    GIT_REPOSITORY https://github.com/gofractally/ucc.git
    GIT_TAG        main
    SOURCE_DIR     ${CMAKE_SOURCE_DIR}/vendor/ucc)
FetchContent_MakeAvailable(psio ucc)

# ── libraries, in dependency order (added as each one migrates) ──────────────
add_subdirectory(libraries/adb)
set(SAL_BUILD_TESTS ${BUILD_TESTS})
add_subdirectory(libraries/sal)
add_subdirectory(libraries/sal2)
add_subdirectory(libraries/gcart)
add_subdirectory(libraries/art)
add_subdirectory(libraries/start_ref)
add_subdirectory(libraries/start_ver)
set(SAL_ART_REF_BUILD_TESTS ${BUILD_TESTS})
add_subdirectory(libraries/sal_art_ref)
add_subdirectory(libraries/psitri)
add_subdirectory(libraries/delta_art)
add_subdirectory(libraries/walart)
add_subdirectory(libraries/psitrimdbx)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

get_filename_component(NURI_CXX_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)

if(DEFINED VCPKG_TARGET_TRIPLET AND NOT "${VCPKG_TARGET_TRIPLET}" STREQUAL "")
  set(_nuri_vcpkg_bin_dirs
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/bin"
  )
else()
  file(GLOB _nuri_vcpkg_bin_dirs
    LIST_DIRECTORIES true
    "${CMAKE_BINARY_DIR}/vcpkg_installed/*/bin"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/*/debug/bin"
  )
endif()

set(_nuri_test_path
  "${CMAKE_BINARY_DIR}/lib"
  ${_nuri_vcpkg_bin_dirs}
  "${NURI_CXX_COMPILER_BIN_DIR}"
  "$ENV{PATH}"
)
list(JOIN _nuri_test_path ";" _nuri_test_path_joined)
string(REPLACE ";" "\\;" _nuri_test_path_escaped "${_nuri_test_path_joined}")

if(WIN32)
  set(NURI_GTEST_LAUNCHER "${CMAKE_CURRENT_BINARY_DIR}/nuri_gtest_launcher.bat")
  file(GENERATE OUTPUT "${NURI_GTEST_LAUNCHER}" CONTENT
"@echo off
setlocal
set \"PATH=${_nuri_test_path_joined}\"
set \"TEST_EXE=%~1\"
shift
\"%TEST_EXE%\" %*
"
  )
else()
  set(NURI_GTEST_LAUNCHER "${CMAKE_CURRENT_BINARY_DIR}/nuri_gtest_launcher.sh")
  file(GENERATE OUTPUT "${NURI_GTEST_LAUNCHER}"
    CONTENT
"#!/usr/bin/env bash
set -euo pipefail
export PATH='${_nuri_test_path_joined}'
exec \"$@\"
"
    FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
endif()

function(nuri_add_gtest_suite target source_file test_prefix)
  cmake_parse_arguments(NURI_TEST "" "" "SUPPORT_FILES" ${ARGN})

  if(DEFINED NURI_TEST_SUPPORT_FILES)
    set(_nuri_test_support_files ${NURI_TEST_SUPPORT_FILES})
  else()
    set(_nuri_test_support_files src/render_graph_test_support.cpp)
  endif()

  add_executable(${target}
      ${source_file}
      ${_nuri_test_support_files}
  )

  target_link_libraries(${target}
      PRIVATE
      nuri_renderer
      GTest::gtest
      GTest::gtest_main
  )

  target_include_directories(${target}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    TEST_LAUNCHER "${NURI_GTEST_LAUNCHER}"
  )

  foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${_cfg}" _cfg_upper)
    set_target_properties(${target} PROPERTIES
      "RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
    )
  endforeach()

  gtest_discover_tests(${target}
    TEST_PREFIX "${test_prefix}"
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PROPERTIES
      ENVIRONMENT "PATH=${_nuri_test_path_escaped}"
  )
endfunction()

nuri_add_gtest_suite(
  nuri_render_graph_metadata_tests
  src/render_graph_metadata_tests.cpp
  "metadata::"
)

nuri_add_gtest_suite(
  nuri_render_graph_compile_behavior_tests
  src/render_graph_compile_behavior_tests.cpp
  "compile_behavior::"
)

nuri_add_gtest_suite(
  nuri_render_graph_executor_tests
  src/render_graph_executor_tests.cpp
  "executor::"
)

nuri_add_gtest_suite(
  nuri_render_graph_renderer_tests
  src/render_graph_renderer_tests.cpp
  "renderer::"
)

nuri_add_gtest_suite(
  nuri_render_graph_telemetry_tests
  src/render_graph_telemetry_tests.cpp
  "telemetry::"
)

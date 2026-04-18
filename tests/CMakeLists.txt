find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

set(NURI_TEST_PCH_HEADER
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tests_pch.h"
)

get_filename_component(NURI_CXX_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(NURI_CXX_COMPILER_ROOT "${NURI_CXX_COMPILER_BIN_DIR}" DIRECTORY)

set(_nuri_clang_runtime_dirs)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  string(REGEX MATCH "^[0-9]+" _nuri_clang_version_major
    "${CMAKE_CXX_COMPILER_VERSION}"
  )
  list(APPEND _nuri_clang_runtime_dirs
    "${NURI_CXX_COMPILER_ROOT}/lib/clang/${_nuri_clang_version_major}/lib/windows"
    "${NURI_CXX_COMPILER_ROOT}/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/windows"
  )
endif()

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
  ${_nuri_clang_runtime_dirs}
  ${_nuri_vcpkg_bin_dirs}
  "${NURI_CXX_COMPILER_BIN_DIR}"
  "$ENV{PATH}"
)
list(JOIN _nuri_test_path ";" _nuri_test_path_joined)

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

add_library(nuri_render_graph_test_support STATIC
  src/render_graph_test_support.cpp
)

target_link_libraries(nuri_render_graph_test_support
  PUBLIC
    nuri_renderer
    GTest::gtest
)

target_include_directories(nuri_render_graph_test_support
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_precompile_headers(nuri_render_graph_test_support PRIVATE
  "${NURI_TEST_PCH_HEADER}"
)

set_target_properties(nuri_render_graph_test_support PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${_cfg}" _cfg_upper)
  set_target_properties(nuri_render_graph_test_support PROPERTIES
    "ARCHIVE_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
    "LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
    "RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
  )
endforeach()

function(nuri_add_gtest_suite target source_file test_prefix)
  add_executable(${target}
    ${source_file}
  )

  target_link_libraries(${target}
    PRIVATE
      nuri_render_graph_test_support
      GTest::gtest_main
  )

  target_precompile_headers(${target} REUSE_FROM nuri_render_graph_test_support)

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
  )
endfunction()

function(nuri_add_gtest_suite_with_sources target test_prefix)
  add_executable(${target}
    ${ARGN}
  )

  target_link_libraries(${target}
    PRIVATE
      nuri_render_graph_test_support
      GTest::gtest_main
  )

  target_include_directories(${target}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/editor/include
  )

  target_precompile_headers(${target} REUSE_FROM nuri_render_graph_test_support)

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
  nuri_transmission_renderer_tests
  src/transmission_renderer_tests.cpp
  "transmission_renderer::"
)

nuri_add_gtest_suite(
  nuri_render_graph_telemetry_tests
  src/render_graph_telemetry_tests.cpp
  "telemetry::"
)

nuri_add_gtest_suite(
  nuri_material_import_tests
  src/material_import_tests.cpp
  "material_import::"
)

nuri_add_gtest_suite(
  nuri_render_scene_light_store_tests
  src/render_scene_light_store_tests.cpp
  "render_scene_light_store::"
)

nuri_add_gtest_suite(
  nuri_render_scene_graph_tests
  src/render_scene_graph_tests.cpp
  "render_scene_graph::"
)

nuri_add_gtest_suite(
  nuri_slot_pool_tests
  src/slot_pool_tests.cpp
  "slot_pool::"
)

nuri_add_gtest_suite(
  nuri_geometry_pool_tests
  src/geometry_pool_tests.cpp
  "geometry_pool::"
)

nuri_add_gtest_suite(
  nuri_renderable_material_resolution_tests
  src/renderable_material_resolution_tests.cpp
  "renderable_material_resolution::"
)

nuri_add_gtest_suite(
  nuri_gltf_light_import_tests
  src/gltf_light_import_tests.cpp
  "gltf_light_import::"
)

nuri_add_gtest_suite(
  nuri_gltf_scene_prefab_import_tests
  src/gltf_scene_prefab_import_tests.cpp
  "gltf_scene_prefab_import::"
)

nuri_add_gtest_suite(
  nuri_material_binary_cache_tests
  src/material_binary_cache_tests.cpp
  "material_binary_cache::"
)

nuri_add_gtest_suite(
  nuri_texture_cache_utils_tests
  src/texture_cache_utils_tests.cpp
  "texture_cache_utils::"
)

nuri_add_gtest_suite(
  nuri_animation_pose_simulation_tests
  src/animation_pose_simulation_tests.cpp
  "animation_pose::"
)

nuri_add_gtest_suite_with_sources(
  nuri_editor_animation_player_service_tests
  "editor_animation_player::"
  src/editor_animation_player_service_tests.cpp
  ${CMAKE_SOURCE_DIR}/editor/src/app/editor_animation_player_service.cpp
)

nuri_add_gtest_suite(
  nuri_scene_animation_tests
  src/scene_animation_tests.cpp
  "scene_animation::"
)

nuri_add_gtest_suite(
  nuri_simulation_runtime_tests
  src/simulation_runtime_tests.cpp
  "simulation_runtime::"
)

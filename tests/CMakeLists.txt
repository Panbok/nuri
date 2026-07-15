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

function(nuri_test_properties target out_labels out_timeout)
  set(_labels "contract")
  set(_timeout 60)
  if("${target}" MATCHES "(tool_core|benchmark|snapshot|autotest)")
    string(APPEND _labels "+tool-core")
  elseif("${target}" MATCHES "(import|cache)")
    string(APPEND _labels "+asset")
    set(_timeout 120)
  elseif("${target}" MATCHES "(render|visibility|transmission|geometry)")
    string(APPEND _labels "+renderer")
  endif()
  set(${out_labels} "${_labels}" PARENT_SCOPE)
  set(${out_timeout} "${_timeout}" PARENT_SCOPE)
endfunction()

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

  nuri_test_properties(${target} _nuri_labels _nuri_timeout)
  gtest_discover_tests(${target}
    TEST_PREFIX "${test_prefix}"
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PROPERTIES
      LABELS "${_nuri_labels}"
      TIMEOUT "${_nuri_timeout}"
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

  nuri_test_properties(${target} _nuri_labels _nuri_timeout)
  gtest_discover_tests(${target}
    TEST_PREFIX "${test_prefix}"
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PROPERTIES
      LABELS "${_nuri_labels}"
      TIMEOUT "${_nuri_timeout}"
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
if(TARGET LVKLibrary)
  target_link_libraries(nuri_render_graph_renderer_tests PRIVATE LVKLibrary)
endif()

nuri_add_gtest_suite(
  nuri_raster_pipeline_state_tests
  src/raster_pipeline_state_tests.cpp
  "raster_pipeline_state::"
)

nuri_add_gtest_suite(
  nuri_render_temporal_frame_service_tests
  src/temporal_frame_service_tests.cpp
  "temporal_service::"
)

nuri_add_gtest_suite(
  nuri_external_temporal_provider_tests
  src/external_temporal_provider_tests.cpp
  "external_temporal_provider::"
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
  nuri_visibility_math_tests
  src/visibility_math_tests.cpp
  "visibility::"
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
  nuri_dynamic_buffer_tests
  src/dynamic_buffer_tests.cpp
  "dynamic_buffer::"
)

nuri_add_gtest_suite(
  nuri_core_primitives_tests
  src/core_primitives_tests.cpp
  "core_primitives::"
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
  nuri_mesh_binary_cache_tests
  src/mesh_binary_cache_tests.cpp
  "mesh_binary_cache::"
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

if(NURI_BUILD_TOOLS)
  nuri_add_gtest_suite(
    nuri_tool_core_baseline_profile_tests
    src/nuri_baseline_profile_tests.cpp
    "baseline_profile::"
  )
  target_link_libraries(nuri_tool_core_baseline_profile_tests
    PRIVATE
      nuri_tool_core
  )

  nuri_add_gtest_suite(
    nuri_tool_core_tests
    src/nuri_tool_core_tests.cpp
    "tool_core::"
  )
  target_link_libraries(nuri_tool_core_tests PRIVATE nuri_tool_core)

  add_executable(nuri_process_test_helper
    helpers/nuri_process_test_helper.cpp
  )
  find_package(Threads REQUIRED)
  target_link_libraries(nuri_process_test_helper PRIVATE Threads::Threads)
  set_target_properties(nuri_process_test_helper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${_cfg}" _cfg_upper)
    set_target_properties(nuri_process_test_helper PROPERTIES
      "RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
    )
  endforeach()

  nuri_add_gtest_suite(
    nuri_tool_core_process_tests
    src/nuri_process_tests.cpp
    "process::"
  )
  target_link_libraries(nuri_tool_core_process_tests PRIVATE nuri_tool_core)
  file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/nuri_process_test_helper_path.h"
    CONTENT "#pragma once\n#define NURI_PROCESS_TEST_HELPER_PATH R\"nuri($<TARGET_FILE:nuri_process_test_helper>)nuri\"\n"
  )
  target_include_directories(nuri_tool_core_process_tests PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
  )
  add_dependencies(nuri_tool_core_process_tests nuri_process_test_helper)

  nuri_add_gtest_suite(
    nuri_tool_core_result_envelope_v2_tests
    src/nuri_result_envelope_v2_tests.cpp
    "result_envelope_v2::"
  )
  target_link_libraries(nuri_tool_core_result_envelope_v2_tests
    PRIVATE
      nuri_tool_core
  )

  add_executable(nuri_benchmark_child_fixture
    helpers/nuri_benchmark_child_fixture.cpp
  )
  target_link_libraries(nuri_benchmark_child_fixture
    PRIVATE
      nuri_benchmarking
      Threads::Threads
  )
  set_target_properties(nuri_benchmark_child_fixture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${_cfg}" _cfg_upper)
    set_target_properties(nuri_benchmark_child_fixture PROPERTIES
      "RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
    )
  endforeach()

  nuri_add_gtest_suite(
    nuri_benchmarking_tests
    src/nuri_benchmarking_tests.cpp
    "benchmarking::"
  )
  target_link_libraries(nuri_benchmarking_tests PRIVATE nuri_benchmarking)
  file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/nuri_benchmark_child_fixture_path.h"
    CONTENT "#pragma once\n#define NURI_BENCHMARK_CHILD_FIXTURE_PATH R\"nuri($<TARGET_FILE:nuri_benchmark_child_fixture>)nuri\"\n"
  )
  target_include_directories(nuri_benchmarking_tests PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>"
  )
  add_dependencies(nuri_benchmarking_tests nuri_benchmark_child_fixture)

  nuri_add_gtest_suite(
    nuri_benchmark_metric_registry_tests
    src/nuri_benchmark_metric_registry_tests.cpp
    "benchmark_metric_registry::"
  )
  target_link_libraries(nuri_benchmark_metric_registry_tests
    PRIVATE
      nuri_benchmarking
  )

  nuri_add_gtest_suite(
    nuri_benchmark_stats_tests
    src/nuri_benchmark_stats_tests.cpp
    "benchmark_stats::"
  )
  target_link_libraries(nuri_benchmark_stats_tests PRIVATE nuri_benchmarking)

  if(TARGET nuri_snapshot_testing)
    nuri_add_gtest_suite(
      nuri_snapshot_testing_tests
      src/nuri_snapshot_testing_tests.cpp
      "snapshot_testing::"
    )
    target_link_libraries(nuri_snapshot_testing_tests
      PRIVATE
        nuri_snapshot_testing
    )
    nuri_add_gtest_suite(
      nuri_snapshot_semantic_tests
      src/nuri_snapshot_semantic_tests.cpp
      "snapshot_semantic::"
    )
    target_link_libraries(nuri_snapshot_semantic_tests
      PRIVATE
        nuri_snapshot_testing
    )
  endif()
  if(TARGET nuri_autotesting)
    nuri_add_gtest_suite(
      nuri_autotesting_tests
      src/nuri_autotesting_tests.cpp
      "autotesting::"
    )
    target_link_libraries(nuri_autotesting_tests
      PRIVATE
        nuri_autotesting
    )
    nuri_add_gtest_suite(
      nuri_autotest_motion_oracle_tests
      src/autotest_motion_oracle_tests.cpp
      "autotest_motion_oracle::"
    )
    target_link_libraries(nuri_autotest_motion_oracle_tests
      PRIVATE
        nuri_autotesting
    )
    nuri_add_gtest_suite(
      nuri_autotest_quality_oracle_tests
      src/autotest_quality_oracle_tests.cpp
      "autotest_quality_oracle::"
    )
    target_link_libraries(nuri_autotest_quality_oracle_tests
      PRIVATE
        nuri_autotesting
    )
  endif()
endif()

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
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin"
  )
else()
  file(GLOB _nuri_vcpkg_bin_dirs
    LIST_DIRECTORIES true
    "${VCPKG_INSTALLED_DIR}/*/bin"
    "${VCPKG_INSTALLED_DIR}/*/debug/bin"
  )
endif()

set(_nuri_test_path
  "${CMAKE_BINARY_DIR}/out/$<CONFIG>/bin"
  "${CMAKE_BINARY_DIR}/out/$<CONFIG>/lib"
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

add_library(nuri_test_base STATIC
  src/test_base.cpp
)
nuri_configure_first_party_target(nuri_test_base)
nuri_register_artifact_target(nuri_test_base)
target_link_libraries(nuri_test_base
  PUBLIC
    GTest::gtest
    nuri::source_root
)
target_include_directories(nuri_test_base
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib
)
target_precompile_headers(nuri_test_base PRIVATE
  "${NURI_TEST_PCH_HEADER}"
)

add_library(nuri_render_graph_test_support STATIC
  src/render_graph_test_support.cpp
)
nuri_configure_first_party_target(nuri_render_graph_test_support)
nuri_register_artifact_target(nuri_render_graph_test_support)

target_link_libraries(nuri_render_graph_test_support
  PUBLIC
    nuri_renderer
    nuri_test_base
)

target_include_directories(nuri_render_graph_test_support
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_precompile_headers(nuri_render_graph_test_support PRIVATE
  "${NURI_TEST_PCH_HEADER}"
)

set(NURI_RENDERER_TEST_TARGETS)
set(NURI_CORE_TEST_TARGETS)
set(NURI_TOOL_CORE_TEST_TARGETS)
set(NURI_BENCHMARK_TEST_TARGETS)
set(NURI_SNAPSHOT_TEST_TARGETS)
set(NURI_AUTOTEST_TEST_TARGETS)

function(nuri_test_properties target out_labels out_timeout)
  set(_labels "contract")
  set(_timeout 60)
  if("${target}" MATCHES "benchmark")
    string(APPEND _labels "+tool-core+benchmark")
  elseif("${target}" MATCHES "snapshot")
    string(APPEND _labels "+tool-core+snapshot")
  elseif("${target}" MATCHES "autotest")
    string(APPEND _labels "+tool-core+autotest")
  elseif("${target}" MATCHES "tool_core")
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
  nuri_configure_first_party_target(${target})
  nuri_register_artifact_target(${target})

  target_link_libraries(${target}
    PRIVATE
      nuri_render_graph_test_support
      GTest::gtest_main
  )

  target_precompile_headers(${target} REUSE_FROM nuri_render_graph_test_support)

  set_target_properties(${target} PROPERTIES
    TEST_LAUNCHER "${NURI_GTEST_LAUNCHER}"
  )

  nuri_test_properties(${target} _nuri_labels _nuri_timeout)
  gtest_discover_tests(${target}
    TEST_PREFIX "${test_prefix}"
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/out/$<CONFIG>/bin"
    PROPERTIES
      LABELS "${_nuri_labels}"
      TIMEOUT "${_nuri_timeout}"
      ENVIRONMENT "NURI_RENDER_GRAPH_WORKER_COUNT=1"
  )
  list(APPEND NURI_RENDERER_TEST_TARGETS ${target})
  set(NURI_RENDERER_TEST_TARGETS "${NURI_RENDERER_TEST_TARGETS}" PARENT_SCOPE)
endfunction()

function(nuri_add_core_gtest_suite target source_file test_prefix)
  add_executable(${target} ${source_file})
  nuri_configure_first_party_target(${target})
  nuri_register_artifact_target(${target})
  target_link_libraries(${target} PRIVATE nuri_test_base GTest::gtest_main)
  target_precompile_headers(${target} REUSE_FROM nuri_test_base)
  set_target_properties(${target} PROPERTIES
    TEST_LAUNCHER "${NURI_GTEST_LAUNCHER}"
  )
  nuri_test_properties(${target} _nuri_labels _nuri_timeout)
  gtest_discover_tests(${target}
    TEST_PREFIX "${test_prefix}"
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/out/$<CONFIG>/bin"
    PROPERTIES
      LABELS "${_nuri_labels}"
      TIMEOUT "${_nuri_timeout}"
      ENVIRONMENT "NURI_RENDER_GRAPH_WORKER_COUNT=1"
  )
  list(APPEND NURI_CORE_TEST_TARGETS ${target})
  set(NURI_CORE_TEST_TARGETS "${NURI_CORE_TEST_TARGETS}" PARENT_SCOPE)
endfunction()

function(nuri_add_gtest_suite_with_sources target test_prefix)
  add_executable(${target}
    ${ARGN}
  )
  nuri_configure_first_party_target(${target})
  nuri_register_artifact_target(${target})

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
    TEST_LAUNCHER "${NURI_GTEST_LAUNCHER}"
  )

  nuri_test_properties(${target} _nuri_labels _nuri_timeout)
  gtest_discover_tests(${target}
    TEST_PREFIX "${test_prefix}"
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/out/$<CONFIG>/bin"
    PROPERTIES
      LABELS "${_nuri_labels}"
      TIMEOUT "${_nuri_timeout}"
      ENVIRONMENT "NURI_RENDER_GRAPH_WORKER_COUNT=1"
  )
  list(APPEND NURI_RENDERER_TEST_TARGETS ${target})
  set(NURI_RENDERER_TEST_TARGETS "${NURI_RENDERER_TEST_TARGETS}" PARENT_SCOPE)
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
  nuri_render_scene_ddgi_bounds_tests
  src/render_scene_ddgi_bounds_tests.cpp
  "render_scene_ddgi_bounds::"
)

nuri_add_gtest_suite(
  nuri_ddgi_atlas_tests
  src/ddgi_atlas_tests.cpp
  "ddgi_atlas::"
)

nuri_add_gtest_suite(
  nuri_ddgi_coverage_tests
  src/ddgi_coverage_tests.cpp
  "ddgi_coverage::"
)

nuri_add_gtest_suite(
  nuri_ddgi_dirty_regions_tests
  src/ddgi_dirty_regions_tests.cpp
  "ddgi_dirty_regions::"
)

nuri_add_gtest_suite(
  nuri_ddgi_scheduler_tests
  src/ddgi_scheduler_tests.cpp
  "ddgi_scheduler::"
)

nuri_add_gtest_suite(
  nuri_ddgi_volume_store_tests
  src/ddgi_volume_store_tests.cpp
  "ddgi_volume_store::"
)

nuri_add_gtest_suite(
  nuri_ddgi_settings_tests
  src/ddgi_settings_tests.cpp
  "ddgi_settings::"
)

nuri_add_gtest_suite(
  nuri_acceleration_structure_contract_tests
  src/acceleration_structure_contract_tests.cpp
  "acceleration_structure::"
)

nuri_add_gtest_suite(
  nuri_render_graph_acceleration_structure_tests
  src/render_graph_acceleration_structure_tests.cpp
  "render_graph_acceleration_structure::"
)

nuri_add_core_gtest_suite(
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
  nuri_asset_cpu_scheduler_tests
  src/asset_cpu_scheduler_tests.cpp
  "asset_cpu_scheduler::"
)

nuri_add_gtest_suite(
  nuri_asset_system_tests
  src/asset_system_tests.cpp
  "asset_system::"
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
target_compile_definitions(nuri_texture_cache_utils_tests
  PRIVATE
    KHRONOS_STATIC
    KTX_FEATURE_KTX1
    KTX_FEATURE_KTX2
    KTX_FEATURE_WRITE
)
if(TARGET KTX::ktx)
  target_link_libraries(nuri_texture_cache_utils_tests PRIVATE KTX::ktx)
elseif(TARGET ktx)
  target_link_libraries(nuri_texture_cache_utils_tests PRIVATE ktx)
endif()
set_source_files_properties(src/texture_cache_utils_tests.cpp PROPERTIES
  SKIP_PRECOMPILE_HEADERS ON
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
  nuri_smaa_lut_baker_tests
  src/smaa_lut_baker_tests.cpp
  "smaa_lut_baker::"
)
target_link_libraries(nuri_smaa_lut_baker_tests PRIVATE nuri_smaa_bakery)

nuri_add_gtest_suite(
  nuri_spatial_aa_renderer_tests
  src/spatial_aa_feature_tests.cpp
  "spatial_aa_renderer::"
)
target_link_libraries(nuri_spatial_aa_renderer_tests PRIVATE nuri_smaa_bakery)

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
  nuri_add_core_gtest_suite(
    nuri_tool_core_baseline_profile_tests
    src/nuri_baseline_profile_tests.cpp
    "baseline_profile::"
  )
  target_link_libraries(nuri_tool_core_baseline_profile_tests
    PRIVATE
      nuri_tool_core
  )

  nuri_add_core_gtest_suite(
    nuri_tool_core_tests
    src/nuri_tool_core_tests.cpp
    "tool_core::"
  )
  target_link_libraries(nuri_tool_core_tests PRIVATE nuri_tool_core)

  add_executable(nuri_process_test_helper
    helpers/nuri_process_test_helper.cpp
  )
  nuri_configure_first_party_target(nuri_process_test_helper)
  nuri_register_artifact_target(nuri_process_test_helper)
  find_package(Threads REQUIRED)
  target_link_libraries(nuri_process_test_helper PRIVATE Threads::Threads)

  nuri_add_core_gtest_suite(
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

  nuri_add_core_gtest_suite(
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
  nuri_configure_first_party_target(nuri_benchmark_child_fixture)
  nuri_register_artifact_target(nuri_benchmark_child_fixture)
  target_link_libraries(nuri_benchmark_child_fixture
    PRIVATE
      nuri_benchmarking
      Threads::Threads
  )
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

if(NURI_BUILD_TOOLS)
  set(NURI_TOOL_CORE_TEST_TARGETS
    nuri_tool_core_baseline_profile_tests
    nuri_tool_core_tests
    nuri_tool_core_process_tests
    nuri_tool_core_result_envelope_v2_tests
  )
  set(NURI_BENCHMARK_TEST_TARGETS
    nuri_benchmarking_tests
    nuri_benchmark_metric_registry_tests
    nuri_benchmark_stats_tests
  )
  if(TARGET nuri_snapshot_testing_tests)
    set(NURI_SNAPSHOT_TEST_TARGETS
      nuri_snapshot_testing_tests
      nuri_snapshot_semantic_tests
    )
  endif()
  if(TARGET nuri_autotesting_tests)
    set(NURI_AUTOTEST_TEST_TARGETS
      nuri_autotesting_tests
      nuri_autotest_motion_oracle_tests
      nuri_autotest_quality_oracle_tests
    )
  endif()
  list(REMOVE_ITEM NURI_CORE_TEST_TARGETS ${NURI_TOOL_CORE_TEST_TARGETS})
  list(REMOVE_ITEM NURI_RENDERER_TEST_TARGETS
    ${NURI_BENCHMARK_TEST_TARGETS}
    ${NURI_SNAPSHOT_TEST_TARGETS}
    ${NURI_AUTOTEST_TEST_TARGETS}
  )
endif()

add_custom_target(nuri-tests-core DEPENDS ${NURI_CORE_TEST_TARGETS})
add_custom_target(nuri-tests-renderer-contract
  DEPENDS ${NURI_RENDERER_TEST_TARGETS}
)
add_custom_target(nuri-tests-tool-core DEPENDS ${NURI_TOOL_CORE_TEST_TARGETS})
add_custom_target(nuri-tests-benchmark DEPENDS ${NURI_BENCHMARK_TEST_TARGETS})
add_custom_target(nuri-tests-snapshot DEPENDS ${NURI_SNAPSHOT_TEST_TARGETS})
add_custom_target(nuri-tests-autotest DEPENDS ${NURI_AUTOTEST_TEST_TARGETS})
add_custom_target(nuri-tests-contract
  DEPENDS
    nuri-tests-core
    nuri-tests-renderer-contract
    nuri-tests-tool-core
    nuri-tests-benchmark
    nuri-tests-snapshot
    nuri-tests-autotest
)

option(NURI_BUILD_BENCHMARK_CLI "Build nuri-bench executable" ON)
option(NURI_BUILD_SNAPSHOT_CLI "Build nuri-snapshot executable" ON)
option(NURI_BUILD_SNAPSHOT_TESTING "Build nuri snapshot testing library" ${NURI_BUILD_SNAPSHOT_CLI})
option(NURI_BUILD_AUTOTEST_CLI "Build nuri-autotest executable" ON)
option(NURI_BUILD_AUTOTESTING "Build nuri autotesting library" ${NURI_BUILD_AUTOTEST_CLI})

if(NOT DEFINED NURI_DEV_CHECKS)
  set(NURI_DEV_CHECKS OFF)
endif()

set(NURI_VCPKG_MANIFEST_FEATURES "${VCPKG_MANIFEST_FEATURES}")

foreach(_nuri_bool IN ITEMS
  NURI_BUILD_SHARED
  NURI_WITH_LOGGING
  NURI_WITH_ASSERTS
  NURI_WITH_TRACY
  NURI_DEV_CHECKS
)
  if(${_nuri_bool})
    set(${_nuri_bool}_VALUE 1)
  else()
    set(${_nuri_bool}_VALUE 0)
  endif()
endforeach()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/cmake/benchmark_build_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/nuri/tools/benchmark/build_config.h"
  @ONLY
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/snapshot/cmake/snapshot_build_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/nuri/tools/snapshot/build_config.h"
  @ONLY
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/autotest/cmake/autotest_build_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/nuri/tools/autotest/build_config.h"
  @ONLY
)

find_package(yyjson CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(nuri_tool_core STATIC
  core/src/atomic_file.cpp
  core/src/baseline_profile.cpp
  core/src/case_catalog.cpp
  core/src/environment_probe.cpp
  core/src/fingerprint.cpp
  core/src/html_report.cpp
  core/src/identifier.cpp
  core/src/json_contract.cpp
  core/src/process.cpp
  core/src/result_envelope_v2.cpp
  core/src/result_protocol.cpp
  core/src/run_workspace.cpp
  core/src/safe_path.cpp
  core/src/sha256.cpp
)
nuri_configure_first_party_target(nuri_tool_core)
nuri_register_artifact_target(nuri_tool_core)

target_include_directories(nuri_tool_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/core/include"
    "${CMAKE_SOURCE_DIR}/lib"
)

target_compile_features(nuri_tool_core PUBLIC cxx_std_20)
target_link_libraries(nuri_tool_core PRIVATE Threads::Threads)

if(TARGET yyjson::yyjson)
  target_link_libraries(nuri_tool_core PRIVATE yyjson::yyjson)
elseif(TARGET yyjson)
  target_link_libraries(nuri_tool_core PRIVATE yyjson)
elseif(TARGET unofficial::yyjson::yyjson)
  target_link_libraries(nuri_tool_core PRIVATE unofficial::yyjson::yyjson)
else()
  message(FATAL_ERROR "nuri_tool_core requires a yyjson package target")
endif()

add_library(nuri_tools_runtime STATIC
  runtime/src/tools_runtime.cpp
)
nuri_configure_first_party_target(nuri_tools_runtime)
nuri_register_artifact_target(nuri_tools_runtime)

target_include_directories(nuri_tools_runtime
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/include"
)

target_link_libraries(nuri_tools_runtime
  PUBLIC
    nuri_renderer
)

target_compile_features(nuri_tools_runtime PUBLIC cxx_std_20)

add_library(nuri_benchmarking STATIC
  benchmark/src/benchmark_baseline.cpp
  benchmark/src/benchmark_case.cpp
  benchmark/src/benchmark_check.cpp
  benchmark/src/benchmark_compare.cpp
  benchmark/src/benchmark_environment.cpp
  benchmark/src/benchmark_graph.cpp
  benchmark/src/benchmark_manifest.cpp
  benchmark/src/benchmark_measurements.cpp
  benchmark/src/benchmark_metric_registry.cpp
  benchmark/src/benchmark_report.cpp
  benchmark/src/benchmark_runner.cpp
  benchmark/src/benchmark_stats.cpp
)
nuri_configure_first_party_target(nuri_benchmarking)
nuri_register_artifact_target(nuri_benchmarking)

target_include_directories(nuri_benchmarking
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/include"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(nuri_benchmarking
  PUBLIC
    nuri_renderer
    nuri_tool_core
  PRIVATE
    nuri_tools_runtime
    nuri::source_root
)

target_compile_features(nuri_benchmarking PUBLIC cxx_std_20)

if(NURI_BUILD_SNAPSHOT_TESTING OR NURI_BUILD_SNAPSHOT_CLI OR NURI_BUILD_AUTOTESTING OR NURI_BUILD_AUTOTEST_CLI)
  add_library(nuri_snapshot_testing STATIC
    snapshot/src/snapshot_baseline.cpp
    snapshot/src/snapshot_capture_artifacts.cpp
    snapshot/src/snapshot_capture_point.cpp
    snapshot/src/snapshot_case.cpp
    snapshot/src/snapshot_compare.cpp
    snapshot/src/snapshot_environment.cpp
    snapshot/src/snapshot_html_report.cpp
    snapshot/src/snapshot_image.cpp
    snapshot/src/snapshot_manifest.cpp
    snapshot/src/snapshot_report.cpp
    snapshot/src/snapshot_runner.cpp
  )
  nuri_configure_first_party_target(nuri_snapshot_testing)
  nuri_register_artifact_target(nuri_snapshot_testing)

  find_package(OpenImageIO CONFIG REQUIRED)

  target_include_directories(nuri_snapshot_testing
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/snapshot/include"
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )

  target_link_libraries(nuri_snapshot_testing
    PUBLIC
      nuri_renderer
      nuri_tool_core
    PRIVATE
      nuri_tools_runtime
      nuri::source_root
      OpenImageIO::OpenImageIO
  )

  target_compile_features(nuri_snapshot_testing PUBLIC cxx_std_20)
endif()

if(NURI_BUILD_AUTOTESTING OR NURI_BUILD_AUTOTEST_CLI)
  add_library(nuri_autotesting STATIC
    autotest/src/autotest_assertion.cpp
    autotest/src/autotest_case.cpp
    autotest/src/autotest_environment.cpp
    autotest/src/autotest_manifest.cpp
    autotest/src/autotest_motion_oracle.cpp
    autotest/src/autotest_quality_oracle.cpp
    autotest/src/autotest_record.cpp
    autotest/src/autotest_report.cpp
    autotest/src/autotest_runner.cpp
    autotest/src/autotest_timeline.cpp
  )
  nuri_configure_first_party_target(nuri_autotesting)
  nuri_register_artifact_target(nuri_autotesting)

  target_include_directories(nuri_autotesting
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/autotest/include"
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )

  target_link_libraries(nuri_autotesting
    PUBLIC
      nuri_renderer
      nuri_snapshot_testing
      nuri_tool_core
    PRIVATE
      nuri_tools_runtime
  )

  target_compile_features(nuri_autotesting PUBLIC cxx_std_20)
endif()

if(NURI_BUILD_BENCHMARK_CLI)
  find_package(CLI11 CONFIG REQUIRED)

  add_executable(nuri-bench
    cli/nuri_benchmark_main.cpp
  )
  nuri_configure_first_party_target(nuri-bench)
  nuri_register_artifact_target(nuri-bench)

  target_link_libraries(nuri-bench
    PRIVATE
      nuri_benchmarking
      CLI11::CLI11
  )

endif()

if(NURI_BUILD_SNAPSHOT_CLI)
  find_package(CLI11 CONFIG REQUIRED)

  add_executable(nuri-snapshot
    cli/nuri_snapshot_main.cpp
  )
  nuri_configure_first_party_target(nuri-snapshot)
  nuri_register_artifact_target(nuri-snapshot)

  target_link_libraries(nuri-snapshot
    PRIVATE
      nuri_snapshot_testing
      CLI11::CLI11
  )

endif()

if(NURI_BUILD_AUTOTEST_CLI)
  find_package(CLI11 CONFIG REQUIRED)

  add_executable(nuri-autotest
    cli/nuri_autotest_main.cpp
  )
  nuri_configure_first_party_target(nuri-autotest)
  nuri_register_artifact_target(nuri-autotest)

  target_link_libraries(nuri-autotest
    PRIVATE
      nuri_autotesting
      CLI11::CLI11
  )

endif()

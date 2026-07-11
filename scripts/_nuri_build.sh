#!/usr/bin/env bash
set -euo pipefail

usage="Usage: $(basename "$0") <debug|release> <lib|app|editor|tests|bench|bench-tests|snapshot|snapshot-tests|autotest|autotest-tests> [cpu|cpu-gpu|off] [devchecks]"

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "${usage}"
  exit 1
fi

mode="$1"
profile="$2"
tracy_mode=""
devchecks="OFF"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ "${mode}" != "debug" && "${mode}" != "release" ]]; then
  echo "${usage}"
  exit 1
fi

if [[ -z "${VCPKG_ROOT:-}" ]]; then
  echo "VCPKG_ROOT is not set. Point it at your vcpkg root."
  exit 1
fi

generator="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  generator="Ninja"
fi

manifest_features="${VCPKG_MANIFEST_FEATURES:-}"
append_manifest_feature() {
  local feature="$1"
  case ";${manifest_features};" in
    *";${feature};"*)
      ;;
    ";;")
      manifest_features="${feature}"
      ;;
    *)
      manifest_features="${manifest_features};${feature}"
      ;;
  esac
}

build_app="OFF"
build_editor="OFF"
build_tests="OFF"
build_tools="OFF"
build_benchmark_cli="ON"
build_snapshot_cli="ON"
build_snapshot_testing="OFF"
build_autotest_cli="OFF"
build_autotesting="OFF"
build_target=""
build_dir_suffix="${mode}"

case "${profile}" in
  lib)
    build_target="nuri_renderer"
    ;;
  app)
    build_app="ON"
    build_target="nuri"
    ;;
  editor)
    build_editor="ON"
    build_target="nuri_editor"
    append_manifest_feature editor
    ;;
  tests)
    build_tests="ON"
    append_manifest_feature tests
    ;;
  bench|benchmark)
    profile="bench"
    build_tools="ON"
    build_snapshot_cli="OFF"
    build_target="nuri-bench"
    build_dir_suffix="${mode}-bench"
    append_manifest_feature benchmark-tools
    ;;
  bench-tests|benchmark-tests)
    profile="bench-tests"
    build_tests="ON"
    build_tools="ON"
    build_benchmark_cli="OFF"
    build_snapshot_cli="OFF"
    build_dir_suffix="${mode}-bench-tests"
    append_manifest_feature benchmark-tools
    append_manifest_feature tests
    ;;
  snapshot|snapshots)
    profile="snapshot"
    build_tools="ON"
    build_benchmark_cli="OFF"
    build_snapshot_testing="ON"
    build_target="nuri-snapshot"
    build_dir_suffix="${mode}-snapshot"
    append_manifest_feature snapshot-tools
    ;;
  autotest|autotests)
    profile="autotest"
    build_tools="ON"
    build_benchmark_cli="OFF"
    build_snapshot_cli="OFF"
    build_snapshot_testing="ON"
    build_autotest_cli="ON"
    build_autotesting="ON"
    build_target="nuri-autotest"
    build_dir_suffix="${mode}-autotest"
    append_manifest_feature autotest-tools
    ;;
  snapshot-tests|snapshots-tests)
    profile="snapshot-tests"
    build_tests="ON"
    build_tools="ON"
    build_benchmark_cli="OFF"
    build_snapshot_cli="OFF"
    build_snapshot_testing="ON"
    build_dir_suffix="${mode}-snapshot-tests"
    append_manifest_feature snapshot-tools
    append_manifest_feature tests
    ;;
  autotest-tests|autotests-tests)
    profile="autotest-tests"
    build_tests="ON"
    build_tools="ON"
    build_benchmark_cli="OFF"
    build_snapshot_cli="OFF"
    build_snapshot_testing="ON"
    build_autotest_cli="OFF"
    build_autotesting="ON"
    build_dir_suffix="${mode}-autotest-tests"
    append_manifest_feature autotest-tools
    append_manifest_feature tests
    ;;
  *)
    echo "${usage}"
    exit 1
    ;;
esac

for arg in "${@:3}"; do
  case "${arg}" in
    cpu|cpu-gpu|off)
      if [[ -n "${tracy_mode}" ]]; then
        echo "${usage}"
        exit 1
      fi
      tracy_mode="${arg}"
      ;;
    devchecks)
      devchecks="ON"
      ;;
    *)
      echo "${usage}"
      exit 1
      ;;
  esac
done

nuri_with_asan="OFF"
nuri_with_logging="OFF"
nuri_with_asserts="OFF"
nuri_with_tracy="OFF"
nuri_with_tracy_gpu="OFF"
nuri_with_tracy_gpu_draw_zones="OFF"
nuri_with_fsr31="${NURI_WITH_FSR31:-OFF}"
if [[ "${nuri_with_fsr31}" != "ON" && "${nuri_with_fsr31}" != "OFF" ]]; then
  echo "NURI_WITH_FSR31 must be ON or OFF"
  exit 1
fi
if [[ "${mode}" == "debug" ]]; then
  nuri_with_asan="ON"
  nuri_with_logging="ON"
  nuri_with_asserts="ON"
  nuri_with_tracy="ON"
  nuri_with_tracy_gpu="ON"
  nuri_with_tracy_gpu_draw_zones="OFF"
fi

if [[ "${mode}" == "release" && "${devchecks}" == "ON" ]]; then
  nuri_with_logging="ON"
  nuri_with_asserts="ON"
fi

case "${tracy_mode}" in
  "")
    ;;
  cpu)
    nuri_with_tracy="ON"
    nuri_with_tracy_gpu="OFF"
    nuri_with_tracy_gpu_draw_zones="OFF"
    ;;
  cpu-gpu)
    nuri_with_tracy="ON"
    nuri_with_tracy_gpu="ON"
    nuri_with_tracy_gpu_draw_zones="ON"
    ;;
  off)
    nuri_with_tracy="OFF"
    nuri_with_tracy_gpu="OFF"
    nuri_with_tracy_gpu_draw_zones="OFF"
    ;;
esac

build_dir="${repo_root}/build/${build_dir_suffix}"

manifest_feature_args=()
if [[ -n "${manifest_features}" ]]; then
  manifest_feature_args=(-DVCPKG_MANIFEST_FEATURES="${manifest_features}")
fi

"${script_dir}/bootstrap_lightweightvk.sh"

configure_args=(
  -S "${repo_root}"
  -B "${build_dir}"
  -G "${generator}"
  -DCMAKE_C_COMPILER=clang
  -DCMAKE_CXX_COMPILER=clang++
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
  -DNURI_TOOL_PROFILE="${profile}"
  -DNURI_DEV_CHECKS="${devchecks}"
  -DNURI_BUILD_APP="${build_app}"
  -DNURI_BUILD_EDITOR="${build_editor}"
  -DNURI_BUILD_TESTS="${build_tests}"
  -DNURI_BUILD_TOOLS="${build_tools}"
  -DNURI_BUILD_BENCHMARK_CLI="${build_benchmark_cli}"
  -DNURI_BUILD_SNAPSHOT_CLI="${build_snapshot_cli}"
  -DNURI_BUILD_SNAPSHOT_TESTING="${build_snapshot_testing}"
  -DNURI_BUILD_AUTOTEST_CLI="${build_autotest_cli}"
  -DNURI_BUILD_AUTOTESTING="${build_autotesting}"
  -DNURI_WITH_ASAN="${nuri_with_asan}"
  -DNURI_WITH_LOGGING="${nuri_with_logging}"
  -DNURI_WITH_ASSERTS="${nuri_with_asserts}"
  -DNURI_WITH_FSR31="${nuri_with_fsr31}"
  "${manifest_feature_args[@]}"
)

case "${mode}" in
  debug)
    configure_args+=(
      -DCMAKE_BUILD_TYPE=Debug
      -DVCPKG_BUILD_TYPE=release
      -DNURI_BUILD_SHARED=ON
      -DNURI_WITH_TRACY="${nuri_with_tracy}"
      -DNURI_WITH_TRACY_GPU="${nuri_with_tracy_gpu}"
      -DNURI_WITH_TRACY_GPU_DRAW_ZONES="${nuri_with_tracy_gpu_draw_zones}"
    )
    ;;
  release)
    configure_args+=(
      -DCMAKE_BUILD_TYPE=Release
      -DVCPKG_BUILD_TYPE=release
      -DNURI_BUILD_SHARED=OFF
      -DNURI_WITH_TRACY="${nuri_with_tracy}"
      -DNURI_WITH_TRACY_GPU="${nuri_with_tracy_gpu}"
      -DNURI_WITH_TRACY_GPU_DRAW_ZONES="${nuri_with_tracy_gpu_draw_zones}"
    )
    ;;
esac

cmake "${configure_args[@]}"

build_args=(--build "${build_dir}")
if [[ -n "${build_target}" ]]; then
  build_args+=(--target "${build_target}")
fi
if [[ "${mode}" == "release" ]]; then
  build_args+=(--config Release)
fi

cmake "${build_args[@]}"

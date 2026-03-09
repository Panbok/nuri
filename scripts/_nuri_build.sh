#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $(basename "$0") <debug|release> <lib|app|editor|tests>"
  exit 1
fi

mode="$1"
profile="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ -z "${VCPKG_ROOT:-}" ]]; then
  echo "VCPKG_ROOT is not set. Point it at your vcpkg root."
  exit 1
fi

generator="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  generator="Ninja"
fi

append_manifest_feature() {
  local feature="$1"
  case ",${manifest_features}," in
    *,"${feature}",*)
      ;;
    ",,")
      manifest_features="${feature}"
      ;;
    *)
      manifest_features="${manifest_features},${feature}"
      ;;
  esac
}

build_app="OFF"
build_editor="OFF"
build_tests="OFF"
build_target=""
manifest_features="${VCPKG_MANIFEST_FEATURES:-}"

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
  *)
    echo "Usage: $(basename "$0") <debug|release> <lib|app|editor|tests>"
    exit 1
    ;;
esac

build_dir="${repo_root}/build/${mode}/${profile}"

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
  -DNURI_BUILD_APP="${build_app}"
  -DNURI_BUILD_EDITOR="${build_editor}"
  -DNURI_BUILD_TESTS="${build_tests}"
  "${manifest_feature_args[@]}"
)

case "${mode}" in
  debug)
    configure_args+=(
      -DCMAKE_BUILD_TYPE=Debug
      -DNURI_BUILD_SHARED=ON
    )
    ;;
  release)
    configure_args+=(
      -DCMAKE_BUILD_TYPE=Release
      -DVCPKG_BUILD_TYPE=release
      -DNURI_BUILD_SHARED=OFF
    )
    ;;
  *)
    echo "Usage: $(basename "$0") <debug|release> <lib|app|editor|tests>"
    exit 1
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

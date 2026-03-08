#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VCPKG_ROOT:-}" ]]; then
  echo "VCPKG_ROOT is not set. Point it at your vcpkg root."
  exit 1
fi

generator="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  generator="Ninja"
fi

build_tests="${NURI_BUILD_TESTS:-OFF}"
build_editor="${NURI_BUILD_EDITOR:-ON}"
manifest_features="${VCPKG_MANIFEST_FEATURES:-}"
if [[ "${build_editor}" == "ON" ]]; then
  case ",${manifest_features}," in
    *,editor,*)
      ;;
    ",,")
      manifest_features="editor"
      ;;
    *)
      manifest_features="${manifest_features},editor"
      ;;
  esac
fi
manifest_feature_args=()
if [[ -n "${manifest_features}" ]]; then
  manifest_feature_args=(-DVCPKG_MANIFEST_FEATURES="${manifest_features}")
fi

./scripts/bootstrap_lightweightvk.sh

cmake -S . -B build/release -G "${generator}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" \
  -DNURI_BUILD_TESTS="${build_tests}" \
  -DNURI_BUILD_EDITOR="${build_editor}" \
  "${manifest_feature_args[@]}" \
  -DVCPKG_BUILD_TYPE=release \
  -DNURI_BUILD_SHARED=OFF

cmake --build build/release --config Release

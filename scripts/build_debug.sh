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

./scripts/bootstrap_lightweightvk.sh

cmake -S . -B build/debug -G "${generator}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" \
  -DNURI_BUILD_SHARED=ON

cmake --build build/debug

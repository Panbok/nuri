#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mode="debug"

if [[ $# -gt 0 ]]; then
  case "$1" in
    debug)
      shift
      ;;
    release)
      mode="release"
      shift
      ;;
    -*)
      ;;
    *)
      echo "Usage: $(basename "$0") [debug|release] [ctest args...]"
      exit 1
      ;;
  esac
fi

manifest_features="${VCPKG_MANIFEST_FEATURES:-}"
case ",${manifest_features}," in
  *,tests,*)
    ;;
  ",,")
    manifest_features="tests"
    ;;
  *)
    manifest_features="${manifest_features},tests"
    ;;
esac

if [[ "$mode" == "debug" ]]; then
  (
    cd "$REPO_ROOT"
    NURI_BUILD_TESTS=ON \
    VCPKG_MANIFEST_FEATURES="$manifest_features" \
    "$SCRIPT_DIR/build_debug.sh"
  )
  build_dir="${REPO_ROOT}/build/debug"
else
  (
    cd "$REPO_ROOT"
    NURI_BUILD_TESTS=ON \
    VCPKG_MANIFEST_FEATURES="$manifest_features" \
    "$SCRIPT_DIR/build_release.sh"
  )
  build_dir="${REPO_ROOT}/build/release"
fi

ctest --test-dir "$build_dir" --output-on-failure "$@"

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

if [[ "${mode}" == "release" ]]; then
  build_dir="${REPO_ROOT}/build_release/tests"
else
  build_dir="${REPO_ROOT}/build_tests"
fi
"${SCRIPT_DIR}/_nuri_build.sh" "${mode}" tests

ctest --test-dir "$build_dir" --output-on-failure "$@"

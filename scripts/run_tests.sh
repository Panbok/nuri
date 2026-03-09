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

build_dir="${REPO_ROOT}/build/${mode}/tests"
"${SCRIPT_DIR}/_nuri_build.sh" "${mode}" tests

ctest --test-dir "$build_dir" --output-on-failure "$@"

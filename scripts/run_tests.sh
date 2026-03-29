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

build_dir="${REPO_ROOT}/build/${mode}"
"${SCRIPT_DIR}/_nuri_build.sh" "${mode}" tests

has_jobs_arg=0
for arg in "$@"; do
  if [[ "${arg}" == "-j" || "${arg}" == "--parallel" ]]; then
    has_jobs_arg=1
    break
  fi
done

ctest_args=(--test-dir "$build_dir" --output-on-failure)
if [[ "${has_jobs_arg}" -eq 0 ]]; then
  if command -v nproc >/dev/null 2>&1; then
    ctest_args+=(-j "$(nproc)")
  else
    ctest_args+=(-j 4)
  fi
fi

ctest "${ctest_args[@]}" "$@"

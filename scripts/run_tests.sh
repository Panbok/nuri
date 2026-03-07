#!/usr/bin/env bash
set -euo pipefail

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

if [[ "$mode" == "debug" ]]; then
  ./scripts/build_debug.sh
  build_dir="build/debug"
else
  ./scripts/build_release.sh
  build_dir="build/release"
fi

ctest --test-dir "$build_dir" --output-on-failure "$@"

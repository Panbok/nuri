#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="debug"
profile="tests"

if [[ $# -gt 0 ]]; then
  case "$1" in
    debug)
      shift
      ;;
    release)
      mode="release"
      shift
      ;;
    bench-tests)
      profile="bench-tests"
      shift
      ;;
    snapshot-tests)
      profile="snapshot-tests"
      shift
      ;;
    autotest-tests)
      profile="autotest-tests"
      shift
      ;;
    *)
      echo "Usage: $(basename "$0") [debug|release] [bench-tests|snapshot-tests|autotest-tests]"
      exit 1
      ;;
  esac
fi

if [[ $# -gt 0 ]]; then
  case "$1" in
    bench-tests)
      profile="bench-tests"
      shift
      ;;
    snapshot-tests)
      profile="snapshot-tests"
      shift
      ;;
    autotest-tests)
      profile="autotest-tests"
      shift
      ;;
    *)
      echo "Usage: $(basename "$0") [debug|release] [bench-tests|snapshot-tests|autotest-tests]"
      exit 1
      ;;
  esac
fi

if [[ $# -ne 0 ]]; then
  echo "Usage: $(basename "$0") [debug|release] [bench-tests|snapshot-tests|autotest-tests]"
  exit 1
fi

"${script_dir}/_nuri_build.sh" "${mode}" "${profile}"

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
  esac
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${script_dir}/_nuri_build.sh" "${mode}" app
"${script_dir}/_nuri_exec.sh" "${mode}" app "$@"

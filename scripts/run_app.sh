#!/usr/bin/env bash
set -euo pipefail

mode="debug"
tracy_mode=""
devchecks=""
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
if [[ $# -gt 0 ]]; then
  case "$1" in
    cpu|off)
      tracy_mode="$1"
      shift
      ;;
  esac
fi
if [[ $# -gt 0 && "$1" == "devchecks" ]]; then
  devchecks="$1"
  shift
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_args=("${mode}" app)
if [[ -n "${tracy_mode}" ]]; then
  build_args+=("${tracy_mode}")
fi
if [[ -n "${devchecks}" ]]; then
  build_args+=("${devchecks}")
fi

"${script_dir}/_nuri_build.sh" "${build_args[@]}"
"${script_dir}/_nuri_exec.sh" "${mode}" app "$@"

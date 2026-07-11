#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="release"

if [[ $# -gt 0 ]]; then
  case "$1" in
    debug)
      mode="debug"
      shift
      ;;
    release)
      shift
      ;;
  esac
fi

exec "${script_dir}/_nuri_build.sh" "${mode}" autotest "$@"

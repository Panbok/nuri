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
    cpu|off|devchecks)
      ;;
    *)
      echo "Usage: $(basename "$0") [release|debug] [cpu|off] [devchecks]"
      exit 1
      ;;
  esac
fi

"${script_dir}/_nuri_build.sh" "${mode}" snapshot "$@"

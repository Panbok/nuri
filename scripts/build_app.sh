#!/usr/bin/env bash
set -euo pipefail

mode="${1:-debug}"
tracy_mode=""
devchecks=""

if [[ $# -gt 3 ]]; then
  echo "Usage: $(basename "$0") [debug|release] [cpu|off] [devchecks]"
  exit 1
fi
if [[ "${mode}" != "debug" && "${mode}" != "release" ]]; then
  echo "Usage: $(basename "$0") [debug|release] [cpu|off] [devchecks]"
  exit 1
fi
for arg in "${@:2}"; do
  case "${arg}" in
    cpu|off)
      if [[ -n "${tracy_mode}" ]]; then
        echo "Usage: $(basename "$0") [debug|release] [cpu|off] [devchecks]"
        exit 1
      fi
      tracy_mode="${arg}"
      ;;
    devchecks)
      devchecks="${arg}"
      ;;
    *)
      echo "Usage: $(basename "$0") [debug|release] [cpu|off] [devchecks]"
      exit 1
      ;;
  esac
done

build_args=("${mode}" app)
if [[ -n "${tracy_mode}" ]]; then
  build_args+=("${tracy_mode}")
fi
if [[ -n "${devchecks}" ]]; then
  build_args+=("${devchecks}")
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_nuri_build.sh" "${build_args[@]}"

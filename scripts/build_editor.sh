#!/usr/bin/env bash
set -euo pipefail

mode="${1:-debug}"
tracy_mode="${2:-}"
if [[ $# -gt 2 ]]; then
  echo "Usage: $(basename "$0") [debug|release] [cpu|cpu-gpu|off]"
  exit 1
fi
if [[ "${mode}" != "debug" && "${mode}" != "release" ]]; then
  echo "Usage: $(basename "$0") [debug|release] [cpu|cpu-gpu|off]"
  exit 1
fi
if [[ -n "${tracy_mode}" && "${tracy_mode}" != "cpu" && "${tracy_mode}" != "cpu-gpu" && "${tracy_mode}" != "off" ]]; then
  echo "Usage: $(basename "$0") [debug|release] [cpu|cpu-gpu|off]"
  exit 1
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_nuri_build.sh" "${mode}" editor "${tracy_mode}"

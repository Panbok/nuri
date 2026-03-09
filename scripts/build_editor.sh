#!/usr/bin/env bash
set -euo pipefail

mode="${1:-debug}"
if [[ $# -gt 1 ]]; then
  echo "Usage: $(basename "$0") [debug|release]"
  exit 1
fi
if [[ "${mode}" != "debug" && "${mode}" != "release" ]]; then
  echo "Usage: $(basename "$0") [debug|release]"
  exit 1
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_nuri_build.sh" "${mode}" editor

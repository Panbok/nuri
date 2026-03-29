#!/usr/bin/env bash
set -euo pipefail

profile="${1:-app}"
if [[ $# -gt 2 ]]; then
  echo "Usage: $(basename "$0") [lib|app|editor|tests] [cpu|cpu-gpu|off]"
  exit 1
fi

build_args=(debug "${profile}")
if [[ $# -ge 2 ]]; then
  build_args+=("$2")
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_nuri_build.sh" "${build_args[@]}"

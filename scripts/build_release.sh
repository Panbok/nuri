#!/usr/bin/env bash
set -euo pipefail

profile="${1:-app}"
if [[ $# -gt 3 ]]; then
  echo "Usage: $(basename "$0") [lib|app|editor|tests] [cpu|cpu-gpu|off] [devchecks]"
  exit 1
fi

build_args=(release "${profile}")
if [[ $# -ge 2 ]]; then
  build_args+=("$2")
fi
if [[ $# -ge 3 ]]; then
  build_args+=("$3")
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_nuri_build.sh" "${build_args[@]}"

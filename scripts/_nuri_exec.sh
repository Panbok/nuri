#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $(basename "$0") <debug|release> <profile> [args...]" >&2
  exit 2
fi
mode="$1"
profile="$2"
shift 2
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${script_dir}/nuri_build.py" legacy-run "${mode}" "${profile}" --no-build -- "$@"

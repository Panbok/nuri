#!/usr/bin/env bash
set -euo pipefail

profile="${1:-app}"
if [[ $# -gt 1 ]]; then
  echo "Usage: $(basename "$0") [lib|app|editor|tests]"
  exit 1
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_nuri_build.sh" release "${profile}"

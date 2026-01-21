#!/usr/bin/env bash
set -euo pipefail

lvk_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/external/lightweightvk"

command -v git >/dev/null 2>&1 || { echo "git not found on PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found on PATH"; exit 1; }

if [[ ! -f "${lvk_dir}/CMakeLists.txt" ]]; then
  if [[ -f ".gitmodules" ]] && grep -q "external/lightweightvk" .gitmodules 2>/dev/null; then
    git submodule update --init --recursive external/lightweightvk
  fi
fi

if [[ ! -f "${lvk_dir}/CMakeLists.txt" ]]; then
  echo "lightweightvk submodule not found at: ${lvk_dir}"
  exit 1
fi

pushd "${lvk_dir}" >/dev/null
python3 third-party/bootstrap.py -b third-party/deps --bootstrap-file=third-party/bootstrap-deps.json -N "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lvk-deps.txt" --break-on-first-error
popd >/dev/null

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $(basename "$0") <debug|release> <app|editor> [args...]"
  exit 1
fi

mode="$1"
profile="$2"
shift 2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
build_dir="${repo_root}/build/${mode}/${profile}"

case "${profile}" in
  app)
    app_path="${build_dir}/nuri"
    ;;
  editor)
    app_path="${build_dir}/nuri_editor"
    ;;
  *)
    echo "Usage: $(basename "$0") <debug|release> <app|editor> [args...]"
    exit 1
    ;;
esac

if [[ ! -x "${app_path}" ]]; then
  echo "Build output not found: ${app_path}"
  exit 1
fi

path_entries=()
while IFS= read -r dir; do
  path_entries+=("${dir}")
done < <(find "${build_dir}/vcpkg_installed" -type d -path '*/bin' 2>/dev/null)

lib_path="${build_dir}/lib"
if [[ -d "${lib_path}" ]]; then
  case "${OSTYPE:-}" in
    darwin*)
      export DYLD_LIBRARY_PATH="${lib_path}:${DYLD_LIBRARY_PATH:-}"
      ;;
    *)
      export LD_LIBRARY_PATH="${lib_path}:${LD_LIBRARY_PATH:-}"
      ;;
  esac
fi

if [[ ${#path_entries[@]} -gt 0 ]]; then
  path_joined="$(printf '%s:' "${path_entries[@]}")"
  export PATH="${path_joined%:}:${PATH}"
fi

exec "${app_path}" "$@"

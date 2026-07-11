#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
mode="release"
build_args=()
first_tool_arg=""
no_build=0

if [[ $# -gt 0 ]]; then
  case "$1" in
    debug)
      mode="debug"
      shift
      ;;
    release)
      shift
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      no_build=1
      shift
      ;;
    cpu|cpu-gpu|off|devchecks)
      build_args+=("$1")
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ ${no_build} -eq 0 ]]; then
  "${script_dir}/_nuri_build.sh" "${mode}" snapshot "${build_args[@]}"
fi

build_dir="${repo_root}/build/${mode}-snapshot"
tool="${build_dir}/nuri-snapshot"
if [[ ! -x "${tool}" ]]; then
  echo "Build output not found: ${tool}"
  exit 1
fi

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

path_entries=()
while IFS= read -r dir; do
  path_entries+=("${dir}")
done < <(find "${build_dir}/vcpkg_installed" -type d -path '*/bin' 2>/dev/null)
if [[ ${#path_entries[@]} -gt 0 ]]; then
  path_joined="$(printf '%s:' "${path_entries[@]}")"
  export PATH="${path_joined%:}:${PATH}"
fi

tool_args=("$@")
if [[ ${#tool_args[@]} -eq 0 ]]; then
  tool_args=(list)
else
  first_tool_arg="${tool_args[0]}"
  case "${first_tool_arg}" in
    list|explain|capture|compare|run|approve|baseline|diff)
      ;;
    *)
      tool_args=(run "${tool_args[@]}")
      ;;
  esac
fi

exec "${tool}" "${tool_args[@]}"

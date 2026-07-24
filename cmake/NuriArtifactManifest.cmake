include_guard(GLOBAL)

function(nuri_register_artifact_target target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "Cannot register missing artifact target: ${target}")
  endif()
  set_property(GLOBAL APPEND PROPERTY NURI_ARTIFACT_TARGETS "${target}")
endfunction()

function(nuri_write_artifact_manifest)
  get_property(_nuri_targets GLOBAL PROPERTY NURI_ARTIFACT_TARGETS)
  if(NOT _nuri_targets)
    message(FATAL_ERROR "No first-party artifact targets were registered")
  endif()
  list(REMOVE_DUPLICATES _nuri_targets)
  list(SORT _nuri_targets)

  file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" _nuri_source_dir)
  file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}" _nuri_binary_dir)
  if(DEFINED VCPKG_INSTALLED_DIR)
    file(TO_CMAKE_PATH "${VCPKG_INSTALLED_DIR}" _nuri_vcpkg_root)
  else()
    set(_nuri_vcpkg_root "")
  endif()
  if(DEFINED VCPKG_TARGET_TRIPLET)
    set(_nuri_vcpkg_triplet "${VCPKG_TARGET_TRIPLET}")
  else()
    set(_nuri_vcpkg_triplet "")
  endif()
  set(_nuri_compiler_runtime_dir "")
  if(WIN32 AND NURI_WITH_ASAN
      AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}"
        "--print-file-name=clang_rt.asan_dynamic-x86_64.dll"
      RESULT_VARIABLE _nuri_runtime_result
      OUTPUT_VARIABLE _nuri_runtime_dll
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _nuri_runtime_result EQUAL 0 OR
        NOT EXISTS "${_nuri_runtime_dll}")
      message(FATAL_ERROR
        "Clang did not resolve its ASan runtime DLL: ${_nuri_runtime_dll}")
    endif()
    get_filename_component(
      _nuri_compiler_runtime_dir "${_nuri_runtime_dll}" DIRECTORY)
    file(TO_CMAKE_PATH
      "${_nuri_compiler_runtime_dir}" _nuri_compiler_runtime_dir)
  endif()

  set(_nuri_entries "")
  set(_nuri_separator "")
  foreach(_nuri_target IN LISTS _nuri_targets)
    get_target_property(_nuri_type "${_nuri_target}" TYPE)
    string(APPEND _nuri_entries
      "${_nuri_separator}    \"${_nuri_target}\": {\n"
      "      \"type\": \"${_nuri_type}\",\n"
      "      \"path\": \"$<TARGET_FILE:${_nuri_target}>\",\n"
      "      \"workingDirectory\": \"${_nuri_source_dir}\"\n"
      "    }"
    )
    set(_nuri_separator ",\n")
  endforeach()

  set(_nuri_content
"{
  \"schemaVersion\": 1,
  \"kind\": \"nuri.cmake_artifact_manifest\",
  \"configuration\": \"$<CONFIG>\",
  \"sourceRoot\": \"${_nuri_source_dir}\",
  \"binaryRoot\": \"${_nuri_binary_dir}\",
  \"treeIdentityDigest\": \"${NURI_TREE_IDENTITY_DIGEST}\",
  \"compileCompatibilityDigest\": \"${NURI_COMPILE_COMPATIBILITY_DIGEST}\",
  \"dependencyIdentityDigest\": \"${NURI_DEPENDENCY_IDENTITY_DIGEST}\",
  \"runtimeSearchPaths\": [
    \"${_nuri_binary_dir}/out/$<CONFIG>/bin\",
    \"${_nuri_binary_dir}/out/$<CONFIG>/lib\",
    \"${_nuri_vcpkg_root}/${_nuri_vcpkg_triplet}/bin\",
    \"${_nuri_vcpkg_root}/${_nuri_vcpkg_triplet}/debug/bin\",
    \"${_nuri_compiler_runtime_dir}\"
  ],
  \"targets\": {
${_nuri_entries}
  }
}
")
  file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/.nuri-artifacts-$<CONFIG>.json"
    CONTENT "${_nuri_content}"
  )
endfunction()

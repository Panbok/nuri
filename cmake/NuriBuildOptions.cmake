include_guard(GLOBAL)

set(NURI_CPU_PROFILE "x86-64-v3" CACHE STRING
    "Explicit first-party CPU code-generation profile")
set_property(CACHE NURI_CPU_PROFILE PROPERTY STRINGS portable x86-64-v3)
option(NURI_NORMALIZE_COMPILER_PATHS
  "Normalize source/build prefixes for compiler-result caching" OFF)

add_library(nuri_build_options INTERFACE)
add_library(nuri::build_options ALIAS nuri_build_options)
target_compile_features(nuri_build_options INTERFACE cxx_std_20)
target_compile_definitions(nuri_build_options INTERFACE
  "$<$<CONFIG:Debug>:NURI_DEBUG>"
)

if(MSVC)
  target_compile_options(nuri_build_options INTERFACE /W4 /wd4013)
  if(NURI_CPU_PROFILE STREQUAL "x86-64-v3")
    target_compile_options(nuri_build_options INTERFACE /arch:AVX2)
  elseif(NOT NURI_CPU_PROFILE STREQUAL "portable")
    message(FATAL_ERROR "Unsupported NURI_CPU_PROFILE: ${NURI_CPU_PROFILE}")
  endif()
else()
  target_compile_options(nuri_build_options INTERFACE
    -Wno-implicit-function-declaration
  )
  if(NURI_CPU_PROFILE STREQUAL "x86-64-v3")
    target_compile_options(nuri_build_options INTERFACE -march=x86-64-v3)
  elseif(NOT NURI_CPU_PROFILE STREQUAL "portable")
    message(FATAL_ERROR "Unsupported NURI_CPU_PROFILE: ${NURI_CPU_PROFILE}")
  endif()
endif()

if(NURI_NORMALIZE_COMPILER_PATHS)
  if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(nuri_build_options INTERFACE
      "/clang:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
      "/clang:-fdebug-prefix-map=${CMAKE_BINARY_DIR}=."
    )
  elseif(MSVC)
    target_compile_options(nuri_build_options INTERFACE
      "/pathmap:${CMAKE_SOURCE_DIR}=."
      "/pathmap:${CMAKE_BINARY_DIR}=."
    )
  else()
    target_compile_options(nuri_build_options INTERFACE
      "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
      "-fdebug-prefix-map=${CMAKE_BINARY_DIR}=."
    )
  endif()
endif()

if(NURI_WITH_ASAN AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(WIN32)
    add_library(nuri_windows_clang_asan_abi INTERFACE)
    add_library(nuri::windows_clang_asan_abi ALIAS
      nuri_windows_clang_asan_abi)
    target_compile_options(nuri_windows_clang_asan_abi INTERFACE
      -fsanitize=address
      "$<$<CONFIG:Debug>:-U_DEBUG>"
    )
    target_link_options(nuri_windows_clang_asan_abi INTERFACE
      -fsanitize=address)
    target_compile_definitions(nuri_windows_clang_asan_abi INTERFACE
      _ITERATOR_DEBUG_LEVEL=0
      _DISABLE_STL_ANNOTATION
    )
    if(NURI_USE_STATIC_CRT)
      target_compile_options(nuri_windows_clang_asan_abi INTERFACE
        -fms-runtime-lib=static
      )
      target_link_options(nuri_windows_clang_asan_abi INTERFACE
        -fms-runtime-lib=static
        "LINKER:/NODEFAULTLIB:libcmtd"
        "LINKER:/DEFAULTLIB:libcmt"
      )
    else()
      target_compile_options(nuri_windows_clang_asan_abi INTERFACE
        -fms-runtime-lib=dll
      )
      target_link_options(nuri_windows_clang_asan_abi INTERFACE
        -fms-runtime-lib=dll
        "LINKER:/NODEFAULTLIB:msvcrtd"
        "LINKER:/NODEFAULTLIB:ucrtd"
        "LINKER:/NODEFAULTLIB:vcruntimed"
        "LINKER:/NODEFAULTLIB:msvcprtd"
        "LINKER:/DEFAULTLIB:msvcrt"
        "LINKER:/DEFAULTLIB:ucrt"
        "LINKER:/DEFAULTLIB:vcruntime"
        "LINKER:/DEFAULTLIB:msvcprt"
      )
    endif()
    target_link_libraries(nuri_build_options INTERFACE
      nuri::windows_clang_asan_abi)
  else()
    target_compile_options(nuri_build_options INTERFACE
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
    )
    target_link_options(nuri_build_options INTERFACE
      -fsanitize=address,undefined
    )
  endif()
endif()

add_library(nuri_source_root INTERFACE)
add_library(nuri::source_root ALIAS nuri_source_root)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" _nuri_source_root)
target_compile_definitions(nuri_source_root INTERFACE
  "PROJECT_SOURCE_DIR=\"${_nuri_source_root}/\""
)

function(nuri_configure_first_party_target target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "Unknown first-party target: ${target}")
  endif()

  get_target_property(_nuri_target_type "${target}" TYPE)
  if(NOT _nuri_target_type STREQUAL "INTERFACE_LIBRARY")
    target_link_libraries("${target}" PRIVATE nuri::build_options)
  endif()

  set(_nuri_output_root "${CMAKE_BINARY_DIR}/out")
  foreach(_nuri_config IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${_nuri_config}" _nuri_config_upper)
    if(_nuri_target_type STREQUAL "EXECUTABLE")
      set_target_properties("${target}" PROPERTIES
        "RUNTIME_OUTPUT_DIRECTORY_${_nuri_config_upper}"
        "${_nuri_output_root}/${_nuri_config}/bin"
      )
    elseif(_nuri_target_type MATCHES "^(STATIC|SHARED|MODULE)_LIBRARY$")
      set_target_properties("${target}" PROPERTIES
        "ARCHIVE_OUTPUT_DIRECTORY_${_nuri_config_upper}"
        "${_nuri_output_root}/${_nuri_config}/lib"
        "LIBRARY_OUTPUT_DIRECTORY_${_nuri_config_upper}"
        "${_nuri_output_root}/${_nuri_config}/lib"
        "RUNTIME_OUTPUT_DIRECTORY_${_nuri_config_upper}"
        "${_nuri_output_root}/${_nuri_config}/bin"
        "PDB_OUTPUT_DIRECTORY_${_nuri_config_upper}"
        "${_nuri_output_root}/${_nuri_config}/symbols"
        "COMPILE_PDB_OUTPUT_DIRECTORY_${_nuri_config_upper}"
        "${_nuri_output_root}/${_nuri_config}/symbols"
      )
    endif()
  endforeach()
endfunction()

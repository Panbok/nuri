cmake_minimum_required(VERSION 3.29)

# Decide CRT linkage as early as possible. CMake's Windows-Clang platform
# module injects /MD-style defaults at language enable time unless told not to.
set(NURI_USE_STATIC_CRT OFF)
if(DEFINED VCPKG_TARGET_TRIPLET AND VCPKG_TARGET_TRIPLET MATCHES "-static")
    set(NURI_USE_STATIC_CRT ON)
endif()

if(WIN32)
    # Prevent CMake's Windows-Clang defaults from forcing /MD.
    set(CMAKE_MSVC_RUNTIME_LIBRARY_DEFAULT 1)
    if(NURI_USE_STATIC_CRT)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
    endif()
endif()

project(nuri LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NURI_BUILD_SHARED "Build nuri as a shared library" ON)
option(NURI_WITH_ASAN "Enable AddressSanitizer for supported toolchains" ON)
option(NURI_WITH_LOGGING "Enable nuri logging macros and runtime logger" ON)
option(NURI_WITH_ASSERTS "Enable NURI_ASSERT checks" ON)
option(NURI_WITH_TRACY "Enable Tracy CPU/GPU profiling" OFF)
option(NURI_WITH_FSR31 "Enable pinned FidelityFX FSR 3.1 integration hooks" OFF)
option(NURI_BUILD_APP "Build nuri app target" ON)
option(NURI_BUILD_EDITOR "Build nuri editor target" ON)
option(NURI_BUILD_TESTS "Build nuri test targets" OFF)
option(NURI_BUILD_TOOLS "Build nuri automation tools" OFF)
set(NURI_GRAPH_CAPABILITY "manual" CACHE STRING
    "Named configured target-graph capability")
set(NURI_TREE_IDENTITY_DIGEST "" CACHE STRING "Canonical tree identity digest")
set(NURI_COMPILE_COMPATIBILITY_DIGEST "" CACHE STRING
    "Compile-compatibility digest")
set(NURI_DEPENDENCY_IDENTITY_DIGEST "" CACHE STRING
    "Dependency identity digest")
set(NURI_NVRHI_SOURCE_DIR "" CACHE PATH
    "Prepared immutable NVRHI source directory")

if(WIN32 AND NURI_WITH_ASAN AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(NURI_USE_STATIC_CRT)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "" FORCE)
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "" FORCE)
    endif()

    foreach(_flag_var IN ITEMS CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG)
        string(REPLACE "/MDd" "" ${_flag_var} "${${_flag_var}}")
        string(REPLACE "/MTd" "" ${_flag_var} "${${_flag_var}}")
    endforeach()
endif()

if(NURI_WITH_ASAN AND WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)
endif()

include(cmake/NuriBuildOptions.cmake)
include(cmake/NuriArtifactManifest.cmake)

add_subdirectory(lib)
if(NURI_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(NURI_BUILD_APP)
  add_subdirectory(app)
endif()
if(NURI_BUILD_EDITOR)
  add_subdirectory(editor)
endif()

if(NURI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

nuri_write_artifact_manifest()

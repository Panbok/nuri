cmake_minimum_required(VERSION 3.27)

project(nuri_lib)

file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS */*.cpp */*.h)

set(NURI_LIBRARY_TYPE STATIC)
if(NURI_BUILD_SHARED)
  set(NURI_LIBRARY_TYPE SHARED)
endif()

add_library(nuri_lib ${NURI_LIBRARY_TYPE}
  ${LIB_SOURCES}
)

if(MSVC)
    target_compile_options(nuri_lib PRIVATE /arch:AVX2)
else()
    target_compile_options(nuri_lib PRIVATE -march=native)
endif()

# Find Vulkan package
find_package(Vulkan REQUIRED)

# CMake's FindVulkan on Windows can result in Vulkan::Vulkan missing Debug locations.
# This project uses Ninja + Debug builds; ensure the imported target is configured.
if(WIN32 AND TARGET Vulkan::Vulkan)
  set_property(TARGET Vulkan::Vulkan PROPERTY MAP_IMPORTED_CONFIG_DEBUG Release)

  if(DEFINED Vulkan_LIBRARY AND Vulkan_LIBRARY)
    set(_vk_implib "${Vulkan_LIBRARY}")
    set(_vk_location "${Vulkan_LIBRARY}")

    foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
      set_target_properties(Vulkan::Vulkan PROPERTIES
        "IMPORTED_LOCATION_${_cfg}" "${_vk_location}"
        "IMPORTED_IMPLIB_${_cfg}" "${_vk_implib}"
      )
    endforeach()

    set_target_properties(Vulkan::Vulkan PROPERTIES
      "IMPORTED_LOCATION" "${_vk_location}"
      "IMPORTED_IMPLIB" "${_vk_implib}"
    )
  endif()
endif()

target_include_directories(nuri_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Vulkan_INCLUDE_DIRS}
)

target_compile_features(nuri_lib PUBLIC cxx_std_20)

find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

if(NURI_WITH_TRACY)
  # Tracy is provided by vcpkg (manifest mode). Target name varies between ports;
  # accept a few common ones.
  find_package(Tracy CONFIG QUIET)
  find_package(tracy CONFIG QUIET)

  set(NURI_TRACY_CLIENT_TARGET "")
  if(TARGET Tracy::TracyClient)
    set(NURI_TRACY_CLIENT_TARGET Tracy::TracyClient)
  elseif(TARGET tracy::tracyclient)
    set(NURI_TRACY_CLIENT_TARGET tracy::tracyclient)
  elseif(TARGET unofficial::tracy::tracyclient)
    set(NURI_TRACY_CLIENT_TARGET unofficial::tracy::tracyclient)
  elseif(TARGET TracyClient)
    set(NURI_TRACY_CLIENT_TARGET TracyClient)
  endif()

  if(NOT NURI_TRACY_CLIENT_TARGET)
    message(FATAL_ERROR "NURI_WITH_TRACY=ON but no TracyClient target was found. Tried: Tracy::TracyClient, tracy::tracyclient, unofficial::tracy::tracyclient, TracyClient.")
  endif()

  target_compile_definitions(nuri_lib PUBLIC NURI_PROFILING=1 TRACY_ENABLE=1)
  target_link_libraries(nuri_lib PUBLIC ${NURI_TRACY_CLIENT_TARGET})
endif()

set(LVK_DEPLOY_DEPS OFF CACHE BOOL "" FORCE)
set(LVK_WITH_GLFW ON CACHE BOOL "" FORCE)
set(LVK_USE_VCPKG_GLFW ON CACHE BOOL "" FORCE)
set(LVK_WITH_IMGUI OFF CACHE BOOL "" FORCE)
set(LVK_WITH_SAMPLES OFF CACHE BOOL "" FORCE)
set(LVK_WITH_TRACY ${NURI_WITH_TRACY} CACHE BOOL "" FORCE)
set(LVK_WITH_TRACY_GPU ${NURI_WITH_TRACY} CACHE BOOL "" FORCE)
set(LVK_WITH_IMPLOT OFF CACHE BOOL "" FORCE)
set(LVK_WITH_OPENXR OFF CACHE BOOL "" FORCE)
set(LVK_WITH_SLANG OFF CACHE BOOL "" FORCE)

set(LVK_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/lightweightvk")
if(NOT EXISTS "${LVK_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "LightweightVK submodule is missing at: ${LVK_SOURCE_DIR}. Run: git submodule update --init --recursive")
endif()

add_subdirectory("${LVK_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/lightweightvk-build")

target_link_libraries(nuri_lib
  PUBLIC
    glm::glm
)

target_link_libraries(nuri_lib PUBLIC assimp::assimp)
target_link_libraries(nuri_lib PUBLIC LVKLibrary)
if(TARGET imgui::imgui)
  target_link_libraries(nuri_lib PUBLIC imgui::imgui)
elseif(TARGET imgui)
  target_link_libraries(nuri_lib PUBLIC imgui)
endif()

if(NURI_BUILD_SHARED)
  target_compile_definitions(nuri_lib PUBLIC NURI_SHARED)
  target_compile_definitions(nuri_lib PRIVATE NURI_EXPORTS)
endif()

set_source_files_properties(
  nuri/resources/stb_image.cpp
  PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

target_precompile_headers(nuri_lib PRIVATE nuri/pch.h)

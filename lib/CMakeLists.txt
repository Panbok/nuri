cmake_minimum_required(VERSION 3.27)

project(nuri_lib)

file(GLOB_RECURSE LIB_SOURCES */*.cpp */*.h)

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

set(LVK_DEPLOY_DEPS OFF CACHE BOOL "" FORCE)
# LVK's ImGui helpers reference lvk::LVKwindow, which is provided when GLFW support is enabled.
set(LVK_WITH_GLFW ON CACHE BOOL "" FORCE)
set(LVK_WITH_SAMPLES OFF CACHE BOOL "" FORCE)
set(LVK_WITH_TRACY OFF CACHE BOOL "" FORCE)
set(LVK_WITH_TRACY_GPU OFF CACHE BOOL "" FORCE)
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

target_link_libraries(nuri_lib PUBLIC LVKLibrary)

if(NURI_BUILD_SHARED)
  target_compile_definitions(nuri_lib PUBLIC NURI_SHARED)
  target_compile_definitions(nuri_lib PRIVATE NURI_EXPORTS)
endif()

target_precompile_headers(nuri_lib PRIVATE nuri/pch.h)

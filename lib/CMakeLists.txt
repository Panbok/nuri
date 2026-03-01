cmake_minimum_required(VERSION 3.27)

project(nuri_lib)

file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS */*.cpp */*.h)

set(NURI_LIBRARY_TYPE STATIC)
if(NURI_BUILD_SHARED)
  set(NURI_LIBRARY_TYPE SHARED)
endif()

add_library(nuri_lib ${NURI_LIBRARY_TYPE}
  ${LIB_SOURCES}
  ${CMAKE_SOURCE_DIR}/vendor/im-guizmo/ImGuizmo.cpp
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
    ${CMAKE_SOURCE_DIR}/vendor/im-guizmo
    PRIVATE ${Vulkan_INCLUDE_DIRS}
)

target_compile_features(nuri_lib PUBLIC cxx_std_20)

find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(Ktx CONFIG QUIET)
if(NOT Ktx_FOUND)
  find_package(ktx CONFIG REQUIRED)
endif()
find_package(yyjson CONFIG REQUIRED)
find_package(unordered_dense CONFIG REQUIRED)
find_package(harfbuzz CONFIG QUIET)
find_package(HarfBuzz CONFIG QUIET)

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
set(MINILOG_RAW_OUTPUT OFF CACHE BOOL "" FORCE)

set(LVK_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/lightweightvk")
if(NOT EXISTS "${LVK_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "LightweightVK submodule is missing at: ${LVK_SOURCE_DIR}. Run: git submodule update --init --recursive")
endif()

add_subdirectory("${LVK_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/lightweightvk-build")

set(MSDF_ATLAS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/msdf-atlas-gen")
if(NOT EXISTS "${MSDF_ATLAS_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "msdf-atlas-gen submodule is missing at: ${MSDF_ATLAS_SOURCE_DIR}. Run: git submodule update --init --recursive")
endif()

set(MSDF_ATLAS_BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
set(MSDF_ATLAS_USE_VCPKG ON CACHE BOOL "" FORCE)
set(MSDF_ATLAS_USE_SKIA OFF CACHE BOOL "" FORCE)
set(MSDF_ATLAS_INSTALL OFF CACHE BOOL "" FORCE)
set(NURI_MSDF_DYNAMIC_RUNTIME ON)
if(NURI_USE_STATIC_CRT)
  set(NURI_MSDF_DYNAMIC_RUNTIME OFF)
endif()
set(MSDF_ATLAS_DYNAMIC_RUNTIME ${NURI_MSDF_DYNAMIC_RUNTIME} CACHE BOOL "" FORCE)
set(MSDF_ATLAS_MSDFGEN_EXTERNAL OFF CACHE BOOL "" FORCE)
set(MSDFGEN_BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
set(MSDFGEN_USE_SKIA OFF CACHE BOOL "" FORCE)
set(MSDFGEN_USE_OPENMP OFF CACHE BOOL "" FORCE)
set(MSDFGEN_INSTALL OFF CACHE BOOL "" FORCE)
set(MSDFGEN_DYNAMIC_RUNTIME ${NURI_MSDF_DYNAMIC_RUNTIME} CACHE BOOL "" FORCE)

add_subdirectory("${MSDF_ATLAS_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/msdf-atlas-gen-build")

if(NURI_WITH_TRACY_GPU_DRAW_ZONES)
  if(NURI_WITH_TRACY AND TARGET LVKVulkan)
    target_compile_definitions(LVKVulkan PUBLIC "LVK_WITH_TRACY_GPU_DRAW_ZONES=1")
  else()
    message(WARNING "NURI_WITH_TRACY_GPU_DRAW_ZONES=ON requires NURI_WITH_TRACY=ON and target LVKVulkan; per-draw Tracy GPU zones remain disabled.")
  endif()
endif()

target_link_libraries(nuri_lib
  PUBLIC
    glm::glm
)

target_link_libraries(nuri_lib PUBLIC assimp::assimp)
target_link_libraries(nuri_lib PUBLIC LVKLibrary)
set(NURI_KTX_TARGET "")
if(TARGET KTX::ktx)
  set(NURI_KTX_TARGET KTX::ktx)
elseif(TARGET ktx)
  set(NURI_KTX_TARGET ktx)
endif()
if(NOT NURI_KTX_TARGET)
  message(FATAL_ERROR "ktx package was found but no target was available. Tried: KTX::ktx, ktx")
endif()
target_link_libraries(nuri_lib PRIVATE ${NURI_KTX_TARGET})
set(NURI_MSDF_ATLAS_TARGET "")
if(TARGET msdf-atlas-gen::msdf-atlas-gen)
  set(NURI_MSDF_ATLAS_TARGET msdf-atlas-gen::msdf-atlas-gen)
elseif(TARGET msdf-atlas-gen)
  set(NURI_MSDF_ATLAS_TARGET msdf-atlas-gen)
endif()
if(NOT NURI_MSDF_ATLAS_TARGET)
  message(FATAL_ERROR "msdf-atlas-gen target was not found. Tried: msdf-atlas-gen::msdf-atlas-gen, msdf-atlas-gen")
endif()
target_link_libraries(nuri_lib PRIVATE ${NURI_MSDF_ATLAS_TARGET})
set(NURI_MESHOPT_TARGET "")
if(TARGET meshoptimizer::meshoptimizer)
  set(NURI_MESHOPT_TARGET meshoptimizer::meshoptimizer)
elseif(TARGET meshoptimizer)
  set(NURI_MESHOPT_TARGET meshoptimizer)
endif()
if(NOT NURI_MESHOPT_TARGET)
  message(FATAL_ERROR "meshoptimizer package was found but no target was available. Tried: meshoptimizer::meshoptimizer, meshoptimizer")
endif()
target_link_libraries(nuri_lib PUBLIC ${NURI_MESHOPT_TARGET})

set(NURI_YYJSON_TARGET "")
if(TARGET yyjson::yyjson)
  set(NURI_YYJSON_TARGET yyjson::yyjson)
elseif(TARGET yyjson)
  set(NURI_YYJSON_TARGET yyjson)
elseif(TARGET unofficial::yyjson::yyjson)
  set(NURI_YYJSON_TARGET unofficial::yyjson::yyjson)
endif()
if(NOT NURI_YYJSON_TARGET)
  message(FATAL_ERROR "yyjson package was found but no target was available. Tried: yyjson::yyjson, yyjson, unofficial::yyjson::yyjson")
endif()
target_link_libraries(nuri_lib PUBLIC ${NURI_YYJSON_TARGET})
set(NURI_UNORDERED_DENSE_TARGET "")
if(TARGET unordered_dense::unordered_dense)
  set(NURI_UNORDERED_DENSE_TARGET unordered_dense::unordered_dense)
elseif(TARGET ankerl::unordered_dense)
  set(NURI_UNORDERED_DENSE_TARGET ankerl::unordered_dense)
elseif(TARGET unordered_dense)
  set(NURI_UNORDERED_DENSE_TARGET unordered_dense)
endif()
if(NOT NURI_UNORDERED_DENSE_TARGET)
  message(FATAL_ERROR "unordered-dense package was found but no target was available. Tried: unordered_dense::unordered_dense, ankerl::unordered_dense, unordered_dense")
endif()
target_link_libraries(nuri_lib PUBLIC ${NURI_UNORDERED_DENSE_TARGET})

set(NURI_HARFBUZZ_TARGET "")
if(TARGET harfbuzz::harfbuzz)
  set(NURI_HARFBUZZ_TARGET harfbuzz::harfbuzz)
elseif(TARGET HarfBuzz::harfbuzz)
  set(NURI_HARFBUZZ_TARGET HarfBuzz::harfbuzz)
elseif(TARGET HarfBuzz::HarfBuzz)
  set(NURI_HARFBUZZ_TARGET HarfBuzz::HarfBuzz)
elseif(TARGET harfbuzz)
  set(NURI_HARFBUZZ_TARGET harfbuzz)
endif()
if(NOT NURI_HARFBUZZ_TARGET)
  message(FATAL_ERROR "Failed to resolve a HarfBuzz target. Ensure vcpkg harfbuzz is installed. Tried: harfbuzz::harfbuzz, HarfBuzz::harfbuzz, HarfBuzz::HarfBuzz, harfbuzz")
endif()
target_link_libraries(nuri_lib PUBLIC ${NURI_HARFBUZZ_TARGET})

if(TARGET imgui::imgui)
  target_link_libraries(nuri_lib PUBLIC imgui::imgui)
elseif(TARGET imgui)
  target_link_libraries(nuri_lib PUBLIC imgui)
endif()
target_link_libraries(nuri_lib PUBLIC implot::implot)

if(NURI_BUILD_SHARED)
  target_compile_definitions(nuri_lib PUBLIC NURI_SHARED)
  target_compile_definitions(nuri_lib PRIVATE NURI_EXPORTS)
endif()

set_source_files_properties(
  nuri/resources/stb_image.cpp
  ${CMAKE_SOURCE_DIR}/vendor/im-guizmo/ImGuizmo.cpp
  PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

set_source_files_properties(
  ${CMAKE_SOURCE_DIR}/vendor/im-guizmo/ImGuizmo.cpp
  PROPERTIES COMPILE_DEFINITIONS IMGUI_DEFINE_MATH_OPERATORS
)

target_precompile_headers(nuri_lib PRIVATE nuri/pch.h)

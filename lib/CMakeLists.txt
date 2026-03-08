cmake_minimum_required(VERSION 3.27)

project(nuri_renderer)

set(NURI_LIBRARY_TYPE STATIC)
if(NURI_BUILD_SHARED)
  set(NURI_LIBRARY_TYPE SHARED)
endif()

set(NURI_RENDERER_SOURCES
  nuri/core/application.cpp
  nuri/core/event_manager.cpp
  nuri/core/input_system.cpp
  nuri/core/layer_stack.cpp
  nuri/core/log.cpp
  nuri/core/runtime_config.cpp
  nuri/gfx/debug_draw_3d.cpp
  nuri/gfx/layers/debug_layer.cpp
  nuri/gfx/layers/opaque_layer.cpp
  nuri/gfx/layers/skybox_layer.cpp
  nuri/gfx/render_graph/render_graph.cpp
  nuri/gfx/render_graph/render_graph_telemetry.cpp
  nuri/gfx/renderer.cpp
  nuri/gfx/shader.cpp
  nuri/platform/glfw_window.cpp
  nuri/platform/lvk_gpu_device.cpp
  nuri/platform/minilog_log.cpp
  nuri/resources/gpu/geometry_pool.cpp
  nuri/resources/gpu/material.cpp
  nuri/resources/gpu/model.cpp
  nuri/resources/gpu/resource_manager.cpp
  nuri/resources/gpu/texture.cpp
  nuri/resources/mesh_importer_assimp.cpp
  nuri/resources/stb_image.cpp
  nuri/resources/storage/font/nfont_binary_codec.cpp
  nuri/resources/storage/mesh/mesh_binary_codec.cpp
  nuri/resources/storage/mesh/mesh_binary_serializer.cpp
  nuri/resources/storage/mesh/mesh_cache_utils.cpp
  nuri/resources/storage/mesh/mesh_cache_writer.cpp
  nuri/scene/camera.cpp
  nuri/scene/camera_controller.cpp
  nuri/scene/camera_system.cpp
  nuri/scene/render_scene.cpp
  nuri/text/detail/font_manager.cpp
  nuri/text/detail/text_layouter.cpp
  nuri/text/detail/text_layer_2d.cpp
  nuri/text/detail/text_layer_3d.cpp
  nuri/text/detail/text_renderer.cpp
  nuri/text/detail/text_shaper.cpp
  nuri/text/detail/text_system.cpp
  nuri/utils/env_utils.cpp
  nuri/utils/fsp_counter.cpp
)

add_library(nuri_renderer ${NURI_LIBRARY_TYPE}
  ${NURI_RENDERER_SOURCES}
)

if(MSVC)
  target_compile_options(nuri_renderer PRIVATE /arch:AVX2)
else()
  target_compile_options(nuri_renderer PRIVATE -march=native)
endif()

find_package(Vulkan REQUIRED)

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

target_include_directories(nuri_renderer
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)

target_compile_features(nuri_renderer PUBLIC cxx_std_20)

find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
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

  target_compile_definitions(nuri_renderer PUBLIC NURI_PROFILING=1 TRACY_ENABLE=1)
  target_link_libraries(nuri_renderer PUBLIC ${NURI_TRACY_CLIENT_TARGET})
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

if(NURI_WITH_TRACY_GPU_DRAW_ZONES)
  if(NURI_WITH_TRACY AND TARGET LVKVulkan)
    target_compile_definitions(LVKVulkan PUBLIC "LVK_WITH_TRACY_GPU_DRAW_ZONES=1")
  else()
    message(WARNING "NURI_WITH_TRACY_GPU_DRAW_ZONES=ON requires NURI_WITH_TRACY=ON and target LVKVulkan; per-draw Tracy GPU zones remain disabled.")
  endif()
endif()

target_link_libraries(nuri_renderer
  PUBLIC
    glm::glm
    assimp::assimp
    LVKLibrary
)

set(NURI_KTX_TARGET "")
if(TARGET KTX::ktx)
  set(NURI_KTX_TARGET KTX::ktx)
elseif(TARGET ktx)
  set(NURI_KTX_TARGET ktx)
endif()
if(NOT NURI_KTX_TARGET)
  message(FATAL_ERROR "ktx package was found but no target was available. Tried: KTX::ktx, ktx")
endif()
target_link_libraries(nuri_renderer PRIVATE ${NURI_KTX_TARGET})

set(NURI_MESHOPT_TARGET "")
if(TARGET meshoptimizer::meshoptimizer)
  set(NURI_MESHOPT_TARGET meshoptimizer::meshoptimizer)
elseif(TARGET meshoptimizer)
  set(NURI_MESHOPT_TARGET meshoptimizer)
endif()
if(NOT NURI_MESHOPT_TARGET)
  message(FATAL_ERROR "meshoptimizer package was found but no target was available. Tried: meshoptimizer::meshoptimizer, meshoptimizer")
endif()
target_link_libraries(nuri_renderer PUBLIC ${NURI_MESHOPT_TARGET})

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
target_link_libraries(nuri_renderer PUBLIC ${NURI_YYJSON_TARGET})

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
target_link_libraries(nuri_renderer PUBLIC ${NURI_UNORDERED_DENSE_TARGET})

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
target_link_libraries(nuri_renderer PUBLIC ${NURI_HARFBUZZ_TARGET})

if(NURI_BUILD_SHARED)
  target_compile_definitions(nuri_renderer PUBLIC NURI_SHARED)
  target_compile_definitions(nuri_renderer PRIVATE NURI_EXPORTS)
endif()

set_source_files_properties(
  nuri/resources/stb_image.cpp
  PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

target_precompile_headers(nuri_renderer PRIVATE nuri/pch.h)

set_target_properties(nuri_renderer PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${_cfg}" _cfg_upper)
  set_target_properties(nuri_renderer PROPERTIES
    "ARCHIVE_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
    "LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
    "RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}/lib"
  )
endforeach()

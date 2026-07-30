set(NURI_LIBRARY_TYPE STATIC)
if(NURI_BUILD_SHARED)
  set(NURI_LIBRARY_TYPE SHARED)
endif()

set(NURI_RENDERER_SOURCES
  nuri/core/application.cpp
  nuri/core/event_manager.cpp
  nuri/core/input_system.cpp
  nuri/core/log.cpp
  nuri/core/runtime_config.cpp
  nuri/core/thread_priority.cpp
  nuri/gfx/ddgi/ddgi_atlas.cpp
  nuri/gfx/ddgi/ddgi_coverage.cpp
  nuri/gfx/ddgi/ddgi_dirty_regions.cpp
  nuri/gfx/ddgi/ddgi_feature.cpp
  nuri/gfx/ddgi/ddgi_scheduler.cpp
  nuri/gfx/gpu_acceleration_structure.cpp
  nuri/gfx/ddgi/ddgi_types.cpp
  nuri/gfx/frame/external_temporal_provider.cpp
  nuri/gfx/frame/temporal_frame_service.cpp
  nuri/gfx/frame/history_registry.cpp
  nuri/gfx/renderers/debug_renderer.cpp
  nuri/gfx/renderers/opaque_renderer.cpp
  nuri/gfx/renderers/scene_draw_database.cpp
  nuri/gfx/renderers/shadow_renderer.cpp
  nuri/gfx/pipeline/features/composite_feature.cpp
  nuri/gfx/pipeline/features/gtao_feature.cpp
  nuri/gfx/pipeline/features/spatial_aa_feature.cpp
  nuri/gfx/pipeline/features/skybox_feature.cpp
  nuri/gfx/pipeline/features/temporal_aa_feature.cpp
  nuri/gfx/pipeline/features/reference_taa_feature.cpp
  nuri/gfx/pipeline/default_render_pipeline.cpp
  nuri/gfx/pipeline/render_pipeline.cpp
  nuri/gfx/pipeline/providers/frame_composition_provider.cpp
  nuri/gfx/pipeline/providers/material_table_gpu_provider.cpp
  nuri/gfx/pipeline/providers/scene_lighting_provider.cpp
  nuri/gfx/ray_tracing/ray_tracing_scene.cpp
  nuri/gfx/renderers/transmission_renderer.cpp
  nuri/gfx/renderers/transparent_renderer.cpp
  nuri/gfx/render_graph/render_graph.cpp
  nuri/gfx/render_graph/render_graph_runtime.cpp
  nuri/gfx/render_graph/render_graph_telemetry.cpp
  nuri/gfx/sim/animation_gpu_services.cpp
  nuri/gfx/sim/animation_scene_frame_provider.cpp
  nuri/gfx/visibility/visibility.cpp
  nuri/gfx/renderer.cpp
  nuri/gfx/shader.cpp
  nuri/platform/glfw_window.cpp
  nuri/resources/gpu/geometry_pool.cpp
  nuri/resources/gpu/material.cpp
  nuri/resources/gpu/model.cpp
  nuri/resources/gpu/resource_manager.cpp
  nuri/resources/gpu/texture.cpp
  nuri/resources/async/asset_cpu_scheduler.cpp
  nuri/resources/async/scene_asset_preparation.cpp
  nuri/resources/async/asset_system.cpp
  nuri/resources/detail/gltf_buffer_utils.cpp
  nuri/resources/detail/gltf_json_utils.cpp
  nuri/resources/mesh_importer_assimp.cpp
  nuri/resources/scene_importer.cpp
    nuri/resources/storage/cache_utils.cpp
  nuri/resources/storage/material/material_binary_serializer.cpp
  nuri/resources/storage/mesh/mesh_binary_serializer.cpp
  nuri/resources/storage/texture/dds_texture_pack.cpp
    nuri/resources/storage/texture/texture_artifact_builder.cpp
    nuri/resources/storage/texture/texture_cache_utils.cpp
    nuri/resources/storage/texture/texture_processing.cpp
  nuri/scene_runtime/scene_runtime_bindings.cpp
  nuri/scene_runtime/scene_runtime_host.cpp
  nuri/scene/camera.cpp
  nuri/scene/camera_controller.cpp
  nuri/scene/camera_system.cpp
  nuri/scene/ddgi_volume.cpp
  nuri/scene/render_scene.cpp
  nuri/scene/scene_animation_player.cpp
  nuri/scene/scene_graph.cpp
  nuri/sim/fixed_step_clock.cpp
  nuri/sim/animation_pose_simulation.cpp
  nuri/sim/backends/animation_pose_simulation_backend.cpp
  nuri/sim/simulation_registry.cpp
  nuri/sim/simulation_scheduler.cpp
  nuri/text/detail/font_manager.cpp
  nuri/text/detail/text_layouter.cpp
  nuri/text/detail/text_renderer.cpp
  nuri/text/detail/text_shaper.cpp
  nuri/text/detail/text_system.cpp
  nuri/utils/env_utils.cpp
)

add_library(nuri_renderer ${NURI_LIBRARY_TYPE}
  ${NURI_RENDERER_SOURCES}
)
nuri_configure_first_party_target(nuri_renderer)
nuri_register_artifact_target(nuri_renderer)

set(NURI_FIDELITYFX_SDK_DIR "${CMAKE_SOURCE_DIR}/external/fidelityfx-sdk")
set(NURI_FIDELITYFX_REQUIRED_FILES
  ffx-api/include/ffx_api/ffx_api.h
  ffx-api/include/ffx_api/ffx_upscale.h
  ffx-api/include/ffx_api/vk/ffx_api_vk.h
  PrebuiltSignedDLL/amd_fidelityfx_vk.dll
  PrebuiltSignedDLL/amd_fidelityfx_vk.lib
  sdk/LICENSE.txt
)
if(NURI_WITH_FSR31)
  foreach(_required_file IN LISTS NURI_FIDELITYFX_REQUIRED_FILES)
    if(NOT EXISTS "${NURI_FIDELITYFX_SDK_DIR}/${_required_file}")
      message(FATAL_ERROR
        "NURI_WITH_FSR31 requires the pinned FidelityFX SDK v1.1.4 at "
        "${NURI_FIDELITYFX_SDK_DIR}; missing ${_required_file}. "
        "See external/fidelityfx-sdk.pin.json.")
    endif()
  endforeach()
  target_compile_definitions(nuri_renderer PRIVATE
    NURI_WITH_FSR31=1
    NURI_FSR31_DEPENDENCY_PRESENT=1
  )
else()
  target_compile_definitions(nuri_renderer PRIVATE
    NURI_WITH_FSR31=0
    NURI_FSR31_DEPENDENCY_PRESENT=0
  )
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)

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

if(NURI_WITH_LOGGING)
  target_compile_definitions(nuri_renderer PUBLIC NURI_LOGGING=1)
endif()

if(NURI_WITH_ASSERTS)
  target_compile_definitions(nuri_renderer PUBLIC NURI_ASSERTS=1)
endif()

if(WIN32)
  target_compile_definitions(nuri_renderer PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(Ktx CONFIG QUIET)
if(NOT Ktx_FOUND)
  find_package(ktx CONFIG REQUIRED)
endif()
find_package(yyjson CONFIG REQUIRED)
find_package(unordered_dense CONFIG REQUIRED)
find_package(harfbuzz CONFIG QUIET)
find_package(HarfBuzz CONFIG QUIET)

add_library(nuri_asset_codecs STATIC
  nuri/resources/stb_image.cpp
  nuri/resources/stb_image_resize.cpp
  nuri/resources/storage/font/nfont_binary_codec.cpp
)
nuri_configure_first_party_target(nuri_asset_codecs)
set_target_properties(nuri_asset_codecs PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(nuri_asset_codecs
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Stb_INCLUDE_DIR}
)
target_link_libraries(nuri_renderer PRIVATE nuri_asset_codecs)

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

set(NVRHI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(NVRHI_INSTALL OFF CACHE BOOL "" FORCE)
set(NVRHI_WITH_VALIDATION ${NURI_WITH_ASSERTS} CACHE BOOL "" FORCE)
set(NVRHI_WITH_VULKAN ON CACHE BOOL "" FORCE)
set(NVRHI_WITH_DX11 OFF CACHE BOOL "" FORCE)
set(NVRHI_WITH_DX12 OFF CACHE BOOL "" FORCE)
set(NVRHI_WITH_NVAPI OFF CACHE BOOL "" FORCE)
set(NVRHI_WITH_AFTERMATH OFF CACHE BOOL "" FORCE)
set(NVRHI_WITH_RTXMU OFF CACHE BOOL "" FORCE)

set(NVRHI_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/nvrhi")
if(NURI_NVRHI_SOURCE_DIR)
  set(NVRHI_SOURCE_DIR "${NURI_NVRHI_SOURCE_DIR}")
endif()
if(NOT EXISTS "${NVRHI_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "NVRHI submodule is missing at: ${NVRHI_SOURCE_DIR}. Run: git submodule update --init --recursive")
endif()

set(NVRHI_ACTIVE_SOURCE_DIR "${NVRHI_SOURCE_DIR}")
set(NVRHI_PATCH_FILE "${CMAKE_SOURCE_DIR}/patches/nvrhi/framebuffer-resolve.patch")
if(EXISTS "${NVRHI_PATCH_FILE}" AND NOT NURI_NVRHI_SOURCE_DIR)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NVRHI_PATCH_FILE}")
  find_package(Git REQUIRED)

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --verify HEAD
    WORKING_DIRECTORY "${NVRHI_SOURCE_DIR}"
    OUTPUT_VARIABLE NVRHI_SOURCE_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT NVRHI_SOURCE_REV)
    set(NVRHI_SOURCE_REV "unknown")
  endif()

  file(SHA256 "${NVRHI_PATCH_FILE}" NVRHI_PATCH_HASH)
  set(NVRHI_PATCHED_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/nvrhi-src-patched")
  set(NVRHI_PATCH_STAMP "${NVRHI_PATCHED_SOURCE_DIR}/.nuri-patch-stamp")
  string(CONCAT NVRHI_EXPECTED_PATCH_STAMP
    "driver=git-apply-ceiling-v1\n"
    "source=${NVRHI_SOURCE_REV}\n"
    "patch=${NVRHI_PATCH_HASH}\n"
  )
  set(NVRHI_CURRENT_PATCH_STAMP "")
  if(EXISTS "${NVRHI_PATCH_STAMP}")
    file(READ "${NVRHI_PATCH_STAMP}" NVRHI_CURRENT_PATCH_STAMP)
  endif()

  set(NVRHI_REFRESH_PATCHED_SOURCE OFF)
  if(NOT EXISTS "${NVRHI_PATCHED_SOURCE_DIR}/CMakeLists.txt")
    set(NVRHI_REFRESH_PATCHED_SOURCE ON)
  elseif(NOT "${NVRHI_CURRENT_PATCH_STAMP}" STREQUAL "${NVRHI_EXPECTED_PATCH_STAMP}")
    set(NVRHI_REFRESH_PATCHED_SOURCE ON)
  endif()

  if(NVRHI_REFRESH_PATCHED_SOURCE)
    file(REMOVE_RECURSE "${NVRHI_PATCHED_SOURCE_DIR}")
    file(MAKE_DIRECTORY "${NVRHI_PATCHED_SOURCE_DIR}")
    file(COPY "${NVRHI_SOURCE_DIR}/" DESTINATION "${NVRHI_PATCHED_SOURCE_DIR}" PATTERN ".git" EXCLUDE)

    set(NVRHI_GIT_APPLY_ENV "GIT_CEILING_DIRECTORIES=${CMAKE_SOURCE_DIR}")
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}" -E env "${NVRHI_GIT_APPLY_ENV}"
        "${GIT_EXECUTABLE}" apply --check --whitespace=nowarn
        "${NVRHI_PATCH_FILE}"
      WORKING_DIRECTORY "${NVRHI_PATCHED_SOURCE_DIR}"
      RESULT_VARIABLE NVRHI_PATCH_CHECK_RESULT
      OUTPUT_VARIABLE NVRHI_PATCH_CHECK_OUTPUT
      ERROR_VARIABLE NVRHI_PATCH_CHECK_ERROR
    )
    if(NOT NVRHI_PATCH_CHECK_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to check NVRHI patch ${NVRHI_PATCH_FILE}:\n${NVRHI_PATCH_CHECK_OUTPUT}\n${NVRHI_PATCH_CHECK_ERROR}")
    endif()

    execute_process(
      COMMAND
        "${CMAKE_COMMAND}" -E env "${NVRHI_GIT_APPLY_ENV}"
        "${GIT_EXECUTABLE}" apply --whitespace=nowarn
        "${NVRHI_PATCH_FILE}"
      WORKING_DIRECTORY "${NVRHI_PATCHED_SOURCE_DIR}"
      RESULT_VARIABLE NVRHI_PATCH_RESULT
      OUTPUT_VARIABLE NVRHI_PATCH_OUTPUT
      ERROR_VARIABLE NVRHI_PATCH_ERROR
    )
    if(NOT NVRHI_PATCH_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to apply NVRHI patch ${NVRHI_PATCH_FILE}:\n${NVRHI_PATCH_OUTPUT}\n${NVRHI_PATCH_ERROR}")
    endif()

    file(WRITE "${NVRHI_PATCH_STAMP}" "${NVRHI_EXPECTED_PATCH_STAMP}")
  endif()

  set(NVRHI_ACTIVE_SOURCE_DIR "${NVRHI_PATCHED_SOURCE_DIR}")
endif()

add_subdirectory("${NVRHI_ACTIVE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/nvrhi-build")

target_link_libraries(nuri_renderer
  PUBLIC
    glm::glm
  PRIVATE
    assimp::assimp
    glfw
    nvrhi
    nvrhi_vk
    glslang::glslang
    glslang::SPIRV
    glslang::glslang-default-resource-limits
    volk::volk
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
target_link_libraries(nuri_renderer PRIVATE ${NURI_MESHOPT_TARGET})

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
target_link_libraries(nuri_renderer PRIVATE ${NURI_HARFBUZZ_TARGET})

if(NURI_BUILD_SHARED)
  target_compile_definitions(nuri_renderer PUBLIC NURI_SHARED)
  target_compile_definitions(nuri_renderer PRIVATE NURI_EXPORTS)
endif()

function(nuri_configure_backend_adapter target_name)
  target_include_directories(${target_name}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${Vulkan_INCLUDE_DIRS}
  )

  target_compile_features(${target_name} PUBLIC cxx_std_20)

  if(NURI_WITH_LOGGING)
    target_compile_definitions(${target_name} PUBLIC NURI_LOGGING=1)
  endif()

  if(NURI_WITH_ASSERTS)
    target_compile_definitions(${target_name} PUBLIC NURI_ASSERTS=1)
  endif()

  if(WIN32)
    target_compile_definitions(${target_name} PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()

  if(NURI_WITH_TRACY)
    target_compile_definitions(${target_name} PUBLIC NURI_PROFILING=1 TRACY_ENABLE=1)
    target_link_libraries(${target_name} PRIVATE ${NURI_TRACY_CLIENT_TARGET})
  endif()

  if(NURI_BUILD_SHARED)
    target_compile_definitions(${target_name} PUBLIC NURI_SHARED)
    target_compile_definitions(${target_name} PRIVATE NURI_EXPORTS)
  endif()

  target_link_libraries(${target_name}
    PRIVATE
      glm::glm
      ${NURI_MESHOPT_TARGET}
      ${NURI_YYJSON_TARGET}
      ${NURI_UNORDERED_DENSE_TARGET}
  )

endfunction()

add_library(nuri_nvrhi_backend OBJECT
  nuri/platform/nvrhi_gpu_device.cpp
)
nuri_configure_first_party_target(nuri_nvrhi_backend)
nuri_configure_backend_adapter(nuri_nvrhi_backend)
target_link_libraries(nuri_nvrhi_backend
  PRIVATE
    glfw
    glslang::glslang
    nvrhi
    nvrhi_vk
    volk::volk
)

target_sources(nuri_renderer
  PRIVATE
    $<TARGET_OBJECTS:nuri_nvrhi_backend>
)

set_source_files_properties(
  nuri/resources/stb_image.cpp
  nuri/resources/stb_image_resize.cpp
  PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

target_precompile_headers(nuri_renderer PRIVATE nuri/pch.h)

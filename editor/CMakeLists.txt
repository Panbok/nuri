cmake_minimum_required(VERSION 3.27)

project(nuri_editor)

set(NURI_EDITOR_SOURCES
  ${CMAKE_SOURCE_DIR}/lib/nuri/resources/stb_image.cpp
  ${CMAKE_SOURCE_DIR}/lib/nuri/resources/storage/font/nfont_binary_codec.cpp
  ${CMAKE_SOURCE_DIR}/lib/nuri/utils/fsp_counter.cpp
  src/bakery/bakery_system.cpp
  src/bakery/brdf_lut_baker.cpp
  src/bakery/envmap_prefilter_baker.cpp
  src/gfx/imgui_gpu_renderer.cpp
  src/main.cpp
  src/platform/imgui_glfw_platform.cpp
  src/resources/storage/font/nfont_compiler.cpp
  src/ui/camera_controller_widget.cpp
  src/ui/editor_layer.cpp
  src/ui/imgui_editor.cpp
  src/ui/imgui_gizmo_controller.cpp
  src/ui/linear_graph.cpp
  src/ui/platform/file_dialog_widget_windows.cpp
  vendor/im-guizmo/ImGuizmo.cpp
)

add_executable(nuri_editor
  ${NURI_EDITOR_SOURCES}
)

set_target_properties(nuri_editor PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${_cfg}" _cfg_upper)
  set_target_properties(nuri_editor PROPERTIES
    "RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}" "${CMAKE_BINARY_DIR}"
  )
endforeach()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
  set_property(TARGET nuri_editor PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endif()

if(MSVC)
  target_compile_options(nuri_editor PRIVATE /arch:AVX2)
else()
  target_compile_options(nuri_editor PRIVATE -march=native)
endif()

find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(Ktx CONFIG QUIET)
if(NOT Ktx_FOUND)
  find_package(ktx CONFIG REQUIRED)
endif()

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

target_include_directories(nuri_editor
  PRIVATE
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/im-guizmo
)

target_link_libraries(nuri_editor
  PRIVATE
    nuri_renderer
    implot::implot
)

if(TARGET imgui::imgui)
  target_link_libraries(nuri_editor PRIVATE imgui::imgui)
elseif(TARGET imgui)
  target_link_libraries(nuri_editor PRIVATE imgui)
endif()

set(NURI_MSDF_ATLAS_TARGET "")
if(TARGET msdf-atlas-gen::msdf-atlas-gen)
  set(NURI_MSDF_ATLAS_TARGET msdf-atlas-gen::msdf-atlas-gen)
elseif(TARGET msdf-atlas-gen)
  set(NURI_MSDF_ATLAS_TARGET msdf-atlas-gen)
endif()
if(NOT NURI_MSDF_ATLAS_TARGET)
  message(FATAL_ERROR "msdf-atlas-gen target was not found. Tried: msdf-atlas-gen::msdf-atlas-gen, msdf-atlas-gen")
endif()
target_link_libraries(nuri_editor PRIVATE ${NURI_MSDF_ATLAS_TARGET})

set(NURI_KTX_TARGET "")
if(TARGET KTX::ktx)
  set(NURI_KTX_TARGET KTX::ktx)
elseif(TARGET ktx)
  set(NURI_KTX_TARGET ktx)
endif()
if(NOT NURI_KTX_TARGET)
  message(FATAL_ERROR "ktx package was found but no target was available. Tried: KTX::ktx, ktx")
endif()
target_link_libraries(nuri_editor PRIVATE ${NURI_KTX_TARGET})

target_compile_features(nuri_editor PRIVATE cxx_std_20)

set_source_files_properties(
  ${CMAKE_SOURCE_DIR}/lib/nuri/resources/stb_image.cpp
  vendor/im-guizmo/ImGuizmo.cpp
  PROPERTIES
    SKIP_PRECOMPILE_HEADERS ON
)

set_source_files_properties(
  vendor/im-guizmo/ImGuizmo.cpp
  PROPERTIES
    COMPILE_DEFINITIONS IMGUI_DEFINE_MATH_OPERATORS
)

target_precompile_headers(nuri_editor PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include/nuri/editor_pch.h"
)

if(WIN32 AND NURI_BUILD_SHARED)
  add_custom_command(TARGET nuri_editor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:nuri_renderer>
      $<TARGET_FILE_DIR:nuri_editor>
  )
endif()

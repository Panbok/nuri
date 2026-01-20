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

target_include_directories(nuri_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Vulkan_INCLUDE_DIRS}
)

target_compile_features(nuri_lib PUBLIC cxx_std_20)

find_package(VulkanHeaders CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(nuri_lib
  PUBLIC
    Vulkan::Headers
    SDL3::SDL3
    imgui::imgui
    glm::glm
)

if(NURI_BUILD_SHARED)
  target_compile_definitions(nuri_lib PUBLIC NURI_SHARED)
  target_compile_definitions(nuri_lib PRIVATE NURI_EXPORTS)
endif()

target_precompile_headers(nuri_lib PRIVATE nuri/pch.h)


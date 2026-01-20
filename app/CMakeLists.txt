cmake_minimum_required(VERSION 3.27)

project(nuri_app)

file(GLOB_RECURSE APP_SOURCES src/*.cpp src/*.h)

add_executable(nuri ${APP_SOURCES})

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set_property(TARGET nuri PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endif()

if(MSVC)
    target_compile_options(nuri PRIVATE /arch:AVX2)
else()
    target_compile_options(nuri PRIVATE -march=native)
endif()

target_link_libraries(nuri
  PRIVATE
    nuri_lib
)

target_include_directories(nuri
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(nuri PRIVATE cxx_std_20)

if(WIN32 AND NURI_BUILD_SHARED)
  add_custom_command(TARGET nuri POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:nuri_lib>
      $<TARGET_FILE_DIR:nuri>
  )
endif()

cmake_minimum_required(VERSION 3.27)

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

project(nuri LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")

if(DEFINED CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE MATCHES "debug")
    add_compile_definitions($<$<CONFIG:Debug>:NURI_DEBUG>)
endif()

set(SAN_FLAGS "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(WIN32)
        set(SAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
    else()
        set(SAN_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
    endif()
endif()

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # ASan on Windows requires the release CRT (msvcrt), not the debug CRT (msvcrtd).
    if(NURI_USE_STATIC_CRT)
        add_compile_options(-fms-runtime-lib=static)
        add_link_options(-fms-runtime-lib=static)
    else()
        add_compile_options(-fms-runtime-lib=dll)
        add_link_options(-fms-runtime-lib=dll)
    endif()
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
    add_compile_options($<$<CONFIG:Debug>:-U_DEBUG>)
    set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)

    # CMake's Windows-Clang defaults inject the Debug CRT via /defaultlib directives
    # (commonly `msvcrtd`). ASan on Windows does not support the Debug CRT.
    # Suppress the debug CRT and explicitly pull the release CRT instead.
    if(NURI_USE_STATIC_CRT)
        # Static CRT variants.
        add_link_options("LINKER:/NODEFAULTLIB:libcmtd")
        add_link_options("LINKER:/DEFAULTLIB:libcmt")
    else()
        # Dynamic CRT variants. Clang+ASan must not use any debug CRT/STL libs.
        add_link_options("LINKER:/NODEFAULTLIB:msvcrtd")
        add_link_options("LINKER:/NODEFAULTLIB:ucrtd")
        add_link_options("LINKER:/NODEFAULTLIB:vcruntimed")
        add_link_options("LINKER:/NODEFAULTLIB:msvcprtd")
        # Note: don't force msvcp140(.lib) here; some setups only provide msvcprt.lib.

        add_link_options("LINKER:/DEFAULTLIB:msvcrt")
        add_link_options("LINKER:/DEFAULTLIB:ucrt")
        add_link_options("LINKER:/DEFAULTLIB:vcruntime")
        add_link_options("LINKER:/DEFAULTLIB:msvcprt")
    endif()
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi -W4 /FC")
    if(SAN_FLAGS)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SAN_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${SAN_FLAGS}")
    endif()
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4013 /arch:AVX2")  # implicit function declaration

else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SAN_FLAGS} -g")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${SAN_FLAGS}")

    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${SAN_FLAGS} -g")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} ${SAN_FLAGS}")
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-function-declaration -march=native")
endif()

option(NURI_BUILD_SHARED "Build nuri as a shared library" ON)
option(NURI_WITH_TRACY "Enable Tracy CPU+GPU profiling" OFF)

add_subdirectory(lib)
add_subdirectory(app)

cmake_minimum_required(VERSION 3.27)

project(nuri LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")
add_compile_definitions($<$<CONFIG:Debug>:NURI_DEBUG>)

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
    add_compile_options(-fms-runtime-lib=dll)
    add_link_options(-fms-runtime-lib=dll)
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
    add_compile_options($<$<CONFIG:Debug>:-U_DEBUG>)
    set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi -W4 /FC")
    if(SAN_FLAGS)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SAN_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${SAN_FLAGS}")
    endif()
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4013 /arch:AVX2")  # implicit function declaration

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # ASan on Windows does not support the debug CRT.
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SAN_FLAGS} -g")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${SAN_FLAGS}")

    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${SAN_FLAGS} -g")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} ${SAN_FLAGS}")
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-function-declaration -march=native")
endif()

option(NURI_BUILD_SHARED "Build nuri as a shared library" ON)

add_subdirectory(lib)
add_subdirectory(app)

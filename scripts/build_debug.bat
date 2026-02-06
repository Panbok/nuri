@echo off
setlocal

if "%VCPKG_ROOT%"=="" (
  echo VCPKG_ROOT is not set. Point it at your vcpkg root.
  exit /b 1
)

where ninja >nul 2>nul
if %errorlevel% neq 0 (
  echo Ninja not found. Install ninja or run CMake with a different generator.
  exit /b 1
)

set "TOOLCHAIN=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"

call "%~dp0bootstrap_lightweightvk.bat"
if errorlevel 1 exit /b 1

cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Debug ^
  -DCMAKE_C_COMPILER=clang ^
  -DCMAKE_CXX_COMPILER=clang++ ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DVCPKG_APPLOCAL_DEPS=OFF ^
  -DVCPKG_BUILD_TYPE=release ^
  -DNURI_BUILD_SHARED=ON ^
  -DNURI_WITH_TRACY=ON
if errorlevel 1 exit /b 1

cmake --build build
if errorlevel 1 exit /b 1

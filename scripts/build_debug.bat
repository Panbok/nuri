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
set "BUILD_TESTS=%NURI_BUILD_TESTS%"
if "%BUILD_TESTS%"=="" set "BUILD_TESTS=OFF"
set "BUILD_EDITOR=%NURI_BUILD_EDITOR%"
if "%BUILD_EDITOR%"=="" set "BUILD_EDITOR=ON"
set "MANIFEST_FEATURES=%VCPKG_MANIFEST_FEATURES%"
if /I "%BUILD_EDITOR%"=="ON" (
  if "%MANIFEST_FEATURES%"=="" (
    set "MANIFEST_FEATURES=editor"
  ) else (
    echo ,%MANIFEST_FEATURES%, | findstr /I /C:",editor," >nul
    if errorlevel 1 set "MANIFEST_FEATURES=%MANIFEST_FEATURES%,editor"
  )
)
set "MANIFEST_FEATURES_ARG="
if not "%MANIFEST_FEATURES%"=="" set "MANIFEST_FEATURES_ARG=-DVCPKG_MANIFEST_FEATURES=%MANIFEST_FEATURES%"

call "%~dp0bootstrap_lightweightvk.bat"
if errorlevel 1 exit /b 1

cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Debug ^
  -DCMAKE_C_COMPILER=clang ^
  -DCMAKE_CXX_COMPILER=clang++ ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DCMAKE_CXX_FLAGS="-DLVK_WITH_TRACY_GPU_DRAW_ZONES=1" ^
  -DVCPKG_APPLOCAL_DEPS=OFF ^
  %MANIFEST_FEATURES_ARG% ^
  -DVCPKG_BUILD_TYPE=release ^
  -DNURI_BUILD_TESTS="%BUILD_TESTS%" ^
  -DNURI_BUILD_EDITOR="%BUILD_EDITOR%" ^
  -DNURI_BUILD_SHARED=ON ^
  -DNURI_WITH_TRACY=ON
if errorlevel 1 exit /b 1

cmake --build build
if errorlevel 1 exit /b 1

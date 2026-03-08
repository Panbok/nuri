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
set "BUILD_DIR=%CD%\build_release"
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

if exist "%BUILD_DIR%\build.ninja" (
  findstr /C:"--dependent-lib=msvcrt" "%BUILD_DIR%\build.ninja" >nul 2>nul
  if %errorlevel% equ 0 (
    echo Stale runtime flags detected in %BUILD_DIR%\build.ninja
    echo Delete %BUILD_DIR% and run this script again.
    exit /b 1
  )
)

call "%~dp0bootstrap_lightweightvk.bat"
if errorlevel 1 exit /b 1

cmake -S . -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=clang ^
  -DCMAKE_CXX_COMPILER=clang++ ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-static ^
  %MANIFEST_FEATURES_ARG% ^
  -DVCPKG_BUILD_TYPE=release ^
  -DNURI_BUILD_TESTS="%BUILD_TESTS%" ^
  -DNURI_BUILD_EDITOR="%BUILD_EDITOR%" ^
  -DNURI_BUILD_SHARED=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config Release

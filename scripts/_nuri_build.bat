@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=%~1"
set "PROFILE=%~2"

if "%MODE%"=="" goto usage
if "%PROFILE%"=="" goto usage

set "BUILD_APP=OFF"
set "BUILD_EDITOR=OFF"
set "BUILD_TESTS=OFF"
set "BUILD_TARGET="
set "MANIFEST_FEATURES=%VCPKG_MANIFEST_FEATURES%"

if /I "%PROFILE%"=="lib" (
  set "BUILD_TARGET=nuri_renderer"
) else if /I "%PROFILE%"=="app" (
  set "BUILD_APP=ON"
  set "BUILD_TARGET=nuri"
) else if /I "%PROFILE%"=="editor" (
  set "BUILD_EDITOR=ON"
  set "BUILD_TARGET=nuri_editor"
  call :append_manifest_feature editor
) else if /I "%PROFILE%"=="tests" (
  set "BUILD_TESTS=ON"
  call :append_manifest_feature tests
) else (
  goto usage
)

if "%VCPKG_ROOT%"=="" (
  echo VCPKG_ROOT is not set. Point it at your vcpkg root.
  exit /b 1
)

where ninja >nul 2>nul
if %errorlevel% neq 0 (
  echo Ninja not found. Install ninja or run CMake with a different generator.
  exit /b 1
)

for %%i in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fi"
call :set_build_dir "%REPO_ROOT%" "%MODE%" "%PROFILE%"
set "TOOLCHAIN=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
set "MANIFEST_FEATURES_ARG="
if not "%MANIFEST_FEATURES%"=="" set "MANIFEST_FEATURES_ARG=-DVCPKG_MANIFEST_FEATURES=%MANIFEST_FEATURES%"

call "%SCRIPT_DIR%bootstrap_lightweightvk.bat"
if errorlevel 1 exit /b 1

if /I "%MODE%"=="debug" goto configure_debug
if /I "%MODE%"=="release" goto configure_release
goto usage

:configure_debug
cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Debug ^
  -DCMAKE_C_COMPILER=clang ^
  -DCMAKE_CXX_COMPILER=clang++ ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DCMAKE_CXX_FLAGS="-DLVK_WITH_TRACY_GPU_DRAW_ZONES=1" ^
  -DVCPKG_APPLOCAL_DEPS=OFF ^
  %MANIFEST_FEATURES_ARG% ^
  -DVCPKG_BUILD_TYPE=release ^
  -DNURI_BUILD_APP="%BUILD_APP%" ^
  -DNURI_BUILD_EDITOR="%BUILD_EDITOR%" ^
  -DNURI_BUILD_TESTS="%BUILD_TESTS%" ^
  -DNURI_BUILD_SHARED=ON ^
  -DNURI_WITH_TRACY=ON
if errorlevel 1 exit /b 1
goto build_target

:configure_release
cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=clang ^
  -DCMAKE_CXX_COMPILER=clang++ ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-static ^
  %MANIFEST_FEATURES_ARG% ^
  -DVCPKG_BUILD_TYPE=release ^
  -DNURI_BUILD_APP="%BUILD_APP%" ^
  -DNURI_BUILD_EDITOR="%BUILD_EDITOR%" ^
  -DNURI_BUILD_TESTS="%BUILD_TESTS%" ^
  -DNURI_BUILD_SHARED=OFF
if errorlevel 1 exit /b 1

:build_target
if "%BUILD_TARGET%"=="" (
  cmake --build "%BUILD_DIR%"
) else (
  cmake --build "%BUILD_DIR%" --target "%BUILD_TARGET%"
)
exit /b %errorlevel%

:append_manifest_feature
if "%MANIFEST_FEATURES%"=="" (
  set "MANIFEST_FEATURES=%~1"
  exit /b 0
)
echo ,%MANIFEST_FEATURES%, | findstr /I /C:",%~1," >nul
if errorlevel 1 set "MANIFEST_FEATURES=%MANIFEST_FEATURES%,%~1"
exit /b 0

:set_build_dir
if /I "%~2"=="release" (
  set "BUILD_DIR=%~1\build_release\%~3"
  exit /b 0
)
if /I "%~3"=="app" (
  set "BUILD_DIR=%~1\build"
  exit /b 0
)
set "BUILD_DIR=%~1\build_%~3"
exit /b 0

:usage
echo Usage: %~nx0 ^<debug^|release^> ^<lib^|app^|editor^|tests^>
exit /b 1

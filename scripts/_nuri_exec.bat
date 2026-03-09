@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=%~1"
set "PROFILE=%~2"
if "%MODE%"=="" goto usage
if "%PROFILE%"=="" goto usage
shift
shift

for %%i in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fi"
call :set_build_dir "%REPO_ROOT%" "%MODE%" "%PROFILE%"
set "APP="
set "VCPKG_BIN="

if /I "%PROFILE%"=="app" (
  set "APP=%BUILD_DIR%\nuri.exe"
  set "VCPKG_BIN=%BUILD_DIR%\vcpkg_installed\x64-windows\bin"
) else if /I "%PROFILE%"=="editor" (
  set "APP=%BUILD_DIR%\nuri_editor.exe"
  set "VCPKG_BIN=%BUILD_DIR%\vcpkg_installed\x64-windows\bin"
) else (
  goto usage
)

if not exist "%APP%" (
  echo Build output not found: %APP%
  exit /b 1
)

if exist "%VCPKG_BIN%" set "PATH=%VCPKG_BIN%;%PATH%"

for /f "delims=" %%i in ('where clang 2^>nul') do (
  set "CLANG_BIN=%%~dpi"
  goto have_clang
)
:have_clang
if defined CLANG_BIN set "PATH=%CLANG_BIN%;%PATH%"

"%APP%" %*
exit /b %errorlevel%

:set_build_dir
set "BUILD_ROOT=%~1\build\%~2"
if /I "%~2"=="release" set "BUILD_ROOT=%~1\build_release"
set "BUILD_DIR=%BUILD_ROOT%\%~3"
exit /b 0

:usage
echo Usage: %~nx0 ^<debug^|release^> ^<app^|editor^> [args...]
exit /b 1

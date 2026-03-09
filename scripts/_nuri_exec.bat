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
set "BUILD_DIR=%REPO_ROOT%\build\%MODE%\%PROFILE%"
set "APP="
set "VCPKG_BIN="

if /I "%PROFILE%"=="app" (
  set "APP=%BUILD_DIR%\app\nuri.exe"
  set "VCPKG_BIN=%BUILD_DIR%\vcpkg_installed\x64-windows\bin"
) else if /I "%PROFILE%"=="editor" (
  set "APP=%BUILD_DIR%\editor\nuri_editor.exe"
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

:usage
echo Usage: %~nx0 ^<debug^|release^> ^<app^|editor^> [args...]
exit /b 1

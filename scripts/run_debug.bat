@echo off
setlocal

call "%~dp0build_debug.bat"
if errorlevel 1 exit /b 1

set "APP=%CD%\build\app\nuri.exe"
if not exist "%APP%" (
  echo Build output not found: %APP%
  exit /b 1
)

set "VCPKG_BIN=%CD%\build\vcpkg_installed\x64-windows\bin"
if exist "%VCPKG_BIN%" set "PATH=%VCPKG_BIN%;%PATH%"

for /f "delims=" %%i in ('where clang 2^>nul') do (
  set "CLANG_BIN=%%~dpi"
  goto :have_clang
)
:have_clang
if defined CLANG_BIN set "PATH=%CLANG_BIN%;%PATH%"

"%APP%"

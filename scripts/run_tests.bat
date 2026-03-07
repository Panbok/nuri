@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if not "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR%\"

set "MODE=debug"

if /I "%~1"=="debug" (
  shift
) else if /I "%~1"=="release" (
  set "MODE=release"
  shift
) else if not "%~1"=="" (
  if not "%~1:~0,1%"=="-" (
    echo Usage: %~nx0 [debug^|release] [ctest args...]
    exit /b 1
  )
)

if /I "%MODE%"=="debug" (
  call "%SCRIPT_DIR%build_debug.bat"
  if errorlevel 1 exit /b 1
  set "BUILD_DIR=build"
) else (
  call "%SCRIPT_DIR%build_release.bat"
  if errorlevel 1 exit /b 1
  set "BUILD_DIR=build_release"
)

ctest --test-dir "%BUILD_DIR%" --output-on-failure %*
exit /b %errorlevel%

@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=debug"
set "TRACY_MODE="
if /I "%~1"=="debug" (
  shift
) else if /I "%~1"=="release" (
  set "MODE=release"
  shift
) else if not "%~1"=="" (
  echo Usage: %~nx0 [debug^|release] [cpu^|cpu-gpu^|off]
  exit /b 1
)

if /I "%~1"=="cpu" (
  set "TRACY_MODE=cpu"
) else if /I "%~1"=="cpu-gpu" (
  set "TRACY_MODE=cpu-gpu"
) else if /I "%~1"=="off" (
  set "TRACY_MODE=off"
) else if not "%~1"=="" (
  echo Usage: %~nx0 [debug^|release] [cpu^|cpu-gpu^|off]
  exit /b 1
)

call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" editor "%TRACY_MODE%"
exit /b %errorlevel%

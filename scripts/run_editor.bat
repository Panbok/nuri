@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=debug"
set "TRACY_MODE="
set "DEVCHECKS="
if /I "%~1"=="debug" (
  shift
) else if /I "%~1"=="release" (
  set "MODE=release"
  shift
)

if /I "%~1"=="cpu" (
  set "TRACY_MODE=cpu"
  shift
) else if /I "%~1"=="off" (
  set "TRACY_MODE=off"
  shift
)
if /I "%~1"=="devchecks" (
  set "DEVCHECKS=devchecks"
  shift
)

call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" editor "%TRACY_MODE%" "%DEVCHECKS%"
if errorlevel 1 exit /b 1

call "%SCRIPT_DIR%_nuri_exec.bat" "%MODE%" editor
exit /b %errorlevel%

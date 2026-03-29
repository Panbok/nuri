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
) else if not "%~1"=="" (
  echo Usage: %~nx0 [debug^|release] [cpu^|cpu-gpu^|off] [devchecks]
  exit /b 1
)

:parse_args
if "%~1"=="" goto run_build
if /I "%~1"=="cpu" (
  if not "%TRACY_MODE%"=="" goto usage
  set "TRACY_MODE=cpu"
) else if /I "%~1"=="cpu-gpu" (
  if not "%TRACY_MODE%"=="" goto usage
  set "TRACY_MODE=cpu-gpu"
) else if /I "%~1"=="off" (
  if not "%TRACY_MODE%"=="" goto usage
  set "TRACY_MODE=off"
) else if /I "%~1"=="devchecks" (
  set "DEVCHECKS=devchecks"
) else (
  goto usage
)
shift
goto parse_args

:run_build
call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" app "%TRACY_MODE%" "%DEVCHECKS%"
exit /b %errorlevel%

:usage
echo Usage: %~nx0 [debug^|release] [cpu^|cpu-gpu^|off] [devchecks]
exit /b 1

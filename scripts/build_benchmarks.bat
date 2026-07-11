@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=release"
set "BUILD_ARGS="
if /I "%~1"=="debug" (
  set "MODE=debug"
  shift
) else if /I "%~1"=="release" (
  shift
) else if not "%~1"=="" (
  if /I not "%~1"=="cpu" if /I not "%~1"=="cpu-gpu" if /I not "%~1"=="off" if /I not "%~1"=="devchecks" goto usage
)

:collect_args
if "%~1"=="" goto build
set "BUILD_ARGS=%BUILD_ARGS% %1"
shift
goto collect_args

:build
call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" bench %BUILD_ARGS%
exit /b %errorlevel%

:usage
echo Usage: %~nx0 [release^|debug] [cpu^|cpu-gpu^|off] [devchecks]
exit /b 1

@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=app"
set "TRACY_MODE=%~2"
set "DEVCHECKS=%~3"
if not "%~4"=="" (
  echo Usage: %~nx0 [lib^|app^|editor^|tests] [cpu^|cpu-gpu^|off] [devchecks]
  exit /b 1
)

call "%SCRIPT_DIR%_nuri_build.bat" release "%PROFILE%" "%TRACY_MODE%" "%DEVCHECKS%"
exit /b %errorlevel%

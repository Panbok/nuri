@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=app"
set "TRACY_MODE=%~2"
if not "%~3"=="" (
  echo Usage: %~nx0 [lib^|app^|editor^|tests] [cpu^|off]
  exit /b 1
)

call "%SCRIPT_DIR%_nuri_build.bat" debug "%PROFILE%" "%TRACY_MODE%"
exit /b %errorlevel%

@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=app"
if not "%~2"=="" (
  echo Usage: %~nx0 [lib^|app^|editor^|tests]
  exit /b 1
)

call "%SCRIPT_DIR%_nuri_build.bat" debug "%PROFILE%"
exit /b %errorlevel%

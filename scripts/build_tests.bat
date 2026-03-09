@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=debug"
if /I "%~1"=="debug" (
  shift
) else if /I "%~1"=="release" (
  set "MODE=release"
  shift
) else if not "%~1"=="" (
  echo Usage: %~nx0 [debug^|release]
  exit /b 1
)

call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" tests
exit /b %errorlevel%

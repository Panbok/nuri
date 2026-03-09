@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=debug"
if /I "%~1"=="debug" (
  shift
) else if /I "%~1"=="release" (
  set "MODE=release"
  shift
)

call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" editor
if errorlevel 1 exit /b 1

call "%SCRIPT_DIR%_nuri_exec.bat" "%MODE%" editor %*
exit /b %errorlevel%

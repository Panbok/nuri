@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=debug"
set "PROFILE=tests"
if /I "%~1"=="debug" (
  shift
) else if /I "%~1"=="release" (
  set "MODE=release"
  shift
) else if /I "%~1"=="bench-tests" (
  set "PROFILE=bench-tests"
  shift
) else if /I "%~1"=="snapshot-tests" (
  set "PROFILE=snapshot-tests"
  shift
) else if /I "%~1"=="autotest-tests" (
  set "PROFILE=autotest-tests"
  shift
) else if not "%~1"=="" (
  goto usage
)

if /I "%~1"=="bench-tests" (
  set "PROFILE=bench-tests"
  shift
) else if /I "%~1"=="snapshot-tests" (
  set "PROFILE=snapshot-tests"
  shift
) else if /I "%~1"=="autotest-tests" (
  set "PROFILE=autotest-tests"
  shift
) else if not "%~1"=="" (
  goto usage
)

call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" "%PROFILE%"
exit /b %errorlevel%

:usage
echo Usage: %~nx0 [debug^|release] [bench-tests^|snapshot-tests^|autotest-tests]
exit /b 1

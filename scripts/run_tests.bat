@echo off
setlocal EnableExtensions EnableDelayedExpansion
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=debug"
set "PROFILE=tests"
set "BUILD_SUFFIX=debug"
set "arg=%~1"

if /I "%arg%"=="debug" (
  shift
) else if /I "%arg%"=="release" (
  set "MODE=release"
  shift
) else if /I "%arg%"=="bench-tests" (
  set "PROFILE=bench-tests"
  shift
) else if /I "%arg%"=="snapshot-tests" (
  set "PROFILE=snapshot-tests"
  shift
) else if /I "%arg%"=="autotest-tests" (
  set "PROFILE=autotest-tests"
  shift
) else if not "%arg%"=="" (
  if not "%arg:~0,1%"=="-" (
    echo Usage: %~nx0 [debug^|release] [bench-tests^|snapshot-tests^|autotest-tests] [ctest args...]
    exit /b 1
  )
)

if /I "%~1"=="bench-tests" (
  set "PROFILE=bench-tests"
  shift
)
if /I "%~1"=="snapshot-tests" (
  set "PROFILE=snapshot-tests"
  shift
)
if /I "%~1"=="autotest-tests" (
  set "PROFILE=autotest-tests"
  shift
)

if /I "%PROFILE%"=="bench-tests" (
  set "BUILD_SUFFIX=%MODE%-bench-tests"
) else if /I "%PROFILE%"=="snapshot-tests" (
  set "BUILD_SUFFIX=%MODE%-snapshot-tests"
) else if /I "%PROFILE%"=="autotest-tests" (
  set "BUILD_SUFFIX=%MODE%-autotest-tests"
) else (
  set "BUILD_SUFFIX=%MODE%"
)

call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" "%PROFILE%"
if errorlevel 1 exit /b 1

set "NURI_CTEST_MODE=%MODE%"
set "NURI_CTEST_PROFILE=%PROFILE%"
set "NURI_CTEST_ARGC=0"
:collect_ctest_args
if "%~1"=="" goto run_ctest
set "NURI_CTEST_ARG_!NURI_CTEST_ARGC!=%~1"
set /a NURI_CTEST_ARGC+=1 >nul
shift
goto collect_ctest_args

:run_ctest
python "%SCRIPT_DIR%_nuri_ctest.py" --from-env
exit /b %errorlevel%

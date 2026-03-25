@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=debug"
set "arg=%~1"

if /I "%arg%"=="debug" (
  shift
) else if /I "%arg%"=="release" (
  set "MODE=release"
  shift
) else if not "%arg%"=="" (
  if not "%arg:~0,1%"=="-" (
    echo Usage: %~nx0 [debug^|release] [ctest args...]
    exit /b 1
  )
)

call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" tests
if errorlevel 1 exit /b 1

for %%i in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fi"
call :set_build_dir "%REPO_ROOT%" "%MODE%" tests

set "CTEST_ARGS="
set "HAS_JOBS_ARG=0"
:collect_ctest_args
if "%~1"=="" goto run_ctest
if /I "%~1"=="-j" set "HAS_JOBS_ARG=1"
if /I "%~1"=="--parallel" set "HAS_JOBS_ARG=1"
set "CTEST_ARGS=%CTEST_ARGS% %1"
shift
goto collect_ctest_args

:run_ctest
set "CTEST_PARALLEL_ARGS="
if "%HAS_JOBS_ARG%"=="0" (
  if defined NUMBER_OF_PROCESSORS (
    set "CTEST_PARALLEL_ARGS= -j %NUMBER_OF_PROCESSORS%"
  ) else (
    set "CTEST_PARALLEL_ARGS= -j 4"
  )
)
ctest --test-dir "%BUILD_DIR%" --output-on-failure%CTEST_PARALLEL_ARGS%%CTEST_ARGS%
exit /b %errorlevel%

:set_build_dir
if /I "%~2"=="release" (
  set "BUILD_DIR=%~1\build_release\%~3"
  exit /b 0
)
if /I "%~3"=="app" (
  set "BUILD_DIR=%~1\build"
  exit /b 0
)
set "BUILD_DIR=%~1\build_%~3"
exit /b 0

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
set "BUILD_DIR=%REPO_ROOT%\build\%MODE%\tests"

set "CTEST_ARGS="
:collect_ctest_args
if "%~1"=="" goto run_ctest
set "CTEST_ARGS=%CTEST_ARGS% %1"
shift
goto collect_ctest_args

:run_ctest
ctest --test-dir "%BUILD_DIR%" --output-on-failure%CTEST_ARGS%
exit /b %errorlevel%

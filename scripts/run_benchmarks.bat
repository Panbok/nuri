@echo off
setlocal EnableExtensions
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=release"
set "BUILD_ARGS="
set "TOOL_ARGS="
set "FIRST_TOOL_ARG="
set "NO_BUILD=0"

if /I "%~1"=="debug" (
  set "MODE=debug"
  shift
) else if /I "%~1"=="release" (
  shift
)

:parse_build_args
if "%~1"=="" goto build
if /I "%~1"=="cpu" (
  set "BUILD_ARGS=%BUILD_ARGS% cpu"
) else if /I "%~1"=="--no-build" (
  set "NO_BUILD=1"
) else if /I "%~1"=="off" (
  set "BUILD_ARGS=%BUILD_ARGS% off"
) else if /I "%~1"=="devchecks" (
  set "BUILD_ARGS=%BUILD_ARGS% devchecks"
) else (
  goto collect_tool_args
)
shift
goto parse_build_args

:collect_tool_args
if "%~1"=="" goto build
if "%FIRST_TOOL_ARG%"=="" set "FIRST_TOOL_ARG=%~1"
set "TOOL_ARGS=%TOOL_ARGS% %1"
shift
goto collect_tool_args

:build
if "%NO_BUILD%"=="0" (
  call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" bench %BUILD_ARGS%
  if errorlevel 1 exit /b 1
)

for %%i in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fi"
set "BUILD_DIR=%REPO_ROOT%\build\%MODE%-bench"
set "TOOL=%BUILD_DIR%\nuri-bench.exe"
if not exist "%TOOL%" (
  echo Build output not found: %TOOL%
  exit /b 1
)

if exist "%BUILD_DIR%\lib" set "PATH=%BUILD_DIR%\lib;%PATH%"
for /d %%d in ("%BUILD_DIR%\vcpkg_installed\*\bin") do set "PATH=%%~fd;%PATH%"
for /d %%d in ("%BUILD_DIR%\vcpkg_installed\*\debug\bin") do set "PATH=%%~fd;%PATH%"

if "%TOOL_ARGS%"=="" (
  set "TOOL_ARGS=list"
  set "FIRST_TOOL_ARG=list"
)
if /I not "%FIRST_TOOL_ARG%"=="list" if /I not "%FIRST_TOOL_ARG%"=="explain" if /I not "%FIRST_TOOL_ARG%"=="run" if /I not "%FIRST_TOOL_ARG%"=="check" if /I not "%FIRST_TOOL_ARG%"=="compare" if /I not "%FIRST_TOOL_ARG%"=="summarize" if /I not "%FIRST_TOOL_ARG%"=="graph" if /I not "%FIRST_TOOL_ARG%"=="baseline" set "TOOL_ARGS=run %TOOL_ARGS%"

"%TOOL%" %TOOL_ARGS%
exit /b %errorlevel%

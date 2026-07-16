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
if not defined FIRST_TOOL_ARG set "FIRST_TOOL_ARG=%~1"
set "TOOL_ARGS=%TOOL_ARGS% %1"
shift
goto collect_tool_args

:build
if "%NO_BUILD%"=="0" (
  call "%SCRIPT_DIR%_nuri_build.bat" "%MODE%" snapshot %BUILD_ARGS%
  if errorlevel 1 exit /b 1
)

for %%i in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fi"
set "BUILD_DIR=%REPO_ROOT%\build\%MODE%-snapshot"
set "TOOL=%BUILD_DIR%\nuri-snapshot.exe"
if not exist "%TOOL%" (
  echo Build output not found: %TOOL%
  exit /b 1
)

if exist "%BUILD_DIR%\lib" set "PATH=%BUILD_DIR%\lib;%PATH%"
for /d %%d in ("%BUILD_DIR%\vcpkg_installed\*\bin") do set "PATH=%%~fd;%PATH%"
for /d %%d in ("%BUILD_DIR%\vcpkg_installed\*\debug\bin") do set "PATH=%%~fd;%PATH%"

if not defined TOOL_ARGS (
  set "TOOL_ARGS=list"
  set "FIRST_TOOL_ARG=list"
)
if /I "%FIRST_TOOL_ARG%"=="list" goto execute_tool
if /I "%FIRST_TOOL_ARG%"=="explain" goto execute_tool
if /I "%FIRST_TOOL_ARG%"=="capture" goto execute_tool
if /I "%FIRST_TOOL_ARG%"=="compare" goto execute_tool
if /I "%FIRST_TOOL_ARG%"=="run" goto execute_tool
if /I "%FIRST_TOOL_ARG%"=="approve" goto execute_tool
if /I "%FIRST_TOOL_ARG%"=="baseline" goto execute_tool
if /I "%FIRST_TOOL_ARG%"=="diff" goto execute_tool
set "TOOL_ARGS=run %TOOL_ARGS%"

:execute_tool
"%TOOL%" %TOOL_ARGS%
exit /b %errorlevel%

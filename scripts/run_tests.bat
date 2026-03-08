@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if not "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR%\"
pushd "%SCRIPT_DIR%.."
if errorlevel 1 exit /b 1

set "MODE=debug"
set "arg=%~1"
set "NURI_BUILD_TESTS=ON"
set "NURI_BUILD_EDITOR=OFF"
set "NURI_TEST_MANIFEST_FEATURES=%VCPKG_MANIFEST_FEATURES%"
if "%NURI_TEST_MANIFEST_FEATURES%"=="" (
  set "NURI_TEST_MANIFEST_FEATURES=tests"
) else (
  echo ,%NURI_TEST_MANIFEST_FEATURES%, | findstr /I /C:",tests," >nul
  if errorlevel 1 set "NURI_TEST_MANIFEST_FEATURES=%NURI_TEST_MANIFEST_FEATURES%,tests"
)
set "VCPKG_MANIFEST_FEATURES=%NURI_TEST_MANIFEST_FEATURES%"

if /I "%arg%"=="debug" (
  shift
) else if /I "%arg%"=="release" (
  set "MODE=release"
  shift
) else if not "%arg%"=="" (
  if not "%arg:~0,1%"=="-" (
    echo Usage: %~nx0 [debug^|release] [ctest args...]
    set "EXIT_CODE=1"
    goto cleanup
  )
)

if /I "%MODE%"=="debug" (
  call "%SCRIPT_DIR%build_debug.bat"
  if errorlevel 1 (
    set "EXIT_CODE=1"
    goto cleanup
  )
  set "BUILD_DIR=build"
) else (
  call "%SCRIPT_DIR%build_release.bat"
  if errorlevel 1 (
    set "EXIT_CODE=1"
    goto cleanup
  )
  set "BUILD_DIR=build_release"
)

set "CTEST_ARGS="
:collect_ctest_args
if "%~1"=="" goto run_ctest
set "CTEST_ARGS=%CTEST_ARGS% %1"
shift
goto collect_ctest_args

:run_ctest
ctest --test-dir "%BUILD_DIR%" --output-on-failure%CTEST_ARGS%
set "EXIT_CODE=%errorlevel%"
:cleanup
popd
exit /b %EXIT_CODE%
